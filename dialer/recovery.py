"""Crash and stuck-state recovery (the "reaper").

Runs every few seconds and is idempotent, so running it twice, or on two
nodes at once, is safe because every change is a CAS or ownership-checked.

1. Expired worker leases (QUEUED/RESERVED calls whose worker went silent).
   Ask the provider "do you have call <our id>?":
     yes     -> ADOPT it: mark INITIATED and let events drive it (the borrower is already ringing)
     no      -> CANCEL: release agent and borrower, no attempt consumed
     unknown -> provider unreachable: retry next sweep
2. Stuck calls (INITIATED/RINGING far longer than any real ring).
   Ask the provider. If the call is gone, fail it. If it is still "ringing",
   hang up and fail it. If the provider is unreachable, keep it until a hard
   deadline (outages end, and the webhooks usually arrive), then fail it
   locally.
3. Orphan agents: busy agents whose call ended or vanished go back to AVAILABLE/WRAP_UP.
4. Orphan borrowers: RESERVED with an expired lease and no live call go back to PENDING.
5. Lost hangups: CONNECTED far longer than any real conversation. Ask the
   provider. If it no longer has a live call, the COMPLETED event was lost:
   complete the call so the agent goes to WRAP_UP. A genuinely live call is left
   alone, and an unreachable provider means try again next sweep.
"""
from collections import Counter
from dataclasses import dataclass

from core.clock import Clock
from core.database import Database
from core.lock_manager import LockManager
from core.state_machines import (
    AGENT_BUSY_STATES,
    AgentState,
    BorrowerState,
    CallState,
    TERMINAL_CALL_STATES,
)
from dialer.call_lifecycle import CallLifecycle
from dialer.safety_controller import SafeDialGateway
from providers.provider_interface import ProviderError


@dataclass
class RecoveryConfig:
    stuck_call_s: float = 120.0      # no real call rings this long
    hard_stuck_s: float = 300.0      # give up waiting for an unreachable provider
    long_connected_s: float = 1800.0 # check with the provider whether a CONNECTED call still exists


class RecoveryService:
    def __init__(self, db: Database, locks: LockManager, lifecycle: CallLifecycle,
                 gateway: SafeDialGateway, clock: Clock, config: RecoveryConfig = None):
        self.db = db
        self.locks = locks
        self.lifecycle = lifecycle
        self.gateway = gateway
        self.clock = clock
        self.cfg = config or RecoveryConfig()
        self.stats: Counter = Counter()

    def sweep(self) -> Counter:
        result: Counter = Counter()
        self._expired_leases(result)
        self._stuck_calls(result)
        self._orphan_agents(result)
        self._orphan_borrowers(result)
        self._lost_hangups(result)
        self.stats.update(result)
        return result

    # ------------------------------------------------------------------ 1
    def _expired_leases(self, result: Counter) -> None:
        now = self.clock.now()
        for call in self.db.find_expired_call_leases(now, [CallState.QUEUED, CallState.RESERVED]):
            if call.state == CallState.QUEUED:     # never dialed (dial requires RESERVED)
                if self.lifecycle.finalize(call.id, CallState.CANCELLED, "worker_crash",
                                           from_states={CallState.QUEUED}):
                    result["lease_cancelled"] += 1
                continue
            try:
                status = self.gateway.lookup(call.id)
            except ProviderError:
                result["lease_lookup_failed"] += 1
                continue
            if status is None:
                if self.lifecycle.finalize(call.id, CallState.CANCELLED, "worker_crash",
                                           from_states={CallState.RESERVED}):
                    result["lease_cancelled"] += 1
                continue
            with self.db.transaction():
                adopted = self.locks.transition_call(
                    call.id, CallState.INITIATED, from_states={CallState.RESERVED},
                    changes={"provider_call_id": status.provider_call_id, "initiated_at": now,
                             "lease_expires_at": None})
                if adopted and adopted.agent_id:
                    self.locks.transition_agent(adopted.agent_id, AgentState.DIALING,
                                                from_states={AgentState.RESERVED}, owned_by_call=call.id)
            if adopted:
                result["lease_adopted"] += 1

    # ------------------------------------------------------------------ 2
    def _stuck_calls(self, result: Counter) -> None:
        now = self.clock.now()
        for call in self.db.find_calls([CallState.INITIATED, CallState.RINGING]):
            started = call.initiated_at or call.created_at
            age = now - started
            if age < self.cfg.stuck_call_s:
                continue
            try:
                status = self.gateway.lookup(call.id)
            except ProviderError:
                if age >= self.cfg.hard_stuck_s:
                    if self.lifecycle.finalize(call.id, CallState.FAILED, "stuck_provider_unreachable",
                                               from_states={CallState.INITIATED, CallState.RINGING}):
                        result["stuck_failed_unreachable"] += 1
                else:
                    result["stuck_waiting_for_provider"] += 1
                continue
            if status is not None and status.is_live:
                self.gateway.hangup(status.provider_call_id)
            if self.lifecycle.finalize(call.id, CallState.FAILED,
                                       "ring_timeout" if status and status.is_live else "lost_by_provider",
                                       from_states={CallState.INITIATED, CallState.RINGING}):
                result["stuck_failed"] += 1

    # ------------------------------------------------------------------ 3
    def _orphan_agents(self, result: Counter) -> None:
        now = self.clock.now()
        for agent in self.db.list_agents(AGENT_BUSY_STATES):
            call = self.db.get_call(agent.reserved_for_call_id) if agent.reserved_for_call_id else None
            if call is not None and call.state not in TERMINAL_CALL_STATES:
                continue
            if agent.state == AgentState.CONNECTED:
                ok = self.locks.transition_agent(
                    agent.id, AgentState.WRAP_UP, from_states={AgentState.CONNECTED},
                    owned_by_call=agent.reserved_for_call_id,
                    changes={"reserved_for_call_id": None, "wrap_until": now + self.lifecycle.cfg.wrap_up_s})
            else:
                ok = self.locks.release_agent(agent.id, agent.reserved_for_call_id)
                if not ok and agent.reserved_for_call_id is None:
                    ok = self.locks.transition_agent(agent.id, AgentState.AVAILABLE,
                                                     from_states={AgentState.RESERVED, AgentState.DIALING},
                                                     owned_by_call=None)
            if ok:
                result["orphan_agent_released"] += 1

    # ------------------------------------------------------------------ 4
    def _orphan_borrowers(self, result: Counter) -> None:
        for b in self.db.find_expired_borrower_leases(self.clock.now()):
            call = self.db.get_call(b.current_call_id) if b.current_call_id else None
            if call is not None and call.state not in TERMINAL_CALL_STATES:
                continue                     # a live call owns it; the lease is irrelevant now
            if self.locks.finish_borrower(b.id, b.current_call_id, BorrowerState.PENDING):
                result["orphan_borrower_released"] += 1

    # ------------------------------------------------------------------ 5
    def _lost_hangups(self, result: Counter) -> None:
        now = self.clock.now()
        for call in self.db.find_calls([CallState.CONNECTED]):
            if now - (call.connected_at or call.created_at) < self.cfg.long_connected_s:
                continue
            try:
                status = self.gateway.lookup(call.id)
            except ProviderError:
                result["long_call_lookup_failed"] += 1
                continue
            if status is not None and status.is_live:
                continue                     # a genuinely long conversation
            if self.lifecycle.finalize(call.id, CallState.COMPLETED, "hangup_event_lost",
                                       from_states={CallState.CONNECTED}):
                result["lost_hangup_completed"] += 1
