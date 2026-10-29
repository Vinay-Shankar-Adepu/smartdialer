"""Call lifecycle side effects, in one place.

Everything that ends a call goes through `finalize()`, and everything that
puts a borrower through to an agent goes through `try_connect()`. Both run
inside a single DB transaction, so the call, the agent and the borrower
change together or not at all. A crash halfway leaves the previous
consistent state, and the next pass simply retries.

ANSWERED != CONNECTED
---------------------
An ANSWERED event only records that the borrower picked up. Bridging to an
agent (`bridge_answered_calls`) is a separate, idempotent pass. If the
process dies between the two, the call sits in ANSWERED and the next pass
connects it or, after `max_answer_hold_s`, abandons it and hangs up.
"""
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from core.clock import Clock
from core.database import Database
from core.lock_manager import LockManager
from core.models import Call
from core.state_machines import AgentState, BorrowerState, CallState

# Failures that are not the borrower's fault: retry soon without spending an attempt.
NO_ATTEMPT_REASONS = frozenset({"provider_timeout", "provider_error", "worker_crash",
                                "no_agent", "permit_rejected", "lost_reservation", "lost_by_provider"})


@dataclass
class LifecycleConfig:
    wrap_up_s: float = 10.0
    retry_delay_s: float = 900.0           # no answer -> try again in 15 min
    abandon_retry_delay_s: float = 60.0    # we abandoned them -> call back quickly
    provider_retry_delay_s: float = 30.0
    max_answer_hold_s: float = 2.0         # answered caller may wait this long for an agent
    lease_s: float = 10.0


class _RetryLater(Exception):
    """Internal: roll back this attempt, the next pass will retry."""


class CallLifecycle:
    def __init__(self, db: Database, locks: LockManager, clock: Clock, config: LifecycleConfig):
        self.db = db
        self.locks = locks
        self.clock = clock
        self.cfg = config
        self.stats: Counter = Counter()

    # ------------------------------------------------------------------ finalize
    def finalize(self, call_id: str, terminal_state: CallState, reason: Optional[str] = None,
                 from_states: Optional[Iterable[CallState]] = None) -> Optional[Call]:
        """Move a call to a terminal state and release everything it held.

        Idempotent: returns None (and does nothing) if the call already ended.
        """
        now = self.clock.now()
        with self.db.transaction():
            call = self.locks.transition_call(
                call_id, terminal_state, from_states=from_states,
                changes={"ended_at": now, "failure_reason": reason, "lease_expires_at": None})
            if call is None:
                return None
            self._release_agents(call, now)
            self._settle_borrower(call, terminal_state, reason, now)
        self.stats[f"finalized_{terminal_state.value}"] += 1
        return call

    def _release_agents(self, call: Call, now: float) -> None:
        # Any agent still reserved for this call, even one the call row does not
        # point at (e.g. a crash between reserving the agent and binding it).
        for agent in self.db.find_agents_reserved_for(call.id):
            if agent.state == AgentState.CONNECTED:
                self.locks.transition_agent(
                    agent.id, AgentState.WRAP_UP, from_states={AgentState.CONNECTED}, owned_by_call=call.id,
                    changes={"reserved_for_call_id": None, "wrap_until": now + self.cfg.wrap_up_s})
            elif agent.state in (AgentState.RESERVED, AgentState.DIALING):
                self.locks.release_agent(agent.id, call.id)

    def _settle_borrower(self, call: Call, state: CallState, reason: Optional[str], now: float) -> None:
        if state == CallState.COMPLETED and call.connected_at is not None:
            self.locks.finish_borrower(call.borrower_id, call.id, BorrowerState.DONE, outcome="contacted")
        elif state == CallState.CANCELLED or reason in NO_ATTEMPT_REASONS:
            delay = self.cfg.provider_retry_delay_s if reason and reason.startswith("provider") else 0.0
            self.locks.finish_borrower(call.borrower_id, call.id, BorrowerState.PENDING,
                                       next_attempt_at=now + delay)
        elif state == CallState.ABANDONED:
            self.locks.finish_borrower(call.borrower_id, call.id, BorrowerState.PENDING, attempts_delta=1,
                                       next_attempt_at=now + self.cfg.abandon_retry_delay_s,
                                       outcome="abandoned")
        else:
            self.locks.finish_borrower(call.borrower_id, call.id, BorrowerState.PENDING, attempts_delta=1,
                                       next_attempt_at=now + self.cfg.retry_delay_s,
                                       outcome=reason or state.value.lower())

    # ------------------------------------------------------------------ connect
    def try_connect(self, call_id: str) -> str:
        """Bridge an ANSWERED call to an agent. Returns 'connected', 'waiting' or 'gone'."""
        now = self.clock.now()
        try:
            with self.db.transaction():
                call = self.db.get_call(call_id)
                if call is None or call.state != CallState.ANSWERED:
                    return "gone"
                agent_id = None
                if call.agent_id:                       # progressive: agent already reserved for it
                    agent = self.db.get_agent(call.agent_id)
                    if (agent and agent.reserved_for_call_id == call.id
                            and agent.state in (AgentState.RESERVED, AgentState.DIALING)):
                        agent_id = agent.id
                if agent_id is None:                    # predictive (or agent lost): grab any idle agent
                    agent_id = self.locks.reserve_any_available_agent(call.id)
                    if agent_id is None:
                        return "waiting"
                if not self.locks.transition_agent(agent_id, AgentState.CONNECTED,
                                                   from_states={AgentState.RESERVED, AgentState.DIALING},
                                                   owned_by_call=call.id):
                    raise _RetryLater()
                if self.locks.transition_call(call.id, CallState.CONNECTED, from_states={CallState.ANSWERED},
                                              changes={"agent_id": agent_id, "connected_at": now}) is None:
                    raise _RetryLater()
        except _RetryLater:
            return "waiting"
        self.stats["connected"] += 1
        return "connected"

    def bridge_answered_calls(self, hangup: Callable[[str], bool]) -> Counter:
        """Connect every ANSWERED call we can; abandon those that waited too long."""
        result: Counter = Counter()
        now = self.clock.now()
        for call in self.db.find_calls([CallState.ANSWERED], order_by="answered_at"):
            outcome = self.try_connect(call.id)
            if outcome == "waiting" and call.answered_at is not None \
                    and now - call.answered_at >= self.cfg.max_answer_hold_s:
                ended = self.finalize(call.id, CallState.ABANDONED, "no_agent_available",
                                      from_states={CallState.ANSWERED})
                if ended is not None:
                    outcome = "abandoned"
                    if ended.provider_call_id:
                        hangup(ended.provider_call_id)
            result[outcome] += 1
        return result
