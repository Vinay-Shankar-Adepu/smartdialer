"""Turns a DialPermit into actual calls: pick borrower, (pick agent), dial.

Per call, in this order:
  1. claim a borrower      (CAS, lease)        borrower PENDING -> RESERVED
  2. insert call row                           call QUEUED (lease, worker_id)
  3. progressive permit: reserve an agent      agent AVAILABLE -> RESERVED (CAS)
  4. mark call RESERVED (+agent_id)            call QUEUED -> RESERVED
  5. gateway.dial(permit, ...)                 safety redeems the permit, provider dials
  6. record INITIATED, agent -> DIALING        lease cleared: events own the call now

Crash safety: every step is individually atomic, and a worker that dies
between steps leaves rows with an expired lease. RecoveryService finds them
and either adopts the call (the provider has it) or cancels it and releases
the agent and borrower. Steps 1-4 never touch the outside world, so
cancelling them is always safe.

The allocator never decides HOW MANY to dial. That is the permit's job.
"""
import itertools
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

from core.clock import Clock
from core.database import Database
from core.lock_manager import LockManager
from core.models import Call
from core.state_machines import AgentState, CallState, TERMINAL_CALL_STATES
from dialer.call_lifecycle import CallLifecycle
from dialer.safety_controller import DialPermit, SafeDialGateway, SafetyViolation
from providers.provider_interface import ProviderCallHandle, ProviderError, ProviderTimeout

DIALED = "dialed"
NO_BORROWER = "no_borrower"
NO_AGENT = "no_agent"
PROVIDER_FAILED = "provider_failed"
REJECTED = "rejected"


class SimulatedCrash(BaseException):
    """Raised by test crash hooks. A BaseException, so no `except Exception`
    cleanup runs, which is exactly what a killed process looks like."""


@dataclass
class AllocationResult:
    outcomes: Counter = field(default_factory=Counter)

    @property
    def dialed(self) -> int:
        return self.outcomes[DIALED]


class CallAllocator:
    MAX_CONSECUTIVE_PROVIDER_ERRORS = 3

    def __init__(self, db: Database, locks: LockManager, lifecycle: CallLifecycle,
                 gateway: SafeDialGateway, clock: Clock, worker_id: str, lease_s: float = 10.0,
                 crash_hook: Optional[Callable[[str, str], None]] = None):
        self.db = db
        self.locks = locks
        self.lifecycle = lifecycle
        self.gateway = gateway
        self.clock = clock
        self.worker_id = worker_id
        self.lease_s = lease_s
        self.crash_hook = crash_hook or (lambda point, call_id: None)
        self._ids = itertools.count(1)

    def execute(self, permit: DialPermit) -> AllocationResult:
        result = AllocationResult()
        consecutive_errors = 0
        for _ in range(permit.max_calls):
            outcome = self._dial_one(permit)
            result.outcomes[outcome] += 1
            if outcome in (NO_BORROWER, NO_AGENT, REJECTED):
                break
            consecutive_errors = consecutive_errors + 1 if outcome == PROVIDER_FAILED else 0
            if consecutive_errors >= self.MAX_CONSECUTIVE_PROVIDER_ERRORS:
                break                 # stop hammering a failing provider; the breaker will see it
        return result

    def _new_call_id(self) -> str:
        return f"call-{self.worker_id}-{next(self._ids):07d}"

    def _dial_one(self, permit: DialPermit) -> str:
        now = self.clock.now()
        call_id = self._new_call_id()

        borrower = self.locks.claim_borrower(call_id, self.worker_id, self.lease_s)
        if borrower is None:
            return NO_BORROWER
        self.crash_hook("after_claim_borrower", call_id)

        self.db.insert_call(Call(id=call_id, borrower_id=borrower.id, state=CallState.QUEUED,
                                 mode=permit.mode, provider=self.gateway.provider_name,
                                 worker_id=self.worker_id, lease_expires_at=now + self.lease_s,
                                 created_at=now))

        agent_id = None
        if permit.requires_agent:
            agent_id = self.locks.reserve_any_available_agent(call_id)
            if agent_id is None:
                self.lifecycle.finalize(call_id, CallState.CANCELLED, "no_agent")
                return NO_AGENT
        self.crash_hook("after_reserve_agent", call_id)

        if self.locks.transition_call(call_id, CallState.RESERVED, from_states={CallState.QUEUED},
                                      changes={"agent_id": agent_id}) is None:
            self.lifecycle.finalize(call_id, CallState.CANCELLED, "lost_reservation")
            return REJECTED
        self.crash_hook("before_dial", call_id)

        try:
            handle = self.gateway.dial(permit, call_id, borrower.phone)
        except SafetyViolation:
            self.lifecycle.finalize(call_id, CallState.CANCELLED, "permit_rejected")
            return REJECTED
        except ProviderTimeout:
            # A timeout may hide a call that WAS placed. Cancel it by our reference.
            self.gateway.cancel_by_call_id(call_id)
            self.lifecycle.finalize(call_id, CallState.FAILED, "provider_timeout")
            return PROVIDER_FAILED
        except ProviderError:
            self.lifecycle.finalize(call_id, CallState.FAILED, "provider_error")
            return PROVIDER_FAILED

        self.crash_hook("after_dial", call_id)
        self._mark_initiated(call_id, handle)
        return DIALED

    def _mark_initiated(self, call_id: str, handle: ProviderCallHandle) -> None:
        now = self.clock.now()
        for _ in range(LockManager.MAX_CAS_RETRIES):
            call = self.db.get_call(call_id)
            if call.state == CallState.RESERVED:
                with self.db.transaction():
                    updated = self.locks.transition_call(
                        call_id, CallState.INITIATED, from_states={CallState.RESERVED},
                        changes={"provider_call_id": handle.provider_call_id, "initiated_at": now,
                                 "lease_expires_at": None})
                    if updated and updated.agent_id:
                        self.locks.transition_agent(updated.agent_id, AgentState.DIALING,
                                                    from_states={AgentState.RESERVED}, owned_by_call=call_id)
                if updated:
                    return
                continue
            if call.state in TERMINAL_CALL_STATES:
                # Recovery decided we were dead and cancelled the call: make sure it doesn't ring.
                self.gateway.hangup(handle.provider_call_id)
                return
            if not call.provider_call_id:     # events got here first; just record the id
                self.locks.update_call_fields(call_id, {"provider_call_id": handle.provider_call_id},
                                              only_in_states={call.state})
            return
