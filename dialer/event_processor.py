"""Applies provider events to calls: idempotent and order-tolerant.

Two layers of protection:
1. Event-id dedup. `provider_events.event_id` is a PRIMARY KEY, so a
   redelivered webhook fails INSERT OR IGNORE and becomes a no-op.
2. State-machine classification. Even a duplicate with a NEW id (buggy
   retries) or a late or out-of-order event is classified against the
   call's current state:
     same state         -> no-op
     lower rank (stale) -> ignored  (RINGING after ANSWERED)
     forward jump       -> applied  (ANSWERED before RINGING)
     after terminal     -> ignored  (ANSWERED after COMPLETED)

The dedup insert and the state change share one transaction. If we crash
midway, both roll back and the redelivered event is processed from scratch.
"""
from collections import Counter

from core.clock import Clock
from core.database import Database
from core.lock_manager import LockManager
from core.models import ProviderEvent
from core.state_machines import (
    AgentState,
    CallState,
    EVENT_TARGET,
    EventDecision,
    TERMINAL_CALL_STATES,
    classify_event_transition,
)
from dialer.call_lifecycle import CallLifecycle

DUPLICATE_EVENT = "DUPLICATE_EVENT_ID"
UNKNOWN_CALL = "UNKNOWN_CALL"

_TIMESTAMP_FIELD = {CallState.RINGING: "ringing_at", CallState.ANSWERED: "answered_at"}


class EventProcessor:
    def __init__(self, db: Database, locks: LockManager, lifecycle: CallLifecycle, clock: Clock):
        self.db = db
        self.locks = locks
        self.lifecycle = lifecycle
        self.clock = clock
        self.stats: Counter = Counter()

    def process(self, event: ProviderEvent) -> str:
        outcome = self._process(event)
        self.stats[outcome] += 1
        return outcome

    def _process(self, event: ProviderEvent) -> str:
        now = self.clock.now()
        with self.db.transaction():
            if not self.db.insert_event_if_new(event, received_at=now):
                return DUPLICATE_EVENT
            call = self.db.get_call(event.call_id)
            if call is None:
                self.db.set_event_outcome(event.event_id, UNKNOWN_CALL)
                return UNKNOWN_CALL

            target, reason = EVENT_TARGET[event.event_type]
            decision = classify_event_transition(call.state, target)
            if decision == EventDecision.APPLY:
                if target in TERMINAL_CALL_STATES:
                    self.lifecycle.finalize(call.id, target, reason)
                else:
                    self._advance(call, target, event, now)
            self.db.set_event_outcome(event.event_id, decision.value)
            return decision.value

    def _advance(self, call, target: CallState, event: ProviderEvent, now: float) -> None:
        changes = {"lease_expires_at": None,           # provider confirmed the call: no longer worker-owned
                   "provider_call_id": call.provider_call_id or event.provider_call_id}
        if call.initiated_at is None:
            changes["initiated_at"] = now
        if target in _TIMESTAMP_FIELD:
            changes[_TIMESTAMP_FIELD[target]] = now
        self.locks.transition_call(call.id, target, from_states={call.state}, changes=changes)
        if call.agent_id:                              # bound agent: RESERVED -> DIALING once the line is live
            self.locks.transition_agent(call.agent_id, AgentState.DIALING, from_states={AgentState.RESERVED},
                                        owned_by_call=call.id)
