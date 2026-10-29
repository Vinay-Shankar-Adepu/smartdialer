"""Atomic reservation and state transitions via optimistic locking.

Every mutable row carries a `version`. A write is

    UPDATE t SET ..., version = version + 1 WHERE id = ? AND version = <read version>

so when N workers read the same row and race to change it, exactly one UPDATE
matches and the rest affect 0 rows and lose. Nobody blocks while holding a
lock, and nothing is left locked if a worker crashes. Losers simply re-read
and retry, or move on to another candidate.

Ownership checks (`reserved_for_call_id == call_id`, `current_call_id ==
call_id`) sit on top of versioning. A worker can only release what it
reserved, so a slow or recovered worker can never free a reservation that now
belongs to someone else.
"""
from typing import Any, Dict, FrozenSet, Iterable, Optional

from core.clock import Clock
from core.database import Database
from core.models import Agent, Borrower, Call
from core.state_machines import (
    AgentState,
    BorrowerState,
    CallState,
    TERMINAL_CALL_STATES,
    assert_agent_transition,
    assert_borrower_transition,
    assert_call_transition,
)

_ANY = object()


class LockManager:
    MAX_CAS_RETRIES = 8

    def __init__(self, db: Database, clock: Clock):
        self.db = db
        self.clock = clock

    # ================================================================ agents
    def reserve_agent(self, agent_id: str, call_id: str) -> bool:
        """Reserve one specific agent for `call_id`. True only for the single winner."""
        agent = self.db.get_agent(agent_id)
        if agent is None or agent.state != AgentState.AVAILABLE:
            return False
        return self._cas_reserve(agent, call_id)

    def reserve_any_available_agent(self, call_id: str, candidates: int = 8) -> Optional[str]:
        """Reserve the longest-idle available agent. Returns its id, or None.

        Losing a CAS race on one candidate just means another worker got that
        agent, so we try the next one.
        """
        for agent in self.db.find_available_agents(limit=candidates):
            if self._cas_reserve(agent, call_id):
                return agent.id
        return None

    def _cas_reserve(self, agent: Agent, call_id: str) -> bool:
        return self.db.cas_update(
            "agents", agent.id, agent.version,
            {"state": AgentState.RESERVED, "reserved_for_call_id": call_id,
             "state_changed_at": self.clock.now(), "wrap_until": None},
            expect={"state": AgentState.AVAILABLE, "reserved_for_call_id": None},
        )

    def release_agent(self, agent_id: str, call_id: str) -> bool:
        """RESERVED/DIALING -> AVAILABLE (or OFFLINE if a logout is pending).

        Only succeeds if the agent is still reserved for `call_id`.
        """
        for _ in range(self.MAX_CAS_RETRIES):
            agent = self.db.get_agent(agent_id)
            if (agent is None or agent.reserved_for_call_id != call_id
                    or agent.state not in (AgentState.RESERVED, AgentState.DIALING)):
                return False
            target = AgentState.OFFLINE if agent.logout_requested else AgentState.AVAILABLE
            changes = {"state": target, "reserved_for_call_id": None,
                       "state_changed_at": self.clock.now(), "wrap_until": None}
            if target == AgentState.OFFLINE:
                changes["logout_requested"] = False
            if self.db.cas_update("agents", agent.id, agent.version, changes):
                return True
        return False

    def transition_agent(self, agent_id: str, to_state: AgentState, *,
                         from_states: Iterable[AgentState], owned_by_call: Any = _ANY,
                         changes: Optional[Dict[str, Any]] = None) -> bool:
        """Generic validated CAS transition for an agent."""
        from_set: FrozenSet[AgentState] = frozenset(from_states)
        for _ in range(self.MAX_CAS_RETRIES):
            agent = self.db.get_agent(agent_id)
            if agent is None or agent.state not in from_set:
                return False
            if owned_by_call is not _ANY and agent.reserved_for_call_id != owned_by_call:
                return False
            assert_agent_transition(agent.state, to_state)
            update = {"state": to_state, "state_changed_at": self.clock.now()}
            update.update(changes or {})
            if self.db.cas_update("agents", agent.id, agent.version, update):
                return True
        return False

    # ================================================================ borrowers
    def claim_borrower(self, call_id: str, worker_id: str, lease_s: float,
                       candidates: int = 8) -> Optional[Borrower]:
        """Atomically claim the next due borrower for `call_id`."""
        now = self.clock.now()
        for borrower in self.db.find_due_borrowers(now, limit=candidates):
            won = self.db.cas_update(
                "borrowers", borrower.id, borrower.version,
                {"state": BorrowerState.RESERVED, "current_call_id": call_id,
                 "lease_owner": worker_id, "lease_expires_at": now + lease_s},
                expect={"state": BorrowerState.PENDING},
            )
            if won:
                return self.db.get_borrower(borrower.id)
        return None

    def finish_borrower(self, borrower_id: str, call_id: Optional[str], to_state: BorrowerState, *,
                        attempts_delta: int = 0, next_attempt_at: Optional[float] = None,
                        outcome: Optional[str] = None) -> bool:
        """Settle a RESERVED borrower. Only the call that owns it may do so."""
        for _ in range(self.MAX_CAS_RETRIES):
            b = self.db.get_borrower(borrower_id)
            if b is None or b.state != BorrowerState.RESERVED or b.current_call_id != call_id:
                return False
            attempts = b.attempts + attempts_delta
            target = to_state
            if target == BorrowerState.PENDING and attempts >= b.max_attempts:
                target = BorrowerState.EXHAUSTED
            assert_borrower_transition(b.state, target)
            changes = {"state": target, "attempts": attempts, "current_call_id": None,
                       "lease_owner": None, "lease_expires_at": None,
                       "outcome": outcome or b.outcome}
            if next_attempt_at is not None:
                changes["next_attempt_at"] = next_attempt_at
            if self.db.cas_update("borrowers", b.id, b.version, changes):
                return True
        return False

    # ================================================================ calls
    def transition_call(self, call_id: str, to_state: CallState, *,
                        from_states: Optional[Iterable[CallState]] = None,
                        changes: Optional[Dict[str, Any]] = None) -> Optional[Call]:
        """Validated CAS transition. Returns the updated call, or None if the
        call is gone, already terminal, or not in `from_states`."""
        from_set = frozenset(from_states) if from_states is not None else None
        for _ in range(self.MAX_CAS_RETRIES):
            call = self.db.get_call(call_id)
            if call is None or call.state in TERMINAL_CALL_STATES:
                return None
            if from_set is not None and call.state not in from_set:
                return None
            assert_call_transition(call.state, to_state)
            update = {"state": to_state}
            update.update(changes or {})
            if self.db.cas_update("calls", call.id, call.version, update):
                return self.db.get_call(call_id)
        return None

    def update_call_fields(self, call_id: str, changes: Dict[str, Any], *,
                           only_in_states: Iterable[CallState]) -> bool:
        """Change non-state fields (e.g. unbind an agent) with CAS protection."""
        allowed = frozenset(only_in_states)
        for _ in range(self.MAX_CAS_RETRIES):
            call = self.db.get_call(call_id)
            if call is None or call.state not in allowed:
                return False
            if self.db.cas_update("calls", call.id, call.version, changes):
                return True
        return False
