"""Domain dataclasses. One row in the database maps to one of these."""
from dataclasses import dataclass
from typing import Optional

from core.state_machines import AgentState, BorrowerState, CallState, ProviderEventType, TERMINAL_CALL_STATES


@dataclass
class Agent:
    id: str
    name: str
    state: AgentState = AgentState.OFFLINE
    version: int = 0                          # optimistic-lock counter, +1 on every write
    reserved_for_call_id: Optional[str] = None
    state_changed_at: float = 0.0             # used to pick the longest-idle agent first
    wrap_until: Optional[float] = None
    logout_requested: bool = False            # graceful logout after the current call


@dataclass
class Borrower:
    id: str
    name: str
    phone: str
    state: BorrowerState = BorrowerState.PENDING
    version: int = 0
    attempts: int = 0
    max_attempts: int = 3
    next_attempt_at: float = 0.0
    current_call_id: Optional[str] = None
    lease_owner: Optional[str] = None         # worker that claimed the borrower
    lease_expires_at: Optional[float] = None
    outcome: Optional[str] = None


@dataclass
class Call:
    id: str
    borrower_id: str
    state: CallState = CallState.QUEUED
    version: int = 0
    agent_id: Optional[str] = None            # None = unbound (predictive) until an agent is bridged
    mode: str = "progressive"                 # mode of the permit the call was dialed under
    provider: Optional[str] = None
    provider_call_id: Optional[str] = None
    worker_id: Optional[str] = None
    lease_expires_at: Optional[float] = None  # set while a worker owns the call (QUEUED/RESERVED)
    created_at: float = 0.0
    initiated_at: Optional[float] = None
    ringing_at: Optional[float] = None
    answered_at: Optional[float] = None
    connected_at: Optional[float] = None
    ended_at: Optional[float] = None
    failure_reason: Optional[str] = None

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_CALL_STATES


@dataclass(frozen=True)
class ProviderEvent:
    """A callback from the telecom provider.

    `event_id` is the provider's idempotency key. A retried webhook reuses it.
    `call_id` is OUR call id, passed to the provider as its client reference
    when dialing, so an event can be matched to a call even when we never
    recorded the provider's call id (for example, the worker crashed after
    dialing).
    """
    event_id: str
    call_id: str
    provider_call_id: str
    event_type: ProviderEventType
    occurred_at: float
