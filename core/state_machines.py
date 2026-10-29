"""Agent, Call and Borrower state machines.

Every state change in the system goes through the transition tables below.
There are two kinds of transitions:

* Internal transitions are made by our own code (reserve an agent, connect a
  call, cancel a call). An illegal internal transition is a bug, so
  `assert_*_transition` raises InvalidTransition.

* Provider-driven transitions come from telecom events, which may be
  duplicated, delayed or out of order. Those are classified by
  `classify_event_transition`, which never raises. It says whether to apply
  the event, treat it as a no-op duplicate, or ignore it as stale.
"""
from enum import Enum
from typing import Dict, FrozenSet


class InvalidTransition(Exception):
    """Raised when code attempts a transition the state machine forbids."""


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------
class AgentState(str, Enum):
    OFFLINE = "OFFLINE"        # not logged in / disconnected
    AVAILABLE = "AVAILABLE"    # logged in, idle, can be reserved
    RESERVED = "RESERVED"      # exclusively allocated to ONE call (reserved_for_call_id)
    DIALING = "DIALING"        # the call they are reserved for is ringing at the provider
    CONNECTED = "CONNECTED"    # talking to a borrower
    WRAP_UP = "WRAP_UP"        # post-call notes; becomes AVAILABLE when wrap_until passes
    PAUSED = "PAUSED"          # logged in but on break, never reserved


AGENT_TRANSITIONS: Dict[AgentState, FrozenSet[AgentState]] = {
    AgentState.OFFLINE: frozenset({AgentState.AVAILABLE}),
    AgentState.AVAILABLE: frozenset({AgentState.RESERVED, AgentState.PAUSED, AgentState.OFFLINE}),
    # RESERVED -> CONNECTED is the predictive path: an answered call grabs an idle agent.
    # RESERVED -> AVAILABLE releases the reservation (dial failed / cancelled).
    AgentState.RESERVED: frozenset({AgentState.DIALING, AgentState.CONNECTED,
                                    AgentState.AVAILABLE, AgentState.OFFLINE}),
    AgentState.DIALING: frozenset({AgentState.CONNECTED, AgentState.AVAILABLE, AgentState.OFFLINE}),
    AgentState.CONNECTED: frozenset({AgentState.WRAP_UP, AgentState.OFFLINE}),
    AgentState.WRAP_UP: frozenset({AgentState.AVAILABLE, AgentState.PAUSED, AgentState.OFFLINE}),
    AgentState.PAUSED: frozenset({AgentState.AVAILABLE, AgentState.OFFLINE}),
}

# Agents in these states hold a reservation and must point at a live call.
AGENT_BUSY_STATES = frozenset({AgentState.RESERVED, AgentState.DIALING, AgentState.CONNECTED})


def can_transition_agent(src: AgentState, dst: AgentState) -> bool:
    return dst in AGENT_TRANSITIONS[src]


def assert_agent_transition(src: AgentState, dst: AgentState) -> None:
    if not can_transition_agent(src, dst):
        raise InvalidTransition(f"agent {src.value} -> {dst.value} is not allowed")


# --------------------------------------------------------------------------
# Call
# --------------------------------------------------------------------------
class CallState(str, Enum):
    QUEUED = "QUEUED"          # call row created for a claimed borrower, nothing dialed yet
    RESERVED = "RESERVED"      # a worker owns the call (lease) and holds its resources; about to dial
    INITIATED = "INITIATED"    # provider accepted the dial request
    RINGING = "RINGING"        # borrower's phone is ringing
    ANSWERED = "ANSWERED"      # borrower picked up; NOT yet bridged to an agent
    CONNECTED = "CONNECTED"    # borrower bridged to an agent
    COMPLETED = "COMPLETED"    # call ended normally
    FAILED = "FAILED"          # no answer / busy / provider error / lost (see failure_reason)
    CANCELLED = "CANCELLED"    # we stopped it before it reached the borrower
    ABANDONED = "ABANDONED"    # borrower answered but no agent was free in time (the predictive risk)


# Rank orders the "happy path" so stale events (lower rank) can be detected.
CALL_RANK: Dict[CallState, int] = {
    CallState.QUEUED: 0,
    CallState.RESERVED: 1,
    CallState.INITIATED: 2,
    CallState.RINGING: 3,
    CallState.ANSWERED: 4,
    CallState.CONNECTED: 5,
    CallState.COMPLETED: 9,
    CallState.FAILED: 9,
    CallState.CANCELLED: 9,
    CallState.ABANDONED: 9,
}

TERMINAL_CALL_STATES = frozenset({CallState.COMPLETED, CallState.FAILED,
                                  CallState.CANCELLED, CallState.ABANDONED})

# Calls that may still produce an answered borrower who needs an agent.
PRE_CONNECT_CALL_STATES = frozenset({CallState.QUEUED, CallState.RESERVED, CallState.INITIATED,
                                     CallState.RINGING, CallState.ANSWERED})

CALL_TRANSITIONS: Dict[CallState, FrozenSet[CallState]] = {
    CallState.QUEUED: frozenset({CallState.RESERVED, CallState.CANCELLED}),
    # From RESERVED, provider events may arrive before the worker records INITIATED
    # (or the worker crashed right after dialing), so forward jumps are legal.
    CallState.RESERVED: frozenset({CallState.INITIATED, CallState.RINGING, CallState.ANSWERED,
                                   CallState.COMPLETED, CallState.FAILED, CallState.CANCELLED}),
    # Forward jumps (INITIATED -> ANSWERED) cover missing or late RINGING events.
    CallState.INITIATED: frozenset({CallState.RINGING, CallState.ANSWERED, CallState.COMPLETED,
                                    CallState.FAILED, CallState.CANCELLED}),
    CallState.RINGING: frozenset({CallState.ANSWERED, CallState.COMPLETED,
                                  CallState.FAILED, CallState.CANCELLED}),
    CallState.ANSWERED: frozenset({CallState.CONNECTED, CallState.ABANDONED,
                                   CallState.COMPLETED, CallState.FAILED}),
    CallState.CONNECTED: frozenset({CallState.COMPLETED, CallState.FAILED}),
    CallState.COMPLETED: frozenset(),
    CallState.FAILED: frozenset(),
    CallState.CANCELLED: frozenset(),
    CallState.ABANDONED: frozenset(),
}


def can_transition_call(src: CallState, dst: CallState) -> bool:
    return dst in CALL_TRANSITIONS[src]


def assert_call_transition(src: CallState, dst: CallState) -> None:
    if not can_transition_call(src, dst):
        raise InvalidTransition(f"call {src.value} -> {dst.value} is not allowed")


# --------------------------------------------------------------------------
# Borrower
# --------------------------------------------------------------------------
class BorrowerState(str, Enum):
    PENDING = "PENDING"        # eligible to be dialed once next_attempt_at has passed
    RESERVED = "RESERVED"      # exactly one call is in progress for this borrower
    DONE = "DONE"              # reached and spoken to
    EXHAUSTED = "EXHAUSTED"    # max attempts used


BORROWER_TRANSITIONS: Dict[BorrowerState, FrozenSet[BorrowerState]] = {
    BorrowerState.PENDING: frozenset({BorrowerState.RESERVED}),
    BorrowerState.RESERVED: frozenset({BorrowerState.PENDING, BorrowerState.DONE,
                                       BorrowerState.EXHAUSTED}),
    BorrowerState.DONE: frozenset(),
    BorrowerState.EXHAUSTED: frozenset(),
}


def assert_borrower_transition(src: BorrowerState, dst: BorrowerState) -> None:
    if dst not in BORROWER_TRANSITIONS[src]:
        raise InvalidTransition(f"borrower {src.value} -> {dst.value} is not allowed")


# --------------------------------------------------------------------------
# Provider events -> call states
# --------------------------------------------------------------------------
class ProviderEventType(str, Enum):
    RINGING = "RINGING"
    ANSWERED = "ANSWERED"
    COMPLETED = "COMPLETED"
    NO_ANSWER = "NO_ANSWER"
    BUSY = "BUSY"
    FAILED = "FAILED"


# event type -> (target call state, failure_reason)
EVENT_TARGET = {
    ProviderEventType.RINGING: (CallState.RINGING, None),
    ProviderEventType.ANSWERED: (CallState.ANSWERED, None),
    ProviderEventType.COMPLETED: (CallState.COMPLETED, None),
    ProviderEventType.NO_ANSWER: (CallState.FAILED, "no_answer"),
    ProviderEventType.BUSY: (CallState.FAILED, "busy"),
    ProviderEventType.FAILED: (CallState.FAILED, "provider_failed"),
}


class EventDecision(str, Enum):
    APPLY = "APPLIED"
    SAME_STATE = "NOOP_SAME_STATE"       # e.g. second ANSWERED while already ANSWERED
    STALE = "IGNORED_STALE"              # e.g. RINGING arriving after ANSWERED
    AFTER_TERMINAL = "IGNORED_TERMINAL"  # anything arriving after the call ended
    ILLEGAL = "REJECTED_ILLEGAL"         # cannot happen in a correct provider; logged


def classify_event_transition(current: CallState, target: CallState) -> EventDecision:
    """Decide what a provider event means for a call in `current` state.

    Pure function and never raises, which makes it easy to test exhaustively.
    """
    if current in TERMINAL_CALL_STATES:
        return EventDecision.AFTER_TERMINAL
    if current == target:
        return EventDecision.SAME_STATE
    if can_transition_call(current, target):
        return EventDecision.APPLY
    if target not in TERMINAL_CALL_STATES and CALL_RANK[target] < CALL_RANK[current]:
        return EventDecision.STALE
    return EventDecision.ILLEGAL
