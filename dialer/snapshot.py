"""Read-only view of the system that pacing and safety decide on.

The pacing engine and the safety controller each build their OWN snapshot
straight from the database. Safety never trusts numbers handed to it by
pacing.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.clock import Clock
from core.database import Database
from core.state_machines import AgentState, PRE_CONNECT_CALL_STATES
from dialer.estimators import clamp


@dataclass
class Snapshot:
    now: float = 0.0
    agents_by_state: Dict[AgentState, int] = field(default_factory=dict)
    wrap_remaining_s: List[float] = field(default_factory=list)
    # Calls with no agent attached (predictive), split by whether they are still ringing.
    unbound_ringing: int = 0           # QUEUED/RESERVED/INITIATED/RINGING without an agent
    unbound_answered_waiting: int = 0  # ANSWERED, waiting for an agent right now
    bound_in_flight: int = 0           # pre-connect calls that already own an agent
    # Recent outcomes (long window)
    window_s: float = 300.0
    answered: int = 0
    unanswered: int = 0
    abandoned: int = 0
    # Recent outcomes (short window), used to detect a sudden answer-rate jump
    short_window_s: float = 60.0
    short_answered: int = 0
    short_unanswered: int = 0
    avg_talk_s: Optional[float] = None
    avg_time_to_answer_s: Optional[float] = None
    wrap_up_s: float = 10.0

    # ---------------------------------------------------------------- derived
    def agents(self, state: AgentState) -> int:
        return self.agents_by_state.get(state, 0)

    @property
    def idle_agents(self) -> int:
        return self.agents(AgentState.AVAILABLE)

    @property
    def connected_agents(self) -> int:
        return self.agents(AgentState.CONNECTED)

    @property
    def logged_in(self) -> int:
        return sum(n for s, n in self.agents_by_state.items() if s != AgentState.OFFLINE)

    @property
    def unbound_in_flight(self) -> int:
        return self.unbound_ringing + self.unbound_answered_waiting

    @property
    def outcomes(self) -> int:
        return self.answered + self.unanswered

    @property
    def short_outcomes(self) -> int:
        return self.short_answered + self.short_unanswered

    @property
    def abandon_rate(self) -> float:
        return self.abandoned / self.answered if self.answered else 0.0

    def agents_free_within(self, seconds: float) -> int:
        return sum(1 for r in self.wrap_remaining_s if r <= seconds)


@dataclass
class FreeAgentForecast:
    """How many agents we expect to be free when calls dialed NOW get answered."""
    lookahead_s: float
    idle: int
    wrap_finishing: int
    talk_finishing: float
    answered_waiting: int

    @property
    def gross(self) -> float:
        return self.idle + self.wrap_finishing + self.talk_finishing

    @property
    def net(self) -> float:
        return max(0.0, self.gross - self.answered_waiting)


def forecast_free_agents(snap: Snapshot, lookahead_s: float, avg_talk_s: float) -> FreeAgentForecast:
    """idle now
     + agents in wrap-up that finish within the lookahead
     + connected agents x (share of a talk that ends early enough to also finish wrap-up)
     - answered callers already waiting (they take the first free agents)."""
    talk_share = clamp((lookahead_s - snap.wrap_up_s) / max(avg_talk_s, 1.0), 0.0, 1.0)
    return FreeAgentForecast(
        lookahead_s=lookahead_s,
        idle=snap.idle_agents,
        wrap_finishing=snap.agents_free_within(lookahead_s),
        talk_finishing=snap.connected_agents * talk_share,
        answered_waiting=snap.unbound_answered_waiting,
    )


class SnapshotReader:
    def __init__(self, db: Database, clock: Clock, wrap_up_s: float,
                 window_s: float = 300.0, short_window_s: float = 60.0):
        self.db = db
        self.clock = clock
        self.wrap_up_s = wrap_up_s
        self.window_s = window_s
        self.short_window_s = short_window_s

    def read(self) -> Snapshot:
        now = self.clock.now()
        lw = now - self.window_s
        sw = now - self.short_window_s
        pre = [s.value for s in PRE_CONNECT_CALL_STATES]
        marks = ",".join("?" for _ in pre)

        wrap = [r["wrap_until"] - now for r in self.db.query(
            "SELECT wrap_until FROM agents WHERE state = ? AND wrap_until IS NOT NULL", (AgentState.WRAP_UP,))]

        inflight = self.db.query_one(
            f"""SELECT
                  COALESCE(SUM(CASE WHEN agent_id IS NULL AND state != 'ANSWERED' THEN 1 ELSE 0 END), 0) AS ur,
                  COALESCE(SUM(CASE WHEN agent_id IS NULL AND state  = 'ANSWERED' THEN 1 ELSE 0 END), 0) AS uw,
                  COALESCE(SUM(CASE WHEN agent_id IS NOT NULL THEN 1 ELSE 0 END), 0) AS b
                FROM calls WHERE state IN ({marks})""", pre)

        stats = self.db.query_one(
            """SELECT
                 COALESCE(SUM(CASE WHEN answered_at >= :lw THEN 1 ELSE 0 END), 0) AS ans,
                 COALESCE(SUM(CASE WHEN answered_at IS NULL AND ended_at >= :lw
                                    AND failure_reason IN ('no_answer','busy') THEN 1 ELSE 0 END), 0) AS unans,
                 COALESCE(SUM(CASE WHEN state = 'ABANDONED' AND answered_at >= :lw THEN 1 ELSE 0 END), 0) AS aband,
                 COALESCE(SUM(CASE WHEN answered_at >= :sw THEN 1 ELSE 0 END), 0) AS s_ans,
                 COALESCE(SUM(CASE WHEN answered_at IS NULL AND ended_at >= :sw
                                    AND failure_reason IN ('no_answer','busy') THEN 1 ELSE 0 END), 0) AS s_unans,
                 AVG(CASE WHEN state = 'COMPLETED' AND connected_at IS NOT NULL AND ended_at >= :lw
                          THEN ended_at - connected_at END) AS talk,
                 AVG(CASE WHEN answered_at >= :lw AND initiated_at IS NOT NULL
                          THEN answered_at - initiated_at END) AS tta
               FROM calls WHERE answered_at >= :lw OR ended_at >= :lw""",
            {"lw": lw, "sw": sw})

        return Snapshot(
            now=now,
            agents_by_state=self.db.count_agents_by_state(),
            wrap_remaining_s=wrap,
            unbound_ringing=inflight["ur"],
            unbound_answered_waiting=inflight["uw"],
            bound_in_flight=inflight["b"],
            window_s=self.window_s,
            answered=stats["ans"],
            unanswered=stats["unans"],
            abandoned=stats["aband"],
            short_window_s=self.short_window_s,
            short_answered=stats["s_ans"],
            short_unanswered=stats["s_unans"],
            avg_talk_s=stats["talk"],
            avg_time_to_answer_s=stats["tta"],
            wrap_up_s=self.wrap_up_s,
        )
