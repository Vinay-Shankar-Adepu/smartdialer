"""Test harness and system-wide invariants."""
import os
import shutil
import tempfile
from typing import Callable, List, Optional

from core.clock import SimClock
from core.database import Database
from core.models import Borrower, Call, ProviderEvent
from core.state_machines import (
    AGENT_BUSY_STATES,
    AgentState,
    BorrowerState,
    CallState,
    ProviderEventType,
    TERMINAL_CALL_STATES,
)
from dialer.call_lifecycle import LifecycleConfig
from dialer.engine import DialerEngine, EngineConfig
from dialer.safety_controller import SafetyConfig
from providers.mock_base import BorrowerBehavior
from providers.provider_factory import create_provider

START = 1000.0


class Harness:
    def __init__(self, mode: str = "progressive", provider: str = "A", agents: int = 3,
                 borrowers: int = 30, answer_rate: float = 1.0, talk_mean_s: float = 60.0,
                 seed: int = 7, db_path: str = ":memory:", crash_hook: Optional[Callable] = None,
                 safety: Optional[SafetyConfig] = None, provider_failure_rate: Optional[float] = 0.0):
        self.clock = SimClock(START)
        self.db = Database(db_path)
        self.behavior = BorrowerBehavior(answer_rate, talk_mean_s=talk_mean_s)
        self.provider = create_provider(provider, self.clock, self.behavior, seed=seed)
        if provider_failure_rate is not None:
            self.provider.failure_rate = provider_failure_rate
        self.config = EngineConfig(lifecycle=LifecycleConfig(wrap_up_s=10.0),
                                   safety=safety or SafetyConfig())
        self.mode = mode
        self.engine = DialerEngine(self.db, self.clock, self.provider, mode, self.config, crash_hook)
        for i in range(agents):
            self.engine.agents.add_agent(f"agent-{i:03d}")
        self.db.insert_borrowers(Borrower(id=f"b-{i:05d}", name=f"B{i}", phone=f"+91{i:010d}")
                                 for i in range(borrowers))

    def spawn_worker(self, worker_id: str, crash_hook: Optional[Callable] = None) -> DialerEngine:
        """A second dialer process sharing the same database and provider."""
        cfg = EngineConfig(worker_id=worker_id, lifecycle=self.config.lifecycle, safety=self.config.safety)
        return DialerEngine(self.db, self.clock, self.provider, self.mode, cfg, crash_hook)

    def run(self, seconds: int, engine: Optional[DialerEngine] = None, check: bool = True) -> list:
        engine = engine or self.engine
        reports = []
        for _ in range(seconds):
            reports.append(engine.tick())
            if check:
                assert_invariants(self.db)
            self.clock.advance(1.0)
        return reports

    def calls(self, states=None) -> List[Call]:
        return self.db.find_calls(states or list(CallState))

    def event(self, call: Call, etype: ProviderEventType, event_id: Optional[str] = None) -> ProviderEvent:
        return ProviderEvent(event_id or f"manual:{call.id}:{etype.value}", call.id,
                             call.provider_call_id or f"pc-{call.id}", etype, self.clock.now())


class TempDbDir:
    """Temp directory for file-backed databases (real multi-connection concurrency)."""

    def __enter__(self) -> str:
        self.path = tempfile.mkdtemp(prefix="smartdialer-")
        return os.path.join(self.path, "dialer.db")

    def __exit__(self, *exc):
        shutil.rmtree(self.path, ignore_errors=True)


def seed_history(db: Database, now: float, answered: int, unanswered: int, abandoned: int = 0,
                 talk_s: float = 60.0, tta_s: float = 8.0, age_s: float = 100.0) -> None:
    """Insert finished calls so the estimators have recent data."""
    t0 = now - age_s
    for i in range(answered):
        db.insert_call(Call(id=f"hist-a-{i}", borrower_id="hist", state=CallState.COMPLETED, mode="predictive",
                            created_at=t0, initiated_at=t0, answered_at=t0 + tta_s, connected_at=t0 + tta_s,
                            ended_at=min(now - 1, t0 + tta_s + talk_s)))
    for i in range(abandoned):
        db.insert_call(Call(id=f"hist-x-{i}", borrower_id="hist", state=CallState.ABANDONED, mode="predictive",
                            created_at=t0, initiated_at=t0, answered_at=t0 + tta_s, ended_at=t0 + tta_s + 2,
                            failure_reason="no_agent_available"))
    for i in range(unanswered):
        db.insert_call(Call(id=f"hist-n-{i}", borrower_id="hist", state=CallState.FAILED, mode="predictive",
                            created_at=t0, initiated_at=t0, ended_at=t0 + 25, failure_reason="no_answer"))


def check_invariants(db: Database) -> List[str]:
    """Cross-table consistency rules that must hold between any two ticks."""
    problems: List[str] = []
    terminal = tuple(s.value for s in TERMINAL_CALL_STATES)
    marks = ",".join("?" for _ in terminal)

    for row in db.query("SELECT reserved_for_call_id, COUNT(*) AS n FROM agents "
                        "WHERE reserved_for_call_id IS NOT NULL GROUP BY 1 HAVING n > 1"):
        problems.append(f"{row['n']} agents reserved for the same call {row['reserved_for_call_id']}")
    for row in db.query(f"SELECT agent_id, COUNT(*) AS n FROM calls WHERE agent_id IS NOT NULL "
                        f"AND state NOT IN ({marks}) GROUP BY 1 HAVING n > 1", terminal):
        problems.append(f"agent {row['agent_id']} bound to {row['n']} live calls")
    for row in db.query(f"SELECT borrower_id, COUNT(*) AS n FROM calls WHERE borrower_id != 'hist' "
                        f"AND state NOT IN ({marks}) GROUP BY 1 HAVING n > 1", terminal):
        problems.append(f"borrower {row['borrower_id']} has {row['n']} live calls")

    for agent in db.list_agents():
        if agent.state in AGENT_BUSY_STATES:
            call = db.get_call(agent.reserved_for_call_id) if agent.reserved_for_call_id else None
            if call is None or call.state in TERMINAL_CALL_STATES:
                problems.append(f"agent {agent.id} {agent.state.value} for dead call {agent.reserved_for_call_id}")
            elif call.agent_id != agent.id:
                problems.append(f"agent {agent.id} reserved for {call.id} but call points at {call.agent_id}")
            if agent.state == AgentState.CONNECTED and call is not None and call.state != CallState.CONNECTED:
                problems.append(f"agent {agent.id} CONNECTED but call {call.id} is {call.state.value}")
        elif agent.reserved_for_call_id is not None:
            problems.append(f"agent {agent.id} is {agent.state.value} but holds reservation")

    for call in db.find_calls([s for s in CallState if s not in TERMINAL_CALL_STATES]):
        if call.agent_id:
            agent = db.get_agent(call.agent_id)
            if agent is None or agent.reserved_for_call_id != call.id:
                problems.append(f"call {call.id} bound to agent {call.agent_id} who does not hold it")
        if call.state == CallState.CONNECTED and not call.agent_id:
            problems.append(f"call {call.id} CONNECTED without agent")
        if call.borrower_id != "hist":
            b = db.get_borrower(call.borrower_id)
            if b is None or b.state != BorrowerState.RESERVED or b.current_call_id != call.id:
                problems.append(f"live call {call.id} but borrower {call.borrower_id} not reserved for it")
    return problems


def assert_invariants(db: Database) -> None:
    problems = check_invariants(db)
    if problems:
        raise AssertionError("invariants violated:\n  " + "\n  ".join(problems[:20]))
