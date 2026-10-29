"""Runs a scenario against the real dialer code on a simulated clock.

Nothing is mocked here except time, the telecom network and the borrowers:
the engine, safety controller, allocator, event processor and recovery are
the production classes, running on a real SQLite database.
"""
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.clock import SimClock
from core.database import Database
from core.models import Borrower
from core.state_machines import AgentState, CallState
from dialer.engine import DialerEngine, EngineConfig
from dialer.call_lifecycle import LifecycleConfig
from providers.mock_base import BorrowerBehavior
from providers.provider_factory import create_provider
from simulation.scenarios import Scenario


@dataclass
class Bucket:
    start_s: float
    talk_agent_s: float = 0.0
    logged_in_agent_s: float = 0.0
    dialed: int = 0
    connected: int = 0
    abandoned: int = 0
    verdicts: Counter = field(default_factory=Counter)


class Simulator:
    def __init__(self, mode: str, scenario: Scenario, seed: int = 42, dt: float = 1.0,
                 bucket_s: float = 300.0, db_path: str = ":memory:"):
        self.mode = mode
        self.scenario = scenario
        self.dt = dt
        self.bucket_s = bucket_s
        self.rng = random.Random(seed)
        self.clock = SimClock(0.0)
        self.db = Database(db_path)
        self.behavior = BorrowerBehavior(scenario.answer_rate, scenario.talk_mean_s,
                                         ring_to_answer_s=scenario.ring_to_answer_s,
                                         no_answer_after_s=scenario.no_answer_after_s)
        # The simulator keeps a provider handle ONLY to inject outages (it plays "the network").
        self.provider = create_provider(scenario.provider, self.clock, self.behavior, seed=seed + 1)
        config = EngineConfig(lifecycle=LifecycleConfig(wrap_up_s=scenario.wrap_up_s))
        self.engine = DialerEngine(self.db, self.clock, self.provider, mode, config)

        for i in range(scenario.num_agents):
            self.engine.agents.add_agent(f"agent-{i:03d}")
        self.db.insert_borrowers(Borrower(id=f"b-{i:06d}", name=f"Borrower {i}", phone=f"+9190000{i:05d}")
                                 for i in range(scenario.borrowers))

        self._pending_actions = sorted(scenario.timeline, key=lambda e: e.at_s)
        self._scheduled_drops: List[tuple] = []
        self._dropped: List[str] = []
        self.buckets: List[Bucket] = []
        self.verdicts: Counter = Counter()
        self.sample_decisions: List[str] = []
        self.timeline_log: List[str] = []
        self.peak_dials_tick = 0
        self.duration = 0.0

    # ---------------------------------------------------------------- timeline
    def _apply_timeline(self, now: float) -> None:
        while self._pending_actions and self._pending_actions[0].at_s <= now:
            ev = self._pending_actions.pop(0)
            if ev.action == "drop_agents":
                online = [a.id for a in self.db.list_agents() if a.state != AgentState.OFFLINE]
                victims = self.rng.sample(online, min(ev.params["count"], len(online)))
                step = ev.params.get("over_s", 0) / max(len(victims), 1)
                self._scheduled_drops += [(now + i * step, v) for i, v in enumerate(victims)]
                self.timeline_log.append(f"t={now:.0f}s  {len(victims)} agents start disappearing "
                                         f"over {ev.params.get('over_s', 0)}s")
            elif ev.action == "restore_agents":
                for agent_id in self._dropped:
                    self.engine.agents.login(agent_id)
                self.timeline_log.append(f"t={now:.0f}s  {len(self._dropped)} agents log back in")
                self._dropped = []
            elif ev.action == "set_answer_rate":
                self.timeline_log.append(f"t={now:.0f}s  answer rate {self.behavior.answer_rate:.0%} "
                                         f"-> {ev.params['rate']:.0%}")
                self.behavior.answer_rate = ev.params["rate"]
            elif ev.action == "set_talk_time":
                self.timeline_log.append(f"t={now:.0f}s  avg talk time {self.behavior.talk_mean_s:.0f}s "
                                         f"-> {ev.params['talk_s']:.0f}s")
                self.behavior.talk_mean_s = ev.params["talk_s"]
            elif ev.action == "provider_outage":
                self.provider.start_outage(ev.params["duration_s"])
                self.timeline_log.append(f"t={now:.0f}s  provider outage for {ev.params['duration_s']}s")
            else:
                raise ValueError(ev.action)
        due = [d for d in self._scheduled_drops if d[0] <= now]
        self._scheduled_drops = [d for d in self._scheduled_drops if d[0] > now]
        for _, agent_id in due:
            self.engine.agents.force_offline(agent_id)
            self._dropped.append(agent_id)

    # ---------------------------------------------------------------- run
    def run(self, duration: float) -> None:
        self.duration = duration
        steps = int(duration / self.dt)
        last_verdict = None
        for _ in range(steps):
            now = self.clock.now()
            self._apply_timeline(now)
            report = self.engine.tick()

            idx = int(now // self.bucket_s)
            while len(self.buckets) <= idx:
                self.buckets.append(Bucket(start_s=len(self.buckets) * self.bucket_s))
            bucket = self.buckets[idx]
            counts = self.db.count_agents_by_state()
            bucket.talk_agent_s += counts[AgentState.CONNECTED] * self.dt
            bucket.logged_in_agent_s += (sum(counts.values()) - counts[AgentState.OFFLINE]) * self.dt
            dialed = report.allocation.dialed if report.allocation else 0
            bucket.dialed += dialed
            bucket.connected += report.bridge["connected"]
            bucket.abandoned += report.bridge["abandoned"]
            verdict = report.safety.verdict.value
            bucket.verdicts[verdict] += 1
            self.verdicts[verdict] += 1
            self.peak_dials_tick = max(self.peak_dials_tick, dialed)

            # Keep a few explanations: the first predictive ones, and every verdict change.
            if verdict != last_verdict or (report.decision.recommended > 0 and len(self.sample_decisions) < 3):
                if len(self.sample_decisions) < 12:
                    line = f"t={now:5.0f}s  [{verdict}{' -> ' + str(report.safety.approved)}] " \
                           f"{report.decision.summary}"
                    if report.safety.reasons:
                        line += f"\n           safety: {'; '.join(report.safety.reasons)}"
                    self.sample_decisions.append(line)
            last_verdict = verdict
            self.clock.advance(self.dt)

    # ---------------------------------------------------------------- results
    def results(self) -> Dict:
        calls = self.db.count_calls_by_state()
        row = self.db.query_one(
            "SELECT COUNT(*) AS dialed, "
            "SUM(CASE WHEN answered_at IS NOT NULL THEN 1 ELSE 0 END) AS answered, "
            "SUM(CASE WHEN connected_at IS NOT NULL THEN 1 ELSE 0 END) AS connected, "
            "SUM(CASE WHEN state = 'ABANDONED' THEN 1 ELSE 0 END) AS abandoned, "
            "SUM(CASE WHEN failure_reason IN ('provider_timeout','provider_error') THEN 1 ELSE 0 END) AS perr "
            "FROM calls WHERE initiated_at IS NOT NULL OR failure_reason IN ('provider_timeout','provider_error')")
        talk = sum(b.talk_agent_s for b in self.buckets)
        logged = sum(b.logged_in_agent_s for b in self.buckets)
        minutes = self.duration / 60 or 1
        answered = row["answered"] or 0
        abandoned = row["abandoned"] or 0
        return {
            "scenario": self.scenario.key,
            "mode": self.mode,
            "utilization": talk / logged if logged else 0.0,
            "dialed": row["dialed"] or 0,
            "answered": answered,
            "connected": row["connected"] or 0,
            "abandoned": abandoned,
            "abandon_rate": abandoned / answered if answered else 0.0,
            "provider_errors": row["perr"] or 0,
            "dials_per_min": (row["dialed"] or 0) / minutes,
            "connects_per_min": (row["connected"] or 0) / minutes,
            "verdicts": dict(self.verdicts),
            "breaker_trips": self.engine.safety.breaker.trips,
            "events": dict(self.engine.events.stats),
            "recovery": dict(self.engine.recovery.stats),
            "safety_violations": self.engine.safety.stats["violations"],
            "calls_by_state": {k.value: v for k, v in calls.items() if v},
            "provider_stats": dict(self.provider.stats),
        }

    def print_results(self, explain: bool = True) -> None:
        r = self.results()
        sc = self.scenario
        print("=" * 100)
        print(f"Scenario {sc.key}: {sc.name}  |  mode={self.mode}  |  {self.duration:.0f}s simulated")
        print(f"  {sc.description}")
        print("-" * 100)
        print(f"  Agent utilization (talk time / logged-in time): {r['utilization']:6.1%}")
        print(f"  Dialed {r['dialed']:6d}  ({r['dials_per_min']:.1f}/min)   Answered {r['answered']:5d}   "
              f"Connected {r['connected']:5d} ({r['connects_per_min']:.1f}/min)")
        print(f"  Abandoned {r['abandoned']:4d}  -> abandon rate {r['abandon_rate']:.2%} (target <= 3%)   "
              f"Provider errors {r['provider_errors']}   Breaker trips {r['breaker_trips']}")
        print(f"  Safety verdicts per tick: {r['verdicts']}   Permit violations: {r['safety_violations']}")
        if r["events"]:
            print(f"  Event outcomes: {r['events']}")
        if r["recovery"]:
            print(f"  Recovery actions: {r['recovery']}")
        if self.timeline_log:
            print("  Timeline:")
            for line in self.timeline_log:
                print(f"    {line}")
        print("  Per-bucket timeline:")
        print(f"    {'window':>13} {'util':>6} {'dialed':>7} {'conn':>5} {'aband':>5}  dominant verdict")
        for b in self.buckets:
            util = b.talk_agent_s / b.logged_in_agent_s if b.logged_in_agent_s else 0
            dominant = b.verdicts.most_common(1)[0][0] if b.verdicts else "-"
            print(f"    {b.start_s:5.0f}-{b.start_s + self.bucket_s:5.0f}s {util:6.1%} {b.dialed:7d} "
                  f"{b.connected:5d} {b.abandoned:5d}  {dominant}")
        if explain and self.sample_decisions:
            print("  Sample pacing decisions (why N calls):")
            for line in self.sample_decisions:
                print(f"    {line}")
        print()
