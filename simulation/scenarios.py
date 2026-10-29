"""Scenarios A-D: answer rates, talk times, agents, providers and timed disruptions."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class TimelineEvent:
    at_s: float
    action: str                      # drop_agents | restore_agents | set_answer_rate | set_talk_time | provider_outage
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Scenario:
    key: str
    name: str
    description: str
    num_agents: int
    answer_rate: float
    talk_mean_s: float
    wrap_up_s: float
    provider: str
    borrowers: int = 20_000
    ring_to_answer_s: Tuple[float, float] = (4.0, 15.0)
    no_answer_after_s: float = 25.0
    timeline: List[TimelineEvent] = field(default_factory=list)


# A-D follow the assignment's table (answer rate / avg talk time). Every scenario has provider
# latency and failures: Provider A = 0.1s latency, 5% failed dials; Provider B = 0.5s latency + 0-3s
# jitter, 10% timeouts, duplicate / out-of-order / stale events.
SCENARIOS: Dict[str, Scenario] = {
    "A": Scenario(
        key="A", name="20% answer, 120s talk",
        description="50 agents, Provider A, 20% answer rate, 120s avg talk, 10s wrap-up.",
        num_agents=50, answer_rate=0.20, talk_mean_s=120, wrap_up_s=10, provider="A",
    ),
    "B": Scenario(
        key="B", name="50% answer, 90s talk",
        description="50 agents, Provider A, 50% answer rate, 90s avg talk, 10s wrap-up.",
        num_agents=50, answer_rate=0.50, talk_mean_s=90, wrap_up_s=10, provider="A",
    ),
    "C": Scenario(
        key="C", name="70% answer, 180s talk, chaotic provider + outage",
        description="50 agents on Provider B (slow, timeouts, ghost calls, duplicate and out-of-order "
                    "events), 70% answer rate, 180s avg talk; full 120s provider outage at t=1200s.",
        num_agents=50, answer_rate=0.70, talk_mean_s=180, wrap_up_s=10, provider="B",
        timeline=[TimelineEvent(1200, "provider_outage", {"duration_s": 120})],
    ),
    "D": Scenario(
        key="D", name="Changing answer rate and talk time + agent drop",
        description="100 agents, Provider A. Starts 50% / 90s; at t=900s 70% / 180s; at t=1500s 40 of "
                    "100 agents vanish within 5s (back at t=2100s); at t=2400s the answer rate "
                    "collapses to 10% with 60s talk.",
        num_agents=100, answer_rate=0.50, talk_mean_s=90, wrap_up_s=10, provider="A",
        timeline=[
            TimelineEvent(900, "set_answer_rate", {"rate": 0.70}),
            TimelineEvent(900, "set_talk_time", {"talk_s": 180}),
            TimelineEvent(1500, "drop_agents", {"count": 40, "over_s": 5}),
            TimelineEvent(2100, "restore_agents", {}),
            TimelineEvent(2400, "set_answer_rate", {"rate": 0.10}),
            TimelineEvent(2400, "set_talk_time", {"talk_s": 60}),
        ],
    ),
}
