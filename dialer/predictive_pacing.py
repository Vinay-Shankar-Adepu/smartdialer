"""Predictive pacing: dial ahead of agents becoming free.

The question it answers
-----------------------
"If I dial N lines now, about L seconds from now some of them will be
answered. How many agents will be free then, and how many lines does that
justify?"

Step by step (every number lands in PacingDecision.explanation)
---------------------------------------------------------------
1. Answer rate p. Recent answered / (answered + no-answer + busy), smoothed
   with a prior (10 imaginary calls at 30%), then we take the UPPER bound of
   the 80% Wilson interval. Overestimating p makes us dial LESS, which is
   the safe way to be wrong.
2. Lookahead L = average dial-to-answer time (how far ahead we dial).
3. Free agents expected at time L:
       idle now
     + agents in wrap-up that finish within L
     + connected agents x max(0, (L - wrap)/avg_talk)   (calls that end early enough)
     - answered callers already waiting for an agent
4. Lines wanted N_total: the largest N with
       N*p + k*sqrt(N*p*(1-p)) <= free agents x aggression
   i.e. expected answers PLUS k standard deviations of binomial noise fit
   inside the free agents. k=1 means overflow in roughly 1 in 6 batches, and an
   overflow only means one caller waits up to 2 s for the next free agent.
5. Aggression is 1.0 while the recent abandon rate is under half the target
   (3%), then scales linearly down to 0.5 as it approaches the target. Above
   the target, the safety controller takes over and forces progressive.
6. Dial N_total - lines already ringing without an agent (never negative).

Example: "why dial 17 instead of 10?" (see tests/test_predictive_logic.py)
   free = 2 idle + 6 wrap-up finishing within L=10s + 0.0 talking - 0 waiting = 8.0 agents
   at a 33% answer rate: p=0.35 (upper bound) -> N_total = 17.3 -> dial 17
   at a 60% answer rate: p=0.62 (upper bound) -> N_total = 10.x -> dial 10
   Same 8 free agents. A lower answer rate needs more lines to produce 8 answers.
"""
import math
from dataclasses import dataclass

from dialer.estimators import clamp, lines_for_free_agents, smoothed_rate, wilson_upper
from dialer.pacing import PREDICTIVE, PacingDecision
from dialer.snapshot import Snapshot, forecast_free_agents


@dataclass
class PredictiveConfig:
    prior_answer_rate: float = 0.3
    prior_weight: float = 10.0
    confidence_z: float = 1.28          # 80% two-sided interval -> 90% one-sided upper bound
    k_sigma: float = 1.0                # binomial safety stock
    target_abandon_rate: float = 0.03
    min_aggression: float = 0.5
    default_time_to_answer_s: float = 10.0
    default_talk_s: float = 120.0


class PredictivePacing:
    mode = PREDICTIVE

    def __init__(self, config: PredictiveConfig = None):
        self.cfg = config or PredictiveConfig()

    def answer_rate(self, snap: Snapshot):
        cfg = self.cfg
        p_point = smoothed_rate(snap.answered, snap.outcomes, cfg.prior_answer_rate, cfg.prior_weight)
        p_upper = wilson_upper(snap.answered + cfg.prior_answer_rate * cfg.prior_weight,
                               snap.outcomes + cfg.prior_weight, cfg.confidence_z)
        return p_point, max(p_upper, 0.01)

    def aggression(self, abandon_rate: float) -> float:
        target = self.cfg.target_abandon_rate
        if abandon_rate <= target / 2:
            return 1.0
        fraction = clamp((abandon_rate - target / 2) / (target / 2), 0.0, 1.0)
        return 1.0 - fraction * (1.0 - self.cfg.min_aggression)

    def recommend(self, snap: Snapshot) -> PacingDecision:
        cfg = self.cfg
        p_point, p = self.answer_rate(snap)
        lookahead = snap.avg_time_to_answer_s or cfg.default_time_to_answer_s
        talk = snap.avg_talk_s or cfg.default_talk_s
        free = forecast_free_agents(snap, lookahead, talk)
        aggression = self.aggression(snap.abandon_rate)
        lines_total = lines_for_free_agents(free.net * aggression, p, cfg.k_sigma)
        recommended = max(0, math.floor(lines_total - snap.unbound_ringing))

        explanation = {
            "answer_rate_point": round(p_point, 3),
            "answer_rate_used_upper": round(p, 3),
            "sample_outcomes": snap.outcomes,
            "lookahead_s": round(lookahead, 1),
            "avg_talk_s": round(talk, 1),
            "idle_agents": free.idle,
            "wrap_finishing": free.wrap_finishing,
            "talk_finishing": round(free.talk_finishing, 2),
            "answered_waiting": free.answered_waiting,
            "free_agents_net": round(free.net, 2),
            "abandon_rate": round(snap.abandon_rate, 4),
            "aggression": round(aggression, 2),
            "k_sigma": cfg.k_sigma,
            "lines_wanted_total": round(lines_total, 2),
            "unbound_ringing": snap.unbound_ringing,
            "recommended": recommended,
        }
        summary = (
            f"Predictive: dial {recommended}. p={p:.2f} (upper bound; estimate {p_point:.2f} from "
            f"{snap.outcomes} calls), "
            f"L={lookahead:.1f}s; free agents = {free.idle} idle + {free.wrap_finishing} wrap-up "
            f"+ {free.talk_finishing:.1f} talking - {free.answered_waiting} waiting = {free.net:.1f}; "
            f"x aggression {aggression:.2f} -> {lines_total:.1f} lines (k={cfg.k_sigma:g}) "
            f"- {snap.unbound_ringing} already ringing = {recommended}"
        )
        return PacingDecision(PREDICTIVE, recommended, summary, explanation)
