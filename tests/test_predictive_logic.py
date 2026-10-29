"""Pacing math is deterministic, explainable and moves in the right direction."""
import math
import unittest

from core.clock import SimClock
from core.database import Database
from core.state_machines import AgentState, CallState
from dialer.estimators import lines_for_free_agents, wilson_upper
from dialer.pacing import PREDICTIVE, PROGRESSIVE, PacingDecision
from dialer.predictive_pacing import PredictiveConfig, PredictivePacing
from dialer.progressive_pacing import ProgressivePacing
from dialer.safety_controller import SafetyConfig, SafetyController, Verdict
from dialer.snapshot import Snapshot
from tests.helpers import Harness, assert_invariants, seed_history


def snap(idle=0, wrap=(), connected=0, answered=300, unanswered=700, ringing=0, waiting=0,
         abandoned=0, tta=10.0, talk=120.0, short=(0, 0), wrap_up_s=10.0):
    return Snapshot(
        now=0.0,
        agents_by_state={AgentState.AVAILABLE: idle, AgentState.WRAP_UP: len(wrap),
                         AgentState.CONNECTED: connected},
        wrap_remaining_s=list(wrap), unbound_ringing=ringing, unbound_answered_waiting=waiting,
        answered=answered, unanswered=unanswered, abandoned=abandoned,
        short_answered=short[0], short_unanswered=short[1],
        avg_time_to_answer_s=tta, avg_talk_s=talk, wrap_up_s=wrap_up_s)


class FixedReader:
    def __init__(self, s):
        self.s = s

    def read(self):
        return self.s


def safety_for(s, **cfg):
    return SafetyController(Database(":memory:"), SimClock(0), FixedReader(s), SafetyConfig(**cfg))


class PredictiveFormulaTest(unittest.TestCase):
    def setUp(self):
        self.pacing = PredictivePacing()

    def test_matches_hand_calculation(self):
        s = snap(idle=4, wrap=(2.0, 5.0, 30.0), connected=10, answered=300, unanswered=700, ringing=3)
        d = self.pacing.recommend(s)
        e = d.explanation
        # p: (300 + 0.3*10) / (1000 + 10) smoothed, then Wilson upper bound at z=1.28
        p_upper = wilson_upper(303, 1010, 1.28)
        self.assertAlmostEqual(e["answer_rate_used_upper"], round(p_upper, 3))
        self.assertAlmostEqual(e["answer_rate_point"], round(303 / 1010, 3))
        # free = 4 idle + 2 wrap-ups finishing within L=10s + 10 talking x (10-10)/120 = 6
        self.assertEqual(e["free_agents_net"], 6.0)
        # N*p + 1*sqrt(N*p*(1-p)) = 6  ->  N; then minus 3 lines already ringing
        b = math.sqrt(p_upper * (1 - p_upper))
        n = ((-b + math.sqrt(b * b + 4 * p_upper * 6)) / (2 * p_upper)) ** 2
        self.assertEqual(d.recommended, math.floor(n - 3))
        self.assertIn(f"dial {d.recommended}", d.summary)

    def test_k_zero_reduces_to_free_over_p(self):
        self.assertAlmostEqual(lines_for_free_agents(10, 0.25, 0.0), 40.0)
        self.assertLess(lines_for_free_agents(10, 0.25, 1.0), 40.0, "safety stock dials fewer")

    def test_why_seventeen_not_ten(self):
        """Interview question: same agents, but 'why 17?' The answer rate."""
        common = dict(idle=2, wrap=(3, 4, 6, 7, 8, 9), connected=30)
        at_33 = self.pacing.recommend(snap(answered=330, unanswered=670, **common))
        at_60 = self.pacing.recommend(snap(answered=600, unanswered=400, **common))
        self.assertEqual(at_33.explanation["free_agents_net"], at_60.explanation["free_agents_net"])
        self.assertEqual((at_33.recommended, at_60.recommended), (17, 10))
        self.assertIn("8.0", at_33.summary)                    # 8 free agents
        self.assertIn("dial 17", at_33.summary)

    def test_lower_answer_rate_dials_more(self):
        lo = self.pacing.recommend(snap(idle=10, answered=150, unanswered=850))
        hi = self.pacing.recommend(snap(idle=10, answered=600, unanswered=400))
        self.assertGreater(lo.recommended, hi.recommended)

    def test_lines_already_ringing_are_subtracted(self):
        base = self.pacing.recommend(snap(idle=10)).recommended
        self.assertEqual(self.pacing.recommend(snap(idle=10, ringing=5)).recommended, base - 5)
        self.assertEqual(self.pacing.recommend(snap(idle=10, ringing=500)).recommended, 0, "never negative")

    def test_no_free_agents_means_no_dials(self):
        self.assertEqual(self.pacing.recommend(snap(idle=0, wrap=(50, 60), connected=40, talk=600)).recommended, 0)

    def test_answered_callers_waiting_consume_free_agents(self):
        free = self.pacing.recommend(snap(idle=5)).recommended
        self.assertLess(self.pacing.recommend(snap(idle=5, waiting=3)).recommended, free)
        self.assertEqual(self.pacing.recommend(snap(idle=5, waiting=5)).recommended, 0)

    def test_agents_finishing_talk_counted_when_lookahead_allows(self):
        short_l = self.pacing.recommend(snap(connected=20, tta=10, talk=100))   # (10-10)/100 = 0
        long_l = self.pacing.recommend(snap(connected=20, tta=30, talk=100))    # (30-10)/100 = 0.2 -> 4 agents
        self.assertEqual(short_l.explanation["talk_finishing"], 0.0)
        self.assertEqual(long_l.explanation["talk_finishing"], 4.0)
        self.assertGreater(long_l.recommended, short_l.recommended)

    def test_aggression_backs_off_near_abandon_target(self):
        a = self.pacing.aggression
        self.assertEqual(a(0.0), 1.0)
        self.assertEqual(a(0.015), 1.0)
        self.assertAlmostEqual(a(0.0225), 0.75)
        self.assertAlmostEqual(a(0.03), 0.5)
        self.assertAlmostEqual(a(0.20), 0.5)

    def test_upper_bound_is_conservative_and_tightens_with_data(self):
        self.assertGreater(wilson_upper(30, 100, 1.28), 0.30)
        self.assertLess(wilson_upper(3000, 10000, 1.28), wilson_upper(30, 100, 1.28))

    def test_cold_start_uses_prior(self):
        d = self.pacing.recommend(snap(idle=10, answered=0, unanswered=0))
        self.assertAlmostEqual(d.explanation["answer_rate_point"], 0.3)
        self.assertGreater(d.explanation["answer_rate_used_upper"], 0.3)

    def test_explanation_lists_every_input(self):
        e = self.pacing.recommend(snap(idle=3)).explanation
        for key in ("answer_rate_point", "answer_rate_used_upper", "lookahead_s", "idle_agents", "wrap_finishing",
                    "talk_finishing", "answered_waiting", "free_agents_net", "aggression", "lines_wanted_total",
                    "unbound_ringing", "recommended"):
            self.assertIn(key, e)

    def test_progressive_is_exactly_idle_agents(self):
        self.assertEqual(ProgressivePacing().recommend(snap(idle=7, wrap=(1, 2))).recommended, 7)


class SafetyOnPacingTest(unittest.TestCase):
    def test_cold_start_falls_back_to_progressive(self):
        s = snap(idle=10, answered=5, unanswered=5)
        d = safety_for(s).evaluate(PacingDecision(PREDICTIVE, 30, "t"))
        self.assertEqual(d.verdict, Verdict.FALLBACK_PROGRESSIVE)
        self.assertEqual(d.approved, 10)
        self.assertTrue(d.permit.requires_agent)
        self.assertIn("low confidence", d.reasons[0])

    def test_healthy_recommendation_is_approved(self):
        s = snap(idle=10, ringing=20)                           # steady state: lines already out
        rec = PredictivePacing().recommend(s)
        d = safety_for(s).evaluate(rec)
        self.assertEqual(d.verdict, Verdict.APPROVED)
        self.assertEqual(d.approved, rec.recommended)
        self.assertEqual(d.effective_mode, PREDICTIVE)

    def test_answer_rate_surge_is_capped(self):
        """5-min window says 20%, but the last minute says 60%: pacing (on the
        5-min estimate) overdials, and safety cuts it back using the 1-min rate."""
        s = snap(idle=10, answered=200, unanswered=800, short=(60, 40))
        rec = PredictivePacing().recommend(s)
        d = safety_for(s).evaluate(rec)
        self.assertEqual(d.verdict, Verdict.REDUCED)
        self.assertLessEqual(d.approved, math.floor(10 * 1.15 / 0.6))
        self.assertAlmostEqual(d.checks["p_safety"], 0.6)

    def test_abandon_rate_above_target_forces_progressive(self):
        s = snap(idle=6, answered=300, unanswered=700, abandoned=12)   # 4%
        d = safety_for(s).evaluate(PacingDecision(PREDICTIVE, 20, "t"))
        self.assertEqual(d.verdict, Verdict.FALLBACK_PROGRESSIVE)
        self.assertEqual(d.approved, 6)
        self.assertIn("abandon rate", d.reasons[0])

    def test_progressive_never_exceeds_idle_minus_unbound(self):
        s = snap(idle=5, ringing=2, waiting=1)
        d = safety_for(s).evaluate(PacingDecision(PROGRESSIVE, 5, "t"))
        self.assertEqual(d.verdict, Verdict.REDUCED)
        self.assertEqual(d.approved, 2)

    def test_blast_radius_cap(self):
        s = snap(idle=4, answered=20, unanswered=980)          # p ~2%: free/p would be ~100+ lines
        rec = PredictivePacing().recommend(s)
        d = safety_for(s).evaluate(rec)
        self.assertLessEqual(d.approved, 4 * 5)

    def test_tick_cap(self):
        s = snap(idle=200)
        d = safety_for(s, max_dials_per_tick=50).evaluate(PacingDecision(PROGRESSIVE, 200, "t"))
        self.assertEqual(d.approved, 50)

    def test_ramp_limits_new_lines_per_tick(self):
        s = snap(idle=20, answered=100, unanswered=900)          # p ~10%: pacing wants ~150 lines
        d = safety_for(s).evaluate(PredictivePacing().recommend(s))
        self.assertEqual(d.approved, 10, "0.5 new lines per free agent per tick")
        self.assertIn("ramp limit", d.reasons[0])

    def test_reduction_reason_names_the_limit_that_applied(self):
        s = snap(idle=4, answered=20, unanswered=980, ringing=19)   # 5 x 4 = 20 lines, 19 ringing
        d = safety_for(s).evaluate(PredictivePacing().recommend(s))
        self.assertEqual(d.approved, 1)
        self.assertIn("5 lines per free agent", d.reasons[0])
        surge = snap(idle=10, answered=200, unanswered=800, short=(60, 40), ringing=16)
        d = safety_for(surge).evaluate(PredictivePacing().recommend(surge))
        self.assertIn("p_safety 0.60", d.reasons[0])

    def test_answer_burst_forces_progressive(self):
        s = snap(idle=0, connected=10, waiting=3)
        d = safety_for(s).evaluate(PacingDecision(PREDICTIVE, 5, "t"))
        self.assertEqual(d.verdict, Verdict.FALLBACK_PROGRESSIVE)
        self.assertEqual(d.approved, 0)
        self.assertIn("answer burst", d.reasons[0])


class AnswerRateShiftTest(unittest.TestCase):
    """Interview question: the model says 70%, reality is 10% (and the reverse)."""

    def _run(self, history_rate, real_rate, seconds=300):
        h = Harness(mode=PREDICTIVE, agents=20, borrowers=5000, answer_rate=real_rate, talk_mean_s=60)
        seed_history(h.db, h.clock.now(), answered=int(300 * history_rate),
                     unanswered=300 - int(300 * history_rate))
        reports = h.run(seconds)
        return h, reports, [c for c in h.calls() if not c.id.startswith("hist")]

    def test_collapse_70_to_10_is_safe(self):
        """Fewer answers than predicted means idle agents, never abandoned callers.
        The estimate then catches up and dialing increases."""
        h, reports, calls = self._run(0.7, 0.1)
        self.assertEqual(sum(c.state == CallState.ABANDONED for c in calls), 0)
        self.assertLess(reports[-1].decision.explanation["answer_rate_point"], 0.4)
        late = sum(r.allocation.dialed for r in reports[-100:] if r.allocation)
        early = sum(r.allocation.dialed for r in reports[10:110] if r.allocation)
        self.assertGreater(late, early, "dials more once it learns the rate is low")

    def test_jump_10_to_70_is_bounded_and_stops_fast(self):
        """The dangerous direction. Lines already ringing can't be recalled, so some abandons are
        possible; the ramp and the answer-burst trigger bound them and stop overdialing quickly."""
        h, reports, calls = self._run(0.1, 0.7)
        first_fallback = next(i for i, r in enumerate(reports)
                              if r.safety.verdict == Verdict.FALLBACK_PROGRESSIVE)
        self.assertLess(first_fallback, 15, "detected within seconds")
        predictive_lines = sum(r.allocation.dialed for r in reports
                               if r.allocation and r.safety.effective_mode == PREDICTIVE)
        self.assertLessEqual(predictive_lines, 20 * 5, "never beyond the blast-radius cap")
        abandoned = sum(c.state == CallState.ABANDONED for c in calls)
        self.assertLessEqual(abandoned, predictive_lines - 20, "at most the lines beyond free agents")
        self.assertLess(abandoned, 45)
        assert_invariants(h.db)


if __name__ == "__main__":
    unittest.main()
