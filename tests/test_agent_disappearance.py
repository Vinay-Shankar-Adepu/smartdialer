"""100 agents, 40 vanish within 5 seconds: the system notices and slows down."""
import unittest

from core.state_machines import AgentState, CallState
from dialer.pacing import PREDICTIVE
from dialer.safety_controller import Verdict
from tests.helpers import Harness, assert_invariants, seed_history


class AgentDisappearanceTest(unittest.TestCase):
    def test_forty_of_hundred_agents_vanish_in_five_seconds(self):
        h = Harness(mode=PREDICTIVE, agents=100, borrowers=6000, answer_rate=0.3, talk_mean_s=90)
        seed_history(h.db, h.clock.now(), answered=90, unanswered=210, talk_s=90)
        warm = h.run(90)
        self.assertEqual(warm[-1].safety.effective_mode, "predictive", "warmed up into predictive")
        dials_before = sum(r.allocation.dialed for r in warm[-10:] if r.allocation)

        victims = [a.id for a in h.db.list_agents() if a.state != AgentState.OFFLINE][:40]
        drop_reports = []
        for second in range(5):                 # 8 agents per second for 5 seconds
            for agent_id in victims[second * 8:(second + 1) * 8]:
                h.engine.agents.force_offline(agent_id)
            drop_reports += h.run(1)

        after = drop_reports[-1].safety
        self.assertEqual(after.checks["logged_in"], 60, "safety recounted agents from the DB")
        fallbacks = [r.safety for r in drop_reports if r.safety.verdict == Verdict.FALLBACK_PROGRESSIVE]
        self.assertTrue(fallbacks, "agent drop must force progressive")
        self.assertTrue(any("agent drop" in reason for f in fallbacks for reason in f.reasons))
        first = next(i for i, r in enumerate(drop_reports) if any("agent drop" in x for x in r.safety.reasons))
        self.assertLessEqual(first, 1, "detected by the 2nd tick (16 of 100 gone = over the 15% threshold)")
        for f in fallbacks:                       # progressive cap: never more calls than free agents
            self.assertLessEqual(f.approved, max(0, f.checks["idle_agents"] - f.checks["unbound_in_flight"]))

        for agent_id in victims:
            agent = h.db.get_agent(agent_id)
            self.assertEqual(agent.state, AgentState.OFFLINE)
            self.assertIsNone(agent.reserved_for_call_id)
        live = h.calls([s for s in CallState if s not in (CallState.COMPLETED, CallState.FAILED,
                                                          CallState.CANCELLED, CallState.ABANDONED)])
        self.assertFalse([c for c in live if c.agent_id in victims], "no live call bound to a vanished agent")

        after_reports = h.run(10)
        dials_after = sum(r.allocation.dialed for r in after_reports if r.allocation)
        self.assertLess(dials_after, dials_before, "pacing slowed down")

        recovered = h.run(150)                    # cool-down expires -> predictive again, on 60 agents
        self.assertEqual(recovered[-1].safety.effective_mode, "predictive")
        assert_invariants(h.db)

    def test_ringing_call_of_vanished_agent_goes_to_another_agent(self):
        h = Harness(agents=1, borrowers=1)
        h.run(1)
        call = h.calls()[0]
        self.assertEqual(call.agent_id, "agent-000")
        h.engine.agents.force_offline("agent-000")
        self.assertIsNone(h.db.get_call(call.id).agent_id, "call unbound from the vanished agent")
        h.engine.agents.add_agent("agent-new")
        h.run(20)
        connected = h.db.get_call(call.id)
        self.assertEqual(connected.state, CallState.CONNECTED)
        self.assertEqual(connected.agent_id, "agent-new")

    def test_answered_call_with_no_agent_left_is_abandoned_and_hung_up(self):
        h = Harness(agents=1, borrowers=1)
        h.run(1)
        call = h.calls()[0]
        h.engine.agents.force_offline("agent-000")
        h.run(25)
        ended = h.db.get_call(call.id)
        self.assertEqual(ended.state, CallState.ABANDONED)
        self.assertFalse(h.provider.lookup(call.id).is_live, "borrower is not left hanging on a silent line")

    def test_agent_vanishing_mid_conversation_ends_call_and_frees_nothing_twice(self):
        h = Harness(agents=1, borrowers=2, talk_mean_s=300)
        h.run(20)
        call = h.calls([CallState.CONNECTED])[0]
        h.engine.agents.force_offline("agent-000")
        ended = h.db.get_call(call.id)
        self.assertEqual(ended.state, CallState.COMPLETED)
        self.assertEqual(ended.failure_reason, "agent_lost")
        self.assertEqual(h.db.get_agent("agent-000").state, AgentState.OFFLINE)
        h.run(5)
        assert_invariants(h.db)


if __name__ == "__main__":
    unittest.main()
