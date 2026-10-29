"""Provider times out: the system degrades safely and recovers on its own."""
import unittest

from core.state_machines import AgentState, BorrowerState, CallState
from dialer.pacing import PREDICTIVE, PacingDecision
from dialer.safety_controller import ProviderCircuitBreaker, Verdict
from tests.helpers import Harness, assert_invariants, seed_history


class ProviderOutageTest(unittest.TestCase):
    def test_dial_during_outage_fails_cleanly(self):
        h = Harness(agents=3, borrowers=10)
        h.provider.start_outage(300)
        report = h.run(1)[0]

        self.assertEqual(report.allocation.outcomes["provider_failed"], 3)
        for call in h.calls():
            self.assertEqual(call.state, CallState.FAILED)
            self.assertEqual(call.failure_reason, "provider_timeout")
        for agent in h.db.list_agents():
            self.assertEqual(agent.state, AgentState.AVAILABLE, "agents must not stay reserved")
        for call in h.calls():
            b = h.db.get_borrower(call.borrower_id)
            self.assertEqual(b.state, BorrowerState.PENDING)
            self.assertEqual(b.attempts, 0, "provider fault is not the borrower's attempt")
            self.assertGreater(b.next_attempt_at, h.clock.now(), "backoff before retrying")

    def test_breaker_opens_rejects_probes_and_recovers(self):
        h = Harness(agents=5, borrowers=200)
        h.provider.start_outage(70)
        reports = h.run(3)
        self.assertEqual(h.engine.safety.breaker.state(), ProviderCircuitBreaker.OPEN)
        self.assertEqual(reports[-1].safety.verdict, Verdict.REJECTED)

        placed_attempts = len(h.calls())
        reports = h.run(25)                    # still OPEN: no dial attempts at all
        self.assertTrue(all(r.safety.verdict == Verdict.REJECTED for r in reports))
        self.assertEqual(len(h.calls()), placed_attempts)

        reports = h.run(10)                    # cool-down over, outage continues: one probe fails, re-open
        probes = [r for r in reports if r.allocation is not None]
        self.assertEqual(len(probes), 1)
        self.assertEqual(probes[0].safety.approved, 1)
        self.assertEqual(h.engine.safety.breaker.state(), ProviderCircuitBreaker.OPEN)

        h.run(60)                              # outage ends at t+70; next probe succeeds
        self.assertEqual(h.engine.safety.breaker.state(), ProviderCircuitBreaker.CLOSED)
        self.assertEqual(h.run(1)[0].safety.verdict, Verdict.APPROVED)
        h.run(30)
        self.assertTrue(any(c.connected_at for c in h.calls()), "dialing resumed and reached borrowers")
        assert_invariants(h.db)

    def test_in_flight_calls_wait_out_the_outage(self):
        h = Harness(agents=3, borrowers=3, talk_mean_s=900)
        h.run(1)                               # 3 calls dialed, ringing
        h.provider.start_outage(200)           # webhooks stop, lookups fail
        h.run(180)
        live = h.calls([CallState.INITIATED, CallState.RINGING])
        self.assertEqual(len(live), 3, "must not kill calls just because the provider is quiet")
        self.assertGreater(h.engine.recovery.stats["stuck_waiting_for_provider"], 0)

        h.run(150)                             # outage over: delayed events flush, calls finish
        self.assertEqual(h.calls([CallState.INITIATED, CallState.RINGING]), [])
        self.assertTrue(all(c.connected_at for c in h.calls()[:3]))
        assert_invariants(h.db)

    def test_hard_deadline_frees_agents_if_provider_never_returns(self):
        h = Harness(agents=2, borrowers=2)
        h.run(1)
        h.provider.start_outage(10_000)
        h.run(320)
        for call in h.calls():
            self.assertEqual(call.state, CallState.FAILED)
            self.assertEqual(call.failure_reason, "stuck_provider_unreachable")
        self.assertTrue(all(a.state == AgentState.AVAILABLE for a in h.db.list_agents()))

    def test_lost_completed_event_is_reaped(self):
        """The provider ends a CONNECTED call but its COMPLETED webhook is lost."""
        h = Harness(agents=1, borrowers=1, talk_mean_s=60)
        h.run(30)
        call = h.calls([CallState.CONNECTED])[0]
        h.provider.poll_events = lambda: []         # every later webhook is lost
        h.engine.safety.cfg.provider_silence_s = 10_000
        h.run(1700)
        self.assertEqual(h.db.get_call(call.id).state, CallState.CONNECTED, "too early to judge")
        h.run(120)
        ended = h.db.get_call(call.id)
        self.assertEqual(ended.state, CallState.COMPLETED)
        self.assertEqual(ended.failure_reason, "hangup_event_lost")
        self.assertNotEqual(h.db.get_agent("agent-000").state, AgentState.CONNECTED)
        assert_invariants(h.db)

    def test_silent_provider_trips_breaker(self):
        """Dials are accepted but webhooks never arrive. We are blind to those calls,
        so dialing must stop even though no request ever failed."""
        h = Harness(agents=3, borrowers=50)
        h.provider.poll_events = lambda: []
        reports = h.run(20)
        self.assertTrue(any("provider silent" in reason for r in reports for reason in r.safety.reasons))
        self.assertEqual(h.engine.safety.breaker.state(), ProviderCircuitBreaker.OPEN)
        self.assertEqual(reports[-1].safety.verdict, Verdict.REJECTED)

    def test_degraded_provider_forces_predictive_into_progressive(self):
        h = Harness(mode=PREDICTIVE, agents=10, borrowers=50)
        seed_history(h.db, h.clock.now(), answered=60, unanswered=60)
        for ok in [True, True, False] * 15 + [True] * 5:   # 30% failures, never 5 in a row: degraded, not open
            h.engine.safety.record_dial_result(ok)
        decision = h.engine.safety.evaluate(PacingDecision(PREDICTIVE, 25, "test"))
        self.assertEqual(decision.verdict, Verdict.FALLBACK_PROGRESSIVE)
        self.assertEqual(decision.effective_mode, "progressive")
        self.assertTrue(decision.permit.requires_agent)
        self.assertTrue(any("provider degraded" in r for r in decision.reasons))
        self.assertLessEqual(decision.approved, 10)

    def test_provider_b_ghost_call_on_timeout_is_hung_up(self):
        h = Harness(provider="B", agents=1, borrowers=1)
        h.provider.timeout_rate = 1.0
        h.provider.ghost_share = 1.0          # every timeout actually placed the call
        h.run(1)
        call = h.calls()[0]
        self.assertEqual(call.state, CallState.FAILED)
        self.assertEqual(call.failure_reason, "provider_timeout")
        status = h.provider.lookup(call.id)
        self.assertIsNotNone(status, "the ghost call exists at the provider")
        self.assertFalse(status.is_live, "...and we hung it up via our client reference")
        h.provider.timeout_rate = 0.0
        h.run(60)                              # late/stale events for it are ignored
        self.assertEqual(h.db.get_call(call.id).state, CallState.FAILED)


if __name__ == "__main__":
    unittest.main()
