"""The same event arriving twice must never cause a second state transition."""
import threading
import unittest

from core.state_machines import AgentState, BorrowerState, CallState, ProviderEventType
from dialer.event_processor import DUPLICATE_EVENT
from tests.helpers import Harness, TempDbDir, assert_invariants


class DuplicateEventsTest(unittest.TestCase):
    def dialed(self, **kw):
        h = Harness(agents=1, borrowers=2, **kw)
        h.engine.tick()                   # one call INITIATED; we feed events by hand (no polling)
        return h, h.calls()[0]

    def test_same_event_id_twice_is_a_noop(self):
        h, call = self.dialed()
        ev = h.event(call, ProviderEventType.ANSWERED)
        self.assertEqual(h.engine.events.process(ev), "APPLIED")
        after_first = h.db.get_call(call.id)

        h.clock.advance(1)
        self.assertEqual(h.engine.events.process(ev), DUPLICATE_EVENT)
        after_second = h.db.get_call(call.id)
        self.assertEqual(after_second.version, after_first.version, "no write happened")
        self.assertEqual(after_second.answered_at, after_first.answered_at)

    def test_same_transition_with_new_event_id_is_a_noop(self):
        """A buggy retry with a fresh id beats id-dedup, so the state machine catches it."""
        h, call = self.dialed()
        h.engine.events.process(h.event(call, ProviderEventType.ANSWERED, "e-1"))
        version = h.db.get_call(call.id).version
        self.assertEqual(h.engine.events.process(h.event(call, ProviderEventType.ANSWERED, "e-2")),
                         "NOOP_SAME_STATE")
        self.assertEqual(h.db.get_call(call.id).version, version)

    def test_duplicate_answered_after_connect_does_not_reconnect(self):
        h, call = self.dialed()
        h.engine.events.process(h.event(call, ProviderEventType.ANSWERED, "e-1"))
        h.engine.lifecycle.bridge_answered_calls(h.engine.gateway.hangup)
        self.assertEqual(h.db.get_call(call.id).state, CallState.CONNECTED)
        self.assertEqual(h.engine.events.process(h.event(call, ProviderEventType.ANSWERED, "e-2")),
                         "IGNORED_STALE")
        self.assertEqual(h.db.get_call(call.id).state, CallState.CONNECTED)

    def test_duplicate_completed_releases_agent_once(self):
        h, call = self.dialed()
        h.engine.events.process(h.event(call, ProviderEventType.ANSWERED))
        h.engine.lifecycle.bridge_answered_calls(h.engine.gateway.hangup)
        h.engine.events.process(h.event(call, ProviderEventType.COMPLETED, "c-1"))
        agent = h.db.get_agent("agent-000")
        borrower = h.db.get_borrower(call.borrower_id)
        self.assertEqual(agent.state, AgentState.WRAP_UP)
        self.assertEqual(borrower.state, BorrowerState.DONE)

        for event_id in ("c-1", "c-2"):      # same id, then a new id
            h.engine.events.process(h.event(call, ProviderEventType.COMPLETED, event_id))
        self.assertEqual(h.db.get_agent("agent-000").version, agent.version, "agent untouched")
        self.assertEqual(h.db.get_borrower(call.borrower_id).version, borrower.version, "borrower untouched")
        assert_invariants(h.db)

    def test_duplicate_no_answer_spends_one_attempt(self):
        h, call = self.dialed(answer_rate=0.0)
        for event_id in ("n-1", "n-1", "n-2"):
            h.engine.events.process(h.event(call, ProviderEventType.NO_ANSWER, event_id))
        b = h.db.get_borrower(call.borrower_id)
        self.assertEqual(b.attempts, 1)
        self.assertEqual(b.state, BorrowerState.PENDING)
        self.assertEqual(h.db.get_agent("agent-000").state, AgentState.AVAILABLE)

    def test_concurrent_duplicate_delivery_applies_once(self):
        """Two webhook handlers receive the same event at the same instant."""
        with TempDbDir() as path:
            h, call = self.dialed(db_path=path)
            ev = h.event(call, ProviderEventType.ANSWERED)
            outcomes = []
            barrier = threading.Barrier(4)

            def deliver():
                barrier.wait()
                outcomes.append(h.engine.events.process(ev))

            threads = [threading.Thread(target=deliver) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)
            self.assertEqual(sorted(outcomes), sorted(["APPLIED"] + [DUPLICATE_EVENT] * 3))
            self.assertEqual(h.db.get_call(call.id).state, CallState.ANSWERED)
            h.db.close()


if __name__ == "__main__":
    unittest.main()
