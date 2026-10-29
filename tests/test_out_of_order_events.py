"""Events arriving out of order (or for unknown calls) leave the system consistent."""
import unittest

from core.models import ProviderEvent
from core.state_machines import (
    AgentState,
    CALL_RANK,
    CallState,
    EVENT_TARGET,
    EventDecision,
    ProviderEventType,
    TERMINAL_CALL_STATES,
    classify_event_transition,
)
from dialer.event_processor import UNKNOWN_CALL
from tests.helpers import Harness, assert_invariants


class OutOfOrderEventsTest(unittest.TestCase):
    def dialed(self, **kw):
        h = Harness(agents=1, borrowers=3, **kw)
        h.engine.tick()
        return h, h.calls()[0]

    def test_answered_before_ringing(self):
        h, call = self.dialed()
        self.assertEqual(h.engine.events.process(h.event(call, ProviderEventType.ANSWERED)), "APPLIED")
        self.assertEqual(h.engine.events.process(h.event(call, ProviderEventType.RINGING)), "IGNORED_STALE")
        self.assertEqual(h.db.get_call(call.id).state, CallState.ANSWERED)
        h.engine.lifecycle.bridge_answered_calls(h.engine.gateway.hangup)
        self.assertEqual(h.db.get_call(call.id).state, CallState.CONNECTED)
        self.assertEqual(h.engine.events.process(h.event(call, ProviderEventType.RINGING, "late-2")),
                         "IGNORED_STALE")
        assert_invariants(h.db)

    def test_completed_before_ringing_and_answered(self):
        h, call = self.dialed()
        self.assertEqual(h.engine.events.process(h.event(call, ProviderEventType.COMPLETED)), "APPLIED")
        ended = h.db.get_call(call.id)
        self.assertEqual(ended.state, CallState.COMPLETED)
        self.assertEqual(h.db.get_agent("agent-000").state, AgentState.AVAILABLE,
                         "never connected, so straight back to AVAILABLE (no wrap-up)")
        for etype in (ProviderEventType.RINGING, ProviderEventType.ANSWERED):
            self.assertEqual(h.engine.events.process(h.event(call, etype)), "IGNORED_TERMINAL")
        self.assertEqual(h.db.get_call(call.id).version, ended.version)
        self.assertEqual(h.db.get_agent("agent-000").state, AgentState.AVAILABLE)
        assert_invariants(h.db)

    def test_assignment_sequences_verbatim(self):
        """The two sequences from the assignment, each answered event with its own id."""
        h, call = self.dialed()
        process, E = h.engine.events.process, ProviderEventType
        self.assertEqual(process(h.event(call, E.ANSWERED, "a1")), "APPLIED")
        h.engine.lifecycle.bridge_answered_calls(h.engine.gateway.hangup)
        self.assertEqual(process(h.event(call, E.ANSWERED, "a2")), "IGNORED_STALE")
        self.assertEqual(process(h.event(call, E.ANSWERED, "a3")), "IGNORED_STALE")
        self.assertEqual(process(h.event(call, E.COMPLETED, "c1")), "APPLIED")
        self.assertEqual(h.db.get_call(call.id).state, CallState.COMPLETED)
        self.assertEqual(h.db.get_agent("agent-000").state, AgentState.WRAP_UP, "connected once, wrapped up once")
        assert_invariants(h.db)

        h.clock.advance(20)
        h.engine.tick()                                        # agent free again; next borrower dialed
        call2 = next(c for c in h.calls() if c.id != call.id)
        self.assertEqual(process(h.event(call2, E.COMPLETED, "x1")), "APPLIED")
        self.assertEqual(process(h.event(call2, E.ANSWERED, "x2")), "IGNORED_TERMINAL")
        self.assertEqual(process(h.event(call2, E.RINGING, "x3")), "IGNORED_TERMINAL")
        self.assertEqual(h.db.get_call(call2.id).state, CallState.COMPLETED)
        self.assertEqual(h.db.get_agent("agent-000").state, AgentState.AVAILABLE)
        assert_invariants(h.db)

    def test_event_arrives_before_worker_records_initiated(self):
        """Fast provider: RINGING is processed while the call is still RESERVED."""
        holder = {}

        def race(where, call_id):
            if where == "after_dial":
                holder["h"].clock.advance(2)
                holder["h"].engine.ingest_events()      # webhook handler wins the race

        h = Harness(agents=1, borrowers=1, crash_hook=race)
        holder["h"] = h
        h.engine.tick()
        call = h.calls()[0]
        self.assertEqual(call.state, CallState.RINGING, "worker must not regress RINGING to INITIATED")
        self.assertIsNotNone(call.provider_call_id)
        self.assertIsNone(call.lease_expires_at)
        self.assertEqual(h.db.get_agent("agent-000").state, AgentState.DIALING)
        assert_invariants(h.db)

    def test_event_for_unknown_call_is_recorded_not_fatal(self):
        h = Harness(agents=1, borrowers=1)
        ev = ProviderEvent("x-1", "call-that-never-existed", "pc-x", ProviderEventType.ANSWERED, 0.0)
        self.assertEqual(h.engine.events.process(ev), UNKNOWN_CALL)
        row = h.db.query_one("SELECT outcome FROM provider_events WHERE event_id = 'x-1'")
        self.assertEqual(row["outcome"], UNKNOWN_CALL)

    def test_classification_is_total_and_never_moves_backwards(self):
        for current in CallState:
            for target, _ in EVENT_TARGET.values():
                decision = classify_event_transition(current, target)
                if decision == EventDecision.APPLY:
                    self.assertNotIn(current, TERMINAL_CALL_STATES)
                    self.assertGreater(CALL_RANK[target], CALL_RANK[current],
                                       f"{current} -> {target} would move backwards")
                if current in TERMINAL_CALL_STATES:
                    self.assertEqual(decision, EventDecision.AFTER_TERMINAL)

    def test_provider_b_chaos_keeps_invariants(self):
        """15 minutes of Provider B: jitter, duplicates, re-ids, stale events,
        timeouts and ghost calls, with invariants checked after every tick."""
        h = Harness(provider="B", agents=8, borrowers=120, answer_rate=0.5, talk_mean_s=40,
                    provider_failure_rate=None, seed=11)
        h.run(900)
        stats = h.engine.events.stats
        self.assertGreater(stats["DUPLICATE_EVENT_ID"], 0)
        self.assertGreater(stats["IGNORED_STALE"] + stats["IGNORED_TERMINAL"] + stats["NOOP_SAME_STATE"], 0)
        self.assertGreater(h.provider.stats["ghost_timeouts"] + h.provider.stats["clean_timeouts"], 0)

        for agent in h.db.list_agents():                     # stop dialing, drain everything
            h.engine.agents.logout(agent.id)
        h.run(400)
        live = [c for c in h.calls() if c.state not in TERMINAL_CALL_STATES]
        self.assertEqual(live, [], "every call eventually reaches a terminal state")
        self.assertTrue(all(a.state == AgentState.OFFLINE for a in h.db.list_agents()))
        self.assertEqual(h.engine.safety.stats["violations"], 0)


if __name__ == "__main__":
    unittest.main()
