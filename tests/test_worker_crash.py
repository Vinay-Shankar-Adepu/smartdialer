"""Worker dies mid-allocation: Agent RESERVED -> Borrower RESERVED -> dialed -> crash.

A crash is simulated by a hook that raises SimulatedCrash (a BaseException),
so no cleanup code runs, the same as `kill -9`. A second worker's recovery
sweep must detect the expired lease and either adopt the live call or
release everything.
"""
import unittest

from core.state_machines import AgentState, BorrowerState, CallState
from dialer.call_allocator import SimulatedCrash
from tests.helpers import Harness, assert_invariants

LEASE = 10.0


def crash_at(point):
    def hook(where, call_id):
        if where == point:
            raise SimulatedCrash(f"worker killed at {where} ({call_id})")
    return hook


class WorkerCrashTest(unittest.TestCase):
    def crashed_harness(self, point, **kw):
        h = Harness(agents=1, borrowers=3, crash_hook=crash_at(point), **kw)
        with self.assertRaises(SimulatedCrash):
            h.engine.tick()
        return h

    def only_call(self, h):
        calls = h.calls()
        self.assertEqual(len(calls), 1)
        return calls[0]

    # ------------------------------------------------------------------ the spec scenario
    def test_crash_after_dial_call_is_adopted_and_completes(self):
        h = self.crashed_harness("after_dial")
        call = self.only_call(h)
        agent = h.db.get_agent("agent-000")
        borrower = h.db.get_borrower(call.borrower_id)
        # Exactly the dangerous state: everything reserved, provider dialed, our DB doesn't know.
        self.assertEqual(call.state, CallState.RESERVED)
        self.assertIsNone(call.provider_call_id)
        self.assertEqual(agent.state, AgentState.RESERVED)
        self.assertEqual(agent.reserved_for_call_id, call.id)
        self.assertEqual(borrower.state, BorrowerState.RESERVED)
        self.assertIsNotNone(h.provider.lookup(call.id), "provider really placed the call")

        survivor = h.spawn_worker("worker-2")
        self.assertEqual(survivor.recovery.sweep()["lease_adopted"], 0, "lease not expired yet: hands off")
        self.assertEqual(h.db.get_call(call.id).state, CallState.RESERVED)

        h.clock.advance(LEASE + 1)
        stats = survivor.recovery.sweep()
        self.assertEqual(stats["lease_adopted"], 1)
        adopted = h.db.get_call(call.id)
        self.assertEqual(adopted.state, CallState.INITIATED)
        self.assertIsNotNone(adopted.provider_call_id)
        self.assertEqual(h.db.get_agent("agent-000").state, AgentState.DIALING)
        assert_invariants(h.db)

        h.run(200, engine=survivor)          # events drive the adopted call to completion
        final = h.db.get_call(call.id)
        self.assertEqual(final.state, CallState.COMPLETED)
        self.assertIsNotNone(final.connected_at, "the borrower was connected to the reserved agent")
        self.assertEqual(h.db.get_borrower(call.borrower_id).state, BorrowerState.DONE)

    def test_crash_after_dial_events_alone_self_heal(self):
        """Even without the sweep, provider events (keyed by OUR call id) move the call on."""
        h = self.crashed_harness("after_dial")
        call = self.only_call(h)
        survivor = h.spawn_worker("worker-2")
        h.clock.advance(2.0)                     # RINGING is due
        survivor.ingest_events()
        healed = h.db.get_call(call.id)
        self.assertEqual(healed.state, CallState.RINGING)
        self.assertIsNone(healed.lease_expires_at, "provider confirmed it: no longer worker-owned")
        self.assertEqual(h.db.get_agent("agent-000").state, AgentState.DIALING)
        h.clock.advance(LEASE + 1)
        self.assertEqual(survivor.recovery.sweep()["lease_cancelled"], 0, "a live call must not be cancelled")
        assert_invariants(h.db)

    # ------------------------------------------------------------------ crash before the provider
    def test_crash_before_dial_releases_agent_and_borrower(self):
        h = self.crashed_harness("before_dial")
        call = self.only_call(h)
        self.assertIsNone(h.provider.lookup(call.id), "nothing was dialed")

        survivor = h.spawn_worker("worker-2")
        h.clock.advance(LEASE + 1)
        self.assertEqual(survivor.recovery.sweep()["lease_cancelled"], 1)

        cancelled = h.db.get_call(call.id)
        self.assertEqual(cancelled.state, CallState.CANCELLED)
        self.assertEqual(cancelled.failure_reason, "worker_crash")
        self.assertEqual(h.db.get_agent("agent-000").state, AgentState.AVAILABLE)
        borrower = h.db.get_borrower(call.borrower_id)
        self.assertEqual(borrower.state, BorrowerState.PENDING)
        self.assertEqual(borrower.attempts, 0, "our crash must not burn the borrower's attempt")
        assert_invariants(h.db)

        h.run(3, engine=survivor)                # system keeps dialing normally
        self.assertTrue(any(c.state != CallState.CANCELLED for c in h.calls()))

    def test_crash_after_reserving_agent_before_binding(self):
        """Agent reserved but the call row doesn't point at it yet: still released."""
        h = self.crashed_harness("after_reserve_agent")
        call = self.only_call(h)
        self.assertEqual(call.state, CallState.QUEUED)
        self.assertIsNone(call.agent_id)
        self.assertEqual(h.db.get_agent("agent-000").reserved_for_call_id, call.id)

        h.clock.advance(LEASE + 1)
        h.spawn_worker("worker-2").recovery.sweep()
        self.assertEqual(h.db.get_call(call.id).state, CallState.CANCELLED)
        self.assertEqual(h.db.get_agent("agent-000").state, AgentState.AVAILABLE)
        assert_invariants(h.db)

    def test_crash_after_claiming_borrower_before_call_row(self):
        h = self.crashed_harness("after_claim_borrower")
        self.assertEqual(h.calls(), [])
        claimed = [b for b in (h.db.get_borrower(f"b-{i:05d}") for i in range(3))
                   if b.state == BorrowerState.RESERVED]
        self.assertEqual(len(claimed), 1)

        h.clock.advance(LEASE + 1)
        stats = h.spawn_worker("worker-2").recovery.sweep()
        self.assertEqual(stats["orphan_borrower_released"], 1)
        self.assertEqual(h.db.get_borrower(claimed[0].id).state, BorrowerState.PENDING)
        assert_invariants(h.db)

    # ------------------------------------------------------------------ slow, not dead
    def test_slow_worker_cannot_dial_after_reaper_cancelled_its_call(self):
        """A paused worker (GC pause, network stall) wakes up after its lease was
        reaped. The safety permit check refuses the dial, so no ghost call."""
        holder = {}

        def stall(where, call_id):
            if where == "before_dial":
                holder["h"].clock.advance(LEASE + 1)
                holder["h"].spawn_worker("reaper").recovery.sweep()

        h = Harness(agents=1, borrowers=3, crash_hook=stall)
        holder["h"] = h
        report = h.engine.tick()
        self.assertEqual(report.allocation.outcomes["rejected"], 1)
        self.assertEqual(h.provider.stats["placed"], 0, "provider must never be reached")
        self.assertEqual(h.engine.safety.stats["violations"], 1)
        self.assertEqual(h.db.get_agent("agent-000").state, AgentState.AVAILABLE)
        assert_invariants(h.db)

    # ------------------------------------------------------------------ ANSWERED but not CONNECTED
    def test_crash_between_answered_and_connected(self):
        h = Harness(agents=1, borrowers=3)
        h.run(1)
        h.clock.advance(16)                      # answer happens within 4-15 s
        h.engine.ingest_events()
        call = h.calls([CallState.ANSWERED])[0]

        original = h.engine.locks.transition_call
        def crash_on_connect(call_id, to_state, **kw):
            if to_state == CallState.CONNECTED:
                raise SimulatedCrash("died while bridging")
            return original(call_id, to_state, **kw)
        h.engine.locks.transition_call = crash_on_connect
        with self.assertRaises(SimulatedCrash):
            h.engine.lifecycle.try_connect(call.id)
        h.engine.locks.transition_call = original

        # Transaction rolled back: still ANSWERED, agent still exclusively held.
        self.assertEqual(h.db.get_call(call.id).state, CallState.ANSWERED)
        self.assertEqual(h.db.get_agent("agent-000").reserved_for_call_id, call.id)
        self.assertEqual(h.db.get_agent("agent-000").state, AgentState.DIALING)
        assert_invariants(h.db)

        self.assertEqual(h.engine.lifecycle.bridge_answered_calls(h.engine.gateway.hangup)["connected"], 1)
        self.assertEqual(h.db.get_call(call.id).state, CallState.CONNECTED)
        self.assertEqual(h.db.get_agent("agent-000").state, AgentState.CONNECTED)
        assert_invariants(h.db)


if __name__ == "__main__":
    unittest.main()
