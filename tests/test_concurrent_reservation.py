"""Two (or many) workers race for the same agent: exactly one wins.

These tests use a FILE database so every thread has its own SQLite
connection. That is real concurrency through SQLite's locking, not a single
shared connection behind a Python lock.
"""
import threading
import unittest

from core.clock import SimClock
from core.database import Database
from core.lock_manager import LockManager
from core.models import Agent, Borrower
from core.state_machines import AgentState, BorrowerState
from dialer.pacing import PREDICTIVE, PROGRESSIVE, PacingDecision
from dialer.safety_controller import SafetyViolation
from tests.helpers import Harness, TempDbDir, assert_invariants, seed_history


class ConcurrentReservationTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TempDbDir()
        self.path = self._tmp.__enter__()
        self.clock = SimClock(1000.0)
        self.db = Database(self.path)
        self.locks = LockManager(self.db, self.clock)

    def tearDown(self):
        self.db.close()
        self._tmp.__exit__(None, None, None)

    def _add_agents(self, n, prefix="agent"):
        for i in range(n):
            self.db.insert_agent(Agent(id=f"{prefix}-{i}", name=f"{prefix}-{i}", state=AgentState.AVAILABLE))

    def _race(self, n_threads, target):
        barrier = threading.Barrier(n_threads)
        errors = []

        def run(i):
            try:
                barrier.wait()
                target(i)
            except Exception as exc:  # surface thread failures in the main thread
                errors.append(exc)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        self.assertEqual(errors, [])

    def test_two_workers_same_agent_exactly_one_wins(self):
        for round_ in range(25):
            agent_id = f"agent42-r{round_}"
            self.db.insert_agent(Agent(id=agent_id, name="Agent 42", state=AgentState.AVAILABLE))
            results = {}
            self._race(2, lambda i: results.__setitem__(i, self.locks.reserve_agent(agent_id, f"call-{i}")))

            self.assertEqual(sorted(results.values()), [False, True], f"round {round_}: {results}")
            winner = next(i for i, won in results.items() if won)
            agent = self.db.get_agent(agent_id)
            self.assertEqual(agent.state, AgentState.RESERVED)
            self.assertEqual(agent.reserved_for_call_id, f"call-{winner}")
            self.assertEqual(agent.version, 1)

    def test_stale_version_cannot_overwrite(self):
        """Deterministic interleaving: A reads, B reserves, A's CAS must fail."""
        self._add_agents(1)
        seen_by_a = self.db.get_agent("agent-0")
        self.assertTrue(self.locks.reserve_agent("agent-0", "call-B"))
        won = self.db.cas_update("agents", "agent-0", seen_by_a.version,
                                 {"state": AgentState.RESERVED, "reserved_for_call_id": "call-A"})
        self.assertFalse(won)
        self.assertEqual(self.db.get_agent("agent-0").reserved_for_call_id, "call-B")

    def test_many_workers_many_agents_no_double_booking(self):
        self._add_agents(30)
        wins = {}
        lock = threading.Lock()

        def worker(i):
            n = 0
            while True:
                call_id = f"call-w{i}-{n}"
                agent_id = self.locks.reserve_any_available_agent(call_id)
                if agent_id is None:
                    if not self.db.find_available_agents(limit=1):
                        return
                    continue                  # lost every candidate to others; look again
                with lock:
                    self.assertNotIn(agent_id, wins, f"{agent_id} reserved twice")
                    wins[agent_id] = call_id
                n += 1

        self._race(12, worker)
        self.assertEqual(len(wins), 30)
        for agent_id, call_id in wins.items():
            agent = self.db.get_agent(agent_id)
            self.assertEqual(agent.state, AgentState.RESERVED)
            self.assertEqual(agent.reserved_for_call_id, call_id)

    def test_concurrent_borrower_claims_are_unique(self):
        self.db.insert_borrowers(Borrower(id=f"b-{i}", name="b", phone="1") for i in range(40))
        claimed = []
        lock = threading.Lock()

        def worker(i):
            n = 0
            while True:
                b = self.locks.claim_borrower(f"call-w{i}-{n}", f"w{i}", lease_s=10)
                if b is None:
                    if not self.db.find_due_borrowers(self.clock.now(), limit=1):
                        return
                    continue
                with lock:
                    claimed.append(b.id)
                n += 1

        self._race(8, worker)
        self.assertEqual(len(claimed), 40)
        self.assertEqual(len(set(claimed)), 40)
        self.assertEqual(self.db.count_borrowers_by_state()[BorrowerState.RESERVED], 40)

    def test_release_requires_ownership(self):
        self._add_agents(1)
        self.assertTrue(self.locks.reserve_agent("agent-0", "call-1"))
        self.assertFalse(self.locks.release_agent("agent-0", "call-2"), "non-owner must not release")
        self.assertEqual(self.db.get_agent("agent-0").state, AgentState.RESERVED)
        self.assertTrue(self.locks.release_agent("agent-0", "call-1"))
        self.assertEqual(self.db.get_agent("agent-0").state, AgentState.AVAILABLE)

    def test_unavailable_agent_cannot_be_reserved(self):
        self.db.insert_agent(Agent(id="paused", name="p", state=AgentState.PAUSED))
        self.assertFalse(self.locks.reserve_agent("paused", "call-1"))
        self.assertFalse(self.locks.reserve_agent("missing", "call-1"))

    def test_pacer_lease_has_one_holder_under_a_race(self):
        winners = []
        barrier = threading.Barrier(8)

        def contend(i):
            barrier.wait()
            if self.db.try_acquire_lease("pacer", f"w{i}", 1000.0, 3.0):
                winners.append(i)

        threads = [threading.Thread(target=contend, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(winners), 1)
        holder = f"w{winners[0]}"
        self.assertTrue(self.db.try_acquire_lease("pacer", holder, 1001.0, 3.0), "holder renews")
        self.assertFalse(self.db.try_acquire_lease("pacer", "other", 1003.5, 3.0), "not expired yet")
        self.assertTrue(self.db.try_acquire_lease("pacer", "other", 1004.0, 3.0), "expired: taken over")


class SinglePacerTest(unittest.TestCase):
    """Two predictive workers on one campaign must not both spend the same free agents."""

    def test_only_one_worker_paces_and_a_survivor_takes_over(self):
        h = Harness(mode=PREDICTIVE, agents=10, borrowers=500, answer_rate=0.3)
        seed_history(h.db, h.clock.now(), answered=90, unanswered=210)
        w1, w2 = h.engine, h.spawn_worker("worker-2")
        for _ in range(20):
            r1, r2 = w1.tick(), w2.tick()
            self.assertEqual((r1.safety is None) + (r2.safety is None), 1, "exactly one pacer per tick")
            self.assertIsNone(r2.safety, "worker-1 got the lease first and keeps renewing it")
            assert_invariants(h.db)
            h.clock.advance(1.0)

        # worker-1 dies. worker-2 keeps handling events, and paces once the lease expires.
        took_over_after = None
        for second in range(10):
            if w2.tick().safety is not None:
                took_over_after = second
                break
            h.clock.advance(1.0)
        self.assertIsNotNone(took_over_after)
        self.assertLessEqual(took_over_after, 3)
        assert_invariants(h.db)

    def test_stalled_pacer_cannot_dial_after_losing_the_lease(self):
        h = Harness(agents=5, borrowers=50)
        w1, w2 = h.engine, h.spawn_worker("worker-2")
        permit = w1.safety.evaluate(PacingDecision(PROGRESSIVE, 1, "t")).permit
        self.assertIsNotNone(permit)
        self.assertIsNotNone(w1.tick().safety, "worker-1 is the pacer")
        h.clock.advance(4.0)                                   # worker-1 stalls; its 3 s lease expires
        self.assertIsNotNone(w2.tick().safety, "worker-2 took over")
        self.assertLess(h.clock.now(), permit.expires_at, "the old permit is still unexpired")
        with self.assertRaises(SafetyViolation) as ctx:
            w1.safety.redeem(permit, "any-call")
        self.assertIn("pacer lease now held by worker-2", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
