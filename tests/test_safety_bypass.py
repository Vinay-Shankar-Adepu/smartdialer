"""The Safety Controller cannot be bypassed: no valid permit, no call."""
import dataclasses
import unittest

from core.state_machines import CallState
from dialer.pacing import PREDICTIVE, PROGRESSIVE, PacingDecision
from dialer.safety_controller import DialPermit, SafeDialGateway, SafetyViolation
from providers.provider_interface import TelecomProvider
from tests.helpers import Harness


class SafetyBypassTest(unittest.TestCase):
    def setUp(self):
        self.h = Harness(agents=2, borrowers=5)
        self.engine = self.h.engine

    def reserved_call(self, with_agent=True):
        """Prepare a RESERVED call exactly as the allocator would, without dialing."""
        call_id = f"manual-{len(self.h.calls())}"
        locks = self.engine.locks
        borrower = locks.claim_borrower(call_id, "test", 10)
        from core.models import Call
        self.h.db.insert_call(Call(id=call_id, borrower_id=borrower.id, created_at=self.h.clock.now(),
                                   lease_expires_at=self.h.clock.now() + 10))
        agent_id = locks.reserve_any_available_agent(call_id) if with_agent else None
        locks.transition_call(call_id, CallState.RESERVED, changes={"agent_id": agent_id})
        return call_id

    def permit(self, mode=PROGRESSIVE, n=1):
        return self.engine.safety.evaluate(PacingDecision(mode, n, "test")).permit

    def assertBlocked(self, permit, call_id):
        with self.assertRaises(SafetyViolation):
            self.engine.gateway.dial(permit, call_id, "+910000000000")
        self.assertEqual(self.h.provider.stats["placed"], 0, "provider must not be reached")

    def test_forged_permit(self):
        forged = DialPermit("made-up", PROGRESSIVE, 99, 0.0, 1e12, "0" * 64)
        self.assertBlocked(forged, self.reserved_call())

    def test_tampered_permit(self):
        real = self.permit(n=1)
        self.assertBlocked(dataclasses.replace(real, max_calls=1000), self.reserved_call())

    def test_expired_permit(self):
        p = self.permit()
        self.h.clock.advance(self.engine.safety.cfg.permit_ttl_s + 1)
        self.assertBlocked(p, self.reserved_call())

    def test_permit_budget_is_enforced(self):
        p = self.permit(n=1)
        first = self.reserved_call()
        self.engine.gateway.dial(p, first, "+910000000000")
        with self.assertRaises(SafetyViolation):
            self.engine.gateway.dial(p, self.reserved_call(), "+910000000001")
        self.assertEqual(self.h.provider.stats["placed"], 1)

    def test_progressive_permit_requires_reserved_agent(self):
        self.assertBlocked(self.permit(PROGRESSIVE), self.reserved_call(with_agent=False))

    def test_same_call_cannot_be_dialed_twice(self):
        p = self.permit(n=2)
        call_id = self.reserved_call()
        self.engine.gateway.dial(p, call_id, "+910000000000")
        with self.assertRaises(SafetyViolation):
            self.engine.gateway.dial(p, call_id, "+910000000000")

    def test_call_must_be_reserved(self):
        call_id = self.reserved_call()
        self.engine.lifecycle.finalize(call_id, CallState.CANCELLED, "test")
        self.assertBlocked(self.permit(), call_id)

    def test_permit_from_another_controller_is_rejected(self):
        other = self.h.spawn_worker("other")
        foreign = other.safety.evaluate(PacingDecision(PROGRESSIVE, 1, "t")).permit
        self.assertBlocked(foreign, self.reserved_call())

    def test_pacing_engines_have_no_way_to_dial(self):
        for pacing in (self.engine.pacing, Harness(mode=PREDICTIVE).engine.pacing):
            for value in vars(pacing).values():
                self.assertNotIsInstance(value, (TelecomProvider, SafeDialGateway))
            self.assertFalse(any(hasattr(pacing, a) for a in ("db", "gateway", "provider", "allocator")))

    def test_only_the_gateway_holds_the_provider(self):
        holders = []
        for name, component in vars(self.engine).items():
            for attr, value in vars(component).items() if hasattr(component, "__dict__") else []:
                if isinstance(value, TelecomProvider):
                    holders.append(f"{name}.{attr}")
        self.assertEqual(holders, ["gateway._provider"])


if __name__ == "__main__":
    unittest.main()
