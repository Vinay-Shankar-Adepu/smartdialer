"""Wires the components together and runs one control-loop tick.

    tick():
      1. ingest provider events          (idempotent, order-tolerant)
      2. finish wrap-ups                 (frees agents before bridging)
      3. bridge ANSWERED calls           (connect, or abandon after hold time)
      4. recovery sweep                  (every sweep_interval_s)
      5. only if this worker holds the "pacer" lease:
         pacing  -> recommendation       (pure function of a snapshot)
         safety  -> verdict + permit     (independent snapshot; the only source of permits)
         allocator.execute(permit)       (every dial redeemed through SafeDialGateway)

Steps 1-4 are safe on any number of workers (CAS + idempotent events). Step 5
is not: two safety controllers reading the same "10 free agents" would each
approve overdial for them. So exactly one worker paces at a time; if it dies,
its lease expires and another worker takes over within pacer_lease_s.

Only the gateway holds the provider. The pacing engine gets nothing but a
Snapshot.
"""
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

from core.clock import Clock
from core.database import Database
from core.lock_manager import LockManager
from dialer.agent_service import AgentService
from dialer.call_allocator import AllocationResult, CallAllocator
from dialer.call_lifecycle import CallLifecycle, LifecycleConfig
from dialer.event_processor import EventProcessor
from dialer.pacing import PREDICTIVE, PROGRESSIVE, PacingDecision
from dialer.predictive_pacing import PredictiveConfig, PredictivePacing
from dialer.progressive_pacing import ProgressivePacing
from dialer.recovery import RecoveryConfig, RecoveryService
from dialer.safety_controller import SafeDialGateway, SafetyConfig, SafetyController, SafetyDecision
from dialer.snapshot import SnapshotReader
from providers.provider_interface import TelecomProvider


@dataclass
class EngineConfig:
    worker_id: str = "worker-1"
    sweep_interval_s: float = 5.0
    pacer_lease_s: float = 3.0
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    predictive: PredictiveConfig = field(default_factory=PredictiveConfig)


@dataclass
class TickReport:
    events: Counter
    bridge: Counter
    recovery: Counter
    decision: Optional[PacingDecision]       # None when another worker holds the pacer lease
    safety: Optional[SafetyDecision]
    allocation: Optional[AllocationResult]


def make_pacing(mode: str, config: EngineConfig):
    if mode == PROGRESSIVE:
        return ProgressivePacing()
    if mode == PREDICTIVE:
        return PredictivePacing(config.predictive)
    raise ValueError(f"unknown mode {mode!r}")


class DialerEngine:
    def __init__(self, db: Database, clock: Clock, provider: TelecomProvider, mode: str,
                 config: Optional[EngineConfig] = None,
                 crash_hook: Optional[Callable[[str, str], None]] = None):
        self.cfg = config or EngineConfig()
        self.db = db
        self.clock = clock
        self.locks = LockManager(db, clock)
        wrap = self.cfg.lifecycle.wrap_up_s
        self.reader = SnapshotReader(db, clock, wrap_up_s=wrap)
        self.safety = SafetyController(db, clock, SnapshotReader(db, clock, wrap_up_s=wrap), self.cfg.safety)
        self.safety.worker_id = self.cfg.worker_id
        self.gateway = SafeDialGateway(provider, self.safety)
        self.lifecycle = CallLifecycle(db, self.locks, clock, self.cfg.lifecycle)
        self.events = EventProcessor(db, self.locks, self.lifecycle, clock)
        self.agents = AgentService(db, self.locks, self.lifecycle, self.gateway, clock)
        self.recovery = RecoveryService(db, self.locks, self.lifecycle, self.gateway, clock, self.cfg.recovery)
        self.allocator = CallAllocator(db, self.locks, self.lifecycle, self.gateway, clock,
                                       self.cfg.worker_id, self.cfg.lifecycle.lease_s, crash_hook)
        self.pacing = make_pacing(mode, self.cfg)
        self._last_sweep = float("-inf")

    @property
    def mode(self) -> str:
        return self.pacing.mode

    def ingest_events(self) -> Counter:
        outcomes: Counter = Counter()
        for event in self.gateway.poll_events():
            outcomes[self.events.process(event)] += 1
        return outcomes

    def tick(self) -> TickReport:
        events = self.ingest_events()
        self.agents.complete_wrapups()
        bridge = self.lifecycle.bridge_answered_calls(self.gateway.hangup)
        recovery: Counter = Counter()
        if self.clock.now() - self._last_sweep >= self.cfg.sweep_interval_s:
            recovery = self.recovery.sweep()
            self._last_sweep = self.clock.now()
        if not self.db.try_acquire_lease("pacer", self.cfg.worker_id, self.clock.now(), self.cfg.pacer_lease_s):
            return TickReport(events, bridge, recovery, None, None, None)
        decision = self.pacing.recommend(self.reader.read())
        verdict = self.safety.evaluate(decision)
        allocation = self.allocator.execute(verdict.permit) if verdict.permit else None
        return TickReport(events, bridge, recovery, decision, verdict, allocation)
