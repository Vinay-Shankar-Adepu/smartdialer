"""Shared machinery for the mock providers.

A mock provider has two halves:
* `BorrowerBehavior` models the WORLD: does the borrower answer, how long
  until they pick up, how long they talk. Scenarios tune it, and it can change
  mid-run (for example, the answer rate jumps).
* `MockProvider` models the TELECOM API: latency, failures, how events are
  delivered. ProviderA and ProviderB subclass it with different behaviors.

Time is taken from the injected Clock. "Latency" shifts event timestamps
instead of calling time.sleep, so simulations run fast and are deterministic.
"""
import heapq
import itertools
import random
import threading
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from core.clock import Clock
from core.models import ProviderEvent
from core.state_machines import ProviderEventType
from providers.provider_interface import (
    ProviderCallHandle,
    ProviderCallStatus,
    ProviderTimeout,
    TelecomProvider,
)


@dataclass
class CallScript:
    outcome: str          # "answered" | "no_answer" | "busy" | "failed"
    ring_s: float         # seconds from dial until answer / no-answer
    talk_s: float = 0.0


class BorrowerBehavior:
    """How borrowers respond to calls. `answer_rate` is mutable at runtime."""

    def __init__(self, answer_rate: float, talk_mean_s: float = 120.0,
                 ring_to_answer_s: Tuple[float, float] = (4.0, 15.0),
                 no_answer_after_s: float = 25.0, busy_share: float = 0.15,
                 talk_min_s: float = 15.0):
        self.answer_rate = answer_rate
        self.talk_mean_s = talk_mean_s
        self.ring_to_answer_s = ring_to_answer_s
        self.no_answer_after_s = no_answer_after_s
        self.busy_share = busy_share          # share of non-answers that are BUSY
        self.talk_min_s = talk_min_s

    def script(self, rng: random.Random, failure_rate: float) -> CallScript:
        if rng.random() < failure_rate:
            return CallScript("failed", ring_s=0.5)
        if rng.random() < self.answer_rate:
            talk = max(self.talk_min_s, rng.gauss(self.talk_mean_s, self.talk_mean_s * 0.35))
            return CallScript("answered", ring_s=rng.uniform(*self.ring_to_answer_s), talk_s=talk)
        if rng.random() < self.busy_share:
            return CallScript("busy", ring_s=1.5)
        return CallScript("no_answer", ring_s=self.no_answer_after_s)


@dataclass
class _MockCall:
    handle: ProviderCallHandle
    script: CallScript
    answered_at: Optional[float]
    end_at: float
    hung_up: bool = False


class MockProvider(TelecomProvider):
    name = "MOCK"
    latency_s = 0.1
    failure_rate = 0.0

    def __init__(self, clock: Clock, behavior: BorrowerBehavior, seed: Optional[int] = None):
        self.clock = clock
        self.behavior = behavior
        self.rng = random.Random(seed)
        self.stats: Counter = Counter()
        self._lock = threading.RLock()
        self._calls: Dict[str, _MockCall] = {}
        self._by_ref: Dict[str, str] = {}
        self._queue: List[Tuple[float, int, ProviderEvent]] = []
        self._seq = itertools.count()
        self._ids = itertools.count(1)
        self._outage_until: Optional[float] = None

    # ------------------------------------------------------------ outage control
    def start_outage(self, duration_s: float) -> None:
        with self._lock:
            self._outage_until = self.clock.now() + duration_s

    def end_outage(self) -> None:
        with self._lock:
            self._outage_until = None

    @property
    def in_outage(self) -> bool:
        return self._outage_until is not None and self.clock.now() < self._outage_until

    def _check_reachable(self, op: str) -> None:
        if self.in_outage:
            self.stats[f"{op}_timeout_outage"] += 1
            raise ProviderTimeout(f"{self.name} unreachable ({op})")

    # ------------------------------------------------------------ API
    def initiate_call(self, call_id: str, phone: str) -> ProviderCallHandle:
        with self._lock:
            self._check_reachable("initiate")
            existing = self._by_ref.get(call_id)
            if existing is not None:
                self.stats["initiate_idempotent_replay"] += 1
                return self._calls[existing].handle
            return self._place_or_raise(call_id)

    def _place_or_raise(self, call_id: str) -> ProviderCallHandle:
        return self._place(call_id).handle

    def _place(self, call_id: str) -> _MockCall:
        pcid = f"{self.name}-{next(self._ids)}"
        script = self.behavior.script(self.rng, self.failure_rate)
        t0 = self.clock.now() + self.latency_s
        timeline: List[Tuple[ProviderEventType, float]]
        answered_at = None
        if script.outcome == "failed":
            timeline = [(ProviderEventType.FAILED, t0 + script.ring_s)]
        elif script.outcome == "busy":
            timeline = [(ProviderEventType.BUSY, t0 + script.ring_s)]
        elif script.outcome == "no_answer":
            timeline = [(ProviderEventType.RINGING, t0 + 1.0),
                        (ProviderEventType.NO_ANSWER, t0 + script.ring_s)]
        else:
            answered_at = t0 + script.ring_s
            timeline = [(ProviderEventType.RINGING, t0 + 1.0),
                        (ProviderEventType.ANSWERED, answered_at),
                        (ProviderEventType.COMPLETED, answered_at + script.talk_s)]
        mc = _MockCall(ProviderCallHandle(pcid, call_id, self.name), script, answered_at,
                       end_at=timeline[-1][1])
        self._calls[pcid] = mc
        self._by_ref[call_id] = pcid
        self.stats["placed"] += 1
        self.stats[f"outcome_{script.outcome}"] += 1
        for etype, at in timeline:
            self._emit(mc, etype, at)
        return mc

    def _emit(self, mc: _MockCall, etype: ProviderEventType, at: float,
              event_id: Optional[str] = None) -> None:
        pcid = mc.handle.provider_call_id
        event = ProviderEvent(event_id or f"{pcid}:{etype.value}", mc.handle.call_id, pcid, etype, at)
        self._deliver(event, at)

    def _deliver(self, event: ProviderEvent, deliver_at: float) -> None:
        """Provider A delivers in order and exactly once. B overrides this."""
        heapq.heappush(self._queue, (deliver_at, next(self._seq), event))

    def poll_events(self) -> List[ProviderEvent]:
        with self._lock:
            if self.in_outage:
                return []          # webhooks are not delivered during an outage
            now = self.clock.now()
            due = []
            while self._queue and self._queue[0][0] <= now:
                due.append(heapq.heappop(self._queue)[2])
            self.stats["events_delivered"] += len(due)
            return due

    def hangup(self, provider_call_id: str) -> None:
        with self._lock:
            self._check_reachable("hangup")
            mc = self._calls.get(provider_call_id)
            if mc is None or mc.hung_up:
                return
            now = self.clock.now()
            mc.hung_up = True
            self.stats["hangups"] += 1
            if self._drop_future_events_on_hangup():
                self._queue = [item for item in self._queue
                               if not (item[2].provider_call_id == provider_call_id and item[2].occurred_at > now)]
                heapq.heapify(self._queue)
                if mc.answered_at is not None and mc.answered_at <= now < mc.end_at:
                    self._emit(mc, ProviderEventType.COMPLETED, now)
            mc.end_at = min(mc.end_at, now)

    def _drop_future_events_on_hangup(self) -> bool:
        return True

    def lookup(self, call_id: str) -> Optional[ProviderCallStatus]:
        with self._lock:
            self._check_reachable("lookup")
            pcid = self._by_ref.get(call_id)
            if pcid is None:
                return None
            mc = self._calls[pcid]
            return ProviderCallStatus(pcid, call_id, is_live=not mc.hung_up and self.clock.now() < mc.end_at)

    # ------------------------------------------------------------ test helpers
    def pending_event_count(self) -> int:
        with self._lock:
            return len(self._queue)
