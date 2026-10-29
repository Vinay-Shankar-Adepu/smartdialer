"""Provider B: slow and chaotic. Exists to break naive dialers.

* 0.5 s request latency
* 10% of initiate_call requests time out. Half of those time out AFTER the
  call was actually placed ("ghost calls"), so a timeout does not mean
  nothing happened.
* each event delivery is delayed by random jitter (0-3 s), so events arrive
  out of order (ANSWERED before RINGING, COMPLETED before ANSWERED)
* 20% of events are delivered twice with the same event id (webhook retry)
* 5% of events are re-sent with a NEW event id (a buggy retry that defeats
  event-id dedup; only state-machine idempotency catches it)
* 30% of hangups leave already-scheduled events in flight (stale events)
"""
import heapq

from core.models import ProviderEvent
from providers.mock_base import MockProvider
from providers.provider_interface import ProviderCallHandle, ProviderTimeout


class ProviderB(MockProvider):
    name = "PROVIDER_B"
    latency_s = 0.5
    failure_rate = 0.02
    timeout_rate = 0.10
    ghost_share = 0.5
    max_jitter_s = 3.0
    duplicate_rate = 0.20
    duplicate_new_id_rate = 0.05
    stale_after_hangup_rate = 0.30

    def _place_or_raise(self, call_id: str) -> ProviderCallHandle:
        if self.rng.random() < self.timeout_rate:
            if self.rng.random() < self.ghost_share:
                self._place(call_id)              # dialed, but the response is lost
                self.stats["ghost_timeouts"] += 1
            else:
                self.stats["clean_timeouts"] += 1
            raise ProviderTimeout(f"{self.name} initiate timed out")
        return self._place(call_id).handle

    def _deliver(self, event: ProviderEvent, deliver_at: float) -> None:
        jittered = deliver_at + self.rng.uniform(0.0, self.max_jitter_s)
        heapq.heappush(self._queue, (jittered, next(self._seq), event))
        if self.rng.random() < self.duplicate_rate:
            self.stats["duplicates_sent"] += 1
            heapq.heappush(self._queue, (jittered + self.rng.uniform(0.0, 2.0), next(self._seq), event))
        if self.rng.random() < self.duplicate_new_id_rate:
            self.stats["duplicates_new_id_sent"] += 1
            clone = ProviderEvent(f"{event.event_id}:retry", event.call_id, event.provider_call_id,
                                  event.event_type, event.occurred_at)
            heapq.heappush(self._queue, (jittered + self.rng.uniform(0.0, 2.0), next(self._seq), clone))

    def _drop_future_events_on_hangup(self) -> bool:
        if self.rng.random() < self.stale_after_hangup_rate:
            self.stats["stale_events_left"] += 1
            return False
        return True
