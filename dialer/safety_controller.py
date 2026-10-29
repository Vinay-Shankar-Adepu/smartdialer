"""Safety Controller: the only component that can authorize a dial.

    Pacing --(PacingDecision)--> SafetyController --(DialPermit)--> CallAllocator
                                        ^                                |
                                        |  redeem(permit, call) on       v
                                        +--------- every dial ---- SafeDialGateway --> Provider

Why it cannot be bypassed
-------------------------
1. Pacing engines are pure functions with no database, provider or allocator
   reference. All they can do is return a number.
2. The raw provider is only ever held by SafeDialGateway. The only method
   that places a call is `SafeDialGateway.dial(permit, call_id, phone)`.
3. `dial()` first calls `SafetyController.redeem()`, which checks that the
   permit was really issued by this controller (HMAC with a per-process
   secret plus an outstanding-permit registry), has not expired, still has
   budget, and that the call is RESERVED in the database. A progressive
   permit also requires the call to own an agent that is RESERVED for it.
   Permits are single-use per call and limited in count.
4. Each permit covers a short window (seconds). Conditions are re-evaluated
   from the database on every tick, never cached from the pacing engine.

What it checks (in order)
-------------------------
  a. Provider circuit breaker      OPEN (5 consecutive or >=50% failures, or accepted dials with
                                   no webhook for 15 s) -> REJECT everything;
                                   HALF_OPEN after 30 s -> 1 probe call decides CLOSED or OPEN again
  b. Progressive request           cap at idle agents - unbound in-flight calls
  c. Predictive request -> fallback to progressive if any of:
       - provider degraded (>=25% of the last 50 dials failed)
       - agents dropped sharply (>=15% of the logged-in peak in the last 30 s)
       - answer burst: >= 3 answered callers waiting with no free agent (model is wrong right now)
       - abandon rate above target (3%)
       - not enough data to trust the answer-rate estimate (warm-up)
       - still in cool-down after one of the above
  d. Predictive capacity cap:
       projected answers = (unbound ringing + new) x p_safety <= free agents x (1 + tolerance)
       where p_safety = max(upper bound of the 5-min answer rate, point estimate of the 1-min one),
       so a sudden jump in answer rate shrinks the cap immediately.
     plus a blast-radius cap of max_lines_per_free_agent, and a ramp cap on new lines per tick
     so exposure builds gradually and early answers are seen before every line is committed.
  e. Global per-tick dial cap (provider calls-per-second limit).
"""
import hashlib
import hmac
import json
import math
import secrets
import threading
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

from core.clock import Clock
from core.database import Database
from core.state_machines import AgentState, CallState
from dialer.estimators import smoothed_rate, wilson_upper
from dialer.pacing import PREDICTIVE, PROGRESSIVE, PacingDecision
from dialer.snapshot import SnapshotReader, forecast_free_agents
from providers.provider_interface import (
    ProviderCallHandle,
    ProviderCallStatus,
    ProviderError,
    TelecomProvider,
)


class Verdict(str, Enum):
    APPROVED = "APPROVED"
    REDUCED = "REDUCED"
    REJECTED = "REJECTED"
    FALLBACK_PROGRESSIVE = "FALLBACK_PROGRESSIVE"


class SafetyViolation(Exception):
    """An attempt to dial without a valid permit. Never caught silently."""


@dataclass(frozen=True)
class DialPermit:
    permit_id: str
    mode: str                 # "progressive" => every call must own a RESERVED agent
    max_calls: int
    issued_at: float
    expires_at: float
    token: str

    @property
    def requires_agent(self) -> bool:
        return self.mode == PROGRESSIVE


@dataclass
class SafetyDecision:
    verdict: Verdict
    requested_mode: str
    effective_mode: str
    recommended: int
    approved: int
    reasons: List[str] = field(default_factory=list)
    checks: Dict[str, float] = field(default_factory=dict)
    permit: Optional[DialPermit] = None


@dataclass
class SafetyConfig:
    target_abandon_rate: float = 0.03        # regulatory-style ceiling (e.g. FCC/Ofcom 3%)
    min_answered_for_abandon_check: int = 20
    min_outcomes_for_predictive: int = 30    # below this, the answer-rate estimate is not trusted
    min_short_window_outcomes: int = 40
    fallback_cooldown_s: float = 60.0
    agent_drop_window_s: float = 30.0
    agent_drop_fraction: float = 0.15
    agent_drop_min: int = 3
    overflow_tolerance: float = 0.15
    max_lines_per_free_agent: float = 5.0
    # Ramp: new predictive lines per tick <= free agents x this. Exposure builds over several
    # ticks, so the first answers (4+ s after dialing) are seen before all lines are committed.
    max_new_lines_per_free_agent_per_tick: float = 0.5
    answer_burst_waiting: int = 3            # answered callers with no agent -> stop overdialing now
    max_dials_per_tick: int = 50
    permit_ttl_s: float = 5.0
    prior_answer_rate: float = 0.3
    prior_weight: float = 10.0
    confidence_z: float = 1.28
    default_time_to_answer_s: float = 10.0
    default_talk_s: float = 120.0
    # provider circuit breaker
    breaker_window: int = 50
    breaker_min_samples: int = 10
    breaker_degraded_rate: float = 0.25      # Provider B's normal 10% timeouts must NOT trip this
    breaker_open_rate: float = 0.5
    breaker_open_consecutive: int = 5        # a hard outage opens the breaker within a tick or two
    breaker_cooldown_s: float = 30.0
    provider_silence_s: float = 15.0         # accepted dials but no webhook for this long -> open breaker


class ProviderCircuitBreaker:
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

    def __init__(self, cfg: SafetyConfig, clock: Clock):
        self.cfg = cfg
        self.clock = clock
        self._results: Deque[bool] = deque(maxlen=cfg.breaker_window)
        self._state = self.CLOSED
        self._open_until = 0.0
        self._probe_busy_until = 0.0
        self._consecutive_failures = 0
        self.trips = 0

    def state(self) -> str:
        if self._state == self.OPEN and self.clock.now() >= self._open_until:
            self._state = self.HALF_OPEN
        return self._state

    @property
    def failure_rate(self) -> float:
        if not self._results:
            return 0.0
        return self._results.count(False) / len(self._results)

    @property
    def samples(self) -> int:
        return len(self._results)

    @property
    def degraded(self) -> bool:
        return self.samples >= self.cfg.breaker_min_samples and self.failure_rate >= self.cfg.breaker_degraded_rate

    def probe_allowed(self) -> bool:
        return self.state() == self.HALF_OPEN and self.clock.now() >= self._probe_busy_until

    def probe_issued(self, until: float) -> None:
        self._probe_busy_until = until

    def record(self, ok: bool) -> None:
        state = self.state()
        if state == self.HALF_OPEN:
            if ok:
                self._state = self.CLOSED
                self._results.clear()
            else:
                self._trip()
            return
        if state == self.OPEN:
            return
        self._results.append(ok)
        self._consecutive_failures = 0 if ok else self._consecutive_failures + 1
        if self._consecutive_failures >= self.cfg.breaker_open_consecutive or (
                self.samples >= self.cfg.breaker_min_samples and self.failure_rate >= self.cfg.breaker_open_rate):
            self._trip()

    def force_open(self) -> None:
        if self._state != self.OPEN:
            self._trip()

    def _trip(self) -> None:
        self._consecutive_failures = 0
        self._state = self.OPEN
        self._open_until = self.clock.now() + self.cfg.breaker_cooldown_s
        self._probe_busy_until = 0.0
        self.trips += 1


class SafetyController:
    def __init__(self, db: Database, clock: Clock, reader: SnapshotReader,
                 config: Optional[SafetyConfig] = None):
        self.db = db
        self.clock = clock
        self.reader = reader
        self.cfg = config or SafetyConfig()
        self.breaker = ProviderCircuitBreaker(self.cfg, clock)
        self.stats: Counter = Counter()
        self._secret = secrets.token_bytes(32)
        self._outstanding: Dict[str, Tuple[DialPermit, int]] = {}
        self._redeemed_calls: Dict[str, float] = {}     # call_id -> expiry of the permit that dialed it
        self._agent_history: Deque[Tuple[float, int]] = deque()
        self._fallback_until = 0.0
        self._fallback_cause: List[str] = []
        self._unacked_dial_since: Optional[float] = None   # oldest accepted dial with no event since
        self._lock = threading.RLock()

    # ================================================================ evaluate
    def evaluate(self, decision: PacingDecision) -> SafetyDecision:
        with self._lock:
            now = self.clock.now()
            snap = self.reader.read()              # independent read: never trust pacing's numbers
            self._expire_permits(now)
            self._record_agents(now, snap.logged_in)

            progressive_cap = max(0, snap.idle_agents - snap.unbound_in_flight)
            tick_cap = self.cfg.max_dials_per_tick
            checks = {"idle_agents": snap.idle_agents, "unbound_in_flight": snap.unbound_in_flight,
                      "progressive_cap": progressive_cap, "logged_in": snap.logged_in,
                      "provider_failure_rate": round(self.breaker.failure_rate, 3)}

            silent_for = now - self._unacked_dial_since if self._unacked_dial_since is not None else 0.0
            if silent_for > self.cfg.provider_silence_s:
                # The provider accepts dials but sends no callbacks. We can't see those calls,
                # so we can't bridge them. Treat it as an outage.
                self.breaker.force_open()
                self._unacked_dial_since = None
                self.stats["silence_trips"] += 1
                return self._finish(decision, Verdict.REJECTED, PROGRESSIVE, 0, checks,
                                    [f"provider silent: no events for {silent_for:.0f}s after accepted dials"])

            breaker_state = self.breaker.state()
            checks["breaker"] = breaker_state
            if breaker_state == ProviderCircuitBreaker.OPEN:
                return self._finish(decision, Verdict.REJECTED, PROGRESSIVE, 0, checks,
                                    ["provider circuit OPEN (dial failures or silence); waiting for cool-down"])
            if breaker_state == ProviderCircuitBreaker.HALF_OPEN:
                n = 1 if (self.breaker.probe_allowed() and progressive_cap > 0 and decision.recommended > 0) else 0
                if n:
                    self.breaker.probe_issued(now + self.cfg.permit_ttl_s)
                return self._finish(decision, Verdict.FALLBACK_PROGRESSIVE if n else Verdict.REJECTED,
                                    PROGRESSIVE, n, checks, ["provider circuit HALF_OPEN: single probe call"])

            if decision.mode == PROGRESSIVE:
                approved = min(decision.recommended, progressive_cap, tick_cap)
                return self._finish(decision, self._verdict(decision.recommended, approved),
                                    PROGRESSIVE, approved, checks,
                                    [] if approved == decision.recommended else
                                    [f"capped at {approved}: idle agents {snap.idle_agents} - "
                                     f"unbound in-flight {snap.unbound_in_flight}, tick cap {tick_cap}"])

            fallback_reasons = self._fallback_reasons(now, snap)
            if fallback_reasons:
                approved = min(progressive_cap, tick_cap)
                return self._finish(decision, Verdict.FALLBACK_PROGRESSIVE, PROGRESSIVE, approved,
                                    checks, fallback_reasons)

            cap, cap_checks = self._predictive_cap(snap)
            checks.update(cap_checks)
            approved = min(decision.recommended, cap, tick_cap)
            reasons = [] if approved == decision.recommended else [self._cap_reason(approved, cap_checks, tick_cap)]
            return self._finish(decision, self._verdict(decision.recommended, approved),
                                PREDICTIVE, approved, checks, reasons)

    def _cap_reason(self, approved: int, cap_checks: Dict[str, float], tick_cap: int) -> str:
        """Name the limit that actually bound the number, so the explanation is never misleading."""
        free = cap_checks["free_agents_net"]
        ringing = cap_checks["unbound_ringing"]
        if approved == tick_cap and cap_checks["predictive_cap"] > tick_cap:
            return f"capped at {approved}: max {tick_cap} dials per tick"
        if approved == cap_checks["ramp_cap"] and approved > 0:
            return (f"capped at {approved}: ramp limit {self.cfg.max_new_lines_per_free_agent_per_tick} "
                    f"new lines per free agent per tick ({free:.1f} free)")
        if cap_checks["ratio_cap_lines"] <= cap_checks["answer_cap_lines"]:
            return (f"capped at {approved}: blast-radius limit {self.cfg.max_lines_per_free_agent:.0f} lines "
                    f"per free agent -> {cap_checks['ratio_cap_lines']:.0f} lines for {free:.1f} free agents, "
                    f"{ringing} already ringing")
        return (f"capped at {approved}: (ringing {ringing} + new) x p_safety {cap_checks['p_safety']:.2f} "
                f"must stay <= {free:.1f} free agents x {1 + self.cfg.overflow_tolerance:.2f} "
                f"-> max {cap_checks['answer_cap_lines']:.0f} lines")

    @staticmethod
    def _verdict(recommended: int, approved: int) -> Verdict:
        if approved >= recommended:
            return Verdict.APPROVED
        return Verdict.REDUCED if approved > 0 else Verdict.REJECTED

    def _record_agents(self, now: float, logged_in: int) -> None:
        self._agent_history.append((now, logged_in))
        while self._agent_history and self._agent_history[0][0] < now - self.cfg.agent_drop_window_s:
            self._agent_history.popleft()

    def _fallback_reasons(self, now: float, snap) -> List[str]:
        triggers: List[str] = []
        if self.breaker.degraded:
            triggers.append(f"provider degraded: {self.breaker.failure_rate:.0%} of last "
                            f"{self.breaker.samples} dials failed")
        peak = max(n for _, n in self._agent_history)
        drop = peak - snap.logged_in
        if drop >= max(self.cfg.agent_drop_min, self.cfg.agent_drop_fraction * peak):
            triggers.append(f"agent drop: logged-in fell {peak} -> {snap.logged_in} "
                            f"within {self.cfg.agent_drop_window_s:.0f}s")
        if snap.unbound_answered_waiting >= self.cfg.answer_burst_waiting:
            # More borrowers are answering than agents can take, right now. Don't wait for the
            # 5-min abandon rate to notice: the model is wrong, so stop adding lines this tick.
            triggers.append(f"answer burst: {snap.unbound_answered_waiting} answered callers waiting "
                            f"with no free agent")
        if snap.answered >= self.cfg.min_answered_for_abandon_check and \
                snap.abandon_rate > self.cfg.target_abandon_rate:
            triggers.append(f"abandon rate {snap.abandon_rate:.1%} > target {self.cfg.target_abandon_rate:.0%}")
        if triggers:
            self._fallback_until = now + self.cfg.fallback_cooldown_s
            self._fallback_cause = triggers
            self.stats["fallback_triggers"] += 1
            return triggers
        if now < self._fallback_until:
            return [f"cool-down ({self._fallback_until - now:.0f}s left) after: " + "; ".join(self._fallback_cause)]
        if snap.outcomes < self.cfg.min_outcomes_for_predictive:
            return [f"low confidence: only {snap.outcomes} call outcomes in the last "
                    f"{snap.window_s:.0f}s (need {self.cfg.min_outcomes_for_predictive})"]
        return []

    def _predictive_cap(self, snap) -> Tuple[int, Dict[str, float]]:
        cfg = self.cfg
        a = snap.answered + cfg.prior_answer_rate * cfg.prior_weight
        n = snap.outcomes + cfg.prior_weight
        p_long = wilson_upper(a, n, cfg.confidence_z)
        # Short window uses the point estimate: an upper bound on ~40 samples is too noisy,
        # while a genuine surge (20% -> 60%) shows up clearly in the point estimate within a minute.
        p_short = 0.0
        if snap.short_outcomes >= cfg.min_short_window_outcomes:
            p_short = snap.short_answered / snap.short_outcomes
        p_safety = max(p_long, p_short, 0.01)
        lookahead = snap.avg_time_to_answer_s or cfg.default_time_to_answer_s
        talk = snap.avg_talk_s or cfg.default_talk_s
        free = forecast_free_agents(snap, lookahead, talk).net
        answer_cap = free * (1 + cfg.overflow_tolerance) / p_safety
        ratio_cap = free * cfg.max_lines_per_free_agent
        ramp_cap = max(1, math.ceil(free * cfg.max_new_lines_per_free_agent_per_tick))
        cap = max(0, min(math.floor(min(answer_cap, ratio_cap) - snap.unbound_ringing), ramp_cap))
        return cap, {"p_long_upper": round(p_long, 3), "p_short_point": round(p_short, 3),
                     "p_safety": round(p_safety, 3), "free_agents_net": round(free, 2),
                     "answer_cap_lines": round(answer_cap, 1), "ratio_cap_lines": round(ratio_cap, 1),
                     "unbound_ringing": snap.unbound_ringing, "ramp_cap": ramp_cap,
                     "predictive_cap": cap, "smoothed_answer_rate": round(
                         smoothed_rate(snap.answered, snap.outcomes, cfg.prior_answer_rate, cfg.prior_weight), 3)}

    def _finish(self, decision: PacingDecision, verdict: Verdict, mode: str, approved: int,
                checks: Dict, reasons: List[str]) -> SafetyDecision:
        now = self.clock.now()
        permit = self._issue_permit(mode, approved, now) if approved > 0 else None
        self.stats[verdict.value] += 1
        self.db.log_decision(now, decision.mode, decision.recommended, verdict.value, mode, approved,
                             decision.summary, json.dumps(reasons))
        return SafetyDecision(verdict, decision.mode, mode, decision.recommended, approved,
                              reasons, checks, permit)

    # ================================================================ permits
    def _sign(self, permit_id: str, mode: str, max_calls: int, issued_at: float, expires_at: float) -> str:
        msg = f"{permit_id}|{mode}|{max_calls}|{issued_at!r}|{expires_at!r}".encode()
        return hmac.new(self._secret, msg, hashlib.sha256).hexdigest()

    def _issue_permit(self, mode: str, max_calls: int, now: float) -> DialPermit:
        pid = uuid.uuid4().hex
        expires = now + self.cfg.permit_ttl_s
        permit = DialPermit(pid, mode, max_calls, now, expires, self._sign(pid, mode, max_calls, now, expires))
        self._outstanding[pid] = (permit, max_calls)
        return permit

    def _expire_permits(self, now: float) -> None:
        for pid in [pid for pid, (p, _) in self._outstanding.items() if p.expires_at < now]:
            del self._outstanding[pid]
        # After a permit expires, the DB check (call RESERVED and undialed) is enough to stop
        # a double dial, so the in-memory guard can be pruned and stays bounded.
        for call_id in [c for c, exp in self._redeemed_calls.items() if exp < now]:
            del self._redeemed_calls[call_id]

    def redeem(self, permit: DialPermit, call_id: str) -> None:
        """Spend one unit of a permit on `call_id`, or raise SafetyViolation."""
        with self._lock:
            expected = self._sign(permit.permit_id, permit.mode, permit.max_calls,
                                  permit.issued_at, permit.expires_at)
            if not hmac.compare_digest(expected, permit.token):
                self._violation("forged or tampered permit")
            entry = self._outstanding.get(permit.permit_id)
            if entry is None:
                self._violation("unknown, exhausted or expired permit")
            _, remaining = entry
            if self.clock.now() > permit.expires_at:
                del self._outstanding[permit.permit_id]
                self._violation("permit expired")
            if call_id in self._redeemed_calls:
                self._violation(f"call {call_id} was already dialed")
            call = self.db.get_call(call_id)
            if call is None or call.state != CallState.RESERVED or call.provider_call_id:
                self._violation(f"call {call_id} is not RESERVED and undialed")
            if permit.requires_agent:
                agent = self.db.get_agent(call.agent_id) if call.agent_id else None
                if agent is None or agent.state != AgentState.RESERVED or agent.reserved_for_call_id != call.id:
                    self._violation(f"progressive permit requires call {call_id} to own a RESERVED agent")
            remaining -= 1
            if remaining > 0:
                self._outstanding[permit.permit_id] = (permit, remaining)
            else:
                del self._outstanding[permit.permit_id]
            self._redeemed_calls[call_id] = permit.expires_at
            self.stats["permits_redeemed"] += 1

    def _violation(self, why: str) -> None:
        self.stats["violations"] += 1
        raise SafetyViolation(why)

    def record_dial_result(self, ok: bool) -> None:
        with self._lock:
            self.breaker.record(ok)
            if ok and self._unacked_dial_since is None:
                self._unacked_dial_since = self.clock.now()

    def record_events_received(self, count: int) -> None:
        if count:
            with self._lock:
                self._unacked_dial_since = None


class SafeDialGateway:
    """The ONLY object that holds the raw provider. Dialing requires a permit."""

    def __init__(self, provider: TelecomProvider, safety: SafetyController):
        self._provider = provider
        self._safety = safety

    @property
    def provider_name(self) -> str:
        return self._provider.name

    def dial(self, permit: DialPermit, call_id: str, phone: str) -> ProviderCallHandle:
        self._safety.redeem(permit, call_id)           # raises SafetyViolation
        try:
            handle = self._provider.initiate_call(call_id, phone)
        except ProviderError:
            self._safety.record_dial_result(False)
            raise
        self._safety.record_dial_result(True)
        return handle

    # Risk-reducing operations do not need a permit.
    def hangup(self, provider_call_id: str) -> bool:
        try:
            self._provider.hangup(provider_call_id)
            return True
        except ProviderError:
            return False

    def lookup(self, call_id: str) -> Optional[ProviderCallStatus]:
        return self._provider.lookup(call_id)          # may raise ProviderError; caller decides

    def cancel_by_call_id(self, call_id: str) -> bool:
        """Best-effort: hang up whatever the provider has for our call id."""
        try:
            status = self._provider.lookup(call_id)
            if status is not None:
                self._provider.hangup(status.provider_call_id)
            return True
        except ProviderError:
            return False

    def poll_events(self):
        events = self._provider.poll_events()
        self._safety.record_events_received(len(events))   # liveness signal for the silence detector
        return events
