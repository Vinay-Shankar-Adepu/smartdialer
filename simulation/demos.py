"""Narrated walkthroughs of the assignment's failure cases, on the real engine.

    python main.py --demo crash          # worker crash: adopt; ANSWERED -> crash -> COMPLETED
    python main.py --demo outage         # provider times out: existing calls, new calls, retries, pacing
    python main.py --demo agent-drop     # 40 of 100 agents vanish within 5 s
    python main.py --demo duplicates     # the same event delivered again and again
    python main.py --demo out-of-order   # COMPLETED, ANSWERED, RINGING
    python main.py --demo all

Each step prints what the database says, so you can see the system end up
consistent rather than take a test's word for it.
"""
from typing import Callable, Optional

from core.clock import SimClock
from core.database import Database
from core.models import Borrower, ProviderEvent
from core.state_machines import AgentState, CallState, ProviderEventType as E
from dialer.call_allocator import SimulatedCrash
from dialer.engine import DialerEngine, EngineConfig
from dialer.pacing import PREDICTIVE, PROGRESSIVE
from providers.mock_base import BorrowerBehavior
from providers.provider_factory import create_provider


class World:
    """One shared database + provider, and any number of workers on it."""

    def __init__(self, agents: int, mode: str = PROGRESSIVE, answer_rate: float = 1.0,
                 talk_s: float = 60.0, borrowers: int = 50,
                 crash_hook: Optional[Callable[[str, str], None]] = None):
        self.clock = SimClock(0.0)
        self.db = Database(":memory:")
        self.behavior = BorrowerBehavior(answer_rate, talk_mean_s=talk_s)
        self.provider = create_provider("A", self.clock, self.behavior, seed=7)
        self.provider.failure_rate = 0.0
        self.mode = mode
        self.engine = self.worker("worker-1", crash_hook)
        for i in range(agents):
            self.engine.agents.add_agent(f"agent-{i:03d}")
        self.db.insert_borrowers(Borrower(id=f"b-{i:04d}", name=f"B{i}", phone=f"+91{i:010d}")
                                 for i in range(borrowers))

    def worker(self, worker_id: str, crash_hook=None) -> DialerEngine:
        return DialerEngine(self.db, self.clock, self.provider, self.mode,
                            EngineConfig(worker_id=worker_id), crash_hook)

    def run(self, seconds: int, engine: Optional[DialerEngine] = None, every=None):
        engine = engine or self.engine
        reports = []
        for _ in range(seconds):
            report = engine.tick()
            reports.append(report)
            if every:
                every(report)
            self.clock.advance(1.0)
        return reports

    def event(self, call_id: str, etype: E, event_id: str) -> ProviderEvent:
        call = self.db.get_call(call_id)
        return ProviderEvent(event_id, call_id, call.provider_call_id or f"pc-{call_id}", etype, self.clock.now())

    def show(self, call_id: str, label: str) -> None:
        call = self.db.get_call(call_id)
        held = [a for a in self.db.list_agents() if a.reserved_for_call_id == call_id]
        b = self.db.get_borrower(call.borrower_id)
        if held:
            agent_txt = f"{held[0].id} {held[0].state.value}"
        elif call.agent_id:
            agent = self.db.get_agent(call.agent_id)
            if agent.reserved_for_call_id:
                agent_txt = f"{agent.id} freed, now on next call"
            else:
                agent_txt = f"{agent.id} {agent.state.value}"
        else:
            agent_txt = "none"
        print(f"  t={self.clock.now():>6.1f}s  {label:<34} call={call.state.value:<10} "
              f"agent={agent_txt:<22} borrower={b.state.value} (attempts {b.attempts})")


def _title(text: str) -> None:
    print(f"\n{'=' * 100}\n{text}\n{'=' * 100}")


# ---------------------------------------------------------------- 1. worker crash
def demo_crash() -> None:
    _title("1. WORKER CRASH: agent reserved > borrower reserved > call initiated > worker killed")

    def kill_after_dial(point: str, call_id: str) -> None:
        if point == "after_dial":
            raise SimulatedCrash(f"kill -9 at {point}")

    w = World(agents=1, crash_hook=kill_after_dial)
    try:
        w.engine.tick()
    except SimulatedCrash as crash:
        print(f"  worker-1 died: {crash}  (no cleanup code ran)")
    call_id = w.db.find_calls(list(CallState))[0].id
    w.show(call_id, "right after the crash")
    print(f"  provider: lookup({call_id}) -> {w.provider.lookup(call_id)}   <- the borrower's phone IS ringing")

    survivor = w.worker("worker-2")
    print(f"  worker-2 sweep before the 10 s lease expires: {dict(survivor.recovery.sweep()) or 'nothing (hands off)'}")
    w.clock.advance(11)
    print(f"  worker-2 sweep after the lease expired:       {dict(survivor.recovery.sweep())}")
    w.show(call_id, "adopted: the agent stays with it")
    w.run(120, engine=survivor)
    w.show(call_id, "events drove it to the end")
    print("  -> nothing leaked: the borrower was reached by the agent that was reserved for them.")

    print("\n  Variant: provider sends ANSWERED, the worker crashes, then COMPLETED arrives")
    w = World(agents=1)
    w.engine.tick()
    call_id = w.db.find_calls(list(CallState))[0].id
    print(f"  worker-1 processes ANSWERED: {w.engine.events.process(w.event(call_id, E.ANSWERED, 'ev-ans'))}")
    w.show(call_id, "ANSWERED, not yet bridged")
    print("  worker-1 dies before bridging. worker-2 takes over:")
    survivor = w.worker("worker-2")
    survivor.lifecycle.bridge_answered_calls(survivor.gateway.hangup)
    w.show(call_id, "worker-2's bridge pass connected it")
    print(f"  provider redelivers ANSWERED: {survivor.events.process(w.event(call_id, E.ANSWERED, 'ev-ans'))}")
    w.clock.advance(30)
    print(f"  COMPLETED arrives:            {survivor.events.process(w.event(call_id, E.COMPLETED, 'ev-done'))}")
    w.show(call_id, "completed, agent wrapping up")


# ---------------------------------------------------------------- 2. provider outage
def demo_outage() -> None:
    _title("2. PROVIDER OUTAGE: the provider starts timing out for 60 s")
    w = World(agents=6, answer_rate=1.0, talk_s=600, borrowers=200)
    for i in range(6, 10):
        w.engine.agents.add_agent(f"agent-{i:03d}", logged_in=False)
    w.run(20)
    existing = w.db.find_calls([CallState.CONNECTED])
    print(f"  t={w.clock.now():.0f}s healthy: {len(existing)} borrowers talking to agents (10-minute calls)")
    w.provider.start_outage(60)
    for i in range(6, 10):
        w.engine.agents.login(f"agent-{i:03d}")
    print(f"  t={w.clock.now():.0f}s OUTAGE for 60 s: dials time out, webhooks stop, lookups fail.")
    print(f"        4 more agents log in now, so the dialer has work to attempt\n")

    errors, last = [], {}

    def line(r):
        if r.allocation:
            errors.append(r.allocation.outcomes["provider_failed"])
        state = w.engine.safety.breaker.state()
        if state != last.get("state") or r.safety.verdict.value != last.get("verdict"):
            last.update(state=state, verdict=r.safety.verdict.value)
            reason = (r.safety.reasons or [""])[0][:58]
            print(f"  t={w.clock.now():>4.0f}s  verdict={r.safety.verdict.value:<21} "
                  f"approved={r.safety.approved:<2} timeouts so far={sum(errors):<3} "
                  f"breaker after this tick={state:<9} {reason}")

    w.run(100, every=line)
    timed_out = [c for c in w.db.find_calls(list(CallState)) if c.failure_reason == "provider_timeout"]
    charged = sum(w.db.get_borrower(c.borrower_id).attempts for c in timed_out)
    print(f"\n  existing calls: " + ", ".join(f"{c.id[-3:]}={w.db.get_call(c.id).state.value}" for c in existing)
          + "  (never killed for the provider being quiet)")
    print(f"  new calls:      {len(timed_out)} dials timed out; each released its agent and borrower at once; "
          f"then the breaker stopped further dials")
    print(f"  retries:        attempts charged to those {len(timed_out)} borrowers: {charged} "
          f"(a provider fault is not the borrower's attempt); they become due again after 30 s")
    print(f"  pacing:         REJECTED while OPEN; HALF_OPEN lets one probe through; its success closes the breaker")


# ---------------------------------------------------------------- 3. agent drop
def demo_agent_drop() -> None:
    _title("3. AGENT AVAILABILITY DROPS: 100 agents, 40 vanish within 5 s (predictive mode)")
    w = World(agents=100, mode=PREDICTIVE, answer_rate=0.3, talk_s=90, borrowers=8000)
    w.run(120)
    print(f"  t={w.clock.now():.0f}s warmed up in predictive mode\n")
    victims = [a.id for a in w.db.list_agents() if a.state != AgentState.OFFLINE][:40]
    for second in range(8):
        if second < 5:
            for agent_id in victims[second * 8:(second + 1) * 8]:
                w.engine.agents.force_offline(agent_id)
        r = w.engine.tick()
        print(f"  +{second}s  logged in={r.safety.checks['logged_in']:<4} verdict={r.safety.verdict.value:<21} "
              f"approved={r.safety.approved:<3} {(r.safety.reasons or [''])[0][:70]}")
        w.clock.advance(1.0)
    print("\n  -> pacing recounts agents from the DB every tick; safety forces progressive as soon as the drop")
    print("     crosses 15% of the 30 s peak, and stays there for a 60 s cool-down.")


# ---------------------------------------------------------------- 4. duplicates
def demo_duplicates() -> None:
    _title("4. DUPLICATE EVENTS: the same provider event arrives multiple times")
    w = World(agents=1)
    w.engine.tick()
    call_id = w.db.find_calls(list(CallState))[0].id
    p = w.engine.events.process
    print(f"  ANSWERED (id ev-1):              {p(w.event(call_id, E.ANSWERED, 'ev-1'))}")
    version = w.db.get_call(call_id).version
    print(f"  ANSWERED (id ev-1) again:        {p(w.event(call_id, E.ANSWERED, 'ev-1'))}")
    print(f"  ANSWERED (id ev-1) again:        {p(w.event(call_id, E.ANSWERED, 'ev-1'))}")
    print(f"  ANSWERED (new id ev-2, resent):  {p(w.event(call_id, E.ANSWERED, 'ev-2'))}")
    print(f"  call row version unchanged: {version} -> {w.db.get_call(call_id).version}")
    w.engine.lifecycle.bridge_answered_calls(w.engine.gateway.hangup)
    w.clock.advance(30)
    print(f"  COMPLETED (id ev-3):             {p(w.event(call_id, E.COMPLETED, 'ev-3'))}")
    print(f"  COMPLETED (id ev-3) again:       {p(w.event(call_id, E.COMPLETED, 'ev-3'))}")
    w.show(call_id, "one transition per real change")


# ---------------------------------------------------------------- 5. out of order
def demo_out_of_order() -> None:
    _title("5. OUT-OF-ORDER EVENTS: COMPLETED, ANSWERED, RINGING  /  ANSWERED before RINGING")
    w = World(agents=1)
    w.engine.tick()
    call_id = w.db.find_calls(list(CallState))[0].id
    p = w.engine.events.process
    for etype, eid in ((E.COMPLETED, "o-1"), (E.ANSWERED, "o-2"), (E.RINGING, "o-3")):
        print(f"  {etype.value:<10} -> {p(w.event(call_id, etype, eid))}")
    w.show(call_id, "terminal state absorbs late events")

    w.clock.advance(1)
    w.engine.tick()
    call_id = [c for c in w.db.find_calls(list(CallState)) if not c.is_terminal][0].id
    for etype, eid in ((E.ANSWERED, "q-1"), (E.RINGING, "q-2")):
        print(f"  {etype.value:<10} -> {p(w.event(call_id, etype, eid))}")
    w.engine.lifecycle.bridge_answered_calls(w.engine.gateway.hangup)
    w.show(call_id, "forward jump applied, stale one ignored")


DEMOS = {
    "crash": demo_crash,
    "outage": demo_outage,
    "agent-drop": demo_agent_drop,
    "duplicates": demo_duplicates,
    "out-of-order": demo_out_of_order,
}


def run_demo(name: str) -> None:
    for key, fn in DEMOS.items():
        if name in (key, "all"):
            fn()
