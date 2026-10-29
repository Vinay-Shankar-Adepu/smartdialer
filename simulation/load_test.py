"""Basic load test: where does the single-node design start to hurt?

    python -m simulation.load_test            # both parts, ~1-2 min
    python -m simulation.load_test --quick    # smaller sizes

Part 1 - reservation race. W threads, each with its OWN SQLite connection to a
file database (WAL), race to reserve every one of N agents. Checks that no
agent is double-booked and that every reservation is unique, and measures
reservations/s and how many CAS attempts lose (hot-row contention: every
worker targets the same longest-idle agents).

Part 2 - control-loop cost. Runs the full engine (events -> bridge -> sweep ->
pacing -> safety -> allocator) against Provider A on a simulated clock with
100 / 1,000 / 10,000 agents, and measures wall-clock milliseconds per 1-second
tick. A tick that takes longer than 1 s means the loop can no longer keep up
in real time - that is the first thing to break.
"""
import argparse
import os
import shutil
import statistics
import sys
import tempfile
import threading
import time
from collections import Counter

from core.clock import SimClock, SystemClock
from core.database import Database
from core.lock_manager import LockManager
from core.models import Agent, Borrower
from core.state_machines import AgentState
from dialer.engine import DialerEngine, EngineConfig
from dialer.safety_controller import SafetyConfig
from providers.mock_base import BorrowerBehavior
from providers.provider_factory import create_provider


# ---------------------------------------------------------------- part 1
def reservation_race(num_agents: int, num_workers: int, db_dir: str) -> dict:
    path = os.path.join(db_dir, f"race-{num_agents}-{num_workers}.db")
    db = Database(path)
    with db.transaction():
        for i in range(num_agents):
            db.insert_agent(Agent(id=f"a{i:05d}", name=f"a{i}", state=AgentState.AVAILABLE,
                                  state_changed_at=float(i)))
    locks = LockManager(db, SystemClock())
    wins: Counter = Counter()
    losses: Counter = Counter()
    barrier = threading.Barrier(num_workers)

    def worker(w: int) -> None:
        barrier.wait()
        n = 0
        while True:
            candidates = db.find_available_agents(limit=8)
            if not candidates:
                return
            for agent in candidates:
                n += 1
                if locks._cas_reserve(agent, f"call-w{w}-{n}"):
                    wins[w] += 1
                    break
                losses[w] += 1

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(num_workers)]
    started = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - started

    reserved = db.query("SELECT reserved_for_call_id FROM agents WHERE state = ?", [AgentState.RESERVED])
    call_ids = [r["reserved_for_call_id"] for r in reserved]
    total_wins = sum(wins.values())
    total_losses = sum(losses.values())
    return {
        "agents": num_agents, "workers": num_workers, "elapsed_s": elapsed,
        "reservations": total_wins, "per_s": total_wins / elapsed,
        "cas_losses": total_losses, "loss_pct": total_losses / max(total_wins + total_losses, 1),
        "all_reserved": len(call_ids) == num_agents,
        "double_booked": total_wins - len(set(call_ids)),   # a call id holding >1 agent, or a lost write
    }


# ---------------------------------------------------------------- part 2
def control_loop_cost(num_agents: int, warmup_s: int, measure_s: int, db_dir: str) -> dict:
    clock = SimClock(0.0)
    db = Database(os.path.join(db_dir, f"loop-{num_agents}.db"))
    behavior = BorrowerBehavior(answer_rate=0.20, talk_mean_s=120)
    provider = create_provider("A", clock, behavior, seed=1)
    # The default 50-dials-per-tick blast-radius cap is sized for ~100 agents; scale it so the
    # test measures the database and the loop, not that cap.
    safety = SafetyConfig(max_dials_per_tick=max(50, num_agents // 2))
    engine = DialerEngine(db, clock, provider, "predictive", EngineConfig(safety=safety))
    with db.transaction():
        for i in range(num_agents):
            db.insert_agent(Agent(id=f"a{i:05d}", name=f"a{i}", state=AgentState.AVAILABLE))
    db.insert_borrowers(Borrower(id=f"b{i:07d}", name=f"B{i}", phone=f"+91{i:010d}")
                        for i in range(num_agents * 30))

    tick_ms, dials, events = [], 0, 0
    for second in range(warmup_s + measure_s):
        t0 = time.perf_counter()
        report = engine.tick()
        took = (time.perf_counter() - t0) * 1000
        clock.advance(1.0)
        if second >= warmup_s:
            tick_ms.append(took)
            events += sum(report.events.values())
            dials += report.allocation.dialed if report.allocation else 0

    tick_ms.sort()
    return {
        "agents": num_agents, "ticks": len(tick_ms),
        "avg_ms": statistics.mean(tick_ms), "p95_ms": tick_ms[int(len(tick_ms) * 0.95) - 1],
        "max_ms": tick_ms[-1], "dials_per_s": dials / measure_s, "events_per_s": events / measure_s,
    }


# ---------------------------------------------------------------- CLI
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SmartDialer basic load test")
    parser.add_argument("--quick", action="store_true", help="smaller sizes (~15 s)")
    args = parser.parse_args(argv)

    race_sizes = [(1000, 1), (1000, 4), (1000, 16)] if args.quick else \
        [(1000, 1), (1000, 4), (1000, 16), (5000, 16)]
    loop_sizes = [(100, 60, 60), (1000, 60, 30)] if args.quick else \
        [(100, 60, 120), (1000, 60, 60), (10000, 30, 20)]

    db_dir = tempfile.mkdtemp(prefix="smartdialer-load-")
    try:
        print("Part 1 - reservation race (file SQLite, WAL, one connection per thread)\n")
        print(f"{'agents':>7}{'workers':>9}{'res/s':>9}{'CAS losses':>12}{'loss %':>8}"
              f"{'all reserved':>14}{'double-booked':>15}")
        ok = True
        for n, w in race_sizes:
            r = reservation_race(n, w, db_dir)
            ok &= r["all_reserved"] and r["double_booked"] == 0
            print(f"{r['agents']:>7}{r['workers']:>9}{r['per_s']:>9.0f}{r['cas_losses']:>12}"
                  f"{r['loss_pct']:>8.1%}{str(r['all_reserved']):>14}{r['double_booked']:>15}")

        print("\nPart 2 - control-loop cost per 1 s tick (predictive, Provider A, 20% answer, 120 s talk)\n")
        print(f"{'agents':>7}{'avg ms':>9}{'p95 ms':>9}{'max ms':>9}{'dials/s':>10}{'events/s':>10}  real-time?")
        for n, warm, measure in loop_sizes:
            print(f"  ... {n} agents", file=sys.stderr)
            r = control_loop_cost(n, warm, measure, db_dir)
            verdict = "yes" if r["p95_ms"] < 1000 else "NO - tick slower than 1 s"
            print(f"{r['agents']:>7}{r['avg_ms']:>9.1f}{r['p95_ms']:>9.1f}{r['max_ms']:>9.1f}"
                  f"{r['dials_per_s']:>10.1f}{r['events_per_s']:>10.1f}  {verdict}")
        print("\nreservation correctness:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        shutil.rmtree(db_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
