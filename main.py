"""SmartDialer CLI.

    python main.py --mode progressive --scenario A
    python main.py --mode predictive  --scenario D --duration 3600
    python main.py --compare                     # every scenario x both modes, summary table
"""
import argparse
import json
import sys
import time

from simulation.scenarios import SCENARIOS
from simulation.simulator import Simulator


def run_one(mode: str, scenario_key: str, duration: int, seed: int, explain: bool, quiet: bool = False):
    started = time.perf_counter()
    sim = Simulator(mode=mode, scenario=SCENARIOS[scenario_key], seed=seed)
    sim.run(duration=duration)
    if not quiet:
        sim.print_results(explain=explain)
        print(f"  (simulated {duration}s in {time.perf_counter() - started:.1f}s wall time)\n")
    return sim.results()


def compare(duration: int, seed: int) -> list:
    rows = []
    for key in SCENARIOS:
        for mode in ("progressive", "predictive"):
            print(f"running scenario {key} / {mode} ...", file=sys.stderr)
            rows.append(run_one(mode, key, duration, seed, explain=False, quiet=True))
    print(f"\nSmartDialer comparison: {duration}s per run, seed={seed}\n")
    header = (f"{'scn':<4}{'mode':<12}{'util':>7}{'dials/min':>11}{'conn/min':>10}"
              f"{'answered':>10}{'abandon':>9}{'prov.err':>9}  safety verdicts (ticks)")
    print(header)
    print("-" * len(header) + "-" * 40)
    for r in rows:
        verdicts = ", ".join(f"{k}={v}" for k, v in sorted(r["verdicts"].items()))
        print(f"{r['scenario']:<4}{r['mode']:<12}{r['utilization']:>7.1%}{r['dials_per_min']:>11.1f}"
              f"{r['connects_per_min']:>10.1f}{r['answered']:>10d}{r['abandon_rate']:>9.2%}"
              f"{r['provider_errors']:>9d}  {verdicts}")
    print()
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SmartDialer: progressive / predictive dialer simulation")
    parser.add_argument("--mode", choices=["progressive", "predictive"], default="progressive")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="A")
    parser.add_argument("--duration", type=int, default=3600, help="simulated seconds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare", action="store_true", help="run all scenarios in both modes")
    parser.add_argument("--no-explain", action="store_true", help="hide sample pacing explanations")
    parser.add_argument("--json", metavar="PATH", help="also write results as JSON")
    args = parser.parse_args(argv)

    if args.compare:
        results = compare(args.duration, args.seed)
    else:
        results = run_one(args.mode, args.scenario, args.duration, args.seed, explain=not args.no_explain)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
