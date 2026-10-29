# SmartDialer

A collections dialing engine with **progressive** and **predictive** pacing behind a **Safety Controller that cannot be bypassed**. It includes atomic agent reservation, idempotent and order-tolerant provider event handling, crash recovery, and a discrete-time simulator.

Built for the CredResolve technical challenge. Pure Python 3.9+ standard library: SQLite, no frameworks.

Design, trade-offs, bottleneck analysis and the answer to *"predictive's utilization with progressive's safety"* are in **[ARCHITECTURE.md](ARCHITECTURE.md)**.

## Setup

```bash
cd smartdialer
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # nothing to install: stdlib only
```

## Run a simulation

```bash
python main.py --mode progressive --scenario A
python main.py --mode predictive  --scenario D --duration 3600
python main.py --compare                               # all scenarios x both modes, summary table
python main.py --mode predictive --scenario C --json results/c.json --seed 7
```

| Flag | Default | Meaning |
|---|---|---|
| `--mode` | `progressive` | `progressive` or `predictive` |
| `--scenario` | `A` | see the scenario table below |
| `--duration` | `3600` | simulated seconds (1 h takes ~10–70 s of wall time) |
| `--seed` | `42` | everything is deterministic for a given seed |
| `--compare` | off | run every scenario in both modes |
| `--no-explain` | off | hide the sample pacing explanations |
| `--json PATH` | - | also write metrics as JSON |

Each run prints utilization, calls dialed (initiated), calls connected, abandon rate, provider errors, safety verdict counts, event-handling outcomes, recovery actions, a 5-minute timeline, and sample pacing decisions that explain why N calls were dialed.

Scenarios A–C use the answer rates and talk times from the assignment's table. Every scenario has provider latency and failures.

| Scenario | Answer rate | Avg talk | Agents | Provider | Disruption |
|---|---|---|---|---|---|
| A | 20% | 120 s | 50 | A (0.1 s latency, 5% failed dials) | - |
| B | 50% | 90 s | 50 | A | - |
| C | 70% | 180 s | 50 | B (0.5 s + 0–3 s jitter, 10% timeouts, duplicate / out-of-order / stale events) | 120 s full outage at t=1200 s |
| D | 50% -> 70% -> 10% | 90 s -> 180 s -> 60 s | 100 | A | rates change at t=900 s and t=2400 s; 40 of 100 agents vanish within 5 s at t=1500 s and return at t=2100 s |

## Run the load test

```bash
python -m simulation.load_test           # ~5 min; --quick for ~30 s
```

Part 1 races 1–16 threads, each with its own SQLite connection, to reserve every agent, then checks that none was double-booked. Part 2 times the full control loop per 1 s tick at 100, 1,000 and 10,000 agents. Output: [`results/load_test.txt`](results/load_test.txt).

## Run the tests

```bash
python -m unittest discover -s tests -t . -v
```

There are 75 tests and they run in about 12 s. Each failure scenario has its own file:

| File | Scenario |
|---|---|
| `tests/test_concurrent_reservation.py` | Two workers, same agent: exactly one reserves (real multi-connection SQLite); only one worker paces at a time, with failover |
| `tests/test_worker_crash.py` | Agent RESERVED -> Borrower RESERVED -> dialed -> crash: adopted or released |
| `tests/test_provider_outage.py` | Timeouts, circuit breaker, silent provider, ghost calls, in-flight calls survive, lost hangup event |
| `tests/test_agent_disappearance.py` | 100 agents, 40 vanish in 5 s: fallback to progressive, calls re-routed |
| `tests/test_duplicate_events.py` | Duplicate events (same id, new id, concurrent): one transition |
| `tests/test_out_of_order_events.py` | ANSWERED before RINGING, COMPLETED before ANSWERED, 15 min of Provider B chaos |
| `tests/test_predictive_logic.py` | Pacing formula, "why 17 instead of 10", safety caps, answer rate 70% -> 10% and 10% -> 70% |
| `tests/test_safety_bypass.py` | Forged, tampered, expired, over-budget and replayed permits are all refused |

`tests/helpers.py` also provides `check_invariants()`, a cross-table consistency check. Among other things, it confirms that no agent is reserved twice and no busy agent points at a dead call. The fuzz and scenario tests run it after every simulated second.

## Results (1 h simulated, seed 42)

```
scn mode           util  dials/min  conn/min  answered  abandon prov.err
A   progressive   50.9%       70.7      13.0       782    0.00%        0
A   predictive    77.4%      106.1      19.8      1193    0.59%        0
B   progressive   67.8%       47.6      23.6      1417    0.00%        0
B   predictive    78.6%       56.5      27.2      1646    0.67%        0
C   progressive   84.4%       23.4      14.2       855    0.00%      136
C   predictive    88.8%       24.9      15.1       915    0.98%      144
D   progressive   59.4%      111.0      29.5      1771    0.00%        0
D   predictive    76.1%      237.2      43.7      2631    0.34%        0
```

Predictive dialing behind the safety controller keeps abandonment **under 1%** (target: 3%). Its utilization gain depends on the answer rate: **+26 points at 20%**, +11 at 50%, +4 at 70%. Full outputs, including safety verdicts per run, are in [`results/`](results/).

## Project layout

```
smartdialer/
├── main.py                   CLI
├── core/
│   ├── state_machines.py     Agent / Call / Borrower states, transition tables, event classifier
│   ├── models.py             dataclasses
│   ├── database.py           SQLite schema, CAS update, queries
│   ├── lock_manager.py       optimistic (version) locking: reserve / release / transition
│   └── clock.py              real vs simulated time
├── dialer/
│   ├── progressive_pacing.py one call per idle agent
│   ├── predictive_pacing.py  explainable overdial: answer rate x free agents x safety stock
│   ├── safety_controller.py  verdicts, signed permits, circuit breaker, SafeDialGateway
│   ├── call_allocator.py     borrower -> (agent) -> dial, crash-safe ordering
│   ├── call_lifecycle.py     finalize / bridge ANSWERED -> CONNECTED / abandon
│   ├── event_processor.py    idempotent, order-tolerant provider events
│   ├── recovery.py           lease reaper, stuck calls, orphans
│   ├── agent_service.py      login / logout / wrap-up / disappearance
│   ├── snapshot.py           read model for pacing and safety
│   ├── estimators.py         Wilson bound, smoothing, binomial line sizing
│   ├── pacing.py             PacingDecision type
│   └── engine.py             wires it all; one control-loop tick
├── providers/
│   ├── provider_interface.py abstract provider (idempotent initiate, lookup, hangup, events)
│   ├── mock_base.py          borrower behavior model + mock telecom mechanics
│   ├── provider_a.py         fast, reliable, ordered
│   ├── provider_b.py         slow, timeouts, ghost calls, duplicates, out-of-order, stale
│   └── provider_factory.py
├── simulation/
│   ├── scenarios.py          Scenarios A–D
│   ├── simulator.py          runs the real engine on a simulated clock
│   └── load_test.py          reservation race + control-loop cost at 100 / 1K / 10K agents
├── tests/                    failure-scenario tests (see above)
└── results/                  saved simulation output
```

A few modules go beyond the suggested structure (`call_lifecycle`, `event_processor`, `recovery`, `agent_service`, `snapshot`, `estimators`, `engine`, `clock`, `mock_base`). Each has one job, so the allocator, pacing and safety files stay focused.
