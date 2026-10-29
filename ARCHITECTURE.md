# SmartDialer: Architecture

This document covers how the system is built, why it is built that way, what it costs, and what breaks first at scale.

Guiding rule: **correctness over cleverness.** Every risky action goes through one choke point (the Safety Controller), every write is a compare-and-swap, and every provider event is applied idempotently.

---

## 1. System architecture

```
                     +-----------------------------------------------------------+
                     |   DialerEngine.tick()  (runs on every worker; pacing,     |
                     |   safety and allocation only on the pacer-lease holder)   |
                     +-----------------------------------------------------------+
                                  |                                   ^
                    Snapshot (own read of the DB)                      |
                                  v                                   |
 +----------------+   PacingDecision   +--------------------+         |
 | Pacing engine  | -----------------> |  Safety Controller | -- independent Snapshot (own DB read)
 | (pure function)|  "dial 17, because |  - circuit breaker |         |
 |  progressive / |   ..."             |  - agent-drop det. |         |
 |  predictive    |                    |  - abandon guard   |         |
 +----------------+                    |  - capacity caps   |         |
   no DB, no provider,                 +--------------------+         |
   no allocator                           |  DialPermit (HMAC-signed, |
                                          |  N calls, 5 s TTL)       |
                                          v                          |
                                 +------------------+                |
                                 |  Call Allocator  |  claim borrower (CAS) -> call row
                                 |                  |  -> reserve agent (CAS, progressive only)
                                 +------------------+  -> call RESERVED
                                          |  dial(permit, call_id)   |
                                          v                          |
                                 +------------------+  redeem(): permit genuine? unexpired?
                                 | SafeDialGateway  |  budget left? call RESERVED? agent owned?
                                 | (only holder of  |                |
                                 |  the provider)   |                |
                                 +------------------+                |
                                          |  initiate_call(call_id = idempotency key)
                                          v                          |
                                 +------------------+   events (dup / out-of-order)
                                 | Telecom provider | ---------------+-----> EventProcessor
                                 |  A (clean) / B   |                        (dedup + state machine)
                                 +------------------+                              |
                                                                                   v
     RecoveryService (every 5 s): expired leases -> adopt or cancel;      CallLifecycle: bridge
     stuck calls -> lookup / hang up; orphan agents & borrowers -> release; ANSWERED -> CONNECTED,
     CONNECTED for 30+ min -> ask provider, complete if the hangup was lost  or ABANDONED after 2 s
```

Separation of concerns:

| Component | Knows about | Does NOT know about |
|---|---|---|
| `PredictivePacing` / `ProgressivePacing` | a `Snapshot` (numbers) | DB, provider, allocator, permits |
| `SafetyController` | DB (its own reads), permits, breaker | how calls are allocated |
| `CallAllocator` | DB, locks, gateway | how many calls to make (the permit says) |
| `SafeDialGateway` | the provider, the safety controller | anything else |
| `EventProcessor` | state machine, lifecycle | pacing, safety |
| `RecoveryService` | leases, provider lookup | pacing |

---

## 2. State machines

### Agent

```
                 login
   +---------+ --------> +-----------+  reserve (CAS)  +----------+  provider accepted  +---------+
   | OFFLINE |           | AVAILABLE | --------------> | RESERVED | ------------------> | DIALING |
   +---------+ <-------- +-----------+ <-------------- +----------+ <------------------ +---------+
        ^       logout     |      ^   release (dial         |         no answer / busy /     |
        |                  | pause|   failed, cancelled)    |         failed: release        |
        |                  v      |                         | predictive bridge              | answered:
        |               +--------+                          | (answered call grabs           | bridge
        |               | PAUSED |                          |  an idle agent)                |
        |               +--------+                          v                                |
        |                                             +-----------+ <------------------------+
        |   any state -> OFFLINE on disappearance     | CONNECTED |
        +-------------------------------------------- +-----------+
                                                            | call ends
                                                            v
                                     AVAILABLE <---- +---------+ ----> PAUSED / OFFLINE
                                     (wrap_until)    | WRAP_UP |       (logout_requested)
                                                     +---------+
```

* **RESERVED** = exclusively held for exactly one call (`reserved_for_call_id`). Only one CAS can win it.
* Busy states (`RESERVED`, `DIALING`, `CONNECTED`) must always point at a live call. This is checked by `tests/helpers.check_invariants` after every simulated tick.
* Releasing is ownership-checked: only the call holding the reservation can release it.

### Call

```
 QUEUED --> RESERVED --> INITIATED --> RINGING --> ANSWERED --> CONNECTED --> COMPLETED
   |           |  \           |  \        |  \        |   \           |
   |           |   \--------->+---\------>+---\------>+    \--> ABANDONED (no agent within 2 s)
   v           v      forward jumps allowed (missing/late events)       |
 CANCELLED  CANCELLED / FAILED            any pre-connect state --> COMPLETED / FAILED / CANCELLED
                                                                   CONNECTED --> COMPLETED / FAILED
 Terminal: COMPLETED, FAILED(reason), CANCELLED, ABANDONED. Nothing leaves a terminal state.
```

| State | Meaning | Who owns it |
|---|---|---|
| QUEUED | row created for a claimed borrower | worker (lease) |
| RESERVED | worker holds the borrower (and the agent, if progressive); about to dial | worker (lease) |
| INITIATED | provider accepted the request | event pipeline |
| RINGING | borrower's phone rings | event pipeline |
| ANSWERED | borrower picked up; **not yet bridged** | bridge pass |
| CONNECTED | borrower talking to an agent | event pipeline |
| ABANDONED | answered, but no agent within `max_answer_hold_s`. This is the predictive risk, and it is measured | - |

**ANSWERED != CONNECTED.** The event processor records ANSWERED. A separate, idempotent bridge pass connects it. If the worker dies in between, the next pass connects the call. If nobody can connect it within 2 s, it becomes ABANDONED and is hung up, so the borrower never sits on a silent line.

### Event classification (pure function, `classify_event_transition`)

| Current state vs event target | Decision | Example |
|---|---|---|
| current is terminal | `IGNORED_TERMINAL` | ANSWERED after COMPLETED |
| same state | `NOOP_SAME_STATE` | second ANSWERED with a new event id |
| legal forward transition | `APPLIED` | ANSWERED before RINGING (jump) |
| lower rank | `IGNORED_STALE` | RINGING after ANSWERED |
| otherwise | `REJECTED_ILLEGAL` (logged) | event for a QUEUED call |

On top of that, `provider_events.event_id` is a PRIMARY KEY, so an exact redelivery is rejected by `INSERT OR IGNORE` before the state machine is even consulted. The dedup insert and the state change share one transaction.

---

## 3. Key design decisions

### 3.1 Optimistic locking (version CAS), not pessimistic locks

```sql
UPDATE agents SET state='RESERVED', reserved_for_call_id=?, version=version+1
WHERE id=? AND version=? AND state='AVAILABLE' AND reserved_for_call_id IS NULL
-- rowcount == 1 -> you won; 0 -> someone else did, try the next agent
```

* **No lock is held across I/O.** The provider call (100–500 ms) happens after the reservation is committed. A pessimistic `SELECT ... FOR UPDATE` held during dialing would serialize every worker behind the slowest provider round-trip.
* **Crash-safe by construction.** A dead worker leaves no held lock behind, only a row with an expired **lease**, which the reaper handles.
* **Losing a race is cheap.** `reserve_any_available_agent` walks the 8 longest-idle candidates and moves on after losing one. Contention does grow with workers, because they all target the same longest-idle agents: the load test measures 0% lost CAS attempts with 1 worker, roughly 7–13% with 4 and 27–36% with 16 (it varies run to run), with zero double-bookings in every run (section 6).
* **It ports directly.** The same pattern works on Postgres (`UPDATE ... WHERE version=`), DynamoDB (conditional writes) and Redis (`WATCH`/Lua).
* **Ownership on top of versions.** Releases check `reserved_for_call_id = me`, so a slow worker waking up late can't free a reservation that now belongs to another call.

Multi-row units of work (finalize a call, release its agent, settle its borrower) run in one `BEGIN IMMEDIATE` transaction, so they commit together or not at all.

### 3.2 Leases plus a reaper for crash recovery

A worker owns a call only while it is QUEUED/RESERVED (`lease_expires_at = now + 10 s`). Once the provider accepts it, events own it. When a lease expires, the reaper:

| Situation | Action |
|---|---|
| QUEUED (never dialed) | cancel, release agent and borrower, no attempt consumed |
| RESERVED, provider says "I have call `<our id>`" | **adopt** it: mark INITIATED and keep the agent (the borrower is already ringing) |
| RESERVED, provider says "never heard of it" | cancel and release |
| provider unreachable | try again on the next sweep |

This works because **our call id is the provider's idempotency key and client reference**. `initiate_call(call_id)` is idempotent, `lookup(call_id)` answers "did that dial go out?", and every event carries our call id. Provider events alone can heal a crashed call: an event for a RESERVED call moves it forward and clears the lease.

**A slow worker is not a dead worker.** If a worker stalls past its lease, the reaper cancels its call. When the worker wakes up and tries to dial, `redeem()` sees the call is no longer RESERVED and raises `SafetyViolation`, so the provider is never reached. Tested in `test_slow_worker_cannot_dial_after_reaper_cancelled_its_call`.

### 3.3 The Safety Controller cannot be bypassed

The guarantee is structural; it does not depend on callers following a convention:

1. Pacing engines are pure functions with no DB, provider or allocator reference. Tested: `test_pacing_engines_have_no_way_to_dial`.
2. Only `SafeDialGateway` holds the provider. Tested: `test_only_the_gateway_holds_the_provider`.
3. `gateway.dial()` calls `safety.redeem(permit, call_id)` first. A permit must be **genuine** (HMAC with a per-process secret plus an outstanding-permit registry), **unexpired** (5 s TTL), **within budget** (N calls), used **once per call**, and the call must be RESERVED and undialed in the DB. A progressive permit additionally requires the call to own a RESERVED agent.
4. Safety reads its own snapshot from the DB and never trusts pacing's numbers.

Every recommendation and verdict is written to `pacing_decisions` (an audit trail of what was asked, what was allowed and why).

The checks, in order:

| # | Check | Effect |
|---|---|---|
| a | Provider silent: accepted dials, no webhook for 15 s | breaker OPEN -> REJECTED |
| b | Breaker OPEN (5 consecutive failures, or >=50% of the last 50) | REJECTED; HALF_OPEN after 30 s: one probe call |
| c | Progressive request | cap = idle agents - unbound in-flight calls |
| d | Predictive, provider degraded (>=25% failures) | FALLBACK_PROGRESSIVE |
| e | Predictive, agents dropped >=15% of the 30 s peak | FALLBACK_PROGRESSIVE (+60 s cool-down) |
| f | Predictive, answer burst: 3+ answered callers waiting with no free agent | FALLBACK_PROGRESSIVE (+60 s cool-down) |
| g | Predictive, abandon rate > 3% (5-min window) | FALLBACK_PROGRESSIVE (+60 s cool-down) |
| h | Predictive, fewer than 30 outcomes in the window (cold start) | FALLBACK_PROGRESSIVE |
| i | Predictive capacity: `(ringing + n) * p_safety <= free * 1.15`, with `p_safety = max(5-min upper bound, 1-min point estimate)` | REDUCED |
| j | Blast radius: at most 5 lines per free agent in flight | REDUCED |
| k | Ramp: at most 0.5 *new* lines per free agent per tick, so exposure builds over several seconds and the first answers are seen before every line is committed | REDUCED |
| l | At most 50 dials per tick (provider CPS) | REDUCED |

`FALLBACK_PROGRESSIVE` issues a *progressive* permit. Every call it pays for must own a reserved agent before dialing, so abandonment becomes impossible.

Every REDUCED verdict names the limit that actually applied (capacity, lines per agent, ramp or tick cap) and its numbers, so "why only 5?" is answered in the log, too.

### 3.4 Predictive pacing you can explain

Each decision is the result of a short calculation, and every intermediate value goes into the explanation:

1. **p** = answer rate over the last 5 min, smoothed with a prior (10 imaginary calls at 30%), then the **upper** bound of an 80% Wilson interval. Overestimating p makes us dial *less*, the safe direction.
2. **L** = average dial-to-answer time. This is how far ahead we must dial.
3. **free** = idle + wrap-ups ending within L + connected x max(0, (L - wrap)/talk) - answered callers already waiting.
4. **N_total** = largest N with `N*p + k*sqrt(N*p*(1-p)) <= free * aggression`. Expected answers plus k=1 standard deviation of binomial noise must fit within the free agents.
5. **aggression** = 1.0 while the abandon rate is under 1.5%, sliding to 0.5 at 3%.
6. **dial** = N_total - lines already ringing without an agent.

> **"Why did you dial 17 instead of 10?"** (`tests/test_predictive_logic.py::test_why_seventeen_not_ten`)
> 2 idle agents + 6 finishing wrap-up within L = 10 s gives 8 free agents in both cases.
> At a 33% answer rate, p = 0.35 (upper bound), and 17.3 lines are needed for 8 answers (+1 sigma), so we dial 17.
> At a 60% answer rate, p = 0.62, so 10 lines suffice.
> The difference comes entirely from the answer rate.

Sample line from a live run:

```
Predictive: dial 12. p=0.34 (upper bound; estimate 0.30 from 374 calls), L=12.1s;
free agents = 1 idle + 4 wrap-up + 0.8 talking - 0 waiting = 5.8; x aggression 1.00
-> 12.3 lines (k=1) - 0 already ringing = 12
```

### 3.5 SQLite (and a simulated clock)

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| In-memory dicts + locks | fastest | no atomicity across process crashes, no real concurrency story, hides bugs | rejected |
| **SQLite** | real transactions, real multi-connection CAS semantics (tests use a file DB with a connection per thread), zero setup, inspectable | single writer; one host | **chosen for the challenge** |
| Postgres | multi-node, row-level locks, SKIP LOCKED | setup cost for a take-home | production target; same SQL pattern |

The simulator uses `:memory:` with one connection for speed (1 h simulated in about 10–70 s). The concurrency tests use a file DB with one connection per thread, so they exercise SQLite's real locking.

Time comes from an injected `Clock`. Production would use `SystemClock`; tests and simulations use `SimClock`. That makes every failure scenario deterministic and reproducible (`--seed`). Provider latency shifts event timestamps instead of calling `time.sleep()`.

### 3.6 Multiple workers: many executors, one pacer

Most of the work is safe on any number of workers because it is CAS-guarded or idempotent: processing events, bridging, recovery sweeps, claiming borrowers, reserving agents.

Pacing is the exception. Two safety controllers that both read "10 free agents" would each approve overdial for those same 10 agents, and the campaign would ring twice as many lines as planned. Progressive mode is immune (every call must win an agent CAS), but predictive is not.

So each tick, a worker first tries to take or renew a 3 s **pacer lease**: one atomic upsert on a `leases` row (`... WHERE holder = me OR expires_at <= now`). Only the holder runs pacing, safety and the allocator. If it dies, the lease expires and another worker takes over within 3 s (`SinglePacerTest`). Kafka, Redis or ZooKeeper aren't needed for this; the database we already trust for reservations does it in one statement.

| Concern | How it's handled |
|---|---|
| Agent allocation | version CAS on the agent row |
| Borrower allocation | CAS claim with a lease (`claim_borrower`), so two workers never call the same borrower |
| Duplicate jobs | our call id is the provider's idempotency key; a permit can be redeemed once per call |
| Retries | provider faults don't consume an attempt; no-answers retry after 15 min, abandons after 60 s; max 3 attempts |
| Concurrent updates | every write is a CAS or ownership-checked; multi-row changes are one transaction |
| Worker crashes | call leases + reaper (adopt or cancel); pacer lease failover |
| Stale state | agent, call and borrower state is re-read from the DB on every tick; writes check versions, so a stale read can only lose a CAS |
| Provider events | event-id dedup + rank-based state machine, handled by any worker |

---

## 4. Failure scenarios and how each is handled

**1. Worker crash** (Agent reserved -> Borrower reserved -> Call initiated -> crash). `test_worker_crash.py` crashes at all 5 points in that sequence.
The call row carries a 10 s lease. When it expires, the reaper asks the provider "do you have call `<our id>`?":
* yes: **adopt** it (INITIATED, the agent stays with it, events drive it from there);
* no: cancel, release the agent and the borrower, and burn no attempt;
* provider unreachable: ask again on the next sweep.

Events alone also heal it: a RINGING/ANSWERED event for a RESERVED call moves it forward and clears the lease. A worker that was only slow, not dead, can't dial later, because its permit redemption sees the call is no longer RESERVED.

**2. Provider outage** (`test_provider_outage.py`, scenario C)
* **New calls:** dials that time out release their agent and borrower immediately. 5 consecutive failures (or >= 50% of the last 50) open the circuit breaker, and safety REJECTs every dial. After 30 s one probe call is allowed; success closes the breaker. Accepted dials with no webhooks for 15 s also open it ("silent provider").
* **Existing calls:** not killed. A quiet provider doesn't mean the borrower hung up. Calls wait, and the delayed events flush when the provider returns. Only after a 300 s hard deadline with the provider still unreachable are they failed locally so agents aren't stuck forever.
* **Retries:** a provider failure is not the borrower's fault. The borrower goes back to PENDING with **no attempt consumed** and a 30 s delay, which spreads retries out instead of hammering a recovering provider. A provider timeout can leave a "ghost" call the provider did place; it is found by our call id and hung up.
* **Pacing:** keeps recommending, but safety rejects everything while the breaker is open, and a degraded provider (>= 25% failures) forces predictive into progressive.

**3. Agent availability drops** (100 agents, 40 vanish within 5 s; `test_agent_disappearance.py`, scenario D).
`force_offline` unbinds calls that haven't been answered yet (they can be bridged to another agent) and ends live ones. Pacing re-reads the DB every 1 s tick, so the next tick already plans for the smaller team. Safety sees "logged-in fell >= 15% of the 30 s peak" and forces progressive. The test asserts this happens by the **2nd tick** (2 s) of the drop, once 16 of 100 agents are gone.

**4. Duplicate events** (`test_duplicate_events.py`). The event id is a PRIMARY KEY, so an exact redelivery is dropped. A duplicate with a new id meets the state machine and becomes a no-op. Four threads delivering the same event concurrently produce exactly one transition. Scenario C absorbs thousands of Provider B duplicates.

**5. Out-of-order events** (`test_out_of_order_events.py`). Each event is classified against the call's current state by rank: forward jumps apply (ANSWERED before RINGING), backward moves are ignored (RINGING after ANSWERED), terminal states absorb everything (ANSWERED after COMPLETED). The assignment's sequences "ANSWERED x3, COMPLETED" and "COMPLETED, ANSWERED, RINGING" are both tested, plus 15 min of Provider B chaos with every invariant checked after each simulated second.

Also tested: two workers reserving the same agent (`test_concurrent_reservation.py`), answer rate 70% -> 10% and 10% -> 70% (`test_predictive_logic.py`), a lost COMPLETED event (`test_provider_outage.py`), and forged, tampered, expired, over-budget, foreign and replayed permits (`test_safety_bypass.py`).

---

## 5. Simulation results (1 h simulated, seed 42)

Scenarios A–C use the assignment's answer-rate / talk-time table; D changes both mid-run (see README for the full definitions).

```
scn mode           util  dials/min  conn/min  answered  abandon prov.err  safety verdicts (ticks)
A   progressive   50.9%       70.7      13.0       782    0.00%        0  APPROVED=3600
A   predictive    77.4%      106.1      19.8      1193    0.59%        0  APPROVED=2786, FALLBACK_PROGRESSIVE=386, REDUCED=428
B   progressive   67.8%       47.6      23.6      1417    0.00%        0  APPROVED=3600
B   predictive    78.6%       56.5      27.2      1646    0.67%        0  APPROVED=3212, FALLBACK_PROGRESSIVE=375, REDUCED=13
C   progressive   84.4%       23.4      14.2       855    0.00%      136  APPROVED=3480, FALLBACK_PROGRESSIVE=4, REJECTED=116
C   predictive    88.8%       24.9      15.1       915    0.98%      144  APPROVED=3028, FALLBACK_PROGRESSIVE=563, REDUCED=9
D   progressive   59.4%      111.0      29.5      1771    0.00%        0  APPROVED=3598, REDUCED=2
D   predictive    76.1%      237.2      43.7      2631    0.34%        0  APPROVED=2245, FALLBACK_PROGRESSIVE=339, REDUCED=684, REJECTED=332
```

*util* = agent talk time / logged-in time. *dials/min* = calls initiated, *conn/min* = calls connected to an agent. Each run's file in `results/` also has a 5-minute timeline and sample "why N calls" explanations.

What the numbers show:

* **The gain from predictive depends on the answer rate.** +26.5 points at 20% (A), +10.8 at 50% (B), +4.4 at 70% with 3-minute calls (C). With a high answer rate and long calls, progressive already keeps agents busy, so predictive buys little and isn't worth much risk. With a low answer rate, progressive leaves agents waiting through rings and no-answers, and that's where predictive pays.
* **The abandon rate stays below 1%** in every scenario, against a 3% target.
* **Scenario C (Provider B + outage):**
  * Progressive: the breaker tripped 4 times and REJECTed 116 ticks, then probes found the provider again and dialing resumed on its own.
  * Predictive: the outage hit while all 50 agents were on 3-minute calls, so there was nothing to dial and nothing failed. In-flight calls waited (8 "stuck, waiting for provider" sweeps) and the delayed events flushed when it came back.
  * Provider B's 683 duplicate event ids, 71 same-state re-sends and 174 stale or post-hangup events were all absorbed with zero invariant violations.
* **Scenario D (everything changes):**
  * 50% -> 70% answer and 90 s -> 180 s talk at t=900: 5 abandons in that 5-min bucket (calls already ringing at the old rate). The answer-burst and abandon guards forced progressive until the estimate caught up.
  * 40 of 100 agents vanish at t=1500: 1 abandon in the bucket; pacing works on 60 agents.
  * Answer rate collapses to 10% at t=2400: **zero abandons**. Predictive utilization drops to ~58% because every agent now needs ~10 lines and the blast-radius cap allows 5. Progressive collapses to ~19%.

---

## 6. Bottleneck analysis: 100 -> 1,000 -> 10,000 agents

Rough load at 10K agents, 20% answer rate and 2-minute calls: about 80 answers/s, so 400 dials/s, and about 1,500–2,000 provider events/s.

**Measured** (`python -m simulation.load_test`, file SQLite with WAL, one laptop; output in `results/load_test.txt`):

```
Part 1 - reservation race                     Part 2 - full control loop, one 1 s tick
 agents  workers  res/s  lost CAS  double     agents  avg ms  p95 ms  real-time?
   1000        1   7221      0.0%       0        100     7.6    22.2  yes
   1000        4   5734      7.6%       0       1000    65.2   170.9  yes
   1000       16   2736     31.3%       0      10000  2238.2  5553.6  NO (2.2-5.6 s per 1 s tick)
   5000       16   5415     14.5%       0
```

A profile of the 1,000-agent loop shows ~60% of the time inside SQLite: a dial costs ~7 small commits (claim borrower, insert call, reserve agent, mark RESERVED, mark INITIATED, ...) at ~6 ms, and each provider event ~2.5 ms. Throughput *falls* as workers are added (7,221 -> 2,736 reservations/s from 1 to 16 workers): they queue on one writer and lose CAS races on the same longest-idle agents.

So, in order:

| Order | What breaks | Why | Fix |
|---|---|---|---|
| 1 | **The one write path** (breaks between 1K and 10K agents, measured) | every reservation, transition and event is a small write transaction; SQLite allows one writer at a time, and a tick at 10K agents needs more writes than fit in a second | Postgres with row-level locking (the CAS SQL is unchanged), batched event writes, and partitioning by campaign so each campaign's writes are independent |
| 2 | **Hot rows on agent reservation** (measured: 13–36% lost CAS at 16 workers) | all workers pick the *same* longest-idle agents | `FOR UPDATE SKIP LOCKED` (Postgres) or random-of-top-K candidates; better, give each allocator its own team of agents so they never compete |
| 3 | **Snapshot queries every tick** | pacing and safety each aggregate the calls table every second; cost grows with call history | counters maintained incrementally by the event processor, per campaign |
| 4 | **Single control loop** | one tick does events, bridging, recovery and dialing in series | webhook ingestors -> queue partitioned by call_id (keeps per-call order) -> event workers; a pacer per campaign (the lease already exists); a separate reaper |
| 5 | **Pacer per campaign** | one pacer is right for safety (no double-spent headroom), but it is a single point per campaign | already lease-based with failover in 3 s; at scale, shard campaigns so each pacer is small, and add a fencing token |
| 6 | **Provider CPS limits** | 400 dials/s exceeds a typical single trunk | multi-provider routing with per-provider breakers and token buckets (the `max_dials_per_tick` cap becomes a per-provider rate limiter) |
| 7 | **Bridge latency** | ANSWERED -> CONNECTED must take under 2 s (the silent-call rule) | event-driven bridging (connect inside the ANSWERED handler, with the polling pass only as a safety net); co-locate with the media server |

The design already isolates these. Pacing is a pure function and the gateway is the only egress, so each piece can be scaled or replaced on its own.

---

## 7. The final question

> **How would you get the utilization benefits of predictive while keeping the safety of progressive?**

Don't choose between them. **Run predictive as an optimization inside a progressive safety envelope, and let the system demote itself automatically** whenever it can't justify the risk. This codebase does exactly that:

1. **Separate deciding from authorizing.** The predictive engine only *recommends* ("dial 17, because..."). A separate Safety Controller, the only issuer of dial permits, decides what actually happens. A pacing bug therefore can't become an over-dial.
2. **Safe by default.** On cold start, after a crash, or with too little data, the system runs **progressive**. That mode cannot abandon anyone, because every call owns an agent before it rings. Predictive must be *earned* by having enough recent outcomes.
3. **Overdial only on evidence, and only conservatively.**
   * Use the *upper* confidence bound of the answer rate.
   * Add a binomial safety stock (k sigma).
   * Count only agents that will really be free when the call is answered.
   * Back off smoothly as the abandon rate approaches the target.
4. **Circuit breakers demote predictive to progressive within one tick** when the abandon rate exceeds 3%, agents drop sharply, the provider degrades, or the answer rate surges. A cool-down prevents flapping. In a full outage or a silent provider, dialing stops entirely and a single probe call tests recovery.
5. **Bound the blast radius when the model is wrong.**
   * Cap lines per free agent, and ramp new lines over several ticks so the first answers are seen before all of them are out.
   * Stop overdialing the moment answered callers are waiting with no agent; don't wait for a 5-minute average.
   * Hold answered callers for at most 2 s, then hang up cleanly (never a silent line) and schedule a priority callback.
6. **Measure the one thing that matters.** The abandon rate is tracked continuously, and it is the metric that turns predictive off.
7. **Spend overdial only where it pays.** At high answer rates progressive is already near predictive (C: 84% vs 89%), so the risk buys little. At low answer rates the gain is large (A: 51% vs 77%).

The result in the simulations: **76–89% utilization (vs 51–84% progressive) at under 1% abandonment**, with automatic fallback to progressive in every disruption scenario.

What it does **not** give is progressive's *guarantee*. A sudden jump in answer rate (10% -> 70%) commits lines before any outcome can reveal the change; the ramp and the answer-burst trigger reduced that test from 56 to 36 abandoned calls (20 agents), but not to zero. Making that bound tighter costs utilization (section 8).

Further improvements I would make next:

* Per-segment answer-rate models (time of day, borrower bucket, retry count), because answer rates vary a lot by segment.
* Use the time since dial to discount lines that have been ringing for a while (their answer probability decays).
* "Progressive-plus": reserve the agent just before their wrap-up ends, which takes most of predictive's gain with zero abandon risk.
* Per-agent skill-based routing in the bridge.

---

## 8. Known limitations (honest list)

* **Sudden answer-rate jumps.** History says 10%, reality is 70%: in a 20-agent test, 36 of ~110 answered calls were abandoned before the guards stopped dialing (`test_jump_10_to_70_is_bounded_and_stops_fast`). Exposure is bounded by `max_lines_per_free_agent` (5) and the ramp (0.5 new lines per free agent per tick). Lowering either lowers this risk and lowers utilization at low answer rates; it is a policy knob, not a bug I can code away. A real deployment would also start each new list or time-of-day segment in progressive.
* **Single-node SQLite.** The concurrency guarantees are real, but throughput is bounded (section 6, measured).
* **Pacer fencing.** If the pacer stalls longer than its 3 s lease *in the middle of a tick*, a second worker can start pacing while the first finishes; the overlap is at most one tick of permits. A fencing token (lease version) checked in `redeem()` would close it.
* **In-memory safety state.** The breaker and agent-drop history live in the pacer's memory. A new pacer starts without them, which is safe: it starts with fresh windows and has to re-earn predictive from DB data.
* **Ghost calls during a full outage.** A provider timeout whose call did go out, while the provider can't even be queried, could ring a borrower with no agent. We hang up by our call id as soon as the provider is reachable, and late events for that call are ignored.
* **Stationary answer-rate model** inside the 5-minute window. The 1-minute check and the guards cover shifts, but a better model would weight recent outcomes more and segment by list / time of day.

---

## 9. Questions we expect in the discussion

**Two workers reserve the same agent at exactly the same time?** Both read version 7. Both send `UPDATE agents SET state='RESERVED', version=8 ... WHERE id=? AND version=7 AND state='AVAILABLE'`. SQLite (Postgres too) applies writes to one row one at a time: the first matches and commits; the second finds version 8 and matches 0 rows. `rowcount` tells each worker whether it won. The loser moves on to the next candidate. Tested with real threads and separate connections; the load test also shows zero double-bookings at 16 workers.

**The database says AVAILABLE but the cache says RESERVED. Which wins?** The database, always. There is no cache in the reservation path, on purpose: the only way to reserve is the CAS against the DB row. If a read cache were added (e.g. for dashboards or candidate lists), a stale entry could only cost a missed opportunity or a lost CAS that retries; it could never cause a double booking, because the write checks the DB's version. Here the cache is wrong: the agent is free, and the next reservation attempt will get them.

**The provider sends ANSWERED, the worker crashes, then COMPLETED arrives?** Event handling is one transaction: event-id insert + state change. If the crash is mid-transaction, nothing committed and the provider's redelivery is applied normally. If ANSWERED committed but the worker died before bridging, the next bridge pass (any worker, every tick) connects the call, or abandons and hangs it up after 2 s. When COMPLETED arrives it finalizes the call in one transaction: the agent goes to WRAP_UP (or AVAILABLE if never connected) and the borrower is settled. Any later ANSWERED is `IGNORED_TERMINAL`. Tested in `test_crash_between_answered_and_connected`.

**The model predicted 70%; it drops to 10%. How does the system protect itself?** That direction is safe for borrowers: fewer people answer than we planned for, so agents sit idle; nobody is abandoned. The estimate catches up within minutes and dialing increases. Tested: zero abandons, and dials rise as the estimate falls. The dangerous direction is the reverse (10% -> 70%), covered in section 8.

**1,000 -> 100,000 agents. What breaks first?** First the single write path: one SQLite writer, ~7 small commits per dial and ~2.5 ms per event (section 6, profiled). Postgres plus partitioning by campaign moves that limit. Then contention: every worker going for the same longest-idle agents (already 27% lost CAS attempts at 16 workers). Then the per-tick snapshot queries over the calls table. Then provider CPS. At 100,000 agents (~4,000 dials/s) the per-campaign design stays the same, but each campaign or team becomes its own shard with its own pacer, and the event stream is partitioned by call id.

**Why 17 calls instead of 10?** Section 3.4: the same 8 free agents, at 33% vs 60% answer rate. The explanation string for every tick shows each number, and every REDUCED verdict names the limit that applied.

**What are you least confident about?** The answer-rate jump above: the risk is bounded and measured, not removed. Second, the 2 s answered-caller hold assumes the bridge runs every second; at scale it must be event-driven (bridge in the ANSWERED handler) to keep that promise.

**What would you do with another week?**
1. Event-driven bridging, plus a fencing token on the pacer lease.
2. Postgres with `FOR UPDATE SKIP LOCKED` candidate selection, and run the load test against it.
3. An answer-rate model segmented by list and time of day, with a recency-weighted estimate.
4. "Progressive-plus": reserve the agent a few seconds before their wrap-up ends. That captures much of predictive's gain with zero abandon risk.
5. A real provider (Plivo) behind the same `TelecomProvider` interface.
