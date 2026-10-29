"""SQLite persistence for agents, borrowers, calls and provider events.

Concurrency model
-----------------
* File databases: every thread gets its own connection (WAL mode, busy
  timeout). Writers are serialized by SQLite itself, and the `version` columns
  provide optimistic compare-and-swap on top of that (see lock_manager.py).
* ":memory:" databases: one shared connection guarded by a re-entrant lock.
  Used by the single-threaded simulator, where it is much faster.

`cas_update` is the only general-purpose write method. It updates a row only
if its version still matches what the caller read, and reports whether it
won. Multi-statement units of work use `transaction()` (BEGIN IMMEDIATE),
which takes the write lock up front so the whole unit commits or rolls back.
"""
import contextlib
import sqlite3
import threading
from dataclasses import asdict, fields
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence

from core.models import Agent, Borrower, Call, ProviderEvent
from core.state_machines import AgentState, BorrowerState, CallState

SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
    id                   TEXT PRIMARY KEY,
    name                 TEXT NOT NULL,
    state                TEXT NOT NULL,
    version              INTEGER NOT NULL DEFAULT 0,
    reserved_for_call_id TEXT,
    state_changed_at     REAL NOT NULL DEFAULT 0,
    wrap_until           REAL,
    logout_requested     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_agents_state ON agents(state, state_changed_at, id);
CREATE INDEX IF NOT EXISTS idx_agents_reserved_for ON agents(reserved_for_call_id);

CREATE TABLE IF NOT EXISTS borrowers (
    id               TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    phone            TEXT NOT NULL,
    state            TEXT NOT NULL,
    version          INTEGER NOT NULL DEFAULT 0,
    attempts         INTEGER NOT NULL DEFAULT 0,
    max_attempts     INTEGER NOT NULL DEFAULT 3,
    next_attempt_at  REAL NOT NULL DEFAULT 0,
    current_call_id  TEXT,
    lease_owner      TEXT,
    lease_expires_at REAL,
    outcome          TEXT
);
CREATE INDEX IF NOT EXISTS idx_borrowers_due ON borrowers(state, next_attempt_at, id);

CREATE TABLE IF NOT EXISTS calls (
    id               TEXT PRIMARY KEY,
    borrower_id      TEXT NOT NULL,
    state            TEXT NOT NULL,
    version          INTEGER NOT NULL DEFAULT 0,
    agent_id         TEXT,
    mode             TEXT NOT NULL,
    provider         TEXT,
    provider_call_id TEXT,
    worker_id        TEXT,
    lease_expires_at REAL,
    created_at       REAL NOT NULL,
    initiated_at     REAL,
    ringing_at       REAL,
    answered_at      REAL,
    connected_at     REAL,
    ended_at         REAL,
    failure_reason   TEXT
);
CREATE INDEX IF NOT EXISTS idx_calls_state ON calls(state);
CREATE INDEX IF NOT EXISTS idx_calls_answered_at ON calls(answered_at);
CREATE INDEX IF NOT EXISTS idx_calls_ended_at ON calls(ended_at);
CREATE INDEX IF NOT EXISTS idx_calls_borrower ON calls(borrower_id);

-- One row per provider event id. The PRIMARY KEY is the idempotency guard:
-- a duplicate delivery fails INSERT OR IGNORE and is treated as a no-op.
CREATE TABLE IF NOT EXISTS provider_events (
    event_id         TEXT PRIMARY KEY,
    call_id          TEXT NOT NULL,
    provider_call_id TEXT,
    event_type       TEXT NOT NULL,
    occurred_at      REAL NOT NULL,
    received_at      REAL NOT NULL,
    outcome          TEXT
);

-- Audit log: every pacing recommendation and the safety verdict on it.
CREATE TABLE IF NOT EXISTS pacing_decisions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    at             REAL NOT NULL,
    requested_mode TEXT NOT NULL,
    recommended    INTEGER NOT NULL,
    verdict        TEXT NOT NULL,
    effective_mode TEXT NOT NULL,
    approved       INTEGER NOT NULL,
    summary        TEXT,
    reasons        TEXT
);

-- Named leases. "pacer" = the one worker allowed to run pacing + safety for the campaign,
-- so two workers can never both spend the same free-agent headroom on predictive lines.
CREATE TABLE IF NOT EXISTS leases (
    name       TEXT PRIMARY KEY,
    holder     TEXT NOT NULL,
    expires_at REAL NOT NULL
);
"""

_MODELS = {"agents": Agent, "borrowers": Borrower, "calls": Call}
_ENUMS = {
    "agents": {"state": AgentState},
    "borrowers": {"state": BorrowerState},
    "calls": {"state": CallState},
}
_COLUMNS = {table: {f.name for f in fields(model)} for table, model in _MODELS.items()}


def _sql(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bool):
        return int(value)
    return value


def _placeholders(values: Sequence[Any]) -> str:
    return ",".join("?" for _ in values)


class Database:
    def __init__(self, path: str = ":memory:", busy_timeout_ms: int = 10_000):
        self.path = path
        self._busy_timeout_ms = busy_timeout_ms
        self._shared = path == ":memory:"
        self._local = threading.local()
        self._guard_lock = threading.RLock() if self._shared else None
        self._shared_conn = self._connect() if self._shared else None
        self.conn.executescript(SCHEMA)

    # ------------------------------------------------------------------ plumbing
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=self._busy_timeout_ms / 1000,
                               isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        if not self._shared:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        if self._shared:
            return self._shared_conn
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    @contextlib.contextmanager
    def _guard(self):
        if self._guard_lock is None:
            yield
        else:
            with self._guard_lock:
                yield

    @contextlib.contextmanager
    def transaction(self):
        """All-or-nothing unit of work. Nested calls join the outer transaction."""
        with self._guard():
            conn = self.conn
            if conn.in_transaction:
                yield conn
                return
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    @staticmethod
    def _params(params):
        if isinstance(params, dict):
            return {k: _sql(v) for k, v in params.items()}
        return [_sql(p) for p in params]

    def execute(self, sql: str, params: Any = ()) -> int:
        with self._guard():
            return self.conn.execute(sql, self._params(params)).rowcount

    def query(self, sql: str, params: Any = ()) -> List[sqlite3.Row]:
        with self._guard():
            return self.conn.execute(sql, self._params(params)).fetchall()

    def query_one(self, sql: str, params: Any = ()) -> Optional[sqlite3.Row]:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def close(self) -> None:
        if self._shared:
            self._shared_conn.close()
        else:
            conn = getattr(self._local, "conn", None)
            if conn is not None:
                conn.close()
                self._local.conn = None

    # ------------------------------------------------------------------ mapping
    @staticmethod
    def _to_model(table: str, row: Optional[sqlite3.Row]):
        if row is None:
            return None
        data = dict(row)
        for column, enum_cls in _ENUMS[table].items():
            data[column] = enum_cls(data[column])
        if table == "agents":
            data["logout_requested"] = bool(data["logout_requested"])
        return _MODELS[table](**data)

    def _insert(self, table: str, model) -> None:
        data = asdict(model)
        cols = list(data)
        self.execute(f"INSERT INTO {table} ({','.join(cols)}) VALUES ({_placeholders(cols)})",
                     [data[c] for c in cols])

    def _get(self, table: str, row_id: str):
        return self._to_model(table, self.query_one(f"SELECT * FROM {table} WHERE id = ?", (row_id,)))

    # ------------------------------------------------------------------ CAS write
    def cas_update(self, table: str, row_id: str, expected_version: int,
                   changes: Dict[str, Any], expect: Optional[Dict[str, Any]] = None) -> bool:
        """Atomically apply `changes` iff the row still has `expected_version`
        (and every column in `expect` still has the given value).

        Returns True for exactly one of any number of concurrent callers that
        read the same version. Everyone else gets False and must re-read.
        """
        allowed = _COLUMNS[table]
        bad = (set(changes) | set(expect or {})) - allowed
        if bad or "version" in changes:
            raise ValueError(f"illegal columns for {table}: {bad or 'version'}")
        set_sql = ", ".join(f"{col} = ?" for col in changes)
        params: List[Any] = [changes[c] for c in changes]
        where = "id = ? AND version = ?"
        params += [row_id, expected_version]
        for col, value in (expect or {}).items():
            if value is None:
                where += f" AND {col} IS NULL"
            else:
                where += f" AND {col} = ?"
                params.append(value)
        sql = f"UPDATE {table} SET {set_sql}, version = version + 1 WHERE {where}"
        return self.execute(sql, params) == 1

    # ------------------------------------------------------------------ agents
    def insert_agent(self, agent: Agent) -> None:
        self._insert("agents", agent)

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        return self._get("agents", agent_id)

    def list_agents(self, states: Optional[Iterable[AgentState]] = None) -> List[Agent]:
        if states is None:
            rows = self.query("SELECT * FROM agents ORDER BY id")
        else:
            st = [s.value for s in states]
            rows = self.query(f"SELECT * FROM agents WHERE state IN ({_placeholders(st)}) ORDER BY id", st)
        return [self._to_model("agents", r) for r in rows]

    def find_available_agents(self, limit: int) -> List[Agent]:
        """Longest-idle agents first (fairness)."""
        rows = self.query("SELECT * FROM agents WHERE state = ? ORDER BY state_changed_at, id LIMIT ?",
                          (AgentState.AVAILABLE, limit))
        return [self._to_model("agents", r) for r in rows]

    def find_agents_reserved_for(self, call_id: str) -> List[Agent]:
        rows = self.query("SELECT * FROM agents WHERE reserved_for_call_id = ?", (call_id,))
        return [self._to_model("agents", r) for r in rows]

    def count_agents_by_state(self) -> Dict[AgentState, int]:
        counts = {s: 0 for s in AgentState}
        for row in self.query("SELECT state, COUNT(*) AS n FROM agents GROUP BY state"):
            counts[AgentState(row["state"])] = row["n"]
        return counts

    # ------------------------------------------------------------------ borrowers
    def insert_borrowers(self, borrowers: Iterable[Borrower]) -> None:
        rows = [asdict(b) for b in borrowers]
        if not rows:
            return
        cols = list(rows[0])
        with self.transaction() as conn:
            conn.executemany(f"INSERT INTO borrowers ({','.join(cols)}) VALUES ({_placeholders(cols)})",
                             [[_sql(r[c]) for c in cols] for r in rows])

    def get_borrower(self, borrower_id: str) -> Optional[Borrower]:
        return self._get("borrowers", borrower_id)

    def find_due_borrowers(self, now: float, limit: int) -> List[Borrower]:
        rows = self.query("SELECT * FROM borrowers WHERE state = ? AND next_attempt_at <= ? "
                          "ORDER BY next_attempt_at, id LIMIT ?", (BorrowerState.PENDING, now, limit))
        return [self._to_model("borrowers", r) for r in rows]

    def find_expired_borrower_leases(self, now: float) -> List[Borrower]:
        rows = self.query("SELECT * FROM borrowers WHERE state = ? AND lease_expires_at < ?",
                          (BorrowerState.RESERVED, now))
        return [self._to_model("borrowers", r) for r in rows]

    def count_borrowers_by_state(self) -> Dict[BorrowerState, int]:
        counts = {s: 0 for s in BorrowerState}
        for row in self.query("SELECT state, COUNT(*) AS n FROM borrowers GROUP BY state"):
            counts[BorrowerState(row["state"])] = row["n"]
        return counts

    # ------------------------------------------------------------------ calls
    def insert_call(self, call: Call) -> None:
        self._insert("calls", call)

    def get_call(self, call_id: str) -> Optional[Call]:
        return self._get("calls", call_id)

    def find_calls(self, states: Iterable[CallState], order_by: str = "created_at") -> List[Call]:
        if order_by not in _COLUMNS["calls"]:
            raise ValueError(order_by)
        st = [s.value for s in states]
        rows = self.query(f"SELECT * FROM calls WHERE state IN ({_placeholders(st)}) ORDER BY {order_by}, id", st)
        return [self._to_model("calls", r) for r in rows]

    def find_expired_call_leases(self, now: float, states: Iterable[CallState]) -> List[Call]:
        st = [s.value for s in states]
        rows = self.query(f"SELECT * FROM calls WHERE state IN ({_placeholders(st)}) "
                          f"AND lease_expires_at IS NOT NULL AND lease_expires_at < ? ORDER BY created_at",
                          st + [now])
        return [self._to_model("calls", r) for r in rows]

    def count_calls_by_state(self) -> Dict[CallState, int]:
        counts = {s: 0 for s in CallState}
        for row in self.query("SELECT state, COUNT(*) AS n FROM calls GROUP BY state"):
            counts[CallState(row["state"])] = row["n"]
        return counts

    # ------------------------------------------------------------------ events
    def insert_event_if_new(self, event: ProviderEvent, received_at: float) -> bool:
        """Returns False if this event id was already recorded (duplicate delivery)."""
        return self.execute(
            "INSERT OR IGNORE INTO provider_events "
            "(event_id, call_id, provider_call_id, event_type, occurred_at, received_at) VALUES (?,?,?,?,?,?)",
            (event.event_id, event.call_id, event.provider_call_id, event.event_type,
             event.occurred_at, received_at)) == 1

    def set_event_outcome(self, event_id: str, outcome: str) -> None:
        self.execute("UPDATE provider_events SET outcome = ? WHERE event_id = ?", (outcome, event_id))

    # ------------------------------------------------------------------ leases
    def try_acquire_lease(self, name: str, holder: str, now: float, ttl_s: float) -> bool:
        """Take or renew lease `name`. One atomic upsert: succeeds only if the lease is free,
        expired, or already ours. True means `holder` owns it until now + ttl_s."""
        return self.execute(
            "INSERT INTO leases (name, holder, expires_at) VALUES (?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET holder = excluded.holder, expires_at = excluded.expires_at "
            "WHERE leases.holder = excluded.holder OR leases.expires_at <= ?",
            (name, holder, now + ttl_s, now)) == 1

    # ------------------------------------------------------------------ audit
    def log_decision(self, at: float, requested_mode: str, recommended: int, verdict: str,
                     effective_mode: str, approved: int, summary: str, reasons: str) -> None:
        self.execute("INSERT INTO pacing_decisions (at, requested_mode, recommended, verdict, "
                     "effective_mode, approved, summary, reasons) VALUES (?,?,?,?,?,?,?,?)",
                     (at, requested_mode, recommended, verdict, effective_mode, approved, summary, reasons))
