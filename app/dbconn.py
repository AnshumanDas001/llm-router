"""Where the database actually lives.

Local dev and a single VM use a plain SQLite file. A deploy on a host with
no durable disk (a container that can be rebuilt or rescheduled at any
moment) points at Turso instead by setting TURSO_DATABASE_URL and
TURSO_AUTH_TOKEN, and the same SQL runs against it unchanged -- Turso is
SQLite over the network.

It connects as an **embedded replica**: a local SQLite file kept in sync
with the remote. Writes go to Turso and are durable there; reads are
answered from the local copy. That matters more than it sounds -- measured
against the Mumbai region from a laptop, a remote round trip was ~150ms and
a local read was ~0.05ms, and a single page load makes several queries. On
boot the replica pulls the remote state, so a container that is wiped and
restarted comes back with every account and chat intact.

The app's SQL is unchanged. Two small gaps in the libsql driver are papered
over here:

  row_factory   the driver has no sqlite3.Row, so chat_db's `row["name"]`
                access would break. `_Row` reproduces the parts actually
                used (by-name and by-index lookup, `in`, dict()).
  sync cadence  writes reach Turso immediately, but the replica only learns
                about *other* writers when it syncs. This app is a single
                process, so its own writes are always visible; a periodic
                sync keeps it honest if that ever stops being true.
"""
import os
import sqlite3
import threading
import time
from pathlib import Path

TURSO_URL = os.getenv("TURSO_DATABASE_URL") or None
TURSO_TOKEN = os.getenv("TURSO_AUTH_TOKEN") or None
# Where the embedded replica is cached. Ephemeral by design: it is rebuilt
# from Turso on boot, so it does not need to survive anything.
REPLICA_PATH = Path(os.getenv("TURSO_REPLICA_PATH", "/tmp/thriftllm-replica.db"))
SYNC_INTERVAL_S = float(os.getenv("TURSO_SYNC_INTERVAL_S", "60"))

DB_PATH = Path(os.getenv("ROUTER_DB_PATH",
                         Path(__file__).resolve().parent.parent / "logs" / "router.db"))

_conn = None
_lock = threading.RLock()
_last_sync = 0.0


def using_turso() -> bool:
    return bool(TURSO_URL and TURSO_TOKEN)


class _Row:
    """sqlite3.Row's interface over a libsql tuple + cursor description."""

    __slots__ = ("_cols", "_vals")

    def __init__(self, cols, vals):
        self._cols, self._vals = cols, vals

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                return self._vals[self._cols.index(key)]
            except ValueError:
                raise IndexError(f"no such column: {key}") from None
        return self._vals[key]

    def keys(self):
        return list(self._cols)

    def __iter__(self):
        return iter(self._vals)

    def __len__(self):
        return len(self._vals)

    def __contains__(self, key):
        return key in self._cols

    def __repr__(self):
        return f"_Row({dict(zip(self._cols, self._vals))})"


class _Cursor:
    """Wraps a libsql cursor so fetches come back as _Row when the caller
    asked for row-by-name access."""

    def __init__(self, cursor, as_row):
        self._cur, self._as_row = cursor, as_row

    def _wrap(self, row):
        if row is None or not self._as_row:
            return row
        cols = [d[0] for d in self._cur.description or []]
        return _Row(cols, row)

    def fetchone(self):
        return self._wrap(self._cur.fetchone())

    def fetchall(self):
        rows = self._cur.fetchall()
        if not self._as_row:
            return rows
        cols = [d[0] for d in self._cur.description or []]
        return [_Row(cols, r) for r in rows]

    def __iter__(self):
        return iter(self.fetchall())

    def __getattr__(self, name):          # lastrowid, rowcount, description
        return getattr(self._cur, name)


class _Connection:
    """The subset of sqlite3.Connection this app uses, over libsql."""

    def __init__(self, conn):
        self._conn = conn
        self.row_factory = None

    def execute(self, sql, params=()):
        with _lock:
            return _Cursor(self._conn.execute(sql, params), self.row_factory is not None)

    def executescript(self, sql):
        with _lock:
            return self._conn.executescript(sql)

    def commit(self):
        with _lock:
            self._conn.commit()

    def sync(self):
        with _lock:
            self._conn.sync()

    def close(self):
        pass      # the connection is shared and long-lived; see _turso_conn


def _turso_conn():
    """One shared embedded-replica connection for the process.

    libsql's replica is a local file with its own write-ahead state; opening
    it repeatedly is both slower and a good way to get two handles fighting
    over the same file. One connection, guarded by a lock, matches how the
    app used SQLite anyway (one writer).
    """
    global _conn, _last_sync
    with _lock:
        if _conn is None:
            import libsql
            REPLICA_PATH.parent.mkdir(parents=True, exist_ok=True)
            raw = libsql.connect(str(REPLICA_PATH), sync_url=TURSO_URL, auth_token=TURSO_TOKEN)
            raw.sync()
            _conn = _Connection(raw)
            _last_sync = time.time()
        elif time.time() - _last_sync > SYNC_INTERVAL_S:
            try:
                _conn.sync()
            except Exception:
                pass      # a failed sync is stale reads, not a broken request
            _last_sync = time.time()
        return _conn


def connect(row_factory=False):
    """A connection to whichever backend is configured. Callers use it
    exactly as they used sqlite3, and close it exactly as before -- closing
    the shared Turso connection is a no-op."""
    if using_turso():
        conn = _turso_conn()
        conn.row_factory = _Row if row_factory else None
        return conn
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    if row_factory:
        conn.row_factory = sqlite3.Row
    return conn


def describe() -> str:
    return f"turso ({TURSO_URL}, replica {REPLICA_PATH})" if using_turso() else f"sqlite ({DB_PATH})"
