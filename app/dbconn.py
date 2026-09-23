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

from dotenv import load_dotenv

# Load .env here, not just in model_config. This module is imported by
# chat_db, which app/main.py imports before anything touches model_config --
# so reading the environment without this found no TURSO_* at all and fell
# back to local SQLite while .env plainly configured Turso. Silently writing
# to the wrong database is the worst failure this file can have, so it reads
# its own configuration rather than depending on import order.
load_dotenv()


def _normalise_env_names():
    """Re-expose any variable whose *name* has surrounding whitespace.

    Deploy consoles let you type a trailing space into the name field, and
    the result is a variable the app can never find -- the failure is silent
    and looks exactly like "the value is wrong". A name can only legitimately
    contain letters, digits and underscores, so whitespace around one is
    always a mistake and is safe to correct.
    """
    for key in list(os.environ):
        stripped = key.strip()
        if stripped != key and stripped not in os.environ:
            os.environ[stripped] = os.environ[key]


_normalise_env_names()

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


def _libsql():
    """The driver, or None if it isn't installed for this interpreter.
    libsql ships no wheel for some Python/arch combinations (3.14 on macOS,
    Linux arm64), and a dev machine shouldn't be unable to start because of
    it -- it falls back to the local file and says so."""
    try:
        import libsql
        return libsql
    except ImportError:
        return None


def using_turso() -> bool:
    return bool(TURSO_URL and TURSO_TOKEN and _libsql() is not None)


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
            libsql = _libsql()
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


# Names the app reads. /health reports which of these are *present* -- never
# their values -- because the usual deploy mistake is a variable that never
# reaches the container (mounted as a file instead of exported, set on the
# build step instead of the service, or simply misspelled), and that is
# indistinguishable from a bad value until you can see which names arrived.
EXPECTED_ENV = [
    "TURSO_DATABASE_URL", "TURSO_AUTH_TOKEN", "OPENROUTER_API_KEY",
    "GROQ_API_KEY", "ROUTER_SECRET_KEY", "COOKIE_SECURE",
]


def env_report() -> dict:
    present = {name: bool(os.getenv(name)) for name in EXPECTED_ENV}
    # Anything that looks like it was meant to be one of ours but isn't
    # spelled the way the app reads it.
    near_misses = sorted(
        k for k in os.environ
        if k not in EXPECTED_ENV and any(w in k.upper() for w in ("TURSO", "OPENROUTER", "GROQ", "ROUTER_SECRET"))
    )
    return {"set": present, "unrecognised_similar_names": near_misses}


def describe() -> str:
    return f"turso ({TURSO_URL}, replica {REPLICA_PATH})" if using_turso() else f"sqlite ({DB_PATH})"


def check() -> list[str]:
    """Warnings worth shouting about at startup. A database that looks
    configured but isn't loses accounts silently, and the symptom shows up a
    day later as 'my login stopped working'."""
    problems = []
    if using_turso():
        return problems
    if TURSO_URL and TURSO_TOKEN and _libsql() is None:
        problems.append(
            "TURSO_* is configured but the `libsql` driver is not installed for this "
            f"Python, so data is going to the local file at {DB_PATH} instead.")
    elif TURSO_URL or TURSO_TOKEN:
        problems.append(
            "TURSO_DATABASE_URL/TURSO_AUTH_TOKEN: only one is set, so Turso is OFF "
            "and data is going to the local file instead.")
    # A container filesystem does not survive the container. Cloud Run, Fly,
    # Railway and friends all set one of these.
    if any(os.getenv(v) for v in ("K_SERVICE", "FLY_APP_NAME", "RAILWAY_ENVIRONMENT", "RENDER")):
        problems.append(
            f"Running on a serverless host with local SQLite at {DB_PATH}. Accounts and "
            "chats will be LOST whenever the instance is recycled. Set TURSO_DATABASE_URL "
            "and TURSO_AUTH_TOKEN.")
    return problems
