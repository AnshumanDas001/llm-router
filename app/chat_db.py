"""Auth + chat-session storage for the product-style demo.

Kept separate from app/db.py (Week 1-4's eval/logging infra) since this is
a genuinely different subsystem -- product state, not research data.
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "logs" / "router.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    title TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL REFERENCES chats(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tier TEXT,
    cost REAL,
    baseline_cost REAL,
    escalated INTEGER,
    latency_ms REAL,
    timestamp TEXT NOT NULL
);

-- BYOM (bring your own model): we never store third-party provider keys.
-- api_keys are OUR service's own issued tokens (for identifying an account
-- on the pass-through routing endpoint); model_configs and
-- calibration_results hold only non-sensitive derived data.

CREATE TABLE IF NOT EXISTS api_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    key_prefix TEXT NOT NULL,
    key_hash TEXT NOT NULL UNIQUE,
    name TEXT,
    created_at TEXT NOT NULL,
    last_used_at TEXT,
    revoked INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS model_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    tier TEXT NOT NULL,
    model_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(user_id, tier)
);

CREATE TABLE IF NOT EXISTS calibration_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    tier TEXT NOT NULL,
    model_name TEXT NOT NULL,
    avg_quality REAL,
    avg_cost REAL,
    avg_latency_ms REAL,
    n_queries INTEGER,
    created_at TEXT NOT NULL,
    UNIQUE(user_id, tier)
);

CREATE TABLE IF NOT EXISTS usage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    source TEXT NOT NULL,
    difficulty TEXT,
    initial_tier TEXT,
    final_tier TEXT,
    escalated INTEGER,
    cost REAL,
    baseline_cost REAL,
    latency_ms REAL,
    timestamp TEXT NOT NULL
);
"""


@contextmanager
def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_chat_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        conn.commit()


# --- users -------------------------------------------------------------

def create_user(username: str, password_hash: str, timestamp: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, timestamp),
        )
        conn.commit()
        return cur.lastrowid


def get_user_by_username(username: str):
    with get_conn() as conn:
        return conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()


def get_user_by_id(user_id: int):
    with get_conn() as conn:
        return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


# --- sessions ------------------------------------------------------------

def create_session(token: str, user_id: int, created_at: str, expires_at: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, user_id, created_at, expires_at),
        )
        conn.commit()


def get_session(token: str):
    with get_conn() as conn:
        return conn.execute("SELECT * FROM sessions WHERE token = ?", (token,)).fetchone()


def delete_session(token: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()


# --- chats -----------------------------------------------------------------

def create_chat(user_id: int, title: str, timestamp: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO chats (user_id, title, created_at) VALUES (?, ?, ?)",
            (user_id, title, timestamp),
        )
        conn.commit()
        return cur.lastrowid


def list_chats(user_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT c.id, c.title, c.created_at, "
            "COALESCE(SUM(m.cost), 0) AS total_cost, "
            "COALESCE(SUM(m.baseline_cost), 0) AS total_baseline_cost "
            "FROM chats c LEFT JOIN chat_messages m ON m.chat_id = c.id "
            "WHERE c.user_id = ? GROUP BY c.id ORDER BY c.id DESC",
            (user_id,),
        ).fetchall()


def get_chat(chat_id: int, user_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id)
        ).fetchone()


def add_chat_message(chat_id: int, role: str, content: str, timestamp: str,
                      tier: str | None = None, cost: float | None = None,
                      baseline_cost: float | None = None, escalated: bool | None = None,
                      latency_ms: float | None = None):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO chat_messages (chat_id, role, content, tier, cost, baseline_cost, "
            "escalated, latency_ms, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (chat_id, role, content, tier, cost, baseline_cost,
             int(escalated) if escalated is not None else None, latency_ms, timestamp),
        )
        conn.commit()


def get_chat_messages(chat_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM chat_messages WHERE chat_id = ? ORDER BY id ASC", (chat_id,)
        ).fetchall()


def get_chat_totals(chat_id: int):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(cost), 0) AS total_cost, "
            "COALESCE(SUM(baseline_cost), 0) AS total_baseline_cost "
            "FROM chat_messages WHERE chat_id = ? AND role = 'assistant'",
            (chat_id,),
        ).fetchone()
        return dict(row)


# --- api keys (ours, for the pass-through routing endpoint) ----------------

def create_api_key(user_id: int, key_prefix: str, key_hash: str, name: str, timestamp: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO api_keys (user_id, key_prefix, key_hash, name, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, key_prefix, key_hash, name, timestamp),
        )
        conn.commit()
        return cur.lastrowid


def list_api_keys(user_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT id, key_prefix, name, created_at, last_used_at, revoked "
            "FROM api_keys WHERE user_id = ? ORDER BY id DESC", (user_id,),
        ).fetchall()


def get_user_by_api_key_hash(key_hash: str):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT u.* FROM api_keys k JOIN users u ON u.id = k.user_id "
            "WHERE k.key_hash = ? AND k.revoked = 0", (key_hash,),
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE key_hash = ?",
                (datetime.now(timezone.utc).isoformat(), key_hash),
            )
            conn.commit()
        return row


def revoke_api_key(key_id: int, user_id: int):
    with get_conn() as conn:
        conn.execute(
            "UPDATE api_keys SET revoked = 1 WHERE id = ? AND user_id = ?", (key_id, user_id),
        )
        conn.commit()


# --- model configs (which model maps to which tier; no keys stored) --------

def set_model_config(user_id: int, tier: str, model_name: str, timestamp: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO model_configs (user_id, tier, model_name, created_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(user_id, tier) DO UPDATE SET model_name = excluded.model_name, "
            "created_at = excluded.created_at",
            (user_id, tier, model_name, timestamp),
        )
        conn.commit()


def get_model_configs(user_id: int) -> dict:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT tier, model_name FROM model_configs WHERE user_id = ?", (user_id,),
        ).fetchall()
        return {r["tier"]: r["model_name"] for r in rows}


# --- calibration results (derived numbers only, never the keys used) -------

def set_calibration_result(user_id: int, tier: str, model_name: str, avg_quality: float,
                            avg_cost: float, avg_latency_ms: float, n_queries: int, timestamp: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO calibration_results (user_id, tier, model_name, avg_quality, avg_cost, "
            "avg_latency_ms, n_queries, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id, tier) DO UPDATE SET model_name = excluded.model_name, "
            "avg_quality = excluded.avg_quality, avg_cost = excluded.avg_cost, "
            "avg_latency_ms = excluded.avg_latency_ms, n_queries = excluded.n_queries, "
            "created_at = excluded.created_at",
            (user_id, tier, model_name, avg_quality, avg_cost, avg_latency_ms, n_queries, timestamp),
        )
        conn.commit()


def get_calibration_results(user_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM calibration_results WHERE user_id = ?", (user_id,),
        ).fetchall()


# --- usage log (both chat UI and API-mode traffic write here) --------------

def log_usage(user_id: int, source: str, difficulty: str, initial_tier: str, final_tier: str,
               escalated: bool, cost: float, baseline_cost: float, latency_ms: float, timestamp: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO usage_log (user_id, source, difficulty, initial_tier, final_tier, "
            "escalated, cost, baseline_cost, latency_ms, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, source, difficulty, initial_tier, final_tier, int(escalated),
             cost, baseline_cost, latency_ms, timestamp),
        )
        conn.commit()


def get_usage_summary(user_id: int):
    with get_conn() as conn:
        totals = conn.execute(
            "SELECT COUNT(*) AS n_calls, COALESCE(SUM(cost), 0) AS total_cost, "
            "COALESCE(SUM(baseline_cost), 0) AS total_baseline_cost "
            "FROM usage_log WHERE user_id = ?", (user_id,),
        ).fetchone()
        by_tier = conn.execute(
            "SELECT final_tier, COUNT(*) AS n FROM usage_log WHERE user_id = ? "
            "GROUP BY final_tier", (user_id,),
        ).fetchall()
        return {"totals": dict(totals), "by_tier": [dict(r) for r in by_tier]}


def get_usage_log(user_id: int, limit: int = 100):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM usage_log WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
