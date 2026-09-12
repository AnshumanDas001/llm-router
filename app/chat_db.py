"""Auth + chat-session storage for the product-style demo.

Kept separate from app/db.py (Week 1-4's eval/logging infra) since this is
a genuinely different subsystem -- product state, not research data.
"""
import sqlite3
from contextlib import contextmanager
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
