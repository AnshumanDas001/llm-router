import os
from contextlib import contextmanager
from pathlib import Path

from app import dbconn

# Overridable so a deployment can point this at a mounted volume; the
# default keeps local development exactly as it was.
DB_PATH = dbconn.DB_PATH   # kept for scripts that report where data lives

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    model_used TEXT NOT NULL,
    tokens_in INTEGER,
    tokens_out INTEGER,
    cost REAL,
    latency_ms REAL,
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id TEXT NOT NULL,
    tier TEXT NOT NULL,
    response_text TEXT NOT NULL,
    tokens_in INTEGER,
    tokens_out INTEGER,
    cost REAL,
    latency_ms REAL,
    timestamp TEXT NOT NULL,
    UNIQUE(query_id, tier)
);

CREATE TABLE IF NOT EXISTS eval_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id TEXT NOT NULL,
    tier TEXT NOT NULL,
    eval_method TEXT NOT NULL,
    score REAL,
    source TEXT NOT NULL,
    notes TEXT,
    timestamp TEXT NOT NULL,
    UNIQUE(query_id, tier, source)
);

CREATE TABLE IF NOT EXISTS cascade_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    difficulty TEXT NOT NULL,
    initial_tier TEXT NOT NULL,
    final_tier TEXT NOT NULL,
    escalated INTEGER NOT NULL,
    escalation_reasons TEXT,
    total_cost REAL,
    total_latency_ms REAL,
    tokens_in INTEGER,
    tokens_out INTEGER,
    timestamp TEXT NOT NULL,
    response_text TEXT
);
"""


@contextmanager
def get_conn():
    conn = dbconn.connect()
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        # Added after the table existed in deployed databases: the shipped
        # answer, so scripts/score_cascade_log.py can grade what the router
        # actually sent back, not just what it cost.
        try:
            conn.execute("ALTER TABLE cascade_log ADD COLUMN response_text TEXT")
        except Exception:
            pass   # already there
        conn.commit()


def log_request(query: str, model_used: str, tokens_in: int, tokens_out: int,
                 cost: float, latency_ms: float, timestamp: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO requests (query, model_used, tokens_in, tokens_out, cost, latency_ms, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (query, model_used, tokens_in, tokens_out, cost, latency_ms, timestamp),
        )
        conn.commit()


def log_eval_response(query_id: str, tier: str, response_text: str, tokens_in: int,
                       tokens_out: int, cost: float, latency_ms: float, timestamp: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO eval_responses "
            "(query_id, tier, response_text, tokens_in, tokens_out, cost, latency_ms, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (query_id, tier, response_text, tokens_in, tokens_out, cost, latency_ms, timestamp),
        )
        conn.commit()


def get_eval_responses(query_id: str | None = None, tier: str | None = None):
    query = "SELECT query_id, tier, response_text, tokens_in, tokens_out, cost, latency_ms, timestamp FROM eval_responses WHERE 1=1"
    params = []
    if query_id:
        query += " AND query_id = ?"
        params.append(query_id)
    if tier:
        query += " AND tier = ?"
        params.append(tier)
    with get_conn() as conn:
        return conn.execute(query, params).fetchall()


def log_eval_score(query_id: str, tier: str, eval_method: str, score: float,
                    source: str, notes: str, timestamp: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO eval_scores "
            "(query_id, tier, eval_method, score, source, notes, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (query_id, tier, eval_method, score, source, notes, timestamp),
        )
        conn.commit()


def log_cascade(query: str, difficulty: str, initial_tier: str, final_tier: str,
                 escalated: bool, escalation_reasons: list[str], total_cost: float,
                 total_latency_ms: float, tokens_in: int, tokens_out: int, timestamp: str,
                 response_text: str | None = None):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO cascade_log (query, difficulty, initial_tier, final_tier, escalated, "
            "escalation_reasons, total_cost, total_latency_ms, tokens_in, tokens_out, timestamp, "
            "response_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (query, difficulty, initial_tier, final_tier, int(escalated),
             " | ".join(escalation_reasons), total_cost, total_latency_ms,
             tokens_in, tokens_out, timestamp, response_text),
        )
        conn.commit()
