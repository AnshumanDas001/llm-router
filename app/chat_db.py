"""Auth + chat-session storage for the product-style demo.

Kept separate from app/db.py (the eval/logging infra) since this is
a genuinely different subsystem -- product state, not research data.
"""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

# Overridable so a deployment can point this at a mounted volume; the
# default keeps local development exactly as it was.
DB_PATH = Path(os.getenv("ROUTER_DB_PATH", Path(__file__).resolve().parent.parent / "logs" / "router.db"))

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
    pinned INTEGER NOT NULL DEFAULT 0,
    mode TEXT NOT NULL DEFAULT 'builtin',
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
    difficulty TEXT,
    escalation_reasons TEXT,
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
    quality_by_difficulty TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(user_id, tier)
);

-- RAGFlow-style provider connections. api_key_encrypted is NULL when the
-- user opted to supply the key per session instead of storing it.
CREATE TABLE IF NOT EXISTS providers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    name TEXT NOT NULL,
    provider TEXT NOT NULL,
    api_key_encrypted TEXT,
    api_base TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(user_id, name)
);

CREATE TABLE IF NOT EXISTS provider_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_id INTEGER NOT NULL REFERENCES providers(id),
    model_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(provider_id, model_name)
);

-- Calibration is keyed by MODEL, not by tier: measured quality is a
-- property of the model itself, so a model calibrated once can be slotted
-- as "cheap" in one session and "mid" in another without re-measuring.
CREATE TABLE IF NOT EXISTS model_calibrations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    model_name TEXT NOT NULL,
    avg_quality REAL,
    avg_cost REAL,
    avg_latency_ms REAL,
    n_queries INTEGER,
    quality_by_difficulty TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(user_id, model_name)
);

-- The three models a chat was created with, frozen at creation so a
-- session's history stays coherent even if the user later reconfigures.
CREATE TABLE IF NOT EXISTS chat_models (
    chat_id INTEGER NOT NULL REFERENCES chats(id),
    tier TEXT NOT NULL,
    model_name TEXT NOT NULL,
    provider_id INTEGER REFERENCES providers(id),
    PRIMARY KEY (chat_id, tier)
);

-- Soft quota for the signed-out landing demo. Not a security boundary
-- (clearing cookies resets it) -- it exists to cap cost per casual visitor.
CREATE TABLE IF NOT EXISTS demo_usage (
    demo_id TEXT PRIMARY KEY,
    n_prompts INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS usage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    api_key_id INTEGER REFERENCES api_keys(id),
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
        # Migration: usage_log predates api_key_id (added when API-mode calls
        # became groupable into per-key "sessions"). SQLite has no
        # ADD COLUMN IF NOT EXISTS, so guard the duplicate-column error.
        try:
            conn.execute("ALTER TABLE usage_log ADD COLUMN api_key_id INTEGER REFERENCES api_keys(id)")
            conn.commit()
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).lower():
                raise
        try:
            conn.execute("ALTER TABLE chats ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")
            conn.commit()
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).lower():
                raise
        try:
            conn.execute("ALTER TABLE chats ADD COLUMN mode TEXT NOT NULL DEFAULT 'builtin'")
            conn.commit()
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).lower():
                raise
        for column, decl in (("difficulty", "TEXT"), ("escalation_reasons", "TEXT")):
            try:
                conn.execute(f"ALTER TABLE chat_messages ADD COLUMN {column} {decl}")
                conn.commit()
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
        try:
            conn.execute("ALTER TABLE calibration_results ADD COLUMN quality_by_difficulty TEXT")
            conn.commit()
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).lower():
                raise

        # Calibration moved from per-(user, tier) to per-(user, model): the
        # same model can now sit in different tiers in different sessions, so
        # its measurements belong to the model. Carry existing rows over once
        # rather than making users re-pay to re-measure. INSERT OR IGNORE
        # makes this a no-op on every later startup.
        conn.execute(
            "INSERT OR IGNORE INTO model_calibrations "
            "(user_id, model_name, avg_quality, avg_cost, avg_latency_ms, n_queries, "
            " quality_by_difficulty, created_at) "
            "SELECT user_id, model_name, avg_quality, avg_cost, avg_latency_ms, n_queries, "
            "       quality_by_difficulty, created_at FROM calibration_results"
        )
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

def create_chat(user_id: int, title: str, timestamp: str, mode: str = "builtin") -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO chats (user_id, title, mode, created_at) VALUES (?, ?, ?, ?)",
            (user_id, title, mode, timestamp),
        )
        conn.commit()
        return cur.lastrowid


def list_chats(user_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT c.id, c.title, c.pinned, c.mode, c.created_at, "
            "COALESCE(SUM(m.cost), 0) AS total_cost, "
            "COALESCE(SUM(m.baseline_cost), 0) AS total_baseline_cost, "
            "(SELECT content FROM chat_messages fm WHERE fm.chat_id = c.id AND fm.role = 'user' "
            "ORDER BY fm.id ASC LIMIT 1) AS preview "
            "FROM chats c LEFT JOIN chat_messages m ON m.chat_id = c.id "
            "WHERE c.user_id = ? GROUP BY c.id ORDER BY c.pinned DESC, c.id DESC",
            (user_id,),
        ).fetchall()


def get_chat_stats(user_id: int):
    """Real aggregate numbers across this user's own chat history -- avg
    latency and overall percent saved -- used by the sidebar status card
    instead of a simulated/fabricated metric."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n_messages, "
            "COALESCE(AVG(m.latency_ms), 0) AS avg_latency_ms, "
            "COALESCE(SUM(m.cost), 0) AS total_cost, "
            "COALESCE(SUM(m.baseline_cost), 0) AS total_baseline_cost "
            "FROM chat_messages m JOIN chats c ON c.id = m.chat_id "
            "WHERE c.user_id = ? AND m.role = 'assistant'",
            (user_id,),
        ).fetchone()
        total_cost = row["total_cost"]
        total_baseline = row["total_baseline_cost"]
        pct_saved = (
            max(0.0, (total_baseline - total_cost) / total_baseline * 100) if total_baseline > 0 else 0.0
        )
        return {
            "n_messages": row["n_messages"],
            "avg_latency_ms": row["avg_latency_ms"],
            "cost_saved": max(0.0, total_baseline - total_cost),
            "pct_saved": pct_saved,
        }


def rename_chat(chat_id: int, user_id: int, title: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE chats SET title = ? WHERE id = ? AND user_id = ?", (title, chat_id, user_id),
        )
        conn.commit()


def set_chat_pinned(chat_id: int, user_id: int, pinned: bool):
    with get_conn() as conn:
        conn.execute(
            "UPDATE chats SET pinned = ? WHERE id = ? AND user_id = ?",
            (int(pinned), chat_id, user_id),
        )
        conn.commit()


def delete_chat(chat_id: int, user_id: int):
    with get_conn() as conn:
        # Ownership check via the WHERE clause on chats; only delete messages
        # if the chat itself actually belonged to this user.
        owned = conn.execute(
            "SELECT id FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id),
        ).fetchone()
        if owned is None:
            return False
        conn.execute("DELETE FROM chat_messages WHERE chat_id = ?", (chat_id,))
        conn.execute("DELETE FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id))
        conn.commit()
        return True


def get_chat(chat_id: int, user_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id)
        ).fetchone()


def add_chat_message(chat_id: int, role: str, content: str, timestamp: str,
                      tier: str | None = None, cost: float | None = None,
                      baseline_cost: float | None = None, escalated: bool | None = None,
                      latency_ms: float | None = None, difficulty: str | None = None,
                      escalation_reasons: list | None = None):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO chat_messages (chat_id, role, content, tier, cost, baseline_cost, "
            "escalated, latency_ms, difficulty, escalation_reasons, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (chat_id, role, content, tier, cost, baseline_cost,
             int(escalated) if escalated is not None else None, latency_ms, difficulty,
             json.dumps(escalation_reasons) if escalation_reasons else None, timestamp),
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
    """Returns a row with the user's fields plus api_key_id (the specific
    key used -- each key is also a "session" for API-mode usage, so callers
    need to know which one to attribute a call to)."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT u.*, k.id AS api_key_id FROM api_keys k JOIN users u ON u.id = k.user_id "
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


# --- providers + their models (RAGFlow-style connections) ------------------

def create_provider(user_id: int, name: str, provider: str, api_key_encrypted: str | None,
                     api_base: str | None, timestamp: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO providers (user_id, name, provider, api_key_encrypted, api_base, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, name, provider, api_key_encrypted, api_base, timestamp),
        )
        conn.commit()
        return cur.lastrowid


def list_providers(user_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT id, name, provider, api_base, created_at, "
            "(api_key_encrypted IS NOT NULL) AS has_stored_key "
            "FROM providers WHERE user_id = ? ORDER BY id DESC", (user_id,),
        ).fetchall()


def get_provider(provider_id: int, user_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM providers WHERE id = ? AND user_id = ?", (provider_id, user_id),
        ).fetchone()


def delete_provider(provider_id: int, user_id: int) -> bool:
    with get_conn() as conn:
        owned = conn.execute(
            "SELECT id FROM providers WHERE id = ? AND user_id = ?", (provider_id, user_id),
        ).fetchone()
        if owned is None:
            return False
        conn.execute("DELETE FROM provider_models WHERE provider_id = ?", (provider_id,))
        conn.execute("DELETE FROM providers WHERE id = ?", (provider_id,))
        conn.commit()
        return True


def set_provider_key(provider_id: int, user_id: int, api_key_encrypted: str | None):
    """Passing None forgets a stored key, reverting that provider to
    per-session entry."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE providers SET api_key_encrypted = ? WHERE id = ? AND user_id = ?",
            (api_key_encrypted, provider_id, user_id),
        )
        conn.commit()


def add_provider_model(provider_id: int, model_name: str, timestamp: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO provider_models (provider_id, model_name, created_at) "
            "VALUES (?, ?, ?)",
            (provider_id, model_name, timestamp),
        )
        conn.commit()


def delete_provider_model(model_id: int, user_id: int) -> bool:
    with get_conn() as conn:
        owned = conn.execute(
            "SELECT m.id FROM provider_models m JOIN providers p ON p.id = m.provider_id "
            "WHERE m.id = ? AND p.user_id = ?", (model_id, user_id),
        ).fetchone()
        if owned is None:
            return False
        conn.execute("DELETE FROM provider_models WHERE id = ?", (model_id,))
        conn.commit()
        return True


def list_user_models(user_id: int):
    """Every model the user has connected, with its provider -- the pool the
    chat-creation dropdowns pick from."""
    with get_conn() as conn:
        return conn.execute(
            "SELECT m.id, m.model_name, m.provider_id, p.name AS provider_name, "
            "p.provider, p.api_base, (p.api_key_encrypted IS NOT NULL) AS has_stored_key "
            "FROM provider_models m JOIN providers p ON p.id = m.provider_id "
            "WHERE p.user_id = ? ORDER BY p.name, m.model_name", (user_id,),
        ).fetchall()


def get_model_with_provider(user_id: int, model_name: str):
    with get_conn() as conn:
        return conn.execute(
            "SELECT m.model_name, p.id AS provider_id, p.provider, p.api_base, "
            "p.api_key_encrypted FROM provider_models m JOIN providers p ON p.id = m.provider_id "
            "WHERE p.user_id = ? AND m.model_name = ? LIMIT 1", (user_id, model_name),
        ).fetchone()


# --- per-model calibration (reusable across tiers and sessions) ------------

def set_model_calibration(user_id: int, model_name: str, avg_quality: float, avg_cost: float,
                           avg_latency_ms: float, n_queries: int, timestamp: str,
                           quality_by_difficulty: dict | None = None):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO model_calibrations (user_id, model_name, avg_quality, avg_cost, "
            "avg_latency_ms, n_queries, quality_by_difficulty, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id, model_name) DO UPDATE SET avg_quality = excluded.avg_quality, "
            "avg_cost = excluded.avg_cost, avg_latency_ms = excluded.avg_latency_ms, "
            "n_queries = excluded.n_queries, "
            "quality_by_difficulty = excluded.quality_by_difficulty, "
            "created_at = excluded.created_at",
            (user_id, model_name, avg_quality, avg_cost, avg_latency_ms, n_queries,
             json.dumps(quality_by_difficulty) if quality_by_difficulty else None, timestamp),
        )
        conn.commit()


def get_model_calibrations(user_id: int) -> dict:
    """{model_name: {avg_quality, avg_cost, avg_latency_ms, n_queries,
    quality_by_difficulty, created_at}}"""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM model_calibrations WHERE user_id = ?", (user_id,),
        ).fetchall()
        return {
            r["model_name"]: {
                "avg_quality": r["avg_quality"], "avg_cost": r["avg_cost"],
                "avg_latency_ms": r["avg_latency_ms"], "n_queries": r["n_queries"],
                "quality_by_difficulty": json.loads(r["quality_by_difficulty"])
                if r["quality_by_difficulty"] else None,
                "created_at": r["created_at"],
            }
            for r in rows
        }


# --- per-chat model selection (frozen at creation) -------------------------

def set_chat_models(chat_id: int, tier_models: dict, provider_ids: dict | None = None):
    with get_conn() as conn:
        for tier, model_name in tier_models.items():
            conn.execute(
                "INSERT OR REPLACE INTO chat_models (chat_id, tier, model_name, provider_id) "
                "VALUES (?, ?, ?, ?)",
                (chat_id, tier, model_name, (provider_ids or {}).get(tier)),
            )
        conn.commit()


def get_chat_models(chat_id: int) -> dict:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT tier, model_name FROM chat_models WHERE chat_id = ?", (chat_id,),
        ).fetchall()
        return {r["tier"]: r["model_name"] for r in rows}


# --- calibration results (derived numbers only, never the keys used) -------

def set_calibration_result(user_id: int, tier: str, model_name: str, avg_quality: float,
                            avg_cost: float, avg_latency_ms: float, n_queries: int, timestamp: str,
                            quality_by_difficulty: dict | None = None):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO calibration_results (user_id, tier, model_name, avg_quality, avg_cost, "
            "avg_latency_ms, n_queries, quality_by_difficulty, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id, tier) DO UPDATE SET model_name = excluded.model_name, "
            "avg_quality = excluded.avg_quality, avg_cost = excluded.avg_cost, "
            "avg_latency_ms = excluded.avg_latency_ms, n_queries = excluded.n_queries, "
            "quality_by_difficulty = excluded.quality_by_difficulty, "
            "created_at = excluded.created_at",
            (user_id, tier, model_name, avg_quality, avg_cost, avg_latency_ms, n_queries,
             json.dumps(quality_by_difficulty) if quality_by_difficulty else None, timestamp),
        )
        conn.commit()


def get_quality_by_tier(user_id: int) -> dict:
    """{tier: {difficulty: score|None}} -- the input the routing policy needs
    to decide where each difficulty should start for this account."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT tier, quality_by_difficulty FROM calibration_results WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        return {
            r["tier"]: json.loads(r["quality_by_difficulty"])
            for r in rows if r["quality_by_difficulty"]
        }


def get_calibration_results(user_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM calibration_results WHERE user_id = ?", (user_id,),
        ).fetchall()


# --- usage log (both chat UI and API-mode traffic write here) --------------
# An API-mode row's api_key_id makes that key a "session" -- see
# get_sessions_overview / get_api_session_detail below, which group by it.

def log_usage(user_id: int, source: str, difficulty: str, initial_tier: str, final_tier: str,
               escalated: bool, cost: float, baseline_cost: float, latency_ms: float, timestamp: str,
               api_key_id: int | None = None):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO usage_log (user_id, api_key_id, source, difficulty, initial_tier, "
            "final_tier, escalated, cost, baseline_cost, latency_ms, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, api_key_id, source, difficulty, initial_tier, final_tier, int(escalated),
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


# --- unified sessions overview (chats + API-key sessions) ------------------

def get_sessions_overview(user_id: int):
    """Every session a user has, of either type, as one flat list -- a chat
    (grouped by chats/chat_messages) or an API key (grouped by api_keys/
    usage_log, one key = one session)."""
    with get_conn() as conn:
        chat_rows = conn.execute(
            "SELECT c.id, c.title, c.pinned, c.mode, c.created_at, COUNT(m.id) AS n_calls, "
            "COALESCE(SUM(m.cost), 0) AS total_cost, "
            "COALESCE(SUM(m.baseline_cost), 0) AS total_baseline_cost "
            "FROM chats c LEFT JOIN chat_messages m ON m.chat_id = c.id AND m.role = 'assistant' "
            "WHERE c.user_id = ? GROUP BY c.id ORDER BY c.id DESC", (user_id,),
        ).fetchall()
        api_rows = conn.execute(
            "SELECT k.id, k.name, k.key_prefix, k.created_at, k.revoked, "
            "COUNT(u.id) AS n_calls, COALESCE(SUM(u.cost), 0) AS total_cost, "
            "COALESCE(SUM(u.baseline_cost), 0) AS total_baseline_cost "
            "FROM api_keys k LEFT JOIN usage_log u ON u.api_key_id = k.id "
            "WHERE k.user_id = ? GROUP BY k.id ORDER BY k.id DESC", (user_id,),
        ).fetchall()

    sessions = []
    for c in chat_rows:
        sessions.append({
            "type": "chat", "id": c["id"], "name": c["title"] or "New chat",
            "created_at": c["created_at"], "n_calls": c["n_calls"],
            "total_cost": c["total_cost"], "total_baseline_cost": c["total_baseline_cost"],
            "cost_saved": max(0.0, c["total_baseline_cost"] - c["total_cost"]),
            "revoked": False, "pinned": bool(c["pinned"]), "mode": c["mode"],
        })
    for k in api_rows:
        sessions.append({
            "type": "api", "id": k["id"], "name": k["name"] or f"API session {k['key_prefix']}",
            "created_at": k["created_at"], "n_calls": k["n_calls"],
            "total_cost": k["total_cost"], "total_baseline_cost": k["total_baseline_cost"],
            "cost_saved": max(0.0, k["total_baseline_cost"] - k["total_cost"]),
            "revoked": bool(k["revoked"]), "pinned": False,
        })
    sessions.sort(key=lambda s: s["created_at"], reverse=True)
    return sessions


def get_api_session_detail(api_key_id: int, user_id: int):
    """The call log for one API-key session -- verifies ownership via the
    user_id join, same 404-not-403 pattern as chat access."""
    with get_conn() as conn:
        key = conn.execute(
            "SELECT * FROM api_keys WHERE id = ? AND user_id = ?", (api_key_id, user_id),
        ).fetchone()
        if key is None:
            return None, []
        calls = conn.execute(
            "SELECT * FROM usage_log WHERE api_key_id = ? ORDER BY id DESC", (api_key_id,),
        ).fetchall()
        return key, calls


# --- signed-out demo quota --------------------------------------------------

def get_demo_count(demo_id: str) -> int:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT n_prompts FROM demo_usage WHERE demo_id = ?", (demo_id,),
        ).fetchone()
        return row["n_prompts"] if row else 0


def bump_demo_count(demo_id: str, timestamp: str) -> int:
    """Increments and returns the new count, atomically."""
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO demo_usage (demo_id, n_prompts, created_at) VALUES (?, 1, ?) "
            "ON CONFLICT(demo_id) DO UPDATE SET n_prompts = n_prompts + 1",
            (demo_id, timestamp),
        )
        conn.commit()
        return conn.execute(
            "SELECT n_prompts FROM demo_usage WHERE demo_id = ?", (demo_id,),
        ).fetchone()["n_prompts"]
