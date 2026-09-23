"""Copy a local SQLite database into Turso.

For the move from "runs on my laptop" to "runs on a host with no disk":
everything already in logs/router.db (accounts, chats, providers,
calibrations, usage) goes to the remote database, so signing in on the
deployed app finds the same accounts.

Safe to re-run. Rows are inserted with INSERT OR IGNORE, keeping their
primary keys, so a second run copies only what's missing and never
duplicates or overwrites what's already remote.

    ./venv/bin/python scripts/migrate_to_turso.py [--dry-run]

Needs the `libsql` driver, which has no wheel for every Python (3.14 on
macOS, Linux arm64). If the interpreter running this can't import it, use
one that can -- the database it writes to is the same either way.
"""
import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import dbconn

# Parents strictly before children: a chat needs its user, a message needs
# its chat, and chat_models needs both its chat and its provider.
TABLES = [
    "users", "sessions", "api_keys",
    "providers", "provider_models",
    "chats", "chat_models", "chat_messages",
    "model_calibrations", "calibration_results", "model_configs",
    "usage_log", "demo_usage", "requests", "cascade_log",
    "eval_responses", "eval_scores",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report what would move, write nothing")
    args = ap.parse_args()

    if not (dbconn.TURSO_URL and dbconn.TURSO_TOKEN):
        sys.exit("TURSO_DATABASE_URL and TURSO_AUTH_TOKEN must be set (see .env).")
    libsql = dbconn._libsql()
    if libsql is None:
        sys.exit(f"the `libsql` driver isn't installed for this Python ({sys.version.split()[0]}); "
                 "run this with one that has it.")
    if not dbconn.DB_PATH.exists():
        sys.exit(f"no local database at {dbconn.DB_PATH}")

    local = sqlite3.connect(dbconn.DB_PATH)
    local.row_factory = sqlite3.Row
    remote = libsql.connect(database=dbconn.TURSO_URL, auth_token=dbconn.TURSO_TOKEN)

    print(f"from {dbconn.DB_PATH}\n  to {dbconn.TURSO_URL}\n")
    total = 0
    for table in TABLES:
        try:
            rows = local.execute(f"SELECT * FROM {table}").fetchall()
        except sqlite3.OperationalError:
            continue                      # table not in this local database
        if not rows:
            continue
        before = remote.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if args.dry_run:
            print(f"  {table:22} {len(rows):5} local rows, {before} already remote")
            continue

        columns = rows[0].keys()
        placeholders = ", ".join("?" for _ in columns)
        sql = f"INSERT OR IGNORE INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
        skipped = 0
        for row in rows:
            try:
                remote.execute(sql, tuple(row))
            except Exception as e:
                # Usually a row pointing at a parent that was deleted locally
                # long ago. Skip it and keep going rather than abandoning the
                # migration half-done.
                if "FOREIGN KEY" not in str(e).upper():
                    raise
                skipped += 1
        remote.commit()
        after = remote.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        note = f"  ({skipped} skipped: orphaned rows)" if skipped else ""
        print(f"  {table:22} {len(rows):5} local -> {after - before:5} new remote (now {after}){note}")
        total += after - before

    print(f"\n{'would copy' if args.dry_run else 'copied'} {total} rows")


if __name__ == "__main__":
    main()
