"""Tests run against a throwaway SQLite file, never the configured database.

.env may point at a real Turso database; load_dotenv() does not override
variables that are already set, so blanking them here -- before any app
module is imported -- keeps every test off it.
"""
import os
import tempfile

os.environ["TURSO_DATABASE_URL"] = ""
os.environ["TURSO_AUTH_TOKEN"] = ""
os.environ["ROUTER_DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="router-test-"), "router.db")
os.environ["ALLOW_ANON_V1"] = ""

from app.storage import connection  # noqa: E402

assert not connection.using_turso(), "tests must never run against Turso"
