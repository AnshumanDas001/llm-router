"""Basic username/password auth with cookie sessions.

Scope note: this is bcrypt password hashing + a random session token in an
HttpOnly cookie, which is a reasonable baseline for a local personal-project
demo. It has not been hardened for public internet deployment (no rate
limiting on login attempts, no CSRF token, cookie isn't marked Secure since
this runs over plain http://localhost) -- don't expose this app to the
internet as-is.
"""
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import Cookie, HTTPException

from app import chat_db

SESSION_COOKIE_NAME = "router_session"
SESSION_LIFETIME_DAYS = 30


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def create_session_for_user(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    expires = now + timedelta(days=SESSION_LIFETIME_DAYS)
    chat_db.create_session(token, user_id, now.isoformat(), expires.isoformat())
    return token


def get_current_user(router_session: str | None = Cookie(default=None)):
    """FastAPI dependency: raises 401 if there's no valid session."""
    if not router_session:
        raise HTTPException(status_code=401, detail="not authenticated")

    session = chat_db.get_session(router_session)
    if session is None:
        raise HTTPException(status_code=401, detail="invalid session")

    expires_at = datetime.fromisoformat(session["expires_at"])
    if expires_at < datetime.now(timezone.utc):
        chat_db.delete_session(router_session)
        raise HTTPException(status_code=401, detail="session expired")

    user = chat_db.get_user_by_id(session["user_id"])
    if user is None:
        raise HTTPException(status_code=401, detail="user not found")
    return user
