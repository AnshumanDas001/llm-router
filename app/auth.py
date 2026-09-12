"""Basic username/password auth with cookie sessions.

Scope note: this is bcrypt password hashing + a random session token in an
HttpOnly cookie, which is a reasonable baseline for a local personal-project
demo. It has not been hardened for public internet deployment (no rate
limiting on login attempts, no CSRF token, cookie isn't marked Secure since
this runs over plain http://localhost) -- don't expose this app to the
internet as-is.
"""
import hashlib
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import Cookie, Header, HTTPException

from app import chat_db

SESSION_COOKIE_NAME = "router_session"
SESSION_LIFETIME_DAYS = 30
API_KEY_PREFIX = "rtr_"


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


def hash_api_key(key: str) -> str:
    # API keys are high-entropy random tokens already (not user-chosen, low
    # entropy like passwords), so a fast hash is the right tool here --
    # bcrypt's deliberate slowness defends against guessing a *weak* secret,
    # which doesn't apply to a 43-char random token. This mirrors how
    # Stripe/GitHub hash their own issued API keys/PATs.
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def generate_api_key() -> tuple[str, str, str]:
    """Returns (full_key, key_prefix, key_hash). Only the hash gets stored;
    the full key is shown to the user exactly once, at creation time."""
    secret = secrets.token_urlsafe(32)
    full_key = f"{API_KEY_PREFIX}{secret}"
    key_prefix = full_key[:len(API_KEY_PREFIX) + 8]
    return full_key, key_prefix, hash_api_key(full_key)


def get_user_from_api_key(authorization: str | None = Header(default=None)):
    """FastAPI dependency for the pass-through routing endpoint: auth via
    `Authorization: Bearer rtr_...`, one of our own issued keys -- never a
    third-party provider key."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing or malformed Authorization header")

    key = authorization.removeprefix("Bearer ").strip()
    if not key.startswith(API_KEY_PREFIX):
        raise HTTPException(status_code=401, detail="invalid API key")

    user = chat_db.get_user_by_api_key_hash(hash_api_key(key))
    if user is None:
        raise HTTPException(status_code=401, detail="invalid or revoked API key")
    return user


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
