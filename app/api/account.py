"""Sign-up, sign-in, and the API keys an account issues for itself."""
from fastapi import APIRouter, Cookie, Depends, HTTPException, Response

from app.api.common import COOKIE_SECURE, now
from app.api.schemas import AuthRequest, CreateApiKeyRequest
from app.auth import (
    SESSION_COOKIE_NAME,
    SESSION_LIFETIME_DAYS,
    create_session_for_user,
    generate_api_key,
    get_current_user,
    hash_password,
    verify_password,
)
from app.storage import chat_db

router = APIRouter()


def _set_session_cookie(response: Response, token: str):
    response.set_cookie(
        key=SESSION_COOKIE_NAME, value=token, httponly=True, samesite="lax",
        secure=COOKIE_SECURE, max_age=SESSION_LIFETIME_DAYS * 24 * 3600,
    )


@router.post("/auth/signup")
def signup(req: AuthRequest, response: Response):
    if len(req.username) < 3 or len(req.password) < 6:
        raise HTTPException(status_code=400, detail="username must be 3+ chars, password 6+ chars")
    if chat_db.get_user_by_username(req.username) is not None:
        raise HTTPException(status_code=409, detail="username already taken")

    user_id = chat_db.create_user(req.username, hash_password(req.password), now())
    _set_session_cookie(response, create_session_for_user(user_id))
    return {"username": req.username}


@router.post("/auth/login")
def login(req: AuthRequest, response: Response):
    user = chat_db.get_user_by_username(req.username)
    if user is None or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="invalid username or password")

    _set_session_cookie(response, create_session_for_user(user["id"]))
    return {"username": user["username"]}


@router.post("/auth/logout")
def logout(response: Response, router_session: str | None = Cookie(default=None)):
    if router_session:
        chat_db.delete_session(router_session)
    response.delete_cookie(SESSION_COOKIE_NAME)
    return {"ok": True}


@router.get("/auth/me")
def me(user=Depends(get_current_user)):
    return {"username": user["username"]}


# --- API keys (ours, never a third-party provider key) -----------------------

@router.post("/api/keys")
def api_create_key(req: CreateApiKeyRequest, user=Depends(get_current_user)):
    full_key, key_prefix, key_hash = generate_api_key()
    key_id = chat_db.create_api_key(user["id"], key_prefix, key_hash, req.name, now())
    # The full key is returned exactly once, here. We only ever stored the hash.
    return {"id": key_id, "key": full_key, "key_prefix": key_prefix}


@router.get("/api/keys")
def api_list_keys(user=Depends(get_current_user)):
    return [
        {
            "id": k["id"], "name": k["name"] or "Unnamed key", "key_prefix": k["key_prefix"],
            "created_at": k["created_at"], "last_used_at": k["last_used_at"],
            "revoked": bool(k["revoked"]),
        }
        for k in chat_db.list_api_keys(user["id"])
    ]


@router.delete("/api/keys/{key_id}")
def api_revoke_key(key_id: int, user=Depends(get_current_user)):
    chat_db.revoke_api_key(key_id, user["id"])
    return {"ok": True}
