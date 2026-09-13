import json
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Cookie, Depends, FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import chat_db, key_vault
from app.auth import (
    SESSION_COOKIE_NAME,
    SESSION_LIFETIME_DAYS,
    create_session_for_user,
    generate_api_key,
    get_current_user,
    get_user_from_api_key,
    hash_password,
    verify_password,
)
from app.baseline_cost import estimate_cost_for_model, estimate_frontier_cost
from app.calibration import DEFAULT_MAX_QUERIES, calibrate_model
from app.cascade import DEFAULT_TIER_MODELS, AllTiersUnavailable, run_cascade
from app.classifier import classify_initial_tier
from app.db import init_db, log_cascade
from app.routing_policy import derive_tier_map

app = FastAPI(title="LLM Router")
app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent / "static"), name="static")


@app.middleware("http")
async def no_cache_static(request, call_next):
    # This app is under active iteration -- a stale cached nav.js/app.css in
    # someone's browser (e.g. from before the Docs link was added) silently
    # hides real changes with no error to notice. Not a concern at this
    # scale/stage; revisit with real cache headers if this ever needs to
    # serve real traffic.
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, must-revalidate"
    return response

DEMO_HTML_PATH = Path(__file__).resolve().parent / "demo.html"
CHAT_APP_HTML_PATH = Path(__file__).resolve().parent / "chat_app.html"
SETTINGS_HTML_PATH = Path(__file__).resolve().parent / "settings.html"
MODELS_HTML_PATH = Path(__file__).resolve().parent / "models.html"
SESSIONS_HTML_PATH = Path(__file__).resolve().parent / "sessions.html"
GUIDE_HTML_PATH = Path(__file__).resolve().parent / "guide.html"


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str | None = None  # ignored; the cascade always decides the tier


class AuthRequest(BaseModel):
    username: str
    password: str


class CreateChatRequest(BaseModel):
    title: str | None = None
    mode: str = "builtin"  # "builtin" (our models) or "byom" (this user's configured models)
    models: dict[str, str] = {}  # tier -> model_name, frozen onto the chat at creation


class UpdateChatRequest(BaseModel):
    title: str | None = None
    pinned: bool | None = None


class SendMessageRequest(BaseModel):
    content: str
    tier_api_keys: dict[str, str] = {}  # BYOM chats only; used transiently, never stored


class ClassifyRequest(BaseModel):
    content: str


class CreateProviderRequest(BaseModel):
    name: str
    provider: str                      # litellm prefix: groq, gemini, openai, ...
    api_key: str | None = None
    store_key: bool = False            # opt in to encrypted storage; default is per-session
    api_base: str | None = None
    models: list[str] = []


class AddModelRequest(BaseModel):
    model_name: str


class CalibrateModelRequest(BaseModel):
    model_name: str
    api_key: str | None = None         # omitted when the provider has a stored key


class CreateApiKeyRequest(BaseModel):
    name: str | None = None


class RouteRequest(BaseModel):
    messages: list[Message]
    models: dict[str, str] = {}         # tier -> model_name, from your connected models
    tier_api_keys: dict[str, str] = {}  # used transiently for this call only, never stored


@app.on_event("startup")
def on_startup():
    init_db()
    chat_db.init_chat_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/demo", response_class=HTMLResponse)
def demo():
    return DEMO_HTML_PATH.read_text()


@app.get("/app", response_class=HTMLResponse)
def chat_app():
    return CHAT_APP_HTML_PATH.read_text()


@app.get("/settings", response_class=HTMLResponse)
def settings_page():
    return SETTINGS_HTML_PATH.read_text()


@app.get("/models", response_class=HTMLResponse)
def models_page():
    return MODELS_HTML_PATH.read_text()


@app.get("/sessions", response_class=HTMLResponse)
def sessions_page():
    return SESSIONS_HTML_PATH.read_text()


@app.get("/guide", response_class=HTMLResponse)
def guide_page():
    # Deliberately not "/docs" -- FastAPI reserves that path for its own
    # auto-generated Swagger UI, and our custom route was shadowing it.
    return GUIDE_HTML_PATH.read_text()


# --- auth ------------------------------------------------------------------

def _set_session_cookie(response: Response, token: str):
    response.set_cookie(
        key=SESSION_COOKIE_NAME, value=token, httponly=True, samesite="lax",
        max_age=SESSION_LIFETIME_DAYS * 24 * 3600,
    )


@app.post("/auth/signup")
def signup(req: AuthRequest, response: Response):
    if len(req.username) < 3 or len(req.password) < 6:
        raise HTTPException(status_code=400, detail="username must be 3+ chars, password 6+ chars")
    if chat_db.get_user_by_username(req.username) is not None:
        raise HTTPException(status_code=409, detail="username already taken")

    user_id = chat_db.create_user(
        req.username, hash_password(req.password), datetime.now(timezone.utc).isoformat(),
    )
    token = create_session_for_user(user_id)
    _set_session_cookie(response, token)
    return {"username": req.username}


@app.post("/auth/login")
def login(req: AuthRequest, response: Response):
    user = chat_db.get_user_by_username(req.username)
    if user is None or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="invalid username or password")

    token = create_session_for_user(user["id"])
    _set_session_cookie(response, token)
    return {"username": user["username"]}


@app.post("/auth/logout")
def logout(response: Response, router_session: str | None = Cookie(default=None)):
    if router_session:
        chat_db.delete_session(router_session)
    response.delete_cookie(SESSION_COOKIE_NAME)
    return {"ok": True}


@app.get("/auth/me")
def me(user=Depends(get_current_user)):
    return {"username": user["username"]}


# --- chats -------------------------------------------------------------

@app.get("/api/chats")
def api_list_chats(user=Depends(get_current_user)):
    chats = chat_db.list_chats(user["id"])
    return [
        {
            "id": c["id"],
            "title": c["title"] or "New chat",
            "preview": c["preview"],
            "pinned": bool(c["pinned"]),
            "mode": c["mode"],
            "created_at": c["created_at"],
            "total_cost": c["total_cost"],
            "total_baseline_cost": c["total_baseline_cost"],
            "cost_saved": max(0.0, c["total_baseline_cost"] - c["total_cost"]),
        }
        for c in chats
    ]


@app.get("/api/chat-stats")
def api_chat_stats(user=Depends(get_current_user)):
    """Real aggregate numbers (not simulated) behind the sidebar's router
    status card -- computed from this user's own chat_messages history."""
    return chat_db.get_chat_stats(user["id"])


@app.post("/api/classify")
def api_classify(req: ClassifyRequest, user=Depends(get_current_user)):
    """Runs only the upfront embedding classifier -- no model call, no cost
    -- so the UI can show a live 'predicted tier' as the user types, same
    logic the cascade itself uses to pick where to start."""
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="content must not be empty")
    tier, difficulty = classify_initial_tier(req.content)
    return {"tier": tier, "difficulty": difficulty}


@app.post("/api/chats")
def api_create_chat(req: CreateChatRequest, user=Depends(get_current_user)):
    if req.mode not in ("builtin", "byom"):
        raise HTTPException(status_code=400, detail="mode must be 'builtin' or 'byom'")

    if req.mode == "byom":
        if not req.models:
            raise HTTPException(
                status_code=400, detail="select at least one model for this session",
            )
        calibrations = chat_db.get_model_calibrations(user["id"])
        for tier, model_name in req.models.items():
            if tier not in ("cheap", "mid", "frontier"):
                raise HTTPException(status_code=400, detail=f"unknown tier '{tier}'")
            if chat_db.get_model_with_provider(user["id"], model_name) is None:
                raise HTTPException(
                    status_code=400, detail=f"'{model_name}' is not a connected model",
                )
            # Calibration is what produces this session's routing map, so an
            # uncalibrated model would leave the router guessing.
            if model_name not in calibrations:
                raise HTTPException(
                    status_code=400,
                    detail=f"'{model_name}' has not been calibrated yet -- calibrate it before "
                           f"starting a session with it",
                )

    chat_id = chat_db.create_chat(
        user["id"], req.title, datetime.now(timezone.utc).isoformat(), mode=req.mode,
    )
    if req.mode == "byom":
        provider_ids = {}
        for tier, model_name in req.models.items():
            row = chat_db.get_model_with_provider(user["id"], model_name)
            provider_ids[tier] = row["provider_id"] if row else None
        chat_db.set_chat_models(chat_id, req.models, provider_ids)
    return {"id": chat_id, "mode": req.mode}


@app.patch("/api/chats/{chat_id}")
def api_update_chat(chat_id: int, req: UpdateChatRequest, user=Depends(get_current_user)):
    chat = chat_db.get_chat(chat_id, user["id"])
    if chat is None:
        raise HTTPException(status_code=404, detail="chat not found")
    if req.title is not None:
        title = req.title.strip()
        if not title:
            raise HTTPException(status_code=400, detail="title must not be empty")
        chat_db.rename_chat(chat_id, user["id"], title)
    if req.pinned is not None:
        chat_db.set_chat_pinned(chat_id, user["id"], req.pinned)
    return {"ok": True}


@app.delete("/api/chats/{chat_id}")
def api_delete_chat(chat_id: int, user=Depends(get_current_user)):
    deleted = chat_db.delete_chat(chat_id, user["id"])
    if not deleted:
        raise HTTPException(status_code=404, detail="chat not found")
    return {"ok": True}


@app.get("/api/chats/{chat_id}")
def api_get_chat(chat_id: int, user=Depends(get_current_user)):
    chat = chat_db.get_chat(chat_id, user["id"])
    if chat is None:
        raise HTTPException(status_code=404, detail="chat not found")

    messages = chat_db.get_chat_messages(chat_id)
    totals = chat_db.get_chat_totals(chat_id)

    # Tiers the browser has to supply a key for: no saved key, and a provider
    # that actually needs one (a local Ollama needs nothing, so prompting for
    # it would be asking for a secret that doesn't exist).
    needs_key = []
    if chat["mode"] == "byom":
        for tier, model_name in chat_db.get_chat_models(chat_id).items():
            row = chat_db.get_model_with_provider(user["id"], model_name)
            if row is None:
                needs_key.append(tier)
            elif not row["api_key_encrypted"] and _provider_needs_key(row["provider"]):
                needs_key.append(tier)
    return {
        "id": chat["id"],
        "title": chat["title"] or "New chat",
        "pinned": bool(chat["pinned"]),
        "mode": chat["mode"],
        "byom_tiers": chat_db.get_chat_models(chat_id) if chat["mode"] == "byom" else {},
        "byom_needs_key": needs_key,
        "messages": [
            {
                "role": m["role"], "content": m["content"], "tier": m["tier"],
                "cost": m["cost"], "baseline_cost": m["baseline_cost"],
                "escalated": bool(m["escalated"]) if m["escalated"] is not None else False,
                "latency_ms": m["latency_ms"], "difficulty": m["difficulty"],
                "escalation_reasons": json.loads(m["escalation_reasons"]) if m["escalation_reasons"] else [],
            }
            for m in messages
        ],
        "total_cost": totals["total_cost"],
        "total_baseline_cost": totals["total_baseline_cost"],
        "cost_saved": max(0.0, totals["total_baseline_cost"] - totals["total_cost"]),
    }


@app.post("/api/chats/{chat_id}/messages")
def api_send_message(chat_id: int, req: SendMessageRequest, user=Depends(get_current_user)):
    chat = chat_db.get_chat(chat_id, user["id"])
    if chat is None:
        raise HTTPException(status_code=404, detail="chat not found")
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="content must not be empty")

    now = datetime.now(timezone.utc).isoformat()
    history = chat_db.get_chat_messages(chat_id)
    chat_db.add_chat_message(chat_id, "user", req.content, now)

    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": req.content})

    is_byom = chat["mode"] == "byom"
    tier_models = chat_db.get_chat_models(chat_id) if is_byom else None
    if is_byom and not tier_models:
        raise HTTPException(
            status_code=400,
            detail="this session has no models attached -- start a new session and pick models",
        )

    tier_api_keys = None
    if is_byom:
        # Stored provider key if the user chose to save one, otherwise the
        # key they supplied for this request from the browser.
        tier_api_keys = {
            tier: key for tier, model_name in tier_models.items()
            if (key := _resolve_provider_key(user["id"], model_name,
                                             req.tier_api_keys.get(tier))) is not None
        }

    try:
        result = run_cascade(
            messages,
            tier_models=tier_models,
            tier_api_keys=tier_api_keys,
            # BYOM routes by what this session's own models measured; the
            # built-in stack keeps the default map, which is what the eval set
            # measured for it.
            difficulty_to_tier=_tier_map_for(user["id"], tier_models) if is_byom else None,
        )
    except AllTiersUnavailable:
        raise HTTPException(
            status_code=503,
            detail="All model tiers are temporarily rate limited. Please try again in a few minutes.",
        )
    except Exception:
        raise HTTPException(status_code=502, detail="Something went wrong generating a response.")

    if is_byom:
        # Baseline: what the strongest tier *this user* configured would have
        # cost -- not our own built-in frontier, which they may not even use.
        order = ["frontier", "mid", "cheap"]
        strongest_configured = next((t for t in order if t in tier_models), result["final_tier"])
        baseline_cost = estimate_cost_for_model(
            tier_models[strongest_configured], result["tokens_in"], result["tokens_out"],
        )
    else:
        baseline_cost = estimate_frontier_cost(result["tokens_in"], result["tokens_out"])

    chat_db.add_chat_message(
        chat_id, "assistant", result["text"], datetime.now(timezone.utc).isoformat(),
        tier=result["final_tier"], cost=result["total_cost"], baseline_cost=baseline_cost,
        escalated=result["escalated"], latency_ms=result["total_latency_ms"],
        difficulty=result["difficulty"], escalation_reasons=result["escalation_reasons"],
    )

    log_cascade(
        query=req.content, difficulty=result["difficulty"], initial_tier=result["initial_tier"],
        final_tier=result["final_tier"], escalated=result["escalated"],
        escalation_reasons=result["escalation_reasons"], total_cost=result["total_cost"],
        total_latency_ms=result["total_latency_ms"], tokens_in=result["tokens_in"],
        tokens_out=result["tokens_out"], timestamp=datetime.now(timezone.utc).isoformat(),
    )

    totals = chat_db.get_chat_totals(chat_id)
    return {
        "content": result["text"],
        "tier": result["final_tier"],
        "difficulty": result["difficulty"],
        "escalated": result["escalated"],
        "escalation_reasons": result["escalation_reasons"],
        "cost": result["total_cost"],
        "baseline_cost": baseline_cost,
        "latency_ms": result["total_latency_ms"],
        "chat_total_cost": totals["total_cost"],
        "chat_total_baseline_cost": totals["total_baseline_cost"],
        "chat_cost_saved": max(0.0, totals["total_baseline_cost"] - totals["total_cost"]),
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    try:
        result = run_cascade([m.model_dump() for m in req.messages])
    except AllTiersUnavailable:
        raise HTTPException(
            status_code=503,
            detail="All model tiers are temporarily rate limited. Please try again in a few minutes.",
        )
    except Exception:
        raise HTTPException(status_code=502, detail="Something went wrong generating a response.")

    log_cascade(
        query=req.messages[-1].content,
        difficulty=result["difficulty"],
        initial_tier=result["initial_tier"],
        final_tier=result["final_tier"],
        escalated=result["escalated"],
        escalation_reasons=result["escalation_reasons"],
        total_cost=result["total_cost"],
        total_latency_ms=result["total_latency_ms"],
        tokens_in=result["tokens_in"],
        tokens_out=result["tokens_out"],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result["final_tier"],
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result["text"]},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": result["tokens_in"],
            "completion_tokens": result["tokens_out"],
            "total_tokens": (result["tokens_in"] or 0) + (result["tokens_out"] or 0),
        },
        "_router": {
            "difficulty": result["difficulty"],
            "initial_tier": result["initial_tier"],
            "final_tier": result["final_tier"],
            "escalated": result["escalated"],
            "escalation_reasons": result["escalation_reasons"],
            "cost": result["total_cost"],
            "latency_ms": round(result["total_latency_ms"], 1),
        },
    }


# --- BYOM: API keys (ours, never a third-party provider key) ---------------

@app.post("/api/keys")
def api_create_key(req: CreateApiKeyRequest, user=Depends(get_current_user)):
    full_key, key_prefix, key_hash = generate_api_key()
    key_id = chat_db.create_api_key(
        user["id"], key_prefix, key_hash, req.name, datetime.now(timezone.utc).isoformat(),
    )
    # The full key is returned exactly once, here. We only ever stored the hash.
    return {"id": key_id, "key": full_key, "key_prefix": key_prefix}


@app.get("/api/keys")
def api_list_keys(user=Depends(get_current_user)):
    keys = chat_db.list_api_keys(user["id"])
    return [
        {
            "id": k["id"], "name": k["name"] or "Unnamed key", "key_prefix": k["key_prefix"],
            "created_at": k["created_at"], "last_used_at": k["last_used_at"],
            "revoked": bool(k["revoked"]),
        }
        for k in keys
    ]


@app.delete("/api/keys/{key_id}")
def api_revoke_key(key_id: int, user=Depends(get_current_user)):
    chat_db.revoke_api_key(key_id, user["id"])
    return {"ok": True}


# --- provider connections (RAGFlow-style) -----------------------------------

# litellm prefix -> label, for the "add a provider" dropdown. The prefix is
# the part that actually matters: getting it wrong ("google/" instead of
# "gemini/") is the single most common calibration failure.
KNOWN_PROVIDERS = [
    {"provider": "groq", "label": "Groq", "needs_key": True},
    {"provider": "gemini", "label": "Google AI Studio (Gemini)", "needs_key": True},
    {"provider": "openai", "label": "OpenAI", "needs_key": True},
    {"provider": "anthropic", "label": "Anthropic", "needs_key": True},
    {"provider": "mistral", "label": "Mistral", "needs_key": True},
    {"provider": "deepseek", "label": "DeepSeek", "needs_key": True},
    {"provider": "together_ai", "label": "Together AI", "needs_key": True},
    {"provider": "ollama", "label": "Ollama (local)", "needs_key": False},
]


def _provider_needs_key(provider: str) -> bool:
    """Unknown providers default to needing a key -- better to ask for one
    that turns out to be unnecessary than to silently call without it."""
    for known in KNOWN_PROVIDERS:
        if known["provider"] == provider:
            return known["needs_key"]
    return True


@app.get("/api/provider-types")
def api_provider_types(user=Depends(get_current_user)):
    return {"providers": KNOWN_PROVIDERS, "key_storage_available": key_vault.storage_available()}


@app.get("/api/providers")
def api_list_providers(user=Depends(get_current_user)):
    providers = chat_db.list_providers(user["id"])
    models = chat_db.list_user_models(user["id"])
    by_provider = {}
    for m in models:
        by_provider.setdefault(m["provider_id"], []).append(
            {"id": m["id"], "model_name": m["model_name"]}
        )
    return [
        {
            "id": p["id"], "name": p["name"], "provider": p["provider"],
            "api_base": p["api_base"], "created_at": p["created_at"],
            "has_stored_key": bool(p["has_stored_key"]),
            "models": by_provider.get(p["id"], []),
        }
        for p in providers
    ]


@app.post("/api/providers")
def api_create_provider(req: CreateProviderRequest, user=Depends(get_current_user)):
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name must not be empty")
    if not req.provider.strip():
        raise HTTPException(status_code=400, detail="provider must not be empty")

    encrypted = None
    if req.store_key:
        if not req.api_key:
            raise HTTPException(status_code=400, detail="no key given to store")
        try:
            encrypted = key_vault.encrypt(req.api_key)
        except key_vault.KeyStorageUnavailable as e:
            # Never silently fall back to not-storing (the user thinks it's
            # saved) or to plaintext (worse). Fail loudly.
            raise HTTPException(status_code=400, detail=str(e))

    now = datetime.now(timezone.utc).isoformat()
    try:
        provider_id = chat_db.create_provider(
            user["id"], name, req.provider.strip(), encrypted, req.api_base, now,
        )
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail=f"you already have a provider named '{name}'")

    for model_name in req.models:
        if model_name.strip():
            chat_db.add_provider_model(provider_id, model_name.strip(), now)
    return {"id": provider_id, "stored_key": encrypted is not None}


@app.delete("/api/providers/{provider_id}")
def api_delete_provider(provider_id: int, user=Depends(get_current_user)):
    if not chat_db.delete_provider(provider_id, user["id"]):
        raise HTTPException(status_code=404, detail="provider not found")
    return {"ok": True}


@app.post("/api/providers/{provider_id}/models")
def api_add_provider_model(provider_id: int, req: AddModelRequest, user=Depends(get_current_user)):
    if chat_db.get_provider(provider_id, user["id"]) is None:
        raise HTTPException(status_code=404, detail="provider not found")
    model_name = req.model_name.strip()
    if not model_name:
        raise HTTPException(status_code=400, detail="model_name must not be empty")
    chat_db.add_provider_model(provider_id, model_name, datetime.now(timezone.utc).isoformat())
    return {"ok": True}


@app.delete("/api/provider-models/{model_id}")
def api_delete_provider_model(model_id: int, user=Depends(get_current_user)):
    if not chat_db.delete_provider_model(model_id, user["id"]):
        raise HTTPException(status_code=404, detail="model not found")
    return {"ok": True}


@app.get("/api/my-models")
def api_my_models(user=Depends(get_current_user)):
    """The pool the chat-creation dropdowns pick from, each annotated with
    whether it has been calibrated -- which is what gates starting a chat."""
    calibrations = chat_db.get_model_calibrations(user["id"])
    out = []
    for m in chat_db.list_user_models(user["id"]):
        cal = calibrations.get(m["model_name"])
        out.append({
            "id": m["id"], "model_name": m["model_name"], "provider_id": m["provider_id"],
            "provider_name": m["provider_name"], "provider": m["provider"],
            "has_stored_key": bool(m["has_stored_key"]),
            "calibrated": cal is not None,
            "quality_by_difficulty": cal["quality_by_difficulty"] if cal else None,
            "avg_cost": cal["avg_cost"] if cal else None,
            "avg_latency_ms": cal["avg_latency_ms"] if cal else None,
        })
    return out


def _resolve_provider_key(user_id: int, model_name: str, supplied: str | None) -> str | None:
    """A model's key comes from the provider's stored key if it has one,
    otherwise from what the caller supplied for this request."""
    row = chat_db.get_model_with_provider(user_id, model_name)
    if row is None:
        return supplied
    if row["api_key_encrypted"]:
        stored = key_vault.decrypt(row["api_key_encrypted"])
        if stored:
            return stored
    return supplied


@app.post("/api/calibrate-model")
def api_calibrate_model(req: CalibrateModelRequest, user=Depends(get_current_user)):
    """Calibrate one connected model. Keyed by model, so the result is
    reused wherever that model is slotted later."""
    return _calibrate_one_model(user["id"], req.model_name, req.api_key)


def _calibrate_one_model(user_id: int, raw_model_name: str, supplied_key: str | None) -> dict:
    model_name = raw_model_name.strip()
    if chat_db.get_model_with_provider(user_id, model_name) is None:
        raise HTTPException(status_code=404, detail=f"'{model_name}' is not a connected model")

    api_key = _resolve_provider_key(user_id, model_name, supplied_key)
    stats = calibrate_model(model_name, api_key, max_queries=DEFAULT_MAX_QUERIES)

    warnings = []
    if stats["n_queries"] == 0:
        if stats["n_rate_limited"] == stats["n_errors"]:
            raise HTTPException(
                status_code=502,
                detail=f"Rate limited on all {stats['n_errors']} calibration calls -- this is a "
                       f"provider quota issue, not a bad model string. Try again later.",
            )
        raise HTTPException(
            status_code=502,
            detail=f"All {stats['n_errors']} calibration calls failed. The provider said: "
                   f"{stats['first_error'] or 'no error detail returned'}",
        )

    unmeasured = [d for d, v in stats["quality_by_difficulty"].items() if v is None]
    if unmeasured:
        cause = "rate limiting" if stats["n_rate_limited"] else "errors on those queries"
        warnings.append(
            f"No {'/'.join(unmeasured)} score -- every query at that difficulty failed ({cause}). "
            f"Routing for it falls back to the strongest model in the session."
        )
    elif stats["n_rate_limited"]:
        warnings.append(
            f"Lost {stats['n_rate_limited']} queries to rate limiting; scores come from the "
            f"{stats['n_queries']} that completed."
        )

    chat_db.set_model_calibration(
        user_id, model_name, stats["avg_quality"], stats["avg_cost"],
        stats["avg_latency_ms"], stats["n_queries"], datetime.now(timezone.utc).isoformat(),
        quality_by_difficulty=stats["quality_by_difficulty"],
    )
    return {"model_name": model_name, "stats": stats, "warnings": warnings}


# --- model/tier introspection ----------------------------------------------

@app.get("/api/tiers")
def api_tiers(user=Depends(get_current_user)):
    """The models behind our own built-in tiers, so the chat UI can show real
    names. A BYOM session's own models come from the chat itself."""
    return {"builtin": DEFAULT_TIER_MODELS}


@app.get("/api/calibration")
def api_get_calibration(user=Depends(get_current_user)):
    """Every model this user has calibrated, keyed by model rather than by
    tier -- one measurement, reusable in any slot of any session."""
    calibrations = chat_db.get_model_calibrations(user["id"])
    return {
        "results": [
            {"model_name": name, **stats} for name, stats in sorted(calibrations.items())
        ],
    }


def _tier_map_for(user_id: int, tier_models: dict) -> dict:
    """The difficulty->tier map implied by the models in *this session*.

    Calibration is stored per model, so the same model keeps its scores
    whether it's slotted as cheap here and mid somewhere else; the map is
    assembled per session from whichever three models it was created with.
    """
    calibrations = chat_db.get_model_calibrations(user_id)
    quality_by_tier = {
        tier: (calibrations.get(model_name) or {}).get("quality_by_difficulty")
        for tier, model_name in tier_models.items()
    }
    return derive_tier_map(quality_by_tier, list(tier_models.keys()))


@app.post("/api/v1/calibrate")
def api_v1_calibrate(req: CalibrateModelRequest, user=Depends(get_user_from_api_key)):
    """Calibrate one connected model via an API key -- same effect as the
    Settings-page flow, for integrations that never touch the web UI."""
    return _calibrate_one_model(user["id"], req.model_name, req.api_key)


# --- BYOM: pass-through routing endpoint (API mode) -------------------------

@app.post("/api/v1/route")
def api_route(req: RouteRequest, user=Depends(get_user_from_api_key)):
    tier_models = req.models
    if not tier_models:
        raise HTTPException(
            status_code=400,
            detail="pass 'models' -- a tier->model map drawn from your connected models, "
                   "e.g. {\"cheap\": \"groq/llama-3.1-8b-instant\", \"frontier\": \"openai/gpt-4o\"}",
        )
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    calibrations = chat_db.get_model_calibrations(user["id"])
    for tier, model_name in tier_models.items():
        if tier not in ("cheap", "mid", "frontier"):
            raise HTTPException(status_code=400, detail=f"unknown tier '{tier}'")
        if chat_db.get_model_with_provider(user["id"], model_name) is None:
            raise HTTPException(
                status_code=400, detail=f"'{model_name}' is not a connected model",
            )
        if model_name not in calibrations:
            raise HTTPException(
                status_code=400,
                detail=f"'{model_name}' has not been calibrated -- POST /api/v1/calibrate first",
            )

    tier_api_keys = {
        tier: key for tier, model_name in tier_models.items()
        if (key := _resolve_provider_key(user["id"], model_name,
                                         req.tier_api_keys.get(tier))) is not None
    }

    try:
        result = run_cascade(
            [m.model_dump() for m in req.messages],
            tier_models=tier_models, tier_api_keys=tier_api_keys,
            difficulty_to_tier=_tier_map_for(user["id"], tier_models),
        )
    except AllTiersUnavailable:
        raise HTTPException(
            status_code=503,
            detail="All configured tiers are temporarily unavailable. Please try again shortly.",
        )
    except Exception:
        raise HTTPException(status_code=502, detail="Something went wrong generating a response.")

    # Baseline: what the strongest tier *this user* configured would have
    # cost for the same token count -- not our own built-in frontier.
    order = ["frontier", "mid", "cheap"]
    strongest_configured = next((t for t in order if t in tier_models), result["final_tier"])
    baseline_cost = estimate_cost_for_model(
        tier_models[strongest_configured], result["tokens_in"], result["tokens_out"],
    )

    now = datetime.now(timezone.utc).isoformat()
    chat_db.log_usage(
        user["id"], "api", result["difficulty"], result["initial_tier"], result["final_tier"],
        result["escalated"], result["total_cost"], baseline_cost, result["total_latency_ms"], now,
        api_key_id=user["api_key_id"],
    )

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result["final_tier"],
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result["text"]},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": result["tokens_in"],
            "completion_tokens": result["tokens_out"],
            "total_tokens": (result["tokens_in"] or 0) + (result["tokens_out"] or 0),
        },
        "_router": {
            "difficulty": result["difficulty"],
            "initial_tier": result["initial_tier"],
            "final_tier": result["final_tier"],
            "escalated": result["escalated"],
            "escalation_reasons": result["escalation_reasons"],
            "cost": result["total_cost"],
            "baseline_cost": baseline_cost,
            "latency_ms": round(result["total_latency_ms"], 1),
        },
    }


# --- BYOM: usage dashboard --------------------------------------------------

@app.get("/api/usage")
def api_usage(user=Depends(get_current_user)):
    summary = chat_db.get_usage_summary(user["id"])
    log = chat_db.get_usage_log(user["id"], limit=100)
    total_cost = summary["totals"]["total_cost"]
    total_baseline = summary["totals"]["total_baseline_cost"]
    return {
        "n_calls": summary["totals"]["n_calls"],
        "total_cost": total_cost,
        "total_baseline_cost": total_baseline,
        "total_cost_saved": max(0.0, total_baseline - total_cost),
        "by_tier": summary["by_tier"],
        "log": [
            {
                "source": r["source"], "difficulty": r["difficulty"],
                "initial_tier": r["initial_tier"], "final_tier": r["final_tier"],
                "escalated": bool(r["escalated"]), "cost": r["cost"],
                "baseline_cost": r["baseline_cost"], "latency_ms": r["latency_ms"],
                "timestamp": r["timestamp"],
            }
            for r in log
        ],
    }


# --- unified sessions (chats + API-key sessions) ----------------------------

@app.get("/api/sessions")
def api_sessions_overview(user=Depends(get_current_user)):
    return chat_db.get_sessions_overview(user["id"])


@app.get("/api/sessions/api/{key_id}")
def api_session_detail(key_id: int, user=Depends(get_current_user)):
    key, calls = chat_db.get_api_session_detail(key_id, user["id"])
    if key is None:
        raise HTTPException(status_code=404, detail="session not found")

    total_cost = sum(c["cost"] or 0.0 for c in calls)
    total_baseline = sum(c["baseline_cost"] or 0.0 for c in calls)
    return {
        "id": key["id"],
        "name": key["name"] or f"API session {key['key_prefix']}",
        "key_prefix": key["key_prefix"],
        "created_at": key["created_at"],
        "revoked": bool(key["revoked"]),
        "total_cost": total_cost,
        "total_baseline_cost": total_baseline,
        "cost_saved": max(0.0, total_baseline - total_cost),
        "calls": [
            {
                "difficulty": c["difficulty"], "initial_tier": c["initial_tier"],
                "final_tier": c["final_tier"], "escalated": bool(c["escalated"]),
                "cost": c["cost"], "baseline_cost": c["baseline_cost"],
                "latency_ms": c["latency_ms"], "timestamp": c["timestamp"],
            }
            for c in calls
        ],
    }
