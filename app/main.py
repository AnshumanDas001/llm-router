import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Cookie, Depends, FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app import chat_db
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
from app.cascade import AllTiersUnavailable, run_cascade
from app.db import init_db, log_cascade

app = FastAPI(title="LLM Router")
DEMO_HTML_PATH = Path(__file__).resolve().parent / "demo.html"
CHAT_APP_HTML_PATH = Path(__file__).resolve().parent / "chat_app.html"
SETTINGS_HTML_PATH = Path(__file__).resolve().parent / "settings.html"


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


class SendMessageRequest(BaseModel):
    content: str


class CreateApiKeyRequest(BaseModel):
    name: str | None = None


class TierModelConfig(BaseModel):
    model: str
    api_key: str | None = None


class CalibrateRequest(BaseModel):
    tiers: dict[str, TierModelConfig]  # keys: "cheap" / "mid" / "frontier"


class RouteRequest(BaseModel):
    messages: list[Message]
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
            "created_at": c["created_at"],
            "total_cost": c["total_cost"],
            "total_baseline_cost": c["total_baseline_cost"],
            "cost_saved": max(0.0, c["total_baseline_cost"] - c["total_cost"]),
        }
        for c in chats
    ]


@app.post("/api/chats")
def api_create_chat(req: CreateChatRequest, user=Depends(get_current_user)):
    chat_id = chat_db.create_chat(user["id"], req.title, datetime.now(timezone.utc).isoformat())
    return {"id": chat_id}


@app.get("/api/chats/{chat_id}")
def api_get_chat(chat_id: int, user=Depends(get_current_user)):
    chat = chat_db.get_chat(chat_id, user["id"])
    if chat is None:
        raise HTTPException(status_code=404, detail="chat not found")

    messages = chat_db.get_chat_messages(chat_id)
    totals = chat_db.get_chat_totals(chat_id)
    return {
        "id": chat["id"],
        "title": chat["title"] or "New chat",
        "messages": [
            {
                "role": m["role"], "content": m["content"], "tier": m["tier"],
                "cost": m["cost"], "baseline_cost": m["baseline_cost"],
                "escalated": bool(m["escalated"]) if m["escalated"] is not None else False,
                "latency_ms": m["latency_ms"],
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

    try:
        result = run_cascade(messages)
    except AllTiersUnavailable:
        raise HTTPException(
            status_code=503,
            detail="All model tiers are temporarily rate limited. Please try again in a few minutes.",
        )
    except Exception:
        raise HTTPException(status_code=502, detail="Something went wrong generating a response.")

    baseline_cost = estimate_frontier_cost(result["tokens_in"], result["tokens_out"])

    chat_db.add_chat_message(
        chat_id, "assistant", result["text"], datetime.now(timezone.utc).isoformat(),
        tier=result["final_tier"], cost=result["total_cost"], baseline_cost=baseline_cost,
        escalated=result["escalated"], latency_ms=result["total_latency_ms"],
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


# --- BYOM: model config (tier -> model name; no keys stored) ----------------

@app.get("/api/models")
def api_get_models(user=Depends(get_current_user)):
    return chat_db.get_model_configs(user["id"])


@app.get("/api/calibration")
def api_get_calibration(user=Depends(get_current_user)):
    results = chat_db.get_calibration_results(user["id"])
    return [
        {
            "tier": r["tier"], "model_name": r["model_name"], "avg_quality": r["avg_quality"],
            "avg_cost": r["avg_cost"], "avg_latency_ms": r["avg_latency_ms"],
            "n_queries": r["n_queries"], "created_at": r["created_at"],
        }
        for r in results
    ]


@app.post("/api/calibrate")
def api_calibrate(req: CalibrateRequest, user=Depends(get_current_user)):
    if not req.tiers:
        raise HTTPException(status_code=400, detail="provide at least one tier to calibrate")
    for tier in req.tiers:
        if tier not in ("cheap", "mid", "frontier"):
            raise HTTPException(status_code=400, detail=f"unknown tier '{tier}'")

    now = datetime.now(timezone.utc).isoformat()
    results = {}
    for tier, cfg in req.tiers.items():
        stats = calibrate_model(cfg.model, cfg.api_key, max_queries=DEFAULT_MAX_QUERIES)
        # Persist the model name + derived numbers; cfg.api_key never gets
        # written anywhere past this point.
        chat_db.set_model_config(user["id"], tier, cfg.model, now)
        chat_db.set_calibration_result(
            user["id"], tier, cfg.model, stats["avg_quality"], stats["avg_cost"],
            stats["avg_latency_ms"], stats["n_queries"], now,
        )
        results[tier] = {"model": cfg.model, **stats}

    # Sanity-check the ordering the cascade assumes (cheap <= mid <= frontier
    # in quality) and warn, rather than silently misroute, if it doesn't hold.
    warnings = []
    order = ["cheap", "mid", "frontier"]
    configured_order = [t for t in order if t in results]
    for a, b in zip(configured_order, configured_order[1:]):
        if results[a]["avg_quality"] > results[b]["avg_quality"] + 0.05:
            warnings.append(
                f"'{a}' scored higher ({results[a]['avg_quality']:.2f}) than "
                f"'{b}' ({results[b]['avg_quality']:.2f}) on the calibration set -- "
                f"the cascade assumes each tier is at least as capable as the one before it."
            )

    return {"results": results, "warnings": warnings}


# --- BYOM: pass-through routing endpoint (API mode) -------------------------

@app.post("/api/v1/route")
def api_route(req: RouteRequest, user=Depends(get_user_from_api_key)):
    tier_models = chat_db.get_model_configs(user["id"])
    if not tier_models:
        raise HTTPException(
            status_code=400,
            detail="no models configured for this account -- calibrate at least one tier first",
        )
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    try:
        result = run_cascade(
            [m.model_dump() for m in req.messages],
            tier_models=tier_models, tier_api_keys=req.tier_api_keys,
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
