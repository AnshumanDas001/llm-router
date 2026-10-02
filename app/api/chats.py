"""Chats in the web app: create, list, rename, and send messages through
the cascade (streamed or not)."""
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api.common import (
    baseline_cost,
    check_routing_mode,
    daily_allowance,
    enforce_daily_limit,
    finish_trace,
    log_result,
    now,
    provider_needs_key,
    resolve_tier_keys,
    sse,
    tier_map_for,
    tokens_used,
    validate_byom_models,
)
from app.api.schemas import ClassifyRequest, CreateChatRequest, SendMessageRequest, UpdateChatRequest
from app.auth import get_current_user
from app.routing.cascade import AllTiersUnavailable, plan_sequence, run_cascade, run_cascade_stream
from app.routing.classifier import classify_initial_tier
from app.routing.policy import TIER_ORDER
from app.storage import chat_db

router = APIRouter()


@router.get("/api/chats")
def api_list_chats(user=Depends(get_current_user)):
    return [
        {
            "id": c["id"],
            "title": c["title"] or "New chat",
            "preview": c["preview"],
            "pinned": bool(c["pinned"]),
            "mode": c["mode"],
            "routing_mode": c["routing_mode"],
            "created_at": c["created_at"],
            "total_cost": c["total_cost"],
            "total_baseline_cost": c["total_baseline_cost"],
            "cost_saved": max(0.0, c["total_baseline_cost"] - c["total_cost"]),
        }
        for c in chat_db.list_chats(user["id"])
    ]


@router.get("/api/chat-stats")
def api_chat_stats(user=Depends(get_current_user)):
    """Real aggregate numbers (not simulated) behind the sidebar's router
    status card -- computed from this user's own chat_messages history."""
    stats = chat_db.get_chat_stats(user["id"])
    stats["daily"] = daily_allowance(user["id"])
    return stats


@router.post("/api/classify")
def api_classify(req: ClassifyRequest, user=Depends(get_current_user)):
    """Runs only the upfront embedding classifier -- no model call, no cost
    -- so the UI can show a live 'predicted tier' as the user types. Given a
    chat, it predicts with that chat's own routing map and mode, so the
    badge shows where the message will really start."""
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="content must not be empty")
    tier_models, direct = None, False
    if req.chat_id is not None:
        chat = chat_db.get_chat(req.chat_id, user["id"])
        if chat is None:
            raise HTTPException(status_code=404, detail="chat not found")
        tier_models = chat_db.get_chat_models(req.chat_id) if chat["mode"] == "byom" else None
        direct = chat["routing_mode"] == "direct"
    tier, difficulty = classify_initial_tier(req.content, tier_map_for(user["id"], tier_models))
    available = [t for t in TIER_ORDER if tier_models is None or t in tier_models]
    if available:
        tier = plan_sequence(tier, available, direct)[0]
    return {"tier": tier, "difficulty": difficulty}


@router.post("/api/chats")
def api_create_chat(req: CreateChatRequest, user=Depends(get_current_user)):
    if req.mode not in ("builtin", "byom"):
        raise HTTPException(status_code=400, detail="mode must be 'builtin' or 'byom'")
    check_routing_mode(req.routing_mode)

    if req.mode == "byom":
        if not req.models:
            raise HTTPException(status_code=400, detail="select at least one model for this session")
        validate_byom_models(user["id"], req.models,
                             "calibrate it before starting a session with it")

    chat_id = chat_db.create_chat(user["id"], req.title, now(), mode=req.mode,
                                  routing_mode=req.routing_mode)
    if req.mode == "byom":
        provider_ids = {}
        for tier, model_name in req.models.items():
            row = chat_db.get_model_with_provider(user["id"], model_name)
            provider_ids[tier] = row["provider_id"] if row else None
        chat_db.set_chat_models(chat_id, req.models, provider_ids)
    return {"id": chat_id, "mode": req.mode}


@router.patch("/api/chats/{chat_id}")
def api_update_chat(chat_id: int, req: UpdateChatRequest, user=Depends(get_current_user)):
    if chat_db.get_chat(chat_id, user["id"]) is None:
        raise HTTPException(status_code=404, detail="chat not found")
    if req.title is not None:
        title = req.title.strip()
        if not title:
            raise HTTPException(status_code=400, detail="title must not be empty")
        chat_db.rename_chat(chat_id, user["id"], title)
    if req.pinned is not None:
        chat_db.set_chat_pinned(chat_id, user["id"], req.pinned)
    if req.routing_mode is not None:
        check_routing_mode(req.routing_mode)
        chat_db.set_chat_routing_mode(chat_id, user["id"], req.routing_mode)
    return {"ok": True}


@router.delete("/api/chats/{chat_id}")
def api_delete_chat(chat_id: int, user=Depends(get_current_user)):
    if not chat_db.delete_chat(chat_id, user["id"]):
        raise HTTPException(status_code=404, detail="chat not found")
    return {"ok": True}


@router.get("/api/chats/{chat_id}")
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
            elif not row["api_key_encrypted"] and provider_needs_key(row["provider"]):
                needs_key.append(tier)
    return {
        "id": chat["id"],
        "title": chat["title"] or "New chat",
        "pinned": bool(chat["pinned"]),
        "mode": chat["mode"],
        "routing_mode": chat["routing_mode"],
        "byom_tiers": chat_db.get_chat_models(chat_id) if chat["mode"] == "byom" else {},
        # which tier each difficulty band starts at, for this chat's models
        "routing_map": tier_map_for(
            user["id"], chat_db.get_chat_models(chat_id) if chat["mode"] == "byom" else None),
        "byom_needs_key": needs_key,
        "messages": [
            {
                "role": m["role"], "content": m["content"], "tier": m["tier"],
                "cost": m["cost"], "baseline_cost": m["baseline_cost"],
                "escalated": bool(m["escalated"]) if m["escalated"] is not None else False,
                "latency_ms": m["latency_ms"], "difficulty": m["difficulty"],
                "escalation_reasons": json.loads(m["escalation_reasons"]) if m["escalation_reasons"] else [],
                "trace": json.loads(m["route_trace"]) if m["route_trace"] else None,
            }
            for m in messages
        ],
        "total_cost": totals["total_cost"],
        "total_baseline_cost": totals["total_baseline_cost"],
        "cost_saved": max(0.0, totals["total_baseline_cost"] - totals["total_cost"]),
    }


def _prepare_send(chat_id: int, req: SendMessageRequest, user) -> dict:
    """Shared setup for both send paths: validate, persist the user turn,
    and resolve this chat's models, keys and routing map -- as the keyword
    arguments run_cascade takes."""
    chat = chat_db.get_chat(chat_id, user["id"])
    if chat is None:
        raise HTTPException(status_code=404, detail="chat not found")
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="content must not be empty")
    enforce_daily_limit(user, builtin=chat["mode"] != "byom")

    history = chat_db.get_chat_messages(chat_id)
    chat_db.add_chat_message(chat_id, "user", req.content, now())
    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": req.content})

    tier_models = chat_db.get_chat_models(chat_id) if chat["mode"] == "byom" else None
    if chat["mode"] == "byom" and not tier_models:
        raise HTTPException(
            status_code=400,
            detail="this session has no models attached -- start a new session and pick models",
        )
    # Stored provider key if the user chose to save one, otherwise the key
    # they supplied for this request from the browser.
    tier_api_keys = resolve_tier_keys(user["id"], tier_models, req.tier_api_keys) if tier_models else None
    return {
        "messages": messages, "tier_models": tier_models, "tier_api_keys": tier_api_keys,
        # BYOM routes by what this session's own models measured; the
        # built-in stack by what calibration measured for it.
        "difficulty_to_tier": tier_map_for(user["id"], tier_models),
        "skip_cheapest": chat["routing_mode"] == "direct",
    }


def _finish_send(chat_id: int, user_id: int, content: str, result: dict,
                 tier_models: dict | None) -> dict:
    """Shared teardown: price the answer, persist it with its decision
    trace, and return what both send paths report back."""
    baseline = baseline_cost(result, tier_models)
    trace = finish_trace(result, user_id, tier_models, baseline)
    if tier_models is None:            # the token allowance covers our models only
        chat_db.add_daily_usage(f"user:{user_id}", tokens=tokens_used(result))
    chat_db.add_chat_message(
        chat_id, "assistant", result["text"], now(),
        tier=result["final_tier"], cost=result["total_cost"], baseline_cost=baseline,
        escalated=result["escalated"], latency_ms=result["total_latency_ms"],
        difficulty=result["difficulty"], escalation_reasons=result["escalation_reasons"],
        route_trace=trace,
    )
    log_result(content, result)
    totals = chat_db.get_chat_totals(chat_id)
    return {
        "baseline_cost": baseline,
        "trace": trace,
        "chat_total_cost": totals["total_cost"],
        "chat_total_baseline_cost": totals["total_baseline_cost"],
        "chat_cost_saved": max(0.0, totals["total_baseline_cost"] - totals["total_cost"]),
    }


@router.post("/api/chats/{chat_id}/messages/stream")
def api_send_message_stream(chat_id: int, req: SendMessageRequest, user=Depends(get_current_user)):
    """Streaming form of the send path. Setup runs before the response starts
    so validation failures are still real HTTP errors rather than an error
    event buried in a 200 stream."""
    plan = _prepare_send(chat_id, req, user)

    def event_stream():
        try:
            for event in run_cascade_stream(**plan):
                if event["type"] == "done":
                    event.update(_finish_send(chat_id, user["id"], req.content, event, plan["tier_models"]))
                yield sse(event)
        except AllTiersUnavailable:
            yield sse({"type": "error",
                       "detail": "All model tiers are temporarily rate limited. Try again shortly."})
        except Exception:
            yield sse({"type": "error", "detail": "Something went wrong generating a response."})

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@router.post("/api/chats/{chat_id}/messages")
def api_send_message(chat_id: int, req: SendMessageRequest, user=Depends(get_current_user)):
    plan = _prepare_send(chat_id, req, user)
    try:
        result = run_cascade(**plan)
    except AllTiersUnavailable:
        raise HTTPException(
            status_code=503,
            detail="All model tiers are temporarily rate limited. Please try again in a few minutes.",
        )
    except Exception:
        raise HTTPException(status_code=502, detail="Something went wrong generating a response.")

    totals = _finish_send(chat_id, user["id"], req.content, result, plan["tier_models"])
    return {
        "content": result["text"],
        "tier": result["final_tier"],
        "difficulty": result["difficulty"],
        "escalated": result["escalated"],
        "escalation_reasons": result["escalation_reasons"],
        "cost": result["total_cost"],
        "latency_ms": result["total_latency_ms"],
        **totals,
    }
