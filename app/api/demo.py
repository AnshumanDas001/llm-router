"""The signed-out demo at /try: the real chat app against the built-in
tiers, capped per device, with nothing persisted but the cap."""
import os
import uuid

from fastapi import APIRouter, Cookie, Header, HTTPException, Response
from fastapi.responses import StreamingResponse

from app.api.common import (
    COOKIE_SECURE,
    DAILY_PROMPT_LIMIT,
    baseline_cost,
    check_routing_mode,
    log_result,
    now,
    sse,
    tier_map_for,
)
from app.api.schemas import ClassifyRequest, DemoRequest
from app.routing.cascade import DEFAULT_TIER_MODELS, AllTiersUnavailable, run_cascade_stream
from app.routing.classifier import classify_initial_tier
from app.storage import chat_db

router = APIRouter()

DEMO_COOKIE_NAME = "tl_demo"
DEMO_PROMPT_LIMIT = int(os.getenv("DEMO_PROMPT_LIMIT", "3"))
DEMO_HISTORY_TURNS = 6            # earlier turns replayed per demo prompt (bounds cost)
DEMO_COOKIE_MAX_AGE = 60 * 60 * 24 * 365


def _demo_id(cookie: str | None, header: str | None) -> str:
    """The demo cap is per device. The id lives in an httponly cookie and,
    mirrored by the page, in localStorage sent back as a header -- clearing
    one leaves the other, so the cap survives what a casual reset would
    otherwise bypass. Clearing all site data does reset it; that's the
    limit of what a browser lets a site do without fingerprinting."""
    for candidate in (cookie, header):
        if candidate and len(candidate) == 32 and candidate.isalnum():
            return candidate
    return uuid.uuid4().hex


def _demo_quota(demo_id: str) -> dict:
    used = chat_db.get_demo_count(demo_id)
    return {"device": demo_id, "used": min(used, DEMO_PROMPT_LIMIT), "limit": DEMO_PROMPT_LIMIT,
            "remaining": max(0, DEMO_PROMPT_LIMIT - used)}


def _set_demo_cookie(response: Response, demo_id: str) -> None:
    response.set_cookie(DEMO_COOKIE_NAME, demo_id, max_age=DEMO_COOKIE_MAX_AGE,
                        httponly=True, samesite="lax", secure=COOKIE_SECURE)


@router.get("/api/try/quota")
def api_try_quota(response: Response, tl_demo: str | None = Cookie(default=None),
                  x_demo_device: str | None = Header(default=None)):
    demo_id = _demo_id(tl_demo, x_demo_device)
    _set_demo_cookie(response, demo_id)
    return _demo_quota(demo_id)


@router.post("/api/try/classify")
def api_try_classify(req: ClassifyRequest):
    """The demo's predicted-tier badges: the embedding classifier only, no
    model call, so it's safe to expose without an account."""
    tier, difficulty = classify_initial_tier(req.content, tier_map_for(0, None))
    return {"tier": tier, "difficulty": difficulty}


@router.get("/api/try/tiers")
def api_try_tiers():
    return {"builtin": DEFAULT_TIER_MODELS}


@router.post("/api/try")
def api_try(req: DemoRequest, tl_demo: str | None = Cookie(default=None),
            x_demo_device: str | None = Header(default=None)):
    """Signed-out demo: the real chat app against the built-in tiers, capped
    per device. The cap bounds cost per casual visitor; it is not meant to
    be unbypassable."""
    content = req.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="content must not be empty")
    check_routing_mode(req.routing_mode)

    demo_id = _demo_id(tl_demo, x_demo_device)
    if chat_db.get_demo_count(demo_id) >= DEMO_PROMPT_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Demo limit reached ({DEMO_PROMPT_LIMIT} prompts on this device). "
                   f"Create a free account for {DAILY_PROMPT_LIMIT} a day.",
        )
    used = chat_db.bump_demo_count(demo_id, now())
    history = [m.model_dump() for m in req.history[-DEMO_HISTORY_TURNS:]]
    messages = history + [{"role": "user", "content": content}]

    def event_stream():
        try:
            for event in run_cascade_stream(messages, difficulty_to_tier=tier_map_for(0, None),
                                            skip_cheapest=(req.routing_mode == "direct")):
                if event["type"] == "done":
                    event["baseline_cost"] = baseline_cost(event, None)
                    event["saved"] = max(0.0, event["baseline_cost"] - event["total_cost"])
                    event["remaining"] = max(0, DEMO_PROMPT_LIMIT - used)
                    log_result(content, event)
                yield sse(event)
        except AllTiersUnavailable:
            yield sse({"type": "error", "detail": "All model tiers are busy right now. Try again shortly."})
        except Exception:
            yield sse({"type": "error", "detail": "Something went wrong generating a response."})

    stream = StreamingResponse(event_stream(), media_type="text/event-stream",
                               headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
    _set_demo_cookie(stream, demo_id)
    return stream
