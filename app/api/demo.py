"""The signed-out demo at /try: the real chat app against the built-in
tiers, capped per device, with nothing persisted but the cap."""
import hashlib
import os
import uuid

from fastapi import APIRouter, Cookie, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from app.api.common import (
    COOKIE_SECURE,
    DAILY_PROMPT_LIMIT,
    baseline_cost,
    builtin_routing,
    check_routing_mode,
    finish_trace,
    log_result,
    now,
    sse,
    tier_map_for,
    tokens_used,
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

# The device cap is easy to reset on purpose (clear site data, new private
# window), so two more limits bound what the demo can cost:
#   per network  prompts per day from one IP address. Generous, because a
#                school or office shares one address. Stored as a salted
#                hash, never the address itself.
#   in total     tokens per day across every demo visitor. Whatever anyone
#                does to the other two, this caps the bill: 150,000 tokens is
#                about $1.40 if every one of them were frontier thinking.
DEMO_IP_DAILY_LIMIT = int(os.getenv("DEMO_IP_DAILY_LIMIT", "6"))
DEMO_DAILY_TOKEN_BUDGET = int(os.getenv("DEMO_DAILY_TOKEN_BUDGET", "150000"))
_IP_SALT = os.getenv("ROUTER_SECRET_KEY") or "thriftllm-demo"


def _network_id(request: Request) -> str:
    ip = request.client.host if request.client else "unknown"
    return "demo-ip:" + hashlib.sha256(f"{_IP_SALT}:{ip}".encode()).hexdigest()[:20]


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
    return {"builtin": DEFAULT_TIER_MODELS, "routing": builtin_routing()}


@router.get("/api/routing")
def api_routing():
    """Public: the built-in routing map and the calibration behind it. No
    model call -- the landing page and the routing explorer draw from it."""
    return builtin_routing()


@router.post("/api/try")
def api_try(req: DemoRequest, request: Request, tl_demo: str | None = Cookie(default=None),
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
    network = _network_id(request)
    if chat_db.get_daily_usage(network)["prompts"] >= DEMO_IP_DAILY_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="The demo limit for your network is used up for today. "
                   f"Create a free account for {DAILY_PROMPT_LIMIT} prompts a day.",
        )
    if chat_db.get_daily_usage("demo-all")["tokens"] >= DEMO_DAILY_TOKEN_BUDGET:
        raise HTTPException(
            status_code=429,
            detail="Today's demo budget is spent. It resets at 00:00 UTC, "
                   "or create a free account to keep going.",
        )
    used = chat_db.bump_demo_count(demo_id, now())
    chat_db.add_daily_usage(network, prompts=1)
    chat_db.add_daily_usage("demo-all", prompts=1)
    history = [m.model_dump() for m in req.history[-DEMO_HISTORY_TURNS:]]
    messages = history + [{"role": "user", "content": content}]

    def event_stream():
        try:
            for event in run_cascade_stream(messages, difficulty_to_tier=tier_map_for(0, None),
                                            skip_cheapest=(req.routing_mode == "direct")):
                if event["type"] == "done":
                    event["baseline_cost"] = baseline_cost(event, None)
                    event["trace"] = finish_trace(event, 0, None, event["baseline_cost"])
                    event["saved"] = max(0.0, event["baseline_cost"] - event["total_cost"])
                    event["remaining"] = max(0, DEMO_PROMPT_LIMIT - used)
                    chat_db.add_daily_usage("demo-all", tokens=tokens_used(event))
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
