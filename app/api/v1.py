"""Programmatic access with an API key: the OpenAI-compatible endpoint on
the built-in stack, and the BYOM route/calibrate pair."""
import logging
import os

from fastapi import APIRouter, Depends, Header, HTTPException

from app.api.common import (
    baseline_cost,
    check_routing_mode,
    completion_response,
    enforce_daily_limit,
    log_result,
    now,
    resolve_tier_keys,
    tier_map_for,
    validate_byom_models,
)
from app.api.models import calibrate_one_model
from app.api.schemas import CalibrateModelRequest, ChatRequest, RouteRequest
from app.auth import get_user_from_api_key
from app.routing.cascade import AllTiersUnavailable, run_cascade
from app.storage import chat_db

router = APIRouter()

# The OpenAI-compatible endpoint spends real money on the built-in tiers, so
# a public deploy must not expose it without a key. The eval harness runs it
# locally without one; ALLOW_ANON_V1=1 opts into that.
ALLOW_ANON_V1 = os.getenv("ALLOW_ANON_V1", "").lower() in ("1", "true", "yes")


@router.post("/v1/chat/completions")
def chat_completions(req: ChatRequest, authorization: str | None = Header(default=None)):
    if not ALLOW_ANON_V1:
        if not authorization:
            raise HTTPException(status_code=401, detail="This endpoint needs an API key (Authorization: Bearer ...)")
        enforce_daily_limit(get_user_from_api_key(authorization))
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    try:
        result = run_cascade([m.model_dump() for m in req.messages],
                             difficulty_to_tier=tier_map_for(0, None))
    except AllTiersUnavailable:
        raise HTTPException(
            status_code=503,
            detail="All model tiers are temporarily rate limited. Please try again in a few minutes.",
        )
    except Exception:
        # The user gets a clean 502; the operator gets the traceback, which
        # a silent except used to swallow (two eval queries once 502'd with
        # no trace of why).
        logging.exception("cascade failed for /v1/chat/completions")
        raise HTTPException(status_code=502, detail="Something went wrong generating a response.")

    log_result(req.messages[-1].content, result)
    return completion_response(result)


@router.post("/api/v1/calibrate")
def api_v1_calibrate(req: CalibrateModelRequest, user=Depends(get_user_from_api_key)):
    """Calibrate one connected model via an API key -- same effect as the
    Settings-page flow, for integrations that never touch the web UI."""
    return calibrate_one_model(user["id"], req.model_name, req.api_key)


@router.post("/api/v1/route")
def api_route(req: RouteRequest, user=Depends(get_user_from_api_key)):
    """BYOM pass-through: route across the caller's own connected models."""
    enforce_daily_limit(user)
    tier_models = req.models
    if not tier_models:
        raise HTTPException(
            status_code=400,
            detail="pass 'models' -- a tier->model map drawn from your connected models, "
                   "e.g. {\"cheap\": \"groq/llama-3.1-8b-instant\", \"frontier\": \"openai/gpt-4o\"}",
        )
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    validate_byom_models(user["id"], tier_models, "POST /api/v1/calibrate first")
    check_routing_mode(req.routing_mode)

    try:
        result = run_cascade(
            [m.model_dump() for m in req.messages],
            tier_models=tier_models,
            tier_api_keys=resolve_tier_keys(user["id"], tier_models, req.tier_api_keys),
            difficulty_to_tier=tier_map_for(user["id"], tier_models),
            skip_cheapest=req.routing_mode == "direct",
        )
    except AllTiersUnavailable:
        raise HTTPException(
            status_code=503,
            detail="All configured tiers are temporarily unavailable. Please try again shortly.",
        )
    except Exception:
        raise HTTPException(status_code=502, detail="Something went wrong generating a response.")

    baseline = baseline_cost(result, tier_models)
    chat_db.log_usage(
        user["id"], "api", result["difficulty"], result["initial_tier"], result["final_tier"],
        result["escalated"], result["total_cost"], baseline, result["total_latency_ms"], now(),
        api_key_id=user["api_key_id"],
    )
    return completion_response(result, baseline_cost=baseline)
