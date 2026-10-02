"""Programmatic access with an API key: the OpenAI-compatible endpoint on
the built-in stack, and the BYOM route/calibrate pair."""
import logging
import os

from fastapi import APIRouter, Depends, Header, HTTPException

from app.api.common import (
    baseline_cost,
    check_routing_mode,
    explain_start,
    completion_response,
    enforce_daily_limit,
    finish_trace,
    log_result,
    tokens_used,
    now,
    resolve_tier_keys,
    tier_map_for,
    validate_byom_models,
)
from app.api.models import calibrate_one_model
from app.api.schemas import CalibrateModelRequest, ChatRequest, ClassifyApiRequest, RouteRequest, RoutingMapRequest
from app.auth import get_user_from_api_key
from app.routing.cascade import AllTiersUnavailable, plan_sequence, run_cascade
from app.routing.classifier import DIFFICULTY_TO_TIER, classify_with_trace
from app.routing.policy import TIER_ORDER
from app.storage import chat_db

router = APIRouter()

# The OpenAI-compatible endpoint spends real money on the built-in tiers, so
# a public deploy must not expose it without a key. The eval harness runs it
# locally without one; ALLOW_ANON_V1=1 opts into that.
ALLOW_ANON_V1 = os.getenv("ALLOW_ANON_V1", "").lower() in ("1", "true", "yes")


@router.post("/v1/chat/completions")
def chat_completions(req: ChatRequest, authorization: str | None = Header(default=None)):
    user = None
    if not ALLOW_ANON_V1 or authorization:
        if not authorization:
            raise HTTPException(status_code=401, detail="This endpoint needs an API key (Authorization: Bearer ...)")
        user = get_user_from_api_key(authorization)
        enforce_daily_limit(user)
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
    baseline = baseline_cost(result, None)
    if user is not None:
        # Counted like every other prompt. This endpoint used to log nothing
        # here, so its calls never counted toward the daily prompt limit.
        chat_db.log_usage(
            user["id"], "openai", result["difficulty"], result["initial_tier"], result["final_tier"],
            result["escalated"], result["total_cost"], baseline, result["total_latency_ms"], now(),
            api_key_id=user["api_key_id"],
        )
        chat_db.add_daily_usage(f"user:{user['id']}", tokens=tokens_used(result))
    return completion_response(result, baseline_cost=baseline,
                               trace=finish_trace(result, 0, None, baseline))


def connect_model(user_id: int, model_name: str, provider: str | None, api_base: str | None) -> None:
    """Make model_name a connected model, so code can calibrate a model in
    one call without visiting the Models page. Models connected this way
    share one provider entry per provider type, named "<provider> (API)";
    no key is stored on it."""
    if chat_db.get_model_with_provider(user_id, model_name) is not None:
        return
    provider = (provider or (model_name.split("/", 1)[0] if "/" in model_name else "")).strip()
    if not provider:
        raise HTTPException(status_code=400,
                            detail=f"can't tell the provider of '{model_name}': use litellm's "
                                   "'provider/model' form or pass 'provider'")
    name = f"{provider} (API)"
    existing = next((p for p in chat_db.list_providers(user_id) if p["name"] == name), None)
    provider_id = existing["id"] if existing else chat_db.create_provider(
        user_id, name, provider, None, api_base, now())
    chat_db.add_provider_model(provider_id, model_name, now())


@router.post("/api/v1/calibrate")
def api_v1_calibrate(req: CalibrateModelRequest, user=Depends(get_user_from_api_key)):
    """Calibrate one model via an API key -- same effect as the Models-page
    flow, for code that never touches the web UI. Connects the model first
    if needed. The provider key is used for this run and never stored."""
    connect_model(user["id"], req.model_name.strip(), req.provider, req.api_base)
    return calibrate_one_model(user["id"], req.model_name, req.api_key)


@router.get("/api/v1/models")
def api_v1_models(user=Depends(get_user_from_api_key)):
    """Every model this account has calibrated, with its per-band numbers,
    so code can reuse a calibration instead of paying for it again."""
    return {"models": [
        {"model_name": name, "quality_by_difficulty": c.get("quality_by_difficulty"),
         "cost_by_difficulty": c.get("cost_by_difficulty"), "avg_quality": c.get("avg_quality"),
         "avg_cost": c.get("avg_cost"), "avg_latency_ms": c.get("avg_latency_ms"),
         "n_queries": c.get("n_queries"), "calibrated_at": c.get("created_at")}
        for name, c in sorted(chat_db.get_model_calibrations(user["id"]).items())
    ]}


@router.post("/api/v1/routing-map")
def api_v1_routing_map(req: RoutingMapRequest, user=Depends(get_user_from_api_key)):
    """Which tier each band starts at for these models, derived from their
    calibration exactly as /api/v1/route will derive it."""
    tier_models = req.models or None
    if tier_models:
        validate_byom_models(user["id"], tier_models, "POST /api/v1/calibrate first")
    return {"map": tier_map_for(user["id"], tier_models)}


@router.post("/api/v1/classify")
def api_v1_classify(req: ClassifyApiRequest, user=Depends(get_user_from_api_key)):
    """Where a prompt would start and why, without calling any model --
    free, and not counted against the daily limit."""
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt must not be empty")
    check_routing_mode(req.routing_mode)
    tier_models = req.models or None
    if tier_models:
        validate_byom_models(user["id"], tier_models, "POST /api/v1/calibrate first")
    band, classification = classify_with_trace(req.prompt)
    start = explain_start(user["id"], tier_models, band)
    mapped = start["chosen"] or DIFFICULTY_TO_TIER[band]
    available = [t for t in TIER_ORDER if tier_models is None or t in tier_models]
    tier = plan_sequence(mapped, available, req.routing_mode == "direct")[0]
    return {"band": band, "tier": tier,
            "trace": {"classification": classification,
                      "start": {**start, "tier": tier, "direct": tier != mapped}}}


@router.post("/api/v1/route")
def api_route(req: RouteRequest, user=Depends(get_user_from_api_key)):
    """BYOM pass-through: route across the caller's own connected models."""
    enforce_daily_limit(user, builtin=False)
    tier_models = req.models
    if not tier_models:
        raise HTTPException(
            status_code=400,
            detail="pass 'models' -- a tier->model map drawn from your connected models, "
                   "e.g. {\"cheap\": \"groq/openai/gpt-oss-20b\", \"frontier\": \"openai/gpt-4o\"}",
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
    return completion_response(result, baseline_cost=baseline,
                               trace=finish_trace(result, user["id"], tier_models, baseline))
