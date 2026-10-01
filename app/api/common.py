"""Helpers shared by more than one router: limits, routing maps, pricing
baselines, provider keys, and the OpenAI-shaped response."""
import json
import os
import time
import uuid
from datetime import datetime, timezone

from fastapi import HTTPException

from app.config import BUILTIN_CALIBRATION
from app.pricing import estimate_cost_for_model, estimate_frontier_cost
from app.routing import scorer
from app.routing.cascade import DEFAULT_TIER_MODELS, judge_for, learned_verifier_on
from app.routing.policy import TIER_ORDER, derive_tier_map
from app.storage import chat_db, key_vault
from app.storage.eval_log import log_cascade

# Cookies carry the Secure flag only when told to. The app is developed over
# plain http://localhost, where a Secure cookie would never be sent back and
# login would silently fail; a deployment behind TLS sets COOKIE_SECURE=1.
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "").lower() in ("1", "true", "yes")

# Signed-in accounts get this many prompts per UTC day across the chat UI and
# the API. It bounds what one free account can spend on the built-in tiers.
DAILY_PROMPT_LIMIT = int(os.getenv("DAILY_PROMPT_LIMIT", "10"))

ROUTING_MODES = ("cascade", "direct")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def check_routing_mode(mode: str) -> None:
    if mode not in ROUTING_MODES:
        raise HTTPException(status_code=400, detail="routing_mode must be 'cascade' or 'direct'")


def enforce_daily_limit(user) -> int:
    used = chat_db.count_prompts_today(user["id"])
    if used >= DAILY_PROMPT_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Daily limit reached ({DAILY_PROMPT_LIMIT} prompts per account). It resets at 00:00 UTC.",
        )
    return used


# --- providers and keys ------------------------------------------------------

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
    {"provider": "openrouter", "label": "OpenRouter (many providers, one key)", "needs_key": True},
    {"provider": "ollama", "label": "Ollama (local)", "needs_key": False},
]


def provider_needs_key(provider: str) -> bool:
    """Unknown providers default to needing a key -- better to ask for one
    that turns out to be unnecessary than to silently call without it."""
    for known in KNOWN_PROVIDERS:
        if known["provider"] == provider:
            return known["needs_key"]
    return True


def resolve_provider_key(user_id: int, model_name: str, supplied: str | None) -> str | None:
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


def resolve_tier_keys(user_id: int, tier_models: dict, supplied: dict) -> dict:
    return {
        tier: key for tier, model_name in tier_models.items()
        if (key := resolve_provider_key(user_id, model_name, supplied.get(tier))) is not None
    }


def validate_byom_models(user_id: int, tier_models: dict, calibrate_hint: str) -> None:
    """Every tier must be a known tier, a connected model, and calibrated --
    calibration is what produces the session's routing map, so an
    uncalibrated model would leave the router guessing."""
    calibrations = chat_db.get_model_calibrations(user_id)
    for tier, model_name in tier_models.items():
        if tier not in TIER_ORDER:
            raise HTTPException(status_code=400, detail=f"unknown tier '{tier}'")
        if chat_db.get_model_with_provider(user_id, model_name) is None:
            raise HTTPException(status_code=400, detail=f"'{model_name}' is not a connected model")
        if model_name not in calibrations:
            raise HTTPException(status_code=400,
                                detail=f"'{model_name}' has not been calibrated -- {calibrate_hint}")


# --- routing map -------------------------------------------------------------

JUDGE_PROBE_TOKENS = (450, 30)   # question + answer in; a low-effort YES/NO out
SCORER_PROBE_TOKENS = (450, 150)  # a second cheap sample plus a 1-token self-check


def judge_cost_estimate(tiers: list[str], tier_models: dict | None) -> float:
    """What one verdict costs. With the LLM judge: the second-cheapest
    configured tier priced at a typical judge call. With the learned
    verifier: the extra *cheap*-tier calls it makes instead -- $0 on a
    local model, and this is exactly what lets the routing policy send
    easy/medium questions to the cheap tier on a stack where the judge
    call alone used to cost more than mid's answer. Zero if there's only
    one tier (nothing to escalate to, so nothing verifies)."""
    if len(tiers) < 2:
        return 0.0
    judge_model = judge_for(tiers, tier_models, None)[0]
    judge = estimate_cost_for_model(judge_model, *JUDGE_PROBE_TOKENS)
    if learned_verifier_on(tier_models):
        # The gate answers some verifications itself, so only its uncertain
        # share reaches the judge. It costs extra cheap-tier calls only if
        # the trained scorer reads features that need them -- the shipped one
        # reads token confidence off the generation already made, and is free.
        own_cost = 0.0
        if scorer.uses(scorer.SIBLING_FEATURES) or scorer.uses(scorer.SELF_VERIFY_FEATURES):
            own_cost = estimate_cost_for_model(DEFAULT_TIER_MODELS[tiers[0]], *SCORER_PROBE_TOKENS)
        return own_cost + scorer.judge_fraction() * judge
    return judge


def tier_map_for(user_id: int, tier_models: dict | None) -> dict:
    """The difficulty->tier map implied by the models in *this session*,
    routing by expected cost with a quality floor (see routing/policy.py).

    Calibration is stored per model, so the same model keeps its numbers
    whether it's slotted as cheap here and mid somewhere else; the map is
    assembled per session from whichever tiers it was created with. The
    built-in stack uses the eval set's measurements of itself.
    """
    if tier_models is None:
        tiers = list(BUILTIN_CALIBRATION.keys())
        quality = {t: BUILTIN_CALIBRATION[t]["quality"] for t in tiers}
        cost = {t: BUILTIN_CALIBRATION[t]["cost"] for t in tiers}
    else:
        tiers = list(tier_models.keys())
        calibrations = chat_db.get_model_calibrations(user_id)
        quality = {t: (calibrations.get(m) or {}).get("quality_by_difficulty") for t, m in tier_models.items()}
        cost = {t: (calibrations.get(m) or {}).get("cost_by_difficulty") for t, m in tier_models.items()}
        # Older calibrations predate per-band cost; without it, fall back to
        # the quality-floor rule rather than pricing every tier at zero.
        if any(v is None for v in cost.values()):
            cost = None
    return derive_tier_map(quality, tiers, cost, judge_cost_estimate(tiers, tier_models))


# --- after an answer ---------------------------------------------------------

def baseline_cost(result: dict, tier_models: dict | None) -> float:
    """What the answer would have cost from the strongest tier available:
    for BYOM, the strongest model *this user* configured -- not our own
    built-in frontier, which they may not even use."""
    if tier_models:
        strongest = next((t for t in reversed(TIER_ORDER) if t in tier_models), result["final_tier"])
        return estimate_cost_for_model(tier_models[strongest], result["tokens_in"], result["tokens_out"])
    return estimate_frontier_cost(result["tokens_in"], result["tokens_out"], result["difficulty"])


def log_result(query: str, result: dict) -> None:
    log_cascade(
        query=query, difficulty=result["difficulty"], initial_tier=result["initial_tier"],
        final_tier=result["final_tier"], escalated=result["escalated"],
        escalation_reasons=result["escalation_reasons"], total_cost=result["total_cost"],
        total_latency_ms=result["total_latency_ms"], tokens_in=result["tokens_in"],
        tokens_out=result["tokens_out"], timestamp=now(), response_text=result["text"],
    )


def completion_response(result: dict, **router_extra) -> dict:
    """The cascade's result in OpenAI chat.completion shape, with routing
    details under `_router`."""
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
            **router_extra,
            "latency_ms": round(result["total_latency_ms"], 1),
        },
    }
