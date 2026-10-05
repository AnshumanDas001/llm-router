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
from app.routing import prompt_cache, scorer
from app.routing.cascade import DEFAULT_TIER_MODELS, judge_for, learned_verifier_on
from app.evaluation.datasets import DIFFICULTIES
from app.routing.policy import QUALITY_THRESHOLD, TIER_ORDER, derive_tier_map, expected_costs
from app.storage import chat_db, key_vault
from app.storage.eval_log import log_cascade

# Cookies carry the Secure flag only when told to. The app is developed over
# plain http://localhost, where a Secure cookie would never be sent back and
# login would silently fail; a deployment behind TLS sets COOKIE_SECURE=1.
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "").lower() in ("1", "true", "yes")

# Signed-in accounts get this many prompts per UTC day across the chat UI and
# the API. It bounds what one free account can spend on the built-in tiers.
DAILY_PROMPT_LIMIT = int(os.getenv("DAILY_PROMPT_LIMIT", "10"))

# ...and this many tokens per UTC day on the built-in models, counted across
# every attempt (escalations included) and including hidden reasoning. A
# prompt count alone doesn't bound spend: one competition-maths question
# can make the frontier think for 30,000 tokens (~$0.29). 50,000 is about
# $0.45 on the frontier at worst, and hundreds of everyday answers.
DAILY_TOKEN_LIMIT = int(os.getenv("DAILY_TOKEN_LIMIT", "50000"))

ROUTING_MODES = ("cascade", "direct")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def check_routing_mode(mode: str) -> None:
    if mode not in ROUTING_MODES:
        raise HTTPException(status_code=400, detail="routing_mode must be 'cascade' or 'direct'")


def enforce_daily_limit(user, builtin: bool = True) -> int:
    """Refuse once the account has used today's prompts -- or, on the
    built-in models, today's tokens. BYOM traffic runs on the user's own
    keys, so only the prompt cap applies to it."""
    used = chat_db.count_prompts_today(user["id"])
    if used >= DAILY_PROMPT_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Daily limit reached ({DAILY_PROMPT_LIMIT} prompts per account). It resets at 00:00 UTC.",
        )
    if builtin and tokens_today(user["id"]) >= DAILY_TOKEN_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Daily token limit reached ({DAILY_TOKEN_LIMIT:,} tokens on the built-in models). "
                   "It resets at 00:00 UTC; chats on your own models still work.",
        )
    return used


def tokens_today(user_id: int) -> int:
    return chat_db.get_daily_usage(f"user:{user_id}")["tokens"]


def tokens_used(result: dict) -> int:
    """Tokens an answer consumed: input and output of every attempt, so an
    escalation counts both answers, and a thinking model's hidden reasoning
    is included (providers bill it as output)."""
    attempts = (result.get("trace") or {}).get("attempts") or []
    total = sum((a.get("tokens_in") or 0) + (a.get("tokens_out") or 0) for a in attempts)
    return total or (result.get("tokens_in") or 0) + (result.get("tokens_out") or 0)


def daily_allowance(user_id: int) -> dict:
    return {"used": chat_db.count_prompts_today(user_id), "limit": DAILY_PROMPT_LIMIT,
            "tokens_used": tokens_today(user_id), "token_limit": DAILY_TOKEN_LIMIT}


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


def _routing_inputs(user_id: int, tier_models: dict | None):
    """(tiers, quality, cost, judge_cost) for the models in this session --
    the same numbers whether deriving the map or explaining it."""
    if tier_models is None:
        tiers = list(BUILTIN_CALIBRATION.keys())
        quality = {t: BUILTIN_CALIBRATION[t]["quality"] for t in tiers}
        cost = {t: BUILTIN_CALIBRATION[t]["cost"] for t in tiers}
    else:
        tiers = [t for t in TIER_ORDER if t in tier_models]
        calibrations = chat_db.get_model_calibrations(user_id)
        quality = {t: (calibrations.get(m) or {}).get("quality_by_difficulty") for t, m in tier_models.items()}
        cost = {t: (calibrations.get(m) or {}).get("cost_by_difficulty") for t, m in tier_models.items()}
        # Older calibrations predate per-band cost; without it, fall back to
        # the quality-floor rule rather than pricing every tier at zero.
        if any(v is None for v in cost.values()):
            cost = None
    return tiers, quality, cost, judge_cost_estimate(tiers, tier_models)


def history_for(messages: list[dict], tier_models: dict | None) -> dict | None:
    """What re-sending this conversation's history costs on each tier, with
    the cached rate where a tier still holds it (app/routing/prompt_cache.py).
    None for a first message, which has no history and routes as before."""
    if len(messages) < 2:
        return None
    models = tier_models or DEFAULT_TIER_MODELS
    tokens = prompt_cache.history_tokens(messages)
    return {t: prompt_cache.history_cost(m, messages, history=tokens)
            for t, m in models.items() if t in TIER_ORDER}


def _extra(history: dict | None) -> dict | None:
    return {t: h["cost"] for t, h in history.items()} if history else None


def tier_map_for(user_id: int, tier_models: dict | None, history: dict | None = None) -> dict:
    """The difficulty->tier map implied by the models in *this session*,
    routing by expected cost with a quality floor (see routing/policy.py).

    Calibration is stored per model, so the same model keeps its numbers
    whether it's slotted as cheap here and mid somewhere else; the map is
    assembled per session from whichever tiers it was created with. The
    built-in stack uses the eval set's measurements of itself.
    """
    tiers, quality, cost, judge = _routing_inputs(user_id, tier_models)
    return derive_tier_map(quality, tiers, cost, judge, _extra(history))


def explain_start(user_id: int, tier_models: dict | None, band: str,
                  history: dict | None = None) -> dict:
    """Why a band starts where it does: every tier's calibrated score on
    the band against the floor, and the expected cost of starting there
    (generation + verification + failure-weighted escalation), with the
    conversation's history priced in where there is one."""
    tiers, quality, cost, judge = _routing_inputs(user_id, tier_models)
    expected = expected_costs(tiers, quality, cost, judge, band, _extra(history)) if cost else {}
    rows = []
    for i, t in enumerate(tiers):
        q = (quality.get(t) or {}).get(band)
        rows.append({
            "tier": t,
            "model": (tier_models or DEFAULT_TIER_MODELS).get(t),
            "quality": q,
            "clears_floor": q is not None and q >= QUALITY_THRESHOLD,
            "last_tier": i == len(tiers) - 1,
            "expected_cost": expected.get(t),
            **({"history_cost": history[t]["cost"], "warm_tokens": history[t]["warm_tokens"]}
               if history and t in history else {}),
        })
    out = {"band": band, "floor": QUALITY_THRESHOLD, "judge_cost": judge, "tiers": rows,
           "chosen": tier_map_for(user_id, tier_models, history).get(band)}
    if history:
        any_h = next(iter(history.values()))
        out["history"] = {"tokens": any_h["tokens"],
                          "warm": [t for t, h in history.items() if h["warm_tokens"]],
                          "hit_rate": {t: h["hit_rate"] for t, h in history.items() if h["warm_tokens"]}}
    return out


def finish_trace(result: dict, user_id: int, tier_models: dict | None, baseline: float) -> dict:
    """The cascade's trace plus the routing explanation and the money."""
    trace = dict(result.get("trace") or {})
    start = dict(trace.get("start") or {})
    history = start.pop("history_by_tier", None)
    trace["start"] = {**start, **explain_start(user_id, tier_models, result["difficulty"], history)}
    trace["totals"] = {"cost": result["total_cost"], "baseline": baseline,
                       "saved": max(0.0, baseline - result["total_cost"]),
                       "latency_ms": round(result["total_latency_ms"])}
    return trace


def builtin_routing() -> dict:
    """The built-in stack's routing decision, for the UI to show rather
    than describe: which tier each band starts at, and the calibrated
    quality that put it there."""
    return {
        "models": DEFAULT_TIER_MODELS,
        "map": tier_map_for(0, None),
        "bands": list(DIFFICULTIES),
        "quality": {t: BUILTIN_CALIBRATION[t]["quality"] for t in TIER_ORDER if t in BUILTIN_CALIBRATION},
        "cost": {t: BUILTIN_CALIBRATION[t]["cost"] for t in TIER_ORDER if t in BUILTIN_CALIBRATION},
        "threshold": QUALITY_THRESHOLD,
    }


# --- after an answer ---------------------------------------------------------

def baseline_cost(result: dict, tier_models: dict | None) -> float:
    """What the answer would have cost from the strongest tier available:
    for BYOM, the strongest model *this user* configured -- not our own
    built-in frontier, which they may not even use.

    In a conversation, that tier would have served every turn, so its
    history would be in its prompt cache: the history is priced at the
    cached rate. That keeps the "saved" figure conservative."""
    history = ((result.get("trace") or {}).get("start") or {}).get("history_by_tier") or {}
    history_tokens = next(iter(history.values()), {}).get("tokens", 0) if history else 0
    if tier_models:
        strongest = next((t for t in reversed(TIER_ORDER) if t in tier_models), result["final_tier"])
        return estimate_cost_for_model(tier_models[strongest], result["tokens_in"], result["tokens_out"],
                                       cached_tokens=history_tokens)
    answer = estimate_frontier_cost(result["tokens_in"], result["tokens_out"], result["difficulty"])
    rates = prompt_cache.prices(DEFAULT_TIER_MODELS[TIER_ORDER[-1]]) if history_tokens else None
    return answer + (history_tokens * rates[1] if rates else 0.0)


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
