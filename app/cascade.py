"""Week 3: the actual cascade -- classify, generate, verify, escalate.

Also degrades gracefully when a tier is temporarily unavailable (rate
limited / daily quota exhausted) instead of hard-failing the whole
request -- discovered the hard way when Groq's daily token cap got hit
mid-demo and a raw provider exception was dumped straight into the chat.

Supports BYOM (bring your own model): pass tier_models (e.g.
{"cheap": "groq/llama-3.1-8b-instant", "frontier": "gpt-4o"}) and
tier_api_keys (the caller's own third-party keys, used transiently for
this call only -- never stored) to route across a user's own models
instead of our built-in three. A user may configure only some tiers;
the cascade only ever steps through the ones actually provided, and
verification strategy is decided by *position* in that sequence (see
app/verifier.py), not by the literal tier name -- so this works the same
way regardless of which tiers a user did or didn't configure.
"""
import time

import litellm

from app.classifier import classify_initial_tier
from app.model_config import TIER_MODEL_LIST, router
from app.verifier import verify_response, verify_with_judge

TIER_ORDER = ["cheap", "mid", "frontier"]
DEFAULT_TIER_MODELS = {t["model_name"]: t["litellm_params"]["model"] for t in TIER_MODEL_LIST}


class AllTiersUnavailable(Exception):
    """Raised only if every tier in the cascade sequence is unreachable."""


def _complete(tier: str, messages: list[dict], tier_models: dict | None,
               tier_api_keys: dict | None):
    if tier_models is None:
        return router.completion(model=tier, messages=messages)
    return litellm.completion(
        model=tier_models[tier], messages=messages,
        api_key=(tier_api_keys or {}).get(tier),
    )


def run_cascade(messages: list[dict], tier_models: dict | None = None,
                 tier_api_keys: dict | None = None) -> dict:
    query = messages[-1]["content"]
    initial_tier, difficulty = classify_initial_tier(query)

    available_tiers = TIER_ORDER if tier_models is None else [
        t for t in TIER_ORDER if t in tier_models
    ]
    if not available_tiers:
        raise AllTiersUnavailable("no tiers configured")
    if initial_tier not in available_tiers:
        # e.g. classifier guessed "mid" but the user only configured cheap/frontier
        initial_tier = available_tiers[0]
    tier_sequence = available_tiers[available_tiers.index(initial_tier):]

    # Only the single cheapest *configured* tier gets a real, paid LLM-judge
    # check (by the next tier up in the full configured list) -- not
    # whichever tier this particular query's sequence happens to start at.
    # A "hard" query skips straight to e.g. mid, but mid still only gets the
    # free structural check: Week 2 showed mid's failure rate is low and its
    # answers are long, making a real judge call there expensive for little
    # safety benefit (it was 82.5% of total cascade cost before this fix).
    judge_model_for = {}
    cheapest_tier = available_tiers[0]
    if len(available_tiers) > 1 and cheapest_tier in tier_sequence:
        judge_tier = available_tiers[1]
        judge_model_for[cheapest_tier] = (
            tier_models[judge_tier] if tier_models is not None else DEFAULT_TIER_MODELS[judge_tier],
            (tier_api_keys or {}).get(judge_tier),
            judge_tier,
        )

    total_cost = 0.0
    total_latency_ms = 0.0
    escalated = False
    escalation_reasons = []
    final_text = None
    final_tier = None
    final_usage = None

    for i, tier in enumerate(tier_sequence):
        start = time.perf_counter()
        try:
            resp = _complete(tier, messages, tier_models, tier_api_keys)
        except litellm.RateLimitError:
            # This tier is temporarily down (daily/per-minute quota). Skip
            # straight to the next tier rather than failing the whole
            # request -- unless there is no next tier, in which case fall
            # back to whatever we already have (or fail if nothing yet).
            escalated = True
            escalation_reasons.append(f"{tier} unavailable (rate limited), skipped")
            continue
        latency_ms = (time.perf_counter() - start) * 1000

        cost = resp._hidden_params.get("response_cost", 0.0) or 0.0
        text = resp.choices[0].message.content

        total_cost += cost
        total_latency_ms += latency_ms
        final_text, final_tier, final_usage = text, tier, resp.usage  # best effort so far

        is_last_tier = i == len(tier_sequence) - 1
        if is_last_tier:
            break  # nothing left to escalate to; trust it unconditionally

        try:
            if tier in judge_model_for:
                judge_model, judge_api_key, judge_tier_label = judge_model_for[tier]
                verify_result = verify_with_judge(query, text, judge_model, judge_api_key, judge_tier_label)
            else:
                verify_result = verify_response(query, text, tier, tier_models, tier_api_keys)
        except litellm.RateLimitError:
            # Fail open: the verifier being unavailable isn't evidence the
            # answer is wrong. Trusting it beats blocking the whole response
            # on an unrelated tier's exhausted quota.
            verify_result = {"passed": True, "cost": 0.0, "latency_ms": 0.0,
                              "reason": "verifier unavailable (rate limited), trusting answer as-is"}
            escalated = True
            escalation_reasons.append(f"{tier}: {verify_result['reason']}")

        total_cost += verify_result["cost"]
        total_latency_ms += verify_result["latency_ms"]

        if verify_result["passed"]:
            break

        escalated = True
        escalation_reasons.append(f"{tier} failed verification: {verify_result['reason']}")

    if final_text is None:
        raise AllTiersUnavailable(
            f"every tier in {tier_sequence} was rate limited or failed; try again shortly"
        )

    return {
        "text": final_text,
        "difficulty": difficulty,
        "initial_tier": initial_tier,
        "final_tier": final_tier,
        "escalated": escalated,
        "escalation_reasons": escalation_reasons,
        "total_cost": total_cost,
        "total_latency_ms": total_latency_ms,
        "tokens_in": final_usage.prompt_tokens if final_usage else None,
        "tokens_out": final_usage.completion_tokens if final_usage else None,
    }
