"""Week 3: the actual cascade -- classify, generate, verify, escalate.

Also degrades gracefully when a tier is temporarily unavailable (rate
limited / daily quota exhausted) instead of hard-failing the whole
request -- discovered the hard way when Groq's daily token cap got hit
mid-demo and a raw provider exception was dumped straight into the chat.
"""
import time

import litellm

from app.classifier import classify_initial_tier
from app.model_config import router
from app.verifier import verify_response

TIER_ORDER = ["cheap", "mid", "frontier"]


class AllTiersUnavailable(Exception):
    """Raised only if every tier in the cascade sequence is unreachable."""


def run_cascade(messages: list[dict]) -> dict:
    query = messages[-1]["content"]
    initial_tier, difficulty = classify_initial_tier(query)

    tier_sequence = TIER_ORDER[TIER_ORDER.index(initial_tier):]

    total_cost = 0.0
    total_latency_ms = 0.0
    escalated = False
    escalation_reasons = []
    final_text = None
    final_tier = None
    final_usage = None

    for tier in tier_sequence:
        start = time.perf_counter()
        try:
            resp = router.completion(model=tier, messages=messages)
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

        try:
            verify_result = verify_response(query, text, tier)
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
