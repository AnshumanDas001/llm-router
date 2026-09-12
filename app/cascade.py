"""Week 3: the actual cascade -- classify, generate, verify, escalate."""
import time

from app.classifier import classify_initial_tier
from app.model_config import router
from app.verifier import verify_response

TIER_ORDER = ["cheap", "mid", "frontier"]


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
        resp = router.completion(model=tier, messages=messages)
        latency_ms = (time.perf_counter() - start) * 1000

        cost = resp._hidden_params.get("response_cost", 0.0) or 0.0
        text = resp.choices[0].message.content

        total_cost += cost
        total_latency_ms += latency_ms

        verify_result = verify_response(query, text, tier)
        total_cost += verify_result["cost"]
        total_latency_ms += verify_result["latency_ms"]

        if verify_result["passed"]:
            final_text, final_tier, final_usage = text, tier, resp.usage
            break

        escalated = True
        escalation_reasons.append(f"{tier} failed verification: {verify_result['reason']}")
        final_text, final_tier, final_usage = text, tier, resp.usage  # fallback if loop ends early

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
