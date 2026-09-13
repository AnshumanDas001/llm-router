"""Estimates what a response would have cost if a stronger tier had
generated it, for the "cost saved" counters in both the built-in demo and
BYOM usage tracking.

This is an estimate, not a re-run: it prices the *actual* tokens_in/out
the cascade produced at the reference model's per-token rate, rather than
calling that model a second time just to measure a baseline. A real
response from that model to the same query would likely use a different
token count, so treat this as directionally correct, not exact -- the
same limitation as the baseline-vs-cascade comparisons throughout
this project.
"""
import litellm

from app.model_config import TIER_MODEL_LIST

FRONTIER_MODEL = next(
    t["litellm_params"]["model"] for t in TIER_MODEL_LIST if t["model_name"] == "frontier"
)


def estimate_cost_for_model(model_name: str, tokens_in: int, tokens_out: int) -> float:
    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model_name, prompt_tokens=tokens_in or 0, completion_tokens=tokens_out or 0,
        )
        return prompt_cost + completion_cost
    except Exception:
        return 0.0


def estimate_frontier_cost(tokens_in: int, tokens_out: int) -> float:
    """Baseline against our own built-in frontier tier (used by the
    default demo/chat, not BYOM)."""
    return estimate_cost_for_model(FRONTIER_MODEL, tokens_in, tokens_out)
