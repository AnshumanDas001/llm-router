"""Estimates what a response would have cost if frontier had generated it,
for the chat UI's running "cost saved" counter.

This is an estimate, not a re-run: it prices the *actual* tokens_in/out the
cascade produced at frontier's per-token rate, rather than calling frontier
a second time just to measure a baseline. A real frontier-only response to
the same query would likely use a different token count, so treat this as
directionally correct, not exact -- it's the same limitation as Week 1's
baseline-vs-cascade comparisons throughout this project.
"""
import litellm

from app.model_config import TIER_MODEL_LIST

FRONTIER_MODEL = next(
    t["litellm_params"]["model"] for t in TIER_MODEL_LIST if t["model_name"] == "frontier"
)


def estimate_frontier_cost(tokens_in: int, tokens_out: int) -> float:
    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=FRONTIER_MODEL, prompt_tokens=tokens_in or 0, completion_tokens=tokens_out or 0,
        )
        return prompt_cost + completion_cost
    except Exception:
        return 0.0
