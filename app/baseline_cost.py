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

from app.model_config import BUILTIN_CALIBRATION, TIER_MODEL_LIST

FRONTIER_MODEL = next(
    t["litellm_params"]["model"] for t in TIER_MODEL_LIST if t["model_name"] == "frontier"
)


def estimate_cost_for_model(model_name: str, tokens_in: int, tokens_out: int) -> float:
    # litellm rejects non-integer token counts, and the except below turns
    # that into a silent $0 -- which is how an average like 14.3 tokens came
    # back as "free". Round so averages price correctly.
    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model_name,
            prompt_tokens=int(round(tokens_in or 0)),
            completion_tokens=int(round(tokens_out or 0)),
        )
        return prompt_cost + completion_cost
    except Exception:
        return 0.0


def estimate_frontier_cost(tokens_in: int, tokens_out: int, difficulty: str | None = None) -> float:
    """Baseline against our own built-in frontier tier (used by the
    default demo/chat, not BYOM).

    Prefers the frontier tier's *calibrated* cost for this difficulty band,
    because pricing tokens at list rate misses hidden reasoning tokens --
    and on the current stack those are the entire difference between mid and
    frontier, which run the same model with thinking off and on. Priced by
    tokens alone the two look identical and the saving reads as zero.

    Falls back to the token estimate when the band is unknown or the tier
    has never been calibrated.
    """
    if difficulty:
        band = ((BUILTIN_CALIBRATION.get("frontier") or {}).get("cost") or {}).get(difficulty)
        if band:
            return band
    return estimate_cost_for_model(FRONTIER_MODEL, tokens_in, tokens_out)
