"""Calibration: run a stratified sample of automatically-gradable questions
against one model and return only derived numbers -- quality, cost and
latency per difficulty band. Used for BYOM models (with the user's key,
used transiently and never stored) and by scripts/calibration/ for the
built-in tiers.

Capped at a modest query count by default -- calibration runs on the
user's own key/budget, and the lesson from building the eval set applies
here too: a smaller, well-targeted set beats a large one if it means
actually running it. Six questions per band is enough to catch a
badly-misconfigured tier (e.g. "mid" actually weaker than "cheap")
without being expensive to run. The expert band is the exception to
"cheap": it is competition maths, and a thinking model can spend several
cents on each of its questions.
"""
import time

import litellm

from app.evaluation.datasets import DIFFICULTIES, load_reference_queries, prompt_for
from app.evaluation.graders import is_automatic, score_one

DEFAULT_MAX_QUERIES = 24


def _load_automatic_queries(max_queries: int | None,
                            bands: tuple[str, ...] = DIFFICULTIES) -> list[dict]:
    """Stratified by difficulty, then interleaved.

    Stratified, because a single blended average can't answer the question
    routing actually needs -- "does this model hold up on *hard* queries?"
    -- since a model that aces easy/medium and collapses on hard still
    posts a respectable overall number. The eval set measured exactly that on our
    own cheap tier (0.916 easy / 0.936 medium / 0.718 hard).

    Interleaved, because running the difficulties in blocks put the hardest
    band last, so a provider rate limit partway through wiped out the whole
    bucket the routing decision most depends on -- while easy and medium
    looked perfect. Round-robin means a truncated run degrades every
    difficulty evenly instead of silently losing one.
    """
    automatic = [q for q in load_reference_queries() if is_automatic(q)]

    # None means every automatic query -- for a tier that costs nothing to
    # measure, the whole set beats a sample.
    per_band = len(automatic) if max_queries is None else max(1, max_queries // len(bands))
    by_band = []
    for band in bands:
        pool = [q for q in automatic if q["difficulty"] == band]
        step = max(1, len(pool) // per_band)
        by_band.append(pool[::step][:per_band])

    interleaved = []
    for i in range(per_band):
        for pool in by_band:
            if i < len(pool):
                interleaved.append(pool[i])
    return interleaved


RATE_LIMIT_RETRY_DELAY_S = 20
MAX_ERROR_CHARS = 300


def _clean_error(exc: Exception) -> str:
    """litellm wraps provider errors in a lot of boilerplate (stack hints,
    'Give Feedback' links). Keep the first line, which is where the provider's
    own message lives, and cap the length for display."""
    message = str(exc).split("\n")[0].strip()
    prefix = "litellm."
    if message.startswith(prefix):
        message = message[len(prefix):]
    return message[:MAX_ERROR_CHARS]


def calibrate_model(model_name: str, api_key: str | None,
                    max_queries: int | None = DEFAULT_MAX_QUERIES,
                    call_params: dict | None = None,
                    bands: tuple[str, ...] = DIFFICULTIES) -> dict:
    """Returns {avg_quality, avg_cost, avg_latency_ms, n_queries, n_errors,
    n_rate_limited, first_error, quality_by_difficulty, cost_by_difficulty,
    n_by_difficulty}. Never persists model_name or api_key itself -- the
    caller decides what (if anything) to store."""
    queries = _load_automatic_queries(max_queries, bands)

    scores, costs, latencies = [], [], []
    scores_by_band = {d: [] for d in bands}
    costs_by_band = {d: [] for d in bands}
    n_errors = 0
    n_rate_limited = 0
    # The provider's own message ("model X does not exist or you do not have
    # access to it") is far more actionable than anything we can infer, so
    # keep the first one instead of swallowing it and guessing at the cause.
    first_error = None

    for q in queries:
        # A rate limit is a transient "try again", not a verdict on the
        # model -- lumping it in with real failures made a throttled run
        # look like a broken model string. Retry once, then give up on this
        # query and record *why* it was dropped.
        for attempt in (1, 2):
            try:
                start = time.perf_counter()
                resp = litellm.completion(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt_for(q)}],
                    # Measure a built-in tier with the parameters it runs
                    # with (reasoning effort, max_tokens, a local api_base);
                    # BYOM gets provider defaults and the caller's key.
                    **{"api_key": api_key, **(call_params or {})},
                )
                latency_ms = (time.perf_counter() - start) * 1000
                # No content is an answer that didn't arrive (a reasoning
                # model that spent its whole budget thinking): score it as
                # wrong, and charge it, rather than crashing on None.
                text = resp.choices[0].message.content or ""
                cost = resp._hidden_params.get("response_cost", 0.0) or 0.0

                score = score_one(q, text)
                scores.append(score)
                scores_by_band[q["difficulty"]].append(score)
                costs_by_band[q["difficulty"]].append(cost)
                costs.append(cost)
                latencies.append(latency_ms)
                break
            except litellm.RateLimitError as e:
                if attempt == 1:
                    time.sleep(RATE_LIMIT_RETRY_DELAY_S)
                    continue
                n_rate_limited += 1
                n_errors += 1
                first_error = first_error or _clean_error(e)
            except Exception as e:
                n_errors += 1
                first_error = first_error or _clean_error(e)
            break

    n = len(scores)
    return {
        "avg_quality": sum(scores) / n if n else 0.0,
        "avg_cost": sum(costs) / n if n else 0.0,
        "avg_latency_ms": sum(latencies) / n if n else 0.0,
        "n_queries": n,
        "n_errors": n_errors,
        "n_rate_limited": n_rate_limited,
        "first_error": first_error,
        # None (not 0.0) where a difficulty got no successful scores, so the
        # routing policy can tell "measured as bad" from "never measured".
        "quality_by_difficulty": {
            d: (sum(s) / len(s) if s else None) for d, s in scores_by_band.items()
        },
        # Per band, not one average: a mid model's easy answers can be 7x
        # cheaper than its overall mean, and routing by the mean makes a
        # cheap tier look like a win on easy questions when it isn't.
        "cost_by_difficulty": {
            d: (sum(c) / len(c) if c else None) for d, c in costs_by_band.items()
        },
        "n_by_difficulty": {d: len(s) for d, s in scores_by_band.items()},
    }
