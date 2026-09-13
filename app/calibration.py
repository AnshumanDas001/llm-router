"""BYOM calibration: run the auto-scoreable eval subset against a user's
own model, using their key transiently (never stored), and return only the
derived quality/cost/latency numbers.

Capped at a modest query count by default -- calibration runs on the
user's own key/budget, and the lesson from building the eval set applies
here too: a smaller, well-targeted set beats a large one if it means
actually running it. 25 diverse auto-scoreable queries is enough to catch
a badly-misconfigured tier (e.g. "mid" actually weaker than "cheap")
without being expensive for the user to run.
"""
import json
import time
from pathlib import Path

import litellm

from app.scoring import score_one

EVAL_QUERIES_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_queries.json"
DEFAULT_MAX_QUERIES = 25


DIFFICULTIES = ("easy", "medium", "hard")


def _load_automatic_queries(max_queries: int) -> list[dict]:
    """Stratified by difficulty, then interleaved.

    Stratified, because a single blended average can't answer the question
    routing actually needs -- "does this model hold up on *hard* queries?"
    -- since a model that aces easy/medium and collapses on hard still
    posts a respectable overall number. The eval set measured exactly that on our
    own cheap tier (0.916 easy / 0.936 medium / 0.718 hard).

    Interleaved, because running the difficulties in blocks put `hard`
    last, so a provider rate limit partway through wiped out the whole hard
    bucket -- the one difficulty the routing decision most depends on --
    while easy and medium looked perfect. Round-robin means a truncated run
    degrades every difficulty evenly instead of silently losing one.
    """
    queries = json.loads(EVAL_QUERIES_PATH.read_text())
    automatic = [q for q in queries if q["eval_method"] != "llm_judge"]

    per_difficulty = max(1, max_queries // len(DIFFICULTIES))
    by_difficulty = []
    for difficulty in DIFFICULTIES:
        pool = [q for q in automatic if q["difficulty"] == difficulty]
        step = max(1, len(pool) // per_difficulty)
        by_difficulty.append(pool[::step][:per_difficulty])

    interleaved = []
    for i in range(per_difficulty):
        for pool in by_difficulty:
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
                     max_queries: int = DEFAULT_MAX_QUERIES) -> dict:
    """Returns {avg_quality, avg_cost, avg_latency_ms, n_queries, n_errors,
    n_rate_limited, quality_by_difficulty}. Never persists model_name or
    api_key itself -- caller decides what (if anything) to store."""
    queries = _load_automatic_queries(max_queries)

    scores, costs, latencies = [], [], []
    scores_by_difficulty = {d: [] for d in DIFFICULTIES}
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
                    model=model_name, api_key=api_key,
                    messages=[{"role": "user", "content": q["query"]}],
                )
                latency_ms = (time.perf_counter() - start) * 1000
                text = resp.choices[0].message.content
                cost = resp._hidden_params.get("response_cost", 0.0) or 0.0

                score = score_one(q, text)
                scores.append(score)
                scores_by_difficulty[q["difficulty"]].append(score)
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
            d: (sum(s) / len(s) if s else None) for d, s in scores_by_difficulty.items()
        },
    }
