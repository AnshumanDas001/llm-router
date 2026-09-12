"""BYOM calibration: run the auto-scoreable eval subset against a user's
own model, using their key transiently (never stored), and return only the
derived quality/cost/latency numbers.

Capped at a modest query count by default -- calibration runs on the
user's own key/budget, and the roadmap's own lesson from Week 2 applies
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


def _load_automatic_queries(max_queries: int) -> list[dict]:
    queries = json.loads(EVAL_QUERIES_PATH.read_text())
    automatic = [q for q in queries if q["eval_method"] != "llm_judge"]
    # Evenly spread across the list rather than just taking the first N, so
    # the subset still spans easy/medium/hard rather than clustering on
    # whichever difficulty happens to sort first.
    step = max(1, len(automatic) // max_queries)
    return automatic[::step][:max_queries]


def calibrate_model(model_name: str, api_key: str | None,
                     max_queries: int = DEFAULT_MAX_QUERIES) -> dict:
    """Returns {avg_quality, avg_cost, avg_latency_ms, n_queries, n_errors}.
    Never persists model_name or api_key itself -- caller decides what (if
    anything) to store."""
    queries = _load_automatic_queries(max_queries)

    scores, costs, latencies = [], [], []
    n_errors = 0

    for q in queries:
        try:
            start = time.perf_counter()
            resp = litellm.completion(
                model=model_name, api_key=api_key,
                messages=[{"role": "user", "content": q["query"]}],
            )
            latency_ms = (time.perf_counter() - start) * 1000
            text = resp.choices[0].message.content
            cost = resp._hidden_params.get("response_cost", 0.0) or 0.0

            scores.append(score_one(q, text))
            costs.append(cost)
            latencies.append(latency_ms)
        except Exception:
            n_errors += 1
            continue

    n = len(scores)
    return {
        "avg_quality": sum(scores) / n if n else 0.0,
        "avg_cost": sum(costs) / n if n else 0.0,
        "avg_latency_ms": sum(latencies) / n if n else 0.0,
        "n_queries": n,
        "n_errors": n_errors,
    }
