"""Run every labeled eval query through all three tiers and store the
full response text (needed for scoring, unlike the baseline log which
only kept metadata). Resumable: skips (query_id, tier) pairs already logged.
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.db import get_conn, init_db, log_eval_response
from app.model_config import router

QUERIES_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_queries.json"

TIERS = ["cheap", "mid", "frontier"]
# Free-tier rate limits differ a lot by provider; throttle accordingly.
THROTTLE_SECONDS = {"cheap": 0.0, "mid": 1.5, "frontier": 4.5}
MAX_RETRIES_ON_RATE_LIMIT = 3


def already_done() -> set[tuple[str, str]]:
    init_db()
    with get_conn() as conn:
        rows = conn.execute("SELECT query_id, tier FROM eval_responses").fetchall()
    return {(r[0], r[1]) for r in rows}


def main():
    queries = json.loads(QUERIES_PATH.read_text())
    done = already_done()

    jobs = [(q, tier) for q in queries for tier in TIERS if (q["id"], tier) not in done]
    print(f"{len(done)} (query, tier) pairs already logged, {len(jobs)} remaining "
          f"out of {len(queries) * len(TIERS)} total.\n")

    for i, (q, tier) in enumerate(jobs, 1):
        for attempt in range(MAX_RETRIES_ON_RATE_LIMIT + 1):
            try:
                start = time.perf_counter()
                resp = router.completion(
                    model=tier,
                    messages=[{"role": "user", "content": q["query"]}],
                )
                latency_ms = (time.perf_counter() - start) * 1000
                usage = resp.usage
                cost = resp._hidden_params.get("response_cost", 0.0) or 0.0
                text = resp.choices[0].message.content

                log_eval_response(
                    query_id=q["id"], tier=tier, response_text=text,
                    tokens_in=usage.prompt_tokens, tokens_out=usage.completion_tokens,
                    cost=cost, latency_ms=latency_ms,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                print(f"[{i}/{len(jobs)}] {q['id']} / {tier}  ({latency_ms:.0f}ms)  -  {q['query'][:50]}")
                break
            except Exception as exc:
                is_rate_limit = "429" in str(exc) or "RateLimitError" in str(exc)
                if is_rate_limit and attempt < MAX_RETRIES_ON_RATE_LIMIT:
                    print(f"[{i}/{len(jobs)}] {q['id']} / {tier}  rate limited, backing off 30s")
                    time.sleep(30)
                    continue
                print(f"[{i}/{len(jobs)}] {q['id']} / {tier}  FAILED: {exc}")
                break
        time.sleep(THROTTLE_SECONDS[tier])

    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM eval_responses").fetchone()[0]
    print(f"\nDone. {total}/{len(queries) * len(TIERS)} (query, tier) pairs logged.")


if __name__ == "__main__":
    main()
