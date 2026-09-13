"""Baseline run: sends every sample query through the frontier-only
router endpoint and prints summary cost/latency stats. Requires the FastAPI
server to be running (uvicorn app.main:app) and GEMINI_API_KEY set in .env.
"""
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.db import get_conn, init_db

ROUTER_URL = "http://localhost:8000/v1/chat/completions"
QUERIES_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_queries.json"

# Gemini free tier caps gemini-3.5-flash-lite at 15 requests/minute. Stay
# comfortably under that rather than tuning it to the wire.
SECONDS_BETWEEN_REQUESTS = 4.5
MAX_RETRIES_ON_RATE_LIMIT = 3


def already_logged_queries() -> set[str]:
    init_db()
    with get_conn() as conn:
        rows = conn.execute("SELECT DISTINCT query FROM requests").fetchall()
    return {r[0] for r in rows}


def main():
    all_queries = json.loads(QUERIES_PATH.read_text())
    done = already_logged_queries()
    queries = [q for q in all_queries if q not in done]

    print(f"{len(done)} queries already logged from a prior run, "
          f"{len(queries)} remaining out of {len(all_queries)} total.\n")

    results = []
    for i, q in enumerate(queries, 1):
        for attempt in range(MAX_RETRIES_ON_RATE_LIMIT + 1):
            try:
                resp = requests.post(
                    ROUTER_URL,
                    json={"messages": [{"role": "user", "content": q}]},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                meta = data["_router"]
                results.append(meta)
                print(f"[{i}/{len(queries)}] cost=${meta['cost']:.5f} "
                      f"latency={meta['latency_ms']:.0f}ms  -  {q[:60]}")
                break
            except Exception as exc:
                is_rate_limit = "429" in str(exc) or "RateLimitError" in str(exc)
                if is_rate_limit and attempt < MAX_RETRIES_ON_RATE_LIMIT:
                    print(f"[{i}/{len(queries)}] rate limited, backing off 30s "
                          f"(attempt {attempt + 1})  -  {q[:60]}")
                    time.sleep(30)
                    continue
                print(f"[{i}/{len(queries)}] FAILED: {exc}  -  {q[:60]}")
                break
        time.sleep(SECONDS_BETWEEN_REQUESTS)

    with get_conn() as conn:
        rows = conn.execute("SELECT cost, latency_ms FROM requests").fetchall()

    if not rows:
        print("\nNo successful requests — check the server is running and keys are set.")
        sys.exit(1)

    total_cost = sum(r[0] for r in rows)
    avg_latency = sum(r[1] for r in rows) / len(rows)

    print("\n--- Baseline summary (frontier-only, all logged requests) ---")
    print(f"Logged requests:     {len(rows)}/{len(all_queries)}")
    print(f"Total cost:          ${total_cost:.5f}")
    print(f"Avg cost/request:    ${total_cost / len(rows):.5f}")
    print(f"Avg latency:         {avg_latency:.0f}ms")


if __name__ == "__main__":
    main()
