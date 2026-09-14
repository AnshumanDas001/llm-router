"""Run the full eval set through the live
cascade router and compute % traffic per tier, cost vs. the
frontier-only baseline, and quality retained vs. that baseline.
"""
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.db import get_conn, init_db

ROUTER_URL = "http://localhost:8000/v1/chat/completions"
QUERIES_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_queries.json"



def main():
    queries = json.loads(QUERIES_PATH.read_text())
    init_db()

    with get_conn() as conn:
        # cascade_log is what the README charts are generated from. A fresh run
        # replaces it, so snapshot the previous run first -- the before/after
        # comparison is usually the whole point of re-running.
        stamp = time.strftime("%Y%m%d_%H%M%S")
        n_prev = conn.execute("SELECT COUNT(*) FROM cascade_log").fetchone()[0]
        if n_prev:
            conn.execute(f"CREATE TABLE cascade_log_{stamp} AS SELECT * FROM cascade_log")
            print(f"Snapshotted previous {n_prev} rows to cascade_log_{stamp} "
                  f"(compare with: scripts/compare_strategies.py --table cascade_log_{stamp})")
        conn.execute("DELETE FROM cascade_log")
        conn.commit()

    print(f"Running {len(queries)} queries through the live cascade router...\n")
    for i, q in enumerate(queries, 1):
        try:
            resp = requests.post(
                ROUTER_URL,
                json={"messages": [{"role": "user", "content": q["query"]}]},
                timeout=120,
            )
            resp.raise_for_status()
            meta = resp.json()["_router"]
            flag = " <ESCALATED>" if meta["escalated"] else ""
            print(f"[{i}/{len(queries)}] {q['difficulty']:6} -> {meta['final_tier']:8} "
                  f"cost=${meta['cost']:.5f}{flag}  -  {q['query'][:50]}")
        except Exception as exc:
            print(f"[{i}/{len(queries)}] FAILED: {exc}  -  {q['query'][:50]}")
        time.sleep(1.5)

    with get_conn() as conn:
        rows = conn.execute("SELECT id FROM cascade_log").fetchall()

    if not rows:
        print("\nNo successful requests.")
        sys.exit(1)

    print(f"\n{len(rows)} queries logged to cascade_log.")
    print("For the cost comparison against mid-only and frontier-only over the same")
    print("tokens, run:  ./venv/bin/python scripts/compare_strategies.py\n")


if __name__ == "__main__":
    main()
