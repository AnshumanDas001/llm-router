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

# The frontier-only baseline (61 queries, gemini-3.5-flash-lite).
BASELINE_TOTAL_COST = 0.05695
BASELINE_AVG_QUALITY = 1.0  # frontier scored 1.000 across the board


def main():
    queries = json.loads(QUERIES_PATH.read_text())
    init_db()

    with get_conn() as conn:
        conn.execute("DELETE FROM cascade_log")  # fresh run for this deliverable
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

    query_to_id = {q["query"]: q["id"] for q in queries}

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT query, difficulty, initial_tier, final_tier, escalated, total_cost FROM cascade_log"
        ).fetchall()

        quality_scores = []
        for query_text, _, _, final_tier, _, _ in rows:
            qid = query_to_id.get(query_text)
            score_row = conn.execute(
                "SELECT score FROM eval_scores WHERE query_id=? AND tier=? "
                "AND source IN ('automatic','judge_claude')",
                (qid, final_tier),
            ).fetchone()
            if score_row:
                quality_scores.append(score_row[0])

    if not rows:
        print("\nNo successful requests.")
        sys.exit(1)

    n = len(rows)
    total_cost = sum(r[5] for r in rows)
    escalated_count = sum(r[4] for r in rows)
    tier_counts = {"cheap": 0, "mid": 0, "frontier": 0}
    for r in rows:
        tier_counts[r[3]] += 1

    print("\n--- cascade summary ---")
    print(f"Total requests: {n}")
    print(f"Escalated: {escalated_count} ({100 * escalated_count / n:.1f}%)")
    print("\n% of traffic per final tier:")
    for tier, count in tier_counts.items():
        print(f"  {tier:10} {count:3} ({100 * count / n:.1f}%)")

    print(f"\nTotal cost (cascade):     ${total_cost:.5f}")
    print(f"Total cost (baseline, frontier-only): ${BASELINE_TOTAL_COST:.5f}")
    savings_pct = 100 * (1 - total_cost / BASELINE_TOTAL_COST)
    print(f"Cost saved vs. baseline:  {savings_pct:.1f}%")

    if quality_scores:
        avg_quality = sum(quality_scores) / len(quality_scores)
        retained_pct = 100 * avg_quality / BASELINE_AVG_QUALITY
        print(f"\nAvg quality (cascade, using each query's eval score for its "
              f"final tier): {avg_quality:.3f} ({len(quality_scores)}/{n} queries matched)")
        print(f"Avg quality (baseline, frontier-only): {BASELINE_AVG_QUALITY:.3f}")
        print(f"Quality retained vs. baseline: {retained_pct:.1f}%")
    print("\nNote: quality here reuses each query's per-tier eval score rather than "
          "re-scoring fresh cascade output -- a reasonable proxy given the same models, "
          "but real given the run-to-run non-determinism seen across eval runs.")


if __name__ == "__main__":
    main()
