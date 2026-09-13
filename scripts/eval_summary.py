"""Combined quality/cost/latency matrix across all three
tiers, for the full 61-query eval set."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.db import get_conn, init_db

TIERS = ["cheap", "mid", "frontier"]


def main():
    init_db()
    with get_conn() as conn:
        print(f"{'tier':<10}{'avg_quality':>12}{'n_queries':>11}{'avg_cost':>12}{'avg_latency_ms':>16}")
        for tier in TIERS:
            scores = conn.execute(
                "SELECT score FROM eval_scores WHERE tier=? AND source IN ('automatic','judge_claude')",
                (tier,),
            ).fetchall()
            avg_quality = sum(s[0] for s in scores) / len(scores)

            perf = conn.execute(
                "SELECT AVG(cost), AVG(latency_ms) FROM eval_responses WHERE tier=?",
                (tier,),
            ).fetchone()

            print(f"{tier:<10}{avg_quality:>12.3f}{len(scores):>11}{perf[0]:>12.5f}{perf[1]:>16.0f}")

        print("\nBy difficulty (frontier-only baseline comparison omitted; this is quality only):")
        print(f"{'tier':<10}{'easy':>8}{'medium':>8}{'hard':>8}")
        import json
        queries = {q["id"]: q["difficulty"] for q in json.loads(
            (Path(__file__).resolve().parent.parent / "data" / "eval_queries.json").read_text()
        )}
        for tier in TIERS:
            rows = conn.execute(
                "SELECT query_id, score FROM eval_scores WHERE tier=? AND source IN ('automatic','judge_claude')",
                (tier,),
            ).fetchall()
            buckets = {"easy": [], "medium": [], "hard": []}
            for qid, score in rows:
                buckets[queries[qid]].append(score)
            print(f"{tier:<10}" + "".join(
                f"{sum(v)/len(v):>8.3f}" for v in buckets.values()
            ))


if __name__ == "__main__":
    main()
