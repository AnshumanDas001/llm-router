"""Grade what the cascade actually shipped.

compare_strategies.py says what a run cost; this says whether the answers
were right. Only the eval queries with an automatic check (exact match,
numeric, schema) are graded -- no judge in the loop, so the number is
ground truth -- and the result is split by where the answer came from, so
a verifier that lets wrong cheap answers through shows up as a low
"answered at cheap" accuracy.

    ./venv/bin/python scripts/score_cascade_log.py [--table cascade_log_<stamp>]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.db import get_conn
from app.scoring import score_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="cascade_log")
    args = ap.parse_args()

    queries = {q["query"]: q for q in json.loads(
        (Path(__file__).resolve().parent.parent / "data" / "eval_queries.json").read_text())}
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT query, difficulty, final_tier, escalated, response_text FROM {args.table}"
        ).fetchall()

    graded = defaultdict(lambda: [0, 0])   # key -> [correct, total]
    skipped = 0
    for query, difficulty, final_tier, escalated, text in rows:
        q = queries.get(query)
        if q is None or q["eval_method"] == "llm_judge" or text is None:
            skipped += 1
            continue
        ok = score_one(q, text) >= 1.0
        for key in ("all", f"difficulty={q['difficulty']}", f"answered at {final_tier}",
                    "escalated" if escalated else "not escalated"):
            graded[key][0] += ok
            graded[key][1] += 1

    print(f"{len(rows)} rows in {args.table}; {len(rows) - skipped} auto-gradable, {skipped} skipped "
          f"(judge-only queries or no stored answer)\n")
    for key in sorted(graded, key=lambda k: (not k.startswith("all"), k)):
        c, n = graded[key]
        print(f"  {key:22} {c:3}/{n:<3} = {c / n:.1%}")


if __name__ == "__main__":
    main()
