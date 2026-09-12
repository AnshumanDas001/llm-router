"""Active learning: turn real cascade escalations into candidate new
reference examples for the classifier, instead of paying for more
hand-crafted eval queries.

Logic: every escalation in cascade_log is a signal that the classifier's
upfront guess was wrong for that specific query. This script finds queries
that (a) actually escalated and (b) aren't already in our labeled reference
set (data/eval_queries.json) -- re-confirming a known query tells us
nothing new -- and proposes a corrected difficulty label for human review.

This does NOT auto-merge into the reference set. Some escalations happen
because a task is genuinely ambiguous (e.g. a 3-way sentiment call with no
clean answer), not because the classifier misjudged difficulty -- merging
those in uncritically would teach the classifier the wrong lesson. A human
reviews data/active_learning_candidates.json and runs merge_candidates.py
on the ones that are real difficulty corrections.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.db import get_conn, init_db

EVAL_QUERIES_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_queries.json"
CANDIDATES_PATH = Path(__file__).resolve().parent.parent / "data" / "active_learning_candidates.json"

# guessed difficulty -> corrected difficulty, when cheap wasn't enough
PROMOTE_FROM_CHEAP = {"easy": "hard", "medium": "hard"}


def main():
    init_db()
    known_queries = {q["query"] for q in json.loads(EVAL_QUERIES_PATH.read_text())}

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT query, difficulty, initial_tier, final_tier, escalated, "
            "escalation_reasons, timestamp FROM cascade_log WHERE escalated = 1"
        ).fetchall()

    candidates = []
    for query, guessed_difficulty, initial_tier, final_tier, escalated, reasons, ts in rows:
        if query in known_queries:
            continue  # already labeled, re-confirming teaches us nothing new

        if initial_tier == "cheap" and final_tier in ("mid", "frontier"):
            suggested_difficulty = PROMOTE_FROM_CHEAP.get(guessed_difficulty, "hard")
            confidence = "medium"  # cheap failing doesn't tell us if mid or frontier was truly needed
        elif initial_tier == "mid" and final_tier == "frontier":
            suggested_difficulty = "hard"  # confirms hard was right, mid genuinely needed help
            confidence = "high"
        else:
            continue

        candidates.append({
            "query": query,
            "classifier_guessed": guessed_difficulty,
            "suggested_difficulty": suggested_difficulty,
            "confidence": confidence,
            "final_tier": final_tier,
            "escalation_reasons": reasons,
            "first_seen": ts,
        })

    CANDIDATES_PATH.write_text(json.dumps(candidates, indent=2) + "\n")
    print(f"Found {len(candidates)} new candidate(s) for review -> {CANDIDATES_PATH}")
    for c in candidates:
        print(f"  [{c['confidence']:6}] {c['classifier_guessed']} -> {c['suggested_difficulty']}  "
              f"({c['final_tier']})  {c['query'][:60]}")


if __name__ == "__main__":
    main()
