"""Merge human-approved active-learning candidates into the classifier's
permanent reference set (data/eval_queries.json).

Usage: mark candidates as approved in data/active_learning_candidates.json
(add `"approved": true` to the ones you've reviewed and trust), then run
this script. Approved candidates get appended with a new id and
eval_method="llm_judge" (a human should later add a real expected answer /
tighter eval_method if the query type supports one -- llm_judge is the
safe default since a harvested query has no pre-computed ground truth).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EVAL_QUERIES_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_queries.json"
CANDIDATES_PATH = Path(__file__).resolve().parent.parent / "data" / "active_learning_candidates.json"


def main():
    if not CANDIDATES_PATH.exists():
        print(f"No candidates file at {CANDIDATES_PATH}. Run harvest_active_learning_candidates.py first.")
        return

    candidates = json.loads(CANDIDATES_PATH.read_text())
    approved = [c for c in candidates if c.get("approved")]
    if not approved:
        print("No candidates marked approved:true. Nothing to merge.")
        return

    existing = json.loads(EVAL_QUERIES_PATH.read_text())
    existing_queries = {q["query"] for q in existing}
    next_num = max(int(q["id"][1:]) for q in existing) + 1

    added = 0
    for c in approved:
        if c["query"] in existing_queries:
            continue  # got added some other way since harvesting
        existing.append({
            "id": f"q{next_num}",
            "query": c["query"],
            "difficulty": c["suggested_difficulty"],
            "task_type": "harvested",
            "eval_method": "llm_judge",
        })
        next_num += 1
        added += 1

    EVAL_QUERIES_PATH.write_text(json.dumps(existing, indent=2) + "\n")

    remaining = [c for c in candidates if not c.get("approved")]
    CANDIDATES_PATH.write_text(json.dumps(remaining, indent=2) + "\n")

    print(f"Merged {added} new example(s) into {EVAL_QUERIES_PATH} "
          f"({len(existing)} total). Restart the router process to pick up "
          f"the refreshed reference set (or call classifier.refresh_reference_cache()).")


if __name__ == "__main__":
    main()
