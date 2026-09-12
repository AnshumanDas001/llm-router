"""Dumps every llm_judge query with its three tier responses to a JSON file
for manual/LLM scoring (no automatic scorer exists for open-ended tasks)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.db import get_conn, init_db

QUERIES_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_queries.json"
OUT_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("judge_export.json")
TIERS = ["cheap", "mid", "frontier"]


def main():
    init_db()
    queries = json.loads(QUERIES_PATH.read_text())
    judge_queries = [q for q in queries if q["eval_method"] == "llm_judge"]

    out = []
    with get_conn() as conn:
        for q in judge_queries:
            entry = {"id": q["id"], "query": q["query"], "task_type": q["task_type"],
                     "difficulty": q["difficulty"], "responses": {}}
            for tier in TIERS:
                row = conn.execute(
                    "SELECT response_text FROM eval_responses WHERE query_id=? AND tier=?",
                    (q["id"], tier),
                ).fetchone()
                entry["responses"][tier] = row[0] if row else None
            out.append(entry)

    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"Exported {len(out)} judge queries x {len(TIERS)} tiers to {OUT_PATH}")


if __name__ == "__main__":
    main()
