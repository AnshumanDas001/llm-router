"""Week 2: automatic scoring for queries with a deterministic correctness check
(exact_match, exact_match_set, exact_match_numeric, exact_match_contains_any,
schema_json, schema_pattern). LLM-judge queries are left for score_judge.py.
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.db import get_conn, init_db, log_eval_score

QUERIES_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_queries.json"
TIERS = ["cheap", "mid", "frontier"]


def score_exact_match(text: str, expected: list[str]) -> float:
    low = text.lower()
    return 1.0 if any(e.lower() in low for e in expected) else 0.0


def score_exact_match_set(text: str, expected: list[str]) -> float:
    low = text.lower()
    found = sum(1 for e in expected if e.lower() in low)
    return found / len(expected)


def score_exact_match_numeric(text: str, expected_value: float, tolerance: float) -> float:
    numbers = re.findall(r"-?\d+\.?\d*", text)
    for n in numbers:
        try:
            if abs(float(n) - expected_value) <= tolerance:
                return 1.0
        except ValueError:
            continue
    return 0.0


def extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def score_schema_json(text: str, expected_json: dict) -> float:
    candidate = extract_first_json_object(text)
    if candidate is None:
        return 0.0
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return 0.0
    matched = sum(1 for k, v in expected_json.items()
                   if k in parsed and str(parsed[k]).lower() == str(v).lower())
    return matched / len(expected_json)


def score_schema_pattern(text: str, pattern: str) -> float:
    return 1.0 if re.search(pattern, text, re.IGNORECASE | re.DOTALL) else 0.0


def score_one(q: dict, response_text: str) -> float:
    method = q["eval_method"]
    if method == "exact_match" or method == "exact_match_contains_any":
        return score_exact_match(response_text, q["expected"])
    if method == "exact_match_set":
        return score_exact_match_set(response_text, q["expected"])
    if method == "exact_match_numeric":
        return score_exact_match_numeric(response_text, q["expected_value"], q["tolerance"])
    if method == "schema_json":
        return score_schema_json(response_text, q["expected_json"])
    if method == "schema_pattern":
        return score_schema_pattern(response_text, q["expected_pattern"])
    raise ValueError(f"not an automatic method: {method}")


def main():
    init_db()
    queries = {q["id"]: q for q in json.loads(QUERIES_PATH.read_text())}
    automatic = {qid: q for qid, q in queries.items() if q["eval_method"] != "llm_judge"}

    with get_conn() as conn:
        # Drop stale automatic scores for queries later reclassified to llm_judge
        # (or removed), so a relabeling doesn't leave orphaned rows behind.
        placeholders = ",".join("?" * len(automatic))
        conn.execute(
            f"DELETE FROM eval_scores WHERE source='automatic' AND query_id NOT IN ({placeholders})",
            list(automatic.keys()),
        )
        conn.commit()
        responses = conn.execute(
            "SELECT query_id, tier, response_text FROM eval_responses"
        ).fetchall()

    scored = 0
    for query_id, tier, response_text in responses:
        q = automatic.get(query_id)
        if q is None:
            continue
        score = score_one(q, response_text)
        log_eval_score(
            query_id=query_id, tier=tier, eval_method=q["eval_method"], score=score,
            source="automatic", notes="", timestamp=datetime.now(timezone.utc).isoformat(),
        )
        scored += 1

    print(f"Scored {scored} (query, tier) pairs across {len(automatic)} automatic-check queries.\n")

    print(f"{'query_id':<8}{'task_type':<16}{'cheap':>8}{'mid':>8}{'frontier':>10}")
    with get_conn() as conn:
        for qid in sorted(automatic):
            row = {}
            for tier in TIERS:
                r = conn.execute(
                    "SELECT score FROM eval_scores WHERE query_id=? AND tier=? AND source='automatic'",
                    (qid, tier),
                ).fetchone()
                row[tier] = r[0] if r else None
            print(f"{qid:<8}{automatic[qid]['task_type']:<16}"
                  f"{row['cheap']:>8.2f}{row['mid']:>8.2f}{row['frontier']:>10.2f}")

    with get_conn() as conn:
        for tier in TIERS:
            avg = conn.execute(
                "SELECT AVG(score) FROM eval_scores WHERE tier=? AND source='automatic'",
                (tier,),
            ).fetchone()[0]
            print(f"\n{tier} avg automatic-check score: {avg:.3f}")


if __name__ == "__main__":
    main()
