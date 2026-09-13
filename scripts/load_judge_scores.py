"""Load Claude's manual judge scores (ground-truth quality scores for
open-ended queries) into eval_scores, source='judge_claude'."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.db import init_db, log_eval_score

SCORES_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(__file__).resolve().parent.parent / "data" / "judge_scores.json"
)

# Re-verified against a full regeneration (the original DB was accidentally
# deleted; see project history). Several items changed on rerun due to LLM
# non-determinism (e.g. q47, q40, q57 flipped correctness) and the mid-tier
# max_tokens fix (q53/54/58/59 no longer truncate). Notes reflect this run's
# actual text, not the original judging pass.
NOTES = {
    ("q02", "cheap"): "self-contradictory framing: calls the same review 'slightly negative overall' and 'leaning towards positive' in one sentence",
    ("q23", "cheap"): "unrequested preamble adds a 4th sentence beyond the requested 3",
    ("q32", "mid"): "verbose with headers/sub-bullets, exceeds 'brief summary' scope",
    ("q44", "cheap"): "code is correct but completely omits the explicitly requested time-complexity explanation",
    ("q46", "cheap"): "claims 'no recursion' and 'not enough iterations' cause O(n^2), which is not true; the partition() helper is internally inconsistent",
    ("q48", "cheap"): "TokenBucket logic is correct this time, but SlidingWindowRateLimiter never expires entries by time -- it's just a permanent counter, not a real sliding window",
    ("q49", "cheap"): "code correct but explicitly requested 'explain why it works' is entirely missing",
    ("q51", "cheap"): "reasonable generic debugging steps, but never hypothesizes about why Monday/9am specifically matters (traffic pattern, cron jobs) the way mid/frontier do",
    ("q55", "cheap"): "correctly IDs appeal to popularity but 'Ad Hominem' and 'Slippery Slope' are both misapplied -- neither fits this argument",
    ("q57", "cheap"): "fails the core requirement: uses a new list (O(n) extra space) instead of the requested in-place O(1)-space merge, and still omits the requested tradeoff explanation",
}


def main():
    init_db()
    scores = json.loads(SCORES_PATH.read_text())
    now = datetime.now(timezone.utc).isoformat()
    count = 0
    for query_id, tiers in scores.items():
        for tier, score in tiers.items():
            note = NOTES.get((query_id, tier), "")
            log_eval_score(
                query_id=query_id, tier=tier, eval_method="llm_judge", score=score,
                source="judge_claude", notes=note, timestamp=now,
            )
            count += 1
    print(f"Loaded {count} judge_claude scores.")


if __name__ == "__main__":
    main()
