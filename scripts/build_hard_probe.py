"""Assemble a set of questions hard enough to tell two strong models apart.

The 116-query eval set was labelled against a 3B local model, so its "hard"
band is things like "write a palindrome check" -- which every hosted model
answers perfectly. Mid and frontier both calibrate at 1.00 on it, which
means the set cannot see any difference between them, and in a full 116-query
run the frontier tier was never reached at all.

This pulls questions from public benchmarks chosen because a *non-reasoning*
model is expected to fail some of them while a reasoning model succeeds:

  MATH level 5   competition problems needing several dependent steps
  BBH            BIG-Bench Hard: tasks selected precisely because
                 chain-of-thought changes the outcome

Both ship ground-truth answers, so grading is automatic and free -- no LLM
judge in the labelling loop, unlike 40 of the original 116.

    ./venv/bin/python scripts/build_hard_probe.py [--n 50] [--out data/hard_probe.json]
"""
import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ROWS_URL = "https://datasets-server.huggingface.co/rows"

# BBH configs where success hinges on tracking several constraints at once,
# rather than on recall or a single lookup.
BBH_CONFIGS = [
    "logical_deduction_seven_objects",
    "tracking_shuffled_objects_seven_objects",
    "multistep_arithmetic_two",
    "word_sorting",
    "dyck_languages",
]


def fetch(dataset: str, config: str, split: str, offset: int, length: int):
    url = (f"{ROWS_URL}?dataset={urllib.parse.quote(dataset, safe='')}"
           f"&config={config}&split={split}&offset={offset}&length={length}")
    with urllib.request.urlopen(url, timeout=60) as r:
        return [x["row"] for x in json.load(r)["rows"]]


def math_questions(n: int) -> list[dict]:
    """MATH-500, level 5 only. Answer goes in \\boxed{} -- the dataset's own
    convention, and an anchor we can extract reliably."""
    out, offset = [], 0
    while len(out) < n and offset < 500:
        for row in fetch("HuggingFaceH4/MATH-500", "default", "test", offset, 100):
            if row.get("level") != 5:
                continue
            out.append({
                "id": f"math5_{row['unique_id'].strip('/').replace('/', '_')}",
                "query": row["problem"] + "\n\nPut your final answer in \\boxed{}.",
                "difficulty": "hard", "task_type": "competition_math",
                "eval_method": "boxed_match", "expected": [row["answer"]],
                "source": "MATH-500 level 5",
            })
            if len(out) >= n:
                break
        offset += 100
    return out


def bbh_questions(per_config: int) -> list[dict]:
    out = []
    for config in BBH_CONFIGS:
        for i, row in enumerate(fetch("lukaemon/bbh", config, "test", 0, per_config)):
            out.append({
                "id": f"bbh_{config}_{i}",
                "query": row["input"] + "\n\nEnd your reply with: Answer: <answer>",
                "difficulty": "hard", "task_type": f"bbh_{config}",
                "eval_method": "suffix_match", "expected": [row["target"]],
                "source": f"BBH {config}",
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="total questions")
    ap.add_argument("--out", default=str(ROOT / "data" / "hard_probe.json"))
    args = ap.parse_args()

    n_math = args.n // 2
    per_config = max(1, (args.n - n_math) // len(BBH_CONFIGS))
    questions = math_questions(n_math) + bbh_questions(per_config)

    Path(args.out).write_text(json.dumps(questions, indent=2))
    by_source = {}
    for q in questions:
        by_source[q["source"]] = by_source.get(q["source"], 0) + 1
    print(f"wrote {len(questions)} questions to {Path(args.out).relative_to(ROOT)}")
    for source, count in sorted(by_source.items()):
        print(f"  {count:3}  {source}")


if __name__ == "__main__":
    main()
