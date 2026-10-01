"""Run a question set against two or more tiers and report where they differ.

The question this exists to answer: does the frontier tier earn its price?
It runs each tier (or a candidate model spec) on a probe set, grades every
answer automatically with app/evaluation/graders.py, and prints accuracy
per task family plus how often each tier is right where the other is wrong.

    ./venv/bin/python -m scripts.probe.run_probe \\
        --questions data/eval/expert_queries.json --models mid,frontier --budget 1.00

Costs real money -- a thinking frontier can spend several cents on one
question -- so it keeps a running total and stops starting new questions
once --budget is spent. Every answer is appended to
data/probe/<questions>_results.jsonl as it arrives, so a crash or a budget
stop costs nothing already paid for, and a re-run resumes where it left off.
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import litellm

from app.config import TIER_CALL_PARAMS, TIER_MODEL_LIST
from app.evaluation.datasets import prompt_for
from app.evaluation.graders import score_one
from app.paths import EXPERT_QUERIES, PROBE_DIR

MODELS = {t["model_name"]: t["litellm_params"]["model"] for t in TIER_MODEL_LIST}


def results_path(questions: Path) -> Path:
    return PROBE_DIR / f"{questions.stem}_results.jsonl"


def load_results(path: Path) -> dict:
    """{tier: [row, ...]} from a results JSONL. Errored calls are left out,
    which is also what makes a re-run retry them."""
    results = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if not row.get("error"):
                    results.setdefault(row["tier"], []).append(row)
    return results


def parse_spec(spec: str) -> tuple[str, str, dict]:
    """'label=model@effort' -> (label, model, params). A bare tier name uses
    that tier's configured model and call params, so the probe measures the
    stack as it actually runs; an explicit spec lets a candidate be tried
    without editing config."""
    if "=" not in spec:
        return spec, MODELS[spec], dict(TIER_CALL_PARAMS[spec])
    label, rest = spec.split("=", 1)
    model, _, effort = rest.partition("@")
    # Generous cap: providers bill tokens used, not tokens allowed, and a
    # reasoning model that runs out mid-thought returns an empty answer --
    # which reads as a wrong answer and silently slanders the model.
    params = {"max_tokens": 32000, "drop_params": True}
    if effort:
        params["reasoning_effort"] = effort
    return label, model, params


def ask(model: str, params: dict, query: str) -> dict:
    start = time.perf_counter()
    for attempt in range(3):
        try:
            resp = litellm.completion(
                model=model, messages=[{"role": "user", "content": query}],
                timeout=900, **params)
            break
        except Exception as e:
            # 402 "in_flight_budget_exhausted" is a throttle, not a failure:
            # OpenRouter caps concurrent spend against the remaining balance
            # and asks you to wait, so treat it like a rate limit.
            transient = isinstance(e, (litellm.RateLimitError, litellm.APIConnectionError,
                                       litellm.Timeout)) or "in_flight" in str(e)
            if attempt == 2 or not transient:
                return {"text": "", "cost": 0.0, "s": 0.0, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            time.sleep(30 * (attempt + 1))
    usage = resp.usage
    details = getattr(usage, "completion_tokens_details", None)
    return {"text": resp.choices[0].message.content or "",
            "cost": resp._hidden_params.get("response_cost", 0.0) or 0.0,
            "s": time.perf_counter() - start,
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "reasoning_tokens": getattr(details, "reasoning_tokens", None),
            "finish_reason": resp.choices[0].finish_reason}


def summarise(results: dict, labels: list[str]) -> None:
    """Accuracy over every answer recorded, grouped by the task family
    stored on each row (not by the questions selected for this run, which
    --only-failed-by narrows to a handful)."""
    print("\n=== accuracy by tier and task family ===")
    for label in labels:
        rows = [r for r in results.get(label, []) if not r.get("error")]
        if not rows:
            continue
        n_ok = sum(r["correct"] for r in rows)
        cost = sum(r["cost"] for r in rows)
        secs = sum(r["seconds"] for r in rows) / len(rows)
        print(f"  {label:9} {n_ok:3}/{len(rows)} = {n_ok / len(rows):5.1%}   "
              f"${cost:.4f} total, ${cost / len(rows):.5f}/question, {secs:.0f}s avg")
        by_family = {}
        for r in rows:
            fam = r.get("task_type") or r.get("source") or "?"
            a, b = by_family.get(fam, (0, 0))
            by_family[fam] = (a + r["correct"], b + 1)
        for fam, (a, b) in sorted(by_family.items()):
            print(f"      {fam:30} {a:3}/{b:<3} {a / b:5.0%}")
        errors = len(results.get(label, [])) - len(rows)
        if errors:
            print(f"      ({errors} calls errored and are excluded)")

    if len(labels) == 2:
        a, b = (results.get(t, []) for t in labels)
        by_id = {r["id"]: r["correct"] for r in b if not r.get("error")}
        both = [r for r in a if not r.get("error") and r["id"] in by_id]
        only_a = sum(1 for r in both if r["correct"] and not by_id[r["id"]])
        only_b = sum(1 for r in both if not r["correct"] and by_id[r["id"]])
        print(f"\n  on the {len(both)} questions both answered:")
        print(f"  {labels[0]} right where {labels[1]} wrong: {only_a}")
        print(f"  {labels[1]} right where {labels[0]} wrong: {only_b}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="mid,frontier",
                    help="comma-separated: a tier name, or label=model@reasoning_effort")
    ap.add_argument("--questions", default=str(EXPERT_QUERIES))
    ap.add_argument("--out", help="results file (default: data/probe/<questions>_results.jsonl)")
    ap.add_argument("--families", help="comma-separated task_type values to keep")
    ap.add_argument("--per-family", type=int, help="keep at most this many questions per family")
    ap.add_argument("--only-failed-by", help="label: keep only questions this label already got wrong in --out")
    ap.add_argument("--budget", type=float, default=1.0,
                    help="stop starting new questions once this many dollars are spent in this run")
    ap.add_argument("--workers", type=int, default=6,
                    help="too many against a low balance trips the provider's in-flight spend cap")
    ap.add_argument("--restart", action="store_true", help="ignore any partial results and start over")
    ap.add_argument("--summary-only", action="store_true", help="print the summary of --out and exit")
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text())
    out = Path(args.out) if args.out else results_path(Path(args.questions))

    if args.families:
        keep = set(args.families.split(","))
        questions = [q for q in questions if q.get("task_type") in keep]
    if args.per_family:
        kept, counts = [], {}
        for q in questions:
            counts[q.get("task_type")] = counts.get(q.get("task_type"), 0) + 1
            if counts[q.get("task_type")] <= args.per_family:
                kept.append(q)
        questions = kept

    specs = [parse_spec(x) for x in args.models.split(",")]
    labels = [label for label, _, _ in specs]
    by_label = {label: (model, params) for label, model, params in specs}

    # Every answer is appended to the JSONL as it arrives, so a crash partway
    # through costs nothing already paid for -- a 12-worker run once died on
    # a provider throttle at 221 of 298 and took every result with it. The
    # same file is what makes a re-run resumable.
    results = {} if args.restart else load_results(out)
    already = {(tier, r["id"]) for tier, rows in results.items() for r in rows}
    if args.summary_only:
        summarise(results, labels)
        return

    if args.only_failed_by:
        failed = {r["id"] for r in results.get(args.only_failed_by, []) if not r["correct"]}
        questions = [q for q in questions if q["id"] in failed]
        print(f"restricted to the {len(questions)} questions {args.only_failed_by!r} got wrong")

    jobs = [(t, q) for t in labels for q in questions if (t, q["id"]) not in already]
    print(f"{len(questions)} questions x {len(specs)} models, {len(jobs)} still to run, "
          f"budget ${args.budget:.2f}: " +
          ", ".join(f"{l}={m}" + (f"@{p.get('reasoning_effort')}" if p.get("reasoning_effort") else "")
                    for l, m, p in specs))
    if already:
        print(f"resuming: {len(already)} answers already recorded in {out.name}")

    lock = threading.Lock()
    spent, done = [0.0], [0]
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("w" if args.restart else "a")

    def work(item):
        label, q = item
        with lock:
            if spent[0] >= args.budget:
                return None
        model, params = by_label[label]
        r = ask(model, params, prompt_for(q))
        ok = bool(score_one(q, r["text"]) >= 1.0) if not r.get("error") else False
        row = {"tier": label, "id": q["id"], "task_type": q.get("task_type"),
               "source": q.get("source"), "correct": ok, "cost": r["cost"], "seconds": r["s"],
               "expected": q.get("expected", q.get("target")), "got": (r["text"] or "")[-600:],
               "completion_tokens": r.get("completion_tokens"),
               "reasoning_tokens": r.get("reasoning_tokens"),
               "finish_reason": r.get("finish_reason"), "error": r.get("error")}
        with lock:
            spent[0] += r["cost"]
            done[0] += 1
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"  [{done[0]:3}/{len(jobs)}] {label:8} {q['id'][:30]:32} "
                  f"{'PASS' if ok else 'FAIL'}  ${r['cost']:.4f}  {r['s']:4.0f}s  "
                  f"spent ${spent[0]:.3f}{'  ' + r['error'] if r.get('error') else ''}", flush=True)
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(work, jobs):
            if row is not None and not row.get("error"):
                results.setdefault(row["tier"], []).append(row)
    fh.close()
    if spent[0] >= args.budget:
        print(f"\nbudget of ${args.budget:.2f} reached; re-run to continue where this stopped")

    summarise(results, labels)


if __name__ == "__main__":
    main()
