"""Run the hard probe against two tiers and report whether they differ.

The question this exists to answer: does the frontier tier earn its price?
On the original eval set mid and frontier both score 1.00, so the set cannot
tell them apart, and routing has no evidence either way. This runs both on
questions chosen to need multi-step reasoning and grades them automatically.

    ./venv/bin/python scripts/run_hard_probe.py [--tiers mid,frontier] [--workers 6]

Costs real money -- roughly $0.01 a question for a reasoning frontier -- so
it prints a running total and takes --limit.
"""
import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import litellm  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
from app.model_config import TIER_CALL_PARAMS, TIER_MODEL_LIST  # noqa: E402

MODELS = {t["model_name"]: t["litellm_params"]["model"] for t in TIER_MODEL_LIST}


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
_print_lock = threading.Lock()


def _norm(text: str) -> str:
    """Compare mathematical answers without tripping over presentation.
    \\left(3,\\frac{\\pi}{2}\\right) and (3, \\frac{\\pi}{2}) are the same
    answer; so are '0.5' and '.5', and '$x$' and 'x'."""
    t = (text or "").strip()
    for junk in ("\\left", "\\right", "$", "\\!", "\\,", " ", "\n", "\\ "):
        t = t.replace(junk, "")
    t = t.replace("dfrac", "frac").replace("tfrac", "frac")
    t = re.sub(r"\\text\{(.*?)\}", r"\1", t)
    t = t.rstrip(".").lstrip("0") if re.fullmatch(r"0\.\d+", t) else t.rstrip(".")
    return t.lower()


def _extract_boxed(text: str) -> str | None:
    """Last \\boxed{...}, matching braces so nested ones survive."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth, out = 0, []
    for ch in text[i:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    return "".join(out)


def _norm_sequence(text: str) -> str:
    """For answers that are a list of items, compare the items and not the
    punctuation between them: BBH's word_sorting expects "syndrome therefrom"
    and models reply "syndrome, therefrom". Marking that wrong measured the
    formatting, not the sorting -- it scored a model 0/5 on a task it had
    actually got right."""
    return " ".join(re.split(r"[,\s]+", (text or "").strip().strip(".")) ).strip().lower()


def grade(q: dict, answer: str) -> bool:
    expected = q["expected"][0]
    if not answer:
        return False
    if q["eval_method"] == "boxed_match":
        got = _extract_boxed(answer)
        if got is None:                      # no \boxed: fall back to last line
            got = answer.strip().splitlines()[-1]
        return _norm(got) == _norm(expected)
    # suffix_match: "Answer: (D)" -- accept the bare letter too
    tail = answer.strip().splitlines()[-1] if answer.strip() else ""
    m = re.search(r"answer\s*:?\s*(.+)$", tail, re.I)
    got = (m.group(1) if m else tail).strip()
    if _norm(got) == _norm(expected) or _norm_sequence(got) == _norm_sequence(expected):
        return True
    letter = re.fullmatch(r"\(([A-Z])\)", expected.strip())
    return bool(letter) and _norm(got) in (_norm(letter.group(1)), _norm(expected))


def ask(model: str, params: dict, query: str) -> dict:
    start = time.perf_counter()
    for attempt in range(3):
        try:
            resp = litellm.completion(
                model=model, messages=[{"role": "user", "content": query}],
                timeout=900, **params)
            break
        except (litellm.RateLimitError, litellm.APIConnectionError, litellm.Timeout) as e:
            if attempt == 2:
                return {"text": "", "cost": 0.0, "s": 0.0, "error": type(e).__name__}
            time.sleep(15 * (attempt + 1))
    u = resp.usage
    return {"text": resp.choices[0].message.content or "",
            "cost": resp._hidden_params.get("response_cost", 0.0) or 0.0,
            "s": time.perf_counter() - start,
            "reasoning_tokens": getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="mid,frontier",
                    help="comma-separated: a tier name, or label=model@reasoning_effort")
    ap.add_argument("--only-failed-by", help="path to a results file: keep only questions this label got wrong")
    ap.add_argument("--questions", default=str(ROOT / "data" / "hard_probe.json"))
    ap.add_argument("--out", default=str(ROOT / "data" / "hard_probe_results.json"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text())
    if args.only_failed_by:
        prior = json.loads(Path(args.only_failed_by).read_text())
        label = next(iter(prior))
        failed = {r["id"] for r in prior[label] if not r["correct"]}
        questions = [q for q in questions if q["id"] in failed]
        print(f"restricted to the {len(questions)} questions {label!r} got wrong\n")
    questions = questions[: args.limit]
    specs = [parse_spec(x) for x in args.models.split(",")]
    tiers = [label for label, _, _ in specs]
    by_label = {label: (model, params) for label, model, params in specs}
    print(f"{len(questions)} questions x {len(specs)}: " +
          ", ".join(f"{l}={m}" + (f"@{p.get('reasoning_effort')}" if p.get("reasoning_effort") else "")
                    for l, m, p in specs) + "\n")

    results, done = {}, [0]

    def work(item):
        tier, q = item
        model, params = by_label[tier]
        r = ask(model, params, q["query"])
        ok = grade(q, r["text"])
        with _print_lock:
            done[0] += 1
            print(f"  [{done[0]:3}/{len(questions)*len(tiers)}] {tier:8} {q['id'][:34]:36} "
                  f"{'PASS' if ok else 'FAIL'}  ${r['cost']:.5f}  {r['s']:.0f}s", flush=True)
        return tier, q, r, ok

    jobs = [(t, q) for t in tiers for q in questions]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for tier, q, r, ok in pool.map(work, jobs):
            results.setdefault(tier, []).append(
                {"id": q["id"], "source": q["source"], "correct": ok, "cost": r["cost"],
                 "seconds": r["s"], "expected": q["expected"][0],
                 "got": (r["text"] or "")[-300:], "reasoning_tokens": r.get("reasoning_tokens")})

    Path(args.out).write_text(json.dumps(results, indent=2))
    print("\n=== accuracy on questions built to need multi-step reasoning ===")
    for tier in tiers:
        rs = results[tier]
        n_ok = sum(r["correct"] for r in rs)
        cost = sum(r["cost"] for r in rs)
        secs = sum(r["seconds"] for r in rs) / len(rs)
        print(f"  {tier:9} {n_ok:3}/{len(rs)} = {n_ok/len(rs):5.1%}   "
              f"${cost:.4f} total, ${cost/len(rs):.5f}/question, {secs:.0f}s avg")
        by_src = {}
        for r in rs:
            a, b = by_src.get(r["source"], (0, 0))
            by_src[r["source"]] = (a + r["correct"], b + 1)
        for src, (a, b) in sorted(by_src.items()):
            print(f"      {src:45} {a}/{b}")

    if len(tiers) == 2:
        a, b = (results[t] for t in tiers)
        by_id = {r["id"]: r["correct"] for r in b}
        only_a = sum(1 for r in a if r["correct"] and not by_id.get(r["id"]))
        only_b = sum(1 for r in a if not r["correct"] and by_id.get(r["id"]))
        print(f"\n  {tiers[0]} right where {tiers[1]} wrong: {only_a}")
        print(f"  {tiers[1]} right where {tiers[0]} wrong: {only_b}   <- what the frontier tier buys")


if __name__ == "__main__":
    main()
