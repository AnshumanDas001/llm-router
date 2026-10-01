"""Calibrate the built-in tiers -- whatever CHEAP/MID/FRONTIER_MODEL point at
-- and save the result so the routing policy uses real numbers for this
stack instead of the defaults measured on the original one.

easy / medium / hard are measured live: ~6 stratified questions per band
for mid and frontier, every gradable question for the nearly-free cheap
tier. The expert band is read from the probe results instead
(data/probe/expert_queries_results.jsonl, written by
scripts/probe/run_probe.py): a thinking frontier spends $0.13-0.29 on one
AIME problem, so those answers are paid for once, kept, and extended with
the probe rather than re-bought on every calibration.

Writes data/calibration/builtin.json tagged with the model names; the app
only uses the file if those still match its configured stack.

    ./venv/bin/python -m scripts.calibration.calibrate_builtin [--tiers cheap,mid] [--expert-only]
"""
import argparse
import json

from app.config import BUILTIN_CALIBRATION_PATH, CHEAP_MODEL, FRONTIER_MODEL, MID_MODEL, TIER_CALL_PARAMS
from app.evaluation.calibration import DEFAULT_MAX_QUERIES, calibrate_model
from app.paths import EXPERT_QUERIES
from app.pricing import estimate_cost_for_model
from scripts.probe.run_probe import load_results, results_path

LIVE_BANDS = ("easy", "medium", "hard")

# (prompt, completion) tokens of a typical answer per band, from the eval set.
TYPICAL_TOKENS = {"easy": (450, 150), "medium": (500, 400), "hard": (600, 900),
                  "expert": (300, 3000)}


def expert_band(models: dict) -> dict:
    """{tier: (quality, cost, n)} for the expert band, from probe results.

    Quality is only reported where the tier answered a fair sample. A probe
    run with --only-failed-by answers just the questions the tier below got
    wrong -- the hardest ones -- so its accuracy there says how much it
    *rescues*, not how good it is, and is kept out of the quality figure.
    Its cost is still a fair (if pessimistic) price for routing."""
    results = load_results(results_path(EXPERT_QUERIES))
    asked = {tier: {r["id"] for r in rows} for tier, rows in results.items()}
    widest = max(asked.values(), key=len, default=set())
    out = {}
    for tier in models:
        rows = results.get(tier, [])
        if not rows:
            continue
        cost = sum(r["cost"] for r in rows) / len(rows)
        fair = asked[tier] >= widest
        quality = sum(r["correct"] for r in rows) / len(rows) if fair else None
        out[tier] = (quality, cost, len(rows))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="cheap,mid,frontier",
                    help="tiers to re-measure live; the others keep their numbers from the existing file")
    ap.add_argument("--expert-only", action="store_true",
                    help="no live calls: refresh only the expert band from the probe results")
    args = ap.parse_args()
    only = set() if args.expert_only else set(args.tiers.split(","))

    models = {"cheap": CHEAP_MODEL, "mid": MID_MODEL, "frontier": FRONTIER_MODEL}
    tiers = {}
    if BUILTIN_CALIBRATION_PATH.exists():
        previous = json.loads(BUILTIN_CALIBRATION_PATH.read_text())
        # Reuse a tier's numbers only if it's still the same model.
        tiers = {t: v for t, v in previous.get("tiers", {}).items()
                 if t not in only and previous.get("models", {}).get(t) == models[t]}

    for tier, model in models.items():
        if tier not in only:
            print(f"keeping   {tier:9} {model}")
            continue
        print(f"calibrating {tier:9} {model} ...", flush=True)
        # The cheap tier gets the whole eval set. Its numbers matter most
        # (they decide what never reaches a paid model) and it's the tier
        # that's nearly free to measure: 8 questions per band once put a
        # 3B model at 0.88 on hard where 76 say 0.73, and a hosted 8B at
        # 1.00 where 120 samples say 0.74 -- either would have routed hard
        # questions to it on noise.
        n = None if tier == "cheap" else DEFAULT_MAX_QUERIES * len(LIVE_BANDS) // 4
        stats = calibrate_model(model, None, max_queries=n, call_params=TIER_CALL_PARAMS[tier],
                                bands=LIVE_BANDS)
        if stats["n_queries"] == 0:
            # A tier that can't be reached (no key yet) still needs a cost
            # for the policy to price escalating into it. List price at the
            # eval set's typical token counts per band; quality is left
            # unmeasured, which the policy treats as "never a start tier".
            print(f"  UNREACHABLE ({stats['first_error']}); pricing from list price, quality unmeasured")
            tiers[tier] = {
                "quality": {d: None for d in TYPICAL_TOKENS},
                "cost": {d: estimate_cost_for_model(model, *toks) for d, toks in TYPICAL_TOKENS.items()},
                "unmeasured": True,
            }
            continue
        q, c = stats["quality_by_difficulty"], stats["cost_by_difficulty"]
        tiers[tier] = {"quality": q, "cost": c, "n": stats["n_by_difficulty"],
                       "latency_ms": stats["avg_latency_ms"], "n_queries": stats["n_queries"]}
        print(f"  quality easy {q['easy']:.2f} / med {q['medium']:.2f} / hard {q['hard']:.2f}"
              f"   cost ${c['easy']:.6f} / ${c['medium']:.6f} / ${c['hard']:.6f}"
              f"   {stats['avg_latency_ms']:.0f}ms  errors={stats['n_errors']}")

    for tier, (quality, cost, n) in expert_band(models).items():
        if tier not in tiers:
            continue
        tiers[tier]["quality"]["expert"] = quality
        tiers[tier]["cost"]["expert"] = cost
        tiers[tier].setdefault("n", {})["expert"] = n
        shown = "unmeasured (rescue run only)" if quality is None else f"{quality:.2f}"
        print(f"  expert    {tier:9} quality {shown}   cost ${cost:.4f}   n={n}   (from probe)")

    BUILTIN_CALIBRATION_PATH.write_text(json.dumps({"models": models, "tiers": tiers}, indent=2))
    print(f"\nwrote {BUILTIN_CALIBRATION_PATH.name}. Restart the server; routing now uses these.")


if __name__ == "__main__":
    main()
