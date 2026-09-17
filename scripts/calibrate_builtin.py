"""Calibrate the built-in tiers -- whatever CHEAP/MID/FRONTIER_MODEL point at
-- and save the result so the routing policy uses real numbers for this
stack instead of the defaults measured on the original one.

Runs ~24 stratified queries per tier against the live models (so it costs
whatever three models charge for ~72 short answers -- cents). Writes
data/builtin_calibration.json tagged with the model names; the app only
uses the file if those still match its configured stack.

    ./venv/bin/python scripts/calibrate_builtin.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.calibration import DEFAULT_MAX_QUERIES, calibrate_model
from app.baseline_cost import estimate_cost_for_model
from app.model_config import BUILTIN_CALIBRATION_PATH, CHEAP_MODEL, FRONTIER_MODEL, MID_MODEL, TIER_CALL_PARAMS

# (prompt, completion) tokens of a typical answer per band, from the eval set.
TYPICAL_TOKENS = {"easy": (450, 150), "medium": (500, 400), "hard": (600, 900)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="cheap,mid,frontier",
                    help="subset to (re)calibrate; the others keep their numbers from the existing file")
    args = ap.parse_args()
    only = set(args.tiers.split(","))

    models = {"cheap": CHEAP_MODEL, "mid": MID_MODEL, "frontier": FRONTIER_MODEL}
    tiers = {}
    if BUILTIN_CALIBRATION_PATH.exists() and only != set(models):
        previous = json.loads(BUILTIN_CALIBRATION_PATH.read_text())
        # Reuse a tier's numbers only if it's still the same model.
        tiers = {t: v for t, v in previous.get("tiers", {}).items()
                 if t not in only and previous.get("models", {}).get(t) == models[t]}
    for tier, model in models.items():
        if tier not in only:
            print(f"keeping   {tier:9} {model} (not selected)")
            continue
        print(f"calibrating {tier:9} {model} ...", flush=True)
        # A free local tier gets the whole eval set: 8 questions per band
        # once measured the cheap model at 0.88 on hard where 60 questions
        # say 0.50-0.64, and 0.88 would have routed hard questions to it.
        n = None if model.startswith("ollama/") else DEFAULT_MAX_QUERIES
        stats = calibrate_model(model, None, max_queries=n, call_params=TIER_CALL_PARAMS[tier])
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
        tiers[tier] = {"quality": q, "cost": c}
        print(f"  quality easy {q['easy']:.2f} / med {q['medium']:.2f} / hard {q['hard']:.2f}"
              f"   cost ${c['easy']:.6f} / ${c['medium']:.6f} / ${c['hard']:.6f}"
              f"   {stats['avg_latency_ms']:.0f}ms  errors={stats['n_errors']}")

    BUILTIN_CALIBRATION_PATH.write_text(json.dumps({"models": models, "tiers": tiers}, indent=2))
    print(f"\nwrote {BUILTIN_CALIBRATION_PATH.name}. Restart the server; routing now uses these.")


if __name__ == "__main__":
    main()
