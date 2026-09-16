"""Calibrate the built-in tiers -- whatever CHEAP/MID/FRONTIER_MODEL point at
-- and save the result so the routing policy uses real numbers for this
stack instead of the defaults measured on the original one.

Runs ~24 stratified queries per tier against the live models (so it costs
whatever three models charge for ~72 short answers -- cents). Writes
data/builtin_calibration.json tagged with the model names; the app only
uses the file if those still match its configured stack.

    ./venv/bin/python scripts/calibrate_builtin.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.calibration import DEFAULT_MAX_QUERIES, calibrate_model
from app.model_config import BUILTIN_CALIBRATION_PATH, CHEAP_MODEL, FRONTIER_MODEL, MID_MODEL


def main():
    models = {"cheap": CHEAP_MODEL, "mid": MID_MODEL, "frontier": FRONTIER_MODEL}
    tiers = {}
    for tier, model in models.items():
        print(f"calibrating {tier:9} {model} ...", flush=True)
        stats = calibrate_model(model, None, max_queries=DEFAULT_MAX_QUERIES)
        if stats["n_queries"] == 0:
            print(f"  FAILED: {stats['first_error']}")
            sys.exit(1)
        q, c = stats["quality_by_difficulty"], stats["cost_by_difficulty"]
        tiers[tier] = {"quality": q, "cost": c}
        print(f"  quality easy {q['easy']:.2f} / med {q['medium']:.2f} / hard {q['hard']:.2f}"
              f"   cost ${c['easy']:.6f} / ${c['medium']:.6f} / ${c['hard']:.6f}"
              f"   {stats['avg_latency_ms']:.0f}ms  errors={stats['n_errors']}")

    BUILTIN_CALIBRATION_PATH.write_text(json.dumps({"models": models, "tiers": tiers}, indent=2))
    print(f"\nwrote {BUILTIN_CALIBRATION_PATH.name}. Restart the server; routing now uses these.")


if __name__ == "__main__":
    main()
