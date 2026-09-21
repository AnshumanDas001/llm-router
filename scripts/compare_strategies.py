"""Price the cascade against the two obvious alternatives on the SAME queries.

The number that justifies a cascade is not "cheaper than frontier" -- a
single mid-tier model can be cheaper than frontier too. It is whether the
cascade beats "just send everything to mid". This reads the most recent
cascade run and prices each query three ways.

Two baselines are printed for each alternative, because neither is right
on its own:

  "over the cascade's tokens"  the tokens the cascade actually produced,
      priced at the other model's rates. Apples-to-apples on length, but a
      different model writes a different answer: a wordy cheap model's
      tokens at mid's rates overstate mid, and a reasoning model's hidden
      thinking tokens aren't in the cascade's counts at all -- a frontier
      that thinks for 3,000 tokens per answer came out *cheaper than mid*
      this way, which is nonsense.
  "from calibration"  what that model's own answers cost per difficulty
      band when it was calibrated, times this run's band mix. Includes its
      thinking tokens. Measured on a sample, so it's an estimate too.

The truth is between them; the README quotes the conservative one.

    ./venv/bin/python scripts/compare_strategies.py [--table cascade_log]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.baseline_cost import estimate_cost_for_model, estimate_frontier_cost
from app.cascade import DEFAULT_TIER_MODELS
from app.db import get_conn
from app.model_config import BUILTIN_CALIBRATION


def from_calibration(tier: str, rows) -> float | None:
    """Sum of this tier's measured per-band cost over the run's queries, or
    None if the tier was never measured (list-price placeholder only)."""
    cal = BUILTIN_CALIBRATION.get(tier) or {}
    if cal.get("unmeasured") or not cal.get("cost"):
        return None
    return sum(cal["cost"].get(r[6]) or 0.0 for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="cascade_log")
    args = ap.parse_args()

    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT total_cost, tokens_in, tokens_out, final_tier, escalated, "
            f"total_latency_ms, difficulty FROM {args.table}"
        ).fetchall()
    if not rows:
        print("no rows"); return

    n = len(rows)
    mid_model = DEFAULT_TIER_MODELS["mid"]
    cascade = sum(r[0] for r in rows)
    frontier = sum(estimate_frontier_cost(r[1], r[2]) for r in rows)
    mid_only = sum(estimate_cost_for_model(mid_model, r[1], r[2]) for r in rows)

    mid_cal = from_calibration("mid", rows)
    frontier_cal = from_calibration("frontier", rows)

    print(f"{n} queries (mid = {mid_model}, frontier = {DEFAULT_TIER_MODELS['frontier']}):\n")
    print(f"  {'strategy':<26}{'over cascade tokens':>21}{'from calibration':>19}")
    print(f"  {'frontier for everything':<26}${frontier:>20.5f}"
          + (f"${frontier_cal:>18.5f}" if frontier_cal is not None else f"{'(not measured)':>19}"))
    print(f"  {'mid for everything':<26}${mid_only:>20.5f}"
          + (f"${mid_cal:>18.5f}" if mid_cal is not None else f"{'(not measured)':>19}"))
    print(f"  {'cascade (actual spend)':<26}${cascade:>20.5f}")

    # The conservative claim: whichever mid-only estimate is lower.
    mid_low = min(x for x in (mid_only, mid_cal) if x is not None)
    mid_high = max(x for x in (mid_only, mid_cal) if x is not None)
    if cascade < mid_low:
        print(f"\n  => cascade beats mid-only by {(1 - cascade / mid_low) * 100:.0f}-{(1 - cascade / mid_high) * 100:.0f}% "
              f"(${mid_low - cascade:.5f} to ${mid_high - cascade:.5f} on this run)")
    elif cascade < mid_high:
        print(f"\n  => inconclusive: cascade is between the two mid-only estimates")
    else:
        print(f"\n  => mid-only beats cascade by ${cascade - mid_high:.5f} ({(cascade / mid_high - 1) * 100:.0f}% of mid-only)")
    if frontier_cal is not None:
        print(f"     vs frontier (from calibration): {(1 - cascade / frontier_cal) * 100:.0f}% cheaper\n")
    else:
        print()

    by = {}
    for r in rows:
        key = "escalated" if r[4] else f"answered at {r[3]}"
        by.setdefault(key, []).append(r)
    print("  where queries ended up:")
    for key, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        print(f"    {key:<20} n={len(rs):3}  avg ${sum(x[0] for x in rs) / len(rs):.5f}"
              f"  avg {sum(x[5] for x in rs) / len(rs):.0f}ms")


if __name__ == "__main__":
    main()
