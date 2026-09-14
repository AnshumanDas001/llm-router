"""Price the cascade against the two obvious alternatives on the SAME queries.

The number that justifies a cascade is not "cheaper than frontier" -- a
single mid-tier model can be cheaper than frontier too. It is whether the
cascade beats "just send everything to mid". This reads the most recent
cascade run and prices each query three ways over the identical token
counts, so the comparison is apples-to-apples.

    ./venv/bin/python scripts/compare_strategies.py [--table cascade_log]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.baseline_cost import estimate_cost_for_model, estimate_frontier_cost
from app.cascade import DEFAULT_TIER_MODELS
from app.db import get_conn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="cascade_log")
    args = ap.parse_args()

    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT total_cost, tokens_in, tokens_out, final_tier, escalated, "
            f"total_latency_ms FROM {args.table}"
        ).fetchall()
    if not rows:
        print("no rows"); return

    n = len(rows)
    mid_model = DEFAULT_TIER_MODELS["mid"]
    cascade = sum(r[0] for r in rows)
    frontier = sum(estimate_frontier_cost(r[1], r[2]) for r in rows)
    mid_only = sum(estimate_cost_for_model(mid_model, r[1], r[2]) for r in rows)

    print(f"{n} queries, priced three ways over identical tokens "
          f"(frontier = {DEFAULT_TIER_MODELS['frontier']}):\n")
    print(f"  {'strategy':<26}{'total':>11}{'per query':>13}{'vs frontier':>13}")
    for label, total in [("frontier for everything", frontier),
                         ("mid for everything", mid_only),
                         ("cascade", cascade)]:
        print(f"  {label:<26}${total:>10.5f}${total / n:>12.5f}"
              f"{(1 - total / frontier) * 100:>12.0f}%")

    d = cascade - mid_only
    verdict = "cascade beats mid-only" if d < 0 else "mid-only beats cascade"
    print(f"\n  => {verdict} by ${abs(d):.5f} ({abs(d) / mid_only * 100:.0f}% of mid-only)\n")

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
