"""Regenerate the README's SVG charts straight from the eval database.

Everything plotted here is measured, not illustrative -- run this after a
new eval and the README updates with it. SVGs are written with a
transparent background and mid-grey text so they stay legible on both
GitHub's light and dark themes.

    ./venv/bin/python scripts/make_readme_charts.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.baseline_cost import estimate_frontier_cost
from app.db import get_conn

OUT_DIR = Path(__file__).resolve().parent.parent / "reports" / "charts"

TIERS = ["cheap", "mid", "frontier"]
DIFFICULTIES = ["easy", "medium", "hard"]
COLORS = {"cheap": "#3987e5", "mid": "#d95926", "frontier": "#199e70"}
INK = "#767676"          # readable on both GitHub themes
INK_STRONG = "#9b9b9b"
GRID = "#8884"
THRESHOLD = 0.80         # app/routing_policy.py QUALITY_THRESHOLD

FONT = ('font-family="IBM Plex Sans, -apple-system, Segoe UI, sans-serif"')
MONO = ('font-family="IBM Plex Mono, SFMono-Regular, Consolas, monospace"')


def _esc(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _text(x, y, content, size=12, fill=INK, anchor="start", weight="400", mono=False):
    face = MONO if mono else FONT
    return (f'<text x="{x:.1f}" y="{y:.1f}" {face} font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{_esc(content)}</text>')


def _svg(width, height, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" role="img" aria-label="{_esc(title)}">\n'
            f'{body}\n</svg>\n')


# --- data ------------------------------------------------------------------

def load_quality_by_difficulty():
    """Difficulty is a property of the query set, not the responses table, so
    join through data/eval_queries.json by query_id (same as eval_summary.py)."""
    import json
    queries_path = Path(__file__).resolve().parent.parent / "data" / "eval_queries.json"
    difficulty_of = {q["id"]: q["difficulty"] for q in json.loads(queries_path.read_text())}

    with get_conn() as conn:
        out = {}
        for tier in TIERS:
            buckets = {d: [] for d in DIFFICULTIES}
            rows = conn.execute(
                "SELECT query_id, score FROM eval_scores "
                "WHERE tier=? AND source IN ('automatic','judge_claude')",
                (tier,),
            ).fetchall()
            for query_id, score in rows:
                difficulty = difficulty_of.get(query_id)
                if difficulty in buckets:
                    buckets[difficulty].append(score)
            out[tier] = {d: (sum(v) / len(v) if v else None) for d, v in buckets.items()}
        return out


def load_tier_summary():
    with get_conn() as conn:
        out = {}
        for tier in TIERS:
            scores = conn.execute(
                "SELECT score FROM eval_scores WHERE tier=? AND source IN ('automatic','judge_claude')",
                (tier,),
            ).fetchall()
            cost, latency = conn.execute(
                "SELECT AVG(cost), AVG(latency_ms) FROM eval_responses WHERE tier=?", (tier,),
            ).fetchone()
            out[tier] = {
                "quality": sum(s[0] for s in scores) / len(scores),
                "cost": cost, "latency": latency, "n": len(scores),
            }
        return out


def load_cascade_totals():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT total_cost, tokens_in, tokens_out, final_tier, escalated FROM cascade_log",
        ).fetchall()
    actual = sum(r[0] for r in rows)
    baseline = sum(estimate_frontier_cost(r[1], r[2]) for r in rows)
    by_tier = {t: sum(1 for r in rows if r[3] == t) for t in TIERS}
    return {
        "n": len(rows), "actual": actual, "baseline": baseline,
        "by_tier": by_tier, "escalated": sum(1 for r in rows if r[4]),
    }


# --- charts ----------------------------------------------------------------

def chart_quality_by_difficulty(data, path):
    W, H = 720, 390
    ml, mr, mt, mb = 54, 150, 68, 54
    plot_w, plot_h = W - ml - mr, H - mt - mb
    y0, y1 = 0.0, 1.0

    def sy(v):
        return mt + plot_h - (v - y0) / (y1 - y0) * plot_h

    parts = [_text(0, 22, "Answer quality by question difficulty", 15, INK_STRONG, weight="600")]
    parts.append(_text(0, 38, "115-query eval set - the cheap tier holds up until questions get hard", 11.5, INK))

    for gv in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = sy(gv)
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+plot_w}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(_text(ml - 9, y + 4, f"{gv:.1f}", 11, INK, anchor="end", mono=True))

    # The line that decides routing: a tier below it doesn't get that difficulty.
    ty = sy(THRESHOLD)
    parts.append(f'<line x1="{ml}" y1="{ty:.1f}" x2="{ml+plot_w}" y2="{ty:.1f}" '
                 f'stroke="#e0a020" stroke-width="1.5" stroke-dasharray="5 4"/>')
    parts.append(_text(ml + plot_w + 8, ty + 4, "0.80 routing", 11, "#e0a020", weight="600"))
    parts.append(_text(ml + plot_w + 8, ty + 17, "threshold", 11, "#e0a020", weight="600"))

    group_w = plot_w / len(DIFFICULTIES)
    bar_w = group_w / (len(TIERS) + 1.4)
    for gi, difficulty in enumerate(DIFFICULTIES):
        gx = ml + gi * group_w
        for ti, tier in enumerate(TIERS):
            v = data[tier][difficulty]
            if v is None:
                continue
            x = gx + group_w / 2 - (len(TIERS) * bar_w) / 2 + ti * bar_w
            y = sy(v)
            h = mt + plot_h - y
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-3:.1f}" height="{h:.1f}" '
                         f'rx="2" fill="{COLORS[tier]}"/>')
            parts.append(_text(x + (bar_w - 3) / 2, y - 6, f"{v:.2f}", 10.5,
                               INK_STRONG, anchor="middle", mono=True))
        parts.append(_text(gx + group_w / 2, mt + plot_h + 22, difficulty, 12.5, INK_STRONG,
                           anchor="middle", weight="600"))

    parts.append(f'<line x1="{ml}" y1="{mt+plot_h}" x2="{ml+plot_w}" y2="{mt+plot_h}" '
                 f'stroke="{INK}" stroke-width="1"/>')

    # Legend sits below the threshold label so the two never collide.
    lx, ly = ml + plot_w + 20, ty + 48
    for tier in TIERS:
        parts.append(f'<rect x="{lx}" y="{ly-9}" width="11" height="11" rx="2" fill="{COLORS[tier]}"/>')
        parts.append(_text(lx + 17, ly, tier, 12, INK_STRONG))
        ly += 21

    path.write_text(_svg(W, H, "\n".join(parts), "Answer quality by question difficulty"))


def chart_cost_quality(summary, path):
    W, H = 720, 370
    ml, mr, mt, mb = 62, 150, 58, 54
    plot_w, plot_h = W - ml - mr, H - mt - mb
    max_cost = max(s["cost"] for s in summary.values()) * 1.18
    y0, y1 = 0.70, 1.02

    def sx(v):
        return ml + (v / max_cost) * plot_w

    def sy(v):
        return mt + plot_h - (v - y0) / (y1 - y0) * plot_h

    parts = [_text(0, 22, "Cost against quality, per tier", 15, INK_STRONG, weight="600")]
    parts.append(_text(0, 38, "Average per query across 115 queries - up and to the left is better", 11.5, INK))

    for gv in [0.7, 0.8, 0.9, 1.0]:
        y = sy(gv)
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+plot_w}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(_text(ml - 9, y + 4, f"{gv:.1f}", 11, INK, anchor="end", mono=True))

    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        cost = max_cost * frac
        x = sx(cost)
        parts.append(_text(x, mt + plot_h + 20, f"${cost:.5f}", 10, INK, anchor="middle", mono=True))

    for tier, s in summary.items():
        x, y = sx(s["cost"]), sy(s["quality"])
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{COLORS[tier]}"/>')
        parts.append(_text(x, y - 14, tier, 12, INK_STRONG, anchor="middle", weight="600"))

    parts.append(f'<line x1="{ml}" y1="{mt+plot_h}" x2="{ml+plot_w}" y2="{mt+plot_h}" '
                 f'stroke="{INK}" stroke-width="1"/>')
    parts.append(_text(ml + plot_w / 2, H - 12, "cost per query (USD)", 11.5, INK, anchor="middle"))
    parts.append(_text(-(mt + plot_h / 2), 16, "quality", 11.5, INK, anchor="middle") .replace(
        "<text ", '<text transform="rotate(-90)" '))

    lx, ly = ml + plot_w + 20, mt + 30
    for tier, s in summary.items():
        parts.append(f'<circle cx="{lx+5}" cy="{ly-4}" r="5" fill="{COLORS[tier]}"/>')
        parts.append(_text(lx + 17, ly, tier, 12, INK_STRONG, weight="600"))
        parts.append(_text(lx + 17, ly + 14, f"{s['latency']/1000:.1f}s per query", 10.5, INK, mono=True))
        ly += 38

    path.write_text(_svg(W, H, "\n".join(parts), "Cost against quality per tier"))


def chart_cascade_savings(totals, path):
    W, H = 720, 210
    ml, mr, mt = 150, 120, 56
    bar_h, gap = 34, 20
    plot_w = W - ml - mr
    max_v = totals["baseline"] * 1.05

    saved_pct = (totals["baseline"] - totals["actual"]) / totals["baseline"] * 100
    parts = [_text(0, 22, "What the cascade actually cost", 15, INK_STRONG, weight="600")]
    parts.append(_text(0, 38, f"{totals['n']} routed queries, measured - versus sending every one to frontier",
                       11.5, INK))

    rows = [
        ("frontier-only", totals["baseline"], COLORS["frontier"]),
        ("cascade", totals["actual"], COLORS["cheap"]),
    ]
    for i, (label, value, color) in enumerate(rows):
        y = mt + i * (bar_h + gap)
        w = (value / max_v) * plot_w
        parts.append(f'<rect x="{ml}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="3" fill="{color}"/>')
        parts.append(_text(ml - 12, y + bar_h / 2 + 4, label, 12.5, INK_STRONG, anchor="end", weight="600"))
        parts.append(_text(ml + w + 10, y + bar_h / 2 + 4, f"${value:.5f}", 12, INK_STRONG, mono=True))

    y_note = mt + len(rows) * (bar_h + gap) + 18
    parts.append(_text(ml, y_note, f"{saved_pct:.0f}% cheaper", 14, COLORS["cheap"], weight="700"))
    routed = "  ".join(f"{t}: {n}" for t, n in totals["by_tier"].items())
    parts.append(_text(ml, y_note + 20, f"routed - {routed}   |   escalated: {totals['escalated']}",
                       11, INK, mono=True))

    path.write_text(_svg(W, H, "\n".join(parts), "Cascade cost versus frontier-only"))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_difficulty = load_quality_by_difficulty()
    summary = load_tier_summary()
    totals = load_cascade_totals()

    chart_quality_by_difficulty(by_difficulty, OUT_DIR / "quality-by-difficulty.svg")
    chart_cost_quality(summary, OUT_DIR / "cost-vs-quality.svg")
    chart_cascade_savings(totals, OUT_DIR / "cascade-savings.svg")

    print(f"wrote 3 charts to {OUT_DIR}")
    for tier, s in summary.items():
        print(f"  {tier:9} quality={s['quality']:.3f} cost=${s['cost']:.5f} latency={s['latency']:.0f}ms")
    print(f"  cascade  {totals['n']} runs  ${totals['actual']:.5f} vs ${totals['baseline']:.5f} baseline")


if __name__ == "__main__":
    main()
