"""Turns calibration numbers into an actual routing decision.

Before this, calibration was inert: it measured each tier's quality and
then only displayed it. Every user -- including BYOM users running models
we've never seen -- routed by one hardcoded map in app/classifier.py
(easy->cheap, medium->cheap, hard->mid), which encodes *our* measurements
of *our* stack. For someone whose "cheap" tier is a frontier-class model,
that map wastes money; for someone whose "cheap" tier is weaker than ours,
it burns a doomed call before escalating.

Two rules, applied per difficulty band:

1. A quality floor. A tier may only *start* a difficulty if its measured
   quality on that band clears QUALITY_THRESHOLD. Below it, too many
   answers are wrong for the judge to be trusted to catch them all.

2. Expected cost, among the tiers that clear the floor. Starting at a
   tier costs its generation, plus its judge call if it's the cheapest
   configured tier, plus -- weighted by how often it fails -- everything
   the escalation then costs. Pick the start with the lowest expected
   total.

Rule 2 is what stops a cascade from losing to its own mid tier. On a stack
whose mid model is already very cheap, the cheap tier's judge call costs
about what mid's own short answer does, so "cheap first" on easy questions
actually loses money -- measured: $0.00084 spent to save $0.00050. Pricing
the whole path makes the policy route around a cheap tier that can't pay
for its verification, and route *to* it on a stack where mid is expensive
enough that it can. The router discovers which stack it's on.
"""
from app.classifier import DIFFICULTY_TO_TIER

TIER_ORDER = ["cheap", "mid", "frontier"]
DIFFICULTIES = ("easy", "medium", "hard")

# The eval set's own numbers for our cheap tier: 0.966 easy / 0.852 medium
# / 0.638 hard, and the call was "hard skips cheap". Any threshold between
# 0.638 and 0.852 reproduces that decision; 0.80 sits inside the band.
QUALITY_THRESHOLD = 0.80


def expected_costs(tiers: list[str], quality_by_tier: dict, cost_by_tier: dict,
                   judge_cost: float, difficulty: str) -> dict:
    """E[total cost | start at tier] for one difficulty, for every tier.

    Solved back to front: the last tier is trusted unconditionally, so its
    expected cost is just its generation cost. Each earlier tier adds its
    own generation, its judge if it's the cheapest configured tier (the
    cascade only pays for a real judge there), and its failure-weighted
    share of whatever the next tier costs.

    A tier with no measured quality on this band is treated as failing
    always -- it can't be a start tier anyway, and this keeps it from
    making the tier below look cheaper than it is.
    """
    costs = {}
    following = None
    for i, tier in reversed(list(enumerate(tiers))):
        gen = (cost_by_tier.get(tier) or {}).get(difficulty) or 0.0
        if following is None:
            costs[tier] = gen
        else:
            quality = (quality_by_tier.get(tier) or {}).get(difficulty)
            p_fail = 1.0 - quality if quality is not None else 1.0
            judge = judge_cost if i == 0 and len(tiers) > 1 else 0.0
            costs[tier] = gen + judge + p_fail * following
        following = costs[tier]
    return costs


def derive_tier_map(quality_by_tier: dict, configured_tiers: list[str] | None = None,
                    cost_by_tier: dict | None = None, judge_cost: float = 0.0) -> dict:
    """quality_by_tier: {tier: {difficulty: score|None}} from calibration.
    cost_by_tier:      {tier: {difficulty: avg cost per answer}} from
                       calibration -- per band, because a mid model's easy
                       answers can cost 7x less than its overall mean. If
                       omitted, only the quality floor applies (cheapest
                       tier that clears it wins), which is the pre-cost rule.

    Returns {difficulty: tier} covering all three difficulties. Falls back
    to the built-in map when there's nothing measured to go on, so an
    uncalibrated account behaves exactly as it did before.
    """
    tiers = [t for t in TIER_ORDER if t in (configured_tiers or quality_by_tier.keys())]
    if not tiers:
        return dict(DIFFICULTY_TO_TIER)

    tier_map = {}
    for difficulty in DIFFICULTIES:
        eligible = [
            t for t in tiers
            if ((quality_by_tier.get(t) or {}).get(difficulty) or 0.0) >= QUALITY_THRESHOLD
        ]
        # The strongest configured tier is always a legal start: if nothing
        # clears the bar the cascade has nowhere better to begin, and it can
        # only escalate upward from wherever it starts.
        if tiers[-1] not in eligible:
            eligible.append(tiers[-1])

        if cost_by_tier is None:
            tier_map[difficulty] = eligible[0]
            continue

        costs = expected_costs(tiers, quality_by_tier, cost_by_tier, judge_cost, difficulty)
        # Ties go to the stronger tier: same money, fewer escalations, and
        # no waiting on a verdict.
        tier_map[difficulty] = min(eligible, key=lambda t: (round(costs[t], 9), -tiers.index(t)))
    return tier_map


def describe_tier_map(tier_map: dict) -> list[str]:
    """Human-readable lines for the Settings page, so a user can see what
    their calibration actually bought them."""
    return [f"{difficulty} -> {tier_map[difficulty]}" for difficulty in DIFFICULTIES]
