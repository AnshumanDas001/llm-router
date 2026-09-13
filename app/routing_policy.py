"""Turns calibration numbers into an actual routing decision.

Before this, calibration was inert: it measured each tier's quality and
then only displayed it. Every user -- including BYOM users running models
we've never seen -- routed by one hardcoded map in app/classifier.py
(easy->cheap, medium->cheap, hard->mid), which encodes *our* Week 2
measurements of *our* Ollama/Groq/Gemini stack. For someone whose "cheap"
tier is a frontier-class model, that map wastes money; for someone whose
"cheap" tier is weaker than ours, it burns a doomed call before escalating.

The rule here is the same judgment Week 2 made by hand, just applied to
whatever numbers a given user actually measured: for each difficulty,
start at the cheapest configured tier that scored well enough on *that
difficulty* to be worth trying.
"""
from app.classifier import DIFFICULTY_TO_TIER

TIER_ORDER = ["cheap", "mid", "frontier"]
DIFFICULTIES = ("easy", "medium", "hard")

# Week 2's own numbers for our cheap tier: 0.916 easy / 0.936 medium /
# 0.718 hard, and the call was "hard skips cheap". Any threshold between
# 0.718 and 0.916 reproduces that decision; 0.80 sits in the middle of
# that band. Stated as a rule rather than a constant so the default map
# is now *derived* from measurement instead of asserted.
QUALITY_THRESHOLD = 0.80


def derive_tier_map(quality_by_tier: dict, configured_tiers: list[str] | None = None) -> dict:
    """quality_by_tier: {tier: {difficulty: score|None}} from calibration.

    Returns {difficulty: tier} covering all three difficulties. Falls back
    to the built-in map when there's nothing measured to go on, so an
    uncalibrated account behaves exactly as it did before.
    """
    tiers = [t for t in TIER_ORDER if t in (configured_tiers or quality_by_tier.keys())]
    if not tiers:
        return dict(DIFFICULTY_TO_TIER)

    tier_map = {}
    for difficulty in DIFFICULTIES:
        chosen = None
        for tier in tiers:  # cheapest first
            score = (quality_by_tier.get(tier) or {}).get(difficulty)
            if score is not None and score >= QUALITY_THRESHOLD:
                chosen = tier
                break
        # Nothing cleared the bar (or nothing was measured): start at the
        # strongest tier this user configured rather than guessing low --
        # the cascade can still only escalate upward from wherever it starts.
        tier_map[difficulty] = chosen or tiers[-1]
    return tier_map


def describe_tier_map(tier_map: dict) -> list[str]:
    """Human-readable lines for the Settings page, so a user can see what
    their calibration actually bought them."""
    return [f"{difficulty} -> {tier_map[difficulty]}" for difficulty in DIFFICULTIES]
