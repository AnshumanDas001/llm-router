"""The routing map is derived from calibration; these pin what it derives."""
from app.config import _BUILTIN_CALIBRATION_DEFAULT as CAL
from app.routing.policy import derive_tier_map

TIERS = ["cheap", "mid", "frontier"]


def _builtin(judge_cost=0.00004):
    return derive_tier_map({t: CAL[t]["quality"] for t in TIERS}, TIERS,
                           {t: CAL[t]["cost"] for t in TIERS}, judge_cost)


def test_builtin_stack_routes_expert_to_frontier():
    tier_map = _builtin()
    assert tier_map["expert"] == "frontier"
    assert tier_map["easy"] == "cheap"


def test_mid_below_floor_cannot_start_a_band():
    quality = {"cheap": {"expert": 0.0}, "mid": {"expert": 0.79}, "frontier": {"expert": None}}
    assert derive_tier_map(quality, TIERS)["expert"] == "frontier"
    quality["mid"]["expert"] = 0.85
    assert derive_tier_map(quality, TIERS)["expert"] == "mid"


def test_calibration_without_expert_band_falls_back_to_strongest_tier():
    """A BYOM calibration made before the expert band existed has no score
    for it, so nothing cheaper has been shown to handle those questions."""
    quality = {"cheap": {"easy": 0.9, "medium": 0.9, "hard": 0.5},
               "mid": {"easy": 1.0, "medium": 1.0, "hard": 0.9}}
    tier_map = derive_tier_map(quality, ["cheap", "mid"])
    assert tier_map == {"easy": "cheap", "medium": "cheap", "hard": "mid", "expert": "mid"}


def test_judge_cost_can_route_around_the_cheap_tier():
    """When a verdict costs about what mid's own answer does, starting at
    the cheap tier loses money even though it clears the floor."""
    quality = {"cheap": {"easy": 0.9}, "mid": {"easy": 1.0}}
    cost = {"cheap": {"easy": 0.00001}, "mid": {"easy": 0.0005}}
    assert derive_tier_map(quality, ["cheap", "mid"], cost, judge_cost=0.00001)["easy"] == "cheap"
    assert derive_tier_map(quality, ["cheap", "mid"], cost, judge_cost=0.0006)["easy"] == "mid"
