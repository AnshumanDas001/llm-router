"""Single source of truth for the three model tiers, unified via LiteLLM's Router.

Swapping/adding a tier later = edit TIER_MODEL_LIST, nothing else changes.
"""
import json
import os

from dotenv import load_dotenv
from litellm import Router

from app.paths import BUILTIN_CALIBRATION as BUILTIN_CALIBRATION_PATH

load_dotenv()

# Every tier is env-overridable with any litellm model string; provider keys
# come from the usual env vars (OPENROUTER_API_KEY, GROQ_API_KEY, ...), which
# litellm reads on its own. The defaults are the measured built-in stack --
# see README "Strategy comparison" for why these four and not the first
# four that were tried. A local Ollama model works as the cheap tier too
# (CHEAP_MODEL=ollama/llama3.2:3b); the routing map is re-derived from
# calibration either way.
CHEAP_MODEL = os.getenv("CHEAP_MODEL", "openrouter/meta-llama/llama-3.1-8b-instruct")


def _cheap_params() -> dict:
    if CHEAP_MODEL.startswith("ollama/"):
        # Talk to Ollama through its OpenAI-compatible endpoint rather than
        # litellm's native ollama provider: the native one refuses to forward
        # `logprobs`, and the learned verifier reads the cheap model's token
        # confidence off exactly that. Cost stays $0 either way (litellm has
        # no price for the model and the app treats unknown as free).
        base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434").rstrip("/")
        return {"model": "openai/" + CHEAP_MODEL.split("/", 1)[1],
                "api_base": base + "/v1", "api_key": "ollama"}
    if CHEAP_MODEL.startswith("openrouter/"):
        # OpenRouter picks a backend per call, and not every backend honours
        # `logprobs` (DeepInfra silently dropped them; Novita returns them).
        # require_parameters restricts routing to backends that support every
        # parameter we send, so the learned verifier gets its signals.
        return {"model": CHEAP_MODEL, "extra_body": {"provider": {"require_parameters": True}}}
    return {"model": CHEAP_MODEL}


# How the cheapest tier's answer is checked before it ships:
#   auto     (default) the learned gate in app/routing/scorer.py runs first: it reads
#            the cheap model's own token confidence, and where it is sure the
#            answer is right (P >= its accept threshold, measured so that no
#            wrong answer slipped through) the answer ships with no judge
#            call at all. Everything else still goes to the judge. Falls back
#            to the judge whenever there is no scorer trained for the current
#            CHEAP_MODEL, or the provider returned no logprobs.
#   judge    always call the judge; never consult the scorer.
#   learned  same as auto (kept as an explicit spelling).
VERIFIER = os.getenv("VERIFIER", "auto")


# Every tier is env-overridable with any litellm model string, so a deploy
# can run a different stack without touching source. Provider keys are read
# by litellm from the standard env vars (GROQ_API_KEY, GEMINI_API_KEY,
# OPENROUTER_API_KEY, ...), so nothing here needs to pass one explicitly.
#
# Notes on the choices:
#   mid      gemini-3.5-flash through OpenRouter (Google's own free tier is
#            capped at 20 requests/day). Runs with thinking off, below.
#   frontier the same model with thinking ON. On everyday and BBH-style
#            questions it buys almost nothing (mid is at 96%); on AIME
#            competition maths mid falls to 70% and thinking rescued 2 of
#            its 3 misses -- that band is what this tier is for (README,
#            "Does the frontier tier earn its slot?"). deepseek-r1 held the
#            slot before and lost outright: it rescued none of the ones
#            tested, cost 3x, and averaged 467s per answer, which no
#            interactive request can absorb.
#
# Two tiers sharing a model is not a mistake: what separates them is whether
# the model is allowed to think, which is the axis that actually predicted
# success here. It does mean their per-token list price is identical, so
# cost comparisons must come from calibration (which measures the reasoning
# tokens) rather than from a price-times-tokens estimate.
MID_MODEL = os.getenv("MID_MODEL", "openrouter/google/gemini-3.5-flash")
FRONTIER_MODEL = os.getenv("FRONTIER_MODEL", "openrouter/google/gemini-3.5-flash")

# Reasoning effort for the mid tier's own answers. A thinking model as mid
# is a trap without this: gemini-3.5-flash spent 720 reasoning tokens on a
# two-sentence answer -- 94% of a $0.0069 bill -- and "minimal" gave the
# same answer for $0.0005. Values are provider-specific ("minimal" is
# Gemini; Groq's gpt-oss takes low/medium/high), so it's per-stack config.
# Unset = provider default. Thinking is what the frontier tier is for.
MID_REASONING_EFFORT = os.getenv("MID_REASONING_EFFORT", "minimal") or None


# Reasoning effort for the frontier tier. Unlike mid, the frontier exists
# *because* it thinks: on AIME problems it spent 13-31k reasoning tokens
# ($0.13-0.29) and got two answers right that mid, writing its working out
# in ~3k visible tokens, got wrong. Left unset it inherits the provider default.
FRONTIER_REASONING_EFFORT = os.getenv("FRONTIER_REASONING_EFFORT", "high") or None


def _frontier_params() -> dict:
    params = {"model": FRONTIER_MODEL, "max_tokens": 32000, "drop_params": True}
    if FRONTIER_REASONING_EFFORT:
        params["reasoning_effort"] = FRONTIER_REASONING_EFFORT
    return params


def _mid_params() -> dict:
    params = {
        "model": MID_MODEL,
        # Without this, long answers were silently truncating mid-sentence
        # (observed on 4 of the longest eval queries).
        "max_tokens": 4096,
    }
    if MID_REASONING_EFFORT:
        params["reasoning_effort"] = MID_REASONING_EFFORT
        params["drop_params"] = True
    return params


# The model that judges the cheapest tier's answers on the built-in stack.
# Unset = the tier above it, which is the natural choice until the mid tier
# is expensive: a judge verdict reads ~450 tokens and writes one word, so
# a small model does it well (gpt-oss-20b caught 94% of wrong cheap answers)
# and the cascade's whole margin on easy/medium questions is the gap between
# what a verdict costs and what a mid answer costs. With the judge tied to
# mid that gap scales with mid's price and the cascade never wins; with a
# $0.00004 judge in front of a $0.001 answer it does. BYOM stacks still use
# their own next tier up.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "groq/openai/gpt-oss-20b") or None

TIER_MODEL_LIST = [
    {"model_name": "cheap", "litellm_params": _cheap_params()},
    {"model_name": "mid", "litellm_params": _mid_params()},
    # A reasoning frontier needs real headroom, and the cap must be generous
    # because providers bill tokens *used*, not tokens allowed. 16000 was not
    # enough: R1 spent 13,697 tokens thinking and returned an EMPTY answer,
    # which reads downstream as a wrong answer -- it silently failed 3 of 12
    # calibration questions that way.
    {"model_name": "frontier", "litellm_params": _frontier_params()},
]

# Per-tier extra call parameters, for code paths that call litellm directly
# rather than through the Router (calibration, the eval harness) and must
# measure the tier as it actually runs.
TIER_CALL_PARAMS = {t["model_name"]: {k: v for k, v in t["litellm_params"].items() if k != "model"}
                    for t in TIER_MODEL_LIST}

router = Router(model_list=TIER_MODEL_LIST)



# What the routing policy needs to know about the built-in tiers, from
# scripts/calibration/calibrate_builtin.py: easy/medium/hard on the eval set
# (cheap: all 76 auto-gradable queries; mid and frontier: 24), expert from
# the AIME probe (data/probe/expert_queries_results.jsonl). Frontier's
# expert quality is None: it was only run on the questions mid got wrong,
# which measures what it rescues, not how good it is -- and as the last
# tier its quality never gates routing anyway.
#
# data/calibration/builtin.json is the live copy and wins when its model
# names match the configured stack; this is the fallback.
_BUILTIN_CALIBRATION_DEFAULT = {
    "cheap": {
        "quality": {"easy": 0.895, "medium": 0.950, "hard": 0.811, "expert": 0.0},
        "cost":    {"easy": 0.000003, "medium": 0.000013, "hard": 0.000025, "expert": 0.00019},
    },
    "mid": {
        "quality": {"easy": 1.000, "medium": 1.000, "hard": 1.000, "expert": 0.70},
        "cost":    {"easy": 0.000477, "medium": 0.002386, "hard": 0.002325, "expert": 0.0193},
    },
    "frontier": {
        "quality": {"easy": 1.000, "medium": 1.000, "hard": 1.000, "expert": None},
        "cost":    {"easy": 0.002252, "medium": 0.008738, "hard": 0.008724, "expert": 0.1939},
    },
}


def _load_builtin_calibration() -> dict:
    if BUILTIN_CALIBRATION_PATH.exists():
        data = json.loads(BUILTIN_CALIBRATION_PATH.read_text())
        # Only trust it for the stack it was measured on.
        current = {"cheap": CHEAP_MODEL, "mid": MID_MODEL, "frontier": FRONTIER_MODEL}
        if data.get("models") == current:
            return data["tiers"]
    return _BUILTIN_CALIBRATION_DEFAULT


BUILTIN_CALIBRATION = _load_builtin_calibration()
