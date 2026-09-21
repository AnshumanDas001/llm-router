"""Single source of truth for the three model tiers, unified via LiteLLM's Router.

Swapping/adding a tier later = edit TIER_MODEL_LIST, nothing else changes.
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from litellm import Router

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
#   learned  the scorer in app/scorer.py (token confidence + a second cheap
#            sample + cheap self-check). No judge call. Requires the trained
#            model file; falls back to the judge if it's missing or the
#            provider returns no logprobs.
#   judge    the LLM judge on the next tier up, always.
#   auto     learned when trained, judge otherwise (default).
VERIFIER = os.getenv("VERIFIER", "judge")


# Every tier is env-overridable with any litellm model string, so a deploy
# can run a different stack without touching source. Provider keys are read
# by litellm from the standard env vars (GROQ_API_KEY, GEMINI_API_KEY,
# OPENROUTER_API_KEY, ...), so nothing here needs to pass one explicitly.
#
# Notes on the choices:
#   mid      gemini-3.5-flash through OpenRouter (Google's own free tier is
#            capped at 20 requests/day). Runs with thinking off, below.
#   frontier deepseek-r1: only reached when mid fails its structural check
#            or is rate-limited. Reasons for thousands of tokens; ~3x mid's
#            price and a minute or more per answer.
MID_MODEL = os.getenv("MID_MODEL", "openrouter/google/gemini-3.5-flash")
FRONTIER_MODEL = os.getenv("FRONTIER_MODEL", "openrouter/deepseek/deepseek-r1")

# Reasoning effort for the mid tier's own answers. A thinking model as mid
# is a trap without this: gemini-3.5-flash spent 720 reasoning tokens on a
# two-sentence answer -- 94% of a $0.0069 bill -- and "minimal" gave the
# same answer for $0.0005. Values are provider-specific ("minimal" is
# Gemini; Groq's gpt-oss takes low/medium/high), so it's per-stack config.
# Unset = provider default. Thinking is what the frontier tier is for.
MID_REASONING_EFFORT = os.getenv("MID_REASONING_EFFORT", "minimal") or None


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
    # A reasoning frontier needs headroom: DeepSeek R1 spent 3,400 thinking
    # tokens on a 330-token answer and, at the provider default, sometimes
    # ran out before saying anything.
    {"model_name": "frontier", "litellm_params": {"model": FRONTIER_MODEL, "max_tokens": 16000}},
]

# Per-tier extra call parameters, for code paths that call litellm directly
# rather than through the Router (calibration, the eval harness) and must
# measure the tier as it actually runs.
TIER_CALL_PARAMS = {t["model_name"]: {k: v for k, v in t["litellm_params"].items() if k != "model"}
                    for t in TIER_MODEL_LIST}

router = Router(model_list=TIER_MODEL_LIST)



# What the routing policy needs to know about the built-in tiers, from
# scripts/calibrate_builtin.py on the eval set (cheap: all 76 auto-gradable
# queries; mid: 24; frontier: 9). data/builtin_calibration.json is the live
# copy and wins when its model names match; this is the fallback.
_BUILTIN_CALIBRATION_DEFAULT = {
    "cheap": {
        "quality": {"easy": 0.895, "medium": 0.950, "hard": 0.811},
        "cost":    {"easy": 0.000003, "medium": 0.000013, "hard": 0.000025},
    },
    "mid": {
        "quality": {"easy": 1.000, "medium": 1.000, "hard": 1.000},
        "cost":    {"easy": 0.000445, "medium": 0.002394, "hard": 0.002123},
    },
    "frontier": {
        "quality": {"easy": 1.000, "medium": 1.000, "hard": 1.000},
        "cost":    {"easy": 0.002516, "medium": 0.008607, "hard": 0.005965},
    },
}


# scripts/calibrate_builtin.py measures whatever stack CHEAP/MID/FRONTIER_MODEL
# point at and writes the result here; when that file exists it wins over the
# constants above, which only describe the default stack.
BUILTIN_CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "data" / "builtin_calibration.json"


def _load_builtin_calibration() -> dict:
    if BUILTIN_CALIBRATION_PATH.exists():
        data = json.loads(BUILTIN_CALIBRATION_PATH.read_text())
        # Only trust it for the stack it was measured on.
        current = {"cheap": CHEAP_MODEL, "mid": MID_MODEL, "frontier": FRONTIER_MODEL}
        if data.get("models") == current:
            return data["tiers"]
    return _BUILTIN_CALIBRATION_DEFAULT


BUILTIN_CALIBRATION = _load_builtin_calibration()
