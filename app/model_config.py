"""Single source of truth for the three model tiers, unified via LiteLLM's Router.

Swapping/adding a tier later = edit TIER_MODEL_LIST, nothing else changes.
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from litellm import Router

load_dotenv()

# The cheap tier is the one most likely to change between environments: a
# local Ollama model on a laptop, but on a cloud box a hosted small model is
# usually faster and the calibration re-derives the routing map either way.
# CHEAP_MODEL takes any litellm string; provider keys come from the usual
# env vars (GROQ_API_KEY etc.), which litellm reads on its own.
CHEAP_MODEL = os.getenv("CHEAP_MODEL", "ollama/llama3.2:3b")


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
    return {"model": CHEAP_MODEL}


# How the cheapest tier's answer is checked before it ships:
#   learned  the scorer in app/scorer.py (token confidence + a second cheap
#            sample + cheap self-check). No judge call. Requires the trained
#            model file; falls back to the judge if it's missing or the
#            provider returns no logprobs.
#   judge    the LLM judge on the next tier up, always.
#   auto     learned when trained, judge otherwise (default).
VERIFIER = os.getenv("VERIFIER", "auto")


# Every tier is env-overridable with any litellm model string, so a deploy
# can run a different stack without touching source. Provider keys are read
# by litellm from the standard env vars (GROQ_API_KEY, GEMINI_API_KEY,
# OPENROUTER_API_KEY, ...), so nothing here needs to pass one explicitly.
#
# Defaults are the measured built-in stack. Notes on the choices:
#   mid      gpt-oss-20b on Groq. The same model via OpenRouter
#            (openrouter/openai/gpt-oss-20b) prices ~2.3x lower per answer
#            and makes the judge call 2.5x cheaper.
#   frontier gemini-3.5-flash replaced flash-lite, which the eval showed was
#            *dominated* by mid (lower quality, 4.7x cost, 2x latency), so
#            escalating to it bought a worse answer for more money.
MID_MODEL = os.getenv("MID_MODEL", "groq/openai/gpt-oss-20b")
FRONTIER_MODEL = os.getenv("FRONTIER_MODEL", "gemini/gemini-3.5-flash")

TIER_MODEL_LIST = [
    {"model_name": "cheap", "litellm_params": _cheap_params()},
    {
        "model_name": "mid",
        "litellm_params": {
            "model": MID_MODEL,
            # Without this, long answers were silently truncating mid-sentence
            # (observed on 4 of the longest eval queries).
            "max_tokens": 4096,
        },
    },
    {"model_name": "frontier", "litellm_params": {"model": FRONTIER_MODEL}},
]

router = Router(model_list=TIER_MODEL_LIST)



# What the routing policy needs to know about the built-in tiers, measured
# on the 115-query eval set (scripts/eval_summary.py, plus per-band cost from
# eval_responses). BYOM tiers get the same numbers from calibration; these
# are the built-in stack's equivalent, stated once here because the eval
# database isn't shipped with the app.
#
# The frontier row is computed, not measured: gemini-3.5-flash priced at the
# token counts the previous frontier model produced per band. Its quality is
# not gated (the top tier is always a legal start), so only its cost matters
# for routing, and only as the escalation target.
_BUILTIN_CALIBRATION_DEFAULT = {
    "cheap": {
        "quality": {"easy": 0.966, "medium": 0.852, "hard": 0.638},
        "cost":    {"easy": 0.0,   "medium": 0.0,   "hard": 0.0},     # local Ollama
    },
    "mid": {
        "quality": {"easy": 1.000, "medium": 0.991, "hard": 1.000},
        "cost":    {"easy": 0.000053, "medium": 0.000107, "hard": 0.000270},
    },
    "frontier": {
        "quality": {"easy": 1.000, "medium": 1.000, "hard": 0.983},   # flash-lite's; not gated
        "cost":    {"easy": 0.000363, "medium": 0.002579, "hard": 0.004325},
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
