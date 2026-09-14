"""Single source of truth for the three model tiers, unified via LiteLLM's Router.

Swapping/adding a tier later = edit TIER_MODEL_LIST, nothing else changes.
"""
import os

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
    params = {"model": CHEAP_MODEL}
    if CHEAP_MODEL.startswith("ollama/"):
        params["api_base"] = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    return params


TIER_MODEL_LIST = [
    {
        "model_name": "cheap",
        "litellm_params": _cheap_params(),
    },
    {
        "model_name": "mid",
        "litellm_params": {
            # Groq's GPT-OSS-20B. See https://console.groq.com/docs/models
            "model": "groq/openai/gpt-oss-20b",
            "api_key": os.getenv("GROQ_API_KEY"),
            # Without this, long answers were silently truncating mid-sentence
            # (observed on 4 of the longest eval queries).
            "max_tokens": 4096,
        },
    },
    {
        "model_name": "frontier",
        "litellm_params": {
            # Gemini 3.5 Flash via Google AI Studio. This replaced Flash-Lite,
            # which the eval showed was *dominated* by the mid tier -- lower
            # quality (0.991 vs 0.997), 4.7x the cost, 2x the latency -- so
            # escalating to it bought a worse answer for more money. A
            # frontier tier only earns its place by being stronger than mid;
            # ~4x mid's price is the cost of that, and escalation is rare.
            # Check https://ai.google.dev/gemini-api/docs/models for current ids.
            "model": os.getenv("FRONTIER_MODEL", "gemini/gemini-3.5-flash"),
            "api_key": os.getenv("GEMINI_API_KEY"),
        },
    },
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
BUILTIN_CALIBRATION = {
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
