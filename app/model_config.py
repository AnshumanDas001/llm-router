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
            # Gemini Flash-Lite via Google AI Studio free tier: ~1000 req/day, 15 RPM.
            # (gemini-3.6-flash is newer/preview and capped at only 20 req/day on free tier.)
            # Check https://ai.google.dev/gemini-api/docs/models for current model ids.
            "model": "gemini/gemini-3.5-flash-lite",
            "api_key": os.getenv("GEMINI_API_KEY"),
        },
    },
]

router = Router(model_list=TIER_MODEL_LIST)

