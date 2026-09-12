"""Single source of truth for the three model tiers, unified via LiteLLM's Router.

Swapping/adding a tier later = edit TIER_MODEL_LIST, nothing else changes.
"""
import os

from dotenv import load_dotenv
from litellm import Router

load_dotenv()

TIER_MODEL_LIST = [
    {
        "model_name": "cheap",
        "litellm_params": {
            "model": "ollama/llama3.2:3b",
            "api_base": os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
        },
    },
    {
        "model_name": "mid",
        "litellm_params": {
            # Groq's GPT-OSS-20B. See https://console.groq.com/docs/models
            "model": "groq/openai/gpt-oss-20b",
            "api_key": os.getenv("GROQ_API_KEY"),
            # Without this, long answers were silently truncating mid-sentence
            # (observed on 4 of the longest Week 2 eval queries).
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

# Week 1: router service is a single-tier passthrough. This is the one line
# that changes in Week 3 when real routing logic replaces the hardcode.
DEFAULT_TIER = "frontier"
