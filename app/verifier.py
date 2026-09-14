"""Verification layer for the cascade.

Two different verification strategies, chosen by *position* in the tier
sequence being tried, not the literal tier name -- this generalizes to a
BYOM user who configures a different subset/order of tiers than our own
cheap/mid/frontier:

- the FIRST (cheapest) tier's answer gets a paid LLM-judge completeness/
  coherence check, run on the SECOND tier. The eval set showed the cheapest tier
  has real, substantive failure modes (dropped sub-requests,
  self-contradictory reasoning), so this call earns its cost.
- any LATER tier's answer gets a free structural check only (non-empty,
  not a refusal). The eval set showed failure rates drop sharply after the first
  tier, and verification cost scales with response length -- a full
  LLM-judge call here was the dominant cost driver in the cascade (82.5%
  of total cost from "hard" queries alone) for very little real safety
  benefit. A free check still catches the genuinely broken case (empty
  output, a refusal) at zero marginal cost.
- the LAST tier in the sequence never gets checked: there's nothing left
  to escalate to, so verifying it can't change the outcome.
"""
import re
import time

import litellm

REFUSAL_PATTERN = re.compile(
    r"\b(i cannot|i can'?t|i'm unable|i am unable|as an ai)\b", re.IGNORECASE
)

VERIFY_PROMPT = """You are a fast quality-control check for an AI response, not a full grader.

Original request:
{query}

Candidate response:
{response}

Does this response address every explicit part of the request, and is it coherent and free of obvious factual or logical errors? Reply with exactly one line: "YES" or "NO: <one short reason>"."""


def _verify_structural(response_text: str) -> dict:
    """Free check: no model call, just pattern sanity checks.

    Deliberately no minimum-length check: a one-word answer like "Neutral"
    to a classification query is a completely valid, complete response, and
    an earlier length threshold (20 chars) flagged it as "suspiciously
    short," triggering a real, costly escalation for no quality reason.
    Empty-response and refusal-pattern checks catch genuine failures
    without penalizing legitimate terseness."""
    stripped = response_text.strip() if response_text else ""
    if not stripped:
        return {"passed": False, "reason": "empty response", "cost": 0.0,
                "latency_ms": 0.0, "verifier_tier": None}
    if REFUSAL_PATTERN.search(stripped[:200]):
        return {"passed": False, "reason": "looks like a refusal", "cost": 0.0,
                "latency_ms": 0.0, "verifier_tier": None}
    return {"passed": True, "reason": "structural check only (free, no LLM-judge call)",
            "cost": 0.0, "latency_ms": 0.0, "verifier_tier": None}


def _verify_llm_judge(query: str, response_text: str, judge_model: str,
                       judge_api_key: str | None, judge_tier_label: str) -> dict:
    """Paid check: ask a stronger tier for a completeness/coherence verdict."""
    prompt = VERIFY_PROMPT.format(query=query, response=response_text)
    start = time.perf_counter()
    result = litellm.completion(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        api_key=judge_api_key,
        # gpt-oss-20b (and other reasoning models) spend tokens on hidden
        # reasoning before the visible answer -- 60 was too tight and got
        # cut off (finish_reason "length", empty content) before ever
        # producing YES/NO.
        max_tokens=300,
        # Those hidden reasoning tokens are billed as output and were the
        # whole cost of the cheap tier: a measured $0.00016 per judge call
        # against $0.00005 nominal, enough that "just use mid" beat the
        # cascade outright. A YES/NO verdict on a short answer doesn't need
        # deliberation; low effort cut a test call from 108 output tokens
        # to 22 with the same verdict. drop_params lets non-reasoning judge
        # models ignore the flag instead of rejecting the request.
        reasoning_effort="low",
        drop_params=True,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    text = result.choices[0].message.content.strip()
    cost = result._hidden_params.get("response_cost", 0.0) or 0.0

    if not text:
        # Fail safe toward escalation rather than silently trusting an
        # unreadable verdict.
        return {"passed": False, "reason": "verifier returned no content (truncated?)",
                "cost": cost, "latency_ms": latency_ms, "verifier_tier": judge_tier_label}

    passed = text.upper().startswith("YES")
    return {"passed": passed, "reason": text, "cost": cost,
            "latency_ms": latency_ms, "verifier_tier": judge_tier_label}


def verify_response(query: str, response_text: str, tier: str,
                     tier_models: dict | None, tier_api_keys: dict | None) -> dict:
    """Returns {passed, reason, cost, latency_ms, verifier_tier}.

    `tier` is the tier that generated response_text. The caller (cascade.py)
    only invokes this for non-last tiers in the sequence, and only passes
    tier="cheap" (our default first tier) or, for BYOM, whichever tier is
    first in the user's configured sequence, when a real LLM-judge check is
    wanted; anything else gets the free structural check. This module
    doesn't need to know the full sequence -- cascade.py decides who the
    judge is and passes it in via judge_model/judge_api_key/judge_tier.
    """
    if not response_text or not response_text.strip():
        return {"passed": False, "reason": "empty response", "cost": 0.0,
                "latency_ms": 0.0, "verifier_tier": None}
    return _verify_structural(response_text)


def verify_with_judge(query: str, response_text: str, judge_model: str,
                       judge_api_key: str | None, judge_tier_label: str) -> dict:
    if not response_text or not response_text.strip():
        return {"passed": False, "reason": "empty response", "cost": 0.0,
                "latency_ms": 0.0, "verifier_tier": None}
    return _verify_llm_judge(query, response_text, judge_model, judge_api_key, judge_tier_label)
