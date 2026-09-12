"""Week 3: verification layer for the cascade.

Two different verification strategies, chosen by which tier generated the
answer -- not a uniform "always ask the next tier up" rule:

- cheap's answers get a paid LLM-judge completeness/coherence check (run on
  mid). Week 2 showed cheap has real, substantive failure modes (dropped
  sub-requests, self-contradictory reasoning), so this call earns its cost.
- mid's answers get a free structural check only (non-empty, not a refusal,
  reasonable length). Week 2 showed mid's failure rate on hard queries is
  near-zero (one minor stylistic issue across 61 queries), and verification
  cost scales with response length -- mid's hard-query answers are often
  long (code + explanation), so a full LLM-judge call here was the dominant
  cost driver in the cascade (82.5% of total cost from hard queries alone)
  for very little actual safety benefit. A free check still catches the
  genuinely broken case (empty output, a refusal) at zero marginal cost.
- frontier is terminal: no verification, nothing left to escalate to.
"""
import re
import time

from app.model_config import router

NEXT_TIER = {"cheap": "mid", "mid": "frontier"}  # frontier has no next tier

VERIFY_PROMPT = """You are a fast quality-control check for an AI response, not a full grader.

Original request:
{query}

Candidate response:
{response}

Does this response address every explicit part of the request, and is it coherent and free of obvious factual or logical errors? Reply with exactly one line: "YES" or "NO: <one short reason>"."""

REFUSAL_PATTERN = re.compile(
    r"\b(i cannot|i can'?t|i'm unable|i am unable|as an ai)\b", re.IGNORECASE
)
MIN_STRUCTURAL_LENGTH = 20  # characters


def _verify_structural(response_text: str) -> dict:
    """Free check: no model call, just pattern/length sanity checks."""
    stripped = response_text.strip() if response_text else ""
    if not stripped:
        return {"passed": False, "reason": "empty response", "cost": 0.0,
                "latency_ms": 0.0, "verifier_tier": None}
    if len(stripped) < MIN_STRUCTURAL_LENGTH:
        return {"passed": False, "reason": "suspiciously short response", "cost": 0.0,
                "latency_ms": 0.0, "verifier_tier": None}
    if REFUSAL_PATTERN.search(stripped[:200]):
        return {"passed": False, "reason": "looks like a refusal", "cost": 0.0,
                "latency_ms": 0.0, "verifier_tier": None}
    return {"passed": True, "reason": "structural check only (free, no LLM-judge call)",
            "cost": 0.0, "latency_ms": 0.0, "verifier_tier": None}


def _verify_llm_judge(query: str, response_text: str, next_tier: str) -> dict:
    """Paid check: ask the next tier up for a completeness/coherence verdict."""
    prompt = VERIFY_PROMPT.format(query=query, response=response_text)
    start = time.perf_counter()
    result = router.completion(
        model=next_tier,
        messages=[{"role": "user", "content": prompt}],
        # gpt-oss-20b (mid) spends tokens on hidden reasoning before the
        # visible answer -- 60 was too tight and got cut off (finish_reason
        # "length", empty content) before it ever produced YES/NO.
        max_tokens=300,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    text = result.choices[0].message.content.strip()
    cost = result._hidden_params.get("response_cost", 0.0) or 0.0

    if not text:
        # Fail safe toward escalation rather than silently trusting an
        # unreadable verdict.
        return {"passed": False, "reason": "verifier returned no content (truncated?)",
                "cost": cost, "latency_ms": latency_ms, "verifier_tier": next_tier}

    passed = text.upper().startswith("YES")
    return {"passed": passed, "reason": text, "cost": cost,
            "latency_ms": latency_ms, "verifier_tier": next_tier}


def verify_response(query: str, response_text: str, tier: str) -> dict:
    """Returns {passed, reason, cost, latency_ms, verifier_tier}."""
    next_tier = NEXT_TIER.get(tier)
    if next_tier is None:
        return {"passed": True, "reason": "terminal tier, no verification",
                "cost": 0.0, "latency_ms": 0.0, "verifier_tier": None}

    if tier == "mid":
        return _verify_structural(response_text)

    if not response_text or not response_text.strip():
        return {"passed": False, "reason": "empty response", "cost": 0.0,
                "latency_ms": 0.0, "verifier_tier": None}

    return _verify_llm_judge(query, response_text, next_tier)
