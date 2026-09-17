"""The cascade: classify, generate, verify, escalate.

Also degrades gracefully when a tier is temporarily unavailable (rate
limited / daily quota exhausted) instead of hard-failing the whole
request -- discovered the hard way when Groq's daily token cap got hit
mid-demo and a raw provider exception was dumped straight into the chat.

Supports BYOM (bring your own model): pass tier_models (e.g.
{"cheap": "groq/llama-3.1-8b-instant", "frontier": "gpt-4o"}) and
tier_api_keys (the caller's own third-party keys, used transiently for
this call only -- never stored) to route across a user's own models
instead of our built-in three. A user may configure only some tiers;
the cascade only ever steps through the ones actually provided, and
verification strategy is decided by *position* in that sequence (see
app/verifier.py), not by the literal tier name -- so this works the same
way regardless of which tiers a user did or didn't configure.
"""
import threading
import time

import litellm

from app import scorer
from app.baseline_cost import estimate_cost_for_model
from app.classifier import classify_initial_tier
from app.model_config import TIER_MODEL_LIST, VERIFIER, router
from app.verifier import verify_response, verify_with_judge

TIER_ORDER = ["cheap", "mid", "frontier"]
DEFAULT_TIER_MODELS = {t["model_name"]: t["litellm_params"]["model"] for t in TIER_MODEL_LIST}


def _approx_tokens(text: str) -> int:
    """Rough token count for pricing a streamed response, where the provider
    sends no usage block. ~4 characters per token is the usual English
    approximation; this only feeds the cost estimate, never routing."""
    return max(1, len(text or "") // 4)


def _plan_sequence(initial_tier: str, available_tiers: list[str], skip_cheapest: bool) -> list[str]:
    """The tiers a request will try, in order.

    skip_cheapest is "direct" routing: start one tier up regardless of what
    the classifier said. On a stack whose mid model is already very cheap,
    the cheapest tier can't undercut it -- its judge call costs about what
    mid's own short answer would -- so a user can opt out of the cheap tier
    and its verification entirely. Because the paid judge only ever attaches
    to available_tiers[0], starting above it disables the judge for free.
    """
    if initial_tier not in available_tiers:
        initial_tier = available_tiers[0]
    start = available_tiers.index(initial_tier)
    if skip_cheapest and len(available_tiers) > 1:
        start = max(start, 1)
    return available_tiers[start:]


def _why(exc: Exception) -> str:
    return "rate limited" if isinstance(exc, litellm.RateLimitError) else "unreachable"


def learned_verifier_on(tier_models: dict | None) -> bool:
    """The learned scorer replaces the LLM judge only on the built-in stack.
    It is trained on the built-in cheap model's confidence profile; a BYOM
    model's logprobs are a different distribution and would need their own
    training run, so BYOM keeps the judge."""
    return VERIFIER != "judge" and tier_models is None and scorer.available()


class _Sibling(threading.Thread):
    """A second sample of the cheap tier, drawn concurrently with the one
    the user sees, for the scorer's consistency features. On a local model
    it costs nothing but the parallel compute; on a hosted cheap tier it
    costs one more cheap answer, still well under a judge call."""

    def __init__(self, tier, messages):
        super().__init__(daemon=True)
        self.tier, self.messages = tier, messages
        self.text, self.cost = None, 0.0
        self.start()

    def run(self):
        try:
            resp = router.completion(model=self.tier, messages=self.messages)
            self.text = resp.choices[0].message.content or ""
            self.cost = resp._hidden_params.get("response_cost", 0.0) or 0.0
        except Exception:
            self.text = None   # scorer falls back to neutral consistency


def _verify_learned(tier, messages, query, text, logprobs, entropy, finish_reason,
                    sibling, difficulty, judge) -> dict:
    """Score the cheap answer with the learned verifier; if the provider
    returned no logprobs (dropped param), fall back to the LLM judge."""
    if not logprobs:
        judge_model, judge_api_key, judge_tier_label = judge
        result = verify_with_judge(query, text, judge_model, judge_api_key, judge_tier_label)
        result["reason"] = "no logprobs from cheap tier; " + result["reason"]
        return result
    start = time.perf_counter()
    cost = 0.0
    p_yes = 0.5
    if scorer.USE_SELF_VERIFY:
        spent = []

        def complete(msgs, **params):
            r = router.completion(model=tier, messages=msgs, **params)
            spent.append(r._hidden_params.get("response_cost", 0.0) or 0.0)
            return r

        try:
            p_yes = scorer.self_verify_p_yes(complete, query, text)
        except Exception:
            pass   # neutral value; the other signals still score
        cost += sum(spent)
    if sibling is not None:
        sibling.join(timeout=120)
        cost += sibling.cost
    siblings = [sibling.text] if sibling is not None and sibling.text else []
    result = scorer.verify_learned(
        query, text, logprobs=logprobs, entropy=entropy, siblings=siblings,
        self_verify_p_yes=p_yes, difficulty=difficulty, finish_reason=finish_reason,
    )
    if result["verdict"] == "unsure":
        # The gate couldn't decide; this is the slice the judge still earns
        # its cost on. Its verdict stands, prefixed so the log shows why
        # a judge call happened at all.
        judge_model, judge_api_key, judge_tier_label = judge
        judged = verify_with_judge(query, text, judge_model, judge_api_key, judge_tier_label)
        judged["reason"] = f"{result['reason']}; judge: {judged['reason']}"
        cost += judged["cost"]
        result = judged
    result["cost"] = cost
    result["latency_ms"] = (time.perf_counter() - start) * 1000
    return result


class AllTiersUnavailable(Exception):
    """Raised only if every tier in the cascade sequence is unreachable."""


def _complete(tier: str, messages: list[dict], tier_models: dict | None,
               tier_api_keys: dict | None):
    if tier_models is None:
        return router.completion(model=tier, messages=messages)
    return litellm.completion(
        model=tier_models[tier], messages=messages,
        api_key=(tier_api_keys or {}).get(tier),
    )


def run_cascade(messages: list[dict], tier_models: dict | None = None,
                 tier_api_keys: dict | None = None,
                 difficulty_to_tier: dict | None = None,
                 skip_cheapest: bool = False) -> dict:
    query = messages[-1]["content"]
    initial_tier, difficulty = classify_initial_tier(query, difficulty_to_tier)

    available_tiers = TIER_ORDER if tier_models is None else [
        t for t in TIER_ORDER if t in tier_models
    ]
    if not available_tiers:
        raise AllTiersUnavailable("no tiers configured")
    tier_sequence = _plan_sequence(initial_tier, available_tiers, skip_cheapest)
    initial_tier = tier_sequence[0]

    # Only the single cheapest *configured* tier gets a real, paid LLM-judge
    # check (by the next tier up in the full configured list) -- not
    # whichever tier this particular query's sequence happens to start at.
    # A "hard" query skips straight to e.g. mid, but mid still only gets the
    # free structural check: the eval set showed mid's failure rate is low and its
    # answers are long, making a real judge call there expensive for little
    # safety benefit (it was 82.5% of total cascade cost before this fix).
    judge_model_for = {}
    cheapest_tier = available_tiers[0]
    if len(available_tiers) > 1 and cheapest_tier in tier_sequence:
        judge_tier = available_tiers[1]
        judge_model_for[cheapest_tier] = (
            tier_models[judge_tier] if tier_models is not None else DEFAULT_TIER_MODELS[judge_tier],
            (tier_api_keys or {}).get(judge_tier),
            judge_tier,
        )

    total_cost = 0.0
    total_latency_ms = 0.0
    escalated = False
    escalation_reasons = []
    final_text = None
    final_tier = None
    final_usage = None

    learned = learned_verifier_on(tier_models)

    for i, tier in enumerate(tier_sequence):
        start = time.perf_counter()
        scored = learned and tier in judge_model_for
        sibling = _Sibling(tier, messages) if scored and scorer.USE_SIBLING else None
        try:
            if scored:
                resp = router.completion(model=tier, messages=messages, **scorer.LOGPROB_PARAMS)
            else:
                resp = _complete(tier, messages, tier_models, tier_api_keys)
        except (litellm.RateLimitError, litellm.APIConnectionError) as exc:
            # This tier is temporarily down -- a quota, or a local model
            # that isn't running (the cheap tier is Ollama; on a box without
            # it, every easy question would otherwise fail outright). Skip
            # to the next tier rather than failing the whole request; if
            # there is no next tier, fall back to whatever we already have.
            escalated = True
            escalation_reasons.append(f"{tier} unavailable ({_why(exc)}), skipped")
            continue
        latency_ms = (time.perf_counter() - start) * 1000

        cost = resp._hidden_params.get("response_cost", 0.0) or 0.0
        text = resp.choices[0].message.content

        total_cost += cost
        total_latency_ms += latency_ms
        final_text, final_tier, final_usage = text, tier, resp.usage  # best effort so far

        is_last_tier = i == len(tier_sequence) - 1
        if is_last_tier:
            break  # nothing left to escalate to; trust it unconditionally

        try:
            if scored:
                lp_content = getattr(resp.choices[0].logprobs, "content", None) if resp.choices[0].logprobs else None
                logprobs, entropy = scorer.token_signals(lp_content)
                verify_result = _verify_learned(
                    tier, messages, query, text, logprobs, entropy,
                    resp.choices[0].finish_reason, sibling, difficulty, judge_model_for[tier])
            elif tier in judge_model_for:
                judge_model, judge_api_key, judge_tier_label = judge_model_for[tier]
                verify_result = verify_with_judge(query, text, judge_model, judge_api_key, judge_tier_label)
            else:
                verify_result = verify_response(query, text, tier, tier_models, tier_api_keys)
        except litellm.RateLimitError:
            # Fail open: the verifier being unavailable isn't evidence the
            # answer is wrong. Trusting it beats blocking the whole response
            # on an unrelated tier's exhausted quota.
            verify_result = {"passed": True, "cost": 0.0, "latency_ms": 0.0,
                              "reason": "verifier unavailable (rate limited), trusting answer as-is"}
            escalated = True
            escalation_reasons.append(f"{tier}: {verify_result['reason']}")

        total_cost += verify_result["cost"]
        total_latency_ms += verify_result["latency_ms"]

        if verify_result["passed"]:
            break

        escalated = True
        escalation_reasons.append(f"{tier} failed verification: {verify_result['reason']}")

    if final_text is None:
        raise AllTiersUnavailable(
            f"every tier in {tier_sequence} was rate limited or failed; try again shortly"
        )

    return {
        "text": final_text,
        "difficulty": difficulty,
        "initial_tier": initial_tier,
        "final_tier": final_tier,
        "escalated": escalated,
        "escalation_reasons": escalation_reasons,
        "total_cost": total_cost,
        "total_latency_ms": total_latency_ms,
        "tokens_in": final_usage.prompt_tokens if final_usage else None,
        "tokens_out": final_usage.completion_tokens if final_usage else None,
    }


def _stream_complete(tier, messages, tier_models, tier_api_keys, scored=False):
    """Same dispatch as _complete, but asks the provider to stream."""
    if tier_models is None:
        params = scorer.LOGPROB_PARAMS if scored else {}
        return router.completion(model=tier, messages=messages, stream=True, **params)
    return litellm.completion(
        model=tier_models[tier], messages=messages,
        api_key=(tier_api_keys or {}).get(tier), stream=True,
    )


def _model_for(tier, tier_models):
    return tier_models[tier] if tier_models is not None else DEFAULT_TIER_MODELS[tier]


def run_cascade_stream(messages: list[dict], tier_models: dict | None = None,
                        tier_api_keys: dict | None = None,
                        difficulty_to_tier: dict | None = None,
                        skip_cheapest: bool = False):
    """Generator form of run_cascade, yielding events as they happen.

    Streaming and verify-then-escalate are in tension: a tier's answer can't
    be judged until it is complete, but waiting for completion is exactly
    what streaming exists to avoid. Rather than withhold every token until
    the verdict is in, this streams optimistically and emits an "escalated"
    event if the check then fails -- the client drops what it showed and
    takes the next tier's answer instead. Escalation ran 13/81 in the eval,
    so the common path streams cleanly, and on the uncommon path the user
    sees the router doing the thing it exists to do.

    Events: routing, token, escalated, done, error.
    """
    query = messages[-1]["content"]
    initial_tier, difficulty = classify_initial_tier(query, difficulty_to_tier)

    available_tiers = TIER_ORDER if tier_models is None else [
        t for t in TIER_ORDER if t in tier_models
    ]
    if not available_tiers:
        raise AllTiersUnavailable("no tiers configured")
    tier_sequence = _plan_sequence(initial_tier, available_tiers, skip_cheapest)
    initial_tier = tier_sequence[0]

    judge_model_for = {}
    cheapest_tier = available_tiers[0]
    if len(available_tiers) > 1 and cheapest_tier in tier_sequence:
        judge_tier = available_tiers[1]
        judge_model_for[cheapest_tier] = (
            _model_for(judge_tier, tier_models), (tier_api_keys or {}).get(judge_tier), judge_tier,
        )

    yield {"type": "routing", "difficulty": difficulty, "initial_tier": initial_tier,
           "direct": skip_cheapest}

    total_cost = 0.0
    total_latency_ms = 0.0
    escalated = False
    escalation_reasons = []
    final_text = None
    final_tier = None
    tokens_in = tokens_out = 0

    learned = learned_verifier_on(tier_models)

    for i, tier in enumerate(tier_sequence):
        start = time.perf_counter()
        text_parts = []
        scored = learned and tier in judge_model_for
        sibling = _Sibling(tier, messages) if scored and scorer.USE_SIBLING else None
        logprobs, entropy, finish_reason = [], [], None
        try:
            yield {"type": "tier_start", "tier": tier}
            for chunk in _stream_complete(tier, messages, tier_models, tier_api_keys, scored):
                choice = chunk.choices[0]
                piece = getattr(choice.delta, "content", None)
                if piece:
                    text_parts.append(piece)
                    yield {"type": "token", "tier": tier, "text": piece}
                if scored:
                    lp = getattr(choice, "logprobs", None)
                    if lp and getattr(lp, "content", None):
                        chosen, ent = scorer.token_signals(lp.content)
                        logprobs += chosen
                        entropy += ent
                    finish_reason = getattr(choice, "finish_reason", None) or finish_reason
        except (litellm.RateLimitError, litellm.APIConnectionError) as exc:
            escalated = True
            escalation_reasons.append(f"{tier} unavailable ({_why(exc)}), skipped")
            yield {"type": "escalated", "from": tier, "reason": _why(exc)}
            continue

        latency_ms = (time.perf_counter() - start) * 1000
        text = "".join(text_parts)

        # Streamed chunks carry no usage/cost, so price it from the token
        # counts using the same pricing table the rest of the app uses.
        tokens_in = _approx_tokens(" ".join(m["content"] for m in messages))
        tokens_out = _approx_tokens(text)
        cost = estimate_cost_for_model(_model_for(tier, tier_models), tokens_in, tokens_out)

        total_cost += cost
        total_latency_ms += latency_ms
        final_text, final_tier = text, tier

        is_last_tier = i == len(tier_sequence) - 1
        if is_last_tier:
            break

        try:
            if scored:
                verify_result = _verify_learned(
                    tier, messages, query, text, logprobs, entropy, finish_reason,
                    sibling, difficulty, judge_model_for[tier])
            elif tier in judge_model_for:
                judge_model, judge_api_key, judge_tier_label = judge_model_for[tier]
                verify_result = verify_with_judge(query, text, judge_model, judge_api_key, judge_tier_label)
            else:
                verify_result = verify_response(query, text, tier, tier_models, tier_api_keys)
        except litellm.RateLimitError:
            verify_result = {"passed": True, "cost": 0.0, "latency_ms": 0.0,
                              "reason": "verifier unavailable (rate limited), trusting answer as-is"}

        total_cost += verify_result["cost"]
        total_latency_ms += verify_result["latency_ms"]

        if verify_result["passed"]:
            break

        escalated = True
        escalation_reasons.append(f"{tier} failed verification: {verify_result['reason']}")
        # Tell the client to discard what it just streamed -- the next tier
        # is about to replace it.
        yield {"type": "escalated", "from": tier, "to": tier_sequence[i + 1],
               "reason": verify_result["reason"]}

    if final_text is None:
        raise AllTiersUnavailable(
            f"every tier in {tier_sequence} was rate limited or failed; try again shortly"
        )

    yield {
        "type": "done", "text": final_text, "difficulty": difficulty,
        "initial_tier": initial_tier, "final_tier": final_tier, "escalated": escalated,
        "escalation_reasons": escalation_reasons, "total_cost": total_cost,
        "total_latency_ms": total_latency_ms, "tokens_in": tokens_in, "tokens_out": tokens_out,
    }
