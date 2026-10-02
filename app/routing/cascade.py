"""The cascade: classify, generate, verify, escalate.

Also degrades gracefully when a tier is temporarily unavailable (rate
limited / daily quota exhausted) instead of hard-failing the whole
request -- discovered the hard way when Groq's daily token cap got hit
mid-demo and a raw provider exception was dumped straight into the chat.

Supports BYOM (bring your own model): pass tier_models (e.g.
{"cheap": "groq/openai/gpt-oss-20b", "frontier": "gpt-4o"}) and
tier_api_keys (the caller's own third-party keys, used transiently for
this call only -- never stored) to route across a user's own models
instead of our built-in three. A user may configure only some tiers;
the cascade only ever steps through the ones actually provided, and
verification strategy is decided by *position* in that sequence (see
app/routing/verifier.py), not by the literal tier name -- so this works the same
way regardless of which tiers a user did or didn't configure.
"""
import threading
import time

import litellm

from app.config import JUDGE_MODEL, TIER_MODEL_LIST, VERIFIER, router
from app.pricing import estimate_cost_for_model
from app.routing import scorer
from app.routing.classifier import DIFFICULTY_TO_TIER, classify_with_trace
from app.routing.policy import TIER_ORDER
from app.routing.verifier import verify_response, verify_with_judge

DEFAULT_TIER_MODELS = {t["model_name"]: t["litellm_params"]["model"] for t in TIER_MODEL_LIST}


def _approx_tokens(text: str) -> int:
    """Rough token count for pricing a streamed response when the provider
    sends no usage block. ~4 characters per token is the usual English
    approximation; this only feeds the cost estimate, never routing."""
    return max(1, len(text or "") // 4)


def plan_sequence(initial_tier: str, available_tiers: list[str], skip_cheapest: bool) -> list[str]:
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


# Errors that mean "this tier can't answer right now", not "this request is
# bad": skip the tier and try the next one. Auth is here because a deploy
# can legitimately leave a tier's key unset (no frontier key yet) and the
# cascade should degrade to the tiers it has rather than 502.
SKIPPABLE = (litellm.RateLimitError, litellm.APIConnectionError, litellm.AuthenticationError)


def _why(exc: Exception) -> str:
    if isinstance(exc, litellm.RateLimitError):
        return "rate limited"
    if isinstance(exc, litellm.AuthenticationError):
        return "no valid credentials"
    return "unreachable"


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
    costs one more cheap answer, still well under a judge call.

    Only started when the trained scorer actually reads those features --
    the shipped one doesn't, so this stays dormant on the default stack."""

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
    if scorer.uses(scorer.SELF_VERIFY_FEATURES):
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
        judged["p_correct"], judged["accept_threshold"] = result["p_correct"], result["accept_threshold"]
        judged["judge_verdict"] = judged["reason"].split("judge: ", 1)[-1]
        cost += judged["cost"]
        result = judged
    result["cost"] = cost
    result["latency_ms"] = (time.perf_counter() - start) * 1000
    return result


class AllTiersUnavailable(Exception):
    """Raised only if every tier in the cascade sequence is unreachable."""


def _plan(messages, tier_models, tier_api_keys, difficulty_to_tier, skip_cheapest):
    """Classify the query and lay out the tiers to try, and who judges.
    Returns (query, difficulty, tier_sequence, judge_model_for, trace)."""
    query = messages[-1]["content"]
    difficulty, classification = classify_with_trace(query)
    tier_map = difficulty_to_tier or DIFFICULTY_TO_TIER
    initial_tier = tier_map.get(difficulty, DIFFICULTY_TO_TIER[difficulty])

    available_tiers = TIER_ORDER if tier_models is None else [
        t for t in TIER_ORDER if t in tier_models
    ]
    if not available_tiers:
        raise AllTiersUnavailable("no tiers configured")
    tier_sequence = plan_sequence(initial_tier, available_tiers, skip_cheapest)

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
        judge_model_for[cheapest_tier] = judge_for(available_tiers, tier_models, tier_api_keys)
    trace = {
        "classification": classification,
        "start": {"mapped_tier": initial_tier, "tier": tier_sequence[0],
                  "direct": skip_cheapest and tier_sequence[0] != initial_tier,
                  "sequence": tier_sequence},
        "attempts": [],
    }
    return query, difficulty, tier_sequence, judge_model_for, trace


def _check_record(tier, scored, judge_model_for, verify_result) -> dict:
    """What checked one tier's answer, and what it said."""
    if scored:
        kind = "learned gate" if verify_result.get("verifier_tier") == "scorer" else "learned gate, then judge"
    elif tier in judge_model_for:
        kind = "judge"
    else:
        kind = "structural"
    record = {"kind": kind, "passed": verify_result["passed"], "reason": verify_result["reason"],
              "cost": verify_result["cost"]}
    for k in ("p_correct", "accept_threshold", "judge_verdict"):
        if k in verify_result:
            record[k] = verify_result[k]
    if kind == "judge":
        record["judge_verdict"] = verify_result["reason"]
        record["judge_model"] = (judge_model_for[tier] or (None,))[0]
    elif "judge" in kind:
        record["judge_model"] = (judge_model_for[tier] or (None,))[0]
    if verify_result.get("rate_limited"):
        record["kind"] = "skipped (verifier rate limited)"
    return record


def _verify(scored, tier, messages, query, text, logprobs, entropy, finish_reason,
            sibling, difficulty, judge_model_for, tier_models, tier_api_keys) -> dict:
    """Check one tier's answer: the learned gate (then the judge if it is
    unsure), the judge alone, or the free structural check -- whichever
    this tier's position in the sequence calls for."""
    try:
        if scored:
            return _verify_learned(tier, messages, query, text, logprobs, entropy,
                                   finish_reason, sibling, difficulty, judge_model_for[tier])
        if tier in judge_model_for:
            judge_model, judge_api_key, judge_tier_label = judge_model_for[tier]
            return verify_with_judge(query, text, judge_model, judge_api_key, judge_tier_label)
        return verify_response(query, text, tier, tier_models, tier_api_keys)
    except litellm.RateLimitError:
        # Fail open: the verifier being unavailable isn't evidence the
        # answer is wrong. Trusting it beats blocking the whole response
        # on an unrelated tier's exhausted quota.
        return {"passed": True, "cost": 0.0, "latency_ms": 0.0, "rate_limited": True,
                "reason": "verifier unavailable (rate limited), trusting answer as-is"}


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
    query, difficulty, tier_sequence, judge_model_for, trace = _plan(
        messages, tier_models, tier_api_keys, difficulty_to_tier, skip_cheapest)
    initial_tier = tier_sequence[0]
    attempts = trace["attempts"]

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
        sibling = _Sibling(tier, messages) if scored and scorer.uses(scorer.SIBLING_FEATURES) else None
        try:
            if scored:
                resp = router.completion(model=tier, messages=messages, **scorer.LOGPROB_PARAMS)
            else:
                resp = _complete(tier, messages, tier_models, tier_api_keys)
        except SKIPPABLE as exc:
            # This tier is temporarily down -- a quota, or a local model
            # that isn't running (the cheap tier is Ollama; on a box without
            # it, every easy question would otherwise fail outright). Skip
            # to the next tier rather than failing the whole request; if
            # there is no next tier, fall back to whatever we already have.
            escalated = True
            escalation_reasons.append(f"{tier} unavailable ({_why(exc)}), skipped")
            attempts.append({"tier": tier, "model": _model_for(tier, tier_models),
                             "outcome": "unavailable", "reason": _why(exc)})
            continue
        latency_ms = (time.perf_counter() - start) * 1000

        cost = resp._hidden_params.get("response_cost", 0.0) or 0.0
        text = resp.choices[0].message.content

        total_cost += cost
        total_latency_ms += latency_ms
        final_text, final_tier, final_usage = text, tier, resp.usage  # best effort so far
        attempt = {"tier": tier, "model": _model_for(tier, tier_models), "cost": cost,
                   "latency_ms": round(latency_ms),
                   "tokens_in": getattr(resp.usage, "prompt_tokens", None),
                   "tokens_out": getattr(resp.usage, "completion_tokens", None)}
        attempts.append(attempt)

        is_last_tier = i == len(tier_sequence) - 1
        if is_last_tier:
            attempt.update(outcome="shipped", check={"kind": "none", "reason": "last tier in the sequence: nothing to escalate to"})
            break  # nothing left to escalate to; trust it unconditionally

        logprobs, entropy = [], []
        if scored:
            lp_content = getattr(resp.choices[0].logprobs, "content", None) if resp.choices[0].logprobs else None
            logprobs, entropy = scorer.token_signals(lp_content)
        verify_result = _verify(scored, tier, messages, query, text, logprobs, entropy,
                                resp.choices[0].finish_reason, sibling, difficulty,
                                judge_model_for, tier_models, tier_api_keys)
        if verify_result.get("rate_limited"):
            escalated = True
            escalation_reasons.append(f"{tier}: {verify_result['reason']}")

        total_cost += verify_result["cost"]
        total_latency_ms += verify_result["latency_ms"]
        attempt["check"] = _check_record(tier, scored, judge_model_for, verify_result)
        attempt["outcome"] = "shipped" if verify_result["passed"] else "escalated"

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
        "trace": trace,
    }


# Ask for a usage block at the end of the stream. Without it a streamed
# answer can only be priced from its visible text, which misses a thinking
# model's hidden reasoning entirely: a frontier AIME answer that cost $0.19
# priced from its text at a fraction of a cent.
STREAM_USAGE = {"stream_options": {"include_usage": True}}


def _stream_complete(tier, messages, tier_models, tier_api_keys, scored=False):
    """Same dispatch as _complete, but asks the provider to stream."""
    if tier_models is None:
        params = scorer.LOGPROB_PARAMS if scored else {}
        return router.completion(model=tier, messages=messages, stream=True, **STREAM_USAGE, **params)
    return litellm.completion(
        model=tier_models[tier], messages=messages,
        api_key=(tier_api_keys or {}).get(tier), stream=True,
        # providers that can't report usage drop the option, not the call
        drop_params=True, **STREAM_USAGE,
    )


def _model_for(tier, tier_models):
    return tier_models[tier] if tier_models is not None else DEFAULT_TIER_MODELS[tier]


def judge_for(available_tiers, tier_models, tier_api_keys):
    """(model, api_key, label) that judges the cheapest tier, or None if
    there's nothing above it to escalate to. Built-in stack: JUDGE_MODEL if
    set, else the tier above; BYOM: always the user's tier above."""
    if len(available_tiers) < 2:
        return None
    judge_tier = available_tiers[1]
    if tier_models is None and JUDGE_MODEL:
        return JUDGE_MODEL, None, "judge"
    return _model_for(judge_tier, tier_models), (tier_api_keys or {}).get(judge_tier), judge_tier


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
    query, difficulty, tier_sequence, judge_model_for, trace = _plan(
        messages, tier_models, tier_api_keys, difficulty_to_tier, skip_cheapest)
    initial_tier = tier_sequence[0]
    attempts = trace["attempts"]

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
        sibling = _Sibling(tier, messages) if scored and scorer.uses(scorer.SIBLING_FEATURES) else None
        logprobs, entropy, finish_reason, usage = [], [], None, None
        try:
            yield {"type": "tier_start", "tier": tier}
            for chunk in _stream_complete(tier, messages, tier_models, tier_api_keys, scored):
                usage = getattr(chunk, "usage", None) or usage
                if not chunk.choices:
                    continue                 # the trailing usage-only chunk
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
        except SKIPPABLE as exc:
            escalated = True
            escalation_reasons.append(f"{tier} unavailable ({_why(exc)}), skipped")
            attempts.append({"tier": tier, "model": _model_for(tier, tier_models),
                             "outcome": "unavailable", "reason": _why(exc)})
            yield {"type": "escalated", "from": tier, "reason": _why(exc)}
            continue

        latency_ms = (time.perf_counter() - start) * 1000
        text = "".join(text_parts)

        # Priced from the provider's usage block (reasoning tokens included)
        # when it sent one, else estimated from the text.
        if usage:
            tokens_in, tokens_out = usage.prompt_tokens, usage.completion_tokens
        else:
            tokens_in = _approx_tokens(" ".join(m["content"] for m in messages))
            tokens_out = _approx_tokens(text)
        cost = estimate_cost_for_model(_model_for(tier, tier_models), tokens_in, tokens_out)

        total_cost += cost
        total_latency_ms += latency_ms
        final_text, final_tier = text, tier
        attempt = {"tier": tier, "model": _model_for(tier, tier_models), "cost": cost,
                   "latency_ms": round(latency_ms), "tokens_in": tokens_in, "tokens_out": tokens_out,
                   "priced_from": "usage" if usage else "estimate"}
        attempts.append(attempt)

        is_last_tier = i == len(tier_sequence) - 1
        if is_last_tier:
            attempt.update(outcome="shipped", check={"kind": "none", "reason": "last tier in the sequence: nothing to escalate to"})
            break

        verify_result = _verify(scored, tier, messages, query, text, logprobs, entropy,
                                finish_reason, sibling, difficulty, judge_model_for,
                                tier_models, tier_api_keys)

        total_cost += verify_result["cost"]
        total_latency_ms += verify_result["latency_ms"]
        attempt["check"] = _check_record(tier, scored, judge_model_for, verify_result)
        attempt["outcome"] = "shipped" if verify_result["passed"] else "escalated"

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
        "trace": trace,
    }
