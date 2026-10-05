"""Provider prompt caching, as a cost the router can price.

Every chat turn re-sends the whole conversation, and providers discount the
part they saw recently: if a request starts with the same messages as one the
same model served a few minutes ago, those tokens bill at the cached rate
(10-50% of the input price). The discount belongs to that one model. Routing
each turn independently, as if every call were the first message of a fresh
chat, can therefore switch away from a model that holds the history cheaply
and pay full price for all of it somewhere else.

Nobody can ask a provider whether a prompt is cached, so this module predicts
it from what the router itself sent:

  remember()       after each call: this model was sent these messages, this
                   many tokens, at this time
  warm_prefix()    before a call: of the messages about to be sent, how many
                   tokens did this model see within its cache lifetime?
  history_cost()   the expected cost of re-sending the conversation history to
                   one model, at the cached rate where it's warm
  observe()        after a call: did the provider actually report the cached
                   tokens predicted? Each model's hit rate is learned from this,
                   so a provider that doesn't cache stops getting the discount.

A conversation is recognised by its messages, not by an id: an earlier request
is a prefix of the next one. That works the same for the chat app, the demo,
the SDK and the OpenAI-compatible endpoint, which carries no conversation id.

State is in memory, per process. It only ever needs to outlive a cache
lifetime (minutes), and the deploy runs one worker. Behind several workers a
request may land where nothing is remembered, and is priced as if cold, which
is today's behaviour.
"""
import hashlib
import os
import threading
import time

import litellm

# How long a provider keeps a prefix. OpenAI and Gemini evict after about five
# to ten minutes idle; Anthropic's default is five. Erring short means a warm
# model is sometimes priced as cold, which is the safe mistake.
TTL_S = float(os.getenv("CACHE_TTL_S", "300"))
# Prefixes shorter than this are never cached (OpenAI and Gemini: 1,024 tokens).
MIN_PREFIX_TOKENS = int(os.getenv("CACHE_MIN_TOKENS", "1024"))
# Before any observation, assume a warm prefix usually hits. Learned per model.
PRIOR_HIT_RATE = 0.9
PRIOR_WEIGHT = 2.0
# Remembered requests are pruned as they expire; this bounds a burst.
MAX_REMEMBERED = 20000

_lock = threading.Lock()
_seen: dict[tuple[str, str], tuple[float, int]] = {}   # (model, prefix hash) -> (when, prompt tokens)
_hits: dict[str, list[float]] = {}                      # model -> [hits, observations]


def _prefix_hashes(messages: list[dict]) -> list[str]:
    """h[j] identifies messages[:j + 1]: each hash chains the one before, so
    every prefix of the conversation has its own key in one pass."""
    out, h = [], hashlib.sha256()
    for m in messages:
        h.update(f"{m.get('role')}\x00{m.get('content')}\x01".encode())
        out.append(h.copy().hexdigest()[:32])
    return out


def _prune(now: float) -> None:
    expired = [k for k, (t, _) in _seen.items() if now - t > TTL_S]
    for k in expired:
        del _seen[k]
    if len(_seen) > MAX_REMEMBERED:
        for k, _ in sorted(_seen.items(), key=lambda kv: kv[1][0])[: len(_seen) - MAX_REMEMBERED]:
            del _seen[k]


def remember(model: str, messages: list[dict], prompt_tokens: int | None, now: float | None = None) -> None:
    """Record that `model` was just sent `messages` (prompt_tokens long)."""
    if not messages or not prompt_tokens:
        return
    now = time.time() if now is None else now
    key = (model, _prefix_hashes(messages)[-1])
    with _lock:
        _seen[key] = (now, int(prompt_tokens))
        _prune(now)


def warm_prefix(model: str, messages: list[dict], now: float | None = None) -> int:
    """Tokens of `messages` that `model` probably still has cached: the
    longest earlier request to this model, within the cache lifetime, whose
    messages begin this one. 0 when cold or too short to be cached."""
    if len(messages) < 2:
        return 0
    now = time.time() if now is None else now
    hashes = _prefix_hashes(messages)
    with _lock:
        for j in range(len(messages) - 2, -1, -1):          # longest strict prefix first
            hit = _seen.get((model, hashes[j]))
            if hit and now - hit[0] <= TTL_S:
                return hit[1] if hit[1] >= MIN_PREFIX_TOKENS else 0
    return 0


def hit_rate(model: str) -> float:
    """Share of predicted-warm calls where the provider really served cached
    tokens, starting from PRIOR_HIT_RATE."""
    with _lock:
        hits, n = _hits.get(model, [0.0, 0.0])
    return (hits + PRIOR_HIT_RATE * PRIOR_WEIGHT) / (n + PRIOR_WEIGHT)


def observe(model: str, predicted: int, cached: int | None) -> None:
    """Learn from the provider's receipt, on calls predicted to be warm.

    A provider that doesn't report a cached count at all counts as a miss:
    with no evidence of a hit, the discount shouldn't keep steering routing.
    Measured: Groq's gpt-oss reports nothing, and Gemini and Llama through
    OpenRouter reported 0 cached tokens on repeated 5,000-token prefixes."""
    if predicted <= 0:
        return
    with _lock:
        rec = _hits.setdefault(model, [0.0, 0.0])
        rec[0] += 1.0 if cached is not None and cached >= 0.5 * predicted else 0.0
        rec[1] += 1.0


def cached_tokens(usage) -> int | None:
    """The cached-input count from a provider's usage block, in either the
    OpenAI shape (prompt_tokens_details.cached_tokens) or Anthropic's
    (cache_read_input_tokens). None when the provider didn't say."""
    if usage is None:
        return None
    details = getattr(usage, "prompt_tokens_details", None)
    value = getattr(details, "cached_tokens", None) if details is not None else None
    if value is None:
        value = getattr(usage, "cache_read_input_tokens", None)
    return int(value) if value is not None else None


def prices(model: str) -> tuple[float, float] | None:
    """(input, cached input) cost per token, or None if the model is unpriced.
    A model with no cached price gets no discount."""
    try:
        info = litellm.get_model_info(model)
    except Exception:
        return None
    full = info.get("input_cost_per_token")
    if full is None:
        return None
    cached = info.get("cache_read_input_token_cost")
    return float(full), float(cached if cached is not None else full)


def history_tokens(messages: list[dict]) -> int:
    """Tokens in everything before the newest message: the part of the
    request that repeats from turn to turn."""
    if len(messages) < 2:
        return 0
    try:
        return int(litellm.token_counter(model="gpt-4o", messages=messages[:-1]))
    except Exception:
        return sum(len(str(m.get("content") or "")) for m in messages[:-1]) // 4


def history_cost(model: str, messages: list[dict], history: int | None = None,
                 now: float | None = None) -> dict:
    """Expected cost of re-sending the conversation history to `model`.

    The calibrated cost per answer already covers the new message and the
    reply, as measured on single questions; this is the rest of the request.
    The cached part bills at the cached rate with the learned hit-rate
    probability, the remainder (and every miss) at the full input rate."""
    tokens = history_tokens(messages) if history is None else history
    warm = min(warm_prefix(model, messages, now), tokens) if tokens else 0
    p = prices(model)
    if not tokens or p is None:
        return {"tokens": tokens, "warm_tokens": warm, "hit_rate": None, "cost": 0.0, "priced": p is not None}
    full, cached = p
    cold = tokens * full
    if warm:
        rate = hit_rate(model)
        warm_cost = warm * cached + (tokens - warm) * full
        cost = rate * warm_cost + (1 - rate) * cold
    else:
        rate, cost = None, cold
    return {"tokens": tokens, "warm_tokens": warm, "hit_rate": rate, "cost": cost, "priced": True}


def reset() -> None:
    """Forget everything (tests)."""
    with _lock:
        _seen.clear()
        _hits.clear()
