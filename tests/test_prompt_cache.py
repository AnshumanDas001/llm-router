"""Prompt-cache-aware routing: the prediction, the learning, the pricing, and
the routing decisions it changes. No model is called."""
from types import SimpleNamespace

import pytest

from app.api import common
from app.pricing import estimate_cost_for_model
from app.routing import cascade, prompt_cache
from app.routing.policy import derive_tier_map


@pytest.fixture(autouse=True)
def fresh_cache():
    prompt_cache.reset()
    yield
    prompt_cache.reset()


def chat(turns: int, words: int = 10) -> list[dict]:
    """A conversation ending in a new user message."""
    msgs = [{"role": "system", "content": "You are helpful. " * words}]
    for i in range(turns):
        msgs += [{"role": "user", "content": f"question {i}"}, {"role": "assistant", "content": f"answer {i}"}]
    return msgs + [{"role": "user", "content": "next question"}]


def test_a_model_is_warm_for_the_conversation_it_just_served():
    first = chat(0)                                  # [system, user]
    prompt_cache.remember("big", first, 20000, now=100.0)
    second = first + [{"role": "assistant", "content": "a"}, {"role": "user", "content": "b"}]
    assert prompt_cache.warm_prefix("big", second, now=130.0) == 20000
    assert prompt_cache.warm_prefix("small", second, now=130.0) == 0      # the cache is per model
    assert prompt_cache.warm_prefix("big", chat(3), now=130.0) == 0       # a different conversation


def test_warmth_expires_and_short_prefixes_never_count():
    first = chat(0)
    second = first + [{"role": "assistant", "content": "a"}, {"role": "user", "content": "b"}]
    prompt_cache.remember("big", first, 20000, now=100.0)
    assert prompt_cache.warm_prefix("big", second, now=100.0 + prompt_cache.TTL_S + 1) == 0
    prompt_cache.remember("big", first, prompt_cache.MIN_PREFIX_TOKENS - 1, now=200.0)
    assert prompt_cache.warm_prefix("big", second, now=210.0) == 0


def test_hit_rate_is_learned_from_the_receipts():
    assert prompt_cache.hit_rate("big") == pytest.approx(prompt_cache.PRIOR_HIT_RATE)
    for _ in range(8):
        prompt_cache.observe("big", predicted=20000, cached=0)            # provider didn't cache
    assert prompt_cache.hit_rate("big") < 0.25
    prompt_cache.observe("big", predicted=0, cached=0)                    # not predicted warm: nothing
    assert prompt_cache.hit_rate("big") < 0.25
    for _ in range(8):
        prompt_cache.observe("quiet", predicted=20000, cached=None)       # never reports: no discount
    assert prompt_cache.hit_rate("quiet") < 0.25
    for _ in range(8):
        prompt_cache.observe("works", predicted=20000, cached=19000)
    assert prompt_cache.hit_rate("works") > 0.9


def test_cached_tokens_are_read_from_either_usage_shape():
    openai = SimpleNamespace(prompt_tokens=100, prompt_tokens_details=SimpleNamespace(cached_tokens=80))
    anthropic = SimpleNamespace(prompt_tokens=100, prompt_tokens_details=None, cache_read_input_tokens=60)
    assert prompt_cache.cached_tokens(openai) == 80
    assert prompt_cache.cached_tokens(anthropic) == 60
    assert prompt_cache.cached_tokens(SimpleNamespace(prompt_tokens=100)) is None


def test_cached_tokens_bill_at_the_cached_rate():
    model = "openrouter/google/gemini-3.5-flash"          # $1.50/M input, $0.15/M cached, $9/M output
    cold = estimate_cost_for_model(model, 20000, 500)
    warm = estimate_cost_for_model(model, 20000, 500, cached_tokens=15000)
    assert cold == pytest.approx(20000 * 1.5e-6 + 500 * 9e-6)
    assert warm == pytest.approx(5000 * 1.5e-6 + 15000 * 1.5e-7 + 500 * 9e-6)


def test_history_cost_uses_the_cached_rate_only_where_warm(monkeypatch):
    monkeypatch.setattr(prompt_cache, "prices", lambda m: (3e-6, 3e-7) if m == "big" else (1e-6, 1e-6))
    msgs = chat(1)
    prompt_cache.remember("big", msgs[:-2], 20000, now=1000.0)
    big = prompt_cache.history_cost("big", msgs, history=20500, now=1010.0)
    small = prompt_cache.history_cost("small", msgs, history=20500, now=1010.0)
    rate = prompt_cache.PRIOR_HIT_RATE
    assert big["warm_tokens"] == 20000 and small["warm_tokens"] == 0
    assert big["cost"] == pytest.approx(rate * (20000 * 3e-7 + 500 * 3e-6) + (1 - rate) * 20500 * 3e-6)
    assert small["cost"] == pytest.approx(20500 * 1e-6)


QUALITY = {"small": {"medium": 0.85}, "big": {"medium": 1.0}}
COST = {"small": {"medium": 0.0026}, "big": {"medium": 0.0078}}   # one answer, no history


def test_a_long_warm_history_keeps_the_conversation_on_the_warm_model():
    """The LinkedIn example: 3x-apart prices, 20k tokens cached on the big model."""
    tiers = {"cheap": QUALITY["small"], "mid": QUALITY["big"]}
    costs = {"cheap": COST["small"], "mid": COST["big"]}
    assert derive_tier_map(tiers, ["cheap", "mid"], costs, 0.0001)["medium"] == "cheap"   # no history
    extra = {"cheap": 20500 * 1e-6, "mid": 20000 * 3e-7 + 500 * 3e-6}                    # big is warm
    assert derive_tier_map(tiers, ["cheap", "mid"], costs, 0.0001, extra)["medium"] == "mid"


def test_warmth_never_overrides_the_quality_floor():
    tiers = {"cheap": {"medium": 0.6}, "mid": {"medium": 1.0}}
    costs = {"cheap": COST["small"], "mid": COST["big"]}
    extra = {"cheap": 0.0, "mid": 1.0}                        # cheap is warm and free, but below 80%
    assert derive_tier_map(tiers, ["cheap", "mid"], costs, 0.0001, extra)["medium"] == "mid"


def test_the_builtin_stack_still_starts_cheap_with_mid_warm():
    """Llama 8B cold ($0.05/M) still undercuts Gemini warm ($0.15/M cached)."""
    msgs = chat(30, words=600)
    prompt_cache.remember(cascade.DEFAULT_TIER_MODELS["mid"], msgs[:-2], 20000)
    history = common.history_for(msgs, None)
    assert history["mid"]["warm_tokens"] > 0 and history["cheap"]["warm_tokens"] == 0
    assert history["cheap"]["cost"] < history["mid"]["cost"]
    assert common.tier_map_for(0, None, history)["easy"] == "cheap"


def test_the_cascade_learns_and_remembers_between_turns(monkeypatch):
    calls = []

    def fake_completion(model, messages, api_key=None, **kw):
        calls.append(model)
        n = len(messages)
        usage = SimpleNamespace(prompt_tokens=5000 * n, completion_tokens=50,
                                prompt_tokens_details=SimpleNamespace(cached_tokens=4000 if n > 2 else 0))
        msg = SimpleNamespace(content="Paris.")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop", logprobs=None)],
                               usage=usage, _hidden_params={"response_cost": 0.001})

    monkeypatch.setattr(cascade.litellm, "completion", fake_completion)
    turn1 = chat(0)
    r1 = cascade.run_cascade(turn1, tier_models={"cheap": "groq/one-model"})
    assert r1["trace"]["attempts"][0]["predicted_cached"] == 0
    turn2 = turn1 + [{"role": "assistant", "content": "Paris."}, {"role": "user", "content": "and Spain?"}]
    r2 = cascade.run_cascade(turn2, tier_models={"cheap": "groq/one-model"})
    attempt = r2["trace"]["attempts"][0]
    assert attempt["predicted_cached"] == 10000          # turn 1's request, remembered
    assert attempt["cached_tokens"] == 4000              # what the provider reported
