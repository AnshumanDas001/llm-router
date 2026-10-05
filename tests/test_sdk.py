"""The Python SDK end to end against the real app, in-process. Calibration
and the cascade are stubbed, so no provider is called and nothing is spent."""
import pytest
from fastapi.testclient import TestClient

import thriftllm
from app.api import models as models_api
from app.api import v1
from app.main import app

FAKE_STATS = {
    "avg_quality": 0.8, "avg_cost": 0.0001, "avg_latency_ms": 900.0, "n_queries": 24,
    "n_errors": 0, "n_rate_limited": 0, "first_error": None,
    "quality_by_difficulty": {"easy": 1.0, "medium": 0.9, "hard": 0.5, "expert": 0.0},
    "cost_by_difficulty": {"easy": 0.00001, "medium": 0.00002, "hard": 0.00003, "expert": 0.0001},
    "n_by_difficulty": {"easy": 6, "medium": 6, "hard": 6, "expert": 6},
}
STRONG_STATS = {**FAKE_STATS, "quality_by_difficulty": {"easy": 1.0, "medium": 1.0, "hard": 0.9, "expert": 0.9},
                "cost_by_difficulty": {"easy": 0.001, "medium": 0.002, "hard": 0.003, "expert": 0.02}}


@pytest.fixture(scope="module")
def sdk():
    with TestClient(app) as http:
        http.post("/auth/signup", json={"username": "sdkuser", "password": "secret123"})
        key = http.post("/api/keys", json={"name": "sdk"}).json()["key"]
        yield thriftllm.Client(api_key=key, base_url="http://testserver", http=http)


def test_calibrate_connects_and_measures(sdk, monkeypatch):
    calls = []

    def fake(model, api_key, max_queries=None, **kw):
        calls.append((model, api_key))
        return dict(STRONG_STATS if "big" in model else FAKE_STATS)

    monkeypatch.setattr(models_api, "calibrate_model", fake)
    small = sdk.calibrate("groq/small-model", api_key="gk")
    assert small.quality["hard"] == 0.5 and not small.clears("hard") and small.clears("easy")
    assert small.api_key == "gk" and calls == [("groq/small-model", "gk")]

    # a second call reuses the stored calibration instead of paying again
    again = sdk.calibrate("groq/small-model", api_key="gk")
    assert again.reused and len(calls) == 1
    sdk.calibrate("groq/small-model", api_key="gk", force=True)
    assert len(calls) == 2

    sdk.calibrate("openai/big-model", api_key="ok")
    assert set(sdk.models()) == {"groq/small-model", "openai/big-model"}


def test_calibrate_needs_a_provider(sdk):
    with pytest.raises(thriftllm.ThriftLLMError) as err:
        sdk.calibrate("no-prefix-model", api_key="x", force=True)
    assert err.value.status == 400 and "provider" in err.value.detail


def test_router_maps_classifies_and_chats(sdk, monkeypatch):
    small, big = sdk.models()["groq/small-model"], sdk.models()["openai/big-model"]
    small.api_key, big.api_key = "gk", "ok"
    router = sdk.router(cheap=small, frontier=big)
    assert router.route_map == {"easy": "cheap", "medium": "cheap", "hard": "frontier", "expert": "frontier"}

    where = router.classify("What is the capital of France?")
    assert where["tier"] == "cheap" and where["trace"]["classification"]["neighbours"]

    seen = {}

    def fake_cascade(messages, tier_models, tier_api_keys, difficulty_to_tier, skip_cheapest, history=None):
        seen.update(models=tier_models, keys=tier_api_keys)
        return {"text": "Paris.", "difficulty": "easy", "initial_tier": "cheap", "final_tier": "cheap",
                "escalated": False, "escalation_reasons": [], "total_cost": 0.00001,
                "total_latency_ms": 800.0, "tokens_in": 10, "tokens_out": 2,
                "trace": {"classification": {"band": "easy", "ms": 6.0, "neighbours": []},
                          "start": {"tier": "cheap"}, "attempts": []}}

    monkeypatch.setattr(v1, "run_cascade", fake_cascade)
    reply = router.chat("What is the capital of France?")
    assert reply.text == "Paris." and reply.tier == "cheap" and reply.band == "easy"
    assert seen["keys"] == {"cheap": "gk", "frontier": "ok"}
    assert reply.trace["start"]["chosen"] == "cheap" and reply.saved >= 0
    assert "Started at cheap" in reply.explain()


def test_router_needs_calibrated_models():
    with pytest.raises(TypeError):
        thriftllm.Router(cheap="groq/small-model", client=object())
    with pytest.raises(ValueError):
        thriftllm.Router(mode="fastest", client=object())


def test_router_with_no_models_uses_the_builtin_stack(sdk, monkeypatch):
    router = sdk.router()
    assert router.route_map["expert"] == "frontier"
    assert router.classify("What is the capital of France?")["tier"] in ("cheap", "mid", "frontier")

    seen = {}

    def fake_cascade(messages, difficulty_to_tier, skip_cheapest=False, **kw):
        seen.update(skip=skip_cheapest, byom=kw.get("tier_models"))
        return {"text": "Paris.", "difficulty": "easy", "initial_tier": "cheap", "final_tier": "cheap",
                "escalated": False, "escalation_reasons": [], "total_cost": 0.00001,
                "total_latency_ms": 800.0, "tokens_in": 10, "tokens_out": 2,
                "trace": {"classification": {"band": "easy", "ms": 6.0, "neighbours": []},
                          "start": {"tier": "cheap"}, "attempts": []}}

    monkeypatch.setattr(v1, "run_cascade", fake_cascade)
    reply = router.chat("What is the capital of France?")
    assert reply.text == "Paris." and reply.tier == "cheap" and seen == {"skip": False, "byom": None}
    assert "Started at cheap" in reply.explain()
    sdk.router(mode="direct").chat("What is the capital of France?")
    assert seen["skip"] is True
