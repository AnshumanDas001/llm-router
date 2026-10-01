"""End-to-end through the HTTP layer with the model calls stubbed out:
every router is mounted, pages render, and a chat round-trips through
storage. Costs nothing -- no provider is ever called."""
import pytest
from fastapi.testclient import TestClient

from app.api import chats
from app.main import app

FAKE_RESULT = {
    "text": "Canberra.", "difficulty": "easy", "initial_tier": "cheap", "final_tier": "cheap",
    "escalated": False, "escalation_reasons": [], "total_cost": 0.00001,
    "total_latency_ms": 12.0, "tokens_in": 10, "tokens_out": 3,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        r = c.post("/auth/signup", json={"username": "tester", "password": "secret123"})
        assert r.status_code == 200, r.text
        yield c


@pytest.mark.parametrize("path", ["/app", "/try", "/settings", "/models", "/sessions", "/guide", "/demo"])
def test_pages_render(client, path):
    r = client.get(path)
    assert r.status_code == 200 and "<html" in r.text.lower()


def test_static_and_health(client):
    assert client.get("/static/app.css").status_code == 200
    health = client.get("/health").json()
    assert health["database"] == "sqlite"


def test_routing_map_is_public(client):
    r = client.get("/api/routing").json()
    assert r["map"]["expert"] == "frontier"
    assert r["bands"] == ["easy", "medium", "hard", "expert"]
    assert r["threshold"] == 0.8


def test_openai_endpoint_needs_a_key(client):
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 401


def test_chat_round_trip(client, monkeypatch):
    chat_id = client.post("/api/chats", json={}).json()["id"]
    monkeypatch.setattr(chats, "run_cascade", lambda **kw: dict(FAKE_RESULT))

    r = client.post(f"/api/chats/{chat_id}/messages", json={"content": "Capital of Australia?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["content"] == "Canberra." and body["tier"] == "cheap"
    assert body["baseline_cost"] > 0

    chat = client.get(f"/api/chats/{chat_id}").json()
    assert [m["role"] for m in chat["messages"]] == ["user", "assistant"]
    assert any(c["id"] == chat_id for c in client.get("/api/chats").json())


def test_classify_uses_the_chats_routing_map(client):
    chat_id = client.post("/api/chats", json={"routing_mode": "direct"}).json()["id"]
    easy = client.post("/api/classify", json={"content": "What is the capital of France?"}).json()
    assert easy["tier"] == "cheap"
    # direct mode skips the cheap tier, and the badge should say so
    direct = client.post("/api/classify", json={"content": "What is the capital of France?",
                                                "chat_id": chat_id}).json()
    assert direct["tier"] == "mid"


def test_competition_maths_routes_to_frontier(client):
    """A competition-style problem written for this test, so it is not one
    of the classifier's own reference questions."""
    problem = ("Let $S$ be the set of positive integers $n < 2000$ such that $n^2 + 3n + 7$ "
               "is divisible by $13$. Find the remainder when the sum of the elements of $S$ "
               "is divided by $1000$.")
    r = client.post("/api/classify", json={"content": problem}).json()
    assert r == {"tier": "frontier", "difficulty": "expert"}


def test_everyday_maths_stays_off_the_frontier(client):
    r = client.post("/api/classify", json={"content": "What is 15% of 240?"}).json()
    assert r["tier"] != "frontier"
