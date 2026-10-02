"""Exercise the whole app over HTTP against a running server, local or
deployed, and report what works.

    ./venv/bin/python -m scripts.ops.e2e_check --base-url http://localhost:8000
    ./venv/bin/python -m scripts.ops.e2e_check --base-url https://your-host --browser

It signs up a throwaway account (e2e-<random>) unless you pass --username
and --password, then walks every feature a user touches: pages, router
startup, the public routing map and demo endpoints, sign-up and sign-in,
chats with streamed answers and their stored route, daily limits, API keys,
the OpenAI-compatible endpoint, the v1 API, and key revocation.

It sends two everyday prompts on the built-in models, which the cheap tier
answers for a small fraction of a cent. --expert adds one competition-maths
prompt, which starts at the thinking frontier and costs $0.13-0.29.
--browser also drives the pages in Chrome (needs `pip install playwright`).
"""
import argparse
import json
import secrets
import sys
import time

import httpx

results = []


def check(name, fn):
    start = time.perf_counter()
    try:
        note = fn()
        results.append((name, True, ""))
        print(f"  PASS  {name}  ({time.perf_counter() - start:.1f}s){'  ' + note if note else ''}")
    except Exception as e:
        results.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"  FAIL  {name}\n        {type(e).__name__}: {e}")


def stream_chat(http: httpx.Client, chat_id: int, content: str) -> dict:
    """Send a message on the streaming endpoint; return the done event and
    the event types seen."""
    events, done = [], None
    with http.stream("POST", f"/api/chats/{chat_id}/messages/stream", json={"content": content},
                     timeout=300) as r:
        if r.status_code != 200:
            raise AssertionError(f"HTTP {r.status_code}: {r.read().decode()[:200]}")
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            events.append(ev["type"])
            if ev["type"] == "error":
                raise AssertionError(ev["detail"])
            if ev["type"] == "done":
                done = ev
    if done is None:
        raise AssertionError(f"stream ended without a done event (saw {events})")
    done["_events"] = events
    return done


def main():
    sys.stdout.reconfigure(line_buffering=True)    # show progress as it happens, even when piped
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--username")
    ap.add_argument("--password")
    ap.add_argument("--expert", action="store_true", help="also send one competition-maths prompt ($0.13-0.29)")
    ap.add_argument("--browser", action="store_true", help="also check the pages in Chrome via playwright")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    http = httpx.Client(base_url=base, timeout=60, follow_redirects=False)
    anon = httpx.Client(base_url=base, timeout=60)
    state = {}
    print(f"ThriftLLM end-to-end check against {base}\n")

    # --- pages and startup --------------------------------------------------
    def pages():
        for path in ("/", "/try", "/demo", "/guide", "/app", "/models", "/sessions", "/settings"):
            r = anon.get(path)
            assert r.status_code == 200 and "<html" in r.text.lower(), f"{path}: HTTP {r.status_code}"
        for asset in ("/static/app.css", "/static/theme.js", "/static/warmup.js", "/static/nav.js"):
            assert anon.get(asset).status_code == 200, asset
        return "8 pages, 4 assets"
    check("every page and shared asset loads", pages)

    def health():
        h = anon.get("/health").json()
        assert h["status"] == "ok"
        return f"database {h['database']}, durable {h['durable']}" + \
            (f", warnings: {h['warnings']}" if h["warnings"] else "")
    check("health", health)

    def router_ready():
        start = time.time()
        while time.time() - start < 180:
            s = anon.get("/api/status", params={"wait": 20}, timeout=40).json()
            if s["ready"]:
                return f"ready after waiting {time.time() - start:.0f}s" + \
                    (f" (model loaded in {s['load_s']}s)" if s.get("load_s") else "")
            if s["stage"] == "failed":
                raise AssertionError(s["error"])
        raise AssertionError("not ready after 3 minutes")
    check("router finishes starting", router_ready)

    # --- public endpoints ------------------------------------------------------
    def routing_map():
        r = anon.get("/api/routing").json()
        assert r["bands"] == ["easy", "medium", "hard", "expert"], r["bands"]
        return " ".join(f"{b}->{r['map'][b]}" for b in r["bands"])
    check("public routing map", routing_map)

    def demo_endpoints():
        q = anon.get("/api/try/quota").json()
        c = anon.post("/api/try/classify", json={"content": "What is the capital of France?"}).json()
        assert c["difficulty"] in ("easy", "medium", "hard", "expert")
        return f"demo quota {q['remaining']}/{q['limit']} left; classify -> {c['difficulty']}/{c['tier']}"
    check("demo quota and classify", demo_endpoints)

    # --- account ------------------------------------------------------------
    def account():
        if args.username:
            r = http.post("/auth/login", json={"username": args.username, "password": args.password})
            assert r.status_code == 200, r.text
            return f"signed in as {args.username}"
        user = f"e2e-{secrets.token_hex(4)}"
        r = http.post("/auth/signup", json={"username": user, "password": secrets.token_urlsafe(12)})
        assert r.status_code == 200, r.text
        assert http.get("/auth/me").json()["username"] == user
        return f"signed up {user} (left in the database)"
    check("sign up or sign in", account)

    def chat_flow():
        chat_id = http.post("/api/chats", json={}).json()["id"]
        state["chat_id"] = chat_id
        p = http.post("/api/classify", json={"content": "What is the capital of Australia?",
                                             "chat_id": chat_id}).json()
        done = stream_chat(http, chat_id, "What is the capital of Australia?")
        assert "canberra" in done["text"].lower(), done["text"][:200]
        for ev in ("routing", "tier_start", "token", "done"):
            assert ev in done["_events"], f"no {ev} event"
        t = done["trace"]
        for part in ("classification", "start", "attempts", "totals"):
            assert t.get(part), f"trace has no {part}"
        state["first"] = done
        return (f"predicted {p['tier']}; answered by {done['final_tier']} "
                f"(band {done['difficulty']}) for ${done['total_cost']:.5f} in {done['total_latency_ms'] / 1000:.1f}s")
    check("chat: stream an answer with its route", chat_flow)

    def stored_route():
        chat = http.get(f"/api/chats/{state['chat_id']}").json()
        msgs = chat["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"], [m["role"] for m in msgs]
        assert msgs[1]["trace"] and msgs[1]["trace"]["attempts"], "route not stored"
        return f"{len(msgs[1]['trace']['attempts'])} attempt(s) recorded"
    check("chat: route is stored with the answer", stored_route)

    def second_turn():
        done = stream_chat(http, state["chat_id"],
                           "A train travels 240 miles in 3 hours. At the same speed, how far does it go in 5 hours?")
        assert "400" in done["text"], done["text"][:200]
        return f"{done['difficulty']} -> {done['final_tier']}, ${done['total_cost']:.5f}"
    check("chat: a follow-up in the same chat", second_turn)

    if args.expert:
        def expert():
            done = stream_chat(http, state["chat_id"],
                               "Let $S$ be the set of positive integers $n < 2000$ such that $n^2 + 3n + 7$ is "
                               "divisible by $13$. Find the remainder when the sum of the elements of $S$ is "
                               "divided by $1000$.")
            assert done["difficulty"] == "expert", done["difficulty"]
            return f"answered by {done['final_tier']} for ${done['total_cost']:.4f} in {done['total_latency_ms'] / 1000:.0f}s"
        check("chat: competition maths starts at the frontier", expert)

    def limits():
        d = http.get("/api/chat-stats").json()["daily"]
        assert d["used"] >= 2, d
        assert d["tokens_used"] > 0, "tokens weren't counted"
        return f"{d['used']}/{d['limit']} prompts, {d['tokens_used']:,}/{d['token_limit']:,} tokens today"
    check("daily prompt and token counters", limits)

    def sessions():
        s = http.get("/api/sessions").json()
        assert any(x["type"] == "chat" and x["id"] == state["chat_id"] for x in s), "chat not in sessions"
        return f"{len(s)} session(s)"
    check("sessions overview", sessions)

    # --- API keys -----------------------------------------------------------------
    def make_key():
        k = http.post("/api/keys", json={"name": "e2e check"}).json()
        state["key"], state["key_id"] = k["key"], k["id"]
        return k["key_prefix"] + "..."
    check("create an API key", make_key)

    api = httpx.Client(base_url=base, timeout=300, headers={"Authorization": f"Bearer {state.get('key', '')}"})

    def openai_endpoint():
        before = http.get("/api/chat-stats").json()["daily"]["used"]
        r = api.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "What is 17 * 23?"}]})
        assert r.status_code == 200, r.text[:200]
        body = r.json()
        assert "391" in body["choices"][0]["message"]["content"]
        assert body["_router"].get("trace"), "no trace in _router"
        after = http.get("/api/chat-stats").json()["daily"]["used"]
        assert after == before + 1, f"prompt count went {before} -> {after}; the call wasn't counted"
        return f"answered by {body['_router']['final_tier']}, counted toward the daily limit"
    check("OpenAI-compatible endpoint with the key", openai_endpoint)

    def v1_api():
        c = api.post("/api/v1/classify", json={"prompt": "Explain the CAP theorem"}).json()
        m = api.post("/api/v1/routing-map", json={}).json()
        models = api.get("/api/v1/models").json()
        assert c["trace"]["start"]["tiers"], c
        return f"classify -> {c['band']}/{c['tier']}; map {m['map']}; {len(models['models'])} calibrated model(s)"
    check("v1 API: classify, routing map, models", v1_api)

    def revoke():
        http.delete(f"/api/keys/{state['key_id']}")
        r = api.post("/api/v1/classify", json={"prompt": "hi"})
        assert r.status_code == 401, f"revoked key still works (HTTP {r.status_code})"
        return "revoked key is refused"
    check("revoke the key", revoke)

    def logout():
        http.post("/auth/logout")
        http.cookies.clear()
        assert http.get("/auth/me").status_code == 401
    check("log out", logout)

    if args.browser:
        browser_checks(base)

    passed = sum(ok for _, ok, _ in results)
    print(f"\n{passed} of {len(results)} checks passed.")
    for name, ok, detail in results:
        if not ok:
            print(f"  failed: {name}: {detail}")
    sys.exit(0 if passed == len(results) else 1)


def browser_checks(base: str) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        results.append(("browser checks", False, "pip install playwright"))
        print("  FAIL  browser checks: pip install playwright")
        return
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1280, "height": 860})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))

        def landing():
            page.goto(base + "/", wait_until="networkidle")
            page.wait_for_function("window.routerReady === true", timeout=180000)
            page.wait_for_function("document.querySelector('#live-band .diff')", timeout=20000)
            rows = page.locator("#matrix tbody tr").count()
            assert rows == 4, f"routing table has {rows} rows"
            return "live panel and routing table render"
        check("browser: landing page", landing)

        def theme():
            before = page.evaluate("document.documentElement.getAttribute('data-theme')")
            page.click("[data-theme-toggle]")
            after = page.evaluate("document.documentElement.getAttribute('data-theme')")
            assert after in ("light", "dark") and after != before, f"{before} -> {after}"
            page.reload(wait_until="networkidle")
            kept = page.evaluate("document.documentElement.getAttribute('data-theme')")
            assert kept == after, "choice wasn't remembered"
            return f"switched to {after}, kept after reload"
        check("browser: light / dark toggle", theme)

        def explorer():
            page.goto(base + "/demo", wait_until="networkidle")
            page.fill("#prompt", "What is the capital of Japan?")
            page.wait_for_selector("#result:not([hidden])", timeout=60000)
            return page.inner_text("#r-band") + " -> " + page.inner_text("#r-tier").split()[0]
        check("browser: routing explorer", explorer)

        def no_errors():
            assert not errors, errors[:3]
        check("browser: no script errors", no_errors)
        browser.close()


if __name__ == "__main__":
    main()
