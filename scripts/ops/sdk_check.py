"""Check the Python SDK against a running ThriftLLM server, end to end.

Uses your app's API key (Settings -> Create key) and real provider keys, so
it calibrates and routes for real. Each step prints PASS or FAIL and the
script carries on, so one run shows everything that works and everything
that doesn't.

    export THRIFTLLM_API_KEY=rtr_...
    export GROQ_API_KEY=...              # or whichever providers your models use
    ./venv/bin/python -m scripts.ops.sdk_check --base-url https://your-host

By default it uses two Groq models (Groq's free tier covers them):
    cheap = groq/openai/gpt-oss-20b, mid = groq/openai/gpt-oss-120b
Pick your own with --cheap / --mid / --frontier. A model that is already
calibrated is reused; --force-calibrate re-runs it (24 real answers each).

Provider keys are read from the usual variable for each model's prefix
(GROQ_API_KEY for groq/..., OPENAI_API_KEY for openai/..., and so on), or
pass them directly: --key cheap=sk-... .
"""
import argparse
import os
import sys
import time
import traceback
from pathlib import Path

try:
    import thriftllm
except ImportError:          # not installed: use the copy in this repo
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sdk" / "python"))
    import thriftllm

PROVIDER_ENV = {
    "groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY", "openrouter": "OPENROUTER_API_KEY",
    "gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "mistral": "MISTRAL_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY", "together_ai": "TOGETHERAI_API_KEY",
}
PROMPTS = [
    ("easy", "What is the capital of Australia?", ["canberra"]),
    ("medium", "A train travels 240 miles in 3 hours. At the same speed, how far does it go in 5 hours?", ["400"]),
]

results = []


def step(name):
    def wrap(fn):
        def run(*a, **kw):
            start = time.perf_counter()
            try:
                out = fn(*a, **kw)
                results.append((name, True, ""))
                print(f"  PASS  {name}  ({time.perf_counter() - start:.1f}s)")
                return out
            except Exception as e:
                detail = f"{type(e).__name__}: {e}"
                results.append((name, False, detail))
                print(f"  FAIL  {name}\n        {detail}")
                if os.getenv("SDK_CHECK_TRACEBACK"):
                    traceback.print_exc()
                return None
        return run
    return wrap


def key_for(model: str, overrides: dict, tier: str) -> str | None:
    if tier in overrides:
        return overrides[tier]
    env = PROVIDER_ENV.get(model.split("/", 1)[0])
    return os.getenv(env) if env else None


def main():
    sys.stdout.reconfigure(line_buffering=True)    # show progress as it happens, even when piped
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default=os.getenv("THRIFTLLM_BASE_URL", "http://localhost:8000"))
    ap.add_argument("--api-key", default=os.getenv("THRIFTLLM_API_KEY"))
    ap.add_argument("--cheap", default="groq/openai/gpt-oss-20b")
    ap.add_argument("--mid", default="groq/openai/gpt-oss-120b")
    ap.add_argument("--frontier", default=None)
    ap.add_argument("--key", action="append", default=[], metavar="TIER=KEY",
                    help="provider key for a tier, overriding the environment")
    ap.add_argument("--force-calibrate", action="store_true")
    args = ap.parse_args()
    if not args.api_key:
        sys.exit("Set THRIFTLLM_API_KEY or pass --api-key rtr_... (Settings -> Create key in the app).")
    overrides = dict(k.split("=", 1) for k in args.key)
    tiers = {t: m for t, m in (("cheap", args.cheap), ("mid", args.mid), ("frontier", args.frontier)) if m}

    print(f"ThriftLLM SDK check against {args.base_url}")
    print("  models: " + ", ".join(f"{t}={m}" for t, m in tiers.items()) + "\n")
    client = thriftllm.Client(api_key=args.api_key, base_url=args.base_url)

    @step("connect with the API key and list calibrated models")
    def connect():
        known = client.models()
        print(f"        {len(known)} calibrated model(s) on this account")
        return known

    connect()

    calibrated = {}
    for tier, model in tiers.items():
        @step(f"calibrate {tier}: {model}")
        def cal(tier=tier, model=model):
            key = key_for(model, overrides, tier)
            if key is None:
                print(f"        no key found for {model}; set {PROVIDER_ENV.get(model.split('/')[0], 'its key')}")
            m = client.calibrate(model, api_key=key, force=args.force_calibrate)
            print(f"        {m}")
            for w in m.warnings:
                print(f"        warning: {w}")
            assert m.quality, "no scores came back"
            return m
        m = cal()
        if m is not None:
            calibrated[tier] = m

    if not calibrated:
        print("\nNothing calibrated, so routing can't be checked.")
        return summary()

    router = thriftllm.Router(**calibrated, client=client)

    @step("routing map")
    def route_map():
        rm = router.route_map
        print(f"        {rm}")
        assert set(rm) == {"easy", "medium", "hard", "expert"}

    route_map()

    @step("classify without calling a model")
    def classify():
        for _, prompt, _ in PROMPTS:
            c = router.classify(prompt)
            print(f"        {c['band']:<7} -> starts at {c['tier']}:  {prompt[:60]}")
            assert c["trace"]["classification"]["neighbours"]

    classify()

    last = {}
    for band, prompt, expect in PROMPTS:
        @step(f"chat ({band}): {prompt[:48]}")
        def chat(prompt=prompt, expect=expect):
            r = router.chat(prompt)
            print(f"        {r.text.strip()[:110]!r}")
            print(f"        band {r.band}, answered by {r.tier}{' after escalating' if r.escalated else ''}, "
                  f"${r.cost:.5f}, {r.latency_ms / 1000:.1f}s, saved ${r.saved:.5f}")
            assert r.text.strip(), "empty answer"
            assert any(e in r.text.lower() for e in expect), f"expected one of {expect} in the answer"
            assert r.trace and r.trace.get("attempts"), "no route trace"
            last["reply"] = r
        chat()

    @step("explain the route")
    def explain():
        text = last["reply"].explain()
        print("        " + text.replace("\n", "\n        "))
        assert "Started at" in text

    if "reply" in last:
        explain()

    @step("a bad key is refused")
    def bad_key():
        try:
            thriftllm.Client(api_key="rtr_not-a-real-key", base_url=args.base_url).models()
        except thriftllm.ThriftLLMError as e:
            assert e.status == 401, e
            return
        raise AssertionError("the server accepted a made-up key")

    bad_key()
    return summary()


def summary():
    passed = sum(ok for _, ok, _ in results)
    print(f"\n{passed} of {len(results)} checks passed.")
    for name, ok, detail in results:
        if not ok:
            print(f"  failed: {name}: {detail}")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
