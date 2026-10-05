"""The ThriftLLM client: calibrate models, build a router from them, send
prompts through it.

Everything runs on a ThriftLLM server; this module only talks to it. Your
provider keys stay in this process and travel with each request. The server
uses them for that one call and never stores them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://localhost:8000"
TIERS = ("cheap", "mid", "frontier")
FLOOR = 0.80

# Calibration runs 24 graded questions against the model, one after another,
# so a thinking model can take several minutes. Routing is faster but a
# frontier answer to a hard problem can still take two.
CALIBRATE_TIMEOUT_S = 900
ROUTE_TIMEOUT_S = 300


class ThriftLLMError(Exception):
    """The server refused a request. `status` is the HTTP status and the
    message is the server's own explanation."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"{detail} (HTTP {status})")
        self.status = status
        self.detail = detail


@dataclass
class CalibratedModel:
    """A model with its measured quality and cost on each difficulty band.

    Pass it to a Router as a tier. The provider key it was calibrated with
    is kept here, in memory only, so the router can send it with each
    request."""

    name: str
    quality: dict[str, float | None]
    cost: dict[str, float | None]
    latency_ms: float | None = None
    n_queries: int | None = None
    reused: bool = False
    warnings: list[str] = field(default_factory=list)
    api_key: str | None = field(default=None, repr=False)

    def clears(self, band: str) -> bool:
        """Whether this model scored at least 80% on the band, which is
        what lets a band start at it."""
        q = self.quality.get(band)
        return q is not None and q >= FLOOR

    def __str__(self) -> str:
        scores = "  ".join(
            f"{band} {'--' if q is None else f'{q:.0%}'}{'' if q is None or q >= FLOOR else ' (below 80%)'}"
            for band, q in self.quality.items())
        how = "reused calibration" if self.reused else f"{self.n_queries} questions"
        return f"{self.name}: {scores}  [{how}]"


@dataclass
class Reply:
    """One answer from the router, with where it went and what it cost."""

    text: str
    tier: str
    band: str
    started_at: str
    escalated: bool
    cost: float
    baseline_cost: float | None
    latency_ms: float
    escalation_reasons: list[str]
    trace: dict[str, Any] | None
    raw: dict[str, Any] = field(repr=False)

    @property
    def saved(self) -> float:
        """What the strongest tier in the router would have cost, minus
        what this answer cost."""
        return max(0.0, (self.baseline_cost or 0.0) - self.cost)

    def explain(self) -> str:
        """The route as readable text: the band and the examples that put
        it there, where it started and why, each attempt and its check,
        and the cost."""
        return explain_trace(self.trace, self.band, self.tier)

    def __str__(self) -> str:
        return self.text


class Router:
    """Routes prompts across up to three calibrated models.

        router = Router(cheap=small, mid=medium, frontier=large)
        reply = router.chat("What is 2+2?")

    With no models it uses the server's built-in stack -- the models the
    ThriftLLM chat app runs on -- so nothing needs calibrating:

        router = Router()

    Each difficulty band starts at the cheapest model that scored at least
    80% on it, priced so that checking and escalating are included. The
    cheapest model's answers are checked by a judge, and a failed check
    moves the question up a tier.

    mode="direct" skips the cheapest tier and its judge, which is faster
    when the middle model is already cheap.
    """

    def __init__(self, cheap: CalibratedModel | None = None, mid: CalibratedModel | None = None,
                 frontier: CalibratedModel | None = None, *, mode: str = "cascade",
                 client: "Client | None" = None):
        self.tiers = {t: m for t, m in (("cheap", cheap), ("mid", mid), ("frontier", frontier)) if m}
        for tier, m in self.tiers.items():
            if not isinstance(m, CalibratedModel):
                raise TypeError(f"{tier} must be a CalibratedModel from calibrate(), got {type(m).__name__}")
        if mode not in ("cascade", "direct"):
            raise ValueError("mode must be 'cascade' or 'direct'")
        self.mode = mode
        self.client = client or default_client()

    def _models(self) -> dict[str, str]:
        return {t: m.name for t, m in self.tiers.items()}

    def _keys(self) -> dict[str, str]:
        return {t: m.api_key for t, m in self.tiers.items() if m.api_key}

    def chat(self, prompt: str | list[dict[str, str]]) -> Reply:
        """Send a prompt (or a list of {"role", "content"} messages) through
        the cascade and return the answer with its route."""
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        if self.tiers:
            data = self.client._post("/api/v1/route", {
                "messages": messages, "models": self._models(), "tier_api_keys": self._keys(),
                "routing_mode": self.mode,
            }, timeout=ROUTE_TIMEOUT_S)
        else:   # the built-in stack, through the OpenAI-compatible endpoint
            data = self.client._post("/v1/chat/completions", {
                "messages": messages, "routing_mode": self.mode,
            }, timeout=ROUTE_TIMEOUT_S)
        r = data.get("_router", {})
        return Reply(
            text=data["choices"][0]["message"]["content"], tier=r.get("final_tier"),
            band=r.get("difficulty"), started_at=r.get("initial_tier"),
            escalated=bool(r.get("escalated")), cost=r.get("cost") or 0.0,
            baseline_cost=r.get("baseline_cost"), latency_ms=r.get("latency_ms") or 0.0,
            escalation_reasons=r.get("escalation_reasons") or [], trace=r.get("trace"), raw=data,
        )

    def classify(self, prompt: str) -> dict[str, Any]:
        """Where a prompt would start, and why, without calling any model.
        Free, and not counted against the daily limit."""
        return self.client._post("/api/v1/classify", {
            "prompt": prompt, "models": self._models(), "routing_mode": self.mode,
        })

    @property
    def route_map(self) -> dict[str, str]:
        """Which tier each difficulty band starts at. Asked of the server,
        which derives it from the models' calibration -- the cheapest tier
        scoring 80% on the band, with checking and escalation priced in --
        exactly as it will when routing."""
        return self.client._post("/api/v1/routing-map", {"models": self._models()})["map"]

    def __repr__(self) -> str:
        tiers = ", ".join(f"{t}={m.name}" for t, m in self.tiers.items()) or "built-in stack"
        return f"Router({tiers}, mode={self.mode!r})"


class Client:
    """A connection to a ThriftLLM server, authenticated with an API key
    from its Settings page (rtr_...)."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 http: httpx.Client | None = None):
        self.api_key = api_key or os.getenv("THRIFTLLM_API_KEY")
        if not self.api_key:
            raise ValueError("no API key: pass api_key='rtr_...' or set THRIFTLLM_API_KEY "
                             "(create one under Settings in the ThriftLLM app)")
        self.base_url = (base_url or os.getenv("THRIFTLLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self._http = http or httpx.Client(base_url=self.base_url)

    def _request(self, method: str, path: str, body: dict | None = None,
                 timeout: float = 60) -> dict[str, Any]:
        resp = self._http.request(method, path, json=body, timeout=timeout,
                                  headers={"Authorization": f"Bearer {self.api_key}"})
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError:
                detail = resp.text
            raise ThriftLLMError(resp.status_code, str(detail))
        return resp.json()

    def _post(self, path: str, body: dict, timeout: float = 60) -> dict[str, Any]:
        return self._request("POST", path, body, timeout)

    def models(self) -> dict[str, CalibratedModel]:
        """Every model this account has calibrated, by name. Their provider
        keys aren't known here; pass api_key to calibrate() to attach one."""
        data = self._request("GET", "/api/v1/models")
        return {m["model_name"]: _model_from(m, reused=True) for m in data["models"]}

    def calibrate(self, model: str, api_key: str | None = None, *, provider: str | None = None,
                  api_base: str | None = None, force: bool = False) -> CalibratedModel:
        """Measure a model on four difficulty bands and return it, ready to
        be a router tier.

        model     litellm's provider/model form: "groq/openai/gpt-oss-20b"
        api_key   the provider's key; used for this run and kept in memory
                  for routing, never stored by the server
        provider  only if the model name has no provider prefix
        api_base  for a self-hosted endpoint such as a local Ollama
        force     re-run even if this model is already calibrated

        A model calibrated before is returned straight away with its stored
        numbers: calibration runs 24 real questions on your key, and a
        thinking model can spend about a dollar on them."""
        if not force:
            known = self.models().get(model)
            if known is not None and known.quality:
                known.api_key = api_key
                return known
        data = self._post("/api/v1/calibrate", {
            "model_name": model, "api_key": api_key, "provider": provider, "api_base": api_base,
        }, timeout=CALIBRATE_TIMEOUT_S)
        stats = data["stats"]
        return CalibratedModel(
            name=data["model_name"], quality=stats["quality_by_difficulty"],
            cost=stats["cost_by_difficulty"], latency_ms=stats.get("avg_latency_ms"),
            n_queries=stats.get("n_queries"), warnings=data.get("warnings") or [], api_key=api_key,
        )

    def router(self, cheap: CalibratedModel | None = None, mid: CalibratedModel | None = None,
               frontier: CalibratedModel | None = None, *, mode: str = "cascade") -> Router:
        return Router(cheap=cheap, mid=mid, frontier=frontier, mode=mode, client=self)

    def close(self) -> None:
        self._http.close()


def _model_from(m: dict, reused: bool) -> CalibratedModel:
    return CalibratedModel(
        name=m["model_name"], quality=m.get("quality_by_difficulty") or {},
        cost=m.get("cost_by_difficulty") or {}, latency_ms=m.get("avg_latency_ms"),
        n_queries=m.get("n_queries"), reused=reused,
    )


# --- module-level shortcuts: calibrate(...), Router(...) -------------------

_default: Client | None = None


def configure(api_key: str | None = None, base_url: str | None = None) -> Client:
    """Set the server and key that calibrate() and Router() use."""
    global _default
    _default = Client(api_key=api_key, base_url=base_url)
    return _default


def default_client() -> Client:
    global _default
    if _default is None:
        _default = Client()
    return _default


def calibrate(model: str, api_key: str | None = None, **kwargs) -> CalibratedModel:
    """calibrate("groq/openai/gpt-oss-20b", api_key=...) on the default
    client; see Client.calibrate."""
    return default_client().calibrate(model, api_key, **kwargs)


# --- reading a route -------------------------------------------------------

def _pct(v: float | None) -> str:
    return "n/m" if v is None else f"{v:.0%}"


def explain_trace(trace: dict | None, band: str | None = None, tier: str | None = None) -> str:
    if not trace:
        return f"classified {band}, answered by {tier} (the server sent no route detail)"
    lines = []
    c = trace.get("classification") or {}
    lines.append(f"1. Classified {c.get('band', band)} in {c.get('ms', '?')}ms, without calling a model.")
    for n in (c.get("neighbours") or [])[:3]:
        lines.append(f"     {n['similarity']:.2f}  {n['band']:<7} {n.get('text', '')[:70]}")
    if c.get("override"):
        lines.append(f"   Override: the vote said {c.get('embedding_band')}, {c['override']}.")
    s = trace.get("start") or {}
    lines.append(f"2. Started at {s.get('tier', s.get('chosen'))}.")
    for r in s.get("tiers") or []:
        mark = "clears" if r.get("clears_floor") else "below"
        cost = r.get("expected_cost")
        lines.append(f"     {r['tier']:<9} {_pct(r.get('quality')):>4} on {s.get('band')} ({mark} 80%)"
                     + (f", expected ${cost:.5f}" if cost is not None else "")
                     + ("   <- chosen" if r["tier"] == s.get("chosen") else ""))
    if s.get("direct"):
        lines.append("   Direct mode skipped the cheapest tier.")
    h = s.get("history") or {}
    if h.get("tokens"):
        warm = ", ".join(h.get("warm") or [])
        lines.append(f"   History: {h['tokens']:,} tokens re-sent, priced into each tier"
                     + (f"; {warm} still has it cached, at the cached rate." if warm else "; no tier has it cached."))
    lines.append("3. What happened:")
    for a in trace.get("attempts") or []:
        if a.get("outcome") == "unavailable":
            lines.append(f"     {a['tier']}: unavailable ({a.get('reason')}), skipped")
            continue
        ch = a.get("check") or {}
        detail = ch.get("kind", "")
        if ch.get("p_correct") is not None:
            detail += f", P(correct) {ch['p_correct']:.2f} vs {ch.get('accept_threshold')}"
        if ch.get("judge_verdict"):
            detail += f", judge: {ch['judge_verdict']}"
        cached = a.get("cached_tokens")
        lines.append(f"     {a['tier']}: ${a.get('cost', 0):.5f}, {a.get('latency_ms')}ms, "
                     + (f"{cached:,} of {a.get('tokens_in') or 0:,} input tokens cached, " if cached else "")
                     + f"{a.get('outcome')} ({detail})")
    t = trace.get("totals") or {}
    if t:
        lines.append(f"4. Cost ${t.get('cost', 0):.5f} against ${t.get('baseline', 0):.5f} "
                     f"for the strongest tier: saved ${t.get('saved', 0):.5f}.")
    return "\n".join(lines)
