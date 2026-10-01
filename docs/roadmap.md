# 4-Week Roadmap: LLM Cost/Latency Router (Model Cascade)

**What this is:** An LLM gateway/router service — infrastructure that sits underneath any LLM-calling application and automatically routes each request to the cheapest model tier that can handle it, escalating to a stronger model only when a verification check fails. Not a chatbot. Think Nginx/load-balancer, but for model selection instead of servers.

**What you'll have at the end:** A working router API, a labeled eval set with real cost/quality numbers across three free-tier models, a documented cascade-escalation mechanism, honest failure-mode analysis, and a minimal demo showing routing decisions live.

**Time budget:** ~8-12 hrs/week, 4 weeks. This is a genuine compression from a more comfortable 5-week pace — the trade is either more weekly hours or a smaller eval set (both options noted below).

**Cost:** $0. Entire stack runs on free tiers.

---

## Locked-in Tier Setup

| Tier | Model | Access | Role |
|---|---|---|---|
| **Cheap** | Llama 3.2 3B or Qwen 2.5 1.5B | Local via **Ollama** | Default destination for most traffic — free, unlimited, fast to iterate on |
| **Mid** | GPT-OSS-20B or Qwen3.6-27B | **Groq** free tier | First escalation step — fast inference, generous free limits |
| **Frontier (stand-in)** | Gemini Flash / Flash-Lite | **Google AI Studio** free tier (1,500 req/day, no card) | Final escalation step — strongest freely-accessible model |

All three are OpenAI-SDK-compatible, unified behind **LiteLLM** so switching/adding a tier later is a config change, not a rewrite.

**Framing note for your README:** be upfront that "frontier" here means strongest freely-accessible model, not a paid flagship. The architecture and cost/quality tradeoff story hold regardless of what sits at the top tier — and you can separately report an extrapolated savings estimate using real paid-tier list prices for context.

---

## Pre-Week 0: Basics Refresher (do this before Week 1 starts, ~3-5 hrs)

Don't skip this — it prevents mid-build stalls.

- **Read**: [RouteLLM blog/paper](https://lmsys.org/blog/2024-07-01-routellm/) — the foundational reference for classifier-based routing
- **Skim**: FrugalGPT (Stanford) — where the cascade-escalation idea comes from
- **Set up accounts**: Ollama (local install), Groq (free API key), Google AI Studio (free API key) — get all three working with a single "hello world" call each *before* Week 1 begins
- **Refresh if needed**: basic scikit-learn (`LogisticRegression` is likely all you need) and `sentence-transformers` for embeddings
- Write, in your own words, one paragraph explaining the difference between classifier-based routing and cascade routing. If you can't, re-read before moving on.

---

## Week 1 — Baseline + Router Skeleton

**Goal:** A working "always call the strongest tier" baseline with full logging, plus the skeleton of the router API.

### Tasks
- Install and verify all three tiers work through **LiteLLM** behind one config
- Build the **router service skeleton** in FastAPI: one endpoint, shaped like a standard chat-completion API (drop-in compatible), that currently just forwards to a single hardcoded tier
- Add **logging**: every request writes `{query, model_used, tokens_in, tokens_out, cost, latency_ms, timestamp}` to a local SQLite DB
- Run 50-80 varied sample queries through the **frontier-tier-only baseline** (Gemini Flash) and record the numbers — this is your reference point for every "cost saved" and "quality retained" claim later

### End of week 1 deliverable
Working FastAPI router (single-tier passthrough for now), full request logging, and a baseline cost/latency dataset.

---

## Week 2 — Eval Set + Quality Baselines

**Goal:** A labeled dataset spanning real difficulty levels, with quality scores from all three tiers, so every later number is credible.

### Tasks
- Write **60-80 queries** (trimmed from the original 150 to fit the 4-week pace) spanning:
  - **Easy** (should route cheap): classification, short factual Q&A, basic reformatting/JSON structuring
  - **Medium**: short summarization, moderate multi-step instructions
  - **Hard** (should route frontier): multi-step reasoning, code generation/debugging, nuance-heavy tasks
- For each query, define how correctness is checked:
  - Exact-match / schema-validation where possible (structured tasks)
  - An **LLM-judge rubric** for open-ended ones — write the rubric, hand-score a subset yourself first, then check judge agreement against your own labels (don't skip this calibration step)
- Run **every query through all three tiers** and record quality scores for each — this gives you the full quality-vs-tier matrix before any routing logic exists

### If time is tight this week
Cut the query count further (40-50) rather than skipping judge calibration — a smaller, well-calibrated eval set beats a larger, uncalibrated one.

### End of week 2 deliverable
Labeled eval set with quality scores for all three tiers on every query.

---

## Week 3 — Classifier + Cascade, Built Together

**Goal:** A working router with both an upfront difficulty check and a verification-based escalation cascade — built as one combined system, not two separate milestones (this is the main compression from the 5-week version).

### Tasks
- **Upfront heuristic/classifier**: start with simple heuristics (query length, keywords suggesting multi-step reasoning, code-block presence) combined with embedding similarity to your Week 2 labeled examples — route to an initial tier guess
- **Verification/cascade layer** on top: after the cheap or mid tier responds, run a cheap check —
  - Structured tasks: does output parse and match expected schema?
  - Open-ended tasks: fast LLM-judge pass checking coherence/completeness (not full scoring, just a red-flag filter)
  - If verification fails → escalate to the next tier, log the escalation
- Wire this into the router service from Week 1, replacing the single-tier passthrough
- Run your full Week 2 eval set through the complete router and compute:
  - **% of traffic per tier**, including how much got escalated
  - **Total cost vs. Week 1 frontier-only baseline** (your headline savings number)
  - **Quality retained** vs. the frontier-only baseline

### End of week 3 deliverable
Fully working cascade router with real cost-saved and quality-retained numbers.

---

## Week 4 — Failure Analysis, Demo, and Writeup

**Goal:** The credibility layer — where does this actually break, and can you show it working live.

### Tasks
- **Find and document "cheap-model drift"** — real cases where a tier was confidently wrong and verification didn't catch it. Concrete examples, not a general statement that this can happen.
- Build a **Pareto chart**: cost vs. quality across configurations (frontier-only, cheap-only naive baseline, your router) — one chart that tells the whole story
- **Minimal demo** (keep this small — it's a showcase, not a product): a CLI or single-page client where a query goes in and the response shows, live, something like *"Routed to: Groq mid-tier · $0.0004 · 340ms"* or *"Escalated to Gemini Flash after verification failed."* This routing-decision visibility is the actual demo-worthy moment.
- **Write the README**: architecture diagram, honest numbers (cost saved %, quality retained %, documented failure cases), and an honest note on the free-tier framing (see above)
- Record a short (30-60 sec) demo clip — a simple query routing cheap, a hard query escalating

### End of week 4 deliverable
Complete repo, README with real numbers and a Pareto chart, working demo, ready to post.

---

## Scope Guardrails (protects your 4-week timeline)

- **Don't exceed 3 tiers.** More tiers = more complexity without proportional signal.
- **Don't skip Week 2's judge calibration**, even under time pressure — everything downstream depends on trusting your quality scores.
- **Don't build a polished web UI.** CLI output or a bare single page is enough — the routing decision is the interesting part, not the interface.
- **Don't try to beat RouteLLM's published numbers.** Report your own honestly, and explain the gap (smaller eval set, solo build, different model tiers) rather than overclaiming.
- **If you must cut something under time pressure, cut eval-set size before you cut judge calibration, cascade logic, or failure-mode documentation.** Those three are what make this "good quality" rather than "a working demo."

---

## Key Resources

- RouteLLM (foundational reference): https://lmsys.org/blog/2024-07-01-routellm/
- FrugalGPT (Stanford, cascade routing origin) — search for the paper, worth a skim for Week 3 design
- LiteLLM docs (multi-provider unification): https://docs.litellm.ai/
- Ollama (local models): https://ollama.com/
- Groq free tier: https://console.groq.com/
- Google AI Studio (Gemini free tier): https://ai.google.dev/

---

*One honest note going in: this timeline assumes the basics refresher (Pre-Week 0) is done first, and that you're not learning FastAPI, scikit-learn, and multi-provider API calling for the first time simultaneously with building the router. If any of those are genuinely new to you, budget extra time in Week 1 rather than compressing Week 2's eval rigor to compensate.*
