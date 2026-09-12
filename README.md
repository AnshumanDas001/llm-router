# LLM Router — Week 1

Cost/latency router that forwards to the cheapest model tier that can handle a
request. Week 1 status: single-tier passthrough (frontier-only), full request
logging, baseline dataset collection.

## Setup

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY and GEMINI_API_KEY
ollama pull llama3.2:3b
ollama serve            # if not already running
```

## Run the router

```bash
./venv/bin/uvicorn app.main:app --port 8000
```

Test it:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}]}'
```

Every request is logged to `logs/router.db` (SQLite): query, model_used,
tokens_in/out, cost, latency_ms, timestamp.

## Run the Week 1 baseline

With the server running and `GEMINI_API_KEY` set (default tier is
`frontier` = Gemini Flash):

```bash
./venv/bin/python3 scripts/run_baseline.py
```

This sends all 60 queries in `data/sample_queries.json` through the
frontier-only baseline and prints total cost / avg latency — the reference
numbers every later "cost saved" claim compares against.

## Tiers (app/model_config.py)

| name | model | provider |
|---|---|---|
| cheap | llama3.2:3b | Ollama (local) |
| mid | openai/gpt-oss-20b | Groq |
| frontier | gemini-3.6-flash | Google AI Studio |
