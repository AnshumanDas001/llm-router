# thriftllm

Python client for [ThriftLLM](https://github.com/AnshumanDas001/llm-router).
Calibrate your models once, put them in a router, and every prompt starts at
the cheapest model that scores at least 80% on its difficulty band. Its
answer is checked, and a failed check moves the prompt up a tier.

## Install

```bash
pip install "git+https://github.com/AnshumanDas001/llm-router.git#subdirectory=sdk/python"
```

That installs the `thriftllm` package from this folder of the repo, plus
httpx. It needs Python 3.10 or newer. To work on the SDK itself, clone the
repo and install it in editable mode instead, so your edits apply without
reinstalling:

```bash
pip install -e sdk/python
```

## Use

You need a ThriftLLM server to talk to (yours, or `./venv/bin/uvicorn
app.main:app` locally) and an API key from its **Settings** page. Then:

```python
import os
import thriftllm

GROQ_KEY = os.environ["GROQ_API_KEY"]

thriftllm.configure(api_key="rtr_...", base_url="https://your-host")
small  = thriftllm.calibrate("groq/openai/gpt-oss-20b",  api_key=GROQ_KEY)
medium = thriftllm.calibrate("groq/openai/gpt-oss-120b", api_key=GROQ_KEY)
router = thriftllm.Router(cheap=small, mid=medium)
reply  = router.chat("What is the capital of Australia?")
print(reply.text, reply.tier, reply.cost)
print(reply.explain())          # the same route "Why this route?" shows
```

Each `calibrate()` runs 24 graded questions against the model on your key
(easy, medium, hard and AIME-level expert) and returns its score on each
band. A model you've calibrated before comes back immediately; pass
`force=True` to re-run. `print(small)` shows the scores, and
`print(router.route_map)` shows which tier each band starts at. Add a third
model with `frontier=` for the questions neither of these clears.

To use the server's built-in models instead, the ones its chat app runs on,
leave the models out: `router = thriftllm.Router()`. Nothing needs
calibrating, and the calls count toward your account's daily allowance.

`reply.explain()` prints the same route the chat app shows under **Why this
route?**: the labelled questions that set the band, each tier's score against
the 80% floor with its expected cost, every attempt and what checked it, and
the cost against the strongest tier.

## Reference

| | |
|---|---|
| `thriftllm.configure(api_key, base_url)` | the server and key the shortcuts use; also read from `THRIFTLLM_API_KEY` and `THRIFTLLM_BASE_URL` |
| `thriftllm.calibrate(model, api_key, *, provider=None, api_base=None, force=False)` | measure a model, or reuse its stored calibration. Returns a `CalibratedModel` |
| `thriftllm.Router(cheap=, mid=, frontier=, mode="cascade")` | up to three calibrated models; with none, the server's built-in models (the ones its chat app runs on), so nothing needs calibrating. `mode="direct"` skips the cheapest tier and its judge |
| `router.chat(prompt)` | a string, or a list of `{"role", "content"}` messages. Returns a `Reply` |
| `router.classify(prompt)` | where it would start and why, with no model call and no cost |
| `router.route_map` | which tier each band starts at |
| `Reply` | `.text`, `.tier`, `.band`, `.started_at`, `.escalated`, `.cost`, `.saved`, `.trace`, `.explain()` |
| `Client(api_key, base_url)` | the same methods on an explicit connection: `client.calibrate(...)`, `client.router(...)`, `client.models()` |

Provider keys never leave your process except inside the request that uses
them, and the server does not store them. Errors from the server raise
`thriftllm.ThriftLLMError` with its explanation, for example a model that
hasn't been calibrated or a daily limit reached.
