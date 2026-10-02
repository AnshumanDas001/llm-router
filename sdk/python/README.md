# thriftllm

Python client for [ThriftLLM](https://github.com/AnshumanDas001/llm-router).
Calibrate your models once, put them in a router, and every prompt starts at
the cheapest model that scores at least 80% on its difficulty band. Its
answer is checked, and a failed check moves the prompt up a tier.

```bash
pip install -e sdk/python        # from a checkout of the repo
```

Create an API key under **Settings** in the ThriftLLM app, then:

```python
import os
import thriftllm

thriftllm.configure(api_key="rtr_...", base_url="https://your-thriftllm-host")

# Each call runs 24 graded questions against the model on your key (easy,
# medium, hard and AIME-level expert) and returns its score on each band.
# A model you've calibrated before comes back immediately; pass force=True to re-run.
small = thriftllm.calibrate("groq/llama-3.1-8b-instant", api_key=os.environ["GROQ_API_KEY"])
medium = thriftllm.calibrate("groq/openai/gpt-oss-20b", api_key=os.environ["GROQ_API_KEY"])
large = thriftllm.calibrate("openai/gpt-5", api_key=os.environ["OPENAI_API_KEY"])
print(small)   # groq/llama-3.1-8b-instant: easy 83%  medium 83%  hard 50% (below 80%)  expert 0% (below 80%)  [24 questions]

router = thriftllm.Router(cheap=small, mid=medium, frontier=large)
print(router.route_map)   # {'easy': 'cheap', 'medium': 'cheap', 'hard': 'mid', 'expert': 'frontier'}

reply = router.chat("What is the capital of Australia?")
print(reply.text)                          # Canberra.
print(reply.tier, reply.band, reply.cost)  # cheap easy 2.1e-05
print(reply.explain())                     # the full route, step by step
```

`reply.explain()` prints the same route the chat app shows under **Why this
route?**: the labelled questions that set the band, each tier's score against
the 80% floor with its expected cost, every attempt and what checked it, and
the cost against the strongest tier.

## Reference

| | |
|---|---|
| `thriftllm.configure(api_key, base_url)` | the server and key the shortcuts use; also read from `THRIFTLLM_API_KEY` and `THRIFTLLM_BASE_URL` |
| `thriftllm.calibrate(model, api_key, *, provider=None, api_base=None, force=False)` | measure a model, or reuse its stored calibration. Returns a `CalibratedModel` |
| `thriftllm.Router(cheap=, mid=, frontier=, mode="cascade")` | at least one tier. `mode="direct"` skips the cheapest tier and its judge |
| `router.chat(prompt)` | a string, or a list of `{"role", "content"}` messages. Returns a `Reply` |
| `router.classify(prompt)` | where it would start and why, with no model call and no cost |
| `router.route_map` | which tier each band starts at |
| `Reply` | `.text`, `.tier`, `.band`, `.started_at`, `.escalated`, `.cost`, `.saved`, `.trace`, `.explain()` |
| `Client(api_key, base_url)` | the same methods on an explicit connection: `client.calibrate(...)`, `client.router(...)`, `client.models()` |

Provider keys never leave your process except inside the request that uses
them, and the server does not store them. Errors from the server raise
`thriftllm.ThriftLLMError` with its explanation, for example a model that
hasn't been calibrated or a daily limit reached.
