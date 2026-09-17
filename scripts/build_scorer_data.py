"""Generate labelled training data for the learned answer scorer.

The first scorer (embeddings of the answer text -> logistic regression)
scored ROC-AUC 0.66: a right answer and a wrong answer to the same
question embed almost identically, so there was nothing to learn from.
This builds a dataset with signals that actually vary with correctness:

  logprobs      the cheap model's own token probabilities on its answer.
                Wrong answers tend to be produced with lower confidence
                (Kadavath et al. 2022, "LMs (Mostly) Know What They Know").
                Free: metadata on the call we already make.
  consistency   the same question sampled again. Wrong answers are
                stochastic, right ones are stable (SelfCheckGPT, Manakul et
                al. 2023). Costs one extra *cheap* generation, no judge.
  self-verify   the cheap model asked "is this correct?", with P(YES)
                read off the logprobs rather than parsed from text.

Every query is sampled K times; each sample becomes one row whose
consistency features compare it with its siblings. Labels: the 76
auto-scoreable queries are scored by app/scoring.py; the 40 open-ended
ones are labelled by the mid judge -- a one-time labelling cost at
training time, not a runtime one, which is exactly FrugalGPT's setup.

Talks to Ollama's OpenAI-compatible endpoint directly because litellm
refuses to forward `logprobs` to its ollama provider. Appends to a JSONL
so an interrupted run resumes where it stopped.

    ./venv/bin/python scripts/build_scorer_data.py
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from app.model_config import CHEAP_MODEL, MID_MODEL  # noqa: E402
from app.scoring import score_one  # noqa: E402
from app.verifier import _verify_llm_judge  # noqa: E402

OUT_PATH = ROOT / "data" / "scorer_training.jsonl"
OLLAMA_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
SAMPLES = {"easy": 3, "medium": 3, "hard": 2}   # hard answers are long and slow
MAX_TOKENS = 1024

SELF_VERIFY_PROMPT = """Question:
{query}

Proposed answer:
{answer}

Is the proposed answer correct and complete? Reply with one word: YES or NO."""


def _ollama_chat(messages, max_tokens, top_logprobs=5, temperature=None):
    body = {"model": CHEAP_MODEL.split("/", 1)[1], "messages": messages,
            "max_tokens": max_tokens, "logprobs": True, "top_logprobs": top_logprobs}
    if temperature is not None:
        body["temperature"] = temperature
    r = httpx.post(f"{OLLAMA_BASE}/v1/chat/completions", json=body, timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]


def generate(query: str) -> dict:
    start = time.perf_counter()
    choice = _ollama_chat([{"role": "user", "content": query}], MAX_TOKENS)
    latency = time.perf_counter() - start
    text = choice["message"]["content"] or ""
    toks = (choice.get("logprobs") or {}).get("content") or []
    # Per token: chosen logprob and the entropy of the top-k alternatives.
    # Entropy over top-5 is a floor on the true entropy, good enough to
    # tell "one obvious continuation" from "several plausible ones".
    chosen, entropy = [], []
    for t in toks:
        chosen.append(t["logprob"])
        ps = [math.exp(a["logprob"]) for a in t.get("top_logprobs") or []]
        entropy.append(-sum(p * math.log(p) for p in ps if p > 0))
    return {"text": text, "logprobs": chosen, "entropy": entropy,
            "finish_reason": choice.get("finish_reason"), "latency_s": latency}


def self_verify(query: str, answer: str) -> float:
    """P(YES) from the cheap model's first-token distribution."""
    prompt = SELF_VERIFY_PROMPT.format(query=query, answer=answer[:4000])
    choice = _ollama_chat([{"role": "user", "content": prompt}], max_tokens=1,
                          top_logprobs=10, temperature=0.0)
    toks = (choice.get("logprobs") or {}).get("content") or []
    if not toks:
        return 0.5
    p_yes = p_no = 0.0
    for alt in toks[0].get("top_logprobs") or []:
        word = alt["token"].strip().upper()
        if word.startswith("YES"):
            p_yes += math.exp(alt["logprob"])
        elif word.startswith("NO"):
            p_no += math.exp(alt["logprob"])
    if p_yes + p_no == 0:
        return 0.5
    return p_yes / (p_yes + p_no)


def label(q: dict, answer: str) -> tuple[float, str]:
    if q["eval_method"] != "llm_judge":
        return score_one(q, answer), "automatic"
    verdict = _verify_llm_judge(q["query"], answer, MID_MODEL, None, "mid")
    return (1.0 if verdict["passed"] else 0.0), "judge_mid"


def main():
    queries = json.loads((ROOT / "data" / "eval_queries.json").read_text())
    done = set()
    if OUT_PATH.exists():
        for line in OUT_PATH.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["query_id"], row["sample"]))
    print(f"{len(queries)} queries, {len(done)} rows already done, cheap={CHEAP_MODEL}")

    with OUT_PATH.open("a") as out:
        for qi, q in enumerate(queries):
            k = SAMPLES[q["difficulty"]]
            todo = [s for s in range(k) if (q["id"], s) not in done]
            if not todo:
                continue
            # Regenerate all K together so every row sees the same siblings.
            samples = [generate(q["query"]) for _ in range(k)]
            for s in range(k):
                if s not in todo:
                    continue
                gen = samples[s]
                score, source = label(q, gen["text"])
                p_yes = self_verify(q["query"], gen["text"]) if gen["text"].strip() else 0.0
                row = {
                    "query_id": q["id"], "sample": s, "query": q["query"],
                    "difficulty": q["difficulty"], "task_type": q["task_type"],
                    "eval_method": q["eval_method"], "answer": gen["text"],
                    "logprobs": gen["logprobs"], "entropy": gen["entropy"],
                    "finish_reason": gen["finish_reason"], "latency_s": gen["latency_s"],
                    "siblings": [samples[j]["text"] for j in range(k) if j != s],
                    "self_verify_p_yes": p_yes, "score": score, "label_source": source,
                }
                out.write(json.dumps(row) + "\n")
                out.flush()
                mean_lp = sum(gen["logprobs"]) / max(1, len(gen["logprobs"]))
                print(f"[{qi+1:3}/{len(queries)}] {q['id']} s{s} {q['difficulty']:6} "
                      f"score={score:.1f} p_yes={p_yes:.2f} mean_lp={mean_lp:+.3f} "
                      f"toks={len(gen['logprobs'])} {gen['latency_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
