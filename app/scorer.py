"""Learned answer scorer: is this cheap-tier answer good enough to ship?

The FrugalGPT-style stand-in for the LLM judge. FrugalGPT (Chen, Zaharia,
Zou 2023) accepts or rejects a cheap model's answer with a small learned
scorer instead of a stronger LLM; ours is a logistic regression trained by
scripts/train_scorer.py on data from scripts/build_scorer_data.py.

The first version featurised the *text* of the answer (sentence embeddings)
and scored ROC-AUC 0.66 -- a right and a wrong answer to the same question
embed almost identically. This version uses signals that actually move
with correctness, all free or nearly so:

  confidence    the generating model's own token logprobs. A model that's
                wrong is usually less sure of it (Kadavath et al. 2022).
                Free: metadata on the call we already make.
  consistency   a second sample of the same question. Wrong answers are
                stochastic, right ones stable (SelfCheckGPT, Manakul et al.
                2023). Costs one extra cheap generation -- never a judge.
  self-verify   the cheap model asked "is this correct?", P(YES) read off
                the logprobs. Weak alone, useful alongside the others.

`features()` is the single definition used by both training and runtime,
so the two can't drift.

Measured against the LLM judge on the same answers, the scorer alone is
not a replacement: the judge catches 94% of wrong cheap answers, the
scorer about half. What it *is* good at is knowing when it's sure. So it
runs as a gate in front of the judge, not instead of it:

    P(correct) >= ACCEPT_THRESHOLD   ship it, no judge call
    P(correct) <  REJECT_THRESHOLD   escalate, no judge call (a judge
                                     verdict here costs more than the
                                     mid answer it would usually lead to)
    otherwise                        ask the judge

The judge still runs on the ambiguous middle, but only there, which is
what brings the per-query verification cost below the mid tier's own
answer -- the condition under which the expected-cost routing policy
sends easy and medium questions to the cheap tier at all.
"""
import math
import re
import threading
from pathlib import Path

import joblib
import numpy as np

from app.classifier import _get_model, _model_lock

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "answer_scorer.joblib"

# Which paid-for signals the runtime collects. Both are cheap-tier calls
# (never a judge); set from the ablation in train_scorer.py -- a feature the
# fitted model doesn't use isn't worth a call.
USE_SIBLING = True        # one extra cheap generation, drawn in parallel
USE_SELF_VERIFY = True    # one 1-token cheap call

# Both thresholds come from the held-out sweep in train_scorer.py, which
# saves them into the model file; these are the fallbacks if it didn't.
ACCEPT_THRESHOLD = 0.9
REJECT_THRESHOLD = 0.3

FEATURE_NAMES = [
    "mean_logprob", "min_logprob", "p10_logprob", "frac_uncertain", "head_logprob",
    "mean_entropy", "max_entropy", "log_tokens", "truncated",
    "self_verify", "consist_max", "consist_mean", "numeric_agree", "qa_cosine",
    "is_easy", "is_medium", "is_hard",
]

_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_clf = None
_clf_lock = threading.Lock()


def _numbers(text: str) -> list[float]:
    return [float(x) for x in _NUM.findall(text.replace(",", ""))[-3:]]


def _numeric_agreement(answer: str, siblings: list[str]) -> float:
    """1 if the final numbers in the answer appear in every sibling, 0 if a
    sibling disagrees, 0.5 when there's nothing numeric to compare.
    Cheap models' arithmetic failures are the classic wrong-but-fluent
    case, and two samples rarely make the *same* arithmetic slip."""
    mine = _numbers(answer)
    if not mine or not siblings:
        return 0.5
    agree = []
    for sib in siblings:
        theirs = _numbers(sib)
        if not theirs:
            continue
        agree.append(1.0 if any(math.isclose(mine[-1], t, rel_tol=1e-3) for t in theirs) else 0.0)
    return float(np.mean(agree)) if agree else 0.5


def features(query: str, answer: str, logprobs: list[float], entropy: list[float],
             siblings: list[str], self_verify_p_yes: float, difficulty: str,
             finish_reason: str | None = None, model=None) -> np.ndarray:
    """One feature row. `siblings` are other samples of the same question;
    an empty list disables the consistency features (they fall to neutral)."""
    lp = np.array(logprobs, dtype=float) if logprobs else np.array([-1.0])
    ent = np.array(entropy, dtype=float) if entropy else np.array([1.0])
    head = lp[:10]

    model = model or _get_model()
    with _model_lock:
        embs = model.encode([query, answer] + list(siblings),
                            convert_to_numpy=True, normalize_embeddings=True)
    qa_cos = float(embs[0] @ embs[1])
    if siblings:
        sims = embs[2:] @ embs[1]
        consist_max, consist_mean = float(sims.max()), float(sims.mean())
    else:
        consist_max = consist_mean = 0.5

    return np.array([
        lp.mean(), lp.min(), np.percentile(lp, 10), (lp < -1.0).mean(), head.mean(),
        ent.mean(), ent.max(), math.log1p(len(lp)), 1.0 if finish_reason == "length" else 0.0,
        self_verify_p_yes, consist_max, consist_mean,
        _numeric_agreement(answer, siblings), qa_cos,
        float(difficulty == "easy"), float(difficulty == "medium"), float(difficulty == "hard"),
    ], dtype=float)


SELF_VERIFY_PROMPT = """Question:
{query}

Proposed answer:
{answer}

Is the proposed answer correct and complete? Reply with one word: YES or NO."""

# What to ask the provider for on the cheap tier's generation so the scorer
# has confidence signals to read. Providers that can't (Groq) reject the
# request outright, so the cascade only adds these when the verifier is on.
LOGPROB_PARAMS = {"logprobs": True, "top_logprobs": 5}


def token_signals(logprob_content) -> tuple[list[float], list[float]]:
    """(chosen logprobs, top-k entropy) from an OpenAI-style logprobs.content
    list -- litellm's `choices[0].logprobs.content`, either a full response
    or one streamed chunk. Objects or dicts; the two look alike."""
    chosen, entropy = [], []
    for t in logprob_content or []:
        lp = t["logprob"] if isinstance(t, dict) else t.logprob
        tops = (t.get("top_logprobs") if isinstance(t, dict) else getattr(t, "top_logprobs", None)) or []
        ps = [math.exp(a["logprob"] if isinstance(a, dict) else a.logprob) for a in tops]
        chosen.append(lp)
        entropy.append(-sum(p * math.log(p) for p in ps if p > 0))
    return chosen, entropy


def self_verify_p_yes(complete, query: str, answer: str) -> float:
    """Ask the cheap model whether its own answer is right, via `complete`
    (messages, **params) -> litellm response. Reads P(YES) off the first
    token's distribution rather than parsing text, so "Yes." and "YES"
    and a hedged 60/40 all come through as numbers."""
    prompt = SELF_VERIFY_PROMPT.format(query=query, answer=answer[:4000])
    resp = complete([{"role": "user", "content": prompt}],
                    max_tokens=1, logprobs=True, top_logprobs=10, temperature=0.0)
    content = getattr(resp.choices[0].logprobs, "content", None) if resp.choices[0].logprobs else None
    if not content:
        return 0.5
    p_yes = p_no = 0.0
    for alt in getattr(content[0], "top_logprobs", None) or []:
        word = alt.token.strip().upper()
        if word.startswith("YES"):
            p_yes += math.exp(alt.logprob)
        elif word.startswith("NO"):
            p_no += math.exp(alt.logprob)
    return p_yes / (p_yes + p_no) if p_yes + p_no else 0.5


def available() -> bool:
    return MODEL_PATH.exists()


def _load():
    global _clf
    with _clf_lock:
        if _clf is None:
            bundle = joblib.load(MODEL_PATH)
            if not isinstance(bundle, dict):    # older single-estimator file
                bundle = {"clf": bundle}
            bundle.setdefault("accept", ACCEPT_THRESHOLD)
            bundle.setdefault("reject", REJECT_THRESHOLD)
            bundle.setdefault("judge_fraction", 1.0)
            _clf = bundle
    return _clf


def judge_fraction() -> float:
    """Share of cheap-tier answers the gate hands to the judge, measured
    held-out at training time. The routing policy prices verification as
    this fraction of a judge call."""
    return _load()["judge_fraction"] if available() else 1.0


def p_correct(**kw) -> float:
    X = features(**kw).reshape(1, -1)
    return float(_load()["clf"].predict_proba(X)[0, 1])


def verify_learned(query: str, answer: str, **signals) -> dict:
    """Same shape as the LLM judge's result, plus "verdict": accept, reject
    or unsure. On unsure, `passed` is None and the cascade asks the judge."""
    if not answer or not answer.strip():
        return {"passed": False, "verdict": "reject", "reason": "empty response",
                "cost": 0.0, "latency_ms": 0.0, "verifier_tier": "scorer"}
    import time
    start = time.perf_counter()
    p = p_correct(query=query, answer=answer, **signals)
    latency_ms = (time.perf_counter() - start) * 1000
    b = _load()
    if p >= b["accept"]:
        verdict, passed = "accept", True
    elif p < b["reject"]:
        verdict, passed = "reject", False
    else:
        verdict, passed = "unsure", None
    return {
        "passed": passed, "verdict": verdict,
        "reason": f"scorer P(correct)={p:.2f} -> {verdict}",
        "cost": 0.0, "latency_ms": latency_ms, "verifier_tier": "scorer",
    }
