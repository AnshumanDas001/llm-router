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
runs as a gate in front of the judge, never instead of it:

    P(correct) >= ACCEPT_THRESHOLD   ship it, no judge call
    P(correct) <  REJECT_THRESHOLD   escalate, no judge call
    otherwise                        ask the judge

Both thresholds are chosen from held-out measurement, and the defaults are
deliberately lopsided because the two errors are not symmetric:

  accept 0.95   the highest band where *zero* wrong answers slipped
                through (0 of 15). At 0.90 two would have. Skipping a
                judge call is only free if the answer really was right.
  reject 0.0    off by default. In the low band most answers the scorer
                doubts are in fact correct (at P<0.05, half of them), so
                escalating on the scorer's word alone buys a needless
                paid answer. Let the judge decide; it is better at this.

How much that is worth depends entirely on what a verdict costs relative
to the answer it protects. With a cheap judge (gpt-oss-20b at $0.00004)
in front of a $0.002 answer, verification is 2% of the bill and the gate
saves a fraction of that. With the judge on the tier above -- where a
verdict costs about what the next answer would -- it is the difference
between a cascade that pays for itself and one that ties its own mid
tier. The gate is the same code either way; only the arithmetic changes.
"""
import math
import re
import threading
from pathlib import Path

import joblib
import numpy as np

from app.classifier import _get_model, _model_lock
from app.model_config import CHEAP_MODEL

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "answer_scorer.joblib"

# Fallbacks, used only if the model file doesn't carry its own. The live
# values are whatever train_scorer.py measured and saved.
ACCEPT_THRESHOLD = 0.95
REJECT_THRESHOLD = 0.0        # 0 = never escalate without asking the judge

# The features a scorer may be trained on. "Free" ones are read off the
# generation the cascade already makes; the other two each cost an extra
# cheap-tier call, so the trained model records which it actually uses and
# the cascade skips the calls nothing depends on.
FREE_FEATURES = [
    "mean_logprob", "min_logprob", "p10_logprob", "frac_uncertain", "head_logprob",
    "mean_entropy", "max_entropy", "log_tokens", "truncated",
    "is_easy", "is_medium", "is_hard",
]
SIBLING_FEATURES = ["consist_max", "consist_mean", "numeric_agree"]
SELF_VERIFY_FEATURES = ["self_verify"]

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
    """A trained scorer exists *for the configured cheap model*. Confidence
    profiles don't transfer between models, so a scorer trained on one is
    treated as absent for another and the cascade uses the judge."""
    if not MODEL_PATH.exists():
        return False
    try:
        # Bundles from before the tag was recorded were trained on the
        # original local cheap model.
        return _load().get("cheap_model", "ollama/llama3.2:3b") == CHEAP_MODEL
    except Exception:
        return False


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
            bundle.setdefault("feature_cols", FEATURE_NAMES)
            bundle["_idx"] = [FEATURE_NAMES.index(c) for c in bundle["feature_cols"]]
            _clf = bundle
    return _clf


def uses(feature_group: list[str]) -> bool:
    """Does the trained scorer read any of these features? The cascade asks
    before paying for the call that produces them."""
    if not available():
        return False
    return any(c in _load()["feature_cols"] for c in feature_group)


def judge_fraction() -> float:
    """Share of cheap-tier answers the gate hands to the judge, measured
    held-out at training time. The routing policy prices verification as
    this fraction of a judge call."""
    return _load()["judge_fraction"] if MODEL_PATH.exists() else 1.0


def p_correct(**kw) -> float:
    bundle = _load()
    X = features(**kw).reshape(1, -1)[:, bundle["_idx"]]
    return float(bundle["clf"].predict_proba(X)[0, 1])


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
