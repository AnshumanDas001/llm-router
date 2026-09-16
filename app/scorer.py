"""Learned answer scorer: is this cheap-tier answer good enough to ship?

The FrugalGPT-style replacement for the LLM judge. g(query, answer) is a
logistic regression over sentence-embedding features of the pair, trained
by scripts/train_scorer.py on the eval database's (query, answer, score)
triples. It runs in about a millisecond on the embedding model the
classifier already holds in memory, so verifying a cheap answer costs
nothing -- which is what lets a cheap tier actually undercut a cheap mid
model instead of paying a judge call that costs as much as the answer.

Features, for a query embedding q and answer embedding a (384-d each):
    [ a,  q * a,  |q - a|,  len(answer), len(query), len ratio ]
The elementwise product and difference are the standard pair features for
"does this answer belong to this question"; the lengths catch the two
failure shapes an embedding can't see -- a curt non-answer and a rambling
one.
"""
import threading
from pathlib import Path

import joblib
import numpy as np

from app.classifier import _get_model, _model_lock

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "answer_scorer.joblib"

# Accept when P(correct) >= this. Chosen from the held-out threshold sweep in
# train_scorer.py to catch most wrong answers without escalating too many
# right ones; see that script's output for the tradeoff at each value.
ACCEPT_THRESHOLD = 0.5

_clf = None
_clf_lock = threading.Lock()


def featurize(model, queries: list[str], answers: list[str]) -> np.ndarray:
    with _model_lock:
        q = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
        a = model.encode(answers, convert_to_numpy=True, normalize_embeddings=True)
    lens = np.array([[len(ans.split()), len(qu.split())] for qu, ans in zip(queries, answers)], dtype=float)
    ratio = (lens[:, :1] + 1) / (lens[:, 1:] + 1)
    return np.hstack([a, q * a, np.abs(q - a), np.log1p(lens), np.log1p(ratio)])


def available() -> bool:
    return MODEL_PATH.exists()


def _get_clf():
    global _clf
    with _clf_lock:
        if _clf is None:
            _clf = joblib.load(MODEL_PATH)
    return _clf


def p_correct(query: str, answer: str) -> float:
    X = featurize(_get_model(), [query], [answer])
    return float(_get_clf().predict_proba(X)[0, 1])


def verify_learned(query: str, answer: str, threshold: float = ACCEPT_THRESHOLD) -> dict:
    """Same shape as the LLM judge's result, so the cascade can swap them."""
    if not answer or not answer.strip():
        return {"passed": False, "reason": "empty response", "cost": 0.0,
                "latency_ms": 0.0, "verifier_tier": "scorer"}
    import time
    start = time.perf_counter()
    p = p_correct(query, answer)
    latency_ms = (time.perf_counter() - start) * 1000
    passed = p >= threshold
    return {
        "passed": passed,
        "reason": f"scorer P(correct)={p:.2f} {'>=' if passed else '<'} {threshold:.2f}",
        "cost": 0.0, "latency_ms": latency_ms, "verifier_tier": "scorer",
    }
