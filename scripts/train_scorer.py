"""Train the learned answer scorer -- the FrugalGPT idea applied here.

FrugalGPT (Chen, Zaharia, Zou 2023) cascades LLMs like this router does, but
its accept/reject decision is a small learned model, not an LLM judge: a
scorer g(query, answer) -> [0, 1] trained on (query, answer, correct?)
examples, with the answer accepted when g clears a threshold. The paper
fine-tunes DistilBERT; with 345 labelled pairs rather than thousands, a
logistic regression over sentence-embedding features is the honest size.

Why it matters here: the LLM judge that verifies each cheap-tier answer
costs about as much as the answer itself on a cheap mid model, which was
the whole reason the cascade couldn't beat "just use mid". A scorer that
runs in a millisecond on the embedding model already in memory costs
nothing per call.

Training data is every (query, answer, score) triple in the eval database,
across all three tiers. Label: score == 1.0 is correct. Everything is
cross-validated and the numbers printed are held-out, because a scorer
that only works on its own training set would ship wrong answers.

    ./venv/bin/python scripts/train_scorer.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.classifier import _get_model
from app.db import get_conn
from app.scorer import MODEL_PATH, featurize

SEED = 7


def load_pairs_from_json():
    """The query text lives in data/eval_queries.json, keyed by id."""
    import json
    queries = {q["id"]: q for q in json.loads(
        (Path(__file__).resolve().parent.parent / "data" / "eval_queries.json").read_text())}
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT r.query_id, r.response_text, s.score, r.tier
            FROM eval_responses r
            JOIN eval_scores s ON s.query_id = r.query_id AND s.tier = r.tier
            WHERE s.source IN ('automatic', 'judge_claude')
        """).fetchall()
    out = []
    for qid, answer, score, tier in rows:
        q = queries.get(qid)
        if q and answer:
            out.append((q["query"], answer, float(score), tier, q["difficulty"]))
    return out


def main():
    pairs = load_pairs_from_json()
    queries = [p[0] for p in pairs]
    answers = [p[1] for p in pairs]
    y = np.array([1 if p[2] >= 1.0 else 0 for p in pairs])   # 1 = correct
    tiers = np.array([p[3] for p in pairs])
    diffs = np.array([p[4] for p in pairs])
    print(f"{len(pairs)} pairs: {y.sum()} correct, {(1 - y).sum()} wrong")

    model = _get_model()
    X = featurize(model, queries, answers)
    print(f"feature matrix: {X.shape}")

    clf = make_pipeline(
        StandardScaler(),
        # Strong L2: ~800 features on 345 rows would memorise otherwise.
        # Balanced class weight, because 90% of answers are correct and an
        # unweighted model would learn to say "accept" and be right 90%.
        LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000),
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    p_correct = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    p_wrong = 1 - p_correct

    print("\n=== held-out (5-fold) ability to spot a WRONG answer ===")
    print(f"  ROC-AUC            : {roc_auc_score(1 - y, p_wrong):.3f}   (0.5 = coin flip)")
    print(f"  average precision  : {average_precision_score(1 - y, p_wrong):.3f}   (base rate {(1-y).mean():.3f})")

    # Runtime only ever scores cheap-tier answers to easy/medium questions
    # -- that's the whole population the router sends to the cheap tier.
    mask = (tiers == "cheap") & np.isin(diffs, ["easy", "medium"])
    if mask.sum():
        yw, pw = (1 - y)[mask], p_wrong[mask]
        print(f"\n  on the {mask.sum()} cheap-tier easy/medium answers (what runs in production):")
        print(f"    wrong answers among them: {yw.sum()}")
        if yw.sum() and yw.sum() < len(yw):
            print(f"    ROC-AUC: {roc_auc_score(yw, pw):.3f}")

    print("\n=== threshold sweep: accept if P(correct) >= tau ===")
    print(f"  {'tau':>5} {'catch wrong':>12} {'false-escalate good':>20} {'escalation rate':>16}")
    for tau in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        reject = p_correct < tau
        caught = (reject & (y == 0)).sum() / max(1, (y == 0).sum())
        false_esc = (reject & (y == 1)).sum() / max(1, (y == 1).sum())
        print(f"  {tau:>5.2f} {caught:>11.0%} {false_esc:>19.0%} {reject.mean():>15.0%}")

    clf.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"\nsaved {MODEL_PATH.relative_to(Path.cwd())} ({MODEL_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
