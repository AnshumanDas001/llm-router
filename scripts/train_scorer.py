"""Train the learned answer scorer on the data from build_scorer_data.py.

Prints held-out (5-fold, grouped by query so no question leaks between
folds) discrimination for the full model and for ablations, plus a
threshold sweep, so the accept threshold in app/scorer.py is a measured
choice. Numbers are for spotting a WRONG answer, since that's the job.

    ./venv/bin/python scripts/train_scorer.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.classifier import _get_model
from app.model_config import CHEAP_MODEL
from app.scorer import ACCEPT_THRESHOLD, FEATURE_NAMES, MODEL_PATH, REJECT_THRESHOLD, features

# One dataset per cheap model: the signals are that model's own confidence.
DATA_PATH = (Path(__file__).resolve().parent.parent / "data"
             / f"scorer_training_{CHEAP_MODEL.replace('/', '-').replace(':', '-')}.jsonl")

# The LLM judge measured on the same cheap answers (scratch benchmark over
# the 117 auto-scored easy/medium rows, mid judge): what the gate's "unsure"
# band gets when it defers. Used to project the hybrid's shipped-wrong rate.
JUDGE_CATCH, JUDGE_FALSE_ESCALATE = 0.94, 0.12

ABLATIONS = {
    "confidence only (free)": ["mean_logprob", "min_logprob", "p10_logprob", "frac_uncertain",
                               "head_logprob", "mean_entropy", "max_entropy", "log_tokens",
                               "truncated", "is_easy", "is_medium", "is_hard"],
    "+ self-verify (1 tiny call)": None,   # filled below
    "+ consistency (1 extra generation)": None,
    "everything": FEATURE_NAMES,
}
ABLATIONS["+ self-verify (1 tiny call)"] = ABLATIONS["confidence only (free)"] + ["self_verify", "qa_cosine"]
ABLATIONS["+ consistency (1 extra generation)"] = ABLATIONS["confidence only (free)"] + [
    "consist_max", "consist_mean", "numeric_agree", "qa_cosine"]


def make_clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000))


def report(name, y_wrong, p_wrong):
    auc = roc_auc_score(y_wrong, p_wrong)
    ap = average_precision_score(y_wrong, p_wrong)
    print(f"  {name:38} ROC-AUC {auc:.3f}   AP {ap:.3f}")
    return auc


def main():
    rows = [json.loads(l) for l in DATA_PATH.read_text().splitlines() if l.strip()]
    model = _get_model()
    X = np.vstack([features(r["query"], r["answer"], r["logprobs"], r["entropy"], r["siblings"],
                            r["self_verify_p_yes"], r["difficulty"], r["finish_reason"], model)
                   for r in rows])
    y = np.array([1 if r["score"] >= 1.0 else 0 for r in rows])
    groups = np.array([r["query_id"] for r in rows])
    diffs = np.array([r["difficulty"] for r in rows])
    print(f"{len(rows)} rows from {len(set(groups))} queries: {y.sum()} correct, {(1-y).sum()} wrong")
    for d in ("easy", "medium", "hard"):
        m = diffs == d
        print(f"  {d:6} {m.sum():3} rows, cheap accuracy {y[m].mean():.2f}")

    y_wrong = 1 - y
    cv = GroupKFold(n_splits=5)

    print("\n=== single-feature AUC for spotting a wrong answer (sign-corrected) ===")
    for i, name in enumerate(FEATURE_NAMES):
        if X[:, i].std() == 0:
            continue
        auc = roc_auc_score(y_wrong, -X[:, i])
        print(f"  {name:16} {max(auc, 1-auc):.3f}  ({'low' if auc >= 0.5 else 'high'} value => wrong)")

    print("\n=== held-out (5-fold grouped by query) ===")
    preds = {}
    for name, cols in ABLATIONS.items():
        idx = [FEATURE_NAMES.index(c) for c in cols]
        p_correct = cross_val_predict(make_clf(), X[:, idx], y, cv=cv, groups=groups,
                                      method="predict_proba")[:, 1]
        preds[name] = p_correct
        report(name, y_wrong, 1 - p_correct)

    p_correct = preds["everything"]
    mask = np.isin(diffs, ["easy", "medium"])
    print(f"\n  on the {mask.sum()} easy/medium rows (what the cheap tier actually sees):")
    print(f"    wrong answers among them: {y_wrong[mask].sum()}")
    report("everything", y_wrong[mask], 1 - p_correct[mask])

    ym, pm = y[mask], p_correct[mask]
    print("\n=== calibration on easy/medium (held-out P(correct) vs. actual) ===")
    for lo, hi in ((0, .3), (.3, .5), (.5, .7), (.7, .8), (.8, .9), (.9, 1.01)):
        b = (pm >= lo) & (pm < hi)
        if b.sum():
            print(f"  P in [{lo:.1f}, {min(hi, 1):.1f}): {b.sum():3} answers, {(1 - ym[b]).sum():2} wrong ({(1 - ym[b]).mean():.0%})")

    print("\n=== scorer alone: escalate if P(correct) < tau ===")
    print(f"  {'tau':>5} {'catch wrong':>12} {'false-escalate good':>20} {'shipped-wrong':>14}")
    for tau in (0.3, 0.5, 0.7, 0.9):
        reject = pm < tau
        caught = (reject & (ym == 0)).sum() / max(1, (ym == 0).sum())
        false_esc = (reject & (ym == 1)).sum() / max(1, (ym == 1).sum())
        shipped_wrong = (~reject & (ym == 0)).sum() / max(1, (~reject).sum())
        print(f"  {tau:>5.2f} {caught:>11.0%} {false_esc:>19.0%} {shipped_wrong:>13.1%}")
    print(f"  no verifier: ships wrong {1 - ym.mean():.1%};  judge alone: ~{(1 - ym.mean()) * (1 - JUDGE_CATCH) / (1 - (1 - ym.mean()) * JUDGE_CATCH - ym.mean() * JUDGE_FALSE_ESCALATE):.1%}")

    print("\n=== gate + judge: accept free if P >= hi, escalate free if P < lo, judge between ===")
    print(f"  {'lo':>4} {'hi':>5} {'judge calls':>12} {'shipped-wrong':>14} {'escalations':>12}")
    for lo, hi in ((0.3, 0.9), (0.3, 0.85), (0.2, 0.9), (0.3, 0.95), (0.0, 0.9)):
        acc, rej = pm >= hi, pm < lo
        judged = ~acc & ~rej
        wrong_j, good_j = (judged & (ym == 0)).sum(), (judged & (ym == 1)).sum()
        caught, false_esc = JUDGE_CATCH * wrong_j, JUDGE_FALSE_ESCALATE * good_j
        escalated = rej.sum() + caught + false_esc
        shipped = len(ym) - escalated
        shipped_wrong = ((acc & (ym == 0)).sum() + wrong_j - caught) / shipped
        marker = "  <- saved" if (lo, hi) == (REJECT_THRESHOLD, ACCEPT_THRESHOLD) else ""
        print(f"  {lo:>4.2f} {hi:>5.2f} {judged.mean():>11.0%} {shipped_wrong:>13.1%} {escalated / len(ym):>11.0%}{marker}")
    judged_frac = float(((pm >= REJECT_THRESHOLD) & (pm < ACCEPT_THRESHOLD)).mean())

    clf = make_clf().fit(X, y)
    coef = clf[-1].coef_[0]
    print("\n=== fitted coefficients (standardised; + means 'more likely correct') ===")
    for name, c in sorted(zip(FEATURE_NAMES, coef), key=lambda t: -abs(t[1]))[:10]:
        print(f"  {name:16} {c:+.2f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "accept": ACCEPT_THRESHOLD, "reject": REJECT_THRESHOLD,
                 "judge_fraction": judged_frac, "n_rows": len(rows), "cheap_model": CHEAP_MODEL}, MODEL_PATH)
    print(f"\nsaved {MODEL_PATH.relative_to(Path.cwd())} for {CHEAP_MODEL}: accept>={ACCEPT_THRESHOLD}, "
          f"reject<{REJECT_THRESHOLD}, judge on {judged_frac:.0%} of cheap answers")


if __name__ == "__main__":
    main()
