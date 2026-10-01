"""Leave-one-out accuracy of the difficulty classifier.

Each labelled question is classified against all the others -- the same
k-NN vote and structural overrides the live router uses -- and compared to
its own label. Free: no model call, only local embeddings.

The two numbers that matter most for the expert band:

  expert recall     expert questions that would start at the frontier
  false expert      other questions pulled into the expert band, each of
                    which pays a thinking answer it didn't need

    ./venv/bin/python -m scripts.classifier.evaluate_loo
"""
from collections import Counter

from sentence_transformers import util

from app.evaluation.datasets import DIFFICULTIES, load_reference_queries
from app.routing.classifier import _get_model, apply_overrides, difficulty_from_similarities

RANK = {d: i for i, d in enumerate(DIFFICULTIES)}


def main():
    queries = load_reference_queries()
    embeddings = _get_model().encode([q["query"] for q in queries], convert_to_tensor=True)
    sims = util.cos_sim(embeddings, embeddings)

    labels = [q["difficulty"] for q in queries]
    confusion = Counter()
    for i, q in enumerate(queries):
        row = sims[i].clone()
        row[i] = -1.0                                   # leave this one out
        guess = difficulty_from_similarities(row, labels)
        confusion[(q["difficulty"], apply_overrides(q["query"], guess))] += 1

    n = len(queries)
    exact = sum(c for (truth, pred), c in confusion.items() if truth == pred)
    over = sum(c for (truth, pred), c in confusion.items() if RANK[pred] > RANK[truth])
    under = sum(c for (truth, pred), c in confusion.items() if RANK[pred] < RANK[truth])
    print(f"{n} labelled questions, leave-one-out")
    print(f"  exact {exact}/{n} = {exact / n:.1%}   over-routed {over}   under-routed {under}\n")

    print("  truth \\ predicted " + "".join(f"{d:>9}" for d in DIFFICULTIES))
    for truth in DIFFICULTIES:
        print(f"  {truth:18}" + "".join(f"{confusion[(truth, p)]:9}" for p in DIFFICULTIES))

    experts = sum(confusion[("expert", p)] for p in DIFFICULTIES)
    if experts:
        recall = confusion[("expert", "expert")] / experts
        false_expert = sum(confusion[(t, "expert")] for t in DIFFICULTIES if t != "expert")
        print(f"\n  expert recall {recall:.1%}   false expert {false_expert}/{n - experts}")


if __name__ == "__main__":
    main()
