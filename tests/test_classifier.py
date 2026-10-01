"""Structural overrides on the embedding vote. The embedding model itself
is exercised by test_api.py and scripts/classifier/evaluate_loo.py."""
from app.routing.classifier import apply_overrides, vote


def test_vote_is_similarity_weighted():
    assert vote([0.9, 0.3, 0.3], ["hard", "easy", "easy"]) == "hard"
    assert vote([0.5, 0.4, 0.4], ["hard", "easy", "easy"]) == "easy"
    assert vote([], []) == "medium"


def test_expert_survives_the_length_rule():
    long_problem = " ".join(["word"] * 60)
    assert apply_overrides(long_problem, "expert") == "expert"
    assert apply_overrides(long_problem, "easy") == "medium"


def test_short_recall_question_is_capped_at_medium():
    assert apply_overrides("What is the worst case of quicksort?", "hard") == "medium"


def test_code_is_at_least_medium():
    assert apply_overrides("python sort a list", "easy") == "medium"
