"""Every grader, including the formatting cases that once marked a right
answer wrong."""
from app.evaluation.graders import final_answer, score_one


def test_bold_initials_still_match():
    q = {"eval_method": "exact_match", "expected": ["hypertext transfer protocol"]}
    assert score_one(q, "**H**yper**T**ext **T**ransfer **P**rotocol") == 1.0


def test_numeric_accepts_percent_and_fraction():
    q = {"eval_method": "exact_match_numeric", "expected_value": 2 / 9, "tolerance": 0.001}
    assert score_one(q, "The probability is 2/9.") == 1.0
    assert score_one(q, "about 22.22%") == 1.0


def test_suffix_match_multiple_choice_with_label():
    q = {"eval_method": "suffix_match", "expected": ["(B)"]}
    assert score_one(q, "...\nAnswer: (B) heptagon") == 1.0
    assert score_one(q, "...\nAnswer: (C) hexagon") == 0.0


def test_suffix_match_word_sorting_ignores_commas():
    q = {"eval_method": "suffix_match", "expected": ["syndrome therefrom"]}
    assert score_one(q, "Answer: syndrome, therefrom") == 1.0


def test_boxed_match_normalises_latex():
    q = {"eval_method": "boxed_match", "expected": ["\\left(3,\\frac{\\pi}{2}\\right)"]}
    assert score_one(q, "so \\boxed{(3, \\dfrac{\\pi}{2})}") == 1.0


def test_final_answer_strips_markup_but_keeps_multiplication():
    assert final_answer("work\n**Answer:** $42$.") == "42"
    assert final_answer("Answer: (25 * 7) + 75") == "(25 * 7) + 75"


def test_answer_integer():
    q = {"eval_method": "answer_integer", "expected": 1081}
    assert score_one(q, "Answer: **1,081**") == 1.0
    assert score_one(q, "\\boxed{1081}") == 1.0
    assert score_one(q, "Answer: 1080 or 1081") == 0.0      # a hedge is not an answer
    assert score_one(q, "") == 0.0


def test_answer_set_and_sequence():
    q = {"eval_method": "answer_set", "expected": ["Ivo", "Gus", "Fay"]}
    assert score_one(q, "Answer: Fay, Gus and Ivo.") == 1.0
    assert score_one({"eval_method": "answer_set", "expected": []}, "Answer: none") == 1.0
    seq = {"eval_method": "answer_sequence", "expected": ["Dara", "Bruno"]}
    assert score_one(seq, "Answer: Dara, Bruno") == 1.0
    assert score_one(seq, "Answer: Bruno, Dara") == 0.0


def test_countdown_checks_value_and_number_use():
    q = {"eval_method": "countdown", "numbers": [25, 1, 2, 9, 7, 75], "target": 253}
    assert score_one(q, "Answer: (25 * 7) + 75 + 2 + 1") == 1.0
    assert score_one(q, "Answer: 25 × 7 + 75 + 2 + 1 = 253") == 1.0
    assert score_one(q, "Answer: 25 * 10 + 2 + 1") == 0.0     # 10 is not one of the numbers
    assert score_one(q, "Answer: 25 * 7 + 75 + 1 + 1 + 1") == 0.0  # 1 used three times
