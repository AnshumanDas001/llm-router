"""Automatic (no-LLM-judge-needed) correctness scoring, shared between the
offline eval scripts (scripts/score_structured.py) and the live BYOM
calibration endpoint -- both need to grade a model's answer against a
known-correct query without a human or judge model in the loop.
"""
import json
import re

AUTOMATIC_METHODS = {
    "exact_match", "exact_match_contains_any", "exact_match_set",
    "exact_match_numeric", "schema_json", "schema_pattern",
}


def score_exact_match(text: str, expected: list[str]) -> float:
    low = text.lower()
    return 1.0 if any(e.lower() in low for e in expected) else 0.0


def score_exact_match_set(text: str, expected: list[str]) -> float:
    low = text.lower()
    found = sum(1 for e in expected if e.lower() in low)
    return found / len(expected)


def score_exact_match_numeric(text: str, expected_value: float, tolerance: float) -> float:
    candidates = []

    # Plain numbers, including comma-formatted thousands (e.g. "275,400").
    for n in re.findall(r"-?\d[\d,]*\.?\d*", text):
        try:
            candidates.append(float(n.replace(",", "")))
        except ValueError:
            continue

    # Percentages, e.g. "22.23%" or LaTeX-escaped "22.23\%" -> 0.2223 -- a
    # common alternate way models express a probability that the plain-number
    # pass above would otherwise misread as e.g. 22.23 and reject as wildly
    # out of tolerance.
    for n in re.findall(r"-?\d[\d,]*\.?\d*\s*\\?%", text):
        try:
            candidates.append(float(n.rstrip("%\\ ").replace(",", "")) / 100)
        except ValueError:
            continue

    # Fractions, plain ("2/9") or LaTeX (\frac{2}{9}) -- probability answers
    # are frequently left as an unreduced fraction rather than converted.
    for num, den in re.findall(r"(\d+)\s*/\s*(\d+)", text):
        try:
            if float(den) != 0:
                candidates.append(float(num) / float(den))
        except ValueError:
            continue
    for num, den in re.findall(r"\\frac\{(\d+)\}\{(\d+)\}", text):
        try:
            if float(den) != 0:
                candidates.append(float(num) / float(den))
        except ValueError:
            continue

    return 1.0 if any(abs(c - expected_value) <= tolerance for c in candidates) else 0.0


def extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def score_schema_json(text: str, expected_json: dict) -> float:
    candidate = extract_first_json_object(text)
    if candidate is None:
        return 0.0
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return 0.0
    matched = sum(1 for k, v in expected_json.items()
                   if k in parsed and str(parsed[k]).lower() == str(v).lower())
    return matched / len(expected_json)


def score_schema_pattern(text: str, pattern: str) -> float:
    return 1.0 if re.search(pattern, text, re.IGNORECASE | re.DOTALL) else 0.0


def score_one(q: dict, response_text: str) -> float:
    method = q["eval_method"]
    if method == "exact_match" or method == "exact_match_contains_any":
        return score_exact_match(response_text, q["expected"])
    if method == "exact_match_set":
        return score_exact_match_set(response_text, q["expected"])
    if method == "exact_match_numeric":
        return score_exact_match_numeric(response_text, q["expected_value"], q["tolerance"])
    if method == "schema_json":
        return score_schema_json(response_text, q["expected_json"])
    if method == "schema_pattern":
        return score_schema_pattern(response_text, q["expected_pattern"])
    raise ValueError(f"not an automatic method: {method}")
