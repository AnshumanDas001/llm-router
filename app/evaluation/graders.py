"""Automatic correctness scoring: grade a model's answer against a question
with a known-correct answer, with no human or judge model in the loop.

One module for every grader, shared by the offline eval scripts, the
benchmark probes (scripts/probe/) and live calibration -- so a question is
scored the same way wherever it is asked. Each question names its grader in
`eval_method`; `score_one` dispatches on it and returns a score in [0, 1].

Every grader here has been caught at least once marking a right answer
wrong for its *formatting* -- a bold initial, a comma between sorted words,
"(B) heptagon" for "(B)". The normalisation in each one is the fix for a
specific case like that; treat any score as a lower bound, and read the
answers before trusting a surprising number.
"""
import ast
import json
import operator
import re
from fractions import Fraction

# Methods whose questions carry a ground truth. Anything else (llm_judge)
# needs a person or a judge model to grade.
AUTOMATIC_METHODS = {
    "exact_match", "exact_match_contains_any", "exact_match_set",
    "exact_match_numeric", "schema_json", "schema_pattern",
    # benchmark-style: the answer is read off a marked final line
    "boxed_match", "suffix_match",
    "answer_integer", "answer_set", "answer_sequence", "countdown",
}


def is_automatic(q: dict) -> bool:
    return q["eval_method"] in AUTOMATIC_METHODS


_EMPHASIS = re.compile(r"[*_`]+")


def _plain(text: str) -> str:
    """Lower-cased with markdown emphasis removed, so "**H**yper**T**ext
    **T**ransfer **P**rotocol" matches "hypertext transfer protocol" -- a
    model that bolds the initials was graded wrong for the formatting."""
    return _EMPHASIS.sub("", text).lower()


# --- free-text answers (the original eval set) ------------------------------

def score_exact_match(text: str, expected: list[str]) -> float:
    low = _plain(text)
    return 1.0 if any(e.lower() in low for e in expected) else 0.0


def score_exact_match_set(text: str, expected: list[str]) -> float:
    low = _plain(text)
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


# --- marked final answers (benchmarks and the expert set) -------------------

def extract_boxed(text: str) -> str | None:
    """Last \\boxed{...}, matching braces so nested ones survive."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth, out = 0, []
    for ch in text[i:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    return "".join(out)


_ANSWER_LINE = re.compile(r"answer\s*[:：]\s*(.+)", re.IGNORECASE)


_MARKUP = re.compile(r"\*\*|__|`")


def final_answer(text: str) -> str:
    """What the model committed to: the last "Answer: ..." line, else the
    last \\boxed{}, else the last non-empty line. Bold, code ticks, LaTeX
    dollar signs and a trailing full stop are stripped -- "**Answer:** $42$."
    is an answer of 42.

    Only *paired* markup goes: a single asterisk is multiplication. Stripping
    every `*` turned "(25 * 7) + 75 + 2 + 1" into "(25  7) + 75 + 2 + 1" and
    failed six correct Countdown answers in a row."""
    text = _MARKUP.sub("", text or "")
    found = None
    for m in _ANSWER_LINE.finditer(text):
        found = m.group(1)
    if found is None:
        found = extract_boxed(text)
    if found is None:
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        found = lines[-1] if lines else ""
    found = found.strip()
    boxed = extract_boxed(found)
    if boxed is not None:
        found = boxed
    return found.replace("$", "").strip().strip("*_").strip().rstrip(".").strip()


def _norm_math(text: str) -> str:
    """Compare mathematical answers without tripping over presentation.
    \\left(3,\\frac{\\pi}{2}\\right) and (3, \\frac{\\pi}{2}) are the same
    answer; so are '0.5' and '.5', and '$x$' and 'x'."""
    t = (text or "").strip()
    for junk in ("\\left", "\\right", "$", "\\!", "\\,", " ", "\n", "\\ "):
        t = t.replace(junk, "")
    t = t.replace("dfrac", "frac").replace("tfrac", "frac")
    t = re.sub(r"\\text\{(.*?)\}", r"\1", t)
    t = t.rstrip(".").lstrip("0") if re.fullmatch(r"0\.\d+", t) else t.rstrip(".")
    return t.lower()


def _norm_sequence(text: str) -> str:
    """For answers that are a list of items, compare the items and not the
    punctuation between them: BBH's word_sorting expects "syndrome therefrom"
    and models reply "syndrome, therefrom". Marking that wrong measured the
    formatting, not the sorting -- it scored a model 0/5 on a task it had
    actually got right."""
    return " ".join(re.split(r"[,\s]+", (text or "").strip().strip("."))).strip().lower()


def score_boxed_match(text: str, expected: str) -> float:
    got = extract_boxed(text)
    if got is None:                      # no \boxed: fall back to last line
        lines = (text or "").strip().splitlines()
        got = lines[-1] if lines else ""
    return 1.0 if _norm_math(got) == _norm_math(expected) else 0.0


def score_suffix_match(text: str, expected: str) -> float:
    """BBH-style: "Answer: (D)" on the last line."""
    tail = (text or "").strip().splitlines()[-1] if (text or "").strip() else ""
    m = re.search(r"answer\s*:?\s*(.+)$", tail, re.I)
    got = (m.group(1) if m else tail).strip()
    if _norm_math(got) == _norm_math(expected) or _norm_sequence(got) == _norm_sequence(expected):
        return 1.0
    # Multiple choice: BBH's target is the bare option, "(B)", but models
    # answer "(B) heptagon" -- the letter *and* what it stands for. Demanding
    # an exact match scored a model 2/17 on a task it had almost entirely
    # right. Accept when the reply picks that option, however it labels it.
    letter = re.fullmatch(r"\(([A-Z])\)", expected.strip())
    if letter:
        chosen = re.match(r"\(?([A-Za-z])[)\.]", got.strip())
        return 1.0 if chosen and chosen.group(1).upper() == letter.group(1) else 0.0
    return 0.0


def score_answer_integer(text: str, expected: int) -> float:
    """An integer on the Answer line. Thousands separators and a leading
    "x =" are tolerated; any other number on the line is not -- "about 40"
    is not 40, and a line holding two candidates is a hedge."""
    got = final_answer(text).replace(",", "").replace("\\,", "")
    got = re.sub(r"^[a-zA-Z]\s*=\s*", "", got)
    numbers = re.findall(r"-?\d+(?:\.0+)?", got)
    if len(numbers) != 1:
        return 0.0
    return 1.0 if int(float(numbers[0])) == int(expected) else 0.0


def _items(text: str) -> list[str]:
    """Split a list answer into items, whatever separates them."""
    text = re.sub(r"\band\b", ",", text, flags=re.IGNORECASE)
    return [x.strip().strip(".").lower() for x in re.split(r"[,;\n]+", text) if x.strip().strip(".")]


def score_answer_set(text: str, expected: list[str]) -> float:
    """An unordered list of names. "None" is a valid answer when the
    expected set is empty."""
    got = _items(final_answer(text))
    if not expected:
        return 1.0 if got in ([], ["none"], ["nobody"], ["no one"]) else 0.0
    return 1.0 if sorted(got) == sorted(e.lower() for e in expected) else 0.0


def score_answer_sequence(text: str, expected: list[str]) -> float:
    """An ordered list: every item, in order."""
    return 1.0 if _items(final_answer(text)) == [e.lower() for e in expected] else 0.0


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv}


def _eval_arith(node, used: list[int]) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval_arith(node.body, used)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval_arith(node.left, used), _eval_arith(node.right, used)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ValueError("division by zero")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        used.append(node.value)
        return Fraction(node.value)
    raise ValueError("not plain arithmetic")


def score_countdown(text: str, numbers: list[int], target: int) -> float:
    """Countdown: an arithmetic expression that hits the target exactly,
    using each given number at most once. Graded by evaluating the
    expression, so any correct solution counts, not just a reference one."""
    expr = final_answer(text)
    expr = expr.split("=")[0]                     # "... = 523" is fine
    expr = (expr.replace("×", "*").replace("·", "*").replace("÷", "/")
                .replace("−", "-").replace("\\times", "*").replace("\\div", "/"))
    expr = re.sub(r"(?<=\d)\s*[xX]\s*(?=[\d(])", "*", expr)
    try:
        used: list[int] = []
        value = _eval_arith(ast.parse(expr.strip(), mode="eval"), used)
    except (SyntaxError, ValueError, ZeroDivisionError):
        return 0.0
    pool = list(numbers)
    for n in used:
        if n not in pool:
            return 0.0
        pool.remove(n)
    return 1.0 if value == target else 0.0


def score_one(q: dict, response_text: str) -> float:
    method = q["eval_method"]
    text = response_text or ""
    if method == "exact_match" or method == "exact_match_contains_any":
        return score_exact_match(text, q["expected"])
    if method == "exact_match_set":
        return score_exact_match_set(text, q["expected"])
    if method == "exact_match_numeric":
        return score_exact_match_numeric(text, q["expected_value"], q["tolerance"])
    if method == "schema_json":
        return score_schema_json(text, q["expected_json"])
    if method == "schema_pattern":
        return score_schema_pattern(text, q["expected_pattern"])
    if method == "boxed_match":
        return score_boxed_match(text, q["expected"][0])
    if method == "suffix_match":
        return score_suffix_match(text, q["expected"][0])
    if method == "answer_integer":
        return score_answer_integer(text, q["expected"])
    if method == "answer_set":
        return score_answer_set(text, q["expected"])
    if method == "answer_sequence":
        return score_answer_sequence(text, q["expected"])
    if method == "countdown":
        return score_countdown(text, q["numbers"], q["target"])
    raise ValueError(f"not an automatic method: {method}")
