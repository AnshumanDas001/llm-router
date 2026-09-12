"""Week 3: upfront difficulty classifier.

Combines cheap keyword heuristics with embedding similarity against the
Week 2 labeled eval set to guess an initial tier before any model call.
"""
import json
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer, util

EVAL_QUERIES_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_queries.json"

CODE_KEYWORDS = re.compile(
    r"\b(function|code|sql|regex|debug|pseudocode|python|javascript|algorithm|"
    r"query|schema|class\b|def\b)\b",
    re.IGNORECASE,
)
MULTI_STEP_KEYWORDS = re.compile(
    r"\b(explain why|walk through|step[- ]by[- ]step|and explain|and describe|"
    r"and classify|compare|tradeoff|design)\b",
    re.IGNORECASE,
)

_model = None
_labeled_embeddings = None
_labeled_queries = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _get_labeled_set():
    global _labeled_embeddings, _labeled_queries
    if _labeled_embeddings is None:
        _labeled_queries = json.loads(EVAL_QUERIES_PATH.read_text())
        model = _get_model()
        texts = [q["query"] for q in _labeled_queries]
        _labeled_embeddings = model.encode(texts, convert_to_tensor=True)
    return _labeled_queries, _labeled_embeddings


def _embedding_difficulty(query: str) -> str:
    """1-nearest-neighbor, not a k-vote: with only 61 labeled examples and
    MiniLM cosine similarities sitting in a fairly flat 0.2-0.4 band for
    short generic queries, both raw-count and similarity-weighted k=5 voting
    let two or three mediocre matches outvote one clearly-best match (e.g.
    "What is 5 + 3?" matched "17*23?" at 0.41 -- correctly easy -- but two
    weaker "hard" matches at 0.38 and 0.22 summed higher). Taking only the
    single best match avoids that dilution."""
    labeled_queries, labeled_embeddings = _get_labeled_set()
    model = _get_model()
    query_emb = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, labeled_embeddings)[0]
    best_idx = int(scores.argmax())
    return labeled_queries[best_idx]["difficulty"]


def classify_difficulty(query: str) -> str:
    """Returns 'easy', 'medium', or 'hard'."""
    embedding_guess = _embedding_difficulty(query)

    # Heuristic override: long, multi-step, or code-heavy queries are at
    # least medium even if embedding similarity suggests otherwise.
    word_count = len(query.split())
    has_code = bool(CODE_KEYWORDS.search(query))
    has_multi_step = bool(MULTI_STEP_KEYWORDS.search(query))

    if word_count > 25 or has_multi_step:
        return "hard" if embedding_guess == "hard" else "medium"
    if has_code and embedding_guess == "easy":
        return "medium"

    return embedding_guess


# Week 2 finding: cheap collapses on "hard" (0.718) but stays strong on
# medium (0.936) and easy (0.916) -- so hard queries skip cheap entirely
# rather than wasting a doomed cheap-tier call before escalating anyway.
DIFFICULTY_TO_TIER = {"easy": "cheap", "medium": "cheap", "hard": "mid"}


def classify_initial_tier(query: str) -> tuple[str, str]:
    """Returns (initial_tier, classified_difficulty)."""
    difficulty = classify_difficulty(query)
    return DIFFICULTY_TO_TIER[difficulty], difficulty
