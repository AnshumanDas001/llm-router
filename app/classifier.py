"""Upfront difficulty classifier.

Combines cheap keyword heuristics with embedding similarity against the
labeled eval set to guess an initial tier before any model call.
"""
import json
import re
import threading
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
# A short question opening with an interrogative ("what is X", "how do I X")
# is recall, which small models handle well -- see RECALL_MAX_WORDS below.
RECALL_QUESTION = re.compile(r"^\s*(what|which|who|when|where|how)\b[^?]{0,70}\??\s*$", re.IGNORECASE)
RECALL_MAX_WORDS = 12

_model = None
_labeled_embeddings = None
_labeled_queries = None

# FastAPI runs sync endpoints in a threadpool, so /api/classify (fired on
# every debounced keystroke) can land while a chat request is classifying.
# Two things follow from that:
#
#   * device="cpu" -- sentence-transformers otherwise auto-selects Apple's
#     MPS backend, and concurrent encodes from two threads killed the whole
#     process with "failed assertion ... IOGPUMetalCommandBuffer". MiniLM-L6
#     is 22M params; a single short query on CPU is a few milliseconds, so
#     the GPU buys nothing here anyway.
#   * one lock around load and encode -- so concurrent callers serialise
#     instead of racing the same model object (and so two first-callers
#     can't both trigger the initial load).
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _model


def _get_labeled_set():
    global _labeled_embeddings, _labeled_queries
    if _labeled_embeddings is None:
        _labeled_queries = json.loads(EVAL_QUERIES_PATH.read_text())
        model = _get_model()
        texts = [q["query"] for q in _labeled_queries]
        _labeled_embeddings = model.encode(texts, convert_to_tensor=True)
    return _labeled_queries, _labeled_embeddings


def refresh_reference_cache():
    """Call after merge_candidates.py adds new examples to eval_queries.json,
    so a long-running server picks them up without a restart."""
    global _labeled_embeddings, _labeled_queries
    _labeled_embeddings = None
    _labeled_queries = None


TOP_K = 5


def _embedding_difficulty(query: str) -> str:
    """Similarity-weighted vote over the k nearest labeled queries.

    This used to take only the single nearest neighbour, on the theory that
    k-voting let mediocre matches outvote one clearly-best match. That was
    reasoned from one example. Measured properly -- leave-one-out across all
    116 labeled queries -- the vote is better on every axis:

        1-NN                 56.0% exact, 21 over-routed, 23 under-routed
        weighted k=5 vote    65.5% exact, 18 over-routed, 11 under-routed

    Single-neighbour is fragile here because the reference set is 52% "hard",
    so one spurious topical match drags a query straight to the mid tier.
    Summing similarity over five neighbours dilutes that.
    """
    with _model_lock:
        labeled_queries, labeled_embeddings = _get_labeled_set()
        model = _get_model()
        query_emb = model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, labeled_embeddings)[0]

        top = scores.topk(min(TOP_K, len(labeled_queries)))
        votes = {}
        for similarity, idx in zip(top.values.tolist(), top.indices.tolist()):
            if similarity <= 0:
                continue
            difficulty = labeled_queries[idx]["difficulty"]
            votes[difficulty] = votes.get(difficulty, 0.0) + similarity
        if not votes:
            return "medium"
        return max(votes, key=votes.get)


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

    # Embedding similarity measures *topic*, not difficulty: "what's the worst
    # case of quicksort" lands next to "explain why naive quicksort degrades
    # to O(n^2)" because both are about quicksort, even though one is recall
    # and the other is analysis. So a short interrogative is capped at medium
    # (which still starts at the cheap tier) rather than trusting a topical
    # match that says hard. If the cheap tier fumbles it, verification
    # escalates -- that safety net is exactly what makes starting low safe.
    if (embedding_guess == "hard" and word_count <= RECALL_MAX_WORDS
            and RECALL_QUESTION.match(query) and not has_multi_step):
        return "medium"

    return embedding_guess


# Measured on the eval set: cheap collapses on "hard" (0.64) but stays strong on
# medium (0.936) and easy (0.916) -- so hard queries skip cheap entirely
# rather than wasting a doomed cheap-tier call before escalating anyway.
# This is the default for our own built-in stack; a BYOM account that has
# calibrated its own models gets a map derived from *its* measurements
# instead (see app/routing_policy.py).
DIFFICULTY_TO_TIER = {"easy": "cheap", "medium": "cheap", "hard": "mid"}


def classify_initial_tier(query: str, difficulty_to_tier: dict | None = None) -> tuple[str, str]:
    """Returns (initial_tier, classified_difficulty)."""
    difficulty = classify_difficulty(query)
    tier_map = difficulty_to_tier or DIFFICULTY_TO_TIER
    return tier_map.get(difficulty, DIFFICULTY_TO_TIER[difficulty]), difficulty
