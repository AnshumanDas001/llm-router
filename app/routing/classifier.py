"""Upfront difficulty classifier.

Combines cheap keyword heuristics with embedding similarity against the
labelled question sets to guess a difficulty band before any model call.
"""
import re
import threading
import time

from sentence_transformers import SentenceTransformer, util

from app.evaluation.datasets import load_reference_queries

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
        _labeled_queries = load_reference_queries()
        texts = [q["query"] for q in _labeled_queries]
        _labeled_embeddings = _get_model().encode(texts, convert_to_tensor=True)
    return _labeled_queries, _labeled_embeddings


def refresh_reference_cache():
    """Call after merge_candidates.py adds new examples to eval_queries.json,
    so a long-running server picks them up without a restart."""
    global _labeled_embeddings, _labeled_queries
    _labeled_embeddings = None
    _labeled_queries = None


TOP_K = 5

# The expert gate looks a little wider and needs at least half the vote.
# Measured leave-one-out over all 176 labelled questions:
#
#   one k=5 vote over all four bands   expert recall 53/60, 10 of 116
#                                       everyday questions pulled to expert
#   expert gate (k=7, share >= 0.5),    expert recall 54/60, 4 of 116
#   then k=5 over the other three       pulled to expert
#
# A false expert pays a thinking answer it didn't need, so the gate is
# tuned against those; the second stage then votes over the original
# three-band examples only, which keeps their own accuracy at 72/116
# against 75 with no expert band at all.
EXPERT_K = 7
EXPERT_MIN_SHARE = 0.5


def vote(similarities: list[float], labels: list[str]) -> str:
    """Similarity-weighted majority over the neighbours given."""
    votes = {}
    for similarity, difficulty in zip(similarities, labels):
        if similarity > 0:
            votes[difficulty] = votes.get(difficulty, 0.0) + similarity
    return max(votes, key=votes.get) if votes else "medium"


def explain_similarities(scores, labels: list[str], texts: list[str] | None = None) -> dict:
    """The embedding vote with its evidence, given one query's cosine
    similarity to every labelled question (a 1-D tensor) and their labels.

    Two stages. First the expert gate: is this a competition-maths style
    problem, by a clear share of its nearest neighbours? If not, a
    similarity-weighted vote over the k nearest everyday examples.

    The vote replaced taking the single nearest neighbour, on measurement
    -- leave-one-out across the 116 everyday questions, with overrides:

        1-NN                 55.2% exact, 22 over-routed, 30 under-routed
        weighted k=5 vote    64.7% exact, 24 over-routed, 17 under-routed

    Single-neighbour is fragile here because the reference set is 52% "hard",
    so one spurious topical match drags a query straight to the mid tier.
    Summing similarity over five neighbours dilutes that.

    Returns {band, expert_share, gate, neighbours, votes} -- everything
    "Why this route?" shows about the classification.
    """
    def rows(top):
        return [{"band": labels[i], "similarity": round(v, 3),
                 **({"text": texts[i]} if texts else {})}
                for v, i in zip(top.values.tolist(), top.indices.tolist())]

    top = scores.topk(min(EXPERT_K, len(labels)))
    weights = [(v, labels[i]) for v, i in zip(top.values.tolist(), top.indices.tolist()) if v > 0]
    total = sum(v for v, _ in weights)
    share = (sum(v for v, label in weights if label == "expert") / total) if total else 0.0
    gate = {"k": EXPERT_K, "share": round(share, 3), "needed": EXPERT_MIN_SHARE}
    if total and share >= EXPERT_MIN_SHARE:
        return {"band": "expert", "expert_share": share, "gate": gate, "neighbours": rows(top),
                "votes": {"expert": round(share, 3)}}

    everyday = scores.clone()
    for i, label in enumerate(labels):
        if label == "expert":
            everyday[i] = -1.0
    top = everyday.topk(min(TOP_K, len(labels)))
    sims, bands = top.values.tolist(), [labels[i] for i in top.indices.tolist()]
    votes = {}
    for v, b in zip(sims, bands):
        if v > 0:
            votes[b] = round(votes.get(b, 0.0) + v, 3)
    return {"band": vote(sims, bands), "expert_share": share, "gate": gate,
            "neighbours": rows(top), "votes": votes}


def difficulty_from_similarities(scores, labels: list[str]) -> str:
    """Just the band; see explain_similarities."""
    return explain_similarities(scores, labels)["band"]


def _embedding_vote(query: str) -> dict:
    with _model_lock:
        labeled_queries, labeled_embeddings = _get_labeled_set()
        query_emb = _get_model().encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, labeled_embeddings)[0]
        return explain_similarities(scores, [q["difficulty"] for q in labeled_queries],
                                    [q["query"] for q in labeled_queries])


def _embedding_difficulty(query: str) -> str:
    return _embedding_vote(query)["band"]


def override(query: str, embedding_guess: str) -> tuple[str, str | None]:
    """Structural corrections to the embedding vote, for what topical
    similarity can't see. Returns (band, the rule that changed it or None)."""
    # Expert is a band of whole problem *types* (competition maths), and
    # its references are long, so the length rule below would demote every
    # one of them to medium. Trust the vote.
    if embedding_guess == "expert":
        return "expert", None

    # Long, multi-step, or code-heavy queries are at least medium even if
    # embedding similarity suggests otherwise.
    word_count = len(query.split())
    has_code = bool(CODE_KEYWORDS.search(query))
    has_multi_step = bool(MULTI_STEP_KEYWORDS.search(query))

    if word_count > 25 or has_multi_step:
        band = "hard" if embedding_guess == "hard" else "medium"
        why = (f"{word_count} words, over the 25-word limit" if word_count > 25
               else "asks for multiple steps")
        return band, (None if band == embedding_guess else f"raised to medium: {why}")
    if has_code and embedding_guess == "easy":
        return "medium", "raised to medium: mentions code"

    # Embedding similarity measures *topic*, not difficulty: "what's the worst
    # case of quicksort" lands next to "explain why naive quicksort degrades
    # to O(n^2)" because both are about quicksort, even though one is recall
    # and the other is analysis. So a short interrogative is capped at medium
    # (which still starts at the cheap tier) rather than trusting a topical
    # match that says hard. If the cheap tier fumbles it, verification
    # escalates -- that safety net is exactly what makes starting low safe.
    if (embedding_guess == "hard" and word_count <= RECALL_MAX_WORDS
            and RECALL_QUESTION.match(query) and not has_multi_step):
        return "medium", f"capped at medium: a short recall question ({word_count} words)"

    return embedding_guess, None


def apply_overrides(query: str, embedding_guess: str) -> str:
    return override(query, embedding_guess)[0]


def classify_with_trace(query: str) -> tuple[str, dict]:
    """(band, trace): the band, and how it was reached -- the nearest
    labelled examples, the vote, the expert gate, any override."""
    start = time.perf_counter()
    voted = _embedding_vote(query)
    band, rule = override(query, voted["band"])
    return band, {
        "band": band, "embedding_band": voted["band"], "override": rule,
        "gate": voted["gate"], "votes": voted["votes"],
        "neighbours": [{**n, "text": n.get("text", "")[:160]} for n in voted["neighbours"]],
        "ms": round((time.perf_counter() - start) * 1000, 1),
    }


def classify_difficulty(query: str) -> str:
    """Returns 'easy', 'medium', 'hard' or 'expert'."""
    return apply_overrides(query, _embedding_difficulty(query))


# The map for an uncalibrated stack. Calibration replaces it with one
# derived from measurements (see app/routing/policy.py); this is only what
# routing falls back to when there is nothing measured to go on.
DIFFICULTY_TO_TIER = {"easy": "cheap", "medium": "cheap", "hard": "mid", "expert": "frontier"}


def classify_initial_tier(query: str, difficulty_to_tier: dict | None = None) -> tuple[str, str]:
    """Returns (initial_tier, classified_difficulty)."""
    difficulty = classify_difficulty(query)
    tier_map = difficulty_to_tier or DIFFICULTY_TO_TIER
    return tier_map.get(difficulty, DIFFICULTY_TO_TIER[difficulty]), difficulty
