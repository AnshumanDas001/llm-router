"""The labelled question sets, and the difficulty bands they define.

Two files, one schema (id, query, difficulty, eval_method, expected...):

  data/eval/eval_queries.json    116 everyday questions labelled easy /
                                 medium / hard against a 3B local model
  data/eval/expert_queries.json  competition maths (AIME 2024-25), built by
                                 scripts/probe/build_expert_set.py

The expert band exists because the other three cannot tell the two paid
tiers apart: mid and frontier both score 1.00 on every question in
eval_queries.json, so the routing policy -- which only ever acts on
measured differences -- had no reason to start anything at the frontier.
Both files serve as the classifier's reference examples and as the
questions calibration grades each tier on.
"""
import json

from app.paths import EVAL_QUERIES, EXPERT_QUERIES

DIFFICULTIES = ("easy", "medium", "hard", "expert")


def prompt_for(q: dict) -> str:
    """What a model is sent: the question plus, for benchmark questions,
    the instruction that pins down where the answer goes so it can be
    graded. The classifier sees `query` alone."""
    return q["query"] + q.get("answer_format", "")


def load_eval_queries() -> list[dict]:
    return json.loads(EVAL_QUERIES.read_text())


def load_expert_queries() -> list[dict]:
    """Empty when the set hasn't been built, so a checkout without it
    still runs on the three original bands."""
    if not EXPERT_QUERIES.exists():
        return []
    return json.loads(EXPERT_QUERIES.read_text())


def load_reference_queries() -> list[dict]:
    """Every labelled question, all four bands."""
    return load_eval_queries() + load_expert_queries()
