"""Every file the app and scripts read or write, in one place.

Paths used to be rebuilt in each module from `Path(__file__).parent.parent`,
which broke silently whenever a file moved. Import them from here instead.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# The labelled eval set: the classifier's reference examples and the
# questions calibration grades every tier on.
EVAL_DIR = DATA / "eval"
EVAL_QUERIES = EVAL_DIR / "eval_queries.json"
EXPERT_QUERIES = EVAL_DIR / "expert_queries.json"
SAMPLE_QUERIES = EVAL_DIR / "sample_queries.json"
JUDGE_SCORES = EVAL_DIR / "judge_scores.json"
ACTIVE_LEARNING_CANDIDATES = EVAL_DIR / "active_learning_candidates.json"

# Measured quality and cost per tier for the built-in stack.
BUILTIN_CALIBRATION = DATA / "calibration" / "builtin.json"

# The learned verifier and the data it is trained on.
SCORER_DIR = DATA / "scorer"
SCORER_MODEL = SCORER_DIR / "answer_scorer.joblib"


def scorer_training_data(cheap_model: str) -> Path:
    return SCORER_DIR / f"training_{cheap_model.replace('/', '-').replace(':', '-')}.jsonl"


# Benchmarks run to compare tiers, with their per-model results.
PROBE_DIR = DATA / "probe"

# Logged cascade runs, eval responses and scores (local SQLite; Turso when
# TURSO_DATABASE_URL is set -- see app/storage/connection.py).
DEFAULT_DB = ROOT / "logs" / "router.db"

# Front-end.
WEB_DIR = ROOT / "app" / "web"
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"
