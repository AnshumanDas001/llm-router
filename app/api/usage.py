"""Spend and savings: the usage dashboard and per-session history."""
from fastapi import APIRouter, Depends, HTTPException

from app.auth import get_current_user
from app.storage import chat_db

router = APIRouter()


@router.get("/api/usage")
def api_usage(user=Depends(get_current_user)):
    summary = chat_db.get_usage_summary(user["id"])
    total_cost = summary["totals"]["total_cost"]
    total_baseline = summary["totals"]["total_baseline_cost"]
    return {
        "n_calls": summary["totals"]["n_calls"],
        "total_cost": total_cost,
        "total_baseline_cost": total_baseline,
        "total_cost_saved": max(0.0, total_baseline - total_cost),
        "by_tier": summary["by_tier"],
        "log": [
            {
                "source": r["source"], "difficulty": r["difficulty"],
                "initial_tier": r["initial_tier"], "final_tier": r["final_tier"],
                "escalated": bool(r["escalated"]), "cost": r["cost"],
                "baseline_cost": r["baseline_cost"], "latency_ms": r["latency_ms"],
                "timestamp": r["timestamp"],
            }
            for r in chat_db.get_usage_log(user["id"], limit=100)
        ],
    }


@router.get("/api/sessions")
def api_sessions_overview(user=Depends(get_current_user)):
    """Chats and API-key sessions in one list."""
    return chat_db.get_sessions_overview(user["id"])


@router.get("/api/sessions/api/{key_id}")
def api_session_detail(key_id: int, user=Depends(get_current_user)):
    key, calls = chat_db.get_api_session_detail(key_id, user["id"])
    if key is None:
        raise HTTPException(status_code=404, detail="session not found")

    total_cost = sum(c["cost"] or 0.0 for c in calls)
    total_baseline = sum(c["baseline_cost"] or 0.0 for c in calls)
    return {
        "id": key["id"],
        "name": key["name"] or f"API session {key['key_prefix']}",
        "key_prefix": key["key_prefix"],
        "created_at": key["created_at"],
        "revoked": bool(key["revoked"]),
        "total_cost": total_cost,
        "total_baseline_cost": total_baseline,
        "cost_saved": max(0.0, total_baseline - total_cost),
        "calls": [
            {
                "difficulty": c["difficulty"], "initial_tier": c["initial_tier"],
                "final_tier": c["final_tier"], "escalated": bool(c["escalated"]),
                "cost": c["cost"], "baseline_cost": c["baseline_cost"],
                "latency_ms": c["latency_ms"], "timestamp": c["timestamp"],
            }
            for c in calls
        ],
    }
