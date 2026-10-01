"""Bring-your-own-model: provider connections, their models, and the
calibration that turns a model into routing numbers."""
import sqlite3

from fastapi import APIRouter, Depends, HTTPException

from app.api.common import KNOWN_PROVIDERS, now, resolve_provider_key
from app.api.schemas import AddModelRequest, CalibrateModelRequest, CreateProviderRequest
from app.auth import get_current_user
from app.evaluation.calibration import DEFAULT_MAX_QUERIES, calibrate_model
from app.routing.cascade import DEFAULT_TIER_MODELS
from app.storage import chat_db, key_vault

router = APIRouter()


@router.get("/api/provider-types")
def api_provider_types(user=Depends(get_current_user)):
    return {"providers": KNOWN_PROVIDERS, "key_storage_available": key_vault.storage_available()}


@router.get("/api/providers")
def api_list_providers(user=Depends(get_current_user)):
    by_provider = {}
    for m in chat_db.list_user_models(user["id"]):
        by_provider.setdefault(m["provider_id"], []).append({"id": m["id"], "model_name": m["model_name"]})
    return [
        {
            "id": p["id"], "name": p["name"], "provider": p["provider"],
            "api_base": p["api_base"], "created_at": p["created_at"],
            "has_stored_key": bool(p["has_stored_key"]),
            "models": by_provider.get(p["id"], []),
        }
        for p in chat_db.list_providers(user["id"])
    ]


@router.post("/api/providers")
def api_create_provider(req: CreateProviderRequest, user=Depends(get_current_user)):
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name must not be empty")
    if not req.provider.strip():
        raise HTTPException(status_code=400, detail="provider must not be empty")

    encrypted = None
    if req.store_key:
        if not req.api_key:
            raise HTTPException(status_code=400, detail="no key given to store")
        try:
            encrypted = key_vault.encrypt(req.api_key)
        except key_vault.KeyStorageUnavailable as e:
            # Never silently fall back to not-storing (the user thinks it's
            # saved) or to plaintext (worse). Fail loudly.
            raise HTTPException(status_code=400, detail=str(e))

    timestamp = now()
    try:
        provider_id = chat_db.create_provider(
            user["id"], name, req.provider.strip(), encrypted, req.api_base, timestamp,
        )
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail=f"you already have a provider named '{name}'")

    for model_name in req.models:
        if model_name.strip():
            chat_db.add_provider_model(provider_id, model_name.strip(), timestamp)
    return {"id": provider_id, "stored_key": encrypted is not None}


@router.delete("/api/providers/{provider_id}")
def api_delete_provider(provider_id: int, user=Depends(get_current_user)):
    if not chat_db.delete_provider(provider_id, user["id"]):
        raise HTTPException(status_code=404, detail="provider not found")
    return {"ok": True}


@router.post("/api/providers/{provider_id}/models")
def api_add_provider_model(provider_id: int, req: AddModelRequest, user=Depends(get_current_user)):
    if chat_db.get_provider(provider_id, user["id"]) is None:
        raise HTTPException(status_code=404, detail="provider not found")
    model_name = req.model_name.strip()
    if not model_name:
        raise HTTPException(status_code=400, detail="model_name must not be empty")
    chat_db.add_provider_model(provider_id, model_name, now())
    return {"ok": True}


@router.delete("/api/provider-models/{model_id}")
def api_delete_provider_model(model_id: int, user=Depends(get_current_user)):
    if not chat_db.delete_provider_model(model_id, user["id"]):
        raise HTTPException(status_code=404, detail="model not found")
    return {"ok": True}


@router.get("/api/my-models")
def api_my_models(user=Depends(get_current_user)):
    """The pool the chat-creation dropdowns pick from, each annotated with
    whether it has been calibrated -- which is what gates starting a chat."""
    calibrations = chat_db.get_model_calibrations(user["id"])
    out = []
    for m in chat_db.list_user_models(user["id"]):
        cal = calibrations.get(m["model_name"])
        out.append({
            "id": m["id"], "model_name": m["model_name"], "provider_id": m["provider_id"],
            "provider_name": m["provider_name"], "provider": m["provider"],
            "has_stored_key": bool(m["has_stored_key"]),
            "calibrated": cal is not None,
            "quality_by_difficulty": cal["quality_by_difficulty"] if cal else None,
            "avg_cost": cal["avg_cost"] if cal else None,
            "avg_latency_ms": cal["avg_latency_ms"] if cal else None,
        })
    return out


@router.post("/api/calibrate-model")
def api_calibrate_model(req: CalibrateModelRequest, user=Depends(get_current_user)):
    """Calibrate one connected model. Keyed by model, so the result is
    reused wherever that model is slotted later."""
    return calibrate_one_model(user["id"], req.model_name, req.api_key)


def calibrate_one_model(user_id: int, raw_model_name: str, supplied_key: str | None) -> dict:
    model_name = raw_model_name.strip()
    if chat_db.get_model_with_provider(user_id, model_name) is None:
        raise HTTPException(status_code=404, detail=f"'{model_name}' is not a connected model")

    api_key = resolve_provider_key(user_id, model_name, supplied_key)
    stats = calibrate_model(model_name, api_key, max_queries=DEFAULT_MAX_QUERIES)

    warnings = []
    if stats["n_queries"] == 0:
        if stats["n_rate_limited"] == stats["n_errors"]:
            raise HTTPException(
                status_code=502,
                detail=f"Rate limited on all {stats['n_errors']} calibration calls -- this is a "
                       f"provider quota issue, not a bad model string. Try again later.",
            )
        raise HTTPException(
            status_code=502,
            detail=f"All {stats['n_errors']} calibration calls failed. The provider said: "
                   f"{stats['first_error'] or 'no error detail returned'}",
        )

    unmeasured = [d for d, v in stats["quality_by_difficulty"].items() if v is None]
    if unmeasured:
        cause = "rate limiting" if stats["n_rate_limited"] else "errors on those queries"
        warnings.append(
            f"No {'/'.join(unmeasured)} score -- every query at that difficulty failed ({cause}). "
            f"Routing for it falls back to the strongest model in the session."
        )
    elif stats["n_rate_limited"]:
        warnings.append(
            f"Lost {stats['n_rate_limited']} queries to rate limiting; scores come from the "
            f"{stats['n_queries']} that completed."
        )

    chat_db.set_model_calibration(
        user_id, model_name, stats["avg_quality"], stats["avg_cost"],
        stats["avg_latency_ms"], stats["n_queries"], now(),
        quality_by_difficulty=stats["quality_by_difficulty"],
        cost_by_difficulty=stats["cost_by_difficulty"],
    )
    return {"model_name": model_name, "stats": stats, "warnings": warnings}


@router.get("/api/tiers")
def api_tiers(user=Depends(get_current_user)):
    """The models behind our own built-in tiers, so the chat UI can show real
    names. A BYOM session's own models come from the chat itself."""
    return {"builtin": DEFAULT_TIER_MODELS}


@router.get("/api/calibration")
def api_get_calibration(user=Depends(get_current_user)):
    """Every model this user has calibrated, keyed by model rather than by
    tier -- one measurement, reusable in any slot of any session."""
    calibrations = chat_db.get_model_calibrations(user["id"])
    return {"results": [{"model_name": name, **stats} for name, stats in sorted(calibrations.items())]}
