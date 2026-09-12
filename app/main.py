import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.cascade import run_cascade
from app.db import init_db, log_cascade

app = FastAPI(title="LLM Router")
DEMO_HTML_PATH = Path(__file__).resolve().parent / "demo.html"


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str | None = None  # ignored; the cascade always decides the tier


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/demo", response_class=HTMLResponse)
def demo():
    return DEMO_HTML_PATH.read_text()


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    try:
        result = run_cascade([m.model_dump() for m in req.messages])
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"upstream call failed: {exc}") from exc

    log_cascade(
        query=req.messages[-1].content,
        difficulty=result["difficulty"],
        initial_tier=result["initial_tier"],
        final_tier=result["final_tier"],
        escalated=result["escalated"],
        escalation_reasons=result["escalation_reasons"],
        total_cost=result["total_cost"],
        total_latency_ms=result["total_latency_ms"],
        tokens_in=result["tokens_in"],
        tokens_out=result["tokens_out"],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result["final_tier"],
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result["text"]},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": result["tokens_in"],
            "completion_tokens": result["tokens_out"],
            "total_tokens": (result["tokens_in"] or 0) + (result["tokens_out"] or 0),
        },
        "_router": {
            "difficulty": result["difficulty"],
            "initial_tier": result["initial_tier"],
            "final_tier": result["final_tier"],
            "escalated": result["escalated"],
            "escalation_reasons": result["escalation_reasons"],
            "cost": result["total_cost"],
            "latency_ms": round(result["total_latency_ms"], 1),
        },
    }
