"""The FastAPI app: wires the routers in app/api/ together, serves the
static front-end, and warms the classifier at boot."""
import logging
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import account, chats, demo, models, pages, usage, v1
from app.paths import STATIC_DIR
from app.routing.classifier import classify_initial_tier
from app.storage import chat_db
from app.storage import connection as dbconn
from app.storage.eval_log import init_db

# Platforms that scale to zero between requests. They matter twice below:
# the database must not live on their ephemeral disk, and warmup has to
# finish before the platform marks the instance ready.
SERVERLESS_ENV_VARS = ("K_SERVICE", "FLY_APP_NAME", "RAILWAY_ENVIRONMENT", "RENDER")

@asynccontextmanager
async def lifespan(app: FastAPI):
    on_startup()
    yield


app = FastAPI(title="LLM Router", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
for module in (pages, account, chats, demo, v1, models, usage):
    app.include_router(module.router)


@app.middleware("http")
async def no_cache_static(request, call_next):
    # This app is under active iteration -- a stale cached nav.js/app.css in
    # someone's browser (e.g. from before the Docs link was added) silently
    # hides real changes with no error to notice. Not a concern at this
    # scale/stage; revisit with real cache headers if this ever needs to
    # serve real traffic.
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, must-revalidate"
    return response


def on_startup():
    # Say where data is going, every boot. This used to be silent, and a
    # misconfiguration looked exactly like a working app until the next
    # restart took the accounts with it.
    log = logging.getLogger("uvicorn.error")
    log.info("database: %s", dbconn.describe())
    for problem in dbconn.check():
        log.warning("DATA LOSS RISK - %s", problem)

    init_db()
    chat_db.init_chat_db()

    # The embedding classifier takes ~14s to become usable: 7.7s importing
    # torch and sentence-transformers, 5.5s loading the MiniLM weights.
    # Left lazy, that lands on whoever sends the first prompt.
    def _warm():
        try:
            classify_initial_tier("warmup")
        except Exception:
            pass  # a failed warmup just means the first real call pays for it

    # On a scale-to-zero host, warm *before* reporting ready. Those platforms
    # bill by request and throttle CPU outside one, so a background thread
    # started here barely runs until traffic arrives -- and then the first
    # visitor waits for the whole load anyway, at throttled speed. Blocking
    # keeps the work inside the startup phase, which gets full CPU, so the
    # request that finally arrives is served warm. On a normal box there is
    # no such throttle and blocking boot would only slow development down.
    if any(os.getenv(v) for v in SERVERLESS_ENV_VARS):
        _warm()
    else:
        threading.Thread(target=_warm, daemon=True).start()


@app.get("/health")
def health():
    """Also reports whether data written here will survive a restart, so a
    misconfigured deploy is visible without reading container logs. Names the
    backend kind only -- never the URL or token."""
    durable = dbconn.using_turso() or not any(os.getenv(v) for v in SERVERLESS_ENV_VARS)
    return {
        "status": "ok",
        "database": "turso" if dbconn.using_turso() else "sqlite",
        "durable": durable,
        "warnings": dbconn.check(),
        "env": dbconn.env_report(),
    }
