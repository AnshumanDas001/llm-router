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
from app.routing import classifier
from app.storage import chat_db
from app.storage import connection as dbconn
from app.storage.eval_log import init_db

# Platforms that scale to zero between requests, whose disk doesn't survive
# a restart -- so the database must live elsewhere (see /health).
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

    # The embedding classifier takes ~14s to become usable on a laptop and a
    # minute or more on a throttled shared vCPU: importing torch and
    # sentence-transformers, then loading MiniLM and encoding the reference
    # set. It used to load before the server reported ready, so a cold start
    # showed a blank page for that whole minute. Now the pages are served at
    # once and the classifier loads in the background; GET /api/status says
    # when it's done, and the pages show a notice until then.
    #
    # On a scale-to-zero host CPU is throttled outside requests, so a
    # background thread alone would crawl. The pages long-poll /api/status
    # while they wait, which keeps a request open -- and the CPU allocated --
    # until the load finishes.
    threading.Thread(target=classifier.warm, daemon=True).start()


@app.get("/api/status")
def router_status(wait: float = 0):
    """Whether the router's classifier has loaded. Pass wait=N (up to 25)
    to hold the request until it's ready or N seconds pass."""
    return classifier.status(wait=max(0.0, min(float(wait), 25.0)))


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
        "router_ready": classifier.status()["ready"],
        "warnings": dbconn.check(),
        "env": dbconn.env_report(),
    }
