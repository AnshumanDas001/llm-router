"""The HTML pages. Each is a static file from app/web/templates."""
from fastapi import APIRouter, Cookie, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse

from app.auth import get_current_user
from app.paths import TEMPLATES_DIR

router = APIRouter()


def _page(name: str) -> str:
    return (TEMPLATES_DIR / f"{name}.html").read_text()


@router.get("/", response_class=HTMLResponse)
def landing(router_session: str | None = Cookie(default=None)):
    """Signed-in users get the app as their home screen; the marketing page
    is for people who aren't in yet."""
    try:
        get_current_user(router_session)
        return RedirectResponse(url="/app", status_code=307)
    except HTTPException:
        return HTMLResponse(_page("landing"))


@router.get("/try", response_class=HTMLResponse)
def try_page():
    """The real chat app in demo mode: same page, one flag. Sends go to
    /api/try, capped per device; nothing is persisted."""
    return _page("chat_app").replace(
        "<script src=\"/static/nav.js\"></script>",
        "<script>window.THRIFT_DEMO = true;</script>\n<script src=\"/static/nav.js\"></script>", 1)


@router.get("/demo", response_class=HTMLResponse)
def demo():
    return _page("demo")


@router.get("/app", response_class=HTMLResponse)
def chat_app():
    return _page("chat_app")


@router.get("/settings", response_class=HTMLResponse)
def settings_page():
    return _page("settings")


@router.get("/models", response_class=HTMLResponse)
def models_page():
    return _page("models")


@router.get("/sessions", response_class=HTMLResponse)
def sessions_page():
    return _page("sessions")


@router.get("/guide", response_class=HTMLResponse)
def guide_page():
    # Deliberately not "/docs" -- FastAPI reserves that path for its own
    # auto-generated Swagger UI, and our custom route was shadowing it.
    return _page("guide")
