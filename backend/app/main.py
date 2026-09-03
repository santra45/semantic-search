from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from backend.app import config
from backend.app.middleware.logging_middleware import APILoggingMiddleware
from backend.app.services.licence_errors import LicenceDenied
from backend.app.routers import (
    dashboard,
    licence_status,
    health,
    magento,
    onboarding,
    operator,
    search,
    sync,
    token_usage,
    webhook_secret,
    webhooks,
)
from backend.app.magento.chatbot.routers import (
    retrieve as magento_chatbot_retrieve,
    sync as magento_chatbot_sync,
    usage as magento_chatbot_usage,
    classify as magento_chatbot_classify,
    agent as magento_chatbot_agent,
)
from backend.app.wordpress.productqa.routers import (
    retrieve as wordpress_productqa_retrieve,
    sync as wordpress_productqa_sync,
)

# The interactive docs are OFF unless AICHATBOT_ENABLE_DOCS is set.
#
# Passing None does not merely hide them — FastAPI never registers the routes,
# so /docs, /redoc and /openapi.json 404 exactly like any other unknown path.
# A 403 would confirm the endpoint exists; a 404 says nothing. The schema is a
# complete index of every licensing, sync, operator and admin route with its
# payload shape and its auth header, so it is worth not publishing.
#
# Swagger's oauth2-redirect route is registered only alongside docs_url, so it
# disappears with them; nothing else in the app reads /openapi.json.
_docs = config.ENABLE_API_DOCS

app = FastAPI(
    title="Semantic Search API",
    version="0.1.0",
    docs_url="/docs" if _docs else None,
    redoc_url="/redoc" if _docs else None,
    openapi_url="/openapi.json" if _docs else None,
)

# Unified request/response logger → logs/api.log
app.add_middleware(APILoggingMiddleware)

# Template setup
templates = Jinja2Templates(directory="backend/app/templates")

# Static files
app.mount("/static", StaticFiles(directory="backend/app/static"), name="static")

app.include_router(search.router, prefix="/api")
app.include_router(webhooks.router, prefix="/api")
app.include_router(sync.router,     prefix="/api")
app.include_router(dashboard.router, prefix="/api")
app.include_router(webhook_secret.router, prefix="/api")
app.include_router(health.router,  prefix="/api")
app.include_router(token_usage.router, prefix="/api")
app.include_router(magento.router, prefix="/api")
app.include_router(onboarding.router)
# Licence heartbeat + chat-quality telemetry. Absolute paths, no prefix.
# Neither writes a usage_events row: a merchant must not pay for a 15-minute
# poll telling them their subscription is paused, nor for reporting on
# themselves. See the note at the top of routers/licence_status.py.
app.include_router(licence_status.router)
# Operator analytics + cost console (Phase 4.4 + 4.5). Serves the HTML at
# /operator (shell only, no data) and gated JSON at /api/operator/* behind
# the X-Operator-Key header. No prefix — paths are absolute in the router.
app.include_router(operator.router)

# Admin console — real accounts, roles, audit (ADMIN_CONSOLE_PLAN.md §6).
# Carries its own /api/admin/auth prefix. Runs alongside /operator rather than
# replacing it; /api/operator/* is retired only once the SPA is live.
#
# Imported HERE and not at the top with the rest, inside a try. This package is
# new, touches no storefront path, and the API must run identically without it —
# so an ImportError from it has no business taking down uvicorn for every
# tenant, which at module scope is exactly what it would do. Admin routes go
# missing (404); merchants notice nothing.
try:
    from backend.app.admin import (
        router_auth as admin_router_auth,
        router_licences as admin_router_licences,
        router_tenants as admin_router_tenants,
        router_usage as admin_router_usage,
        router_write as admin_router_write,
    )

    app.include_router(admin_router_auth.router)
    app.include_router(admin_router_tenants.router)
    app.include_router(admin_router_licences.router)
    app.include_router(admin_router_usage.router)
    app.include_router(admin_router_write.router)
except Exception as _admin_exc:  # pragma: no cover
    import logging

    logging.getLogger(__name__).error(
        "admin console routes NOT mounted (%s: %s). Storefront traffic is "
        "unaffected; /api/admin/* will 404 until this is fixed.",
        type(_admin_exc).__name__, _admin_exc,
    )

# Magento chatbot backend — now pure retrieval.
# All routing / agent dispatch happens on the Magento side; this backend only
# answers three questions: "give me matching products", "give me matching
# content", and optionally "summarize these sources" (admin-toggled).
app.include_router(magento_chatbot_retrieve.router, prefix="/api")
app.include_router(magento_chatbot_sync.router, prefix="/api")
app.include_router(magento_chatbot_usage.router, prefix="/api")
app.include_router(magento_chatbot_classify.router, prefix="/api")
# Phase 3.1 — tool-calling intent router. Sits alongside /classify;
# Magento side picks which to call based on aichatbot/llm/tool_calling_mode.
app.include_router(magento_chatbot_agent.router, prefix="/api")

# WooCommerce per-product Q&A — the `ai-product-qa-woo` plugin.
# Kept in its own package rather than folded into the Magento routers above:
# the platforms disagree on the product lookup key (WooCommerce products often
# have no SKU, so lookups go by post ID) and the answer prompt should be
# tunable for one storefront without moving the other. Shared infrastructure
# (Qdrant, embeddings, licensing, token accounting) is still shared.
app.include_router(wordpress_productqa_retrieve.router, prefix="/api")
app.include_router(wordpress_productqa_sync.router, prefix="/api")

# ── Structured licence refusals (ADMIN_CONSOLE_PLAN.md §8.1) ────────────────
#
# LicenceDenied is an HTTPException subclass, so without this handler FastAPI's
# built-in one would render only {"detail": ...} and silently drop error_code,
# license_status and merchant_message — the fields the client-side kill switch
# is built on. Registered for the subclass specifically; every other
# HTTPException keeps FastAPI's default behaviour untouched.
@app.exception_handler(LicenceDenied)
async def _licence_denied_handler(request, exc: LicenceDenied):
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.body(),
        headers=exc.headers,
    )


@app.get("/")
def root():
    return {"status": "ok", "message": "Semantic Search API is running"}


# ── Admin console SPA (ADMIN_CONSOLE_PLAN.md §10.4) ─────────────────────────
#
# REGISTERED LAST, AND THAT IS NOT A STYLE CHOICE. The catch-all below matches
# /admin/<anything>; FastAPI resolves routes in registration order, so anything
# declared after it that shares a prefix becomes unreachable. Every /api route
# is declared above, which is what keeps them answering JSON instead of being
# handed index.html — a failure that looks like "the whole console broke at
# once" and reads as a frontend bug.
#
# The directory only exists after the build has run:
#     docker compose --profile build run --rm admin-ui-build
# so both the mount and the routes are conditional. A backend deployed without
# a built frontend serves the API exactly as before and answers /admin with a
# 503 that says what to run, rather than crashing at import time on a missing
# directory.
_ADMIN_DIST = Path(__file__).resolve().parent / "static" / "admin"
_ADMIN_INDEX = _ADMIN_DIST / "index.html"

# Created rather than checked. The mount is evaluated ONCE at import, and
# static files do not trigger uvicorn's --reload (it filters to *.py) — so a
# conditional mount that found no directory at boot would stay missing after the
# build ran, serving index.html with every asset 404ing until someone restarted
# the container. An empty directory mounts fine and 404s until the files land.
(_ADMIN_DIST / "assets").mkdir(parents=True, exist_ok=True)
app.mount(
    "/admin/assets",
    StaticFiles(directory=str(_ADMIN_DIST / "assets")),
    name="admin-assets",
)


@app.get("/admin")
@app.get("/admin/{path:path}")
def admin_spa(path: str = ""):
    """Serve the SPA shell for every /admin route.

    Client-side routing means /admin/tenants/abc is a real URL the user can
    bookmark and reload, but the server has no such route — it returns the shell
    and React Router reads the path. That is also why this cannot 404 on unknown
    paths: it has no way to tell a mistyped URL from a route it does not know
    about, and the SPA renders its own not-found.
    """
    if not _ADMIN_INDEX.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                "The admin console has not been built. Run: "
                "docker compose --profile build run --rm admin-ui-build"
            ),
        )
    return FileResponse(str(_ADMIN_INDEX))