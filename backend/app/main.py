from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from backend.app.middleware.logging_middleware import APILoggingMiddleware
from backend.app.routers import (
    dashboard,
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

app = FastAPI(
    title="Semantic Search API",
    version="0.1.0"
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

@app.get("/")
def root():
    return {"status": "ok", "message": "Semantic Search API is running"}