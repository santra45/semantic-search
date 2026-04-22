from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from backend.app.routers import (
    chatbot,
    dashboard,
    health,
    ingest,
    magento,
    onboarding,
    search,
    sync,
    token_usage,
    webhook_secret,
    webhooks,
)
from backend.app.magento.chatbot.routers import (
    agent as magento_chatbot_agent,
    sync as magento_chatbot_sync,
)

app = FastAPI(
    title="Semantic Search API",
    version="0.1.0"
)

# Template setup
templates = Jinja2Templates(directory="backend/app/templates")

# Static files
app.mount("/static", StaticFiles(directory="backend/app/static"), name="static")

app.include_router(search.router, prefix="/api")
app.include_router(ingest.router,   prefix="/api")
app.include_router(webhooks.router, prefix="/api")
app.include_router(sync.router,     prefix="/api")
app.include_router(dashboard.router, prefix="/api")
app.include_router(webhook_secret.router, prefix="/api")
app.include_router(health.router,  prefix="/api")
app.include_router(token_usage.router, prefix="/api")
app.include_router(magento.router, prefix="/api")
app.include_router(chatbot.router, prefix="/api")
app.include_router(onboarding.router)

# Multi-agent Magento chatbot (LangGraph + per-tenant Magento REST).
# Chat history now lives in the Magento DB — this backend owns only the
# stateless agent endpoints and the sync ingest endpoints.
app.include_router(magento_chatbot_agent.router, prefix="/api")
app.include_router(magento_chatbot_sync.router, prefix="/api")

@app.get("/")
def root():
    return {"status": "ok", "message": "Semantic Search API is running"}