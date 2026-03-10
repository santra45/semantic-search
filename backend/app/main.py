from fastapi import FastAPI
from backend.app.routers import search, ingest, webhooks, sync, dashboard, webhook_secret

app = FastAPI(
    title="Semantic Search API",
    version="0.1.0"
)

app.include_router(search.router, prefix="/api")
app.include_router(ingest.router,   prefix="/api")
app.include_router(webhooks.router, prefix="/api")
app.include_router(sync.router,     prefix="/api")
app.include_router(dashboard.router, prefix="/api")
app.include_router(webhook_secret.router, prefix="/api")


@app.get("/")
def root():
    return {"status": "ok", "message": "Semantic Search API is running"}