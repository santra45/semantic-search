from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from backend.app.routers import search, ingest, webhooks, sync, dashboard, webhook_secret, onboarding

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
app.include_router(onboarding.router)

@app.get("/")
def root():
    return {"status": "ok", "message": "Semantic Search API is running"}