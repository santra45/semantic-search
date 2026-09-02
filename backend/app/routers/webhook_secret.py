from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import text
from backend.app.services.database import get_db
from backend.app.services import request_auth
from fastapi import Depends

router = APIRouter()

# Derived from the plugins, not invented: `semantic-search-woo` is the only
# thing that registers a webhook secret. The Q&A plugin pushes changes rather
# than receiving webhooks, and neither Magento module uses this at all.
_WEBHOOK_PRODUCTS = frozenset({"woo_search"})

class WebhookSecretPayload(BaseModel):
    license_key: str
    webhook_secret: str


@router.post("/register-webhook-secret")
def register_webhook_secret(
    payload: WebhookSecretPayload,
    request: Request,
    db: Session = Depends(get_db)
):
    # This endpoint writes a shared secret that authenticates every subsequent
    # webhook, so it is the last place that should have been authenticating on
    # its own. It now goes through the shared chokepoint like everything else —
    # which also means a wrong key is a 403 rather than the 404 it used to
    # answer, since "no such licence" and "not your licence" are the same
    # refusal and telling them apart was a small oracle.
    license_info = request_auth.authorize_request(
        request=request,
        db=db,
        authorization=None,          # this endpoint takes the key in the body
        x_api_key=None,
        request_license=payload.license_key,
        allowed_products=_WEBHOOK_PRODUCTS,
    )
    client_id = license_info["client_id"]

    # save webhook secret
    db.execute(
        text("""
        UPDATE clients
        SET webhook_secret = :secret
        WHERE id = :client_id
        """),
        {
            "secret": payload.webhook_secret,
            "client_id": client_id
        }
    )

    db.commit()

    return {"status": "saved", "client_id": client_id}