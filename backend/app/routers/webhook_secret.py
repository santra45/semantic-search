from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import text
from backend.app.services.database import get_db
from fastapi import Depends

router = APIRouter()

class WebhookSecretPayload(BaseModel):
    license_key: str
    webhook_secret: str


@router.post("/register-webhook-secret")
def register_webhook_secret(
    payload: WebhookSecretPayload,
    db: Session = Depends(get_db)
):

    # find client using license key
    client = db.execute(
        text("""
        SELECT id
        FROM clients
        WHERE license_key = :license_key
        """),
        {"license_key": payload.license_key}
    ).fetchone()

    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    # save webhook secret
    db.execute(
        text("""
        UPDATE clients
        SET webhook_secret = :secret
        WHERE id = :client_id
        """),
        {
            "secret": payload.webhook_secret,
            "client_id": client.id
        }
    )

    db.commit()

    return {"status": "saved", "client_id": client.id}