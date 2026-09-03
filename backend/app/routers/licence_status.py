"""GET /api/license/status and POST /api/telemetry/chat-quality.

ADMIN_CONSOLE_PLAN.md §8.2 and §8.4. Both are called by the merchant's own
plugin on a cron, both authenticate with the licence key, and NEITHER IS EVER
BILLED — see the note on _no_usage_row below.
"""
from __future__ import annotations

import logging
from datetime import date as date_type
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.services import catalog, licence_errors, licensing_service
from backend.app.services.database import get_db
from backend.app.services.request_auth import resolve_license_key

logger = logging.getLogger(__name__)

router = APIRouter(tags=["licence"])

# ── Why these two endpoints do not call authorize_request() ──────────────────
#
# authorize_request() binds a tenant context so downstream usage writers can
# attribute rows. That is exactly what must NOT happen here.
#
# A heartbeat polled every 15 minutes by five products across every store is,
# by a wide margin, the noisiest possible source of ledger rows — and a merchant
# must not pay to be told their subscription is paused, nor to report on
# themselves. So both endpoints resolve the key directly, skip the context
# binding, and write no usage_events row at all: not even a non-billable one.
#
# They also skip the domain gate, deliberately. It reads Origin/Referer, which a
# server-side cron does not send; requiring it would make the heartbeat work
# from a shopper's browser and fail from the scheduled job that is the entire
# point.
_no_usage_row = True  # documentation, referenced by the comment above


def _resolve_or_deny(db: Session, authorization: Optional[str],
                     request_license: Optional[str]) -> dict:
    key = resolve_license_key(authorization, request_license)
    if not key:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    context = licensing_service.resolve_key(db, key)
    if context is None:
        # Same structured shape as every other refusal, so a plugin polling this
        # endpoint learns the licence is dead in exactly the form it already
        # knows how to read. This is the PRIMARY way a widget finds out.
        raise licence_errors.denied_for_key(
            db, key, detail="License key is not valid for this request."
        )
    return context


@router.get("/api/license/status")
def licence_status(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_license_key: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Is this licence alive, and what does it allow?

    Cheap, cacheable, no LLM cost, never billed. The plugin's cron polls it and
    refreshes a short-lived cache entry; that cache is what bounds re-enable
    latency, since the reactive path only reacts once a shopper has already been
    refused.

    `poll_after` is served BY THE SERVER so the cadence can be retuned later
    without another plugin release across five products.
    """
    ctx = _resolve_or_deny(db, authorization, x_license_key)

    subscription = db.execute(
        text("""
            SELECT s.disabled_reason, s.expires_at, s.status,
                   si.indexed_items, si.catalogue_limit, si.environment,
                   c.name AS client_name
            FROM subscriptions s
            JOIN sites si  ON si.id = s.site_id
            JOIN clients c ON c.id = si.client_id
            WHERE s.id = :sid
        """),
        {"sid": ctx.get("subscription_id")},
    ).fetchone()

    period_requests = db.execute(
        text("""
            SELECT billable_requests FROM usage_counters
            WHERE subscription_id = :sid
              AND period = DATE_FORMAT(UTC_TIMESTAMP(), '%Y-%m')
        """),
        {"sid": ctx.get("subscription_id")},
    ).scalar()

    return {
        "active": True,
        "status": licence_errors.STATUS_ACTIVE,
        "client_name": subscription.client_name if subscription else None,
        "environment": ctx.get("environment"),
        "product": ctx.get("product_code"),
        "plan": ctx.get("plan"),
        "limits": {
            "catalogue_limit": int(subscription.catalogue_limit) if subscription else None,
            "request_limit": ctx.get("request_limit"),
        },
        "usage": {
            "indexed_items": int(subscription.indexed_items) if subscription else None,
            # None, not 0: no counter row for this period means nothing billable
            # has been recorded yet, which is not the same as zero requests.
            "requests_this_period": int(period_requests) if period_requests is not None else None,
        },
        "expires_at": ctx.get("licence_expires_at"),
        "merchant_message": subscription.disabled_reason if subscription else None,
        # Seconds. 15 minutes is the figure §9.1 bounds restore latency on.
        "poll_after": 900,
    }


# ── Chat quality telemetry ───────────────────────────────────────────────────

class ChatQualityRatings(BaseModel):
    up: int = Field(default=0, ge=0)
    down: int = Field(default=0, ge=0)


class ChatQualityBody(BaseModel):
    """Counts and timings. NOTHING a shopper typed.

    If a field is ever proposed here that carries message text, a query, or a
    customer identifier, that is the privacy decision being reopened — not a
    schema tweak. Conversations live in the merchant's own database on purpose.
    """
    date: date_type
    store_code: str = Field(default="default", max_length=64)
    sessions: int = Field(default=0, ge=0)
    messages: int = Field(default=0, ge=0)
    avg_response_ms: int = Field(default=0, ge=0)
    p95_response_ms: int = Field(default=0, ge=0)
    ratings: ChatQualityRatings = Field(default_factory=ChatQualityRatings)
    by_agent: dict[str, int] = Field(default_factory=dict)
    escalations: int = Field(default=0, ge=0)
    zero_result_turns: int = Field(default=0, ge=0)


@router.post("/api/telemetry/chat-quality")
def post_chat_quality(
    body: ChatQualityBody,
    authorization: Optional[str] = Header(None),
    x_license_key: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Daily aggregate from the merchant's own database.

    UPSERT on (subscription_id, store_code, date), which is what makes the cron
    safely re-runnable and the 30-day first-run backfill unable to double-count.
    A re-post for the same day CORRECTS the row.

    Keyed on subscription_id rather than (client_id, product_code): the licence
    already resolves to exactly one subscription, so there is nothing to infer,
    and it stays correct when one client runs two stores on the same product —
    which the client+product key would silently collide.
    """
    ctx = _resolve_or_deny(db, authorization, x_license_key)
    subscription_id = ctx.get("subscription_id")

    import json
    db.execute(
        text("""
            INSERT INTO chat_quality_daily
                (subscription_id, store_code, date, sessions, messages,
                 avg_response_ms, p95_response_ms, rating_up, rating_down,
                 escalations, zero_result_turns, by_agent_json)
            VALUES
                (:sid, :store, :date, :sessions, :messages,
                 :avg_ms, :p95_ms, :up, :down, :esc, :zero, :agents)
            ON DUPLICATE KEY UPDATE
                sessions          = VALUES(sessions),
                messages          = VALUES(messages),
                avg_response_ms   = VALUES(avg_response_ms),
                p95_response_ms   = VALUES(p95_response_ms),
                rating_up         = VALUES(rating_up),
                rating_down       = VALUES(rating_down),
                escalations       = VALUES(escalations),
                zero_result_turns = VALUES(zero_result_turns),
                by_agent_json     = VALUES(by_agent_json),
                received_at       = CURRENT_TIMESTAMP
        """),
        {
            "sid": subscription_id,
            "store": body.store_code or "default",
            "date": body.date,
            "sessions": body.sessions,
            "messages": body.messages,
            "avg_ms": body.avg_response_ms,
            "p95_ms": body.p95_response_ms,
            "up": body.ratings.up,
            "down": body.ratings.down,
            "esc": body.escalations,
            "zero": body.zero_result_turns,
            # Agent NAMES and counts. Names are ours, not the shopper's.
            "agents": json.dumps(body.by_agent) if body.by_agent else None,
        },
    )
    db.commit()

    logger.info(
        "telemetry: chat quality for %s %s (%s) — %d sessions, %d messages",
        ctx.get("product_code"), body.date, body.store_code, body.sessions, body.messages,
    )
    return {"success": True, "date": str(body.date), "store_code": body.store_code}
