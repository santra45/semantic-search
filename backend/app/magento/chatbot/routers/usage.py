"""
Tenant-scoped token usage endpoint.

  GET /api/magento/chatbot/usage/stats?days=30

Rolls up everything this tenant has been charged for — embeddings (ingest +
search), reranks, and chat answers — and returns totals + daily series + per-
query-type and per-model breakdowns. Keyed by client_id from the license key.
Consumed by Czargroup/AIChatbot's admin dashboard (ApiClient.php:684 ->
Handle.php:193 -> admin-dashboard.js:700).

No LLM is invoked here; just a MySQL read.

WHAT CHANGED, AND WHY THIS WAS RETURNING 500
--------------------------------------------
All four queries below named `token_usage_tracking`, which the v2 billing
migration renamed to `token_usage_tracking_archive_v1`. Every call raised
pymysql 1146 straight out of the router - HTTP 500 on a live, authenticated,
merchant-facing endpoint. The bound client_id in the traceback was the correct
tenant, so authorisation was working perfectly and the failure was entirely
downstream of it.

They now read usage_ledger_read.LEDGER, which is the frozen v1 archive and the
live v2 `usage_events` presented as one table under the v1 column names. Not a
repoint to usage_events alone: that table is empty until v2 licences are issued,
so a repoint would have turned this 500 into a merchant staring at a dashboard
of zeros for their own spend.

WHAT THIS ENDPOINT ACTUALLY SHOWS TODAY
---------------------------------------
The tenant's real pre-migration spend that falls inside the `days` window, and
nothing after 2026-08-03 (the archive's last row). Traffic served since then is
recorded in neither ledger - v1 JWT keys resolve no v2 context, so
usage_service.record() refuses the row - which means a small `days` window
returns genuine, correct, EMPTY results, and it will keep doing so until the
first v2 licence is issued.

That is why the response now carries `usage_source`. Without it a merchant on
`days=7` sees a confident 0.00 for their monthly cost and has no way to tell it
from a measurement; with it, `usage_source.current` is false and the note says
in words that the figure is not current. Same reasoning as the `source` key
token_usage_service.get_client_usage_stats() returns. Every pre-existing key in
this response keeps its name and its meaning, because the PHP and JS above
render them and cannot be redeployed from this repo.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Header, Query, Request
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.services.database import get_db
from backend.app.services.usage_ledger_read import LEDGER, provenance

from backend.app.magento.chatbot.routers.common import authorize_request

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/magento/chatbot/usage/stats")
def usage_stats(
    request: Request,
    days: int = Query(30, ge=1, le=365),
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    license_data = authorize_request(
        request=request, db=db,
        authorization=authorization, x_api_key=x_api_key,
        request_license=None,
    )
    client_id = license_data["client_id"]
    since = datetime.utcnow() - timedelta(days=days)

    # Totals
    totals_row = db.execute(
        text(
            f"""
            SELECT
                COUNT(*)                    AS requests,
                COALESCE(SUM(input_tokens),0)  AS input_tokens,
                COALESCE(SUM(output_tokens),0) AS output_tokens,
                COALESCE(SUM(total_tokens),0)  AS total_tokens,
                COALESCE(SUM(total_cost),0)    AS total_cost
            FROM {LEDGER} u
            WHERE client_id = :client_id AND created_at >= :since
            """
        ),
        {"client_id": client_id, "since": since},
    ).fetchone()

    totals = {
        "requests":      int(totals_row.requests or 0)      if totals_row else 0,
        "input_tokens":  int(totals_row.input_tokens or 0)  if totals_row else 0,
        "output_tokens": int(totals_row.output_tokens or 0) if totals_row else 0,
        "total_tokens":  int(totals_row.total_tokens or 0)  if totals_row else 0,
        "total_cost":    round(float(totals_row.total_cost or 0), 8) if totals_row else 0.0,
    }

    # By query type
    by_query_type_rows = db.execute(
        text(
            f"""
            SELECT query_type,
                   COUNT(*)                    AS requests,
                   COALESCE(SUM(input_tokens),0)  AS input_tokens,
                   COALESCE(SUM(output_tokens),0) AS output_tokens,
                   COALESCE(SUM(total_tokens),0)  AS total_tokens,
                   COALESCE(SUM(total_cost),0)    AS total_cost
            FROM {LEDGER} u
            WHERE client_id = :client_id AND created_at >= :since
            GROUP BY query_type
            ORDER BY total_cost DESC
            """
        ),
        {"client_id": client_id, "since": since},
    ).fetchall()
    by_query_type = [
        {
            "query_type":    r.query_type,
            "requests":      int(r.requests or 0),
            "input_tokens":  int(r.input_tokens or 0),
            "output_tokens": int(r.output_tokens or 0),
            "total_tokens":  int(r.total_tokens or 0),
            "total_cost":    round(float(r.total_cost or 0), 8),
        }
        for r in by_query_type_rows
    ]

    # By model
    by_model_rows = db.execute(
        text(
            f"""
            SELECT llm_provider, llm_model,
                   COUNT(*)                    AS requests,
                   COALESCE(SUM(total_tokens),0) AS total_tokens,
                   COALESCE(SUM(total_cost),0)   AS total_cost
            FROM {LEDGER} u
            WHERE client_id = :client_id AND created_at >= :since
            GROUP BY llm_provider, llm_model
            ORDER BY total_cost DESC
            """
        ),
        {"client_id": client_id, "since": since},
    ).fetchall()
    by_model = [
        {
            "llm_provider": r.llm_provider,
            "llm_model":    r.llm_model,
            "requests":     int(r.requests or 0),
            "total_tokens": int(r.total_tokens or 0),
            "total_cost":   round(float(r.total_cost or 0), 8),
        }
        for r in by_model_rows
    ]

    # Daily series
    series_rows = db.execute(
        text(
            f"""
            SELECT DATE(created_at) AS day,
                   COALESCE(SUM(total_tokens),0) AS tokens,
                   COALESCE(SUM(total_cost),0)   AS cost
            FROM {LEDGER} u
            WHERE client_id = :client_id AND created_at >= :since
            GROUP BY DATE(created_at)
            ORDER BY day ASC
            """
        ),
        {"client_id": client_id, "since": since},
    ).fetchall()
    series = [
        {
            "day":    r.day.isoformat() if hasattr(r.day, "isoformat") else str(r.day),
            "tokens": int(r.tokens or 0),
            "cost":   round(float(r.cost or 0), 8),
        }
        for r in series_rows
    ]

    return {
        "range_days":    days,
        "totals":        totals,
        "by_query_type": by_query_type,
        "by_model":      by_model,
        "series":        series,
        # Added, never substituted for an existing key. Handle.php and
        # admin-dashboard.js read the four above by name and ignore what they do
        # not recognise, so this is safe to ship ahead of the admin panel that
        # will render it - and until it does, `usage_source.current` is the only
        # place the "these figures stop at the migration" fact exists at all.
        "usage_source":  provenance(db),
    }
