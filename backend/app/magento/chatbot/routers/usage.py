"""
Tenant-scoped token usage endpoint.

  GET /api/magento/chatbot/usage/stats?days=30

Rolls up everything this tenant has been charged for in the backend's
`token_usage_tracking` ledger — embeddings (ingest + search), reranks, and
chat answers — and returns totals + daily series + per-query-type and
per-model breakdowns. Keyed by client_id from the license key.

No LLM is invoked here; just a MySQL read.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Header, Query, Request
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.services.database import get_db

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
            """
            SELECT
                COUNT(*)                    AS requests,
                COALESCE(SUM(input_tokens),0)  AS input_tokens,
                COALESCE(SUM(output_tokens),0) AS output_tokens,
                COALESCE(SUM(total_tokens),0)  AS total_tokens,
                COALESCE(SUM(total_cost),0)    AS total_cost
            FROM token_usage_tracking
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
            """
            SELECT query_type,
                   COUNT(*)                    AS requests,
                   COALESCE(SUM(input_tokens),0)  AS input_tokens,
                   COALESCE(SUM(output_tokens),0) AS output_tokens,
                   COALESCE(SUM(total_tokens),0)  AS total_tokens,
                   COALESCE(SUM(total_cost),0)    AS total_cost
            FROM token_usage_tracking
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
            """
            SELECT llm_provider, llm_model,
                   COUNT(*)                    AS requests,
                   COALESCE(SUM(total_tokens),0) AS total_tokens,
                   COALESCE(SUM(total_cost),0)   AS total_cost
            FROM token_usage_tracking
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
            """
            SELECT DATE(created_at) AS day,
                   COALESCE(SUM(total_tokens),0) AS tokens,
                   COALESCE(SUM(total_cost),0)   AS cost
            FROM token_usage_tracking
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
    }
