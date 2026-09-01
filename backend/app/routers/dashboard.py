from fastapi import APIRouter, HTTPException, Depends, Header, Query, Request
from sqlalchemy.orm import Session
from sqlalchemy import text
from backend.app.services.database import get_db
from backend.app.services import request_auth
from backend.app.services.qdrant_service import get_client_product_count
from datetime import datetime
from typing import Optional

router = APIRouter()


# Derived from the plugins, not invented: `semantic-search-woo` and
# Czargroup/AISearch are the only things that build /api/dashboard/stats,
# /api/analytics/* and /api/status. Neither Q&A module reports through here —
# their admin screens read /api/token-usage/me/*, which has its own gate.
#
# A module that starts calling these endpoints 403s until it is added, which is
# the loud direction of failure and the one this gate exists to produce.
_REPORTING_PRODUCTS = frozenset({"woo_search", "magento_search"})


def get_client(
    request: Request,
    db: Session,
    authorization: Optional[str],
    license_key: Optional[str],
) -> dict:
    """Authenticate a reporting caller and return their license_data.

    Was a bare validate_license_key plus a separate header-or-query helper,
    which resolved no v2 key, applied no domain gate and bound no tenant
    context. request_auth does all three and resolves header-then-query itself,
    so the second helper had nothing left to do.
    """
    return request_auth.authorize_request(
        request=request,
        db=db,
        authorization=authorization,
        x_api_key=None,
        request_license=license_key,
        allowed_products=_REPORTING_PRODUCTS,
    )

    raise HTTPException(status_code=401, detail="Missing Authorization header")


# ─── Dashboard Stats ───────────────────────────────────────────────────────────

@router.get("/dashboard/stats")
def dashboard_stats(
    request: Request,
    authorization: Optional[str] = Header(None),
    license_key: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """
    Returns everything the dashboard overview needs in one call.
    """
    client = get_client(request, db, authorization, license_key)

    client_id = client["client_id"]
    domain    = client["domain"]
    month     = datetime.utcnow().strftime("%Y-%m")

    # Usage this month
    usage = db.execute(text("""
        -- The archive, not usage_logs: the v2 migration renamed the table and
        -- this SELECT was 500ing the whole endpoint. Reading history rather
        -- than reporting a zero it never measured.
        SELECT search_count, ingest_count
        FROM usage_logs_archive_v1
        WHERE client_id = :client_id AND month = :month
    """), {"client_id": client_id, "month": month}).fetchone()

    search_count = usage.search_count if usage else 0
    ingest_count = usage.ingest_count if usage else 0

    # Products indexed in Qdrant
    indexed_count = get_client_product_count(client_id, domain)

    # Search quota percentage
    search_limit   = client["search_limit"]
    product_limit  = client["product_limit"]
    search_pct     = round((search_count / search_limit * 100), 1) if search_limit else 0
    product_pct    = round((indexed_count / product_limit * 100), 1) if product_limit else 0

    # Last 10 searches
    recent = db.execute(text("""
        SELECT query, results_count, response_time_ms, cached, searched_at
        FROM search_logs
        WHERE client_id = :client_id
        ORDER BY searched_at DESC
        LIMIT 10
    """), {"client_id": client_id}).fetchall()

    return {
        "client_name":    client["client_name"],
        "plan":           client["plan"],
        "month":          month,

        "searches": {
            "used":       search_count,
            "limit":      search_limit,
            "percentage": search_pct
        },

        "products": {
            "indexed":    indexed_count,
            "limit":      product_limit,
            "percentage": product_pct
        },

        "ingestions": {
            "this_month": ingest_count
        },

        "recent_searches": [
            {
                "query":           row.query,
                "results_count":   row.results_count,
                "response_time_ms": row.response_time_ms,
                "cached":          bool(row.cached),
                "searched_at":     row.searched_at.isoformat()
            }
            for row in recent
        ]
    }


# ─── Analytics ─────────────────────────────────────────────────────────────────

@router.get("/analytics/top-queries")
def top_queries(
    request: Request,
    authorization: Optional[str] = Header(None),
    license_key: Optional[str] = Query(None),
    days: int = 7,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Most searched queries in the last N days."""
    client    = get_client(request, db, authorization, license_key)

    client_id = client["client_id"]

    rows = db.execute(text("""
        SELECT query, COUNT(*) as count,
               AVG(results_count) as avg_results,
               AVG(response_time_ms) as avg_time
        FROM search_logs
        WHERE client_id = :client_id
          AND searched_at >= DATE_SUB(NOW(), INTERVAL :days DAY)
        GROUP BY query
        ORDER BY count DESC
        LIMIT :limit
    """), {"client_id": client_id, "days": days, "limit": limit}).fetchall()

    return {
        "days":    days,
        "queries": [
            {
                "query":       row.query,
                "count":       row.count,
                "avg_results": round(float(row.avg_results), 1),
                "avg_time_ms": round(float(row.avg_time))
            }
            for row in rows
        ]
    }


@router.get("/analytics/zero-results")
def zero_results(
    request: Request,
    authorization: Optional[str] = Header(None),
    license_key: Optional[str] = Query(None),
    days: int = 7,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Queries that returned zero results — shows gaps in catalog."""
    client    = get_client(request, db, authorization, license_key)

    client_id = client["client_id"]

    rows = db.execute(text("""
        SELECT query, COUNT(*) as count
        FROM search_logs
        WHERE client_id   = :client_id
          AND results_count = 0
          AND searched_at >= DATE_SUB(NOW(), INTERVAL :days DAY)
        GROUP BY query
        ORDER BY count DESC
        LIMIT :limit
    """), {"client_id": client_id, "days": days, "limit": limit}).fetchall()

    return {
        "days":    days,
        "queries": [
            {"query": row.query, "count": row.count}
            for row in rows
        ]
    }


@router.get("/analytics/summary")
def analytics_summary(
    request: Request,
    authorization: Optional[str] = Header(None),
    license_key: Optional[str] = Query(None),
    days: int = 7,
    db: Session = Depends(get_db)
):
    """Cache hit rate, avg response time, daily volume for charts."""
    client    = get_client(request, db, authorization, license_key)

    client_id = client["client_id"]

    # Overall stats
    stats = db.execute(text("""
        SELECT
            COUNT(*)                                        as total_searches,
            SUM(CASE WHEN cached = 1 THEN 1 ELSE 0 END)   as cached_count,
            SUM(CASE WHEN results_count = 0 THEN 1 ELSE 0 END) as zero_count,
            AVG(response_time_ms)                          as avg_time,
            AVG(CASE WHEN cached = 0
                THEN response_time_ms END)                 as avg_time_uncached
        FROM search_logs
        WHERE client_id = :client_id
          AND searched_at >= DATE_SUB(NOW(), INTERVAL :days DAY)
    """), {"client_id": client_id, "days": days}).fetchone()

    total   = stats.total_searches or 0
    cached  = stats.cached_count   or 0
    zero    = stats.zero_count      or 0

    cache_hit_rate  = round((cached / total * 100), 1) if total else 0
    zero_result_pct = round((zero   / total * 100), 1) if total else 0

    # Daily volume for Chart.js bar chart
    daily = db.execute(text("""
        SELECT
            DATE(searched_at) as day,
            COUNT(*)          as count
        FROM search_logs
        WHERE client_id = :client_id
          AND searched_at >= DATE_SUB(NOW(), INTERVAL :days DAY)
        GROUP BY DATE(searched_at)
        ORDER BY day ASC
    """), {"client_id": client_id, "days": days}).fetchall()

    return {
        "days":              days,
        "total_searches":    total,
        "cache_hit_rate":    cache_hit_rate,
        "zero_result_rate":  zero_result_pct,
        "avg_response_ms":   round(float(stats.avg_time or 0)),
        "avg_uncached_ms":   round(float(stats.avg_time_uncached or 0)),

        # Chart.js data — labels + values arrays
        "daily_volume": {
            "labels": [str(row.day) for row in daily],
            "values": [row.count    for row in daily]
        }
    }


# ─── Status Check ──────────────────────────────────────────────────────────────

@router.get("/status")
def status_check(
    request: Request,
    authorization: Optional[str] = Header(None),
    license_key: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Full status check — used by the Status tab in the plugin."""
    client    = get_client(request, db, authorization, license_key)

    client_id = client["client_id"]
    domain    = client["domain"]

    indexed_count = get_client_product_count(client_id, domain)

    # Check webhooks registered in WooCommerce
    # (We store webhook IDs in license_keys table — added later)
    # For now just return license info

    return {
        "api_reachable":   True,
        "license_valid":   True,
        "client_name":     client["client_name"],
        "client_id":       client_id,
        "plan":            client["plan"],
        "domain":          client["domain"],
        "product_limit":   client["product_limit"],
        "search_limit":    client["search_limit"],
        "indexed_count":   indexed_count,
        "license_expires": client["license_expires"]
    }