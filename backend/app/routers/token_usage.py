"""
Token Usage API Router

Provides endpoints for accessing token usage statistics and costs. Mounted at
/api in main.py:53 and consumed by four shipped admin panels -
Czargroup/AIProductQA, Czargroup/AISearch, semantic-search-woo and
ai-product-qa-woo all fetch the /me/* family.

WHY FIVE OF THESE ENDPOINTS WERE RETURNING 500
----------------------------------------------
Every raw-SQL query in this file named `token_usage_tracking`, which the v2
billing migration renamed to `token_usage_tracking_archive_v1`. Measured against
the running stack: /clients, /models, /hourly, /me/models and /me/hourly all
raised pymysql 1146 and returned HTTP 500. They now read
usage_ledger_read.LEDGER - the frozen v1 archive and the live v2 `usage_events`
presented as one table under the v1 column names - and each carries a
`usage_source` block saying which half its figures came from.

Read usage_ledger_read's docstring before changing any of this. The short
version: usage_events is empty until v2 licences are issued, so pointing these
at it alone would have replaced a 500 with a dashboard of confident zeros, and
during this window a zero here means NOT MEASURED rather than NO SPEND.

TWO THINGS IN THIS FILE THAT ARE STILL WRONG AND WERE NOT FIXED HERE
--------------------------------------------------------------------
1. THE FOUR TRACKER-BACKED ENDPOINTS READ THE ARCHIVE ONLY. /client/{id}/stats,
   /me/stats, /summary and /me/summary delegate to
   token_usage_service.TokenUsageTracker, which was repointed at
   token_usage_tracking_archive_v1 and stops there. They return 200 and they
   agree with the union endpoints today only because usage_events is empty. The
   day the first v2 row lands, /me/stats and /me/models in this same router
   start describing different universes, and the tracker's own `source` key
   inside `data` is the only thing that says which. The fix belongs in
   token_usage_service.get_client_usage_stats/get_usage_summary, not here -
   reimplementing their shape in the router would duplicate the thing this
   change exists to de-duplicate.

2. FOUR ENDPOINTS HAVE NO AUTHENTICATION AT ALL. /clients, /summary, /models and
   /hourly take no Authorization header and check nothing: they return
   all-tenant cost data, per-client_id, to anyone who can reach the port. The
   /me/* siblings directly above and below them each call _get_client_from_auth.
   This predates the migration and is untouched here on purpose - adding a gate
   is a behaviour change that would break whatever internal tooling calls them,
   and it does not belong inside a fix for a renamed table. It does need doing.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Header
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Optional, List
from datetime import datetime, timedelta

from backend.app.services.database import get_db
from backend.app.services.token_usage_service import TokenUsageTracker
from backend.app.services.license_service import validate_license_key, extract_license_key_from_authorization
from backend.app.services.usage_ledger_read import LEDGER, provenance
# Reused, not reimplemented: operator.py's gate is locked by default - with no
# AICHATBOT_OPERATOR_KEY configured it 403s rather than falling open, which is
# the behaviour every cross-tenant endpoint below needs.
from backend.app.routers.operator import require_operator

router = APIRouter(prefix="/token-usage", tags=["token-usage"])

def _get_client_from_auth(authorization: Optional[str], db: Session) -> dict:
    token = extract_license_key_from_authorization(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    try:
        return validate_license_key(token, db)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))

@router.get("/client/{client_id}/stats")
def get_client_usage_stats(
    client_id: str,
    start_date: Optional[datetime] = Query(None, description="Start date for filtering"),
    end_date: Optional[datetime] = Query(None, description="End date for filtering"),
        # CROSS-TENANT: this reads other people's billing data, so it is gated on
    # the operator key and not on a licence. It answered 200 to an unauthenticated
    # request until this guard existed - it had been masked only by a 500, because
    # it queried a table the v2 migration renamed away. Repointing it at the live
    # ledger turned an inert endpoint into an open one.
    _: bool = Depends(require_operator),
db: Session = Depends(get_db)
):
    """
    Get token usage statistics for a specific client.
    
    - **client_id**: Client identifier
    - **start_date**: Optional start date for filtering (ISO format)
    - **end_date**: Optional end date for filtering (ISO format)
    
    Returns detailed usage statistics broken down by query type, provider, and model.
    """
    try:
        tracker = TokenUsageTracker(db)
        stats = tracker.get_client_usage_stats(
            client_id=client_id,
            start_date=start_date,
            end_date=end_date
        )
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get client stats: {str(e)}")

@router.get("/me/stats")
def get_my_usage_stats(
    authorization: Optional[str] = Header(None),
    start_date: Optional[datetime] = Query(None, description="Start date for filtering"),
    end_date: Optional[datetime] = Query(None, description="End date for filtering"),
    db: Session = Depends(get_db),
):
    client = _get_client_from_auth(authorization, db)
    tracker = TokenUsageTracker(db)
    try:
        stats = tracker.get_client_usage_stats(
            client_id=client["client_id"],
            start_date=start_date,
            end_date=end_date,
        )
        return {"success": True, "data": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get client stats: {str(e)}")

@router.get("/summary")
def get_usage_summary(
    start_date: Optional[datetime] = Query(None, description="Start date for filtering"),
    end_date: Optional[datetime] = Query(None, description="End date for filtering"),
        # CROSS-TENANT: this reads other people's billing data, so it is gated on
    # the operator key and not on a licence. It answered 200 to an unauthenticated
    # request until this guard existed - it had been masked only by a 500, because
    # it queried a table the v2 migration renamed away. Repointing it at the live
    # ledger turned an inert endpoint into an open one.
    _: bool = Depends(require_operator),
db: Session = Depends(get_db)
):
    """
    Get token usage summary across all clients.
    
    - **start_date**: Optional start date for filtering (ISO format)
    - **end_date**: Optional end date for filtering (ISO format)
    
    Returns aggregated usage statistics for all clients.
    """
    try:
        tracker = TokenUsageTracker(db)
        summary = tracker.get_usage_summary(
            start_date=start_date,
            end_date=end_date
        )
        return {
            "success": True,
            "data": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get usage summary: {str(e)}")

@router.get("/me/summary")
def get_my_usage_summary(
    authorization: Optional[str] = Header(None),
    start_date: Optional[datetime] = Query(None, description="Start date for filtering"),
    end_date: Optional[datetime] = Query(None, description="End date for filtering"),
    db: Session = Depends(get_db),
):
    client = _get_client_from_auth(authorization, db)
    tracker = TokenUsageTracker(db)
    stats = tracker.get_client_usage_stats(
        client_id=client["client_id"],
        start_date=start_date,
        end_date=end_date,
    )
    totals = stats.get("totals", {})
    return {
        "success": True,
        "data": {
            "client_id": client["client_id"],
            "total_requests": totals.get("total_requests", 0),
            # Split as well as combined — the integrations' usage panels bill
            # input and output separately, and deriving one from the other is
            # not possible from a single figure.
            "total_input_tokens": totals.get("total_input_tokens", 0),
            "total_output_tokens": totals.get("total_output_tokens", 0),
            "total_tokens": totals.get("total_tokens", 0),
            "total_cost": totals.get("total_cost", 0.0),
            "period": stats.get("period", {}),
        },
    }

@router.get("/clients")
def get_active_clients(
    min_requests: int = Query(1, description="Minimum number of requests to be considered active"),
    days_back: int = Query(30, description="Number of days to look back for active clients"),
        # CROSS-TENANT: this reads other people's billing data, so it is gated on
    # the operator key and not on a licence. It answered 200 to an unauthenticated
    # request until this guard existed - it had been masked only by a 500, because
    # it queried a table the v2 migration renamed away. Repointing it at the live
    # ledger turned an inert endpoint into an open one.
    _: bool = Depends(require_operator),
db: Session = Depends(get_db)
):
    """
    Get list of active clients with their basic stats.
    
    - **min_requests**: Minimum number of requests to be considered active
    - **days_back**: Number of days to look back for activity
    
    Returns a list of clients with their request counts and costs.
    """
    try:
        start_date = datetime.utcnow() - timedelta(days=days_back)
        
        sql = text(f"""
        SELECT
            client_id,
            COUNT(*) as request_count,
            SUM(total_tokens) as total_tokens,
            SUM(total_cost) as total_cost,
            COUNT(DISTINCT query_type) as query_types_used,
            MIN(created_at) as first_request,
            MAX(created_at) as last_request
        FROM {LEDGER} u
        WHERE created_at >= :start_date
        GROUP BY client_id
        HAVING COUNT(*) >= :min_requests
        ORDER BY total_cost DESC
        """)
        
        result = db.execute(sql, {"start_date": start_date, "min_requests": min_requests})
        rows = result.fetchall()
        
        clients = []
        for row in rows:
            clients.append({
                "client_id": row.client_id,
                "request_count": row.request_count,
                "total_tokens": row.total_tokens,
                "total_cost": float(row.total_cost),
                "query_types_used": row.query_types_used,
                "first_request": row.first_request,
                "last_request": row.last_request
            })
        
        return {
            "success": True,
            "data": {
                "period": {
                    "start_date": start_date,
                    "days_back": days_back
                },
                "clients": clients,
                "usage_source": provenance(db),
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get active clients: {str(e)}")

@router.get("/models")
def get_model_usage(
    start_date: Optional[datetime] = Query(None, description="Start date for filtering"),
    end_date: Optional[datetime] = Query(None, description="End date for filtering"),
        # CROSS-TENANT: this reads other people's billing data, so it is gated on
    # the operator key and not on a licence. It answered 200 to an unauthenticated
    # request until this guard existed - it had been masked only by a 500, because
    # it queried a table the v2 migration renamed away. Repointing it at the live
    # ledger turned an inert endpoint into an open one.
    _: bool = Depends(require_operator),
db: Session = Depends(get_db)
):
    """
    Get usage statistics broken down by LLM models.
    
    - **start_date**: Optional start date for filtering (ISO format)
    - **end_date**: Optional end date for filtering (ISO format)
    
    Returns usage stats for each model across all providers.
    """
    try:
        where_clause = "WHERE 1=1"
        params = {}
        
        if start_date:
            where_clause += " AND created_at >= :start_date"
            params['start_date'] = start_date
            
        if end_date:
            where_clause += " AND created_at <= :end_date"
            params['end_date'] = end_date
        
        sql = text(f"""
        SELECT 
            llm_provider,
            llm_model,
            query_type,
            COUNT(*) as request_count,
            SUM(input_tokens) as total_input_tokens,
            SUM(output_tokens) as total_output_tokens,
            SUM(total_tokens) as total_tokens,
            SUM(input_cost) as total_input_cost,
            SUM(output_cost) as total_output_cost,
            SUM(total_cost) as total_cost,
            COUNT(DISTINCT client_id) as unique_clients,
            AVG(total_cost) as avg_cost_per_request
        FROM {LEDGER} u
        {where_clause}
        GROUP BY llm_provider, llm_model, query_type
        ORDER BY total_cost DESC
        """)
        
        result = db.execute(sql, params)
        rows = result.fetchall()
        
        model_stats = []
        for row in rows:
            model_stats.append({
                "llm_provider": row.llm_provider,
                "llm_model": row.llm_model,
                "query_type": row.query_type,
                "request_count": row.request_count,
                "total_input_tokens": row.total_input_tokens,
                "total_output_tokens": row.total_output_tokens,
                "total_tokens": row.total_tokens,
                "total_input_cost": float(row.total_input_cost),
                "total_output_cost": float(row.total_output_cost),
                "total_cost": float(row.total_cost),
                "unique_clients": row.unique_clients,
                "avg_cost_per_request": float(row.avg_cost_per_request)
            })
        
        return {
            "success": True,
            "data": {
                "period": {
                    "start_date": start_date,
                    "end_date": end_date
                },
                "models": model_stats,
                "usage_source": provenance(db),
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model usage: {str(e)}")

@router.get("/me/models")
def get_my_model_usage(
    authorization: Optional[str] = Header(None),
    start_date: Optional[datetime] = Query(None, description="Start date for filtering"),
    end_date: Optional[datetime] = Query(None, description="End date for filtering"),
    db: Session = Depends(get_db),
):
    client = _get_client_from_auth(authorization, db)
    try:
        where_clause = "WHERE client_id = :client_id"
        params = {"client_id": client["client_id"]}

        if start_date:
            where_clause += " AND created_at >= :start_date"
            params["start_date"] = start_date

        if end_date:
            where_clause += " AND created_at <= :end_date"
            params["end_date"] = end_date

        sql = text(f"""
        SELECT 
            llm_provider,
            llm_model,
            query_type,
            COUNT(*) as request_count,
            SUM(input_tokens) as total_input_tokens,
            SUM(output_tokens) as total_output_tokens,
            SUM(total_tokens) as total_tokens,
            SUM(input_cost) as total_input_cost,
            SUM(output_cost) as total_output_cost,
            SUM(total_cost) as total_cost,
            AVG(total_cost) as avg_cost_per_request
        FROM {LEDGER} u
        {where_clause}
        GROUP BY llm_provider, llm_model, query_type
        ORDER BY total_cost DESC
        """)

        result = db.execute(sql, params)
        rows = result.fetchall()
        models = []
        for row in rows:
            models.append({
                "llm_provider": row.llm_provider,
                "llm_model": row.llm_model,
                "query_type": row.query_type,
                "request_count": row.request_count,
                "total_input_tokens": row.total_input_tokens,
                "total_output_tokens": row.total_output_tokens,
                "total_tokens": row.total_tokens,
                "total_input_cost": float(row.total_input_cost or 0),
                "total_output_cost": float(row.total_output_cost or 0),
                "total_cost": float(row.total_cost or 0),
                "avg_cost_per_request": float(row.avg_cost_per_request or 0),
            })

        return {
            "success": True,
            "data": {
                "period": {"start_date": start_date, "end_date": end_date},
                "models": models,
                "usage_source": provenance(db),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model usage: {str(e)}")

@router.get("/hourly")
def get_hourly_usage(
    client_id: Optional[str] = Query(None, description="Filter by specific client"),
    hours_back: int = Query(24, description="Number of hours to look back"),
        # CROSS-TENANT: this reads other people's billing data, so it is gated on
    # the operator key and not on a licence. It answered 200 to an unauthenticated
    # request until this guard existed - it had been masked only by a 500, because
    # it queried a table the v2 migration renamed away. Repointing it at the live
    # ledger turned an inert endpoint into an open one.
    _: bool = Depends(require_operator),
db: Session = Depends(get_db)
):
    """
    Get hourly token usage breakdown.
    
    - **client_id**: Optional client ID to filter by
    - **hours_back**: Number of hours to look back (default: 24)
    
    Returns hourly usage data for time series analysis.
    """
    try:
        start_date = datetime.utcnow() - timedelta(hours=hours_back)
        
        where_clause = "WHERE created_at >= :start_date"
        params = {"start_date": start_date}
        
        if client_id:
            where_clause += " AND client_id = :client_id"
            params['client_id'] = client_id
        
        sql = text(f"""
        SELECT 
            DATE_FORMAT(created_at, '%Y-%m-%d %H:00:00') as hour,
            COUNT(*) as request_count,
            SUM(total_tokens) as total_tokens,
            SUM(total_cost) as total_cost,
            COUNT(DISTINCT client_id) as unique_clients
        FROM {LEDGER} u
        {where_clause}
        GROUP BY DATE_FORMAT(created_at, '%Y-%m-%d %H:00:00')
        ORDER BY hour ASC
        """)
        
        result = db.execute(sql, params)
        rows = result.fetchall()
        
        hourly_data = []
        for row in rows:
            hourly_data.append({
                "hour": row.hour,
                "request_count": row.request_count,
                "total_tokens": row.total_tokens,
                "total_cost": float(row.total_cost),
                "unique_clients": row.unique_clients
            })
        
        return {
            "success": True,
            "data": {
                "period": {
                    "start_date": start_date,
                    "hours_back": hours_back
                },
                "client_id": client_id,
                "hourly_data": hourly_data,
                "usage_source": provenance(db),
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get hourly usage: {str(e)}")

@router.get("/me/hourly")
def get_my_hourly_usage(
    authorization: Optional[str] = Header(None),
    hours_back: int = Query(24, description="Number of hours to look back"),
    db: Session = Depends(get_db),
):
    client = _get_client_from_auth(authorization, db)
    start_date = datetime.utcnow() - timedelta(hours=hours_back)

    where_clause = "WHERE created_at >= :start_date AND client_id = :client_id"
    params = {"start_date": start_date, "client_id": client["client_id"]}

    try:
        sql = text(f"""
        SELECT 
            DATE_FORMAT(created_at, '%Y-%m-%d %H:00:00') as hour,
            llm_provider,
            llm_model,
            query_type,
            COUNT(*) as request_count,
            SUM(total_tokens) as total_tokens,
            SUM(total_cost) as total_cost
        FROM {LEDGER} u
        {where_clause}
        GROUP BY DATE_FORMAT(created_at, '%Y-%m-%d %H:00:00'), llm_provider, llm_model, query_type
        ORDER BY hour ASC
        """)

        result = db.execute(sql, params)
        rows = result.fetchall()
        hourly_data = []
        for row in rows:
            hourly_data.append({
                "hour": row.hour,
                "llm_provider": row.llm_provider,
                "llm_model": row.llm_model,
                "query_type": row.query_type,
                "request_count": row.request_count,
                "total_tokens": row.total_tokens,
                "total_cost": float(row.total_cost or 0),
            })

        return {
            "success": True,
            "data": {
                "period": {"start_date": start_date, "hours_back": hours_back},
                "client_id": client["client_id"],
                "hourly_data": hourly_data,
                # An empty hourly_data over the default 24-hour window is the
                # EXPECTED result right now, not a bug: the archive's last row
                # predates today by weeks and the live ledger has none. Without
                # this block the four admin panels that render this chart show a
                # flat line indistinguishable from a quiet day.
                "usage_source": provenance(db),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get hourly usage: {str(e)}")
