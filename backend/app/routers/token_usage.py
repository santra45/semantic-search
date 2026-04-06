"""
Token Usage API Router

Provides endpoints for accessing token usage statistics and costs.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Header
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Optional, List
from datetime import datetime, timedelta

from backend.app.services.database import get_db
from backend.app.services.token_usage_service import TokenUsageTracker
from backend.app.services.license_service import validate_license_key, extract_license_key_from_authorization

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
    return {
        "success": True,
        "data": {
            "client_id": client["client_id"],
            "total_requests": stats.get("totals", {}).get("total_requests", 0),
            "total_tokens": stats.get("totals", {}).get("total_tokens", 0),
            "total_cost": stats.get("totals", {}).get("total_cost", 0.0),
            "period": stats.get("period", {}),
        },
    }

@router.get("/clients")
def get_active_clients(
    min_requests: int = Query(1, description="Minimum number of requests to be considered active"),
    days_back: int = Query(30, description="Number of days to look back for active clients"),
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
        
        sql = text("""
        SELECT 
            client_id,
            COUNT(*) as request_count,
            SUM(total_tokens) as total_tokens,
            SUM(total_cost) as total_cost,
            COUNT(DISTINCT query_type) as query_types_used,
            MIN(created_at) as first_request,
            MAX(created_at) as last_request
        FROM token_usage_tracking
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
                "clients": clients
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get active clients: {str(e)}")

@router.get("/models")
def get_model_usage(
    start_date: Optional[datetime] = Query(None, description="Start date for filtering"),
    end_date: Optional[datetime] = Query(None, description="End date for filtering"),
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
        FROM token_usage_tracking
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
                "models": model_stats
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
        FROM token_usage_tracking
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

        return {"success": True, "data": {"period": {"start_date": start_date, "end_date": end_date}, "models": models}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model usage: {str(e)}")

@router.get("/hourly")
def get_hourly_usage(
    client_id: Optional[str] = Query(None, description="Filter by specific client"),
    hours_back: int = Query(24, description="Number of hours to look back"),
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
        FROM token_usage_tracking
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
                "hourly_data": hourly_data
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
        FROM token_usage_tracking
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
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get hourly usage: {str(e)}")
