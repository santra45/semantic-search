"""/api/admin/usage, /audit, /health — read only.

ADMIN_CONSOLE_PLAN.md §7.1.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.admin import queries as q
from backend.app.admin.deps import AdminActor, require_viewer
from backend.app.services import usage_ledger_read
from backend.app.services.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin-read"])

# GROUP BY, like ORDER BY, cannot be parameterised — the value is SQL. So the
# cost explorer's dimensions are an allowlist of fragments written here, mapping
# a caller's short name to (select expression, group expression).
_USAGE_DIMENSIONS = {
    "client": ("c.name", "c.name"),
    "site": ("si.domain", "si.domain"),
    "product": ("ue.product_code", "ue.product_code"),
    "model": ("CONCAT(ue.provider, '/', ue.model)", "ue.provider, ue.model"),
    "call_type": ("ue.call_type", "ue.call_type"),
    "environment": ("si.environment", "si.environment"),
    "key_owner": ("ue.key_owner", "ue.key_owner"),
    "kind": ("ue.kind", "ue.kind"),
    "day": ("DATE(ue.created_at)", "DATE(ue.created_at)"),
}


@router.get("/usage")
def usage_explorer(
    group_by: str = Query("product", description="|".join(_USAGE_DIMENSIONS)),
    days: int = Query(30),
    environment: Optional[str] = Query(None),
    client_id: Optional[str] = Query(None, max_length=36),
    product: Optional[str] = Query(None, max_length=64),
    limit: int = Query(100),
    actor: AdminActor = Depends(require_viewer),
    db: Session = Depends(get_db),
):
    """Cost explorer over usage_events.

    usage_events ONLY, so this covers the v2 era. The v1 archive is not unioned
    in here even though usage_ledger_read.LEDGER could: LEDGER deliberately does
    not project `billable` or `product_code` because v1 has no equivalent, and
    every dimension worth grouping by is one of those or needs the site join
    that v1 rows cannot satisfy. Blending them would produce a per-product chart
    where half the rows are filed under a product nobody can name.

    `usage_source` on the response says which half is being read, and the
    archive totals are reported separately below rather than mixed in.
    """
    days = q.clamp_days(days)
    limit = q.clamp_limit(limit, default=100, ceiling=1000)
    if group_by not in _USAGE_DIMENSIONS:
        group_by = "product"
    select_expr, group_expr = _USAGE_DIMENSIONS[group_by]

    where = ["ue.created_at >= UTC_TIMESTAMP() - INTERVAL :days DAY"]
    params: dict = {"days": days, "limit": limit}
    if environment in ("development", "production"):
        where.append("si.environment = :environment")
        params["environment"] = environment
    if client_id:
        where.append("ue.client_id = :client_id")
        params["client_id"] = client_id
    if product:
        where.append("ue.product_code = :product")
        params["product"] = product

    where_sql = " AND ".join(where)

    rows_ = q.rows(db, f"""
        SELECT {select_expr}         AS bucket,
               {q.BILLABLE_REQUESTS} AS requests,
               {q.ALL_ROWS_COST}     AS cost,
               SUM(ue.total_tokens)  AS tokens,
               SUM(ue.input_tokens)  AS input_tokens,
               SUM(ue.output_tokens) AS output_tokens,
               COUNT(*)              AS ledger_rows
        FROM usage_events ue
        JOIN sites si  ON si.id = ue.site_id
        JOIN clients c ON c.id = ue.client_id
        WHERE {where_sql}
        GROUP BY {group_expr}
        ORDER BY cost DESC
        LIMIT :limit
    """, params)

    totals = q.one(db, f"""
        SELECT {q.BILLABLE_REQUESTS} AS requests,
               {q.ALL_ROWS_COST}     AS cost,
               SUM(ue.total_tokens)  AS tokens,
               COUNT(*)              AS ledger_rows
        FROM usage_events ue
        JOIN sites si  ON si.id = ue.site_id
        JOIN clients c ON c.id = ue.client_id
        WHERE {where_sql}
    """, {k: v for k, v in params.items() if k != "limit"})

    return {
        "group_by": group_by,
        "window_days": days,
        "dimensions": sorted(_USAGE_DIMENSIONS),
        "rows": [
            {"bucket": q.iso(r["bucket"]) if group_by == "day" else r["bucket"],
             "requests": q.i(r["requests"]), "cost": q.f(r["cost"]),
             "tokens": q.i(r["tokens"]), "input_tokens": q.i(r["input_tokens"]),
             "output_tokens": q.i(r["output_tokens"]),
             "ledger_rows": q.i(r["ledger_rows"])}
            for r in rows_
        ],
        "totals": {
            "requests": q.i_or_none(totals.get("requests")),
            "cost": q.f_or_none(totals.get("cost")),
            "tokens": q.i_or_none(totals.get("tokens")),
            # Rows vs requests, side by side and labelled. One customer action
            # is one billable row plus several non-billable ones, so these two
            # differing by ~4x is correct and expected — showing only one of
            # them is how a chart ends up quadruple-counting requests.
            "ledger_rows": q.i_or_none(totals.get("ledger_rows")),
        },
        "usage_source": usage_ledger_read.provenance(db),
    }


@router.get("/audit")
def audit_log(
    actor_email: Optional[str] = Query(None, max_length=255),
    action: Optional[str] = Query(None, max_length=64),
    target_type: Optional[str] = Query(None, max_length=32),
    target_id: Optional[str] = Query(None, max_length=64),
    days: int = Query(30),
    limit: int = Query(100),
    offset: int = Query(0),
    actor: AdminActor = Depends(require_viewer),
    db: Session = Depends(get_db),
):
    """Who changed what.

    Viewer-readable on purpose: an audit trail only half the team can see is a
    weaker deterrent and a worse debugging tool. Snapshots hold configuration —
    plans, limits, flags — not secrets.
    """
    days = q.clamp_days(days)
    limit, offset = q.clamp_limit(limit, default=100, ceiling=500), q.clamp_offset(offset)

    where = ["created_at >= UTC_TIMESTAMP() - INTERVAL :days DAY"]
    params: dict = {"days": days, "limit": limit, "offset": offset}
    if actor_email:
        where.append("actor_email = :actor_email")
        params["actor_email"] = actor_email
    if action:
        where.append("action = :action")
        params["action"] = action
    if target_type:
        where.append("target_type = :target_type")
        params["target_type"] = target_type
    if target_id:
        where.append("target_id = :target_id")
        params["target_id"] = target_id
    where_sql = " AND ".join(where)

    rows_ = q.rows(db, f"""
        SELECT id, admin_user_id, actor_email, action, target_type, target_id,
               before_json, after_json, reason, evicted, ip, created_at
        FROM admin_audit_log
        WHERE {where_sql}
        ORDER BY created_at DESC, id DESC
        LIMIT :limit OFFSET :offset
    """, params)

    total = q.scalar(db, f"SELECT COUNT(*) FROM admin_audit_log WHERE {where_sql}",
                     {k: v for k, v in params.items() if k not in ("limit", "offset")})

    return {
        "total": q.i(total),
        "limit": limit,
        "offset": offset,
        "window_days": days,
        "entries": [
            {
                "id": q.i(r["id"]),
                "admin_user_id": r["admin_user_id"],
                "actor_email": r["actor_email"],
                # A break-glass row has no admin_user_id. Flagged rather than
                # left for the UI to infer from a null, because "who did this"
                # answered as "somebody with the operator key" is a different
                # and much weaker answer than a named account.
                "is_break_glass": r["admin_user_id"] is None,
                "action": r["action"],
                "target_type": r["target_type"],
                "target_id": r["target_id"],
                "before": r["before_json"],
                "after": r["after_json"],
                "reason": r["reason"],
                # How many cached auth contexts the write actually forgot.
                # 0 on a row that should have evicted something is the
                # five-minute-stale-toggle bug, visible after the fact.
                # -1 means the eviction itself failed; None means the action had
                # no cache dimension.
                "evicted": r["evicted"],
                "ip": r["ip"],
                "created_at": q.iso(r["created_at"]),
            }
            for r in rows_
        ],
    }


@router.get("/health")
def health(
    actor: AdminActor = Depends(require_viewer),
    db: Session = Depends(get_db),
):
    """Reachability of everything the console depends on.

    Each probe is independent and none of them raise: a dead Qdrant must not
    stop the page reporting that MySQL is fine. That is the whole value of a
    health panel — it is read when something is already broken.
    """
    checks: dict = {}

    try:
        db.execute(text("SELECT 1")).scalar()
        checks["mysql"] = {"ok": True}
    except Exception as exc:
        q._safe_rollback(db)
        checks["mysql"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    try:
        from backend.app.services.cache_service import r as redis_client
        checks["redis"] = {"ok": bool(redis_client.ping())}
    except Exception as exc:
        # Redis down degrades the auth cache and the login throttle; it does not
        # stop the API. Reported as not-ok, never raised.
        checks["redis"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    try:
        # The module-level client, not a factory — qdrant_service builds one at
        # import and there is no accessor. Importing a name that does not exist
        # would be caught below and reported as "Qdrant is down", which is a
        # health panel lying about the thing it exists to report.
        from backend.app.services.qdrant_service import qdrant
        collections = qdrant.get_collections()
        checks["qdrant"] = {"ok": True, "collections": len(collections.collections)}
    except Exception as exc:
        checks["qdrant"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    schema = {
        name: q.table_exists(db, name)
        for name in ("clients", "sites", "products", "subscriptions", "licences",
                     "usage_events", "usage_counters", "admin_users",
                     "admin_sessions", "admin_audit_log", "licence_events",
                     "alert_rules", "chat_quality_daily", "schema_migrations")
    }

    migrations = q.rows(db, """
        SELECT version, filename, applied_at FROM schema_migrations ORDER BY version
    """) if schema.get("schema_migrations") else []

    return {
        "checks": checks,
        "schema": schema,
        "migrations": [
            {"version": m["version"], "filename": m["filename"],
             "applied_at": q.iso(m["applied_at"])}
            for m in migrations
        ],
        "flags": {
            # Every quota number the console shows is observational while this
            # is off. Surfaced so the UI can label limits "not enforced" rather
            # than implying a ceiling that does not exist.
            "quota_enforcement": os.getenv("AICHATBOT_QUOTA_ENFORCEMENT", "") or None,
            # While this is set, anyone holding it has owner rights and their
            # actions log as 'break-glass'. The console should say so loudly.
            "operator_key_configured": bool(os.getenv("AICHATBOT_OPERATOR_KEY", "")),
        },
        "usage_source": usage_ledger_read.provenance(db),
    }
