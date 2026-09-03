"""/api/admin/overview, /tenants, /tenants/{id}, /sites/{id} — read only.

ADMIN_CONSOLE_PLAN.md §7.1. Every endpoint is viewer-or-above and every query
is fail-soft: one broken panel degrades to empty rather than 500-ing a page of
ten.

TWO THINGS THAT MAKE THESE QUERIES LOOK ODD, BOTH DELIBERATE:

1. Requests and cost come from different row sets. usage_events writes ONE
   billable row per customer action plus several non-billable ones for
   embeddings, reranks and tool calls, so requests count `billable = 1` and cost
   sums everything. See queries.BILLABLE_REQUESTS / ALL_ROWS_COST.

2. Licences are never folded with MAX() or COUNT(DISTINCT) into a tenant row.
   Under v2 one client legitimately holds many licences — one per subscription —
   so an aggregate over them answers a question nobody asked. They are returned
   as a list, or counted per subscription.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.admin import queries as q
from backend.app.admin.deps import AdminActor, require_viewer
from backend.app.services import license_key, usage_ledger_read
from backend.app.services.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin-read"])

# ORDER BY cannot be parameterised, so sortable columns are an allowlist of
# fragments this file wrote rather than anything a caller can influence.
_TENANT_SORTS = {
    "name": "c.name ASC",
    "created": "c.created_at DESC",
    "cost": "cost DESC",
    "requests": "requests DESC",
    "sites": "site_count DESC",
}


def _environment_clause(environment: Optional[str], alias: str = "si") -> tuple[str, dict]:
    """Optional environment filter.

    Every list and every KPI takes one. Today 100% of rows are `development`,
    and the first real customer lands ALONGSIDE them rather than instead of
    them — so a console without this filter starts silently mixing demo traffic
    into headline numbers on the day it matters most, and retrofitting it means
    a period where nobody can tell which figures were already wrong.
    """
    if environment in ("development", "production"):
        return f" AND {alias}.environment = :environment ", {"environment": environment}
    return "", {}


@router.get("/overview")
def overview(
    days: int = Query(30, description="Window for the series and KPIs, 1-365."),
    environment: Optional[str] = Query(None, description="development | production"),
    actor: AdminActor = Depends(require_viewer),
    db: Session = Depends(get_db),
):
    days = q.clamp_days(days)
    env_sql, env_params = _environment_clause(environment)
    params = {"days": days, **env_params}

    # ── Estate counts. Absent stays absent: scalar_or_none so a failed probe
    # renders "—" rather than claiming there are zero tenants.
    estate = {
        "clients": q.i_or_none(q.scalar_or_none(db, "SELECT COUNT(*) FROM clients WHERE is_active = 1")),
        "sites": q.i_or_none(q.scalar_or_none(
            db, f"SELECT COUNT(*) FROM sites si WHERE si.is_active = 1 {env_sql}", env_params)),
        "subscriptions": q.i_or_none(q.scalar_or_none(db, f"""
            SELECT COUNT(*) FROM subscriptions s
            JOIN sites si ON si.id = s.site_id
            WHERE s.status IN ('active','trial') {env_sql}
        """, env_params)),
        "licences": q.i_or_none(q.scalar_or_none(db, f"""
            SELECT COUNT(*) FROM licences l
            JOIN subscriptions s ON s.id = l.subscription_id
            JOIN sites si ON si.id = s.site_id
            WHERE l.is_active = 1 {env_sql}
        """, env_params)),
    }

    # ── Spend. usage_events only, so this is the v2 era; `provenance` below
    # says so, and the v1 archive is reported separately rather than being
    # silently blended into a per-product chart it has no columns for.
    totals = q.one(db, f"""
        SELECT {q.BILLABLE_REQUESTS} AS requests,
               {q.ALL_ROWS_COST}     AS cost,
               SUM(ue.total_tokens)  AS tokens,
               COUNT(*)              AS ledger_rows
        FROM usage_events ue
        JOIN sites si ON si.id = ue.site_id
        WHERE ue.created_at >= UTC_TIMESTAMP() - INTERVAL :days DAY {env_sql}
    """, params)

    series = q.rows(db, f"""
        SELECT DATE(ue.created_at)   AS day,
               {q.BILLABLE_REQUESTS} AS requests,
               {q.ALL_ROWS_COST}     AS cost,
               SUM(ue.total_tokens)  AS tokens
        FROM usage_events ue
        JOIN sites si ON si.id = ue.site_id
        WHERE ue.created_at >= UTC_TIMESTAMP() - INTERVAL :days DAY {env_sql}
        GROUP BY DATE(ue.created_at)
        ORDER BY day
    """, params)

    by_product = q.rows(db, f"""
        SELECT ue.product_code       AS product_code,
               {q.BILLABLE_REQUESTS} AS requests,
               {q.ALL_ROWS_COST}     AS cost,
               SUM(ue.total_tokens)  AS tokens
        FROM usage_events ue
        JOIN sites si ON si.id = ue.site_id
        WHERE ue.created_at >= UTC_TIMESTAMP() - INTERVAL :days DAY {env_sql}
        GROUP BY ue.product_code
        ORDER BY cost DESC
    """, params)

    # key_owner separates spend we funded from spend the merchant funded. Every
    # row is 'czargroup' today because every site is a development site; the
    # split is here so the day that changes is visible rather than assumed.
    by_key_owner = q.rows(db, f"""
        SELECT ue.key_owner      AS key_owner,
               {q.ALL_ROWS_COST} AS cost,
               COUNT(*)          AS rows_counted
        FROM usage_events ue
        JOIN sites si ON si.id = ue.site_id
        WHERE ue.created_at >= UTC_TIMESTAMP() - INTERVAL :days DAY {env_sql}
        GROUP BY ue.key_owner
    """, params)

    by_environment = q.rows(db, """
        SELECT si.environment    AS environment,
               COUNT(DISTINCT si.id) AS sites,
               SUM(CASE WHEN ue.billable = 1 THEN 1 ELSE 0 END) AS requests,
               SUM(ue.total_cost) AS cost
        FROM sites si
        LEFT JOIN usage_events ue
               ON ue.site_id = si.id
              AND ue.created_at >= UTC_TIMESTAMP() - INTERVAL :days DAY
        WHERE si.is_active = 1
        GROUP BY si.environment
    """, {"days": days})

    top_tenants = q.rows(db, f"""
        SELECT c.id                  AS client_id,
               c.name                AS name,
               {q.BILLABLE_REQUESTS} AS requests,
               {q.ALL_ROWS_COST}     AS cost
        FROM usage_events ue
        JOIN clients c ON c.id = ue.client_id
        JOIN sites si  ON si.id = ue.site_id
        WHERE ue.created_at >= UTC_TIMESTAMP() - INTERVAL :days DAY {env_sql}
        GROUP BY c.id, c.name
        ORDER BY cost DESC
        LIMIT 10
    """, params)

    expiring = q.rows(db, f"""
        SELECT l.id AS licence_id, l.licence_key, l.expires_at,
               s.product_code, si.domain, si.environment, c.name AS client_name
        FROM licences l
        JOIN subscriptions s ON s.id = l.subscription_id
        JOIN sites si        ON si.id = s.site_id
        JOIN clients c       ON c.id = si.client_id
        WHERE l.is_active = 1
          AND l.expires_at IS NOT NULL
          AND l.expires_at <= UTC_TIMESTAMP() + INTERVAL 30 DAY
          {env_sql}
        ORDER BY l.expires_at
    """, env_params)

    # Catalogue headroom is the ONLY ceiling actually enforced today —
    # request quota is behind AICHATBOT_QUOTA_ENFORCEMENT, which is not set.
    near_ceiling = q.rows(db, f"""
        SELECT si.id AS site_id, si.domain, si.environment, si.index_plan,
               si.indexed_items, si.catalogue_limit,
               ROUND(100.0 * si.indexed_items / NULLIF(si.catalogue_limit, 0), 1) AS pct
        FROM sites si
        WHERE si.is_active = 1
          AND si.catalogue_limit > 0
          AND si.indexed_items >= si.catalogue_limit * 0.8
          {env_sql}
        ORDER BY pct DESC
    """, env_params)

    # ── Coverage: which sites can we actually SEE?
    #
    # The console's first honest question, and the one every headline figure
    # depends on. Six of eight stores present v1 JWTs, which resolve no v2
    # context, so usage_service.record() writes nothing for them — their spend
    # is not zero, it is unobserved. Without this block the overview reports
    # totals over an estate it silently only partly measures.
    #
    # Returned per site rather than as a ratio so the UI can name the dark ones.
    coverage_sites = q.rows(db, f"""
        SELECT si.id, si.domain, si.environment,
               (SELECT COUNT(*) FROM usage_events ue
                 WHERE ue.site_id = si.id
                   AND ue.created_at >= UTC_TIMESTAMP() - INTERVAL :days DAY) AS event_rows,
               (SELECT COUNT(*) FROM licences l
                  JOIN subscriptions s2 ON s2.id = l.subscription_id
                 WHERE s2.site_id = si.id AND l.is_active = 1) AS live_licences
        FROM sites si
        WHERE si.is_active = 1 {env_sql}
        ORDER BY si.domain
    """, params)

    return {
        "window_days": days,
        "environment": environment,
        "estate": estate,
        "coverage": {
            "sites": [
                {"site_id": r["id"], "domain": r["domain"],
                 "environment": r["environment"],
                 "event_rows": q.i(r["event_rows"]),
                 "live_licences": q.i(r["live_licences"]),
                 # Reporting means "produced ledger rows in this window". A site
                 # with a v2 licence and no traffic is NOT reporting either, and
                 # that is correct: the console cannot distinguish it from one
                 # that cannot report, and pretending otherwise would be the
                 # same lie in the other direction.
                 "reporting": q.i(r["event_rows"]) > 0}
                for r in coverage_sites
            ],
            "sites_total": len(coverage_sites),
            "sites_reporting": sum(1 for r in coverage_sites if q.i(r["event_rows"]) > 0),
        },
        "totals": {
            "requests": q.i_or_none(totals.get("requests")),
            "cost": q.f_or_none(totals.get("cost")),
            "tokens": q.i_or_none(totals.get("tokens")),
            "ledger_rows": q.i_or_none(totals.get("ledger_rows")),
        },
        "series": [
            {"day": q.iso(r["day"]), "requests": q.i(r["requests"]),
             "cost": q.f(r["cost"]), "tokens": q.i(r["tokens"])}
            for r in series
        ],
        "by_product": [
            {"product_code": r["product_code"], "requests": q.i(r["requests"]),
             "cost": q.f(r["cost"]), "tokens": q.i(r["tokens"])}
            for r in by_product
        ],
        "by_key_owner": [
            {"key_owner": r["key_owner"], "cost": q.f(r["cost"]),
             "rows": q.i(r["rows_counted"])}
            for r in by_key_owner
        ],
        "by_environment": [
            {"environment": r["environment"], "sites": q.i(r["sites"]),
             "requests": q.i(r["requests"]), "cost": q.f(r["cost"])}
            for r in by_environment
        ],
        "top_tenants": [
            {"client_id": r["client_id"], "name": r["name"],
             "requests": q.i(r["requests"]), "cost": q.f(r["cost"])}
            for r in top_tenants
        ],
        "licences_expiring_30d": [
            {"licence_id": r["licence_id"],
             # prefix_of, never the column. A list payload must not carry a
             # usable credential; the full key is a separate, owner-only reveal.
             "key_prefix": license_key.prefix_of(r["licence_key"]),
             "product_code": r["product_code"], "domain": r["domain"],
             "environment": r["environment"], "client_name": r["client_name"],
             "expires_at": q.iso(r["expires_at"])}
            for r in expiring
        ],
        "sites_near_catalogue_ceiling": [
            {"site_id": r["site_id"], "domain": r["domain"],
             "environment": r["environment"], "index_plan": r["index_plan"],
             "indexed_items": q.i(r["indexed_items"]),
             "catalogue_limit": q.i(r["catalogue_limit"]),
             "pct": q.f(r["pct"])}
            for r in near_ceiling
        ],
        # Which half of the ledger these figures came from, and whether the live
        # half has started. During the dual-read window a small number and an
        # absent one are the only distinction worth making about any of them.
        "usage_source": usage_ledger_read.provenance(db),
        # Quota figures are observational until this is armed. Shipping a limit
        # an operator believes is a ceiling is worse than showing no limit.
        "quota_enforced": _quota_enforced(),
    }


def _quota_enforced() -> bool:
    import os
    return os.getenv("AICHATBOT_QUOTA_ENFORCEMENT", "").lower() in ("1", "true", "on", "yes")


@router.get("/tenants")
def list_tenants(
    search: Optional[str] = Query(None, max_length=200),
    environment: Optional[str] = Query(None),
    product: Optional[str] = Query(None, max_length=64),
    status: Optional[str] = Query(None, description="active | inactive"),
    sort: Optional[str] = Query(None),
    limit: int = Query(50),
    offset: int = Query(0),
    days: int = Query(30),
    actor: AdminActor = Depends(require_viewer),
    db: Session = Depends(get_db),
):
    limit, offset, days = q.clamp_limit(limit), q.clamp_offset(offset), q.clamp_days(days)
    order_by = q.safe_sort(sort, _TENANT_SORTS, "cost")

    where = ["1 = 1"]
    params: dict = {"days": days, "limit": limit, "offset": offset}

    if search:
        where.append("(c.name LIKE :search OR c.email LIKE :search OR si.domain LIKE :search)")
        params["search"] = f"%{search}%"
    if status == "active":
        where.append("c.is_active = 1")
    elif status == "inactive":
        where.append("c.is_active = 0")
    if environment in ("development", "production"):
        where.append("si.environment = :environment")
        params["environment"] = environment
    if product:
        where.append("s.product_code = :product")
        params["product"] = product

    where_sql = " AND ".join(where)

    # LEFT JOINs throughout: a client with no site, or a site with no
    # subscription, is a real and interesting state — usually a half-finished
    # onboarding — and an INNER JOIN would hide exactly the rows an operator is
    # looking for.
    #
    # COUNT(DISTINCT ...) on the ids, because the joins multiply rows: three
    # subscriptions on two sites would otherwise report six sites.
    rows_ = q.rows(db, f"""
        SELECT c.id         AS client_id,
               c.name       AS name,
               c.email      AS email,
               c.is_active  AS is_active,
               c.created_at AS created_at,
               COUNT(DISTINCT si.id) AS site_count,
               COUNT(DISTINCT s.id)  AS subscription_count,
               GROUP_CONCAT(DISTINCT si.environment) AS environments,
               GROUP_CONCAT(DISTINCT s.product_code) AS products,
               (SELECT {q.BILLABLE_REQUESTS} FROM usage_events ue
                 WHERE ue.client_id = c.id
                   AND ue.created_at >= UTC_TIMESTAMP() - INTERVAL :days DAY) AS requests,
               (SELECT {q.ALL_ROWS_COST} FROM usage_events ue
                 WHERE ue.client_id = c.id
                   AND ue.created_at >= UTC_TIMESTAMP() - INTERVAL :days DAY) AS cost
        FROM clients c
        LEFT JOIN sites si        ON si.client_id = c.id
        LEFT JOIN subscriptions s ON s.site_id = si.id
        WHERE {where_sql}
        GROUP BY c.id, c.name, c.email, c.is_active, c.created_at
        ORDER BY {order_by}
        LIMIT :limit OFFSET :offset
    """, params)

    total = q.scalar(db, f"""
        SELECT COUNT(DISTINCT c.id)
        FROM clients c
        LEFT JOIN sites si        ON si.client_id = c.id
        LEFT JOIN subscriptions s ON s.site_id = si.id
        WHERE {where_sql}
    """, {k: v for k, v in params.items() if k not in ("limit", "offset")})

    return {
        "total": q.i(total),
        "limit": limit,
        "offset": offset,
        "window_days": days,
        "tenants": [
            {
                "client_id": r["client_id"],
                "name": r["name"],
                "email": r["email"],
                "is_active": bool(r["is_active"]),
                "created_at": q.iso(r["created_at"]),
                "site_count": q.i(r["site_count"]),
                "subscription_count": q.i(r["subscription_count"]),
                "environments": sorted((r["environments"] or "").split(",")) if r["environments"] else [],
                "products": sorted((r["products"] or "").split(",")) if r["products"] else [],
                # None, not 0. A tenant still on a v1 JWT resolves no v2 context,
                # so usage_service.record() writes nothing and there is no row to
                # count. Rendering that as 0 reads as "used the product, spent
                # nothing" when it means "not measured at all".
                "requests": q.i_or_none(r["requests"]),
                "cost": q.f_or_none(r["cost"]),
            }
            for r in rows_
        ],
        "usage_source": usage_ledger_read.provenance(db),
    }


@router.get("/tenants/{client_id}")
def tenant_detail(
    client_id: str,
    days: int = Query(30),
    actor: AdminActor = Depends(require_viewer),
    db: Session = Depends(get_db),
):
    days = q.clamp_days(days)
    client = q.one(db, """
        SELECT id, name, email, company, plan, is_active, created_at
        FROM clients WHERE id = :id
    """, {"id": client_id})
    if not client:
        raise HTTPException(status_code=404, detail="No such tenant.")

    sites = q.rows(db, """
        SELECT id, domain, platform, platform_version, store_name, collection_name,
               environment, index_plan, catalogue_limit, indexed_items, is_active,
               created_at
        FROM sites WHERE client_id = :id ORDER BY domain
    """, {"id": client_id})

    subs = q.rows(db, """
        SELECT s.id, s.site_id, s.product_code, s.status, s.plan, s.request_limit,
               s.disabled_reason, s.expires_at, si.domain, si.environment,
               (SELECT COUNT(*) FROM licences l
                 WHERE l.subscription_id = s.id AND l.is_active = 1) AS active_licences
        FROM subscriptions s
        JOIN sites si ON si.id = s.site_id
        WHERE si.client_id = :id
        ORDER BY si.domain, s.product_code
    """, {"id": client_id})

    licences = q.rows(db, """
        SELECT l.id, l.licence_key, l.is_active, l.issued_at, l.expires_at,
               l.revoked_at, s.product_code, si.domain, si.environment
        FROM licences l
        JOIN subscriptions s ON s.id = l.subscription_id
        JOIN sites si        ON si.id = s.site_id
        WHERE si.client_id = :id
        ORDER BY l.issued_at DESC
    """, {"id": client_id})

    series = q.rows(db, f"""
        SELECT DATE(created_at) AS day,
               {q.BILLABLE_REQUESTS} AS requests,
               {q.ALL_ROWS_COST}     AS cost,
               SUM(total_tokens)     AS tokens
        FROM usage_events
        WHERE client_id = :id AND created_at >= UTC_TIMESTAMP() - INTERVAL :days DAY
        GROUP BY DATE(created_at) ORDER BY day
    """, {"id": client_id, "days": days})

    by_product = q.rows(db, f"""
        SELECT product_code, {q.BILLABLE_REQUESTS} AS requests,
               {q.ALL_ROWS_COST} AS cost, SUM(total_tokens) AS tokens
        FROM usage_events
        WHERE client_id = :id AND created_at >= UTC_TIMESTAMP() - INTERVAL :days DAY
        GROUP BY product_code ORDER BY cost DESC
    """, {"id": client_id, "days": days})

    by_model = q.rows(db, f"""
        SELECT provider, model, {q.ALL_ROWS_COST} AS cost,
               SUM(total_tokens) AS tokens, COUNT(*) AS calls
        FROM usage_events
        WHERE client_id = :id AND created_at >= UTC_TIMESTAMP() - INTERVAL :days DAY
        GROUP BY provider, model ORDER BY cost DESC
    """, {"id": client_id, "days": days})

    return {
        "client": {
            "id": client["id"], "name": client["name"], "email": client["email"],
            "company": client.get("company"), "plan": client.get("plan"),
            "is_active": bool(client["is_active"]),
            "created_at": q.iso(client["created_at"]),
        },
        "sites": [_site_payload(s) for s in sites],
        "subscriptions": [
            {"id": s["id"], "site_id": s["site_id"], "domain": s["domain"],
             "environment": s["environment"], "product_code": s["product_code"],
             "status": s["status"], "plan": s["plan"],
             "request_limit": q.i(s["request_limit"]),
             "disabled_reason": s["disabled_reason"],
             "expires_at": q.iso(s["expires_at"]),
             "active_licences": q.i(s["active_licences"])}
            for s in subs
        ],
        "licences": [
            {"id": l["id"], "key_prefix": license_key.prefix_of(l["licence_key"]),
             # Says whether a full key COULD be revealed, without revealing it.
             # Pre-2026-09-03 licences stored no plaintext and never can.
             "has_plaintext": l["licence_key"] is not None,
             "is_active": bool(l["is_active"]), "product_code": l["product_code"],
             "domain": l["domain"], "environment": l["environment"],
             "issued_at": q.iso(l["issued_at"]), "expires_at": q.iso(l["expires_at"]),
             "revoked_at": q.iso(l["revoked_at"])}
            for l in licences
        ],
        "series": [{"day": q.iso(r["day"]), "requests": q.i(r["requests"]),
                    "cost": q.f(r["cost"]), "tokens": q.i(r["tokens"])} for r in series],
        "by_product": [{"product_code": r["product_code"], "requests": q.i(r["requests"]),
                        "cost": q.f(r["cost"]), "tokens": q.i(r["tokens"])} for r in by_product],
        "by_model": [{"provider": r["provider"], "model": r["model"], "cost": q.f(r["cost"]),
                      "tokens": q.i(r["tokens"]), "calls": q.i(r["calls"])} for r in by_model],
        "window_days": days,
        "usage_source": usage_ledger_read.provenance(db),
    }


def _site_payload(s: dict) -> dict:
    limit = q.i(s["catalogue_limit"])
    used = q.i(s["indexed_items"])
    return {
        "id": s["id"], "domain": s["domain"], "platform": s["platform"],
        "platform_version": s.get("platform_version"),
        "store_name": s.get("store_name"), "collection_name": s.get("collection_name"),
        "environment": s["environment"], "index_plan": s["index_plan"],
        "catalogue_limit": limit, "indexed_items": used,
        # Guarded: an unlimited plan stores 0 and would divide by zero.
        "catalogue_pct": round(100.0 * used / limit, 1) if limit else None,
        "is_active": bool(s["is_active"]), "created_at": q.iso(s.get("created_at")),
    }


@router.get("/sites/{site_id}")
def site_detail(
    site_id: str,
    days: int = Query(30),
    actor: AdminActor = Depends(require_viewer),
    db: Session = Depends(get_db),
):
    days = q.clamp_days(days)
    site = q.one(db, """
        SELECT si.*, c.name AS client_name, c.id AS client_id
        FROM sites si JOIN clients c ON c.id = si.client_id
        WHERE si.id = :id
    """, {"id": site_id})
    if not site:
        raise HTTPException(status_code=404, detail="No such site.")

    subs = q.rows(db, """
        SELECT s.id, s.product_code, s.status, s.plan, s.request_limit,
               s.disabled_reason, s.expires_at,
               (SELECT COUNT(*) FROM licences l
                 WHERE l.subscription_id = s.id AND l.is_active = 1) AS active_licences,
               (SELECT uc.billable_requests FROM usage_counters uc
                 WHERE uc.subscription_id = s.id
                   AND uc.period = DATE_FORMAT(UTC_TIMESTAMP(), '%Y-%m')) AS requests_this_period
        FROM subscriptions s
        WHERE s.site_id = :id ORDER BY s.product_code
    """, {"id": site_id})

    return {
        "site": {**_site_payload(site),
                 "client_id": site["client_id"], "client_name": site["client_name"]},
        "subscriptions": [
            {"id": s["id"], "product_code": s["product_code"], "status": s["status"],
             "plan": s["plan"], "request_limit": q.i(s["request_limit"]),
             "disabled_reason": s["disabled_reason"],
             "expires_at": q.iso(s["expires_at"]),
             "active_licences": q.i(s["active_licences"]),
             # None when no counter row exists for this period — that is "no
             # billable traffic recorded yet", not "zero requests made".
             "requests_this_period": q.i_or_none(s["requests_this_period"])}
            for s in subs
        ],
        "window_days": days,
        "quota_enforced": _quota_enforced(),
    }
