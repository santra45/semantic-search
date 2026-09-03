"""/api/admin/licences, /products, /plans — read only.

ADMIN_CONSOLE_PLAN.md §7.1.

────────────────────────────────────────────────────────────────────────────
THE KEY REVEAL IS A SEPARATE, OWNER-ONLY ENDPOINT.

licences.licence_key holds the plaintext since 2026-09-03, which is what makes
"resend my key" answerable without rotating a working install. It also makes
this table a secrets table, so:

  * every LIST and every DETAIL returns the prefix only
  * the full key comes from GET /licences/{id}/key, owner-only, logged
  * `has_plaintext` says whether a reveal COULD work, without revealing anything

Putting the key in the list payload would spray every credential in the estate
into browser memory, proxy logs and anyone's devtools on the first page load —
which is the same mistake as putting it in a log line, at greater scale.
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.admin import queries as q
from backend.app.admin.deps import AdminActor, require_owner, require_viewer
from backend.app.services import catalog, license_key
from backend.app.services.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin-read"])

_LICENCE_SORTS = {
    "issued": "l.issued_at DESC",
    "expires": "l.expires_at IS NULL, l.expires_at ASC",
    "domain": "si.domain ASC",
    "product": "s.product_code ASC",
}


@router.get("/licences")
def list_licences(
    status: Optional[str] = Query(None, description="active | revoked | expired"),
    product: Optional[str] = Query(None, max_length=64),
    environment: Optional[str] = Query(None),
    expiring_days: Optional[int] = Query(None, description="Only those expiring within N days."),
    search: Optional[str] = Query(None, max_length=200),
    sort: Optional[str] = Query(None),
    limit: int = Query(50),
    offset: int = Query(0),
    actor: AdminActor = Depends(require_viewer),
    db: Session = Depends(get_db),
):
    limit, offset = q.clamp_limit(limit), q.clamp_offset(offset)
    where = ["1 = 1"]
    params: dict = {"limit": limit, "offset": offset}

    if status == "active":
        # "Active" means the flag AND not past its expiry. A licence whose
        # expires_at has passed still carries is_active = 1 — nothing sweeps
        # them — and resolve_key() denies it at request time. Listing it as
        # active would disagree with what the API actually does.
        where.append("l.is_active = 1 AND (l.expires_at IS NULL OR l.expires_at > UTC_TIMESTAMP())")
    elif status == "revoked":
        where.append("l.is_active = 0")
    elif status == "expired":
        where.append("l.expires_at IS NOT NULL AND l.expires_at <= UTC_TIMESTAMP()")

    if product:
        where.append("s.product_code = :product")
        params["product"] = product
    if environment in ("development", "production"):
        where.append("si.environment = :environment")
        params["environment"] = environment
    if expiring_days is not None:
        where.append("l.expires_at IS NOT NULL AND l.expires_at <= UTC_TIMESTAMP() + INTERVAL :exp DAY")
        params["exp"] = q.clamp_days(expiring_days)
    if search:
        # Never search licence_key. Matching on a credential means the search
        # term IS a credential, and it would land in access logs and browser
        # history as a query string.
        where.append("(si.domain LIKE :search OR c.name LIKE :search)")
        params["search"] = f"%{search}%"

    where_sql = " AND ".join(where)
    order_by = q.safe_sort(sort, _LICENCE_SORTS, "issued")

    rows_ = q.rows(db, f"""
        SELECT l.id, l.licence_key, l.is_active, l.issued_at, l.expires_at, l.revoked_at,
               s.id AS subscription_id, s.product_code, s.status AS subscription_status,
               si.id AS site_id, si.domain, si.environment,
               c.id AS client_id, c.name AS client_name,
               (SELECT MAX(ue.created_at) FROM usage_events ue
                 WHERE ue.subscription_id = s.id) AS last_seen
        FROM licences l
        JOIN subscriptions s ON s.id = l.subscription_id
        JOIN sites si        ON si.id = s.site_id
        JOIN clients c       ON c.id = si.client_id
        WHERE {where_sql}
        ORDER BY {order_by}
        LIMIT :limit OFFSET :offset
    """, params)

    total = q.scalar(db, f"""
        SELECT COUNT(*)
        FROM licences l
        JOIN subscriptions s ON s.id = l.subscription_id
        JOIN sites si        ON si.id = s.site_id
        JOIN clients c       ON c.id = si.client_id
        WHERE {where_sql}
    """, {k: v for k, v in params.items() if k not in ("limit", "offset")})

    return {
        "total": q.i(total),
        "limit": limit,
        "offset": offset,
        "licences": [
            {
                "id": r["id"],
                "key_prefix": license_key.prefix_of(r["licence_key"]),
                "has_plaintext": r["licence_key"] is not None,
                "is_active": bool(r["is_active"]),
                "subscription_id": r["subscription_id"],
                "subscription_status": r["subscription_status"],
                "product_code": r["product_code"],
                "site_id": r["site_id"],
                "domain": r["domain"],
                "environment": r["environment"],
                "client_id": r["client_id"],
                "client_name": r["client_name"],
                "issued_at": q.iso(r["issued_at"]),
                "expires_at": q.iso(r["expires_at"]),
                "revoked_at": q.iso(r["revoked_at"]),
                # Attributed via the subscription, not the licence: usage_events
                # has no licence_id column, so a rotation does not reset this.
                "last_seen": q.iso(r["last_seen"]),
            }
            for r in rows_
        ],
    }


@router.get("/licences/{licence_id}")
def licence_detail(
    licence_id: str,
    actor: AdminActor = Depends(require_viewer),
    db: Session = Depends(get_db),
):
    row = q.one(db, """
        SELECT l.id, l.licence_key, l.is_active, l.issued_at, l.expires_at, l.revoked_at,
               s.id AS subscription_id, s.product_code, s.status AS subscription_status,
               s.plan, s.request_limit, s.disabled_reason,
               si.id AS site_id, si.domain, si.environment, si.platform,
               c.id AS client_id, c.name AS client_name
        FROM licences l
        JOIN subscriptions s ON s.id = l.subscription_id
        JOIN sites si        ON si.id = s.site_id
        JOIN clients c       ON c.id = si.client_id
        WHERE l.id = :id
    """, {"id": licence_id})
    if not row:
        raise HTTPException(status_code=404, detail="No such licence.")

    # THE WHOLE CHAIN, BY SUBSCRIPTION — not just this licence.
    #
    # issue_licence() DELETEs the licence it rotates out, so a superseded key's
    # own events are keyed to a licence_id with no row behind it. Querying by
    # licence_id alone would therefore show a "rotated" event with no trace of
    # what it replaced, and the predecessor's history would exist in the table
    # while being unreachable from every page in the console.
    #
    # subscription_id survives all of that, which is why the column is there.
    # `is_this_licence` marks which rows belong to the key being viewed so the
    # UI can show the chain without pretending it is all one key.
    events = [
        {**e, "is_this_licence": e["licence_id"] == licence_id}
        for e in q.rows(db, """
            SELECT id, licence_id, event, detail, key_prefix, actor_email, created_at
            FROM licence_events
            WHERE subscription_id = :sid
            ORDER BY created_at DESC, id DESC
        """, {"sid": row["subscription_id"]})
    ]

    usage = q.one(db, f"""
        SELECT {q.BILLABLE_REQUESTS} AS requests,
               {q.ALL_ROWS_COST}     AS cost,
               SUM(total_tokens)     AS tokens,
               MIN(created_at)       AS first_seen,
               MAX(created_at)       AS last_seen
        FROM usage_events WHERE subscription_id = :sid
    """, {"sid": row["subscription_id"]})

    return {
        "licence": {
            "id": row["id"],
            "key_prefix": license_key.prefix_of(row["licence_key"]),
            "has_plaintext": row["licence_key"] is not None,
            "is_active": bool(row["is_active"]),
            "issued_at": q.iso(row["issued_at"]),
            "expires_at": q.iso(row["expires_at"]),
            "revoked_at": q.iso(row["revoked_at"]),
        },
        "subscription": {
            "id": row["subscription_id"], "product_code": row["product_code"],
            "status": row["subscription_status"], "plan": row["plan"],
            "request_limit": q.i(row["request_limit"]),
            "disabled_reason": row["disabled_reason"],
        },
        "site": {"id": row["site_id"], "domain": row["domain"],
                 "environment": row["environment"], "platform": row["platform"]},
        "client": {"id": row["client_id"], "name": row["client_name"]},
        "events": [
            {"id": e["id"], "event": e["event"], "detail": e["detail"],
             "key_prefix": e["key_prefix"], "actor_email": e["actor_email"],
             "created_at": q.iso(e["created_at"]),
             # False means this event describes a key that this subscription
             # used to hold — usually one that no longer exists as a row.
             "is_this_licence": e["is_this_licence"]}
            for e in events
        ],
        # Attributed to the SUBSCRIPTION, and the label says so. usage_events
        # carries no licence_id, so this figure spans every key the subscription
        # has ever held — presenting it as this licence's usage would be wrong
        # the moment anything was rotated.
        "usage_for_subscription": {
            "requests": q.i_or_none(usage.get("requests")),
            "cost": q.f_or_none(usage.get("cost")),
            "tokens": q.i_or_none(usage.get("tokens")),
            "first_seen": q.iso(usage.get("first_seen")),
            "last_seen": q.iso(usage.get("last_seen")),
        },
    }


@router.get("/licences/{licence_id}/key")
def reveal_licence_key(
    licence_id: str,
    actor: AdminActor = Depends(require_owner),
    db: Session = Depends(get_db),
):
    """The full plaintext key. OWNER ONLY, and every call is logged.

    Separate from the detail endpoint on purpose: a reveal should be a
    deliberate act with its own permission and its own log line, not a field
    that rides along on every page load of a screen a viewer can open.

    No audit_log row, because mutate() is for mutations and this changes
    nothing. The log line is the record. If reveals ever need to be queryable
    rather than greppable, that is a licence_events row with event='revealed',
    not a widening of admin_audit_log to cover reads.
    """
    row = q.one(db, """
        SELECT l.licence_key, l.is_active, s.product_code, si.domain
        FROM licences l
        JOIN subscriptions s ON s.id = l.subscription_id
        JOIN sites si        ON si.id = s.site_id
        WHERE l.id = :id
    """, {"id": licence_id})
    if not row:
        raise HTTPException(status_code=404, detail="No such licence.")

    if row["licence_key"] is None:
        # Not an error in the caller — a fact about the row. Licences minted
        # before 2026-09-03 stored only a hash, and SHA-256 is one-way.
        raise HTTPException(
            status_code=409,
            detail="This licence was issued before plaintext keys were stored, "
                   "so it cannot be shown. Rotating the key is the only way to "
                   "produce one that can.",
        )

    logger.warning(
        "admin: %s REVEALED the licence key for %s on %s (licence %s)",
        actor.email, row["product_code"], row["domain"], licence_id,
    )
    return {
        "licence_id": licence_id,
        "key": row["licence_key"],
        "is_active": bool(row["is_active"]),
        "product_code": row["product_code"],
        "domain": row["domain"],
    }


@router.get("/products")
def list_products(
    days: int = Query(30),
    environment: Optional[str] = Query(None),
    actor: AdminActor = Depends(require_viewer),
    db: Session = Depends(get_db),
):
    days = q.clamp_days(days)
    env_sql, env_params = ("", {})
    if environment in ("development", "production"):
        env_sql = " AND si.environment = :environment "
        env_params = {"environment": environment}

    rows_ = q.rows(db, f"""
        SELECT p.code, p.name, p.platform, p.key_segment, p.is_sellable,
               COUNT(DISTINCT s.id)  AS subscriptions,
               COUNT(DISTINCT CASE WHEN s.status IN ('active','trial') THEN s.id END) AS active_subscriptions,
               COUNT(DISTINCT si.id) AS sites,
               COUNT(DISTINCT si.client_id) AS clients
        FROM products p
        LEFT JOIN subscriptions s ON s.product_code = p.code
        -- The environment filter goes in the ON clause, not a WHERE. In a WHERE
        -- it would drop products that have no site in that environment; here
        -- they survive with zero counts, which is the answer an operator asking
        -- "what is adoption like in production" actually needs.
        LEFT JOIN sites si        ON si.id = s.site_id {env_sql}
        GROUP BY p.code, p.name, p.platform, p.key_segment, p.is_sellable
        ORDER BY p.platform, p.code
    """, env_params)

    spend = {
        r["product_code"]: r
        for r in q.rows(db, f"""
            SELECT ue.product_code,
                   {q.BILLABLE_REQUESTS} AS requests,
                   {q.ALL_ROWS_COST}     AS cost,
                   SUM(ue.total_tokens)  AS tokens
            FROM usage_events ue
            JOIN sites si ON si.id = ue.site_id
            WHERE ue.created_at >= UTC_TIMESTAMP() - INTERVAL :days DAY {env_sql}
            GROUP BY ue.product_code
        """, {"days": days, **env_params})
    }

    return {
        "window_days": days,
        "environment": environment,
        "products": [
            {
                "code": r["code"], "name": r["name"], "platform": r["platform"],
                "key_segment": r["key_segment"],
                # "Can it be BOUGHT", not "can it be used". A withdrawn product
                # stops being offered by onboarding; existing subscriptions keep
                # resolving. The chokepoint deliberately does not read this.
                "is_sellable": bool(r["is_sellable"]),
                "subscriptions": q.i(r["subscriptions"]),
                "active_subscriptions": q.i(r["active_subscriptions"]),
                "sites": q.i(r["sites"]),
                "clients": q.i(r["clients"]),
                "requests": q.i_or_none((spend.get(r["code"]) or {}).get("requests")),
                "cost": q.f_or_none((spend.get(r["code"]) or {}).get("cost")),
                "tokens": q.i_or_none((spend.get(r["code"]) or {}).get("tokens")),
            }
            for r in rows_
        ],
    }


@router.get("/plans")
def list_plans(
    actor: AdminActor = Depends(require_viewer),
    db: Session = Depends(get_db),
):
    """BOTH ladders, side by side, with what sits on each.

    They are independent and cannot be derived from each other:

      INDEX_PLANS  catalogue size, bought ONCE PER SITE, because a store's
                   modules share one Qdrant collection. Lands on sites.
      MODULE_PLANS request quota, bought PER MODULE. Lands on subscriptions.

    Returned as two lists rather than merged, because a single "plan" control
    per tenant would silently edit the wrong one — and dropping a site's index
    plan below what is already indexed has no clean resolution.
    """
    site_counts = {
        r["index_plan"]: q.i(r["n"])
        for r in q.rows(db, "SELECT index_plan, COUNT(*) AS n FROM sites GROUP BY index_plan")
    }
    sub_counts = {
        r["plan"]: q.i(r["n"])
        for r in q.rows(db, "SELECT plan, COUNT(*) AS n FROM subscriptions GROUP BY plan")
    }

    def _ladder(source: dict, counts: dict, count_key: str) -> list[dict]:
        out = []
        for code, spec in source.items():
            entry = {"code": code, count_key: counts.get(code, 0)}
            # Copied rather than cherry-picked: catalog.py is the authority for
            # what a plan IS, and naming its fields here would mean this endpoint
            # silently drops any field added there.
            if isinstance(spec, dict):
                entry.update({k: v for k, v in spec.items() if not k.startswith("_")})
            out.append(entry)
        return out

    return {
        "index_plans": _ladder(catalog.INDEX_PLANS, site_counts, "sites"),
        "module_plans": _ladder(catalog.MODULE_PLANS, sub_counts, "subscriptions"),
        # MODULE_PLANS carries display prices as strings ("$29"), not amounts,
        # so nothing here can compute revenue. Said explicitly rather than
        # letting a UI infer it from a missing field.
        "revenue_computable": False,
    }
