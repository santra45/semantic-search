"""/api/admin/* — the write endpoints. The control plane.

ADMIN_CONSOLE_PLAN.md §7.3. Every one of these goes through mutate(), and none
of them calls a service mutator directly. That is not a style rule: mutate() is
what ties the audit row to the change and what evicts auth_cache, and an
endpoint that bypasses it is a toggle that silently does nothing for 300s.

────────────────────────────────────────────────────────────────────────────
THE THREE SCOPES, AND WHY THE UI MUST NOT COLLAPSE THEM.

v2 has three levels that all read as "turn this off" and mean different things:

  clients.is_active       the whole CUSTOMER, every store, every module
  sites.is_active         one STORE INSTALL, every module on it
  subscriptions.status    one MODULE on one store

An operator asked to "suspend Acme" needs to be shown which of the three they
are pulling. That is why these are separate endpoints with separate blast-radius
previews rather than one /disable that guesses.
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import contextlib
import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.admin import queries as q
from backend.app.admin.audit import mutate
from backend.app.admin.deps import AdminActor, require_operator, require_owner
from backend.app.services import catalog, license_key, licensing_service, tenancy_service
from backend.app.services.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin-write"])


# ── Request bodies ───────────────────────────────────────────────────────────

class ReasonBody(BaseModel):
    reason: str = Field(min_length=3, max_length=500)


class OptionalReasonBody(BaseModel):
    reason: Optional[str] = Field(default=None, max_length=500)


class IndexPlanBody(BaseModel):
    index_plan: str = Field(max_length=32)
    reason: Optional[str] = Field(default=None, max_length=500)


class EnvironmentBody(BaseModel):
    environment: str = Field(max_length=32)
    # Type-to-confirm, enforced server-side rather than trusted to the UI. A
    # confirmation only the browser checks is a confirmation an API client skips.
    confirm_domain: str = Field(max_length=255)
    reason: Optional[str] = Field(default=None, max_length=500)


class SubscriptionCreateBody(BaseModel):
    site_id: str = Field(max_length=36)
    product_code: str = Field(max_length=64)
    plan: Optional[str] = Field(default=None, max_length=32)
    status: Optional[str] = Field(default=None, max_length=16)


class PlanBody(BaseModel):
    plan: str = Field(max_length=32)
    reason: Optional[str] = Field(default=None, max_length=500)


class TermBody(BaseModel):
    # None means open-ended, which is a real and different thing from "expired".
    expires_at: Optional[datetime] = None
    extend_days: Optional[int] = Field(default=None, ge=1, le=3650)
    reason: Optional[str] = Field(default=None, max_length=500)


class IssueLicenceBody(BaseModel):
    valid_days: Optional[int] = Field(default=365, ge=1, le=3650)
    reason: Optional[str] = Field(default=None, max_length=500)


class RevokeBody(BaseModel):
    reason: str = Field(min_length=3, max_length=500)
    confirm_prefix: str = Field(max_length=64)


class ProductWithdrawBody(BaseModel):
    reason: str = Field(min_length=3, max_length=500)
    # The count of affected subscriptions, echoed back. A global product switch
    # can dark-out every merchant on that line at once, so the operator has to
    # have read the blast radius before the request is even well-formed.
    confirm_affected: int = Field(ge=0)


# ── Snapshots ────────────────────────────────────────────────────────────────
#
# before/after pairs for the audit row. Deliberately whole rows rather than the
# one column being changed: the console's 24h revert replays before_json, so it
# has to be complete enough to restore from, and a reader years later should not
# need this file to know what the row looked like.

def _snap_client(db: Session, client_id: str) -> dict:
    return q.one(db, "SELECT id, name, email, plan, is_active FROM clients WHERE id = :id",
                 {"id": client_id})


def _snap_site(db: Session, site_id: str) -> dict:
    return q.one(db, """
        SELECT id, client_id, domain, environment, index_plan, catalogue_limit,
               indexed_items, is_active
        FROM sites WHERE id = :id
    """, {"id": site_id})


def _snap_subscription(db: Session, subscription_id: str) -> dict:
    return q.one(db, """
        SELECT id, site_id, product_code, status, plan, request_limit,
               disabled_reason, expires_at
        FROM subscriptions WHERE id = :id
    """, {"id": subscription_id})


def _snap_licence(db: Session, licence_id: str) -> dict:
    # licence_key is NEVER snapshotted. admin_audit_log is readable by any
    # viewer, so putting the plaintext in before_json would hand every viewer
    # every credential through the audit screen — undoing the whole reason the
    # reveal is a separate owner-only endpoint.
    return q.one(db, """
        SELECT id, subscription_id, is_active, issued_at, expires_at, revoked_at
        FROM licences WHERE id = :id
    """, {"id": licence_id})


def _snap_product(db: Session, code: str) -> dict:
    return q.one(db, "SELECT code, name, platform, is_sellable FROM products WHERE code = :code",
                 {"code": code})


@contextlib.contextmanager
def service_errors():
    """Turn a service's refusal into the status code it deserves.

    tenancy_service and licensing_service raise ValueError for "you asked for
    something incoherent" and LookupError for "no such id". Left alone both
    become a 500, which tells an operator the console is broken when in fact
    they were told something useful and specific — set_subscription_status'
    "Activating a trial needs the plan they bought. Call set_subscription_plan
    instead" is better guidance than anything this layer would write, and it
    should reach the screen rather than the log.

    Wraps mutate() from OUTSIDE, so mutate has already rolled back and logged by
    the time this converts the exception.
    """
    try:
        yield
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _resumed_status(plan: Optional[str]) -> str:
    """The status a paused subscription should return to.

    NOT a hardcoded 'active'. status and plan are coupled: the service refuses
    to make a trial-plan subscription active, because doing so would bill a
    paying customer against trial limits and there is no defensible way to guess
    which plan they actually bought. So a subscription still on the trial plan
    resumes to 'trial'; one on a paid plan resumes to 'active'.

    Found by a 500 in testing, not by reading — the first version sent every
    resume to 'active' and worked only for subscriptions that had been upgraded.
    """
    return "trial" if (plan or catalog.TRIAL_MODULE_PLAN) == catalog.TRIAL_MODULE_PLAN else "active"


def _licence_event(db: Session, *, licence_id: str, subscription_id: str,
                   event: str, key_prefix: Optional[str], actor: AdminActor,
                   detail: Optional[str] = None) -> None:
    """Append to licence_events, inside the caller's transaction.

    Separate from admin_audit_log because the two answer different questions:
    that one is "who did it", this one is "what happened to this key", including
    events with no human actor. It also OUTLIVES the licence — issue_licence()
    deletes the row it rotates out, so for a rotated key this is the only record
    that it existed.
    """
    db.execute(
        text("""
            INSERT INTO licence_events
                (licence_id, subscription_id, event, detail, key_prefix, actor_email)
            VALUES (:lid, :sid, :event, :detail, :prefix, :email)
        """),
        {"lid": licence_id, "sid": subscription_id, "event": event,
         "detail": detail, "prefix": key_prefix, "email": actor.email},
    )


# ── Blast radius ─────────────────────────────────────────────────────────────

@router.get("/clients/{client_id}/blast-radius")
def client_blast_radius(
    client_id: str,
    actor: AdminActor = Depends(require_operator),
    db: Session = Depends(get_db),
):
    """What disabling this customer would take offline.

    Shown in the confirm dialog before the button arms. The point is that
    "disable Acme" is an abstract phrase and "3 sites, 5 subscriptions, 5 live
    keys, ~1,240 requests/day" is not.
    """
    row = q.one(db, """
        SELECT c.id, c.name, c.is_active,
               COUNT(DISTINCT si.id) AS sites,
               COUNT(DISTINCT s.id)  AS subscriptions,
               COUNT(DISTINCT CASE WHEN l.is_active = 1 THEN l.id END) AS live_licences
        FROM clients c
        LEFT JOIN sites si        ON si.client_id = c.id
        LEFT JOIN subscriptions s ON s.site_id = si.id
        LEFT JOIN licences l      ON l.subscription_id = s.id
        WHERE c.id = :id GROUP BY c.id, c.name, c.is_active
    """, {"id": client_id})
    if not row:
        raise HTTPException(status_code=404, detail="No such tenant.")

    daily = q.scalar_or_none(db, """
        SELECT ROUND(COUNT(*) / 7.0, 1) FROM usage_events
        WHERE client_id = :id AND billable = 1
          AND created_at >= UTC_TIMESTAMP() - INTERVAL 7 DAY
    """, {"id": client_id})

    return {
        "client_id": row["id"], "name": row["name"], "is_active": bool(row["is_active"]),
        "sites": q.i(row["sites"]), "subscriptions": q.i(row["subscriptions"]),
        "live_licences": q.i(row["live_licences"]),
        "requests_per_day": q.f_or_none(daily),
    }


@router.get("/products/{code}/blast-radius")
def product_blast_radius(
    code: str,
    actor: AdminActor = Depends(require_owner),
    db: Session = Depends(get_db),
):
    row = q.one(db, """
        SELECT p.code, p.name, p.is_sellable,
               COUNT(DISTINCT s.id)  AS subscriptions,
               COUNT(DISTINCT si.client_id) AS clients
        FROM products p
        LEFT JOIN subscriptions s ON s.product_code = p.code
        LEFT JOIN sites si        ON si.id = s.site_id
        WHERE p.code = :code GROUP BY p.code, p.name, p.is_sellable
    """, {"code": code})
    if not row:
        raise HTTPException(status_code=404, detail="No such product.")
    return {
        "code": row["code"], "name": row["name"],
        "is_sellable": bool(row["is_sellable"]),
        "subscriptions": q.i(row["subscriptions"]),
        "clients": q.i(row["clients"]),
        # Says plainly that this is NOT a runtime kill switch, so nobody reaches
        # for it during an incident expecting traffic to stop.
        "note": "Withdrawing stops the product being offered by onboarding. "
                "Existing subscriptions keep resolving and keep working.",
    }


# ── Clients ──────────────────────────────────────────────────────────────────

@router.post("/clients/{client_id}/disable")
def disable_client(
    client_id: str,
    body: ReasonBody,
    actor: AdminActor = Depends(require_operator),
    db: Session = Depends(get_db),
):
    if not _snap_client(db, client_id):
        raise HTTPException(status_code=404, detail="No such tenant.")
    with service_errors(), mutate(db, actor, action="client.disable", target=("client", client_id),
                reason=body.reason, before=_snap_client(db, client_id)) as m:
        m.result = tenancy_service.set_client_active(db, client_id, False)
        m.after = _snap_client(db, client_id)
    return {"success": True, "client_id": client_id, "is_active": False}


@router.post("/clients/{client_id}/enable")
def enable_client(
    client_id: str,
    body: OptionalReasonBody = OptionalReasonBody(),
    actor: AdminActor = Depends(require_operator),
    db: Session = Depends(get_db),
):
    if not _snap_client(db, client_id):
        raise HTTPException(status_code=404, detail="No such tenant.")
    with service_errors(), mutate(db, actor, action="client.enable", target=("client", client_id),
                reason=body.reason, before=_snap_client(db, client_id)) as m:
        m.result = tenancy_service.set_client_active(db, client_id, True)
        m.after = _snap_client(db, client_id)
    return {"success": True, "client_id": client_id, "is_active": True}


# ── Sites ────────────────────────────────────────────────────────────────────

@router.post("/sites/{site_id}/disable")
def disable_site(
    site_id: str,
    body: ReasonBody,
    actor: AdminActor = Depends(require_operator),
    db: Session = Depends(get_db),
):
    if not _snap_site(db, site_id):
        raise HTTPException(status_code=404, detail="No such site.")
    with service_errors(), mutate(db, actor, action="site.disable", target=("site", site_id),
                reason=body.reason, before=_snap_site(db, site_id)) as m:
        m.result = tenancy_service.set_site_active(db, site_id, False)
        m.after = _snap_site(db, site_id)
    return {"success": True, "site_id": site_id, "is_active": False}


@router.post("/sites/{site_id}/enable")
def enable_site(
    site_id: str,
    body: OptionalReasonBody = OptionalReasonBody(),
    actor: AdminActor = Depends(require_operator),
    db: Session = Depends(get_db),
):
    if not _snap_site(db, site_id):
        raise HTTPException(status_code=404, detail="No such site.")
    with service_errors(), mutate(db, actor, action="site.enable", target=("site", site_id),
                reason=body.reason, before=_snap_site(db, site_id)) as m:
        m.result = tenancy_service.set_site_active(db, site_id, True)
        m.after = _snap_site(db, site_id)
    return {"success": True, "site_id": site_id, "is_active": True}


@router.patch("/sites/{site_id}/index-plan")
def set_site_index_plan(
    site_id: str,
    body: IndexPlanBody,
    actor: AdminActor = Depends(require_operator),
    db: Session = Depends(get_db),
):
    """Change a site's CATALOGUE-SIZE plan.

    One of two independent ladders. This one is bought once per site, because a
    store's modules share one Qdrant collection; the request-quota ladder lives
    on subscriptions and is a different endpoint. A single "plan" control that
    edited both would silently change the wrong one.
    """
    site = _snap_site(db, site_id)
    if not site:
        raise HTTPException(status_code=404, detail="No such site.")
    if not catalog.is_valid_index_plan(body.index_plan):
        raise HTTPException(
            status_code=422,
            detail=f"Unknown index plan '{body.index_plan}'. "
                   f"Expected one of: {', '.join(catalog.INDEX_PLANS)}.",
        )

    # Refuse a downgrade below what is already indexed. There is no clean
    # resolution to a site sitting over its own ceiling: the catalogue is
    # already in Qdrant, nothing deletes it, and every subsequent sync fails the
    # headroom check with no way for the merchant to act on it.
    new_limit = catalog.catalogue_limit_for(body.index_plan)
    indexed = q.i(site.get("indexed_items"))
    if new_limit and indexed > new_limit:
        raise HTTPException(
            status_code=409,
            detail=f"This site has {indexed} items indexed and '{body.index_plan}' "
                   f"allows {new_limit}. Downgrading would leave it permanently "
                   f"over its ceiling. Reduce the catalogue first.",
        )

    with service_errors(), mutate(db, actor, action="site.index_plan", target=("site", site_id),
                reason=body.reason, before=site) as m:
        m.result = tenancy_service.set_index_plan(db, site_id, body.index_plan)
        m.after = _snap_site(db, site_id)
    return {"success": True, "site_id": site_id, "index_plan": body.index_plan,
            "catalogue_limit": new_limit}


@router.post("/sites/{site_id}/environment")
def promote_site(
    site_id: str,
    body: EnvironmentBody,
    actor: AdminActor = Depends(require_owner),
    db: Session = Depends(get_db),
):
    """Move a site between development and production. OWNER ONLY.

    This is not a label. sites.environment decides usage_events.key_owner for
    every future row — whether spend is booked as ours or the merchant's — and
    it decides whether newly minted keys are czg_live_ or czg_test_. Promoting
    leaves the site's EXISTING keys inconsistent with it, so the response says
    to reissue.
    """
    site = _snap_site(db, site_id)
    if not site:
        raise HTTPException(status_code=404, detail="No such site.")
    if body.environment not in ("development", "production"):
        raise HTTPException(status_code=422, detail="environment must be 'development' or 'production'.")
    if body.confirm_domain.strip().lower() != (site["domain"] or "").strip().lower():
        raise HTTPException(
            status_code=422,
            detail="confirm_domain does not match this site's domain.",
        )

    live_keys = q.i(q.scalar(db, """
        SELECT COUNT(*) FROM licences l
        JOIN subscriptions s ON s.id = l.subscription_id
        WHERE s.site_id = :id AND l.is_active = 1
    """, {"id": site_id}))

    with service_errors(), mutate(db, actor, action="site.environment", target=("site", site_id),
                reason=body.reason, before=site) as m:
        m.result = tenancy_service.set_site_environment(db, site_id, body.environment)
        m.after = _snap_site(db, site_id)

    return {
        "success": True, "site_id": site_id, "environment": body.environment,
        "live_licences_needing_reissue": live_keys,
        "note": "Existing keys still carry the previous environment in their "
                "prefix. Reissue them so the key and the site agree.",
    }


# ── Subscriptions ────────────────────────────────────────────────────────────

@router.post("/subscriptions")
def create_subscription(
    body: SubscriptionCreateBody,
    actor: AdminActor = Depends(require_operator),
    db: Session = Depends(get_db),
):
    """Sell a module to a site. Creates no licence — that is a separate,
    owner-only step, so adding a subscription cannot accidentally mint a
    credential."""
    site = _snap_site(db, body.site_id)
    if not site:
        raise HTTPException(status_code=404, detail="No such site.")
    if not catalog.is_valid_product(body.product_code):
        raise HTTPException(status_code=422, detail=f"Unknown product '{body.product_code}'.")
    if body.plan and not catalog.is_valid_module_plan(body.plan):
        raise HTTPException(status_code=422, detail=f"Unknown module plan '{body.plan}'.")

    if licensing_service.get_subscription(db, body.site_id, body.product_code):
        raise HTTPException(
            status_code=409,
            detail="That site already has a subscription for this product.",
        )

    kwargs = {"site_id": body.site_id, "product_code": body.product_code}
    if body.plan:
        kwargs["plan"] = body.plan
    if body.status:
        kwargs["status"] = body.status

    with service_errors(), mutate(db, actor, action="subscription.create",
                target=("subscription", f"{body.site_id}:{body.product_code}"),
                before=None) as m:
        result = licensing_service.create_subscription(db, **kwargs)
        m.result = result
        m.after = _snap_subscription(db, result["id"])
        # A brand-new subscription has no licence and therefore no cached
        # context to forget. Explicit, so it is clear this was considered.
        if not result.get("key_hashes"):
            m.evict = []

    return {"success": True, "subscription": m.after}


@router.post("/subscriptions/{subscription_id}/pause")
def pause_subscription(
    subscription_id: str,
    body: ReasonBody,
    actor: AdminActor = Depends(require_operator),
    db: Session = Depends(get_db),
):
    """Turn ONE module off for ONE store. The narrowest of the three scopes.

    Also where disabled_reason is set — the merchant-facing sentence, distinct
    from the audit reason written for colleagues.
    """
    before = _snap_subscription(db, subscription_id)
    if not before:
        raise HTTPException(status_code=404, detail="No such subscription.")

    with service_errors(), mutate(db, actor, action="subscription.pause",
                target=("subscription", subscription_id),
                reason=body.reason, before=before) as m:
        m.result = licensing_service.set_subscription_status(db, subscription_id, "suspended")
        db.execute(
            text("UPDATE subscriptions SET disabled_reason = :r WHERE id = :id"),
            {"r": body.reason, "id": subscription_id},
        )
        m.after = _snap_subscription(db, subscription_id)
    return {"success": True, "subscription_id": subscription_id, "status": "suspended"}


@router.post("/subscriptions/{subscription_id}/resume")
def resume_subscription(
    subscription_id: str,
    body: OptionalReasonBody = OptionalReasonBody(),
    actor: AdminActor = Depends(require_operator),
    db: Session = Depends(get_db),
):
    before = _snap_subscription(db, subscription_id)
    if not before:
        raise HTTPException(status_code=404, detail="No such subscription.")
    target_status = _resumed_status(before.get("plan"))
    with service_errors(), mutate(db, actor, action="subscription.resume",
                target=("subscription", subscription_id),
                reason=body.reason, before=before) as m:
        m.result = licensing_service.set_subscription_status(db, subscription_id, target_status)
        # Cleared on resume: a stale reason would otherwise keep showing in the
        # merchant's storefront after the module came back.
        db.execute(
            text("UPDATE subscriptions SET disabled_reason = NULL WHERE id = :id"),
            {"id": subscription_id},
        )
        m.after = _snap_subscription(db, subscription_id)
    return {"success": True, "subscription_id": subscription_id, "status": target_status}


@router.patch("/subscriptions/{subscription_id}/plan")
def set_subscription_plan(
    subscription_id: str,
    body: PlanBody,
    actor: AdminActor = Depends(require_operator),
    db: Session = Depends(get_db),
):
    """Change a subscription's REQUEST-QUOTA plan — the per-module ladder."""
    before = _snap_subscription(db, subscription_id)
    if not before:
        raise HTTPException(status_code=404, detail="No such subscription.")
    if not catalog.is_valid_module_plan(body.plan):
        raise HTTPException(
            status_code=422,
            detail=f"Unknown module plan '{body.plan}'. "
                   f"Expected one of: {', '.join(catalog.MODULE_PLANS)}.",
        )
    with service_errors(), mutate(db, actor, action="subscription.plan",
                target=("subscription", subscription_id),
                reason=body.reason, before=before) as m:
        m.result = licensing_service.set_subscription_plan(db, subscription_id, body.plan)
        m.after = _snap_subscription(db, subscription_id)
    return {"success": True, "subscription_id": subscription_id, "plan": body.plan,
            "request_limit": catalog.request_limit_for(body.plan),
            # Said on every response that changes a limit, because a limit
            # nobody enforces reads as a ceiling to whoever set it.
            "quota_enforced": _quota_enforced()}


@router.patch("/subscriptions/{subscription_id}/term")
def set_subscription_term(
    subscription_id: str,
    body: TermBody,
    actor: AdminActor = Depends(require_operator),
    db: Session = Depends(get_db),
):
    before = _snap_subscription(db, subscription_id)
    if not before:
        raise HTTPException(status_code=404, detail="No such subscription.")

    if body.extend_days is not None and body.expires_at is not None:
        raise HTTPException(
            status_code=422,
            detail="Pass expires_at or extend_days, not both.",
        )

    if body.extend_days is not None:
        # Extend from the CURRENT expiry when there is one, not from today —
        # extending a term that has three months left should give it three
        # months plus the extension, not restart the clock and lose them.
        base = before.get("expires_at") or datetime.utcnow()
        if base < datetime.utcnow():
            base = datetime.utcnow()
        expires_at = base + timedelta(days=body.extend_days)
    else:
        expires_at = body.expires_at

    with service_errors(), mutate(db, actor, action="subscription.term",
                target=("subscription", subscription_id),
                reason=body.reason, before=before) as m:
        m.result = licensing_service.set_subscription_term(db, subscription_id, expires_at)
        m.after = _snap_subscription(db, subscription_id)
    return {"success": True, "subscription_id": subscription_id,
            "expires_at": q.iso(expires_at)}


# ── Licences ─────────────────────────────────────────────────────────────────

@router.post("/subscriptions/{subscription_id}/licence")
def issue_licence(
    subscription_id: str,
    body: IssueLicenceBody = IssueLicenceBody(),
    actor: AdminActor = Depends(require_owner),
    db: Session = Depends(get_db),
):
    """Mint a key, or rotate the existing one. OWNER ONLY.

    ROTATION IS DESTRUCTIVE TO A WORKING INSTALL. The previous key is deleted
    and its cached contexts evicted, so the merchant's module stops working the
    moment this returns and stays broken until the new key is pasted in. The
    response carries the plaintext once — this is the moment to copy it.
    """
    sub = _snap_subscription(db, subscription_id)
    if not sub:
        raise HTTPException(status_code=404, detail="No such subscription.")
    site = _snap_site(db, sub["site_id"])

    existing = [l for l in licensing_service.list_licences(db, subscription_id) if l["is_active"]]

    with service_errors(), mutate(db, actor,
                action="licence.rotate" if existing else "licence.issue",
                target=("subscription", subscription_id),
                reason=body.reason,
                before={"active_licences": [l["key_prefix"] for l in existing]}) as m:
        result = licensing_service.issue_licence(
            db, subscription_id=subscription_id,
            environment=site["environment"], valid_days=body.valid_days,
        )
        m.result = result

        # The timeline the deleted rows no longer provide. Written for the keys
        # that were superseded AND for the new one, so a rotation reads as two
        # events rather than an unexplained appearance.
        for old in existing:
            _licence_event(db, licence_id=old["id"], subscription_id=subscription_id,
                           event="superseded", key_prefix=old["key_prefix"], actor=actor,
                           detail="Replaced by %s" % result["key_prefix"])
        _licence_event(db, licence_id=result["id"], subscription_id=subscription_id,
                       event="rotated" if existing else "issued",
                       key_prefix=result["key_prefix"], actor=actor,
                       detail=body.reason)

        # after_json carries the PREFIX only. admin_audit_log is viewer-readable.
        m.after = {"licence_id": result["id"], "key_prefix": result["key_prefix"],
                   "expires_at": result.get("expires_at")}

    return {
        "success": True,
        "licence_id": result["id"],
        # Shown once. It is also stored in licences.licence_key, so this is a
        # convenience rather than the only chance — but the console should treat
        # it as the moment to hand it over.
        "key": result["key"],
        "key_prefix": result["key_prefix"],
        "expires_at": result.get("expires_at"),
        "superseded": len(existing),
        "note": "The previous key stopped working immediately."
                if existing else "First key for this subscription.",
    }


@router.post("/licences/{licence_id}/revoke")
def revoke_licence(
    licence_id: str,
    body: RevokeBody,
    actor: AdminActor = Depends(require_owner),
    db: Session = Depends(get_db),
):
    """Kill a key permanently. OWNER ONLY, and there is no undo.

    Unlike a rotation the row is KEPT with revoked_at set, because a revoke is
    not replaced by anything: deleting it would leave the subscription looking
    like it never had a licence.
    """
    before = _snap_licence(db, licence_id)
    if not before:
        raise HTTPException(status_code=404, detail="No such licence.")

    row = q.one(db, "SELECT licence_key, subscription_id FROM licences WHERE id = :id",
                {"id": licence_id})
    prefix = license_key.prefix_of(row.get("licence_key"))

    # Type-to-confirm, server-side. The operator has to have looked at WHICH key
    # they are killing, not merely clicked a button next to it.
    if prefix and body.confirm_prefix.strip() != prefix:
        raise HTTPException(
            status_code=422,
            detail="confirm_prefix does not match this licence's prefix.",
        )

    with service_errors(), mutate(db, actor, action="licence.revoke", target=("licence", licence_id),
                reason=body.reason, before=before) as m:
        m.result = licensing_service.revoke_licence(db, licence_id)
        _licence_event(db, licence_id=licence_id,
                       subscription_id=row["subscription_id"], event="revoked",
                       key_prefix=prefix, actor=actor, detail=body.reason)
        m.after = _snap_licence(db, licence_id)
    return {"success": True, "licence_id": licence_id, "key_prefix": prefix}


# ── Products ─────────────────────────────────────────────────────────────────

@router.post("/products/{code}/withdraw")
def withdraw_product(
    code: str,
    body: ProductWithdrawBody,
    actor: AdminActor = Depends(require_owner),
    db: Session = Depends(get_db),
):
    """Stop offering a product. OWNER ONLY.

    NOT A KILL SWITCH, and the difference matters during an incident: existing
    subscriptions keep resolving and keep working. This only stops onboarding
    offering it. Wiring is_sellable into the request chokepoint would break
    every live subscription on the product, which is the opposite of what the
    column's own contract promises.
    """
    before = _snap_product(db, code)
    if not before:
        raise HTTPException(status_code=404, detail="No such product.")

    affected = q.i(q.scalar(db, "SELECT COUNT(*) FROM subscriptions WHERE product_code = :c",
                            {"c": code}))
    if body.confirm_affected != affected:
        raise HTTPException(
            status_code=422,
            detail=f"confirm_affected must equal the number of existing "
                   f"subscriptions ({affected}). Read the blast radius first.",
        )

    with service_errors(), mutate(db, actor, action="product.withdraw", target=("product", code),
                reason=body.reason, before=before) as m:
        db.execute(text("UPDATE products SET is_sellable = 0 WHERE code = :c"), {"c": code})
        m.after = _snap_product(db, code)
        # products.is_sellable is not part of any cached auth context — the
        # chokepoint deliberately never reads it — so there is genuinely nothing
        # to forget. Said explicitly rather than left to look like an oversight.
        m.evict = []
    return {"success": True, "code": code, "is_sellable": False,
            "existing_subscriptions_unaffected": affected}


@router.post("/products/{code}/restore")
def restore_product(
    code: str,
    body: OptionalReasonBody = OptionalReasonBody(),
    actor: AdminActor = Depends(require_owner),
    db: Session = Depends(get_db),
):
    before = _snap_product(db, code)
    if not before:
        raise HTTPException(status_code=404, detail="No such product.")
    with service_errors(), mutate(db, actor, action="product.restore", target=("product", code),
                reason=body.reason, before=before) as m:
        db.execute(text("UPDATE products SET is_sellable = 1 WHERE code = :c"), {"c": code})
        m.after = _snap_product(db, code)
        m.evict = []
    return {"success": True, "code": code, "is_sellable": True}


def _quota_enforced() -> bool:
    import os
    return os.getenv("AICHATBOT_QUOTA_ENFORCEMENT", "").lower() in ("1", "true", "on", "yes")
