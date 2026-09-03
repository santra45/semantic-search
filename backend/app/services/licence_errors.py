"""Structured licence refusals: a machine-readable code beside the human string.

ADMIN_CONSOLE_PLAN.md §8.1.

Today a dead licence produces `403 {"detail": "License key is inactive"}`. A
plugin receiving that cannot tell "your subscription was switched off" from "the
API is having a bad day", and that distinction is the entire basis of a safe
kill switch: one means hide the widget, the other means leave it exactly as it
is and try again later. Getting it backwards means one backend hiccup darkens
every storefront at once.

────────────────────────────────────────────────────────────────────────────
`detail` STAYS A STRING, AND STAYS BYTE-IDENTICAL.

Every shipped plugin build reads `detail` and renders it. Turning it into an
object to make room for the new fields would change what five deployed clients
display, in the same release that asks them to trust a new field — so the extra
keys are SIBLINGS of `detail`, not replacements. Old builds ignore them; new
builds act on them; nobody is forced to upgrade.

That is also why this is an exception subclass plus an app-level handler rather
than `HTTPException(detail={...})`: FastAPI serialises `detail` verbatim, so a
dict there would reshape the body.

`X-License-Status` is a header specifically so a plugin can act without parsing
a body at all — including on responses that carry none.
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.services import license_key as license_key_format

logger = logging.getLogger(__name__)

# The closed set a plugin may switch on. Adding a value is a client-visible
# change: a build that does not know it must fall through to "leave the widget
# alone", so new values are only ever safe if unknown ones fail open.
STATUS_ACTIVE = "active"
STATUS_DISABLED = "disabled"          # switched off by us — client/site/subscription
STATUS_EXPIRED = "expired"            # term ran out
STATUS_NOT_ENTITLED = "not_entitled"  # right key, wrong product for this endpoint
STATUS_QUOTA_EXCEEDED = "quota_exceeded"
STATUS_DOMAIN_MISMATCH = "domain_mismatch"
STATUS_INVALID = "invalid"            # unknown key, never issued, malformed

ALL_STATUSES = (
    STATUS_ACTIVE, STATUS_DISABLED, STATUS_EXPIRED, STATUS_NOT_ENTITLED,
    STATUS_QUOTA_EXCEEDED, STATUS_DOMAIN_MISMATCH, STATUS_INVALID,
)

# Only these mean "stop rendering the widget". Everything else — including
# STATUS_INVALID — leaves it alone. An invalid key is far more often a merchant
# mid-configuration than a revoked one, and hiding their widget while they paste
# a key is a worse failure than showing it briefly to nobody.
DEAD_STATUSES = frozenset({STATUS_DISABLED, STATUS_EXPIRED, STATUS_NOT_ENTITLED})


class LicenceDenied(HTTPException):
    """A refusal that carries a code, a status, and something to show a shopper."""

    def __init__(
        self,
        *,
        status_code: int,
        detail: str,
        error_code: str,
        licence_status: str,
        merchant_message: Optional[str] = None,
        retry_after: Optional[int] = None,
    ) -> None:
        headers = {"X-License-Status": licence_status}
        if retry_after is not None:
            headers["Retry-After"] = str(retry_after)
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code
        self.licence_status = licence_status
        self.merchant_message = merchant_message
        self.retry_after = retry_after

    def body(self) -> dict:
        return {
            # FIRST and unchanged. Everything below is additive.
            "detail": self.detail,
            "error_code": self.error_code,
            "license_status": self.licence_status,
            "merchant_message": self.merchant_message,
            "retry_after": self.retry_after,
        }


def diagnose(db: Session, presented_key: str) -> dict:
    """Why was this key refused? Read-only, and only on the failure path.

    resolve_key() collapses five different refusals into None — inactive
    licence, inactive site, inactive client, expired licence, non-active
    subscription — because on the hot path the answer is the same and one query
    is cheaper than five. Here the answer is the entire point, so the cost of a
    second lookup is paid once per DENIED request, which is rare by definition.

    Returns `status`, `merchant_message` (from subscriptions.disabled_reason,
    the sentence written FOR the merchant) and a little context for the log.
    Never raises: a diagnosis that fails must still produce a refusal, so the
    fallback is STATUS_INVALID, which is the safe reading of "we cannot tell".
    """
    unknown = {"status": STATUS_INVALID, "merchant_message": None, "product_code": None}
    if not presented_key:
        return unknown
    try:
        row = db.execute(
            text("""
                SELECT l.is_active        AS licence_active,
                       l.expires_at       AS licence_expires_at,
                       l.revoked_at       AS revoked_at,
                       s.status           AS subscription_status,
                       s.expires_at       AS subscription_expires_at,
                       s.disabled_reason  AS disabled_reason,
                       s.product_code     AS product_code,
                       si.is_active       AS site_active,
                       c.is_active        AS client_active
                FROM licences l
                JOIN subscriptions s ON s.id  = l.subscription_id
                JOIN sites si        ON si.id = s.site_id
                JOIN clients c       ON c.id  = si.client_id
                WHERE l.key_hash = :h
            """),
            {"h": license_key_format.hash_key(presented_key)},
        ).fetchone()
    except Exception:
        logger.exception("licence diagnosis failed; refusing as invalid")
        try:
            db.rollback()
        except Exception:
            pass
        return unknown

    if row is None:
        # No such key. Includes a rotated-out one: issue_licence() deletes the
        # row it supersedes, so "was rotated" and "never existed" are genuinely
        # indistinguishable here — licence_events is where that question is
        # answerable, and it is not on this path.
        return unknown

    from datetime import datetime
    now = datetime.utcnow()
    msg = row.disabled_reason
    ctx = {"product_code": row.product_code}

    # ORDER MATTERS: report the reason a human would act on first. A licence
    # that is both expired and belongs to a suspended client should say
    # "expired" only if that is the thing to fix — but a suspended CLIENT
    # outranks it, because renewing the term would not bring them back.
    if not row.client_active or not row.site_active:
        return {"status": STATUS_DISABLED, "merchant_message": msg, **ctx}
    if row.subscription_status not in ("active", "trial"):
        return {"status": STATUS_DISABLED, "merchant_message": msg, **ctx}
    if row.revoked_at is not None or not row.licence_active:
        return {"status": STATUS_DISABLED, "merchant_message": msg, **ctx}
    if row.licence_expires_at is not None and row.licence_expires_at <= now:
        return {"status": STATUS_EXPIRED, "merchant_message": msg, **ctx}
    if row.subscription_expires_at is not None and row.subscription_expires_at <= now:
        return {"status": STATUS_EXPIRED, "merchant_message": msg, **ctx}

    # Everything looks fine and it was still refused. Worth a log line: it means
    # resolve_key() and this function disagree, which is a bug in one of them.
    logger.warning(
        "licence diagnosis: key resolves to a healthy-looking row but was "
        "refused (product=%s). resolve_key() and diagnose() disagree.",
        row.product_code,
    )
    return {"status": STATUS_INVALID, "merchant_message": msg, **ctx}


def denied_for_key(
    db: Session,
    presented_key: str,
    *,
    detail: str,
    status_code: int = 403,
) -> LicenceDenied:
    """Build the refusal for a key that would not resolve.

    `detail` is passed in by the caller and reproduced verbatim, so existing
    call sites keep emitting exactly the string they emit today.
    """
    d = diagnose(db, presented_key)
    status = d["status"]
    return LicenceDenied(
        status_code=status_code,
        detail=detail,
        error_code=f"license_{status}",
        licence_status=status,
        merchant_message=d.get("merchant_message"),
        # Only for statuses a client should retry. Telling a plugin to come back
        # in five minutes about a revoked key invites a permanent poll.
        retry_after=300 if status in (STATUS_DISABLED, STATUS_EXPIRED) else None,
    )


def not_entitled(detail: str, merchant_message: Optional[str] = None) -> LicenceDenied:
    """Right key, wrong product for this endpoint."""
    return LicenceDenied(
        status_code=403,
        detail=detail,
        error_code="license_not_entitled",
        licence_status=STATUS_NOT_ENTITLED,
        merchant_message=merchant_message,
    )


def quota_exceeded(detail: str, retry_after: int = 3600) -> LicenceDenied:
    return LicenceDenied(
        status_code=429,
        detail=detail,
        error_code="license_quota_exceeded",
        licence_status=STATUS_QUOTA_EXCEEDED,
        merchant_message=None,
        retry_after=retry_after,
    )


def domain_mismatch(detail: str) -> LicenceDenied:
    return LicenceDenied(
        status_code=403,
        detail=detail,
        error_code="license_domain_mismatch",
        licence_status=STATUS_DOMAIN_MISMATCH,
    )
