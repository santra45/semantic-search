"""mutate() — the ONLY way the admin API is allowed to change anything.

    with mutate(db, actor,
                action="subscription.pause",
                target=("subscription", subscription_id),
                reason=body.reason) as m:
        m.before = snapshot_subscription(db, subscription_id)
        result   = licensing_service.set_subscription_status(db, subscription_id, "paused")
        m.after  = snapshot_subscription(db, subscription_id)
        m.evict  = result["key_hashes"]

Three things happen together, and that is the whole point:

  1. the mutation and its admin_audit_log row commit in ONE transaction
  2. auth_cache is evicted after that commit, with the count recorded
  3. neither can be skipped, because the block raises if you try

────────────────────────────────────────────────────────────────────────────
WHY THE EVICTION IS FORCED RATHER THAN TRUSTED.

auth_cache holds a resolved licence context for 300 seconds and eviction is
deliberately the CALLER's job — every mutator in tenancy_service and
licensing_service returns `key_hashes` and evicts nothing itself, so that a
missing eviction is a visible omission at the call site.

Visible to a reader. Not visible to an operator, who sees HTTP 200 and a green
toast while the disabled tenant keeps authorising for five more minutes. That
failure gets "tested" by someone who waits, refreshes, sees the key rejected,
and concludes the toggle works.

So: m.evict is mandatory. A block that exits without setting it raises, in
tests and in production alike, and an endpoint that genuinely has nothing to
evict says so with `m.evict = []`. Explicit nothing, never implicit nothing.
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import contextlib
import json
import logging
from typing import Any, Iterable, Optional, Sequence

from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.admin.deps import AdminActor
from backend.app.services import auth_cache

logger = logging.getLogger(__name__)

# Actions that take something away from a merchant. The API rejects these with
# 422 when `reason` is blank, because "who turned this off and why" asked six
# months later is answerable only if somebody was made to type it at the time.
REASON_REQUIRED = frozenset({
    "client.disable",
    "site.disable",
    "subscription.pause",
    "licence.revoke",
    "product.withdraw",
})

_UNSET = object()


class MutationContext:
    """The handle yielded by mutate(). Fill in before/after/evict."""

    __slots__ = ("before", "after", "_evict", "detail")

    def __init__(self) -> None:
        self.before: Optional[dict] = None
        self.after: Optional[dict] = None
        self.detail: Optional[str] = None
        self._evict: Any = _UNSET

    @property
    def evict(self) -> Sequence[str]:
        return () if self._evict is _UNSET else self._evict

    @evict.setter
    def evict(self, hashes: Iterable[str]) -> None:
        if hashes is None:
            raise ValueError(
                "m.evict = None is not a way to say 'nothing to evict'. Use "
                "m.evict = [] so the intent is in the diff."
            )
        if isinstance(hashes, str):
            # A bare string is iterable, so this would otherwise evict one
            # character at a time and report a healthy-looking count while
            # forgetting nothing. licensing_service's own comments flag the same
            # trap; catching it here means it cannot reach the cache at all.
            raise TypeError(
                "m.evict wants a list of key hashes, not a single string. "
                "A str would be iterated character by character."
            )
        self._evict = list(hashes)

    @property
    def evict_was_set(self) -> bool:
        return self._evict is not _UNSET


def _json(value: Optional[dict]) -> Optional[str]:
    """Snapshot to JSON, never raising.

    default=str so a datetime or a Decimal from a database row serialises
    instead of exploding — an audit row that fails to write rolls back the
    mutation it describes, and losing a real disable because a timestamp would
    not serialise is a bad trade for type purity.
    """
    if value is None:
        return None
    try:
        return json.dumps(value, default=str)
    except Exception:
        logger.exception("admin audit: snapshot would not serialise, storing marker")
        return json.dumps({"_error": "snapshot not serialisable"})


@contextlib.contextmanager
def mutate(
    db: Session,
    actor: AdminActor,
    *,
    action: str,
    target: tuple[str, str],
    reason: Optional[str] = None,
):
    """Run a mutation, log it, evict the cache it invalidates.

    Raises ValueError before touching anything when a reason is required and
    missing — the caller should have returned 422, and failing here is the
    backstop for when it forgot.
    """
    target_type, target_id = target

    if action in REASON_REQUIRED and not (reason or "").strip():
        raise ValueError(
            f"'{action}' takes something offline and requires a reason. "
            f"Reject the request with 422 rather than calling mutate()."
        )

    ctx = MutationContext()
    try:
        yield ctx
    except Exception:
        # The mutation failed. Roll back so no half-change survives, and write
        # NO audit row: this log is a record of what happened, and an entry for
        # a change that did not happen is worse than no entry.
        _safe_rollback(db)
        logger.exception(
            "admin audit: %s on %s/%s FAILED for %s — rolled back, nothing logged",
            action, target_type, target_id, actor.email,
        )
        raise

    if not ctx.evict_was_set:
        _safe_rollback(db)
        raise RuntimeError(
            f"'{action}' finished without setting m.evict, so the mutation was "
            f"rolled back.\n"
            f"auth_cache holds a resolved context for 300s and eviction is the "
            f"caller's job: every tenancy_service/licensing_service mutator "
            f"returns 'key_hashes' for this. Set m.evict = result['key_hashes'], "
            f"or m.evict = [] if this action genuinely has no cached state "
            f"behind it."
        )

    # The audit row goes in the SAME transaction as the mutation the caller just
    # made. If this INSERT fails, the mutation goes with it — an unlogged
    # disable is worse than a failed one.
    db.execute(
        text("""
            INSERT INTO admin_audit_log
                (admin_user_id, actor_email, action, target_type, target_id,
                 before_json, after_json, reason, evicted, ip)
            VALUES
                (:uid, :email, :action, :ttype, :tid,
                 :before, :after, :reason, :evicted, :ip)
        """),
        {
            "uid": actor.admin_user_id,
            "email": actor.email,
            "action": action,
            "ttype": target_type,
            "tid": str(target_id),
            "before": _json(ctx.before),
            "after": _json(ctx.after),
            "reason": (reason or None),
            # Written as NULL now and updated after the eviction below, because
            # the honest number is not known until the cache has been told.
            "evicted": None,
            "ip": actor.ip,
        },
    )
    audit_id = db.execute(text("SELECT LAST_INSERT_ID()")).scalar()
    db.commit()

    # AFTER the commit, never before. auth_cache repopulates from whatever the
    # database says at the moment it is asked, so evicting first races the
    # commit and can pull the PRE-change row straight back into the cache — a
    # correct-looking eviction that restores exactly what it was meant to clear.
    evicted = 0
    if ctx.evict:
        try:
            evicted = auth_cache.invalidate_many(ctx.evict)
        except Exception:
            # Redis being down must not undo a committed mutation. Loud, because
            # the write IS live and the cache is not: up to 300 seconds of stale
            # authorisation, and this line is the only warning anyone gets.
            logger.exception(
                "admin audit: %s on %s/%s COMMITTED but eviction FAILED for %d "
                "key(s). Cached contexts stay valid for up to 300s.",
                action, target_type, target_id, len(ctx.evict),
            )
            evicted = -1

    try:
        db.execute(
            text("UPDATE admin_audit_log SET evicted = :n WHERE id = :id"),
            {"n": evicted, "id": audit_id},
        )
        db.commit()
    except Exception:
        # A missing count is a blemish on a row that is otherwise correct. Never
        # let it undo the mutation.
        _safe_rollback(db)
        logger.exception("admin audit: could not record eviction count on row %s", audit_id)

    if ctx.evict and evicted == 0:
        # Asked to forget specific keys and forgot none. Either they had already
        # expired, or the wrong hashes were handed over — and the second is the
        # bug this column exists to surface.
        logger.warning(
            "admin audit: %s on %s/%s evicted 0 of %d key(s). If those keys were "
            "live, the change will not take effect for up to 300s.",
            action, target_type, target_id, len(ctx.evict),
        )

    logger.info(
        "admin audit: %s %s/%s by %s (evicted=%s)%s",
        action, target_type, target_id, actor.email, evicted,
        f" reason={reason!r}" if reason else "",
    )


def _safe_rollback(db: Session) -> None:
    try:
        db.rollback()
    except Exception:
        logger.exception("admin audit: rollback failed")
