"""mutate() — the ONLY way the admin API is allowed to change anything.

    with mutate(db, actor,
                action="subscription.pause",
                target=("subscription", subscription_id),
                reason=body.reason,
                before=snapshot_subscription(db, subscription_id)) as m:
        m.result = licensing_service.set_subscription_status(db, subscription_id, "paused")
        m.after  = snapshot_subscription(db, subscription_id)

Three things happen together:

  1. the audit row is written BEFORE the mutation and commits WITH it
  2. auth_cache is evicted after that commit, with the count recorded
  3. the eviction list is taken from the mutator's own return value, so it
     cannot be forgotten

────────────────────────────────────────────────────────────────────────────
WHY THE AUDIT ROW GOES FIRST. (Corrected 2026-09-03 — the first version of this
file had it the other way round and its docstring claimed a guarantee it did
not provide.)

Every mutator in tenancy_service and licensing_service calls db.commit() itself.
set_client_active, set_site_active, set_site_environment, set_index_plan,
create_subscription, set_subscription_{plan,status,term}, issue_licence and
revoke_licence — all of them. So a wrapper that ran the mutation and THEN
inserted its audit row was describing something already committed: if the INSERT
failed, there was nothing left to roll back, and the promise that "an unlogged
disable cannot happen" was decoration.

It went unnoticed because the test used a fake session that never committed. A
test double that is more transactional than the real thing will confirm any
ordering you like.

Inserting first inverts it. The row is written into the caller's open
transaction and the service's own commit persists BOTH, atomically, with no
extra commit here. If the service raises before committing, the rollback takes
the audit row with it. If the audit INSERT itself fails, the mutation never runs.
`after_json` and `evicted` are filled in afterwards by UPDATE, because neither
value exists yet at insert time — but the ROW's existence, which is the part
that matters, is already tied to the change it describes.

WHY THE EVICTION LIST IS TAKEN, NOT GIVEN.

auth_cache holds a resolved licence context for 300 seconds and eviction is
deliberately the caller's job. A missed eviction is visible to a READER and
invisible to an OPERATOR, who sees 200 and a green toast while a disabled tenant
keeps authorising for five more minutes.

The first version made `m.evict` mandatory and rolled back if it was unset —
which, given the internal commits above, could not actually roll anything back.
So instead of asking the endpoint to remember, `m.result` takes the mutator's
return dict and this file reads `key_hashes` out of it. Every shipped mutator
returns one. An endpoint whose action genuinely has no cached state behind it
says so with `m.evict = []`, explicitly.
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
    """The handle yielded by mutate()."""

    __slots__ = ("after", "detail", "_evict", "_result")

    def __init__(self) -> None:
        self.after: Optional[dict] = None
        self.detail: Optional[str] = None
        self._evict: Any = _UNSET
        self._result: Optional[dict] = None

    # ── The normal path: hand over what the service returned ─────────────────
    @property
    def result(self) -> Optional[dict]:
        return self._result

    @result.setter
    def result(self, value: Optional[dict]) -> None:
        """Take the mutator's return dict and read `key_hashes` out of it.

        This is the whole anti-forgetting mechanism. `key_hash` (singular, the
        NEW key from issue_licence) is deliberately ignored — evicting it would
        throw away the entry for the key that was just created, which is the one
        thing a rotation must not do.
        """
        self._result = value
        if isinstance(value, dict) and "key_hashes" in value:
            self.evict = value["key_hashes"]

    # ── The explicit path: for actions with no cached state ──────────────────
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
            # A bare string is iterable, so this would evict one character at a
            # time and report a healthy-looking count while forgetting nothing.
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

    default=str so a datetime or Decimal from a row serialises instead of
    exploding — losing a real change because a timestamp would not serialise is
    a bad trade for type purity.
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
    before: Optional[dict] = None,
):
    """Run a mutation, log it, evict the cache it invalidates.

    `before` is a parameter rather than an attribute set inside the block,
    because the audit row has to be INSERTed before the mutation runs — see the
    module docstring. There is no point in the block at which this file could
    read it otherwise.
    """
    target_type, target_id = target

    if action in REASON_REQUIRED and not (reason or "").strip():
        raise ValueError(
            f"'{action}' takes something offline and requires a reason. "
            f"Reject the request with 422 rather than calling mutate()."
        )

    # Written into the caller's OPEN transaction, uncommitted. The service
    # mutator's own db.commit() is what persists this row, together with the
    # change it describes.
    db.execute(
        text("""
            INSERT INTO admin_audit_log
                (admin_user_id, actor_email, action, target_type, target_id,
                 before_json, reason, ip)
            VALUES
                (:uid, :email, :action, :ttype, :tid, :before, :reason, :ip)
        """),
        {
            "uid": actor.admin_user_id,
            "email": actor.email,
            "action": action,
            "ttype": target_type,
            "tid": str(target_id),
            "before": _json(before),
            "reason": (reason or None),
            "ip": actor.ip,
        },
    )
    # Read now, on this connection, while it is still the last insert. Reading
    # it after the service has run would return whatever IT last inserted.
    audit_id = db.execute(text("SELECT LAST_INSERT_ID()")).scalar()

    ctx = MutationContext()
    try:
        yield ctx
    except Exception:
        # Roll back: discards the audit row and any uncommitted part of the
        # mutation. A change that did not happen must not leave a log entry
        # saying it did.
        _safe_rollback(db)
        logger.exception(
            "admin audit: %s on %s/%s FAILED for %s — rolled back, nothing logged",
            action, target_type, target_id, actor.email,
        )
        raise

    if not ctx.evict_was_set:
        # NOT recoverable by rollback: the service committed, so the change is
        # live and so is the audit row. Loud rather than silent, and the message
        # says the change stands — an operator who reads "failed" and retries
        # would apply it twice.
        logger.critical(
            "admin audit: %s on %s/%s set neither m.result nor m.evict. The "
            "mutation is COMMITTED and cached auth contexts were NOT evicted, "
            "so it will not take effect for up to 300s. Set m.result to the "
            "service's return value, or m.evict = [] if this action has no "
            "cached state behind it.",
            action, target_type, target_id,
        )

    # Ensure the audit row is durable even if the service somehow did not
    # commit. A no-op when it did.
    try:
        db.commit()
    except Exception:
        _safe_rollback(db)
        logger.exception("admin audit: commit failed after %s", action)
        raise

    # AFTER the commit, never before. auth_cache repopulates from whatever the
    # database says at the moment it is asked, so evicting first races the
    # commit and can pull the PRE-change row straight back into the cache.
    evicted = 0
    if ctx.evict:
        try:
            evicted = auth_cache.invalidate_many(ctx.evict)
        except Exception:
            # Redis being down must not undo a committed mutation. Loud, because
            # the write IS live and the cache is not.
            logger.exception(
                "admin audit: %s on %s/%s COMMITTED but eviction FAILED for %d "
                "key(s). Cached contexts stay valid for up to 300s.",
                action, target_type, target_id, len(ctx.evict),
            )
            evicted = -1

    # NULL means "this action had no cached state behind it"; 0 means "it had
    # some and forgot none". Those are opposite facts and the column previously
    # collapsed them: an explicit `m.evict = []` and a mutator that returned
    # three hashes none of which were evicted both stored 0.
    #
    # That made the number unreadable exactly where it matters. 0 is the
    # five-minute-stale-toggle bug and should be alarming; a create with nothing
    # cached is routine and should not be. Distinguishing them here is what lets
    # the audit screen flag one without crying wolf about the other.
    evicted_recorded = evicted if ctx.evict else None

    try:
        db.execute(
            text("UPDATE admin_audit_log SET after_json = :after, evicted = :n WHERE id = :id"),
            {"after": _json(ctx.after), "n": evicted_recorded, "id": audit_id},
        )
        db.commit()
    except Exception:
        # A missing after-snapshot is a blemish on a row that is otherwise
        # correct. Never let it undo the mutation.
        _safe_rollback(db)
        logger.exception("admin audit: could not complete row %s", audit_id)

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
