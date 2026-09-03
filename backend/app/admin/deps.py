"""FastAPI dependencies: who is calling, and are they allowed to.

    @router.get("/api/admin/tenants")
    def tenants(actor = Depends(require_viewer), db = Depends(get_db)): ...

    @router.post("/api/admin/clients/{id}/disable")
    def disable(actor = Depends(require_operator), db = Depends(get_db)): ...

Two ways in, and only two: a session cookie, or the break-glass operator key.
"""
from __future__ import annotations

import hmac
import logging
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from backend.app.admin import auth_service
from backend.app.config import OPERATOR_KEY
from backend.app.services.database import get_db

logger = logging.getLogger(__name__)

# Ordered, and compared by index. A set of allowed roles per endpoint would let
# "operator" be granted somewhere "owner" was not, which is not a hierarchy any
# more and is impossible to reason about at a glance.
ROLE_ORDER = ("viewer", "operator", "owner")

BREAK_GLASS_EMAIL = "break-glass"
BREAK_GLASS_ROLE = "owner"


@dataclass(frozen=True)
class AdminActor:
    """Who is making this request. Frozen: a dependency result that an endpoint
    can mutate is a role that can be edited mid-request."""
    admin_user_id: Optional[str]   # None for break-glass — no row to point at
    email: str
    name: str
    role: str
    session_id: Optional[str]      # hashed id, for "revoke my other sessions"
    ip: Optional[str]

    @property
    def is_break_glass(self) -> bool:
        return self.admin_user_id is None

    def can(self, min_role: str) -> bool:
        try:
            return ROLE_ORDER.index(self.role) >= ROLE_ORDER.index(min_role)
        except ValueError:
            # An unrecognised role denies rather than crashes, and denying is
            # the only safe reading of a value this code does not understand.
            logger.error("admin rbac: unknown role %r, denying", self.role)
            return False


def client_ip(request: Request) -> Optional[str]:
    """Best-effort caller IP.

    X-Forwarded-For FIRST because the console sits behind a reverse proxy, so
    request.client.host is the proxy on every request and would make the audit
    log and the login throttle useless — one IP for the entire internet.

    This header is caller-controlled and trivially spoofed. That is tolerable
    for an audit hint and for throttling an internal console reachable only by
    staff; it would NOT be tolerable as an authorisation input, and nothing here
    uses it as one.
    """
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()[:45]
    return request.client.host if request.client else None


def _check_break_glass(request: Request) -> bool:
    """True when a valid X-Operator-Key is presented.

    compare_digest, not ==. String equality short-circuits on the first
    differing byte and leaks the key one character at a time to anyone who can
    measure the response.

    An unset OPERATOR_KEY denies. Without that check an empty header would match
    an empty configured key and hand out owner on a deployment that simply never
    configured one.
    """
    presented = request.headers.get("x-operator-key")
    if not presented or not OPERATOR_KEY:
        return False
    return hmac.compare_digest(str(presented), str(OPERATOR_KEY))


def get_current_admin(
    request: Request,
    db: Session = Depends(get_db),
) -> AdminActor:
    """Resolve the caller. 401 if they are nobody.

    SESSION FIRST, break-glass second. An operator who is logged in AND has the
    key in their environment should be attributed to their account in the audit
    log, not to "break-glass" — otherwise the trail degrades to anonymous for
    whoever happens to have the key exported.
    """
    raw = request.cookies.get(auth_service.SESSION_COOKIE_NAME)
    if raw:
        session = auth_service.lookup_session(db, raw)
        if session:
            return AdminActor(
                admin_user_id=session["admin_user_id"],
                email=session["email"],
                name=session["name"],
                role=session["role"],
                session_id=session["session_id"],
                ip=client_ip(request),
            )

    if _check_break_glass(request):
        # WARNING, not info. This is an unattributable owner-level action and
        # every one of them should be visible in the log without going looking.
        # ADMIN_CONSOLE_PLAN.md §13.14: the key leaves production once real
        # accounts exist, or RBAC and the audit trail are decorative.
        logger.warning(
            "admin: BREAK-GLASS access from %s %s — no admin_user, audit rows "
            "will be attributed to '%s'",
            client_ip(request), request.url.path, BREAK_GLASS_EMAIL,
        )
        return AdminActor(
            admin_user_id=None,
            email=BREAK_GLASS_EMAIL,
            name="Break-glass operator key",
            role=BREAK_GLASS_ROLE,
            session_id=None,
            ip=client_ip(request),
        )

    # 401, not 403: the caller is unauthenticated, not under-privileged, and the
    # console needs the difference to know whether to redirect to /login or show
    # "you cannot do that".
    raise HTTPException(status_code=401, detail="Authentication required.")


def require_admin(min_role: str = "viewer"):
    """Dependency factory. `Depends(require_admin("operator"))`.

    Validated at import time, not per request: a typo'd role name should break
    at boot in an obvious way, not silently deny every call to one endpoint
    while the rest of the console works fine.
    """
    if min_role not in ROLE_ORDER:
        raise ValueError(
            f"Unknown role {min_role!r}. Expected one of {ROLE_ORDER}."
        )

    def _dependency(actor: AdminActor = Depends(get_current_admin)) -> AdminActor:
        if not actor.can(min_role):
            logger.warning(
                "admin rbac: %s (%s) denied, needs %s", actor.email, actor.role, min_role
            )
            raise HTTPException(
                status_code=403,
                detail=f"This action requires the '{min_role}' role or higher.",
            )
        return actor

    return _dependency


# Built once at import. Constructing these inside a decorator on every route
# would give each endpoint its own dependency object, defeating FastAPI's
# per-request dependency cache and re-running the session lookup per Depends.
require_viewer = require_admin("viewer")
require_operator = require_admin("operator")
require_owner = require_admin("owner")
