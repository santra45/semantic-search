"""/api/admin/auth/* — login, logout, me, change password.

ADMIN_CONSOLE_PLAN.md §6.2. The only endpoints in the admin package that a
caller may reach without already being authenticated.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.admin import auth_service, throttle
from backend.app.admin.deps import AdminActor, client_ip, require_viewer
from backend.app.services.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/auth", tags=["admin-auth"])

# One string for both "no such account" and "wrong password". Two different
# messages turn this endpoint into an account-enumeration oracle, and pairing
# that with burn_verify() (which closes the timing side of the same leak) is
# what makes the two indistinguishable rather than merely similarly worded.
_BAD_CREDENTIALS = "Email or password is incorrect."


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=1, max_length=auth_service.MAX_PASSWORD_BYTES)


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=auth_service.MAX_PASSWORD_BYTES)
    new_password: str = Field(min_length=1, max_length=auth_service.MAX_PASSWORD_BYTES)


def _actor_payload(actor: AdminActor) -> dict:
    return {
        "id": actor.admin_user_id,
        "email": actor.email,
        "name": actor.name,
        "role": actor.role,
        "is_break_glass": actor.is_break_glass,
    }


@router.post("/login")
def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    ip = client_ip(request)

    if throttle.is_blocked(ip):
        wait = throttle.retry_after(ip)
        logger.warning("admin auth: login blocked, too many failures from %s", ip)
        raise HTTPException(
            status_code=429,
            detail="Too many failed sign-in attempts. Try again later.",
            headers={"Retry-After": str(wait)},
        )

    email = (body.email or "").strip().lower()
    admin = auth_service.get_admin_by_email(db, email)

    # burn_verify() on the miss, so an unknown email costs the same ~55 ms as a
    # real one. Returning early here instead would make account existence
    # readable straight off the response time, whatever the body says.
    if admin is None:
        auth_service.burn_verify(body.password)
        throttle.record_failure(ip)
        logger.info("admin auth: failed sign-in for unknown address from %s", ip)
        raise HTTPException(status_code=401, detail=_BAD_CREDENTIALS)

    if not auth_service.verify_password(body.password, admin["password_hash"]):
        throttle.record_failure(ip)
        logger.info("admin auth: failed sign-in for %s from %s", email, ip)
        raise HTTPException(status_code=401, detail=_BAD_CREDENTIALS)

    # Deactivated accounts fail AFTER the password check and with the same
    # message. Checking first would tell an attacker holding a wrong password
    # that the address exists; telling the truth here would tell someone holding
    # the RIGHT password of a disabled account that they were disabled rather
    # than deleted, which is information the console owes them from a human, not
    # from a login form.
    if not admin["is_active"]:
        throttle.record_failure(ip)
        logger.warning("admin auth: sign-in attempt on DEACTIVATED account %s from %s", email, ip)
        raise HTTPException(status_code=401, detail=_BAD_CREDENTIALS)

    raw = auth_service.create_session(
        db, admin["id"], ip=ip, user_agent=request.headers.get("user-agent")
    )
    auth_service.touch_last_login(db, admin["id"])
    throttle.clear(ip)

    response.set_cookie(
        key=auth_service.SESSION_COOKIE_NAME,
        value=raw,
        max_age=auth_service.SESSION_TTL_HOURS * 3600,
        path=auth_service.SESSION_COOKIE_PATH,
        # httponly: script on the page cannot read it, so an XSS anywhere on
        # this origin cannot walk away with a live operator session.
        httponly=True,
        secure=auth_service.SESSION_COOKIE_SECURE,
        # lax, not strict: the console is a real SPA that people reach from
        # links and bookmarks, and strict drops the cookie on those first
        # navigations, showing a login screen to somebody who is signed in.
        # lax still withholds it from cross-site POSTs, which is the CSRF case.
        samesite=auth_service.SESSION_COOKIE_SAMESITE,
    )

    logger.info("admin auth: %s signed in from %s", email, ip)
    return {
        "success": True,
        "admin": {
            "id": admin["id"],
            "email": admin["email"],
            "name": admin["name"],
            "role": admin["role"],
            "is_break_glass": False,
        },
    }


@router.post("/logout")
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    """Revoke the current session.

    No auth dependency, and that is deliberate: logging out with an already
    expired cookie should be a quiet success, not a 401 that leaves the browser
    holding a dead cookie it was trying to get rid of. Always 200.
    """
    raw = request.cookies.get(auth_service.SESSION_COOKIE_NAME)
    revoked = auth_service.revoke_session(db, raw) if raw else False

    # Cleared with the SAME path/secure/samesite the cookie was set with —
    # browsers match on those, and a mismatch silently leaves it in place.
    response.delete_cookie(
        key=auth_service.SESSION_COOKIE_NAME,
        path=auth_service.SESSION_COOKIE_PATH,
        httponly=True,
        secure=auth_service.SESSION_COOKIE_SECURE,
        samesite=auth_service.SESSION_COOKIE_SAMESITE,
    )
    return {"success": True, "revoked": revoked}


@router.get("/me")
def me(actor: AdminActor = Depends(require_viewer)):
    """Who am I. The SPA's first call — it decides login screen vs console."""
    return _actor_payload(actor)


@router.post("/password")
def change_password(
    body: PasswordChangeRequest,
    actor: AdminActor = Depends(require_viewer),
    db: Session = Depends(get_db),
):
    """Change your own password. Revokes every OTHER session.

    Any role, because this is self-service: `viewer` is the floor for being
    signed in at all, not a statement about privilege.
    """
    if actor.is_break_glass:
        # The operator key is a static environment variable. There is no row to
        # update and nothing this endpoint could change.
        raise HTTPException(
            status_code=400,
            detail="Break-glass access has no password. Rotate "
                   "AICHATBOT_OPERATOR_KEY in the environment instead.",
        )

    admin = auth_service.get_admin_by_email(db, actor.email)
    if admin is None or not auth_service.verify_password(
        body.current_password, admin["password_hash"]
    ):
        # Throttled on the caller's IP like a login: this endpoint verifies a
        # password, so it is a guessing oracle if left unmetered — and it is
        # reachable by anyone who has borrowed an unlocked browser.
        throttle.record_failure(actor.ip)
        logger.warning("admin auth: wrong current password for %s", actor.email)
        raise HTTPException(status_code=401, detail="Current password is incorrect.")

    problem = auth_service.validate_password_strength(body.new_password)
    if problem:
        raise HTTPException(status_code=422, detail=problem)

    if body.new_password == body.current_password:
        raise HTTPException(
            status_code=422, detail="New password must differ from the current one."
        )

    killed = auth_service.set_password(
        db, actor.admin_user_id, body.new_password, keep_session_id=actor.session_id
    )
    throttle.clear(actor.ip)
    return {"success": True, "other_sessions_revoked": killed}
