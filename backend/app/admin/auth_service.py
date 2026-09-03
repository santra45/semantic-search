"""Password hashing and session lifecycle for admin_users.

────────────────────────────────────────────────────────────────────────────
WHY STDLIB scrypt AND NOT passlib[bcrypt], WHICH THE PLAN SPECIFIED.

Because passlib and bcrypt are both absent from the image, and dependencies are
installed at BUILD time while source is bind-mounted. Adding one turns Phase 2
from `git pull` into an image rebuild on a box with ~760 MB of free RAM running
a live API under --reload. Worse, the failure mode of getting it wrong is not
degraded auth: a module-scope `import passlib` that fails takes uvicorn down for
every tenant, storefronts included.

hashlib.scrypt is in the standard library, is already usable in the image
(verified: 55 ms at n=2**14 on the live container), and is a memory-hard KDF —
by modern standards a better choice than bcrypt, not a compromise. The cost is
that the encoding and the verify are written out below instead of being handed
to a library, which is ~40 lines of well-trodden ground.

If passlib is ever added for another reason, `verify_password` already reads the
algorithm tag off the stored string, so migrating is: add the new branch, and
re-hash on next successful login.
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import uuid
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# ── Session cookie ───────────────────────────────────────────────────────────

SESSION_COOKIE_NAME = "admin_session"
SESSION_COOKIE_PATH = "/"
SESSION_COOKIE_SAMESITE = "lax"
SESSION_TTL_HOURS = 12

# Secure defaults ON. The console is served same-origin from an https host, so
# the only environment this inconveniences is a developer on plain http, who can
# opt out explicitly. The reverse default — insecure unless told otherwise — is
# how a session cookie ends up going out over http in production because nobody
# set the variable.
SESSION_COOKIE_SECURE = os.getenv("ADMIN_COOKIE_SECURE", "1") not in ("0", "false", "False")


# ── Password hashing ─────────────────────────────────────────────────────────

# n=2**14, r=8, p=1 costs ~55 ms and ~16 MB per hash (128*n*r bytes). The memory
# figure is the one to keep in mind on a small box: concurrent logins each want
# 16 MB while hashing. Login is throttled to 5 attempts per IP per 15 minutes,
# so the ceiling is set by how many operators exist, not by an attacker.
_SCRYPT_N = 2 ** 14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
_SALT_BYTES = 16
# maxmem must be >= 128*n*r or scrypt raises. Computed from the constants rather
# than hardcoded so raising _SCRYPT_N cannot leave this behind — and generous
# enough to also cover a stored hash whose embedded parameters are higher than
# today's constants, which is exactly what happens the first time they are
# raised and old hashes are still being verified.
_SCRYPT_MAXMEM = 128 * _SCRYPT_N * _SCRYPT_R * 4


MIN_PASSWORD_LENGTH = 12

# scrypt hashes the raw bytes, so a 400 KB "password" is 400 KB of work. Capped
# well above anything a human or a password manager produces.
MAX_PASSWORD_BYTES = 1024


def validate_password_strength(password: str) -> Optional[str]:
    """None if acceptable, else the reason it is not.

    Length only, deliberately. Character-class rules ("one uppercase, one
    symbol") measurably push people toward Password1! and are not what makes a
    passphrase hard to guess. This is an internal console with a handful of
    accounts and a login throttle; length plus a password manager is the win.
    """
    if password is None or password == "":
        return "Password is required."
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        return f"Password must be at most {MAX_PASSWORD_BYTES} bytes."
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
    return None


def hash_password(password: str) -> str:
    """`scrypt$n$r$p$salt_hex$hash_hex` — self-describing, so parameters can be
    raised later without invalidating existing hashes.

    The parameters live IN the string rather than being read from the constants
    at verify time. Bump _SCRYPT_N with the parameters implicit and every
    existing password stops verifying, silently, because the same input now
    derives a different key.
    """
    problem = validate_password_strength(password)
    if problem:
        raise ValueError(problem)

    salt = os.urandom(_SALT_BYTES)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_SCRYPT_DKLEN,
        maxmem=_SCRYPT_MAXMEM,
    )
    return "scrypt$%d$%d$%d$%s$%s" % (
        _SCRYPT_N, _SCRYPT_R, _SCRYPT_P, salt.hex(), digest.hex()
    )


def verify_password(password: str, password_hash: str) -> bool:
    """Constant-time verify. False for anything malformed, never an exception.

    A raise here would leak, through a 500 instead of a 401, that this
    particular account's hash is corrupt — and would turn one bad row into an
    endpoint that errors instead of denying.
    """
    if not password or not password_hash:
        return False
    try:
        algo, n, r, p, salt_hex, want_hex = password_hash.split("$")
        if algo != "scrypt":
            logger.error("admin auth: unknown password hash algorithm %r", algo)
            return False
        if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
            return False
        got = hashlib.scrypt(
            password.encode("utf-8"),
            salt=bytes.fromhex(salt_hex),
            n=int(n), r=int(r), p=int(p), dklen=len(want_hex) // 2,
            maxmem=_SCRYPT_MAXMEM,
        )
    except Exception:
        logger.exception("admin auth: malformed password hash, denying")
        return False
    return hmac.compare_digest(got.hex(), want_hex)



def burn_verify(password: str) -> None:
    """Hash against a throwaway salt and discard the result.

    Called when the email is unknown. Without it, "no such account" returns in
    under a millisecond while a real account costs 55 ms, and that gap is a
    reliable oracle for enumerating who has an account here. The response text
    being identical does not help if the timing is not.
    """
    try:
        hashlib.scrypt(
            (password or "").encode("utf-8")[:MAX_PASSWORD_BYTES],
            salt=os.urandom(_SALT_BYTES),
            n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_SCRYPT_DKLEN,
            maxmem=_SCRYPT_MAXMEM,
        )
    except Exception:
        pass


# ── Session tokens ───────────────────────────────────────────────────────────

def new_session_token() -> str:
    """The value that goes in the cookie. 256 bits, URL-safe."""
    return secrets.token_urlsafe(32)


def hash_token(raw_token: str) -> str:
    """What admin_sessions.id stores.

    SHA-256 and not a KDF, on purpose and for the opposite reason to passwords:
    this input is 256 bits of os.urandom, not a human choice, so there is no
    dictionary to attack and stretching it would only add latency to every
    single request. The hash is here so a database dump does not contain live
    session tokens.
    """
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def create_session(
    db: Session,
    admin_user_id: str,
    ip: Optional[str] = None,
    user_agent: Optional[str] = None,
    ttl_hours: int = SESSION_TTL_HOURS,
) -> str:
    """Insert a session row and return the RAW token for the cookie.

    The raw token is returned and never stored. This is the only moment it
    exists server-side; the caller sets it on the response and forgets it.
    """
    raw = new_session_token()
    db.execute(
        text("""
            INSERT INTO admin_sessions
                (id, admin_user_id, ip, user_agent, expires_at)
            VALUES
                (:id, :uid, :ip, :ua, :exp)
        """),
        {
            "id": hash_token(raw),
            "uid": admin_user_id,
            "ip": (ip or None),
            # Truncated, not rejected: a long User-Agent is a browser being a
            # browser, and failing a login over it would be absurd.
            "ua": (user_agent or "")[:255] or None,
            "exp": datetime.utcnow() + timedelta(hours=ttl_hours),
        },
    )
    db.commit()
    return raw


def lookup_session(db: Session, raw_token: str) -> Optional[dict]:
    """The admin behind a cookie, or None. Liveness is checked in SQL.

    Both conditions, every time: revoked_at IS NULL **and** expires_at in the
    future. Checking only one is the bug that makes logout look like it worked
    until the session would have expired anyway.

    Joins admin_users and requires is_active there too, so deactivating an
    operator kills their live sessions on the next request rather than at expiry.
    """
    if not raw_token:
        return None
    row = db.execute(
        text("""
            SELECT s.id            AS session_id,
                   s.expires_at    AS expires_at,
                   u.id            AS admin_user_id,
                   u.email         AS email,
                   u.name          AS name,
                   u.role          AS role
            FROM admin_sessions s
            JOIN admin_users u ON u.id = s.admin_user_id
            WHERE s.id           = :id
              AND s.revoked_at IS NULL
              AND s.expires_at   > UTC_TIMESTAMP()
              AND u.is_active    = 1
        """),
        {"id": hash_token(raw_token)},
    ).fetchone()
    if row is None:
        return None
    return {
        "session_id": row.session_id,
        "admin_user_id": row.admin_user_id,
        "email": row.email,
        "name": row.name,
        "role": row.role,
        "expires_at": row.expires_at,
    }


def revoke_session(db: Session, raw_token: str) -> bool:
    """Log out one session. True if a live session was actually revoked."""
    if not raw_token:
        return False
    result = db.execute(
        text("""
            UPDATE admin_sessions SET revoked_at = UTC_TIMESTAMP()
            WHERE id = :id AND revoked_at IS NULL
        """),
        {"id": hash_token(raw_token)},
    )
    db.commit()
    return bool(result.rowcount)


def revoke_all_sessions_for_user(
    db: Session,
    admin_user_id: str,
    except_session_id: Optional[str] = None,
    commit: bool = True,
) -> int:
    """Kill every live session for an operator. Returns how many.

    `except_session_id` (a HASHED id, not a raw token) keeps the caller's own
    session alive — used on password change, where logging the user out of the
    browser they just changed it in is hostile, but leaving their other sessions
    running defeats the point of changing it.

    commit=False lets this join a caller's transaction; a password update and
    the session sweep that follows it must not be separately committable, or a
    crash between them leaves the old password's sessions alive.
    """
    sql = """
        UPDATE admin_sessions SET revoked_at = UTC_TIMESTAMP()
        WHERE admin_user_id = :uid AND revoked_at IS NULL
    """
    params = {"uid": admin_user_id}
    if except_session_id:
        sql += " AND id <> :keep"
        params["keep"] = except_session_id
    result = db.execute(text(sql), params)
    if commit:
        db.commit()
    return int(result.rowcount or 0)


def touch_last_login(db: Session, admin_user_id: str) -> None:
    db.execute(
        text("UPDATE admin_users SET last_login_at = UTC_TIMESTAMP() WHERE id = :id"),
        {"id": admin_user_id},
    )
    db.commit()


# ── Accounts ─────────────────────────────────────────────────────────────────

def new_admin_id() -> str:
    return str(uuid.uuid4())


def get_admin_by_email(db: Session, email: str) -> Optional[dict]:
    row = db.execute(
        text("""
            SELECT id, email, name, password_hash, role, is_active
            FROM admin_users WHERE email = :email
        """),
        {"email": (email or "").strip().lower()},
    ).fetchone()
    return dict(row._mapping) if row else None


def get_admin_by_id(db: Session, admin_user_id: str) -> Optional[dict]:
    row = db.execute(
        text("""
            SELECT id, email, name, role, is_active, last_login_at, created_at
            FROM admin_users WHERE id = :id
        """),
        {"id": admin_user_id},
    ).fetchone()
    return dict(row._mapping) if row else None


def set_password(
    db: Session,
    admin_user_id: str,
    new_password: str,
    keep_session_id: Optional[str] = None,
) -> int:
    """Change a password and revoke every other session. Returns sessions killed.

    One transaction, one commit, deliberately. Split them and a crash in between
    leaves the password changed while sessions opened under the old one keep
    working — which is precisely the situation someone changing their password
    under suspicion is trying to end.
    """
    problem = validate_password_strength(new_password)
    if problem:
        raise ValueError(problem)

    db.execute(
        text("UPDATE admin_users SET password_hash = :h WHERE id = :id"),
        {"h": hash_password(new_password), "id": admin_user_id},
    )
    killed = revoke_all_sessions_for_user(
        db, admin_user_id, except_session_id=keep_session_id, commit=False
    )
    db.commit()
    logger.info(
        "admin auth: password changed for %s, %d other session(s) revoked",
        admin_user_id, killed,
    )
    return killed
