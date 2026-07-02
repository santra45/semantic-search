"""Request rate limiting — Redis-first, MySQL fallback.

Primary path: Redis atomic INCR + TTL over two fixed windows (a short burst
window + an hourly window) keyed per client_id+ip. Keys auto-expire, so there is
zero cleanup and zero table bloat; the increment is atomic, so there is no
read-then-write race and no duplicate-counting; and it is shared across workers.

Fallback (Redis unavailable): the caller applies its in-process burst damper for
the short window and calls `mysql_enforce_hourly` for the sustained window. The
MySQL path is hardened vs the original leaky implementation — it SUMs across any
duplicate rows (so pre-existing dupes can't silently multiply the limit) and
deletes the client+ip's expired rows first (window reset + no unbounded
accumulation).

Fails OPEN on any backend error — a rate limiter must never be the reason a
paying merchant's bot goes dark.
"""

from __future__ import annotations

import logging
import time

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger("rate_limiter")

BURST_WINDOW_SEC = 15
BURST_MAX = 30
HOUR_WINDOW_SEC = 3600
HOUR_MAX = 1000

_BURST_MSG = "Too many requests in a short time. Please slow down."
_HOUR_MSG = "Rate limit exceeded. Please try again later."


def redis_enforce(client_id: str, ip: str) -> bool:
    """Check both windows in Redis. Returns True when Redis handled the check
    (request allowed). Raises HTTPException(429) when over a limit. Returns
    False when Redis is unavailable so the caller can fall back to MySQL.

    Fixed-window pattern: `SET key 0 EX ttl NX` creates the counter with its TTL
    exactly once (on the first hit of the window), then `INCR` bumps it WITHOUT
    touching the TTL — so a continuously-active client's window still expires and
    resets on schedule instead of sliding forever.
    """
    try:
        from backend.app.services.cache_service import r

        bkey = f"rl:b:{client_id}:{ip}"
        hkey = f"rl:h:{client_id}:{ip}"
        pipe = r.pipeline()
        pipe.set(bkey, 0, ex=BURST_WINDOW_SEC, nx=True)
        pipe.incr(bkey)
        pipe.set(hkey, 0, ex=HOUR_WINDOW_SEC, nx=True)
        pipe.incr(hkey)
        res = pipe.execute()
        burst_count = int(res[1])
        hour_count = int(res[3])
    except Exception as exc:
        logger.warning("rate limiter: Redis unavailable (%s) — falling back to MySQL", exc)
        return False

    if burst_count > BURST_MAX:
        raise HTTPException(status_code=429, detail=_BURST_MSG)
    if hour_count > HOUR_MAX:
        raise HTTPException(status_code=429, detail=_HOUR_MSG)
    return True


def mysql_enforce_hourly(db: Session, client_id: str, ip: str) -> None:
    """Fixed-window hourly limit in the `rate_limits` table — the Redis-down
    fallback. Hardened vs the original: expired rows for this client+ip are
    deleted first (window reset + no accumulation), and the in-window count is
    SUMmed across any duplicate rows so dupes can't multiply the effective
    limit. Fails OPEN on any DB error.
    """
    now = int(time.time())
    cutoff = now - HOUR_WINDOW_SEC
    try:
        db.execute(
            text(
                "DELETE FROM rate_limits "
                "WHERE client_id = :c AND ip_address = :ip AND window_start <= :cutoff"
            ),
            {"c": client_id, "ip": ip, "cutoff": cutoff},
        )
        row = db.execute(
            text(
                "SELECT COALESCE(SUM(request_count), 0) AS total FROM rate_limits "
                "WHERE client_id = :c AND ip_address = :ip AND window_start > :cutoff"
            ),
            {"c": client_id, "ip": ip, "cutoff": cutoff},
        ).fetchone()
        total = int(row.total) if row and row.total is not None else 0
    except Exception as exc:
        logger.warning("rate limiter: MySQL fallback read failed (%s) — allowing", exc)
        _safe_rollback(db)
        return  # fail open

    if total >= HOUR_MAX:
        raise HTTPException(status_code=429, detail=_HOUR_MSG)

    # Record this hit: bump exactly ONE in-window row (LIMIT 1 so pre-existing
    # dupes don't each get incremented), else insert a fresh window row.
    try:
        updated = db.execute(
            text(
                "UPDATE rate_limits SET request_count = request_count + 1 "
                "WHERE client_id = :c AND ip_address = :ip AND window_start > :cutoff "
                "LIMIT 1"
            ),
            {"c": client_id, "ip": ip, "cutoff": cutoff},
        )
        if getattr(updated, "rowcount", 0) == 0:
            db.execute(
                text(
                    "INSERT INTO rate_limits (client_id, ip_address, request_count, window_start) "
                    "VALUES (:c, :ip, 1, :now)"
                ),
                {"c": client_id, "ip": ip, "now": now},
            )
        db.commit()
    except Exception as exc:
        logger.warning("rate limiter: MySQL fallback write failed (%s)", exc)
        _safe_rollback(db)


def _safe_rollback(db: Session) -> None:
    try:
        db.rollback()
    except Exception:
        pass
