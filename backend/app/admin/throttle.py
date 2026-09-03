"""Login attempt throttling: 5 failures per IP per 15 minutes.

Not services/rate_limiter.py, which keys on (client_id, ip) and is about
merchant licences — there is no client_id at a login, which is the whole point:
the caller has not proved who they are yet.

Redis when it is there, an in-process dict when it is not. The fallback is
deliberately weak and deliberately present: with several uvicorn workers each
holds its own counter, so the real limit is 5 x workers. That is still three
orders of magnitude below what makes online password guessing viable, and the
alternative — failing login closed when Redis blips — locks every operator out
of the console at exactly the moment they most need to look at something.

Successful logins clear the counter, so a person who fumbles their password
twice and then gets it right is not carrying three strikes for a quarter hour.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 5
WINDOW_SECONDS = 15 * 60

_KEY_PREFIX = "admin:login:fail:"

# ip -> (count, window_started_at). Bounded by _MAX_LOCAL_ENTRIES so a spray
# across forged X-Forwarded-For values cannot grow it without limit — this dict
# lives for the lifetime of the process.
_local: dict[str, tuple[int, float]] = {}
_MAX_LOCAL_ENTRIES = 10_000


# ── Redis, behind a circuit breaker ──────────────────────────────────────────
#
# auth_cache's module docstring makes the point that matters here: a try/except
# around a Redis call catches Redis FAILING, not Redis HANGING. redis-py
# defaults socket_timeout and socket_connect_timeout to None, and the shared
# client this borrows sets neither — so an unreachable host does not raise, it
# blocks, and every login blocks with it.
#
# That is worse than the outage this fallback exists to survive: a login
# endpoint that hangs is indistinguishable from a dead console, and it hangs on
# the exact "Redis blipped" scenario the local counter is meant to cover.
#
# So the first failure opens the circuit and the next _COOLDOWN seconds go
# straight to the in-process counter without touching Redis. A dead Redis costs
# one slow request per cooldown, not one per attempt. Mutating the shared
# client's socket timeouts would fix it more directly and is off the table —
# that pool is shared with the rate limiter and the auth cache, and reaching
# into it here would change their behaviour invisibly.

_REDIS_DOWN_UNTIL = 0.0
_REDIS_COOLDOWN_SECONDS = 30.0


def _redis():
    """The stack's shared Redis client, or None when it is unusable.

    Imported inside the function, matching auth_cache._redis(): cache_service
    builds its client at module import, so a missing package or a bad
    REDIS_PORT would turn a degraded throttle into an ImportError that takes the
    whole app down at boot.
    """
    if time.time() < _REDIS_DOWN_UNTIL:
        return None
    try:
        from backend.app.services.cache_service import r
        return r
    except Exception:
        _trip_breaker()
        return None


def _trip_breaker() -> None:
    """Stop using Redis for a while. Called on any failure, including a hang."""
    global _REDIS_DOWN_UNTIL
    was_up = time.time() >= _REDIS_DOWN_UNTIL
    _REDIS_DOWN_UNTIL = time.time() + _REDIS_COOLDOWN_SECONDS
    if was_up:
        # Once per cooldown, not once per attempt — a dead Redis under load
        # would otherwise fill the log faster than anything else in it.
        logger.warning(
            "admin throttle: Redis unusable, falling back to the in-process "
            "counter for %.0fs. The limit is now per worker, so the effective "
            "ceiling is %d x workers.",
            _REDIS_COOLDOWN_SECONDS, MAX_ATTEMPTS,
        )


def _key(ip: str) -> str:
    # IPs only, and truncated. Never the email — a Redis key naming an account
    # is an account-enumeration list sitting in a datastore with no access
    # control of its own.
    return _KEY_PREFIX + (ip or "unknown")[:45]


def _local_prune(now: float) -> None:
    if len(_local) < _MAX_LOCAL_ENTRIES:
        return
    for k, (_, started) in list(_local.items()):
        if now - started >= WINDOW_SECONDS:
            _local.pop(k, None)
    if len(_local) >= _MAX_LOCAL_ENTRIES:
        # Still full of live windows: drop the oldest half rather than refuse to
        # track anything new. Losing counters degrades the throttle; refusing
        # would either leak memory or start denying logins.
        for k, _ in sorted(_local.items(), key=lambda kv: kv[1][1])[: _MAX_LOCAL_ENTRIES // 2]:
            _local.pop(k, None)


def failures(ip: str) -> int:
    """Failed attempts in the current window."""
    client = _redis()
    if client is not None:
        try:
            raw = client.get(_key(ip))
            return int(raw) if raw else 0
        except Exception:
            _trip_breaker()

    now = time.time()
    count, started = _local.get(_key(ip), (0, now))
    if now - started >= WINDOW_SECONDS:
        return 0
    return count


def is_blocked(ip: str) -> bool:
    return failures(ip) >= MAX_ATTEMPTS


def record_failure(ip: str) -> int:
    """Count one failed attempt. Returns the new total.

    The TTL is set only when the counter is created, so the window runs from the
    FIRST failure. Refreshing it on every attempt would let a persistent guesser
    hold themselves blocked forever — which sounds fine until it is an operator
    who mistyped, locked out for as long as they keep trying.
    """
    client = _redis()
    if client is not None:
        try:
            k = _key(ip)
            count = client.incr(k)
            if count == 1:
                client.expire(k, WINDOW_SECONDS)
            return int(count)
        except Exception:
            _trip_breaker()

    now = time.time()
    _local_prune(now)
    k = _key(ip)
    count, started = _local.get(k, (0, now))
    if now - started >= WINDOW_SECONDS:
        count, started = 0, now
    count += 1
    _local[k] = (count, started)
    return count


def clear(ip: str) -> None:
    """Reset on successful login."""
    client = _redis()
    if client is not None:
        try:
            client.delete(_key(ip))
        except Exception:
            _trip_breaker()
    _local.pop(_key(ip), None)


def retry_after(ip: str) -> Optional[int]:
    """Seconds until the window expires, for the Retry-After header.

    Redis TTL when available; otherwise computed from the local window. Falls
    back to the full window rather than None, because a client told to retry
    immediately will, and be denied again.
    """
    client = _redis()
    if client is not None:
        try:
            ttl = client.ttl(_key(ip))
            if ttl and ttl > 0:
                return int(ttl)
        except Exception:
            _trip_breaker()
    entry = _local.get(_key(ip))
    if entry:
        remaining = WINDOW_SECONDS - (time.time() - entry[1])
        if remaining > 0:
            return int(remaining)
    return WINDOW_SECONDS
