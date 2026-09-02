"""Redis in front of licensing_service.resolve_key(), and the eviction rules
that make caching an authorisation decision safe.

WHY THIS IS A SEPARATE MODULE
-----------------------------
resolve_key() is five joins on the hottest path in the system: every sync
batch, every chat turn, every search. Caching it is obvious. Caching it
CORRECTLY is not, because what gets cached is not a lookup — it is a live
authorisation decision spanning four rows that four different admin actions can
each withdraw:

    clients.is_active -> sites.is_active -> subscriptions.status -> licences

A cache in front of that is a machine for keeping cancelled accounts alive.
Suspend a merchant for non-payment, watch them keep syncing for another five
minutes, and the cache has manufactured the exact incident it was added to
prevent. So the eviction rules live here, next to the writer, rather than being
sprinkled through whichever router happened to need them.

THE TTL IS A BACKSTOP, NOT THE MECHANISM
----------------------------------------
Say it in full, because "it's only five minutes" is how this gets wired wrong:
the TTL is what limits the damage when an invalidation hook is MISSING. It is
not the design. Every event below must evict explicitly. If the only bust that
ever got wired is the licence-level one, deactivating a client silently leaves
every key underneath it serving.

WIRING CHECKLIST — WHICH EVENT CALLS WHICH INVALIDATOR
------------------------------------------------------
Evict AFTER the mutation commits, never before. Evicting first leaves a window
in which a concurrent request reads the OLD committed row and writes it back
with a fresh full TTL, which is strictly worse than not evicting at all.

THE ORDINARY CASE IS ONE LINE. Every mutator in tenancy_service and
licensing_service that changes a value living in the cached context returns a
"key_hashes" list — the same field name, always a list, present even when it is
empty — so the wiring is:

    result = licensing_service.set_subscription_status(db, sub_id, "suspended")
    auth_cache.invalidate_many(result["key_hashes"])

and that one helper is correct for all of them. It did not used to be:
issue_licence returned "superseded_key_hashes", create_subscription and
set_index_plan returned nothing at all, so `evict(result.get("key_hashes", []))`
was silently a no-op on exactly the calls where a stale entry matters most. The
invalidate_for_* functions below are for the cases where there is no result to
read the list off — a hand-written UPDATE, a delete, a repair script.

  LICENCE LEVEL — the mutator already knows the hash, no lookup needed
    licensing_service.revoke_licence()      -> invalidate_many(r["key_hashes"])
    licensing_service.issue_licence()       -> invalidate_many(r["key_hashes"]),
                                               which is the SUPERSEDED keys. The
                                               singular r["key_hash"] beside it
                                               is the NEW key and is not
                                               something to evict; evicting the
                                               list is what stops the rotated-out
                                               key authorising for the rest of
                                               the TTL, which is the entire
                                               point of rotating it.
    any hand-written UPDATE licences SET is_active = 0
                                            -> invalidate(that key_hash), or
                                               invalidate_for_subscription() if
                                               you only have the subscription
    a licence EXPIRING                      -> no event exists to hook. Handled
                                               inside this module: put() clamps
                                               the TTL to the time left on the
                                               licence, and get() re-checks the
                                               expiry on every read. Nothing to
                                               wire, but do not "tidy away"
                                               either of those two guards.

  SUBSCRIPTION LEVEL — invalidate_for_subscription(db, subscription_id)
    set_subscription_status() to suspended / cancelled  (the money case)
    set_subscription_status() to active / trial         (an un-suspend that does
                                                         not take effect for
                                                         five minutes is also a
                                                         support ticket)
    set_subscription_plan()                             (request_limit rides in
                                                         the cached context; a
                                                         stale entry enforces
                                                         the plan the merchant
                                                         just stopped paying for)
    set_subscription_term()                             (subscriptions.expires_at
                                                         is the other half of the
                                                         TTL clamp below)

  SITE LEVEL — invalidate_for_site(db, site_id)
    tenancy_service.set_site_active()       (the per-store kill switch. This
                                             function exists BECAUSE this
                                             checklist named the event and
                                             nothing in the tree performed it:
                                             sites.is_active was a hand-written
                                             single-column UPDATE in a MySQL
                                             client, which by construction
                                             cannot fire an eviction hook, so
                                             this entry described a bust that
                                             could not happen)
    tenancy_service.set_index_plan()        (catalogue_limit is in the context)
    tenancy_service.set_site_environment()  (key_owner is DERIVED from it and is
                                             stamped onto every usage_events
                                             row; a stale entry books a client's
                                             own API spend as Czargroup COGS, or
                                             invoices them for ours)
    sites.collection_name / domain / store_name edited
                                            (a stale collection_name points the
                                             store at a collection that no
                                             longer exists, and Qdrant answers a
                                             missing collection with zero
                                             results rather than an error — the
                                             store just goes quiet)

  CLIENT LEVEL — invalidate_for_client(db, client_id)
    tenancy_service.set_client_active()     (THE one the brief calls out, and
                                             the other event that had no
                                             producer until that function was
                                             written. One client can own many
                                             sites and many subscriptions; this
                                             is the only call that reaches all
                                             of them)
    clients.name edited                     (cosmetic, but it is in the context)

  DELETES — capture the hashes BEFORE the DELETE, evict after
    licences FK-cascade off subscriptions, subscriptions off sites, sites off
    clients. Once the rows are gone there is nothing left to walk, so
    key_hashes_for() returns [] and every invalidate_for_* here returns 0 having
    evicted nothing, while looking like it worked. Call
    licensing_service.key_hashes_for() first, keep the list, then
    invalidate_many() after the delete commits.

  DELIBERATELY NOT AN EVICTION EVENT
    sites.indexed_items changes on every single product upsert. Evicting on it
    would leave the cache cold for every sync batch, which is most of the
    traffic — it would cost more than it saves and buy nothing, because
    indexed_items is not an authorisation input.
    The consequence, and it matters: indexed_items in a cached context is a
    STALE SNAPSHOT. Never enforce a catalogue ceiling against it. Ask the
    database — tenancy_service.has_catalogue_headroom() — which reads the live
    counter. catalogue_limit from the cache is fine (it only moves on a plan
    change, which does evict); the count sitting next to it is not.

FAIL OPEN, ALWAYS — WITH ONE EXPOSURE THAT IS NOT FIXED HERE
------------------------------------------------------------
Every Redis call in this file is wrapped. Redis being down degrades resolution
to the MySQL path it was already doing before this module existed: slower, never
broken. An auth cache that can refuse authentication is a worse liability than
no cache at all. The one place this is NOT a shrug is a failed EVICTION — that
does not degrade to a database lookup, it degrades to a revoked key continuing
to serve — so read/write failures log at WARNING and eviction failures log at
ERROR.

Say the limit of that promise out loud, because it is the difference between a
degraded service and no service: try/except catches Redis FAILING, not Redis
HANGING. redis-py defaults socket_timeout and socket_connect_timeout to None,
i.e. block forever, and backend/app/services/cache_service.py:7 — which owns the
shared client _redis() deliberately reuses — sets neither. A Redis that is
TCP-connected but not answering (a BGSAVE fork stall, the box swapping, a
half-open connection through a NAT or a firewall) therefore parks client.get()
in get() for as long as it likes, on the authentication path, on every request,
until the worker pool is exhausted and the API stops answering altogether.
Connection refused raises immediately and behaves exactly as promised;
unresponsive does not.

The fix is one line and it is in that file, not here:

    redis.Redis(..., socket_timeout=0.25, socket_connect_timeout=0.25)

Building a second client here to get timeouts would be worse than the problem —
double the sockets, drift from REDIS_HOST/REDIS_PORT, and two different opinions
about whether Redis is reachable — and mutating the shared client's pool from
this module would silently change the rate limiter's behaviour too. So it stays
written down here until somebody owns that file.

WHAT IS NOT CACHED
------------------
Denials. A key that resolve_key() refused is never written here, on purpose:
caching a denial makes a freshly re-issued key mysteriously invalid for the next
five minutes, and a merchant who has just been sent a new key is by far the most
likely person to be presenting one that failed a moment ago.

That is the whole of the reason, and it is worth being blunt about what is NOT
part of it. license_key.looks_valid() is a typo filter, not a security control —
license_key.py says so plainly itself. Its checksum is CRC32 of the body, which
is public and recomputable in one line, so anyone can emit an unlimited stream
of keys that pass the cheap gate; each one costs a connection out of the
SQLAlchemy pool (5 + 10 overflow by default) and an index dive on
uq_licences_key_hash, the hottest index in the system. "Only well-formed keys
reach the database" is therefore true and means nothing.

Nothing meters that today either: rate_limiter.redis_enforce() keys on
client_id, and client_id only exists AFTER a licence has resolved, so the
pre-resolution path is entirely unlimited. Before the routers are wired,
something has to sit in front of resolve_key() and count by IP. Not caching
denials is still the right call; it just is not what protects that endpoint, and
this paragraph is what the next person sizing this will read.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Iterable, Optional

from sqlalchemy.orm import Session

from backend.app.services import licensing_service
from backend.app.services.licensing_service import CONTEXT_FIELDS, cache_key_for

logger = logging.getLogger(__name__)


# ── Tunables ─────────────────────────────────────────────────────────────────

# Five minutes. Long enough that a busy store resolves once and then rides the
# cache; short enough that an invalidator nobody wired still self-corrects
# inside a support call rather than inside a billing period.
DEFAULT_TTL_SECONDS = 300

# How long an evicted hash stays poisoned against re-population. See
# _tombstone_key(): this closes the window where a request that read the
# pre-change row is still in flight when the eviction lands and writes its
# now-stale context back afterwards, with a full fresh TTL. That window is the
# width of one resolve_key() call plus whatever the GIL and the network add, so
# a few seconds covers it with room to spare. It is not a lock and does not need
# to be one.
TOMBSTONE_TTL_SECONDS = 15

# One DEL per pipeline round trip is fine for a licence; a client with hundreds
# of sites would otherwise build one enormous command. Chunking keeps a single
# invalidation from monopolising a connection, and keeps a partial failure
# partial.
EVICTION_CHUNK = 500

# Redis being down means EVERY request on the auth path takes the fallback and
# would log a warning. At any real request rate that buries the one line
# explaining the outage under thousands of identical ones. First failure logs
# immediately, then at most one line a minute carrying the suppressed count.
# Eviction failures are never throttled — they are rare, and they are the
# security-relevant ones.
_WARN_INTERVAL_SECONDS = 60
_last_warned_at = 0.0
_warnings_suppressed = 0


# ── Cached context shape ─────────────────────────────────────────────────────
#
# The presence check is licensing_service.CONTEXT_FIELDS, imported above and
# never restated. This module used to carry three tuples of its own —
# _REQUIRED_IDENTITY_FIELDS, _REQUIRED_QUOTA_FIELDS, _REQUIRED_EXPIRY_FIELDS —
# holding thirteen hand-copied key names with nothing binding them to
# _context_from_row(). Their stated justification was that keeping a copy of the
# FULL list here would be the drift to avoid — which was true, and was also
# exactly what those three tuples were, only partial. Now that the definition
# exports its own key set, checking the WHOLE shape is both stricter and cheaper
# to maintain than checking part of it, and the field list exists in one place.
#
# This is still not a substitute for CACHE_VERSION. The version orphans every
# entry written by a previous DEPLOY in one go; this catches the entry that is
# individually corrupt — a truncated write, a value someone SET by hand while
# debugging, a half-populated dict from a caller that built its own. Either of
# those reaching the auth path is a KeyError on a live request or, far worse, a
# .get() returning None that reads as "no limit".
#
# What follows are the fields whose TYPE is also checked, and the names are run
# through _context_subset() so that a rename in _context_from_row() fails on
# these lines, at import, rather than turning every cache write into an ERROR.


def _context_subset(*names: str) -> tuple[str, ...]:
    """Field names that must still exist in licensing_service.CONTEXT_FIELDS.

    Presence is checked against the whole imported set; a TYPE check needs to
    know which fields are strings and which are ints, and that is a genuine
    subset this module has to name. Naming it is safe only if the names are
    bound to the definition, which is what this does — at import, once, so the
    failure is a process that does not start rather than an authorisation path
    that logs.

    ImportError rather than ValueError because that is what the failure IS: this
    module asking a dependency for names it no longer exports.
    """
    unknown = [name for name in names if name not in CONTEXT_FIELDS]
    if unknown:
        raise ImportError(
            "auth_cache validates " + ", ".join(unknown) + " on every cached "
            "context, but licensing_service.CONTEXT_FIELDS no longer contains "
            + ("them" if len(unknown) > 1 else "it") + ". _context_from_row() "
            "changed shape: update the lists here AND bump CACHE_VERSION, or "
            "every entry written by the running deploy outlives the rename. "
            "The context now carries: " + ", ".join(sorted(CONTEXT_FIELDS)) + "."
        )
    return names


# Non-empty strings. key_hash is in here for two reasons: it must be a real
# digest for put()'s cross-check below to mean anything, and a context missing
# it would otherwise skip that check silently rather than be rejected.
_TEXT_FIELDS = _context_subset(
    "client_id",
    "site_id",
    "subscription_id",
    "licence_id",
    "product_code",
    "platform",
    "key_owner",
    "collection_name",
    "status",
    "key_hash",
)

# Ints, and checked as ints rather than for truthiness: a request_limit of 0 is
# a data error worth surfacing as a quota of zero, not something to silently
# reject as "missing" and paper over with a database read.
_INT_FIELDS = _context_subset("request_limit", "catalogue_limit")

# The two expiries. None is a legitimate value here meaning "never expires", so
# there is no type to assert — presence is covered by the CONTEXT_FIELDS check.
# This tuple exists because _seconds_until_expiry() has to know WHICH fields
# carry an expiry, which is a fact about the shape and not a validation rule.
_EXPIRY_FIELDS = _context_subset("licence_expires_at", "subscription_expires_at")

# The only characters _fingerprint() lets through into a log line. Everything a
# genuine key_hash or key_prefix is made of, and nothing that can end a line,
# move a cursor or start an escape sequence.
_SAFE_LOG_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
)


# ── Redis plumbing ───────────────────────────────────────────────────────────

def _warn_throttled(message: str, *args) -> None:
    """WARNING with a per-process rate limit, for the hot-path failures.

    The counter is plain module state with no lock. Under-counting a suppressed
    warning by one during a race is not worth a mutex on the authentication
    path; this is a log throttle, not a ledger.
    """
    global _last_warned_at, _warnings_suppressed

    now = time.monotonic()
    if now - _last_warned_at < _WARN_INTERVAL_SECONDS:
        _warnings_suppressed += 1
        return

    suppressed = _warnings_suppressed
    _warnings_suppressed = 0
    _last_warned_at = now

    if suppressed:
        logger.warning(
            message + " (%d further cache warnings suppressed in the last %ds)",
            *args,
            suppressed,
            _WARN_INTERVAL_SECONDS,
        )
    else:
        logger.warning(message, *args)


def _redis():
    """The stack's existing Redis client, or None when it cannot be had.

    Imported inside the function, and inside the try, exactly as rate_limiter
    does it. cache_service runs `import redis` and constructs the client at
    module import time, so a missing package or an unparseable REDIS_PORT would
    turn a degraded cache into an ImportError that takes the whole app down on
    boot. Authentication has to survive Redis being ABSENT, not merely down.

    Deliberately not a second client. redis-py's Redis object is a thread-safe
    connection pool; a private one here would double the sockets, drift from
    REDIS_HOST/REDIS_PORT the moment someone changes one of them, and give the
    rate limiter and the auth cache two different opinions about whether Redis
    is reachable.
    """
    try:
        from backend.app.services.cache_service import r

        return r
    except Exception as exc:
        _warn_throttled(
            "auth cache: no Redis client available (%s) - every licence "
            "resolution will hit MySQL",
            exc,
        )
        return None


def _tombstone_key(cache_key: str) -> str:
    """The "do not re-populate this hash yet" marker for a cache key.

    Derived from cache_key_for()'s output so it inherits CACHE_VERSION: bumping
    the version has to orphan the tombstones along with the entries they guard,
    or a stale tombstone from the previous shape would suppress writes of the
    new one. Collision-free by construction — cache_key_for() only ever emits a
    key ending in 64 hex characters, so nothing it produces can equal one of
    these.
    """
    return f"{cache_key}:evicted"


# ── Expiry ───────────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    """Naive UTC — the same convention licensing_service._utcnow() uses.

    Must match it exactly. The expiry strings in the context came from MySQL
    TIMESTAMP columns via _iso(), which are naive, and comparing an aware
    datetime against a naive one raises TypeError on the authentication path.
    Not datetime.utcnow(), which is deprecated from 3.12 and this stack is 3.13.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _seconds_until_expiry(context: dict) -> Optional[int]:
    """Seconds until this context stops being true, or None if it never does.

    Takes the EARLIER of the licence and subscription expiries, because either
    one lapsing ends the authorisation: a licence is minted for 365 days while a
    subscription's term is whatever was sold, so they routinely disagree.

    May return zero or negative — the caller decides what that means. put()
    treats it as "do not cache", get() treats it as "this entry is dead".

    An expiry string that is present but unparseable returns 0, i.e. expired.
    Returning None there would read as "no expiry" and keep a licence alive
    forever on the strength of a corrupt field, which is the one result a parse
    error must never produce. Falling through to MySQL costs nothing.
    """
    now = _utcnow()
    soonest: Optional[float] = None

    for field in _EXPIRY_FIELDS:
        raw = context.get(field)
        if not raw:
            continue
        try:
            parsed = datetime.fromisoformat(raw)
        except (TypeError, ValueError):
            logger.warning(
                "auth cache: unparseable %s (%r) in the cached context for "
                "subscription %s - treating the entry as expired",
                field, raw, context.get("subscription_id"),
            )
            return 0
        if parsed.tzinfo is not None:
            # Should not happen (_iso() serialises naive values), but an aware
            # value compared against a naive `now` is a TypeError on the auth
            # path rather than a wrong answer, so normalise instead of trusting.
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)

        remaining = (parsed - now).total_seconds()
        if soonest is None or remaining < soonest:
            soonest = remaining

    return None if soonest is None else int(soonest)


# ── Shape validation ─────────────────────────────────────────────────────────

def _context_is_well_formed(context) -> bool:
    """Does this look like something resolve_key() produced?

    Presence is checked against the WHOLE of CONTEXT_FIELDS, not a subset. A
    field this module does not read itself is still one a caller downstream
    will, and a context missing it is corrupt whether or not the corruption
    happens to be in a field the cache cares about. `in` rather than truthiness,
    because both expiries are legitimately None.

    Returns a bool rather than raising: both callers have somewhere better to
    put the diagnosis, and neither may turn a bad cache entry into a failed
    authentication.
    """
    if not isinstance(context, dict):
        return False

    if not CONTEXT_FIELDS.issubset(context):
        return False

    for field in _TEXT_FIELDS:
        value = context.get(field)
        if not isinstance(value, str) or not value:
            return False

    for field in _INT_FIELDS:
        # bool is an int subclass, so True would sail through an isinstance
        # check while being a nonsense quota. Exclude it explicitly.
        value = context.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            return False

    return True


def _missing_fields(context) -> list[str]:
    """Which CONTEXT_FIELDS keys are absent, for the log line only.

    _context_is_well_formed() answers yes or no on the hot path and must stay
    that cheap. This runs once, on the rejection path, so the ERROR names the
    gap instead of leaving whoever reads it to diff two dicts by hand.
    """
    if not isinstance(context, dict):
        return sorted(CONTEXT_FIELDS)
    return sorted(CONTEXT_FIELDS - set(context))


def _fingerprint(value) -> str:
    """Render a rejected key_hash for a log line without echoing it.

    NEVER %r the value. The one realistic way a bad key_hash reaches this
    module is a caller passing the PLAINTEXT licence key where a hash was
    expected — invalidate(presented_key) from a rotate or logout path that has
    the key in hand — and cache_key_for() exists precisely because that is the
    anticipated mistake. Its own message is carefully worded not to repeat what
    it rejected; a handler that then prints the value undoes that and writes a
    customer's working credential into the application log at ERROR, which
    licensing_service._deny's docstring correctly calls the least
    access-controlled surface in the system.

    Not license_key.mask() either, tempting as it looks. parse_for_logging()
    length-checks only the secret and checksum segments, so the environment and
    product segments pass through at any length and with any bytes — newlines
    included — and mask() splices them into the returned string. Feeding that to
    a log line lets an unauthenticated caller forge whole log ENTRIES, which is
    a strictly worse outcome than a value nobody can read. Eight characters
    through a base62 filter, with the length reported separately, tells an
    operator what shape of thing was passed and nothing else: a real key starts
    'czg_live', a digest is hex, a None says so.
    """
    if not isinstance(value, str):
        return f"<{type(value).__name__}>"
    # An explicit alphabet, not str.isalnum(): isalnum() is Unicode-aware and
    # would pass Devanagari digits, fullwidth latin and a long tail of other
    # things that render unpredictably in a terminal tailing the log.
    head = "".join(c if c in _SAFE_LOG_CHARS else "." for c in value[:8])
    return f"'{head}...' ({len(value)} chars)"


# ── Read ─────────────────────────────────────────────────────────────────────

def get(key_hash: str) -> Optional[dict]:
    """Cached tenant context for a key hash, or None to go and ask MySQL.

    Takes the HASH. cache_key_for() raises if handed anything that is not a
    64-character lowercase digest, which is the guard that stops a caller
    passing the plaintext licence key and writing a customer's working
    credential into a Redis key name — read its docstring before changing this
    call. That ValueError is a programming error and is deliberately NOT caught:
    it fires on the first call in any environment, rather than intermittently in
    production.

    Every other failure returns None, which means "cache miss", which means the
    caller resolves against the database exactly as it did before this module
    existed.
    """
    cache_key = cache_key_for(key_hash)

    client = _redis()
    if client is None:
        return None

    try:
        raw = client.get(cache_key)
    except Exception as exc:
        _warn_throttled("auth cache: read failed (%s) - falling back to MySQL", exc)
        return None

    if raw is None:
        return None

    try:
        context = json.loads(raw)
    except Exception as exc:
        # A value that will not parse will not parse on the next request either,
        # so drop it rather than leaving every caller to pay the same failure
        # until the TTL runs out.
        logger.warning(
            "auth cache: corrupt entry at %s (%s) - dropping it", cache_key, exc
        )
        _delete_quietly(client, cache_key)
        return None

    if not _context_is_well_formed(context):
        logger.warning(
            "auth cache: the entry at %s does not match the expected context "
            "shape (missing: %s) - dropping it. If resolve_key()'s context "
            "changed, bump licensing_service.CACHE_VERSION so the whole "
            "population is orphaned at once instead of one entry at a time.",
            cache_key, _missing_fields(context) or "nothing, so a value has the "
            "wrong type - check request_limit and catalogue_limit",
        )
        _delete_quietly(client, cache_key)
        return None

    # The read side of the invariant put() enforces: the entry is named after
    # the hash we asked for, and the context inside it must agree. put() already
    # refuses to write a mismatch, so this catches the entry it did not write —
    # one SET by hand while debugging, one left behind by an older writer, one
    # restored from a dump. Serving it would authenticate this caller as
    # whichever tenant the context actually describes, and every other guard in
    # this function would pass it: it parses, it is well formed, and it is not
    # expired. It is simply somebody else's.
    if context.get("key_hash") != key_hash:
        logger.error(
            "auth cache: the entry at %s holds a context for a DIFFERENT "
            "licence (%s, subscription %s) - dropping it and resolving against "
            "MySQL. Serving it would have authenticated one tenant as another. "
            "Nothing in this module writes such an entry; find what did.",
            cache_key, _fingerprint(context.get("key_hash")),
            context.get("subscription_id"),
        )
        _delete_quietly(client, cache_key)
        return None

    # Re-check expiry on every read even though put() clamps the TTL to it.
    # Belt and braces on purpose: the clamp depends on this process's clock at
    # write time, an expires_at can be moved EARLIER by an admin edit that
    # forgot to evict, and a licence quietly outliving its own expiry date is
    # not something anyone notices until an audit.
    remaining = _seconds_until_expiry(context)
    if remaining is not None and remaining <= 0:
        logger.info(
            "auth cache: the cached context for subscription %s is past its "
            "expiry - dropping it and resolving against MySQL",
            context.get("subscription_id"),
        )
        _delete_quietly(client, cache_key)
        return None

    return context


def _delete_quietly(client, cache_key: str) -> None:
    """Best-effort DEL for the housekeeping paths inside get().

    Separate from invalidate() because the severity is different: these deletes
    tidy up an entry that has ALREADY been rejected, so failing to remove it
    costs one wasted round trip per request until the TTL expires and grants
    nobody any authority. invalidate() failing is a revoked key that keeps
    working, and that logs at ERROR.
    """
    try:
        client.delete(cache_key)
    except Exception as exc:
        _warn_throttled("auth cache: could not drop a bad entry (%s)", exc)


# ── Write ────────────────────────────────────────────────────────────────────

def put(key_hash: str, context: dict, ttl: int = DEFAULT_TTL_SECONDS) -> bool:
    """Cache a resolved context. Returns True only if it was actually stored.

    Refuses rather than raises on a malformed context: this runs immediately
    after a SUCCESSFUL resolve_key() on a live request, and the cache turning a
    working authentication into a 500 inverts the entire point of the module. It
    logs at ERROR instead, because a context that fails the shape check means
    resolve_key() changed and CACHE_VERSION was not bumped.

    The stored TTL is the SMALLER of *ttl* and the time left on the licence or
    subscription. Without the clamp, resolving a licence four minutes before it
    expires would keep it authorised for a minute past it — there is no event to
    hook when a timestamp simply passes, so this clamp and the re-check in get()
    are the only things standing between an expiry date and a licence that
    ignores it.
    """
    if ttl <= 0:
        # A programming error, and the only one this function raises on: it
        # lives in the caller's constant, not in the request, so it fails on the
        # first call in any environment rather than intermittently in production.
        raise ValueError(f"auth cache: ttl must be positive, got {ttl}.")

    cache_key = cache_key_for(key_hash)

    if not _context_is_well_formed(context):
        logger.error(
            "auth cache: refusing to cache a context that does not match the "
            "expected shape (subscription %s, missing: %s). Nothing breaks - "
            "every request will resolve against MySQL - but check whether "
            "_context_from_row() changed without a CACHE_VERSION bump.",
            context.get("subscription_id") if isinstance(context, dict) else None,
            _missing_fields(context) or "nothing, so a value has the wrong type",
        )
        return False

    # THE ENTRY IS NAMED AFTER key_hash; the context says which licence it
    # actually describes. If those two disagree, this write files tenant A's
    # complete authorisation context — client_id, site_id, subscription_id,
    # collection_name, request_limit, catalogue_limit — under tenant B's hash,
    # and get() then hands it to B. That is a cross-tenant authentication
    # bypass, serving A's Qdrant collection to B for the full TTL, with no
    # failure signal anywhere: the cache works perfectly, it is just answering
    # the wrong question.
    #
    # The realistic way in is a transposition, not malice: middleware holding a
    # request-scoped hash variable while resolving a different key, a retry loop
    # reusing a stale local, a test helper wiring two fixtures together. Nobody
    # has written that middleware yet, which is the reason to add the guard now
    # rather than after it exists. _context_from_row() puts key_hash in the
    # context explicitly and its comment says it is there so the cache layer can
    # name its own entry without re-hashing — so the data was already present
    # and this was one comparison away the whole time.
    #
    # Compared raw. Both sides are supposed to be the same lowercase digest, so
    # anything else — a stray space, a different case — is itself the bug, and
    # refusing costs a cache miss while accepting costs an authorisation.
    #
    # Neither value goes into the line whole, and neither goes through
    # cache_key_for(): the context's key_hash has only been checked as a
    # non-empty string, so cache_key_for() could raise on it, and an exception
    # escaping put() would turn a SUCCESSFUL authentication into a 500 — the one
    # thing this module must never do. _fingerprint() cannot raise and cannot
    # echo a credential.
    if context["key_hash"] != key_hash:
        logger.error(
            "auth cache: REFUSING a context for subscription %s because it "
            "names a different licence than the hash it would be filed under "
            "(entry %s, context names %s). Caching it would authenticate one "
            "tenant as another for the whole TTL, with A's collection_name and "
            "A's limits. Nothing breaks - resolution falls through to MySQL - "
            "but the caller has transposed a key_hash and needs fixing.",
            context.get("subscription_id"),
            cache_key, _fingerprint(context["key_hash"]),
        )
        return False

    remaining = _seconds_until_expiry(context)
    if remaining is not None:
        if remaining <= 0:
            # resolve_key() gates on both expiries, so reaching here means one
            # of them passed between that gate and this call, or a clock moved.
            # Either way, caching it would extend an authorisation that has just
            # ended.
            logger.info(
                "auth cache: not caching subscription %s - its licence or "
                "subscription expired between resolution and the cache write",
                context.get("subscription_id"),
            )
            return False
        ttl = min(ttl, remaining)

    try:
        payload = json.dumps(context)
    except Exception as exc:
        # _context_from_row() promises str/int/bool/None only. A datetime or a
        # Decimal getting in there raises here, on the first cache write in
        # production and in no unit test that calls resolve_key() directly.
        logger.error(
            "auth cache: the context for subscription %s is not "
            "JSON-serialisable (%s) - not cached",
            context.get("subscription_id"), exc,
        )
        return False

    client = _redis()
    if client is None:
        return False

    tombstone = _tombstone_key(cache_key)
    try:
        # Write, then ask whether an eviction landed while we were writing. The
        # order looks backwards and is not: checking the tombstone FIRST and
        # then writing leaves the same race in the other direction, and this way
        # costs one round trip instead of two. An entry that loses this race is
        # readable for the length of a single pipeline before the DEL removes
        # it, versus for the whole TTL if the check is skipped entirely.
        pipe = client.pipeline()
        pipe.setex(cache_key, int(ttl), payload)
        pipe.get(tombstone)
        result = pipe.execute()

        if result[1] is not None:
            client.delete(cache_key)
            logger.info(
                "auth cache: discarded a context for subscription %s that was "
                "written over a live eviction",
                context.get("subscription_id"),
            )
            return False
    except Exception as exc:
        _warn_throttled("auth cache: write failed (%s) - not cached", exc)
        return False

    return True


# ── Eviction ─────────────────────────────────────────────────────────────────

def invalidate(key_hash: str) -> bool:
    """Evict one key hash. True when Redis confirmed it processed the delete.

    True does NOT mean an entry existed - deleting a hash that was never cached
    is a no-op and still returns True. The distinction that matters is "reached
    Redis" versus "did not", because only the second leaves a possibly stale
    entry serving.
    """
    return invalidate_many([key_hash]) == 1


def invalidate_many(key_hashes: Iterable[str]) -> int:
    """Evict a batch. Returns how many hashes Redis confirmed.

    A caller comparing the return value against len(hashes) is the only signal
    that an eviction was incomplete, which is why this counts confirmations
    rather than deletions.

    One bad hash does not abort the batch. An invalidation triggered by
    "deactivate this client" can carry hundreds of hashes, and refusing to evict
    the other 499 because one row holds a malformed key_hash would turn a data
    error into a security one.
    """
    seen: set[str] = set()
    cache_keys: list[str] = []
    for key_hash in key_hashes:
        try:
            cache_key = cache_key_for(key_hash)
        except ValueError as exc:
            # _fingerprint(), never %r. The one realistic way to reach this
            # branch is a caller handing us a plaintext licence key instead of
            # a hash, and printing it here would write a customer's working
            # credential to the log at ERROR — cache_key_for() guards against
            # exactly that mistake and words its own message not to echo the
            # value, so echoing it in the handler undoes the guard. It also
            # bounds the line: an attacker-supplied 500-character string would
            # otherwise be reprinted in full, once per hash.
            logger.error(
                "auth cache: cannot evict %s - %s. That hash stays cached until "
                "its TTL expires.",
                _fingerprint(key_hash), exc,
            )
            continue
        if cache_key in seen:
            continue
        seen.add(cache_key)
        cache_keys.append(cache_key)

    if not cache_keys:
        return 0

    client = _redis()
    if client is None:
        logger.error(
            "auth cache: Redis unreachable while evicting %d licence context(s) "
            "- any cached entry among them keeps serving for up to %ds. This is "
            "the one cache failure that is not harmless.",
            len(cache_keys), DEFAULT_TTL_SECONDS,
        )
        return 0

    confirmed = 0
    for start in range(0, len(cache_keys), EVICTION_CHUNK):
        chunk = cache_keys[start:start + EVICTION_CHUNK]
        try:
            pipe = client.pipeline()
            pipe.delete(*chunk)
            # Poison each hash against re-population by a resolve() that read
            # the pre-change row and has not written its result back yet. See
            # TOMBSTONE_TTL_SECONDS.
            for cache_key in chunk:
                pipe.setex(_tombstone_key(cache_key), TOMBSTONE_TTL_SECONDS, "1")
            pipe.execute()
            confirmed += len(chunk)
        except Exception as exc:
            logger.error(
                "auth cache: eviction failed for %d licence context(s) (%s) - "
                "they keep serving for up to %ds. Retry the invalidation; it is "
                "idempotent.",
                len(chunk), exc, DEFAULT_TTL_SECONDS,
            )

    return confirmed


def _invalidate_by_selector(db: Session, selector: str, value: str) -> int:
    """Walk down to every affected licence, then evict each one's cached context.

    The walk is licensing_service.key_hashes_for() and is deliberately not
    re-implemented here. That function exists FOR this module (its docstring
    says so), licences and subscriptions are owned over there, and a second copy
    of the four join paths in this file is one schema change away from evicting
    the wrong set - silently, because eviction has no observable success case to
    fail.

    Note what key_hashes_for() deliberately does not do: filter on is_active.
    The hashes most urgently needing eviction are the ones that were just
    deactivated, so filtering them out would delete every cache entry except the
    dangerous one, while looking like it worked.

    Returns 0 on a database failure rather than raising. The mutation that
    triggered this has already committed; raising here turns a successful revoke
    into a 500, the operator retries, and the second attempt hits an
    already-suspended subscription and does nothing. An ERROR line naming the
    row is more actionable than an exception that undoes nothing.
    """
    if not value:
        raise ValueError(
            f"auth cache: {selector} is required. Without it there is nothing "
            f"to walk down from, and an eviction that quietly evicts nothing is "
            f"how a suspended account keeps serving."
        )

    try:
        hashes = licensing_service.key_hashes_for(db, **{selector: value})
    except Exception as exc:
        logger.error(
            "auth cache: could not list the licences for %s=%s (%s) - NOTHING "
            "was evicted. Any cached context underneath it keeps serving for up "
            "to %ds.",
            selector, value, exc, DEFAULT_TTL_SECONDS,
        )
        return 0

    if not hashes:
        # Routine and not an error: a subscription can exist with no licence
        # ever issued, and a site or client can be deactivated before anyone
        # minted a key for it. Logged anyway, because it is also exactly what a
        # delete that cascaded before the eviction looks like - see the DELETES
        # note in the module docstring.
        logger.info(
            "auth cache: no licences under %s=%s, nothing to evict", selector, value
        )
        return 0

    evicted = invalidate_many(hashes)
    if evicted != len(hashes):
        logger.error(
            "auth cache: evicted only %d of %d licence context(s) under %s=%s",
            evicted, len(hashes), selector, value,
        )
    return evicted


def invalidate_for_subscription(db: Session, subscription_id: str) -> int:
    """Every cached context for one subscription's licences.

    Call after: a status change (suspend, cancel, reactivate, back to trial), a
    plan change, or a change to expires_at. The plan case is the easiest to
    forget and the expensive one - request_limit rides in the cached context, so
    an upgrade that does not land keeps enforcing the smaller quota the merchant
    just stopped paying for.
    """
    return _invalidate_by_selector(db, "subscription_id", subscription_id)


def invalidate_for_site(db: Session, site_id: str) -> int:
    """Every cached context for every subscription on one site.

    Call after: deactivating the site, an index_plan change, an environment
    flip, or an edit to domain / store_name / collection_name. One store can
    hold up to five subscriptions (three Magento modules, two WooCommerce
    plugins) and each has its own licences, so evicting the one licence you
    happen to be looking at reaches at most a fifth of what changed.
    """
    return _invalidate_by_selector(db, "site_id", site_id)


def invalidate_for_client(db: Session, client_id: str) -> int:
    """Every cached context under one client, across all of their sites.

    Call after deactivating a client. This is the invalidator the brief singles
    out and the one most likely to be missed, because clients.is_active is a
    single-column UPDATE that looks nothing like a licence operation - and it
    has the widest blast radius in the schema: one client, many sites, many
    subscriptions per site, many licences per subscription. Skip it and every
    key underneath a deactivated account keeps authorising for the full TTL.
    """
    return _invalidate_by_selector(db, "client_id", client_id)
