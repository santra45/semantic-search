"""
Subscriptions, licences, and the one function that turns a presented key into a
tenant: resolve_key().

WHAT THIS MODULE IS FOR
-----------------------
Three Magento modules call the identical backend endpoints
(/magento/chatbot/agent/sync/*). The route a request arrived on therefore
cannot tell you which product made the call, which is why per-product usage and
billing were impossible in v1. The licence key is the only thing on the wire
that differs between them, so resolving that key to exactly one subscription is
the single mechanism that recovers product identity. Everything else in the
billing rewrite — usage_events.product_code, the per-module quota, the
per-product cost dashboard — is downstream of this file being right.

THE TWO SCOPES, RESTATED BECAUSE THIS FILE STRADDLES THEM
---------------------------------------------------------
Every module on one store shares ONE Qdrant collection, so:

  * catalogue size is consumed once per STORE  -> sites.catalogue_limit
  * licences and request quota are per MODULE  -> subscriptions.request_limit

The context resolve_key() returns carries both, from their own levels. It does
NOT derive one from the other, and nothing downstream should either: a store's
ceiling belongs to its index_plan, and cancelling a module must never move it.

THE PRODUCT SEGMENT IN THE KEY IS NOT A CREDENTIAL
--------------------------------------------------
Read the warning block at the top of license_key.py. This module never calls
parse_for_logging() for anything but a log line, never branches on
`unverified_product`, and never lets a caller supply a product code alongside a
key. product_code in the returned context comes from the joined subscription
row and from nowhere else. If a future change wants the product before the hash
lookup, it wants the hash lookup.

WHY THE LIVENESS GATES ARE EVALUATED IN PYTHON AND NOT IN THE WHERE CLAUSE
--------------------------------------------------------------------------
Folding `AND l.is_active = 1 AND s.status IN (...) AND si.is_active = 1 ...`
into the SELECT is one fewer round of Python and is exactly what v1 did in
get_client_license(). It also collapses five distinct answers into one: "no
row". A merchant whose site was deactivated, whose subscription lapsed, and who
pasted a key from the wrong store all get the same silence, and support cannot
tell them apart without opening a MySQL client. So the SELECT matches on
key_hash alone and each gate is checked separately with its own log line. The
extra cost is a handful of comparisons on an already-fetched row.

THE CACHE CONTRACT THIS MODULE OWES THE CACHE LAYER
---------------------------------------------------
Another module owns Redis. Three things here exist purely so it and this file
cannot drift apart:

  * cache_key_for(key_hash) — the only place a cache key string is spelled, so
    the writer and the invalidator cannot disagree about naming.
  * CONTEXT_FIELDS — the key set _context_from_row() actually produces, DERIVED
    from that function rather than typed out beside it. Three modules used to
    carry hand-copied versions of that list (usage_service._REQUIRED_CTX_FIELDS
    and auth_cache's three _REQUIRED_* tuples) with nothing binding any of them
    to the definition, and the two failure modes were asymmetric: auth_cache
    rejects a wrong-shaped context loudly, while usage_service.record() caught
    its own ValueError, logged it and returned False — losing the billing row
    while the shopper still got their answer and the request still cost money.
    That is the wp_product_qa failure (a Python allowlist and a schema drifting
    apart, eating the INSERT) reproduced one layer up. Import the set; do not
    restate it.
  * key_hashes_for(...) — every mutation that can revoke authority is reachable
    from a licence, a subscription, a site or a client, and the cached entry is
    keyed by key_hash. Without a way to go the other way, an invalidation hook
    on "client deactivated" has nothing to delete and a dead account keeps
    serving until the TTL expires. The TTL is a backstop, not the mechanism.
    tenancy_service.set_site_active() and set_client_active() exist so those
    two events have a producer at all; they call straight into this function.

Every mutating function below returns a "key_hashes" list — the same field
name, always a list, present even when it is empty — so that a caller who
writes one eviction helper is correct for all of them. That is not decoration:
the previous shape had issue_licence returning "superseded_key_hashes" and
create_subscription returning nothing, so `evict(result.get("key_hashes", []))`
was silently a no-op on the two calls where a stale entry matters most.

WHICH EXCEPTION MEANS WHAT
--------------------------
The same two-exception contract tenancy_service's docstring sets out, restated
here because it is the contract a ROUTER implements and the router author will
have read whichever of the two files they opened first:

  * ValueError  — the caller supplied something unusable: an unknown plan, an
    unknown status, a valid_days already in the past, a plan and status that
    contradict each other. Every message is written to be read by whoever typed
    it, because onboarding renders str(e) straight at a customer.
  * LookupError — an id that names no row, or a products table that disagrees
    with catalog.py. An internal inconsistency, never a form error, and
    deliberately NOT a subclass of ValueError so a router's customer-facing
    `except ValueError: return error=str(e)` cannot render a raw uuid, or an
    instruction to reseed a database table, into a signup form.

The getters (get_subscription, get_subscription_by_id) return None for "no such
row" and raise nothing; that is what a getter is for. resolve_key() also
returns None rather than raising, and its own docstring says why.

WHO OWNS THE TRANSACTION
------------------------
Every function here that writes COMMITS before it returns. The caller does not
own the boundary and must not assume one. The consequence is worth stating
plainly rather than discovering: an onboarding flow calling
find_or_create_client -> find_or_create_site -> create_subscription ->
issue_licence performs four independent commits, and a failure in the fourth
leaves the first three durable, with no single rollback that undoes them. The
compensating cleanup for that half-built tenant is
set_subscription_status(id, 'cancelled') and tenancy_service.set_site_active(
site_id, False) — neither of which deletes anything, deliberately, because a
half-built tenant an operator can finish is worth more than one that was
silently unwound.

A refusal is NOT a failed statement, and no refusal path here calls
db.rollback(). The only rollbacks below are after an IntegrityError, where the
session's transaction is genuinely dead and the next execute() would raise
PendingRollbackError. Rolling back on a refusal discards whatever else the
caller had pending on that Session, which is a far harder bug to find than the
tidiness it was reaching for.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.services import catalog, license_key

logger = logging.getLogger(__name__)


# ── Vocabulary ───────────────────────────────────────────────────────────────
#
# Not enums in MySQL, on purpose (see schema_v2.py) — the vocabulary lives here
# and a value the database has never seen degrades to an odd-looking row rather
# than a lost INSERT. That puts the burden of validation on this module, which
# is why every writer below checks against these tuples before it writes.

SUBSCRIPTION_STATUSES = ("trial", "active", "suspended", "cancelled")

# The statuses resolve_key() will authorise. Deliberately an ALLOW-list even
# though the brief phrases the rule as "reject cancelled or suspended". A
# deny-list on an auth path fails open: one hand-written UPDATE setting
# 'canceled' (one l), or a future status nobody remembered to add to the deny
# list, and a cancelled account keeps serving. An allow-list fails closed —
# the merchant sees an authorisation error, which generates a ticket, which is
# how you find out. The reverse never generates anything.
RESOLVABLE_STATUSES = ("trial", "active")

# sites.environment. Distinct from license_key.ENVIRONMENTS ('live'/'test') and
# routinely confused with it, hence the explicit map below rather than passing
# either string straight through to mint().
SITE_ENVIRONMENTS = ("development", "production")

KEY_ENVIRONMENT_FOR_SITE = {
    "production": "live",
    "development": "test",
}

DEFAULT_LICENCE_DAYS = 365


# ── Cache key naming ─────────────────────────────────────────────────────────
#
# The version segment is not decoration. The cached value is the whole context
# dict spanning client -> site -> subscription -> licence; if the shape of that
# dict changes, every entry written by the previous deploy deserialises into
# code expecting the new shape and produces a KeyError on the hot path — or
# worse, a missing key that reads as None and silently disables a quota check.
# Bumping CACHE_VERSION orphans the entire old population instantly, with no
# SCAN, no DEL, and no window where both shapes are live.
#
# Bump it whenever _context_from_row() gains, loses or renames a field.
#
# CACHE_VERSION and CONTEXT_FIELDS solve two halves of one problem and neither
# covers the other's half. CACHE_VERSION orphans entries written by a previous
# DEPLOY; CONTEXT_FIELDS (defined next to _context_from_row, because it is
# derived from it) stops the field list itself being restated in three other
# modules. A rename needs both: bump the version so no old entry survives, and
# let the derived set carry the new name to every consumer that imports it.
CACHE_PREFIX = "licence"
CACHE_VERSION = "v2"


def cache_key_for(key_hash: str) -> str:
    """The Redis key under which a resolved context is cached.

    Takes the HASH, never the plaintext key, and refuses anything that is not a
    64-character lowercase hex digest. That guard is the point of the function
    existing at all: the natural mistake is `cache_key_for(presented_key)`,
    which would put a customer's working credential into Redis as part of a key
    name, visible to anyone with MONITOR, KEYS or a memory dump — reintroducing
    exactly the exposure that storing only hashes was meant to end. It is also
    a bug that would never show up in testing, because the cache would still
    work perfectly; the entries would just be named after the secret.

    Lowercase is enforced too. licences.key_hash is ascii_bin and
    license_key.hash_key() returns hexdigest(), so an uppercased hash misses in
    both MySQL and Redis, and the two would miss on different keys.
    """
    value = (key_hash or "").strip()
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(
            "cache_key_for() expects a 64-char lowercase SHA-256 hex digest, "
            "not a licence key. Hash it with license_key.hash_key() first — "
            "never name a cache entry after the plaintext credential."
        )
    return f"{CACHE_PREFIX}:{CACHE_VERSION}:{value}"


def key_hashes_for(
    db: Session,
    *,
    licence_id: Optional[str] = None,
    subscription_id: Optional[str] = None,
    site_id: Optional[str] = None,
    client_id: Optional[str] = None,
) -> list[str]:
    """Every key_hash whose cached context depends on the named row.

    The cached entry spans four levels, so authority can be withdrawn at any of
    them: revoke a licence, suspend a subscription, deactivate a site,
    deactivate a client. Only the first of those knows its own key_hash; the
    other three have to walk down to the licences to find out what to evict.
    This is that walk, and it exists here rather than in the cache module
    because licences and subscriptions are owned here.

    Deliberately does NOT filter on is_active, and this is the one place the
    obvious reading of "every ACTIVE licence under this scope" is wrong. The
    hashes you most need to evict are the ones that were just deactivated:
    filtering on is_active deletes every cache entry EXCEPT the dangerous one,
    and does it while looking like the invalidation ran. Revoke a licence, then
    evict, and an is_active filter hands back a list that no longer contains
    the key you revoked — which keeps working for the full TTL. Extra hashes in
    the list cost one wasted DEL each; a missing one costs an authorisation.

    Exactly one selector, so a caller that passes none does not silently get
    every hash in the database and evict the whole cache.

    Raises ValueError on nought or several selectors — a caller mistake, not an
    id that names no row, so it is not the LookupError case. A selector naming
    a row that does not exist returns [], which is correct: nothing is cached
    under a client that does not exist, and there is no eviction to make.
    """
    selectors = {
        "licence_id": licence_id,
        "subscription_id": subscription_id,
        "site_id": site_id,
        "client_id": client_id,
    }
    given = [name for name, value in selectors.items() if value]
    if len(given) != 1:
        raise ValueError(
            f"key_hashes_for() takes exactly one selector, got {given or 'none'}."
        )

    if licence_id:
        sql = "SELECT key_hash FROM licences WHERE id = :value"
    elif subscription_id:
        sql = "SELECT key_hash FROM licences WHERE subscription_id = :value"
    elif site_id:
        sql = """
            SELECT l.key_hash
            FROM licences l
            JOIN subscriptions s ON s.id = l.subscription_id
            WHERE s.site_id = :value
        """
    else:
        sql = """
            SELECT l.key_hash
            FROM licences l
            JOIN subscriptions s ON s.id = l.subscription_id
            JOIN sites si       ON si.id = s.site_id
            WHERE si.client_id = :value
        """

    rows = db.execute(text(sql), {"value": selectors[given[0]]}).fetchall()
    return [row.key_hash for row in rows]


# ── Time ─────────────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    """Naive UTC, which is what every gate in this file compares against.

    Naive and not tz-aware because MySQL TIMESTAMP columns come back naive:
    handing pymysql an aware datetime writes a value offset from every naive
    one already in the table, and the comparison that decides whether a licence
    has expired would then be off by the server's UTC offset — silently, and
    only for rows written after the change.

    Not datetime.utcnow(), which this codebase uses elsewhere, because it is
    deprecated from 3.12 and this stack runs 3.13. Same value, no warning on
    every authenticated request.

    One thing to know before adding a gate: issued_at, started_at and created_at
    are written by MySQL's own DEFAULT CURRENT_TIMESTAMP, i.e. by the database
    server's clock in its session time zone, not by this one. They are fine for
    display and useless as gates. Every column this module actually gates on —
    licences.expires_at, subscriptions.expires_at — is written from Python, so
    both sides of the comparison come from the same clock.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _as_naive_utc(value: datetime, field: str) -> datetime:
    """Force a caller-supplied datetime into the frame the columns are stored in.

    pymysql formats a datetime with strftime and drops the tzinfo, so an aware
    datetime in, say, +05:30 is WRITTEN as its local wall clock and then
    compared against _utcnow() — a five and a half hour shift, applied
    silently, to a value that decides whether a subscription still authorises.
    Converting here means a caller who does the tz-aware thing gets the right
    answer instead of a subtly wrong one.

    usage_service carries a byte-similar copy of this. Not shared, because
    usage_service imports this module for CONTEXT_FIELDS and importing back
    would be a cycle; and this is five lines of coercion with no policy in it,
    which is the only kind of duplication worth accepting.
    """
    if not isinstance(value, datetime):
        raise ValueError(f"{field} must be a datetime, got {type(value).__name__}.")
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _iso(value) -> Optional[str]:
    """Datetime -> ISO string, for a context dict that has to survive JSON.

    The resolved context is handed to the cache layer, which json.dumps() it.
    A datetime raises TypeError there, and it would raise it on the first cache
    write in production rather than in any unit test that only calls
    resolve_key() directly. So the context carries strings, and every consumer
    gets the identical shape whether it came from Redis or from MySQL.
    """
    return value.isoformat() if isinstance(value, datetime) else None


# ── Environment and key ownership ────────────────────────────────────────────

def key_owner_for(environment: Optional[str]) -> str:
    """Whose API keys paid for a call made under this environment.

    'client' on production, 'czargroup' everywhere else. This is not a cosmetic
    label: in production the merchant supplies their own LLM and embedding keys,
    so usage_events.total_cost is what THEY spent and is not Czargroup's cost of
    goods; on a development site the calls run on Czargroup's keys and the same
    column is COGS. Summing total_cost across both values produces a number that
    means nothing, which is why usage_events carries key_owner on every row.

    Anything unrecognised resolves to 'czargroup' on purpose. The failure is
    then "we booked a client's spend as our own cost", which understates margin
    and shows up in a reconciliation; the other direction invoices a customer
    for tokens they never bought.
    """
    return "client" if environment == "production" else "czargroup"


def _key_environment(environment: str) -> str:
    """Map whatever vocabulary the caller has onto license_key.ENVIRONMENTS.

    There are two environment vocabularies in this system and they are one
    substitution apart: sites.environment is development|production, and a
    licence key's second segment is test|live. Callers overwhelmingly have the
    former, because they just read a site row, and mint() only accepts the
    latter — it raises ValueError on 'production', which is a confusing failure
    at the end of a signup flow. Accepting both here and translating means the
    obvious call works and there is exactly one place the mapping is written.
    """
    value = (environment or "").strip().lower()
    if value in KEY_ENVIRONMENT_FOR_SITE:
        return KEY_ENVIRONMENT_FOR_SITE[value]
    if value in license_key.ENVIRONMENTS:
        return value
    raise ValueError(
        f"Unknown environment '{environment}'. Expected one of "
        f"{SITE_ENVIRONMENTS} or {license_key.ENVIRONMENTS}."
    )


# ── Subscriptions ────────────────────────────────────────────────────────────

def _subscription_dict(row) -> dict:
    return {
        "id": row.id,
        "site_id": row.site_id,
        "product_code": row.product_code,
        "status": row.status,
        "plan": row.plan,
        "request_limit": int(row.request_limit),
        "started_at": _iso(row.started_at),
        "expires_at": _iso(row.expires_at),
    }


_SUBSCRIPTION_COLUMNS = """
    id, site_id, product_code, status, plan, request_limit, started_at, expires_at
"""


def get_subscription(db: Session, site_id: str, product_code: str) -> Optional[dict]:
    """The subscription for one (site, product), or None.

    This pair is the subscription's real identity — uq_subscriptions_site_product
    — and the id is a surrogate. Lookups from onboarding and from the operator
    console come in on the pair.
    """
    row = db.execute(
        text(f"""
            SELECT {_SUBSCRIPTION_COLUMNS}
            FROM subscriptions
            WHERE site_id = :site_id AND product_code = :product_code
        """),
        {"site_id": site_id, "product_code": product_code},
    ).fetchone()
    return _subscription_dict(row) if row else None


def get_subscription_by_id(db: Session, subscription_id: str) -> Optional[dict]:
    row = db.execute(
        text(f"""
            SELECT {_SUBSCRIPTION_COLUMNS}
            FROM subscriptions
            WHERE id = :id
        """),
        {"id": subscription_id},
    ).fetchone()
    return _subscription_dict(row) if row else None


# ── The plan/status pair ─────────────────────────────────────────────────────
#
# 'trial' exists twice in the vocabulary — as a SUBSCRIPTION_STATUSES value and
# as a MODULE_PLANS rung — and the two used to be validated independently and
# never against each other. That let create_subscription(plan='pro',
# status='trial') write a trial carrying Pro's 500,000-request allowance, and it
# let the ordinary support action of flipping a lapsed customer back to
# status='trial' leave them on the plan they stopped paying for. The mirror
# image, plan='trial' with status='active', puts a paying customer on 250
# requests a month and cuts them off on day one.
#
# Two independent representations of one concept is exactly what splitting
# INDEX_PLANS from MODULE_PLANS was meant to end, so the pair is now enforced:
#
#     status == 'trial'   <=>  plan == 'trial'
#     status == 'active'   =>  plan != 'trial'
#     status in ('suspended', 'cancelled') — plan is whatever it was when the
#         subscription stopped serving. RESOLVABLE_STATUSES excludes both, so no
#         allowance is enforced against either, and forcing a cancelled trial off
#         its trial plan would rewrite history to no purpose.
#
# create_subscription REFUSES an incoherent pair: the caller named both values
# and one of them is a mistake worth surfacing. The two setters COERCE the field
# the caller did not name and log it, because a setter handed one value and
# required to restate the other is precisely how the two drift apart again.


def _assert_plan_status_coherent(plan: str, status: str) -> None:
    """Refuse a (plan, status) pair that describes two different products."""
    if status == "trial" and plan != catalog.TRIAL_MODULE_PLAN:
        raise ValueError(
            f"status='trial' cannot carry plan='{plan}' — that would grant "
            f"{catalog.request_limit_for(plan):,} requests to a trial. Open it "
            f"as status='trial' with the default plan, or as status='active' "
            f"with the plan they bought."
        )
    if status == "active" and plan == catalog.TRIAL_MODULE_PLAN:
        raise ValueError(
            f"plan='{catalog.TRIAL_MODULE_PLAN}' cannot be status='active' — "
            f"that bills a paying customer against "
            f"{catalog.request_limit_for(catalog.TRIAL_MODULE_PLAN):,} requests "
            f"a month. Name the plan they bought: "
            + ", ".join(catalog.MODULE_PLAN_ORDER)
            + "."
        )


def _plan_for_status(status: str, current_plan: str) -> str:
    """The plan a subscription must hold once it moves to *status*."""
    if status == "trial":
        # Returning someone to trial without moving the plan is the support
        # action that leaves a lapsed Pro customer on 500,000 requests.
        return catalog.TRIAL_MODULE_PLAN
    if status == "active" and current_plan == catalog.TRIAL_MODULE_PLAN:
        # There is no defensible default here. DEFAULT_MODULE_PLAN is the radio
        # button the pricing page comes up with checked, not evidence that
        # anybody bought anything, and guessing it would either give away
        # Starter or cut a Pro customer to a tenth of what they paid for.
        raise ValueError(
            "Activating a trial needs the plan they bought. Call "
            "set_subscription_plan(subscription_id, plan) instead — it moves "
            "the status to 'active' as part of the same change."
        )
    return current_plan


def _status_for_plan(plan: str, current_status: str) -> str:
    """The status a subscription must hold once it moves to *plan*."""
    if plan == catalog.TRIAL_MODULE_PLAN:
        return "trial"
    if current_status == "trial":
        # Upgrade in place — the same key keeps working, which is what
        # MODULE_PLANS['trial'] promises on the pricing page. Leaving the status
        # at 'trial' while the plan says Pro is the pair this rule exists to
        # forbid, and it is the common path, not an exotic one.
        return "active"
    # 'suspended' and 'cancelled' survive a plan change untouched: an operator
    # correcting the plan on a suspended account is not un-suspending it, and
    # reactivating on a plan edit would put a non-paying store back online.
    return current_status


def create_subscription(
    db: Session,
    site_id: str,
    product_code: str,
    plan: str = catalog.TRIAL_MODULE_PLAN,
    status: str = "trial",
    expires_at: Optional[datetime] = None,
) -> dict:
    """Open a subscription for (site, product), or return the one already open.

    FIND-OR-CREATE, NEVER UPSERT. A second call for the same pair returns the
    existing row untouched and reports created=False. It deliberately does not
    apply the plan or status it was passed: this is the function onboarding
    calls, onboarding gets re-run (a merchant reinstalls, bookmarks the page,
    double-clicks Submit), and its default status is 'trial'. An upsert here
    would silently drop a paying customer from 'active'/'pro' back to a 250-
    request trial the second time they visited the page. Changing a live
    subscription is set_subscription_plan()/set_subscription_status(), which are
    named after what they do.

    Defaults to status='trial' on plan='trial', which is a REAL row with a small
    request_limit and not the absence of a subscription. An absent subscription
    would mean a licence resolving to nothing, and every resolver would need a
    second code path for "authorised but unmetered" — which is precisely the
    path that forgets to write a usage row. The two defaults move together on
    purpose; see _assert_plan_status_coherent() for why the pair is checked and
    not just each half.

    *expires_at* sets a fixed term (naive UTC, or tz-aware and converted here).
    None is open-ended, which is what a monthly rolling subscription is: it ends
    by moving to status='cancelled', not by a date. Pass a date only for a term
    that was actually sold as one, because resolve_key() gates on it
    independently of the licence and a store goes dark the moment it passes.

    Returns "key_hashes" like every other mutator here, always empty: a
    subscription that has just been created holds no licences, and the
    find-path returns a row it did not touch. It is present so that a caller's
    one eviction helper works on the result of every function in this module
    without a special case — an empty list is a fact, a missing key is a trap.
    """
    if not catalog.is_valid_product(product_code):
        raise ValueError(
            f"Unknown product '{product_code}'. Pick one of: "
            + ", ".join(sorted(catalog.PRODUCTS))
        )
    if not catalog.is_valid_module_plan(plan):
        raise ValueError(
            f"Unknown module plan '{plan}'. Expected one of: "
            + ", ".join(catalog.MODULE_PLAN_ORDER)
            + f", or '{catalog.TRIAL_MODULE_PLAN}'."
        )
    if status not in SUBSCRIPTION_STATUSES:
        raise ValueError(
            f"Unknown subscription status '{status}'. Expected one of "
            f"{SUBSCRIPTION_STATUSES}."
        )
    _assert_plan_status_coherent(plan, status)

    if expires_at is not None:
        expires_at = _as_naive_utc(expires_at, "expires_at")
        if expires_at <= _utcnow():
            raise ValueError(
                f"expires_at={expires_at.isoformat()} is in the past, so this "
                f"subscription would refuse every request the moment it was "
                f"created. Pass a future date, or None for open-ended."
            )

    existing = get_subscription(db, site_id, product_code)
    if existing:
        return {**existing, "created": False, "key_hashes": []}

    # request_limit is written from the plan, here, at the moment the row is
    # created. It is not looked up at quota-check time. That is what stops the
    # v1 failure where catalog.PLANS advertised one number and
    # license_service.PLAN_LIMITS granted another: the granted number is a
    # column, and catalog.py is the only thing that can produce it.
    # request_limit_for() raises on an unknown rung rather than falling back to
    # the smallest, which matters because this value is then enforced against
    # every request for the life of the subscription.
    subscription_id = str(uuid.uuid4())
    params = {
        "id": subscription_id,
        "site_id": site_id,
        "product_code": product_code,
        "status": status,
        "plan": plan,
        "request_limit": catalog.request_limit_for(plan),
        "expires_at": expires_at,
    }

    try:
        db.execute(
            text("""
                INSERT INTO subscriptions
                    (id, site_id, product_code, status, plan, request_limit,
                     expires_at)
                VALUES
                    (:id, :site_id, :product_code, :status, :plan, :request_limit,
                     :expires_at)
            """),
            params,
        )
        db.commit()
    except IntegrityError:
        # The SELECT above and this INSERT are not one atomic operation, and
        # onboarding double-submits are the common case rather than an exotic
        # one. uq_subscriptions_site_product is what actually enforces "a store
        # buys a module once" — v1 hand-rolled the same check in the router and
        # had the same race with nothing to catch it. Losing the race is not an
        # error; it means somebody else created exactly the row we wanted.
        #
        # The rollback is mandatory before the re-read: after an IntegrityError
        # the session's transaction is dead and the next execute() on it raises
        # PendingRollbackError, which would surface as a 500 on a request that
        # actually succeeded.
        db.rollback()
        existing = get_subscription(db, site_id, product_code)
        if existing is None:
            # A constraint other than the (site_id, product_code) unique key —
            # most likely fk_subscriptions_site or fk_subscriptions_product,
            # i.e. a site_id that does not exist or a product code missing from
            # the products table because the seed never ran. Re-raise: silently
            # returning None here would let onboarding mint a key against a
            # subscription that does not exist.
            raise
        return {**existing, "created": False, "key_hashes": []}

    logger.info(
        "subscription opened: site=%s product=%s plan=%s status=%s limit=%s "
        "expires=%s",
        site_id, product_code, plan, status, params["request_limit"],
        expires_at.isoformat() if expires_at else "never",
    )

    # Re-read rather than assembling the dict from the params we just sent.
    # started_at is written by MySQL's DEFAULT CURRENT_TIMESTAMP, so the only
    # way to return its real value is to ask; hand-building the dict would put
    # a None there and the caller would get a different shape from the created
    # path than from the found path — the classic reason a signup flow works on
    # a fresh account and breaks on a repeat visit.
    created = get_subscription_by_id(db, subscription_id)
    if created is None:
        # Only reachable if something outside this function deleted the row in
        # the microseconds after commit. LookupError, not a None return: the
        # caller's next move is to mint a licence against this id and it must
        # not do that.
        raise LookupError(
            f"Subscription '{subscription_id}' vanished immediately after insert."
        )
    return {**created, "created": True, "key_hashes": []}


def _require_subscription(db: Session, subscription_id: str) -> dict:
    """The subscription with this id, or LookupError. The setters' front half.

    Every setter below establishes existence with a SELECT before it writes,
    and none of them interprets a rowcount. That is not defensive habit, it is
    the fix for a specific trap: PyMySQL does not set CLIENT_FOUND_ROWS and
    database.py passes no connect_args, so result.rowcount is rows CHANGED, not
    rows matched. Re-applying a status or a plan a subscription already holds
    changes nothing and reports 0 — indistinguishable, to the old code, from
    "no such subscription".

    That ambiguity had teeth. auth_cache tells an operator whose eviction failed
    to "retry the invalidation; it is idempotent", and through these setters it
    was not: suspend for non-payment, watch the eviction fail because Redis
    blinked, retry set_subscription_status(id, 'suspended'), get rowcount 0, get
    None back, evict nothing — and the suspended account keeps serving for the
    full TTL with no API path left to force the eviction. Reading first makes a
    zero rowcount mean exactly one thing, "the row already held these values",
    which is a success that still owes the caller its key_hashes.

    LookupError, not a None return, for an id that names no row. See the module
    docstring: a caller that does result["key_hashes"] on a None gets a
    TypeError about NoneType instead of a clean not-found, and a router
    implementing the documented ValueError contract would render the raw uuid at
    a customer.
    """
    existing = get_subscription_by_id(db, subscription_id)
    if existing is None:
        raise LookupError(f"No subscription '{subscription_id}'.")
    return existing


def _updated_subscription(db: Session, subscription_id: str, hashes: list[str]) -> dict:
    """Re-read after a committed write, carrying the hashes the caller must evict."""
    updated = get_subscription_by_id(db, subscription_id)
    if updated is None:
        # Deleted between the UPDATE and this read. The FK cascade took its
        # licences with it, so the hashes are still live cache entries pointing
        # at rows that no longer exist — name them in the message rather than
        # losing the only copy to the exception.
        raise LookupError(
            f"Subscription '{subscription_id}' was deleted mid-update. These "
            f"key hashes still need evicting: {hashes}."
        )
    return {**updated, "key_hashes": hashes}


def set_subscription_plan(db: Session, subscription_id: str, plan: str) -> dict:
    """Move a subscription to another MODULE_PLANS rung.

    plan, request_limit and status move together in one UPDATE and there is no
    way to write one without the others. Writing the label alone is the drift
    bug in its purest form: the dashboard would show Pro while the quota kept
    enforcing Starter's 10,000, and the merchant would be cut off at a tenth of
    what they just paid for with nothing in the logs saying why.

    status moves with them because 'trial' is both a status and a plan and the
    two must agree — see _status_for_plan(). Upgrading a trial to a sellable
    rung activates it in the same change ("upgrade in place, the same key keeps
    working", which is what the pricing page promises); moving any subscription
    onto the trial rung returns it to status='trial'. A suspended or cancelled
    subscription keeps its status: correcting the plan on a suspended account is
    not a decision to put it back online.

    Returns the updated row plus key_hashes — the cached context carries
    request_limit, so leaving those entries in Redis enforces the OLD limit for
    up to the full TTL after an upgrade. An upgrade that does not take effect
    for five minutes is a support ticket; a downgrade that does not is revenue.

    Raises LookupError if the id names no row, ValueError on an unknown plan or
    on a move the pair rules forbid.
    """
    if not catalog.is_valid_module_plan(plan):
        raise ValueError(
            f"Unknown module plan '{plan}'. Expected one of: "
            + ", ".join(catalog.MODULE_PLAN_ORDER)
            + f", or '{catalog.TRIAL_MODULE_PLAN}'."
        )

    existing = _require_subscription(db, subscription_id)
    status = _status_for_plan(plan, existing["status"])

    # catalog.request_limit_for() is the only thing in the system that can
    # produce this number, and it raises on an unknown rung rather than falling
    # back to the smallest. That is the guard against the v1 failure where
    # catalog.PLANS advertised one allowance and license_service.PLAN_LIMITS
    # granted another.
    request_limit = catalog.request_limit_for(plan)

    db.execute(
        text("""
            UPDATE subscriptions
            SET plan = :plan, request_limit = :request_limit, status = :status
            WHERE id = :id
        """),
        {
            "id": subscription_id,
            "plan": plan,
            "request_limit": request_limit,
            "status": status,
        },
    )

    hashes = key_hashes_for(db, subscription_id=subscription_id)
    db.commit()

    if status != existing["status"]:
        logger.info(
            "subscription %s moved to plan %s (limit %s), and status %s -> %s "
            "with it",
            subscription_id, plan, request_limit, existing["status"], status,
        )
    else:
        logger.info(
            "subscription %s moved to plan %s (limit %s)",
            subscription_id, plan, request_limit,
        )

    return _updated_subscription(db, subscription_id, hashes)


def set_subscription_status(db: Session, subscription_id: str, status: str) -> dict:
    """Suspend, cancel, activate or return a subscription to trial.

    Cancelling is a status, never a DELETE. The row has to survive so its
    usage_counters and its historical usage_events stay explicable years later —
    a billing dispute about a module that was cancelled in March is unanswerable
    if the subscription row went with it.

    plan and request_limit move with the status when the pair would otherwise
    contradict itself: returning a lapsed customer to status='trial' also drops
    them to the trial allowance, because leaving them on the Pro limit they
    stopped paying for is a free 500,000 requests a month that nothing in the
    system would ever flag. Activating a trial is the one move refused outright
    — it needs the plan they bought, which only set_subscription_plan() has.

    Returns key_hashes for the same reason set_subscription_plan() does, and
    here it is the more serious of the two: RESOLVABLE_STATUSES excludes
    'suspended' and 'cancelled', but a cached context was written when the
    status was still 'active' and nothing re-checks it until the entry expires.
    Suspending an account for non-payment and having it keep serving for another
    five minutes is the exact failure the brief calls out.

    Raises LookupError if the id names no row, ValueError on an unknown status
    or on an activation with no plan behind it.
    """
    if status not in SUBSCRIPTION_STATUSES:
        raise ValueError(
            f"Unknown subscription status '{status}'. Expected one of "
            f"{SUBSCRIPTION_STATUSES}."
        )

    existing = _require_subscription(db, subscription_id)
    plan = _plan_for_status(status, existing["plan"])
    request_limit = catalog.request_limit_for(plan)

    # All three columns are written every time, even when only the status is
    # moving and plan comes back unchanged. That is deliberate: re-asserting
    # request_limit from catalog.py costs nothing and quietly repairs a row
    # whose allowance was hand-edited away from what its plan actually grants —
    # the drift that makes "what does this customer get" unanswerable from the
    # database.
    db.execute(
        text("""
            UPDATE subscriptions
            SET status = :status, plan = :plan, request_limit = :request_limit
            WHERE id = :id
        """),
        {
            "id": subscription_id,
            "status": status,
            "plan": plan,
            "request_limit": request_limit,
        },
    )

    hashes = key_hashes_for(db, subscription_id=subscription_id)
    db.commit()

    if plan != existing["plan"]:
        logger.info(
            "subscription %s status -> %s, and plan %s -> %s with it (limit %s)",
            subscription_id, status, existing["plan"], plan, request_limit,
        )
    else:
        logger.info("subscription %s status -> %s", subscription_id, status)

    return _updated_subscription(db, subscription_id, hashes)


def set_subscription_term(
    db: Session,
    subscription_id: str,
    expires_at: Optional[datetime],
) -> dict:
    """Set or clear the fixed term a subscription was sold on.

    THE ONLY WRITER of subscriptions.expires_at, and until it existed there was
    none. create_subscription did not list the column, set_subscription_plan
    wrote plan and request_limit, set_subscription_status wrote status, and no
    code path anywhere wrote a term. So resolve_key()'s subscription-expiry gate
    — the one whose comment explains it must fire independently of the licence
    because "a lapsed subscription keeps serving until the key it happens to
    hold runs out" — could never fire on a row this API created, and
    auth_cache's TTL clamp on subscription_expires_at was dead alongside it.
    Both read as implemented and tested while being unreachable, which is the
    worst state an authorisation gate can be in. A fixed-term deal could only be
    entered with a hand-written UPDATE in a MySQL client, which by construction
    also skips the eviction below.

    expires_at=None CLEARS the term, which is what a rolling monthly
    subscription is: it ends by moving to status='cancelled', not on a date
    somebody has to remember.

    A term already in the past is allowed and logged at WARNING. Backdating is
    how an operator ends a term that should have ended last week, and refusing
    it sends them back to the MySQL client this function exists to replace — but
    it takes the store offline at the next cache miss, so it must never happen
    by accident.

    Raises LookupError if the id names no row.
    """
    if expires_at is not None:
        expires_at = _as_naive_utc(expires_at, "expires_at")

    # subscriptions.expires_at is a TIMESTAMP in schema_v2.py as this is
    # written, and TIMESTAMP tops out at 2038-01-19. A twelve-year enterprise
    # term is therefore error 1292 on this UPDATE under strict SQL mode, or a
    # silently truncated authorisation date under non-strict. That is a column
    # type decision, not something to paper over here: a range guard in this
    # function would be wrong the day the column becomes DATETIME, and nothing
    # in this module depends on TIMESTAMP's conversion behaviour because it
    # writes UTC explicitly.
    _require_subscription(db, subscription_id)

    db.execute(
        text("UPDATE subscriptions SET expires_at = :expires_at WHERE id = :id"),
        {"id": subscription_id, "expires_at": expires_at},
    )

    hashes = key_hashes_for(db, subscription_id=subscription_id)
    db.commit()

    if expires_at is None:
        logger.info("subscription %s term cleared (open-ended)", subscription_id)
    elif expires_at <= _utcnow():
        logger.warning(
            "subscription %s term set to %s, which is already past. It stops "
            "authorising as soon as its cached contexts are evicted.",
            subscription_id, expires_at.isoformat(),
        )
    else:
        logger.info(
            "subscription %s expires at %s", subscription_id, expires_at.isoformat()
        )

    return _updated_subscription(db, subscription_id, hashes)


# ── Licences ─────────────────────────────────────────────────────────────────

def issue_licence(
    db: Session,
    subscription_id: str,
    environment: str,
    valid_days: Optional[int] = DEFAULT_LICENCE_DAYS,
) -> dict:
    """Mint a key for a subscription and return the plaintext EXACTLY ONCE.

    THE PLAINTEXT IS NOT RECOVERABLE. Only the SHA-256 hash and a short display
    prefix are stored, so the string in the returned dict is the only copy that
    will ever exist. If the caller drops it — swallows an exception after this
    returns, logs it instead of showing it, renders it into a template that
    fails — the merchant's only remedy is another call to this function, which
    issues a DIFFERENT key. v1 kept the whole plaintext JWT in a TEXT column,
    which made recovery easy and a database dump a handover of every customer's
    working credential.

    *environment* accepts either vocabulary (development|production or
    test|live); see _key_environment(). It is a parameter rather than being read
    off the site because minting a test key against a production site is a
    legitimate support action — but the two disagreeing is far more often a
    mistake, so a mismatch is logged loudly and allowed rather than raised.

    valid_days=None issues an open-ended licence (expires_at NULL). Reserved for
    internal and demo keys: an open-ended key is one that no longer expires if
    the customer stops paying and everyone forgets it exists.

    Returns "key_hashes" — the keys this rotation SUPERSEDED, which is the same
    field name every other mutator here uses for the same purpose — alongside
    the singular "key_hash" of the key just minted. Evicting the plural and not
    the singular is the whole contract; they are named apart so a caller cannot
    do it the other way round.

    Raises LookupError if the subscription id names no row, or if its
    product_code is absent from catalog.PRODUCTS. Raises ValueError only on
    arguments the caller got wrong: an unknown environment, a non-positive
    valid_days.
    """
    subscription = db.execute(
        text("""
            SELECT s.id, s.site_id, s.product_code, s.status,
                   si.environment AS site_environment, si.domain
            FROM subscriptions s
            JOIN sites si ON si.id = s.site_id
            WHERE s.id = :id
        """),
        {"id": subscription_id},
    ).fetchone()

    if subscription is None:
        # LookupError, not ValueError. subscription_id is not something a
        # customer typed — it came out of create_subscription moments ago — so
        # a missing row is an internal inconsistency, and a router implementing
        # the documented `except ValueError: return error=str(e)` contract would
        # otherwise render this raw uuid into a signup form.
        raise LookupError(f"No subscription '{subscription_id}' to issue a licence for.")

    product = catalog.get_product(subscription.product_code)
    if product is None:
        # subscriptions.product_code is a foreign key into products, which is
        # seeded from catalog.PRODUCTS — so this means the seed is ahead of the
        # code, or a product was renamed in catalog.py without a migration.
        # Minting anyway would produce a key whose segment is unguessable and
        # whose product nothing downstream can price.
        #
        # LookupError for the same reason as above, and more sharply: "reseed
        # the products table" is an instruction for an operator with a shell,
        # and ValueError is the class this codebase renders straight at a
        # merchant.
        raise LookupError(
            f"Subscription '{subscription_id}' names product "
            f"'{subscription.product_code}', which catalog.PRODUCTS does not "
            f"define. products is out of step with catalog.py — reseed it."
        )

    if valid_days is not None and valid_days <= 0:
        raise ValueError(
            f"valid_days={valid_days} would issue a licence that is already "
            f"expired. Pass a positive number of days, or None for open-ended."
        )

    key_env = _key_environment(environment)
    site_env = (subscription.site_environment or "").strip().lower()
    if site_env in KEY_ENVIRONMENT_FOR_SITE and KEY_ENVIRONMENT_FOR_SITE[site_env] != key_env:
        logger.warning(
            "issuing a '%s' key for %s, whose site %s is environment='%s'. "
            "usage_events.key_owner follows the SITE, not the key, so cost on "
            "this licence will be booked as %s regardless of the key's label.",
            key_env, subscription.product_code, subscription.domain, site_env,
            key_owner_for(site_env),
        )

    # Mint BEFORE touching any existing licence. mint() validates the
    # environment and the key segment and raises on either, and if that happens
    # after the deactivation below the subscription is left with no working key
    # at all and no new one to replace it — a store goes dark because of a typo
    # in an argument.
    minted = license_key.mint(key_env, product["key_segment"])

    expires_at = _utcnow() + timedelta(days=valid_days) if valid_days else None
    licence_id = str(uuid.uuid4())

    # Read the outgoing hashes before the UPDATE overwrites nothing but still
    # makes them unfindable as "active" — key_hashes_for() does not filter on
    # is_active precisely so this could be done either side, but reading first
    # keeps the returned list to the keys this call actually superseded rather
    # than every key the subscription has ever held.
    superseded = [
        row.key_hash
        for row in db.execute(
            text("""
                SELECT key_hash FROM licences
                WHERE subscription_id = :subscription_id AND is_active = 1
            """),
            {"subscription_id": subscription_id},
        ).fetchall()
    ]

    # Deactivation and insertion share one transaction and one commit. Split
    # across two commits, a crash in between leaves the subscription with zero
    # active licences, which resolve_key() reads as a revoked key — the store
    # stops working and the operator sees a rotation that "succeeded".
    #
    # revoked_at is set rather than left NULL: a support ticket six months later
    # asking why a key stopped working is answerable from the row, and the
    # distinction between "rotated out" and "never issued" is otherwise gone.
    db.execute(
        text("""
            UPDATE licences
            SET is_active = 0, revoked_at = :now
            WHERE subscription_id = :subscription_id AND is_active = 1
        """),
        {"subscription_id": subscription_id, "now": _utcnow()},
    )

    db.execute(
        text("""
            INSERT INTO licences
                (id, subscription_id, key_hash, licence_key, is_active, expires_at)
            VALUES
                (:id, :subscription_id, :key_hash, :licence_key, 1, :expires_at)
        """),
        {
            "id": licence_id,
            "subscription_id": subscription_id,
            "key_hash": minted["key_hash"],
            # The plaintext, stored on purpose since 2026-09-03 so a key can be
            # shown again without rotating a working install. See the note above
            # LICENCES_TABLE for what that costs.
            "licence_key": minted["key"],
            "expires_at": expires_at,
        },
    )
    db.commit()

    logger.info(
        "licence %s issued for %s on %s (prefix %s, superseded %d)",
        licence_id, subscription.product_code, subscription.domain,
        minted["key_prefix"], len(superseded),
    )

    return {
        "id": licence_id,
        "subscription_id": subscription_id,
        "site_id": subscription.site_id,
        "product_code": subscription.product_code,
        # The one and only copy. Show it to the customer and let it go.
        "key": minted["key"],
        # Not a secret — it is a digest of 192 bits of OS randomness and cannot
        # be walked back to the key. It is returned because the cache layer
        # needs it to name the entry, and because the caller may want to prove
        # later which licence it created without keeping the plaintext.
        "key_hash": minted["key_hash"],
        "key_prefix": minted["key_prefix"],
        "expires_at": _iso(expires_at),
        # Evict these or the key this rotation replaced keeps authorising for
        # the whole TTL, which defeats the point of rotating it.
        #
        # Named "key_hashes" like every other mutator in this module, and NOT
        # "superseded_key_hashes" as it once was. The singular "key_hash" above
        # is the NEW key and must never be evicted; the plural is always the
        # list to forget. A caller writing one helper —
        # auth_cache.invalidate_many(result["key_hashes"]) — was previously
        # correct for three functions here and silently a no-op for this one,
        # which is the single worst place to miss: the whole point of a rotation
        # is that the old key stops working.
        "key_hashes": superseded,
    }


def revoke_licence(db: Session, licence_id: str) -> dict:
    """Kill one key. Idempotent, and always returns the hash to evict.

    Raises LookupError when the licence id names no row — the same class every
    other id-takes-no-row case in this module raises, so a router has one branch
    for "that id is not a thing" rather than four different answers across three
    modules. Revoking an already revoked licence is NOT that case: it is a no-op
    that still reports its key_hash, because the common reason to call revoke
    twice is that the first eviction failed, and refusing to hand back the hash
    the second time would make that unrecoverable without a manual FLUSHDB.

    The row is never deleted. licences.key_hash carries UNIQUE, so a deleted row
    also frees its hash — and while a 192-bit collision is not a real risk, the
    row is the only record that a key which a merchant may still have pasted
    somewhere ever existed.
    """
    row = db.execute(
        text("""
            SELECT id, subscription_id, key_hash, licence_key, is_active, revoked_at
            FROM licences
            WHERE id = :id
        """),
        {"id": licence_id},
    ).fetchone()

    if row is None:
        raise LookupError(f"No licence '{licence_id}'.")

    already_revoked = not row.is_active
    if not already_revoked:
        db.execute(
            text("""
                UPDATE licences
                SET is_active = 0, revoked_at = :now
                WHERE id = :id
            """),
            {"id": licence_id, "now": _utcnow()},
        )
        db.commit()
        # prefix_of(), never the raw column: this table holds plaintext now and
        # a revocation is exactly the moment nobody wants the dead key printed.
        logger.info(
            "licence %s revoked (prefix %s)",
            licence_id, license_key.prefix_of(row.licence_key),
        )

    return {
        "id": row.id,
        "subscription_id": row.subscription_id,
        "key_prefix": license_key.prefix_of(row.licence_key),
        "already_revoked": already_revoked,
        # Singular list, under the same name every other mutator here uses.
        # Three different shapes for "things to evict" is how one of them gets
        # iterated as a string and evicts nothing, one character at a time,
        # while reporting a healthy-looking count.
        "key_hashes": [row.key_hash],
    }


def list_licences(db: Session, subscription_id: str) -> list[dict]:
    """Every licence ever issued for a subscription, newest first.

    This exists because revoke_licence() takes a licence id and nothing else in
    the system hands one out, so without it the operator console has no way to
    reach a specific key.

    RETURNS PLAINTEXT KEYS in `key` since 2026-09-03 — that is the point of the
    column, and it is why nothing may render this straight to a merchant. `key`
    is None for the seven pre-2026-09-03 licences, whose plaintext was never
    stored; `key_prefix` is always populated and is what a list view should
    show. A caller that wants a safe listing should read `key_prefix` and ignore
    `key` rather than assume this function redacts anything.
    """
    rows = db.execute(
        text("""
            SELECT id, licence_key, is_active, issued_at, expires_at, revoked_at
            FROM licences
            WHERE subscription_id = :subscription_id
            ORDER BY issued_at DESC, id DESC
        """),
        {"subscription_id": subscription_id},
    ).fetchall()

    return [
        {
            "id": row.id,
            "key": row.licence_key,
            "key_prefix": license_key.prefix_of(row.licence_key),
            "is_active": bool(row.is_active),
            "issued_at": _iso(row.issued_at),
            "expires_at": _iso(row.expires_at),
            "revoked_at": _iso(row.revoked_at),
        }
        for row in rows
    ]


# ── Resolution: the auth hot path ────────────────────────────────────────────

def _deny(presented_key: str, reason: str) -> None:
    """Log a refusal against the masked prefix and return None.

    mask() and nothing else. A denied key is still a valid credential for some
    other tenant more often than you would like — a merchant running two stores
    pastes the wrong one — and a log file is the least controlled surface in the
    system. The prefix identifies the key well enough to match it against the
    licences table by eye and is useless for authenticating anything.
    """
    logger.info("licence denied (%s): %s", license_key.mask(presented_key), reason)
    return None


def _context_from_row(row) -> dict:
    """The resolved tenant context. Bump CACHE_VERSION if this shape changes.

    Every value is a str, int, bool or None so the whole dict survives
    json.dumps() into Redis and comes back identical. Nothing here is a
    datetime, a Decimal, or a database Row.
    """
    environment = row.environment

    return {
        # ── Identity, all of it from the database and none of it from the key
        "client_id": row.client_id,
        "site_id": row.site_id,
        "subscription_id": row.subscription_id,
        "licence_id": row.licence_id,
        # THE authoritative product identity. The key's own segment is a human
        # label and is not consulted anywhere in this function.
        "product_code": row.product_code,
        "platform": row.platform,

        # ── Store
        "domain": row.domain,
        "store_name": row.store_name,
        # STORED, never recomputed from `domain`. get_collection_name() maps
        # shop.example.com, shop-example-com and shop_example_com onto the same
        # string, and the live collections were named from an unnormalised
        # host — recomputing it from the normalised domain points a store at a
        # collection that does not exist, and Qdrant answers a missing
        # collection with zero results rather than an error. The store just goes
        # quiet.
        "collection_name": row.collection_name,
        "environment": environment,
        "key_owner": key_owner_for(environment),

        # ── The two scopes, each from its own level
        "index_plan": row.index_plan,
        "catalogue_limit": int(row.catalogue_limit),
        "indexed_items": int(row.indexed_items),
        "plan": row.plan,
        "status": row.status,
        "request_limit": int(row.request_limit),

        # ── Provenance
        "client_name": row.client_name,
        # DERIVED, not the stored column. The context is cached in Redis, logged
        # by auth_cache's fingerprinter, and stamped onto usage rows — putting a
        # usable key in it would spread the plaintext to three more places that
        # have no business holding one. The field keeps its name so nothing
        # downstream (auth_cache's required-field tuples included) has to change.
        "key_prefix": license_key.prefix_of(row.licence_key),
        # Lets the cache layer name its own entry without re-hashing, and lets
        # any caller invalidate the context it is holding.
        "key_hash": row.key_hash,
        "licence_expires_at": _iso(row.licence_expires_at),
        "subscription_expires_at": _iso(row.subscription_expires_at),
    }


class _ContextShapeProbe:
    """A stand-in row that answers every column with 0. Used exactly once.

    CONTEXT_FIELDS below is built by RUNNING _context_from_row() against this,
    rather than by typing its key names out a second time beside it. That is the
    entire point. Three other modules used to carry hand-copied versions of the
    list — usage_service._REQUIRED_CTX_FIELDS and auth_cache's three _REQUIRED_*
    tuples — with nothing binding any of them to the function that defines the
    shape, and they agreed only because nobody had renamed anything yet. A
    derived set cannot drift: rename a key in the function above and
    CONTEXT_FIELDS carries the new name in the same edit, so a consumer that
    hard-codes the old one fails where it NAMES the field rather than at 3am on
    the hot path where it reads it.

    0 rather than None because the function coerces: int(row.catalogue_limit)
    needs a number, and _iso(0) returns None exactly the way _iso(NULL) does. If
    a field added later needs a shape 0 cannot provide, this raises at import
    and the application does not boot. That is the right way round — the
    alternative is a CONTEXT_FIELDS that quietly lost a key, which is the
    failure it exists to prevent, and a boot that fails on the line below is
    about as cheap as diagnosis gets.
    """

    def __getattr__(self, name: str) -> int:
        return 0


# The authoritative key set of a resolved context. Import this; never restate
# it. usage_service uses it to prove at import time that the columns it copies
# onto a usage_events row still exist in the context, and auth_cache uses it to
# reject an individually corrupt cache entry. Both used to answer that question
# from their own copy of the list, and the two failure modes were asymmetric:
# auth_cache rejected loudly, usage_service.record() caught, logged and returned
# False — losing the billing row while the shopper still got their answer.
CONTEXT_FIELDS: frozenset[str] = frozenset(_context_from_row(_ContextShapeProbe()))


def resolve_key(db: Session, presented_key: str) -> Optional[dict]:
    """Presented licence key -> full tenant context, or None.

    THE AUTHENTICATION ENTRY POINT. Everything a request handler is allowed to
    know about who is calling comes from here.

    Order is fixed and each step exists for a reason:

      1. looks_valid() — shape and checksum, no database. Rejects a truncated
         paste, a stray quote, a copied Bearer prefix, and every scanner probing
         for credentials, without spending a query on any of them.
      2. hash_key() the presented string. The plaintext is never compared
         against anything, because nothing stores the plaintext.
      3. One SELECT on uq_licences_key_hash, joining down to the client.
      4. Five liveness gates, each with its own answer.

    Returns None rather than raising. Every caller is a request handler that has
    to turn a refusal into a 401 or a 403, and a control-flow exception on the
    hot path only guarantees that one of them will eventually let it escape as a
    500 — which reads to a merchant as "the service is broken" rather than "your
    key is not valid".
    """
    # 1 — cheap structural reject.
    #
    # A key that fails here has never touched the database, which is the point:
    # this endpoint is reachable from any storefront and the shape check is
    # roughly free next to a MySQL round trip. It proves nothing about
    # authorisation — a correctly-formatted key that was never issued sails
    # through and dies at step 3.
    if not license_key.looks_valid(presented_key):
        return _deny(presented_key, "malformed or checksum mismatch")

    # 2 — the only form the database has ever seen.
    key_hash = license_key.hash_key(presented_key)

    # 3 — one probe on the hottest index in the system, then down the chain.
    #
    # Matching on key_hash ALONE is deliberate; see the module docstring. The
    # joins are all primary-key lookups, so the whole thing is four index dives
    # regardless of how big any of these tables get.
    row = db.execute(
        text("""
            SELECT
                l.id               AS licence_id,
                l.key_hash         AS key_hash,
                l.licence_key      AS licence_key,
                l.is_active        AS licence_active,
                l.revoked_at       AS revoked_at,
                l.expires_at       AS licence_expires_at,

                s.id               AS subscription_id,
                s.product_code     AS product_code,
                s.status           AS status,
                s.plan             AS plan,
                s.request_limit    AS request_limit,
                s.expires_at       AS subscription_expires_at,

                p.platform         AS platform,

                si.id              AS site_id,
                si.domain          AS domain,
                si.store_name      AS store_name,
                si.collection_name AS collection_name,
                si.environment     AS environment,
                si.index_plan      AS index_plan,
                si.catalogue_limit AS catalogue_limit,
                si.indexed_items   AS indexed_items,
                si.is_active       AS site_active,
                si.platform        AS site_platform,

                c.id               AS client_id,
                c.name             AS client_name,
                c.is_active        AS client_active
            FROM licences l
            JOIN subscriptions s ON s.id  = l.subscription_id
            JOIN products p      ON p.code = s.product_code
            JOIN sites si        ON si.id = s.site_id
            JOIN clients c       ON c.id  = si.client_id
            WHERE l.key_hash = :key_hash
        """),
        {"key_hash": key_hash},
    ).fetchone()

    if row is None:
        # Either the key was never issued, or one of the joins found nothing —
        # a subscription whose site row was deleted, say. Both are "this key
        # authorises nothing", and neither is worth distinguishing to a caller
        # who could be probing.
        return _deny(presented_key, "no licence for this key hash")

    # 4 — liveness, innermost outwards. Order is not load-bearing, but reporting
    # the most specific true reason is: "your licence expired" and "your account
    # is deactivated" send a merchant to completely different places.

    now = _utcnow()

    if not row.licence_active:
        return _deny(
            presented_key,
            f"licence {row.licence_id} revoked at {row.revoked_at}"
            if row.revoked_at else f"licence {row.licence_id} is inactive",
        )

    # Checked separately from is_active even though a revoked licence is always
    # inactive: a key can be revoked by a process that failed to flip the flag,
    # and a revoked_at with is_active=1 is a data error that must fail closed.
    if row.revoked_at is not None:
        return _deny(presented_key, f"licence {row.licence_id} carries revoked_at")

    if row.licence_expires_at is not None and row.licence_expires_at < now:
        return _deny(
            presented_key,
            f"licence {row.licence_id} expired at {row.licence_expires_at}",
        )

    # Allow-list, not a deny-list. See RESOLVABLE_STATUSES.
    if row.status not in RESOLVABLE_STATUSES:
        return _deny(
            presented_key,
            f"subscription {row.subscription_id} status '{row.status}' is not "
            f"one of {RESOLVABLE_STATUSES}",
        )

    # A subscription can outlive its licence or the other way round — a licence
    # is issued for 365 days, a subscription's term is whatever was sold — so
    # both expiries gate independently. Missing this one means a lapsed
    # subscription keeps serving until the key it happens to hold runs out.
    if row.subscription_expires_at is not None and row.subscription_expires_at < now:
        return _deny(
            presented_key,
            f"subscription {row.subscription_id} expired at "
            f"{row.subscription_expires_at}",
        )

    if not row.site_active:
        return _deny(presented_key, f"site {row.domain} is deactivated")

    if not row.client_active:
        return _deny(presented_key, f"client {row.client_id} is deactivated")

    # Not a gate. products.platform is the authority (it is what usage_events
    # gets stamped with and what the per-platform cost report groups by), and
    # sites.platform describes the store. They disagreeing means a Magento
    # subscription was written against a WooCommerce site or vice versa, which
    # is a data error worth finding — but denying over it would take a paying
    # store offline to fix a reporting dimension.
    if row.site_platform and row.site_platform != row.platform:
        logger.warning(
            "subscription %s is product %s (platform %s) but site %s is "
            "platform %s. usage_events will carry %s.",
            row.subscription_id, row.product_code, row.platform,
            row.domain, row.site_platform, row.platform,
        )

    return _context_from_row(row)
