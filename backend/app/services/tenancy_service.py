"""Who the customer is, and which store install they are running.

This module owns exactly two tables - `clients` and `sites` - and nothing else
in the v2 model touches them directly. Subscriptions, licences and usage rows
all hang off a site id that came from here.

THE SPLIT THIS MODULE EXISTS TO ENFORCE
---------------------------------------
Every module installed on one store shares ONE Qdrant collection, named per
(client, domain). That single fact is why `sites` is a table at all:

  * catalogue size is consumed once per STORE  -> sites.catalogue_limit
  * licences and request quota are per MODULE  -> subscriptions.request_limit

A store running AIChatbot, AIProductQA and AISearch indexes its catalogue once
and pays for that once, while buying three module licences with three separate
request allowances. So a site's ceiling comes from its own `index_plan` and is
never derived from the subscriptions sitting on it - see the long note in
catalog.py for why deriving it breaks the moment a module is cancelled.

TWO STRINGS THAT MUST NOT DRIFT
-------------------------------
`domain` and `collection_name` are the two values in here that other systems
compare against rather than merely display, and both fail silently when they
are wrong:

  * `domain` is what a licence key is bound to and what DomainAuthorizer
    compares at request time. One character out and the key 403s on the store's
    first real request.
  * `collection_name` names a Qdrant collection. A read against a name that
    does not exist returns ZERO RESULTS, not an error - so a wrong value here
    surfaces days later as "search stopped working", with nothing in the logs.

Both are therefore produced by rules copied from, or imported from, the code
that already produces them. Read the comments on `normalise_domain` and
`derive_collection_name` before touching either.

WHICH EXCEPTION MEANS WHAT
--------------------------
Two, and the split is deliberate because onboarding renders one of them
straight at a customer (`except ValueError: return error=str(e)`):

  * ValueError  - the caller or the customer supplied something unusable, or
    asked for something we refuse: a malformed store URL, an unknown plan, an
    account or site that is deactivated, a downgrade below what is already
    indexed. Every message is written to be read by a merchant.
  * LookupError - an id that names no row. That is an internal inconsistency,
    never a form error, and it deliberately does NOT inherit from ValueError so
    a router's customer-facing handler cannot catch it and render a raw uuid.
    It should reach the generic 500 handler and be logged with a traceback.

WHO OWNS THE TRANSACTION
------------------------
Every function here that writes COMMITS before it returns; the caller does not
own the boundary and must not assume one. licensing_service says the same thing
in the same words, because a router calling both needs one rule and not two.

The consequence, stated rather than discovered: an onboarding flow calling
find_or_create_client -> find_or_create_site ->
licensing_service.create_subscription -> issue_licence performs four
independent commits, and a failure in the fourth leaves the first three
durable. There is no single rollback that undoes them. The compensating cleanup
for that half-built tenant is set_site_active(site_id, False) and
licensing_service.set_subscription_status(id, 'cancelled') — neither of which
deletes anything, deliberately, because a half-built tenant an operator can
finish is worth more than one that was silently unwound.

A refusal is NOT a failed statement, and no refusal path here calls
db.rollback(). The only rollbacks below are after an IntegrityError, where the
session's transaction is genuinely dead and the next execute() would raise
PendingRollbackError. Rolling back on a refusal — which set_index_plan and
adjust_indexed_items both used to do — silently discards whatever else the
caller had pending on that Session, and a caller who inserted an unrelated row
before the call loses it with no error anywhere.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------
It never mints, reads or validates a licence, and it never writes a
subscription. `find_or_create_site` gives you a site id; turning that into a
sellable, authorised install is the licensing layer's job. Keeping the storage
scope ignorant of the module scope is what stops a future change from
re-deriving one from the other.

There is exactly ONE crossing, and it is a read: every mutator here that
changes a value living in the cached authorisation context calls
`licensing_service.key_hashes_for()` and hands the resulting hashes back to its
caller. That import used to be refused on scope-purity grounds, with a comment
telling the caller to do the walk itself, and the result was the defect the
review singled out: nothing anywhere wrote `sites.is_active` or
`clients.is_active` at all. Deactivating a site or a client was a hand-written
single-column UPDATE in a MySQL client, which by construction cannot fire an
eviction hook, so `auth_cache.invalidate_for_site()` and
`invalidate_for_client()` — two of the four evictions the design says MUST
happen — had no producer and every key underneath a suspended account kept
authorising for the full 300-second TTL. Scope purity that costs a mandated
eviction is not purity, it is a missing function.

The crossing is one-directional and stays that way: licensing_service does not
import this module, which is why the import below can sit at module scope
instead of being deferred into function bodies.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.services import catalog, licensing_service

logger = logging.getLogger(__name__)


# ── Environments ─────────────────────────────────────────────────────────────
#
# Not an enum in MySQL and not a boolean here, because this value decides who
# paid for the tokens a request spent. usage_events.key_owner is stamped from
# it at write time: a development site burns Czargroup's own provider keys
# (key_owner='czargroup'), a production site burns the merchant's own keys they
# configured in their admin (key_owner='client').
#
# The consequence is that usage_events.total_cost means two entirely different
# things depending on this column - our cost of goods on one side, the
# merchant's own provider bill on the other - and summing across both produces
# a number that is not anybody's money. Flipping a site to 'production' is
# therefore a deliberate operator action, never a side effect of a customer
# re-submitting the signup form. See find_or_create_site.

# Imported, not restated. This vocabulary is read in two places - here, to
# validate what goes into sites.environment, and in licensing_service, to derive
# key_owner from it - and a second hand-written copy is exactly the drift that
# CONTEXT_FIELDS was introduced to end. Licensing owns the tuple because
# resolve_key() cannot import this module without a cycle, while this module
# already imports licensing for key_hashes_for().
ENVIRONMENTS = licensing_service.SITE_ENVIRONMENTS
DEFAULT_ENVIRONMENT = "development"


# ── The two strings that must not drift ──────────────────────────────────────

def normalise_domain(url: str) -> str:
    """Pull the bare host out of a store URL: lowercased, no www, no port.

    The rules here are a deliberate character-for-character copy of
    onboarding.extract_domain(). Copied rather than imported for one reason
    only: importing a router into a service inverts the dependency direction
    and drags FastAPI, Jinja2 and the whole template stack into anything that
    wants to create a site row - including migration scripts and tests.

    That makes this a second copy of a rule, which is the exact shape of bug
    this rewrite exists to remove, so be clear about what keeps them honest:
    the value is what a licence key is BOUND to, and what DomainAuthorizer
    compares against at request time. If the two implementations ever disagree,
    onboarding mints a key on one string and this module stores the other, and
    the store 403s on its first real request. onboarding.extract_domain should
    be deleted in favour of this function the moment that router is rewritten
    for v2 - it is the caller, and the rule belongs on this side of the line.

    Raises ValueError, with wording written for a customer to read, exactly as
    extract_domain does: every raise here is reachable from a signup form.
    """
    raw = (url or "").strip()
    if not raw:
        raise ValueError("Enter your store URL.")

    # urlparse puts a bare "shop.example.com" in .path, not .netloc, so give it
    # a scheme when the customer omitted one rather than reading the wrong field.
    if "://" not in raw:
        raw = "https://" + raw

    host = (urlparse(raw).netloc or "").lower().strip()
    if not host:
        raise ValueError("That store URL doesn't look right. Use the full address, like https://yourstore.com")

    host = host.split("@")[-1]      # strip any user:pass@
    host = host.split(":")[0]       # strip :8080
    if host.startswith("www."):
        host = host[4:]

    if "." not in host:
        # Allow bare hostnames for local development; reject the typo case
        # ("myyshop") that would otherwise mint an unusable key.
        if host not in {"localhost"}:
            raise ValueError(f"'{host}' isn't a full domain. Use the address shoppers visit, like https://yourstore.com")

    return host


def derive_collection_name(client_id: str, domain: str) -> str:
    """The Qdrant collection a (client, domain) pair reads and writes.

    Imported from qdrant_service rather than reimplemented, because that
    function is what every sync and every retrieval already calls: a private
    copy here that drifted by one character would point new sites at a
    collection nothing else ever writes to, and Qdrant answers a read against a
    name that does not exist with an empty result rather than an error.

    The import is deferred into the function body on purpose. qdrant_service
    constructs a QdrantClient at module scope, and qdrant-client 1.17 performs
    a server version handshake inside that constructor - so a module-level
    import would make "create a row in MySQL" depend on the vector store being
    reachable, and would block for a connection timeout when it is not.

    (scripts/migrate_v2_schema.py keeps a byte-identical copy instead of
    importing, under the stricter rule that a schema migration must not require
    the qdrant client library to be installed at all. Both files say so; if the
    naming rule ever changes, all three have to move together and every row in
    sites.collection_name has to be re-seeded from what Qdrant actually holds.)
    """
    from backend.app.services.qdrant_service import get_collection_name

    return get_collection_name(client_id, domain)


# ── Row shaping ──────────────────────────────────────────────────────────────
#
# One converter per table, used by every read in this module, so that a client
# dict from find_or_create_client is indistinguishable from one out of
# get_client. Callers that round-trip a row through a template or a JSON
# response should never have to ask which function produced it.

def _iso(value) -> Optional[str]:
    """Datetime -> ISO string. The same convention licensing_service._iso uses.

    These two converters used to hand back raw datetime objects while
    licensing_service's subscription and licence converters handed back ISO
    strings, so `json.dumps(site)` raised "Object of type datetime is not JSON
    serializable" and `json.dumps(subscription)` did not. Both modules wrote a
    converter-per-table precisely so a caller would never have to ask which
    function produced a dict, and then chose opposite answers.

    It matters more than it looks because FastAPI's jsonable_encoder papers over
    it: the mismatch passes in a router and fails the first time somebody
    json.dumps a site into a cache entry, a log line, a webhook payload or a
    background job — by which point the shape is load-bearing in three places.
    The resolved authorisation context is already str/int/bool/None-only for
    exactly this reason; these rows now match it.
    """
    return value.isoformat() if isinstance(value, datetime) else None


def _client_to_dict(row) -> dict:
    return {
        "id": row.id,
        "name": row.name,
        "email": row.email,
        "company": row.company,
        "is_active": bool(row.is_active),
        "created_at": _iso(row.created_at),
    }


# sites.webhook_secret is deliberately absent from this projection. It is an
# HMAC secret, and the operator console serialises whatever a service hands
# back straight into a JSON response - which is precisely how a secret ends up
# in a browser's network tab. Anything that genuinely needs it should get its
# own narrow accessor that returns the secret and nothing else, so that the read
# is visible in a grep.
#
# The one caller that was going to need one may not exist for long: the
# WooCommerce plugins are moving to the push method ai-product-qa-woo already
# uses, where the plugin hooks WordPress actions itself and presents the
# plaintext licence key as a Bearer token on every call. Inbound webhook
# verification - and with it backend/app/routers/webhooks.py - goes away in that
# design. Do not add an accessor speculatively; add it when a live path asks.
_SITE_COLUMNS = """
    id, client_id, domain, platform, platform_version, store_name,
    collection_name, environment, index_plan, catalogue_limit,
    indexed_items, is_active, created_at, updated_at
"""


def _site_to_dict(row) -> dict:
    return {
        "id": row.id,
        "client_id": row.client_id,
        "domain": row.domain,
        "platform": row.platform,
        "platform_version": row.platform_version,
        "store_name": row.store_name,
        "collection_name": row.collection_name,
        "environment": row.environment,
        "index_plan": row.index_plan,
        "catalogue_limit": int(row.catalogue_limit),
        "indexed_items": int(row.indexed_items),
        "is_active": bool(row.is_active),
        "created_at": _iso(row.created_at),
        "updated_at": _iso(row.updated_at),
    }


# ── The one crossing into the module scope ───────────────────────────────────
#
# Two one-line wrappers so the dependency on licensing_service is spelled out in
# exactly two places and every mutator below reads the same. See the module
# docstring for why the crossing exists at all; the short version is that
# without it, `sites.is_active` and `clients.is_active` had no writer anywhere
# in the tree, so two of the four evictions the cache design mandates had no
# producer and a suspended account went on authorising for the full TTL.
#
# Both deliberately return EVERY licence hash under the scope, revoked ones
# included - key_hashes_for() explains why filtering on is_active is the bug
# that deletes every cache entry except the dangerous one.


def _key_hashes_for_site(db: Session, site_id: str) -> list[str]:
    """Every cached authorisation entry that depends on this site row."""
    return licensing_service.key_hashes_for(db, site_id=site_id)


def _key_hashes_for_client(db: Session, client_id: str) -> list[str]:
    """Every cached authorisation entry beneath this client, across all its sites.

    One client can own many sites, each holding up to one subscription per
    product, each holding licences - so this is the widest eviction in the
    system and the only one that reaches everything a suspended customer owns.
    """
    return licensing_service.key_hashes_for(db, client_id=client_id)


# ── clients ──────────────────────────────────────────────────────────────────

def get_client_by_email(db: Session, email: str) -> Optional[dict]:
    """The client account for *email*, active or not.

    Returns inactive clients too. Liveness is a decision for the caller, and a
    function that hid them would make "that account is inactive" indis-
    tinguishable from "no such account" - the first needs a support
    conversation, the second needs a signup.
    """
    row = db.execute(text("""
        SELECT id, name, email, company, is_active, created_at
        FROM clients
        WHERE email = :email
    """), {"email": _normalise_email(email)}).fetchone()

    return _client_to_dict(row) if row else None


def get_client(db: Session, client_id: str) -> Optional[dict]:
    """The client account with this id, or None."""
    row = db.execute(text("""
        SELECT id, name, email, company, is_active, created_at
        FROM clients
        WHERE id = :id
    """), {"id": client_id}).fetchone()

    return _client_to_dict(row) if row else None


def find_or_create_client(
    db: Session,
    name: str,
    email: str,
    company: Optional[str] = None,
) -> dict:
    """The client account for *email*, creating it if this is a first purchase.

    Find-or-create, not create. A customer buying their second module is the
    ordinary case under per-product licensing, and v1's create_client() raised
    "a client with that email already exists" at exactly this point - which is
    what made multi-product signup impossible through the UI. There is one
    account per email and it accumulates sites and subscriptions.

    Raises ValueError if the account exists but is deactivated. Silently
    reactivating it would undo a deliberate operator action - the one kill
    switch that stops every module on every store this customer owns - through
    a public signup form. Reactivation is set_client_active(client_id, True),
    which is an operator call and evicts.

    Returns "key_hashes", always empty, for the same reason find_or_create_site
    does: one shape from every mutator in the two service modules, so a caller
    never has to know which of them owes an eviction. It is genuinely empty
    here - the only column this function can write on an existing row is
    `company`, which is not in the cached authorisation context, and it never
    touches `name`, which is.
    """
    email = _normalise_email(email)
    name = (name or "").strip()
    company = (company or "").strip() or None

    if not email:
        raise ValueError("Enter your email address.")
    if not name:
        raise ValueError("Enter your name.")

    existing = get_client_by_email(db, email)
    if existing:
        if not existing["is_active"]:
            raise ValueError("That account is inactive. Get in touch and we'll sort it out.")

        # Fill company in if we did not have it and the customer has now
        # supplied it: onboarding only started collecting it with v2, so most
        # existing rows are NULL and an invoice needs the value. Deliberately
        # only fills a hole - it never overwrites an existing name or company,
        # because those may have been corrected by an operator and a signup
        # form is not the place to silently undo that.
        if company and not existing["company"]:
            db.execute(text("""
                UPDATE clients SET company = :company WHERE id = :id
            """), {"company": company, "id": existing["id"]})
            db.commit()
            existing["company"] = company

        return {**existing, "key_hashes": []}

    client_id = str(uuid.uuid4())
    try:
        db.execute(text("""
            INSERT INTO clients (id, name, email, company)
            VALUES (:id, :name, :email, :company)
        """), {"id": client_id, "name": name, "email": email, "company": company})
        db.commit()
    except IntegrityError:
        # uq_clients_email fired: another request created this account between
        # our SELECT and our INSERT. Two browser tabs on the signup form is
        # enough to produce it. The constraint did its job, so recover to the
        # winner's row rather than surfacing a database error for something
        # that is not an error at all. The rollback is mandatory - SQLAlchemy
        # leaves the session in a failed transaction and the SELECT below would
        # raise PendingRollbackError instead of running.
        db.rollback()
        existing = get_client_by_email(db, email)
        if existing is None:
            raise
        if not existing["is_active"]:
            raise ValueError("That account is inactive. Get in touch and we'll sort it out.")
        return {**existing, "key_hashes": []}

    logger.info("tenancy: created client %s for %s", client_id, email)

    # Read the row back rather than assembling the dict from what we just sent.
    # created_at is a server-side CURRENT_TIMESTAMP default, and a Python
    # datetime.now() standing in for it would disagree with the stored value
    # whenever the app process and MySQL disagree about the clock or the zone -
    # which is the sort of thing that only shows up in an invoice date.
    client = get_client(db, client_id)
    if client is None:
        raise LookupError(f"Client '{client_id}' vanished immediately after insert.")
    return {**client, "key_hashes": []}


def _normalise_email(email: str) -> str:
    """Lowercased and stripped, so one person is one account.

    MySQL is already doing half of this: clients.email is utf8mb4_general_ci,
    so 'Sam@Shop.com' and 'sam@shop.com' collide in uq_clients_email and a
    lookup on either finds the other. Normalising in Python as well keeps the
    stored value predictable for anything that compares it outside the database
    - log lines, invoices, exports - where no collation is helping.

    Do NOT "tidy" that column to a _bin collation on the strength of this
    function. The two together are belt and braces; the collation alone is what
    stops a returning customer who capitalised their email getting a second
    client row and losing sight of every site they already own.
    """
    return (email or "").strip().lower()


# ── sites ────────────────────────────────────────────────────────────────────

def get_site(db: Session, site_id: str) -> Optional[dict]:
    """The site with this id, or None."""
    row = db.execute(text(f"""
        SELECT {_SITE_COLUMNS}
        FROM sites
        WHERE id = :id
    """), {"id": site_id}).fetchone()

    return _site_to_dict(row) if row else None


def sites_for_client(db: Session, client_id: str) -> list[dict]:
    """Every store install this client owns, oldest first.

    Includes deactivated sites. The operator console needs to show them - a
    customer asking "why did my staging store stop working" is answered by the
    row being there with is_active=0, and not at all by an empty list.
    """
    rows = db.execute(text(f"""
        SELECT {_SITE_COLUMNS}
        FROM sites
        WHERE client_id = :client_id
        ORDER BY created_at ASC, domain ASC
    """), {"client_id": client_id}).fetchall()

    return [_site_to_dict(row) for row in rows]


def find_or_create_site(
    db: Session,
    client_id: str,
    domain: str,
    platform: str,
    store_name: Optional[str] = None,
    platform_version: Optional[str] = None,
    index_plan: str = catalog.DEFAULT_INDEX_PLAN,
) -> dict:
    """The site row for (client, domain), creating it on first sight.

    *domain* is accepted in whatever shape the customer typed it - a full URL,
    a bare host, with or without www - and normalised here. What gets stored is
    what a licence key will be bound to, so this must be the only place the
    value is decided; a caller pre-normalising with its own rules is how the
    two strings drift apart.

    Idempotent, and it has to be: a customer buying a second module for a store
    they already run comes through this function again with the same domain,
    and must land on the same site, the same collection and the same catalogue
    ceiling they already paid for. That is the whole point of the storage scope
    being separate from the module scope.

    What a repeat call MAY change is descriptive: store_name and
    platform_version, which are support-triage fields the merchant re-reports
    on every signup and whose current value is more useful than their first.
    What it may NOT change is anything commercial or structural -
    index_plan, catalogue_limit, environment and platform are untouched here.
    Those are purchases and operator decisions; see set_index_plan for the
    supported way to move a ceiling.

    THERE IS NO `environment` PARAMETER, and that is the point. Every site this
    function creates starts on DEFAULT_ENVIRONMENT ('development'); promoting
    one is set_site_environment(), which is an operator action and evicts. The
    parameter used to be here, validated and defaulted like any other, sitting
    on a signature otherwise made of values that come straight off a signup form
    - store_name, platform_version, the domain the merchant typed. One router
    binding it to a request body and the merchant chooses who pays for their own
    inference: sites.environment is what licensing_service.key_owner_for()
    reads to stamp usage_events.key_owner, so 'production' means the spend is
    the merchant's own and 'development' means it is Czargroup's cost of goods.
    Defaulting to 'development' also fails in the safe direction - we book their
    spend as ours, which understates margin and surfaces in a reconciliation,
    rather than invoicing them for tokens they never bought.

    Returns "key_hashes" alongside the site, always: [] on the create path
    (a brand-new site holds no licences) and, on the repeat path, whatever
    _refresh_site_metadata() invalidated. Same convention as every mutator in
    licensing_service, so one eviction helper works on all of them.

    Raises ValueError on: an unusable domain, an unknown platform, an unknown
    index plan, a deactivated client, a deactivated site, a platform that
    disagrees with the existing row, or a collection name already claimed by a
    different site. Every one of those is customer- or operator-facing wording.
    A client_id naming no row raises LookupError.
    """
    domain = normalise_domain(domain)

    if not catalog.is_valid_platform(platform):
        raise ValueError(
            f"Unknown platform '{platform}'. Expected one of: "
            + ", ".join(sorted(catalog.PLATFORMS))
        )

    # Raises on an unknown rung rather than coercing to the smallest one. v1's
    # onboarding did `if not is_valid_plan(plan): plan = DEFAULT_PLAN`, which is
    # fine for a tampered radio button and catastrophic here: this number is
    # written to catalogue_limit and then enforced against every sync, so a
    # silent fallback means a customer who bought 100,000 items being cut off
    # at 500 with no error anywhere explaining it.
    catalogue_limit = catalog.catalogue_limit_for(index_plan)

    # The foreign key would catch a missing client, but as errno 1452 naming a
    # constraint rather than a problem anyone can act on. The active check has
    # no equivalent in the schema at all, and a deactivated customer acquiring
    # new store installs is a state nothing downstream is written to expect.
    client = get_client(db, client_id)
    if client is None:
        # LookupError, not ValueError: client_id is not something a customer
        # typed, it came out of find_or_create_client moments ago. A missing row
        # here means it was deleted underneath us, which is an internal
        # inconsistency to log and not a message to render on a signup form.
        raise LookupError(f"No client '{client_id}'.")
    if not client["is_active"]:
        raise ValueError("That account is inactive. Get in touch and we'll sort it out.")

    existing = _site_by_client_domain(db, client_id, domain)
    if existing:
        return _adopt_existing_site(db, existing, platform, store_name, platform_version)

    site_id = str(uuid.uuid4())
    collection_name = derive_collection_name(client_id, domain)

    # uq_sites_collection would catch this at INSERT time, but the message it
    # produces names a constraint rather than the two stores involved. The
    # regex behind derive_collection_name is not injective - it replaces every
    # non-alphanumeric with '_', so shop.example.com, shop-example-com and
    # shop_example_com all collapse to one name - and two sites on one
    # collection is a cross-tenant read of somebody else's catalogue. Worth a
    # SELECT to be able to say which site already holds it.
    clash = db.execute(text("""
        SELECT id, domain FROM sites WHERE collection_name = :collection_name
    """), {"collection_name": collection_name}).fetchone()
    if clash:
        raise ValueError(
            f"{domain} would share a search index with {clash.domain} (site {clash.id}). "
            f"That is not something we can set up automatically - get in touch and we "
            f"will sort it out."
        )

    try:
        db.execute(text("""
            INSERT INTO sites
                (id, client_id, domain, platform, platform_version, store_name,
                 collection_name, environment, index_plan, catalogue_limit)
            VALUES
                (:id, :client_id, :domain, :platform, :platform_version, :store_name,
                 :collection_name, :environment, :index_plan, :catalogue_limit)
        """), {
            "id": site_id,
            "client_id": client_id,
            "domain": domain,
            "platform": platform,
            "platform_version": (platform_version or "").strip() or None,
            "store_name": (store_name or "").strip() or None,
            "collection_name": collection_name,
            "environment": DEFAULT_ENVIRONMENT,
            "index_plan": index_plan,
            "catalogue_limit": catalogue_limit,
        })
        db.commit()
    except IntegrityError:
        # uq_sites_client_domain, or uq_sites_collection losing the same race.
        # Concurrent signups for one store are ordinary: a merchant installing
        # two modules in two tabs hits this. Recover to the row that won.
        db.rollback()
        existing = _site_by_client_domain(db, client_id, domain)
        if existing is None:
            # Not our unique key, then - most likely the collection name was
            # claimed by a DIFFERENT client between the clash check and here.
            # Re-raise: this is the cross-tenant case and it must not be
            # smoothed over into a successful-looking return.
            raise
        # Same adoption path as the ordinary repeat call, so the loser of the
        # race is still held to the platform check. Two concurrent signups for
        # one domain naming different platforms is narrow but real, and letting
        # it through here would produce exactly the shared-collection mess the
        # check exists to stop.
        return _adopt_existing_site(db, existing, platform, store_name, platform_version)

    logger.info(
        "tenancy: created site %s client=%s domain=%s platform=%s env=%s "
        "index_plan=%s limit=%s collection=%s",
        site_id, client_id, domain, platform, DEFAULT_ENVIRONMENT,
        index_plan, catalogue_limit, collection_name,
    )
    site = get_site(db, site_id)
    if site is None:
        # Only reachable if something outside this function deleted the row in
        # the microseconds after commit. Loud, because a caller about to mint a
        # licence against this id needs to not do that.
        raise LookupError(f"Site '{site_id}' vanished immediately after insert.")
    # Nothing to evict: this site did not exist a moment ago, so nothing is
    # cached under it. The key is present anyway so the created path and the
    # repeat path hand back the same shape - a caller that has to test which
    # branch it got is the classic reason a signup flow works on a fresh account
    # and breaks on a repeat visit.
    return {**site, "key_hashes": []}


def _site_by_client_domain(db: Session, client_id: str, domain: str) -> Optional[dict]:
    """Lookup on uq_sites_client_domain - the identity of a store install."""
    row = db.execute(text(f"""
        SELECT {_SITE_COLUMNS}
        FROM sites
        WHERE client_id = :client_id AND domain = :domain
    """), {"client_id": client_id, "domain": domain}).fetchone()

    return _site_to_dict(row) if row else None


def _platform_name(code: str) -> str:
    """Display name for a platform code, falling back to the code itself.

    Defensive because this is only ever called while building an error message,
    and a site row carrying a platform the catalogue no longer lists - a code
    retired after the row was written, or the migration's LEGACY_PLATFORM
    default landing on something since renamed - would turn a helpful
    ValueError into a KeyError from inside the f-string.
    """
    return catalog.PLATFORMS.get(code, {}).get("name", code)


def _adopt_existing_site(
    db: Session,
    site: dict,
    platform: str,
    store_name: Optional[str],
    platform_version: Optional[str],
) -> dict:
    """Return an already-registered site, after checking it is usable as one.

    The single path for "this store install already exists", shared by the
    ordinary repeat signup and by the loser of a concurrent-insert race, so
    that both are held to the same two checks.
    """
    if not site["is_active"]:
        # Same reasoning as the inactive-client case: is_active=0 is a
        # deliberate operator action, and a signup form must not undo it. It
        # matters more here than it looks, because minting a licence against a
        # deactivated site succeeds - the site is only consulted later, on the
        # request path, where the merchant gets a 403 with nothing anywhere
        # explaining why the key they were just handed does not work.
        raise ValueError(
            f"{site['domain']} is currently deactivated. Get in touch and we'll "
            f"sort it out."
        )

    if site["platform"] != platform:
        # Not a tolerable mismatch, because both platforms would be sharing one
        # Qdrant collection: collection_name is derived from (client, domain)
        # only, so a Magento sync and a WooCommerce sync against this row would
        # write product points with different payload shapes into the same
        # collection and read each other's back. A genuine platform migration
        # is rare, deliberate, and needs a human to decide what happens to the
        # existing points - picking the wrong product on the signup form is
        # neither of those and is far more common.
        raise ValueError(
            f"{site['domain']} is already registered as a {_platform_name(site['platform'])} "
            f"store, so it cannot also be set up as {_platform_name(platform)}. Get in "
            f"touch if you have moved the store to a different platform."
        )

    return _refresh_site_metadata(db, site, store_name, platform_version)


def _refresh_site_metadata(
    db: Session,
    site: dict,
    store_name: Optional[str],
    platform_version: Optional[str],
) -> dict:
    """Update the two support-triage fields, if the caller supplied new values.

    Only ever fills or replaces store_name and platform_version, and only when
    a non-empty value was passed: a repeat signup that omits them must not
    blank out what we already know. Neither field is read by anything that
    authorises, meters or bills, which is exactly why they are the only two a
    re-submitted form is allowed to move.

    store_name IS in the cached authorisation context, though, so this returns
    key_hashes like everything else that touches a cached value - populated only
    when a change was actually written. Nothing authorises on it and a stale
    copy hurts nobody, so this is the one eviction in the module that is
    cosmetic; it is returned anyway because the alternative is a reader having
    to work out which of two nearly identical dicts owes an eviction, and that
    is how the licence-level ones get missed too.
    """
    store_name = (store_name or "").strip() or None
    platform_version = (platform_version or "").strip() or None

    changes: dict[str, object] = {}
    if store_name and store_name != site["store_name"]:
        changes["store_name"] = store_name
    if platform_version and platform_version != site["platform_version"]:
        changes["platform_version"] = platform_version

    if not changes:
        return {**site, "key_hashes": []}

    # The interpolated fragment is built from this function's own two literal
    # column names, never from anything a caller passed - the caller's values
    # go through bound parameters below like everywhere else. Worth stating
    # plainly, because an f-string next to a SET clause is the shape everyone
    # is trained to flag on sight.
    assignments = ", ".join(f"{column} = :{column}" for column in changes)
    db.execute(
        text(f"UPDATE sites SET {assignments} WHERE id = :id"),
        {**changes, "id": site["id"]},
    )
    hashes = _key_hashes_for_site(db, site["id"]) if "store_name" in changes else []
    db.commit()

    site.update(changes)
    return {**site, "key_hashes": hashes}


# ── Catalogue ceiling and the counter it is checked against ──────────────────

def set_index_plan(db: Session, site_id: str, index_plan: str) -> dict:
    """Move a site to a different INDEX_PLANS rung, ceiling and all.

    index_plan and catalogue_limit always move together. They are two columns
    describing one purchase, and the failure mode of letting them diverge is a
    dashboard that shows a customer the plan they bought while the sync path
    enforces a different number.

    REFUSES A DOWNGRADE below what the site has already indexed, and says both
    numbers. Accepting it would strand the store above its own ceiling with no
    way back: nothing can be un-indexed to get under the line, refusing every
    subsequent sync bricks a store that is still paying, and keeping the old
    ceiling quietly makes the number on the dashboard a lie. The caller has to
    deal with it - by having the merchant reduce what they sync, or by taking
    the site down and rebuilding the collection.

    Note the guard is only as honest as sites.indexed_items, which the v2
    migration seeds at 0 for every backfilled site and which needs a reconcile
    pass against Qdrant before it can be trusted. Until that has run, this
    check will wave through downgrades it should be refusing.

    Returns the site plus "key_hashes". It used to return the site alone and
    document the eviction in a comment sixty lines below the signature, which
    made forgetting it invisible: the caller got back a dict that looked
    complete. Both columns this function writes ride in the cached
    authorisation context, so the symptom of a missed eviction is a merchant who
    upgrades and keeps being refused syncs against the old ceiling for the rest
    of the TTL - and it reads as a plan-change bug, not a cache bug.

    Raises LookupError if the id names no row, ValueError on a refused
    downgrade or an unknown rung. Neither path rolls back: see the module
    docstring.
    """
    catalogue_limit = catalog.catalogue_limit_for(index_plan)

    # The downgrade guard lives in the WHERE clause, not in an if statement
    # after a SELECT. A sync running concurrently can push indexed_items past
    # the new ceiling in the window between a read and a write, and the version
    # of this bug that gets shipped is always the one where the check passed on
    # a number that was already stale.
    result = db.execute(text("""
        UPDATE sites
        SET index_plan = :index_plan, catalogue_limit = :catalogue_limit
        WHERE id = :id AND indexed_items <= :catalogue_limit
    """), {"index_plan": index_plan, "catalogue_limit": catalogue_limit, "id": site_id})

    if result.rowcount == 0:
        # rowcount is rows CHANGED, not rows matched: without CLIENT_FOUND_ROWS
        # (and PyMySQL does not set it) MySQL reports 0 for an UPDATE that
        # assigned a row its existing values. So a zero here has three possible
        # meanings and only a read can tell them apart - treating it as "the
        # downgrade was refused" would reject re-applying the plan a site is
        # already on.
        #
        # Neither refusal below rolls back, and that is a change from how this
        # used to read. A zero rowcount means nothing was written, so there is
        # nothing to undo - while db.rollback() on the caller's Session throws
        # away every other statement they had pending, silently, on a path that
        # is merely saying no. The rollbacks in this module are all after an
        # IntegrityError, where the transaction is genuinely dead.
        site = get_site(db, site_id)
        if site is None:
            raise LookupError(f"No site '{site_id}'.")
        if site["indexed_items"] > catalogue_limit:
            raise ValueError(
                f"The {catalog.INDEX_PLANS[index_plan]['name']} plan holds "
                f"{catalogue_limit:,} items and this store already has "
                f"{site['indexed_items']:,} indexed. Reduce what you sync to "
                f"{catalogue_limit:,} or fewer first, then change the plan."
            )
        # No-op: same plan, same ceiling, row untouched. Still a success, and it
        # still hands back the hashes - an operator re-applying a plan is most
        # often somebody whose first eviction failed, and returning [] here
        # would leave them no way to force it short of a manual FLUSHDB.
        hashes = _key_hashes_for_site(db, site_id)
        db.commit()
        return {**site, "key_hashes": hashes}

    # Read the hashes inside the same transaction as the UPDATE, and hand them
    # back rather than DELeting here. licensing_service caches the resolved
    # context against key_hash for five minutes and that context carries
    # index_plan and catalogue_limit, both of which just moved. The DEL itself
    # stays with the caller for one reason: auth_cache says evict AFTER the
    # mutation commits, never before, because evicting first leaves a window in
    # which a concurrent request reads the OLD committed row and writes it back
    # with a fresh full TTL.
    hashes = _key_hashes_for_site(db, site_id)
    db.commit()

    logger.info(
        "tenancy: site %s moved to index plan %s (ceiling %s), %d cached key(s) "
        "to evict",
        site_id, index_plan, catalogue_limit, len(hashes),
    )
    site = get_site(db, site_id)
    if site is None:
        raise LookupError(
            f"Site '{site_id}' was deleted mid-update. These key hashes still "
            f"need evicting: {hashes}."
        )
    return {**site, "key_hashes": hashes}


def adjust_indexed_items(db: Session, site_id: str, delta: int) -> int:
    """Move the site's indexed-entity count by *delta*, and return the new total.

    Always an atomic UPDATE, never a read-modify-write. Two syncs running at
    once - a full catalogue push and a webhook-driven single-product update are
    routine together - would each read the same starting number and each write
    back their own increment, losing one of them entirely. Doing the arithmetic
    inside the UPDATE makes the row lock do the serialising.

    The CAST is load-bearing. indexed_items is INT UNSIGNED, and MySQL
    evaluates `indexed_items + (-5)` in unsigned arithmetic: on a row holding 3
    that is error 1690, "BIGINT UNSIGNED value is out of range", raised BEFORE
    GREATEST ever sees the value. So the clamp does not clamp - it throws, and
    it throws on the delete path of a store whose counter has drifted low,
    which is exactly when you least want a sync to fall over. CAST forces the
    addition into signed arithmetic; GREATEST then guarantees the value written
    back is non-negative, so assigning it to an unsigned column is safe.

    There is no upper clamp. INT UNSIGNED tops out at 4,294,967,295, forty
    thousand times the largest plan, so a delta that overflows it is a bug in
    the caller and failing loudly is the correct outcome.

    Raises LookupError if the site does not exist, so that a sync counting into
    a site id that was never created cannot be mistaken for a successful no-op.

    Deliberately does NOT evict the licence resolver cache, unlike
    set_index_plan. indexed_items is in the cached context, but this is the
    hottest write in the system - a full catalogue sync calls it in a loop - and
    evicting on every call would mean re-resolving four tables per batch and
    leave the cache permanently empty for exactly the tenant doing the most
    work. The number in a cached context is therefore knowingly up to five
    minutes stale, which is fine because nothing enforces a ceiling from it:
    has_catalogue_headroom reads sites directly. Do not "optimise" that check
    into reading the cached context - a sync would then be measured against a
    count from before it started, and the ceiling would stop meaning anything.
    """
    db.execute(text("""
        UPDATE sites
        SET indexed_items = GREATEST(0, CAST(indexed_items AS SIGNED) + :delta)
        WHERE id = :id
    """), {"delta": int(delta), "id": site_id})

    # Existence is checked by reading the value back, NOT by the UPDATE's
    # rowcount. rowcount counts rows changed, so it is legitimately 0 whenever
    # the arithmetic produced the number already stored - delta=0, or any
    # negative delta against a counter sitting at 0 - and a site that exists
    # would be reported as missing.
    current = db.execute(text("""
        SELECT indexed_items FROM sites WHERE id = :id
    """), {"id": site_id}).scalar()

    if current is None:
        # No rollback. The UPDATE matched no row, so it wrote nothing and there
        # is nothing to undo - while rolling back the caller's Session would
        # discard every other statement they had pending. This is the hottest
        # write in the system and it is routinely called from inside a larger
        # sync unit of work, which is exactly the caller whose pending work
        # would disappear.
        raise LookupError(f"No site '{site_id}'.")

    db.commit()
    return int(current)


def has_catalogue_headroom(db: Session, site_id: str, adding: int) -> tuple[bool, int, int]:
    """Can this site hold *adding* more entities? Returns (ok, current, limit).

    Returns the two numbers alongside the verdict rather than just a boolean,
    because every caller that refuses a sync has to tell the merchant why, and
    "over your plan limit" without the figures generates a support ticket every
    single time.

    Advisory, not a lock. The count can move between this call and the write
    that follows it, so this is the check that produces a good error message
    early in a sync, not the thing that guarantees the ceiling holds. The
    binding enforcement belongs with the writer, next to adjust_indexed_items.

    Counted in logical entities - a configurable product, a CMS page - not in
    Qdrant points, because that is the unit sites.catalogue_limit and the
    INDEX_PLANS ladder are both denominated in. A product that chunks into
    forty vectors is one item against the ceiling.
    """
    if adding < 0:
        # Deletions trivially fit, so a negative here is a sign error at the
        # call site rather than a question. Answering "yes, plenty of room"
        # would let a mis-signed sync sail past the ceiling it was meant to
        # check against.
        raise ValueError(f"has_catalogue_headroom() takes a non-negative count, got {adding}.")

    row = db.execute(text("""
        SELECT indexed_items, catalogue_limit FROM sites WHERE id = :id
    """), {"id": site_id}).fetchone()

    if row is None:
        raise LookupError(f"No site '{site_id}'.")

    current = int(row.indexed_items)
    limit = int(row.catalogue_limit)
    return (current + adding <= limit), current, limit


# ── Operator switches ────────────────────────────────────────────────────────
#
# THE FUNCTIONS THAT DID NOT EXIST, AND WHY THAT WAS THE BUG.
#
# Until these were written, nothing in any of the four modules wrote
# `sites.is_active` or `clients.is_active`. Not one UPDATE anywhere in the tree:
# the only writer of any is_active column was licensing_service.revoke_licence,
# which touches `licences`. Deactivating a store install, or suspending a
# customer for non-payment, was a hand-written single-column UPDATE typed into a
# MySQL client - and an UPDATE typed into a MySQL client cannot, by
# construction, fire an eviction hook.
#
# The consequence was not theoretical. auth_cache.invalidate_for_site() and
# invalidate_for_client() are correct and were unreachable: two of the four
# evictions the cache design says MUST happen had no producer at all, so a
# client suspended for non-payment kept authorising on every cached hash
# underneath it for the full 300-second TTL - and one client can hold many
# sites, each holding a subscription per product, each holding licences. The
# failure was also invisible from the outside, which is why it survived: with a
# cold cache resolve_key() denies correctly, so anyone testing sees exactly the
# right answer.
#
# Each function below therefore does the UPDATE and hands back the key_hashes
# that must now be evicted, in the same shape licensing_service's mutators use.
# An operator console cannot switch an account off without being handed, in the
# return value, precisely the list it owes the cache. That is the whole design:
# the eviction is a VISIBLE omission at the call site rather than an invisible
# one, and the TTL goes back to being the backstop it was meant to be.
#
# All three are idempotent and all three return the hashes even when the row
# already held the requested value. The common reason to run one of these twice
# is that the first eviction failed, and refusing to hand the list back the
# second time makes that unrecoverable short of a manual FLUSHDB.


def set_site_active(db: Session, site_id: str, active: bool) -> dict:
    """Switch one store install off, or back on.

    Deactivating is the per-store kill switch: resolve_key() checks
    sites.is_active and denies with "site <domain> is deactivated", which stops
    every module on that store while leaving the client's other stores running.
    It is not a delete and nothing cascades - the subscriptions, licences,
    usage_events and counters underneath it all stay exactly where they are, so
    the store can be switched back on without re-onboarding and a billing
    question about it is still answerable a year later.

    REACTIVATING also returns hashes, and they are not usually needed: denials
    are never cached, so there is normally nothing stale to remove. They are
    returned anyway because "normally" is doing work in that sentence - if the
    eviction that accompanied the DEACTIVATION failed, a live pre-deactivation
    context is still sitting in Redis, and this is the call that clears it.

    Refuses to reactivate a site whose CLIENT is deactivated, because that
    succeeds while changing nothing a merchant can observe: resolve_key() denies
    at the client gate immediately afterwards, and the operator is left staring
    at is_active=1 on a store that will not serve.

    Raises LookupError if the id names no row.
    """
    site = get_site(db, site_id)
    if site is None:
        raise LookupError(f"No site '{site_id}'.")

    if active and not site["is_active"]:
        client = get_client(db, site["client_id"])
        if client is not None and not client["is_active"]:
            raise ValueError(
                f"{site['domain']} cannot be switched back on while the account "
                f"that owns it ({client['email']}) is deactivated - it would "
                f"still refuse every request. Reactivate the client first with "
                f"set_client_active()."
            )

    db.execute(
        text("UPDATE sites SET is_active = :is_active WHERE id = :id"),
        {"id": site_id, "is_active": 1 if active else 0},
    )

    # Inside the transaction, before the commit, and handed back rather than
    # DELeted here: auth_cache is explicit that eviction happens AFTER the
    # mutation commits, because evicting first leaves a window in which a
    # concurrent request reads the OLD committed row and writes it back with a
    # fresh full TTL - strictly worse than not evicting at all.
    hashes = _key_hashes_for_site(db, site_id)
    db.commit()

    logger.info(
        "tenancy: site %s (%s) is_active -> %s, %d cached key(s) to evict",
        site_id, site["domain"], 1 if active else 0, len(hashes),
    )

    updated = get_site(db, site_id)
    if updated is None:
        raise LookupError(
            f"Site '{site_id}' was deleted mid-update. These key hashes still "
            f"need evicting: {hashes}."
        )
    return {**updated, "key_hashes": hashes}


def set_client_active(db: Session, client_id: str, active: bool) -> dict:
    """Switch a whole customer account off, or back on.

    THE WIDEST SWITCH IN THE SYSTEM. resolve_key() checks clients.is_active
    last, so this denies every key on every store this customer owns, across
    every product, in one column - which is what makes it the non-payment
    suspension and also what makes the missing eviction so expensive. The
    returned key_hashes is the only list that reaches all of them; walking sites
    one at a time and calling set_site_active() would leave the client flag
    itself unchanged and is not the same operation.

    Deactivating deliberately does NOT touch the sites underneath. They keep
    is_active=1 and simply stop serving, so switching the account back on
    restores exactly the state the customer had - rather than a state some
    cascade invented, in which a store the operator had deliberately switched
    off six months ago quietly comes back up.

    Nothing is deleted here either. clients rows outlive cancellation because
    usage_events and usage_counters hang off them and a billing dispute about a
    module cancelled in March is unanswerable if the account row went with it.

    Raises LookupError if the id names no row.
    """
    client = get_client(db, client_id)
    if client is None:
        raise LookupError(f"No client '{client_id}'.")

    db.execute(
        text("UPDATE clients SET is_active = :is_active WHERE id = :id"),
        {"id": client_id, "is_active": 1 if active else 0},
    )

    hashes = _key_hashes_for_client(db, client_id)
    db.commit()

    logger.info(
        "tenancy: client %s (%s) is_active -> %s, %d cached key(s) to evict "
        "across every site it owns",
        client_id, client["email"], 1 if active else 0, len(hashes),
    )

    updated = get_client(db, client_id)
    if updated is None:
        raise LookupError(
            f"Client '{client_id}' was deleted mid-update. These key hashes "
            f"still need evicting: {hashes}."
        )
    return {**updated, "key_hashes": hashes}


def set_site_environment(db: Session, site_id: str, environment: str) -> dict:
    """Promote a store to 'production', or demote it back to 'development'.

    THE ONLY WRITER of sites.environment after the row is created, which is why
    find_or_create_site no longer takes it as a parameter - see that function's
    docstring for why a value this expensive must not sit on a signature made of
    signup-form fields.

    This column decides WHO PAID for the tokens every request spends.
    licensing_service.key_owner_for() reads it and stamps
    usage_events.key_owner on every ledger row: 'production' means the merchant
    supplied their own LLM and embedding keys and total_cost is THEIR spend,
    'development' means the calls ran on Czargroup's keys and the same column is
    our cost of goods. Summing across both produces a number that is nobody's
    money, which is why the value is denormalised onto every row and why a stale
    cached context here is a billing error rather than a cosmetic one: it books
    a client's own API spend as Czargroup COGS, or invoices them for ours.

    The promotion to 'production' is logged at WARNING. It is rare, it is
    irreversible in the ledger (rows already written keep the key_owner they
    were stamped with, correctly), and it should be findable in a log search
    when somebody asks why a merchant's invoice changed shape.

    Raises LookupError if the id names no row, ValueError on an unknown
    environment.
    """
    if environment not in ENVIRONMENTS:
        raise ValueError(
            f"Unknown environment '{environment}'. Expected one of: "
            + ", ".join(ENVIRONMENTS)
        )

    site = get_site(db, site_id)
    if site is None:
        raise LookupError(f"No site '{site_id}'.")

    db.execute(
        text("UPDATE sites SET environment = :environment WHERE id = :id"),
        {"id": site_id, "environment": environment},
    )

    hashes = _key_hashes_for_site(db, site_id)
    db.commit()

    log = logger.warning if environment == "production" else logger.info
    log(
        "tenancy: site %s (%s) environment %s -> %s. usage_events.key_owner on "
        "new rows becomes '%s'; %d cached key(s) to evict or the old owner "
        "keeps being stamped for the rest of the TTL.",
        site_id, site["domain"], site["environment"], environment,
        licensing_service.key_owner_for(environment), len(hashes),
    )

    updated = get_site(db, site_id)
    if updated is None:
        raise LookupError(
            f"Site '{site_id}' was deleted mid-update. These key hashes still "
            f"need evicting: {hashes}."
        )
    return {**updated, "key_hashes": hashes}
