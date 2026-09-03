"""
WooCommerce webhook receiver for semantic-search-woo. DEPRECATED — this whole
module is decommissioned by the v2 schema. Do not build on it.

WHAT IS HERE AND WHAT FIRES IT
------------------------------
Three endpoints, all registered under /api by main.py, all POSTed by
WooCommerce ITSELF rather than by our plugin's PHP. SSW_Webhook_Manager
(semantic-search-woo/includes/class-webhook-manager.php) creates them through
the store's own WooCommerce REST API when the merchant saves a licence key,
and bakes the tenant identity and both provider keys into the delivery URL as
query parameters, because a WooCommerce webhook has no other way to carry
them:

  POST /api/webhook/product-created   topic product.created
  POST /api/webhook/product-updated   topic product.updated
  POST /api/webhook/product-deleted   topic product.deleted

Each one authenticates the same way: `?client_id=` names the tenant, and the
X-WC-Webhook-Signature header is HMAC-SHA256 of the raw body under a shared
secret the plugin generated and pushed to /api/register-webhook-secret, which
stored it on `clients.webhook_secret`. Note what that means — the caller
proves it knows the webhook secret, NOT that it holds a licence key. Nothing
on this path ever presents one.

WHERE THE PLAINTEXT LICENCE KEY COMES FROM, AND WHY THAT KILLS THIS FILE
-----------------------------------------------------------------------
Both provider keys arrive here encrypted, and llm_key_service.decrypt_key
derives its AES key as sha256(license_key) — so unwrapping them needs the
merchant's plaintext licence key. Since the caller is WooCommerce and not our
plugin, there is nothing on the wire to take it from, so this file goes and
reads it out of the database instead:

  * process_upsert()  -> get_client_license(db, client_id)["license_key"]
                         handed to resolve_embedding_key(). The KEK. This is
                         the dependency the whole decommission is about.
  * process_upsert()  -> the same dict's ["domain"] and ["product_limit"] for
                         the Qdrant collection and the catalogue ceiling.
  * product_deleted() -> get_client_license(db, client_id)["domain"], only to
                         name the collection.

schema_v2 stores `licences.key_hash` and no plaintext column at all, so after
the migration get_client_license has nothing to select and this path has no
way to obtain a KEK. It is not fixable from the server side: a hash cannot be
un-hashed, and no amount of rework here changes the fact that WooCommerce's
own webhook does not carry a credential we can derive one from.

WHAT REPLACES IT
----------------
The push method ai-product-qa-woo already uses: the PLUGIN hooks WordPress
actions itself and calls the API with `Authorization: Bearer <license_key>` on
every request (AIPQA_API_Client::build_headers, ai-product-qa-woo/includes/
class-api-client.php). The plaintext key is then always presented, the KEK is
always in scope, and the server resolves identity from the key it was handed
rather than from a client_id in a query string. semantic-search-woo already
has half of this — SSW_Sync::register_hooks() pushes product upserts through
/api/sync/batch — so the switch is finishing the job, not starting it.
docs/webhook-migration.md is the plan, file by file.

UNTIL THEN
----------
This module stays wired up because the deployed plugin still points at it. Add
nothing to it, and route no new work through it.

The one thing that was added, and why it is not a violation of that rule: the
embedding this path pays for now writes a usage_events row through
usage_service, which reads the tenant off a request-scoped context that only an
auth chokepoint sets — and this path has no chokepoint. So the spend it was
already causing was landing nowhere. _webhook_usage_context() below resolves
that context from the STORE instead of from a key, purely so the money is
attributable while the plugin is still pointing here. It routes no new work
anywhere, and it is deleted with the rest of this file. See its own comment
block.
"""

import hmac
import hashlib
import base64
import logging
import os
import json
import anyio
from contextlib import nullcontext
from datetime import datetime, timezone
from fastapi import APIRouter, Request, HTTPException, Header, Query, Depends
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import bindparam, text
from backend.app.services.embedder import embed_document
from backend.app.services.qdrant_service import (
    upsert_product,
    delete_product,
    get_client_product_count,
    product_exists,
    retrieve_content_by_entity_ids,
)
from backend.app.services.cache_service import invalidate_client_results
from backend.app.services.license_service import increment_ingest_count, validate_license_key, get_client_license
from backend.app.services.database import get_db
from backend.app.services.product_service import build_product_text, extract_payload  # ← import
from backend.app.services.llm_key_service import decrypt_key
from backend.app.services.embedding_key_service import (
    resolve_embedding_key,
    resolve_embedding_model,
)
from backend.app.wordpress.services.product_formatter import LEGACY_MANAGED_BY, MANAGED_BY
from backend.app.services import request_context, tenancy_service, usage_service
# _context_from_row is private and imported anyway. That is deliberate and it is
# the lesser of the two evils available: it is the function that DEFINES the
# resolved-context shape — licensing_service.CONTEXT_FIELDS is built by running
# it, and usage_service checks its own column list against that at import — so
# calling it is the only way to produce a context here that cannot drift away
# from the one resolve_key() produces. The alternative is hand-typing twenty-odd
# keys beside it, which is exactly the hand-copied-shape failure those two
# modules spend a page of docstring describing (a private _REQUIRED_CTX_FIELDS
# list that agreed with the schema only until somebody renamed something, and
# ate every billing row when they did).
#
# It is called with a Row from _STORE_SUBSCRIPTIONS_SQL below, whose column
# ALIASES are copied from resolve_key()'s own SELECT for the same reason. If the
# context grows a field this query does not supply, _context_from_row() raises
# AttributeError, which _webhook_usage_context() catches and reports as a
# context-shape failure — the row is then unattributed and loud, never
# attributed and wrong. Promote it to a public name if a second caller ever
# needs it; do not copy it.
from backend.app.services.licensing_service import (
    RESOLVABLE_STATUSES,
    _context_from_row,
)
from backend.app.services.tenancy_service import normalise_domain

router    = APIRouter()

# The new lines below log rather than print, unlike the rest of this file. They
# are the other half of usage_service's NO CONTEXT / CALLER BUG warnings — you
# read them together or not at all — and print() gives a line no level and no
# logger name, so the one filter that finds the usage markers would not find
# these. The existing print()s are left alone.
logger = logging.getLogger(__name__)


def verify_signature(body: bytes, signature: str, secret: str) -> bool:
    mac = hmac.new(secret.encode("utf-8"), body, hashlib.sha256)
    expected = base64.b64encode(mac.digest()).decode("utf-8")
    return hmac.compare_digest(expected, signature)


# ─── Attributing the spend this path causes ───────────────────────────────────
#
# THE PROBLEM. These endpoints authenticate on an HMAC of clients.webhook_secret
# and never on a licence key, so they never pass through authorize_request() and
# nothing has ever bound a request-scoped tenant context for them. The v2 wiring
# then put a usage write underneath them: process_upsert() calls
# embed_document(), embedder._embed() calls usage_service.track(), and track()
# reads that ambient context. With nothing bound, every embedding a merchant's
# catalogue sync pays for is refused as NO CONTEXT and the spend is
# unattributed — a store re-indexing its whole catalogue produces no usage
# evidence at all, and the log fills with warnings pointing at a call site whose
# arguments were fine.
#
# THE IDENTITY IS RECOVERABLE EVEN THOUGH THE CREDENTIAL IS NOT. The handler
# already knows client_id (query string, and the HMAC is what proves it) and the
# store's domain (off the v1 licence it already reads for the collection name),
# and sites/subscriptions carry everything usage_events needs. So the context is
# resolved from the STORE rather than from a presented key.
#
# WHAT THIS IS NOT, AND MUST NOT BECOME. It is not authentication.
# resolve_key() returns a context because the caller proved possession of a key
# and passed five liveness gates; this returns one because the caller proved
# possession of the webhook secret, which already authorised exactly this —
# writing to this store's index — and nothing here widens that by a single
# byte. It names who pays for work the endpoint was already doing. Never call it
# from a path that HAS a key: that path has resolve_key(), which checks the
# licence, and using this instead would be a way to skip those checks.
#
# SCHEDULED TO GO AWAY WITH THE REST OF THIS FILE. See the module docstring and
# docs/webhook-migration.md. Once semantic-search-woo pushes with
# `Authorization: Bearer <licence key>` the way ai-product-qa-woo already does,
# these three endpoints stop existing, the request carries a key, resolve_key()
# answers, the chokepoint binds, and this whole block is deleted with them. It
# exists to keep the ledger honest during the wait, not as a second way to
# identify a tenant.
#
# That move is in progress rather than hypothetical: commit 4fa0698 ported
# ai-product-qa-woo's key handling into semantic-search-woo wholesale rather
# than inventing a second answer, which is the same direction the migration doc
# describes for the request shape. Expect the plugin side to land before this
# side, so do not build anything that assumes these handlers keep receiving
# traffic.
#
#
# WHICH SUBSCRIPTION A CATALOGUE SYNC BELONGS TO
# ----------------------------------------------
# A site can hold MORE THAN ONE subscription: catalog.py sells woo_search and
# woo_product_qa separately and a WooCommerce store can run both. This write
# serves neither in particular. There is one Qdrant collection per store, both
# modules read it, and one embedding of one product is what both of them will
# answer from — the cost is a property of the STORE, which is what schema_v2
# says about that scope in as many words: catalogue size is consumed once per
# store, against sites.catalogue_limit.
#
# usage_events has no site-only row. subscription_id is NOT NULL, so the ledger
# forces a per-store cost into a per-module bucket, and when a store runs two
# modules there is no honest way to pick one. So this refuses to pick:
#
#   exactly one subscription on the store  ->  that one. Not a guess: it is the
#                                              only module the store has.
#   more than one, one of them live        ->  the live one. A cancelled module
#                                              is not reading this index any
#                                              more; the live one is what the
#                                              sync serves.
#   more than one live, or more than one
#   matching site                          ->  NO CONTEXT, and a WARNING naming
#                                              every candidate.
#   none                                   ->  NO CONTEXT, quietly. That is the
#                                              ordinary state of a store that
#                                              has no v2 rows yet, and it is
#                                              every store today.
#
# The cost of refusing is bounded; the cost of guessing is not. A refused row
# still logs its provider, model, tokens and cost through usage_service's NO
# CONTEXT line, so the spend stays recoverable from the log. A row attributed to
# the wrong module is indistinguishable from a real one forever after and lands
# in usage_counters.total_cost for a subscription that did not incur it. Note
# what is and is not at risk: an embedding row is billable=False, so
# billable_requests never moves and no merchant's quota is drawn down either
# way. The damage from a wrong pick is confined to per-product cost reporting —
# which is the entire reason the site/subscription split exists, so it is not a
# small thing to get wrong.
#
# Two narrowings considered and rejected. Filtering candidates to WooCommerce
# products cannot separate the case that is actually ambiguous (two Woo modules
# on one Woo store) and would only paper over a data error in the other
# direction. Splitting the cost across the live subscriptions would put
# fractional rows in a ledger that is one row per model call by design, and no
# provider invoice can be reconciled against invented halves.
#
# LIVENESS IS A TIE-BREAKER HERE, NOT A GATE, and that is the one place this
# deliberately does not mirror resolve_key(). That function answers "may this
# caller act", so a cancelled subscription or a deactivated site is a refusal.
# This one answers "who spent this money", and the money was spent either way —
# discarding the only candidate because the store lapsed would lose the record
# of spend precisely when somebody most wants it. So every subscription counts
# as a candidate, and status/expiry are consulted only to break a tie.
#
# kind='sync', AND IT IS NOT SETTABLE FROM HERE. The only usage write this path
# can reach is embedder._embed() by way of embed_document(), and embedder
# derives kind from task_type, not from the caller's label: RETRIEVAL_DOCUMENT
# is reachable only through embed_document(), so every row a webhook delivery
# produces is KIND_SYNC and cannot be anything else. That is the right answer —
# a webhook delivery is a catalogue sync, never shopper traffic — and it is
# worth stating because the guarantee lives in another module. If a serving
# writer (a completion, a rerank) is ever called from inside the bound block in
# process_upsert(), it will pass its own kind and file indexing spend as serve;
# keep those calls out of that block.

# Column ALIASES copied from licensing_service.resolve_key()'s SELECT, because
# _context_from_row() reads the row by those names. The licence columns are
# NULL: there is no licence in play here, and inventing an id for one would put
# a fiction on the context. Nothing reads them — usage_events stamps the six
# tenant fields and licence_id is not one of them.
#
# No liveness predicate in the WHERE clause on purpose; see the tie-breaker note
# above. status and expires_at come back as data so the decision is made in
# Python where it can be logged.
_STORE_SUBSCRIPTIONS_SQL = text("""
    SELECT
        NULL               AS licence_id,
        NULL               AS key_hash,
        NULL               AS licence_key,
        NULL               AS licence_expires_at,

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

        c.id               AS client_id,
        c.name             AS client_name
    FROM sites si
    JOIN subscriptions s ON s.site_id = si.id
    JOIN products p      ON p.code    = s.product_code
    JOIN clients c       ON c.id      = si.client_id
    WHERE si.client_id = :client_id
      AND si.domain IN :domains
    ORDER BY s.product_code
""").bindparams(bindparam("domains", expanding=True))


def _utcnow() -> datetime:
    """Naive UTC, matching licensing_service._utcnow().

    The subscription expiry columns are DATETIME holding naive UTC and are
    compared against a naive UTC value everywhere else in the codebase; an aware
    datetime here would compare against a value in a different frame, or raise.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _store_domains(license_data: dict) -> list:
    """The forms of this store's host that could be sitting in sites.domain.

    Both are tried because the two tables disagree by design. The v1 licence
    carries allowed_domain exactly as it was typed at issue time — never
    lowercased, never www-stripped — while sites.domain is normalised by
    tenancy_service.normalise_domain(). Matching on only one of them silently
    finds nothing for every tenant whose stored URL has a capital letter, a
    www. or a port, and "found nothing" here means the spend goes unattributed
    with no indication that a string comparison is the reason.

    Both go into one IN clause rather than two queries: the pair identifies one
    store, and if it somehow matched two sites the caller refuses to attribute
    anything anyway.
    """
    raw = (license_data or {}).get("domain") or ""
    candidates = []
    if raw:
        candidates.append(raw)
    try:
        normalised = normalise_domain(raw)
    except ValueError:
        # normalise_domain() is written for a signup form and raises on anything
        # that is not a usable host. That is not a reason to give up: the raw
        # value may still match sites.domain verbatim, and the query decides.
        normalised = ""
    if normalised and normalised not in candidates:
        candidates.append(normalised)
    return candidates


def _is_live(row, now: datetime) -> bool:
    """Is this subscription one the store is currently paying for?

    The same two conditions resolve_key() applies — an allow-list of statuses,
    and an expiry that has not passed — used here only to break a tie between
    several subscriptions on one store. Never to reject the only candidate.
    """
    if row.status not in RESOLVABLE_STATUSES:
        return False
    return row.subscription_expires_at is None or row.subscription_expires_at >= now


def _webhook_usage_context(db: Session, client_id: str, license_data: dict) -> Optional[dict]:
    """The tenant context to attribute this delivery's spend to, or None.

    None is a legitimate and, today, the usual answer: it means this store has
    no v2 rows yet, or it has two modules and no honest way to choose between
    them. The caller then runs the ingest with nothing bound, usage_service
    refuses the row as NO CONTEXT at WARNING with the amounts on the line, and
    the spend stays recoverable from the log. That is the designed degradation,
    not a failure — an unattributed row you can find beats a confidently wrong
    one that you cannot.

    NEVER RAISES. An accounting lookup must not be the reason a merchant's
    product stops being indexed, which is the same rule usage_service states for
    itself. Every failure below returns None and says so.

    The returned dict is the shape licensing_service.resolve_key() produces,
    plus an interaction_id. It deliberately carries no `license_key` (there is
    no key on this path — that is the whole reason this file is being retired)
    and no `auth_path` (neither resolver answered). Nothing that reads the
    ambient context wants either: usage_service.record() is the only consumer,
    and it reads the six tenant fields and interaction_id.
    """
    domains = _store_domains(license_data)
    if not domains:
        logger.debug(
            "webhook usage: no domain on the v1 licence for client %s, so the "
            "site cannot be identified; this delivery's embedding spend will be "
            "unattributed.",
            client_id,
        )
        return None

    try:
        rows = db.execute(
            _STORE_SUBSCRIPTIONS_SQL,
            {"client_id": client_id, "domains": domains},
        ).fetchall()
    except Exception as exc:
        # Read-only and outside anything this request has half-written, so
        # swallowing it costs the attribution and nothing else. Raising would
        # make the billing lookup the reason a product failed to index, which is
        # the one thing this must never be.
        logger.warning(
            "webhook usage: resolving the tenant for client %s failed (%s: %s). "
            "The ingest continues; this delivery's embedding spend will be "
            "unattributed.",
            client_id, type(exc).__name__, exc,
        )
        return None

    if not rows:
        # The ordinary state during the dual-read window, and DEBUG for the same
        # reason usage_service logs NO CONTEXT at WARNING rather than ERROR: it
        # fires on legitimate traffic for every store that has not been migrated
        # onto sites/subscriptions, which today is all of them. usage_service
        # already emits one WARNING per refused row carrying the amounts, so a
        # second line here would only double the volume of the same finding.
        logger.debug(
            "webhook usage: no site/subscription rows for client %s at %s. "
            "Expected until this store is migrated onto the v2 tables; this "
            "delivery's embedding spend will be unattributed.",
            client_id, domains,
        )
        return None

    # More than one SITE matched. Only reachable when a client holds two site
    # rows whose domains differ by case, a www. or a port — which
    # uq_sites_client_domain permits and which is the same store written twice.
    # Liveness must not be used to break this tie: it would answer a question
    # about WHICH STORE with evidence about which module, and pick a store on a
    # coincidence.
    site_ids = sorted({row.site_id for row in rows})
    if len(site_ids) > 1:
        logger.warning(
            "webhook usage: AMBIGUOUS STORE - client %s has %d site rows "
            "matching %s (%s), so this delivery cannot be attributed to one of "
            "them. Its embedding spend will be unattributed rather than booked "
            "against a guess. Merge the duplicate site rows.",
            client_id, len(site_ids), domains, ", ".join(site_ids),
        )
        return None

    if len(rows) > 1:
        now = _utcnow()
        live = [row for row in rows if _is_live(row, now)]
        if len(live) != 1:
            # THE CASE THIS FUNCTION EXISTS TO REFUSE. A catalogue sync feeds
            # the one index both modules read, so it belongs to the store and
            # not to either subscription, and usage_events has no row shape that
            # can say that. Picking one would book a merchant's indexing cost
            # against a product that did not incur it, in a table whose whole
            # purpose is per-product cost, and nothing downstream could ever
            # tell that row from a real one.
            #
            # Named at WARNING with every candidate so the NO CONTEXT volume
            # from this path is attributable to a decision somebody made on
            # purpose rather than read as migration noise.
            logger.warning(
                "webhook usage: AMBIGUOUS - site %s (client %s, %s) holds %d "
                "subscriptions, %d of them live: %s. A webhook ingest is a "
                "catalogue sync that serves whichever module reads the index, "
                "so it belongs to no single subscription. Leaving this "
                "delivery's embedding spend unattributed rather than booking it "
                "against one of them; usage_service will log the amounts.",
                site_ids[0], client_id, domains, len(rows), len(live),
                ", ".join(f"{row.product_code}/{row.subscription_id}({row.status})" for row in rows),
            )
            return None
        rows = live

    row = rows[0]

    try:
        context = _context_from_row(row)
    except Exception as exc:
        # The context grew a field this query does not select, or changed shape
        # under it. ERROR because it is a code defect that will lose every row
        # from this path until somebody edits the SELECT above — the same
        # distinction usage_service draws between CALLER BUG and DATABASE, and
        # for the same reason: retrying will not help.
        logger.error(
            "webhook usage: CONTEXT SHAPE - licensing_service._context_from_row "
            "could not build a context from _STORE_SUBSCRIPTIONS_SQL (%s: %s). "
            "That query's column aliases have drifted from the ones "
            "resolve_key() selects; add the missing column here. Every webhook "
            "delivery's spend is unattributed until it is fixed.",
            type(exc).__name__, exc,
        )
        return None

    # One id per webhook delivery, which is honest grouping of what we can see:
    # WooCommerce fires one delivery per product save and this handler embeds
    # once, so the id threads exactly one row. It is minted rather than left
    # NULL because record() reads NULL as "a write site failed to thread an id",
    # which is a real finding and should not be manufactured here. If
    # WooCommerce's own delivery id turns out to be on the wire, feed it through
    # request_context.interaction_id_from_header() instead of minting — that
    # would tie the row to the delivery record in the merchant's own webhook
    # log. Do not read a header into this without that sanitiser: the value ends
    # up in log lines, and a newline in it forges a whole log entry.
    context["interaction_id"] = usage_service.new_interaction_id()
    return context


def _has_full_woo_payload(client_id: str, domain: str, product_id: str) -> bool:
    """Was this product's point written by the full WooCommerce formatter?

    Every WooCommerce writer resolves the same per-tenant collection from the
    same licence, and a product's point id is derived from
    (client_id, content_type, product_id) — so they all write to the SAME
    record. Both plugins' PHP syncs now produce identical bytes, so between
    those two the write order stopped mattering.

    This webhook is the exception, and it cannot be brought into line. It runs
    off WooCommerce's REST product object, which carries no variation detail,
    no merchant notes, and attribute display labels with no taxonomy code — so
    the attributes it does carry key the payload differently. There is no way
    to write this path that produces the same point.

    Overwriting the full record with this one is a silent downgrade: search
    keeps working, and "what sizes does this come in" starts answering "I don't
    have that information" with nothing logged anywhere to explain it. So when
    the marker is present, this webhook stands down and lets the plugin's own
    realtime sync — which posts the full payload — do the update instead.

    Fails OPEN. If the lookup errors, indexing something is better than
    indexing nothing: a store must never stop syncing because a Qdrant read
    hiccuped.
    """
    try:
        hits = retrieve_content_by_entity_ids(
            client_id=client_id,
            domain=domain,
            entity_ids=[product_id],
            content_types=["product"],
            limit=1,
        )
    except Exception as exc:
        print(f"⚠️  payload check failed for {product_id}: {exc} — proceeding")
        return False

    # Both values accepted: points written before the two formatters were
    # merged carry the old marker, and they are just as full as the new ones.
    return bool(hits) and str(hits[0].get("managed_by") or "") in (MANAGED_BY, LEGACY_MANAGED_BY)


def process_upsert(product: dict, action: str, client_id: str, db: Session, license_data: dict = None, llm_api_key_encrypted: str = None, embedding_api_key_encrypted: str = None, embedding_provider: str = None, embedding_model: str = None) -> dict:
    """
    Shared logic for created + updated webhooks.
    Both do the same thing — embed and upsert.
    """
    product_id = str(product["id"])

    if product.get("type") == "variation":
        return {"status": "skipped", "reason": "variation"}

    # Resolved BEFORE the unpublish branch, not after it. Removing a point
    # needs the domain that names the collection, and the domain only exists on
    # license_data — see the fix immediately below.
    #
    # get_client_license is DEPRECATED and this is the call that keeps it
    # alive: license_data["license_key"] is the plaintext key, which is the KEK
    # for the encrypted provider keys further down. Read the module header
    # before touching this.
    if not license_data:
        license_data = get_client_license(db, client_id)

    if product.get("status") != "publish":
        # qdrant_service.delete_product is (client_id, domain, product_id).
        # This was called with two arguments, so every unpublished product
        # raised TypeError, the endpoint's blanket `except Exception` turned it
        # into a 500, and the point was never removed — a product pulled out of
        # `publish` kept answering shoppers from the index indefinitely.
        #
        # Nothing else covered it: SSW_Sync::sync_single() in
        # semantic-search-woo deliberately stands down for a non-published
        # product on the stated grounds that "removal is the product-updated
        # webhook's job". It was not doing that job. The same two-argument call
        # The same two-argument call was live at routers/ingest.py and
        # routers/magento.py; ingest.py has since been deleted as dead code and
        # magento.py's copy is somebody else's file.
        # Asked BEFORE the delete: delete_product() reports nothing about
        # whether it removed anything, and an unpublish webhook for a product
        # that was never indexed is ordinary. Decrementing on that would walk
        # the counter down every time one arrived.
        _was_indexed = product_exists(client_id, license_data["domain"], product_id)
        delete_product(client_id, license_data["domain"], product_id)
        invalidate_client_results(client_id)
        if _was_indexed:
            tenancy_service.adjust_indexed_items_for(
                db, {"client_id": client_id, "domain": license_data["domain"]}, -1
            )
        print(f"🗑️  Webhook [{action}]: removed product {product_id}")
        return {"status": "removed", "product_id": product_id}

    exists = product_exists(client_id, license_data["domain"], product_id)

    # Stand down if the point already holds the full WooCommerce payload — see
    # the helper above. Checked before the embed so a store that stands down
    # isn't paying for embeddings it then throws away.
    if exists and _has_full_woo_payload(client_id, license_data["domain"], product_id):
        print(f"⏭️  Webhook [{action}]: {product_id} holds the full payload — not overwriting")
        return {"status": "skipped", "reason": "full_payload_present", "product_id": product_id}

    # Only block NEW products — an update to something already indexed adds
    # nothing to the count and must never be refused for capacity.
    if not exists:
        _ok, _current, _limit = tenancy_service.check_catalogue_headroom(
            db, {"client_id": client_id, "domain": license_data["domain"],
                 "catalogue_limit": license_data.get("product_limit")}, 1
        )
        if not _ok:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Catalogue limit reached. This store holds {_current:,} of "
                    f"{_limit:,} indexed items. Remove content you no longer "
                    f"need, or move to a larger plan."
                ),
            )
    # For updates, check if we're adding a new product or updating existing
    # If product doesn't exist in vector store, count it as new
    # if current_count >= license_data["product_limit"]:
    #     raise HTTPException(
    #         status_code=400,
    #         detail=f"Product limit exceeded. Current: {current_count}, Limit: {license_data['product_limit']}"
    #     )

    # Everything from here down is the ingest, and it is the only part of this
    # function that spends money. Resolved now rather than at the top so a
    # variation, an unpublish or a stand-down costs no query — those branches
    # returned above and none of them embeds anything.
    ctx = _webhook_usage_context(db, client_id, license_data)

    # THE PAIRED SETTER, not bind_context(). request_context() is a context
    # manager that restores whatever was bound on the way out, and it is the
    # right one here for a reason the chokepoints do not have: they RETURN their
    # license_data and let the request's own context copy expire around it,
    # while this owns a block with a beginning and an end. A bare bind would
    # leave the value set for whatever else runs on this worker thread inside
    # the same context copy after process_upsert() returns — and the endpoints
    # below catch broadly and keep going, so "after process_upsert() returns" is
    # a real place with real code in it. Anything that could write a usage row
    # there would book this store's identity onto somebody else's spend.
    #
    # nullcontext() when there is no context, deliberately: binding an empty or
    # part-filled dict is exactly the shape usage_service had to grow a
    # _is_tenant_shaped() check to survive, and it turns "nobody could say who
    # spent this" (NO CONTEXT, WARNING, correct) into "a caller passed a broken
    # dict" (CALLER BUG, ERROR, wrong, and it names an innocent call site).
    # Leaving the scope genuinely empty is the honest representation of not
    # knowing.
    #
    # The whole ingest is inside the block, not just the embed_document() line.
    # Nothing else in here writes usage today, and wrapping the region rather
    # than the one call means the next writer added to it is attributed without
    # anybody having to notice.
    with (request_context.request_context(ctx) if ctx is not None else nullcontext()):
        # The merchant's embedding key if they configured one, else their LLM
        # key. The raw blobs are no longer printed — a webhook fires on every
        # product save, and this was writing an encrypted key into the log on
        # each one.
        #
        # THIS IS THE LINE THAT CANNOT BE PORTED. The third argument is the
        # merchant's plaintext licence key, read out of license_keys.license_key
        # by get_client_license above, and llm_key_service.decrypt_key turns it
        # into the AES key as sha256(license_key). v2 stores only a SHA-256
        # hash, so there is no plaintext left to read and no way to derive the
        # KEK on a request that never presented a key. Every other caller of
        # resolve_embedding_key takes the key from the request — routers/sync.py
        # and routers/ingest.py from the body, the WordPress namespace from the
        # Authorization header — and that is what this path has to become.
        #
        # Note what the block above does NOT fix: the context it binds is
        # resolved from the store, so it can name who pays, but it cannot
        # produce a KEK. Attribution and decryption are separate problems and
        # only the first one is solvable from this side of the wire.
        embedding_api_key = resolve_embedding_key(
            embedding_api_key_encrypted,
            llm_api_key_encrypted,
            license_data["license_key"],
        )
        embedding_model = resolve_embedding_model(embedding_model, embedding_provider)

        # Uses product_service — raw WooCommerce format with nested categories/tags/attributes
        text    = build_product_text(product)
        # kind='sync' on the row this writes, and it is not passed from here:
        # embedder derives it from task_type, and embed_document() is the only
        # way to reach RETRIEVAL_DOCUMENT. A webhook delivery is a catalogue
        # sync, so that is the correct value — do not swap this for embed_query()
        # to reuse its cache, which would file indexing spend as shopper traffic.
        vector  = embed_document(text, embedding_api_key, client_id, model=embedding_model)
        payload = extract_payload(product)
        payload["embedded_text"] = text

        upsert_product(client_id, license_data["domain"], product_id, vector, payload)
        invalidate_client_results(client_id)
        if not exists:
            increment_ingest_count(db, client_id, count=1)
            # Exact, not recounted: this path already knows whether the point
            # was new, so a webhook costs no extra Qdrant round trip the way a
            # batch does.
            tenancy_service.adjust_indexed_items_for(
                db, {"client_id": client_id, "domain": license_data["domain"]}, +1
            )
        print(f"✅ Webhook [{action}]: indexed {product_id} - {product.get('name')}")

        return {"status": action, "product_id": product_id}

async def parse_webhook_body(request: Request) -> tuple:
    """
    Read body once. Return (raw_bytes, parsed_json_or_none).
    Handles WooCommerce ping (form-encoded) and real webhook (JSON).
    """
    body         = await request.body()
    content_type = request.headers.get("content-type", "")

    # WooCommerce ping — not JSON, just acknowledge it
    if "application/json" not in content_type:
        return body, None

    if not body:
        return body, None

    try:
        return body, json.loads(body)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/webhook/product-created")
def product_created(
    request: Request,
    client_id: str = Query(...),   # ← reads ?client_id= from URL
    llm_api_key: Optional[str] = Query(None),  # ← reads ?llm_api_key= from URL
    embedding_api_key: Optional[str] = Query(None),  # ← reads ?embedding_api_key= from URL
    embedding_provider: Optional[str] = Query(None),
    embedding_model: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    x_wc_webhook_signature: Optional[str] = Header(None)
):
     # Verify client exists and is active
    client = db.execute(text("""
        SELECT id, webhook_secret
        FROM clients
        WHERE id = :client_id AND is_active = 1
    """), {"client_id": client_id}).fetchone()

    if not client:
        raise HTTPException(status_code=403, detail="Invalid client")
    
    body, product = anyio.from_thread.run(parse_webhook_body, request)

    # Ping request — just acknowledge
    if product is None:
        return {"status": "ok", "reason": "ping"}
    
    # Webhook signature is required for security
    if not x_wc_webhook_signature:
        raise HTTPException(status_code=401, detail="Webhook signature header missing")

    # Verify webhook signature - REQUIRED for security
    secret = client.webhook_secret

    if not secret:
        raise HTTPException(status_code=500, detail="Webhook secret not registered")

    if not verify_signature(body, x_wc_webhook_signature, secret):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    try:
        return process_upsert(
            product=product,
            action="created",
            client_id=client_id,
            db=db,
            llm_api_key_encrypted=llm_api_key,
            embedding_api_key_encrypted=embedding_api_key,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
        )
    except HTTPException as e:
        # Re-raise HTTPExceptions to preserve status codes
        raise e
    except Exception as e:
        print(f"❌ Webhook [created] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/webhook/product-updated")
def product_updated(
    request: Request,
    client_id: str = Query(...),   # ← reads ?client_id= from URL
    llm_api_key: Optional[str] = Query(None),  # ← reads ?llm_api_key= from URL
    embedding_api_key: Optional[str] = Query(None),  # ← reads ?embedding_api_key= from URL
    embedding_provider: Optional[str] = Query(None),
    embedding_model: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    x_wc_webhook_signature: Optional[str] = Header(None)
):
    # Verify client exists and is active
    client = db.execute(text("""
        SELECT id, webhook_secret
        FROM clients
        WHERE id = :client_id AND is_active = 1
    """), {"client_id": client_id}).fetchone()

    if not client:
        raise HTTPException(status_code=403, detail="Invalid client")
    
    body, product = anyio.from_thread.run(parse_webhook_body, request)

    if product is None:
        return {"status": "ok", "reason": "ping"}
    
    # Webhook signature is required for security
    if not x_wc_webhook_signature:
        raise HTTPException(status_code=401, detail="Webhook signature header missing")

    # Verify webhook signature - REQUIRED for security
    secret = client.webhook_secret

    if not secret:
        raise HTTPException(status_code=500, detail="Webhook secret not registered")

    if not verify_signature(body, x_wc_webhook_signature, secret):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    try:
        return process_upsert(
            product=product,
            action="updated",
            client_id=client_id,
            db=db,
            llm_api_key_encrypted=llm_api_key,
            embedding_api_key_encrypted=embedding_api_key,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
        )
    except HTTPException as e:
        # Re-raise HTTPExceptions to preserve status codes
        raise e
    except Exception as e:
        print(f"❌ Webhook [updated] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/webhook/product-deleted")
def product_deleted(
    request: Request,
    client_id: str = Query(...),   # ← reads ?client_id= from URL
    db: Session = Depends(get_db),
    x_wc_webhook_signature: Optional[str] = Header(None)
):
    # Verify client exists and is active
    client = db.execute(text("""
        SELECT id, webhook_secret
        FROM clients
        WHERE id = :client_id AND is_active = 1
    """), {"client_id": client_id}).fetchone()

    if not client:
        raise HTTPException(status_code=403, detail="Invalid client")
    
    body, product = anyio.from_thread.run(parse_webhook_body, request)

    if product is None:
        return {"status": "ok", "reason": "ping"}
    
    # Webhook signature is required for security
    if not x_wc_webhook_signature:
        raise HTTPException(status_code=401, detail="Webhook signature header missing")

    # Verify webhook signature - REQUIRED for security
    secret = client.webhook_secret

    if not secret:
        raise HTTPException(status_code=500, detail="Webhook secret not registered")

    if not verify_signature(body, x_wc_webhook_signature, secret):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    product_id = str(product.get("id", ""))
    if not product_id:
        return {"status": "skipped", "reason": "no product id"}

    # No tenant context is bound around this one, and that is not an oversight.
    # Removing a point costs nothing: no embedding, no completion, no usage
    # write anywhere below this line — delete_product() and
    # invalidate_client_results() are a Qdrant delete and a Redis delete. There
    # is no spend to attribute, so binding a context here would resolve a site
    # on every deletion to hand it to nobody. If a usage write is ever added to
    # this handler, wrap it the way process_upsert() wraps its ingest.
    try:
        # Get license data to retrieve domain. DEPRECATED call — this one wants
        # nothing but the domain, so under v2 it becomes
        # resolve_key(db, presented_key)["collection_name"] once the plugin
        # presents a key. It cannot simply be repointed at the new schema while
        # the caller is WooCommerce, because WooCommerce presents no key.
        license_data = get_client_license(db, client_id)
        _was_indexed = product_exists(client_id, license_data["domain"], product_id)
        delete_product(client_id, license_data["domain"], product_id)
        invalidate_client_results(client_id)
        if _was_indexed:
            tenancy_service.adjust_indexed_items_for(
                db, {"client_id": client_id, "domain": license_data["domain"]}, -1
            )
        print(f"🗑️  Webhook [deleted]: removed product {product_id}")
        return {"status": "deleted", "product_id": product_id}
    except Exception as e:
        print(f"❌ Webhook [deleted] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
