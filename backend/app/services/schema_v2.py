"""
The v2 billing schema, as CREATE TABLE text. One constant per table.

WHY THIS FILE EXISTS SEPARATELY FROM THE MIGRATION
--------------------------------------------------
The v1 schema is currently described in three places that disagree: the stale
dump in init/01-schema.sql (missing license_keys.product_code and platform),
scripts/create_token_usage_table.py (which has a query_type the live enum does
not), and whatever ALTERs happen to have been run by hand. Anyone asking "what
does this database actually look like" has to read migration control flow to
find out. That is how the wp_product_qa rows went unbilled for months: the
Python allowlist and the MySQL enum drifted and nobody could see both at once.

So the DDL lives here as flat, readable strings, and the migration is nothing
but the control flow that applies them. If you want to know the shape of the
schema, this file is the answer and there is no second answer.

THE TWO SCOPES
--------------
Every module installed on one store shares ONE Qdrant collection, named per
(client, domain). That single fact forces the whole layout:

  * catalogue size is consumed once per STORE   -> `sites.catalogue_limit`
  * licences and request quota are per MODULE   -> `subscriptions.request_limit`

They cannot collapse into one level. A store running AIChatbot, AIProductQA and
AISearch indexes its catalogue once and pays for it once, but buys three
module licences with three separate request allowances. Anything that tries to
derive the catalogue ceiling from the subscriptions (max of their plans, say)
breaks the moment a module is cancelled: the ceiling drops below what is
already indexed and there is no clean resolution.

CONVENTIONS THAT ARE LOAD-BEARING, NOT TASTE
--------------------------------------------
COLLATION. `clients.id` is varchar(36) utf8mb4_general_ci. InnoDB refuses a
foreign key whose referencing column has a different character set or collation
from the referenced one, and it refuses it with errno 3780 and a message that
names neither column. Every VARCHAR here therefore spells out
COLLATE utf8mb4_general_ci rather than inheriting a server default that may not
match. The one deliberate exception is licences.key_hash - see there.

UUID COLUMNS ARE VARCHAR(36). Not CHAR(36), not BINARY(16). The referencing and
referenced columns of a foreign key must have the same type, clients.id is
already varchar(36), and every id in this codebase is minted app-side with
str(uuid.uuid4()). A tighter type would buy a few bytes and cost a class of
constraint-creation failures that only shows up at migration time.

NO ENUMS. Anywhere. status, kind, key_owner, call_type, environment, plan codes
are all VARCHAR. The vocabulary lives in Python. This is not a style
preference: token_usage_tracking.query_type is an enum, token_usage_service
keeps a parallel Python allowlist, the two drifted, and every WooCommerce
Product Q&A answer failed its INSERT into a MySQL warning that nobody read.
A label the database has not seen before must degrade to a row with an
unfamiliar string in it - findable with one SELECT DISTINCT - never to a lost
billing row.

DECIMAL, NEVER FLOAT, for money. FLOAT cannot represent 0.1 and a month of
summed per-call costs would not reconcile against the provider invoice.
"""

from __future__ import annotations

# ── Shared type notes referenced from the DDL below ──────────────────────────
#
# Checked by the migration before it creates anything. If the live clients.id
# does not match this, every foreign key below will be rejected and the
# migration would leave a half-built schema behind, so it aborts instead.
EXPECTED_CLIENT_ID_COLLATION = "utf8mb4_general_ci"


# ── clients ──────────────────────────────────────────────────────────────────
#
# Who pays. Deliberately thin: a client is a billing relationship and nothing
# else. It loses v1's `plan` column, because a plan is now a property of a
# subscription and a client with three modules has three of them; and it loses
# `webhook_secret`, which moves to `sites` - a WooCommerce webhook is registered
# by a store, not by a customer, so a client with two Woo stores was previously
# overwriting one store's secret with the other's on every re-registration.

CLIENTS_TABLE = """
CREATE TABLE `clients` (
  `id`         VARCHAR(36)  COLLATE utf8mb4_general_ci NOT NULL,
  `name`       VARCHAR(255) COLLATE utf8mb4_general_ci NOT NULL,
  `email`      VARCHAR(255) COLLATE utf8mb4_general_ci NOT NULL,
  `company`    VARCHAR(255) COLLATE utf8mb4_general_ci DEFAULT NULL
               COMMENT 'Legal/trading name for invoices. Nullable: onboarding does not collect it yet.',
  `is_active`  TINYINT(1) NOT NULL DEFAULT 1
               COMMENT 'Account-level kill switch. One of the five liveness gates on the auth path.',
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  -- One account per email. Onboarding is find-or-create on this: a returning
  -- customer buying a second module must land on their existing client row,
  -- not be rejected as a duplicate the way create_client() used to reject them.
  UNIQUE KEY `uq_clients_email` (`email`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci
"""


# ── products ─────────────────────────────────────────────────────────────────
#
# Reference data, seeded from backend/app/services/catalog.py. Every column
# except `is_sellable` is a mirror and must NEVER be hand edited: catalog.py is
# what the onboarding UI renders, what licence minting reads, and what usage
# rows are stamped with, so a row here that catalog.py does not know about is a
# product nothing can sell, authorise, or bill.
#
# `is_sellable` is the one exception and it is deliberate. catalog.py defines no
# such field, so the seeder writes 1 on insert and never touches it again -
# withdrawing a product is an operator UPDATE against this column, and it is the
# only edit anyone should ever make to this table by hand. If a `sellable` flag
# is ever added to catalog.PRODUCTS, delete this paragraph and seed it, because
# then there would be two answers to "is this product still on sale" and the
# operator's would win silently on a re-seed.
#
# It exists as a table at all only so subscriptions.product_code can be a real
# foreign key. That constraint is what makes it impossible to write a
# subscription - and therefore a usage_events row - against a product code that
# was typo'd or renamed.

PRODUCTS_TABLE = """
CREATE TABLE `products` (
  `code`        VARCHAR(64)  COLLATE utf8mb4_general_ci NOT NULL
                COMMENT 'catalog.PRODUCTS key. PERMANENT: written into billing history; add codes, never repurpose one.',
  `platform`    VARCHAR(32)  COLLATE utf8mb4_general_ci NOT NULL
                COMMENT 'catalog.PLATFORMS key. Derived from the product, never supplied independently.',
  `name`        VARCHAR(128) COLLATE utf8mb4_general_ci NOT NULL,
  `key_segment` VARCHAR(8)   COLLATE utf8mb4_general_ci NOT NULL
                COMMENT 'Human label inside a licence key (mchat, wpqa...). NOT a credential: never authorise on it.',
  `is_sellable` TINYINT(1) NOT NULL DEFAULT 1
                COMMENT 'Withdrawn products go false; existing subscriptions keep resolving, onboarding stops offering it.',
  PRIMARY KEY (`code`),
  -- catalog.assert_key_segments_unique() states the same rule in Python, but a
  -- Python guard only runs where somebody wired it into an import path, and as
  -- this is written that function is defined and never called. This UNIQUE key
  -- is the copy that always runs. Two products sharing a segment makes their
  -- keys visually indistinguishable, which is the entire reason the segment
  -- exists - and the collision is worse than cosmetic on the way in: an
  -- INSERT ... ON DUPLICATE KEY UPDATE that trips this key instead of the
  -- primary key updates the INCUMBENT product's row and never inserts the new
  -- one. seed_products does an explicit find-then-write for that reason.
  UNIQUE KEY `uq_products_key_segment` (`key_segment`),
  -- Onboarding renders the product picker filtered by platform.
  KEY `idx_products_platform` (`platform`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci
"""


# ── sites ────────────────────────────────────────────────────────────────────
#
# One row per store install. This is the STORAGE scope: the Qdrant collection,
# the catalogue ceiling, the indexed-item count and the webhook secret all
# belong to a store, not to a customer and not to a module.

SITES_TABLE = """
CREATE TABLE `sites` (
  `id`               VARCHAR(36)  COLLATE utf8mb4_general_ci NOT NULL,
  `client_id`        VARCHAR(36)  COLLATE utf8mb4_general_ci NOT NULL,
  `domain`           VARCHAR(255) COLLATE utf8mb4_general_ci NOT NULL
                     COMMENT 'Bare host, lowercased, no www, no port, no scheme. Must equal what DomainAuthorizer compares.',
  `platform`         VARCHAR(32)  COLLATE utf8mb4_general_ci NOT NULL,
  `platform_version` VARCHAR(32)  COLLATE utf8mb4_general_ci DEFAULT NULL
                     COMMENT 'Magento/WordPress version the merchant reported at onboarding. Support triage only.',
  `store_name`       VARCHAR(255) COLLATE utf8mb4_general_ci DEFAULT NULL,
  `collection_name`  VARCHAR(255) COLLATE utf8mb4_general_ci NOT NULL
                     COMMENT 'STORED, not derived. See the note below before you change how it is produced.',
  `environment`      VARCHAR(16)  COLLATE utf8mb4_general_ci NOT NULL DEFAULT 'development'
                     COMMENT 'development|production. Decides usage_events.key_owner at write time.',
  `index_plan`       VARCHAR(32)  COLLATE utf8mb4_general_ci NOT NULL
                     COMMENT 'INDEX_PLANS rung: free|small|medium|large. Bought once per site.',
  `catalogue_limit`  INT UNSIGNED NOT NULL
                     COMMENT 'Ceiling in logical entities. Comes from this row index_plan, NEVER from a subscription.',
  `indexed_items`    INT UNSIGNED NOT NULL DEFAULT 0
                     COMMENT 'Logical entities in the collection. Maintained at the Qdrant boundary; needs periodic reconcile.',
  `webhook_secret`   VARCHAR(255) COLLATE utf8mb4_general_ci DEFAULT NULL
                     COMMENT 'HMAC secret for this store WooCommerce webhooks. Moved off clients: webhooks are per store.',
  `is_active`        TINYINT(1) NOT NULL DEFAULT 1,
  `created_at`       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at`       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  -- REQUIRED INDEX 5. One install per (customer, store). This is the identity
  -- of a site: onboarding upserts on it, and the webhook path resolves through
  -- it. It also serves as the index InnoDB demands for fk_sites_client, since
  -- client_id is its leftmost column - so there is deliberately no separate
  -- idx_sites_client to maintain on every write.
  UNIQUE KEY `uq_sites_client_domain` (`client_id`, `domain`),
  -- Not in the brief, added on purpose. qdrant_service.get_collection_name
  -- replaces every non-alphanumeric with '_', so shop.example.com,
  -- shop-example-com and shop_example_com all produce the same collection.
  -- Two sites landing on one collection is a cross-tenant read of somebody
  -- else's catalogue. This makes that structurally impossible instead of
  -- merely unlikely, and it fails at INSERT time where it is cheap to notice.
  UNIQUE KEY `uq_sites_collection` (`collection_name`),
  CONSTRAINT `fk_sites_client` FOREIGN KEY (`client_id`) REFERENCES `clients` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci
"""

# collection_name is STORED rather than recomputed for one specific reason.
# Today the collection a request lands on is named from the JWT's `domain`
# claim, which is never lowercased and never www-stripped. The new
# sites.domain is normalised. If resolution recomputed the name from the
# normalised domain, every collection whose key was minted with a mixed-case
# or www-prefixed host would be renamed away from its own data - and a Qdrant
# read against a name that does not exist returns zero results, not an error.
# The store would simply go quiet, and nobody would find out until a merchant
# complained days later. Seed this column from the names Qdrant already holds.


# ── subscriptions ────────────────────────────────────────────────────────────
#
# One row per (site x product): the MODULE scope. This is the table that makes
# per-product billing possible at all. Three Magento modules call the identical
# backend endpoints, so the route a request arrived on can never identify the
# product; resolving a licence to exactly one subscription is the only thing
# that can.

SUBSCRIPTIONS_TABLE = """
CREATE TABLE `subscriptions` (
  `id`            VARCHAR(36) COLLATE utf8mb4_general_ci NOT NULL,
  `site_id`       VARCHAR(36) COLLATE utf8mb4_general_ci NOT NULL,
  `product_code`  VARCHAR(64) COLLATE utf8mb4_general_ci NOT NULL,
  `status`        VARCHAR(16) COLLATE utf8mb4_general_ci NOT NULL DEFAULT 'trial'
                  COMMENT 'trial|active|suspended|cancelled. A trial is a real subscription with a small request_limit.',
  `plan`          VARCHAR(32) COLLATE utf8mb4_general_ci NOT NULL
                  COMMENT 'MODULE_PLANS rung: trial|starter|growth|pro. Unrelated to sites.index_plan.',
  `request_limit` INT UNSIGNED NOT NULL
                  COMMENT 'Billable requests per calendar month. Checked against usage_counters.billable_requests.',
  `started_at`    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `expires_at`    DATETIME NULL DEFAULT NULL
                  COMMENT 'NULL means open-ended. Past means the resolver denies, same as a revoked licence.',
  `created_at`    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at`    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  -- REQUIRED INDEX 4. A store buys a given module once. v1 had no such
  -- constraint and onboarding had to hand-roll the check with a
  -- (client_id, allowed_domain, product_code) SELECT before every mint; that
  -- is a race, and this is the same predicate expressed where it cannot lose.
  UNIQUE KEY `uq_subscriptions_site_product` (`site_id`, `product_code`),
  -- Required by InnoDB for fk_subscriptions_product: product_code is the
  -- SECOND column of the unique key above, so that key cannot serve the
  -- constraint. Doubles as the index for "every store running AIChatbot".
  KEY `idx_subscriptions_product` (`product_code`),
  -- Deleting a site takes its subscriptions with it. Cancelling a module is
  -- status='cancelled', never a DELETE - the row has to survive so its
  -- usage_counters and its historical usage_events stay explicable.
  CONSTRAINT `fk_subscriptions_site` FOREIGN KEY (`site_id`) REFERENCES `sites` (`id`) ON DELETE CASCADE,
  CONSTRAINT `fk_subscriptions_product` FOREIGN KEY (`product_code`) REFERENCES `products` (`code`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci
"""

# WHY expires_at IS DATETIME WHEN EVERY OTHER TIME COLUMN HERE IS TIMESTAMP.
# Applies to subscriptions.expires_at above and licences.expires_at below, and
# to those two columns only.
#
# TIMESTAMP cannot represent an instant after 2038-01-19. Every other time
# column in this schema is written by the server as "now" and is therefore
# structurally in the past, but these two are handed a future date from Python:
# licensing_service.issue_licence takes an arbitrary valid_days, and an
# enterprise term of twelve years is not a hypothetical. Under strict SQL mode
# a date past the ceiling is error 1292 on the INSERT - the licence fails at
# the last step, after the plaintext key has already been minted and handed
# out. Non-strict mode is worse: the value is silently truncated and the
# licence quietly gets an expiry nobody chose, which is an authorisation
# decision made by an overflow.
#
# DATETIME has no epoch bound, and neither column wants TIMESTAMP's other
# behaviour either. TIMESTAMP converts on write and back on read using the
# session time zone; DATETIME stores the literal value. licensing_service
# writes naive UTC and compares in Python against naive UTC, so a literal store
# is exactly what it means.
#
# The wrinkle to know about: on a server whose session time zone is not UTC,
# these DATETIME columns hold UTC while the TIMESTAMP columns beside them
# (issued_at, started_at, created_at) read back in local time, so the two are
# in different frames within one row - a licence can display as issued after it
# expires. That is not caused by this choice; it is the same session time zone
# skew that already splits usage_counters.period from usage_events.created_at.
# The one fix for all of it is pinning the connection:
# create_engine(..., connect_args={"init_command": "SET time_zone='+00:00'"}).


# ── licences ─────────────────────────────────────────────────────────────────
#
# One row per issued key. Stores the SHA-256 hash and a displayable prefix, and
# never the key itself. v1 stored the whole plaintext JWT in a TEXT column,
# which made a database dump a handover of every customer's working credential.
#
# Consequence to understand before you touch anything downstream: the plaintext
# cannot be read back. Anything that used to recover a key from the database -
# onboarding handing a returning customer their existing key, the WooCommerce
# webhook path deriving the AES KEK for a merchant's wrapped LLM key - has to
# be redesigned, not reimplemented. See the migration's report.

LICENCES_TABLE = """
CREATE TABLE `licences` (
  `id`              VARCHAR(36) COLLATE utf8mb4_general_ci NOT NULL,
  `subscription_id` VARCHAR(36) COLLATE utf8mb4_general_ci NOT NULL,
  `key_hash`        CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL
                    COMMENT 'SHA-256 hex of the presented key, lowercase. The plaintext is never stored.',
  `key_prefix`      VARCHAR(32) COLLATE utf8mb4_general_ci NOT NULL
                    COMMENT 'czg_live_mchat_7Kq2 - enough to identify a key in a list, useless for guessing it.',
  `is_active`       TINYINT(1) NOT NULL DEFAULT 1,
  `issued_at`       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  -- DATETIME, not TIMESTAMP. See the note above SUBSCRIPTIONS_TABLE: this
  -- column is handed issue_licence's valid_days and a long-dated enterprise
  -- term would hit TIMESTAMP's 2038 ceiling, after the key was already minted.
  `expires_at`      DATETIME NULL DEFAULT NULL,
  `revoked_at`      TIMESTAMP NULL DEFAULT NULL
                    COMMENT 'Set on rotation. Kept rather than deleted so a support ticket about a dead key is answerable.',
  PRIMARY KEY (`id`),
  -- REQUIRED INDEX 1, and the hottest index in the system: one probe per
  -- authenticated request, before anything else happens.
  --
  -- ascii_bin, not the table's utf8mb4_general_ci, and this is deliberate on
  -- both counts. A hex digest is ascii by construction, so ascii stores it in
  -- 64 bytes instead of 256 and keeps the index four times smaller and hotter
  -- in the buffer pool. The _bin half is the important one: general_ci
  -- compares case-insensitively, so 'AB..' and 'ab..' would collide in this
  -- UNIQUE index and a lookup would match a hash that is not byte-identical.
  -- license_key.hash_key() returns hexdigest(), which is always lowercase.
  -- Never .upper() a hash on the way in or out or lookups silently miss and
  -- the merchant sees "License key not found".
  UNIQUE KEY `uq_licences_key_hash` (`key_hash`),
  -- "Show me the keys issued for this module" on the operator console, and the
  -- index InnoDB requires for the constraint below.
  KEY `idx_licences_subscription` (`subscription_id`),
  CONSTRAINT `fk_licences_subscription` FOREIGN KEY (`subscription_id`) REFERENCES `subscriptions` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci
"""


# ── usage_events ─────────────────────────────────────────────────────────────
#
# The ledger. Replaces token_usage_tracking, which carried client_id as its
# only tenant dimension and therefore could not tell you which of a store's
# three Magento modules spent the money.
#
# DENORMALISED ON PURPOSE, AND WITH NO FOREIGN KEYS. A billing row has to be
# readable years after the site was deleted, the subscription cancelled and the
# product withdrawn, without a five-table join to anything that may no longer
# exist. Foreign keys here would either block those deletions forever or
# cascade the evidence away. The identifiers are stamped at write time from the
# resolved licence context and are then frozen facts, not references.
#
# The write is also the hottest INSERT path in the codebase - embedder.py emits
# one row per embedded chunk, so a 25,000-item catalogue sync is a 25,000-row
# burst - which is a second reason not to make it pay for constraint checks.

USAGE_EVENTS_TABLE = """
CREATE TABLE `usage_events` (
  `id`              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  `client_id`       VARCHAR(36)  COLLATE utf8mb4_general_ci NOT NULL,
  `site_id`         VARCHAR(36)  COLLATE utf8mb4_general_ci NOT NULL,
  `subscription_id` VARCHAR(36)  COLLATE utf8mb4_general_ci NOT NULL,
  `product_code`    VARCHAR(64)  COLLATE utf8mb4_general_ci NOT NULL,
  `platform`        VARCHAR(32)  COLLATE utf8mb4_general_ci NOT NULL,
  `key_owner`       VARCHAR(16)  COLLATE utf8mb4_general_ci NOT NULL DEFAULT 'czargroup'
                    COMMENT 'czargroup|client, from sites.environment. Default is deliberate: an unstamped row books the spend as ours, never as the merchant. NEVER SUM total_cost across both values.',
  `kind`            VARCHAR(16)  COLLATE utf8mb4_general_ci NOT NULL
                    COMMENT 'sync|serve. NO DEFAULT: it is NOT derivable from call_type, so a writer that omits it must fail.',
  `billable`        TINYINT(1) NOT NULL DEFAULT 0
                    COMMENT 'TRUE on exactly one row per customer-visible action. That row is what quota counts.',
  `interaction_id`  VARCHAR(64)  COLLATE utf8mb4_general_ci DEFAULT NULL
                    COMMENT 'Groups every row of one shopper turn. Minted server-side; NULL means a write site failed to thread it.',
  `call_type`       VARCHAR(64)  COLLATE utf8mb4_general_ci NOT NULL
                    COMMENT 'chat_answer, embed_document, product_rerank... VARCHAR by hard-won experience, never an enum.',
  `provider`        VARCHAR(50)  COLLATE utf8mb4_general_ci NOT NULL DEFAULT '',
  `model`           VARCHAR(100) COLLATE utf8mb4_general_ci NOT NULL DEFAULT '',
  `input_tokens`    INT UNSIGNED NOT NULL DEFAULT 0,
  `output_tokens`   INT UNSIGNED NOT NULL DEFAULT 0,
  `total_tokens`    INT UNSIGNED NOT NULL DEFAULT 0,
  `input_cost`      DECIMAL(12,8) NOT NULL DEFAULT 0.00000000,
  `output_cost`     DECIMAL(12,8) NOT NULL DEFAULT 0.00000000,
  `total_cost`      DECIMAL(12,8) NOT NULL DEFAULT 0.00000000,
  `created_at`      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  -- REQUIRED INDEX 2. Quota reads and every tenant-facing usage panel are
  -- "this module, this window": WHERE subscription_id = ? AND created_at >= ?.
  -- subscription_id leads because it is the equality predicate; created_at
  -- second turns the range into a contiguous scan of that prefix.
  KEY `idx_usage_events_subscription_created` (`subscription_id`, `created_at`),
  -- REQUIRED INDEX 3. The operator-side pivot the whole rewrite exists for:
  -- "what does AI Chatbot cost across all tenants this month", which v1 could
  -- only approximate by guessing at query_type names. Note what this index
  -- does NOT serve: usage_service.usage_by_product GROUP BYs on product_code
  -- rather than filtering on it, so it has no equality predicate for this
  -- index to seek on. That query is the one below.
  KEY `idx_usage_events_product_created` (`product_code`, `created_at`),
  -- REQUIRED for the per-tenant usage panel. usage_service.usage_by_product is
  -- WHERE created_at >= ? AND created_at < ? AND client_id = ?, and it does a
  -- COUNT(DISTINCT site_id), so it reads every row it touches. Without a
  -- tenant-scoped index one merchant opening their own panel range-scans every
  -- tenant's rows for the month on the largest table in the database and
  -- filters client_id in the server. v1's token_usage_tracking carried exactly
  -- this key (idx_client_created); dropping it in v2 would have been a
  -- regression, not a simplification.
  KEY `idx_usage_events_client_created` (`client_id`, `created_at`),
  -- Reconstruct one shopper turn for a support ticket, and audit the rule that
  -- exactly one row per interaction carries billable=1. Without it both are
  -- full scans of the largest table in the database.
  KEY `idx_usage_events_interaction` (`interaction_id`),
  -- The operator console's platform-wide totals and daily series have no
  -- tenant predicate at all, only a time window.
  KEY `idx_usage_events_created` (`created_at`),
  -- The four invariants v1's token_usage_tracking enforced, carried forward
  -- deliberately rather than left to Python. usage_service does clamp and
  -- quantize every one of these before it binds them, but this file's whole
  -- thesis (lines 48-55) is that an invariant living only in a Python
  -- allowlist is how wp_product_qa went unbilled. A repair script, a second
  -- writer or a hand-written INSERT reaches this table without going through
  -- usage_service.record(), and a row where total_cost != input_cost +
  -- output_cost breaks reconciliation in the one direction nobody notices:
  -- the row still looks arithmetically plausible until somebody adds the
  -- columns up.
  --
  -- Names are prefixed with the table, unlike v1's bare chk_total_cost_match.
  -- CHECK constraint names are unique per DATABASE in MySQL 8.0, not per
  -- table, and token_usage_tracking keeps its four through the archive rename
  -- - reusing the v1 names would fail this CREATE with errno 3822 on any
  -- deployment that still holds the old ledger.
  --
  -- Known interaction, and it is the only way a well-behaved writer can trip
  -- the two match constraints. usage_service clamps each total to what the
  -- column can hold - total_tokens to min(in + out, 4294967295) for INT
  -- UNSIGNED, total_cost to the DECIMAL(12,8) ceiling - while the two halves
  -- are clamped independently, so a sum that overflows lands on the ceiling
  -- and stops equalling in + out. Both need a broken input to reach: billions
  -- of tokens reported on one call, or a single call costing four figures,
  -- each of which usage_service already logs at ERROR as a parsing or
  -- price-table bug. If either ever fires in anger the answer is to clamp the
  -- halves at the write site so their sum still fits, NOT to drop these
  -- constraints - a rejected INSERT is a lost billing row, and this file is
  -- built around never losing one.
  CONSTRAINT `chk_usage_events_total_cost` CHECK (`total_cost` = `input_cost` + `output_cost`),
  CONSTRAINT `chk_usage_events_total_tokens` CHECK (`total_tokens` = `input_tokens` + `output_tokens`),
  -- INT UNSIGNED already refuses a negative token count, so this pair is a
  -- backstop on that column and the real guard on the DECIMALs, which are
  -- signed and would happily store a negative cost.
  CONSTRAINT `chk_usage_events_tokens_non_negative` CHECK (`input_tokens` >= 0 AND `output_tokens` >= 0 AND `total_tokens` >= 0),
  CONSTRAINT `chk_usage_events_costs_non_negative` CHECK (`input_cost` >= 0 AND `output_cost` >= 0 AND `total_cost` >= 0)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci
"""

# One index that looks obviously necessary and is deliberately absent, so
# nobody adds it back without reading this:
#
#   (site_id, created_at)   - a store has at most one subscription per product,
#                             so "everything this store spent" is an IN over at
#                             most five subscription ids against the index that
#                             already exists. Not worth a sixth secondary index
#                             on the hottest INSERT path in the system.
#
# RETRY IDEMPOTENCY WAS TRADED AWAY, ON PURPOSE, AND THIS IS THE RECORD OF IT.
# v1's token_usage_tracking carried UNIQUE KEY request_id, which made
# double-counting a retried write physically impossible. usage_events has no
# such column: usage_service.record() mints no request id, and a nullable
# unique column would be decorative - MySQL permits unlimited NULLs in a UNIQUE
# index, so it would constrain only the writers that already bothered. The
# thing standing between a retried request and a double-billed row is therefore
# usage_service.record() being called exactly once per attempt, and nothing
# else. If a retrying caller is ever added on this path, the fix is a real
# NOT NULL request_id minted at the write site plus a UNIQUE key here - not a
# Python check, for the reason the paragraph above gives.
#
# On the column types: input/output/total_cost are DECIMAL(12,8) where the v1
# archive used DECIMAL(10,8). Every archived value fits, so the archive and the
# ledger stay directly comparable - which matters, because the archive is being
# kept specifically to size the new plan ladders - but a single expensive
# completion can no longer overflow the ~$100 ceiling that (10,8) imposed.
#
# created_at is TIMESTAMP to match the archive column for column, so a
# before/after comparison over the same time buckets has no type artefact in
# it. Known and dated: TIMESTAMP cannot represent an instant after 2038-01-19.
# Ordering rows WITHIN one interaction is done by `id`, not by created_at -
# several rows of a turn routinely land in the same second.


# ── usage_counters ───────────────────────────────────────────────────────────
#
# Monthly rollup, incremented by the same write that stamps billable=1. It
# exists so a quota check is one primary-key read instead of an aggregate over
# the ledger.
#
# The rule that makes it correct: usage_counters is a CACHE OF THE LEDGER, and
# it is only ever incremented from the billable row. v1's usage_logs was
# incremented from two endpoints out of fifteen, so the only quota counter in
# the system counted AI Search traffic and nothing else, and every chatbot
# tenant was structurally un-quotable. If you find yourself adding a second
# place that increments this table, that is the bug reappearing.
#
# There is deliberately NO ingest counter here. Sync volume is usage_events
# WHERE kind='sync', and catalogue size is sites.indexed_items. v1 conflated
# both into usage_logs.ingest_count, a per-client number describing a
# per-store fact, read by exactly one operator query.

USAGE_COUNTERS_TABLE = """
CREATE TABLE `usage_counters` (
  `subscription_id`   VARCHAR(36) COLLATE utf8mb4_general_ci NOT NULL,
  `period`            CHAR(7) COLLATE utf8mb4_general_ci NOT NULL
                      COMMENT 'YYYY-MM. Calendar month, not a rolling window: an invoice line has to name a month.',
  `billable_requests` INT UNSIGNED NOT NULL DEFAULT 0
                      COMMENT 'Count of usage_events rows with billable=1. Compared against subscriptions.request_limit.',
  `total_tokens`      BIGINT UNSIGNED NOT NULL DEFAULT 0
                      COMMENT 'BIGINT: 500k pro-plan requests at a few thousand tokens each overflows INT inside one month.',
  `total_cost`        DECIMAL(16,8) NOT NULL DEFAULT 0.00000000
                      COMMENT 'Wider than the ledger row on purpose - this is a sum of hundreds of thousands of them.',
  `updated_at`        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  -- Composite primary key, no surrogate id. The access pattern is entirely
  -- "this subscription, this month" - one read on the quota path, one
  -- INSERT ... ON DUPLICATE KEY UPDATE on the write path - and a surrogate id
  -- would only add a second index to maintain and a uuid to generate for a row
  -- that is never addressed any other way.
  PRIMARY KEY (`subscription_id`, `period`),
  -- The monthly billing run reads a whole period across every subscription;
  -- period is the second column of the PK so the PK cannot serve that.
  KEY `idx_usage_counters_period` (`period`),
  -- A counter for a subscription that no longer exists is meaningless, unlike
  -- a ledger row, which is evidence. Hence a cascade here and none there.
  CONSTRAINT `fk_usage_counters_subscription` FOREIGN KEY (`subscription_id`) REFERENCES `subscriptions` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci
"""


# ── Creation order ───────────────────────────────────────────────────────────
#
# Foreign key dependency order, and the migration walks it as-is. products
# first because subscriptions references it and it has no dependencies of its
# own; clients before sites; sites before subscriptions; subscriptions before
# both licences and usage_counters. usage_events references nothing, so its
# position is free - it sits with the other ledger table for readability.

V2_TABLES: list[tuple[str, str]] = [
    ("clients",        CLIENTS_TABLE),
    ("products",       PRODUCTS_TABLE),
    ("sites",          SITES_TABLE),
    ("subscriptions",  SUBSCRIPTIONS_TABLE),
    ("licences",       LICENCES_TABLE),
    ("usage_events",   USAGE_EVENTS_TABLE),
    ("usage_counters", USAGE_COUNTERS_TABLE),
]


# ── Bringing an existing v1 `clients` up to the v2 shape ─────────────────────
#
# clients is the one v2 table that already exists in every deployment, so it is
# reached by ALTER rather than CREATE. Kept here as data, next to the CREATE it
# has to converge on, because a reader comparing the two needs both in view.
#
# CONVERGE, do not merely extend. A fresh install runs CLIENTS_TABLE and an
# upgraded install runs these ALTERs, and if the two do not end up structurally
# identical then "what does this database look like" has two answers again -
# the exact condition the header of this file was written to end. So the three
# lists below cover every way v1's clients (init/01-schema.sql) differs from
# CLIENTS_TABLE, not just the column that is missing.

CLIENTS_ADD_COLUMNS: list[tuple[str, str]] = [
    (
        "company",
        "ADD COLUMN `company` VARCHAR(255) COLLATE utf8mb4_general_ci DEFAULT NULL "
        "COMMENT 'Legal/trading name for invoices. Nullable: onboarding does not collect it yet.' "
        "AFTER `email`",
    ),
]

# Columns that exist but are declared differently from CLIENTS_TABLE. Applied
# only when information_schema says the live column actually diverges - a
# MODIFY is a table rebuild, and running one on every migration re-run for a
# column that is already right is a cost with no buyer.
#
#   is_active - v1 declares it `tinyint(1) DEFAULT '1'`, i.e. NULLABLE. This is
#               not cosmetic and it is not a fresh-install-only concern: a
#               clients row carrying NULL here is treated as falsy by the auth
#               path and denied, so a customer would be refused with no
#               explanation and no column anyone would think to look at. The
#               DDL that makes NULL impossible is the fix, and it was never
#               being applied.
#
# The migration refuses this MODIFY while any row still holds NULL rather than
# backfilling one. Choosing 1 activates accounts that are being denied today;
# choosing 0 permanently deactivates accounts that v1's DEFAULT '1' says were
# meant to be live. That is a decision about who can log in, and it belongs to
# a human with the row in front of them.
CLIENTS_ALTER_COLUMNS: list[tuple[str, str]] = [
    (
        "is_active",
        "MODIFY COLUMN `is_active` TINYINT(1) NOT NULL DEFAULT 1 "
        "COMMENT 'Account-level kill switch. One of the five liveness gates on the auth path.'",
    ),
]

# Same unique key, different name: v1 calls it `email`, CLIENTS_TABLE calls it
# `uq_clients_email`. Renaming an index touches no rows and no data - it is
# metadata only - and it means a DESCRIBE of an upgraded database matches a
# DESCRIBE of a fresh one, so a shape diff has nothing spurious in it to
# explain away. (from, to, ddl); RENAME INDEX needs MySQL 5.7 or later.
CLIENTS_RENAME_INDEXES: list[tuple[str, str, str]] = [
    ("email", "uq_clients_email", "RENAME INDEX `email` TO `uq_clients_email`"),
]

# Dropped only AFTER the backfill has read them. Both are columns whose value
# has moved to a different level, not columns whose data is being discarded:
#
#   plan           -> subscriptions.plan. A client with three modules on two
#                     plans cannot be described by one string, which is exactly
#                     what operator.py papered over with MAX() aggregates.
#   webhook_secret -> sites.webhook_secret. A webhook is registered by a store.
#                     Per-client, a customer with two Woo stores overwrote one
#                     store's secret every time they re-registered the other.
CLIENTS_DROP_COLUMNS: list[tuple[str, str]] = [
    ("plan",           "DROP COLUMN `plan`"),
    ("webhook_secret", "DROP COLUMN `webhook_secret`"),
]


# ── Tables the v2 model retires ──────────────────────────────────────────────
#
# THE THREE chat_* TABLES ARE NOT EMPTY HUSKS. An earlier version of this
# comment said they were, and that claim was the entire justification for
# dropping them. It was wrong, so here is the actual situation.
#
# They are absent from init/01-schema.sql and absent from the currently
# deployed database. That is not evidence that nothing uses them. It is
# evidence of one thing only: /magento/chatbot/message has never been called
# against THIS database. conversation_service.ensure_chat_tables() runs
# CREATE TABLE IF NOT EXISTS for all three at the top of every one of its
# public functions, and routers/chatbot.py calls start_or_get_conversation()
# and append_turn() on the live chat route. So the tables come into existence
# the first time a shopper sends a message, and from that moment every chat
# turn the AI Chatbot has ever served is a row in them -
# chat_analytics_service and operator.py already build reporting on top of
# that. On a deployment where the chat route has been used, these three tables
# hold the production conversation history and nothing else does.
#
# Which is why drop_legacy_tables refuses to drop ANY table that has rows,
# with the row counts printed before the destructive phase runs rather than
# after. MySQL cannot roll back DDL: there is no undo for getting this wrong
# once. Dropping them where they ARE empty is still not sufficient on its own -
# delete conversation_service.py in the same change or the next stray request
# recreates them.
#
# security_logs and client_api_keys are referenced by domain_auth_service and
# have never existed. _log_security_event swallows its own exception, so every
# authorised request has been paying for a failing INSERT; _get_client_api_keys
# does not, so any request carrying an X-API-Key header raises straight out of
# the router today.
#
# usage_logs is NOT in this list any more - it is archived instead, see below.
#
# Order is children first. The chat_* tables happen to carry no foreign keys,
# but relying on that is how a drop script breaks the day someone adds one.
LEGACY_TABLES_TO_DROP: list[str] = [
    "chat_feedback",
    "chat_messages",
    "chat_conversations",
    "security_logs",
    "client_api_keys",
]


# ── The v1 tables that are archived rather than dropped ──────────────────────
#
# (source, target) rename pairs. A rename moves no rows, costs nothing, and
# guarantees an honest cutover: code still writing to the old name fails
# loudly instead of quietly appending to a table nobody reads any more.
#
#   token_usage_tracking - real observed token counts and real per-model costs.
#       The only evidence in existence for what a chat turn actually costs, and
#       therefore the only input to sizing MODULE_PLANS. Renamed, NEVER
#       dropped; the assertion below makes that structural rather than a
#       promise in a comment.
#
#   usage_logs - superseded by usage_counters (search_count) and
#       sites.indexed_items (ingest_count), and its reembed_count /
#       reembed_limit columns were never wired to anything. But search_count is
#       the only per-tenant, per-month record of actual request volume that has
#       ever been kept, which is exactly the input for sizing MODULE_PLANS'
#       request rungs - the same reason the token ledger is being kept. The
#       backfill creates zero usage_counters rows, so dropping this would
#       destroy that history to save the cost of one rename.
V1_TABLES_TO_ARCHIVE: list[tuple[str, str]] = [
    ("token_usage_tracking", "token_usage_tracking_archive_v1"),
    ("usage_logs",           "usage_logs_archive_v1"),
]

# Named separately because the token ledger is the one archive other code has
# to reason about by name, and because a reader looking for "what happened to
# token_usage_tracking" should find it as a constant, not as list[0].
ARCHIVE_TABLE_FROM = V1_TABLES_TO_ARCHIVE[0][0]
ARCHIVE_TABLE_TO = V1_TABLES_TO_ARCHIVE[0][1]


# v1 tables that survive this migration untouched and are deliberately out of
# scope. Listed so the migration can report their row counts: a table nobody
# names in the report is a table nobody remembers is still there.
#
# All four are still keyed on client_id rather than site_id, which is the same
# per-client/per-store confusion this migration is fixing for webhook_secret -
# client_magento_credentials is PRIMARY KEY (client_id), so a customer with two
# Magento stores gets one credentials row for both. That is a real defect and
# it is not addressed here.
V1_TABLES_LEFT_ALONE: list[str] = [
    "search_logs",
    "rate_limits",
    "agent_client_vocab",
    "client_magento_credentials",
]


# A table cannot be both archived and dropped, and getting that pair wrong is
# unrecoverable in the direction that matters. Checked here, at import, so it
# is a startup failure rather than a discovery made by reading the report after
# the DROP has already committed.
_archive_sources = {source for source, _ in V1_TABLES_TO_ARCHIVE}
_drop_and_archive = _archive_sources.intersection(LEGACY_TABLES_TO_DROP)
if _drop_and_archive:
    raise RuntimeError(
        f"{sorted(_drop_and_archive)} appear in both V1_TABLES_TO_ARCHIVE and "
        f"LEGACY_TABLES_TO_DROP. Archiving is a rename and dropping is not "
        f"reversible; pick one."
    )
