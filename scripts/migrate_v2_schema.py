"""
Rebuild the billing schema around the two scopes that actually exist.

WHAT IS WRONG WITH v1
---------------------
Everything hangs off client_id, because the schema was written when AISearch
was the only product. Three Magento modules now call the identical backend
endpoints, so the route a request arrived on cannot tell you which product made
it. A store running AIChatbot, AIProductQA and AISearch drains one shared
counter, and its three admin panels each render the same client-wide number
while implying it belongs to that module. Per-product usage is not a hard query
to write - it is unanswerable, because nothing ever recorded the answer.

WHAT REPLACES IT
----------------
Seven tables, and one architectural fact behind them: every module on a store
shares ONE Qdrant collection. So catalogue size is consumed once per STORE and
lives on `sites`, while licences and request quota are per MODULE and live on
`subscriptions`. The DDL and the reasoning are in
backend/app/services/schema_v2.py; this file is only the control flow that
applies it.

HOW TO RUN IT
-------------
    python scripts/migrate_v2_schema.py                    # dry run, default
    python scripts/migrate_v2_schema.py --apply            # build and backfill
    python scripts/migrate_v2_schema.py --apply --yes-drop-data   # + teardown

The default is a DRY RUN that writes nothing and prints the full plan: every
table it would create, every column it would add or modify, every site and
subscription it would backfill, every table it would rename, and every drop it
would perform. Read that plan before you pass --apply, because half of what
this script does cannot be undone - MySQL commits DDL implicitly and cannot
roll it back, and there is no dual-read period.

--yes-drop-data is a SECOND gate, on top of --apply, and only the destructive
phase is behind it: dropping clients.plan and clients.webhook_secret, dropping
the retired tables, and redacting the plaintext licence keys. Without it the
script builds and backfills everything and leaves every v1 artefact in place,
which is a perfectly good state to stop in and look around.

THIS IS A CUTOVER, NOT AN ADDITION
----------------------------------
Running this with both flags leaves the currently-deployed code unable to serve
requests. That is intended and unavoidable. What stops working, in full:

  * token_usage_tracking is renamed out from under token_usage_service
  * usage_logs is renamed out from under increment_search_count
  * clients.plan is dropped out from under license_service
  * license_keys.license_key is blanked, which takes with it onboarding's
    "hand a returning customer their existing key" path and webhooks.py's
    derivation of the AES KEK for merchant-supplied API keys

There are no live customers - every row in this database is a development site
- so there is no dual-read period to design for and none has been built. Land
the code rewrite in the same deploy.

CHAT HISTORY IS NOT PART OF THE CUTOVER
---------------------------------------
chat_conversations, chat_messages and chat_feedback are in the drop list, and
they are the one thing here that could destroy data nobody chose to give up.
They are absent from this deployment only because /magento/chatbot/message has
never been called against it; conversation_service.ensure_chat_tables() creates
them on demand, so on a deployment where the chat route HAS been used they hold
every chat turn ever served. drop_legacy_tables therefore refuses any table
with rows in it, whatever the flags say, and the row counts are printed BEFORE
the destructive phase runs rather than in a summary afterwards.

WHAT THIS DOES NOT TOUCH
------------------------
search_logs, rate_limits, agent_client_vocab and client_magento_credentials
survive unchanged and are still keyed on client_id rather than site_id. That is
the same per-client/per-store confusion this migration is fixing for
webhook_secret, left in place four more times: client_magento_credentials is
PRIMARY KEY (client_id), so a customer with two Magento stores has exactly one
credentials row for both. Out of scope here, deliberately, but their row counts
are in the report so nobody forgets they are still there.

WHAT CANNOT BE MIGRATED
-----------------------
Licence keys. v1 stored the whole plaintext JWT; v2 stores only the SHA-256 of
a new opaque czg_live_... token, and you cannot turn one into the other. So NO
`licences` rows are created here at all. Every existing key must be reissued
through onboarding, and the report at the end counts them. The site and
subscription rows this script builds are what onboarding will attach those new
licences to, so the reissue is a mint, not a re-onboarding.

SAFE TO RUN REPEATEDLY
----------------------
Every CREATE, ALTER, RENAME and DROP checks information_schema first, and every
backfill INSERT checks for its own row before writing. Note that MySQL commits
DDL implicitly and cannot roll it back: idempotence is the only recovery
mechanism there is, which is why it is checked per statement rather than per
phase.

STYLE NOTE INHERITED FROM scripts/migrate_license_product_platform.py: `keys`
is a MySQL reserved word and blew up that migration when used as a column
alias. No alias below is a reserved word - `key_count`, `row_total`, and so on.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import uuid
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from backend.app.services import catalog, schema_v2
from backend.app.services.database import engine

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ── Backfill policy ──────────────────────────────────────────────────────────
#
# The ladders are catalog.py's, read out of it rather than restated here.
#
# An earlier version of this file declared its own copies and justified them by
# saying the real ladders "belong in catalog.py once the code rewrite lands".
# The rewrite has landed: catalog.INDEX_PLANS and catalog.MODULE_PLANS carry
# both, catalog.assert_plan_ladders_sane() checks at import that each one
# strictly ascends (which is the property ladder_rung's linear scan depends on),
# and this module already imports catalog. The copies survived anyway, and the
# one number they did not share - the trial allowance - had already drifted to
# 1,000 against catalog.py's 250. That is precisely the drift the comment
# claimed to be avoiding, demonstrated inside the file that claimed it.
#
# What ladder_rung adds is the question catalog.py cannot answer: which rung
# does a limit that was already granted to a live dev store fall on.

# Bought once per SITE. Catalogue items in the shared Qdrant collection.
INDEX_PLAN_LADDER = tuple(
    (code, catalog.INDEX_PLANS[code]["catalogue_limit"])
    for code in catalog.INDEX_PLAN_ORDER
)

# Bought per SUBSCRIPTION. Billable requests per calendar month. The sellable
# rungs carry the same numbers v1's search_limit_per_month ladder advertised,
# so the mapping from an existing key is exact rather than approximate.
# TRIAL_MODULE_PLAN is deliberately absent from MODULE_PLAN_ORDER - it is a
# status a subscription starts in, not a rung anyone can be mapped onto.
MODULE_PLAN_LADDER = tuple(
    (code, catalog.MODULE_PLANS[code]["request_limit"])
    for code in catalog.MODULE_PLAN_ORDER
)

# Used only for the placeholder subscription attached to a legacy key that
# named no product at all. Every such key in this database came from
# scripts/create_client.py or from onboarding before product scoping existed,
# both of which were Magento-first. It is a value to correct at reissue, not a
# claim about what the merchant bought - which is why every site that gets it
# is named individually in the report.
LEGACY_PLATFORM = "magento"

# Which product a placeholder trial subscription is written against. A
# subscription needs a real product_code (it is a foreign key), and an
# unscoped v1 key names none, so one has to be chosen. The flagship of the
# platform is the least-wrong choice and the report says so out loud.
FLAGSHIP_PRODUCT = {
    "magento":     "magento_chatbot",
    "woocommerce": "woo_search",
}

# What clients.id must be, not just what collation it must carry. See preflight.
EXPECTED_CLIENT_ID_TYPE = "varchar(36)"


# ── Collection naming ────────────────────────────────────────────────────────

def derive_collection_name(client_id: str, domain: str) -> str:
    """Byte-identical copy of qdrant_service.get_collection_name.

    Copied rather than imported on purpose: importing qdrant_service
    constructs a QdrantClient at module scope, and a schema migration must not
    require the vector store to be reachable - or its client library to be
    installed - to add a column to MySQL.

    If get_collection_name ever changes, this must change with it, and every
    row in sites.collection_name has to be re-seeded from what Qdrant actually
    holds. A collection name that does not exist reads back as zero results
    rather than an error, so a mismatch here surfaces as "search stopped
    working" days later, not as a stack trace.
    """
    client_safe = re.sub(r"[^a-zA-Z0-9]", "_", client_id)
    domain_safe = re.sub(r"[^a-zA-Z0-9]", "_", domain)
    return f"products_{domain_safe}_{client_safe}"


def normalise_domain(raw: str) -> str:
    """Bare host, lowercased, no www - matching onboarding.extract_domain.

    Same transform, but it never raises. extract_domain rejects a bare word
    like 'myyshop' because minting a key on it would produce a credential that
    fails on its first request; here the row already exists and refusing it
    would just abandon a site. Anything odd is carried through and reported.
    """
    host = (raw or "").strip().lower()
    host = host.split("://")[-1]
    host = host.split("/")[0]
    host = host.split("@")[-1]
    host = host.split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    return host


# ladder_rung's third return value. Not a bool, because there are three
# outcomes and the two that are not "exact" need different sentences in the
# report - and when this was a bool the interesting one was the one it could
# not express.
RUNG_EXACT = ""
RUNG_RAISED = "raised"
RUNG_CAPPED = "capped"


def ladder_rung(ladder, granted: int):
    """Smallest rung that still covers *granted*: (code, limit, adjustment).

    Never shrinks within the ladder. A store sitting on a hand-edited
    3,000-item ceiling gets 'small' (5,000), not 'free' (500) - because
    catalogue_limit now comes from the site's own index_plan, and rounding down
    would put a store below what it has already indexed, with no clean
    resolution. Rounding up costs nothing: the ceiling is not a bill.

    ABOVE THE TOP RUNG IS THE CASE TO UNDERSTAND. A store hand-granted 250,000
    items when the largest plan sells 100,000 has no honest representation
    here. The previous version kept the granted number and paired it with the
    top rung's plan code, and reported nothing - it returned raised=False, so
    the one branch that most needed a human was the only silent one. That
    writes a row catalog.py cannot reproduce: sites.index_plan='large' beside
    catalogue_limit=250000, when catalog.catalogue_limit_for('large') is
    100000. tenancy_service.set_index_plan(site, 'large') - an operator
    re-applying the plan already on the row, a no-op by every appearance - then
    silently cuts that store's ceiling by 150,000 items. Same shape for
    MODULE_PLAN_LADDER and licensing_service.set_subscription_plan.

    So it caps at the top rung and returns RUNG_CAPPED, and both call sites
    turn that into a loud warning naming the store and the size of the cut.
    Every (plan, limit) pair this migration writes is therefore one catalog.py
    can reproduce exactly, which is the only version of the two-ladder design
    that holds. The cut is visible and fixable - add a rung above the top one
    in catalog.py and re-run - where the silent rewrite later was neither.
    """
    for code, limit in ladder:
        if granted <= limit:
            return code, limit, (RUNG_RAISED if limit != granted else RUNG_EXACT)

    top_code, top_limit = ladder[-1]
    return top_code, top_limit, RUNG_CAPPED


# ── The run itself: connection, mode, and everything it has to say ───────────

class Migration:
    """The connection, the two gates, and the report being assembled.

    Every phase below takes one of these instead of a raw connection, and the
    reason is the dry run. A migration that can only be understood by running
    it is a migration nobody reads first, and this one drops tables. So writes
    go through exactly one method - change() - which either executes the
    statement or records that it would have, and reads go through methods that
    always execute. There is no second way to write, which is what makes "the
    default run touches nothing" a property of the code rather than a promise
    in a docstring.

    Two gates, not one:

      apply_changes     - False by default. Nothing is written at all.
      allow_destructive - False by default. Everything that REMOVES something
                          (a table, a column, a stored secret) is refused and
                          reported as refused, even under --apply.

    A dry run still reads freely, including from tables that do not exist yet -
    see table_exists and scalar_if_table. That matters: against a pristine v1
    database none of the seven v2 tables is there, and a probe that raised
    error 1146 instead of answering "no row" would make the dry run useless
    exactly where it is needed most.
    """

    def __init__(self, conn, *, apply_changes: bool, allow_destructive: bool):
        self.conn = conn
        self.apply_changes = apply_changes
        self.allow_destructive = allow_destructive
        self.actions: list[str] = []
        self.warnings: list[str] = []
        # Cheap and worth it: table_exists is called dozens of times per run,
        # several times per site, and information_schema is not free. Reset
        # whenever a statement could have changed the answer.
        self._table_cache: dict[str, bool] = {}

    # ── reads ────────────────────────────────────────────────────────────────

    def rows(self, sql: str, params: dict | None = None):
        return self.conn.execute(text(sql), params or {}).mappings().all()

    def scalar(self, sql: str, params: dict | None = None):
        return self.conn.execute(text(sql), params or {}).scalar()

    def scalar_if_table(self, table: str, sql: str, params: dict | None = None):
        """A 'does this row already exist' probe against a table that may not.

        Absent table means no row, which is the same answer a fresh install
        gives - and it is the answer a dry run needs, because create_tables
        has not run.
        """
        if not self.table_exists(table):
            return None
        return self.scalar(sql, params)

    def table_exists(self, table: str) -> bool:
        if table not in self._table_cache:
            self._table_cache[table] = self.scalar("""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema = DATABASE() AND table_name = :table
            """, {"table": table}) > 0
        return self._table_cache[table]

    def column_exists(self, table: str, column: str) -> bool:
        return self.scalar("""
            SELECT COUNT(*) FROM information_schema.columns
            WHERE table_schema = DATABASE()
              AND table_name   = :table
              AND column_name  = :column
        """, {"table": table, "column": column}) > 0

    def column_meta(self, table: str, column: str):
        """One row of information_schema.columns, or None. `column_type` is the
        full declaration ('varchar(36)'), which `data_type` alone is not."""
        found = self.rows("""
            -- Aliased explicitly. MySQL 8 labels information_schema columns
            -- in UPPERCASE and SQLAlchemy's mapping keys are case sensitive, so
            -- an unaliased `collation_name` comes back as COLLATION_NAME and
            -- every lookup below raises NoSuchColumnError. Aliasing pins the
            -- label regardless of what the server decides to return.
            SELECT column_type    AS column_type,
                   collation_name AS collation_name,
                   is_nullable    AS is_nullable
            FROM information_schema.columns
            WHERE table_schema = DATABASE()
              AND table_name   = :table
              AND column_name  = :column
        """, {"table": table, "column": column})
        return found[0] if found else None

    def table_columns(self, table: str) -> set[str]:
        found = self.rows("""
            SELECT column_name AS column_name FROM information_schema.columns
            WHERE table_schema = DATABASE() AND table_name = :table
        """, {"table": table})
        return {row["column_name"] for row in found}

    def index_exists(self, table: str, index: str) -> bool:
        return self.scalar("""
            SELECT COUNT(*) FROM information_schema.statistics
            WHERE table_schema = DATABASE()
              AND table_name   = :table
              AND index_name   = :index
        """, {"table": table, "index": index}) > 0

    def foreign_keys(self, table: str) -> list[str]:
        found = self.rows("""
            SELECT constraint_name AS constraint_name FROM information_schema.table_constraints
            WHERE table_schema    = DATABASE()
              AND table_name      = :table
              AND constraint_type = 'FOREIGN KEY'
        """, {"table": table})
        return [row["constraint_name"] for row in found]

    def row_count(self, table: str):
        """Row count, or None when the table is not there. Both are reportable."""
        if not self.table_exists(table):
            return None
        return self.scalar(f"SELECT COUNT(*) AS row_total FROM `{table}`")

    # ── the one write path ───────────────────────────────────────────────────

    def change(self, sql: str, params: dict | None = None, *,
               label: str | None = None, destructive: bool = False) -> bool:
        """Execute a statement that changes something, or report that it would.

        Returns whether it actually ran, so a caller can count what happened
        without asking the mode flags itself. *label* is the human phrasing of
        the statement ('create table sites'); pass None inside a loop that
        reports an aggregate instead of one line per row.
        """
        if destructive and not self.allow_destructive:
            if label:
                self.note("REFUSED", f"{label} - needs --yes-drop-data")
            return False

        if not self.apply_changes:
            if label:
                self.note("planned", label)
            return False

        self.conn.execute(text(sql), params or {})
        self._table_cache.clear()
        if label:
            self.note("done", label)
        return True

    def commit(self) -> None:
        """Commit an applying run; roll a dry run back.

        The rollback is belt and braces - change() refuses every write in a dry
        run, so there should be nothing to roll back. It is here because the
        cost is zero and the failure it covers (a phase added later that
        executes through self.conn directly) is otherwise silent. It cannot
        help with DDL: MySQL commits CREATE, ALTER, RENAME and DROP implicitly
        whatever this connection believes about its transaction.
        """
        if self.apply_changes:
            self.conn.commit()
        else:
            self.conn.rollback()

    # ── the report ───────────────────────────────────────────────────────────

    def note(self, verb: str, label: str) -> None:
        self.actions.append(f"{verb:<8} {label}")

    def detail(self, text_line: str) -> None:
        """A continuation line under the note above it."""
        self.actions.append(f"{'':<8} {text_line}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)


# ── Phases ───────────────────────────────────────────────────────────────────

def preflight(m: Migration) -> None:
    """Refuse to start rather than leave a half-built schema behind.

    MySQL cannot roll back DDL, so a foreign key rejected on table four leaves
    three orphan tables and a database nobody can describe. The check here is
    for exactly that failure, which only surfaces at that point.
    """
    if not m.table_exists("clients"):
        # A database with no clients table is a fresh install, not a broken
        # one. create_tables builds clients first, with the collation schema_v2
        # declares, so there is nothing to verify and nothing to back fill.
        m.note("note", "no `clients` table - treating this as a fresh database")
        return

    # InnoDB rejects a foreign key whose referencing column differs from the
    # referenced one - in charset, in collation, OR in type - with errno 3780
    # and a message that names neither column. Every FK in schema_v2 points at
    # clients.id eventually, so a mismatch here means nothing gets built.
    #
    # Both halves are checked. The collation half is the one that bites in
    # practice, but VARCHAR(36) against CHAR(36) is rejected identically, and
    # schema_v2's own commentary is explicit that every uuid column is
    # VARCHAR(36) and not CHAR(36) - a check that does not enforce the rule its
    # commentary establishes is the kind of guard that reads like one that does.
    meta = m.column_meta("clients", "id")
    if meta is None:
        raise RuntimeError(
            "clients exists but has no `id` column. Nothing in schema_v2 can "
            "reference it. Look at the table by hand before running anything."
        )

    collation = meta["collation_name"]
    column_type = (meta["column_type"] or "").lower()

    if collation != schema_v2.EXPECTED_CLIENT_ID_COLLATION:
        raise RuntimeError(
            f"clients.id is {collation!r} but schema_v2 declares every foreign "
            f"key column as {schema_v2.EXPECTED_CLIENT_ID_COLLATION!r}. InnoDB "
            f"would reject the constraints. Either convert clients.id or "
            f"change the COLLATE clauses in schema_v2.py to match - but change "
            f"all of them, not the ones that happened to fail."
        )

    if column_type != EXPECTED_CLIENT_ID_TYPE:
        raise RuntimeError(
            f"clients.id is {column_type!r} but schema_v2 declares every uuid "
            f"column as {EXPECTED_CLIENT_ID_TYPE!r}. InnoDB requires the "
            f"referencing and referenced columns of a foreign key to have the "
            f"same type, and rejects a mismatch with the same errno 3780 that "
            f"a collation mismatch gives. Convert clients.id; do not widen the "
            f"schema_v2 columns to match a legacy type."
        )

    m.note("checked", f"clients.id is {column_type} {collation} - foreign keys will hold")


def _ddl_columns(ddl: str) -> list[str]:
    """The column names a CREATE TABLE constant declares, in order.

    Parsed rather than hand-listed so the two can never disagree. A column
    definition is the only thing in these constants that starts with a
    backticked identifier: PRIMARY KEY, UNIQUE KEY, KEY, CONSTRAINT and the
    comment lines all start with a word or a dash.
    """
    return re.findall(r"^\s*`([^`]+)`\s+[A-Z]", ddl, flags=re.MULTILINE)


def create_tables(m: Migration) -> None:
    """CREATE each v2 table that is absent, in foreign key dependency order.

    A table that is already present is verified rather than trusted. The old
    version guarded on presence alone, which is fine on the first run and
    misleading on every one after it: add a column to schema_v2.py, re-run the
    migration that calls itself safe to run repeatedly, and it reports
    'present table X (skipped)' over a table that no longer matches the file
    this project calls the single source of truth. The divergence then surfaces
    as an unknown-column error at runtime, in whichever service touches the new
    column first.
    """
    for table, ddl in schema_v2.V2_TABLES:
        if not m.table_exists(table):
            m.change(ddl, label=f"create table {table}")
            continue

        m.note("present", f"table {table} (skipped)")

        # clients is exempt: it is the one v2 table that already exists in
        # every deployment and is reached by ALTER, so align_clients below owns
        # its convergence. Shape-checking it here would report the v1 columns
        # as divergences one phase before they are fixed.
        if table != "clients":
            _verify_table_shape(m, table, ddl)


def _verify_table_shape(m: Migration, table: str, ddl: str) -> None:
    """Warn per column that schema_v2.py and the live table disagree about.

    Columns only. Indexes, constraints and column types would each need their
    own comparison and their own tolerance for how MySQL normalises a
    declaration; the column set is the divergence that actually happens (a
    column added to the DDL after a run) and it is the one that produces an
    unknown-column error rather than a slow query.
    """
    declared = set(_ddl_columns(ddl))
    live = m.table_columns(table)

    missing = sorted(declared - live)
    extra = sorted(live - declared)

    if missing:
        m.warn(
            f"{table} is missing {missing!r}, which schema_v2.py declares. This "
            f"migration does not ALTER v2 tables - it only creates absent ones - "
            f"so the column has to be added by hand or the table rebuilt. Until "
            f"then the first service to touch it gets an unknown-column error."
        )
    if extra:
        m.warn(
            f"{table} carries {extra!r}, which schema_v2.py does not declare. A "
            f"fresh install would not have them, so this database and a new one "
            f"are two different shapes - which is the condition schema_v2.py "
            f"exists to end."
        )


def align_clients(m: Migration) -> None:
    """Bring an existing v1 `clients` to the v2 shape. Drops happen later.

    Converge, do not merely extend: a fresh install runs CLIENTS_TABLE and an
    upgraded one runs these, and if they do not end up identical then "what
    does this database look like" has two answers again.
    """
    if not m.table_exists("clients"):
        m.note("note", "clients will be CREATEd with the v2 shape - no ALTERs needed")
        return

    for column, ddl in schema_v2.CLIENTS_ADD_COLUMNS:
        if m.column_exists("clients", column):
            m.note("present", f"clients.{column} (skipped)")
            continue
        m.change(f"ALTER TABLE `clients` {ddl}", label=f"add clients.{column}")

    for column, ddl in schema_v2.CLIENTS_ALTER_COLUMNS:
        _align_clients_column(m, column, ddl)

    for old_name, new_name, ddl in schema_v2.CLIENTS_RENAME_INDEXES:
        if m.index_exists("clients", new_name):
            m.note("present", f"clients index {new_name} (skipped)")
            continue
        if not m.index_exists("clients", old_name):
            # Neither name is there. That is a clients table whose unique key
            # on email has gone missing entirely, which onboarding's
            # find-or-create depends on - say so rather than inventing one.
            m.warn(
                f"clients has neither index `{old_name}` nor `{new_name}`. The "
                f"unique key on email is what makes onboarding find-or-create "
                f"instead of duplicate-or-reject. Add it by hand."
            )
            continue
        m.change(f"ALTER TABLE `clients` {ddl}",
                 label=f"rename clients index {old_name} -> {new_name}")


def _align_clients_column(m: Migration, column: str, ddl: str) -> None:
    """Apply one CLIENTS_ALTER_COLUMNS entry, only where it is needed.

    A MODIFY is a table rebuild, so it runs only when information_schema says
    the live column actually diverges - and NOT AT ALL while any row would be
    destroyed by it. clients.is_active is the whole reason this list exists: v1
    declares it nullable, and the auth path treats NULL as falsy and denies, so
    a NULL here is a customer refused with no explanation. Making it NOT NULL
    is the fix; backfilling the NULLs to get there is a decision about who can
    log in, because 1 activates accounts that are being denied today and 0
    permanently deactivates accounts that v1's DEFAULT '1' says were meant to
    be live. Refuse, name the count, and let a human choose.
    """
    meta = m.column_meta("clients", column)
    if meta is None:
        m.warn(f"clients has no `{column}` column to align. Check the table by hand.")
        return

    if meta["is_nullable"] != "YES":
        m.note("present", f"clients.{column} already NOT NULL (skipped)")
        return

    null_rows = m.scalar(f"SELECT COUNT(*) AS row_total FROM `clients` WHERE `{column}` IS NULL")
    if null_rows:
        m.warn(
            f"clients.{column} is nullable and {null_rows} row(s) hold NULL, so it "
            f"cannot be made NOT NULL yet. The auth path reads a NULL here as "
            f"false and denies the customer with no explanation. Decide per row "
            f"whether each account is live (set 1) or suspended (set 0), then "
            f"re-run - do not let a migration guess which."
        )
        m.note("REFUSED", f"modify clients.{column} - {null_rows} NULL row(s)")
        return

    m.change(f"ALTER TABLE `clients` {ddl}", label=f"modify clients.{column} to NOT NULL")


def seed_products(m: Migration) -> None:
    """Mirror catalog.PRODUCTS into the products table.

    Done here rather than left to a separate seed script because
    subscriptions.product_code is a foreign key onto this table: the backfill
    two phases down cannot write a single row until these exist.

    Written as find-then-write rather than INSERT ... ON DUPLICATE KEY UPDATE,
    which is what it used to be. products has TWO unique keys - PRIMARY KEY
    (code) and uq_products_key_segment (key_segment) - and ON DUPLICATE fires
    on whichever one conflicts first. If a catalogue edit moved a key_segment
    to a different product code, the upsert would find the segment collision
    and UPDATE THE OTHER PRODUCT'S ROW - overwriting its platform and name,
    never inserting the intended one, and reporting a clean 'seeded 5 rows'
    either way. products.code is permanent and written into billing history, so
    a mutation of the wrong row is expensive to unpick.

    is_sellable is written on insert and never touched again: catalog.py
    defines no such field, and it is the one operator-owned column on this
    table. See the note above PRODUCTS_TABLE.
    """
    inserted = updated = unchanged = 0

    for product in catalog.PRODUCTS.values():
        params = {
            "code":        product["code"],
            "platform":    product["platform"],
            "name":        product["name"],
            "key_segment": product["key_segment"],
        }

        holder = m.scalar_if_table("products", """
            SELECT code FROM products WHERE key_segment = :key_segment AND code <> :code
        """, params)
        if holder:
            m.warn(
                f"product '{product['code']}' wants key segment "
                f"'{product['key_segment']}', which product '{holder}' already holds. "
                f"Neither row was touched. Two products sharing a segment makes "
                f"their licence keys visually indistinguishable, and resolving it "
                f"means deciding which product keeps the segment - catalog.py is "
                f"where that decision belongs."
            )
            continue

        existing = m.scalar_if_table("products", """
            SELECT code FROM products WHERE code = :code
        """, params)

        if not existing:
            m.change("""
                INSERT INTO products (code, platform, name, key_segment, is_sellable)
                VALUES (:code, :platform, :name, :key_segment, 1)
            """, params)
            inserted += 1
            continue

        stale = m.scalar("""
            SELECT COUNT(*) AS row_total FROM products
            WHERE code = :code
              AND (platform <> :platform OR name <> :name OR key_segment <> :key_segment)
        """, params)
        if not stale:
            unchanged += 1
            continue

        m.change("""
            UPDATE products
               SET platform = :platform, name = :name, key_segment = :key_segment
             WHERE code = :code
        """, params)
        updated += 1

    verb = "done" if m.apply_changes else "planned"
    m.note(verb, f"seed products from catalog.PRODUCTS: {inserted} new, "
                 f"{updated} updated, {unchanged} already correct")


def _read_license_keys(m: Migration) -> list[dict]:
    """Every v1 key, joined to a live client.

    product_code and platform are selected conditionally because
    init/01-schema.sql defines license_keys WITHOUT them - they exist only
    where scripts/migrate_license_product_platform.py has been run. A database
    built from init alone would fail this SELECT on an unknown column, which is
    the same latent trap that makes validate_license_key error there today.

    Note what is NOT selected: license_keys.license_key, the plaintext JWT.
    Nothing the backfill does needs the secret, which is what makes it safe for
    the teardown phase to blank that column afterwards.
    """
    has_product = m.column_exists("license_keys", "product_code")
    has_platform = m.column_exists("license_keys", "platform")

    product_expr = "lk.product_code" if has_product else "NULL"
    platform_expr = "lk.platform" if has_platform else "NULL"

    rows = m.rows(f"""
        SELECT lk.id                     AS license_id,
               lk.client_id              AS client_id,
               lk.allowed_domain         AS allowed_domain,
               lk.product_limit          AS product_limit,
               lk.search_limit_per_month AS search_limit,
               lk.is_active              AS is_active,
               lk.expires_at             AS expires_at,
               lk.created_at             AS created_at,
               {product_expr}            AS product_code,
               {platform_expr}           AS platform
        FROM license_keys lk
        JOIN clients c ON c.id = lk.client_id
        ORDER BY lk.created_at
    """)

    return [dict(row) for row in rows]


def backfill_sites(m: Migration, key_rows: list[dict]) -> dict:
    """One site per (client, normalised domain). Returns {(client, domain): id}.

    Every site is environment='development' because every row in this database
    is a development site - the brief is explicit that no live customer exists.
    That matters beyond bookkeeping: usage_events.key_owner is derived from
    this column, and calling a dev site 'production' would book Czargroup's own
    API spend as the merchant's.

    The returned map includes sites this run only PLANNED to create, so a dry
    run can go on to report the subscriptions and webhook secrets that would
    hang off them instead of stopping at the first missing row.
    """
    groups: dict[tuple[str, str], dict] = defaultdict(lambda: {
        "raw_domains": [],
        "product_limit": 0,
        "platforms": set(),
        "product_codes": set(),
    })

    for row in key_rows:
        domain = normalise_domain(row["allowed_domain"])
        if not domain:
            m.warn(
                f"licence {row['license_id']} has an empty allowed_domain - "
                f"skipped; it cannot name a store."
            )
            continue

        bucket = groups[(row["client_id"], domain)]
        bucket["raw_domains"].append((row["created_at"], row["is_active"], row["allowed_domain"]))
        bucket["product_limit"] = max(bucket["product_limit"], int(row["product_limit"] or 0))
        if row["platform"]:
            bucket["platforms"].add(row["platform"])
        if row["product_code"]:
            bucket["product_codes"].add(row["product_code"])

    site_ids: dict[tuple[str, str], str] = {}
    planned_collections: set[str] = set()
    created = reused = 0

    for (client_id, domain), bucket in groups.items():
        existing = m.scalar_if_table("sites", """
            SELECT id FROM sites WHERE client_id = :client_id AND domain = :domain
        """, {"client_id": client_id, "domain": domain})

        if existing:
            site_ids[(client_id, domain)] = existing
            reused += 1
            continue

        # The collection name must come from the RAW allowed_domain, never from
        # the normalised one. The live collection was named from the JWT's
        # `domain` claim, which is whatever string was handed to
        # generate_license_key - unlowercased, www intact. Recomputing it from
        # the normalised host renames the collection away from its own data,
        # and a Qdrant read against a name that does not exist returns zero
        # results rather than failing. Pick the most recent active key's raw
        # value: that is the one whose token is actually in a store's config.
        candidates = sorted(bucket["raw_domains"], key=lambda item: (bool(item[1]), item[0]))
        raw_domain = candidates[-1][2]

        distinct_raw = {item[2] for item in bucket["raw_domains"]}
        if len(distinct_raw) > 1:
            m.warn(
                f"{domain}: keys were minted against {sorted(distinct_raw)!r}, which "
                f"are different Qdrant collections today. Seeded from {raw_domain!r}; "
                f"the others' points are unreachable until re-synced."
            )

        collection_name = derive_collection_name(client_id, raw_domain)

        clash = m.scalar_if_table("sites", """
            SELECT id FROM sites WHERE collection_name = :collection_name
        """, {"collection_name": collection_name})
        if clash or collection_name in planned_collections:
            # get_collection_name's regex is not injective: '.', '-' and '_'
            # all collapse to '_'. Two stores on one collection is a
            # cross-tenant read, so refuse the second rather than let the
            # UNIQUE constraint abort the whole migration. planned_collections
            # catches the pair within one run, which the live SELECT cannot see
            # in a dry run and cannot see in an applying run either until the
            # first INSERT has happened.
            m.warn(
                f"{domain}: derives collection {collection_name!r}, already claimed by "
                f"site {clash or 'another store in this same run'}. NOT created - these "
                f"two stores are sharing one Qdrant collection today and that has to be "
                f"resolved by hand."
            )
            continue

        platform = _resolve_platform(m, domain, bucket)

        granted = bucket["product_limit"] or INDEX_PLAN_LADDER[0][1]
        index_plan, catalogue_limit, adjustment = ladder_rung(INDEX_PLAN_LADDER, granted)
        if adjustment == RUNG_RAISED:
            m.warn(
                f"{domain}: catalogue ceiling raised {granted} -> {catalogue_limit} to "
                f"land on the '{index_plan}' rung. Ceilings are never rounded down; a "
                f"store below what it has indexed has no fix."
            )
        elif adjustment == RUNG_CAPPED:
            m.warn(
                f"{domain}: was granted a {granted}-item catalogue, which is above every "
                f"rung catalog.INDEX_PLANS sells. CUT to {catalogue_limit} on the "
                f"'{index_plan}' rung, losing {granted - catalogue_limit} items of "
                f"ceiling. The alternative was writing index_plan='{index_plan}' beside "
                f"catalogue_limit={granted}, a pair catalog.py cannot reproduce, which "
                f"set_index_plan('{index_plan}') would silently cut later anyway. If "
                f"this store really is that size, add a rung above '{index_plan}' to "
                f"catalog.INDEX_PLANS and re-run before anyone syncs."
            )

        site_id = str(uuid.uuid4())
        m.change("""
            INSERT INTO sites
                (id, client_id, domain, platform, collection_name,
                 environment, index_plan, catalogue_limit, indexed_items, is_active)
            VALUES
                (:id, :client_id, :domain, :platform, :collection_name,
                 'development', :index_plan, :catalogue_limit, 0, 1)
        """, {
            "id":              site_id,
            "client_id":       client_id,
            "domain":          domain,
            "platform":        platform,
            "collection_name": collection_name,
            "index_plan":      index_plan,
            "catalogue_limit": catalogue_limit,
        })
        site_ids[(client_id, domain)] = site_id
        planned_collections.add(collection_name)
        created += 1

    # indexed_items is left at 0 for every new row, deliberately. The honest
    # value is a count from Qdrant, which this script cannot reach, and three
    # of the four product-delete call sites are currently broken (they pass two
    # arguments to a three-argument delete_product), so the live collections
    # hold points for products the stores believe they deleted. Seeding a
    # number from a source that is known to be wrong is worse than seeding
    # zero: run the reconcile once the delete arity is fixed.
    verb = "done" if m.apply_changes else "planned"
    m.note(verb, f"backfill sites: {created} created, {reused} already present")
    m.detail("sites.indexed_items left at 0 - seed it from Qdrant, see the code comment")
    return site_ids


def _resolve_platform(m: Migration, domain: str, bucket: dict) -> str:
    """A site's platform, with every ambiguous case named rather than picked.

    sites.platform is load-bearing: tenancy_service._adopt_existing_site raises
    on a platform mismatch precisely because a Magento and a WooCommerce sync
    writing into one collection produce incompatible payload shapes. It is also
    denormalised onto every usage_events row, so a mis-stamp contaminates
    billing history that is meant to be readable years later.

    The two-platform case used to be resolved by sorted(...)[0], i.e. by
    alphabet, i.e. always 'magento', in silence - while the no-platform case
    beside it warned and _platform_from_products below refused the identical
    ambiguity. Two products on two platforms sharing one domain is unlikely,
    and it is exactly the case that most needs a human.
    """
    platforms = sorted(bucket["platforms"])

    if len(platforms) > 1:
        chosen = platforms[0]
        m.warn(
            f"{domain}: its v1 keys name {platforms!r} - two platforms on one store. "
            f"Written as '{chosen}'. One store cannot be both: a Magento and a "
            f"WooCommerce sync into the same Qdrant collection produce incompatible "
            f"payloads, and this value is stamped onto every billing row from here "
            f"on. Confirm which it is before reissuing."
        )
        return chosen

    if platforms:
        return platforms[0]

    derived = _platform_from_products(bucket["product_codes"])
    if derived:
        return derived

    m.warn(
        f"{domain}: no platform on any of its keys; defaulted to "
        f"'{LEGACY_PLATFORM}'. Correct it before reissuing."
    )
    return LEGACY_PLATFORM


def _platform_from_products(product_codes: set) -> str | None:
    """Derive a site's platform from the products its keys were scoped to.

    catalog.platform_of is the authority: the platform follows from the product
    the customer chose, and accepting the two independently is how a key ends
    up claiming a (platform, product) pair that does not exist.
    """
    platforms = {catalog.platform_of(code) for code in product_codes}
    platforms.discard(None)
    return sorted(platforms)[0] if len(platforms) == 1 else None


def backfill_subscriptions(m: Migration, key_rows: list[dict], site_ids: dict) -> None:
    """One subscription per (site, product) found on the v1 keys.

    Rows that carry a product_code become status='active' - they were real
    issued keys for a real module. Rows with no product_code were legacy
    all-products keys; they become a single status='trial' subscription against
    the platform's flagship product, because a subscription needs a real
    product_code and an unscoped key names none. Every one of those is listed
    in the report: the product is a placeholder to confirm, not a fact.

    The trial pair - plan and request_limit - comes from catalog.py and nowhere
    else. It used to be plan='starter' with a locally-declared
    TRIAL_REQUEST_LIMIT of 1,000, which was wrong twice over: catalog says a
    starter buys 10,000, so the row was internally inconsistent by a factor of
    ten and licensing_service.set_subscription_plan(id, 'starter') - visibly a
    no-op to an operator confirming the plan already on screen - would have
    silently granted the full 10,000; and licensing_service.create_subscription
    writes plan='trial' for the identical situation, so a
    `WHERE plan='trial'` written against the service layer's vocabulary would
    not have found a single row this migration created.
    """
    groups: dict[tuple[str, str], dict] = defaultdict(lambda: {
        "search_limit": 0,
        "expires_at": None,
        "started_at": None,
        "scoped": False,
        "domain": "",
    })

    for row in key_rows:
        domain = normalise_domain(row["allowed_domain"])
        site_id = site_ids.get((row["client_id"], domain))
        if not site_id:
            continue  # site was skipped upstream and already reported

        product_code = row["product_code"]
        scoped = bool(product_code)

        if not scoped:
            platform = row["platform"] or LEGACY_PLATFORM
            product_code = FLAGSHIP_PRODUCT.get(platform, FLAGSHIP_PRODUCT[LEGACY_PLATFORM])
            m.warn(
                f"{domain}: key {row['license_id']} named no product, so its placeholder "
                f"trial subscription is against '{product_code}'. Confirm what this store "
                f"actually runs before reissuing."
            )

        if product_code not in catalog.PRODUCTS:
            # A product_code the catalogue no longer knows cannot become a
            # subscription: products.code is a foreign key and catalog.py is
            # the only thing that seeds it. Product codes are permanent by
            # policy, so this means the catalogue was edited, not that the key
            # is odd - and the fix belongs in catalog.py.
            m.warn(
                f"{domain}: key {row['license_id']} names product '{product_code}', "
                f"absent from catalog.PRODUCTS. No subscription created."
            )
            continue

        bucket = groups[(site_id, product_code)]
        bucket["search_limit"] = max(bucket["search_limit"], int(row["search_limit"] or 0))
        bucket["scoped"] = bucket["scoped"] or scoped
        bucket["domain"] = domain

        if row["expires_at"] and (bucket["expires_at"] is None or row["expires_at"] > bucket["expires_at"]):
            bucket["expires_at"] = row["expires_at"]
        if row["created_at"] and (bucket["started_at"] is None or row["created_at"] < bucket["started_at"]):
            bucket["started_at"] = row["created_at"]

    created = reused = 0

    for (site_id, product_code), bucket in groups.items():
        existing = m.scalar_if_table("subscriptions", """
            SELECT id FROM subscriptions
            WHERE site_id = :site_id AND product_code = :product_code
        """, {"site_id": site_id, "product_code": product_code})

        if existing:
            reused += 1
            continue

        if bucket["scoped"]:
            status = "active"
            granted = bucket["search_limit"] or MODULE_PLAN_LADDER[0][1]
            plan, request_limit, adjustment = ladder_rung(MODULE_PLAN_LADDER, granted)
            _report_module_rung(m, bucket["domain"], product_code,
                                granted, plan, request_limit, adjustment)
        else:
            # A trial's allowance comes from the trial, not from whatever the
            # legacy key happened to grant. The whole point of writing it as
            # 'trial' rather than 'active' is that nobody has decided what this
            # store bought yet - so the plan says 'trial' too, and the number
            # beside it is the one catalog.py gives for that plan.
            status = "trial"
            plan = catalog.TRIAL_MODULE_PLAN
            request_limit = catalog.request_limit_for(catalog.TRIAL_MODULE_PLAN)

        m.change("""
            INSERT INTO subscriptions
                (id, site_id, product_code, status, plan, request_limit,
                 started_at, expires_at)
            VALUES
                (:id, :site_id, :product_code, :status, :plan, :request_limit,
                 COALESCE(:started_at, CURRENT_TIMESTAMP), :expires_at)
        """, {
            "id":            str(uuid.uuid4()),
            "site_id":       site_id,
            "product_code":  product_code,
            "status":        status,
            "plan":          plan,
            "request_limit": request_limit,
            "started_at":    bucket["started_at"],
            "expires_at":    bucket["expires_at"],
        })
        created += 1

    verb = "done" if m.apply_changes else "planned"
    m.note(verb, f"backfill subscriptions: {created} created, {reused} already present")
    m.detail("licences: 0 created - v1 keys are plaintext JWTs and cannot be hashed forward")


def _report_module_rung(m: Migration, domain: str, product_code: str, granted: int,
                        plan: str, request_limit: int, adjustment: str) -> None:
    """The module-ladder half of what backfill_sites says about the index ladder.

    This existed nowhere at all before: backfill_subscriptions discarded
    ladder_rung's third value entirely (`plan, request_limit, _ = ...`), so a
    quota that moved during the migration moved in complete silence.
    """
    if adjustment == RUNG_RAISED:
        m.warn(
            f"{domain}/{product_code}: monthly request allowance raised {granted} -> "
            f"{request_limit} to land on the '{plan}' rung. Allowances are never "
            f"rounded down; a tenant refused mid-month against a ceiling they used "
            f"to have has no explanation anyone can give them."
        )
    elif adjustment == RUNG_CAPPED:
        m.warn(
            f"{domain}/{product_code}: v1 granted {granted} requests a month, above "
            f"every rung catalog.MODULE_PLANS sells. CUT to {request_limit} on the "
            f"'{plan}' rung, losing {granted - request_limit} requests. The "
            f"alternative was writing plan='{plan}' beside request_limit={granted}, a "
            f"pair catalog.py cannot reproduce, which set_subscription_plan('{plan}') "
            f"would silently halve later anyway. Add a rung above '{plan}' to "
            f"catalog.MODULE_PLANS and re-run if the allowance is real."
        )


def move_webhook_secrets(m: Migration, site_ids: dict) -> set[str]:
    """Copy clients.webhook_secret onto that client's sites. Returns the orphans.

    A WooCommerce webhook is registered by a store, but v1 stored its HMAC
    secret per client - so a customer with two Woo stores overwrote one store's
    secret every time they re-registered the other, and three verification call
    sites then checked signatures against whichever won.

    The returned set is every client that HOLDS a secret and OWNS NO SITE, so
    the copy has nowhere to land. That case is the reason this function returns
    anything at all: a client with a secret but no site row - no license_keys
    row, a key with an empty allowed_domain, a site skipped for a collection
    clash - would otherwise have its secret dropped later in the same run by
    drop_client_columns, and the report that tells the operator to resolve the
    clash by hand would be printed after the data they need was already gone.
    A partial move used to be indistinguishable from a complete one, because
    the moved count was never compared against the number of clients holding a
    secret in the first place.
    """
    if not m.column_exists("clients", "webhook_secret"):
        m.note("present", "clients.webhook_secret already moved (skipped)")
        return set()

    holders = [
        row["id"] for row in m.rows("""
            SELECT id FROM clients
            WHERE webhook_secret IS NOT NULL AND webhook_secret <> ''
        """)
    ]

    # Every client that will own at least one site once this run finishes:
    # rows already in `sites`, plus rows this run created or planned to create.
    covered = {client_id for (client_id, _domain) in site_ids}
    if m.table_exists("sites"):
        covered.update(row["client_id"] for row in m.rows("SELECT client_id FROM sites"))

    orphans = {client_id for client_id in holders if client_id not in covered}

    would_move = 0
    if m.table_exists("sites"):
        would_move = m.scalar("""
            SELECT COUNT(*) AS row_total
            FROM sites s
            JOIN clients c ON c.id = s.client_id
            WHERE c.webhook_secret IS NOT NULL
              AND c.webhook_secret <> ''
              AND s.webhook_secret IS NULL
        """)

        m.change("""
            UPDATE sites s
            JOIN clients c ON c.id = s.client_id
            SET s.webhook_secret = c.webhook_secret
            WHERE c.webhook_secret IS NOT NULL
              AND c.webhook_secret <> ''
              AND s.webhook_secret IS NULL
        """)

        # A client with one secret and several stores had exactly one working
        # webhook registration; copying the secret to all of them does not make
        # the others work, it just stops them being silently wrong.
        for row in m.rows("""
            SELECT s.client_id AS client_id, COUNT(*) AS site_count
            FROM sites s
            JOIN clients c ON c.id = s.client_id
            WHERE c.webhook_secret IS NOT NULL AND c.webhook_secret <> ''
            GROUP BY s.client_id
            HAVING COUNT(*) > 1
        """):
            m.warn(
                f"client {row['client_id']} has {row['site_count']} sites sharing one "
                f"webhook secret. Only one of those stores was ever registered with it - "
                f"re-register the rest."
            )

    for client_id in sorted(orphans):
        m.warn(
            f"client {client_id} holds a webhook_secret and owns no site row, so there "
            f"is nowhere to move it to. Dropping clients.webhook_secret would destroy "
            f"it permanently and WooCommerce HMAC verification would then fail against "
            f"a secret nobody can recover. Give the client a site (or record the secret "
            f"somewhere) before running with --yes-drop-data."
        )

    verb = "done" if m.apply_changes else "planned"
    m.note(verb, f"move webhook_secret from {len(holders)} client(s) holding one "
                 f"onto {would_move} site(s); {len(orphans)} reach no site")
    if not m.table_exists("sites"):
        # Reading "onto 0 site(s)" here would be misleading rather than
        # informative: on a v1 database that create_tables has not yet built
        # on, the count is zero because there is nothing to count, not because
        # the copy would find nothing to do.
        m.detail("(0 because `sites` does not exist yet - nothing has been built to move them onto)")
    return orphans


def drop_client_columns(m: Migration, orphaned_secrets: set[str]) -> None:
    """Drop clients.plan and clients.webhook_secret, after the backfill.

    Both have moved to a level where they can be true. `plan` is per
    subscription now - a client with three modules on two plans was never
    describable by one string, which is why operator.py had to collapse it with
    MAX() and report a meaningless value for anyone with more than one licence.

    webhook_secret is refused while any client's secret reached no site,
    whatever the flags say. The value is not recoverable and re-registering a
    WooCommerce webhook without knowing why is a support ticket that starts
    from nothing.
    """
    for column, ddl in schema_v2.CLIENTS_DROP_COLUMNS:
        if not m.column_exists("clients", column):
            m.note("absent", f"clients.{column} (skipped)")
            continue

        if column == "webhook_secret" and orphaned_secrets:
            m.note("REFUSED", f"drop clients.{column} - {len(orphaned_secrets)} "
                              f"secret(s) reached no site (see warnings)")
            continue

        m.change(f"ALTER TABLE `clients` {ddl}",
                 label=f"drop clients.{column}", destructive=True)


def archive_v1_tables(m: Migration) -> None:
    """Rename the v1 ledger tables aside. Never drop them.

    token_usage_tracking holds the only real observed token counts and
    per-model costs in existence, and usage_logs.search_count is the only
    record of actual request volume per tenant per month. Those two numbers are
    what the new plan ladders have to be sized against, and the backfill
    creates zero usage_counters rows, so nothing else will ever hold them
    again. A rename moves no rows and costs nothing.

    It also forces an honest cutover: code still writing to the old name fails
    loudly instead of appending to a table nobody reads.

    A rename is NOT destructive in the sense --yes-drop-data guards. Nothing is
    lost and a second RENAME puts it back, so it runs under --apply alone.
    """
    for source, target in schema_v2.V1_TABLES_TO_ARCHIVE:
        if m.table_exists(target):
            # Both present means a previous run archived one table and
            # something recreated the original. Merging them is a judgement
            # call about real billing data, so make it visible instead of
            # guessing.
            if m.table_exists(source):
                m.warn(
                    f"both `{source}` and `{target}` exist. Left alone - a second archive "
                    f"would either overwrite the first or invent a name. Decide by hand "
                    f"which rows belong where."
                )
            m.note("present", f"{target} (skipped)")
            continue

        if not m.table_exists(source):
            m.note("absent", f"{source} - nothing to archive (skipped)")
            continue

        if not m.change(f"RENAME TABLE `{source}` TO `{target}`",
                        label=f"archive {source} -> {target}"):
            continue

        # The v1 tables have foreign keys onto clients(id), which follow the
        # table through the rename. Archive data must not constrain live
        # tables: leaving them means deleting a decommissioned dev client is
        # blocked by a row in a table nobody queries. Dropping a constraint
        # touches no rows.
        for constraint in m.foreign_keys(target):
            m.change(f"ALTER TABLE `{target}` DROP FOREIGN KEY `{constraint}`",
                     label=f"drop {target} foreign key {constraint}")


def redact_license_key_plaintext(m: Migration) -> None:
    """Blank license_keys.license_key, keeping the row as the reissue worklist.

    THIS IS THE POINT OF THE WHOLE REWRITE AND IT IS EASY TO MISS. v2 stores
    only the SHA-256 of a key, so that a database dump stops being a handover
    of every customer's working credential. That property is false for as long
    as v1's license_keys.license_key column is populated - it is a TEXT column
    holding every merchant's plaintext JWT, still valid, and the same string is
    the AES KEK that wraps their stored LLM, embedding and Magento API keys. A
    clean v2 schema sitting next to that table has fixed nothing.

    The row itself is kept: it is the worklist for the reissue campaign.
    _read_license_keys proves the worklist does not need the secret - it reads
    id, client_id, allowed_domain, the limits, is_active, expires_at,
    created_at, product_code and platform, and never the key.

    What this breaks, so nobody has to find out by watching it break:

      * onboarding.py's "already licensed for this product? hand back the same
        key" path SELECTs this column. It cannot be reimplemented against v2 at
        all - the plaintext is gone by design - so it has to become a mint.
      * webhooks.py / license_service.get_client_license derive the KEK from
        it. Both are being decommissioned: semantic-search-woo moves to the
        push method ai-product-qa-woo already uses, where the plugin presents
        the plaintext key on every request as a Bearer token and the server
        never reads one back out of the database.
      * every wrapped API key blob stored against these clients becomes
        undecryptable. Reissuing invalidates them anyway - a new licence key is
        a new KEK - so this destroys nothing the reissue was not already going
        to, but the merchant does have to re-enter their API keys.

    Destructive and irreversible, hence --yes-drop-data. Idempotent: a second
    run finds nothing left to blank.
    """
    if not m.table_exists("license_keys"):
        m.note("absent", "license_keys - no plaintext to redact (skipped)")
        return

    if not m.column_exists("license_keys", "license_key"):
        m.note("absent", "license_keys.license_key already gone (skipped)")
        return

    remaining = m.scalar("""
        SELECT COUNT(*) AS row_total FROM license_keys
        WHERE license_key IS NOT NULL AND license_key <> ''
    """)

    if not remaining:
        m.note("present", "license_keys.license_key already blank on every row (skipped)")
        return

    m.change("UPDATE license_keys SET license_key = '' WHERE license_key <> ''",
             label=f"redact plaintext licence key on {remaining} license_keys row(s)",
             destructive=True)


def drop_legacy_tables(m: Migration) -> None:
    """Drop the retired tables - but never one that has rows in it.

    The row guard is not a nicety and it is not gated on any flag. Three of
    these tables are chat_conversations, chat_messages and chat_feedback, and
    their absence from a deployment proves only that the chat route has never
    been called there: conversation_service.ensure_chat_tables() creates all
    three on demand at the top of every one of its public functions, and
    routers/chatbot.py calls into it on the live message route. Anywhere the
    chatbot has actually served a shopper, these hold the entire conversation
    history and nothing else does. MySQL cannot roll back a DROP.

    So a table with rows is reported and left alone, and the run carries on
    with the rest. An operator who genuinely means to destroy chat history can
    do it in one statement in a MySQL client, with the row count from this
    report in front of them - which is a decision, rather than a side effect of
    a migration that called itself safe to run repeatedly.
    """
    for table in schema_v2.LEGACY_TABLES_TO_DROP:
        if not m.table_exists(table):
            m.note("absent", f"table {table} (skipped)")
            continue

        rows = m.row_count(table)
        if rows:
            m.warn(
                f"TABLE `{table}` HAS {rows} ROW(S) AND WAS NOT DROPPED. The drop list "
                f"assumes this table is an unused artefact; {rows} rows say otherwise. "
                f"For the chat_* tables that is production conversation history - "
                f"conversation_service creates them on demand, so their absence "
                f"elsewhere is not evidence they are unused. Read the rows, decide what "
                f"they are, and drop the table by hand if you still want to. There is no "
                f"undo for a DROP."
            )
            m.note("REFUSED", f"drop table {table} - {rows} row(s)")
            continue

        m.change(f"DROP TABLE `{table}`", label=f"drop empty table {table}", destructive=True)


# ── Reporting ────────────────────────────────────────────────────────────────

# Everything whose row count belongs in the before/after, including the v1
# tables this migration does not touch. A table nobody names in the report is a
# table nobody remembers is still there - and all four of the untouched ones
# are still keyed on client_id rather than site_id, which is the same
# per-client/per-store confusion being fixed here for webhook_secret.
REPORTED_TABLES: list[str] = list(dict.fromkeys([
    "clients",
    "license_keys",
    "products",
    "sites",
    "subscriptions",
    "licences",
    "usage_events",
    "usage_counters",
    *[source for source, _ in schema_v2.V1_TABLES_TO_ARCHIVE],
    *[target for _, target in schema_v2.V1_TABLES_TO_ARCHIVE],
    *schema_v2.LEGACY_TABLES_TO_DROP,
    *schema_v2.V1_TABLES_LEFT_ALONE,
]))


def snapshot(m: Migration) -> dict:
    return {table: m.row_count(table) for table in REPORTED_TABLES}


def _fmt(count) -> str:
    return "-" if count is None else str(count)


def reissue_worklist(m: Migration) -> dict:
    """How many v1 keys have to be reissued, and how urgently.

    All of them, strictly: none can be carried forward. The split matters for
    sequencing - an active, unexpired key is a store that stops working the
    moment this lands, and those merchants need a new key in hand first.
    """
    if not m.table_exists("license_keys"):
        return {"total": 0, "active": 0, "live": 0}

    row = m.rows("""
        SELECT COUNT(*) AS total_count,
               SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END) AS active_count,
               SUM(CASE WHEN is_active = 1
                         AND (expires_at IS NULL OR expires_at > NOW())
                        THEN 1 ELSE 0 END) AS live_count
        FROM license_keys
    """)[0]

    return {
        "total":  int(row["total_count"] or 0),
        "active": int(row["active_count"] or 0),
        "live":   int(row["live_count"] or 0),
    }


def announce_destructive_phase(m: Migration, counts: dict) -> None:
    """Row counts and the exact teardown plan, printed BEFORE any of it runs.

    This used to be part of the summary at the end, which meant the operator
    learned how many rows they had destroyed strictly after destroying them.
    The before-snapshot was being taken correctly and then shown too late to be
    worth anything, which is the most annoying possible way to get this wrong.
    """
    logger.info("")
    logger.info("  about to enter the destructive phase")

    # "Removed" means dropped or blanked, and it deliberately does NOT cover the
    # archive renames, which are listed separately below because they run under
    # --apply alone. Saying "nothing below will be removed" over a block that
    # ended with the rename list told an operator running --apply that their
    # ledger was untouched, moments before it was renamed out from under them.
    if not m.allow_destructive:
        logger.info("    --yes-drop-data was not given: nothing will be dropped or")
        logger.info("    blanked. The archive renames further down still run under --apply.")
    elif not m.apply_changes:
        logger.info("    dry run: nothing below will be dropped, blanked or renamed.")
    else:
        logger.info("    --apply --yes-drop-data: THIS WILL REMOVE THE FOLLOWING. MySQL")
        logger.info("    cannot roll back DDL. Stop here if any row count surprises you.")

    logger.info("")
    logger.info("    tables on the drop list, with the rows they hold right now:")
    for table in schema_v2.LEGACY_TABLES_TO_DROP:
        count = counts.get(table)
        if count is None:
            logger.info("      %-28s not present", table)
        elif count:
            logger.info("      %-28s %d row(s)  <-- NOT DROPPED, has data", table, count)
        else:
            logger.info("      %-28s empty", table)

    logger.info("")
    logger.info("    columns to be dropped:   %s",
                ", ".join(f"clients.{column}" for column, _ in schema_v2.CLIENTS_DROP_COLUMNS))

    # The blast radius of the redaction, quantified BEFORE it runs. The count was
    # already computed inside redact_license_key_plaintext(), but only reached the
    # operator through print_report - after the UPDATE had committed - which is
    # the same "told too late to act on it" failure the row counts above fix.
    plaintext = m.scalar_if_table(
        "license_keys",
        "SELECT COUNT(*) FROM license_keys "
        "WHERE license_key IS NOT NULL AND license_key <> ''",
    )
    if plaintext:
        logger.info("    plaintext to be blanked: license_keys.license_key on %d row(s)", plaintext)
        logger.info("      Those keys are the AES key-encryption key for every merchant API")
        logger.info("      key wrapped under them (llm_key_service derives sha256 of the")
        logger.info("      licence key). Blanking them is irreversible and there is no copy:")
        logger.info("      each affected merchant must be reissued a key AND must re-enter")
        logger.info("      their provider API keys afterwards. Have that list ready first.")
    else:
        logger.info("    plaintext to be blanked: none present")

    logger.info("")
    logger.info("    renamed under --apply (NOT gated on --yes-drop-data), never dropped:")
    for source, target in schema_v2.V1_TABLES_TO_ARCHIVE:
        rows = counts.get(source)
        logger.info("      %-28s -> %-34s %s", source, target,
                    f"{rows} row(s)" if rows is not None else "not present")
    logger.info("")


def print_report(m: Migration, before: dict, after: dict, worklist: dict) -> None:
    logger.info("")
    logger.info("  actions")
    for action in m.actions:
        logger.info("    %s", action)

    logger.info("")
    logger.info("  row counts                    %10s  %10s", "before", "after")
    for table in REPORTED_TABLES:
        if before[table] is None and after[table] is None:
            continue
        logger.info("    %-28s %10s  %10s", table, _fmt(before[table]), _fmt(after[table]))

    if m.warnings:
        logger.info("")
        logger.info("  needs a human")
        # Deduplicated because a site with several unscoped keys raises the same
        # placeholder warning once per key, and a report nobody finishes reading
        # is a report that hid something.
        for warning in dict.fromkeys(m.warnings):
            logger.info("    - %s", warning)

    logger.info("")
    logger.info("  licence keys")
    logger.info("    %d v1 key(s) on file, %d active, %d active and unexpired.",
                worklist["total"], worklist["active"], worklist["live"])
    logger.info("    ALL of them must be reissued through onboarding. v1 stored the")
    logger.info("    plaintext JWT; v2 stores only the SHA-256 of a new opaque token, and")
    logger.info("    one cannot be derived from the other. license_keys is kept as the")
    logger.info("    worklist for that reissue, with the plaintext column blanked - the")
    logger.info("    worklist never needed the secret.")
    logger.info("")
    logger.info("    Reissuing also re-wraps stored API keys: the plaintext licence key is")
    logger.info("    the AES KEK for every merchant-supplied LLM and embedding key, so a")
    logger.info("    new key invalidates the old wrapped blobs.")
    logger.info("")

    if not m.apply_changes:
        logger.info("  DRY RUN - nothing above was written. Re-run with --apply to build")
        logger.info("  and backfill; add --yes-drop-data to also run the teardown.")
        logger.info("")


# ── Entry point ──────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Migrate the billing schema from the v1 client-scoped model to "
                    "the v2 site/subscription model.",
        epilog="With no flags this is a dry run: it reads the database, prints "
               "everything it would do, and writes nothing.",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually write. Without this the run is a dry run.",
    )
    parser.add_argument(
        "--yes-drop-data", action="store_true", dest="yes_drop_data",
        help="Also run the destructive phase: drop clients.plan and "
             "clients.webhook_secret, drop the retired tables (empty ones only), "
             "and blank the plaintext licence keys. Requires --apply to take "
             "effect; on its own it just includes the teardown in the dry-run plan.",
    )
    return parser.parse_args(argv)


def migrate(argv=None, _expose: dict | None = None) -> None:
    args = _parse_args(argv)

    with engine.connect() as conn:
        m = Migration(conn,
                      apply_changes=args.apply,
                      allow_destructive=args.yes_drop_data)
        # Handed straight out so _run() can report what committed if this
        # raises. Populated before any work starts, never read on success.
        if _expose is not None:
            _expose["m"] = m

        logger.info("")
        if m.apply_changes:
            logger.info("  APPLYING%s",
                        " (including the destructive phase)" if m.allow_destructive else "")
        else:
            logger.info("  DRY RUN - reading only, nothing will be written")

        preflight(m)
        before = snapshot(m)

        # DDL first. MySQL commits each of these implicitly regardless of what
        # this connection thinks its transaction is doing, so they are ordered
        # to leave a usable database at every point they could stop.
        create_tables(m)
        align_clients(m)
        m.commit()

        # Backfill. This half genuinely is transactional: it is all DML, and a
        # site without its subscriptions is worse than neither.
        seed_products(m)
        key_rows = _read_license_keys(m) if m.table_exists("license_keys") else []
        if not key_rows:
            m.note("note", "backfill skipped - no license_keys rows to read")

        # _read_license_keys inner-joins clients, so a key whose client row has
        # gone takes itself out of the backfill silently. The foreign key should
        # make that impossible, but foreign keys added under FOREIGN_KEY_CHECKS=0
        # do not validate existing rows, so say it out loud rather than trust it.
        on_file = m.row_count("license_keys") or 0
        if on_file != len(key_rows):
            m.warn(
                f"{on_file - len(key_rows)} license_keys row(s) reference a client that "
                f"is not in `clients`. Not migrated - they name no reachable customer."
            )
        site_ids = backfill_sites(m, key_rows)
        backfill_subscriptions(m, key_rows, site_ids)
        orphaned_secrets = move_webhook_secrets(m, site_ids)
        m.commit()

        # Teardown last, so nothing above ever reads a table this has removed.
        # The row counts go out FIRST, on their own, before a single statement
        # of this phase runs - see announce_destructive_phase.
        announce_destructive_phase(m, snapshot(m))
        drop_client_columns(m, orphaned_secrets)
        archive_v1_tables(m)
        redact_license_key_plaintext(m)
        drop_legacy_tables(m)
        m.commit()

        worklist = reissue_worklist(m)
        after = snapshot(m)

    print_report(m, before, after, worklist)


def _run(argv=None) -> int:
    """Entry point that guarantees a report even when the migration fails.

    migrate() had no exception handling, so print_report was reachable only on
    the fully-successful path. A run that died partway - a lock timeout, a
    column that would not convert, a disk filling - printed the traceback and
    nothing else, leaving the operator to work out from the traceback alone
    which phases had already committed. That is the worst moment to be guessing:
    this script commits per phase, so a mid-run failure means the database is in
    a state that is neither v1 nor v2.

    m.actions and m.warnings accumulate as the run goes, so everything needed to
    say what DID happen is already in hand by the time an exception unwinds.
    """
    m_ref: dict = {}
    try:
        migrate(argv, _expose=m_ref)
        return 0
    except Exception:
        migration = m_ref.get("m")
        logger.error("")
        logger.error("  MIGRATION ABORTED - the exception follows the summary below.")
        logger.error("  This script commits per phase, so the steps listed as done ARE")
        logger.error("  committed and will be skipped on the next run. Read them before")
        logger.error("  re-running, and do not assume the database is untouched.")
        if migration is not None:
            logger.error("")
            for action in migration.actions:
                logger.error("    done   %s", action)
            for warning in migration.warnings:
                logger.error("    warn   %s", warning)
            if not migration.actions:
                logger.error("    nothing had been committed when the run failed.")
        logger.error("")
        raise


if __name__ == "__main__":
    sys.exit(_run(sys.argv[1:]))
