"""
The billing ledger: usage_events, and the monthly rollup it feeds.

WHAT THIS MODULE OWNS
---------------------
Two tables and nothing else. `usage_events` is the evidence - one row per LLM or
embedding call, denormalised so a billing line stays explicable years after the
site was renamed, the subscription cancelled and the product withdrawn.
`usage_counters` is a cache of that evidence, one row per (subscription,
calendar month), so a quota check costs one primary-key read instead of an
aggregate over the largest table in the database.

Nothing else in the codebase may write to either table. If a second place that
increments usage_counters ever appears, that is v1's bug coming back: usage_logs
was incremented by two endpoints out of fifteen, so the only quota counter in
the system counted AI Search traffic and every chatbot tenant was structurally
un-quotable.

WHY THE ROW IS DENORMALISED, AND WHERE THE VALUES COME FROM
-----------------------------------------------------------
client_id, site_id, subscription_id, product_code, platform and key_owner are
copied onto every row from the context licensing_service.resolve_key() returned.
They are frozen facts at write time, not foreign keys - usage_events carries no
constraints at all, precisely so a row survives the deletion of everything it
names. None of them is re-derived here: in particular key_owner comes from ctx,
because it follows sites.environment and this module has no business reading
the sites table to second-guess the resolver. Re-deriving it from anything on
the wire is how a client's own spend gets booked as Czargroup's cost of goods.

That also means product identity arrives the only way it legitimately can. Three
Magento modules call the identical endpoints, so the route cannot tell them
apart; the licence resolved to exactly one subscription, and that subscription
named the product. Never accept a product_code from a request body here.

HOW THAT CONTEXT REACHES A SERVICE THAT WAS NEVER GIVEN ONE
-----------------------------------------------------------
Four of the write sites are shared services - embedder, llm_completion_service,
llm_rerank_service, chat_response_service - and every one of them receives a
bare client_id and nothing else. client_id alone cannot name a site, a
subscription or a product, so it cannot produce a row this module will accept.

The obvious fix is to thread a ctx parameter down to them. It is the wrong one:
embed_query and embed_document have twenty-five call sites between them, that
diff is beyond anyone's ability to review honestly, and every site it missed
would go on writing rows that are unattributable in a way no test would catch.

So record() reads the context implicitly when it is not passed one, from
backend/app/services/request_context, which holds what the auth chokepoint
resolved for the request currently being served. A shared service needs no new
parameter and no new call sites; it needs to call this module instead of the v1
one.

Implicit is not a synonym for optional. An EXPLICIT ctx always wins, and there
is one place that has to use it: a StreamingResponse generator runs after its
handler has returned, in a fresh copy of the event loop's context, so the
request-scoped value is provably absent there. That write site captures the ctx
in its closure and passes it. Everywhere else the ambient value is the ctx, and
a missing one is refused rather than guessed at - see NO CONTEXT below.

ACCOUNTING MUST NEVER BE THE REASON AN ANSWER FAILS
---------------------------------------------------
By the time record() is called the LLM call is already paid for and the shopper
is waiting. So every failure inside record() is swallowed - but never silently.
Read the comment above the TokenUsageTracker call in
backend/app/wordpress/productqa/routers/retrieve.py: a bare `except: pass` in
this position is what hid `wp_product_qa` being rejected as an unregistered
query_type, and every WooCommerce Q&A answer went unbilled for months because
the failure had nowhere to surface. Every swallow below logs at WARNING or
above, names the tenant, the call type, the provider/model and the amount, so
the cost is recoverable from the log file even when the row is not.

TELLING A CODE BUG FROM AN OUTAGE, IN THE LOG
---------------------------------------------
Swallowing is right. Swallowing everything into one undifferentiated WARNING is
not, because the three ways record() loses a row need three different responses
and the log line is the only place anyone will ever see any of them. So each
carries a marker, and nothing else in this module emits any of them:

  usage: CALLER BUG   The arguments are wrong - a ctx that is not the one
                      resolve_key() returned, or an empty call_type. NOT
                      transient: that call site will lose every row it writes
                      until somebody edits the code, and no amount of waiting or
                      retrying changes it. Logged at ERROR. One grep answers
                      "is this gap in the ledger a bad deploy or a bad network".
  usage: NO CONTEXT   Nobody could say WHO the spend belongs to: no ctx was
                      passed and the request scope was empty. Not a defect at
                      the call site and not an outage - during the dual-read
                      window it is the ordinary, correct state of a request that
                      authenticated on a v1 JWT, because a v1 key resolves to no
                      subscription. Logged at WARNING, with the amounts.
  usage: DATABASE     The statement failed - MySQL blinked, the connection
                      dropped, a deadlock, a disk full. Self-healing, and only
                      interesting by volume or by persistence. Logged at
                      WARNING, with the amounts, so the spend is recoverable.

CALLER BUG and NO CONTEXT look alike from a distance and must not be merged:
one means somebody reshaped a dict and every row from that site is lost until a
human edits code, the other means this particular request had no v2 identity to
copy. Merging them buries a release worth rolling back inside the migration's
own background noise. NO CONTEXT is WARNING and not ERROR for the same reason -
every key in the database is still a v1 key, so it fires on legitimate traffic
until licences are issued, and an ERROR per embedding call would train everyone
to filter the marker out before it ever carried signal. Its volume is the
migration's progress bar: when it stops, the cutover is finished.

The distinction used to be invisible: a renamed context field and a dropped
connection produced the same shape of line, and the first is a release to roll
back while the second is a pager. It is also why the ctx field list no longer
lives on this side of the import at all - see _TENANT_FIELDS, which makes that
particular caller bug a boot failure rather than a log line.

That is also why call_type is a free VARCHAR with no allowlist in this module.
An unfamiliar label must degrade to a row you can find with one SELECT DISTINCT,
never to a rejected INSERT. Same for `kind` and `key_owner`: an unexpected value
is logged and then written, not refused.

TRANSACTION CONTRACT - READ THIS BEFORE CALLING record()
--------------------------------------------------------
record() COMMITS the session it is handed. It has to: an accounting row that is
never committed is a lost billing row, which is the entire class of bug this
rewrite exists to kill. Call it once the request's own writes are final, not in
the middle of a multi-statement unit of work you are not ready to commit.

That rule is why track() exists next to it. A shared service holds no Session,
and the tempting fix - reach for the router's - is the one that breaks: a
catalogue push calls the embedder once per chunk, so record() would commit the
sync router's transaction twenty-five thousand times, each commit taking
whatever the router had half-written along with it. track() opens a short-lived
session of its own and closes it, which is exactly the shape v1's track_usage()
already had and the reason it was safe to call from anywhere.

That is the same rule tenancy_service and licensing_service state in their own
docstrings - a service commits its own writes and the caller does not own the
boundary - so all three can be reasoned about together instead of one at a time.
Where this module goes further: no REFUSAL path here rolls the caller's
transaction back, and no failed statement is allowed to poison it either. The
only db.rollback() below is after a failed COMMIT, where the transaction is
already dead and the alternative is handing the caller a session that raises
PendingRollbackError on its next statement.

Each statement it issues runs inside its own SAVEPOINT, which is the only reason
that contract is safe. A failed INSERT poisons a SQLAlchemy session - every
later statement on it raises PendingRollbackError - so without a savepoint the
choice would be between propagating the failure (breaking the answer) and
calling db.rollback() (throwing away whatever the caller had pending). The
savepoint rolls back exactly the statement that failed and leaves the caller's
work, and the session, intact.

The two statements get SEPARATE savepoints on purpose. The ledger row is
evidence and cannot be reconstructed; the counter is a cache of the ledger and
can be rebuilt by aggregating it. So if the counter upsert fails, the ledger row
must still land, and it does.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.services import catalog
# The session factory track() opens its own connection from. database.py imports
# nothing from this package, so there is no cycle, and create_engine() does not
# connect - importing this module still costs no round trip.
from backend.app.services.database import SessionLocal
# The authoritative key set of a resolved context, imported rather than
# restated. See _TENANT_FIELDS below for what that buys and what it cost not to
# have it. No cycle: licensing_service imports catalog and license_key and
# nothing else from this package, so this can sit at module scope rather than
# being deferred into a function body - a deferred import here would move the
# drift check back onto the hot path, which is the thing being fixed.
from backend.app.services.licensing_service import CONTEXT_FIELDS

logger = logging.getLogger(__name__)


# ── The request-scoped context ───────────────────────────────────────────────
#
# Holds the identity the auth chokepoint resolved for the request being served.
# record() reads it whenever a caller passes no ctx - see the module docstring
# for why the four shared services cannot pass one.
#
# The import is GUARDED, and that is a considered exception to this module's own
# fail-at-boot rule rather than an oversight. _context_subset() below refuses to
# import on a field RENAME, because that is a drift inside a module that exists
# and a five-minute fix. This is a different failure: request_context.py is new,
# and it is entirely possible for a deploy - or an rsync to the bind-mounted
# mirror, which copies file by file - to land usage_service.py a moment before
# it. A hard import there takes the entire API down, on a race nobody controls,
# to protect a feature that degrades perfectly well without it: explicit-ctx
# callers keep working, and implicit ones refuse their row and say so.
#
# So the absence is LOUD but not fatal - one ERROR here naming the module, and
# one NO CONTEXT warning per write that needed it. Nothing is silent and nothing
# is misattributed. Once request_context.py has shipped everywhere, make this a
# plain import: by then a missing module really is a broken deploy and refusing
# to boot is the correct response.
#
# Only get_context is imported. This module has no business setting or resetting
# a scope it does not own, and importing the setter would invite exactly that.
_CONTEXT_IMPORT_ERROR: Optional[str] = None
try:
    from backend.app.services.request_context import get_context as _get_request_context
except (ImportError, AttributeError) as _import_exc:
    _get_request_context = None
    _CONTEXT_IMPORT_ERROR = f"{type(_import_exc).__name__}: {_import_exc}"
    logger.error(
        "usage: NO CONTEXT - backend/app/services/request_context.get_context "
        "could not be imported (%s). Every write site that relies on the "
        "implicit request context - the shared services that receive only a "
        "client_id - will refuse its row and log NO CONTEXT until this "
        "resolves. Callers passing ctx= explicitly are unaffected.",
        _CONTEXT_IMPORT_ERROR,
    )


# ── Vocabulary ───────────────────────────────────────────────────────────────
#
# Known values only, NOT an allowlist. Nothing below refuses a write because a
# string is not in one of these tuples - see the docstring. They exist so an
# unexpected value produces a log line that names it, which is the difference
# between "we found the typo in an afternoon" and "wp_product_qa went unbilled
# for months".

KIND_SYNC = "sync"
KIND_SERVE = "serve"
KINDS = (KIND_SYNC, KIND_SERVE)

# Set from sites.environment by licensing_service.key_owner_for(). Repeated here
# only so a mangled ctx is visible in a log line.
KEY_OWNERS = ("czargroup", "client")


# ── Column limits, spelled out where the writer can see them ─────────────────
#
# schema_v2.py is the authority for these; they are duplicated here because the
# alternative is MySQL enforcing them. In strict mode an over-long value is
# error 1406 "Data too long" and the whole row is lost; in non-strict mode it is
# a truncation warning nobody reads. Both outcomes are worse than truncating in
# Python with a WARNING that names the field and prints the value in full.

_MAX_CALL_TYPE = 64        # usage_events.call_type      VARCHAR(64)
_MAX_PROVIDER = 50         # usage_events.provider       VARCHAR(50)
_MAX_MODEL = 100           # usage_events.model          VARCHAR(100)
_MAX_INTERACTION_ID = 64   # usage_events.interaction_id VARCHAR(64)
_MAX_KIND = 16             # usage_events.kind           VARCHAR(16)
_MAX_KEY_OWNER = 16        # usage_events.key_owner      VARCHAR(16)

# usage_events token columns are INT UNSIGNED. A value above this is a parsing
# bug in whatever read the provider's response, not a real token count.
_MAX_TOKENS = 4_294_967_295

# usage_events costs are DECIMAL(12,8): four integer digits, eight decimal
# places. One completion cannot legitimately cost four figures, so a value over
# the ceiling means a price table is wrong by orders of magnitude.
_COST_PLACES = Decimal("0.00000000")
_COST_CEILING = Decimal("9999.99999999")

# usage_counters.period is CHAR(7). A caller passing '2026-8' would read a row
# that cannot exist and be told the subscription has used nothing this month,
# which on the quota path means unlimited service.
_PERIOD_RE = re.compile(r"^\d{4}-(?:0[1-9]|1[0-2])$")

# ── The tenant identity copied onto every row ────────────────────────────────
#
# Six of the fields licensing_service.resolve_key() produces. Every one is NOT
# NULL in the schema and none has a defensible default: a row stamped with an
# empty product_code is worse than no row, because it still turns up in a SUM
# and can never be assigned to anybody.
#
# The names are CHECKED against licensing_service.CONTEXT_FIELDS at import,
# which is the entire point of the block below. This list used to be a private
# hand-copy - _REQUIRED_CTX_FIELDS - with nothing binding it to the function
# that defines the shape, and the two consumers failed asymmetrically: auth_cache
# rejected a wrong-shaped context loudly, while record() caught its own
# ValueError, logged a line and returned False. The shopper still got their
# answer, the request still cost money, and every billing row was lost. That is
# the wp_product_qa failure - a Python list and a schema drifting apart, eating
# the INSERT - reproduced one layer up, in the module written to end it.
#
# Now a rename in _context_from_row() fails on the line below, at import, before
# the process serves a request. A billing module that will not boot is a
# five-minute problem; one that boots and silently stops billing is not.

def _context_subset(*names: str) -> tuple[str, ...]:
    """Field names that must still exist in licensing_service.CONTEXT_FIELDS.

    usage_events has six tenant columns while the resolved context carries
    twenty-odd fields, so wanting a subset is legitimate. Choosing that subset
    by typing the names a second time is not - that is precisely the drift
    CONTEXT_FIELDS exists to end. Checking them against the authoritative set at
    import costs one pass over a frozenset, once, at boot.

    ImportError rather than ValueError because that is what the failure IS: this
    module asking a dependency for names it no longer exports. It is also the
    exception a reader expects when an application refuses to start, and it
    names both lists so the fix does not need a bisect.
    """
    unknown = [name for name in names if name not in CONTEXT_FIELDS]
    if unknown:
        raise ImportError(
            "usage_service stamps " + ", ".join(unknown) + " onto every "
            "usage_events row, but licensing_service.CONTEXT_FIELDS no longer "
            "contains " + ("them" if len(unknown) > 1 else "it") + ". "
            "_context_from_row() changed shape. Update the column list here - "
            "and usage_events itself if a column was renamed - rather than "
            "putting the old name back. The context now carries: "
            + ", ".join(sorted(CONTEXT_FIELDS)) + "."
        )
    return names


_TENANT_FIELDS = _context_subset(
    "client_id",
    "site_id",
    "subscription_id",
    "product_code",
    "platform",
    "key_owner",
)


# ── Time ─────────────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    """Naive UTC, matching licensing_service._utcnow().

    Naive because MySQL TIMESTAMP columns come back naive, and handing pymysql
    an aware datetime writes a value offset from every naive one already in the
    table. Not datetime.utcnow(), which is deprecated from 3.12 and this stack
    runs 3.13.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _as_naive_utc(value: datetime, field: str) -> datetime:
    """Force a caller-supplied datetime into the frame the columns are stored in.

    pymysql formats a datetime with strftime and drops the tzinfo, so an aware
    datetime in, say, +05:30 is compared as its local wall clock against values
    written in UTC - a five and a half hour window shift, applied silently, with
    the query still returning plausible-looking rows. Converting here means a
    caller who does the tz-aware thing gets the right answer instead of a subtly
    wrong one.
    """
    if not isinstance(value, datetime):
        raise ValueError(f"{field} must be a datetime, got {type(value).__name__}.")
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def current_period(moment: Optional[datetime] = None) -> str:
    """The 'YYYY-MM' bucket usage_counters rows are keyed by.

    UTC, in Python, in exactly one place - because the write path and both read
    paths have to agree on which row they mean. If record() incremented a UTC
    month while within_request_quota() read a local one, a merchant crossing a
    month boundary would be metered against a counter nobody was filling.

    Known asymmetry, and it is a live defect rather than a design choice:
    usage_events.created_at is written by MySQL's own DEFAULT CURRENT_TIMESTAMP,
    i.e. by the database server's clock in its SESSION time zone, and nothing
    sets one - database.py builds the engine with no connect_args and neither
    docker-compose.yml nor the override sets TZ. On the IST box this stack runs
    on that is +05:30, so a row written in the last five and a half hours of a
    month lands in usage_events under one month and in usage_counters under
    another. The counter is the number a merchant's quota is enforced against
    and the ledger is the evidence for it, so they disagree about which invoice
    the request belongs to.

    The fix is one line and it is not in this file:

        create_engine(..., connect_args={"init_command": "SET time_zone='+00:00'"})

    at backend/app/services/database.py:15, which today reads
    create_engine(DB_URL, pool_pre_ping=True). That one edit makes
    CURRENT_TIMESTAMP, this function, usage_by_product's window and
    licences.issued_at all agree at once. Until it lands, anything reconciling
    the counter against the ledger must bucket usage_events in UTC in Python,
    never with DATE_FORMAT(created_at, '%Y-%m').

    UTC wins over the server's clock here either way, because an invoice line
    names Czargroup's calendar month and that should not be a property of
    whichever time zone a container image happened to ship with.
    """
    return (moment or _utcnow()).strftime("%Y-%m")


def new_interaction_id() -> str:
    """A fresh id threading every row of one customer-visible action together.

    Mint it ONCE at the top of a turn and pass the same value to every record()
    call the turn makes - the intent classifier, the query decomposer, the
    rerank, the answer. That is what makes "show me everything the shopper's
    third question cost" a single indexed lookup, and what makes the rule that
    exactly one row per interaction carries billable=1 auditable.

    record() deliberately does NOT mint one for you when you pass None. A NULL
    interaction_id means a write site failed to thread it, which is a finding;
    auto-minting would turn that finding into four rows that each look like a
    separate, correctly-threaded interaction.
    """
    return str(uuid.uuid4())


# ── Coercion, so a bad value costs a log line and not the row ────────────────

def _brief(value, limit: int = 48) -> str:
    """Shorten a value for the context suffix carried on every log line here.

    Not cosmetic. That suffix is appended to a dozen different warnings, so an
    over-long model name or a garbage call_type would be reprinted in full on
    each one and turn a diagnosable failure into several kilobytes of log that
    nobody scrolls through. The offending value is still printed in full,
    exactly once, by the truncation warning that noticed it.
    """
    s = "" if value is None else str(value)
    return s if len(s) <= limit else s[:limit] + "..."


def _fit(value, limit: int, field: str, where: str) -> str:
    """Coerce to str and truncate to the column width, loudly.

    The alternative is MySQL deciding: error 1406 and a lost row under strict
    mode, a warning nobody reads otherwise.
    """
    s = "" if value is None else str(value)
    if len(s) <= limit:
        return s
    logger.warning(
        "usage: %s '%s' is %d chars, column holds %d - truncating (%s)",
        field, s, len(s), limit, where,
    )
    return s[:limit]


def _tokens(value, field: str, where: str) -> int:
    """Coerce to a value an INT UNSIGNED column will actually accept.

    Negatives are the trap worth knowing about: MySQL evaluates arithmetic on an
    unsigned column in unsigned space, so a negative count is error 1264 or 1690
    rather than a clamp, and it takes the whole row with it. A negative token
    count is always a bug in whatever parsed the provider's response, so writing
    0 and saying so keeps the cost and the attribution while flagging the bug.
    """
    try:
        n = int(value or 0)
    except (TypeError, ValueError):
        logger.warning("usage: %s is not a number (%r) - recording 0 (%s)", field, value, where)
        return 0
    if n < 0:
        logger.warning("usage: %s is negative (%d) - recording 0 (%s)", field, n, where)
        return 0
    if n > _MAX_TOKENS:
        logger.error(
            "usage: %s is %d, above the INT UNSIGNED ceiling - clamping (%s). "
            "This is a response-parsing bug, not a real token count.",
            field, n, where,
        )
        return _MAX_TOKENS
    return n


def _fit_pair(first, second, ceiling, label: str, where: str):
    """Shrink a pair so their sum fits *ceiling*, preserving total = a + b.

    Exists because the ledger's CHECK constraints require the stored total to
    equal the two halves exactly. Clamping each half independently and then
    clamping the sum satisfies the columns but violates the identity, and MySQL
    rejects the row rather than truncating it - losing a billing row that
    record() has already counted in usage_counters.

    Reaching here at all means an upstream token or cost parse produced numbers
    orders of magnitude beyond anything a real request generates, so this logs
    at ERROR: the clamp keeps the row and the attribution, but the input is a
    bug somewhere else and should not pass quietly.
    """
    total = first + second
    if total <= ceiling:
        return first, second

    # Keep the input half whole where possible - it is the prompt we built, so
    # it is the number we can most nearly vouch for.
    kept = first if first <= ceiling else ceiling
    remainder = ceiling - kept

    logger.error(
        "usage: %s sum %s exceeds the column ceiling %s (%s). Clamped to "
        "%s + %s so the stored total still equals its halves; the row is kept "
        "but the upstream count is wrong and should be investigated.",
        label, total, ceiling, where, kept, remainder,
    )
    return kept, remainder


def _money(value, field: str, where: str) -> Decimal:
    """Coerce to a Decimal quantized to the ledger's eight decimal places.

    Decimal and not float, and quantized HERE rather than left to MySQL, for one
    specific reason: record() adds the same number to the ledger row and to the
    monthly counter. If both sides went in as floats, MySQL would round each
    independently on the way into DECIMAL and SUM(usage_events.total_cost) would
    stop matching usage_counters.total_cost by a few units in the last place per
    row - which over a month of a busy tenant is a reconciliation nobody can
    close. Rounding once, in Python, makes the two identical by construction.

    Decimal(str(x)) rather than Decimal(x): a float carries its full binary
    expansion into Decimal (1.1 becomes 1.100000000000000088817841...), and
    while quantize would round that away, the repr is what the provider's price
    table actually meant.
    """
    try:
        amount = Decimal(str(value if value is not None else 0))
    except (InvalidOperation, TypeError, ValueError):
        logger.warning("usage: %s is not a number (%r) - recording 0 (%s)", field, value, where)
        return _COST_PLACES

    if not amount.is_finite():
        # inf/NaN reach here when a price table divides by a zero token count.
        logger.warning("usage: %s is %s - recording 0 (%s)", field, amount, where)
        return _COST_PLACES
    if amount < 0:
        logger.warning("usage: %s is negative (%s) - recording 0 (%s)", field, amount, where)
        return _COST_PLACES
    if amount > _COST_CEILING:
        logger.error(
            "usage: %s is %s, above the DECIMAL(12,8) ceiling - clamping to %s (%s). "
            "A single call cannot cost four figures; check the price table.",
            field, amount, _COST_CEILING, where,
        )
        return _COST_CEILING
    # Known and harmless: pymysql escapes a Decimal with str(), and str() of a
    # value below 1e-6 uses exponent notation - a one-cent-per-million embedding
    # call goes onto the wire as 1E-8. MySQL parses that as an approximate
    # literal and converts it to DECIMAL(12,8) on assignment. It round-trips
    # exactly, because every value that triggers the notation has at most three
    # significant digits and a double carries fifteen. Do not "fix" it by
    # binding a quoted string: a quoted numeric is a different implicit
    # conversion with its own strict-mode behaviour.
    return amount.quantize(_COST_PLACES, rounding=ROUND_HALF_UP)


def _implicit_ctx() -> Optional[dict]:
    """The context resolved for the request being served, or None.

    NEVER RAISES, and that is the whole contract. By the time anything here runs
    the model call has been made and paid for and a shopper is waiting on the
    answer; a bug in the context module turning into a 500 would make this
    module the reason a request failed, which is the one thing the docstring
    above forbids outright. An unreadable scope is treated as an absent one -
    the row is refused, loudly, and the answer still goes out.

    NEVER LOG THE RETURNED DICT WHOLE, and check that again before adding a
    `%r` or a `%s` on it to any line in this module. request_context.get_context
    hands back the chokepoint's own license_data, which carries the presented
    licence key in PLAIN TEXT under "license_key" - and that key is the
    key-encryption key every merchant-supplied LLM and embedding key is
    encrypted under, so one interpolation of it here would write credentials
    for every tenant into the log file this module is otherwise designed to
    make people read. Everything below logs field NAMES, the six tenant
    identifiers, and amounts. Nothing logs a value off the context it did not
    name explicitly.
    """
    if _get_request_context is None:
        return None
    try:
        return _get_request_context()
    except Exception as exc:
        logger.warning(
            "usage: NO CONTEXT - reading the request-scoped context raised "
            "(%s: %s). Treating it as absent; the row will be refused.",
            type(exc).__name__, exc,
        )
        return None


def _is_tenant_shaped(ctx) -> bool:
    """True when *ctx* carries every identifier a usage row needs.

    The one question worth asking before deciding a context is usable. A dict is
    not evidence of identity: during the dual-read window a v1 JWT resolves to a
    populated license_data with no site, subscription or product anywhere in it,
    and that is a legitimate request rather than a broken one.
    """
    if not isinstance(ctx, dict):
        return False
    return all(ctx.get(field) for field in _TENANT_FIELDS)


def _tenant_fields(ctx: dict) -> dict:
    """The six identifiers stamped onto every row, or ValueError naming the gaps.

    Deliberately strict where the rest of this module is forgiving. Everything
    else record() touches describes WHAT was spent and degrades usefully to a
    truncated string or a zero; these six say WHO spent it, and a row that
    cannot answer that is not a cheaper billing row, it is a permanent piece of
    noise in every product and tenant aggregate that follows.

    A ctx straight from licensing_service.resolve_key() always has all six -
    _TENANT_FIELDS is checked against CONTEXT_FIELDS at import, so the six names
    cannot have drifted away from the ones that function produces. Reaching the
    raise therefore means a caller hand-built a context or reshaped one, and the
    caller is what needs fixing: this is the CALLER BUG case in the module
    docstring, never a transient one.
    """
    if not isinstance(ctx, dict):
        raise ValueError(f"ctx must be the dict resolve_key() returns, got {type(ctx).__name__}.")

    missing = [f for f in _TENANT_FIELDS if not ctx.get(f)]
    if missing:
        raise ValueError(
            "ctx is missing " + ", ".join(missing) +
            " - it must be the context licensing_service.resolve_key() returned."
        )

    return {f: str(ctx[f]) for f in _TENANT_FIELDS}


# ── The write path ───────────────────────────────────────────────────────────

_INSERT_EVENT = text("""
    INSERT INTO usage_events (
        client_id, site_id, subscription_id, product_code, platform,
        key_owner, kind, billable, interaction_id, call_type,
        provider, model,
        input_tokens, output_tokens, total_tokens,
        input_cost, output_cost, total_cost
    ) VALUES (
        :client_id, :site_id, :subscription_id, :product_code, :platform,
        :key_owner, :kind, :billable, :interaction_id, :call_type,
        :provider, :model,
        :input_tokens, :output_tokens, :total_tokens,
        :input_cost, :output_cost, :total_cost
    )
""")

# The increments are written as arithmetic on the stored column, never as a
# read-then-write, so the row lock does the serialising. Two shoppers on the
# same module in the same millisecond is the ordinary case, and a
# read-modify-write loses one of the two increments every time it happens - a
# quota counter that undercounts exactly when the tenant is busiest.
#
# The bound parameter is repeated in the UPDATE clause instead of the more usual
# VALUES(col): VALUES() is deprecated from MySQL 8.0.20, its replacement (the
# `AS new` row alias) does not exist in MariaDB, and this form needs neither.
_UPSERT_COUNTER = text("""
    INSERT INTO usage_counters (
        subscription_id, period, billable_requests, total_tokens, total_cost
    ) VALUES (
        :subscription_id, :period, :requests, :tokens, :cost
    )
    ON DUPLICATE KEY UPDATE
        billable_requests = billable_requests + :requests,
        total_tokens      = total_tokens + :tokens,
        total_cost        = total_cost + :cost
""")


def record(
    db: Session,
    ctx: dict,
    call_type: str,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    input_cost,
    output_cost,
    kind: str,
    billable: bool = False,
    interaction_id: Optional[str] = None,
) -> bool:
    """Write one ledger row, and fold it into the month's counter.

    *ctx* is the context licensing_service.resolve_key() returned. client_id,
    site_id, subscription_id, product_code, platform and key_owner are copied
    off it onto the row - not looked up, not derived - so the row still reads
    correctly years later when the site has been renamed and the subscription
    cancelled.

    PASS None FOR *ctx* to use the context of the request being served. That is
    how a shared service holding nothing but a client_id writes an attributable
    row without growing a parameter and twenty-five call sites of plumbing. An
    explicit ctx always wins over the ambient one; pass it explicitly from any
    code that runs after its handler returned - a StreamingResponse generator -
    because the request scope is provably empty there. When neither channel
    produces a context the row is refused and logged as NO CONTEXT, never
    written with null or invented tenant columns.

    *kind* has no default and is not derivable from *call_type*: an embedding
    call is 'sync' during a catalogue push and 'serve' when it embeds a
    shopper's question, and the same call_type covers both. Getting it wrong
    mixes indexing spend into shopper traffic on every dashboard.

    *billable* must be True on EXACTLY ONE row per customer-visible action. That
    row is what the quota counts. The other rows of the same turn - the intent
    classification, the decomposition, the rerank - share its interaction_id and
    contribute their cost, but must not each count as a request; a chatbot turn
    would otherwise burn five requests of a merchant's allowance for one answer.

    *interaction_id* defaults to the one on *ctx*, which the chokepoint minted
    for the turn - so a write site normally passes nothing at all. Pass it only
    to override that. See new_interaction_id() for why a missing one is left as
    NULL rather than invented here.

    Returns True if the ledger row landed. THIS IS FOR TESTS AND LOGGING ONLY.
    Never turn a False into a customer-facing error, and never retry on it - the
    answer has already been produced and paid for, and this module failing is
    not the shopper's problem.

    COMMITS the session (see the module docstring). Call it when the request's
    own writes are final.
    """
    where = (
        f"call_type={_brief(call_type)} provider={_brief(provider)} "
        f"model={_brief(model)}"
    )

    # ── Which channel supplied the identity ──────────────────────────────────
    #
    # An explicit ctx always wins over the ambient one. That ordering is not a
    # style preference, it is the only thing that makes the streaming answer
    # path attributable: its generator body runs after the handler returned, in
    # a fresh copy of the event loop's context, so the request-scoped value is
    # provably empty there and that write site has to capture the ctx in its
    # closure and pass it. Preferring the ambient value would overwrite a
    # correctly captured context with nothing, on the busiest endpoint, and the
    # only visible symptom would be billing rows that stopped appearing.
    #
    # Falsy rather than `is None`, deliberately. The request scope hands out a
    # mutable box that is EMPTIED at the end of the request rather than
    # replaced, precisely so every copy of the context expires at once. An
    # expired or never-filled scope therefore reads as {}, not as None. Testing
    # identity here would accept that empty dict as a real context and hand it
    # to _tenant_fields(), turning a missing identity into a CALLER BUG at
    # ERROR - the wrong diagnosis, at the wrong severity, naming an innocent
    # call site.
    ctx_source = "arg"
    if not _is_tenant_shaped(ctx):
        ctx = _implicit_ctx()
        ctx_source = "scope"
    if not _is_tenant_shaped(ctx):
        # Shape, not truthiness. Defence in depth behind the chokepoint's own
        # v2-only bind: a v1 JWT license_data is a truthy dict carrying none of
        # the six tenant identifiers, and testing `if not ctx` let it through to
        # _tenant_fields(), which raised and reported the loss as CALLER BUG at
        # ERROR - blaming a call site whose arguments were fine for a request
        # that simply had no v2 identity to copy. Reducing it to None here means
        # the NO CONTEXT branch below sees it, which is the honest diagnosis.
        ctx = None

    where = f"{where} ctx={ctx_source}"

    if not ctx:
        # NO CONTEXT gets its own marker because neither of the other two would
        # send whoever greps it anywhere useful. This is not a defect at this
        # call site - the arguments are fine - and it is not an outage. The
        # REQUEST carried no v2 identity, so there is nothing to copy onto the
        # row, and refusing it is the only honest outcome: usage_events has six
        # NOT NULL tenant columns and no defensible default for any of them, and
        # a row that cannot say who spent the money is not a cheaper billing row
        # but permanent noise in every aggregate that follows.
        #
        # The three things that cause it, in the order worth checking:
        #   1. The request authenticated on the v1 JWT path. A v1 key resolves
        #      to no subscription, so there is no v2 context to set. Expected,
        #      and today it is every request - see the module docstring on why
        #      this is a WARNING and not an ERROR.
        #   2. The write happens outside the request scope - a StreamingResponse
        #      generator, or anything else running after the handler returned.
        #      Pass ctx= explicitly there. The ambient value cannot reach that
        #      code and no amount of setting it harder will change that.
        #   3. The endpoint has no chokepoint, so nothing set a context at all.
        #
        # The amounts go in the line so the spend stays recoverable from the log
        # even though the row is refused - the same rule every swallow here
        # follows.
        logger.warning(
            "usage: NO CONTEXT - refusing an unattributable row (%s). No ctx "
            "argument and nothing in the request scope%s. Expected while a "
            "tenant is still on a v1 key; otherwise this write site is outside "
            "the request scope and must pass ctx= explicitly, or its endpoint "
            "has no chokepoint. tokens in=%s out=%s cost in=%s out=%s",
            where,
            f" ({_CONTEXT_IMPORT_ERROR})" if _CONTEXT_IMPORT_ERROR else "",
            input_tokens, output_tokens, input_cost, output_cost,
        )
        return False

    # Everything up to the first db.execute() is pure. A ctx that cannot name a
    # tenant fails here, before any statement has touched the caller's
    # transaction, so the swallow below cannot leave a half-written interaction.
    try:
        fields = _tenant_fields(ctx)
    except ValueError as exc:
        # ERROR and marked CALLER BUG, because this is not a database hiccup and
        # must not read like one. A failed INSERT is infrastructure: transient,
        # self-healing, worth a pager only on volume. This is a code defect -
        # somebody handed record() a dict resolve_key() did not produce - and it
        # will lose every row that call site writes until a human edits it. The
        # two need opposite responses, and the log line is the only place anyone
        # will ever see either. See the module docstring.
        #
        # It can no longer be caused by a field RENAME: _TENANT_FIELDS is
        # checked against licensing_service.CONTEXT_FIELDS at import, so that
        # variant fails at boot instead of arriving here one lost row at a time.
        # What is left is a hand-built or reshaped context, which is the case
        # this line is worded for.
        #
        # The amounts go in the line so the spend is still recoverable from the
        # log when the row is not.
        logger.error(
            "usage: CALLER BUG (context shape) - refusing an unattributable row "
            "(%s): %s. This is a code defect, not a database failure: retrying "
            "will not help and every row from this call site is lost until it is "
            "fixed. tokens in=%s out=%s cost in=%s out=%s",
            where, exc, input_tokens, output_tokens, input_cost, output_cost,
        )
        return False

    where = (
        f"subscription={fields['subscription_id']} product={fields['product_code']} "
        f"{where}"
    )

    # interaction_id rides in on the context, so no write site has to thread it.
    # The chokepoint mints one per turn and puts it in the resolved context;
    # every row of that turn then shares it at zero plumbing cost, which is what
    # makes "show me everything the shopper's third question cost" one indexed
    # lookup, and what makes the exactly-one-billable-row-per-interaction rule
    # auditable at all.
    #
    # An explicit argument still wins, because a caller that genuinely knows
    # better - stitching several requests into one unit of work - has to be able
    # to say so, and because the streaming write site passes an explicit ctx and
    # must be free to pass an explicit id with it.
    #
    # This is still NOT minting. new_interaction_id() explains why record()
    # refuses to invent one: a NULL means a write site ran with no interaction
    # to belong to, and that is a finding worth keeping. Reading a value the
    # chokepoint already put on the context is not inventing one - if it did not
    # put one there, this stays NULL and the finding survives intact.
    if interaction_id is None:
        interaction_id = ctx.get("interaction_id")

    if fields["key_owner"] not in KEY_OWNERS:
        # Written anyway. key_owner decides whether total_cost is the merchant's
        # spend or Czargroup's cost of goods, so an unrecognised value is a
        # reporting problem worth flagging - but refusing the row would delete
        # the cost evidence entirely, which is strictly worse.
        logger.warning(
            "usage: unrecognised key_owner '%s' (%s) - expected one of %s. "
            "Cost on this row cannot be classified as COGS or merchant spend.",
            fields["key_owner"], where, KEY_OWNERS,
        )

    kind_value = _fit(kind, _MAX_KIND, "kind", where)
    if kind_value not in KINDS:
        logger.warning(
            "usage: unfamiliar kind '%s' (%s) - expected one of %s. Recording it "
            "as given; indexing and serving spend will not separate cleanly.",
            kind_value, where, KINDS,
        )

    in_tokens = _tokens(input_tokens, "input_tokens", where)
    out_tokens = _tokens(output_tokens, "output_tokens", where)
    in_cost = _money(input_cost, "input_cost", where)
    out_cost = _money(output_cost, "output_cost", where)

    # total_* are stored, not computed on read, so that a report can sum one
    # column instead of two and so the archive table's shape is preserved. They
    # are derived from the already-clamped halves, which keeps
    # SUM(input_cost + output_cost) exactly equal to SUM(total_cost) - a
    # reconciliation that stops holding the moment either side is rounded twice.
    #
    # The halves must be clamped so their SUM fits, not just so each half fits.
    # chk_usage_events_total_tokens and chk_usage_events_total_cost require
    # total = input + output exactly, and clamping the sum on its own breaks that
    # identity: input=3e9 and output=2e9 are each under the INT UNSIGNED ceiling
    # so neither half logs anything, while the sum clamps to 4294967295 and the
    # CHECK then rejects the whole row. The failure is the bad one - record()
    # catches, logs, returns False, and the billing row is lost while the monthly
    # counter has already moved, so the ledger and the counter disagree with
    # nothing to reconcile them from.
    #
    # Truncating the output half rather than scaling both: input tokens are the
    # prompt we sent and are the half we can most nearly vouch for, so if one
    # number has to give it should be the one we did not construct.
    in_tokens, out_tokens = _fit_pair(
        in_tokens, out_tokens, _MAX_TOKENS, "tokens", where,
    )
    in_cost, out_cost = _fit_pair(
        in_cost, out_cost, _COST_CEILING, "cost", where,
    )

    total_tokens = in_tokens + out_tokens
    total_cost = _money(in_cost + out_cost, "total_cost", where)

    billable_flag = 1 if billable else 0

    params = {
        **fields,
        "kind": kind_value,
        "billable": billable_flag,
        "interaction_id": (
            _fit(interaction_id, _MAX_INTERACTION_ID, "interaction_id", where)
            if interaction_id else None
        ),
        "call_type": _fit(call_type, _MAX_CALL_TYPE, "call_type", where),
        "provider": _fit(provider, _MAX_PROVIDER, "provider", where),
        "model": _fit(model, _MAX_MODEL, "model", where),
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "total_tokens": total_tokens,
        "input_cost": in_cost,
        "output_cost": out_cost,
        "total_cost": total_cost,
    }

    if not params["call_type"]:
        # NOT NULL with no default. An empty string would satisfy MySQL and then
        # sit in the dashboard as an unnamed bucket forever. Same marker as the
        # ctx failure above and for the same reason: a write site that passes no
        # call_type passes no call_type on every request it serves.
        logger.error(
            "usage: CALLER BUG (empty call_type) - refusing the row (%s). "
            "tokens in=%d out=%d cost=%s",
            where, in_tokens, out_tokens, total_cost,
        )
        return False

    # ── Statement 1: the evidence ────────────────────────────────────────────
    #
    # Its own savepoint. See the module docstring: a failed statement poisons
    # the session, and rolling the whole transaction back would take the
    # caller's uncommitted work with it.
    event_written = False
    try:
        with db.begin_nested():
            db.execute(_INSERT_EVENT, params)
        event_written = True
    except Exception as exc:
        # DATABASE, not CALLER BUG: the parameters were already accepted by
        # every check above, so what failed here is the statement - a dropped
        # connection, a deadlock, a full disk, a column that does not match
        # schema_v2.py. WARNING because it is transient and self-healing;
        # interesting by volume and by persistence, not one at a time.
        logger.warning(
            "usage: DATABASE - ledger row not written (%s): %s. tokens in=%d "
            "out=%d cost=%s billable=%s interaction=%s",
            where, exc, in_tokens, out_tokens, total_cost, billable_flag,
            params["interaction_id"],
        )

    # ── Statement 2: the rollup ──────────────────────────────────────────────
    #
    # Attempted even when the ledger row failed, and that is deliberate. The
    # counter is what enforces quota, and a control that stops counting whenever
    # the ledger write hiccups is a control that whatever is breaking the ledger
    # can defeat. The request really did happen; the log line above is the
    # evidence for the row that did not land, and the divergence between the
    # counter and SUM(usage_events) is itself the signal that something is
    # wrong. The reverse ordering - refusing to count what we cannot evidence -
    # reads as principled and hands a broken tenant unlimited service.
    #
    # billable_requests moves only on a billable row. total_tokens and
    # total_cost accumulate every row, so the month's cost in the counter equals
    # the month's cost in the ledger, and only the request COUNT is the
    # one-per-action number.
    counter_written = False
    requests = 1 if billable else 0
    if requests or total_tokens or total_cost:
        try:
            with db.begin_nested():
                db.execute(_UPSERT_COUNTER, {
                    "subscription_id": fields["subscription_id"],
                    "period": current_period(),
                    "requests": requests,
                    "tokens": total_tokens,
                    "cost": total_cost,
                })
            counter_written = True
        except Exception as exc:
            logger.warning(
                "usage: DATABASE - monthly counter not updated for subscription "
                "%s period %s (%s): %s. requests=+%d tokens=+%d cost=+%s. The "
                "ledger row is %s; rebuild the counter from usage_events if this "
                "persists.",
                fields["subscription_id"], current_period(), where, exc,
                requests, total_tokens, total_cost,
                "written" if event_written else "also missing",
            )
    else:
        # Nothing to add: a non-billable row with no tokens and no cost, e.g. a
        # cache hit recorded for traceability. Skipping the statement avoids
        # creating an all-zero counter row and avoids a pointless write on the
        # hottest INSERT path in the system - a full catalogue sync calls this
        # function once per embedded chunk.
        counter_written = True

    if not (event_written or counter_written):
        # Both savepoints rolled back, so there is nothing of ours to commit and
        # the caller's transaction is exactly as we found it. Leaving it alone
        # is the honest outcome; committing here would only make this module
        # responsible for someone else's half-finished work.
        return False

    try:
        db.commit()
    except Exception as exc:
        # A commit can still fail on a deadlock or a dropped connection. Same
        # rule: log it, do not raise it at a shopper.
        logger.warning(
            "usage: DATABASE - commit failed after recording (%s): %s. "
            "tokens=%d cost=%s billable=%d - this spend is in no table.",
            where, exc, total_tokens, total_cost, billable_flag,
        )
        try:
            db.rollback()
        except Exception:
            # The session is unusable either way; the caller's `finally: close()`
            # will deal with it. Raising here would defeat the entire point.
            logger.warning(
                "usage: DATABASE - rollback after a failed commit also failed (%s)",
                where,
            )
        return False

    # DEBUG, and it earns its place during the migration specifically: `ctx=arg`
    # versus `ctx=scope` in this line is the only direct evidence that the
    # request-scoped context actually reached a shared service. The alternative
    # way to find out is to read usage_events afterwards and infer it, which
    # cannot distinguish "the scope worked" from "the caller passed a ctx".
    logger.debug(
        "usage: recorded %s tokens=%d cost=%s billable=%d interaction=%s (%s)",
        params["call_type"], total_tokens, total_cost, billable_flag,
        params["interaction_id"], where,
    )
    return event_written


def track(
    call_type: str,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    input_cost,
    output_cost,
    kind: str,
    *,
    ctx: Optional[dict] = None,
    billable: bool = False,
    interaction_id: Optional[str] = None,
) -> bool:
    """record(), for a call site that holds neither a Session nor a context.

    This is the entry point the four shared services use - embedder,
    llm_completion_service, llm_rerank_service, chat_response_service. Each of
    them receives a client_id and nothing else: no Session, no request, no
    context. They get the tenant from the request scope, and the transaction
    from here.

    IT OPENS ITS OWN SESSION, and that is the point rather than a convenience.
    record() commits what it is handed, so the tempting alternative - reaching
    for the router's session - is the one that breaks: a catalogue push calls
    the embedder once per chunk, so a shared session would commit the sync
    router's transaction once per embedded chunk, twenty-five thousand times on
    a real catalogue, each commit taking whatever the router had half-written
    with it. A short-lived session cannot do that to anybody. It is also exactly
    the shape v1's track_usage() had - open, write, close in a finally - which
    is why that function was safe to call from anywhere, and keeping the shape
    keeps the diff at those four sites to the name being called.

    It is one connection checkout per write, from a pool built with SQLAlchemy's
    defaults (five plus ten overflow, database.py sets no size). That is the
    honest cost of not threading a Session through twenty-five call sites, and
    it is the same cost v1 paid; it is worth knowing about on the sync path,
    where the write rate is highest.

    *ctx* is keyword-only and normally omitted - the request scope supplies it.
    Pass it from code running outside that scope, which in this tree means a
    StreamingResponse generator and the callers reached from one. Everything
    else behaves exactly as record() documents, including returning False rather
    than raising: never turn this value into a customer-facing error.
    """
    try:
        db = SessionLocal()
    except Exception as exc:
        # Pool exhausted, or the database is unreachable. DATABASE rather than
        # CALLER BUG - nothing about the arguments is wrong - and the amounts go
        # in the line so the spend is still recoverable from the log.
        logger.warning(
            "usage: DATABASE - could not open a session to record %s spend "
            "(provider=%s model=%s): %s. tokens in=%s out=%s cost in=%s out=%s",
            _brief(call_type), _brief(provider), _brief(model), exc,
            input_tokens, output_tokens, input_cost, output_cost,
        )
        return False

    try:
        return record(
            db, ctx, call_type, provider, model,
            input_tokens, output_tokens, input_cost, output_cost,
            kind, billable=billable, interaction_id=interaction_id,
        )
    finally:
        try:
            db.close()
        except Exception:
            # Returning the connection to the pool is best effort. Letting this
            # propagate would convert a bookkeeping detail into the 500 that
            # every swallow in record() exists to prevent - and it would do it
            # from a finally, so it would replace record()'s return value on the
            # way out and lose the outcome as well.
            logger.warning(
                "usage: DATABASE - closing the session opened for %s failed",
                _brief(call_type),
            )


# ── The read paths ───────────────────────────────────────────────────────────

def counter_for(db: Session, subscription_id: str, period: Optional[str] = None) -> dict:
    """The monthly rollup for one subscription. Zeros when the month is untouched.

    Returns zeros rather than None for a month with no rows, because "this
    module has not been used yet this calendar month" is the ordinary state on
    the first of every month and every caller would otherwise need the same
    `or {}` dance - which is where a None slips through into a comparison and
    quietly disables a check.

    RAISES on a lookup failure, and does not fail open. This is the dashboard
    and invoicing entry point: showing a merchant a confident zero because MySQL
    was unreachable is a lie with a number attached. The quota path has the
    opposite requirement and gets it in within_request_quota(), which wraps this.
    """
    period = period or current_period()
    if not _PERIOD_RE.match(period):
        # A malformed period reads a row that cannot exist and reports no usage,
        # which on the quota path means unlimited service. Fail here instead.
        raise ValueError(f"period must be 'YYYY-MM', got '{period}'.")

    row = db.execute(text("""
        SELECT billable_requests, total_tokens, total_cost, updated_at
        FROM usage_counters
        WHERE subscription_id = :subscription_id AND period = :period
    """), {"subscription_id": subscription_id, "period": period}).fetchone()

    if row is None:
        return {
            "subscription_id": subscription_id,
            "period": period,
            "billable_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "updated_at": None,
        }

    return {
        "subscription_id": subscription_id,
        "period": period,
        "billable_requests": int(row.billable_requests or 0),
        "total_tokens": int(row.total_tokens or 0),
        # float for display. An invoice must read the DECIMAL column itself
        # rather than this value - eight decimal places of a monthly sum do not
        # survive a round trip through a double.
        "total_cost": float(row.total_cost or 0),
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def within_request_quota(
    db: Session,
    subscription_id: str,
    request_limit: int,
) -> tuple[bool, int, int]:
    """Has this subscription got a billable request left this month?

    Returns (ok, used, limit). All three, not just the verdict, because every
    caller that refuses a request has to tell the merchant why, and "over your
    plan limit" without the two numbers generates a support ticket every time.

    *request_limit* is passed in rather than read from subscriptions: the
    resolved context already carries it (ctx['request_limit']), and re-reading
    it here would add a second round trip to the hot path for a value the caller
    is holding. A limit of 0 means no billable requests, not unlimited.

    Reads usage_counters and never aggregates usage_events. That is the entire
    reason the counter exists - the ledger is the largest table in the database
    and an aggregate over it on every request is how a quota check becomes the
    slowest thing in the stack.

    FAILS OPEN. Any lookup error returns (True, 0, request_limit) and logs. A
    quota check must never be the reason a paying merchant's storefront goes
    dark: the worst case of failing open is a tenant briefly overshooting an
    allowance we can reconcile from the ledger afterwards, and the worst case of
    failing closed is every bot on the platform going silent the moment MySQL
    blinks. Note that the 0 in the `used` slot on the error path is
    indistinguishable from a genuinely unused month - the log line is the only
    discriminator, which is why it is a WARNING and not a debug.

    Advisory, not a lock. The counter can move between this call and the write
    that follows it. It is the check that produces a good refusal early, not a
    guarantee that the ceiling holds to the request.
    """
    try:
        limit = int(request_limit)
    except (TypeError, ValueError):
        logger.warning(
            "usage: request_limit for subscription %s is not a number (%r) - "
            "allowing the request.",
            subscription_id, request_limit,
        )
        return True, 0, 0

    try:
        used = counter_for(db, subscription_id)["billable_requests"]
    except Exception as exc:
        logger.warning(
            "usage: quota lookup failed for subscription %s - allowing the "
            "request (failing open): %s",
            subscription_id, exc,
        )
        return True, 0, limit

    # Strictly less-than: `used` counts requests already recorded, so a
    # subscription sitting at 9,999 of 10,000 has one left and one at 10,000 has
    # none. Using <= here would hand every plan one free request a month, which
    # nobody would ever notice and which would make the advertised number wrong.
    return used < limit, used, limit


def usage_by_product(db: Session, client_id: str, start: datetime, end: datetime) -> list[dict]:
    """The dashboard pivot: one row per (product, kind) for this client's window.

    Grouped by kind as well as product because indexing spend and shopper
    traffic have completely different shapes and averaging them together hides
    both. A 25,000-item catalogue sync is 25,000 cheap embedding rows in a
    burst; a month of chat is a smaller number of expensive completions. One
    combined "cost per product" number describes neither.

    NO COMBINED total_cost FIELD, on purpose. Cost is split into czargroup_cost
    and client_cost by usage_events.key_owner, and there is deliberately no
    column that adds them, because that sum means nothing: on a development site
    the calls ran on Czargroup's API keys and the cost is our cost of goods, and
    on a production site the merchant supplied their own keys and the identical
    column is what THEY spent. A client with a staging site and a live one has
    both kinds of row and no honest way to total them. Anyone who genuinely
    wants the sum has to write the addition themselves, which is the point.

    *start* is inclusive and *end* is exclusive. Half-open so a caller walking
    month by month cannot double-count the row that lands exactly on a boundary.

    Both are required even though usage_events carries
    idx_usage_events_client_created (client_id, created_at). That index is what
    makes this a seek into one tenant's slice rather than a range scan over
    every tenant's month; the window is what keeps the slice bounded. The
    COUNT(DISTINCT site_id) means every row inside it is read, so an "all time"
    variant of this function would still be a table scan wearing a helpful name.

    THE WINDOW AND THE COLUMN ARE ON DIFFERENT CLOCKS, and this is the caveat
    that makes these numbers wrong rather than merely slow. *start* and *end*
    are normalised to naive UTC just below; usage_events.created_at is written
    by MySQL's DEFAULT CURRENT_TIMESTAMP in the server's session time zone,
    which nothing sets - see current_period(), which carries the same split and
    the one-line fix for it in database.py. On a +05:30 server every month
    boundary attributes five and a half hours of spend to the wrong period, and
    a caller walking month by month double-counts one edge and drops the other.
    The half-open window is built so a boundary row cannot be counted twice, and
    then the boundary itself is measured on a different clock from the data.

    Until that line lands, treat a figure that straddles a month boundary as
    approximate and do not put one in front of a merchant disputing an invoice -
    which is the reason the ledger exists, so it is worth fixing rather than
    working around. The AUTHORISATION gates are not affected: licences.expires_at
    and subscriptions.expires_at are both written and read from Python, so they
    round-trip consistently whatever the session time zone is.
    """
    start = _as_naive_utc(start, "start")
    end = _as_naive_utc(end, "end")
    if start >= end:
        # Returning [] would render as "no usage in this period", which is a
        # different and much more damaging statement than "your window is
        # inverted".
        raise ValueError(f"start ({start}) must be before end ({end}).")

    rows = db.execute(text("""
        SELECT
            product_code,
            platform,
            kind,
            COUNT(*)                                                     AS events,
            SUM(CASE WHEN billable THEN 1 ELSE 0 END)                    AS billable_requests,
            SUM(input_tokens)                                            AS input_tokens,
            SUM(output_tokens)                                           AS output_tokens,
            SUM(total_tokens)                                            AS total_tokens,
            SUM(CASE WHEN key_owner = 'czargroup' THEN total_cost ELSE 0 END) AS czargroup_cost,
            SUM(CASE WHEN key_owner = 'client'    THEN total_cost ELSE 0 END) AS client_cost,
            COUNT(DISTINCT site_id)                                      AS sites
        FROM usage_events
        WHERE created_at >= :start
          AND created_at <  :end
          AND client_id = :client_id
        GROUP BY product_code, platform, kind
        ORDER BY product_code, kind
    """), {"client_id": client_id, "start": start, "end": end}).fetchall()

    # platform is in the GROUP BY only to satisfy ONLY_FULL_GROUP_BY, which is on
    # by default from MySQL 5.7. It is functionally dependent on product_code
    # (products.platform is derived from the product, never supplied
    # independently), so it cannot split a product across two rows.

    out: list[dict] = []
    for row in rows:
        # The display name is resolved from catalog.PRODUCTS rather than joined
        # from the products table, and falls back to the raw code. usage_events
        # carries no foreign keys precisely so a row outlives the things it
        # names; a join would drop the history of a withdrawn product from this
        # report at exactly the moment somebody was trying to work out what it
        # had cost.
        product = catalog.get_product(row.product_code) or {}
        out.append({
            "product_code": row.product_code,
            "product_name": product.get("name", row.product_code),
            "platform": row.platform,
            "kind": row.kind,
            "events": int(row.events or 0),
            "billable_requests": int(row.billable_requests or 0),
            "input_tokens": int(row.input_tokens or 0),
            "output_tokens": int(row.output_tokens or 0),
            "total_tokens": int(row.total_tokens or 0),
            # Display figures; see the note in counter_for() about floats.
            "czargroup_cost": float(row.czargroup_cost or 0),
            "client_cost": float(row.client_cost or 0),
            "sites": int(row.sites or 0),
        })
    return out
