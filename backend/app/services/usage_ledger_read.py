"""The READ side of the usage ledger, across the v1 -> v2 dual-read window.

WHY THIS MODULE EXISTS
----------------------
The v2 billing migration renamed `token_usage_tracking` to
`token_usage_tracking_archive_v1` and replaced it with `usage_events`. Every
reader that still names the old table raises pymysql 1146
"Table 'semanticsearch.token_usage_tracking' doesn't exist". Measured against
the running stack, that was SEVEN live HTTP 500s, not one:

    GET /api/magento/chatbot/usage/stats   (magento/chatbot/routers/usage.py)
    GET /api/magento/chatbot/usage         (services/chat_analytics_service.py)
    GET /api/token-usage/clients           (routers/token_usage.py)
    GET /api/token-usage/models            (routers/token_usage.py)
    GET /api/token-usage/hourly            (routers/token_usage.py)
    GET /api/token-usage/me/models         (routers/token_usage.py)
    GET /api/token-usage/me/hourly         (routers/token_usage.py)

plus nine SELECTs in routers/operator.py, which are individually fail-soft and
so render zeros instead of erroring - the worse failure of the two.

Twenty-one statements in four files read that table. They are NOT four
independent bugs and they must not get four independent fixes: if the union
below is spelled out at each site, the day somebody adds a column or changes a
mapping is the day two admin dashboards start disagreeing about a merchant's
bill and nobody can say which one is lying. One derived table, defined once,
imported by all four.

WHY THE UNION, RATHER THAN A REPOINT
------------------------------------
There were three options per reader and only one of them survives contact with
the current state of the database:

  * Point at usage_events (the live v2 ledger). usage_events is EMPTY and stays
    empty until v2 licences are issued - `licences` has 0 rows, so no request
    ever resolves a v2 context and no row can be written. A naive repoint turns
    an HTTP 500 into a dashboard of confident zeros, which is strictly worse: a
    500 gets fixed and a zero gets believed.
  * Point at token_usage_tracking_archive_v1 (2,794 rows, 2026-04-03 to
    2026-08-03, 3 tenants). Honest about the past, permanently blind to the
    future - every one of these endpoints would need editing a second time the
    day the first licence is issued, and until somebody did, a merchant's
    "last 24 hours" panel would render empty forever.
  * Read BOTH, as one ledger, and say which half each figure came from. That is
    what this module does. The archive half is frozen, so its contribution never
    changes; the live half starts contributing the moment usage_service.record()
    lands its first row, with no further deploy.

None of the readers here is a candidate for deletion: every one of them is
mounted, reachable, and rendered by a real admin screen - five of them by
shipped PHP/JS in Czargroup/AIChatbot, Czargroup/AIProductQA, Czargroup/AISearch,
semantic-search-woo and ai-product-qa-woo. Deleting an endpoint five clients
call is not a fix for a renamed table.

WHAT THE NUMBERS ACTUALLY MEAN TODAY
------------------------------------
Read this before quoting any figure these endpoints return.

The v1 JWT keys that carry 100% of current traffic do NOT produce usage rows.
usage_service.record() refuses a row it cannot attribute to a v2 context, and
with 0 licences there is no v2 context, so today's traffic is recorded NOWHERE -
not in the archive (frozen at the migration) and not in usage_events (needs a
licence). So during the dual-read window every figure below is pre-migration
history with a hole after 2026-08-03, and a zero means "not measured", not
"no spend". That is precisely why provenance() exists and why every response in
these four files now carries it: the one thing a caller must be able to tell is
whether a number is small or absent.

THE COST COLUMN CANNOT BE SPLIT BY OWNER ACROSS THIS UNION
----------------------------------------------------------
usage_events.key_owner separates spend on Czargroup's API keys (our cost of
goods) from spend on the merchant's own keys (their bill), and
usage_service.usage_by_product() refuses to emit a column that adds them,
because that sum describes nothing. v1 had no such column: all 2,794 archive
rows are an unlabelled mix and cannot be retro-split. The union therefore stamps
the archive half `v1_unknown` rather than guessing 'czargroup' or 'client' -
a third value a reader has to notice, instead of a wrong answer they will not.

Nothing in this change adds an owner split to the four readers' JSON, because
their combined `total_cost` key is already rendered by five shipped frontends
and quietly changing what it means is how a merchant gets an invoice they
cannot reconcile. Anyone adding one must handle `v1_unknown` explicitly.

READ-ONLY, DELIBERATELY
-----------------------
Nothing here writes. usage_service.py is the only writer of usage_events and
usage_counters, and its docstring forbids a second one in as many words - two
writers of one counter is the exact v1 bug this migration exists to end. If you
are here to record usage, you are in the wrong module.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# The two halves of the ledger, named so `grep V1_ARCHIVE_TABLE` finds every
# place still reading v1 history in one hop. token_usage_service.py holds its
# own copy of the archive name for the two queries it owns; these must stay in
# step, which is an argument for that module importing this one rather than for
# a third spelling appearing somewhere.
V1_ARCHIVE_TABLE = "token_usage_tracking_archive_v1"
V2_LIVE_TABLE = "usage_events"

# The value stamped on archive rows in place of usage_events.key_owner. NOT
# 'czargroup' and NOT 'client': see the module docstring. Anything grouping by
# key_owner has to handle this third value.
V1_KEY_OWNER = "v1_unknown"

# Chat-attributable call types, for the readers that report chat spend alone.
#
# The list this replaces was ('chat_answer', 'chat_context', 'chat_rewrite'),
# and it was WRONG against its own data: chat_context and chat_rewrite have
# never been written (0 rows in the archive), while chat_intent (6 rows) and
# chat_tool_call (6 rows) are real, billed chat spend that the filter dropped on
# the floor. Measured on the archive, the old list under-reported the chat cost
# of the three tenants that have any by 0.00359062 of 0.01236495 - 29.04% of it,
# missing, on a merchant-facing cost panel. Both names are kept even though they
# are empty today: they are still legal values of the archive's query_type ENUM,
# so a row could exist on a deployment other than this one.
CHAT_CALL_TYPES: tuple[str, ...] = (
    "chat_answer",
    "chat_context",
    "chat_rewrite",
    "chat_intent",
    "chat_tool_call",
)

# ── The derived table ────────────────────────────────────────────────────────
#
# Use it as a table: f"SELECT ... FROM {LEDGER} u WHERE u.client_id = :cid".
# It is a constant, never built from caller input, so an f-string here is not a
# SQL injection surface - every value a caller supplies still goes through bound
# parameters at the call site.
#
# The v1 column NAMES win, on both halves. Not because they are better -
# `call_type` is a better name than `query_type` - but because five shipped
# frontends and four routers already speak them, and renaming a JSON key is a
# silent break in a PHP admin panel nobody in this repo can redeploy. The
# mapping is: call_type -> query_type, provider -> llm_provider,
# model -> llm_model.
#
# TWO THINGS THAT LOOK LIKE NOISE AND ARE NOT:
#
#   COLLATE on every string column. The two tables were created under different
#   collations - the archive is utf8mb4_0900_ai_ci (it predates the migration),
#   usage_events is utf8mb4_general_ci - and MySQL answers a UNION across them
#   with ERROR 1271 "Illegal mix of collations", not with a warning. Reproduced
#   before this was written. The _utf8mb4 introducers on the literals are part
#   of the same fix: an unqualified literal takes character_set_connection, so
#   without it the same statement works under the app's utf8mb4 connection and
#   fails under a client connected as latin1 (ERROR 1253). Both were observed.
#
#   CAST(query_type AS CHAR). The archive's query_type is an ENUM. COLLATE
#   happens to force it to a string too, but that is a side effect of the
#   collation rewrite rather than a documented guarantee, and a UNION branch
#   that resolved an ENUM to its integer ordinal would turn every query_type in
#   every dashboard into a number without erroring anywhere.
#
# usage_events.billable and usage_events.product_code are deliberately NOT
# projected. v1 has no equivalent for either, and the honest v1 value is
# "unknowable" - which SUM(CASE WHEN billable ...) would silently render as
# "zero billable requests before the migration". No reader here needs them; add
# them only alongside a decision about what the archive half should say.
LEDGER = f"""(
    SELECT
        _utf8mb4'v1_archive' COLLATE utf8mb4_general_ci        AS ledger,
        client_id            COLLATE utf8mb4_general_ci        AS client_id,
        CAST(query_type AS CHAR) COLLATE utf8mb4_general_ci    AS query_type,
        llm_provider         COLLATE utf8mb4_general_ci        AS llm_provider,
        llm_model            COLLATE utf8mb4_general_ci        AS llm_model,
        _utf8mb4'{V1_KEY_OWNER}' COLLATE utf8mb4_general_ci    AS key_owner,
        input_tokens, output_tokens, total_tokens,
        input_cost,   output_cost,   total_cost,
        created_at
    FROM {V1_ARCHIVE_TABLE}
    UNION ALL
    SELECT
        _utf8mb4'v2_live' COLLATE utf8mb4_general_ci           AS ledger,
        client_id, call_type, provider, model, key_owner,
        input_tokens, output_tokens, total_tokens,
        input_cost,   output_cost,   total_cost,
        created_at
    FROM {V2_LIVE_TABLE}
)"""


# ── Provenance ───────────────────────────────────────────────────────────────

# The paragraph explaining the dual-read window is printed once per process; the
# machine-readable block below rides on every response. Same split as
# token_usage_service._explain_retirement_once(), for the same reason: the
# explanation is worth reading once and worthless repeated 25,000 times during a
# catalogue sync, while the per-response field is what a caller can actually act
# on.
_WINDOW_EXPLAINED = False


def _explain_window_once(v1_last: Any, v2_last: Any) -> None:
    """Say once, loudly, that the live ledger has not started."""
    global _WINDOW_EXPLAINED
    if _WINDOW_EXPLAINED:
        return
    _WINDOW_EXPLAINED = True
    logger.warning(
        "usage: DUAL-READ WINDOW - %s is empty, so every usage figure served by "
        "the reporting endpoints is pre-migration history from %s and stops at "
        "%s. Current traffic is recorded in NEITHER table: v1 JWT keys resolve "
        "no v2 context, and usage_service.record() refuses a row it cannot "
        "attribute, so a zero from these endpoints means NOT MEASURED and not "
        "NO SPEND. It resolves itself the moment the first v2 licence is issued "
        "and the first usage_events row lands. This paragraph is printed once "
        "per process; the per-response `usage_source` block carries the same "
        "facts to callers.",
        V2_LIVE_TABLE, V1_ARCHIVE_TABLE, v1_last,
    )


def _iso(value: Any) -> Optional[str]:
    return value.isoformat() if hasattr(value, "isoformat") else (None if value is None else str(value))


def provenance(db: Session) -> dict:
    """Where the figures in this response came from, and whether they are current.

    Attached to every reporting response in the four files that read the ledger.
    It is the difference between a dashboard showing a small number and a
    dashboard showing an absent one, which during this migration window is the
    only question worth asking about any of these figures.

    Costs two O(1) index lookups - MAX(created_at) against an index on each half
    - and is deliberately NOT cached. A cached "the live ledger is empty" would
    go stale at exactly the moment it stops being true, which is the moment the
    whole block exists to report.

    FAIL-SOFT, and says so rather than lying. If the lookup itself fails the
    block comes back with `error` set and the timestamps None. A caller must be
    able to tell "no v2 rows yet" from "I could not find out", because the first
    is the expected state and the second means the numbers beside it are
    untrustworthy too.
    """
    try:
        row = db.execute(text(f"""
            SELECT (SELECT MAX(created_at) FROM {V1_ARCHIVE_TABLE}) AS v1_last,
                   (SELECT MAX(created_at) FROM {V2_LIVE_TABLE})    AS v2_last
        """)).fetchone()
        v1_last = row.v1_last if row else None
        v2_last = row.v2_last if row else None
    except Exception as exc:
        # Do not rollback here: the caller owns the session and may have work in
        # flight. A failed SELECT does not poison a session the way a failed
        # INSERT does, and a reporting helper rolling back a router's
        # transaction is the bug token_usage_service was fixed for.
        logger.warning("usage: provenance lookup failed: %s", exc)
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "v1_archive": {"table": V1_ARCHIVE_TABLE, "last_event_at": None},
            "v2_live": {"table": V2_LIVE_TABLE, "has_rows": None, "last_event_at": None},
            "current": None,
            "note": (
                "Could not determine which ledgers these figures came from. "
                "Treat every number in this response as unverified."
            ),
        }

    # MAX(created_at) IS NULL is the emptiness test: it is an index lookup,
    # where COUNT(*) on usage_events is a scan of what becomes the largest table
    # in the database.
    v2_has_rows = v2_last is not None
    if not v2_has_rows:
        _explain_window_once(_iso(v1_last), _iso(v2_last))

    return {
        "v1_archive": {
            "table": V1_ARCHIVE_TABLE,
            "frozen": True,
            "last_event_at": _iso(v1_last),
        },
        "v2_live": {
            "table": V2_LIVE_TABLE,
            "has_rows": v2_has_rows,
            "last_event_at": _iso(v2_last),
        },
        # The single field a dashboard should branch on. False means every
        # figure beside it stopped at v1_archive.last_event_at.
        "current": v2_has_rows,
        "note": (
            "Figures combine the frozen v1 archive with the live v2 ledger."
            if v2_has_rows else
            "NOT CURRENT. The v2 ledger is empty (no licences issued yet), so "
            "these figures are pre-migration history that stops at "
            f"v1_archive.last_event_at. Traffic served since then is recorded in "
            "neither table - a zero here means NOT MEASURED, not NO SPEND."
        ),
    }
