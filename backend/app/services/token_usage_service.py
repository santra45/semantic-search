"""
The v1 usage writer. RETIRED: its table no longer exists.

WHAT HAPPENED
-------------
The v2 billing migration renamed `token_usage_tracking` to
`token_usage_tracking_archive_v1` and replaced it with `usage_events` plus the
`usage_counters` rollup, both owned by backend/app/services/usage_service.py.
Every INSERT this module issued has failed since that migration ran, with
pymysql error 1146 "Table 'semanticsearch.token_usage_tracking' doesn't exist",
and every one of its callers wraps the call in a swallow - two of them in a bare
`except Exception: pass`. So the failure had nowhere to surface and 100% of usage
data was being discarded with no log line, no error and no missing-row alarm.

That is the same failure this module already caused once, at a smaller scale:
`wp_product_qa` was never added to the archived table's query_type ENUM (look at
it - the value is still absent from
token_usage_tracking_archive_v1.query_type today), so every WooCommerce Q&A
answer was rejected at the column and went unbilled for months, hidden by the
identical swallow. A writer whose failures cannot be seen will eventually stop
writing, and nobody will find out from the code.

SO IT NOW FAILS LOUDLY INSTEAD OF QUIETLY
-----------------------------------------
create_usage_record() no longer issues the INSERT. It logs at ERROR, marked
`usage: V1 WRITER RETIRED`, naming the renamed table and the replacement, and
prints the tenant, the call type and the amounts so the spend stays recoverable
from the log file even though no row is written. One grep counts how much
traffic is still routed through the dead writer, which is the number that says
whether the migration of the write sites is finished.

It does NOT raise, and it does NOT touch the session. Both are deliberate:

  * Raising delivers nothing. All eight call sites already swallow, so the
    exception was never reaching anybody - the ERROR line is the only channel
    that works. And at the three sites that construct TokenUsageTracker(db) with
    the REQUEST's session, the old `except: rollback(); raise` actively rolled
    back the router's own uncommitted work to report a metrics failure. An
    accounting write must never be the reason a shopper's request fails or a
    router loses its transaction.
  * Attempting the INSERT first costs a round trip to be told what we already
    know, and it is that failed statement which poisons the caller's session.

WHY IT DOES NOT DELEGATE TO usage_service
-----------------------------------------
Delegating looks kinder - nothing would be lost while the write sites are
migrated one at a time - and it is the wrong call, for three reasons in
increasing order of seriousness.

  1. It would buy almost nothing today. This module's signature carries a bare
     client_id; usage_service.record() needs six tenant fields off a resolved
     v2 context. With no licences issued yet, a delegated call would find no
     context, refuse the row and log NO CONTEXT - the same outcome as failing
     loudly, reached more expensively.
  2. It would hide the un-migrated write sites. A v1 writer that quietly keeps
     working is how a half-finished migration ships: the sites nobody converted
     look exactly like the ones somebody did, and the discrepancy surfaces at
     invoice time.
  3. It would DOUBLE-BILL. usage_service's own docstring forbids a second writer
     of those tables in as many words, because that is precisely v1's bug: two
     endpoints out of fifteen incremented the only quota counter in the system.
     The eight write sites are being converted to call usage_service directly
     right now; any site that ends up doing both - a half-applied edit, a path
     somebody revives later - would write the same interaction twice, and with
     billable=True on chat_answer that double-counts a merchant's quota and
     their invoice. A missing row is recoverable from the log. A duplicated
     billable row is a wrong invoice that looks exactly like real traffic and
     that nobody will ever notice.

THE READS STILL WORK, AGAINST THE ARCHIVE
-----------------------------------------
get_client_usage_stats() and get_usage_summary() were pointed at the same
renamed table and were therefore raising raw MySQL 1146 at the four endpoints in
backend/app/routers/token_usage.py that call them. They now read
token_usage_tracking_archive_v1, which still holds every pre-migration row, so
historical reporting keeps working unchanged. Both return a `source` key naming
that table, because a dashboard silently showing a figure that stops at the
migration date is worse than one that errors: the number looks current and is
not. New usage lives in usage_events - use usage_service.usage_by_product() and
usage_service.counter_for() for anything after the migration.

STILL BROKEN, AND NOT IN THIS FILE
----------------------------------
Fix the class, not the instance: repointing these two queries fixes two of the
twenty-six statements that still read the dead table. The other twenty-four are
raw SQL in five files this module does not own, and every one of them raises
1146 today:

    backend/app/routers/operator.py              9 SELECTs
    backend/app/routers/token_usage.py           5 SELECTs (its own raw SQL,
                                                 separate from the two here)
    backend/app/magento/chatbot/routers/usage.py 4 SELECTs

Both routers are mounted in main.py. The same migration archived `usage_logs`,
whose three writer functions in license_service.py are called UNGUARDED from ten
places - which is why /search and /magento/search return 500 on every request.
None of that is fixed here.
"""

import uuid
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import text

from backend.app.services.database import get_db

logger = logging.getLogger(__name__)

# The table this module used to write. Named as a constant so the two remaining
# queries cannot drift apart from each other, and so `grep _V1_ARCHIVE_TABLE`
# finds every place still reading v1 history in one hop.
_V1_ARCHIVE_TABLE = "token_usage_tracking_archive_v1"

# The long explanation is printed once per process; the per-call line below is
# printed every time. Both are ERROR and both carry the same marker, so one grep
# still counts every lost write, while the paragraph explaining what to do about
# it does not repeat 25,000 times during a catalogue sync. Rate-limiting the
# per-call line instead would hide the one number that matters - how much
# traffic is still on the dead writer.
_RETIREMENT_EXPLAINED = False


def _explain_retirement_once() -> None:
    """Print the what-and-why of the retirement the first time it bites."""
    global _RETIREMENT_EXPLAINED
    if _RETIREMENT_EXPLAINED:
        return
    _RETIREMENT_EXPLAINED = True
    logger.error(
        "usage: V1 WRITER RETIRED - token_usage_service.create_usage_record is "
        "no longer writing anything. Its table `token_usage_tracking` was "
        "renamed to `%s` by the v2 billing migration. Usage now belongs in "
        "usage_events via usage_service.record() (or usage_service.track() from "
        "a shared service, which opens its own session and reads the tenant from "
        "the request context). Every call from here on logs the spend it could "
        "not record; migrate the call site. This paragraph is printed once per "
        "process.",
        _V1_ARCHIVE_TABLE,
    )


def _log_retired_write(
    client_id: str,
    query_type: str,
    llm_provider: str,
    llm_model: str,
    input_tokens: int,
    output_tokens: int = 0,
    input_cost: float = 0.0,
    output_cost: float = 0.0,
    request_id: Optional[str] = None,
) -> str:
    """The single place a lost v1 write is reported. Returns the request_id.

    Module-level, and shared by both public entry points, because this log line
    is now the ONLY record that the spend ever happened. Two copies of it would
    drift, and a drifted copy would mean one of the two entry points quietly
    reporting less than the other - which is the shape of every bug in this
    file's history.

    Takes no session and does not touch the database. Never raises: a caller
    swallowing an exception from here is what hid the problem in the first
    place, so there is nothing left to swallow.
    """
    # Generated before anything else so the log line and the return value agree,
    # and so a caller-supplied id is echoed back unchanged.
    request_id = request_id or str(uuid.uuid4())

    total_tokens = input_tokens + output_tokens
    total_cost = input_cost + output_cost

    _explain_retirement_once()

    # The QUERY_TYPES allowlist used to raise ValueError here, BEFORE anything
    # was attempted. It is folded into the message instead of being a second
    # exit: a raise would preempt the line below and put the call straight back
    # into the invisible bucket it has been in since the migration, which is the
    # whole problem being fixed. It is still worth reporting, because an
    # unregistered type is exactly how wp_product_qa went unbilled - and it
    # tells whoever migrates the site that v2 needs no allowlist at all, where
    # call_type is a free VARCHAR by design.
    unregistered = "" if query_type in QUERY_TYPES.values() else (
        f" (note: '{query_type}' was never registered in QUERY_TYPES either, "
        f"so v1 would have refused it regardless)"
    )

    # Every argument that carries meaning goes in the line. The row is gone;
    # this is the only remaining evidence the spend happened, and it has to be
    # enough to reconstruct a billing entry by hand if it ever matters.
    logger.error(
        "usage: V1 WRITER RETIRED - not recorded: client=%s query_type=%s "
        "provider=%s model=%s tokens in=%s out=%s total=%s cost in=%s out=%s "
        "total=%s request_id=%s%s. Migrate this call site to "
        "usage_service.record()/track(); `%s` is archived and nothing writes "
        "to it any more.",
        client_id, query_type, llm_provider, llm_model,
        input_tokens, output_tokens, total_tokens,
        input_cost, output_cost, total_cost,
        request_id, unregistered, _V1_ARCHIVE_TABLE,
    )
    return request_id

# Query types that can be tracked
QUERY_TYPES = {
    'embed_search': 'embed_search',
    'embed_document': 'embed_document',
    'product_rerank': 'product_rerank',
    'content_rerank': 'content_rerank',
    'chat_answer': 'chat_answer',
    'chat_context': 'chat_context',
    'chat_rewrite': 'chat_rewrite',
    # Intent classification — used by the Magento chatbot's LLMClassifier
    # when the heuristic layer falls below confidence threshold and we
    # need a single small structured-JSON call to pick the right agent.
    'chat_intent': 'chat_intent',
    # Tool-calling intent router — Phase 3.1. Single LLM call where every
    # agent is registered as a tool; LLM picks the appropriate tool +
    # extracts args. Separate query type from `chat_intent` so admin
    # dashboard can A/B costs between the legacy classifier and the new
    # tool-call router during the shadow / live rollout window.
    'chat_tool_call': 'chat_tool_call',
    # Query decomposer — Phase 3.3. Small JSON-mode call that splits a
    # compositional customer query into 1-3 semantic sub-queries which
    # the retrieval pipeline then embeds + fuses via Qdrant RRF.
    # Separate query type so admin dashboard can see decomposer cost
    # broken out from classifier and answer-generation buckets.
    'chat_query_decompose': 'chat_query_decompose',
    # Single-shot grounded answer for the WooCommerce product Q&A widget.
    # Kept apart from 'chat_answer' because the two have completely different
    # cost shapes — one product's payload versus a multi-turn conversation
    # with retrieved context — and averaging them together hides both.
    'wp_product_qa': 'wp_product_qa',
}

class TokenUsageTracker:
    """Service for tracking token usage and costs per client request."""
    
    def __init__(self, db: Optional[Session] = None):
        self.db = db or next(get_db())
    
    def create_usage_record(
        self,
        client_id: str,
        query_type: str,
        llm_provider: str,
        llm_model: str,
        input_tokens: int,
        output_tokens: int = 0,
        input_cost: float = 0.0,
        output_cost: float = 0.0,
        request_text_length: int = 0,
        response_text_length: int = 0,
        request_id: Optional[str] = None
    ) -> str:
        """RETIRED. Writes nothing, logs the spend at ERROR, never raises.

        The signature is unchanged so that no call site breaks while the eight
        write sites are migrated to usage_service. What changed is that this no
        longer pretends to record anything: the target table was renamed by the
        v2 migration, so the INSERT could only fail, and every caller swallows
        the exception. See the module docstring for the full reasoning,
        including why this does not delegate to usage_service instead.

        Issues NO statement and touches NO session, deliberately. The old code
        caught its own failure, called self.db.rollback() and re-raised - and at
        the three call sites that construct TokenUsageTracker(db) with the
        request's own session, that rollback threw away the router's uncommitted
        work in order to report a metrics failure nobody could see.

        Returns the request_id it would have used, so a caller that correlates
        on the return value keeps working. That id now appears only in the log
        line this delegates to.
        """
        return _log_retired_write(
            client_id=client_id,
            query_type=query_type,
            llm_provider=llm_provider,
            llm_model=llm_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            request_id=request_id,
        )

    def get_client_usage_stats(
        self, 
        client_id: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """Historical usage for one client, read from the v1 ARCHIVE.

        Reads token_usage_tracking_archive_v1, so every pre-migration row is
        still reportable and the endpoints in backend/app/routers/token_usage.py
        that call this stop raising a raw MySQL 1146.

        THE NUMBERS STOP AT THE MIGRATION. Nothing has written to this table
        since, and nothing will again. Anything after that date is in
        usage_events - usage_service.usage_by_product() for the breakdown,
        usage_service.counter_for() for the month's rollup. The returned dict
        carries a `source` key naming the archive for exactly this reason: a
        dashboard that silently renders a total which ends a month ago is worse
        than one that errors, because the figure looks current and is not.

        Args:
            client_id: Client identifier
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Dictionary containing usage statistics for the archived period
        """

        where_clause = "WHERE client_id = :client_id"
        params = {'client_id': client_id}
        
        if start_date:
            where_clause += " AND created_at >= :start_date"
            params['start_date'] = start_date
            
        if end_date:
            where_clause += " AND created_at <= :end_date"
            params['end_date'] = end_date
        
        sql = f"""
        SELECT 
            query_type,
            llm_provider,
            llm_model,
            COUNT(*) as request_count,
            SUM(input_tokens) as total_input_tokens,
            SUM(output_tokens) as total_output_tokens,
            SUM(total_tokens) as total_tokens,
            SUM(input_cost) as total_input_cost,
            SUM(output_cost) as total_output_cost,
            SUM(total_cost) as total_cost,
            AVG(total_cost) as avg_cost_per_request,
            MIN(created_at) as first_request,
            MAX(created_at) as last_request
        FROM {_V1_ARCHIVE_TABLE}
        {where_clause}
        GROUP BY query_type, llm_provider, llm_model
        ORDER BY total_cost DESC
        """
        
        try:
            result = self.db.execute(text(sql), params)
            rows = result.fetchall()
            
            stats = {
                'client_id': client_id,
                'period': {
                    'start_date': start_date,
                    'end_date': end_date
                },
                # Names the table these figures came from. An added key is safe
                # for every caller (all four are FastAPI endpoints returning
                # this as JSON), and without it a dashboard renders a total that
                # silently stopped growing at the migration as though it were
                # this month's spend.
                'source': _V1_ARCHIVE_TABLE,
                'usage_by_type': []
            }
            
            for row in rows:
                stats['usage_by_type'].append({
                    'query_type': row.query_type,
                    'llm_provider': row.llm_provider,
                    'llm_model': row.llm_model,
                    'request_count': row.request_count,
                    'total_input_tokens': row.total_input_tokens,
                    'total_output_tokens': row.total_output_tokens,
                    'total_tokens': row.total_tokens,
                    'total_input_cost': float(row.total_input_cost),
                    'total_output_cost': float(row.total_output_cost),
                    'total_cost': float(row.total_cost),
                    'avg_cost_per_request': float(row.avg_cost_per_request),
                    'first_request': row.first_request,
                    'last_request': row.last_request
                })
            
            # Calculate totals across all types. Input and output are kept
            # apart as well as summed: they price differently on every provider
            # (output is 3-10x input), so a single token count can't be turned
            # back into a cost, and any surface showing "tokens used" wants the
            # split.
            total_stats = {
                'total_requests': sum(row['request_count'] for row in stats['usage_by_type']),
                'total_input_tokens': sum(row['total_input_tokens'] or 0 for row in stats['usage_by_type']),
                'total_output_tokens': sum(row['total_output_tokens'] or 0 for row in stats['usage_by_type']),
                'total_tokens': sum(row['total_tokens'] for row in stats['usage_by_type']),
                'total_cost': sum(row['total_cost'] for row in stats['usage_by_type'])
            }
            stats['totals'] = total_stats
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get client usage stats: {e}")
            raise
    
    def get_usage_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """Cross-client usage summary, read from the v1 ARCHIVE.

        Same caveat as get_client_usage_stats(): this covers only the period
        before the v2 migration, because nothing has written to the archived
        table since. The returned dict names its source for that reason.

        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Dictionary containing the archived period's usage summary
        """

        where_clause = "WHERE 1=1"
        params = {}
        
        if start_date:
            where_clause += " AND created_at >= :start_date"
            params['start_date'] = start_date
            
        if end_date:
            where_clause += " AND created_at <= :end_date"
            params['end_date'] = end_date
        
        sql = f"""
        SELECT 
            COUNT(DISTINCT client_id) as unique_clients,
            COUNT(*) as total_requests,
            SUM(input_tokens) as total_input_tokens,
            SUM(output_tokens) as total_output_tokens,
            SUM(total_tokens) as total_tokens,
            SUM(input_cost) as total_input_cost,
            SUM(output_cost) as total_output_cost,
            SUM(total_cost) as total_cost,
            AVG(total_cost) as avg_cost_per_request
        FROM {_V1_ARCHIVE_TABLE}
        {where_clause}
        """
        
        try:
            result = self.db.execute(text(sql), params)
            row = result.fetchone()
            
            return {
                'period': {
                    'start_date': start_date,
                    'end_date': end_date
                },
                # See get_client_usage_stats() - the figures below end at the
                # migration, and a caller has to be able to tell.
                'source': _V1_ARCHIVE_TABLE,
                'unique_clients': row.unique_clients,
                'total_requests': row.total_requests,
                'total_input_tokens': row.total_input_tokens or 0,
                'total_output_tokens': row.total_output_tokens or 0,
                'total_tokens': row.total_tokens or 0,
                'total_input_cost': float(row.total_input_cost or 0),
                'total_output_cost': float(row.total_output_cost or 0),
                'total_cost': float(row.total_cost or 0),
                'avg_cost_per_request': float(row.avg_cost_per_request or 0)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get usage summary: {e}")
            raise

# Convenience function for easy tracking
def track_usage(
    client_id: str,
    query_type: str,
    llm_provider: str,
    llm_model: str,
    input_tokens: int,
    output_tokens: int = 0,
    input_cost: float = 0.0,
    output_cost: float = 0.0,
    request_text_length: int = 0,
    response_text_length: int = 0,
    request_id: Optional[str] = None
) -> str:
    """RETIRED, exactly like create_usage_record(). Logs the spend, writes nothing.

    The three shared services that call this - embedder, llm_completion_service,
    llm_rerank_service - hold no Session and no request context, which is why this convenience function existed. Its replacement is
    usage_service.track(), which has the same "open a short-lived session, close
    it in a finally" shape and gets the tenant from the request context instead
    of from a bare client_id.

    NO LONGER OPENS A SESSION. It used to construct TokenUsageTracker(), which
    calls next(get_db()) - a connection checkout per call, on the hottest write
    path in the system, for a writer that no longer touches the database. That
    checkout was pure cost even before the migration made the write fail.
    """
    # Straight to the shared reporter, bypassing TokenUsageTracker entirely so
    # no instance and therefore no session is created. request_text_length and
    # response_text_length are accepted and dropped: they only ever fed two
    # columns of the archived table, usage_events has no equivalent, and keeping
    # them in the signature is what lets the four call sites migrate one at a
    # time instead of all at once.
    return _log_retired_write(
        client_id=client_id,
        query_type=query_type,
        llm_provider=llm_provider,
        llm_model=llm_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
        request_id=request_id,
    )
