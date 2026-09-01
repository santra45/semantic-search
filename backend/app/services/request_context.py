"""The tenant context of the request being served, reachable without plumbing.

NOT TO BE CONFUSED WITH backend/app/magento/chatbot/agents/request_context.py,
which already exists and exports a RequestContext dataclass holding the Magento
admin client, the store code and the shopper's quote id. That one is a value
object a handler builds and passes down by hand. This one is the ambient tenant
identity a shared service reads where it stands. Both names were already in
flight when this module was specified; if you are renaming things, rename that
one to agent_context.py rather than this one, because this is the module the
usage writers import.

WHY THIS EXISTS
---------------
Four of the eight places that write a usage row are shared services - embedder,
llm_completion_service, llm_rerank_service, chat_response_service - and every
one of them receives a bare client_id and nothing else. usage_service.record()
needs six identifiers off the resolved licence (client_id, site_id,
subscription_id, product_code, platform, key_owner) plus the interaction_id
that threads one turn together. embed_query() and embed_document() alone have
25 call sites, so threading a context argument down to them means editing every
intermediate signature in between - a large diff whose failure mode is one
caller left passing the old shape, silently writing an unattributable row.

So the context travels out of band. The auth chokepoint binds it once per
request; the writers read it where they stand.

THE BRIEF'S SAFETY STORY IS BACKWARDS, AND THAT CHANGES HOW YOU READ THIS FILE
------------------------------------------------------------------------------
The specification for this module said its purpose was to stop one tenant's
identity bleeding into the next request on a reused worker thread. Three
separate measurements against the running container (real uvicorn, the pinned
starlette/anyio, sequential requests deliberately landing on the same AnyIO
worker thread, with no reset anywhere) all read the variable as unset on entry
every single time.

That is not luck. Every request is served in its own asyncio Task, and Tasks
copy the context at creation; every sync `def` endpoint is dispatched through
anyio.to_thread.run_sync, which does copy_context() per dispatch and runs the
callable inside that copy. A value set inside either is discarded when the
request ends. Cross-request leakage on this stack is structurally impossible,
and it stays impossible for as long as the app is ASGI-on-asyncio.

The failure this module actually has is the opposite one: the value being
silently ABSENT where a writer expects it, producing a dropped or
unattributable billing row rather than a loud error. Everything below is shaped
around that:

  * bind_context() hands back no token, so there is nothing to forget to reset.
  * request_context() is the only way to take a scope you own, and it cannot be
    entered without the paired reset.
  * streaming_response() is how you return a StreamingResponse from this
    codebase, because a SYNC streaming body is the one place a bound context
    provably does NOT reach and the naive fix passes a smoke test before
    failing where it counts. (An async body on an `async def` endpoint keeps
    the binding - measured - and needs none of this; pinned_stream() says so
    and refuses one.) See its docstring, and pinned_stream() below it.
  * get_context() logs at ERROR when it is read as absent from inside a
    streaming body, which is the shape that used to lose a whole turn's
    billable row with no error anywhere. See _in_streaming_body().
  * bind_context() logs at ERROR if it finds a different tenant already bound,
    which is the only way the brief's leakage story could ever come true. If
    that line ever fires, the premises above have changed and this comment is
    the thing to re-verify.

NO APPLICATION IMPORTS, EVER
----------------------------
This module must not import anything from backend.app. usage_service.record()
imports it to read the ambient context; usage_service already imports
licensing_service. The moment this file imports usage_service back - for
new_interaction_id(), which is the tempting one - that is a circular import and
the app stops booting. Minting stays with the caller.
interaction_id_from_header() below only sanitises.

That rule used to be written here as "stdlib only", which is a stricter line
than the reason behind it justifies and which streaming_response() now crosses:
it imports starlette's StreamingResponse. starlette is a leaf dependency that
cannot import this application, so it cannot participate in the cycle the rule
exists to prevent, and the alternative - a helper that constructs the response
without naming its class - is worse code for a rule that never meant that.
Third-party leaf imports are fine. backend.app imports are not, at module scope
or inside a function.
"""

from __future__ import annotations

import contextvars
import logging
import re
import sys
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Mapping, Optional

from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

logger = logging.getLogger(__name__)


# -- The variable ------------------------------------------------------------

# Private, and it stays private. Exporting the ContextVar itself would let a
# call site do _CTX.set() without a reset, which is the one thing the API shape
# below is arranged to prevent.
#
# default=None rather than no default: a writer running outside any request -
# a management script, a test that forgot a fixture - should read "nothing is
# bound" and take its own branch, not raise LookupError from inside a billing
# call and lose the row to an exception handler that was written for database
# errors.
_CTX: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "aichatbot_request_tenant_context", default=None
)


# -- Reading -----------------------------------------------------------------

# How far up the stack _in_streaming_body() will look before giving up. The
# frame it wants is a handful above the writer - get_context, _implicit_ctx,
# record, the shared service, the generator body, _next - but an active
# retrieval tool loop puts a dozen langchain frames in between, so this is
# generous. It only ever runs when the context is already missing, i.e. on a
# path that is about to refuse a billing row anyway.
_STACK_SCAN_LIMIT = 60


def _in_streaming_body() -> bool:
    """True when the caller is running inside a StreamingResponse body pull.

    THE ONLY THING THAT SEPARATES A LOST BILLING ROW FROM AN EXPECTED ONE.

    A missing context is routine in this codebase and will stay routine for the
    whole dual-read window: a v1 JWT request has no v2 identity to bind, and
    several entry points (the WooCommerce webhooks, the AI Search family) have
    no chokepoint at all. Logging every absent context at ERROR would bury the
    one case that is genuinely a defect.

    That case is a streaming body that was not pinned. starlette pulls a sync
    streaming body with iterate_in_threadpool(), which dispatches
    `_next(as_iterator)` onto an AnyIO worker thread through
    anyio.to_thread.run_sync - and that dispatch takes a FRESH copy_context(),
    so nothing the chokepoint bound is visible inside. Because `_next` is the
    callable handed to the thread, its frame sits directly beneath the
    generator's on that thread's stack, and walking f_back from a writer finds
    it. Nothing else in starlette/concurrency.py reaches a worker thread's
    stack: run_in_threadpool() awaits on the event-loop thread and its frame is
    on a different stack entirely, so an ordinary sync `def` endpoint does not
    match here (verified against the running container - a plain sync handler
    reading an absent context stays silent).

    A pinned body cannot reach this function, because pinned_stream() refuses an
    empty ctx, so inside a pinned stream the context is never None. Absent
    context plus this frame therefore means exactly one thing: someone returned
    StreamingResponse(gen(), ...) instead of streaming_response(gen(), ctx).

    Only the SYNC body shape is detected, and that is the whole population worth
    detecting: an async body never goes near iterate_in_threadpool, is iterated
    on the event loop inside the request's own Task, and keeps the chokepoint's
    binding (measured - see pinned_stream). There is no silent loss to catch.

    Matched on the module path rather than only the function name so that
    renaming `_next` does not silently retire the guard; the name is kept as a
    second, independent condition in case the file moves instead. If starlette
    ever does both, this goes quiet and streaming_response() is the defence
    that remains - which is why that helper exists rather than this check alone.
    """
    frame = sys._getframe()
    depth = 0
    while frame is not None and depth < _STACK_SCAN_LIMIT:
        code = frame.f_code
        path = code.co_filename.replace("\\", "/")
        if "starlette" in path and (
            path.endswith("/concurrency.py") or code.co_name == "_next"
        ):
            return True
        frame = frame.f_back
        depth += 1
    return False


def get_context() -> Optional[dict]:
    """The tenant context bound for this request, or None.

    None means one of three things and the caller has to treat them the same
    way: no chokepoint ran (a router that still authenticates inline), the
    caller is not on a request at all, or the caller is inside a streaming
    response body that was not built with streaming_response().

    Only the third of those is a defect, and it is the one that used to be
    invisible, so it is the one that gets a log line here - see
    _in_streaming_body() for why the other two stay quiet. Logged here rather
    than in usage_service because this is the module that knows what a pinned
    stream is, and because a streaming body that reads an absent context has
    already lost the row by the time any writer sees the None.

    The dict is the SAME OBJECT the chokepoint returned to the handler, not a
    copy. That is on purpose: what a shared service reads here is exactly what
    it would have been handed if the value had been threaded down by hand, so
    there is one shape to reason about rather than two. It also means it
    carries the presented licence key in plain text under "license_key",
    because that key is the KEK every merchant-supplied LLM key is encrypted
    under and roughly twenty call sites need it. Never log this dict whole.
    """
    ctx = _CTX.get()

    # THE GUARD FOR THE FAILURE THAT HAS NO OTHER SYMPTOM.
    #
    # Everything else that goes wrong with this module is at least visible: a
    # refused usage row logs, a rebind logs, a bad ctx shape raises. A streaming
    # endpoint written as StreamingResponse(gen(), ...) logs nothing anywhere -
    # the answer streams perfectly, the shopper is happy, and the turn's
    # billable row silently does not exist. Measured in the container: every
    # chunk and the terminal write read None.
    #
    # The near-miss variant is the reason this is at ERROR rather than WARNING.
    # Binding the ContextVar at the top of the generator gives the right tenant
    # in chunk 1 and None from chunk 2 onwards, because the generator FRAME
    # survives across yields and the Context does not. A reviewer who checks the
    # first token concludes it works and ships it. This line fires from chunk 2
    # on, which is the only moment that shape is distinguishable from a correct
    # one.
    #
    # Deliberately not rate-limited. One line per lost row is the correct
    # volume: each is a real billing row that will not exist, and a stream that
    # fires this fifty times is fifty rows gone.
    if ctx is None and _in_streaming_body():
        logger.error(
            "request context: ABSENT INSIDE A STREAMING BODY - this turn's "
            "usage rows are being lost. The endpoint returned a bare "
            "StreamingResponse, so the body is pulled through "
            "iterate_in_threadpool() in a fresh context copy and nothing the "
            "auth chokepoint bound can reach it. Fix the endpoint, not the "
            "writer: return request_context.streaming_response(gen(), "
            "license_data, media_type=...) instead of StreamingResponse(gen(), "
            "...). See request_context.streaming_response()."
        )

    return ctx


def current_interaction_id() -> Optional[str]:
    """The id threading every row of the current turn, or None.

    Convenience over get_context() because a writer that only needs the id
    should not have to know the context is a dict, nor guess the key name.

    Reads _CTX directly rather than going through get_context(), so it does NOT
    trip the unpinned-stream guard. That is deliberate: a missing interaction_id
    costs a grouping, not a row - the row is still written, just harder to thread
    back to its turn - and the guard's whole value is that it only ever fires on
    an actual lost row. Anything that would lose a row goes through
    get_context().
    """
    ctx = _CTX.get()
    return ctx.get("interaction_id") if isinstance(ctx, dict) else None


# -- Binding -----------------------------------------------------------------

def bind_context(ctx: dict) -> None:
    """Bind *ctx* as the tenant context for the rest of this request.

    For the auth chokepoints and nothing else. It deliberately returns no
    token, because there is no reset for the caller to own: authorize_request()
    RETURNS the dict rather than wrapping the handler body, so it could not
    hold a scope open even if it wanted to, and the runtime ends the binding
    anyway when the request's context copy is discarded (see the module
    docstring). An API that handed back a token here would be handing back a
    token nobody can honour.

    If you want a scope with a beginning and an end - a nested block, a loop
    over tenants, a test - use request_context() instead. That one cannot be
    entered without its reset.

    Raises on a non-dict, which is a programming error that will fire on the
    first request in any environment rather than intermittently in production.
    """
    if not isinstance(ctx, dict):
        raise TypeError(
            f"request context must be the dict the chokepoint built, got "
            f"{type(ctx).__name__}."
        )

    # THE GUARD THAT WOULD CATCH THE BUG THE BRIEF WAS WORRIED ABOUT.
    #
    # Binding twice in one request is legitimate exactly once - an endpoint
    # that authorises a second key - and is otherwise the signature of a
    # context that outlived the request that bound it. On this stack that
    # cannot happen (per-Task and per-dispatch context copies, measured), so
    # this line should never fire. If it does, something structural has changed
    # - a ThreadPoolExecutor, a non-asyncio server, a handler reusing a context
    # object - and the next thing to happen is one tenant's spend being booked
    # against another. Loud, with both ids, so the report names the pair.
    previous = _CTX.get()
    if isinstance(previous, dict):
        before = previous.get("client_id")
        after = ctx.get("client_id")
        if before and after and before != after:
            logger.error(
                "request context: rebinding from client %s to client %s inside "
                "one execution context. Either an endpoint authorised two keys "
                "(fine, ignore this) or a context survived the request that "
                "bound it (not fine - usage rows are about to be attributed to "
                "the wrong tenant). See the module docstring.",
                before, after,
            )

    _CTX.set(ctx)


@contextmanager
def request_context(ctx: dict) -> Iterator[dict]:
    """Bind *ctx* for the duration of the block, and restore whatever was there.

    The only scoped setter, and the reason there is no public set/reset pair:
    the two cannot be separated, so a caller cannot write the set and leave the
    reset for a later commit that never lands.

    Restores rather than clears, so nesting behaves - an inner block for a
    second tenant hands the outer one back its own context on exit instead of
    leaving the rest of the request unattributed, which is the failure mode a
    bare set-then-clear pair produces and which shows up as a NULL client_id on
    a usage row written after the inner block, a long way from the cause.
    """
    if not isinstance(ctx, dict):
        raise TypeError(
            f"request context must be a dict, got {type(ctx).__name__}."
        )
    token = _CTX.set(ctx)
    try:
        yield ctx
    finally:
        _CTX.reset(token)


# -- Streaming ---------------------------------------------------------------

class _PinnedIterator:
    """An iterator that runs every step inside one fixed contextvars.Context."""

    __slots__ = ("_iterator", "_context")

    def __init__(self, iterator: Iterator, context: contextvars.Context) -> None:
        self._iterator = iterator
        self._context = context

    def __iter__(self) -> "_PinnedIterator":
        return self

    def __next__(self):
        # StopIteration propagates out of Context.run untouched, which is what
        # starlette's iterate_in_threadpool is watching for to end the response.
        return self._context.run(next, self._iterator)

    def close(self) -> None:
        """Close the wrapped generator INSIDE the pinned context.

        WHO ACTUALLY CALLS THIS, BECAUSE IT IS NOT STARLETTE.

        The previous version of this docstring asserted that a generator's
        finally block "runs during close()" as if the framework guaranteed the
        call. It does not, and the difference is the whole point. starlette
        1.3.1's iterate_in_threadpool() does `as_iterator = iter(iterator)` and
        never closes it; StreamingResponse never touches body_iterator on the
        way out either. Abandon a stream mid-flight and this object is dropped
        by refcount, the wrapped generator is finalised by the interpreter, and
        its finally block runs on whatever thread the collection happened on,
        OUTSIDE the pinned context - and, worse, after the request's db session
        has been torn down. FastAPI's AsyncExitStackMiddleware wraps the router,
        so every `yield` dependency (get_db among them) unwinds the moment
        Response.__call__ returns; a write from a GC-driven close has no live
        Session to write to even if it had a tenant.

        Measured against the running container, real uvicorn, a real socket RST
        after the first chunk: chunks 0-3 read client_id=probe-client, the
        finally block read client_id=None. The instrumented close() never fired.

        So the caller is ours. _PinnedStreamingResponse.__call__() invokes this
        in a finally, which runs on every exit - clean end, client disconnect,
        cancelled shutdown - and runs while the request's dependencies are still
        alive. That is what streaming_response() returns, and it is the only
        thing that makes the guarantee above real.

        A bare StreamingResponse(pinned_stream(gen, ctx), ...) gets the pinning
        for the chunks and NOT this. Do not write one; use streaming_response().

        Synchronous on the event-loop thread, on purpose. Dispatching the close
        to a worker would make it an await, and an await inside a cancelled
        scope raises before the callable runs - losing the cleanup at exactly
        the moment (server shutdown) it is least recoverable. The cost is that a
        disconnect-time finally block blocks the loop for its duration, so keep
        one to a single small write.
        """
        closer = getattr(self._iterator, "close", None)
        if closer is None:
            return
        try:
            self._context.run(closer)
        except Exception as exc:
            # Never let cleanup replace a clean client disconnect with a
            # traceback. Two things reach here: a finally block in the body that
            # raised (its own bug, but the response is already over), and
            # ValueError("generator already executing") if a future change ever
            # lets the pull thread outlive the response - anyio's
            # to_thread.run_sync defaults to abandon_on_cancel=False, which is
            # what currently makes that impossible, so if this ever fires with
            # that message the default has been overridden somewhere.
            logger.warning(
                "request context: closing a pinned stream body raised (%s: %s). "
                "The response was already finished; nothing is retried.",
                type(exc).__name__, exc,
            )


class _PinnedStreamingResponse(StreamingResponse):
    """A StreamingResponse that closes its pinned body, inside the pinned context.

    The whole reason streaming_response() returns a response object rather than
    leaving the caller to build one. starlette never closes a streaming body
    (see _PinnedIterator.close for the measurement), so the only place a
    deterministic close can be hung is the response's own __call__, which is
    what this overrides.
    """

    def __init__(self, body: _PinnedIterator, **kwargs: Any) -> None:
        super().__init__(body, **kwargs)
        # Our own handle, because self.body_iterator is NOT this object:
        # StreamingResponse.__init__ replaces a sync iterable with the
        # iterate_in_threadpool() async generator that wraps it, and that
        # wrapper has no close() worth calling.
        self._pinned_body = body

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # EVERY exit runs this: the stream ending normally, the shopper
            # closing the tab (starlette's disconnect handling differs by ASGI
            # spec_version - 2.3 collapses the task group and returns, 2.4+
            # raises ClientDisconnect - and a finally covers both, where the
            # `background=` hook starlette offers covers only one), and a
            # cancelled shutdown. Closing an already-exhausted generator is a
            # no-op, so the normal path costs nothing.
            self._pinned_body.close()


def streaming_response(
    iterable: Iterable,
    ctx: dict,
    *,
    media_type: Optional[str] = None,
    status_code: int = 200,
    headers: Optional[Mapping[str, str]] = None,
    background: Any = None,
) -> _PinnedStreamingResponse:
    """RETURN THIS, NOT StreamingResponse, FROM EVERY STREAMING ENDPOINT.

    THE FUNCTION THAT EXISTS SO THE NEXT ONE OF THESE IS NOT WRITTEN WRONG.

    pinned_stream() below is correct and was, for one release, a manual per-site
    opt-in with exactly one caller. Nothing stopped the next streaming endpoint
    from being written the obvious way - StreamingResponse(gen(), ...) - and
    that shape was measured losing the context in every chunk AND at the
    terminal write, with no error raised anywhere. A billing row that silently
    does not exist is found in a reconciliation months later, if at all. So the
    pinning is no longer something to remember; it is what the constructor you
    are supposed to reach for does.

    Two independent things keep a naive endpoint from shipping quietly:
    returning this instead of building a response by hand, and the ERROR that
    get_context() logs when it is read as absent from inside a streaming body.
    The second catches the endpoint that ignores the first.

    *ctx* is the dict the chokepoint returned - license_data. Passing it rather
    than reading get_context() here is deliberate: at the streaming site it is
    a local the handler is already holding, and requiring it means a caller who
    forgot to authorise cannot silently pin nothing.

    Pass it whatever auth path the request took. A v1 JWT license_data has none
    of the six tenant identifiers on it, but it is still the right thing to pin:
    usage_service decides by SHAPE and will refuse the row as NO CONTEXT at
    WARNING either way, whereas pinning conditionally would leave the context
    genuinely absent inside the body and trip get_context()'s guard on every v1
    streamed turn - a false ERROR on the whole of today's traffic. The
    chokepoint's bind_context() IS v2-only; that asymmetry is deliberate and
    the reason is written at the call site in retrieve.py.

    Call this LAST in the handler, for the reason in pinned_stream().

    The keyword arguments are StreamingResponse's own, forwarded unchanged, so
    there is nothing to learn here beyond the name. Note that *background* runs
    after the body is exhausted and does NOT run on a client disconnect on this
    stack - disconnect cleanup belongs in the generator's finally, which
    _PinnedStreamingResponse closes into the pinned context.
    """
    body = pinned_stream(iterable, ctx)
    return _PinnedStreamingResponse(
        body,
        status_code=status_code,
        headers=headers,
        media_type=media_type,
        background=background,
    )


def pinned_stream(iterable: Iterable, ctx: dict) -> _PinnedIterator:
    """Wrap a streaming body so *ctx* is visible inside it.

    The mechanism behind streaming_response(), and separate from it only
    because the two do different jobs: this pins the context to the iteration,
    that one also owns the response object and therefore the close. Reach for
    streaming_response() unless you are building a response starlette does not
    - handing this to a bare StreamingResponse gets the chunks right and loses
    the disconnect-time cleanup (see _PinnedIterator.close).

    THE ONE PLACE A BOUND CONTEXT DOES NOT REACH, and the reason this function
    exists rather than a comment saying "be careful".

    A sync `def` endpoint that returns StreamingResponse(event_stream(), ...)
    has already returned by the time a single byte of the generator runs.
    starlette pulls the generator through iterate_in_threadpool, one
    anyio.to_thread.run_sync per item, and each of those dispatches takes a
    FRESH copy_context() from the event-loop task - a task that never saw the
    handler's binding, because the handler ran in its own discarded copy.
    Measured in the container: the context bound by the chokepoint reads None
    in the first chunk, the second, and the terminal one.

    The obvious fix is worse than no fix. Setting the variable at the top of
    the generator works for chunk 1 and reads None from chunk 2 onwards,
    because the generator FRAME survives across yields and the Context does
    not. Anyone verifying by eyeballing the first token concludes it works and
    ships it; the usage write is the LAST thing the generator does, so the row
    it drops is the billable one for the whole turn - which is precisely the
    silent loss this rewrite exists to end.

    So the context is pinned to the Context object itself, and every next() and
    the close() run inside it.

    Pin LAST. copy_context() snapshots every contextvar as it stands right now,
    so anything bound after this call is invisible inside the stream.

    NONE OF THIS APPLIES TO AN ASYNC BODY, and the check below says so rather
    than letting iter() raise something cryptic. starlette only reaches for
    iterate_in_threadpool() when the body is a plain iterable; an AsyncIterable
    is iterated directly with `async for` on the event loop, inside the same
    Task the handler ran in, so an `async def` endpoint's binding is simply
    still there. Measured the same way as everything else here - async endpoint,
    async generator, bare StreamingResponse, real socket RST after the first
    chunk: every chunk AND the finally block read the bound tenant. There is
    nothing for this function to fix there.
    """
    if not isinstance(ctx, dict) or not ctx:
        raise TypeError(
            "pinned_stream() needs the tenant context the chokepoint returned; "
            f"got {type(ctx).__name__}. Pass the handler's license_data - "
            "pinning an empty context is the same silent loss as not pinning."
        )

    if hasattr(iterable, "__aiter__"):
        raise TypeError(
            "pinned_stream() takes a SYNC iterable; got an async one "
            f"({type(iterable).__name__}). An async body does not need pinning: "
            "starlette iterates it on the event loop inside the request's own "
            "Task, so the chokepoint's context is still bound and every chunk "
            "sees it (measured). Return StreamingResponse(agen(), ...) directly "
            "- but check first that the endpoint is `async def`, because a sync "
            "`def` handler runs in a worker thread whose context is discarded "
            "before the body starts, and neither shape saves it."
        )

    context = contextvars.copy_context()
    context.run(_CTX.set, ctx)
    return _PinnedIterator(iter(iterable), context)


# -- interaction_id, from the wire -------------------------------------------

# usage_events.interaction_id is VARCHAR(64). usage_service truncates anything
# longer and warns about it on every single row, so trim here instead and keep
# the log readable.
MAX_INTERACTION_ID = 64

# AIChatbot's RequestTimer stamps this on every backend call of one shopper
# turn, which is the grouping usage_events actually wants: a turn is three or
# four HTTP requests (tool-call, retrieve products, retrieve content, answer)
# and minting per request would split one turn across four ids and orphan the
# retrieval spend from the answer it paid for. The other modules do not send it
# yet - for them the fallback mint really is one id per HTTP request, and
# fixing that is a plugin-side change, not something this module can do.
INTERACTION_ID_HEADER = "X-Request-Id"

# Conservative on purpose. This value is chosen by the caller, lands in a
# VARCHAR(64) and is printed in log lines on the way, so a newline in it forges
# a whole log entry - an attacker choosing what the operator's incident
# timeline says. license_key.parse_for_logging() rejects for the same reason
# and with the same reasoning. UUIDs, hex ids and dotted or colon-separated
# trace ids all pass; anything else is not worth salvaging when minting a fresh
# id costs nothing.
_INTERACTION_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,%d}$" % MAX_INTERACTION_ID)


def interaction_id_from_header(raw: Optional[str]) -> Optional[str]:
    """A caller-supplied request id, if it is safe to use, else None.

    Returns None rather than a sanitised version of a bad value: a mangled id
    still groups rows, just under a string nobody can trace back to anything,
    whereas None tells the caller to mint one that is at least internally
    consistent. The caller mints - see the module docstring for why this file
    cannot import usage_service to do it here.
    """
    if not raw:
        return None
    candidate = raw.strip()
    if not _INTERACTION_ID_RE.match(candidate):
        return None
    return candidate
