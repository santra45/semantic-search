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
  * pinned_stream() exists because a StreamingResponse generator is the one
    place a bound context provably does NOT reach, and the naive fix passes a
    smoke test before failing where it counts. See its docstring.
  * bind_context() logs at ERROR if it finds a different tenant already bound,
    which is the only way the brief's leakage story could ever come true. If
    that line ever fires, the premises above have changed and this comment is
    the thing to re-verify.

STDLIB ONLY, DELIBERATELY
-------------------------
This module imports nothing from the rest of the application, and must not
start. usage_service.record() is going to import it to read the ambient
context; usage_service already imports licensing_service. The moment this file
imports usage_service back - for new_interaction_id(), which is the tempting
one - that is a circular import and the app stops booting. Minting stays with
the caller. interaction_id_from_header() below only sanitises.
"""

from __future__ import annotations

import contextvars
import logging
import re
from contextlib import contextmanager
from typing import Iterable, Iterator, Optional

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

def get_context() -> Optional[dict]:
    """The tenant context bound for this request, or None.

    None means one of three things and the caller has to treat them the same
    way: no chokepoint ran (a router that still authenticates inline), the
    caller is not on a request at all, or the caller is inside a
    StreamingResponse generator, which cannot see a bound context - see
    pinned_stream().

    The dict is the SAME OBJECT the chokepoint returned to the handler, not a
    copy. That is on purpose: what a shared service reads here is exactly what
    it would have been handed if the value had been threaded down by hand, so
    there is one shape to reason about rather than two. It also means it
    carries the presented licence key in plain text under "license_key",
    because that key is the KEK every merchant-supplied LLM key is encrypted
    under and roughly twenty call sites need it. Never log this dict whole.
    """
    return _CTX.get()


def current_interaction_id() -> Optional[str]:
    """The id threading every row of the current turn, or None.

    Convenience over get_context() because a writer that only needs the id
    should not have to know the context is a dict, nor guess the key name.
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

        Not decorative. A generator's finally block is where a usage write
        belongs if the shopper disconnects mid-stream, and that block runs
        during close(). Closing from outside the pinned context would leave
        exactly that write - the one for a turn that was paid for and not
        delivered - without a tenant.
        """
        closer = getattr(self._iterator, "close", None)
        if closer is not None:
            self._context.run(closer)


def pinned_stream(iterable: Iterable, ctx: dict) -> _PinnedIterator:
    """Wrap a StreamingResponse body so *ctx* is visible inside it.

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
    the final close() run inside it.

    Pin LAST. copy_context() snapshots every contextvar as it stands right now,
    so anything bound after this call is invisible inside the stream.

    Passing *ctx* explicitly, rather than reading get_context() here, is also
    deliberate: at the streaming site the resolved dict is a local the handler
    is already holding, and requiring it means a caller who forgot to
    authorise cannot silently pin nothing.
    """
    if not isinstance(ctx, dict) or not ctx:
        raise TypeError(
            "pinned_stream() needs the tenant context the chokepoint returned; "
            f"got {type(ctx).__name__}. Pass the handler's license_data - "
            "pinning an empty context is the same silent loss as not pinning."
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
