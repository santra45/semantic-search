"""
Pure retrieval endpoints. The backend is a Qdrant proxy now — zero chat logic.

  POST /api/magento/chatbot/retrieve/products    — semantic / structured product search
  POST /api/magento/chatbot/retrieve/content     — CMS pages / blocks / widgets / store config
  POST /api/magento/chatbot/retrieve/answer      — optional RAG summary over provided sources

All three are license + domain guarded. They never see chat history and never
touch any identifiers beyond client_id (already anonymous).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from backend.app.services.database import get_db
from backend.app.services.embedder import embed_query
from backend.app.services.qdrant_service import (
    get_collection_name,
    qdrant,
    search_content as qdrant_search_content,
    search_products as qdrant_search_products,
)
from backend.app.services.token_usage_service import TokenUsageTracker
from backend.app.utils.llm_logger import log_llm_call

from backend.app.magento.chatbot.routers.common import (
    authorize_request,
    decrypt_llm_key,
    maybe_persist_magento_creds,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Schemas ──────────────────────────────────────────────────────────────────


class ProductRetrieveRequest(BaseModel):
    license_key: Optional[str] = None
    query: Optional[str] = None
    skus: list[str] = Field(default_factory=list)
    content_types: list[str] = Field(default_factory=lambda: ["product"])
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    only_in_stock: bool = False
    attribute_filters: dict[str, str] = Field(default_factory=dict)  # {"color": "red", "size": "m"}
    category_id: Optional[str] = None
    limit: int = 8
    rerank: bool = False  # admin-toggled — small LLM rerank of top-N
    # When `rerank=True` the reranker uses these to pick the right provider/model
    # (otherwise falls back to the service's defaults, which may not match the
    # tenant's billing config and will lose cost tracking).
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    # Customer's current store view. When set, retrieval is scoped to
    # points tagged with this store_code (set by the Magento side at
    # sync time from store->getCode()). Absent / empty means "no store
    # filter" — used by legacy single-store deployments that pre-date
    # per-store sync.
    store_code: Optional[str] = None
    # Sort intent extracted by the PHP classifier from phrases like
    # "cheapest", "most expensive", "lowest priced". The Qdrant search
    # itself returns by semantic similarity (vector cosine); we fetch a
    # larger relevance-ordered pool, then apply this sort in Python over
    # the on-topic candidates, then slice back to `limit`. So semantic
    # relevance still gates which products appear, but the order honours
    # the customer's stated preference.
    #
    # sort_by   — currently "price" only. Future-proofed for "name",
    #             "rating", "newest" etc. Unknown values fall back to
    #             vector relevance (no resort) instead of erroring.
    # sort_order — "asc" | "desc". Defaults to "asc" when sort_by is set
    #              and order is omitted.
    sort_by: Optional[str] = None
    sort_order: Optional[str] = None

    @field_validator("sort_by", mode="before")
    @classmethod
    def _coerce_sort_by(cls, value):
        if value in (None, "", [], {}):
            return None
        v = str(value).strip().lower()
        # Whitelist known sort keys; silently drop unknowns rather than
        # 422-ing so a forward-incompatible classifier output never
        # breaks the retrieval call.
        return v if v in {"price", "name", "rating", "newest"} else None

    @field_validator("sort_order", mode="before")
    @classmethod
    def _coerce_sort_order(cls, value):
        if value in (None, "", [], {}):
            return None
        v = str(value).strip().lower()
        return v if v in {"asc", "desc"} else None

    @field_validator("attribute_filters", mode="before")
    @classmethod
    def _coerce_attribute_filters(cls, value):
        """PHP json-encodes empty arrays as `[]`, which doesn't match a `dict`
        schema. Accept `None`/`[]`/list-of-{name,value}-dicts and coerce."""
        if value in (None, "", [], {}):
            return {}
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items() if v not in (None, "")}
        if isinstance(value, list):
            out: dict[str, str] = {}
            for entry in value:
                if isinstance(entry, dict):
                    name = entry.get("name") or entry.get("key") or entry.get("code")
                    val  = entry.get("value") or entry.get("option")
                    if name and val:
                        out[str(name)] = str(val)
            return out
        return {}

    @field_validator("skus", "content_types", mode="before")
    @classmethod
    def _coerce_string_list(cls, value):
        if value in (None, "", [], {}):
            return []
        if isinstance(value, str):
            return [s.strip() for s in value.split(",") if s.strip()]
        if isinstance(value, list):
            return [str(v) for v in value if v not in (None, "")]
        return []

    @field_validator("min_price", "max_price", mode="before")
    @classmethod
    def _coerce_optional_float(cls, value):
        if value in (None, "", [], {}):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @field_validator("category_id", mode="before")
    @classmethod
    def _coerce_optional_str(cls, value):
        if value in (None, "", [], {}):
            return None
        return str(value)


class ContentRetrieveRequest(BaseModel):
    license_key: Optional[str] = None
    query: str
    content_types: list[str] = Field(default_factory=lambda: ["cms_page", "cms_block"])
    limit: int = 5
    # See ProductRetrieveRequest.store_code — same semantics.
    store_code: Optional[str] = None

    @field_validator("content_types", mode="before")
    @classmethod
    def _coerce_types(cls, value):
        if value in (None, "", [], {}):
            return ["cms_page", "cms_block"]
        if isinstance(value, str):
            return [s.strip() for s in value.split(",") if s.strip()]
        if isinstance(value, list):
            return [str(v) for v in value if v not in (None, "")]
        return ["cms_page", "cms_block"]


class AnswerRequest(BaseModel):
    license_key: Optional[str] = None
    query: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_api_key_encrypted: Optional[str] = None
    # Optional store contact details forwarded by the Magento agents
    # (PolicyAgent / StoreInfoAgent / GenericChatAgent). When set, the
    # prompt instructs the LLM to surface phone/email in any refusal so
    # the customer always has a path forward. Pulled live from Magento's
    # general/store_information/* config — universal across clients.
    #
    # Shape: {"phone": "+44 ...", "email": "support@..."}
    contact: dict[str, str] = Field(default_factory=dict)
    # Optional last-few-turns of conversation history for multi-turn
    # follow-up queries ("what's the price of next-day delivery" after a
    # product question about delivery options). Each item:
    #   {"role": "user"|"assistant", "content": "..."}
    # Capped at 6 turns by the prompt builder — older context is ignored.
    conversation_history: list[dict[str, str]] = Field(default_factory=list)

    # Purpose switches the prompt template:
    #
    #   "answer"   (default) — strict grounded Q&A. Refuses when evidence
    #                          doesn't support a claim. Used by PolicyAgent,
    #                          StoreInfoAgent, GenericChatAgent for policy /
    #                          info questions where confident-but-wrong
    #                          answers are the failure mode to avoid.
    #
    #   "preamble"           — confirmation framing. Used by
    #                          ProductSearchAgent's NL answer-line. Vector
    #                          search has ALREADY filtered to relevant
    #                          products; the LLM's job is to acknowledge
    #                          the customer's request in one sentence
    #                          using their phrasing, NOT to second-guess
    #                          whether the evidence supports the claim.
    #                          Skipping the strict refusal rule is the
    #                          whole point.
    #
    #   "comparison"         — comparative framing. Used by GenericChatAgent's
    #                          "X vs Y" branch. Asks the LLM to contrast
    #                          both sides using product evidence per
    #                          operand.
    purpose: Optional[str] = "answer"
    # Free-form extra instruction the caller wants surfaced in the
    # prompt. ProductSearchAgent uses this to pass framing hints
    # ("results are sorted cheapest first", "the customer described
    # their situation as advisory"). Distinct from `query` so the LLM
    # doesn't confuse the instruction with the customer's question.
    instruction: Optional[str] = None

    @field_validator("contact", mode="before")
    @classmethod
    def _coerce_contact(cls, value):
        if value in (None, "", [], {}):
            return {}
        if isinstance(value, dict):
            out: dict[str, str] = {}
            for key in ("phone", "email"):
                v = value.get(key)
                if v is not None and str(v).strip() != "":
                    out[key] = str(v).strip()
            return out
        return {}

    @field_validator("conversation_history", mode="before")
    @classmethod
    def _coerce_history(cls, value):
        if value in (None, "", [], {}):
            return []
        if not isinstance(value, list):
            return []
        out: list[dict[str, str]] = []
        for entry in value:
            if not isinstance(entry, dict):
                continue
            role = str(entry.get("role") or "").strip().lower()
            content = str(entry.get("content") or "").strip()
            if role in ("user", "assistant") and content != "":
                out.append({"role": role, "content": content})
        return out[-6:]


# ── Routes ───────────────────────────────────────────────────────────────────


@router.post("/magento/chatbot/retrieve/products")
def retrieve_products(
    req: ProductRetrieveRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_llm_api_key_encrypted: Optional[str] = Header(None, alias="X-LLM-API-Key-Encrypted"),
    x_magento_creds: Optional[str] = Header(None, alias="X-Magento-Admin-Creds-Encrypted"),
    db: Session = Depends(get_db),
):
    license_data = authorize_request(
        request=request, db=db,
        authorization=authorization, x_api_key=x_api_key,
        request_license=req.license_key,
    )
    maybe_persist_magento_creds(
        db=db,
        client_id=license_data["client_id"],
        license_key=license_data["license_key"],
        encrypted_creds_header=x_magento_creds,
    )

    client_id = license_data["client_id"]
    domain = license_data["domain"]

    # ── Direct SKU lookup path (no embedding, no LLM) ────────────────────────
    if req.skus:
        hits = _lookup_by_skus(client_id, domain, req.skus, store_code=req.store_code)
        return {
            "results": hits,
            "count": len(hits),
            "mode": "sku",
        }

    # ── Semantic search path ─────────────────────────────────────────────────
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query or skus is required")

    embedding_api_key = decrypt_llm_key(x_llm_api_key_encrypted, license_data["license_key"])
    query_vector = embed_query(req.query.strip(), embedding_api_key, client_id)

    # Fetch over-broad so the post-filter has room to narrow down. When a
    # sort intent is present we widen even further — we want enough
    # on-topic candidates that the price-sorted slice doesn't just show
    # the same N closest-by-vector items. Universal: every catalog with
    # > limit relevant items benefits from the wider pool.
    fan_out = 3 if (req.attribute_filters or req.category_id) else 1
    if req.sort_by:
        fan_out = max(fan_out, 5)
    raw_limit = max(req.limit, req.limit * fan_out)

    hits = qdrant_search_products(
        client_id=client_id,
        domain=domain,
        query_vector=query_vector,
        limit=raw_limit,
        min_price=req.min_price,
        max_price=req.max_price,
        only_in_stock=req.only_in_stock,
        content_types=req.content_types or ["product"],
        store_code=req.store_code,
    )

    # Attribute + category filtering — these are stored as payload booleans,
    # so we filter server-side here (Qdrant's FieldCondition also works but
    # the Python side is simpler for dynamic keys and lets us keep the
    # qdrant_service generic).
    if req.attribute_filters:
        for attr, value in req.attribute_filters.items():
            key = f"attr_{_slug(attr)}_{_slug(value)}"
            hits = [h for h in hits if h.get(key) is True]
    if req.category_id:
        key = f"cat_{req.category_id}"
        hits = [h for h in hits if h.get(key) is True]

    # ── Apply customer-requested sort on the on-topic candidate pool ─────
    # Runs AFTER attribute/category narrowing so the sort operates on
    # items that already match what the customer asked for. Vector
    # relevance gated which products made it this far; the sort decides
    # the final ordering. Stable sort preserves relevance order within
    # equal sort keys, so a tie on price (rare in practice) keeps the
    # most-relevant item on top.
    if req.sort_by:
        hits = _apply_sort(hits, req.sort_by, req.sort_order or "asc")

    hits = hits[: req.limit]

    mode = "semantic"
    if req.rerank and hits:
        try:
            hits = _llm_rerank(
                req.query.strip(),
                hits,
                license_data=license_data,
                llm_api_key=embedding_api_key,
                llm_provider=req.llm_provider,
                llm_model=req.llm_model,
                db=db,
            )
            mode = "semantic+rerank"
        except Exception as exc:
            logger.warning("retrieve/products rerank failed: %s", exc)

    return {
        "results": hits,
        "count": len(hits),
        "mode": mode,
    }


@router.post("/magento/chatbot/retrieve/content")
def retrieve_content(
    req: ContentRetrieveRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_llm_api_key_encrypted: Optional[str] = Header(None, alias="X-LLM-API-Key-Encrypted"),
    db: Session = Depends(get_db),
):
    license_data = authorize_request(
        request=request, db=db,
        authorization=authorization, x_api_key=x_api_key,
        request_license=req.license_key,
    )

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    embedding_api_key = decrypt_llm_key(x_llm_api_key_encrypted, license_data["license_key"])
    query_vector = embed_query(req.query.strip(), embedding_api_key, license_data["client_id"])

    hits = qdrant_search_content(
        client_id=license_data["client_id"],
        domain=license_data["domain"],
        query_vector=query_vector,
        limit=req.limit,
        content_types=req.content_types or ["cms_page", "cms_block"],
        store_code=req.store_code,
    )
    return {"results": hits, "count": len(hits)}


@router.post("/magento/chatbot/retrieve/answer")
def retrieve_answer(
    req: AnswerRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_llm_api_key_encrypted: Optional[str] = Header(None, alias="X-LLM-API-Key-Encrypted"),
    db: Session = Depends(get_db),
):
    """Optional RAG summary. Admin-toggled on the Magento side; if the
    Magento module never calls this endpoint the LLM is never invoked."""
    license_data = authorize_request(
        request=request, db=db,
        authorization=authorization, x_api_key=x_api_key,
        request_license=req.license_key,
    )

    if not req.query.strip() or not req.sources:
        raise HTTPException(status_code=400, detail="query and sources are required")

    api_key = decrypt_llm_key(x_llm_api_key_encrypted, license_data["license_key"])

    from backend.app.magento.chatbot.agents.llm_factory import build_llm

    llm = build_llm(
        provider=req.llm_provider,
        model=req.llm_model,
        api_key=api_key,
        temperature=0.2,
    )

    prompt = _build_answer_prompt(
        query=req.query.strip(),
        sources=req.sources,
        contact=req.contact,
        conversation_history=req.conversation_history,
        purpose=req.purpose or "answer",
        instruction=req.instruction,
    )

    from langchain_core.messages import HumanMessage

    provider_name = (req.llm_provider or "google").lower()
    model_name    = req.llm_model or "gemini-2.0-flash-lite"

    with log_llm_call(
        provider=provider_name,
        model=model_name,
        purpose="chat_answer",
        prompt=prompt,
        client_id=license_data["client_id"],
    ) as _log_ctx:
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
        except Exception as exc:
            logger.warning("retrieve/answer LLM invoke failed: %s", exc)
            raise HTTPException(status_code=502, detail="LLM unavailable")

        answer_text = _extract_text(resp.content).strip()
        usage = getattr(resp, "usage_metadata", None) or {}
        input_tokens  = int(usage.get("input_tokens",  0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)

        # Cost is computed from the model pricing table — matches what the
        # /classify endpoint returns so the Magento per-message billing row
        # can sum cost across both call types without per-shape branches.
        # Split into input_cost / output_cost so the token_usage_tracking
        # row populates both component columns (the tracker computes
        # total_cost = input_cost + output_cost internally). The previous
        # version computed `cost` as a single value and never passed it to
        # the tracker — every chat_answer row ended up with zeroed cost
        # columns despite token counts being correct.
        from backend.app.services.llm_rerank_service import MODEL_PRICING
        pricing = MODEL_PRICING.get(model_name, {})
        input_cost  = input_tokens  * pricing.get("input",  0.0)
        output_cost = output_tokens * pricing.get("output", 0.0)
        cost = input_cost + output_cost

        _log_ctx.record(
            response_text=answer_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=float(cost),
            extra={"sources": len(req.sources or [])},
        )
    try:
        TokenUsageTracker(db).create_usage_record(
            client_id=license_data["client_id"],
            query_type="chat_answer",
            llm_provider=req.llm_provider or "google",
            llm_model=req.llm_model or "gemini-2.0-flash-lite",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=float(input_cost),
            output_cost=float(output_cost),
            request_text_length=len(prompt),
            response_text_length=len(answer_text),
        )
    except Exception:
        pass

    return {
        "answer": answer_text,
        "grounded": True,
        "usage": {
            "input":    input_tokens,
            "output":   output_tokens,
            "cost":     float(cost),
            "provider": provider_name,
            "model":    model_name,
        },
    }


@router.post("/magento/chatbot/retrieve/answer/stream")
def retrieve_answer_stream(
    req: AnswerRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_llm_api_key_encrypted: Optional[str] = Header(None, alias="X-LLM-API-Key-Encrypted"),
    db: Session = Depends(get_db),
):
    """Streaming variant of /retrieve/answer.

    Returns NDJSON (newline-delimited JSON), one event per line:
        {"event": "token", "text": "..."}    — incremental chunk
        {"event": "done",  "answer": "...", "usage": {...}}    — terminal

    NDJSON over chunked transfer encoding rather than SSE because:
      - PHP can read it as plain chunks via cURL WRITEFUNCTION (the Magento
        proxy in Controller/Ajax/Stream.php) without needing an SSE parser.
      - Frontend can use fetch + ReadableStream.getReader without
        EventSource — fetch is what we already use for everything else.
      - Restart-friendly: each line is a complete JSON object so a partial
        chunk just means "wait for more bytes".

    Behaviour parity with /retrieve/answer: same auth, same prompt, same
    cost computation, same TokenUsageTracker write at the end. The only
    difference is the response shape.
    """
    license_data = authorize_request(
        request=request, db=db,
        authorization=authorization, x_api_key=x_api_key,
        request_license=req.license_key,
    )

    if not req.query.strip() or not req.sources:
        raise HTTPException(status_code=400, detail="query and sources are required")

    api_key = decrypt_llm_key(x_llm_api_key_encrypted, license_data["license_key"])

    from backend.app.magento.chatbot.agents.llm_factory import build_llm

    llm = build_llm(
        provider=req.llm_provider,
        model=req.llm_model,
        api_key=api_key,
        temperature=0.2,
    )

    # Same prompt as the one-shot endpoint — keeps answer style consistent
    # whether the merchant has streaming on or off.
    prompt = _build_answer_prompt(
        query=req.query.strip(),
        sources=req.sources,
        contact=req.contact,
        conversation_history=req.conversation_history,
        purpose=req.purpose or "answer",
        instruction=req.instruction,
    )

    from langchain_core.messages import HumanMessage

    provider_name = (req.llm_provider or "google").lower()
    model_name    = req.llm_model or "gemini-2.0-flash-lite"

    def event_stream():
        # Accumulate as we go so we can compute total cost + write the
        # tracker row once at the end. The langchain stream() yields
        # AIMessageChunk objects; the FINAL chunk usually carries
        # usage_metadata, but we max-merge across chunks to be safe.
        full_answer: list[str] = []
        in_tokens  = 0
        out_tokens = 0
        try:
            for chunk in llm.stream([HumanMessage(content=prompt)]):
                token_text = _extract_text(getattr(chunk, "content", "")) or ""
                if token_text:
                    full_answer.append(token_text)
                    yield json.dumps({"event": "token", "text": token_text}) + "\n"
                meta = getattr(chunk, "usage_metadata", None) or {}
                if meta:
                    in_tokens  = max(in_tokens,  int(meta.get("input_tokens",  0) or 0))
                    out_tokens = max(out_tokens, int(meta.get("output_tokens", 0) or 0))
        except Exception as exc:
            logger.warning("retrieve/answer/stream LLM stream failed: %s", exc)
            yield json.dumps({"event": "error", "message": "LLM unavailable"}) + "\n"
            return

        answer_text = "".join(full_answer).strip()

        # Cost calculation — split into input/output components so the
        # tracker row matches what /retrieve/answer writes (same fix from
        # the cost-tracking bug we patched a few turns ago).
        from backend.app.services.llm_rerank_service import MODEL_PRICING
        pricing = MODEL_PRICING.get(model_name, {})
        input_cost  = in_tokens  * pricing.get("input",  0.0)
        output_cost = out_tokens * pricing.get("output", 0.0)
        cost = input_cost + output_cost

        try:
            TokenUsageTracker(db).create_usage_record(
                client_id=license_data["client_id"],
                query_type="chat_answer",
                llm_provider=provider_name,
                llm_model=model_name,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                input_cost=float(input_cost),
                output_cost=float(output_cost),
                request_text_length=len(prompt),
                response_text_length=len(answer_text),
            )
        except Exception:
            pass

        yield json.dumps({
            "event":  "done",
            "answer": answer_text,
            "usage": {
                "input":    in_tokens,
                "output":   out_tokens,
                "cost":     float(cost),
                "provider": provider_name,
                "model":    model_name,
            },
        }) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_answer_prompt(
    query: str,
    sources: list[dict],
    contact: Optional[dict] = None,
    conversation_history: Optional[list[dict]] = None,
    purpose: str = "answer",
    instruction: Optional[str] = None,
) -> str:
    """Single source of truth for the /retrieve/answer prompt — shared
    between the one-shot and streaming endpoints so style and rules stay
    in sync.

    Three modes, switched by `purpose`:

      "answer" (default) — strict grounded Q&A. Refuses when the
        evidence doesn't support a claim. Right behaviour for policy /
        info questions where confident-but-wrong is the failure mode.

      "preamble" — confirmation framing for ProductSearchAgent's
        natural-language answer line. Vector search has ALREADY filtered
        to relevant products; the prompt's job is to acknowledge the
        customer's request in one sentence, NOT to second-guess whether
        the evidence supports the claim. The strict refusal rule is
        DROPPED entirely — if the LLM hesitates here, it produces "I
        don't have specific information about indoor water features"
        while looking at four indoor water features, which is exactly
        the failure the senior reported.

      "comparison" — used by GenericChatAgent's comparative branch.
        Contrasts both sides explicitly.

    Across all modes:
      - **Conversation context** (last 6 turns) for pronoun resolution.
      - **Contact details on refusal** in "answer" mode only — preamble
        doesn't refuse.
      - **Direction-of-flow rule** for damage scenarios.
    """
    sources_blob = "\n\n".join(
        _format_source_for_prompt(s) for s in (sources or [])[:6]
    )

    history_block = ""
    if conversation_history:
        history_lines = []
        for turn in conversation_history[-6:]:
            role = (turn.get("role") or "").strip().lower()
            text = (turn.get("content") or "").strip()
            if not text or role not in ("user", "assistant"):
                continue
            history_lines.append(
                f"{'CUSTOMER' if role == 'user' else 'ASSISTANT'}: {text}"
            )
        if history_lines:
            history_block = (
                "\n\nConversation so far (most recent last) — use to resolve "
                "pronouns and follow-ups, but don't quote it verbatim:\n"
                + "\n".join(history_lines)
            )

    instruction_block = ""
    if instruction:
        instruction_block = f"\n\nAdditional framing instruction: {instruction.strip()}"

    if purpose == "preamble":
        # CONFIRMATION mode — used by ProductSearchAgent's NL answer-line.
        # Drops the strict refusal rule. The product cards render below
        # the message regardless, so the prompt's job is ONE sentence.
        return (
            "You are writing the lead-in sentence for a product search result. "
            "Vector search has ALREADY confirmed these products match the customer's request — "
            "your job is to acknowledge that match in natural language, NOT to second-guess "
            "whether the evidence supports it.\n\n"
            "Rules:\n"
            " - Reply with ONE short sentence. Two at most.\n"
            " - Use the customer's own phrasing where it fits (e.g. if they asked about "
            "\"indoor water features\", use \"indoor water features\" in your reply — not "
            "an internal attribute name).\n"
            " - For yes/no questions, start with \"Yes\" or \"Yes — \".\n"
            " - For \"which X would you suggest\" / \"recommend\" / \"help me choose\" "
            "questions, frame as advice: \"For a small garden, here are some compact "
            "options to consider:\", \"These would suit a beginner:\".\n"
            " - Do NOT list product names or SKUs. Cards render below this sentence.\n"
            " - Do NOT refuse or claim insufficient information. The match is real.\n"
            " - Do NOT use markdown headings or bullets — plain prose only.\n"
            + instruction_block
            + "\n\n"
            f"Customer question: {query}"
            + history_block
            + f"\n\nMatched products (for context — DO NOT list these by name):\n{sources_blob}"
        )

    # Default "answer" purpose + "comparison" (same prompt, comparison
    # gets an extra hint via `instruction`).
    #
    # NOTE: there used to be a `contact_line` directive here that asked
    # the LLM to mention phone/email inline whenever it refused. That
    # was the right idea at the time but it conflicts with the chip
    # rendering UX — the frontend now shows clickable tel:/mailto:
    # chips below every chatbot reply, so the LLM dumping the same
    # numbers into its text body produced an ugly "duplicate contact"
    # experience AND triggered refusals in cases where the evidence
    # actually contained the answer (the LLM saw the "include contact"
    # instruction and concluded it should refuse to use it). We rely on
    # the chip channel for contact handoff; the LLM should produce
    # clean answers using only the source content.
    return (
        "You are a concise store assistant. Answer the customer's question using ONLY the sources below.\n\n"
        "Rules:\n"
        " - If the sources don't contain the answer, say honestly in one sentence that you don't have "
        "that specific information. Do NOT include phone numbers or email addresses in your reply — the "
        "interface renders contact options as separate clickable chips below your message; mentioning "
        "them inline duplicates that and makes the reply feel cluttered.\n"
        " - Keep the answer to 1-2 short sentences. For \"tell me about\" requests, you may use 2-3 sentences.\n"
        " - Put any concrete number, measurement, timeframe, or money value in **bold** (e.g. **30 days**, "
        "**$50**, **1.5 kg**).\n"
        " - Never invent SKUs, prices, dates, dimensions, or policy terms that aren't in the sources.\n"
        " - No markdown headings. No bulleted lists unless the customer explicitly asked for a list.\n"
        " - Plain prose only — write the way a helpful human store assistant would.\n"
        " - **Direction of flow matters.** If the customer describes RECEIVING a damaged, broken, or "
        "defective product, and the evidence only covers RETURNING such items (e.g. 'damaged-on-return' "
        "or 'sender's responsibility'), do NOT apply that evidence to the customer's situation. Say plainly "
        "that the policy in evidence is for customer-initiated returns, and that received-damaged claims "
        "need to be raised with the support team separately.\n"
        " - **Comparative questions** (\"difference between X and Y\", \"X vs Y\"): contrast both sides "
        "using only what the sources say about each. If the evidence covers only one side, name what "
        "you do know and acknowledge the gap on the other.\n"
        " - **When the question is about store info** (address, hours, phone, email, payment options, "
        "shipping methods, etc.) and a `[store_config: ...]` source is present, USE the facts in that "
        "source verbatim. The composite contains the literal values from the store's admin config — they "
        "are authoritative. Do not say you don't have the information when a store_config source is "
        "provided that contains the answer.\n"
        " - **FACTUAL EXTRACTION across messy sources.** Merchants frequently store key facts (address, "
        "phone, hours, payment methods, policy text) inside CMS blocks with unrelated names (`footer_v2`, "
        "`sirena_footer`, `info_main`) or as inline HTML in pages whose identifier is cryptic (`pp` for "
        "privacy policy, `home` for an FAQ-shaped page). When the customer asks for a SPECIFIC fact "
        "(e.g. \"what's the store address?\", \"do you accept PayPal?\", \"when are you open?\"), scan "
        "the BODY of every source — including blocks whose title looks unrelated — for that fact. If a "
        "block contains an address-shaped sequence, a phone number, opening hours, etc., that's likely "
        "where the merchant has put the answer. Pull it out and present it cleanly. Do NOT refuse on the "
        "basis that the source's title doesn't match the question; the merchant's content hygiene is "
        "their problem, not the customer's."
        + instruction_block
        + "\n\n"
        f"Customer question: {query}"
        + history_block
        + f"\n\nSources:\n{sources_blob}"
    )


def _format_source_for_prompt(s: dict) -> str:
    """Flatten one source into the text block the RAG summarizer sees.

    Per-content-type formatting because different shapes need different
    framing for the LLM:
      - product             → sku, variants, price, stock, attributes, description.
      - cms_page / cms_block → URL, heading, meta description, content body.
      - store_config         → the full composite body (contact_info, shipping_options,
                               payment_options, store_identity, tax_info, store_rules,
                               locale_info, social_links). Address/phone/hours live
                               in `contact_info`; truncating these at 800 chars meant
                               the LLM saw the anchors but not the actual facts.
      - everything else      → generic title + body fallback.

    When the source carries a `comparison_side` tag (set by
    GenericChatAgent's comparative branch), the side is prepended so the
    LLM can cleanly attribute facts to the right operand.
    """
    ct = (s.get("content_type") or "").lower()
    title = s.get("title") or s.get("name") or s.get("identifier") or s.get("sku") or s.get("label") or ""
    side = (s.get("comparison_side") or "").strip()
    side_prefix = f"[COMPARE-SIDE: {side}] " if side else ""

    if ct == "product" or s.get("sku") or s.get("type_id"):
        return side_prefix + _format_product_source(s, title)

    if ct in ("cms_page", "cms_block"):
        return side_prefix + _format_cms_source(s, ct, title)

    if ct == "store_config":
        return side_prefix + _format_store_config_source(s, title)

    body = (s.get("summary") or s.get("content") or s.get("description") or "")[:800]
    return f"{side_prefix}[{ct or 'source'}] {title}\n{body}"


def _format_store_config_source(s: dict, title: str) -> str:
    """Lay out a store_config composite for the LLM.

    StoreConfigContentProvider on the Magento side packs each composite
    (contact_info, shipping_options, payment_options, store_identity,
    tax_info, store_rules, locale_info, social_links) into a single
    payload with:

      - `label`   — human title (e.g. "Store contact information")
      - `content` — anchors + facts ("Phone number: ...; Email: ...;
                    Mailing address: street, city, postcode, country;
                    Business hours: ...")
      - `value`   — facts only, no anchors
      - `summary` — first 300 chars of facts, used for card snippets

    Previously the generic fallback read `summary` first and capped at
    800 chars — so the LLM saw 300 chars of facts and lost half the
    composite. That's why "what's the store address" got refused even
    when the address WAS indexed: the address was past the 300-char cap.

    Now we explicitly read `value` (facts only) preferring it over
    `content` (which leads with anchor text the LLM doesn't need to
    see), cap at 6000 chars (composites rarely exceed 2000), and tag
    the source with its composite key so the LLM knows which kind of
    info it's looking at.
    """
    key = str(s.get("key") or s.get("entity_id") or "").strip() or "store_config"

    # `value` is "facts only" (no anchor preamble). Prefer it because
    # the anchors are retrieval-only signal — at answer time they just
    # take up token budget. Fall back to `content` (which contains
    # anchors + facts) when `value` is empty (old payloads from before
    # the composite split).
    body = (
        s.get("value")
        or s.get("content")
        or s.get("summary")
        or ""
    )
    body = str(body)

    label = title or s.get("label") or key.replace("_", " ").title()

    parts = [f"[store_config: {key}] {label}"]
    if body:
        parts.append(body)
    return "\n".join(parts)


def _format_cms_source(s: dict, ct: str, title: str) -> str:
    """Lay out a CMS page / block with all its useful framing context.

    The LLM benefits from seeing the page's display heading and URL — both
    let it write a more specific answer ("see our **Return Policy** page
    for full details"). meta_description is added explicitly so it doesn't
    get truncated when the body is long.

    Body cap: 15000 chars per source. CMS policy pages routinely run
    several thousand words (return policy, warranty terms, FAQ pages);
    the old 4000-char cap chopped them mid-sentence and the LLM could
    only see the start. With Gemini's 1M-token context window plus
    typical 6 sources × ~3700 tokens per source = ~22000 tokens, this
    fits comfortably while letting full policy content reach the model.
    If you find the LLM losing the early sources to "lost in the
    middle", drop this back to 8000-10000.
    """
    parts: list[str] = [f"[{ct}] {title}"]

    heading = (s.get("content_heading") or "").strip()
    if heading and heading.lower() != str(title).lower():
        parts.append(f"Heading: {heading}")

    permalink = (s.get("permalink") or "").strip()
    if permalink:
        parts.append(f"URL: {permalink}")

    meta_desc = (s.get("meta_description") or "").strip()
    if meta_desc:
        parts.append(f"Summary: {meta_desc}")

    keywords = (s.get("meta_keywords") or "").strip()
    if keywords:
        parts.append(f"Keywords: {keywords}")

    body = s.get("content") or s.get("summary") or ""
    if body:
        parts.append("")
        parts.append(str(body))

    return "\n".join(parts)


def _format_product_source(s: dict, title: str) -> str:
    parts: list[str] = [f"[product] {title}"]
    sku = s.get("sku")
    if sku:
        parts.append(f"SKU: {sku}")
    type_id = s.get("type_id")
    if type_id:
        parts.append(f"Type: {type_id}")
    if s.get("stock_status"):
        parts.append(f"Stock: {s.get('stock_status')}")

    price = s.get("price")
    if price:
        currency = s.get("currency") or ""
        parts.append(f"Price: {price} {currency}".strip())

    categories = s.get("categories")
    if categories:
        parts.append(f"Categories: {categories}")

    brand = s.get("brand")
    if brand:
        parts.append(f"Brand: {brand}")

    # Attributes block — the LLM needs these to answer "what's the weight",
    # "what's it made of", "who makes it", and any custom-attribute question
    # the merchant exposed (battery life, dimensions, screen size etc.). The
    # values are stored as flat top-level keys on the payload (color="Red",
    # weight="1.5", etc.) by format_product. We iterate everything that
    # ISN'T a known structural field, isn't a filter-flag boolean, and is a
    # short scalar — that gives us the attribute set without a hard-coded
    # whitelist that would miss merchant-defined attributes.
    attribute_lines = _extract_attribute_lines(s)
    if attribute_lines:
        parts.append("Attributes:\n" + "\n".join(attribute_lines))

    # The critical bit: variant attributes + children so the LLM can
    # answer "what sizes does this come in".
    variant_attrs = s.get("variant_attributes") or {}
    if isinstance(variant_attrs, dict) and variant_attrs:
        lines = []
        for attr_code, values in variant_attrs.items():
            if not values:
                continue
            vals = values if isinstance(values, list) else [values]
            lines.append(f"  - {attr_code}: {', '.join(str(v) for v in vals)}")
        if lines:
            parts.append("Available variants:\n" + "\n".join(lines))

    children = s.get("children") or []
    if isinstance(children, list) and children:
        # Show up to 20 child SKUs with their attributes and stock.
        child_lines = []
        for ch in children[:20]:
            if not isinstance(ch, dict):
                continue
            attrs = ch.get("attributes") or {}
            attr_bits = ", ".join(f"{k}={v}" for k, v in attrs.items()) if attrs else ""
            stock = ch.get("stock_status") or ""
            price_c = ch.get("price") or ""
            child_lines.append(
                f"  - {ch.get('sku', '')}  {attr_bits}  {price_c}  {stock}".strip()
            )
        if child_lines:
            parts.append("Child SKUs:\n" + "\n".join(child_lines))

    desc = (s.get("description") or s.get("short_description") or s.get("summary") or "")
    if desc:
        parts.append(f"Description: {str(desc)[:1500]}")

    return "\n".join(parts)


# Structural fields on a product source — they're surfaced explicitly above,
# so we skip them when listing free-form attributes. Anything NOT in this set
# and not prefixed `attr_` / `cat_` is treated as a merchant attribute the
# LLM should see.
_KNOWN_PRODUCT_FIELDS = frozenset({
    "sku", "name", "title", "summary", "description", "short_description",
    "permalink", "image_url", "price", "currency", "currency_symbol",
    "regular_price", "sale_price", "on_sale", "average_rating",
    "categories", "category_paths", "category_ids", "tags",
    "stock_status", "type_id", "is_configurable", "has_variants",
    "variant_attributes", "children", "child_skus",
    "content_type", "entity_id", "client_id", "store_code", "embedded_text",
    "score", "product_id", "page_id", "post_id", "value", "label", "key",
    "identifier", "status", "meta_description", "updated_at",
    "brand", "gender",   # already surfaced explicitly
})


def _extract_attribute_lines(s: dict) -> list[str]:
    """Pull merchant attributes (weight, material, custom fields) out of the
    payload and render them as a bulleted list. Skips structural fields,
    filter booleans, and anything that isn't a short scalar so we don't dump
    JSON blobs or 2000-char descriptions into the prompt twice."""
    lines: list[str] = []
    for key, value in s.items():
        if key in _KNOWN_PRODUCT_FIELDS:
            continue
        if key.startswith("attr_") or key.startswith("cat_"):
            continue
        if value in (None, "", [], {}):
            continue
        # Only short scalars — long text is the description path.
        if isinstance(value, (int, float)):
            text = str(value)
        elif isinstance(value, str):
            text = value.strip()
            if not text or len(text) > 200:
                continue
        else:
            continue
        label = key.replace("_", " ").strip().title()
        lines.append(f"  - {label}: {text}")
    return lines


def _extract_text(content: Any) -> str:
    """LangChain's AIMessage.content can be a plain string OR a list of content
    blocks (Gemini returns the list shape, e.g. [{'type': 'text', 'text': '...'}]).
    Normalize both to a flat string so downstream code can always .strip() etc."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # {'type': 'text', 'text': '...'} or {'text': '...'}
                text = block.get("text") or block.get("content") or ""
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def _slug(value: str) -> str:
    import re as _re

    s = str(value or "").strip().lower()
    s = _re.sub(r"%", " percent", s)
    s = _re.sub(r"[^a-z0-9]+", "_", s)
    s = _re.sub(r"_+", "_", s)
    return s.strip("_")


def _apply_sort(hits: list[dict], sort_by: str, sort_order: str) -> list[dict]:
    """Apply a customer-requested sort to the (already-filtered) candidate
    pool. Handles missing / non-numeric payload fields gracefully — items
    with no sortable value drop to the end so they don't compete with
    properly-priced items for the top slots.

    sort_by   — "price" | "name" | "rating" | "newest". Anything else
                falls through (returns hits unchanged) rather than
                erroring, so a forward-compatible classifier output
                doesn't break retrieval if it sends a sort the backend
                hasn't learned yet.
    sort_order — "asc" | "desc". Defaults to ascending for unknown values.
    """
    reverse = sort_order == "desc"

    if sort_by == "price":
        def key(h: dict) -> tuple[int, float]:
            raw = h.get("price")
            try:
                v = float(raw)
                if v <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                # No usable price → push to the back (group=1) regardless
                # of asc/desc so the top of the list always shows items
                # with real prices.
                return (1, 0.0)
            return (0, v)
        return sorted(hits, key=key, reverse=reverse)

    if sort_by == "name":
        def key_name(h: dict) -> tuple[int, str]:
            name = str(h.get("name") or h.get("title") or "").strip().lower()
            return (0 if name else 1, name)
        return sorted(hits, key=key_name, reverse=reverse)

    if sort_by == "rating":
        def key_rating(h: dict) -> tuple[int, float]:
            try:
                v = float(h.get("average_rating") or 0)
            except (TypeError, ValueError):
                v = 0.0
            return (0 if v > 0 else 1, v)
        return sorted(hits, key=key_rating, reverse=reverse)

    if sort_by == "newest":
        def key_newest(h: dict) -> tuple[int, str]:
            ts = str(h.get("updated_at") or h.get("created_at") or "").strip()
            return (0 if ts else 1, ts)
        # "newest" naturally implies descending date order — flip when
        # asc is explicitly requested.
        return sorted(hits, key=key_newest, reverse=not (sort_order == "asc"))

    return hits


def _lookup_by_skus(
    client_id: str,
    domain: str,
    skus: list[str],
    store_code: Optional[str] = None,
) -> list[dict]:
    """Direct SKU lookup via Qdrant scroll with filter — no embedding required.

    When `store_code` is provided we also filter on it so multi-store
    deployments return the customer's store-view variant of each SKU.
    Without it (legacy single-store) we get whichever variant happens to
    score first, which for single-store collections is fine (only one
    variant exists anyway).
    """
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

    collection = get_collection_name(client_id, domain)
    try:
        existing = {c.name for c in qdrant.get_collections().collections}
        if collection not in existing:
            return []

        must = [FieldCondition(key="sku", match=MatchAny(any=skus))]
        if store_code:
            must.append(FieldCondition(key="store_code", match=MatchValue(value=store_code)))

        points, _ = qdrant.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=must),
            limit=max(10, len(skus) * 2),
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:
        logger.warning("SKU lookup failed: %s", exc)
        return []

    return [dict(p.payload or {}) for p in points]


def _llm_rerank(
    query: str,
    hits: list[dict],
    *,
    license_data: dict,
    llm_api_key: Optional[str],
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    db: Session,
) -> list[dict]:
    """Delegate to the existing llm_rerank_service already used by the /magento/search endpoint."""
    from backend.app.services.llm_rerank_service import llm_rerank_products

    return llm_rerank_products(
        query,
        hits,
        len(hits),
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        client_id=license_data["client_id"],
    ) or hits
