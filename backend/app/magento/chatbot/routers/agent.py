"""
Stateless multi-agent chatbot endpoints.

  POST   /api/magento/chatbot/agent/session/start   (lightweight bootstrap — no persistence)
  POST   /api/magento/chatbot/agent/message         (runs ONE agent turn, returns the reply)

History is owned entirely by the Magento module. The storefront POSTs the recent
conversation on every message as `history: [{role, content}, ...]`, we seed the
graph with it, and we return the assistant turn without persisting anything on
this server. The only server-side trail is the anonymized token-usage ledger
(`token_usage_tracking`) which is keyed by client_id for billing and holds no
customer identifiers.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.services.database import get_db
from backend.app.services.token_usage_service import TokenUsageTracker

from backend.app.magento.chatbot.agents.orchestrator import AgentOrchestrator
from backend.app.magento.chatbot.routers.common import (
    authorize_request,
    build_request_context,
    maybe_persist_magento_creds,
)

router = APIRouter()


# ── Schemas ──────────────────────────────────────────────────────────────────


class SessionStartRequest(BaseModel):
    license_key: Optional[str] = None
    store_code: str = "default"


class HistoryTurn(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class MessageRequest(BaseModel):
    license_key: Optional[str] = None
    message: str
    store_code: str = "default"
    customer_id: Optional[str] = None
    is_customer_login: bool = False
    guest_session_id: Optional[str] = None
    quote_id: Optional[str] = None
    thread_id: Optional[str] = None  # opaque id from the module — used only for LangGraph run config
    history: list[HistoryTurn] = Field(default_factory=list)
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_api_key_encrypted: Optional[str] = None


# ── Routes ───────────────────────────────────────────────────────────────────


@router.post("/magento/chatbot/agent/session/start")
async def agent_session_start(
    req: SessionStartRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_magento_creds: Optional[str] = Header(None, alias="X-Magento-Admin-Creds-Encrypted"),
    db: Session = Depends(get_db),
):
    """Lightweight handshake. Validates auth + persists admin creds if the header is fresh.
    No session is created — the Magento module owns all session state."""
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

    return {
        "ok": True,
        "client_id": license_data["client_id"],
        "store_code": req.store_code,
    }


@router.post("/magento/chatbot/agent/message")
async def agent_message(
    req: MessageRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_llm_api_key_encrypted: Optional[str] = Header(None, alias="X-LLM-API-Key-Encrypted"),
    x_magento_creds: Optional[str] = Header(None, alias="X-Magento-Admin-Creds-Encrypted"),
    db: Session = Depends(get_db),
):
    """One agent turn. Stateless — history is passed in and never stored."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

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

    ctx = await build_request_context(
        db=db,
        license_data=license_data,
        store_code=req.store_code,
        customer_id=req.customer_id,
        is_customer_login=req.is_customer_login,
        guest_session_id=req.guest_session_id,
        quote_id=req.quote_id,
        llm_provider=req.llm_provider,
        llm_model=req.llm_model,
        llm_api_key_encrypted=x_llm_api_key_encrypted,
    )

    # Translate client-supplied history into LangChain messages.
    history_messages: list[Any] = []
    for turn in req.history[-16:]:  # cap — don't let the client blow up the prompt
        content = (turn.content or "").strip()
        if not content:
            continue
        if turn.role == "user":
            history_messages.append(HumanMessage(content=content))
        elif turn.role == "assistant":
            history_messages.append(AIMessage(content=content))
        # silently ignore tool/system roles — storefront shouldn't be replaying them

    # Thread id is purely a per-request tag for LangGraph's config (graph metrics, tracing).
    # No Postgres checkpointer in the privacy-preserving mode.
    thread_id = (req.thread_id or f"{license_data['client_id']}::ephemeral::{int(time.time()*1000)}")

    orchestrator = AgentOrchestrator(ctx)

    t0 = time.time()
    try:
        result = await orchestrator.chat(
            req.message.strip(),
            thread_id=thread_id,
            history=history_messages,
            checkpointer=None,  # stateless
        )
    finally:
        if ctx.magento_client is not None:
            await ctx.magento_client.close()

    elapsed_ms = int((time.time() - t0) * 1000)
    usage = result.get("token_usage", {}) or {}
    current_agent = result.get("current_agent") or "product"

    # Only the anonymized billing ledger survives on this server.
    try:
        TokenUsageTracker(db).create_usage_record(
            client_id=license_data["client_id"],
            query_type="chat_answer",
            llm_provider=req.llm_provider or "google",
            llm_model=req.llm_model or "gemini-2.0-flash-lite",
            input_tokens=int(usage.get("input", 0) or 0),
            output_tokens=int(usage.get("output", 0) or 0),
            request_text_length=len(req.message),
            response_text_length=len(result["content"] or ""),
        )
    except Exception:
        pass

    return {
        "thread_id": thread_id,
        "agent": current_agent,
        "answer": result["content"],
        "direct_tool": result.get("direct_tool"),
        "usage": usage,
        "response_time_ms": elapsed_ms,
    }
