"""
Multi-agent chatbot HTTP endpoints.

  POST   /api/magento/chatbot/agent/session/start
  POST   /api/magento/chatbot/agent/message
  POST   /api/magento/chatbot/agent/link-guest
  POST   /api/magento/chatbot/agent/reset
  GET    /api/magento/chatbot/agent/history
"""

from __future__ import annotations

import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.services.database import get_db
from backend.app.services.token_usage_service import TokenUsageTracker

from backend.app.magento.chatbot.agents.checkpointer import (
    get_saver,
    migrate_thread_history,
)
from backend.app.magento.chatbot.agents.orchestrator import AgentOrchestrator
from backend.app.magento.chatbot.services import chat_history_service
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
    customer_id: Optional[str] = None
    is_customer_login: bool = False
    guest_session_id: Optional[str] = None
    quote_id: Optional[str] = None


class MessageRequest(BaseModel):
    license_key: Optional[str] = None
    message: str
    store_code: str = "default"
    customer_id: Optional[str] = None
    is_customer_login: bool = False
    guest_session_id: Optional[str] = None
    quote_id: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_api_key_encrypted: Optional[str] = None


class LinkGuestRequest(BaseModel):
    license_key: Optional[str] = None
    store_code: str = "default"
    guest_session_id: str
    customer_id: str


class ResetRequest(BaseModel):
    license_key: Optional[str] = None
    store_code: str = "default"
    customer_id: Optional[str] = None
    guest_session_id: Optional[str] = None


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

    thread_id = chat_history_service.build_thread_id(
        client_id=license_data["client_id"],
        is_customer_login=req.is_customer_login,
        customer_id=req.customer_id,
        guest_session_id=req.guest_session_id,
        store_code=req.store_code,
    )

    session = chat_history_service.start_or_update_session(
        db,
        client_id=license_data["client_id"],
        thread_id=thread_id,
        store_code=req.store_code,
        customer_id=req.customer_id,
        guest_session_id=req.guest_session_id,
        is_customer_login=req.is_customer_login,
        quote_id=req.quote_id,
    )

    messages = chat_history_service.get_session_messages(
        db, client_id=license_data["client_id"], session_id=session["session_id"], limit=50
    )
    return {
        "session": session,
        "history": messages,
        "thread_id": thread_id,
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

    thread_id = chat_history_service.build_thread_id(
        client_id=license_data["client_id"],
        is_customer_login=req.is_customer_login,
        customer_id=req.customer_id,
        guest_session_id=req.guest_session_id,
        store_code=req.store_code,
    )

    session = chat_history_service.start_or_update_session(
        db,
        client_id=license_data["client_id"],
        thread_id=thread_id,
        store_code=req.store_code,
        customer_id=req.customer_id,
        guest_session_id=req.guest_session_id,
        is_customer_login=req.is_customer_login,
        quote_id=req.quote_id,
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

    # Persist the user message up front so it survives an orchestrator failure.
    chat_history_service.append_message(
        db,
        session_id=session["session_id"],
        client_id=license_data["client_id"],
        thread_id=thread_id,
        role="user",
        content=req.message.strip(),
    )

    # Load short history (last 8 user/assistant turns) to seed the graph when
    # the Postgres checkpointer is unavailable (dev/minimal deployments).
    prior_msgs = chat_history_service.get_session_messages(
        db, client_id=license_data["client_id"], session_id=session["session_id"], limit=30
    )
    history: list[Any] = []
    for m in prior_msgs[:-1]:  # exclude the user message we just inserted
        if m["role"] == "user" and m["content"]:
            history.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant" and m["content"]:
            history.append(AIMessage(content=m["content"]))
    history = history[-16:]

    saver = await get_saver()
    orchestrator = AgentOrchestrator(ctx)

    t0 = time.time()
    try:
        result = await orchestrator.chat(
            req.message.strip(),
            thread_id=thread_id,
            history=history if saver is None else None,  # checkpointer restores history itself
            checkpointer=saver,
        )
    finally:
        if ctx.magento_client is not None:
            await ctx.magento_client.close()

    elapsed_ms = int((time.time() - t0) * 1000)
    usage = result.get("token_usage", {}) or {}
    current_agent = result.get("current_agent") or "product"

    # Persist the assistant reply + token usage.
    chat_history_service.append_message(
        db,
        session_id=session["session_id"],
        client_id=license_data["client_id"],
        thread_id=thread_id,
        role="assistant",
        agent_name=current_agent,
        content=result["content"],
        input_tokens=int(usage.get("input", 0) or 0),
        output_tokens=int(usage.get("output", 0) or 0),
        total_cost=0.0,
        response_time_ms=elapsed_ms,
        tool_name=result.get("direct_tool"),
    )

    # Token-usage ledger (dashboard source of truth)
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
        "session_id": session["session_id"],
        "thread_id": thread_id,
        "agent": current_agent,
        "answer": result["content"],
        "direct_tool": result.get("direct_tool"),
        "usage": usage,
        "response_time_ms": elapsed_ms,
    }


@router.post("/magento/chatbot/agent/link-guest")
async def agent_link_guest(
    req: LinkGuestRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    """Merge a guest conversation into the customer's thread on first login."""
    license_data = authorize_request(
        request=request, db=db,
        authorization=authorization, x_api_key=x_api_key,
        request_license=req.license_key,
    )

    guest_thread = chat_history_service.build_thread_id(
        client_id=license_data["client_id"],
        is_customer_login=False,
        customer_id=None,
        guest_session_id=req.guest_session_id,
        store_code=req.store_code,
    )
    customer_thread = chat_history_service.build_thread_id(
        client_id=license_data["client_id"],
        is_customer_login=True,
        customer_id=req.customer_id,
        guest_session_id=None,
        store_code=req.store_code,
    )

    rows_updated = chat_history_service.merge_guest_into_customer(
        db,
        client_id=license_data["client_id"],
        guest_thread_id=guest_thread,
        customer_thread_id=customer_thread,
        customer_id=req.customer_id,
    )
    await migrate_thread_history(guest_thread, customer_thread)

    return {
        "migrated_rows": rows_updated,
        "from_thread": guest_thread,
        "to_thread": customer_thread,
    }


@router.post("/magento/chatbot/agent/reset")
async def agent_reset(
    req: ResetRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    """Mark a session closed. Checkpointer state is kept so the admin can still see history."""
    license_data = authorize_request(
        request=request, db=db,
        authorization=authorization, x_api_key=x_api_key,
        request_license=req.license_key,
    )
    thread_id = chat_history_service.build_thread_id(
        client_id=license_data["client_id"],
        is_customer_login=bool(req.customer_id),
        customer_id=req.customer_id,
        guest_session_id=req.guest_session_id,
        store_code=req.store_code,
    )
    from sqlalchemy import text
    closed = db.execute(
        text(
            """
            UPDATE agent_chat_sessions SET status = 'closed'
            WHERE client_id = :client_id AND thread_id = :thread_id AND status = 'active'
            """
        ),
        {"client_id": license_data["client_id"], "thread_id": thread_id},
    )
    db.commit()
    return {"closed": int(closed.rowcount or 0), "thread_id": thread_id}


@router.get("/magento/chatbot/agent/history")
async def agent_history(
    request: Request,
    session_id: Optional[str] = Query(None),
    thread_id: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    license_data = authorize_request(
        request=request, db=db,
        authorization=authorization, x_api_key=x_api_key,
        request_license=None,
    )

    if session_id:
        session = chat_history_service.get_session(
            db, client_id=license_data["client_id"], session_id=session_id
        )
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        messages = chat_history_service.get_session_messages(
            db, client_id=license_data["client_id"], session_id=session_id
        )
        return {"session": session, "messages": messages}

    if thread_id:
        from sqlalchemy import text
        row = db.execute(
            text(
                "SELECT * FROM agent_chat_sessions WHERE client_id = :client_id AND thread_id = :thread_id ORDER BY last_activity_at DESC LIMIT 1"
            ),
            {"client_id": license_data["client_id"], "thread_id": thread_id},
        ).fetchone()
        if not row:
            return {"session": None, "messages": []}
        session = chat_history_service.get_session(
            db, client_id=license_data["client_id"], session_id=row.id
        )
        messages = chat_history_service.get_session_messages(
            db, client_id=license_data["client_id"], session_id=row.id
        )
        return {"session": session, "messages": messages}

    raise HTTPException(status_code=400, detail="session_id or thread_id is required")
