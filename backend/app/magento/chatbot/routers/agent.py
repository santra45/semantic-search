"""
Tool-calling intent router endpoint (Phase 3.1).

  POST /api/magento/chatbot/agent/tool-call

Replaces the legacy heuristic+/classify pair on the Magento Router's
"live" mode path. One LLM call with twelve tools registered (each
tool corresponds 1:1 with a Magento agent); the response carries the
picked tool name + extracted arguments. The Magento side maps tool →
intent and dispatches the same agent it would have dispatched before
— this endpoint replaces *classification only*, not execution.

Failure posture:
    Never raises 5xx to the caller. The orchestrator inside
    tool_call_classifier.select_tool() returns a structured fallback
    ({tool_name: "general_chat", confidence: 0.0, error: "..."}) on
    any internal failure so the Magento Router can either dispatch
    that or — preferred — fall back to its regex heuristic.

Cost tracking:
    The orchestrator writes a `chat_tool_call` row to
    `token_usage_tracking` per call. Admin dashboard reads this
    bucket alongside `chat_intent` for the shadow/live A/B view.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from backend.app.magento.chatbot.agents.tool_call_classifier import select_tool
from backend.app.magento.chatbot.routers.common import (
    authorize_request,
    decrypt_llm_key,
)
from backend.app.services.database import get_db

router = APIRouter()


class _CustomerContext(BaseModel):
    """Slim view of the customer state the LLM needs for routing.

    We deliberately do NOT pass customer_id, email, or any PII — the
    LLM never needs to know who the customer is, only the auth posture
    so it doesn't pick `view_profile` for a brand-new guest with empty
    context. The Magento side enforces auth at agent.canHandle() time
    regardless of what the LLM picks.
    """
    is_logged_in: bool = False
    customer_name: Optional[str] = None
    store_code: Optional[str] = None


class ToolCallRequest(BaseModel):
    license_key: Optional[str] = None
    llm_api_key_encrypted: Optional[str] = None

    query: str = Field(min_length=1, max_length=4000)

    # Last few conversation turns for follow-up resolution ("the cheaper
    # one"). Same shape /retrieve/answer accepts — capped at 6 turns
    # there, capped at 12 here so the LLM has slightly more context
    # for routing decisions (vs answer generation).
    conversation_history: list[dict[str, str]] = Field(default_factory=list)

    customer_context: _CustomerContext = Field(default_factory=_CustomerContext)

    # Provider/model overrides — admin can pick a cheap fast model for
    # routing (e.g. gemini-2.5-flash-lite) and keep a premium model for
    # RAG answers. Both fall back to "google" / "gemini-2.5-flash"
    # when omitted.
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None

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
            if role in ("user", "assistant") and content:
                out.append({"role": role, "content": content})
        # 12-turn cap — orchestrator further trims to last 6 inside the
        # prompt build. Capping here too prevents a misbehaving caller
        # from inflating the request body.
        return out[-12:]


@router.post("/magento/chatbot/agent/tool-call")
def tool_call(
    req: ToolCallRequest,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_llm_api_key_encrypted: Optional[str] = Header(None, alias="X-LLM-API-Key-Encrypted"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    license_data = authorize_request(
        request=request, db=db,
        authorization=authorization, x_api_key=x_api_key,
        request_license=req.license_key,
    )

    api_key = decrypt_llm_key(
        x_llm_api_key_encrypted or req.llm_api_key_encrypted,
        license_data["license_key"],
    )

    # Run the orchestrator. It NEVER raises — always returns a structured
    # result (with `error` set on failure). We pass the response straight
    # through so the Magento side can inspect `tool_name`, `arguments`,
    # `confidence`, `usage`, `error` and decide whether to dispatch or
    # fall back to its heuristic layer.
    result = select_tool(
        query=req.query.strip(),
        conversation_history=req.conversation_history,
        customer_context=req.customer_context.model_dump(),
        provider=(req.llm_provider or "google"),
        model=(req.llm_model or "gemini-2.5-flash"),
        api_key=api_key,
        client_id=license_data["client_id"],
    )

    return result
