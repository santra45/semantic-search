"""
LLM classification endpoint.

  POST /api/magento/chatbot/classify

Generic single-shot LLM completion proxy for the Magento chatbot's
intent classifier. Replaces the in-PHP LLMClient that called Gemini/
OpenAI/Anthropic directly — moving the call here gives us:

  * Token + cost tracking on the central billing ledger
  * Encrypted LLM API key transit (X-LLM-API-Key-Encrypted header)
  * License-key authorization on every classification call

Caller passes the prompt verbatim; this endpoint does NOT build or modify
prompts. Provider/model fall back to sensible defaults when omitted.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.magento.chatbot.routers.common import (
    authorize_request,
    decrypt_llm_key,
)
from backend.app.services.database import get_db
from backend.app.services.llm_completion_service import complete

router = APIRouter()

# Restrict the externally-supplied query_type to types we actually use
# from the Magento chatbot. Stops a misbehaving caller from polluting
# unrelated buckets in the usage dashboard.
_ALLOWED_QUERY_TYPES = {"chat_intent", "chat_rewrite"}


class ClassifyRequest(BaseModel):
    license_key: Optional[str] = None
    llm_api_key_encrypted: Optional[str] = None
    prompt: str
    json_mode: bool = False
    max_tokens: int = Field(default=512, ge=1, le=8192)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    provider: Optional[str] = None
    model: Optional[str] = None
    query_type: str = "chat_intent"


@router.post("/magento/chatbot/classify")
def classify(
    req: ClassifyRequest,
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

    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    api_key = decrypt_llm_key(
        x_llm_api_key_encrypted or req.llm_api_key_encrypted,
        license_data["license_key"],
    )

    # Whitelist query_type so any caller can't slot tokens into arbitrary
    # buckets. Anything outside the set is silently coerced to chat_intent.
    query_type = req.query_type if req.query_type in _ALLOWED_QUERY_TYPES else "chat_intent"

    try:
        text, usage = complete(
            req.prompt,
            json_mode=req.json_mode,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            provider=(req.provider or "gemini"),
            model=req.model,
            api_key=api_key,
            client_id=license_data["client_id"],
            query_type=query_type,
        )
    except ValueError as ve:
        # Missing API key, unknown provider, etc. — caller-fixable.
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        # Provider error / network blip — surface as 502 so callers can
        # fall back to heuristic-only routing.
        raise HTTPException(status_code=502, detail=f"LLM provider error: {exc}")

    # `usage` carries input/output tokens, cost, provider, and model so the
    # Magento side can persist a per-message billing row (matches the shape
    # /retrieve/answer returns).
    return {"text": text, "usage": usage}
