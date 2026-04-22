"""
Deprecated — the multi-agent LangGraph stack has been retired.

Chat routing and agent dispatch now live in the Magento module
(`Czargroup\AIChatbot\Model\Agent\Router`). This backend only exposes the
Qdrant-backed retrieve endpoints (see routers/retrieve.py).

Router kept as an empty placeholder so stale imports don't break.
"""

from fastapi import APIRouter

router = APIRouter()
