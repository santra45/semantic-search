"""
Semantic intent classifier — ports magento_chatbot/utils/intent_router.py.

Uses Gemini embeddings with cosine similarity over a catalog of intent utterances.
The utterance embeddings are pre-computed once at first use and cached in-process.

Per-request LLM API keys override the backend's Gemini key if provided — same
decryption path the embedder uses for search.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
from google import genai
from google.genai import types

from backend.app.config import GEMINI_API_KEY
from backend.app.magento.chatbot.services.config import INTENT_ROUTER_ENABLED

logger = logging.getLogger(__name__)

INTENT_DEFINITIONS: dict[str, dict] = {
    "get_customer_email":     {"agent": "profile", "utterances": ["my email", "email", "what is my email", "email address", "email id", "my emil", "show my email"]},
    "get_customer_name":      {"agent": "profile", "utterances": ["my name", "name", "what is my name", "show my name", "who am i"]},
    "get_customer_phone":     {"agent": "profile", "utterances": ["my phone", "phone", "phone number", "phone no", "my number", "contact number", "mobile number"]},
    "get_customer_addresses": {"agent": "profile", "utterances": ["my address", "address", "billing address", "shipping address", "delivery address"]},
    "get_customer_profile":   {"agent": "profile", "utterances": ["my profile", "profile", "my details", "my account", "account details", "personal details"]},
    "view_cart":              {"agent": "cart", "utterances": ["my cart", "cart", "view cart", "show cart", "what is in my cart", "shopping cart", "cart items"]},
    "remove_from_cart":       {"agent": "cart", "utterances": ["remove from cart", "delete from cart", "take out of cart", "remove item from my cart"]},
    "clear_cart":             {"agent": "cart", "utterances": ["clear cart", "empty cart", "clear my cart", "empty my cart", "remove all items"]},
    "get_customer_orders":    {"agent": "order", "utterances": ["my orders", "orders", "order history", "show my orders", "list my orders"]},
    "get_last_order":         {"agent": "order", "utterances": ["last order", "my last order", "recent order", "latest order", "previous order"]},
    "get_order_status":       {"agent": "order", "utterances": ["order status", "my order status", "check order status", "status of my order"]},
    "get_shipping_status":    {"agent": "order", "utterances": ["shipping status", "tracking", "track order", "track my order", "delivery status", "tracking number"]},
    "find_products":          {"agent": "product", "utterances": [
        "search products", "find products", "show products", "list products",
        "red products", "blue products", "products under 50", "products above 100",
        "product details", "tell me about product", "show product details",
        "shirt products", "jacket products", "dress products", "shoes",
    ]},
}


class IntentRouter:
    _instance: Optional["IntentRouter"] = None
    SIMILARITY_THRESHOLD = 0.65

    def __new__(cls) -> "IntentRouter":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._client: Optional[genai.Client] = None
        self._query_cache: dict[str, list[float]] = {}
        self._intent_embeddings: dict[str, list[list[float]]] = {}
        self._setup_done = False
        self._setup_failed = False
        self._initialized = True

    def _ensure_setup(self, api_key: Optional[str]) -> bool:
        if not INTENT_ROUTER_ENABLED:
            return False
        if self._setup_done:
            return True
        if self._setup_failed:
            return False

        key = api_key or GEMINI_API_KEY
        if not key:
            self._setup_failed = True
            return False

        try:
            self._client = genai.Client(api_key=key)
            utterances: list[str] = []
            mapping: list[str] = []
            for name, data in INTENT_DEFINITIONS.items():
                for utterance in data["utterances"]:
                    utterances.append(utterance)
                    mapping.append(name)

            all_embeddings: list[list[float]] = []
            batch_size = 10
            for i in range(0, len(utterances), batch_size):
                if i > 0:
                    time.sleep(10)  # respect Gemini RPM limits during bootstrap
                batch = utterances[i : i + batch_size]
                res = self._client.models.embed_content(
                    model="models/gemini-embedding-001",
                    contents=batch,
                    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
                )
                all_embeddings.extend(e.values for e in res.embeddings)

            for name in INTENT_DEFINITIONS:
                self._intent_embeddings[name] = []
            for idx, emb in enumerate(all_embeddings):
                self._intent_embeddings[mapping[idx]].append(emb)

            self._setup_done = True
            return True
        except Exception as exc:
            logger.warning("IntentRouter setup failed: %s", exc)
            self._setup_failed = True
            return False

    def _embed_query(self, query: str) -> Optional[list[float]]:
        if query in self._query_cache:
            return self._query_cache[query]
        if not self._client:
            return None
        try:
            res = self._client.models.embed_content(
                model="models/gemini-embedding-001",
                contents=[query],
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
            )
            vec = res.embeddings[0].values
            self._query_cache[query] = vec
            return vec
        except Exception as exc:
            logger.debug("Intent query embedding failed: %s", exc)
            return None

    def classify(
        self,
        query: str,
        *,
        llm_api_key: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """Return (tool_name, agent_type) or (None, None) below threshold."""
        if not query.strip():
            return None, None
        if not self._ensure_setup(llm_api_key):
            return None, None

        q_vec = self._embed_query(query.lower().strip())
        if not q_vec:
            return None, None

        q_np = np.array(q_vec)
        q_norm = np.linalg.norm(q_np)
        if q_norm == 0:
            return None, None

        best_score = 0.0
        best_intent: Optional[str] = None
        for intent, embs in self._intent_embeddings.items():
            if not embs:
                continue
            mat = np.array(embs)
            dots = mat @ q_np
            norms = np.linalg.norm(mat, axis=1) * q_norm
            scores = dots / norms
            peak = float(np.max(scores))
            if peak > best_score:
                best_score = peak
                best_intent = intent

        if best_intent and best_score >= self.SIMILARITY_THRESHOLD:
            return best_intent, INTENT_DEFINITIONS[best_intent]["agent"]
        return None, None


def get_intent_router() -> IntentRouter:
    return IntentRouter()
