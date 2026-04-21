"""Per-request context threaded into every tool and node.

Every chat request builds a RequestContext, hands it to the orchestrator, and the
orchestrator hands it to tools via closure. This keeps per-tenant state on the
stack instead of module-level singletons — critical for multi-tenancy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from backend.app.magento.chatbot.services.magento_client import MagentoClient


@dataclass
class RequestContext:
    # Tenant
    client_id: str
    domain: str
    license_key: str

    # Shopper
    store_code: str
    customer_id: Optional[int] = None
    is_customer_login: bool = False
    guest_session_id: Optional[str] = None
    quote_id: Optional[str] = None

    # LLM preferences (from Magento module config)
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None  # already decrypted

    # Attached services (injected by the orchestrator / router)
    magento_client: Optional[MagentoClient] = None

    # Scratch space for tools / nodes to share state during a single chat turn
    scratch: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_customer_id(self) -> int:
        if self.is_customer_login and self.customer_id and int(self.customer_id) > 0:
            return int(self.customer_id)
        return 0
