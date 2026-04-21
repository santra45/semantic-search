"""
Async Magento REST client — per-client, per-request.

Ported from magento_chatbot/services/magento_client.py with two critical changes:
  * Credentials are resolved per-client from MySQL (see magento_creds_service).
  * The admin token is minted via admin_token_service and cached in Redis per client_id.

No module-level singletons — each chat request creates one MagentoClient bound to the
tenant whose license key authenticated the request.
"""

from __future__ import annotations

import ssl
from typing import Any, Dict, List, Optional, Set
from urllib.parse import quote

import aiohttp

from backend.app.magento.chatbot.services import admin_token_service


class MagentoClient:
    """One instance per (client_id, store_code) for the lifetime of a request."""

    def __init__(
        self,
        *,
        client_id: str,
        base_url: str,
        api_version: str = "V1",
        verify_ssl: bool = True,
        store_code: str = "default",
        admin_token: Optional[str] = None,
        timeout_s: int = 30,
    ) -> None:
        self.client_id = client_id
        self.base_url = base_url.rstrip("/")
        self.api_version = api_version
        self.verify_ssl = verify_ssl
        self.store_code = store_code
        self.admin_token = admin_token
        self.timeout_s = timeout_s
        self._session: Optional[aiohttp.ClientSession] = None
        self._currency_symbol: Optional[str] = None
        self._media_url: Optional[str] = None

    # ── internals ────────────────────────────────────────────────────────────

    def _ssl_ctx(self) -> ssl.SSLContext:
        ctx = ssl.create_default_context()
        if not self.verify_ssl:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        return ctx

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                ssl=self._ssl_ctx(),
                limit=50,
                limit_per_host=10,
                keepalive_timeout=30,
                enable_cleanup_closed=True,
            )
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.timeout_s),
            )
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def _request(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict] = None,
        store_scoped: bool = True,
        suppress: Optional[Set[int]] = None,
    ) -> Optional[Any]:
        if store_scoped:
            url = f"{self.base_url}/rest/{self.store_code}/{self.api_version}/{endpoint}"
        else:
            url = f"{self.base_url}/rest/{self.api_version}/{endpoint}"

        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.admin_token:
            headers["Authorization"] = f"Bearer {self.admin_token}"

        session = self._get_session()
        try:
            async with session.request(method, url, json=data, headers=headers) as resp:
                if resp.status in (200, 201):
                    try:
                        return await resp.json()
                    except Exception:
                        return await resp.text()
                if resp.status == 204:
                    return {"success": True}
                if resp.status == 401:
                    # Token expired — invalidate so the next request re-mints.
                    admin_token_service.invalidate_token(self.client_id)
                if suppress and resp.status in suppress:
                    return None
                return None
        except Exception:
            return None

    # ── catalog (used only when the backend needs to fall back to live REST; the
    #    chatbot primarily answers from Qdrant) ────────────────────────────────

    async def get_product_by_sku(self, sku: str) -> Optional[Dict]:
        return await self._request(f"products/{quote(sku, safe='')}", suppress={404})

    async def get_products_by_skus(self, skus: List[str]) -> List[Dict]:
        """Used by product agent after Qdrant returns SKUs — fetches live stock + price."""
        if not skus:
            return []
        sku_filter = ",".join(quote(s, safe="") for s in skus)
        endpoint = (
            f"products?"
            f"searchCriteria[filterGroups][0][filters][0][field]=sku"
            f"&searchCriteria[filterGroups][0][filters][0][value]={sku_filter}"
            f"&searchCriteria[filterGroups][0][filters][0][conditionType]=in"
            f"&searchCriteria[pageSize]={len(skus)}"
        )
        result = await self._request(endpoint)
        return (result or {}).get("items", []) if isinstance(result, dict) else []

    async def get_categories(self) -> Optional[Dict]:
        return await self._request("categories")

    # ── cart ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_numeric_cart_id(qid: str) -> bool:
        return bool(qid) and str(qid).isdigit()

    @staticmethod
    def is_guest_cart_id(qid: str) -> bool:
        """Guest carts use masked IDs (non-numeric); logged-in carts use numeric quote_id."""
        return not MagentoClient._is_numeric_cart_id(qid)

    async def get_cart(self, cart_id: str, is_guest: bool) -> Optional[Dict]:
        if is_guest:
            endpoint = f"guest-carts/{cart_id}"
            base = await self._request(endpoint)
            if base is not None and not base.get("items"):
                items = await self._request(f"guest-carts/{cart_id}/items")
                if isinstance(items, list):
                    base["items"] = items
            return base

        if self._is_numeric_cart_id(cart_id):
            base = await self._request(f"carts/{cart_id}")
            if base is not None and not base.get("items"):
                items = await self._request(f"carts/{cart_id}/items")
                if isinstance(items, list):
                    base["items"] = items
            return base

        return None

    async def get_cart_totals(self, cart_id: str, is_guest: bool) -> Optional[Dict]:
        if is_guest:
            return await self._request(f"guest-carts/{cart_id}/totals")
        if self._is_numeric_cart_id(cart_id):
            return await self._request(f"carts/{cart_id}/totals")
        return None

    async def add_to_cart(self, cart_id: str, sku: str, qty: int, is_guest: bool) -> Dict:
        data = {"cartItem": {"sku": sku, "qty": qty, "quoteId": cart_id}}
        if is_guest:
            endpoint = f"guest-carts/{cart_id}/items"
        else:
            if not self._is_numeric_cart_id(cart_id):
                return {"success": False, "error_message": "Invalid quote ID."}
            endpoint = f"carts/{cart_id}/items"

        result = await self._request(endpoint, method="POST", data=data)
        if result and not (isinstance(result, dict) and result.get("error")):
            return {"success": True}
        return {"success": False, "error_message": "Magento refused the add-to-cart request."}

    async def update_cart_item(self, cart_id: str, item_id: int, qty: int, is_guest: bool) -> bool:
        data = {"cartItem": {"qty": qty, "item_id": item_id, "quoteId": cart_id}}
        if is_guest:
            endpoint = f"guest-carts/{cart_id}/items/{item_id}"
        else:
            if not self._is_numeric_cart_id(cart_id):
                return False
            endpoint = f"carts/{cart_id}/items/{item_id}"
        result = await self._request(endpoint, method="PUT", data=data)
        return result is not None

    async def remove_from_cart(self, cart_id: str, item_id: int, is_guest: bool) -> bool:
        if is_guest:
            endpoint = f"guest-carts/{cart_id}/items/{item_id}"
        else:
            if not self._is_numeric_cart_id(cart_id):
                return False
            endpoint = f"carts/{cart_id}/items/{item_id}"
        result = await self._request(endpoint, method="DELETE")
        return result is not None

    async def get_currency_symbol(self) -> str:
        if self._currency_symbol:
            return self._currency_symbol
        symbols = {
            "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CNY": "¥",
            "INR": "₹", "AUD": "A$", "CAD": "C$", "AED": "AED ",
        }
        try:
            config = await self._request("store/storeConfigs")
            if config and isinstance(config, list) and len(config) > 0:
                code = config[0].get("base_currency_code", "USD")
                self._currency_symbol = symbols.get(code, f"{code} ")
                self._media_url = config[0].get("base_media_url", f"{self.base_url}/media/")
                return self._currency_symbol
        except Exception:
            pass
        self._currency_symbol = "$"
        return self._currency_symbol

    def get_checkout_url(self) -> str:
        return f"{self.base_url}/checkout/"

    # ── orders ───────────────────────────────────────────────────────────────

    async def get_customer_orders(self, customer_id: int, page_size: int = 10) -> List[Dict]:
        endpoint = (
            f"orders?searchCriteria[filter_groups][0][filters][0][field]=customer_id"
            f"&searchCriteria[filter_groups][0][filters][0][value]={customer_id}"
            f"&searchCriteria[pageSize]={page_size}"
            f"&searchCriteria[sortOrders][0][field]=created_at"
            f"&searchCriteria[sortOrders][0][direction]=DESC"
        )
        result = await self._request(endpoint)
        return (result or {}).get("items", []) if isinstance(result, dict) else []

    async def get_order_by_increment_id(self, increment_id: str, customer_id: int) -> Optional[Dict]:
        endpoint = (
            f"orders?searchCriteria[filter_groups][0][filters][0][field]=increment_id"
            f"&searchCriteria[filter_groups][0][filters][0][value]={increment_id}"
            f"&searchCriteria[filter_groups][1][filters][0][field]=customer_id"
            f"&searchCriteria[filter_groups][1][filters][0][value]={customer_id}"
            f"&searchCriteria[pageSize]=1"
        )
        result = await self._request(endpoint)
        items = (result or {}).get("items", []) if isinstance(result, dict) else []
        return items[0] if items else None

    # ── customer ─────────────────────────────────────────────────────────────

    async def get_customer_by_id(self, customer_id: int) -> Optional[Dict]:
        return await self._request(f"customers/{customer_id}")
