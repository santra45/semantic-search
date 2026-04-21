"""
Mint and cache a Magento admin integration token per tenant.

Redis key: magento_admin_token:{client_id}
TTL: ADMIN_TOKEN_TTL_SECONDS (default 4h — Magento tokens default to 4h lifetime).
"""

from __future__ import annotations

import asyncio
from typing import Optional

import aiohttp
import ssl

from backend.app.services.cache_service import r as redis_client
from backend.app.magento.chatbot.services.config import ADMIN_TOKEN_TTL_SECONDS


_TOKEN_KEY_PREFIX = "magento_admin_token:"


def _cache_key(client_id: str) -> str:
    return f"{_TOKEN_KEY_PREFIX}{client_id}"


def get_cached_token(client_id: str) -> Optional[str]:
    try:
        return redis_client.get(_cache_key(client_id))
    except Exception:
        return None


def invalidate_token(client_id: str) -> None:
    try:
        redis_client.delete(_cache_key(client_id))
    except Exception:
        pass


def _ssl_ctx(verify: bool) -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    if not verify:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


async def mint_token(
    *,
    client_id: str,
    base_url: str,
    username: str,
    password: str,
    api_version: str = "V1",
    verify_ssl: bool = True,
    timeout_s: int = 30,
) -> Optional[str]:
    """Hit /rest/{V1}/integration/admin/token and cache the returned token."""
    url = f"{base_url.rstrip('/')}/rest/{api_version}/integration/admin/token"
    payload = {"username": username, "password": password}

    connector = aiohttp.TCPConnector(ssl=_ssl_ctx(verify_ssl))
    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout_s),
            ) as resp:
                if resp.status != 200:
                    return None
                raw = (await resp.text()).strip().strip('"')
                if not raw:
                    return None
                try:
                    redis_client.setex(_cache_key(client_id), ADMIN_TOKEN_TTL_SECONDS, raw)
                except Exception:
                    pass
                return raw
    except Exception:
        return None


async def get_or_mint_token(
    *,
    client_id: str,
    creds: dict,
) -> Optional[str]:
    """Return cached token or mint a new one. `creds` comes from magento_creds_service.get_credentials."""
    cached = get_cached_token(client_id)
    if cached:
        return cached

    return await mint_token(
        client_id=client_id,
        base_url=creds["base_url"],
        username=creds["username"],
        password=creds["password"],
        api_version=creds.get("api_version", "V1"),
        verify_ssl=bool(creds.get("verify_ssl", True)),
    )


# Sync wrapper for places that can't await (e.g. startup probes)
def mint_token_sync(*args, **kwargs) -> Optional[str]:
    return asyncio.run(mint_token(*args, **kwargs))
