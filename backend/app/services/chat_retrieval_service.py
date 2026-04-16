from __future__ import annotations

from typing import Any

from backend.app.services.embedder import embed_query
from backend.app.services.product_service import strip_html
from backend.app.services.qdrant_service import search_content

DEFAULT_ALLOWED_TYPES = [
    "product",
    "cms_page",
    "cms_block",
    "widget",
    "review",
    "policy",
    "faq",
    "store_config",
]


def _content_title(item: dict[str, Any]) -> str:
    return (
        item.get("title")
        or item.get("name")
        or item.get("question")
        or item.get("identifier")
        or f"{item.get('content_type', 'content').replace('_', ' ').title()} #{item.get('entity_id', '')}"
    )


def _content_excerpt(item: dict[str, Any]) -> str:
    excerpt = (
        item.get("summary")
        or item.get("content")
        or item.get("description")
        or item.get("short_description")
        or ""
    )
    return strip_html(str(excerpt)).strip()[:280]


def _source_card(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "content_type": item.get("content_type", "product"),
        "entity_id": item.get("entity_id", ""),
        "title": _content_title(item),
        "permalink": item.get("permalink", ""),
        "excerpt": _content_excerpt(item),
        "score": float(item.get("score", 0)),
    }


def _prioritize_results(results: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    products = [item for item in results if item.get("content_type") == "product"]
    support = [item for item in results if item.get("content_type") != "product"]

    prioritized = []
    prioritized.extend(products[: min(3, limit)])
    prioritized.extend(support[: max(0, min(3, limit - len(prioritized)))])

    for item in results:
        if item in prioritized:
            continue
        prioritized.append(item)
        if len(prioritized) >= limit:
            break

    return prioritized[:limit]


def _grounding_confidence(results: list[dict[str, Any]]) -> tuple[bool, float]:
    if not results:
        return False, 0.0

    top_score = float(results[0].get("score", 0))
    supporting = len([item for item in results if float(item.get("score", 0)) >= 0.35])
    grounded = top_score >= 0.42 or supporting >= 2
    confidence = min(1.0, max(top_score, supporting / 4))
    return grounded, round(confidence, 4)


def retrieve_evidence(
    client_id: str,
    domain: str,
    message: str,
    embedding_api_key: str | None = None,
    allowed_content_types: list[str] | None = None,
    limit: int = 6,
) -> dict[str, Any]:
    query_vector = embed_query(
        message,
        api_key=embedding_api_key,
        client_id=client_id,
        query_type="chat_context",
    )

    raw_results = search_content(
        client_id=client_id,
        domain=domain,
        query_vector=query_vector,
        limit=max(limit * 2, 8),
        content_types=allowed_content_types or DEFAULT_ALLOWED_TYPES,
    )

    results = _prioritize_results(raw_results, limit)
    grounded, confidence = _grounding_confidence(results)

    return {
        "grounded": grounded,
        "confidence": confidence,
        "matches": results,
        "sources": [_source_card(item) for item in results],
    }
