from __future__ import annotations

import json
from typing import Any

from backend.app.services.embedder import embed_document
from backend.app.services.product_service import build_product_text, extract_payload, strip_html
from backend.app.services.qdrant_service import delete_content_item, upsert_content_item

ALLOWED_CONTENT_TYPES = {
    "product",
    "cms_page",
    "cms_block",
    "widget",
    "review",
    "policy",
    "faq",
    "store_config",
}


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def _normalized_metadata(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata") or {}
    return metadata if isinstance(metadata, dict) else {"raw": _stringify(metadata)}


def normalize_item(item: dict[str, Any]) -> dict[str, Any]:
    content_type = _stringify(item.get("content_type")).strip()
    entity_id = _stringify(item.get("entity_id") or item.get("product_id")).strip()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise ValueError(f"Unsupported content_type: {content_type}")
    if not entity_id:
        raise ValueError("entity_id is required")

    title = _stringify(item.get("title") or item.get("name") or item.get("question")).strip()
    content = _stringify(item.get("content") or item.get("description")).strip()
    summary = _stringify(item.get("summary") or item.get("excerpt") or item.get("short_description")).strip()

    return {
        **item,
        "content_type": content_type,
        "entity_id": entity_id,
        "title": title,
        "content": content,
        "summary": summary,
        "permalink": _stringify(item.get("permalink")).strip(),
        "status": _stringify(item.get("status") or "active").strip(),
        "updated_at": _stringify(item.get("updated_at")).strip(),
        "metadata": _normalized_metadata(item),
    }


def build_content_text(item: dict[str, Any]) -> str:
    content_type = item["content_type"]
    if content_type == "product":
        product_item = {
            **item,
            "name": item.get("name") or item.get("title") or "",
            "description": item.get("description") or item.get("content") or "",
            "short_description": item.get("short_description") or item.get("summary") or "",
        }
        return build_product_text(product_item)

    parts = []
    title = item.get("title", "")
    summary = strip_html(item.get("summary", "")).strip()
    content = strip_html(item.get("content", "")).strip()
    metadata = item.get("metadata", {})

    if title:
        parts.append(f"Title: {title}")
    if content_type:
        parts.append(f"Content type: {content_type.replace('_', ' ')}")
    if summary:
        parts.append(f"Summary: {summary[:500]}")
    if content:
        parts.append(f"Content: {content[:1500]}")

    for key, value in metadata.items():
        label = key.replace("_", " ").strip().title()
        rendered_value = strip_html(_stringify(value)).strip()
        if label and rendered_value:
            parts.append(f"{label}: {rendered_value[:500]}")

    return "\n".join(parts)


def build_payload(item: dict[str, Any]) -> dict[str, Any]:
    content_type = item["content_type"]
    if content_type == "product":
        payload = extract_payload(item)
        payload.update(
            {
                "title": item.get("name") or item.get("title") or "",
                "summary": strip_html(item.get("summary") or item.get("short_description") or "")[:500],
                "content": strip_html(item.get("content") or item.get("description") or "")[:1500],
                "status": item.get("status") or "active",
                "updated_at": item.get("updated_at") or "",
                "metadata": item.get("metadata") or {},
            }
        )
        return payload

    base_payload = {
        "title": item.get("title") or "",
        "summary": strip_html(item.get("summary") or "")[:500],
        "content": strip_html(item.get("content") or "")[:1500],
        "permalink": item.get("permalink") or "",
        "status": item.get("status") or "active",
        "updated_at": item.get("updated_at") or "",
        "metadata": item.get("metadata") or {},
    }
    base_payload.update(item.get("metadata") or {})
    return base_payload


def ingest_items(
    client_id: str,
    domain: str,
    items: list[dict[str, Any]],
    embedding_api_key: str | None = None,
) -> dict[str, Any]:
    success_ids = []
    failed_ids = []
    counts_by_type: dict[str, int] = {}

    for raw_item in items:
        try:
            item = normalize_item(raw_item)
            embedded_text = build_content_text(item)
            vector = embed_document(
                embedded_text,
                api_key=embedding_api_key,
                client_id=client_id,
                query_type="embed_document",
            )
            payload = build_payload(item)
            payload["embedded_text"] = embedded_text
            upsert_content_item(
                client_id=client_id,
                domain=domain,
                content_type=item["content_type"],
                entity_id=item["entity_id"],
                vector=vector,
                payload=payload,
            )
            success_ids.append(
                {
                    "entity_id": item["entity_id"],
                    "content_type": item["content_type"],
                }
            )
            counts_by_type[item["content_type"]] = counts_by_type.get(item["content_type"], 0) + 1
        except Exception:
            failed_ids.append(
                {
                    "entity_id": _stringify(raw_item.get("entity_id") or raw_item.get("product_id")),
                    "content_type": _stringify(raw_item.get("content_type")),
                }
            )

    return {
        "success_count": len(success_ids),
        "failed_count": len(failed_ids),
        "success_ids": success_ids,
        "failed_ids": failed_ids,
        "counts_by_type": counts_by_type,
    }


def delete_items(client_id: str, domain: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    deleted = []
    failed = []

    for raw_item in items:
        try:
            item = normalize_item(raw_item)
            delete_content_item(client_id, domain, item["content_type"], item["entity_id"])
            deleted.append(
                {
                    "entity_id": item["entity_id"],
                    "content_type": item["content_type"],
                }
            )
        except Exception:
            failed.append(
                {
                    "entity_id": _stringify(raw_item.get("entity_id") or raw_item.get("product_id")),
                    "content_type": _stringify(raw_item.get("content_type")),
                }
            )

    return {
        "deleted_count": len(deleted),
        "failed_count": len(failed),
        "deleted_items": deleted,
        "failed_items": failed,
    }
