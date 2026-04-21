"""
Per-client attribute & category vocabularies, persisted in MySQL.

Replaces the flat JSON files used by magento_chatbot (attribute_vocab.json,
category_vocab.json) with a multi-tenant table so two customers can't step on
each other. Ingested batches merge into the existing vocabulary — they never
replace it wholesale — because a batch sync only carries a slice of the catalog.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.magento.chatbot.db.schema import ensure_agent_schema


def _load(db: Session, client_id: str, store_code: str, vocab_type: str) -> dict:
    row = db.execute(
        text(
            """
            SELECT vocab_json
            FROM agent_client_vocab
            WHERE client_id = :client_id AND store_code = :store_code AND vocab_type = :vocab_type
            LIMIT 1
            """
        ),
        {"client_id": client_id, "store_code": store_code, "vocab_type": vocab_type},
    ).fetchone()
    if not row or not row.vocab_json:
        return {}
    try:
        return json.loads(row.vocab_json)
    except Exception:
        return {}


def _save(db: Session, client_id: str, store_code: str, vocab_type: str, payload: Any) -> None:
    serialized = json.dumps(payload, ensure_ascii=True)
    db.execute(
        text(
            """
            INSERT INTO agent_client_vocab (client_id, store_code, vocab_type, vocab_json)
            VALUES (:client_id, :store_code, :vocab_type, :vocab_json)
            ON DUPLICATE KEY UPDATE vocab_json = VALUES(vocab_json)
            """
        ),
        {
            "client_id": client_id,
            "store_code": store_code,
            "vocab_type": vocab_type,
            "vocab_json": serialized,
        },
    )
    db.commit()


def merge_attributes(
    db: Session,
    client_id: str,
    store_code: str,
    new_attributes: dict[str, set[str]],
) -> dict[str, list[str]]:
    """Merge `{attr_key: {values...}}` into the stored attribute vocabulary."""
    if not new_attributes:
        return {}
    ensure_agent_schema(db)
    existing = _load(db, client_id, store_code, "attribute")
    merged: dict[str, set[str]] = {k: set(v) for k, v in existing.items() if isinstance(v, list)}
    for key, values in new_attributes.items():
        merged.setdefault(key, set()).update(values)

    flattened = {k: sorted(v) for k, v in merged.items()}
    _save(db, client_id, store_code, "attribute", flattened)
    return flattened


def merge_categories(
    db: Session,
    client_id: str,
    store_code: str,
    new_categories: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    if not new_categories:
        return []
    ensure_agent_schema(db)
    existing = _load(db, client_id, store_code, "category")
    lookup: dict[str, dict[str, str]] = {}
    if isinstance(existing, list):
        for entry in existing:
            if isinstance(entry, dict) and entry.get("id"):
                lookup[str(entry["id"])] = {"id": str(entry["id"]), "name": entry.get("name", "")}

    for cid, entry in new_categories.items():
        if cid:
            lookup[str(cid)] = {"id": str(cid), "name": entry.get("name", "")}

    merged = sorted(lookup.values(), key=lambda e: int(e["id"]) if e["id"].isdigit() else e["id"])
    _save(db, client_id, store_code, "category", merged)
    return merged


def get_attributes(db: Session, client_id: str, store_code: str = "default") -> dict[str, list[str]]:
    ensure_agent_schema(db)
    data = _load(db, client_id, store_code, "attribute")
    return {k: v for k, v in data.items() if isinstance(v, list)}


def get_categories(db: Session, client_id: str, store_code: str = "default") -> list[dict[str, str]]:
    ensure_agent_schema(db)
    data = _load(db, client_id, store_code, "category")
    return data if isinstance(data, list) else []
