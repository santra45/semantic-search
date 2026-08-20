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


def merge_specs(
    db: Session,
    client_id: str,
    store_code: str,
    new_specs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Merge extracted specification keys into the stored spec vocabulary.

    Shape: `{key: {unit, label, count}}`. This vocabulary does two jobs from
    one table, which is why it is worth keeping accurate:

      * fed BACK into extraction, it stops key names fragmenting — without
        it one product yields `flow_rate`, the next `flowrate`, and a filter
        on either silently misses the other;
      * fed FORWARD into the retrieval tool's description, it tells the
        assistant which specs this particular store can be filtered by, so
        the feature works on a pump catalog and a furniture catalog with no
        configuration and no hardcoded spec list.

    `count` is how many products have carried the key, which drives ordering
    when the list is shown (to the extractor and to the assistant) and makes
    a one-off typo key visibly rare next to a real one.
    """
    if not new_specs:
        return {}
    ensure_agent_schema(db)
    existing = _load(db, client_id, store_code, "spec")
    merged: dict[str, dict[str, Any]] = {
        k: dict(v) for k, v in existing.items() if isinstance(v, dict)
    }

    for key, meta in new_specs.items():
        if not key or not isinstance(meta, dict):
            continue
        entry = merged.setdefault(key, {"unit": "", "label": "", "count": 0})
        entry["count"] = int(entry.get("count") or 0) + 1
        if not entry.get("label"):
            entry["label"] = str(meta.get("label") or "")
        # First unit seen for a key wins. A second unit showing up is the
        # mixed-unit case (B7) — recorded so it is visible in the admin view
        # rather than silently averaged away, but never used to convert.
        unit = str(meta.get("unit") or "")
        if unit:
            if not entry.get("unit"):
                entry["unit"] = unit
            elif entry["unit"] != unit:
                others = set(entry.get("other_units") or [])
                others.add(unit)
                entry["other_units"] = sorted(others)

    _save(db, client_id, store_code, "spec", merged)
    return merged


def get_specs(db: Session, client_id: str, store_code: str = "default") -> dict[str, dict[str, Any]]:
    """The store's specification vocabulary, most common key first."""
    ensure_agent_schema(db)
    data = _load(db, client_id, store_code, "spec")
    if not isinstance(data, dict):
        return {}
    clean = {k: v for k, v in data.items() if isinstance(v, dict)}
    return dict(sorted(clean.items(), key=lambda kv: (-int(kv[1].get("count") or 0), kv[0])))
