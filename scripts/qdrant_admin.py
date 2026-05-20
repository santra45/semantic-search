#!/usr/bin/env python3
"""
Qdrant admin CLI for the AIChatbot project.

Inspect, filter, and delete points in the per-tenant Qdrant collections that
back the chatbot's retrieval. Designed for ops/dev work — diagnosing "why
isn't my CMS page surfacing", verifying a sync ran cleanly, removing stale
points after schema changes, exporting payloads for offline review.

Quick start
-----------
    # 1. Find the collection name (one per tenant)
    python qdrant_admin.py collections

    # 2. See what's indexed
    python qdrant_admin.py stats -c COLLECTION
    python qdrant_admin.py show  -c COLLECTION --type cms_page --limit 5

    # 3. Look up a specific item
    python qdrant_admin.py get   -c COLLECTION 42 --type cms_page

    # 4. Vector-search by text (needs GEMINI_API_KEY)
    python qdrant_admin.py search -c COLLECTION "return policy" --type cms_page

    # 5. Delete (asks for confirmation by default; --yes to skip; --dry-run to preview)
    python qdrant_admin.py delete -c COLLECTION --type cms_block

    # 6. Export to JSON for inspection
    python qdrant_admin.py export -c COLLECTION --type cms_page -o pages.json

    # 7. Nuke a tenant entirely (asks you to retype the collection name)
    python qdrant_admin.py purge  -c COLLECTION

Environment
-----------
    QDRANT_HOST       host (default: localhost)
    QDRANT_PORT       port (default: 6333)
    QDRANT_URL        full URL — overrides host/port
    QDRANT_API_KEY    optional, for cloud Qdrant
    GEMINI_API_KEY    only needed for `search`
    EMBED_MODEL       embedding model (default: gemini-embedding-001)

Filters
-------
The most useful filters are exposed as flags on most commands:
    --type        product | category | cms_page | cms_block | widget |
                  store_config | promotion | (any other)
    --entity-id   the id Magento sent (string match against payload.entity_id)
    --store-code  scope to one store view

Larger ad-hoc filters: use `export` to dump JSON, then grep / jq it.

Chunked content
---------------
`cms_page`, `cms_block`, and `category` are split into N Qdrant points
per source item (~500-char chunks with ~200-char overlap). `stats` shows
this divergence in two columns — ENTITIES (parent items, matches admin
dashboard) and POINTS (raw Qdrant rows). For these types `show` /
`get` / `search` / `delete` / `export` operate on raw POINTS, not
entities — that's usually what you want when debugging or removing data
(deleting by content_type wipes every chunk of every page in one go).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        FieldCondition,
        Filter,
        IsEmptyCondition,
        MatchValue,
        PayloadField,
    )
except ImportError:
    sys.stderr.write(
        "ERROR: qdrant-client not installed.\n"
        "       Install: pip install qdrant-client\n"
    )
    sys.exit(1)


# ── Connection / config ──────────────────────────────────────────────────────


def make_client() -> QdrantClient:
    """Build a Qdrant client from env vars. URL takes precedence over host/port."""
    url = os.environ.get("QDRANT_URL")
    api_key = os.environ.get("QDRANT_API_KEY")
    if url:
        return QdrantClient(url=url, api_key=api_key) if api_key else QdrantClient(url=url)
    host = os.environ.get("QDRANT_HOST", "localhost")
    port = int(os.environ.get("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port)


def build_filter(
    content_type: Optional[str] = None,
    entity_id: Optional[str] = None,
    store_code: Optional[str] = None,
) -> Optional[Filter]:
    """Build a Qdrant must-filter from the most-used field combos. None when no filter."""
    must = []
    if content_type:
        must.append(FieldCondition(key="content_type", match=MatchValue(value=content_type)))
    if entity_id:
        must.append(FieldCondition(key="entity_id", match=MatchValue(value=str(entity_id))))
    if store_code:
        must.append(FieldCondition(key="store_code", match=MatchValue(value=store_code)))
    return Filter(must=must) if must else None


# ── Output helpers ───────────────────────────────────────────────────────────


def truncate(value: Any, n: int = 200) -> str:
    """Render any payload value as a one-line, length-bounded string."""
    if value is None:
        return ""
    if isinstance(value, str):
        s = value.replace("\n", " | ").replace("\r", "")
        return s if len(s) <= n else s[:n] + f"... [+{len(s) - n} chars]"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return f"[list of {len(value)}]"
    if isinstance(value, dict):
        return f"{{dict with {len(value)} keys}}"
    return repr(value)


# Fields shown first in any point dump — the identifying / routing ones.
_HEADER_FIELDS = ("content_type", "entity_id", "title", "label", "name", "sku", "identifier")


def print_point(point, fields: Optional[list[str]] = None, full: bool = False) -> None:
    """Render one point. Compact by default; --full or --fields to expand."""
    payload = dict(point.payload or {})
    score = getattr(point, "score", None)

    print("-" * 78)
    print(f"id: {point.id}")
    if score is not None:
        print(f"score: {score:.4f}")

    # Header fields first.
    for key in _HEADER_FIELDS:
        if key in payload and payload[key] not in (None, "", [], {}):
            print(f"  {key + ':':<18}{truncate(payload[key], 100)}")

    if fields:
        # Custom field selection.
        for key in fields:
            if key in _HEADER_FIELDS:
                continue
            if key not in payload:
                print(f"  {key + ':':<18}(missing)")
                continue
            cap = 1500 if full else 300
            print(f"  {key + ':':<18}{truncate(payload[key], cap)}")
    elif full:
        # All remaining payload fields, capped at 1500 chars each.
        for key in sorted(payload.keys()):
            if key in _HEADER_FIELDS:
                continue
            print(f"  {key + ':':<18}{truncate(payload[key], 1500)}")
    else:
        # Compact: a single body preview line, picking the most useful field.
        for key in ("summary", "content", "meta_description", "value"):
            if payload.get(key):
                print(f"  {key + ':':<18}{truncate(payload[key], 200)}")
                break


# ── Commands ─────────────────────────────────────────────────────────────────


def cmd_collections(args, client: QdrantClient) -> None:
    cols = client.get_collections().collections
    if not cols:
        print("(no collections)")
        return
    print(f"{'NAME':<60}  POINTS")
    print("-" * 78)
    for c in sorted(cols, key=lambda x: x.name):
        try:
            count = client.count(collection_name=c.name, exact=True).count
        except Exception:
            count = "?"
        print(f"{c.name:<60}  {count}")


def cmd_info(args, client: QdrantClient) -> None:
    info = client.get_collection(args.collection)
    vectors = info.config.params.vectors
    print(f"Collection:        {args.collection}")
    print(f"Status:            {info.status}")
    print(f"Vector size:       {vectors.size if vectors else '?'}")
    print(f"Distance:          {vectors.distance if vectors else '?'}")
    print(f"Points count:      {info.points_count}")
    print(f"Indexed vectors:   {info.indexed_vectors_count}")


# Common content types we recognise — used to drive the `stats` table.
# Keep in step with backend KNOWN_CONTENT_TYPES (app/services/qdrant_service.py).
_KNOWN_TYPES = ("product", "category", "cms_page", "cms_block", "widget",
                "store_config", "promotion", "policy", "faq", "review",
                "page", "post")

# Content types that get chunked at sync time (see backend
# CHUNKABLE_CONTENT_TYPES). For these the raw Qdrant point count
# inflates by chunk fan-out — a 50-page CMS catalog with ~5 chunks
# per page shows up as 250 points. `cmd_stats` queries an additional
# "entity count" with a chunk_index=0-OR-missing filter so the table
# can show both numbers (entities = what the merchant indexed, points
# = what's stored in Qdrant). Keep in step with the same set in the
# backend.
_CHUNKABLE_TYPES = frozenset({"cms_page", "cms_block", "category"})


def build_entity_filter(content_type: str) -> Filter:
    """Filter that counts ONE row per logical entity even for chunked types.

    Matches points where chunk_index=0 OR the field is missing. For
    chunked content (cms_page / cms_block / category post-Phase 1.3 +
    category) only the first chunk has chunk_index=0; for legacy
    pre-chunking points the field is absent entirely. Either way the
    parent entity is represented once.

    For non-chunkable types this filter is equivalent to a plain
    content_type filter because no point in those types carries
    chunk_index at all, so the IsEmpty branch always matches.

    Mirror of backend qdrant_service.count_content_type's filter — keep
    in sync if the chunkable-detection logic ever changes there.
    """
    return Filter(
        must=[
            FieldCondition(key="content_type", match=MatchValue(value=content_type)),
        ],
        should=[
            FieldCondition(key="chunk_index", match=MatchValue(value=0)),
            IsEmptyCondition(is_empty=PayloadField(key="chunk_index")),
        ],
    )


def cmd_stats(args, client: QdrantClient) -> None:
    """Two-column count: ENTITIES (logical items the merchant indexed) +
    POINTS (raw Qdrant points; chunked types fan out N points per entity).

    Both columns are useful in different contexts:
      * ENTITIES is what the admin dashboard shows and what the merchant
        thinks of as "indexed content" — 50 CMS pages, 30 categories.
      * POINTS is the actual Qdrant storage shape — 250 chunks for those
        50 pages. Useful when debugging "why is my collection so big" or
        verifying chunking is actually fanning out.
    """
    try:
        total_points = client.count(collection_name=args.collection, exact=True).count
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Per-type counts. For chunkable types we query twice (entities + points);
    # for non-chunkable types entities == points so we just copy.
    rows: list[tuple[str, int, int]] = []  # (type, entities, points)
    accounted_points = 0
    total_entities = 0
    for t in _KNOWN_TYPES:
        try:
            n_points = client.count(
                collection_name=args.collection,
                count_filter=build_filter(content_type=t),
                exact=True,
            ).count
        except Exception:
            n_points = 0
        if not n_points:
            continue

        if t in _CHUNKABLE_TYPES:
            try:
                n_entities = client.count(
                    collection_name=args.collection,
                    count_filter=build_entity_filter(t),
                    exact=True,
                ).count
            except Exception:
                # If the entity-count query fails for any reason, fall
                # back to the point count rather than 0 — we'd rather
                # over-report than show a confusing zero alongside a
                # non-zero points column.
                n_entities = n_points
        else:
            n_entities = n_points

        rows.append((t, n_entities, n_points))
        total_entities += n_entities
        accounted_points += n_points

    # Sort by raw points desc — biggest contributors to collection size first.
    rows.sort(key=lambda r: -r[2])

    print(f"{'CONTENT TYPE':<20}  {'ENTITIES':>10}  {'POINTS':>10}")
    print("-" * 46)
    for t, e, p in rows:
        # Asterisk marks chunked types so the merchant can immediately see
        # which rows explain any entities-vs-points divergence below.
        marker = " *" if t in _CHUNKABLE_TYPES else "  "
        print(f"{t + marker:<20}  {e:>10,}  {p:>10,}")
    if total_points != accounted_points:
        other = total_points - accounted_points
        print(f"{'(other)':<20}  {'?':>10}  {other:>10,}")
    print("-" * 46)
    print(f"{'TOTAL':<20}  {total_entities:>10,}  {total_points:>10,}")
    if total_entities != total_points:
        # Footnote only when there's actually a divergence, to avoid
        # cluttering output on collections that hold no chunked content.
        print()
        print("  * chunked content type — entities = parent items "
              "(cms_page / cms_block / category),")
        print("    points = chunks stored in Qdrant. See "
              "CHUNKABLE_CONTENT_TYPES in qdrant_service.py.")


def cmd_show(args, client: QdrantClient) -> None:
    f = build_filter(
        content_type=args.type,
        entity_id=args.entity_id,
        store_code=args.store_code,
    )
    points, _next_offset = client.scroll(
        collection_name=args.collection,
        scroll_filter=f,
        limit=args.limit,
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        print("(no matching points)")
        return
    print(f"Showing {len(points)} point(s)")
    fields = [f.strip() for f in args.fields.split(",")] if args.fields else None
    for p in points:
        print_point(p, fields=fields, full=args.full)
    print()


def cmd_get(args, client: QdrantClient) -> None:
    f = build_filter(entity_id=args.entity_id, content_type=args.type)
    points, _ = client.scroll(
        collection_name=args.collection,
        scroll_filter=f,
        limit=10,
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        type_str = f" (type={args.type})" if args.type else ""
        print(f"(no point with entity_id={args.entity_id}{type_str})")
        return
    for p in points:
        print_point(p, full=True)
    print()


def cmd_search(args, client: QdrantClient) -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.stderr.write(
            "ERROR: GEMINI_API_KEY not set. The search command needs to embed\n"
            "       your query first. Set the env var or use `show --type X`\n"
            "       with a payload-field filter instead.\n"
        )
        sys.exit(1)

    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        sys.stderr.write(
            "ERROR: google-genai not installed. Install: pip install google-genai\n"
        )
        sys.exit(1)

    model = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
    genai_client = genai.Client(api_key=api_key)
    try:
        resp = genai_client.models.embed_content(
            model=model,
            contents=args.query,
            config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
    except Exception as e:
        sys.stderr.write(f"ERROR: embedding failed: {e}\n")
        sys.exit(1)

    vector = resp.embeddings[0].values
    f = build_filter(content_type=args.type, store_code=args.store_code)

    result = client.query_points(
        collection_name=args.collection,
        query=vector,
        query_filter=f,
        limit=args.limit,
        with_payload=True,
    )
    if not result.points:
        print("(no matches)")
        return
    print(f"Top {len(result.points)} matches for: {args.query!r}")
    fields = [f.strip() for f in args.fields.split(",")] if args.fields else None
    for p in result.points:
        print_point(p, fields=fields, full=args.full)
    print()


def _scroll_ids(client: QdrantClient, collection: str, qfilter: Optional[Filter],
                batch: int = 500):
    """Yield point IDs matching the filter, in batches of `batch`."""
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=qfilter,
            limit=batch,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        if not points:
            break
        yield [p.id for p in points]
        if offset is None:
            break


def cmd_delete(args, client: QdrantClient) -> None:
    f = build_filter(
        content_type=args.type,
        entity_id=args.entity_id,
        store_code=args.store_code,
    )
    if f is None:
        sys.stderr.write(
            "ERROR: at least one filter is required (--type, --entity-id, "
            "or --store-code).\n"
            "       To delete the whole collection use `purge` instead.\n"
        )
        sys.exit(1)

    # Pre-count so the user sees the blast radius before confirming.
    n = client.count(
        collection_name=args.collection,
        count_filter=f,
        exact=True,
    ).count
    if n == 0:
        print("(no matching points to delete)")
        return

    # Show a small sample so the user can sanity-check we're about to delete
    # the right thing.
    sample, _ = client.scroll(
        collection_name=args.collection,
        scroll_filter=f,
        limit=3,
        with_payload=True,
        with_vectors=False,
    )
    print(f"Will delete {n} point(s) from {args.collection}.")
    if sample:
        print("Sample (first 3):")
        for p in sample:
            payload = dict(p.payload or {})
            label = (
                payload.get("title")
                or payload.get("name")
                or payload.get("label")
                or payload.get("sku")
                or payload.get("identifier")
                or payload.get("entity_id")
                or str(p.id)
            )
            ct = payload.get("content_type") or "?"
            print(f"  - [{ct}] {truncate(label, 80)}")

    if args.dry_run:
        print("(dry run — pass --yes to actually delete)")
        return
    if not args.yes:
        resp = input(f"Delete {n} point(s)? [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted.")
            return

    deleted = 0
    for ids in _scroll_ids(client, args.collection, f):
        client.delete(collection_name=args.collection, points_selector=ids)
        deleted += len(ids)
    print(f"Deleted {deleted} point(s).")


def cmd_purge(args, client: QdrantClient) -> None:
    try:
        n = client.count(collection_name=args.collection, exact=True).count
    except Exception as e:
        sys.stderr.write(f"ERROR: collection not found or unreachable: {e}\n")
        sys.exit(1)

    print(f"WARNING: this will DELETE the entire collection {args.collection!r} "
          f"({n} point(s)).")
    if args.dry_run:
        print("(dry run — pass --yes to actually purge)")
        return
    if not args.yes:
        # Require typing the collection name back to confirm — keeps tab-complete
        # accidents from nuking production data.
        confirm = input(f"Type the collection name to confirm: ").strip()
        if confirm != args.collection:
            print("Aborted (collection name did not match).")
            return

    client.delete_collection(collection_name=args.collection)
    print(f"Purged {args.collection}.")


def cmd_export(args, client: QdrantClient) -> None:
    f = build_filter(content_type=args.type, store_code=args.store_code)

    out: list[dict[str, Any]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=args.collection,
            scroll_filter=f,
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=args.include_vectors,
        )
        if not points:
            break
        for p in points:
            row: dict[str, Any] = {"id": str(p.id), "payload": dict(p.payload or {})}
            if args.include_vectors and getattr(p, "vector", None) is not None:
                row["vector"] = list(p.vector)
            out.append(row)
        if offset is None:
            break

    path = args.output or f"{args.collection}.json"
    with open(path, "w", encoding="utf-8") as f_out:
        json.dump(out, f_out, ensure_ascii=False, indent=2)
    print(f"Exported {len(out)} point(s) to {path}")


# ── argparse setup ───────────────────────────────────────────────────────────


def _common_filter_args(p: argparse.ArgumentParser) -> None:
    """Add the standard --type / --entity-id / --store-code triple."""
    p.add_argument("--type",
                   help="Content type filter, e.g. product, category, cms_page, "
                        "cms_block, widget, store_config, promotion")
    p.add_argument("--entity-id",
                   help="Filter to a specific Magento entity_id")
    p.add_argument("--store-code",
                   help="Scope to one Magento store view")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="qdrant_admin.py",
        description="Qdrant admin CLI for the AIChatbot project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:

  # List collections (one per tenant)
  python qdrant_admin.py collections

  # Stats on a collection
  python qdrant_admin.py stats -c products_storedomain_com_clientid

  # Show 5 cms_page points (compact)
  python qdrant_admin.py show -c COL --type cms_page --limit 5

  # Show all CMS-page payload fields for entity_id 12
  python qdrant_admin.py show -c COL --type cms_page --entity-id 12 --full

  # Custom field selection
  python qdrant_admin.py show -c COL --type cms_page --fields title,permalink,content

  # Vector search (needs GEMINI_API_KEY)
  python qdrant_admin.py search -c COL "return policy" --type cms_page

  # Dry-run a delete to see what would be removed
  python qdrant_admin.py delete -c COL --type cms_block --dry-run

  # Actually delete (asks for confirmation)
  python qdrant_admin.py delete -c COL --type cms_block

  # Delete one specific item without confirmation
  python qdrant_admin.py delete -c COL --type cms_page --entity-id 42 --yes

  # Export all CMS pages to JSON for offline review
  python qdrant_admin.py export -c COL --type cms_page -o pages.json
""",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("collections", help="List all collections")
    p.set_defaults(func=cmd_collections)

    p = sub.add_parser("info", help="Collection metadata")
    p.add_argument("-c", "--collection", required=True)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("stats", help="Count points by content_type")
    p.add_argument("-c", "--collection", required=True)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("show", help="List points with optional filters")
    p.add_argument("-c", "--collection", required=True)
    _common_filter_args(p)
    p.add_argument("--limit", type=int, default=10,
                   help="Max points to show (default: 10)")
    p.add_argument("--fields",
                   help="Comma-separated payload fields to show (e.g. title,permalink,content)")
    p.add_argument("--full", action="store_true",
                   help="Show every payload field (truncated at 1500 chars each)")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("get", help="Look up a point by entity_id")
    p.add_argument("entity_id")
    p.add_argument("-c", "--collection", required=True)
    p.add_argument("--type", help="Optional content_type to disambiguate")
    p.set_defaults(func=cmd_get)

    p = sub.add_parser("search", help="Vector search by text query (needs GEMINI_API_KEY)")
    p.add_argument("query")
    p.add_argument("-c", "--collection", required=True)
    p.add_argument("--type")
    p.add_argument("--store-code")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--fields",
                   help="Comma-separated payload fields to show")
    p.add_argument("--full", action="store_true")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("delete", help="Delete points matching filters")
    p.add_argument("-c", "--collection", required=True)
    _common_filter_args(p)
    p.add_argument("--dry-run", action="store_true",
                   help="Show count + sample without deleting")
    p.add_argument("-y", "--yes", action="store_true",
                   help="Skip confirmation prompt")
    p.set_defaults(func=cmd_delete)

    p = sub.add_parser("purge", help="Delete the entire collection")
    p.add_argument("-c", "--collection", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-y", "--yes", action="store_true",
                   help="Skip the type-the-collection-name confirmation")
    p.set_defaults(func=cmd_purge)

    p = sub.add_parser("export", help="Dump payloads to JSON")
    p.add_argument("-c", "--collection", required=True)
    p.add_argument("--type")
    p.add_argument("--store-code")
    p.add_argument("-o", "--output",
                   help="Output file (default: <collection>.json)")
    p.add_argument("--include-vectors", action="store_true",
                   help="Include the embedding vectors in the dump (large!)")
    p.set_defaults(func=cmd_export)

    args = parser.parse_args()
    client = make_client()
    try:
        args.func(args, client)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()
