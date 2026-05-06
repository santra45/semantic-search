"""
qdrant_query_inspect.py - inspect what Qdrant returns for a customer-style query.

What this is for: when a shopper asks the chatbot something like
"How do I clean and maintain my Stainless Steel Water Feature?", you want to
know what the vector index actually surfaces -- products? cms_blocks? FAQs?
This script embeds your query the exact same way the backend does
(Gemini gemini-embedding-001 with task_type=RETRIEVAL_QUERY) and runs the
exact same Qdrant call, but prints results grouped by content_type so the
source of an answer is obvious.

Quick start:

    # 1. List collections to find your client's index
    python scripts/qdrant_query_inspect.py --list

    # 2. Search by collection name
    python scripts/qdrant_query_inspect.py \\
        --collection products_example_com_clientabc \\
        --query "How do I clean my Stainless Steel Water Feature?"

    # 3. Or search by client-id + domain (same naming rule as backend)
    python scripts/qdrant_query_inspect.py \\
        --client-id 9bd2cf13-f6d9-44e5-80e8-331544a1e8cb \\
        --domain example.com \\
        --query "How do I clean my Stainless Steel Water Feature?"

    # 4. Restrict to non-product content (mimics what PolicyAgent / GenericChatAgent does)
    python scripts/qdrant_query_inspect.py \\
        --collection products_example_com_clientabc \\
        --query "..." \\
        --types cms_block,cms_page,faq,policy

    # 5. Dump full per-hit payload for deeper debugging
    python scripts/qdrant_query_inspect.py \\
        --collection products_example_com_clientabc \\
        --query "..." \\
        --show-payload

Tip: run once with no --types filter to see the full ranking, then run again
with the type filter to see only what the agent would actually use.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
)

load_dotenv()


# ── Config ───────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_HOST    = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT    = int(os.getenv("QDRANT_PORT", 6333))
EMBED_MODEL    = "gemini-embedding-001"


# ── Collection naming (mirror of backend/app/services/qdrant_service.py) ─────
def derive_collection_name(client_id: str, domain: str) -> str:
    client_safe = re.sub(r"[^a-zA-Z0-9]", "_", client_id)
    domain_safe = re.sub(r"[^a-zA-Z0-9]", "_", domain)
    return f"products_{domain_safe}_{client_safe}"


# ── Pretty-printer per hit ───────────────────────────────────────────────────
def render_hit(idx: int, hit, show_payload: bool) -> None:
    p = hit.payload or {}
    ct = p.get("content_type", "?")
    title = (
        p.get("title")
        or p.get("name")
        or p.get("question")
        or p.get("identifier")
        or "(untitled)"
    )
    eid = (
        p.get("entity_id")
        or p.get("product_id")
        or p.get("page_id")
        or p.get("post_id")
        or ""
    )

    print(f"  #{idx:>2}  [{hit.score:.4f}]  type={ct:<12}  id={eid}")
    print(f"        title : {title}")

    if ct == "product":
        meta = []
        if "sku" in p:          meta.append(f"sku={p['sku']}")
        if "price" in p:        meta.append(f"price={p['price']}")
        if "stock_status" in p: meta.append(f"stock={p['stock_status']}")
        if meta:
            print(f"        meta  : {'  '.join(meta)}")
        if p.get("permalink"):
            print(f"        url   : {p['permalink']}")

    elif ct in ("cms_block", "cms_page", "page", "policy"):
        if p.get("identifier"):
            print(f"        ident : {p['identifier']}")
        if p.get("permalink"):
            print(f"        url   : {p['permalink']}")
        body = p.get("content") or p.get("summary") or p.get("description") or ""
        if body:
            snippet = body[:240].strip().replace("\n", " ")
            ellipsis = "..." if len(body) > 240 else ""
            print(f"        body  : {snippet}{ellipsis}")

    elif ct == "faq":
        if p.get("answer"):
            ans = p["answer"][:240].strip().replace("\n", " ")
            ellipsis = "..." if len(p["answer"]) > 240 else ""
            print(f"        answer: {ans}{ellipsis}")

    else:
        body = p.get("summary") or p.get("content") or p.get("description") or ""
        if body:
            snippet = body[:240].strip().replace("\n", " ")
            ellipsis = "..." if len(body) > 240 else ""
            print(f"        body  : {snippet}{ellipsis}")

    if show_payload:
        # Strip the bulky fields that just clutter the terminal
        compact = {k: v for k, v in p.items() if k not in ("embedded_text",)}
        print("        payload:")
        for line in json.dumps(compact, default=str, indent=2).splitlines():
            print(f"          {line}")
    print()


# ── Commands ─────────────────────────────────────────────────────────────────
def cmd_list(qdrant: QdrantClient) -> None:
    cols = qdrant.get_collections().collections
    if not cols:
        print("\nNo collections in this Qdrant instance.\n")
        return
    print(f"\n{len(cols)} collection(s):\n")
    for c in sorted(cols, key=lambda x: x.name):
        try:
            cnt = qdrant.count(collection_name=c.name, exact=False).count
        except Exception:
            cnt = "?"
        print(f"  {c.name}   (~{cnt} points)")
    print()


def cmd_query(
    qdrant: QdrantClient,
    collection: str,
    query: str,
    types: list[str] | None,
    limit: int,
    threshold: float,
    show_payload: bool,
) -> None:
    # 1. Verify collection
    existing = {c.name for c in qdrant.get_collections().collections}
    if collection not in existing:
        print(f"\nERROR: collection {collection!r} does not exist.\n", file=sys.stderr)
        if existing:
            print("Available collections:", file=sys.stderr)
            for c in sorted(existing):
                print(f"  {c}", file=sys.stderr)
        else:
            print("(no collections at all -- has any data been ingested?)", file=sys.stderr)
        sys.exit(2)

    # 2. Header
    print()
    print(f"Query      : {query!r}")
    print(f"Collection : {collection}")
    print(f"Types      : {','.join(types) if types else '(all)'}")
    print(f"Limit      : {limit}")
    if threshold > 0:
        print(f"Threshold  : >= {threshold}")
    print("-" * 78)

    # 3. Embed (matches backend/app/services/embedder.py exactly)
    if not GEMINI_API_KEY:
        print("\nERROR: GEMINI_API_KEY missing from environment / .env\n", file=sys.stderr)
        sys.exit(1)

    gemini = genai.Client(api_key=GEMINI_API_KEY)
    emb = gemini.models.embed_content(
        model=EMBED_MODEL,
        contents=query,
        config={"task_type": "RETRIEVAL_QUERY"},
    )
    vector = emb.embeddings[0].values
    print(f"Embedded   : {len(vector)} dims (model={EMBED_MODEL})")

    # 4. Build optional content_type filter (same shape backend uses)
    qfilter = None
    if types:
        if len(types) == 1:
            qfilter = Filter(must=[FieldCondition(
                key="content_type", match=MatchValue(value=types[0])
            )])
        else:
            qfilter = Filter(must=[FieldCondition(
                key="content_type", match=MatchAny(any=types)
            )])

    # 5. Search
    result = qdrant.query_points(
        collection_name=collection,
        query=vector,
        query_filter=qfilter,
        limit=limit,
        with_payload=True,
    )
    hits = [h for h in result.points if (h.score or 0) >= threshold]

    if not hits:
        print("\n(no results -- try lowering --threshold, removing --types, or different query text)\n")
        return

    # 6. Summary by content_type
    by_type: dict[str, list] = defaultdict(list)
    for h in hits:
        ct = (h.payload or {}).get("content_type", "?")
        by_type[ct].append(h)

    print(f"\n{len(hits)} hit(s)  -  breakdown by content_type:")
    rows = sorted(by_type.items(), key=lambda kv: -max(h.score for h in kv[1]))
    for ct, group in rows:
        top = max(h.score for h in group)
        print(f"  {ct:<14}  count={len(group):<3}  top_score={top:.4f}")
    print()

    # 7. Full ranking (this is what the chatbot would see)
    print("All hits ordered by score (this is what the chatbot would see):\n")
    for i, h in enumerate(sorted(hits, key=lambda x: -(x.score or 0)), 1):
        render_hit(i, h, show_payload)


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--query", "-q",
                   help="Customer-style query to embed and search.")
    p.add_argument("--collection", "-c",
                   help="Qdrant collection name (overrides --client-id/--domain).")
    p.add_argument("--client-id",
                   help="Client UUID. Combine with --domain to derive the collection.")
    p.add_argument("--domain",
                   help="Storefront domain, e.g. example.com.")
    p.add_argument("--types", "-t",
                   help="Comma-separated content_type filter "
                        "(product,cms_block,cms_page,faq,policy,page,post,...).")
    p.add_argument("--limit", "-n", type=int, default=10,
                   help="Max hits to return (default 10).")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="Drop hits below this cosine score (0..1).")
    p.add_argument("--show-payload", action="store_true",
                   help="Dump full payload for each hit (verbose).")
    p.add_argument("--list", action="store_true",
                   help="List all Qdrant collections and exit.")
    args = p.parse_args()

    try:
        qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        # cheap probe -- raises early if Qdrant is unreachable
        qdrant.get_collections()
    except Exception as e:
        print(f"\nERROR: cannot reach Qdrant at {QDRANT_HOST}:{QDRANT_PORT} - {e}\n",
              file=sys.stderr)
        sys.exit(1)

    if args.list:
        cmd_list(qdrant)
        return

    if not args.query:
        p.error("--query is required (or pass --list to discover collections).")

    if args.collection:
        coll = args.collection
    elif args.client_id and args.domain:
        coll = derive_collection_name(args.client_id, args.domain)
    else:
        p.error("Specify --collection, OR both --client-id and --domain. "
                "Use --list to see existing collections.")

    types = [t.strip() for t in args.types.split(",") if t.strip()] if args.types else None
    cmd_query(qdrant, coll, args.query, types, args.limit, args.threshold, args.show_payload)


if __name__ == "__main__":
    main()
