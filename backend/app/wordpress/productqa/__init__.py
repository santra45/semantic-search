"""Per-product Q&A for WooCommerce.

Serves the `ai-product-qa-woo` plugin: a mini-chat on the single-product page
that answers questions about the one product in front of the shopper.

Wire surface (all under /api/wordpress/productqa):
    POST retrieve/product   — the on-page product's indexed payload, by post ID
    POST retrieve/content   — merchant FAQ grounding
    POST retrieve/answer    — one-shot grounded answer
    POST sync/batch         — content ingest (honours X-Full-Sync)
    POST sync/delete        — single delete
    GET  sync/status        — indexed counts
    POST sync/purge         — purge one content_type (FAQ replace-in-place)
"""
