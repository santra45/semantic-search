"""WordPress-facing backend surface.

Deliberately parallel to `backend.app.magento` rather than shared with it.
The two platforms send different product shapes, need different lookup keys
(WooCommerce products routinely have no SKU — the post ID is the only
guaranteed identifier), and their prompts should be able to evolve
independently: tuning a Magento agent's wording must not silently change what
a WooCommerce shopper is told.

What IS shared lives in `backend.app.services` — Qdrant, embeddings,
licensing, domain auth, token accounting. Those are platform-agnostic
infrastructure and duplicating them would mean fixing every bug twice.
"""
