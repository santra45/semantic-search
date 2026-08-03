"""Shared across every WordPress integration on this backend.

Distinct from `app.wordpress.productqa.services`, which is specific to the
product Q&A plugin. Anything in here is used by more than one WordPress
integration and must not be tuned for either one in isolation:

  * `product_formatter` — the single WooCommerce product formatter. Both the
    search plugin (`semantic-search-woo`, via /api/sync/batch) and the Q&A
    plugin (`ai-product-qa-woo`, via /api/wordpress/productqa/sync/batch) run
    products through it, because both write to the same Qdrant point for the
    same product and the later write wins. Identical output makes that race a
    non-event. Change it for one caller and you change it for both — which is
    the point, not an accident.

Still deliberately separate from the Magento namespace: a prompt or format
tweak for a WooCommerce shopper must not silently change what a Magento one
is told.
"""
