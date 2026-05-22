"""Canonical slug function for payload-key fragments.

Single source of truth for the slug used in:

  • sync-time payload writes (product_formatter.py — `normalize_token`)
  • runtime Qdrant FieldCondition keys (qdrant_service.py)
  • runtime post-filter fallback (retrieve.py — compat mode)

If these ever produce different output for the same input, every
attribute / category filter lookup silently misses and the bot
returns the wrong subset of products. Centralising here makes drift
impossible (one function, one definition).

Algorithm:
  lowercase → "%" maps to " percent" → non-alphanumerics become "_"
  → consecutive "_" collapse → leading/trailing "_" stripped.

Examples:
  "Stainless Steel"   → "stainless_steel"
  "10% off"           → "10_percent_off"
  "ABC-123"           → "abc_123"
  "  spaced "         → "spaced"
  ""                  → ""
"""

from __future__ import annotations

import re
from typing import Any


def slug(value: Any) -> str:
    """Slugify a string into the snake_case payload-key fragment.

    Accepts any value; coerces to string. Empty / falsy input returns "".
    """
    s = str(value or "").strip().lower()
    s = re.sub(r"%", " percent", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")
