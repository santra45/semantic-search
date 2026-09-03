"""
Licence key format: minting, hashing, and (non-authoritative) parsing.

    czg_live_mchat_7Kq2mXvR9nBpL4wTzYsE6dHfA3jCgN8u_a4f9c2
    └┬┘ └─┬┘ └─┬─┘ └───────────────┬──────────────┘ └──┬─┘
  issuer  env  product        192-bit random        checksum

Why an opaque token rather than the JWT this replaces: validate_license_key
already hit the database on every request — it has to, to check is_active,
expires_at and the client's status — and the row deliberately beat the JWT's
claims because a token is a snapshot from issue time. So the JWT was charging
~350 characters for statelessness nothing used.

──────────────────────────────────────────────────────────────────────────────
THE PRODUCT SEGMENT IS NOT A CREDENTIAL.

`mchat` is a label so a human pasting three keys into three module configs can
tell them apart. It is not evidence of anything. The authoritative binding is
licence → subscription → product, in the database, resolved by hash.

Editing the segment breaks the checksum and then the hash, so a tampered key is
rejected — that part is safe. The danger is subtler: a future code path that
parses the prefix *before* the hash lookup, to pick a route or tag a metric,
and then trusts what it read. That is attacker-controlled input.

Two structural defences, not just this comment:

  * parse_for_logging() names every field it returns `unverified_*`, so a caller
    reading `parsed["unverified_product"]` has to actively ignore the warning.
  * resolve_* helpers never accept a parsed label. Product identity comes back
    from the database row or not at all.

If you find yourself wanting the product before the lookup, you want the
lookup.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import hashlib
import re
import secrets
import zlib
from typing import Optional

# ── Format constants ─────────────────────────────────────────────────────────

ISSUER = "czg"

# Only two, deliberately. A third value is a one-line change to add later and
# effectively permanent once dev sites hold keys minted with it; anything
# needing isolation from production needs its own account and billing anyway,
# at which point `test` already covers it.
ENVIRONMENTS = ("live", "test")

SEPARATOR = "_"

# base62. These keys are copy-pasted from a dashboard into an admin field, not
# transcribed by hand, so the ambiguous-character problem base58 solves doesn't
# apply here and the extra alphabet keeps the string shorter.
_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_BASE = len(_ALPHABET)

SECRET_BYTES = 24        # 192 bits
SECRET_CHARS = 33        # ceil(192 / log2(62))
CHECKSUM_CHARS = 6       # 62^6 ≈ 5.7e10, comfortably over CRC32's 2^32

# How much of the secret appears in the stored, displayable prefix. Enough to
# tell two keys apart in a list; far too little to be worth guessing from.
PREFIX_SECRET_CHARS = 4

_SEGMENT_RE = re.compile(r"^[a-z][a-z0-9]{1,7}$")


# ── base62 ───────────────────────────────────────────────────────────────────

def _b62_encode(value: int, width: int) -> str:
    """Encode *value* in base62, left-padded to exactly *width* chars."""
    if value < 0:
        raise ValueError("cannot encode a negative value")
    out = []
    while value:
        value, rem = divmod(value, _BASE)
        out.append(_ALPHABET[rem])
    s = "".join(reversed(out)) or _ALPHABET[0]
    if len(s) > width:
        raise ValueError(f"value needs {len(s)} chars, width is {width}")
    return s.rjust(width, _ALPHABET[0])


_ALPHABET_SET = frozenset(_ALPHABET)


def _is_base62(value: str) -> bool:
    """True when every character is in the minting alphabet.

    Length alone is not a charset check, and these two fields reach a log line
    on the unauthenticated path — see parse_for_logging().
    """
    return bool(value) and _ALPHABET_SET.issuperset(value)


def _random_secret() -> str:
    """SECRET_CHARS of base62 drawn from SECRET_BYTES of OS randomness."""
    value = int.from_bytes(secrets.token_bytes(SECRET_BYTES), "big")
    return _b62_encode(value, SECRET_CHARS)


# ── Checksum ─────────────────────────────────────────────────────────────────

def _checksum(body: str) -> str:
    """CRC32 of everything before the checksum, base62, fixed width.

    Scope, so nobody mistakes this for more than it is: it catches a truncated
    paste or a mistyped character in an admin field *before* a network call,
    and it lets secret scanners match this format without drowning in false
    positives. It is not a security control — it is public and recomputable —
    and it does not save a database round trip on a genuine key.
    """
    return _b62_encode(zlib.crc32(body.encode("ascii")) & 0xFFFFFFFF, CHECKSUM_CHARS)


# ── Minting ──────────────────────────────────────────────────────────────────

def mint(environment: str, product_segment: str) -> dict:
    """Create a new licence key.

    Returns the plaintext key, the hash to look it up by, and the displayable
    prefix. Since 2026-09-03 the plaintext IS persisted, in `licences.licence_key`
    — see the note above LICENCES_TABLE in schema_v2.py for the trade that was
    made and what it costs. `key_hash` is still what resolution matches on, so
    nothing on the hot path changed.
    """
    if environment not in ENVIRONMENTS:
        raise ValueError(
            f"Unknown environment '{environment}'. Expected one of {ENVIRONMENTS}."
        )
    if not _SEGMENT_RE.match(product_segment or ""):
        raise ValueError(
            f"Bad product segment '{product_segment}'. Expected 2-8 lowercase "
            f"alphanumerics starting with a letter."
        )

    secret = _random_secret()
    body = SEPARATOR.join((ISSUER, environment, product_segment, secret))
    key = body + SEPARATOR + _checksum(body)

    return {
        "key": key,
        "key_hash": hash_key(key),
        "key_prefix": SEPARATOR.join(
            (ISSUER, environment, product_segment, secret[:PREFIX_SECRET_CHARS])
        ),
    }


def prefix_of(key: Optional[str]) -> Optional[str]:
    """The displayable head of a stored key: `czg_live_mchat_7Kq2`.

    Exists because the database now holds the whole key (see mint()). Every log
    line and list view that used to read a `key_prefix` COLUMN must run the
    plaintext through here instead — otherwise the change from storing a prefix
    to storing a key silently turns every one of those call sites into a
    credential dump, and log files are the one place a leaked key is least
    likely to be noticed and most likely to be shipped somewhere else.

    Truncates rather than parsing: a malformed or legacy value still yields
    something safe to print, which a parse would not.
    """
    if not key:
        return None
    parts = key.split(SEPARATOR)
    if len(parts) < 4:
        # Not our format. Show enough to correlate, never enough to use.
        return key[:12] + "…"
    issuer, environment, product_segment, secret = parts[0], parts[1], parts[2], parts[3]
    return SEPARATOR.join(
        (issuer, environment, product_segment, secret[:PREFIX_SECRET_CHARS])
    )


# ── Hashing ──────────────────────────────────────────────────────────────────

def hash_key(key: str) -> str:
    """SHA-256 of the whole key, hex. This is what the database stores.

    SHA-256 and not bcrypt/argon2 on purpose: a password KDF exists to make
    guessing a low-entropy secret expensive, and this secret carries 192 bits
    of OS randomness. A deliberately slow hash here would only tax every
    request on the hot path to defend against an attack nobody can mount.

    Storing the hash rather than the key means a database dump is not a
    handover of every customer's working credential — which is exactly what
    the old plaintext-JWT column was.
    """
    return hashlib.sha256((key or "").strip().encode("utf-8")).hexdigest()


# ── Shape check (cheap, pre-lookup) ──────────────────────────────────────────

def looks_valid(key: str) -> bool:
    """Structural check only: right shape, right issuer, checksum agrees.

    True here means "worth a database lookup", never "authorised". Use it to
    reject obvious garbage — a truncated paste, a stray quote — before
    spending a query. Anything that passes still has to be resolved by hash.
    """
    parsed = parse_for_logging(key)
    return bool(parsed and parsed["unverified_checksum_ok"])


# ── Parsing — for logs and display ONLY ──────────────────────────────────────

def parse_for_logging(key: str) -> Optional[dict]:
    """Split a key into its visible segments. NOT an authorisation decision.

    Every field comes back prefixed `unverified_` because every field is
    attacker-controlled: the string was handed to us by whoever made the
    request. Use these to write a log line or render a masked key in a
    dashboard. Never to choose a code path, route a request, pick a rate
    limit, or decide what a caller may do.

    Returns None if the key isn't the right shape at all.
    """
    raw = (key or "").strip()
    parts = raw.split(SEPARATOR)
    if len(parts) != 5:
        return None

    issuer, environment, product_segment, secret, checksum = parts

    if issuer != ISSUER:
        return None

    # Every segment is constrained to a known vocabulary or a known charset
    # before any of it is returned, because the one guaranteed consumer of this
    # function is a log line on the UNAUTHENTICATED path: resolve_key() denies
    # an unknown key and logs mask(presented_key) at INFO.
    #
    # Splitting on "_" only guarantees the parts contain no underscore. It says
    # nothing about newlines, ANSI escapes, or control bytes, so environment and
    # product_segment were previously free-form attacker input on their way into
    # the log. A newline there forges a whole log line — an attacker choosing
    # what the operator's incident timeline says is a real problem even though
    # the request itself is refused.
    #
    # Rejecting outright rather than sanitising: anything failing these checks
    # was never minted here, so there is nothing to preserve. The tampering case
    # worth keeping legible — a real key with an edited segment — still parses
    # and reports unverified_checksum_ok False, which is what a reader needs.
    if environment not in ENVIRONMENTS:
        return None
    if not _SEGMENT_RE.match(product_segment):
        return None
    if len(secret) != SECRET_CHARS or len(checksum) != CHECKSUM_CHARS:
        return None
    if not _is_base62(secret) or not _is_base62(checksum):
        return None

    body = SEPARATOR.join((issuer, environment, product_segment, secret))

    return {
        "unverified_environment": environment,
        "unverified_product": product_segment,
        "unverified_checksum_ok": secrets.compare_digest(checksum, _checksum(body)),
        # Safe to log and to show: PREFIX_SECRET_CHARS of the secret and no
        # more — the same cut prefix_of() makes. Identical output, different
        # input: this one parses a presented key, prefix_of() truncates a stored
        # one without trusting its shape.
        "display_prefix": SEPARATOR.join(
            (issuer, environment, product_segment, secret[:PREFIX_SECRET_CHARS])
        ),
    }


def mask(key: str) -> str:
    """Render a key for a log line or a support ticket: prefix, then nothing."""
    parsed = parse_for_logging(key)
    return f"{parsed['display_prefix']}…" if parsed else "<malformed key>"
