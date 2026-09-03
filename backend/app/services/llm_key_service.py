"""Unwrapping the provider secrets a merchant stores in their module config.

THE LICENCE KEY IS THE KEY-ENCRYPTION KEY
-----------------------------------------
Every merchant-supplied secret in this system — the LLM API key, the embedding
API key, the Magento admin-credentials blob — is encrypted on the PHP side
under the licence key the merchant pasted into the same settings screen, and is
opened here under sha256(that licence key). The licence key is never stored
next to the ciphertext: it arrives on the request, and it is the only thing
that can open the blob.

That makes the licence key a KEK, and it makes reissuing a licence a
DESTRUCTIVE operation on every blob wrapped under the old one. See _kek() for
the re-wrap requirement that falls out of it.

WHY THIS FILE IS SO LOUD ABOUT FAILING
--------------------------------------
decrypt_key returns None on failure and does not raise, and every one of its
nine call sites reads None as "the merchant configured nothing". For Google —
the provider behind every embedding and the default chat model — that is not an
error at any layer below either:

    embedder.get_client(None)             -> genai.Client(api_key=None)
    llm_completion_service.py:72          -> genai.Client() with no key
    agents/llm_factory.py:104             -> api_key or GEMINI_API_KEY

All three fall through to the SERVER's key from the environment and succeed.
So a tenant whose blobs stopped opening keeps working perfectly, on our API
budget, and the first symptom is the Google bill a month later. Nothing above
this module is positioned to notice, because a router cannot tell "no key
configured" from "key would not open" — both arrive as None.

Hence the warnings below. They live here rather than at the call sites because
this is the single point all nine funnel through, and because the interesting
fields (which KEK, which tenant) are only assembled here.

WHAT IS DELIBERATELY *NOT* LOGGED
---------------------------------
Never the licence key, never the ciphertext, never the decrypted plaintext, and
never str(exc) from the decrypt path — see _failure_reason() for the specific
exception whose own message quotes a byte of the plaintext.
"""

from __future__ import annotations

import base64
import binascii
import logging
from hashlib import sha256
from typing import Optional

from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

from backend.app.services import license_key as license_key_format
from backend.app.services import request_context

logger = logging.getLogger(__name__)


# How much of the KEK digest goes into a log line.
#
# What this prints is a prefix of sha256(licence key) — the SAME digest
# licences.key_hash stores — so an operator can match a warning here against a
# row in `licences` while neither the licence key nor the merchant's provider
# key ever appears in the log. Twelve hex characters is 48 bits: enough that
# two tenants never collide in a day's logs, far too little to be worth
# anything against a 192-bit key.
#
# It is also the one field that states the reissue diagnosis outright, because
# the whole failure IS "the digest changed": two warnings for the same
# client_id carrying two different kek= values is a reissued licence and
# nothing else.
_KEK_LOG_CHARS = 12

# v1 licence keys are JWTs. Recognised by the base64 of '{"' that every JWT
# header starts with — used ONLY to label a log line, never to pick a code path
# (the same rule license_key.parse_for_logging() spells out at length: the
# string is attacker-controlled until the database says otherwise).
_JWT_PREFIX = "eyJ"


def _kek(license_key: str) -> bytes:
    """The AES key for a blob: sha256 of the presented licence key, raw bytes.

    ── RE-WRAP REQUIREMENT. READ BEFORE ISSUING A v2 LICENCE TO A LIVE TENANT ─

    This digest, and therefore every stored ciphertext, is a function of the
    licence key STRING. A tenant moving from a v1 JWT to an opaque czg_ key
    gets a completely different digest, so every blob already sitting in their
    module config — LLM key, embedding key, Magento admin credentials — stops
    opening the moment the new key is presented.

    Nothing about that failure is visible from the merchant's side. The module
    keeps sending the same header, decrypt_key returns None, and the request
    succeeds on OUR server key (see the module docstring for the three
    fall-through sites). The merchant's settings screen still shows a key
    saved. The only evidence is the warning decrypt_key now writes, and the
    bill.

    So: ISSUING A v2 LICENCE FOR A TENANT THAT HAS STORED BLOBS MUST RE-WRAP
    THOSE BLOBS UNDER THE NEW KEY, in the same save that changes the licence —
    that is the only moment both the old and the new key exist together, and
    once the old one is gone the ciphertext is unrecoverable.

    There is precedent to copy rather than a design to invent: commit 4fa0698,
    "fix(semantic-search-woo): re-wrap stored keys when the licence key
    changes", solved exactly this on the WooCommerce side with
    store_provider_key(), which handles the three cases per key — a new raw key
    typed (wrap under the licence now in force), the licence changed with no
    new key (unwrap with the OLD licence, re-wrap under the new one), and
    nothing changed (leave the blob alone) — and clears a blob that cannot be
    unwrapped with the old licence rather than leaving dead ciphertext behind,
    telling the merchant to paste the key in again.

    That work reaches into PHP in the Magento modules and the WooCommerce
    plugins and is out of scope for this file. What is in scope is that the
    failure is no longer silent, so nobody has to discover it from an invoice.
    """
    return sha256(license_key.strip().encode("utf-8")).digest()


# ── Log fields: identify the failure without leaking anything ────────────────
#
# Both helpers are public rather than underscored because embedding_key_service
# builds the same fields for its own server-key fallback line. They belong to
# this pair of modules; nothing else should need them.


def licence_for_log(license_key: Optional[str]) -> str:
    """Which licence key was used as the KEK, in a form safe to write down."""
    if not license_key or not license_key.strip():
        return "licence=<absent> kek=<none>"

    parsed = license_key_format.parse_for_logging(license_key)
    if parsed:
        # display_prefix is issuer_env_product_first4 — the same shape
        # license_key.prefix_of() derives, and safe to display. Note the
        # database now stores the WHOLE key in licences.licence_key; this line
        # must keep going through the parser and never read that column.
        shape = f"v2 {parsed['display_prefix']}"
    elif license_key.strip().startswith(_JWT_PREFIX):
        # No part of a JWT goes in the line. The whole token is the credential,
        # and its payload is base64 of the tenant's claims, i.e. readable.
        shape = "v1 JWT"
    else:
        shape = "unrecognised format"

    return f"licence={shape} kek={_kek(license_key).hex()[:_KEK_LOG_CHARS]}"


def tenant_for_log() -> str:
    """Who the failing request belongs to, read from the ambient context.

    Only populated on the v2 auth path: the chokepoints bind the context ONLY
    when auth_path == v2, so a v1 JWT request reports nothing here. That is the
    right way round for this warning — a v1 request's KEK has not changed, so a
    failure on it is a corrupt option value rather than the reissue case, and
    licence_for_log() already says which key was tried.

    get_context() rather than a private read of the ContextVar, on purpose. It
    logs at ERROR when an absent context is read from inside a streaming body,
    and that line would be TRUE here, not a false alarm: a streaming body that
    cannot see the context is losing its usage rows regardless of what this
    module is doing. (Measured today: all nine decrypt_key call sites sit in
    handler bodies, none inside a generator, so it does not fire.)
    """
    ctx = request_context.get_context()
    if not isinstance(ctx, dict):
        return "tenant=<no v2 context bound>"

    # NAMED SCALARS ONLY, never the dict. get_context()'s docstring says it in
    # as many words: the context carries the PLAINTEXT licence key under
    # "license_key", because that key is the KEK ~20 call sites need. This is
    # the module that would be handing out the KEK if it ever logged the whole
    # thing.
    fields = [
        f"{name}={ctx.get(name)}"
        for name in ("client_id", "site_id", "subscription_id", "product_code")
        if ctx.get(name)
    ]
    return " ".join(fields) if fields else "tenant=<context bound but unnamed>"


def _failure_reason(exc: BaseException) -> str:
    """Why the unwrap failed, in words, WITHOUT the exception's own message.

    Deliberately not str(exc). UnicodeDecodeError renders as "can't decode byte
    0x8f in position 3", and that byte comes out of the decrypted buffer — on a
    partially-correct unwrap it is a byte of the merchant's plaintext API key,
    the one thing this module must never write to a log. The exception type
    plus a hand-written cause carries everything an operator needs and nothing
    they must not have.
    """
    name = type(exc).__name__

    # binascii.Error is a SUBCLASS of ValueError and its message for a
    # mis-padded base64 string is literally "Incorrect padding", so it has to
    # be tested before the padding branch below or every malformed blob would
    # be misdiagnosed as a wrong KEK.
    if isinstance(exc, binascii.Error):
        # Spelled out rather than taken from __name__, which renders as the
        # bare and meaningless "Error".
        return (
            "binascii.Error: the stored value is not valid base64, so it was "
            "never a blob this module wrote - a truncated paste, or a "
            "plaintext key saved into the ciphertext column"
        )

    if isinstance(exc, UnicodeDecodeError):
        return (
            f"{name}: the cipher ran and the padding happened to check out, "
            "but the result is not UTF-8 - a wrong key-encryption key that got "
            "unlucky (1 in 256 of them do)"
        )

    if isinstance(exc, ValueError):
        # pycryptodome's unpad raises ValueError("Padding is incorrect.") when
        # the last block does not decrypt to a valid PKCS#7 tail. On a wrong
        # key that happens 255 times out of 256, which makes it THE signature
        # of the reissue case rather than of a corrupted blob.
        if "padding" in str(exc).lower():
            return (
                f"{name}: PKCS#7 padding did not check out, which is exactly "
                "what a WRONG key-encryption key looks like - the licence key "
                "presented is not the one this blob was wrapped under"
            )
        return (
            f"{name}: the ciphertext is not the shape this module writes - "
            "bad IV length or a truncated payload"
        )

    return name


def decrypt_key(
    encrypted_blob: Optional[str],
    license_key: Optional[str],
    *,
    purpose: str = "stored key",
) -> Optional[str]:
    """Unwrap one merchant-supplied secret. None means "could not" — loudly.

    *purpose* only labels the log line ("llm key", "embedding key", ...). It is
    keyword-only with a default because eight of the nine call sites live in
    files this change does not touch and pass two positional arguments; adding
    a required argument would have been a nine-file diff whose failure mode is
    a TypeError at runtime on a path nobody exercises in dev.

    Returning None rather than raising is deliberate and UNCHANGED. A corrupt
    option value in one merchant's admin must degrade to the server key, not
    500 every sync and every search for that tenant. What changed is that it
    now says so at WARNING, with enough to identify the tenant and the key.
    """
    # Nothing configured is not a failure. Most installs have never set a
    # provider key at all and every caller correctly reads None as "use the
    # server's". If this logged, the warning below would drown in it within a
    # day and stop meaning anything — which is how the ERROR marker in
    # usage_service got filtered out before it ever carried signal.
    if not encrypted_blob:
        return None

    # The KEK itself is missing: a caller reached here without the plaintext
    # licence key. Distinct from a decrypt failure and worth its own line,
    # because there is nothing wrong with the ciphertext — the key to open it
    # was never passed. Both auth chokepoints set license_data["license_key"]
    # explicitly to prevent exactly this (see the comment above that assignment
    # in magento/chatbot/routers/common.py), and routers/webhooks.py documents
    # at length that it is the one path that cannot supply it under v2.
    if not license_key or not license_key.strip():
        logger.warning(
            "provider key: %s NOT DECRYPTED - no licence key was passed, so "
            "there is no key-encryption key to derive from. The caller reads "
            "this as 'no key configured' and falls back to the server's own "
            "API key, so this tenant's spend lands on OUR budget. %s",
            purpose,
            tenant_for_log(),
        )
        return None

    try:
        # The PHP side stores "<base64>.<uuid>"; the dot-suffix is a version
        # marker, not ciphertext.
        blob = encrypted_blob.split(".")[0] if "." in encrypted_blob else encrypted_blob

        data = base64.b64decode(blob)
        iv, payload = data[:16], data[16:]

        cipher = AES.new(_kek(license_key), AES.MODE_CBC, iv)
        return unpad(cipher.decrypt(payload), AES.block_size).decode("utf-8")

    except Exception as exc:
        logger.warning(
            "provider key: %s FAILED TO DECRYPT (%s). %s %s. The caller reads "
            "this as 'no key configured' and falls back to the server's own "
            "API key, so this tenant's spend lands on OUR budget with no other "
            "symptom until the invoice arrives. FIRST THING TO CHECK: was this "
            "licence reissued? The key is the key-encryption key, so a new key "
            "orphans every blob wrapped under the old one - see the re-wrap "
            "requirement in llm_key_service._kek().",
            purpose,
            _failure_reason(exc),
            tenant_for_log(),
            licence_for_log(license_key),
        )
        return None
