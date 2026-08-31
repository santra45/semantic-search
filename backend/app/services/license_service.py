import uuid
import os
from datetime import datetime, timedelta
from typing import Optional
from jose import jwt, JWTError
from sqlalchemy.orm import Session
from sqlalchemy import text

from backend.app.services import catalog

SECRET_KEY = os.getenv("JWT_SECRET")
ALGORITHM  = "HS256"

PLAN_LIMITS = {
    "starter": {
        "product_limit":        500,
        "search_limit_per_month": 10000
    },
    "growth": {
        "product_limit":        5000,
        "search_limit_per_month": 100000
    },
    "pro": {
        "product_limit":        25000,
        "search_limit_per_month": 500000
    }
}

# PLAN_LIMITS above is v1's single ladder and is now frozen: it feeds only the
# JWT path in this module, which the v2 schema replaces. It is deliberately no
# longer cross-checked against catalog, because there is nothing left to check
# it against - catalog.PLANS split into INDEX_PLANS (per site) and MODULE_PLANS
# (per subscription), and neither is the shape assert_plans_match compared.
#
# Nothing new should read these numbers. New allowances come from
# catalog.catalogue_limit_for() and catalog.request_limit_for(), which the v2
# services already use.


# ─── Generate ──────────────────────────────────────────────────────────────────

def create_client(db: Session, name: str, email: str, plan: str = "starter") -> dict:
    """Create a new client record in MySQL."""
    # Check if email already exists
    existing_client = db.execute(text("""
        SELECT id, name, email, plan FROM clients 
        WHERE email = :email AND is_active = 1
    """), {"email": email}).fetchone()
    
    if existing_client:
        raise ValueError(f"A client with email '{email}' already exists. Please use a different email address.")
    
    client_id = str(uuid.uuid4())

    db.execute(text("""
        INSERT INTO clients (id, name, email, plan)
        VALUES (:id, :name, :email, :plan)
    """), {"id": client_id, "name": name, "email": email, "plan": plan})

    db.commit()

    return {"id": client_id, "name": name, "email": email, "plan": plan}


def get_client_by_email(db: Session, email: str) -> Optional[dict]:
    """Get a client by email address."""
    result = db.execute(text("""
        SELECT id, name, email, plan, is_active FROM clients 
        WHERE email = :email
    """), {"email": email}).fetchone()
    
    if not result:
        return None
    
    return {
        "id": result.id,
        "name": result.name,
        "email": result.email,
        "plan": result.plan,
        "is_active": bool(result.is_active)
    }


def generate_license_key(
    db: Session,
    client_id: str,
    allowed_domain: str,
    plan: str = "starter",
    valid_days: int = 365,
    product_code: Optional[str] = None,
) -> str:
    """
    Generate a JWT license key for a client and store it in MySQL.
    The JWT contains everything needed to authenticate a request
    without hitting the database on every search.

    *product_code* scopes the key to one thing the customer bought — see
    catalog.PRODUCTS. It matters because several Magento modules share backend
    endpoints, so the route a request arrives on cannot identify the product;
    the key is the only trustworthy carrier of that identity. The platform is
    derived from the product rather than passed in, so a caller can't mint a
    key claiming a (platform, product) pair that doesn't exist.

    Passing None issues a legacy all-products key. That path exists for the
    keys already deployed in customer stores and for internal tooling; new
    keys from onboarding always carry a product.
    """
    limits     = PLAN_LIMITS.get(plan, PLAN_LIMITS["starter"])
    license_id = str(uuid.uuid4())
    expires_at = datetime.utcnow() + timedelta(days=valid_days)

    # Raises ValueError with customer-safe wording on an unknown code.
    platform = catalog.validate_selection(product_code) if product_code else None

    payload = {
        "license_id":     license_id,
        "client_id":      client_id,
        "plan":           plan,
        "domain":         allowed_domain,
        "product_limit":  limits["product_limit"],
        "search_limit":   limits["search_limit_per_month"],
        "exp":            expires_at,
        "iat":            datetime.utcnow(),
    }

    # Only stamp the claims when there's a product. An explicit null in the JWT
    # and an absent claim would both have to be handled on the read side, and
    # absent is the one that already matches every key issued before this
    # change — so there's exactly one shape for "unscoped".
    if product_code:
        payload["product_code"] = product_code
        payload["platform"]     = platform

    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    # Store in MySQL
    db.execute(text("""
        INSERT INTO license_keys
            (id, client_id, license_key, allowed_domain, product_limit,
             search_limit_per_month, expires_at, product_code, platform)
        VALUES
            (:id, :client_id, :token, :domain, :product_limit,
             :search_limit, :expires_at, :product_code, :platform)
    """), {
        "id":            license_id,
        "client_id":     client_id,
        "token":         token,
        "domain":        allowed_domain,
        "product_limit": limits["product_limit"],
        "search_limit":  limits["search_limit_per_month"],
        "expires_at":    expires_at,
        "product_code":  product_code,
        "platform":      platform,
    })

    db.commit()

    return token


# ─── Validate ──────────────────────────────────────────────────────────────────

def validate_license_key(token: str, db: Session) -> dict:
    """
    Validate a license key and return client info.
    Raises ValueError with a clear message if invalid.
    """
    # Step 1 — decode JWT
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as e:
        raise ValueError(f"Invalid license key: {e}")

    client_id  = payload.get("client_id")
    license_id = payload.get("license_id")
    expires    = datetime.fromtimestamp(payload.get("exp")).isoformat()

    # Step 2 — check license is active in MySQL
    result = db.execute(text("""
        SELECT lk.is_active, lk.expires_at, lk.product_limit,
               lk.search_limit_per_month, lk.product_code, lk.platform,
               c.is_active as client_active, c.name
        FROM license_keys lk
        JOIN clients c ON c.id = lk.client_id
        WHERE lk.id = :license_id
        AND   lk.client_id = :client_id
    """), {"license_id": license_id, "client_id": client_id}).fetchone()

    if not result:
        raise ValueError("License key not found")

    if not result.is_active:
        raise ValueError("License key is inactive")

    if not result.client_active:
        raise ValueError("Client account is inactive")

    if result.expires_at and result.expires_at < datetime.utcnow():
        raise ValueError("License key has expired")

    # The row wins over the JWT claim. A key's product can be changed in the
    # database — an upgrade, a correction — and the token in the customer's
    # store is a snapshot from issue time that nobody can reissue on demand.
    # Reading the row means such a change takes effect without asking the
    # customer to paste a new key.
    product_code = result.product_code or payload.get("product_code")
    platform     = result.platform     or payload.get("platform")

    return {
        "client_id":     client_id,
        "license_id":    license_id,
        "plan":          payload.get("plan"),
        "domain":        payload.get("domain"),
        "product_limit": result.product_limit,
        "search_limit":  result.search_limit_per_month,
        "client_name":   result.name,
        "license_expires": expires,
        # None on keys issued before per-product licensing. Callers must treat
        # that as "legacy, all products" rather than as an error — those keys
        # are live in customer stores and cannot be recalled.
        "product_code":  product_code,
        "platform":      platform,
    }


# ─── Usage Tracking ────────────────────────────────────────────────────────────

def increment_search_count(db: Session, client_id: str):
    """Increment search count for current month."""
    month = datetime.utcnow().strftime("%Y-%m")

    db.execute(text("""
        INSERT INTO usage_logs (id, client_id, month, search_count)
        VALUES (:id, :client_id, :month, 1)
        ON DUPLICATE KEY UPDATE search_count = search_count + 1
    """), {
        "id":        str(uuid.uuid4()),
        "client_id": client_id,
        "month":     month
    })

    db.commit()


def get_monthly_usage(db: Session, client_id: str) -> dict:
    """Get current month's usage for a client."""
    month = datetime.utcnow().strftime("%Y-%m")

    result = db.execute(text("""
        SELECT search_count, ingest_count
        FROM usage_logs
        WHERE client_id = :client_id AND month = :month
    """), {"client_id": client_id, "month": month}).fetchone()

    if not result:
        return {"search_count": 0, "ingest_count": 0, "month": month}

    return {
        "search_count": result.search_count,
        "ingest_count": result.ingest_count,
        "month":        month
    }


def check_search_quota(db: Session, client_id: str, search_limit: int) -> bool:
    """Returns True if client is within their monthly search limit."""
    usage = get_monthly_usage(db, client_id)
    return usage["search_count"] < search_limit


def extract_license_key_from_authorization(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None

    parts = authorization.strip().split()
    if len(parts) != 2:
        return None

    scheme, token = parts[0], parts[1]
    if scheme.lower() != "bearer":
        return None

    return token or None


# ─── Search Logging ────────────────────────────────────────────────────────────

def log_search(
    db: Session,
    client_id: str,
    query: str,
    results_count: int,
    response_time_ms: int,
    cached: bool
):
    """Insert a row into search_logs table."""
    db.execute(text("""
        INSERT INTO search_logs
            (id, client_id, query, results_count, response_time_ms, cached)
        VALUES
            (:id, :client_id, :query, :results_count, :response_time_ms, :cached)
    """), {
        "id":               str(uuid.uuid4()),
        "client_id":        client_id,
        "query":            query,
        "results_count":    results_count,
        "response_time_ms": response_time_ms,
        "cached":           1 if cached else 0
    })
    db.commit()


# ─── License Lookup ────────────────────────────────────────────────────────────

def get_client_license(db: Session, client_id: str) -> dict:
    """
    DEPRECATED. Cannot be ported to the v2 schema. Do not add callers.

    WHY IT CANNOT SURVIVE
    ---------------------
    This is the only function in the tree that reads a customer's PLAINTEXT
    licence key back out of the database (`lk.license_key`), and the field it
    returns it in is load-bearing rather than incidental: `llm_key_service.
    decrypt_key` derives its AES key as sha256(license_key), so the licence key
    is the key-encryption key for every merchant-supplied LLM and embedding key
    on that store. webhooks.py feeds `license_data["license_key"]` straight
    into `resolve_embedding_key`, and that is the only reason it calls this at
    all.

    schema_v2 stores `licences.key_hash` — a SHA-256 hex digest — and has no
    plaintext column anywhere. A hash cannot be un-hashed, so after the
    migration this SELECT has nothing left to select. There is no version of
    this function that works against v2; it is not a rewrite, it is a removal.

    WHAT REPLACED IT
    ----------------
    The push method ai-product-qa-woo already uses. Its plugin hooks
    WordPress actions directly (`woocommerce_update_product`,
    `woocommerce_delete_product`, `save_post_page`, ...), defers the work to a
    scheduled single event, and calls the API itself with
    `Authorization: Bearer <license_key>` on every request — see
    AIPQA_API_Client::build_headers() in ai-product-qa-woo/includes/
    class-api-client.php. The plaintext key is therefore always in scope FROM
    THE REQUEST, so the KEK is available without any database lookup and the
    server never needs to be able to read a key back.

    semantic-search-woo is being switched to the same shape, which removes the
    last two callers. docs/webhook-migration.md is the plan, file by file.

    THE OTHER HALF: IT IS ALSO WRONG UNDER PER-PRODUCT LICENSING
    ------------------------------------------------------------
    Even a caller that only wants the domain or the limits — never the key —
    must not be moved onto this. `ORDER BY lk.expires_at DESC LIMIT 1` assumes
    one active licence per client. v2 makes that false by design: a client
    holds one subscription per module per site, each with its own licence, so
    "the newest one" is an arbitrary pick between two products that may sit on
    different plans with different quotas. It also folds every liveness gate
    into the WHERE clause, which collapses five distinct denial reasons into
    "no row" — the exact failure licensing_service's docstring calls out, where
    a deactivated site, a lapsed subscription and a key from the wrong store
    are indistinguishable to support.

    The v2 answer to "who is this caller" is
    `licensing_service.resolve_key(db, presented_key)`: it takes the key the
    caller actually presented, hashes it, and returns client_id, site_id,
    subscription_id, collection_name and both limits from their own levels,
    with a separate log line per denial reason.

    CALLERS AS OF 2026-08-27 (both die with webhooks.py)
    ---------------------------------------------------
      * backend/app/routers/webhooks.py:20   import
      * backend/app/routers/webhooks.py      process_upsert() — for the KEK
      * backend/app/routers/webhooks.py      product_deleted() — for the domain

    Nothing else in the backend imports it. The same two live on the deployed
    checkout at C:/xampp/htdocs/semantic-search. Prose references (not
    callers): ADMIN_CONSOLE_PLAN.md:162, licensing_service.py's docstring, and
    the migration report in scripts/migrate_v2_schema.py.

    Get active license data for a client by client_id.
    Raises ValueError if no active license found.
    """
    result = db.execute(text("""
        SELECT lk.id, lk.license_key, lk.is_active, lk.expires_at, lk.product_limit,
               lk.search_limit_per_month, lk.allowed_domain,
               c.name as client_name
        FROM license_keys lk
        JOIN clients c ON c.id = lk.client_id
        WHERE lk.client_id = :client_id
        AND   lk.is_active = 1
        AND   c.is_active = 1
        ORDER BY lk.expires_at DESC
        LIMIT 1
    """), {"client_id": client_id}).fetchone()

    if not result:
        raise ValueError("No active license found for client")

    if result.expires_at and result.expires_at < datetime.utcnow():
        raise ValueError("License has expired")

    return {
        "license_id": result.id,
        "client_id": client_id,
        "license_key": result.license_key,
        "client_name": result.client_name,
        "domain": result.allowed_domain,
        "product_limit": result.product_limit,
        "search_limit": result.search_limit_per_month,
        "license_expires": result.expires_at.isoformat() if result.expires_at else None
    }


# ─── Ingest Logging ────────────────────────────────────────────────────────────

def increment_ingest_count(db: Session, client_id: str, count: int = 1):
    """Increment ingest count for current month."""
    month = datetime.utcnow().strftime("%Y-%m")

    db.execute(text("""
        INSERT INTO usage_logs (id, client_id, month, ingest_count)
        VALUES (:id, :client_id, :month, :count)
        ON DUPLICATE KEY UPDATE ingest_count = ingest_count + :count
    """), {
        "id":        str(uuid.uuid4()),
        "client_id": client_id,
        "month":     month,
        "count":     count
    })

    db.commit()