import uuid
import os
from datetime import datetime, timedelta
from typing import Optional
from jose import jwt, JWTError
from sqlalchemy.orm import Session
from sqlalchemy import text

SECRET_KEY = os.getenv("JWT_SECRET", "change-this-in-production")
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
    valid_days: int = 365
) -> str:
    """
    Generate a JWT license key for a client and store it in MySQL.
    The JWT contains everything needed to authenticate a request
    without hitting the database on every search.
    """
    limits     = PLAN_LIMITS.get(plan, PLAN_LIMITS["starter"])
    license_id = str(uuid.uuid4())
    expires_at = datetime.utcnow() + timedelta(days=valid_days)

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

    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    # Store in MySQL
    db.execute(text("""
        INSERT INTO license_keys
            (id, client_id, license_key, allowed_domain, product_limit,
             search_limit_per_month, expires_at)
        VALUES
            (:id, :client_id, :token, :domain, :product_limit,
             :search_limit, :expires_at)
    """), {
        "id":            license_id,
        "client_id":     client_id,
        "token":         token,
        "domain":        allowed_domain,
        "product_limit": limits["product_limit"],
        "search_limit":  limits["search_limit_per_month"],
        "expires_at":    expires_at
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
               lk.search_limit_per_month,
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

    return {
        "client_id":     client_id,
        "license_id":    license_id,
        "plan":          payload.get("plan"),
        "domain":        payload.get("domain"),
        "product_limit": result.product_limit,
        "search_limit":  result.search_limit_per_month,
        "client_name":   result.name,
        "license_expires": expires
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