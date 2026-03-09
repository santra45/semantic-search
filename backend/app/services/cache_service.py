import redis
import json
import hashlib
import os

# Connect to Redis
r = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True    # returns strings not bytes
)

def make_key(prefix: str, text: str) -> str:
    """
    Hash the text so long queries don't become huge keys.
    'blue dress for a party' → 'search:client_A:a3f2b1...'
    Uses SHA-256 for better collision resistance with minimal performance cost.
    """
    text_hash = hashlib.sha256(text.lower().strip().encode()).hexdigest()
    return f"{prefix}:{text_hash}"


def get_cached_embedding(query: str):
    """Returns cached embedding vector or None if not cached."""
    key  = make_key("embed", query)
    data = r.get(key)

    if data:
        return json.loads(data)   # list of floats
    return None


def set_cached_embedding(query: str, vector: list):
    """Cache an embedding vector for 24 hours."""
    key = make_key("embed", query)
    r.setex(key, 86400, json.dumps(vector))   # 86400 = 24 hours


def get_cached_results(client_id: str, query: str):
    """Returns cached search results or None if not cached."""
    key  = make_key(f"search:{client_id}", query)
    data = r.get(key)

    if data:
        return json.loads(data)   # list of result dicts
    return None


def set_cached_results(client_id: str, query: str, results: list):
    """Cache search results for 1 hour."""
    key = make_key(f"search:{client_id}", query)
    r.setex(key, 3600, json.dumps(results))   # 3600 = 1 hour


def invalidate_client_results(client_id: str):
    """
    Clear all cached search results for a client.
    Called when a product is added/updated/deleted via webhook
    so stale results don't get served.
    """
    # Scan for matching keys and delete them
    cursor = 0
    deleted = 0
    while True:
        cursor, keys = r.scan(cursor, match=f"search:{client_id}:*", count=100)
        if keys:
            r.delete(*keys)
            deleted += len(keys)
        if cursor == 0:
            break

    print(f" Cache: cleared {deleted} cached results for {client_id}")
    return deleted