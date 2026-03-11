import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
QDRANT_HOST     = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLL     = os.getenv("QDRANT_COLLECTION", "local_shared_products")
EMBED_MODEL     = "gemini-embedding-001"
EMBED_DIM       = 3072

# WordPress fallback configuration
WORDPRESS_API_URL = os.getenv("WC_LOCAL_URL")
FALLBACK_SCORE_THRESHOLD = float(os.getenv("FALLBACK_SCORE_THRESHOLD", 0.58))