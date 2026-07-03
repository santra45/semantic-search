import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
QDRANT_HOST     = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", 6333))
# QDRANT_COLL removed — each client now has its own collection: products_{client_id}
EMBED_MODEL     = "gemini-embedding-001"
EMBED_DIM       = 3072

# WordPress fallback configuration
WORDPRESS_API_URL = os.getenv("WC_LOCAL_URL")
FALLBACK_SCORE_THRESHOLD = float(os.getenv("FALLBACK_SCORE_THRESHOLD", 0.58))

# Operator analytics dashboard — gate for the all-tenant ops console (Phase
# 4.4 + 4.5). Empty by default = the dashboard AND its API are LOCKED (403)
# until an operator key is configured. Set AICHATBOT_OPERATOR_KEY to a long
# random secret; it's checked against the X-Operator-Key request header.
OPERATOR_KEY = os.getenv("AICHATBOT_OPERATOR_KEY", "")