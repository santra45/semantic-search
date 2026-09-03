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

# Interactive API docs — /docs (Swagger UI), /redoc, /openapi.json.
#
# OFF by default, which means FastAPI never REGISTERS the routes: they 404 the
# same way a typo does, rather than 403-ing and confirming there is something
# there. That matters more than it sounds — the schema is a complete map of
# every licensing, sync and admin endpoint, their payload shapes and their
# auth headers, handed to anyone who asks. It is a reconnaissance document.
#
# Set AICHATBOT_ENABLE_DOCS=1 for local development. Leaving it set in
# production is reported as a finding on the console's System screen.
ENABLE_API_DOCS = os.getenv("AICHATBOT_ENABLE_DOCS", "").lower() in ("1", "true", "on", "yes")
