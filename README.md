# Semantic Search API

> AI-powered semantic search for e-commerce — find products by meaning, not just keywords.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Traditional keyword search fails when customers search *"affordable running shoes for wide feet"* but your catalog says *"budget athletic footwear, wide fit"*. This API bridges that gap using Google Gemini embeddings and Qdrant vector search to match intent, not just words.

---

## Features

- 🤖 **Semantic Search** — Google Gemini embeddings understand natural language queries
- ⚡ **Vector Similarity** — Qdrant powers fast, accurate nearest-neighbor search
- 🛒 **WooCommerce Ready** — Real-time product sync via webhooks
- 🔐 **JWT Auth** — License key authentication with per-domain validation
- 📊 **Usage Analytics** — Per-client quota tracking and dashboard
- 🧠 **Redis Caching** — Reduces latency for repeated or similar queries

---

## How It Works

```
Client Query → JWT Auth → Gemini Embedding → Qdrant Vector Search → Ranked Results
                                                       ↑
                                          WooCommerce Webhook Sync
```

Products are embedded once on ingestion and stored in Qdrant. At query time, the search term is embedded and compared against the product vectors — results are ranked by semantic similarity, not keyword overlap.

---

## Quick Start

### Prerequisites

| Dependency | Version |
|------------|---------|
| Python | 3.9+ |
| MySQL | 8.0+ |
| Redis | 6.0+ |
| Qdrant | latest |

### Installation

```bash
# 1. Clone and enter the project
git clone https://github.com/your-org/semantic-search.git
cd semantic-search

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt        # Production
pip install -r requirements-dev.txt   # Development (adds linting, testing tools)

# 4. Configure environment
cp .env.example .env
# Edit .env — see Configuration section below

# 5. Install pre-commit hooks (dev only)
pre-commit install
```

### Configuration

Copy `.env.example` to `.env` and fill in your values:

```env
# ── Database ──────────────────────────────────────────
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=your_username
MYSQL_PASSWORD=your_password
MYSQL_DB=semanticsearch

# ── Vector Database ───────────────────────────────────
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=products   # Name of your Qdrant collection

# ── AI / Embeddings ──────────────────────────────────
GEMINI_API_KEY=your_gemini_api_key   # Get from console.cloud.google.com

# ── Authentication ────────────────────────────────────
JWT_SECRET=your_super_secret_jwt_key  # Use a strong random string (32+ chars)

# ── Cache ─────────────────────────────────────────────
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# ── WooCommerce ───────────────────────────────────────
WC_WEBHOOK_SECRET=your_webhook_secret  # Set same value in WooCommerce settings
```

> ⚠️ Never commit your `.env` file. It's in `.gitignore` by default.

### Running the Server

```bash
# Development (auto-reload on code changes)
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Once running, open the interactive docs:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## API Reference

All endpoints require `Authorization: Bearer <your_license_key>` unless noted.

### Search

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/search` | Semantic product search |

### Product Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ingest` | Bulk ingest products |
| `POST` | `/ingest/delete` | Remove a product |
| `POST` | `/sync/batch` | Batch sync products |
| `GET`  | `/sync/status` | Check sync status |

### Webhooks (WooCommerce)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/webhook/product-created` | Handle new product |
| `POST` | `/webhook/product-updated` | Handle product update |
| `POST` | `/webhook/product-deleted` | Handle product deletion |

### Dashboard

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/dashboard/client/{client_id}` | Client usage and analytics |

---

## Rate Limits

| Plan | Monthly Searches |
|------|-----------------|
| Starter | 10,000 |
| Growth | 100,000 |
| Pro | 500,000 |

---

## Project Structure

```
semantic-search/
├── backend/
│   └── app/
│       ├── routers/          # API route handlers
│       ├── services/         # Business logic (search, embedding, sync)
│       ├── config.py         # Settings and env loading
│       └── main.py           # FastAPI app entry point
├── scripts/                  # One-off and maintenance scripts
├── tests/                    # Test suite (mirrors backend/ structure)
├── .github/
│   ├── workflows/            # CI/CD pipelines
│   └── PULL_REQUEST_TEMPLATE.md
├── requirements.txt          # Production dependencies
├── requirements-dev.txt      # Dev dependencies
├── pyproject.toml            # Tool configuration (black, isort, mypy)
├── .env.example              # Environment variable template
└── README.md
```

---

## Development

### Code Quality Tools

| Tool | Purpose | Run with |
|------|---------|----------|
| `black` | Code formatting | `black backend/` |
| `isort` | Import sorting | `isort backend/` |
| `flake8` | Linting | `flake8 backend/` |
| `mypy` | Type checking | `mypy backend/` |
| `bandit` | Security scanning | `bandit -r backend/` |
| `safety` | Dependency CVEs | `safety check` |

Pre-commit hooks run all of these automatically before each commit.

### Running Tests

```bash
# Full test suite
pytest

# With coverage report
pytest --cov=backend --cov-report=html

# Single file
pytest tests/test_search.py

# Verbose output
pytest -v
```

### Before Opening a PR

- [ ] Tests pass (`pytest`)
- [ ] No linting errors (`flake8 backend/`)
- [ ] Type checks pass (`mypy backend/`)
- [ ] New features have tests
- [ ] `.env.example` updated if new env vars were added

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes with clear, atomic commits using [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat: add multi-language search support`
   - `fix: resolve timeout on large batch ingestion`
   - `docs: add webhook setup guide`
4. Push and open a Pull Request against `main`
5. Fill in the PR template — describe what changed, why, and how to test it

Please keep PRs focused on a single concern. Large, unrelated changes are harder to review and slower to merge.

---

## Security

- Do **not** open a GitHub issue for security vulnerabilities
- Email **security@semanticsearch.com** with details
- We aim to respond within 48 hours

---

## Support

| Channel | Link |
|---------|------|
| 📧 Email | support@semanticsearch.com |
| 🐛 Bug reports | [GitHub Issues](https://github.com/your-org/semantic-search/issues) |
| 📖 Full docs | [ReadTheDocs](https://semantic-search.readthedocs.io/) |

---

## License

MIT License — see [LICENSE](LICENSE) for details.