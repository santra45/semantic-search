# Semantic Search API

AI-powered semantic search API for e-commerce products using vector embeddings and advanced search capabilities.

## Features

- 🤖 **AI-Powered Search**: Uses Google Gemini embeddings for semantic understanding
- 🔍 **Vector Search**: Fast and accurate vector similarity search with Qdrant
- 🛒 **E-commerce Ready**: Optimized for product catalogs with WooCommerce integration
- 🔐 **Secure**: JWT-based authentication with domain validation
- 📊 **Analytics**: Usage tracking and quota management
- ⚡ **High Performance**: Redis caching for fast responses
- 🔄 **Webhooks**: Real-time product synchronization

## Quick Start

### Prerequisites

- Python 3.9+
- MySQL 8.0+
- Redis 6.0+
- Qdrant Vector Database

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-org/semantic-search.git
cd semantic-search
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
# Production
pip install -r requirements.txt

# Development
pip install -r requirements-dev.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Set up pre-commit hooks** (development only)
```bash
pre-commit install
```

### Configuration

Create a `.env` file with the following variables:

```env
# Database
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=your_username
MYSQL_PASSWORD=your_password
MYSQL_DB=semanticsearch

# Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=local_shared_products

# AI/ML
GEMINI_API_KEY=your_gemini_api_key

# Authentication
JWT_SECRET=your_super_secret_jwt_key

# Cache
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# WooCommerce
WC_WEBHOOK_SECRET=your_webhook_secret
```

### Running the Application

**Development:**
```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

**Production:**
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Project Structure

```
semantic-search/
├── backend/
│   └── app/
│       ├── routers/          # API endpoints
│       ├── services/         # Business logic
│       ├── config.py         # Configuration
│       └── main.py          # FastAPI app
├── scripts/                  # Utility scripts
├── tests/                    # Test suite
├── requirements.txt          # Production dependencies
├── requirements-dev.txt      # Development dependencies
├── pyproject.toml           # Project configuration
└── README.md                # This file
```

## Development

### Code Quality

This project uses several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **bandit**: Security scanning
- **safety**: Dependency vulnerability checking

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backend

# Run specific test file
pytest tests/test_search.py
```

### Security Scanning

```bash
# Security audit
bandit -r backend/

# Dependency vulnerability check
safety check
```

## API Endpoints

### Search
- `POST /search` - Semantic product search

### Product Management
- `POST /ingest` - Bulk product ingestion
- `POST /ingest/delete` - Delete product
- `POST /sync/batch` - Batch product synchronization
- `GET /sync/status` - Get sync status

### Webhooks
- `POST /webhook/product-created` - Product creation webhook
- `POST /webhook/product-updated` - Product update webhook
- `POST /webhook/product-deleted` - Product deletion webhook

### Dashboard
- `GET /dashboard/client/{client_id}` - Client dashboard data

## Authentication

The API uses JWT-based authentication with domain validation. Include your license key in the `Authorization` header:

```
Authorization: Bearer <your_license_key>
```

## Rate Limiting

API requests are rate-limited based on your subscription plan:
- **Starter**: 10,000 searches/month
- **Growth**: 100,000 searches/month
- **Pro**: 500,000 searches/month

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📧 Email: support@semanticsearch.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-org/semantic-search/issues)
- 📖 Documentation: [ReadTheDocs](https://semantic-search.readthedocs.io/)

## Security

For security concerns, please email security@semanticsearch.com rather than opening a public issue.
