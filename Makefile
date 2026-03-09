.PHONY: help install install-dev test lint format security clean run dev docker-build docker-run

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code"
	@echo "  security     - Run security checks"
	@echo "  clean        - Clean cache and build files"
	@echo "  run          - Run production server"
	@echo "  dev          - Run development server"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run with Docker"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

# Testing
test:
	pytest tests/ -v --cov=backend --cov-report=html --cov-report=term

test-fast:
	pytest tests/ -v -x --ff

# Code Quality
lint:
	flake8 backend/ scripts/
	mypy backend/
	bandit -r backend/ -f json -o bandit-report.json

format:
	black backend/ scripts/
	isort backend/ scripts/

# Security
security:
	safety check --json --output safety-report.json
	bandit -r backend/

# Cleanup
clean:
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name "*.pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	rm -f bandit-report.json safety-report.json

# Development
dev:
	uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

run:
	uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --workers 4

# Docker
docker-build:
	docker build -t semantic-search-api .

docker-run:
	docker run -p 8000:8000 --env-file .env semantic-search-api

# Database
db-init:
	mysql -u root -p < scripts/init_database.sql

db-migrate:
	alembic upgrade head

# Dependencies
update-deps:
	pip-compile requirements.in
	pip-compile requirements-dev.in

# Pre-commit
pre-commit:
	pre-commit run --all-files
