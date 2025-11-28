.PHONY: help install dev-install up down logs test lint format clean

help:
	@echo "Sports Betting Intelligence Platform - Makefile Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install       - Install production dependencies with Poetry"
	@echo "  make dev-install   - Install all dependencies including dev tools"
	@echo ""
	@echo "Docker:"
	@echo "  make up            - Start all Docker services"
	@echo "  make down          - Stop all Docker services"
	@echo "  make logs          - View logs from all services"
	@echo "  make restart       - Restart all services"
	@echo ""
	@echo "Development:"
	@echo "  make run           - Run backend locally (requires Docker DBs)"
	@echo "  make celery        - Run Celery worker locally"
	@echo "  make frontend      - Run Next.js frontend"
	@echo ""
	@echo "Testing:"
	@echo "  make test          - Run all tests with coverage"
	@echo "  make test-unit     - Run unit tests only"
	@echo "  make test-integration - Run integration tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          - Run linters (flake8, mypy)"
	@echo "  make format        - Format code (black, isort)"
	@echo "  make check         - Run all quality checks"
	@echo ""
	@echo "Data & Models:"
	@echo "  make fetch-data    - Fetch latest match data"
	@echo "  make train         - Train ML models"
	@echo "  make backtest      - Run backtesting"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean         - Clean cache and temporary files"
	@echo "  make migrate       - Run database migrations"

# Setup
install:
	poetry install --no-dev

dev-install:
	poetry install

# Docker
up:
	docker-compose up -d
	@echo "Services started. Backend: http://localhost:8000"

down:
	docker-compose down

logs:
	docker-compose logs -f

restart:
	docker-compose restart

# Development
run:
	poetry run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

celery:
	poetry run celery -A src.data_collection.infrastructure.tasks.celery_app worker --loglevel=info

frontend:
	cd frontend && npm run dev

# Testing
test:
	poetry run pytest -v --cov=src --cov-report=html --cov-report=term-missing

test-unit:
	poetry run pytest tests/unit -v

test-integration:
	poetry run pytest tests/integration -v

# Code Quality
lint:
	poetry run flake8 src tests
	poetry run mypy src

format:
	poetry run black src tests
	poetry run isort src tests

check: format lint test

# Data & Models
fetch-data:
	poetry run python scripts/fetch_matches.py

train:
	poetry run python scripts/train_models.py

backtest:
	poetry run python scripts/run_backtest.py

# Utilities
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf .coverage

migrate:
	poetry run alembic upgrade head

migrate-create:
	@read -p "Enter migration message: " msg; \
	poetry run alembic revision --autogenerate -m "$$msg"
