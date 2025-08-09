# MeshAI SDK Development Makefile

.PHONY: help install test test-all format lint clean build deploy

# Default target
help:
	@echo "MeshAI SDK Development Commands"
	@echo "================================"
	@echo "install     Install dependencies for development"
	@echo "test        Run quick tests"
	@echo "test-all    Run all tests including integration and performance"
	@echo "format      Format code with black and isort"
	@echo "lint        Run linting and type checking"
	@echo "security    Run security checks"
	@echo "clean       Clean up build artifacts and cache"
	@echo "build       Build Docker image"
	@echo "deploy      Deploy to Cloud Run via Cloud Build"
	@echo "local       Start services locally with docker-compose"
	@echo "integration Run integration tests against live API"

# Development setup
install:
	pip install -e ".[dev,all]"
	pre-commit install

# Testing
test:
	python scripts/run-tests.py --quick

test-all:
	python scripts/run-tests.py --all

integration:
	pytest tests/integration/ -v -m integration

performance:
	python scripts/run-tests.py --performance

# Code quality
format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

lint:
	python scripts/run-tests.py --lint

security:
	python scripts/run-tests.py --security

# Build and deployment
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/

build:
	docker build -t meshai-sdk:latest .

deploy:
	gcloud builds submit --config deployment/gcp/cloudbuild.yaml .

# Local development
local:
	docker-compose -f docker-compose.dev.yml up -d

local-stop:
	docker-compose -f docker-compose.dev.yml down

# Database operations
db-init:
	python scripts/init-database.py

db-migrate:
	cd src/meshai/database/migrations && alembic upgrade head

# Health checks
health:
	@echo "Checking service health..."
	@curl -s https://api.meshai.dev/health | jq . || echo "API not accessible"

# Coverage
coverage:
	python scripts/run-tests.py --coverage
	@echo "Coverage report generated in htmlcov/index.html"

# Pre-commit
pre-commit:
	pre-commit run --all-files