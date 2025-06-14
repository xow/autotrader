# Autotrader Bot Makefile
# Provides common development tasks and build commands

# Project configuration
PYTHON := python3
PIP := pip
VENV := venv
PYTEST := pytest
BLACK := black
ISORT := isort
FLAKE8 := flake8
MYPY := mypy

# Source directories
SRC_DIR := .
TEST_DIR := tests
DOCS_DIR := docs

# Default target
.DEFAULT_GOAL := help

# Help target
.PHONY: help
help: ## Show this help message
	@echo "Autotrader Bot - Development Commands"
	@echo "===================================="
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Environment setup
.PHONY: setup
setup: ## Set up development environment
	@echo "Setting up development environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Activating virtual environment and installing dependencies..."
	. $(VENV)/bin/activate && $(PIP) install --upgrade pip
	. $(VENV)/bin/activate && $(PIP) install -r requirements.txt
	. $(VENV)/bin/activate && $(PIP) install -r requirements-dev.txt
	@echo "Setup complete! Activate with: source $(VENV)/bin/activate"

.PHONY: setup-minimal
setup-minimal: ## Set up minimal environment (production dependencies only)
	@echo "Setting up minimal environment..."
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && $(PIP) install --upgrade pip
	. $(VENV)/bin/activate && $(PIP) install -r requirements.txt
	@echo "Minimal setup complete!"

# Dependencies management
.PHONY: install
install: ## Install project dependencies
	$(PIP) install -r requirements.txt

.PHONY: install-dev
install-dev: ## Install development dependencies
	$(PIP) install -r requirements-dev.txt

.PHONY: freeze
freeze: ## Generate requirements.txt from current environment
	$(PIP) freeze > requirements.txt

.PHONY: update
update: ## Update all dependencies to latest versions
	$(PIP) install --upgrade -r requirements.txt
	$(PIP) install --upgrade -r requirements-dev.txt

# Application execution
.PHONY: run
run: ## Run the autotrader bot
	$(PYTHON) autotrader.py

.PHONY: run-debug
run-debug: ## Run the autotrader bot in debug mode
	$(PYTHON) autotrader.py --log-level DEBUG

.PHONY: run-config
run-config: ## Run with custom configuration file
	@echo "Usage: make run-config CONFIG=path/to/config.json"
	$(PYTHON) autotrader.py --config $(CONFIG)

.PHONY: daemon
daemon: ## Run the autotrader bot as a background daemon
	nohup $(PYTHON) autotrader.py > autotrader.out 2>&1 &
	@echo "Bot started in background. Check autotrader.out for output."

.PHONY: stop
stop: ## Stop the background daemon
	pkill -f "python.*autotrader.py" || echo "No running autotrader processes found"

# Testing
.PHONY: test
test: ## Run all tests
	$(PYTEST) $(TEST_DIR)/ -v

.PHONY: test-unit
test-unit: ## Run unit tests only
	$(PYTEST) $(TEST_DIR)/unit/ -v

.PHONY: test-integration
test-integration: ## Run integration tests only
	$(PYTEST) $(TEST_DIR)/integration/ -v

.PHONY: test-coverage
test-coverage: ## Run tests with coverage report
	$(PYTEST) --cov=. --cov-report=html --cov-report=term $(TEST_DIR)/

.PHONY: test-watch
test-watch: ## Run tests in watch mode (requires pytest-watch)
	ptw -- $(TEST_DIR)/

# Code quality
.PHONY: lint
lint: ## Run all linting checks
	@echo "Running flake8..."
	$(FLAKE8) $(SRC_DIR) --count --select=E9,F63,F7,F82 --show-source --statistics
	$(FLAKE8) $(SRC_DIR) --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	@echo "Running mypy..."
	$(MYPY) $(SRC_DIR) --ignore-missing-imports

.PHONY: format
format: ## Format code with black and isort
	@echo "Formatting imports with isort..."
	$(ISORT) $(SRC_DIR) $(TEST_DIR)
	@echo "Formatting code with black..."
	$(BLACK) $(SRC_DIR) $(TEST_DIR)

.PHONY: format-check
format-check: ## Check if code is properly formatted
	$(BLACK) --check $(SRC_DIR) $(TEST_DIR)
	$(ISORT) --check-only $(SRC_DIR) $(TEST_DIR)

.PHONY: type-check
type-check: ## Run type checking with mypy
	$(MYPY) $(SRC_DIR) --ignore-missing-imports

# Build and distribution
.PHONY: build
build: ## Build the project (run tests and linting)
	@echo "Building project..."
	$(MAKE) test
	$(MAKE) lint
	@echo "Build successful!"

.PHONY: clean
clean: ## Clean build artifacts and temporary files
	@echo "Cleaning build artifacts..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.log" -delete
	find . -type f -name "*.tmp" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	@echo "Clean complete!"

.PHONY: clean-data
clean-data: ## Clean training data and model files (CAUTION: This removes learned models!)
	@echo "WARNING: This will delete all training data and model files!"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	rm -f *.keras
	rm -f *.pkl
	rm -f training_data.json
	rm -f *.log
	@echo "Training data cleaned!"

# Documentation
.PHONY: docs
docs: ## Generate documentation
	@echo "Generating documentation..."
	@mkdir -p $(DOCS_DIR)
	$(PYTHON) -m pydoc -w autotrader
	@echo "Documentation generated in $(DOCS_DIR)/"

.PHONY: docs-serve
docs-serve: ## Serve documentation locally
	@echo "Starting documentation server..."
	$(PYTHON) -m http.server 8000 -d $(DOCS_DIR)

# Development utilities
.PHONY: shell
shell: ## Start Python shell with project context
	$(PYTHON) -i -c "from autotrader import *; print('Autotrader development shell ready!')"

.PHONY: jupyter
jupyter: ## Start Jupyter notebook server
	jupyter notebook

.PHONY: profile
profile: ## Profile the application performance
	$(PYTHON) -m cProfile -o profile.stats autotrader.py
	@echo "Profile saved to profile.stats"

.PHONY: benchmark
benchmark: ## Run performance benchmarks
	$(PYTHON) -m pytest benchmarks/ -v --benchmark-only

# Git utilities
.PHONY: commit
commit: ## Run pre-commit checks and commit changes
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) test
	git add -A
	@echo "All checks passed! Ready to commit."
	@echo "Run: git commit -m 'Your commit message'"

.PHONY: pre-commit
pre-commit: ## Run pre-commit hooks
	$(MAKE) format-check
	$(MAKE) lint
	$(MAKE) test

# Monitoring and maintenance
.PHONY: status
status: ## Show bot status and recent logs
	@echo "=== Autotrader Bot Status ==="
	@ps aux | grep "python.*autotrader.py" | grep -v grep || echo "No running processes found"
	@echo ""
	@echo "=== Recent Logs ==="
	@tail -20 autotrader.log 2>/dev/null || echo "No log file found"

.PHONY: logs
logs: ## Show live log output
	tail -f autotrader.log

.PHONY: health-check
health-check: ## Run system health checks
	@echo "Running health checks..."
	$(PYTHON) -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
	$(PYTHON) -c "import numpy as np; print(f'NumPy version: {np.__version__}')"
	$(PYTHON) -c "import pandas as pd; print(f'Pandas version: {pd.__version__}')"
	@echo "All dependencies are working!"

# Environment information
.PHONY: info
info: ## Show environment information
	@echo "=== Environment Information ==="
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Pip version: $$($(PIP) --version)"
	@echo "Virtual environment: $$(which python)"
	@echo "Working directory: $$(pwd)"
	@echo ""
	@echo "=== Project Files ==="
	@ls -la *.py 2>/dev/null || echo "No Python files found"
	@echo ""
	@echo "=== Dependencies ==="
	@$(PIP) list --format=columns 2>/dev/null | head -10

# Backup and restore
.PHONY: backup
backup: ## Backup training data and model files
	@echo "Creating backup..."
	@mkdir -p backups
	@tar -czf backups/autotrader-backup-$$(date +%Y%m%d-%H%M%S).tar.gz \
		*.keras *.pkl *.json *.log 2>/dev/null || true
	@echo "Backup created in backups/ directory"

.PHONY: restore
restore: ## Restore from backup (specify BACKUP=filename)
	@echo "Usage: make restore BACKUP=backups/autotrader-backup-YYYYMMDD-HHMMSS.tar.gz"
	@if [ -n "$(BACKUP)" ]; then \
		echo "Restoring from $(BACKUP)..."; \
		tar -xzf $(BACKUP); \
		echo "Restore complete!"; \
	fi

# Quick commands for common workflows
.PHONY: quick-test
quick-test: format lint test ## Quick development cycle: format, lint, and test

.PHONY: deploy-check
deploy-check: clean build docs ## Full deployment readiness check

.PHONY: fresh-start
fresh-start: clean clean-data setup ## Complete fresh start (removes all data!)

# Development workflow targets
.PHONY: dev
dev: setup run-debug ## Set up and run in development mode

.PHONY: prod
prod: build run ## Production build and run

# Show available make targets
.PHONY: targets
targets: ## List all available make targets
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'
