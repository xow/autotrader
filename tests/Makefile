# Makefile for autotrader testing

.PHONY: help install test test-unit test-integration test-performance test-fast test-slow coverage clean lint format

# Default target
help:
	@echo "Available targets:"
	@echo "  install          - Install test dependencies"
	@echo "  test             - Run all tests"
	@echo "  test-unit        - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-performance - Run performance tests only"
	@echo "  test-fast        - Run fast tests only"
	@echo "  test-slow        - Run slow tests only"
	@echo "  coverage         - Generate coverage report"
	@echo "  clean            - Clean test artifacts"
	@echo "  lint             - Run code linting"
	@echo "  format           - Format test code"
	@echo "  setup-reports    - Create reports directory"

# Install test dependencies
install:
	pip install -r tests/requirements.txt

# Setup reports directory
setup-reports:
	mkdir -p tests/reports

# Run all tests
test: setup-reports
	pytest

# Run unit tests only
test-unit: setup-reports
	pytest -m unit

# Run integration tests only
test-integration: setup-reports
	pytest -m integration

# Run performance tests only
test-performance: setup-reports
	pytest -m performance

# Run fast tests (exclude slow ones)
test-fast: setup-reports
	pytest -m "not slow"

# Run slow tests only
test-slow: setup-reports
	pytest -m slow

# Run tests with specific verbosity
test-verbose: setup-reports
	pytest -v -s

# Run tests in parallel
test-parallel: setup-reports
	pytest -n auto

# Generate coverage report only
coverage: setup-reports
	pytest --cov=autotrader --cov-report=html --cov-report=term

# Run specific test file
test-file: setup-reports
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make test-file FILE=test_filename.py"; \
	else \
		pytest tests/$(FILE); \
	fi

# Run specific test function
test-function: setup-reports
	@if [ -z "$(FUNC)" ]; then \
		echo "Usage: make test-function FUNC=test_function_name"; \
	else \
		pytest -k $(FUNC); \
	fi

# Run tests with debugger on failure
test-debug: setup-reports
	pytest --pdb

# Run tests and open coverage report
test-coverage-open: coverage
	@if command -v open >/dev/null 2>&1; then \
		open htmlcov/index.html; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open htmlcov/index.html; \
	else \
		echo "Coverage report generated in htmlcov/index.html"; \
	fi

# Clean test artifacts
clean:
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf tests/reports
	rm -rf .coverage
	rm -f coverage.xml
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pytest_cache" -type d -exec rm -rf {} +

# Lint test code
lint:
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 tests/ --max-line-length=120; \
	else \
		echo "flake8 not installed. Install with: pip install flake8"; \
	fi

# Format test code
format:
	@if command -v black >/dev/null 2>&1; then \
		black tests/ --line-length=120; \
	else \
		echo "black not installed. Install with: pip install black"; \
	fi

# Run tests with memory profiling
test-memory: setup-reports
	@if command -v mprof >/dev/null 2>&1; then \
		mprof run pytest tests/test_performance.py::TestPerformanceMetrics::test_memory_usage_data_collection; \
		mprof plot; \
	else \
		echo "memory-profiler not installed. Install with: pip install memory-profiler"; \
	fi

# Benchmark tests
benchmark: setup-reports
	pytest tests/test_performance.py --benchmark-only

# Run tests with different Python versions (if available)
test-python-versions:
	@for python in python3.8 python3.9 python3.10 python3.11; do \
		if command -v $$python >/dev/null 2>&1; then \
			echo "Testing with $$python"; \
			$$python -m pytest tests/test_autotrader.py -v; \
		fi; \
	done

# Continuous testing (watch for changes)
test-watch:
	@if command -v watchmedo >/dev/null 2>&1; then \
		watchmedo shell-command \
			--patterns="*.py" \
			--recursive \
			--command="make test-fast" \
			.; \
	else \
		echo "watchdog not installed. Install with: pip install watchdog"; \
	fi

# Generate test report
report: test
	@echo "Test report generated:"
	@echo "  HTML: tests/reports/report.html"
	@echo "  JSON: tests/reports/report.json"
	@echo "  Coverage: htmlcov/index.html"

# Validate test setup
validate:
	@echo "Validating test setup..."
	@python -c "import pytest; print(f'pytest version: {pytest.__version__}')"
	@python -c "import numpy; print(f'numpy version: {numpy.__version__}')"
	@python -c "import tensorflow; print(f'tensorflow version: {tensorflow.__version__}')"
	@echo "Test setup validation complete"

# Run smoke tests (quick validation)
smoke: setup-reports
	pytest tests/test_autotrader.py::TestContinuousAutoTrader::test_initialization_default_values -v

# Run regression tests
regression: setup-reports
	pytest tests/test_integration.py -v

# Performance regression test
perf-regression: setup-reports
	pytest tests/test_performance.py::TestPerformanceMetrics::test_data_collection_performance -v
	pytest tests/test_performance.py::TestPerformanceMetrics::test_model_prediction_performance -v

# Test with different configurations
test-configs:
	@echo "Testing with different configurations..."
	@export AUTOTRADER_TEST_MODE=unit && pytest tests/test_autotrader.py
	@export AUTOTRADER_TEST_MODE=integration && pytest tests/test_integration.py

# Security tests (basic)
test-security:
	@echo "Running basic security tests..."
	@if command -v bandit >/dev/null 2>&1; then \
		bandit -r tests/ -f json -o tests/reports/security.json; \
	else \
		echo "bandit not installed. Install with: pip install bandit"; \
	fi

# All quality checks
quality: clean lint format test coverage
	@echo "All quality checks completed"
