# Autotrader Bot - Agent Guidelines

## Build/Test/Lint Commands
- **Run all tests**: `make test` or `pytest tests/`
- **Run single test**: `pytest tests/test_<filename>.py` or `pytest tests/test_<filename>.py::test_function_name`
- **Run with coverage**: `make test-coverage` or `pytest --cov=. --cov-report=html tests/`
- **Lint code**: `make lint` (runs flake8 + mypy)
- **Format code**: `make format` (runs black + isort)
- **Build/validate**: `make build` (runs tests + lint)
- **Run app**: `make run` or `python autotrader.py`

## Architecture
- **Main entry**: `autotrader.py` with `ContinuousAutoTrader` class
- **Core modules**: `autotrader/` package with `core/`, `api/`, `ml/`, `trading/`, `data/`, `utils/`
- **ML Engine**: LSTM-based neural network with TensorFlow for price prediction
- **Data Source**: BTCMarkets API for live BTC-AUD trading data
- **Storage**: Pickle files for model/state persistence (`autotrader_model.keras`, `trader_state.pkl`)

## Code Style
- **Formatting**: Black (88 char line length) + isort for imports
- **Type hints**: Required, mypy strict mode enabled
- **Docstrings**: Required for all public functions/classes
- **Testing**: pytest with fixtures in `conftest.py`, separate unit/integration tests
- **Error handling**: Structured logging with `structlog`, graceful exception handling
- **Naming**: snake_case for functions/variables, PascalCase for classes
