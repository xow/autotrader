# Development Guide

This guide covers the development setup, workflows, and best practices for the Autotrader Bot project.

## Table of Contents
- [Quick Start](#quick-start)
- [Development Environment](#development-environment)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Code Quality Standards](#code-quality-standards)
- [Testing](#testing)
- [Debugging](#debugging)
- [Performance](#performance)
- [Contributing](#contributing)

## Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/xow/autotrader.git
cd autotrader
make setup
```

### 2. Activate Environment
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Run Development Server
```bash
make run-debug
```

## Development Environment

### Prerequisites
- **Python 3.8+** (3.9+ recommended)
- **Git** for version control
- **Make** for build automation
- **Virtual environment** support

### Environment Setup

#### Automatic Setup (Recommended)
```bash
make setup
```

#### Manual Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### IDE Configuration

#### VS Code
Recommended extensions:
- Python
- Pylance  
- Black Formatter
- isort
- GitLens
- Better Comments

#### PyCharm
Recommended settings:
- Enable type checking
- Configure Black as formatter
- Set up pytest as test runner
- Enable Git integration

## Project Structure

```
autotrader/
├── autotrader.py           # Main application entry point
├── src/                    # Source code (future modularization)
│   ├── __init__.py
│   ├── models/            # ML models and neural networks
│   ├── data/              # Data processing and persistence  
│   ├── trading/           # Trading logic and simulation
│   ├── api/               # BTCMarkets API integration
│   └── utils/             # Utility functions and helpers
├── tests/                 # Test suites
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   ├── performance/       # Performance tests
│   └── fixtures/          # Test data and fixtures
├── docs/                  # Documentation
├── config/                # Configuration files
├── scripts/               # Build and utility scripts
├── requirements.txt       # Production dependencies
├── requirements-dev.txt   # Development dependencies
├── Makefile              # Build automation
├── .gitignore            # Git ignore rules
├── .pre-commit-config.yaml # Pre-commit hooks
└── pyproject.toml        # Project configuration
```

## Development Workflow

### Daily Development Cycle

1. **Pull Latest Changes**
   ```bash
   git pull origin main
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Write code following style guidelines
   - Add/update tests
   - Update documentation

4. **Run Quality Checks**
   ```bash
   make quick-test  # format, lint, test
   ```

5. **Commit Changes**
   ```bash
   make commit  # Runs pre-commit checks
   git commit -m "feat: add new feature"
   ```

6. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

### Common Commands

```bash
# Development
make run              # Run the bot
make run-debug        # Run with debug logging
make shell            # Python shell with context

# Testing  
make test             # Run all tests
make test-unit        # Unit tests only
make test-coverage    # Tests with coverage

# Code Quality
make lint             # Run linting
make format           # Format code
make type-check       # Type checking

# Utilities
make clean            # Clean build artifacts
make docs             # Generate documentation
make health-check     # Verify environment
```

## Code Quality Standards

### Code Style
- **PEP 8** compliance enforced by `black`
- **Type hints** required for all functions
- **Docstrings** for all public methods (Google style)
- **Import sorting** with `isort`

### Example Code Style
```python
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

class TradingBot:
    """A continuous learning trading bot for cryptocurrency markets.
    
    This class implements the core trading logic with machine learning
    capabilities for autonomous operation.
    
    Args:
        initial_balance: Starting portfolio balance in AUD
        config: Configuration dictionary for bot parameters
        
    Attributes:
        balance: Current portfolio balance
        model: The neural network model for predictions
    """
    
    def __init__(self, initial_balance: float, config: Optional[Dict] = None) -> None:
        self.balance = initial_balance
        self.config = config or {}
        self._model: Optional[tf.keras.Model] = None
        
    def predict_price(self, market_data: List[float]) -> float:
        """Predict future price based on historical market data.
        
        Args:
            market_data: List of historical price points
            
        Returns:
            Predicted price as a float
            
        Raises:
            ValueError: If market_data is empty or invalid
        """
        if not market_data:
            raise ValueError("Market data cannot be empty")
            
        # Implementation here
        return predicted_price
```

### Commit Messages
Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new trading strategy
fix: resolve memory leak in data processor  
docs: update API documentation
test: add unit tests for portfolio manager
refactor: simplify neural network architecture
perf: optimize data preprocessing pipeline
```

## Testing

### Test Structure
```
tests/
├── unit/                  # Fast, isolated tests
│   ├── test_models.py
│   ├── test_trading.py
│   └── test_utils.py
├── integration/           # Component interaction tests
│   ├── test_api_integration.py
│   └── test_data_flow.py
├── performance/           # Performance and load tests
│   └── test_benchmarks.py
└── fixtures/             # Test data
    ├── market_data.json
    └── model_states.pkl
```

### Running Tests
```bash
# All tests
make test

# Specific test types
make test-unit
make test-integration

# With coverage
make test-coverage

# Watch mode (reruns on changes)
make test-watch

# Specific test file
pytest tests/unit/test_models.py -v

# Specific test function
pytest tests/unit/test_models.py::test_model_training -v
```

### Writing Tests
```python
import pytest
from unittest.mock import Mock, patch
from autotrader import TradingBot

class TestTradingBot:
    
    @pytest.fixture
    def trading_bot(self):
        """Create a trading bot instance for testing."""
        return TradingBot(initial_balance=10000.0)
    
    def test_initial_balance(self, trading_bot):
        """Test that initial balance is set correctly."""
        assert trading_bot.balance == 10000.0
    
    @patch('autotrader.requests.get')
    def test_api_call(self, mock_get, trading_bot):
        """Test API integration with mocked responses."""
        mock_get.return_value.json.return_value = {'price': 50000.0}
        
        result = trading_bot.get_current_price()
        
        assert result == 50000.0
        mock_get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_functionality(self, trading_bot):
        """Test asynchronous operations."""
        result = await trading_bot.process_data_async()
        assert result is not None
```

## Debugging

### Logging
The application uses structured logging:

```python
import logging
logger = logging.getLogger(__name__)

# Different log levels
logger.debug("Debug information")
logger.info("General information")  
logger.warning("Warning message")
logger.error("Error occurred")
logger.critical("Critical error")
```

### Debug Mode
```bash
# Run with debug logging
make run-debug

# Or set environment variable
export LOG_LEVEL=DEBUG
python autotrader.py
```

### Interactive Debugging
```python
# Add breakpoint in code
import pdb; pdb.set_trace()

# Or use ipdb (enhanced debugger)
import ipdb; ipdb.set_trace()

# Or use built-in breakpoint() (Python 3.7+)
breakpoint()
```

### Profiling
```bash
# Profile application
make profile

# Memory profiling
python -m memory_profiler autotrader.py

# Line profiling (add @profile decorator)
kernprof -l -v autotrader.py
```

## Performance

### Optimization Guidelines
1. **Profile before optimizing** - Use data to guide decisions
2. **Optimize bottlenecks first** - Focus on the slowest parts
3. **Consider memory vs speed tradeoffs** - Monitor both metrics
4. **Use appropriate data structures** - Lists vs sets vs dicts
5. **Leverage NumPy/Pandas** - Vectorized operations are faster

### Monitoring
```python
# Memory usage monitoring
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

# Performance timing
import time
from contextlib import contextmanager

@contextmanager
def timer():
    start = time.time()
    yield
    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds")

# Usage
with timer():
    expensive_operation()
```

### Performance Testing
```bash
# Benchmark tests
make benchmark

# Load testing
locust -f tests/performance/locustfile.py
```

## Contributing

### Before Contributing
1. **Read the codebase** - Understand the architecture
2. **Check existing issues** - Avoid duplicate work
3. **Discuss major changes** - Create an issue first
4. **Follow conventions** - Code style and patterns

### Pull Request Process
1. **Create feature branch** from `main`
2. **Write tests** for new functionality
3. **Update documentation** as needed
4. **Run quality checks** (`make quick-test`)
5. **Write descriptive PR** description
6. **Request review** from maintainers

### Code Review Checklist
- [ ] Code follows style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No security vulnerabilities
- [ ] Performance impact considered
- [ ] Error handling is appropriate
- [ ] Type hints are included

### Issue Reporting
When reporting bugs:
1. **Search existing issues** first
2. **Use issue templates** provided
3. **Include reproduction steps**
4. **Provide environment details**
5. **Add relevant logs/screenshots**

### Feature Requests
When requesting features:
1. **Describe the problem** you're solving
2. **Explain the proposed solution**
3. **Consider alternatives** you've evaluated
4. **Estimate impact** on existing functionality

## Best Practices

### Security
- Never commit secrets or API keys
- Use environment variables for sensitive data
- Validate all external inputs
- Keep dependencies updated
- Use security linting tools

### Error Handling
```python
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def robust_function(data: str) -> Optional[Dict]:
    """Example of robust error handling."""
    try:
        result = process_data(data)
        return result
    except ValueError as e:
        logger.warning(f"Invalid data format: {e}")
        return None
    except ConnectionError as e:
        logger.error(f"Network error: {e}")
        raise  # Re-raise for caller to handle
    except Exception as e:
        logger.critical(f"Unexpected error: {e}")
        raise
```

### Configuration Management
```python
import os
from typing import Dict, Any
from pathlib import Path
import yaml

def load_config() -> Dict[str, Any]:
    """Load configuration with environment override."""
    config_path = Path("config") / "default.yaml"
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Environment variable overrides
    config['api_key'] = os.getenv('API_KEY', config.get('api_key'))
    config['debug'] = os.getenv('DEBUG', 'false').lower() == 'true'
    
    return config
```

### Resource Management
```python
from contextlib import contextmanager
import sqlite3

@contextmanager
def database_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect('data.db')
    try:
        yield conn
    finally:
        conn.close()

# Usage
with database_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM trades")
    results = cursor.fetchall()
```

## Troubleshooting

### Common Issues

#### Environment Problems
```bash
# Recreate virtual environment
rm -rf venv
make setup

# Update dependencies
make update
```

#### Test Failures
```bash
# Clear test cache
rm -rf .pytest_cache
pytest --cache-clear

# Verbose test output
pytest -v -s tests/
```

#### Import Errors
```bash
# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Install in development mode
pip install -e .
```

#### Performance Issues
```bash
# Profile the application
make profile

# Check memory usage
python -m memory_profiler autotrader.py
```

### Getting Help
- **Documentation**: Check project docs and README
- **GitHub Issues**: Search existing issues
- **Code Comments**: Look for inline documentation
- **Git History**: Check commit messages for context

---

For more information, see the main [README.md](README.md) and project [SPEC.md](specs/SPEC.md).
