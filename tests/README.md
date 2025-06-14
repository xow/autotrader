# AutoTrader Testing Framework

Comprehensive testing framework for the AutoTrader cryptocurrency trading bot.

## Overview

This testing framework provides thorough coverage of all autotrader components including:

- **Unit Tests**: Individual component testing
- **Integration Tests**: System component interaction testing  
- **Performance Tests**: Benchmarking and performance validation
- **API Tests**: External API integration testing
- **ML Tests**: Machine learning model and algorithm testing
- **Data Tests**: Data management and persistence testing
- **Trading Tests**: Trading simulation and portfolio management testing

## Quick Start

### Prerequisites

```bash
# Install test dependencies
make install

# Or manually:
pip install -r tests/requirements.txt
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test categories
make test-unit          # Unit tests only
make test-integration   # Integration tests only
make test-performance   # Performance tests only

# Run fast tests (exclude slow ones)
make test-fast

# Generate coverage report
make coverage
```

## Test Categories

### Unit Tests (`test_autotrader.py`)
Tests individual components and methods:
- Initialization and configuration
- Market data fetching and processing
- Technical indicator calculations
- Feature preparation and scaling
- Model creation and training
- State management and persistence

### API Integration Tests (`test_api_integration.py`)
Tests BTCMarkets API interactions:
- API endpoint connectivity
- Response handling and parsing
- Error recovery and retry logic
- Rate limiting and timeout handling
- Data validation and extraction

### Machine Learning Tests (`test_machine_learning.py`)
Tests ML components:
- LSTM model architecture and training
- Technical indicator algorithms
- Feature engineering and scaling
- Prediction generation and confidence scoring
- Model performance tracking

### Data Management Tests (`test_data_management.py`)
Tests data persistence and management:
- Training data save/load cycles
- State persistence across restarts
- File corruption handling
- Memory management and cleanup
- Data integrity validation

### Trading Simulation Tests (`test_trading_simulation.py`)
Tests trading logic and portfolio management:
- Buy/sell/hold signal execution
- Fee calculation and balance tracking
- Risk management (RSI overrides)
- Position sizing and trade validation
- Performance metrics calculation

### Integration Tests (`test_integration.py`)
Tests complete system workflows:
- End-to-end trading cycles
- Continuous operation simulation
- Error recovery and resilience
- Concurrent operations
- Memory management under load

### Performance Tests (`test_performance.py`)
Benchmarks and performance validation:
- Data collection performance
- Model prediction latency
- Memory usage monitoring
- CPU utilization tracking
- Throughput measurement
- Scalability testing

## Test Configuration

### pytest.ini
Configures test discovery, output formatting, coverage reporting, and test categorization with markers.

### conftest.py
Provides test fixtures including:
- Temporary directories and file operations
- Mock market data and API responses  
- Sample training datasets
- TensorFlow model mocking
- Technical indicator test data
- Isolated trader instances

## Running Specific Tests

### By Category
```bash
pytest -m unit           # Unit tests
pytest -m integration    # Integration tests  
pytest -m performance    # Performance tests
pytest -m api           # API tests
pytest -m ml            # ML tests
pytest -m data          # Data management tests
pytest -m trading       # Trading simulation tests
```

### By File
```bash
pytest tests/test_autotrader.py
pytest tests/test_api_integration.py
make test-file FILE=test_autotrader.py
```

### By Function
```bash
pytest -k test_initialization
pytest tests/test_autotrader.py::TestContinuousAutoTrader::test_initialization_default_values
make test-function FUNC=test_initialization
```

## Test Utilities

### test_utils.py
Provides utility classes and functions:

- **MockMarketDataGenerator**: Realistic market data simulation
- **TechnicalIndicatorValidator**: Validates indicator calculations
- **TestDataBuilder**: Creates data with specific patterns (trending, volatile, sideways)
- **PerformanceMetrics**: Calculates trading performance metrics
- **FileTestHelper**: File operation testing utilities
- **ModelTestHelper**: ML model testing utilities
- **APITestHelper**: API testing utilities

### Example Usage
```python
from test_utils import MockMarketDataGenerator, TestDataBuilder

# Generate realistic market data
market_gen = MockMarketDataGenerator(base_price=45000.0)
tick_data = market_gen.generate_tick()

# Create trending data for testing
trending_data = TestDataBuilder.create_trending_data(
    start_price=40000, end_price=50000, num_points=100
)
```

## Coverage and Reporting

### Coverage Reports
```bash
make coverage                    # Generate coverage report
make test-coverage-open         # Generate and open in browser
```

Coverage reports include:
- Line coverage percentage
- Branch coverage analysis
- Missing line identification
- HTML and XML output formats

### Test Reports
```bash
make report                     # Generate comprehensive test report
```

Generates:
- HTML test report (`tests/reports/report.html`)
- JSON test report (`tests/reports/report.json`)
- Coverage report (`htmlcov/index.html`)

## Performance Testing

### Benchmarking
```bash
make test-performance           # Run performance tests
make benchmark                  # Run benchmark tests only
```

Performance tests measure:
- Data collection throughput
- Prediction latency
- Memory usage patterns
- CPU utilization
- File I/O performance
- Concurrent operation efficiency

### Memory Profiling
```bash
make test-memory               # Memory usage profiling
```

## Continuous Integration

### Quality Checks
```bash
make quality                   # Run all quality checks
make lint                      # Code linting
make format                    # Code formatting
```

### Regression Testing
```bash
make regression                # Run regression tests
make perf-regression          # Performance regression tests
```

## Debugging

### Debug Mode
```bash
make test-debug               # Run with debugger on failure
pytest --pdb                 # Drop into debugger on failure
```

### Verbose Output
```bash
make test-verbose             # Verbose test output
pytest -v -s                 # Verbose with print statements
```

### Watch Mode
```bash
make test-watch              # Continuous testing (watch for changes)
```

## Test Data and Mocking

### Mock Data Generation
Tests use realistic mock data that simulates:
- Market price movements with volatility
- Technical indicator patterns
- API response structures
- Trading scenarios (bull/bear/sideways markets)

### Isolated Testing
Each test runs in isolation with:
- Temporary directories for file operations
- Mocked external dependencies (API calls, TensorFlow)
- Clean state initialization
- Automatic cleanup after completion

## Best Practices

### Test Writing Guidelines
1. **Test single responsibilities**: Each test should focus on one specific behavior
2. **Use descriptive names**: Test names should clearly describe what is being tested
3. **Setup and teardown**: Use fixtures for consistent test setup and cleanup
4. **Mock external dependencies**: Don't rely on external services during testing
5. **Test error conditions**: Include tests for failure scenarios and edge cases

### Performance Testing Guidelines
1. **Set realistic benchmarks**: Performance thresholds should be achievable on target hardware
2. **Measure consistently**: Use the same environment and conditions for performance tests
3. **Test under load**: Include tests that simulate realistic usage patterns
4. **Monitor resources**: Track memory, CPU, and I/O usage during tests

### Integration Testing Guidelines
1. **Test real workflows**: Integration tests should mirror actual usage patterns
2. **Verify data flow**: Ensure data flows correctly between components
3. **Test error propagation**: Verify that errors are handled properly across components
4. **Validate state consistency**: Check that system state remains consistent

## Common Issues and Solutions

### TensorFlow Warnings
TensorFlow may produce warnings during testing. These are filtered in `pytest.ini` but can be re-enabled by removing the filter.

### File Permission Issues
Tests create temporary files and directories. Ensure the test runner has appropriate file system permissions.

### Memory Usage
Performance tests monitor memory usage. On systems with limited memory, some tests may need adjustment.

### API Rate Limiting
API tests use mocking by default. Live API tests should be run carefully to avoid rate limiting.

## Contributing

When adding new tests:
1. Follow the existing test structure and naming conventions
2. Add appropriate markers for test categorization
3. Include both positive and negative test cases
4. Update this documentation for new test categories or utilities
5. Ensure tests are deterministic and don't depend on external state

### Test Coverage Goals
- Maintain >80% code coverage
- Cover all critical paths and error conditions
- Include performance benchmarks for key operations
- Test all external integrations with mocking

## Makefile Targets

The included Makefile provides convenient targets for common testing tasks:

```bash
make help                     # Show all available targets
make install                  # Install test dependencies
make test                     # Run all tests
make clean                    # Clean test artifacts
make validate                 # Validate test setup
make smoke                    # Run quick smoke tests
```

For a complete list of targets and their descriptions, run `make help`.
