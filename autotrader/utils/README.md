# AutoTrader Error Handling Framework

A comprehensive error handling and exception management system designed for 24/7 autonomous trading operations.

## Overview

This framework provides:

- **Custom Exception Classes**: Specialized exceptions for different error types
- **Retry Mechanisms**: Exponential backoff with jitter for resilient operations
- **Circuit Breakers**: Prevent cascade failures and provide graceful degradation
- **Recovery Strategies**: Automatic error recovery for common issues
- **Alerting System**: Multi-channel notifications for critical errors
- **Health Monitoring**: System health checks and diagnostics

## Quick Start

```python
from autotrader.utils import (
    with_retry, RetryConfig, error_context, 
    APIConnectionError, ModelTrainingError
)

# Basic retry with exponential backoff
@with_retry(RetryConfig(max_attempts=3, base_delay=1.0))
def fetch_market_data():
    # Your API call here
    pass

# Error context for automatic error handling
with error_context("data_processing", {"source": "btc_markets"}):
    process_market_data()

# Custom exceptions with context
raise APIConnectionError(
    "Failed to connect to API",
    endpoint="https://api.btcmarkets.net",
    status_code=500
)
```

## Exception Hierarchy

### Base Exception
- `AutoTraderBaseException`: Base class for all AutoTrader exceptions

### API Exceptions
- `APIException`: Base for API-related errors
- `APIConnectionError`: Connection failures
- `APIRateLimitError`: Rate limit exceeded
- `APIAuthenticationError`: Authentication failures
- `APIDataFormatError`: Invalid data format

### ML Exceptions
- `MLException`: Base for ML-related errors
- `ModelLoadingError`: Model loading failures
- `ModelTrainingError`: Training failures
- `ModelPredictionError`: Prediction failures
- `InsufficientDataError`: Not enough data for operations

### Data Validation Exceptions
- `DataValidationException`: Base for data validation errors
- `InvalidPriceDataError`: Invalid price data
- `InvalidVolumeDataError`: Invalid volume data
- `MissingDataFieldError`: Required fields missing
- `DataCorruptionError`: Data corruption detection

### Trading Exceptions
- `TradingException`: Base for trading errors
- `InsufficientBalanceError`: Insufficient funds
- `InvalidTradeParametersError`: Invalid trade parameters
- `TradeExecutionError`: Trade execution failures

### System Exceptions
- `SystemException`: Base for system errors
- `FileOperationError`: File I/O failures
- `ConfigurationError`: Configuration issues
- `NetworkException`: Network-related errors
- `CriticalSystemError`: Critical system failures

## Retry Mechanisms

### RetryConfig

Configure retry behavior:

```python
config = RetryConfig(
    max_attempts=5,        # Maximum retry attempts
    base_delay=1.0,        # Base delay in seconds
    max_delay=300.0,       # Maximum delay cap
    exponential_base=2.0,  # Exponential backoff base
    jitter=True,           # Add randomization
    backoff_factor=1.0     # Additional backoff multiplier
)

@with_retry(config)
def unreliable_operation():
    # Operation that might fail
    pass
```

### Built-in Recovery Strategies

The framework includes automatic recovery for:

- **API Connection Errors**: Automatic retry with backoff
- **Rate Limiting**: Respects retry-after headers
- **Network Timeouts**: Random backoff to prevent thundering herd
- **Memory Issues**: Automatic garbage collection

## Circuit Breaker Pattern

Prevent cascade failures:

```python
from autotrader.utils import error_handler

# Get circuit breaker for specific operation
cb = error_handler.get_circuit_breaker("api_calls")

# Use circuit breaker
try:
    result = cb.call(risky_api_call)
except Exception as e:
    # Circuit breaker may prevent call if too many failures
    pass
```

Circuit breaker states:
- **CLOSED**: Normal operation
- **OPEN**: Blocking calls due to failures
- **HALF_OPEN**: Testing if service recovered

## Error Context Management

Automatic context capture:

```python
with error_context("model_training", {"model_type": "LSTM", "samples": 1000}):
    train_model()
    # Any errors automatically include context
```

## Safe Execution

Execute functions with automatic error handling:

```python
from autotrader.utils import safe_execute

# Returns default value on error
result = safe_execute(
    lambda: risky_operation(),
    default_return="fallback_value"
)
```

## Data Validation

Validate data with custom rules:

```python
from autotrader.utils import validate_data

data = {"price": 50000, "volume": 1.5}
rules = {
    "price": lambda d: d.get("price", 0) > 0,
    "volume": lambda d: 0 < d.get("volume", 0) < 100
}

validate_data(data, rules)  # Raises DataValidationException if invalid
```

## Alerting System

### Email Configuration

Set environment variables:

```bash
export ALERT_SMTP_SERVER="smtp.gmail.com"
export ALERT_SMTP_PORT="587"
export ALERT_EMAIL_USERNAME="your_email@gmail.com"
export ALERT_EMAIL_PASSWORD="your_app_password"
export ALERT_EMAIL_RECIPIENT="admin@yourcompany.com"
```

### Alert Severities

- **CRITICAL**: Immediate attention required
- **HIGH**: Significant issues requiring prompt attention
- **MEDIUM**: Notable issues that should be addressed
- **LOW**: Minor issues for monitoring

### Manual Alerts

```python
from autotrader.utils.error_handler import ErrorAlert, error_handler

alert = ErrorAlert(
    error_type="CUSTOM_ALERT",
    timestamp=datetime.now().isoformat(),
    message="Custom alert message",
    severity="HIGH",
    context={"additional": "data"}
)

error_handler.alerting_system.send_alert(alert)
```

## Health Monitoring

Check system health:

```python
from autotrader.utils import check_system_health

health = check_system_health()
print(f"System status: {health['overall_status']}")
print(f"Memory usage: {health['checks']['memory']['usage_percent']}%")
```

## Emergency Shutdown

For critical failures:

```python
from autotrader.utils import emergency_shutdown

if critical_condition_detected():
    emergency_shutdown("Data integrity compromised")
```

## Integration with AutoTrader

### Example: Enhanced API Calls

```python
from autotrader.utils import with_retry, RetryConfig, APIConnectionError
import requests

@with_retry(RetryConfig(max_attempts=3, base_delay=2.0))
def fetch_market_data():
    try:
        response = requests.get("https://api.btcmarkets.net/v3/markets/tickers")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise APIConnectionError(
            f"Failed to fetch market data: {str(e)}",
            endpoint="https://api.btcmarkets.net/v3/markets/tickers",
            status_code=getattr(e.response, 'status_code', None)
        )
```

### Example: ML Model Training

```python
from autotrader.utils import (
    with_retry, error_context, ModelTrainingError, 
    InsufficientDataError
)

@with_retry(RetryConfig(max_attempts=2))
def train_model(data):
    if len(data) < 100:
        raise InsufficientDataError(
            "Need at least 100 samples for training",
            required_samples=100,
            available_samples=len(data)
        )
    
    with error_context("model_training", {"samples": len(data)}):
        try:
            # Training logic here
            model.fit(data)
        except Exception as e:
            raise ModelTrainingError(
                f"Training failed: {str(e)}",
                training_samples=len(data)
            )
```

### Example: Trade Execution

```python
from autotrader.utils import (
    error_context, InsufficientBalanceError, 
    TradeExecutionError
)

def execute_trade(trade_type, amount, price):
    with error_context("trade_execution", {
        "trade_type": trade_type,
        "amount": amount,
        "price": price
    }):
        if trade_type == "BUY" and balance < amount * price:
            raise InsufficientBalanceError(
                "Insufficient balance for trade",
                required_amount=amount * price,
                available_balance=balance,
                trade_type=trade_type
            )
        
        try:
            # Execute trade logic
            pass
        except Exception as e:
            raise TradeExecutionError(
                f"Trade execution failed: {str(e)}",
                trade_type=trade_type,
                amount=amount,
                price=price
            )
```

## Best Practices

### 1. Use Specific Exceptions

```python
# Good
raise APIConnectionError("Connection timeout", endpoint=url)

# Avoid
raise Exception("Something went wrong")
```

### 2. Include Context

```python
# Good
raise ModelTrainingError(
    "Training failed", 
    training_samples=len(data),
    epoch=current_epoch,
    loss=current_loss
)
```

### 3. Use Error Context

```python
# Good - automatic context capture
with error_context("data_processing"):
    process_data()

# Also good - explicit context
with error_context("api_call", {"endpoint": url}):
    make_api_call()
```

### 4. Configure Retries Appropriately

```python
# For API calls - more attempts, longer delays
api_config = RetryConfig(max_attempts=5, base_delay=2.0, max_delay=60.0)

# For quick operations - fewer attempts, shorter delays  
quick_config = RetryConfig(max_attempts=3, base_delay=0.5, max_delay=5.0)
```

### 5. Monitor Error Rates

```python
# Check error rates regularly
health = check_system_health()
if health['overall_status'] == 'UNHEALTHY':
    # Take corrective action
    pass
```

## Testing

Run the comprehensive test suite:

```bash
python -m pytest tests/test_error_handling.py -v
```

The test suite covers:
- Exception creation and serialization
- Retry mechanisms and backoff
- Circuit breaker functionality
- Recovery strategies
- Alerting system
- Data validation
- Integration scenarios

## Configuration

### Environment Variables

- `ALERT_SMTP_SERVER`: SMTP server for email alerts
- `ALERT_SMTP_PORT`: SMTP port (default: 587)
- `ALERT_EMAIL_USERNAME`: Email username for alerts
- `ALERT_EMAIL_PASSWORD`: Email password/app password
- `ALERT_EMAIL_RECIPIENT`: Email address to receive alerts

### Logging Configuration

The framework integrates with Python's logging system:

```python
import logging

# Configure logging for error handling
logging.getLogger('autotrader.utils').setLevel(logging.INFO)
```

## Performance Considerations

- **Circuit Breakers**: Prevent wasted resources on failing operations
- **Exponential Backoff**: Reduces load on failing services
- **Jitter**: Prevents thundering herd problems
- **Context Limits**: Error context data is limited to prevent memory issues
- **Alert Throttling**: Prevents alert spam through frequency limits

## Monitoring and Observability

The framework provides built-in monitoring:

- Error count tracking
- Error rate calculations
- System health checks
- Performance metrics
- Alert history

Use these metrics to:
- Identify recurring issues
- Optimize retry configurations
- Plan capacity and scaling
- Improve system reliability
