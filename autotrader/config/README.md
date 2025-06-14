# AutoTrader Configuration Management

This directory contains the configuration management system for the AutoTrader bot, providing secure and flexible configuration handling for different environments.

## Overview

The configuration system supports:
- Environment-based configurations (development, staging, production)
- Environment variable overrides for sensitive data
- Configuration validation and error handling
- Secure API key management
- ML model parameter configuration
- Trading parameter management

## Files Structure

```
autotrader/config/
├── __init__.py                 # Package initialization
├── config.py                   # Core configuration classes
├── settings.py                 # Global settings manager
├── config_development.json     # Development environment config
├── config_staging.json         # Staging environment config
├── config_production.json      # Production environment config
├── .env.template              # Environment variables template
└── README.md                  # This documentation
```

## Quick Start

### 1. Setup Environment Variables

Copy the environment template and fill in your credentials:

```bash
cp autotrader/config/.env.template .env
```

Edit `.env` with your BTCMarkets API credentials:

```bash
BTCMARKETS_API_KEY=your_actual_api_key
BTCMARKETS_API_SECRET=your_actual_api_secret
AUTOTRADER_ENV=development
```

### 2. Basic Usage

```python
from autotrader.config import Settings

# Get global settings instance
settings = Settings()

# Access configuration values
api_key, api_secret = settings.get_api_credentials()
initial_balance = settings.initial_balance
sequence_length = settings.sequence_length

# Check environment
if settings.is_production:
    print("Running in production mode")
```

### 3. Load Custom Configuration

```python
from autotrader.config import Config

# Load from specific file
config = Config.from_file("my_config.json")

# Create and save configuration
config = Config()
config.trading.initial_balance = 5000.0
config.to_file("my_config.json")
```

## Configuration Structure

### API Configuration (`APIConfig`)

Manages BTCMarkets API settings:

- `base_url`: BTCMarkets API base URL
- `api_key`: API key (loaded from environment)
- `api_secret`: API secret (loaded from environment) 
- `timeout`: Request timeout in seconds
- `max_retries`: Maximum retry attempts
- `retry_delay`: Delay between retries

### Trading Configuration (`TradingConfig`)

Controls trading behavior:

- `initial_balance`: Starting balance in AUD
- `trade_amount`: Amount per trade in BTC
- `fee_rate`: Trading fee rate (0.1% = 0.001)
- `market_pair`: Trading pair (e.g., "BTC-AUD")
- `buy_confidence_threshold`: Minimum confidence for buy signals
- `sell_confidence_threshold`: Maximum confidence for sell signals
- `rsi_overbought`/`rsi_oversold`: RSI filter levels

### ML Configuration (`MLConfig`)

Machine learning model parameters:

- `sequence_length`: LSTM sequence length
- `max_training_samples`: Maximum training data points
- `lstm_units`: Number of LSTM units
- `learning_rate`: Model learning rate
- `epochs`: Training epochs
- `batch_size`: Training batch size

### Operational Configuration (`OperationalConfig`)

System operation settings:

- `data_collection_interval`: How often to collect data (seconds)
- `save_interval`: How often to save state (seconds)
- `training_interval`: How often to retrain model (seconds)
- `log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `log_file`: Log file path

## Environment-Specific Settings

### Development Environment
- Smaller balances and trade amounts for testing
- More frequent saves and training
- Debug logging enabled
- Conservative trading thresholds

### Staging Environment  
- Medium balances for integration testing
- Moderate intervals
- Production-like settings but safer
- Comprehensive logging

### Production Environment
- Full balances and trade amounts
- Longer intervals to reduce API calls
- Conservative trading thresholds
- Warning-level logging only

## Environment Variables

Override any configuration with environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `AUTOTRADER_ENV` | Environment (development/staging/production) | `production` |
| `BTCMARKETS_API_KEY` | BTCMarkets API key | `your-api-key` |
| `BTCMARKETS_API_SECRET` | BTCMarkets API secret | `your-api-secret` |
| `TRADING_INITIAL_BALANCE` | Starting balance | `10000.0` |
| `TRADING_TRADE_AMOUNT` | Trade amount in BTC | `0.01` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `DATA_COLLECTION_INTERVAL` | Data collection interval | `60` |

## Security Best Practices

### API Key Management
- Never hardcode API keys in configuration files
- Use environment variables for credentials
- Rotate API keys regularly
- Use minimal required permissions

### Configuration Files
- Never commit `.env` files to version control
- Keep production configs separate from code
- Validate all configuration values
- Use different API keys for different environments

### Logging
- Don't log sensitive information
- Use appropriate log levels for environments
- Rotate log files in production
- Monitor for configuration errors

## Advanced Usage

### Custom Validation

```python
from autotrader.config import Config

class CustomConfig(Config):
    def _validate_config(self):
        super()._validate_config()
        
        # Add custom validation
        if self.trading.trade_amount > self.trading.initial_balance * 0.1:
            raise ValueError("Trade amount too large")
```

### Dynamic Configuration Updates

```python
from autotrader.config import get_settings

settings = get_settings()

# Update configuration at runtime
settings.config.trading.trade_amount = 0.02
settings.config._validate_config()  # Validate changes

# Reload from file
settings.reload_config("new_config.json")
```

### Configuration Monitoring

```python
from autotrader.config import validate_environment

# Validate current environment setup
try:
    validate_environment()
    print("Environment is properly configured")
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Testing Configuration

```python
import pytest
from autotrader.config import Config, Environment

def test_development_config():
    config = Config()
    config.environment = Environment.DEVELOPMENT
    
    assert config.is_development()
    assert not config.is_production()
    assert config.trading.initial_balance > 0

def test_api_credentials():
    config = Config()
    # Test with mock environment variables
    import os
    os.environ["BTCMARKETS_API_KEY"] = "test-key"
    os.environ["BTCMARKETS_API_SECRET"] = "test-secret"
    
    key, secret = config.get_api_credentials()
    assert key == "test-key"
    assert secret == "test-secret"
```

## Troubleshooting

### Common Issues

1. **Missing API Credentials**
   - Check `.env` file exists and has correct values
   - Verify environment variables are loaded
   - Ensure no extra spaces in values

2. **Configuration Validation Errors**
   - Check all numeric values are positive
   - Verify thresholds are between 0 and 1
   - Ensure file paths are writable

3. **Environment Loading Issues**
   - Verify `AUTOTRADER_ENV` is set correctly
   - Check config file exists for environment
   - Validate JSON syntax in config files

### Debug Configuration Loading

```python
from autotrader.config import Settings
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Load settings and check values
settings = Settings()
print(f"Environment: {settings.environment}")
print(f"Config loaded: {settings.config.to_dict()}")
```

## Migration Guide

When updating configuration structure:

1. Update the dataclass definitions in `config.py`
2. Update default configuration files
3. Update environment variable loading
4. Test with existing configuration files
5. Provide migration documentation

For questions or issues, refer to the main project documentation or create an issue in the repository.
