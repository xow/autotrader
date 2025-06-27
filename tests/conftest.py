# -*- coding: utf-8 -*-
"""
pytest configuration file with fixtures for autotrader testing.
"""
import pytest
import os
import tempfile
import shutil
import json
import pickle
import numpy as np
import tensorflow as tf
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
from datetime import datetime, timedelta
from collections import deque
from sklearn.preprocessing import StandardScaler # Import StandardScaler

import sys

@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_config():
    """Basic test configuration."""
    return {
        "initial_balance": 10000.0,
        "sequence_length": 20,
        "max_training_samples": 100,
        "save_interval_seconds": 300,
        "training_interval_seconds": 600
    }


@pytest.fixture
def mock_market_data():
    """Mock market data response from BTCMarkets API."""
    return [
        {
            "marketId": "BTC-AUD",
            "lastPrice": 45000.50,
            "volume24h": 123.45,
            "bestBid": 44995.00,
            "bestAsk": 45005.00,
            "high24h": 46000.00,
            "low24h": 44000.00
        }
    ]


@pytest.fixture
def sample_training_data():
    """Generate sample training data for testing."""
    data = []
    base_price = 45000
    
    for i in range(50):
        # Generate realistic price variation
        price_change = np.random.normal(0, 100)
        price = base_price + price_change
        base_price = price
        
        # Generate other market data
        volume = np.random.uniform(100, 500)
        spread = np.random.uniform(5, 20)
        
        data_point = {
            "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat() + "Z", # ISO 8601 format with 'Z' for UTC
            "price": float(price),
            "volume": float(volume),
            "bid": float(price - spread/2),
            "ask": float(price + spread/2),
            "high24h": price + np.random.uniform(0, 200),
            "low24h": price - np.random.uniform(0, 200),
            "spread": spread,
            "sma_5": price + np.random.uniform(-50, 50),
            "sma_20": price + np.random.uniform(-100, 100),
            "ema_12": price + np.random.uniform(-30, 30),
            "ema_26": price + np.random.uniform(-80, 80),
            "rsi": np.random.uniform(20, 80),
            "macd": np.random.uniform(-100, 100),
            "macd_signal": np.random.uniform(-100, 100),
            "bb_upper": price + 200,
            "bb_lower": price - 200,
            "volume_sma": volume + np.random.uniform(-50, 50)
        }
        data.append(data_point)
    
    return data


@pytest.fixture
def mock_requests():
    """Mock requests module for API testing."""
    with patch("requests.get") as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "marketId": "BTC-AUD",
                "lastPrice": 45000.50, # Ensure this is float
                "volume24h": 123.45,  # Ensure this is float
                "bestBid": 44995.00,  # Ensure this is float
                "bestAsk": 45005.00,  # Ensure this is float
                "high24h": 46000.00,  # Ensure this is float
                "low24h": 44000.00   # Ensure this is float
            }
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        yield mock_get



@pytest.fixture
def mock_tensorflow():
    """Mock TensorFlow for model testing."""
    mock_sequential = Mock()
    mock_model = Mock()
    mock_sequential.return_value = mock_model

    with patch("tensorflow.keras.models.Sequential", return_value=mock_sequential) as Sequential:
        mock_model.fit.return_value.history = {
            "loss": [0.5, 0.4, 0.3],
            "accuracy": [0.6, 0.7, 0.8],
            "val_loss": [0.6, 0.5, 0.4],
            "val_accuracy": [0.5, 0.6, 0.7]
        }
        mock_model.predict.return_value = np.array([[0.7]])
        mock_model.save.return_value = None
        # Add mock layers to satisfy the dropout count test
        mock_model.layers = [Mock(name='Dropout'), Mock(name='Dropout')]
        mock_model.input_shape = (None, 20, 12) # Set input shape for consistency

        yield {"sequential": Sequential, "model": mock_model}

@pytest.fixture
def mock_file_operations(temp_dir):
    """Mock file operations with temporary directory."""
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    
    yield temp_dir
    
    os.chdir(original_cwd)


@pytest.fixture
def isolated_trader(temp_dir, test_config, mock_logging, initial_balance: float = None):
    """Create an isolated trader instance for testing with a mocked Settings object."""
    original_cwd = os.getcwd()
    os.chdir(temp_dir)

    from autotrader.core.continuous_autotrader import ContinuousAutoTrader
    import autotrader.core.continuous_autotrader as trader_module
    from autotrader.config.settings import Settings, get_settings

    # Ensure the Settings singleton is reset before each test using isolated_trader
    Settings._reset_singleton()

    # Create a mock Settings instance with nested config objects
    mock_settings_instance = Mock(spec=Settings)
    mock_settings_instance.config = Mock()
    mock_settings_instance.config.api = Mock()
    mock_settings_instance.config.trading = Mock()
    mock_settings_instance.config.ml = Mock()
    mock_settings_instance.config.operations = Mock()

    # Set default values for mocked settings based on test_config or common defaults
    mock_settings_instance.initial_balance = initial_balance if initial_balance is not None else test_config["initial_balance"]
    mock_settings_instance.config.trading.initial_balance = initial_balance if initial_balance is not None else test_config["initial_balance"]
    mock_settings_instance.config.trading.trade_amount = 0.01
    mock_settings_instance.config.trading.fee_rate = 0.001
    mock_settings_instance.config.trading.market_pair = "BTC-AUD"
    mock_settings_instance.config.trading.buy_confidence_threshold = 0.65
    mock_settings_instance.config.trading.sell_confidence_threshold = 0.35
    mock_settings_instance.config.trading.rsi_overbought = 80.0
    mock_settings_instance.config.trading.rsi_oversold = 20.0
    mock_settings_instance.config.trading.max_position_size = 0.1
    mock_settings_instance.config.trading.risk_per_trade = 0.02

    mock_settings_instance.config.ml.model_filename = "autotrader_model.keras"
    mock_settings_instance.config.ml.scalers_filename = "scalers.pkl"
    mock_settings_instance.config.ml.sequence_length = test_config["sequence_length"]
    mock_settings_instance.config.ml.max_training_samples = test_config["max_training_samples"]
    mock_settings_instance.config.ml.lstm_units = 50
    mock_settings_instance.config.ml.dropout_rate = 0.2
    mock_settings_instance.config.ml.learning_rate = 0.001
    mock_settings_instance.config.ml.dense_units = 25
    mock_settings_instance.config.ml.epochs = 10
    mock_settings_instance.config.ml.batch_size = 16
    mock_settings_instance.config.ml.feature_count = 12
    mock_settings_instance.config.ml.volume_sma_period = 10 # Assuming a default value
    mock_settings_instance.config.ml.scaling_method = "standard" # Add scaling method
    mock_settings_instance.config.ml.sma_periods = [5, 10, 20, 50]
    mock_settings_instance.config.ml.ema_periods = [12, 26, 50]
    mock_settings_instance.config.ml.rsi_period = 14
    mock_settings_instance.config.ml.macd_fast = 12
    mock_settings_instance.config.ml.macd_slow = 26
    mock_settings_instance.config.ml.macd_signal = 9
    mock_settings_instance.config.ml.bb_period = 20
    mock_settings_instance.config.ml.bb_std = 2
    mock_settings_instance.config.ml.volatility_window = 10
    mock_settings_instance.config.ml.lag_periods = [1, 2, 3, 5, 10]
    mock_settings_instance.config.ml.rolling_windows = [5, 10, 20]
    mock_settings_instance.config.ml.use_sma = True
    mock_settings_instance.config.ml.use_ema = True
    mock_settings_instance.config.ml.use_rsi = True
    mock_settings_instance.config.ml.use_macd = True
    mock_settings_instance.config.ml.use_bollinger = True
    mock_settings_instance.config.ml.use_volume_indicators = True
    mock_settings_instance.config.ml.use_price_ratios = True
    mock_settings_instance.config.ml.use_price_differences = True
    mock_settings_instance.config.ml.use_log_returns = True
    mock_settings_instance.config.ml.use_volatility = True
    mock_settings_instance.config.ml.use_time_features = True
    mock_settings_instance.config.ml.use_cyclical_encoding = True
    mock_settings_instance.config.ml.use_lag_features = True
    mock_settings_instance.config.ml.use_rolling_stats = True

    mock_settings_instance.config.operations.data_collection_interval = 60
    mock_settings_instance.config.operations.save_interval = test_config["save_interval_seconds"]
    mock_settings_instance.config.operations.training_interval = test_config["training_interval_seconds"]
    mock_settings_instance.config.operations.log_level = "INFO"
    mock_settings_instance.config.operations.log_file = "autotrader.log"
    mock_settings_instance.config.operations.training_data_filename = "training_data.json"
    mock_settings_instance.config.operations.state_filename = "trader_state.pkl"

    mock_settings_instance.config.api.base_url = "https://api.btcmarkets.net/v3"
    mock_settings_instance.config.api.timeout = 10
    mock_settings_instance.config.api.max_retries = 3

    # Patch get_settings to return our mock instance
    with patch('autotrader.config.settings.get_settings', return_value=mock_settings_instance), \
         patch.object(trader_module, 'logger', mock_logging), \
         patch('requests.get') as mock_requests_get: # Patch requests.get here
        
        # Configure mock_requests_get if needed for specific tests
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "marketId": "BTC-AUD",
                "lastPrice": 45000.50, # Ensure this is float
                "volume24h": 123.45,  # Ensure this is float
                "bestBid": 44995.00,  # Ensure this is float
                "bestAsk": 45005.00,  # Ensure this is float
                "high24h": 46000.00,  # Ensure this is float
                "low24h": 44000.00,   # Ensure this is float
                "timestamp": datetime.now().isoformat() + "Z" # ISO 8601 format with 'Z' for UTC
            }
        ]
        mock_response.raise_for_status.return_value = None
        mock_requests_get.return_value = mock_response

        # Instantiate ContinuousAutoTrader within the patch context
        trader = ContinuousAutoTrader()
        
        # Explicitly ensure FeatureEngineer is not fitted for tests expecting raw features
        # This overrides any fitting that might occur during ContinuousAutoTrader.__init__
        # (e.g., if scalers.pkl somehow exists or is mocked to exist)
        trader.feature_engineer.is_fitted_ = False
        trader.scalers_fitted = False
        
        # Explicitly set model after instantiation
        # This ensures the test's mocks are used, overriding any real ones created in __init__
        trader.model = Mock() # A generic mock for the model
        
        trader.training_data = deque(maxlen=trader.max_training_samples) # Ensure deque is initialized

        yield trader

    # Cleanup
    os.chdir(original_cwd)


@pytest.fixture
def mock_technical_indicators():
    """Mock technical indicator calculations."""
    def _mock_indicators(prices, volumes):
        return {
            "sma_5": float(np.mean(prices[-5:]) if len(prices) >= 5 else prices[-1]),
            "sma_20": float(np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]),
            "ema_12": float(prices[-1] * 0.9),
            "ema_26": float(prices[-1] * 0.8),
            "rsi": 50.0,
            "macd": 10.0,
            "macd_signal": 8.0,
            "bb_upper": float(prices[-1] * 1.05),
            "bb_lower": float(prices[-1] * 0.95),
            "volume_sma": float(np.mean(volumes[-10:]) if len(volumes) >= 10 else volumes[-1])
        }
    return _mock_indicators


@pytest.fixture
def prediction_test_data():
    """Test data for prediction validation."""
    return {
        "buy_signal": {"signal": "BUY", "confidence": 0.8, "price": 45000, "rsi": 45},
        "sell_signal": {"signal": "SELL", "confidence": 0.2, "price": 45000, "rsi": 65},
        "hold_signal": {"signal": "HOLD", "confidence": 0.5, "price": 45000, "rsi": 50},
        "uncertain_signal": {"signal": "HOLD", "confidence": 0.55, "price": 45000, "rsi": 50}
    }


@pytest.fixture
def mock_logging():
    """Mock logging for testing, capturing structured log messages."""
    with patch("structlog.get_logger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Configure the mock logger to capture all arguments
        def _mock_log_method(event, *args, **kwargs):
            mock_logger.calls.append({"event": event, "args": args, "kwargs": kwargs})

        mock_logger.info.side_effect = _mock_log_method
        mock_logger.debug.side_effect = _mock_log_method
        mock_logger.warning.side_effect = _mock_log_method
        mock_logger.error.side_effect = _mock_log_method
        mock_logger.exception.side_effect = _mock_log_method

        mock_logger.calls = [] # Initialize a list to store calls

        yield mock_logger


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Disable TensorFlow warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.get_logger().setLevel("ERROR")
    
    # Set random seeds for reproducible tests
    np.random.seed(42)
    tf.random.set_seed(42)
    
    yield
    
    # Cleanup after test
    if "TF_CPP_MIN_LOG_LEVEL" in os.environ:
        del os.environ["TF_CPP_MIN_LOG_LEVEL"]


@pytest.fixture(autouse=True)
def reset_settings_singleton():
    """Reset the Settings singleton before each test to ensure a clean state."""
    from autotrader.config.settings import Settings
    Settings._instance = None
    Settings._config = None
    yield
    Settings._instance = None
    Settings._config = None


# Utility fixtures for specific test scenarios

@pytest.fixture
def empty_training_data():
    """Empty training data for initialization tests."""
    return []


@pytest.fixture
def minimal_training_data():
    """Minimal training data that won't trigger model training."""
    return [
        {
            "timestamp": datetime.now().isoformat(),
            "price": 45000,
            "volume": 100,
            "rsi": 50
        }
    ]


@pytest.fixture
def corrupted_model_file(temp_dir):
    """Create a corrupted model file for error testing."""
    model_path = os.path.join(temp_dir, "autotrader_model.keras")
    with open(model_path, "w") as f:
        f.write("corrupted_model_data")
    return model_path


@pytest.fixture
def performance_metrics():
    """Mock performance metrics for testing."""
    return {
        "total_trades": 50,
        "profitable_trades": 30,
        "win_rate": 0.6,
        "total_profit": 1500.0,
        "max_drawdown": -200.0,
        "sharpe_ratio": 1.2
    }
