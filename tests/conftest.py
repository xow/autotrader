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
            "lastPrice": "45000.50",
            "volume24h": "123.45",
            "bestBid": "44995.00",
            "bestAsk": "45005.00",
            "high24h": "46000.00",
            "low24h": "44000.00"
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
            "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
            "price": price,
            "volume": volume,
            "bid": price - spread/2,
            "ask": price + spread/2,
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
                "lastPrice": "45000.50",
                "volume24h": "123.45",
                "bestBid": "44995.00",
                "bestAsk": "45005.00",
                "high24h": "46000.00",
                "low24h": "44000.00"
            }
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        yield mock_get


@pytest.fixture
def mock_tensorflow():
    """Mock TensorFlow for model testing."""
    with patch("tensorflow.keras.models.load_model") as mock_load, \
         patch("tensorflow.keras.Sequential") as mock_sequential:
        
        # Mock model
        mock_model = Mock()
        mock_model.fit.return_value.history = {
            "loss": [0.5, 0.4, 0.3],
            "accuracy": [0.6, 0.7, 0.8],
            "val_loss": [0.6, 0.5, 0.4],
            "val_accuracy": [0.5, 0.6, 0.7]
        }
        mock_model.predict.return_value = np.array([[0.7]])
        mock_model.save.return_value = None
        
        mock_sequential.return_value = mock_model
        mock_load.return_value = mock_model
        
        yield {
            "model": mock_model,
            "load_model": mock_load,
            "sequential": mock_sequential
        }


@pytest.fixture
def mock_file_operations(temp_dir):
    """Mock file operations with temporary directory."""
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    
    yield temp_dir
    
    os.chdir(original_cwd)


@pytest.fixture
def isolated_trader(temp_dir, test_config):
    """Create an isolated trader instance for testing."""
    # Change to temp directory for isolated file operations
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    
    # Import after changing directory to avoid file conflicts
    from autotrader import ContinuousAutoTrader
    
    trader = ContinuousAutoTrader(initial_balance=test_config["initial_balance"])
    trader.save_interval_seconds = test_config["save_interval_seconds"]
    trader.training_interval_seconds = test_config["training_interval_seconds"]
    trader.max_training_samples = test_config["max_training_samples"]
    trader.sequence_length = test_config["sequence_length"]
    
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
    """Mock logging for testing."""
    with patch("logging.getLogger") as mock_logger_factory:
        mock_logger = Mock()
        mock_logger_factory.return_value = mock_logger
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
