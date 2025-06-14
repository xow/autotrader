# -*- coding: utf-8 -*-
"""
Utility functions and test helpers for autotrader testing.
"""
import numpy as np
import json
import os
import tempfile
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import random


class MockMarketDataGenerator:
    """Generate realistic mock market data for testing."""
    
    def __init__(self, base_price: float = 45000.0, volatility: float = 0.02):
        self.base_price = base_price
        self.volatility = volatility
        self.current_price = base_price
    
    def generate_tick(self) -> Dict[str, Any]:
        """Generate a single market data tick."""
        # Generate price change with some randomness
        price_change = np.random.normal(0, self.current_price * self.volatility)
        self.current_price = max(1000, self.current_price + price_change)  # Keep price reasonable
        
        # Generate other market data
        volume = np.random.exponential(100)  # Exponential distribution for volume
        spread = np.random.uniform(5, 25)
        
        return {
            "marketId": "BTC-AUD",
            "lastPrice": f"{self.current_price:.2f}",
            "volume24h": f"{volume:.4f}",
            "bestBid": f"{self.current_price - spread/2:.2f}",
            "bestAsk": f"{self.current_price + spread/2:.2f}",
            "high24h": f"{self.current_price * (1 + np.random.uniform(0, 0.05)):.2f}",
            "low24h": f"{self.current_price * (1 - np.random.uniform(0, 0.05)):.2f}"
        }
    
    def generate_historical_data(self, num_points: int) -> List[Dict[str, Any]]:
        """Generate historical market data."""
        data = []
        for i in range(num_points):
            tick = self.generate_tick()
            # Add timestamp
            timestamp = (datetime.now() - timedelta(minutes=num_points-i)).isoformat()
            tick["timestamp"] = timestamp
            data.append(tick)
        return data


class TechnicalIndicatorValidator:
    """Validate technical indicator calculations."""
    
    @staticmethod
    def validate_rsi(rsi_value: float) -> bool:
        """Validate RSI is within correct bounds."""
        return 0 <= rsi_value <= 100
    
    @staticmethod
    def validate_moving_average(prices: np.ndarray, ma_value: float, period: int) -> bool:
        """Validate moving average calculation."""
        if len(prices) < period:
            return True  # Can't validate insufficient data
        
        expected_ma = np.mean(prices[-period:])
        return abs(ma_value - expected_ma) < 0.01
    
    @staticmethod
    def validate_bollinger_bands(bb_upper: float, bb_lower: float, sma: float) -> bool:
        """Validate Bollinger Bands ordering."""
        return bb_upper >= sma >= bb_lower
    
    @staticmethod
    def validate_macd_signal_relationship(macd: float, macd_signal: float) -> bool:
        """Validate MACD and signal line relationship."""
        # No strict rule, but both should be reasonable values
        return abs(macd) < 1000 and abs(macd_signal) < 1000


class TestDataBuilder:
    """Build test data with specific patterns."""
    
    @staticmethod
    def create_trending_data(start_price: float, end_price: float, num_points: int) -> List[Dict[str, Any]]:
        """Create data with a clear trend."""
        price_step = (end_price - start_price) / (num_points - 1)
        
        data = []
        for i in range(num_points):
            price = start_price + (i * price_step)
            # Add some noise
            price += np.random.normal(0, price * 0.005)
            
            data_point = {
                "timestamp": (datetime.now() - timedelta(minutes=num_points-i)).isoformat(),
                "price": price,
                "volume": np.random.uniform(50, 200),
                "bid": price - np.random.uniform(5, 15),
                "ask": price + np.random.uniform(5, 15),
                "high24h": price + np.random.uniform(0, 100),
                "low24h": price - np.random.uniform(0, 100),
                "spread": np.random.uniform(10, 30)
            }
            data.append(data_point)
        
        return data
    
    @staticmethod
    def create_volatile_data(base_price: float, num_points: int, volatility: float = 0.05) -> List[Dict[str, Any]]:
        """Create highly volatile market data."""
        data = []
        current_price = base_price
        
        for i in range(num_points):
            # High volatility price changes
            price_change = np.random.normal(0, current_price * volatility)
            current_price = max(1000, current_price + price_change)
            
            data_point = {
                "timestamp": (datetime.now() - timedelta(minutes=num_points-i)).isoformat(),
                "price": current_price,
                "volume": np.random.exponential(150),  # Higher volume in volatile markets
                "bid": current_price - np.random.uniform(20, 50),
                "ask": current_price + np.random.uniform(20, 50),
                "high24h": current_price * (1 + np.random.uniform(0.02, 0.1)),
                "low24h": current_price * (1 - np.random.uniform(0.02, 0.1)),
                "spread": np.random.uniform(30, 100)  # Wider spreads in volatile markets
            }
            data.append(data_point)
        
        return data
    
    @staticmethod
    def create_sideways_data(base_price: float, num_points: int, range_pct: float = 0.02) -> List[Dict[str, Any]]:
        """Create sideways (ranging) market data."""
        data = []
        price_range = base_price * range_pct
        
        for i in range(num_points):
            # Price oscillates around base price
            price = base_price + np.random.uniform(-price_range, price_range)
            
            data_point = {
                "timestamp": (datetime.now() - timedelta(minutes=num_points-i)).isoformat(),
                "price": price,
                "volume": np.random.uniform(80, 120),
                "bid": price - np.random.uniform(5, 15),
                "ask": price + np.random.uniform(5, 15),
                "high24h": price + np.random.uniform(0, 50),
                "low24h": price - np.random.uniform(0, 50),
                "spread": np.random.uniform(10, 25)
            }
            data.append(data_point)
        
        return data


class PerformanceMetrics:
    """Calculate trading performance metrics."""
    
    @staticmethod
    def calculate_profit_loss(initial_balance: float, final_balance: float) -> float:
        """Calculate absolute profit/loss."""
        return final_balance - initial_balance
    
    @staticmethod
    def calculate_return_percentage(initial_balance: float, final_balance: float) -> float:
        """Calculate percentage return."""
        return ((final_balance - initial_balance) / initial_balance) * 100
    
    @staticmethod
    def calculate_max_drawdown(balance_history: List[float]) -> float:
        """Calculate maximum drawdown from balance history."""
        if len(balance_history) < 2:
            return 0.0
        
        peak = balance_history[0]
        max_drawdown = 0.0
        
        for balance in balance_history[1:]:
            if balance > peak:
                peak = balance
            else:
                drawdown = (peak - balance) / peak
                max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    @staticmethod
    def calculate_win_rate(trades: List[Dict[str, Any]]) -> float:
        """Calculate win rate from trade history."""
        if not trades:
            return 0.0
        
        profitable_trades = sum(1 for trade in trades if trade.get('profit', 0) > 0)
        return profitable_trades / len(trades)
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = [r - risk_free_rate for r in returns]
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns)
        
        if std_return == 0:
            return 0.0
        
        return mean_return / std_return


class FileTestHelper:
    """Helper functions for file-based testing."""
    
    @staticmethod
    def create_temp_file(content: str, suffix: str = ".tmp") -> str:
        """Create a temporary file with content."""
        fd, path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, 'w') as tmp:
                tmp.write(content)
            return path
        except:
            os.close(fd)
            raise
    
    @staticmethod
    def create_corrupted_json_file(path: str):
        """Create a corrupted JSON file for testing error handling."""
        with open(path, 'w') as f:
            f.write('{"invalid": json content without closing brace')
    
    @staticmethod
    def create_corrupted_pickle_file(path: str):
        """Create a corrupted pickle file for testing error handling."""
        with open(path, 'wb') as f:
            f.write(b'corrupted_pickle_data_not_deserializable')
    
    @staticmethod
    def verify_file_exists(path: str) -> bool:
        """Verify a file exists and is readable."""
        return os.path.exists(path) and os.access(path, os.R_OK)
    
    @staticmethod
    def get_file_size(path: str) -> int:
        """Get file size in bytes."""
        return os.path.getsize(path) if os.path.exists(path) else 0


class ModelTestHelper:
    """Helper functions for testing ML models."""
    
    @staticmethod
    def create_mock_lstm_history():
        """Create mock LSTM training history."""
        return {
            'loss': [0.8, 0.6, 0.4, 0.3, 0.25],
            'accuracy': [0.5, 0.6, 0.7, 0.75, 0.8],
            'val_loss': [0.9, 0.7, 0.5, 0.4, 0.35],
            'val_accuracy': [0.45, 0.55, 0.65, 0.7, 0.75]
        }
    
    @staticmethod
    def validate_model_architecture(model):
        """Validate basic model architecture requirements."""
        # Check if model has layers
        if not hasattr(model, 'layers'):
            return False
        
        # Check if model is compiled
        if not hasattr(model, 'optimizer'):
            return False
        
        return True
    
    @staticmethod
    def generate_prediction_test_cases():
        """Generate test cases for prediction validation."""
        return [
            # (confidence, expected_signal)
            (0.9, "BUY"),
            (0.8, "BUY"),
            (0.7, "BUY"),
            (0.65, "BUY"),
            (0.6, "HOLD"),
            (0.5, "HOLD"),
            (0.4, "HOLD"),
            (0.35, "SELL"),
            (0.3, "SELL"),
            (0.2, "SELL"),
            (0.1, "SELL")
        ]


class APITestHelper:
    """Helper functions for API testing."""
    
    @staticmethod
    def create_mock_api_response(market_id: str = "BTC-AUD", price: float = 45000.0):
        """Create a mock API response."""
        return [
            {
                "marketId": market_id,
                "lastPrice": f"{price:.2f}",
                "volume24h": f"{np.random.uniform(100, 500):.4f}",
                "bestBid": f"{price - np.random.uniform(5, 25):.2f}",
                "bestAsk": f"{price + np.random.uniform(5, 25):.2f}",
                "high24h": f"{price * (1 + np.random.uniform(0, 0.05)):.2f}",
                "low24h": f"{price * (1 - np.random.uniform(0, 0.05)):.2f}"
            }
        ]
    
    @staticmethod
    def create_invalid_api_response():
        """Create an invalid API response for error testing."""
        return [
            {
                "invalid_field": "invalid_data",
                "missing_required_fields": True
            }
        ]
    
    @staticmethod
    def simulate_network_error():
        """Simulate various network errors."""
        import requests
        errors = [
            requests.exceptions.ConnectionError("Connection failed"),
            requests.exceptions.Timeout("Request timeout"),
            requests.exceptions.HTTPError("404 Not Found"),
            requests.exceptions.RequestException("General request error")
        ]
        return random.choice(errors)


def setup_test_environment():
    """Setup common test environment variables and configurations."""
    # Set random seeds for reproducible tests
    np.random.seed(42)
    random.seed(42)
    
    # Disable TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    return True


def cleanup_test_files(file_patterns: List[str]):
    """Clean up test files matching given patterns."""
    import glob
    
    for pattern in file_patterns:
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
            except OSError:
                pass  # File might already be deleted


def assert_approximately_equal(actual: float, expected: float, tolerance: float = 0.01):
    """Assert that two float values are approximately equal."""
    assert abs(actual - expected) <= tolerance, f"Expected {expected}, got {actual} (tolerance: {tolerance})"


def assert_valid_prediction(prediction: Dict[str, Any]):
    """Assert that a prediction dictionary has valid structure and values."""
    required_fields = ["signal", "confidence", "price"]
    
    for field in required_fields:
        assert field in prediction, f"Missing required field: {field}"
    
    assert prediction["signal"] in ["BUY", "SELL", "HOLD"], f"Invalid signal: {prediction['signal']}"
    assert 0 <= prediction["confidence"] <= 1, f"Invalid confidence: {prediction['confidence']}"
    assert prediction["price"] > 0, f"Invalid price: {prediction['price']}"


def assert_valid_technical_indicators(indicators: Dict[str, float]):
    """Assert that technical indicators have valid values."""
    required_indicators = [
        'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 
        'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'volume_sma'
    ]
    
    for indicator in required_indicators:
        assert indicator in indicators, f"Missing indicator: {indicator}"
        assert isinstance(indicators[indicator], (int, float)), f"Invalid type for {indicator}"
        assert not np.isnan(indicators[indicator]), f"NaN value for {indicator}"
    
    # Specific validations
    assert 0 <= indicators['rsi'] <= 100, f"Invalid RSI: {indicators['rsi']}"
    assert indicators['bb_upper'] >= indicators['bb_lower'], "Invalid Bollinger Bands ordering"
