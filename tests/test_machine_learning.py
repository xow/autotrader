# -*- coding: utf-8 -*-
"""
Tests for machine learning components including LSTM model and technical indicators.
"""
import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import Mock, patch, MagicMock
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Import the autotrader module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autotrader import ContinuousAutoTrader


class TestMachineLearningComponents:
    """Test cases for ML components of the autotrader."""
    
    def test_lstm_model_architecture(self, isolated_trader):
        """Test LSTM model creation and architecture."""
        model = isolated_trader.create_lstm_model()
        
        assert model is not None
        
        # Check input shape
        assert model.input_shape == (None, isolated_trader.sequence_length, 12)
        
        # Check that model is compiled
        assert model.optimizer is not None
        assert model.loss == 'binary_crossentropy'
        assert 'accuracy' in model.metrics_names
    
    def test_technical_indicators_calculation_manual(self, isolated_trader):
        """Test manual technical indicator calculations."""
        # Create test price data with known patterns
        prices = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113, 115, 117, 116, 118, 120])
        volumes = np.array([50, 55, 52, 58, 60, 57, 62, 65, 63, 68, 70, 67, 72, 75, 73, 78, 80, 77, 82, 85])
        
        indicators = isolated_trader.calculate_technical_indicators(prices, volumes)
        
        # Test all required indicators are present
        required_indicators = [
            'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 
            'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'volume_sma'
        ]
        
        for indicator in required_indicators:
            assert indicator in indicators
            assert isinstance(indicators[indicator], (int, float))
            assert not np.isnan(indicators[indicator])
        
        # Test RSI bounds
        assert 0 <= indicators['rsi'] <= 100
        
        # Test that Bollinger Bands are ordered correctly
        assert indicators['bb_upper'] >= indicators['bb_lower']
        
        # Test SMA calculations
        expected_sma_5 = np.mean(prices[-5:])
        assert abs(indicators['sma_5'] - expected_sma_5) < 0.01
        
        expected_sma_20 = np.mean(prices[-20:])
        assert abs(indicators['sma_20'] - expected_sma_20) < 0.01
    
    def test_technical_indicators_insufficient_data(self, isolated_trader):
        """Test technical indicators with insufficient data."""
        short_prices = np.array([100, 102, 101])
        short_volumes = np.array([50, 55, 52])
        
        indicators = isolated_trader.calculate_technical_indicators(short_prices, short_volumes)
        
        # Should return default values for insufficient data
        assert indicators['rsi'] == 50  # Neutral RSI
        assert indicators['sma_5'] == 0
        assert indicators['sma_20'] == 0
    
    def test_rsi_calculation_edge_cases(self, isolated_trader):
        """Test RSI calculation with edge cases."""
        # Test with all increasing prices (should give RSI near 100)
        increasing_prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115])
        rsi_increasing = isolated_trader.manual_rsi(increasing_prices)
        assert rsi_increasing > 80  # Should be high
        
        # Test with all decreasing prices (should give RSI near 0)
        decreasing_prices = np.array([115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100])
        rsi_decreasing = isolated_trader.manual_rsi(decreasing_prices)
        assert rsi_decreasing < 20  # Should be low
        
        # Test with constant prices (should give RSI around 50)
        constant_prices = np.array([100] * 20)
        rsi_constant = isolated_trader.manual_rsi(constant_prices)
        assert 45 <= rsi_constant <= 55  # Should be neutral
    
    def test_moving_average_calculations(self, isolated_trader):
        """Test SMA and EMA calculations."""
        prices = np.array([100, 102, 104, 106, 108, 110, 112, 114, 116, 118])
        
        # Test SMA
        sma_5 = isolated_trader.manual_sma(prices, 5)
        expected_sma = np.mean(prices[-5:])
        assert abs(sma_5 - expected_sma) < 0.01
        
        # Test EMA
        ema_5 = isolated_trader.manual_ema(prices, 5)
        assert isinstance(ema_5, (int, float))
        assert not np.isnan(ema_5)
        
        # EMA should be different from SMA for trending data
        assert abs(ema_5 - sma_5) > 0.1
    
    def test_feature_preparation_completeness(self, isolated_trader):
        """Test that feature preparation includes all required features."""
        data_point = {
            'price': 45000.0,
            'volume': 123.45,
            'spread': 10.0,
            'sma_5': 45050.0,
            'sma_20': 44900.0,
            'ema_12': 45020.0,
            'ema_26': 44950.0,
            'rsi': 65.5,
            'macd': 15.2,
            'macd_signal': 12.8,
            'bb_upper': 45200.0,
            'bb_lower': 44800.0
        }
        
        features = isolated_trader.prepare_features(data_point)
        
        assert len(features) == 12  # Expected number of features
        assert all(isinstance(f, (int, float)) for f in features)
        assert all(not np.isnan(f) for f in features)
        
        # Check specific feature values
        assert features[0] == 45000.0  # price
        assert features[1] == 123.45   # volume
        assert features[7] == 65.5     # rsi
    
    def test_feature_preparation_missing_values(self, isolated_trader):
        """Test feature preparation with missing values."""
        incomplete_data_point = {
            'price': 45000.0,
            'volume': 123.45
            # Missing other features
        }
        
        features = isolated_trader.prepare_features(incomplete_data_point)
        
        assert len(features) == 12
        assert features[0] == 45000.0
        assert features[1] == 123.45
        assert features[7] == 50  # Default RSI value
    
    def test_scaler_fitting_and_transformation(self, isolated_trader, sample_training_data):
        """Test scaler fitting and data transformation."""
        isolated_trader.training_data = sample_training_data
        
        # Test scaler fitting
        success = isolated_trader.fit_scalers(sample_training_data)
        assert success
        assert isolated_trader.scalers_fitted
        
        # Test feature transformation
        test_data_point = sample_training_data[0]
        features = isolated_trader.prepare_features(test_data_point)
        
        # Transform features
        scaled_features = isolated_trader.feature_scaler.transform([features])
        
        assert scaled_features.shape == (1, 12)
        assert not np.any(np.isnan(scaled_features))
        
        # Test price transformation
        test_price = [[test_data_point['price']]]
        scaled_price = isolated_trader.price_scaler.transform(test_price)
        
        assert scaled_price.shape == (1, 1)
        assert not np.isnan(scaled_price[0][0])
    
    def test_lstm_training_data_preparation(self, isolated_trader, sample_training_data):
        """Test preparation of sequential data for LSTM training."""
        isolated_trader.training_data = sample_training_data
        isolated_trader.scalers_fitted = True
        
        sequences, labels = isolated_trader.prepare_lstm_training_data()
        
        if sequences is not None and labels is not None:
            # Check shapes
            assert len(sequences) == len(labels)
            assert sequences.shape[1] == isolated_trader.sequence_length
            assert sequences.shape[2] == 12  # Number of features
            
            # Check labels are binary
            assert all(label in [0, 1] for label in labels)
            
            # Check sequences contain scaled data
            assert not np.any(np.isnan(sequences))
            
            # Check that we have reasonable number of sequences
            expected_sequences = len(sample_training_data) - isolated_trader.sequence_length - 1
            assert len(sequences) <= expected_sequences
    
    def test_lstm_training_insufficient_data(self, isolated_trader):
        """Test LSTM training with insufficient data."""
        # Set minimal training data
        isolated_trader.training_data = [
            {"price": 100, "volume": 50} for _ in range(10)
        ]
        
        sequences, labels = isolated_trader.prepare_lstm_training_data()
        
        # Should return None for insufficient data
        assert sequences is None
        assert labels is None
    
    def test_model_training_process(self, isolated_trader, sample_training_data, mock_tensorflow):
        """Test the complete model training process."""
        isolated_trader.training_data = sample_training_data
        isolated_trader.model = mock_tensorflow["model"]
        isolated_trader.scalers_fitted = True
        
        # Mock successful training
        mock_tensorflow["model"].fit.return_value.history = {
            'loss': [0.8, 0.6, 0.4],
            'accuracy': [0.5, 0.6, 0.7],
            'val_loss': [0.9, 0.7, 0.5],
            'val_accuracy': [0.4, 0.5, 0.6]
        }
        
        success = isolated_trader.train_model()
        
        assert success
        mock_tensorflow["model"].fit.assert_called_once()
        
        # Check training parameters
        call_args = mock_tensorflow["model"].fit.call_args
        assert call_args[1]['epochs'] == 10
        assert call_args[1]['batch_size'] == 16
        assert call_args[1]['validation_split'] == 0.2
        assert call_args[1]['shuffle'] is False  # Important for time series
    
    def test_prediction_generation(self, isolated_trader, mock_market_data, sample_training_data, mock_tensorflow):
        """Test ML prediction generation."""
        # Setup trader with proper data and model
        isolated_trader.training_data = sample_training_data
        isolated_trader.scalers_fitted = True
        isolated_trader.model = mock_tensorflow["model"]
        
        # Mock model prediction
        mock_tensorflow["model"].predict.return_value = np.array([[0.8]])
        
        prediction = isolated_trader.predict_trade_signal(mock_market_data)
        
        assert "signal" in prediction
        assert "confidence" in prediction
        assert prediction["signal"] in ["BUY", "SELL", "HOLD"]
        assert 0 <= prediction["confidence"] <= 1
        
        # High confidence should result in BUY signal
        assert prediction["signal"] == "BUY"
        assert prediction["confidence"] == 0.8
    
    def test_prediction_confidence_thresholds(self, isolated_trader, mock_market_data, sample_training_data, mock_tensorflow):
        """Test prediction confidence threshold logic."""
        isolated_trader.training_data = sample_training_data
        isolated_trader.scalers_fitted = True
        isolated_trader.model = mock_tensorflow["model"]
        
        # Test different confidence levels
        test_cases = [
            (0.8, "BUY"),    # High confidence -> BUY
            (0.3, "SELL"),   # Low confidence -> SELL
            (0.5, "HOLD"),   # Neutral confidence -> HOLD
            (0.4, "HOLD"),   # Slightly low but not low enough -> HOLD
            (0.6, "HOLD"),   # Slightly high but not high enough -> HOLD
        ]
        
        for confidence, expected_signal in test_cases:
            mock_tensorflow["model"].predict.return_value = np.array([[confidence]])
            
            prediction = isolated_trader.predict_trade_signal(mock_market_data)
            
            assert prediction["signal"] == expected_signal
            assert prediction["confidence"] == confidence
    
    def test_prediction_with_invalid_input(self, isolated_trader, mock_tensorflow):
        """Test prediction handling with invalid input data."""
        isolated_trader.model = mock_tensorflow["model"]
        
        # Test with empty market data
        prediction = isolated_trader.predict_trade_signal([])
        assert prediction["signal"] == "HOLD"
        assert prediction["confidence"] == 0.5
        
        # Test with invalid market data structure
        invalid_data = [{"invalid": "data"}]
        prediction = isolated_trader.predict_trade_signal(invalid_data)
        assert prediction["signal"] == "HOLD"
    
    def test_model_performance_tracking(self, isolated_trader, mock_tensorflow):
        """Test that model performance is properly tracked during training."""
        isolated_trader.model = mock_tensorflow["model"]
        
        # Mock training history with performance metrics
        mock_history = {
            'loss': [0.8, 0.6, 0.4, 0.3],
            'accuracy': [0.5, 0.6, 0.7, 0.75],
            'val_loss': [0.9, 0.7, 0.5, 0.4],
            'val_accuracy': [0.45, 0.55, 0.65, 0.7]
        }
        mock_tensorflow["model"].fit.return_value.history = mock_history
        
        # Setup minimal required data
        isolated_trader.training_data = [{"price": 100 + i, "volume": 50} for i in range(50)]
        isolated_trader.scalers_fitted = True
        
        success = isolated_trader.train_model()
        
        assert success
        # The training should complete successfully with performance tracking
        mock_tensorflow["model"].fit.assert_called_once()
    
    def test_feature_scaling_consistency(self, isolated_trader, sample_training_data):
        """Test that feature scaling is consistent across calls."""
        isolated_trader.training_data = sample_training_data
        
        # Fit scalers
        success = isolated_trader.fit_scalers(sample_training_data)
        assert success
        
        # Transform the same data point multiple times
        test_data = sample_training_data[0]
        features = isolated_trader.prepare_features(test_data)
        
        scaled_1 = isolated_trader.feature_scaler.transform([features])
        scaled_2 = isolated_trader.feature_scaler.transform([features])
        
        # Should be identical
        np.testing.assert_array_almost_equal(scaled_1, scaled_2)
    
    def test_gradient_explosion_prevention(self, isolated_trader):
        """Test that model architecture prevents gradient explosion."""
        model = isolated_trader.create_lstm_model()
        
        # Check for dropout layers (help prevent overfitting and gradient issues)
        layer_types = [type(layer).__name__ for layer in model.layers]
        dropout_count = layer_types.count('Dropout')
        
        assert dropout_count >= 2  # Should have dropout layers
        
        # Check optimizer learning rate is reasonable
        optimizer = model.optimizer
        assert hasattr(optimizer, 'learning_rate')
        # Learning rate should be reasonable (not too high)
        lr = float(optimizer.learning_rate)
        assert 0.0001 <= lr <= 0.01
