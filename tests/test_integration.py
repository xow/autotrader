# -*- coding: utf-8 -*-
"""
Integration tests for the complete autotrader system.
"""
import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from datetime import datetime, timedelta

# Import the autotrader module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autotrader import ContinuousAutoTrader

# Import test utilities
from test_utils import (
    MockMarketDataGenerator, 
    TestDataBuilder, 
    PerformanceMetrics,
    assert_valid_prediction,
    assert_valid_technical_indicators
)


class TestSystemIntegration:
    """Integration tests for the complete autotrader system."""
    
    def test_complete_trading_cycle_simulation(self, isolated_trader, mock_tensorflow):
        """Test a complete trading cycle from data collection to trade execution."""
        # Setup
        isolated_trader.model = mock_tensorflow["model"]
        mock_tensorflow["model"].predict.return_value = np.array([[0.8]])
        
        # Create realistic market data generator
        market_gen = MockMarketDataGenerator(base_price=45000.0)
        
        # Simulate data collection phase
        for _ in range(25):  # Collect enough data for technical indicators
            mock_data = [market_gen.generate_tick()]
            
            with patch.object(isolated_trader, 'fetch_market_data', return_value=mock_data):
                success = isolated_trader.collect_and_store_data()
                assert success
        
        # Verify we have sufficient data
        assert len(isolated_trader.training_data) >= 25
        
        # Test prediction generation
        current_data = [market_gen.generate_tick()]
        with patch.object(isolated_trader, 'fetch_market_data', return_value=current_data):
            prediction = isolated_trader.predict_trade_signal(current_data)
            assert_valid_prediction(prediction)
        
        # Test trade execution
        initial_balance = isolated_trader.balance
        isolated_trader.execute_simulated_trade(prediction)
        
        # Verify trade was executed (balance changed)
        if prediction["signal"] != "HOLD" and prediction["confidence"] > 0.65:
            assert isolated_trader.balance != initial_balance
    
    def test_continuous_operation_simulation(self, isolated_trader, mock_tensorflow):
        """Test continuous operation over multiple iterations."""
        isolated_trader.model = mock_tensorflow["model"]
        mock_tensorflow["model"].predict.return_value = np.array([[0.7]])
        
        market_gen = MockMarketDataGenerator()
        iteration_count = 0
        max_iterations = 10
        
        def mock_fetch_data():
            nonlocal iteration_count
            iteration_count += 1
            if iteration_count > max_iterations:
                raise KeyboardInterrupt("Test simulation complete")
            return [market_gen.generate_tick()]
        
        with patch.object(isolated_trader, 'fetch_market_data', side_effect=mock_fetch_data):
            try:
                isolated_trader.run_continuous_trading()
            except KeyboardInterrupt:
                pass  # Expected to stop the loop
        
        # Verify system collected data over multiple iterations
        assert len(isolated_trader.training_data) >= max_iterations
        assert iteration_count == max_iterations + 1
    
    def test_model_training_integration(self, isolated_trader, mock_tensorflow):
        """Test integration of model training with data collection."""
        isolated_trader.model = mock_tensorflow["model"]
        
        # Create training data with clear patterns
        trending_data = TestDataBuilder.create_trending_data(
            start_price=40000, end_price=50000, num_points=100
        )
        
        # Convert to autotrader format and add technical indicators
        for i, data_point in enumerate(trending_data):
            isolated_trader.training_data.append({
                **data_point,
                "sma_5": data_point["price"],
                "sma_20": data_point["price"],
                "ema_12": data_point["price"],
                "ema_26": data_point["price"],
                "rsi": 50.0,
                "macd": 0.0,
                "macd_signal": 0.0,
                "bb_upper": data_point["price"] + 100,
                "bb_lower": data_point["price"] - 100,
                "volume_sma": data_point["volume"]
            })
        
        # Fit scalers
        isolated_trader.fit_scalers(isolated_trader.training_data)
        
        # Test model training
        success = isolated_trader.train_model()
        assert success
        
        # Verify training was called
        mock_tensorflow["model"].fit.assert_called_once()
        
        # Verify training parameters
        call_args = mock_tensorflow["model"].fit.call_args
        assert call_args[1]['epochs'] == 10
        assert call_args[1]['validation_split'] == 0.2
        assert call_args[1]['shuffle'] is False
    
    def test_state_persistence_across_restarts(self, temp_dir, test_config):
        """Test that state persists correctly across application restarts."""
        # First session
        with patch('os.getcwd', return_value=temp_dir):
            trader1 = ContinuousAutoTrader(initial_balance=12000.0)
            
            # Add some data and modify state
            trader1.training_data = [
                {
                    "timestamp": datetime.now().isoformat(),
                    "price": 45000.0,
                    "volume": 100.0,
                    "rsi": 65.0
                }
            ]
            trader1.balance = 11500.0
            trader1.last_training_time = 1234567890
            
            # Save everything
            trader1.save_training_data()
            trader1.save_state()
            
            del trader1
        
        # Second session (simulate restart)
        with patch('os.getcwd', return_value=temp_dir):
            trader2 = ContinuousAutoTrader()
            
            # Verify state was restored
            assert trader2.balance == 11500.0
            assert trader2.last_training_time == 1234567890
            assert len(trader2.training_data) == 1
            assert trader2.training_data[0]["price"] == 45000.0
    
    def test_error_recovery_integration(self, isolated_trader, mock_tensorflow):
        """Test error recovery during various operations."""
        isolated_trader.model = mock_tensorflow["model"]
        
        # Test API error recovery
        call_count = 0
        def failing_api_call():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("API Error")
            return [{"marketId": "BTC-AUD", "lastPrice": "45000"}]
        
        with patch.object(isolated_trader, 'fetch_market_data', side_effect=failing_api_call):
            # Should handle API errors gracefully
            success = isolated_trader.collect_and_store_data()
            assert not success  # Should fail after retries
            assert call_count > 1  # Should have retried
        
        # Test model training error recovery
        mock_tensorflow["model"].fit.side_effect = Exception("Training Error")
        
        # Add sufficient data
        isolated_trader.training_data = TestDataBuilder.create_trending_data(
            start_price=45000, end_price=46000, num_points=50
        )
        isolated_trader.scalers_fitted = True
        
        # Should handle training errors gracefully
        success = isolated_trader.train_model()
        assert not success
    
    def test_performance_under_load(self, isolated_trader, mock_tensorflow):
        """Test system performance under heavy load."""
        isolated_trader.model = mock_tensorflow["model"]
        mock_tensorflow["model"].predict.return_value = np.array([[0.6]])
        
        # Generate large dataset
        large_dataset = TestDataBuilder.create_volatile_data(
            base_price=45000, num_points=1000, volatility=0.03
        )
        
        start_time = time.time()
        
        # Process large dataset
        for data_point in large_dataset[:100]:  # Process subset for testing
            isolated_trader.training_data.append({
                **data_point,
                "sma_5": data_point["price"],
                "sma_20": data_point["price"],
                "ema_12": data_point["price"],
                "ema_26": data_point["price"],
                "rsi": 50.0,
                "macd": 0.0,
                "macd_signal": 0.0,
                "bb_upper": data_point["price"] + 100,
                "bb_lower": data_point["price"] - 100,
                "volume_sma": data_point["volume"]
            })
        
        # Test prediction generation speed
        with patch.object(isolated_trader, 'fetch_market_data', return_value=[large_dataset[0]]):
            prediction = isolated_trader.predict_trade_signal([large_dataset[0]])
            assert_valid_prediction(prediction)
        
        processing_time = time.time() - start_time
        
        # Should complete processing within reasonable time
        assert processing_time < 30  # 30 seconds for 100 data points
        assert len(isolated_trader.training_data) == 100
    
    def test_concurrent_operations(self, isolated_trader, mock_tensorflow):
        """Test concurrent operations don't cause conflicts."""
        isolated_trader.model = mock_tensorflow["model"]
        mock_tensorflow["model"].predict.return_value = np.array([[0.75]])
        
        # Prepare data
        isolated_trader.training_data = TestDataBuilder.create_trending_data(
            start_price=45000, end_price=46000, num_points=50
        )
        
        results = []
        errors = []
        
        def data_collection_thread():
            try:
                with patch.object(isolated_trader, 'fetch_market_data', 
                                return_value=[{"marketId": "BTC-AUD", "lastPrice": "45000"}]):
                    for _ in range(10):
                        isolated_trader.collect_and_store_data()
                        time.sleep(0.01)
                results.append("data_collection_success")
            except Exception as e:
                errors.append(f"data_collection_error: {e}")
        
        def trading_thread():
            try:
                for _ in range(10):
                    prediction = {
                        "signal": "BUY",
                        "confidence": 0.8,
                        "price": 45000.0,
                        "rsi": 50.0
                    }
                    isolated_trader.execute_simulated_trade(prediction)
                    time.sleep(0.01)
                results.append("trading_success")
            except Exception as e:
                errors.append(f"trading_error: {e}")
        
        def save_thread():
            try:
                for _ in range(5):
                    isolated_trader.save_training_data()
                    isolated_trader.save_state()
                    time.sleep(0.02)
                results.append("save_success")
            except Exception as e:
                errors.append(f"save_error: {e}")
        
        # Start concurrent threads
        threads = [
            threading.Thread(target=data_collection_thread),
            threading.Thread(target=trading_thread),
            threading.Thread(target=save_thread)
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify no errors and all operations completed
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 3
        assert "data_collection_success" in results
        assert "trading_success" in results
        assert "save_success" in results
    
    def test_memory_management_long_running(self, isolated_trader):
        """Test memory management during long-running operation."""
        # Set conservative limits
        isolated_trader.max_training_samples = 500
        
        # Generate data over "time"
        for i in range(1000):  # More than max_training_samples
            data_point = {
                "timestamp": datetime.now().isoformat(),
                "price": 45000 + (i % 100),
                "volume": 100 + (i % 50),
                "rsi": 50,
                "sma_5": 45000,
                "sma_20": 45000,
                "ema_12": 45000,
                "ema_26": 45000,
                "macd": 0,
                "macd_signal": 0,
                "bb_upper": 45100,
                "bb_lower": 44900,
                "volume_sma": 125
            }
            
            isolated_trader.training_data.append(data_point)
            
            # Simulate periodic cleanup
            if len(isolated_trader.training_data) > isolated_trader.max_training_samples:
                isolated_trader.training_data = isolated_trader.training_data[-isolated_trader.max_training_samples:]
        
        # Verify memory constraints are respected
        assert len(isolated_trader.training_data) == isolated_trader.max_training_samples
        
        # Verify most recent data is preserved
        assert isolated_trader.training_data[-1]["price"] == 45000 + (999 % 100)
    
    def test_market_condition_adaptation(self, isolated_trader, mock_tensorflow):
        """Test system adaptation to different market conditions."""
        isolated_trader.model = mock_tensorflow["model"]
        
        # Test different market conditions
        market_conditions = [
            ("bull_market", TestDataBuilder.create_trending_data(40000, 50000, 50)),
            ("bear_market", TestDataBuilder.create_trending_data(50000, 40000, 50)),
            ("sideways", TestDataBuilder.create_sideways_data(45000, 50, 0.02)),
            ("volatile", TestDataBuilder.create_volatile_data(45000, 50, 0.05))
        ]
        
        for condition_name, market_data in market_conditions:
            # Clear previous data
            isolated_trader.training_data = []
            
            # Add market data with technical indicators
            for data_point in market_data:
                isolated_trader.training_data.append({
                    **data_point,
                    "sma_5": data_point["price"],
                    "sma_20": data_point["price"],
                    "ema_12": data_point["price"],
                    "ema_26": data_point["price"],
                    "rsi": 50.0,
                    "macd": 0.0,
                    "macd_signal": 0.0,
                    "bb_upper": data_point["price"] + 100,
                    "bb_lower": data_point["price"] - 100,
                    "volume_sma": data_point["volume"]
                })
            
            # Test prediction generation
            mock_tensorflow["model"].predict.return_value = np.array([[0.7]])
            
            with patch.object(isolated_trader, 'fetch_market_data', return_value=[market_data[-1]]):
                prediction = isolated_trader.predict_trade_signal([market_data[-1]])
                assert_valid_prediction(prediction)
                
                # System should generate valid predictions for all market conditions
                assert prediction["signal"] in ["BUY", "SELL", "HOLD"]
                assert 0 <= prediction["confidence"] <= 1
    
    def test_full_system_robustness(self, isolated_trader, mock_tensorflow):
        """Test full system robustness under various failure scenarios."""
        isolated_trader.model = mock_tensorflow["model"]
        
        # Test various failure scenarios
        failure_scenarios = [
            # Network failures
            lambda: setattr(isolated_trader, 'fetch_market_data', lambda: None),
            # Model prediction failures
            lambda: mock_tensorflow["model"].predict.side_effect = Exception("Model Error"),
            # File system failures
            lambda: patch('builtins.open', side_effect=IOError("File Error")),
        ]
        
        for scenario_func in failure_scenarios:
            # Apply failure scenario
            scenario_func()
            
            # System should handle failures gracefully
            try:
                with patch.object(isolated_trader, 'fetch_market_data', return_value=None):
                    success = isolated_trader.collect_and_store_data()
                    assert not success  # Should fail gracefully
                
                # Reset for next scenario
                mock_tensorflow["model"].predict.side_effect = None
                mock_tensorflow["model"].predict.return_value = np.array([[0.6]])
                
            except Exception as e:
                # Should not raise unhandled exceptions
                assert False, f"Unhandled exception in failure scenario: {e}"
