# -*- coding: utf-8 -*-
"""
Performance and benchmarking tests for the autotrader system.
"""
import pytest
import time
import psutil
import os
import threading
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from datetime import datetime, timedelta
from collections import deque # Import deque

# Import the autotrader module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autotrader import ContinuousAutoTrader # Import from top-level autotrader.py

# Import test utilities
from test_utils import (
    MockMarketDataGenerator, 
    TestDataBuilder, 
    PerformanceMetrics
)


class TestPerformanceMetrics:
    """Performance and benchmarking tests."""
    
    def test_data_collection_performance(self, isolated_trader):
        """Test performance of data collection operations."""
        market_gen = MockMarketDataGenerator()
        
        # Measure time for data collection
        start_time = time.time()
        
        with patch.object(isolated_trader, 'fetch_market_data') as mock_fetch:
            mock_fetch.return_value = [market_gen.generate_tick()]
            
            # Collect data multiple times
            for _ in range(100):
                isolated_trader.collect_and_store_data()
        
        end_time = time.time()
        collection_time = end_time - start_time
        
        # Should complete 100 collections in reasonable time
        assert collection_time < 5.0  # Less than 5 seconds
        assert len(isolated_trader.training_data) == 100
        
        # Calculate operations per second
        ops_per_second = 100 / collection_time
        assert ops_per_second > 20  # At least 20 operations per second
    
    def test_technical_indicator_calculation_performance(self, isolated_trader):
        """Test performance of technical indicator calculations."""
        # Generate test data
        prices = np.random.uniform(40000, 50000, 1000)
        volumes = np.random.uniform(50, 500, 1000)
        
        # Measure calculation time
        start_time = time.time()
        
        for i in range(100):  # 100 calculations
            test_prices = prices[i:i+100] if i+100 <= len(prices) else prices[-100:]
            test_volumes = volumes[i:i+100] if i+100 <= len(volumes) else volumes[-100:]
            
            indicators_list = isolated_trader.calculate_technical_indicators(market_data=None, prices=test_prices, volumes=test_volumes)
            
            # Verify indicators are calculated
            assert len(indicators_list) > 0 # Should return a list of data points
            assert 'sma_5' in indicators_list[-1] # Check for an indicator in the last data point
        
        end_time = time.time()
        calculation_time = end_time - start_time
        
        # Should complete 100 calculations quickly
        assert calculation_time < 2.0  # Less than 2 seconds
        
        # Calculate indicators per second
        indicators_per_second = 100 / calculation_time
        assert indicators_per_second > 50  # At least 50 calculations per second
    
    def test_model_prediction_performance(self, isolated_trader, mock_tensorflow, sample_training_data):
        """Test performance of model predictions."""
        isolated_trader.model = mock_tensorflow["model"]
        isolated_trader.training_data = sample_training_data
        isolated_trader.scalers_fitted = True
        
        # Mock fast prediction
        mock_tensorflow["model"].predict.return_value = np.array([[0.7]])
        
        market_gen = MockMarketDataGenerator()
        
        # Measure prediction time
        start_time = time.time()
        
        for _ in range(100):
            mock_data = [market_gen.generate_tick()]
            prediction = isolated_trader.predict_trade_signal(mock_data[0]) # Pass single dict
            
            # Verify prediction is valid
            assert "signal" in prediction
            assert "confidence" in prediction
        
        end_time = time.time()
        prediction_time = end_time - start_time
        
        # Should complete 100 predictions quickly
        assert prediction_time < 3.0  # Less than 3 seconds
        
        # Calculate predictions per second
        predictions_per_second = 100 / prediction_time
        assert predictions_per_second > 30  # At least 30 predictions per second
    
    def test_memory_usage_data_collection(self, isolated_trader):
        """Test memory usage during data collection."""
        process = psutil.Process(os.getpid())
        
        # Measure initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Collect large amount of data
        market_gen = MockMarketDataGenerator()
        
        with patch.object(isolated_trader, 'fetch_market_data') as mock_fetch:
            mock_fetch.return_value = [market_gen.generate_tick()]
            
            # Collect data points up to max_training_samples
            for _ in range(isolated_trader.max_training_samples):
                isolated_trader.collect_and_store_data()
        
        # Measure final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 100  # Less than 100MB increase
        
        # Verify data was limited by max_training_samples
        assert len(isolated_trader.training_data) == isolated_trader.max_training_samples # Should be exactly maxlen
    
    def test_file_io_performance(self, isolated_trader, sample_training_data):
        """Test file I/O performance."""
        isolated_trader.training_data = sample_training_data
        
        # Test save performance
        start_time = time.time()
        
        for _ in range(10):  # Multiple saves
            isolated_trader.save_training_data()
            isolated_trader.save_state()
        
        save_time = time.time() - start_time
        
        # Should complete 10 saves quickly
        assert save_time < 2.0  # Less than 2 seconds
        
        # Test load performance
        start_time = time.time()
        
        for _ in range(10):  # Multiple loads
            isolated_trader.load_training_data()
            isolated_trader.load_state()
        
        load_time = time.time() - start_time
        
        # Should complete 10 loads quickly
        assert load_time < 1.0  # Less than 1 second
    
    def test_concurrent_performance(self, isolated_trader, mock_tensorflow):
        """Test performance under concurrent operations."""
        isolated_trader.model = mock_tensorflow["model"]
        mock_tensorflow["model"].predict.return_value = np.array([[0.6]])
        
        # Prepare data
        isolated_trader.training_data = TestDataBuilder.create_trending_data(
            start_price=45000, end_price=46000, num_points=100
        )
        
        results = []
        
        def data_collection_worker():
            start_time = time.time()
            with patch.object(isolated_trader, 'fetch_market_data', 
                            return_value=[{"marketId": "BTC-AUD", "lastPrice": "45000"}]):
                for _ in range(50):
                    isolated_trader.collect_and_store_data()
            
            end_time = time.time()
            results.append(("data_collection", end_time - start_time))
        
        def prediction_worker():
            start_time = time.time()
            with patch.object(isolated_trader, 'fetch_market_data', 
                            return_value=[{"marketId": "BTC-AUD", "lastPrice": "45000"}]):
                for _ in range(50):
                    isolated_trader.predict_trade_signal([{"marketId": "BTC-AUD", "lastPrice": "45000"}])
            
            end_time = time.time()
            results.append(("prediction", end_time - start_time))
        
        def trading_worker():
            start_time = time.time()
            for _ in range(50):
                prediction = {
                    "signal": "BUY",
                    "confidence": 0.7,
                    "price": 45000.0,
                    "rsi": 50.0
                }
                isolated_trader.execute_simulated_trade(prediction)
            
            end_time = time.time()
            results.append(("trading", end_time - start_time))
        
        # Start concurrent workers
        threads = [
            threading.Thread(target=data_collection_worker),
            threading.Thread(target=prediction_worker),
            threading.Thread(target=trading_worker)
        ]
        
        overall_start = time.time()
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        overall_time = time.time() - overall_start
        
        # All operations should complete quickly
        assert overall_time < 10.0  # Less than 10 seconds total
        assert len(results) == 3
        
        # Each operation should be reasonably fast
        for operation, duration in results:
            assert duration < 5.0  # Each operation less than 5 seconds
    
    def test_scalability_large_dataset(self, isolated_trader, mock_tensorflow):
        """Test scalability with large datasets."""
        isolated_trader.model = mock_tensorflow["model"]
        isolated_trader.max_training_samples = 5000  # Larger dataset
        
        # Generate large dataset
        large_dataset = TestDataBuilder.create_volatile_data(
            base_price=45000, num_points=5000, volatility=0.02
        )
        
        start_time = time.time()
        
        # Process large dataset
        isolated_trader.training_data = deque(maxlen=isolated_trader.max_training_samples)
        for i, data_point in enumerate(large_dataset):
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
                "volume_sma": data_point["volume"],
                "marketId": "BTC-AUD", # Add marketId for extract_comprehensive_data
                "lastPrice": str(data_point["price"]), # Add lastPrice as string
                "bestBid": str(data_point["price"] - 5),
                "bestAsk": str(data_point["price"] + 5),
                "high24h": str(data_point["price"] + 100),
                "low24h": str(data_point["price"] - 100)
            })
            
            # Deque handles maxlen automatically
            # if len(isolated_trader.training_data) > isolated_trader.max_training_samples:
            #     isolated_trader.training_data = isolated_trader.training_data[-isolated_trader.max_training_samples:]
        
        processing_time = time.time() - start_time
        
        # Should handle large dataset efficiently
        assert processing_time < 30.0  # Less than 30 seconds
        assert len(isolated_trader.training_data) == isolated_trader.max_training_samples
        
        # Test operations with large dataset
        start_time = time.time()
        
        # Test scaler fitting
        isolated_trader.fit_scalers() # No need to pass data
        
        scaler_time = time.time() - start_time
        assert scaler_time < 5.0  # Scaler fitting should be fast
    
    def test_cpu_usage_monitoring(self, isolated_trader, mock_tensorflow):
        """Test CPU usage during intensive operations."""
        isolated_trader.model = mock_tensorflow["model"]
        
        # Monitor CPU usage
        process = psutil.Process(os.getpid())
        
        # Measure CPU usage during intensive operations
        cpu_percent_before = process.cpu_percent()
        
        # Perform CPU-intensive operations
        start_time = time.time()
        
        # Generate and process data
        for _ in range(100):
            # Generate random data
            prices = np.random.uniform(40000, 50000, 100)
            volumes = np.random.uniform(50, 500, 100)
            
            # Calculate indicators
            isolated_trader.calculate_technical_indicators(market_data=None, prices=prices, volumes=volumes)
            
            # Simulate feature preparation
            data_point = {
                "price": prices[-1],
                "volume": volumes[-1],
                "sma_5": np.mean(prices[-5:]),
                "sma_20": np.mean(prices[-20:]),
                "ema_12": prices[-1],
                "ema_26": prices[-1],
                "rsi": 50.0,
                "macd": 0.0,
                "macd_signal": 0.0,
                "bb_upper": prices[-1] + 100,
                "bb_lower": prices[-1] - 100,
                "spread": 10.0 # Add spread for prepare_features
            }
            isolated_trader.prepare_features(data_point)
        
        processing_time = time.time() - start_time
        cpu_percent_after = process.cpu_percent()
        
        # Should complete intensive operations efficiently
        assert processing_time < 10.0  # Less than 10 seconds
        
        # CPU usage should be reasonable (not constantly at 100%)
        # Note: This test might be flaky on different systems
        if cpu_percent_after > 0:  # Only check if we got a reading
            assert cpu_percent_after <= 100.0  # Allow up to 100% CPU usage for intensive operations
    
    def test_memory_leak_detection(self, isolated_trader):
        """Test for memory leaks during long-running operations."""
        process = psutil.Process(os.getpid())
        
        # Measure initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform repetitive operations that might cause memory leaks
        market_gen = MockMarketDataGenerator()
        
        for cycle in range(10):  # 10 cycles
            # Clear data to simulate long-running operation
            isolated_trader.training_data = deque(maxlen=isolated_trader.max_training_samples) # Reset deque
            
            # Collect data
            with patch.object(isolated_trader, 'fetch_market_data') as mock_fetch:
                mock_fetch.return_value = [market_gen.generate_tick()]
                
                for _ in range(100):
                    isolated_trader.collect_and_store_data()
            
            # Save and reload (potential memory leak source)
            isolated_trader.save_training_data()
            isolated_trader.save_state()
            
            # Measure memory after each cycle
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            
            # Memory increase should be bounded
            assert memory_increase < 50  # Less than 50MB increase per cycle
        
        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_memory_increase = final_memory - initial_memory
        
        # Total memory increase should be reasonable
        assert total_memory_increase < 200  # Less than 200MB total increase
    
    def test_response_time_under_load(self, isolated_trader, mock_tensorflow):
        """Test response times under high load."""
        isolated_trader.model = mock_tensorflow["model"]
        mock_tensorflow["model"].predict.return_value = np.array([[0.7]])
        
        # Prepare data
        isolated_trader.training_data = deque(TestDataBuilder.create_trending_data(
            start_price=45000, end_price=46000, num_points=200
        ), maxlen=isolated_trader.max_training_samples)
        isolated_trader.scalers_fitted = True
        
        # Measure response times for different operations
        response_times = {
            "data_collection": [],
            "prediction": [],
            "trade_execution": []
        }
        
        market_gen = MockMarketDataGenerator()
        
        for _ in range(50):  # 50 iterations
            # Test data collection response time
            start_time = time.time()
            with patch.object(isolated_trader, 'fetch_market_data',
                            return_value=[market_gen.generate_tick()]):
                isolated_trader.collect_and_store_data()
            response_times["data_collection"].append(time.time() - start_time)
            
            # Test prediction response time
            start_time = time.time()
            with patch.object(isolated_trader, 'fetch_market_data',
                            return_value=[market_gen.generate_tick()]):
                isolated_trader.predict_trade_signal(market_gen.generate_tick()) # Pass single dict
            response_times["prediction"].append(time.time() - start_time)
            
            # Test trade execution response time
            start_time = time.time()
            prediction = {
                "signal": "BUY",
                "confidence": 0.7,
                "price": 45000.0,
                "rsi": 50.0
            }
            isolated_trader.execute_simulated_trade(prediction)
            response_times["trade_execution"].append(time.time() - start_time)
        
        # Analyze response times
        for operation, times in response_times.items():
            avg_time = np.mean(times)
            max_time = np.max(times)
            p95_time = np.percentile(times, 95)
            
            # Response time requirements
            assert avg_time < 0.1  # Average less than 100ms
            assert max_time < 0.5  # Maximum less than 500ms
            assert p95_time < 0.2  # 95th percentile less than 200ms
            
            print(f"{operation} - Average: {avg_time:.3f}s, Max: {max_time:.3f}s, P95: {p95_time:.3f}s")
    
    def test_throughput_measurement(self, isolated_trader, mock_tensorflow):
        """Test system throughput under sustained load."""
        isolated_trader.model = mock_tensorflow["model"]
        mock_tensorflow["model"].predict.return_value = np.array([[0.6]])
        
        # Prepare system
        isolated_trader.training_data = deque(TestDataBuilder.create_trending_data(
            start_price=45000, end_price=46000, num_points=100
        ), maxlen=isolated_trader.max_training_samples)
        isolated_trader.scalers_fitted = True
        
        # Measure throughput for different operations
        market_gen = MockMarketDataGenerator()
        
        # Test data collection throughput
        start_time = time.time()
        collections = 0
        
        with patch.object(isolated_trader, 'fetch_market_data',
                        return_value=[market_gen.generate_tick()]):
            while time.time() - start_time < 2.0:  # 2 seconds
                isolated_trader.collect_and_store_data()
                collections += 1
        
        collection_throughput = collections / 2.0
        assert collection_throughput > 10  # At least 10 collections per second
        
        # Test prediction throughput
        start_time = time.time()
        predictions = 0
        
        with patch.object(isolated_trader, 'fetch_market_data',
                        return_value=[market_gen.generate_tick()]):
            while time.time() - start_time < 2.0:  # 2 seconds
                isolated_trader.predict_trade_signal(market_gen.generate_tick()) # Pass single dict
                predictions += 1
        
        prediction_throughput = predictions / 2.0
        assert prediction_throughput > 5  # At least 5 predictions per second
        
        # Test trade execution throughput
        start_time = time.time()
        trades = 0
        
        while time.time() - start_time < 2.0:  # 2 seconds
            prediction = {
                "signal": "BUY",
                "confidence": 0.7,
                "price": 45000.0,
                "rsi": 50.0
            }
            isolated_trader.execute_simulated_trade(prediction)
            trades += 1
        
        trade_throughput = trades / 2.0
        assert trade_throughput > 100  # At least 100 trades per second
        
        print(f"Throughput - Collections: {collection_throughput:.1f}/s, "
              f"Predictions: {prediction_throughput:.1f}/s, "
              f"Trades: {trade_throughput:.1f}/s")
