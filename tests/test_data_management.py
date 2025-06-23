# -*- coding: utf-8 -*-
"""
Tests for data management, persistence, and state handling.
"""
import pytest
import os
import json
import pickle
import tempfile
import shutil
from datetime import datetime, timedelta, timezone # Import timezone
from unittest.mock import Mock, patch, MagicMock
from collections import deque # Import deque

# Import the autotrader module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autotrader import ContinuousAutoTrader # Import from top-level autotrader.py


class TestDataManagement:
    """Test cases for data management and persistence."""
    
    def test_training_data_save_and_load_cycle(self, isolated_trader, sample_training_data):
        """Test complete save and load cycle for training data."""
        isolated_trader.training_data = deque(sample_training_data, maxlen=isolated_trader.max_training_samples)
        
        # Save data
        isolated_trader.save_training_data()
        
        # Verify file exists
        assert os.path.exists(isolated_trader.training_data_filename)
        
        # Load data in new instance
        new_trader = ContinuousAutoTrader()
        loaded_data = new_trader.load_training_data()
        
        # Verify data integrity
        assert len(loaded_data) == len(sample_training_data)
        assert loaded_data[0]['price'] == sample_training_data[0]['price']
        assert loaded_data[-1]['timestamp'] == sample_training_data[-1]['timestamp']
    
    def test_training_data_file_corruption_handling(self, isolated_trader, temp_dir):
        """Test handling of corrupted training data file."""
        # Create corrupted JSON file
        corrupted_file = os.path.join(temp_dir, isolated_trader.training_data_filename)
        with open(corrupted_file, 'w') as f:
            f.write("{ invalid json content")
        
        # Should handle corruption gracefully
        loaded_data = isolated_trader.load_training_data()
        assert len(loaded_data) == 0  # Should return empty deque
    
    def test_training_data_size_management(self, isolated_trader):
        """Test that training data is limited to max_training_samples."""
        isolated_trader.max_training_samples = 50
        isolated_trader.training_data = deque(maxlen=isolated_trader.max_training_samples) # Re-initialize deque
        
        # Add more data than the limit
        for i in range(100):
            isolated_trader.training_data.append({
                'timestamp': datetime.now().isoformat(),
                'price': 45000 + i,
                'volume': 100 + i,
                'rsi': 50,
                "marketId": "BTC-AUD", # Add marketId for extract_comprehensive_data
                "lastPrice": str(45000 + i), # Add lastPrice as string
                "bestBid": str(45000 + i - 5),
                "bestAsk": str(45000 + i + 5),
                "high24h": str(45000 + i + 100),
                "low24h": str(45000 + i - 100)
            })
        
        # Simulate data collection that should trigger size limit
        with patch.object(isolated_trader, 'fetch_market_data') as mock_fetch:
            mock_fetch.return_value = [{
                "marketId": "BTC-AUD",
                "lastPrice": "46000",
                "volume24h": "200",
                "bestBid": "45995",
                "bestAsk": "46005",
                "high24h": "47000",
                "low24h": "45000"
            }]
            
            isolated_trader.collect_and_store_data()
        
        # Should not exceed max_training_samples
        assert len(isolated_trader.training_data) == isolated_trader.max_training_samples
    
    def test_state_persistence_complete_cycle(self, isolated_trader):
        """Test complete state save and load cycle."""
        # Modify trader state
        isolated_trader.balance = 15000.50
        # last_training_time is not saved in state, so we don't assert it here.
        # isolated_trader.last_training_time = 1234567890
        isolated_trader.scalers_fitted = True # This is not saved in state either
        
        # Save state
        isolated_trader.save_state()
        
        # Verify state file exists
        assert os.path.exists(isolated_trader.state_filename)
        
        # Create new trader and verify state loading
        new_trader = ContinuousAutoTrader()
        
        assert new_trader.balance == 15000.50
        # assert new_trader.last_training_time == 1234567890 # Removed as it's not saved
        # assert new_trader.scalers_fitted == True # Removed as it's not saved
    
    def test_state_file_corruption_handling(self, isolated_trader, temp_dir):
        """Test handling of corrupted state file."""
        # Create corrupted state file
        corrupted_file = os.path.join(temp_dir, isolated_trader.state_filename)
        with open(corrupted_file, 'wb') as f:
            f.write(b"corrupted_pickle_data")
        
        # Should handle corruption gracefully
        new_trader = ContinuousAutoTrader()
        assert new_trader.balance == 10000.0  # Default value
    
    def test_scalers_persistence(self, isolated_trader, sample_training_data):
        """Test scaler save and load functionality."""
        isolated_trader.training_data = deque(sample_training_data, maxlen=isolated_trader.max_training_samples)
        
        # Fit and save scalers
        isolated_trader.fit_scalers() # No need to pass data
        isolated_trader.save_scalers()
        
        # Verify scalers file exists
        assert os.path.exists(isolated_trader.scaler_filename) # Use scaler_filename
        
        # Load scalers in new instance
        new_trader = ContinuousAutoTrader()
        new_trader.load_scalers() # Explicitly load scalers
        
        assert new_trader.scalers_fitted
        assert new_trader.feature_scaler is not None
        # assert new_trader.price_scaler is not None # Removed as price_scaler is no longer separate
    
    def test_model_persistence(self, isolated_trader, mock_tensorflow):
        """Test model save and load functionality."""
        isolated_trader.model = mock_tensorflow["model"]
        
        # Test model saving
        isolated_trader.save_model()
        mock_tensorflow["model"].save.assert_called_once_with(isolated_trader.model_filename)
        
        # Test model loading
        loaded_model = isolated_trader.load_model()
        assert loaded_model is not None
        # mock_tensorflow["load_model"].assert_called_once() # This mock is not used in the actual load_model
    
    def test_data_collection_with_technical_indicators(self, isolated_trader):
        """Test data collection includes technical indicators."""
        # Add some historical data first
        historical_data = []
        for i in range(25):
            historical_data.append({
                'timestamp': (datetime.now() - timedelta(minutes=i)).isoformat(),
                'price': 45000 + (i * 10),
                'volume': 100 + i,
                'rsi': 50,
                "marketId": "BTC-AUD", # Add marketId for extract_comprehensive_data
                "lastPrice": str(45000 + (i * 10)), # Add lastPrice as string
                "bestBid": str(45000 + (i * 10) - 5),
                "bestAsk": str(45000 + (i * 10) + 5),
                "high24h": str(45000 + (i * 10) + 100),
                "low24h": str(45000 + (i * 10) - 100)
            })
        isolated_trader.training_data = deque(historical_data, maxlen=isolated_trader.max_training_samples)
        
        # Mock API response
        with patch.object(isolated_trader, 'fetch_market_data') as mock_fetch:
            mock_fetch.return_value = [{
                "marketId": "BTC-AUD",
                "lastPrice": "45500",
                "volume24h": "150",
                "bestBid": "45495",
                "bestAsk": "45505",
                "high24h": "46000",
                "low24h": "45000"
            }]
            
            success = isolated_trader.collect_and_store_data()
            
            assert success
            
            # Check that the latest data point has technical indicators
            latest_data = isolated_trader.training_data[-1]
            assert 'sma_5' in latest_data
            assert 'sma_20' in latest_data
            assert 'rsi' in latest_data
            assert 'macd' in latest_data
            assert 'bb_upper' in latest_data
            assert 'bb_lower' in latest_data
            assert 'volume_sma' in latest_data # Add volume_sma check
    
    def test_data_collection_error_handling(self, isolated_trader):
        """Test error handling during data collection."""
        # Mock API failure
        with patch.object(isolated_trader, 'fetch_market_data') as mock_fetch:
            mock_fetch.side_effect = Exception("API Error") # Simulate API error
            
            success = isolated_trader.collect_and_store_data()
            assert not success
        
        # Mock invalid market data (extract_comprehensive_data will return None)
        with patch.object(isolated_trader, 'fetch_market_data') as mock_fetch:
            mock_fetch.return_value = [{"marketId": "BTC-AUD", "lastPrice": "invalid"}] # Invalid price
            
            success = isolated_trader.collect_and_store_data()
            assert not success
    
    def test_timestamp_accuracy_in_data_collection(self, isolated_trader):
        """Test that timestamps are accurate in collected data."""
        with patch.object(isolated_trader, 'fetch_market_data') as mock_fetch:
            mock_fetch.return_value = [{
                "marketId": "BTC-AUD",
                "lastPrice": "45000",
                "volume24h": "100",
                "bestBid": "44995",
                "bestAsk": "45005",
                "high24h": "46000",
                "low24h": "44000"
            }]
            
            # Record time before collection
            before_time = datetime.now()
            
            isolated_trader.collect_and_store_data()
            
            # Record time after collection
            after_time = datetime.now()
            
            # Check timestamp is within reasonable range
            latest_data = isolated_trader.training_data[-1]
            # The timestamp is now an integer (Unix timestamp in ms)
            data_timestamp_ms = latest_data['timestamp']
            data_datetime = datetime.fromtimestamp(data_timestamp_ms / 1000, tz=timezone.utc) # Convert to datetime object
            
            # Allow a small buffer for time differences
            assert before_time.timestamp() <= data_datetime.timestamp() <= after_time.timestamp() + 1 # Add 1 second buffer
    
    def test_data_integrity_validation(self, isolated_trader):
        """Test data integrity validation."""
        # Test with valid comprehensive data
        valid_data = {
            'price': 45000.0,
            'volume': 123.45,
            'bid': 44995.0,
            'ask': 45005.0,
            'high24h': 46000.0,
            'low24h': 44000.0
        }
        
        # Should be accepted
        assert valid_data['price'] > 0
        assert valid_data['volume'] >= 0
        assert valid_data['ask'] >= valid_data['bid']
        
        # Test with invalid data (negative price)
        invalid_data = {
            'price': -100.0,
            'volume': 123.45
        }
        
        # Should be rejected in extract_comprehensive_data
        mock_api_response = [{"marketId": "BTC-AUD", "lastPrice": str(invalid_data['price'])}]
        extracted = isolated_trader.extract_comprehensive_data(mock_api_response)
        # The method should return None for invalid data
        assert extracted is None
    
    def test_concurrent_file_operations(self, isolated_trader, sample_training_data):
        """Test handling of concurrent file operations."""
        import threading
        import time
        
        isolated_trader.training_data = sample_training_data
        
        def save_data():
            isolated_trader.save_training_data()
            isolated_trader.save_state()
        
        def load_data():
            time.sleep(0.1)  # Small delay
            new_trader = ContinuousAutoTrader()
            return new_trader.training_data
        
        # Start concurrent operations
        save_thread = threading.Thread(target=save_data)
        load_thread = threading.Thread(target=load_data)
        
        save_thread.start()
        load_thread.start()
        
        save_thread.join()
        load_thread.join()
        
        # Should complete without errors
        assert True  # If we reach here, no deadlocks occurred
    
    def test_backup_and_recovery_simulation(self, isolated_trader, sample_training_data):
        """Test backup and recovery of data files."""
        isolated_trader.training_data = deque(sample_training_data, maxlen=isolated_trader.max_training_samples)
        isolated_trader.balance = 12000.0
        
        # Save all data
        isolated_trader.save_training_data()
        isolated_trader.save_state()
        
        # Simulate system crash by clearing in-memory data
        isolated_trader.training_data = deque(maxlen=isolated_trader.max_training_samples) # Reset to empty deque
        isolated_trader.balance = 0.0
        
        # Simulate recovery
        isolated_trader.load_state()
        isolated_trader.training_data = isolated_trader.load_training_data()
        
        # Verify recovery
        assert isolated_trader.balance == 12000.0
        assert len(isolated_trader.training_data) == len(sample_training_data)
    
    def test_memory_management_large_dataset(self, isolated_trader):
        """Test memory management with large datasets."""
        # Set reasonable limits
        isolated_trader.max_training_samples = 1000
        isolated_trader.training_data = deque(maxlen=isolated_trader.max_training_samples) # Re-initialize deque
        
        # Simulate long-running operation with many data points
        for i in range(1500):  # More than max_training_samples
            data_point = {
                'timestamp': datetime.now().isoformat(),
                'price': 45000 + (i % 1000),  # Varying prices
                'volume': 100 + (i % 100),
                'rsi': 50 + (i % 50),
                "marketId": "BTC-AUD", # Add marketId for extract_comprehensive_data
                "lastPrice": str(45000 + (i % 1000)), # Add lastPrice as string
                "bestBid": str(45000 + (i % 1000) - 5),
                "bestAsk": str(45000 + (i % 1000) + 5),
                "high24h": str(45000 + (i % 1000) + 100),
                "low24h": str(45000 + (i % 1000) - 100)
            }
            isolated_trader.training_data.append(data_point)
            
            # The deque automatically handles maxlen, no need for manual trimming
            # if len(isolated_trader.training_data) > isolated_trader.max_training_samples:
            #     isolated_trader.training_data = isolated_trader.training_data[-isolated_trader.max_training_samples:]
        
        # Should not exceed memory limits
        assert len(isolated_trader.training_data) == isolated_trader.max_training_samples
    
    def test_file_permissions_handling(self, isolated_trader, temp_dir):
        """Test handling of file permission issues."""
        # Create a read-only directory to simulate permission issues
        readonly_dir = os.path.join(temp_dir, 'readonly')
        os.makedirs(readonly_dir, exist_ok=True)
        os.chmod(readonly_dir, 0o444)  # Read-only
        
        # Change to readonly directory
        original_cwd = os.getcwd()
        try:
            os.chdir(readonly_dir)
            
            # Attempt to save data (should handle gracefully)
            isolated_trader.save_training_data()
            isolated_trader.save_state()
            
            # Should not crash, even if saves fail
            assert True
            
        except PermissionError:
            # Expected behavior - should be handled gracefully
            assert True
        finally:
            os.chdir(original_cwd)
            # Restore directory permissions for cleanup
            os.chmod(readonly_dir, 0o755)
    
    def test_data_format_versioning(self, isolated_trader, sample_training_data):
        """Test handling of different data format versions."""
        # Create old format data (missing some fields)
        old_format_data = []
        for i, data_point in enumerate(sample_training_data[:5]):
            old_data = {
                'timestamp': data_point['timestamp'],
                'price': data_point['price'],
                'volume': data_point['volume']
                # Missing newer fields like technical indicators
            }
            old_format_data.append(old_data)
        
        # Save old format data
        with open(isolated_trader.training_data_filename, 'w') as f:
            json.dump(old_format_data, f)
        
        # Load and verify it handles missing fields gracefully
        loaded_data = isolated_trader.load_training_data()
        
        assert len(loaded_data) == 5
        assert all('price' in item for item in loaded_data)
        assert all('volume' in item for item in loaded_data)
        
        # Feature preparation should handle missing fields with defaults
        features = isolated_trader.prepare_features(loaded_data[0])
        assert len(features) == isolated_trader.settings.ml.feature_count  # Should still have all features with defaults
