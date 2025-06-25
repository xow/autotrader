"""
Tests for the AutoTrader configuration management system.
"""

import os
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from autotrader.config.config import (
    Config, Environment, APIConfig, TradingConfig, 
    MLConfig, OperationalConfig, load_config
)
from autotrader.config.settings import Settings, get_settings


class TestConfig:
    """Test the Config class"""
    
    def test_default_config_creation(self):
        """Test creating a default configuration"""
        config = Config()
        
        assert config.environment == Environment.DEVELOPMENT
        assert config.api.base_url == "https://api.btcmarkets.net/v3"
        assert config.trading.initial_balance == 10000.0
        assert config.ml.sequence_length == 20
        assert config.operations.data_collection_interval == 60
    
    def test_config_validation_success(self):
        """Test successful configuration validation"""
        config = Config()
        # Should not raise any exceptions
        config._validate_config()
    
    def test_config_validation_failures(self):
        """Test configuration validation failures"""
        config = Config()
        
        # Test negative balance
        config.trading.initial_balance = -1000
        with pytest.raises(ValueError, match="initial_balance must be positive"):
            config._validate_config()
        
        # Reset and test invalid fee rate
        config = Config()
        config.trading.fee_rate = 1.5
        with pytest.raises(ValueError, match="fee_rate must be between 0 and 1"):
            config._validate_config()
    
    def test_environment_overrides(self):
        """Test environment variable overrides"""
        test_env_vars = {
            'BTCMARKETS_API_KEY': 'test-key',
            'BTCMARKETS_API_SECRET': 'test-secret',
            'TRADING_INITIAL_BALANCE': '5000.0',
            'AUTOTRADER_ENV': 'production',
            'LOG_LEVEL': 'DEBUG'
        }
        
        with patch.dict(os.environ, test_env_vars):
            config = Config()
            
            assert config.api.api_key == 'test-key'
            assert config.api.api_secret == 'test-secret'
            assert config.trading.initial_balance == 5000.0
            assert config.environment == Environment.PRODUCTION
            assert config.operations.log_level == 'DEBUG'
    
    def test_api_credentials(self):
        """Test API credential handling"""
        config = Config()
        
        # Test without credentials
        api_key, api_secret = config.get_api_credentials()
        assert api_key is None
        assert api_secret is None
        assert not config.has_valid_api_credentials()
        
        # Test with environment variables
        with patch.dict(os.environ, {
            'BTCMARKETS_API_KEY': 'env-key',
            'BTCMARKETS_API_SECRET': 'env-secret'
        }):
            api_key, api_secret = config.get_api_credentials()
            assert api_key == 'env-key'
            assert api_secret == 'env-secret'
            assert config.has_valid_api_credentials()
    
    def test_config_file_operations(self):
        """Test saving and loading configuration files"""
        config = Config()
        config.trading.initial_balance = 15000.0
        config.ml.sequence_length = 25
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save configuration
            config.to_file(temp_path)
            
            # Load configuration
            loaded_config = Config.from_file(temp_path)
            
            assert loaded_config.trading.initial_balance == 15000.0
            assert loaded_config.ml.sequence_length == 25
            
            # API credentials should be None in saved file
            with open(temp_path, 'r') as f:
                saved_data = json.load(f)
            assert saved_data['api']['api_key'] is None
            assert saved_data['api']['api_secret'] is None
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_config_to_dict(self):
        """Test configuration to dictionary conversion"""
        config = Config()
        config_dict = config.to_dict()
        
        assert 'environment' in config_dict
        assert 'api' in config_dict
        assert 'trading' in config_dict
        assert 'ml' in config_dict
        assert 'operations' in config_dict
        
        assert config_dict['environment'] == 'development'
        assert config_dict['trading']['initial_balance'] == 10000.0


class TestSettings:
    """Test the Settings singleton class"""
    
    def test_settings_singleton(self):
        """Test that Settings implements singleton pattern."""
        # Ensure a fresh instance is created for this test
        Settings._instance = None
        Settings._config = None

        settings1 = Settings()
        settings2 = Settings()
        
        assert settings1 is settings2
        assert get_settings() is settings1
        
        # Clean up for subsequent tests (though autouse fixture should handle this)
        Settings._instance = None
        Settings._config = None
    
    def test_settings_properties(self):
        """Test settings property access"""
        settings = Settings()
        
        # Test environment properties
        assert isinstance(settings.environment, Environment)
        assert isinstance(settings.is_production, bool)
        assert isinstance(settings.is_development, bool)
        
        # Test API properties
        assert isinstance(settings.api_base_url, str)
        assert isinstance(settings.api_timeout, int)
        assert isinstance(settings.has_api_credentials, bool)
        
        # Test trading properties
        assert isinstance(settings.initial_balance, float)
        assert isinstance(settings.trade_amount, float)
        assert isinstance(settings.market_pair, str)
        
        # Test ML properties
        assert isinstance(settings.sequence_length, int)
        assert isinstance(settings.learning_rate, float)
        
        # Test operational properties
        assert isinstance(settings.data_collection_interval, int)
        assert isinstance(settings.log_level, str)
    
    @patch('autotrader.config.settings.logger')
    def test_settings_reload(self, mock_logger):
        """Test settings reload functionality"""
        settings = Settings()
        original_balance = settings.initial_balance
        
        # Create a temporary config with different values
        temp_config = Config()
        temp_config.trading.initial_balance = 20000.0
        
        settings.update_config(temp_config)
        
        assert settings.initial_balance == 20000.0
        mock_logger.info.assert_called_with("Configuration updated")


class TestConfigValidation:
    """Test configuration validation functions"""
    
    def test_load_config_with_missing_file(self):
        """Test loading config when file doesn't exist"""
        non_existent_path = "/tmp/non_existent_config.json"
        config = load_config(non_existent_path)
        
        # Should return default config
        assert isinstance(config, Config)
        assert config.environment == Environment.DEVELOPMENT
    
    def test_load_config_with_invalid_json(self):
        """Test loading config with invalid JSON"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            # Should return default config on error
            assert isinstance(config, Config)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    @patch.dict(os.environ, {'AUTOTRADER_ENV': 'production'})
    def test_load_config_environment_detection(self):
        """Test automatic environment detection"""
        config = load_config()
        assert config.environment == Environment.PRODUCTION


class TestDataClasses:
    """Test individual dataclass configurations"""
    
    def test_api_config(self):
        """Test APIConfig dataclass"""
        api_config = APIConfig()
        
        assert api_config.base_url == "https://api.btcmarkets.net/v3"
        assert api_config.timeout == 10
        assert api_config.max_retries == 3
        assert api_config.api_key is None
    
    def test_trading_config(self):
        """Test TradingConfig dataclass"""
        trading_config = TradingConfig()
        
        assert trading_config.initial_balance == 10000.0
        assert trading_config.trade_amount == 0.01
        assert trading_config.market_pair == "BTC-AUD"
        assert trading_config.fee_rate == 0.001
    
    def test_ml_config(self):
        """Test MLConfig dataclass"""
        ml_config = MLConfig()
        
        assert ml_config.sequence_length == 20
        assert ml_config.lstm_units == 50
        assert ml_config.learning_rate == 0.001
        assert ml_config.feature_count == 12
    
    def test_operational_config(self):
        """Test OperationalConfig dataclass"""
        ops_config = OperationalConfig()
        
        assert ops_config.data_collection_interval == 60
        assert ops_config.save_interval == 1800
        assert ops_config.log_level == "INFO"
        assert ops_config.enable_detailed_logging is True


class TestEnvironmentSpecificConfigs:
    """Test environment-specific configuration differences"""
    
    def test_development_environment(self):
        """Test development environment configuration"""
        config = Config()
        config.environment = Environment.DEVELOPMENT
        
        assert config.is_development()
        assert not config.is_production()
    
    def test_production_environment(self):
        """Test production environment configuration"""
        config = Config()
        config.environment = Environment.PRODUCTION
        
        assert config.is_production()
        assert not config.is_development()
    
    def test_staging_environment(self):
        """Test staging environment configuration"""
        config = Config()
        config.environment = Environment.STAGING
        
        assert not config.is_production()
        assert not config.is_development()


class TestConfigIntegration:
    """Integration tests for the configuration system"""
    
    def test_full_configuration_flow(self):
        """Test complete configuration loading and usage flow"""
        # Set up environment
        test_env = {
            'AUTOTRADER_ENV': 'development',
            'BTCMARKETS_API_KEY': 'test-integration-key',
            'TRADING_INITIAL_BALANCE': '7500.0'
        }
        
        with patch.dict(os.environ, test_env):
            # Load settings
            settings = Settings()
            settings.reload_config() # Explicitly reload config to pick up env vars
    
            # Verify environment loading
            assert settings.environment == Environment.DEVELOPMENT
            assert settings.initial_balance == 7500.0
            
            # Verify API credentials
            api_key, api_secret = settings.get_api_credentials()
            assert api_key == 'test-integration-key'
            
            # Test configuration validation
            config = settings.config
            config._validate_config()  # Should not raise
    
    def test_error_handling(self):
        """Test error handling in configuration system"""
        # Test with invalid environment variable
        with patch.dict(os.environ, {'TRADING_INITIAL_BALANCE': 'invalid_number'}):
            # Should handle gracefully and use default
            config = Config()
            assert config.trading.initial_balance == 10000.0  # default value


if __name__ == '__main__':
    pytest.main([__file__])
