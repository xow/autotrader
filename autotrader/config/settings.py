"""
Settings management for AutoTrader Bot.

This module provides global settings access and environment-specific configurations.
It acts as a singleton pattern to ensure consistent configuration across the application.
"""

import os
from pathlib import Path
from typing import Optional
import logging

from .config import Config, Environment, APIConfig, TradingConfig, MLConfig, OperationalConfig
# Removed explicit import of load_config as it will be removed from config.py

logger = logging.getLogger(__name__)


class Settings:
    """
    Global settings manager using singleton pattern.
    
    This class ensures that configuration is loaded once and shared across
    the entire application, while providing methods to reload or update
    configuration when needed.
    """
    
    _instance: Optional["Settings"] = None
    _config: Optional[Config] = None
    
    def __new__(cls) -> "Settings":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only load config if it hasn't been loaded yet for this instance
        if self._config is None:
            self._load_config()

    @classmethod
    def _reset_singleton(cls):
        """Resets the singleton instance for testing purposes."""
        cls._instance = None
        cls._config = None
    
    def _load_config(self):
        """Load configuration based on environment"""
        try:
            # Determine config path based on environment
            env_name = os.getenv("AUTOTRADER_ENV", "development").lower()
            try:
                env = Environment(env_name)
            except ValueError:
                env = Environment.DEVELOPMENT
            
            config_dir = Path(__file__).parent
            config_path = config_dir / f"config_{env.value}.json"
            
            self._config = Config.from_file(config_path)
            logger.info(f"Settings loaded for environment: {self._config.environment.value}")
        except Exception as e:
            logger.error("Failed to load configuration", exc_info=e)
            # Fall back to default configuration
            self._config = Config()
            logger.info("Using default configuration")
    
    @property
    def config(self) -> Config:
        """Get the current configuration"""
        if self._config is None:
            self._load_config()
        return self._config
    
    def reload_config(self, config_path: Optional[str] = None):
        """Reload configuration from file"""
        if config_path:
            self._config = Config.from_file(config_path)
        else:
            self._load_config()
        logger.info("Configuration reloaded")
    
    def update_config(self, config: Config):
        """Update the current configuration"""
        self._config = config
        logger.info("Configuration updated")
    
    # Convenience properties for commonly accessed settings
    
    @property
    def environment(self) -> Environment:
        """Current environment"""
        return self.config.environment
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.config.is_production()
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.config.is_development()
    
    # API Settings
    
    @property
    def api_base_url(self) -> str:
        """BTCMarkets API base URL"""
        return self.config.api.base_url
    
    @property
    def api_timeout(self) -> int:
        """API request timeout in seconds"""
        return self.config.api.timeout
    
    @property
    def api_max_retries(self) -> int:
        """Maximum API retry attempts"""
        return self.config.api.max_retries
    
    def get_api_credentials(self) -> tuple[Optional[str], Optional[str]]:
        """Get API credentials"""
        return self.config.get_api_credentials()
    
    @property
    def has_api_credentials(self) -> bool:
        """Check if API credentials are available"""
        return self.config.has_valid_api_credentials()
    
    # Trading Settings
    
    @property
    def initial_balance(self) -> float:
        """Initial trading balance"""
        return self.config.trading.initial_balance
    
    @property
    def trade_amount(self) -> float:
        """Amount per trade in BTC"""
        return self.config.trading.trade_amount
    
    @property
    def fee_rate(self) -> float:
        """Trading fee rate"""
        return self.config.trading.fee_rate
    
    @property
    def market_pair(self) -> str:
        """Trading market pair"""
        return self.config.trading.market_pair
    
    @property
    def buy_confidence_threshold(self) -> float:
        """Confidence threshold for buy signals"""
        return self.config.trading.buy_confidence_threshold
    
    @property
    def sell_confidence_threshold(self) -> float:
        """Confidence threshold for sell signals"""
        return self.config.trading.sell_confidence_threshold
    
    @property
    def rsi_overbought(self) -> float:
        """RSI overbought level"""
        return self.config.trading.rsi_overbought
    
    @property
    def rsi_oversold(self) -> float:
        """RSI oversold level"""
        return self.config.trading.rsi_oversold

    @property
    def max_position_size(self) -> float:
        """Maximum position size in BTC"""
        return self.config.trading.max_position_size

    @property
    def risk_per_trade(self) -> float:
        """Risk percentage per trade"""
        return self.config.trading.risk_per_trade
    
    # ML Settings
    @property
    def ml(self) -> MLConfig:
        """ML configuration"""
        return self.config.ml
    
    @property
    def model_filename(self) -> str:
        """ML model filename"""
        return self.config.ml.model_filename

    @property
    def scalers_filename(self) -> str:
        """ML scalers filename"""
        return self.config.ml.scalers_filename
    
    @property
    def sequence_length(self) -> int:
        """LSTM sequence length"""
        return self.config.ml.sequence_length
    
    @property
    def max_training_samples(self) -> int:
        """Maximum training samples to keep"""
        return self.config.ml.max_training_samples
    
    @property
    def lstm_units(self) -> int:
        """Number of LSTM units"""
        return self.config.ml.lstm_units

    @property
    def dropout_rate(self) -> float:
        """ML model dropout rate"""
        return self.config.ml.dropout_rate

    @property
    def learning_rate(self) -> float:
        """ML model learning rate"""
        return self.config.ml.learning_rate

    @property
    def dense_units(self) -> int:
        """Number of dense units in ML model"""
        return self.config.ml.dense_units

    @property
    def feature_count(self) -> int:
        """Number of features for ML model"""
        return self.config.ml.feature_count
    
    @property
    def training_epochs(self) -> int:
        """Number of training epochs"""
        return self.config.ml.epochs
    
    @property
    def batch_size(self) -> int:
        """Training batch size"""
        return self.config.ml.batch_size

    @property
    def volume_sma_period(self) -> int:
        """Volume SMA period for technical indicators"""
        return self.config.ml.volume_sma_period
    
    # Operational Settings
    @property
    def operations(self) -> OperationalConfig:
        """Operational configuration"""
        return self.config.operations
    
    @property
    def data_collection_interval(self) -> int:
        """Data collection interval in seconds"""
        return self.config.operations.data_collection_interval
    
    @property
    def save_interval(self) -> int:
        """Save interval in seconds"""
        return self.config.operations.save_interval
    
    @property
    def training_interval(self) -> int:
        """Model training interval in seconds"""
        return self.config.operations.training_interval
    
    @property
    def log_level(self) -> str:
        """Logging level"""
        return self.config.operations.log_level
    
    @property
    def log_file(self) -> str:
        """Log file path"""
        return self.config.operations.log_file
    
    @property
    def training_data_filename(self) -> str:
        """Training data filename"""
        return self.config.operations.training_data_filename
    
    @property
    def state_filename(self) -> str:
        """Trader state filename"""
        return self.config.operations.state_filename

    @property
    def enable_detailed_logging(self) -> bool:
        """Enable detailed logging"""
        return self.config.operations.enable_detailed_logging
    
    def save_config(self, config_path: Optional[str] = None):
        """Save current configuration to file"""
        if config_path is None:
            config_path = self.config.get_environment_config_path()
        self.config.to_file(config_path)
    
    def create_default_configs(self):
        """Create default configuration files for all environments"""
        config_dir = Path(__file__).parent
        
        for env in Environment:
            config = Config()
            config.environment = env
            
            # Environment-specific adjustments
            if env == Environment.PRODUCTION:
                config.operations.log_level = "WARNING"
                config.operations.enable_detailed_logging = False
                config.trading.buy_confidence_threshold = 0.7  # More conservative
                config.trading.sell_confidence_threshold = 0.3
            elif env == Environment.DEVELOPMENT:
                config.operations.log_level = "DEBUG"
                config.operations.enable_detailed_logging = True
                # Removed hardcoded initial_balance for development to allow env var override
            
            config_path = config_dir / f"config_{env.value}.json"
            config.to_file(config_path)
            logger.info(f"Created default config for {env.value}: {config_path}")


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance"""
    return settings


def setup_logging():
    """Setup logging based on current configuration"""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    log_file = settings.log_file
    
    # Clear existing handlers
    logging.getLogger().handlers.clear()
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Logging configured - Level: {settings.log_level}, File: {log_file}")


def validate_environment():
    """Validate the current environment configuration"""
    errors = []
    warnings = []
    
    # Check API credentials in production
    if settings.is_production and not settings.has_api_credentials:
        errors.append("Production environment requires valid API credentials")
    
    # Check file permissions
    try:
        test_file = Path(settings.log_file)
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.touch(exist_ok=True)
    except PermissionError:
        errors.append(f"Cannot write to log file: {settings.log_file}")
    
    # Check model file paths
    model_path = Path(settings.model_filename)
    if settings.is_production and not model_path.exists():
        warnings.append(f"Model file not found: {settings.model_filename}")
    
    # Validate trading parameters
    if settings.trade_amount * settings.api_timeout > settings.initial_balance * 0.5:
        warnings.append("Trade amount seems high relative to initial balance")
    
    if errors:
        raise ValueError(f"Environment validation failed: {'; '.join(errors)}")
    
    if warnings:
        for warning in warnings:
            logger.warning(warning)
    
    logger.info("Environment validation passed")


# Initialize settings on import
if __name__ != "__main__":
    setup_logging()
