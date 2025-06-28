"""
Core configuration management for AutoTrader Bot.

This module provides secure configuration management with support for:
- Environment-based configurations (dev, staging, prod)
- Environment variable overrides
- Configuration validation
- Secure handling of API keys and sensitive data
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Supported environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class APIConfig:
    """BTCMarkets API configuration"""
    base_url: str = "https://api.btcmarkets.net/v3"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    timeout: int = 10
    max_retries: int = 3
    retry_delay: float = 2.0


@dataclass
class TradingConfig:
    """Trading parameters configuration"""
    initial_balance: float = 10000.0
    trade_amount: float = 0.01  # BTC
    fee_rate: float = 0.001  # 0.1%
    market_pair: str = "BTC-AUD"
    
    # Risk management
    max_position_size: float = 0.1  # Max 10% of balance per trade
    risk_per_trade: float = 0.02 # Risk percentage per trade
    stop_loss_percent: float = 0.02  # 2% stop loss
    take_profit_percent: float = 0.04  # 4% take profit
    
    # Signal thresholds
    buy_confidence_threshold: float = 0.65
    sell_confidence_threshold: float = 0.35
    min_confidence_diff: float = 0.15
    
    # RSI filters
    rsi_overbought: float = 80.0
    rsi_oversold: float = 20.0


@dataclass
class MLConfig:
    """Machine Learning model configuration"""
    model_filename: str = "autotrader_model.keras"
    scalers_filename: str = "scalers.pkl"
    sequence_length: int = 20
    max_training_samples: int = 2000
    
    # Model architecture
    lstm_units: int = 50
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    dense_units: int = 25
    
    # Training parameters
    epochs: int = 10
    batch_size: int = 16
    validation_split: float = 0.2
    shuffle: bool = False  # Don't shuffle time series
    
    # Feature configuration
    feature_count: int = 0 # Will be dynamically calculated
    enable_technical_indicators: bool = True
    volume_sma_period: int = 10 # Default period for volume SMA
    
    # Feature Engineering Configuration (for FeatureEngineer)
    scaling_method: str = "standard"  # standard, minmax, robust, quantile
    sma_periods: Optional[List[int]] = field(default_factory=lambda: [5, 10, 20, 50])
    ema_periods: Optional[List[int]] = field(default_factory=lambda: [12, 26, 50])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: int = 2
    use_sma: bool = True
    use_ema: bool = True
    use_rsi: bool = True
    use_macd: bool = True
    use_bollinger: bool = True
    use_volume_indicators: bool = True
    use_price_ratios: bool = True
    use_price_differences: bool = True
    use_log_returns: bool = True
    use_volatility: bool = True
    volatility_window: int = 10
    use_time_features: bool = True
    use_cyclical_encoding: bool = True
    use_lag_features: bool = True
    lag_periods: Optional[List[int]] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    use_rolling_stats: bool = True
    rolling_windows: Optional[List[int]] = field(default_factory=lambda: [5, 10, 20])

    def __post_init__(self):
        """Post-initialization validation and setup for MLConfig."""
        self.feature_count = self._calculate_dynamic_feature_count()

    def _calculate_dynamic_feature_count(self) -> int:
        """
        Dynamically calculates the number of features based on the MLConfig's
        feature engineering settings.
        """
        # Import FeatureEngineer and FeatureConfig locally to avoid circular imports
        from autotrader.ml.feature_engineer import FeatureEngineer, FeatureConfig
        import pandas as pd
        import numpy as np
        
        # Create a FeatureConfig instance from MLConfig's relevant fields
        fe_config = FeatureConfig(
            use_sma=self.use_sma,
            sma_periods=self.sma_periods,
            use_ema=self.use_ema,
            ema_periods=self.ema_periods,
            use_rsi=self.use_rsi,
            rsi_period=self.rsi_period,
            use_macd=self.use_macd,
            macd_fast=self.macd_fast,
            macd_slow=self.macd_slow,
            macd_signal=self.macd_signal,
            use_bollinger=self.use_bollinger,
            bb_period=self.bb_period,
            bb_std=self.bb_std,
            use_volume_indicators=self.use_volume_indicators,
            use_price_ratios=self.use_price_ratios,
            use_price_differences=self.use_price_differences,
            use_log_returns=self.use_log_returns,
            use_volatility=self.use_volatility,
            volatility_window=self.volatility_window,
            use_time_features=self.use_time_features,
            use_cyclical_encoding=self.use_cyclical_encoding,
            use_lag_features=self.use_lag_features,
            lag_periods=self.lag_periods,
            use_rolling_stats=self.use_rolling_stats,
            rolling_windows=self.rolling_windows,
            scaling_method=self.scaling_method # Pass scaling method as well
        )
        
        # Instantiate a dummy FeatureEngineer
        dummy_fe = FeatureEngineer(config=fe_config)
        
        # Create a dummy DataFrame with enough data points for all indicators
        # The maximum period for any indicator or rolling stat is 50 (sma_periods, ema_periods, rolling_windows)
        # plus 26 for macd_slow, plus 14 for rsi/atr. Let's use 100 for safety.
        dummy_data_points = 100
        dummy_data = {
            'price': np.random.rand(dummy_data_points) * 10000,
            'volume': np.random.rand(dummy_data_points) * 1000,
            'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=dummy_data_points, freq='H')),
            'high': np.random.rand(dummy_data_points) * 10000 + 100, # Add high/low for hl_volatility
            'low': np.random.rand(dummy_data_points) * 10000 - 100,
            'spread': np.random.rand(dummy_data_points) * 10 # Add spread to dummy data
        }
        dummy_df = pd.DataFrame(dummy_data)
        
        # Generate features using the dummy FeatureEngineer
        # We use _generate_all_features because it doesn't require fitting the scaler
        # and gives us the raw feature columns.
        features_df = dummy_fe._generate_all_features(dummy_df)
        
        return len(features_df.columns)


@dataclass
class OperationalConfig:
    """Operational settings configuration"""
    data_collection_interval: int = 60  # seconds
    save_interval: int = 1800  # 30 minutes
    training_interval: int = 600  # 10 minutes
    log_level: str = "INFO"
    log_file: str = "autotrader.log"
    
    # Data persistence
    training_data_filename: str = "training_data.json"
    state_filename: str = "trader_state.pkl"
    
    # Monitoring
    status_report_interval: int = 10  # iterations
    enable_detailed_logging: bool = True


@dataclass
class Config:
    """Main configuration class"""
    environment: Environment = Environment.DEVELOPMENT
    api: APIConfig = field(default_factory=APIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    operations: OperationalConfig = field(default_factory=OperationalConfig)
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        self._validate_config()
        self._load_environment_overrides()
    
    def _validate_config(self):
        """Validate configuration values"""
        errors = []
        
        # Validate trading config
        if self.trading.initial_balance <= 0:
            errors.append("Trading initial_balance must be positive")
        
        if self.trading.trade_amount <= 0:
            errors.append("Trading trade_amount must be positive")
        
        if not (0 < self.trading.fee_rate < 1):
            errors.append("Trading fee_rate must be between 0 and 1")
        
        if not (0 < self.trading.max_position_size <= 1):
            errors.append("Trading max_position_size must be between 0 and 1")
        
        # Validate thresholds
        if not (0 < self.trading.buy_confidence_threshold < 1):
            errors.append("Buy confidence threshold must be between 0 and 1")
        
        if not (0 < self.trading.sell_confidence_threshold < 1):
            errors.append("Sell confidence threshold must be between 0 and 1")
        
        # Validate ML config
        if self.ml.sequence_length <= 0:
            errors.append("ML sequence_length must be positive")
        
        if self.ml.max_training_samples <= self.ml.sequence_length:
            errors.append("ML max_training_samples must be greater than sequence_length")
        
        # Validate operational config
        if self.operations.data_collection_interval <= 0:
            errors.append("Data collection interval must be positive")
        
        if self.operations.save_interval <= 0:
            errors.append("Save interval must be positive")
        
        if self.operations.training_interval <= 0:
            errors.append("Training interval must be positive")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def _load_environment_overrides(self):
        """Load configuration overrides from environment variables"""
        # API configuration
        if api_key := os.getenv("BTCMARKETS_API_KEY"):
            self.api.api_key = api_key
        
        if api_secret := os.getenv("BTCMARKETS_API_SECRET"):
            self.api.api_secret = api_secret
        
        if base_url := os.getenv("BTCMARKETS_BASE_URL"):
            self.api.base_url = base_url
        
        # Trading configuration
        if initial_balance := os.getenv("TRADING_INITIAL_BALANCE"):
            try:
                self.trading.initial_balance = float(initial_balance)
            except ValueError:
                logger.warning(f"Invalid TRADING_INITIAL_BALANCE: {initial_balance}")
        
        if trade_amount := os.getenv("TRADING_TRADE_AMOUNT"):
            try:
                self.trading.trade_amount = float(trade_amount)
            except ValueError:
                logger.warning(f"Invalid TRADING_TRADE_AMOUNT: {trade_amount}")
        
        # Environment setting
        if env := os.getenv("AUTOTRADER_ENV"):
            try:
                self.environment = Environment(env.lower())
            except ValueError:
                logger.warning(f"Invalid AUTOTRADER_ENV: {env}")
        
        # Operational configuration
        if log_level := os.getenv("LOG_LEVEL"):
            self.operations.log_level = log_level.upper()
        
        if data_interval := os.getenv("DATA_COLLECTION_INTERVAL"):
            try:
                self.operations.data_collection_interval = int(data_interval)
            except ValueError:
                logger.warning(f"Invalid DATA_COLLECTION_INTERVAL: {data_interval}")
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "Config":
        """Load configuration from JSON file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls()
        
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Create nested dataclass instances
            config = cls()
            
            if 'api' in config_data:
                config.api = APIConfig(**config_data['api'])
            
            if 'trading' in config_data:
                config.trading = TradingConfig(**config_data['trading'])
            
            if 'ml' in config_data:
                config.ml = MLConfig(**config_data['ml'])
            
            if 'operations' in config_data:
                config.operations = OperationalConfig(**config_data['operations'])
            
            if 'environment' in config_data:
                config.environment = Environment(config_data['environment'])
            
            logger.info(f"Configuration loaded from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            logger.info("Using default configuration")
            return cls()
    
    def to_file(self, config_path: Union[str, Path]):
        """Save configuration to JSON file"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            config_dict = {
                'environment': self.environment.value,
                'api': asdict(self.api),
                'trading': asdict(self.trading),
                'ml': asdict(self.ml),
                'operations': asdict(self.operations)
            }
            
            # Remove sensitive data from saved config
            config_dict['api']['api_key'] = None
            config_dict['api']['api_secret'] = None
            
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving config to {config_path}: {e}")
    
    def get_environment_config_path(self) -> Path:
        """Get the config file path for current environment"""
        config_dir = Path(__file__).parent
        return config_dir / f"config_{self.environment.value}.json"
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == Environment.DEVELOPMENT
    
    def get_api_credentials(self) -> tuple[Optional[str], Optional[str]]:
        """Get API credentials, checking environment variables first"""
        api_key = self.api.api_key or os.getenv("BTCMARKETS_API_KEY")
        api_secret = self.api.api_secret or os.getenv("BTCMARKETS_API_SECRET")
        return api_key, api_secret
    
    def has_valid_api_credentials(self) -> bool:
        """Check if valid API credentials are available"""
        api_key, api_secret = self.get_api_credentials()
        return bool(api_key and api_secret)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'environment': self.environment.value,
            'api': asdict(self.api),
            'trading': asdict(self.trading),
            'ml': asdict(self.ml),
            'operations': asdict(self.operations)
        }
