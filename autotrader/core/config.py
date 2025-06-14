"""
Configuration management for the Autotrader Bot

Handles environment-based configurations, API keys, trading parameters,
and ML model settings with validation and security.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field
from configparser import ConfigParser

from pydantic import BaseModel, validator, Field
from dotenv import load_dotenv

from ..utils.exceptions import ConfigurationError
from ..utils.logging_config import get_logger


@dataclass
class APIConfig:
    """BTCMarkets API configuration."""
    base_url: str = "https://api.btcmarkets.net"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    rate_limit_requests: int = 50
    rate_limit_window: int = 60
    timeout: float = 30.0
    retry_attempts: int = 3
    
    def __post_init__(self):
        if not self.api_key or not self.api_secret:
            # Try to load from environment
            self.api_key = self.api_key or os.getenv("BTCMARKETS_API_KEY")
            self.api_secret = self.api_secret or os.getenv("BTCMARKETS_API_SECRET")


@dataclass  
class MLConfig:
    """Machine learning configuration."""
    model_type: str = "lstm"
    sequence_length: int = 60
    features: list = field(default_factory=lambda: ["close", "volume", "rsi", "macd"])
    hidden_units: int = 128
    num_layers: int = 2
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    validation_split: float = 0.2
    
    # Continuous learning settings
    online_learning: bool = True
    adaptive_learning_rate: bool = True
    experience_replay_size: int = 10000
    update_frequency: int = 1  # Update model every N data points
    
    # Model persistence
    checkpoint_interval: int = 100  # Save every N training steps
    max_checkpoints: int = 10
    model_save_path: str = "models/"


@dataclass
class TradingConfig:
    """Trading simulation configuration."""
    initial_balance: float = 10000.0
    currency_pair: str = "BTC-AUD"
    min_trade_amount: float = 25.0  # Minimum trade in AUD
    max_position_size: float = 0.1  # Maximum 10% of portfolio per trade
    
    # Risk management
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    confidence_threshold: float = 0.6  # Minimum prediction confidence
    
    # Trading strategy
    strategy_type: str = "momentum"
    rebalance_frequency: int = 24  # Hours between rebalancing
    
    # Fees simulation
    maker_fee: float = 0.0085  # 0.85%
    taker_fee: float = 0.0085  # 0.85%


@dataclass
class DataConfig:
    """Data management configuration."""
    data_dir: str = "data/"
    max_data_points: int = 100000
    data_retention_days: int = 365
    compression_enabled: bool = True
    
    # Real-time data
    websocket_url: str = "wss://socket.btcmarkets.net/v2"
    data_frequency: str = "1m"  # 1 minute intervals
    buffer_size: int = 1000
    
    # Data validation
    outlier_detection: bool = True
    outlier_threshold: float = 3.0  # Standard deviations
    data_quality_checks: bool = True


@dataclass
class SystemConfig:
    """System operation configuration."""
    log_level: str = "INFO"
    log_dir: str = "logs/"
    
    # Autonomous operation
    max_runtime_hours: Optional[int] = None  # None = unlimited
    health_check_interval: int = 300  # Seconds
    auto_restart: bool = True
    
    # Resource management
    max_memory_mb: int = 2048
    cpu_limit_percent: float = 80.0
    
    # Monitoring
    metrics_enabled: bool = True
    alert_email: Optional[str] = None
    status_report_interval: int = 3600  # Seconds


class Config:
    """
    Main configuration class that manages all settings.
    """
    
    def __init__(self, config_file: Optional[str] = None, env_file: Optional[str] = None):
        self.logger = get_logger(__name__)
        
        # Load environment variables
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()  # Load from .env file if exists
        
        # Initialize configuration sections
        self.api = APIConfig()
        self.ml = MLConfig()
        self.trading = TradingConfig()
        self.data = DataConfig()
        self.system = SystemConfig()
        
        # Load configuration from file if provided
        if config_file:
            self.load_from_file(config_file)
        
        # Override with environment variables
        self._load_from_environment()
        
        # Validate configuration
        self.validate()
    
    def load_from_file(self, config_file: str):
        """
        Load configuration from YAML or INI file.
        
        Args:
            config_file: Path to configuration file
        """
        config_path = Path(config_file)
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_file}")
        
        try:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                self._load_yaml(config_path)
            elif config_path.suffix.lower() in ['.ini', '.cfg']:
                self._load_ini(config_path)
            else:
                raise ConfigurationError(f"Unsupported config file format: {config_path.suffix}")
            
            self.logger.info(f"Configuration loaded from {config_file}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def _load_yaml(self, config_path: Path):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Update configuration sections
        if 'api' in config_data:
            self._update_dataclass(self.api, config_data['api'])
        if 'ml' in config_data:
            self._update_dataclass(self.ml, config_data['ml'])
        if 'trading' in config_data:
            self._update_dataclass(self.trading, config_data['trading'])
        if 'data' in config_data:
            self._update_dataclass(self.data, config_data['data'])
        if 'system' in config_data:
            self._update_dataclass(self.system, config_data['system'])
    
    def _load_ini(self, config_path: Path):
        """Load configuration from INI file."""
        config = ConfigParser()
        config.read(config_path)
        
        # Map INI sections to configuration objects
        section_mapping = {
            'api': self.api,
            'ml': self.ml,
            'trading': self.trading,
            'data': self.data,
            'system': self.system
        }
        
        for section_name, config_obj in section_mapping.items():
            if section_name in config:
                section_data = dict(config[section_name])
                # Convert string values to appropriate types
                section_data = self._convert_types(section_data, config_obj)
                self._update_dataclass(config_obj, section_data)
    
    def _update_dataclass(self, obj: Any, data: Dict[str, Any]):
        """Update dataclass fields with new values."""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
    
    def _convert_types(self, data: Dict[str, str], reference_obj: Any) -> Dict[str, Any]:
        """Convert string values to appropriate types based on reference object."""
        converted = {}
        
        for key, value in data.items():
            if hasattr(reference_obj, key):
                reference_type = type(getattr(reference_obj, key))
                
                try:
                    if reference_type == bool:
                        converted[key] = value.lower() in ('true', '1', 'yes', 'on')
                    elif reference_type == int:
                        converted[key] = int(value)
                    elif reference_type == float:
                        converted[key] = float(value)
                    elif reference_type == list:
                        converted[key] = [item.strip() for item in value.split(',')]
                    else:
                        converted[key] = value
                except ValueError:
                    self.logger.warning(f"Failed to convert {key}={value} to {reference_type}")
                    converted[key] = value
            else:
                converted[key] = value
        
        return converted
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        # API configuration
        if os.getenv("BTCMARKETS_API_KEY"):
            self.api.api_key = os.getenv("BTCMARKETS_API_KEY")
        if os.getenv("BTCMARKETS_API_SECRET"):
            self.api.api_secret = os.getenv("BTCMARKETS_API_SECRET")
        
        # System configuration
        if os.getenv("LOG_LEVEL"):
            self.system.log_level = os.getenv("LOG_LEVEL")
        if os.getenv("LOG_DIR"):
            self.system.log_dir = os.getenv("LOG_DIR")
        
        # Trading configuration
        if os.getenv("INITIAL_BALANCE"):
            self.trading.initial_balance = float(os.getenv("INITIAL_BALANCE"))
        if os.getenv("CURRENCY_PAIR"):
            self.trading.currency_pair = os.getenv("CURRENCY_PAIR")
    
    def validate(self):
        """Validate configuration settings."""
        errors = []
        
        # Validate API configuration
        if not self.api.api_key:
            errors.append("API key is required (BTCMARKETS_API_KEY)")
        if not self.api.api_secret:
            errors.append("API secret is required (BTCMARKETS_API_SECRET)")
        
        # Validate ML configuration
        if self.ml.learning_rate <= 0 or self.ml.learning_rate >= 1:
            errors.append("Learning rate must be between 0 and 1")
        if self.ml.sequence_length <= 0:
            errors.append("Sequence length must be positive")
        
        # Validate trading configuration
        if self.trading.initial_balance <= 0:
            errors.append("Initial balance must be positive")
        if self.trading.confidence_threshold < 0 or self.trading.confidence_threshold > 1:
            errors.append("Confidence threshold must be between 0 and 1")
        
        # Validate system configuration
        if self.system.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            errors.append("Invalid log level")
        
        if errors:
            raise ConfigurationError(f"Configuration validation failed: {'; '.join(errors)}")
        
        self.logger.info("Configuration validation passed")
    
    def save_to_file(self, config_file: str):
        """
        Save current configuration to file.
        
        Args:
            config_file: Path to save configuration file
        """
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = {
            'api': self._dataclass_to_dict(self.api),
            'ml': self._dataclass_to_dict(self.ml),
            'trading': self._dataclass_to_dict(self.trading),
            'data': self._dataclass_to_dict(self.data),
            'system': self._dataclass_to_dict(self.system)
        }
        
        # Remove sensitive information before saving
        if 'api_key' in config_data['api']:
            config_data['api']['api_key'] = "[REDACTED]"
        if 'api_secret' in config_data['api']:
            config_data['api']['api_secret'] = "[REDACTED]"
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Configuration saved to {config_file}")
    
    def _dataclass_to_dict(self, obj: Any) -> Dict[str, Any]:
        """Convert dataclass to dictionary."""
        return {
            field.name: getattr(obj, field.name)
            for field in obj.__dataclass_fields__.values()
        }
    
    def get_setting(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a specific configuration setting.
        
        Args:
            section: Configuration section (api, ml, trading, data, system)
            key: Setting key
            default: Default value if setting not found
            
        Returns:
            Configuration value or default
        """
        section_obj = getattr(self, section, None)
        if section_obj is None:
            return default
        
        return getattr(section_obj, key, default)
    
    def set_setting(self, section: str, key: str, value: Any):
        """
        Set a specific configuration setting.
        
        Args:
            section: Configuration section
            key: Setting key
            value: New value
        """
        section_obj = getattr(self, section, None)
        if section_obj is None:
            raise ConfigurationError(f"Invalid configuration section: {section}")
        
        if not hasattr(section_obj, key):
            raise ConfigurationError(f"Invalid configuration key: {section}.{key}")
        
        setattr(section_obj, key, value)
        self.logger.info(f"Configuration updated: {section}.{key} = {value}")
    
    def __str__(self) -> str:
        """String representation of configuration (excluding sensitive data)."""
        sections = {
            'api': self._dataclass_to_dict(self.api),
            'ml': self._dataclass_to_dict(self.ml),
            'trading': self._dataclass_to_dict(self.trading),
            'data': self._dataclass_to_dict(self.data),
            'system': self._dataclass_to_dict(self.system)
        }
        
        # Mask sensitive information
        if 'api_key' in sections['api']:
            sections['api']['api_key'] = "[REDACTED]"
        if 'api_secret' in sections['api']:
            sections['api']['api_secret'] = "[REDACTED]"
        
        return yaml.dump(sections, default_flow_style=False, indent=2)


# Global configuration instance
_global_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def init_config(config_file: Optional[str] = None, env_file: Optional[str] = None) -> Config:
    """
    Initialize the global configuration.
    
    Args:
        config_file: Path to configuration file
        env_file: Path to environment file
        
    Returns:
        Initialized configuration instance
    """
    global _global_config
    _global_config = Config(config_file, env_file)
    return _global_config
