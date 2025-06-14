"""
Logging Settings and Configuration for AutoTrader Bot

This module provides default settings and configuration options for the logging system.
Settings can be overridden via environment variables or configuration files.
"""

import os
from typing import Dict, Any
from pathlib import Path


class LoggingSettings:
    """Centralized logging settings with environment variable support"""
    
    # Default settings
    DEFAULT_LOG_DIR = "logs"
    DEFAULT_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    DEFAULT_BACKUP_COUNT = 10
    DEFAULT_CONSOLE_LEVEL = "INFO"
    DEFAULT_FILE_LEVEL = "DEBUG"
    DEFAULT_STRUCTURED_LOGGING = True
    DEFAULT_LOG_RETENTION_DAYS = 30
    
    # Log file names
    LOG_FILES = {
        'main': 'autotrader.log',
        'trading': 'trading_decisions.log',
        'training': 'training_progress.log',
        'model': 'model_performance.log',
        'system': 'system_events.log',
        'api': 'api_calls.log',
        'performance': 'performance_metrics.log',
        'error': 'errors.log'
    }
    
    # Log levels for different environments
    ENVIRONMENT_CONFIGS = {
        'development': {
            'console_level': 'DEBUG',
            'file_level': 'DEBUG',
            'structured_logging': False,
            'max_file_size': 10 * 1024 * 1024,  # 10MB for dev
        },
        'production': {
            'console_level': 'INFO',
            'file_level': 'DEBUG',
            'structured_logging': True,
            'max_file_size': 100 * 1024 * 1024,  # 100MB for prod
        },
        'testing': {
            'console_level': 'WARNING',
            'file_level': 'INFO',
            'structured_logging': True,
            'max_file_size': 5 * 1024 * 1024,  # 5MB for tests
        }
    }
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get logging configuration based on environment"""
        
        # Detect environment
        env = os.getenv('AUTOTRADER_ENV', 'development').lower()
        
        # Start with default config
        config = {
            'log_dir': cls._get_env_value('LOG_DIR', cls.DEFAULT_LOG_DIR),
            'max_file_size': cls._get_env_int('MAX_FILE_SIZE', cls.DEFAULT_MAX_FILE_SIZE),
            'backup_count': cls._get_env_int('BACKUP_COUNT', cls.DEFAULT_BACKUP_COUNT),
            'console_level': cls._get_env_value('CONSOLE_LEVEL', cls.DEFAULT_CONSOLE_LEVEL),
            'file_level': cls._get_env_value('FILE_LEVEL', cls.DEFAULT_FILE_LEVEL),
            'structured_logging': cls._get_env_bool('STRUCTURED_LOGGING', cls.DEFAULT_STRUCTURED_LOGGING),
            'log_retention_days': cls._get_env_int('LOG_RETENTION_DAYS', cls.DEFAULT_LOG_RETENTION_DAYS)
        }
        
        # Apply environment-specific overrides
        if env in cls.ENVIRONMENT_CONFIGS:
            config.update(cls.ENVIRONMENT_CONFIGS[env])
        
        # Apply any environment variable overrides
        config.update(cls._get_env_overrides())
        
        return config
    
    @classmethod
    def _get_env_value(cls, key: str, default: str) -> str:
        """Get environment variable with AUTOTRADER_LOG_ prefix"""
        return os.getenv(f'AUTOTRADER_LOG_{key}', default)
    
    @classmethod
    def _get_env_int(cls, key: str, default: int) -> int:
        """Get integer environment variable with AUTOTRADER_LOG_ prefix"""
        try:
            return int(os.getenv(f'AUTOTRADER_LOG_{key}', str(default)))
        except ValueError:
            return default
    
    @classmethod
    def _get_env_bool(cls, key: str, default: bool) -> bool:
        """Get boolean environment variable with AUTOTRADER_LOG_ prefix"""
        value = os.getenv(f'AUTOTRADER_LOG_{key}', str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    @classmethod
    def _get_env_overrides(cls) -> Dict[str, Any]:
        """Get any environment variable overrides"""
        overrides = {}
        
        # Check for specific overrides
        env_mappings = {
            'AUTOTRADER_LOG_CONSOLE_LEVEL': 'console_level',
            'AUTOTRADER_LOG_FILE_LEVEL': 'file_level',
            'AUTOTRADER_LOG_MAX_FILE_SIZE': 'max_file_size',
            'AUTOTRADER_LOG_BACKUP_COUNT': 'backup_count',
            'AUTOTRADER_LOG_STRUCTURED': 'structured_logging',
            'AUTOTRADER_LOG_RETENTION_DAYS': 'log_retention_days'
        }
        
        for env_var, config_key in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Convert to appropriate type
                if config_key in ['max_file_size', 'backup_count', 'log_retention_days']:
                    try:
                        overrides[config_key] = int(value)
                    except ValueError:
                        pass
                elif config_key == 'structured_logging':
                    overrides[config_key] = value.lower() in ('true', '1', 'yes', 'on')
                else:
                    overrides[config_key] = value
        
        return overrides
    
    @classmethod
    def get_log_file_path(cls, log_type: str, log_dir: str = None) -> Path:
        """Get the full path for a specific log file type"""
        if log_dir is None:
            log_dir = cls._get_env_value('LOG_DIR', cls.DEFAULT_LOG_DIR)
        
        filename = cls.LOG_FILES.get(log_type, f'{log_type}.log')
        return Path(log_dir) / filename
    
    @classmethod
    def create_log_directories(cls, log_dir: str = None) -> None:
        """Create log directories if they don't exist"""
        if log_dir is None:
            log_dir = cls._get_env_value('LOG_DIR', cls.DEFAULT_LOG_DIR)
        
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organized logging
        subdirs = ['trading', 'training', 'system', 'api', 'archive']
        for subdir in subdirs:
            (log_path / subdir).mkdir(exist_ok=True)


# Environment configuration presets
DEVELOPMENT_CONFIG = {
    'log_dir': 'logs',
    'console_level': 'DEBUG',
    'file_level': 'DEBUG',
    'structured_logging': False,
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
}

PRODUCTION_CONFIG = {
    'log_dir': '/var/log/autotrader',
    'console_level': 'INFO',
    'file_level': 'DEBUG',
    'structured_logging': True,
    'max_file_size': 100 * 1024 * 1024,  # 100MB
    'backup_count': 20,
    'log_retention_days': 90
}

TESTING_CONFIG = {
    'log_dir': 'test_logs',
    'console_level': 'WARNING',
    'file_level': 'INFO',
    'structured_logging': True,
    'max_file_size': 5 * 1024 * 1024,  # 5MB
    'backup_count': 3,
    'log_retention_days': 7
}


def get_logging_config_for_environment(env: str = None) -> Dict[str, Any]:
    """Get logging configuration for a specific environment"""
    if env is None:
        env = os.getenv('AUTOTRADER_ENV', 'development').lower()
    
    if env == 'production':
        return PRODUCTION_CONFIG.copy()
    elif env == 'testing':
        return TESTING_CONFIG.copy()
    else:
        return DEVELOPMENT_CONFIG.copy()


def setup_environment_logging():
    """Setup logging based on current environment"""
    from .logging_config import setup_logging
    
    config = LoggingSettings.get_config()
    
    # Create log directories
    LoggingSettings.create_log_directories(config['log_dir'])
    
    # Setup logging with environment-specific config
    return setup_logging(**config)
