"""
Utility functions and helpers for the Autotrader Bot

Contains logging, error handling, configuration utilities,
and common helper functions.
"""

# Import all required components
from .logging_config import setup_logging, get_logger
from .exceptions import (
    AutotraderError,
    APIError,
    DataError,
    APIConnectionError,
    APIRateLimitError,
    ModelTrainingError,
    InvalidPriceDataError,
    TradingError,
    TradingException,
    CriticalSystemError,
    NetworkError,
    NetworkTimeoutError,
    RecoveryError,
    ValidationError,
    ConfigurationError,
    StateError
)

from .error_handler import (
    ErrorHandler,
    ErrorRecoveryManager,
    retry_with_backoff,
    with_retry,
    error_context,
    validate_data,
    RetryConfig,
    CircuitBreaker,
    safe_execute,
    async_retry_with_backoff
)

from .alerting import (
    AlertingSystem,
    ErrorAlert,
    alerting_system
)

# For backward compatibility with existing tests
AutoTraderBaseException = AutotraderError

# Export all required components
__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    
    # Error Handling
    "ErrorHandler",
    "ErrorRecoveryManager",
    "RetryConfig",
    "CircuitBreaker",
    "retry_with_backoff",
    "with_retry",
    "error_context",
    "validate_data",
    "safe_execute",
    "async_retry_with_backoff",
    
    # Alerting
    "AlertingSystem",
    "ErrorAlert",
    "alerting_system",
    
    # Exceptions
    "AutotraderError",
    "AutoTraderBaseException",  # For backward compatibility
    "APIError",
    "APIConnectionError",
    "APIRateLimitError",
    "DataError",
    "InvalidPriceDataError",
    "ModelTrainingError",
    "TradingError",
    "TradingException",
    "CriticalSystemError",
    "NetworkError",
    "NetworkTimeoutError",
    "RecoveryError",
    "ValidationError",
    "ConfigurationError",
    "StateError"
]
