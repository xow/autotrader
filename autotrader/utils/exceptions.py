"""
Custom exceptions for the Autotrader Bot

Defines specialized exception classes for different types of errors
that can occur during autonomous operation.
"""

from typing import Optional, Dict, Any


class AutotraderError(Exception):
    """
    Base exception class for all autotrader-related errors.
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
    
    def __str__(self) -> str:
        base_msg = self.message
        if self.error_code:
            base_msg = f"[{self.error_code}] {base_msg}"
        if self.context:
            base_msg += f" Context: {self.context}"
        return base_msg


class APIError(AutotraderError):
    """
    Exception raised for API-related errors.
    """
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
        endpoint: Optional[str] = None
    ):
        context = {
            "status_code": status_code,
            "response_data": response_data,
            "endpoint": endpoint
        }
        super().__init__(message, "API_ERROR", context)
        self.status_code = status_code
        self.response_data = response_data
        self.endpoint = endpoint


class DataError(AutotraderError):
    """
    Exception raised for data-related errors.
    """
    
    def __init__(
        self,
        message: str,
        data_type: Optional[str] = None,
        validation_errors: Optional[list] = None
    ):
        context = {
            "data_type": data_type,
            "validation_errors": validation_errors
        }
        super().__init__(message, "DATA_ERROR", context)
        self.data_type = data_type
        self.validation_errors = validation_errors


class MLError(AutotraderError):
    """
    Exception raised for machine learning related errors.
    """
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        training_step: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None
    ):
        context = {
            "model_name": model_name,
            "training_step": training_step,
            "metrics": metrics
        }
        super().__init__(message, "ML_ERROR", context)
        self.model_name = model_name
        self.training_step = training_step
        self.metrics = metrics


class TradingError(AutotraderError):
    """
    Exception raised for trading-related errors.
    """
    
    def __init__(
        self,
        message: str,
        trade_action: Optional[str] = None,
        portfolio_state: Optional[Dict[str, Any]] = None
    ):
        context = {
            "trade_action": trade_action,
            "portfolio_state": portfolio_state
        }
        super().__init__(message, "TRADING_ERROR", context)
        self.trade_action = trade_action
        self.portfolio_state = portfolio_state


class ConfigurationError(AutotraderError):
    """
    Exception raised for configuration-related errors.
    """
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None
    ):
        context = {
            "config_key": config_key,
            "config_value": config_value
        }
        super().__init__(message, "CONFIG_ERROR", context)
        self.config_key = config_key
        self.config_value = config_value


class StateError(AutotraderError):
    """
    Exception raised for state management errors.
    """
    
    def __init__(
        self,
        message: str,
        state_type: Optional[str] = None,
        checkpoint_path: Optional[str] = None
    ):
        context = {
            "state_type": state_type,
            "checkpoint_path": checkpoint_path
        }
        super().__init__(message, "STATE_ERROR", context)
        self.state_type = state_type
        self.checkpoint_path = checkpoint_path


class NetworkError(AutotraderError):
    """
    Exception raised for network-related errors.
    """
    
    def __init__(
        self,
        message: str,
        connection_type: Optional[str] = None,
        retry_count: Optional[int] = None
    ):
        context = {
            "connection_type": connection_type,
            "retry_count": retry_count
        }
        super().__init__(message, "NETWORK_ERROR", context)
        self.connection_type = connection_type
        self.retry_count = retry_count


class ValidationError(AutotraderError):
    """
    Exception raised for validation errors.
    """
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        expected_type: Optional[str] = None,
        actual_value: Optional[Any] = None
    ):
        context = {
            "field_name": field_name,
            "expected_type": expected_type,
            "actual_value": actual_value
        }
        super().__init__(message, "VALIDATION_ERROR", context)
        self.field_name = field_name
        self.expected_type = expected_type
        self.actual_value = actual_value


class RecoveryError(AutotraderError):
    """
    Exception raised when recovery from an error fails.
    """
    
    def __init__(
        self,
        message: str,
        original_error: Optional[Exception] = None,
        recovery_attempts: Optional[int] = None
    ):
        context = {
            "original_error": str(original_error) if original_error else None,
            "recovery_attempts": recovery_attempts
        }
        super().__init__(message, "RECOVERY_ERROR", context)
        self.original_error = original_error
        self.recovery_attempts = recovery_attempts


class TimeoutError(AutotraderError):
    """
    Exception raised for timeout-related errors.
    """
    
    def __init__(
        self,
        message: str,
        timeout_duration: Optional[float] = None,
        operation: Optional[str] = None
    ):
        context = {
            "timeout_duration": timeout_duration,
            "operation": operation
        }
        super().__init__(message, "TIMEOUT_ERROR", context)
        self.timeout_duration = timeout_duration
        self.operation = operation
