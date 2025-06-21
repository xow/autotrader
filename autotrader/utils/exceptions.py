"""
Custom exceptions for the Autotrader Bot

Defines specialized exception classes for different types of errors
that can occur during autonomous operation.
"""

import time
from typing import Optional, Dict, Any, List


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
        self.timestamp = time.time()
    
    def __str__(self) -> str:
        base_msg = self.message
        if self.error_code:
            base_msg = f"[{self.error_code}] {base_msg}"
        if self.context:
            base_msg += f" Context: {self.context}"
        return base_msg
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the exception to a dictionary representation.
        
        Returns:
            Dict containing the exception details including message, error code, and context.
        """
        return {
            "message": self.message,
            "error_code": self.error_code,
            "error_type": self.__class__.__name__,
            "context": self.context,
            "timestamp": self.timestamp
        }


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


class APIConnectionError(APIError):
    """
    Exception raised when there are issues connecting to an API.
    """
    
    def __init__(
        self,
        message: str,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        response_text: Optional[str] = None,
        **kwargs
    ):
        context = {
            "endpoint": endpoint,
            "status_code": status_code,
            "response_text": response_text,
            **kwargs
        }
        super().__init__(
            message=message,
            status_code=status_code,
            response_data={"text": response_text} if response_text else None,
            endpoint=endpoint
        )
        # Ensure these are set directly on the instance for backward compatibility
        self.endpoint = endpoint
        self.status_code = status_code
        self.response_text = response_text
        # Also store them in context for consistency
        self.context.update({
            "endpoint": endpoint,
            "status_code": status_code,
            "response_text": response_text,
            **kwargs
        })


class APIRateLimitError(APIError):
    """
    Exception raised when API rate limits are exceeded.
    """
    
    def __init__(
        self,
        message: str,
        rate_limit: Optional[int] = None,
        reset_time: Optional[float] = None,
        endpoint: Optional[str] = None
    ):
        context = {
            "rate_limit": rate_limit,
            "reset_time": reset_time,
            "endpoint": endpoint
        }
        super().__init__(message, 429, None, endpoint)
        self.rate_limit = rate_limit
        self.reset_time = reset_time
        self.retry_after = reset_time - time.time() if reset_time else None
        
    def get_retry_after(self) -> Optional[float]:
        """
        Get the number of seconds to wait before retrying the request.
        
        Returns:
            Optional[float]: Number of seconds to wait, or None if reset_time is not set
        """
        if self.reset_time:
            return max(0, self.reset_time - time.time())
        return None

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


class InvalidPriceDataError(DataError):
    """
    Exception raised when price data is invalid or malformed.
    """
    
    def __init__(
        self,
        message: str,
        data: Optional[Any] = None,
        missing_fields: Optional[List[str]] = None,
        validation_errors: Optional[Dict[str, str]] = None
    ):
        context = {
            "data_type": type(data).__name__ if data is not None else None,
            "missing_fields": missing_fields or [],
            "validation_errors": validation_errors or {}
        }
        super().__init__(
            message=message,
            data_type="price_data",
            validation_errors=context["validation_errors"]
        )
        self.data = data
        self.missing_fields = context["missing_fields"]


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


class ModelTrainingError(MLError):
    """
    Exception raised for model training related errors.
    """
    
    def __init__(
        self,
        message: str,
        training_samples: Optional[int] = None,
        epoch: Optional[int] = None,
        loss: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        metrics = metrics or {}
        if loss is not None:
            metrics["loss"] = loss
            
        context = {
            "training_samples": training_samples,
            "epoch": epoch,
            "loss": loss,
            "metrics": metrics,
            **{k: v for k, v in kwargs.items() if k != 'model_name'}
        }
        
        super().__init__(
            message=message,
            model_name=kwargs.get("model_name"),
            training_step=epoch,
            metrics=metrics
        )
        # Ensure these are set directly on the instance for backward compatibility
        self.training_samples = training_samples
        self.epoch = epoch
        self.loss = loss
        self.metrics = metrics
        # Also store them in context for consistency
        self.context.update(context)


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


class NetworkTimeoutError(AutotraderError):
    """
    Exception raised for network timeout errors.
    
    This exception is raised when a network operation exceeds the specified timeout duration.
    """
    
    def __init__(
        self,
        message: str,
        timeout_duration: Optional[float] = None,
        operation: Optional[str] = None
    ):
        """
        Initialize the NetworkTimeoutError.
        
        Args:
            message: Human-readable error message
            timeout_duration: Duration in seconds that was exceeded
            operation: Description of the operation that timed out
        """
        context = {
            "timeout_duration": timeout_duration,
            "operation": operation,
            "error_type": "network_timeout"
        }
        super().__init__(message, "NETWORK_TIMEOUT_ERROR", context)
        self.timeout_duration = timeout_duration
        self.operation = operation
        
    def __str__(self) -> str:
        """Return a string representation of the error."""
        if self.operation and self.timeout_duration:
            return f"Network operation '{self.operation}' timed out after {self.timeout_duration} seconds"
        elif self.operation:
            return f"Network operation '{self.operation}' timed out"
        elif self.timeout_duration:
            return f"Network operation timed out after {self.timeout_duration} seconds"
        return super().__str__()


# Alias for backward compatibility
TimeoutError = NetworkTimeoutError


# Alias for backward compatibility
TradingException = TradingError


class CriticalSystemError(AutotraderError):
    """
    Exception raised for critical system-level errors that may require
    immediate attention or system restart.
    """
    
    def __init__(
        self,
        message: str,
        component: Optional[str] = None,
        system_state: Optional[Dict[str, Any]] = None,
        requires_restart: bool = False
    ):
        """
        Initialize the CriticalSystemError.
        
        Args:
            message: Human-readable error message
            component: The system component where the error occurred
            system_state: Dictionary containing relevant system state information
            requires_restart: Whether the system needs to be restarted
        """
        context = {
            "component": component,
            "system_state": system_state or {},
            "requires_restart": requires_restart,
            "timestamp": time.time()
        }
        super().__init__(
            message=message,
            error_code="CRITICAL_SYSTEM_ERROR",
            context=context
        )
        self.component = component
        self.system_state = context["system_state"]
        self.requires_restart = requires_restart
        self.timestamp = context["timestamp"]
    
    def __str__(self) -> str:
        msg = f"Critical system error in {self.component}: {self.message}" if self.component else f"Critical system error: {self.message}"
        if self.requires_restart:
            msg += " (System restart required)"
        return msg
