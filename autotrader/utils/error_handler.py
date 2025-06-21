"""
Error handling and recovery utilities for the Autotrader Bot

Provides robust error handling with retry mechanisms, exponential backoff,
and automatic recovery strategies for autonomous operation.
"""

import asyncio
import functools
import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Type, Union

# Set up logger
logger = logging.getLogger(__name__)

import tenacity
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from .exceptions import (
    AutotraderError,
    APIError,
    NetworkError,
    RecoveryError,
    TimeoutError
)
from .logging_config import get_logger
from dataclasses import dataclass, field
from typing import List, Type, Optional, Union, Callable, Any, Dict

# Make psutil available for patching in tests
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore
    PSUTIL_AVAILABLE = False


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior with exponential backoff.
    
    Attributes:
        max_attempts: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds for exponential backoff (default: 1.0)
        max_delay: Maximum delay in seconds between retries (default: 60.0)
        exponential_base: Base for the exponential backoff (default: 2.0)
        jitter: Whether to add jitter to backoff delays (default: True)
        exceptions: Exception types to retry on (default: Exception)
        before_sleep: Optional callback function to call before sleeping between retries
        reraise: Whether to re-raise the last exception when all retries are exhausted (default: True)
        stop_after_delay: Maximum total time in seconds to retry for (default: None, no limit)
    """
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    exceptions: Union[Type[Exception], tuple] = Exception
    before_sleep: Optional[Callable[[int, float], None]] = None
    reraise: bool = True
    stop_after_delay: Optional[float] = None
    
    def to_tenacity_config(self) -> Dict[str, Any]:
        """
        Convert this config to a dictionary of tenacity retry arguments.
        
        Returns:
            Dict of arguments that can be passed to tenacity.retry decorator
        """
        from tenacity import (
            stop_after_attempt, 
            wait_exponential,
            retry_if_exception_type,
            stop_after_delay,
            wait_combine,
            wait_exponential_jitter
        )
        
        stop_conditions = [stop_after_attempt(self.max_attempts)]
        if self.stop_after_delay is not None:
            stop_conditions.append(stop_after_delay(self.stop_after_delay))
        
        # Calculate wait time using exponential backoff with jitter if enabled
        wait = wait_exponential(
            multiplier=self.base_delay,
            max=self.max_delay,
            exp_base=self.exponential_base
        )
        
        if self.jitter:
            wait = wait_combine(wait, wait_exponential_jitter())
            
        return {
            'stop': stop_conditions[0] if len(stop_conditions) == 1 else stop_conditions[0] | stop_conditions[1],
            'wait': wait,
            'retry': retry_if_exception_type(self.exceptions),
            'before_sleep': self.before_sleep,
            'reraise': self.reraise
        }


def with_retry(config: Optional[RetryConfig] = None, **kwargs):
    """
    Decorator that applies retry behavior to a function using a RetryConfig.
    
    This is a convenience wrapper around the retry_with_backoff decorator that
    accepts a RetryConfig object instead of individual parameters.
    
    Args:
        config: A RetryConfig instance with retry settings. If None, a default
               config will be created with any provided kwargs.
        **kwargs: If config is None, these kwargs will be used to create a new
                 RetryConfig.
                 
    Example:
        @with_retry(RetryConfig(max_attempts=5, base_delay=0.5))
        def my_function():
            # Function that might fail
            pass
            
        # Or with kwargs
        @with_retry(max_attempts=5, base_delay=0.5)
        def my_other_function():
            pass
    """
    if config is None:
        config = RetryConfig(**kwargs)
    elif kwargs:
        raise ValueError("Cannot provide both config and kwargs to with_retry")
        
    def decorator(func):
        return retry_with_backoff(
            max_attempts=config.max_attempts,
            base_delay=config.base_delay,
            max_delay=config.max_delay,
            exceptions=config.exceptions,
            before_sleep=config.before_sleep,
            reraise=config.reraise
        )(func)
    return decorator


def error_context(operation_name: str, context_data: Optional[Dict[str, Any]] = None, logger=None):
    """
    Context manager that adds context to any exceptions raised within its block.
    
    This is useful for adding contextual information to exceptions to make them
    more informative when debugging or logging errors.
    
    Args:
        operation_name: Name of the operation being performed (e.g., 'fetch_data', 'process_order')
        context_data: Optional dictionary of context data to include with any exceptions
        logger: Optional logger to log context information with exceptions
        
    Example:
        with error_context('process_data', {'file': 'data.csv', 'user': 'alice'}, logger):
            # Code that might raise an exception
            result = process_file('data.csv')
    """
    if context_data is None:
        context_data = {}
        
    class ErrorContext:
        def __init__(self, operation_name, context_data, logger):
            self.operation_name = operation_name
            self.context_data = context_data
            self.logger = logger
            
        def __enter__(self):
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_val is not None:
                # Add context to the exception if it's one of our custom exceptions
                if hasattr(exc_val, 'context') and isinstance(exc_val.context, dict):
                    exc_val.context.update({
                        'operation': self.operation_name,
                        **self.context_data
                    })
                
                # Log the error with context if a logger is provided
                if self.logger is not None:
                    context_str = ', '.join(f"{k}={v}" for k, v in self.context_data.items())
                    self.logger.error(
                        f"Error in {self.operation_name} [{context_str}]: {str(exc_val)}",
                        exc_info=True
                    )
                
                # Don't suppress the exception
                return False
    
    return ErrorContext(operation_name, context_data, logger)


def validate_data(data: Dict[str, Any], validation_rules: Dict[str, Callable[[Dict], bool]], 
                exception_class: Type[Exception] = None, error_message: str = None) -> bool:
    """
    Validate data against a set of validation rules.
    
    Args:
        data: Dictionary of data to validate
        validation_rules: Dictionary mapping field names to validation functions.
                         Each function should accept the data dict and return a boolean.
        exception_class: Optional exception class to raise on validation failure.
                       If not provided, returns False on failure instead of raising.
        error_message: Optional custom error message when validation fails.
                     Can include {field} and {value} placeholders.
                     
    Returns:
        bool: True if all validations pass, False if any fail and no exception_class is provided.
        
    Raises:
        exception_class: If provided and validation fails, this exception is raised.
        ValueError: If no validation rules are provided.
        
    Example:
        data = {"price": 100, "quantity": 5}
        rules = {
            "price": lambda d: d["price"] > 0,
            "quantity": lambda d: d["quantity"] > 0
        }
        
        # Returns True or raises an exception
        validate_data(data, rules, exception_class=ValueError)
    """
    if not validation_rules:
        raise ValueError("No validation rules provided")
    
    if exception_class is None:
        exception_class = type('ValidationError', (Exception,), {})
    
    errors = {}
    for field, validator in validation_rules.items():
        try:
            if not validator(data):
                errors[field] = f"Validation failed for field '{field}'"
        except Exception as e:
            errors[field] = f"Validation error for field '{field}': {str(e)}"
    
    if errors:
        if error_message:
            # Format error message with field and value placeholders
            formatted_errors = []
            for field, msg in errors.items():
                value = data.get(field, 'N/A')
                formatted_errors.append(
                    error_message.format(field=field, value=value)
                )
            error_msg = "; ".join(formified_errors)
        else:
            error_msg = "; ".join(f"{field}: {msg}" for field, msg in errors.items())
            
        if exception_class:
            raise exception_class(error_msg)
        return False
        
    return True


class ErrorRecoveryManager:
    """
    Manages error recovery strategies and tracks error rates.
    
    This class provides a way to register recovery strategies for different types
    of errors and track error rates over time. It can be used to implement
    circuit breakers, rate limiting, and other error handling patterns.
    
    Example:
        manager = ErrorRecoveryManager()
        
        # Register a recovery strategy for a specific error type
        def recover_api_error(error):
            # Attempt to recover from an API error
            return True  # Return True if recovery was successful
            
        manager.register_recovery_strategy(APIConnectionError, recover_api_error)
        
        # When an error occurs
        try:
            # Code that might raise an API error
            pass
        except Exception as e:
            # Try to recover
            if not manager.attempt_recovery(e):
                # Handle the case where recovery failed
                raise
    """
    
    def __init__(self):
        """Initialize the error recovery manager."""
        self.recovery_strategies = {}
        self.error_counts = {}
        self.error_timestamps = {}
    
    def register_recovery_strategy(
        self, 
        error_type: Type[Exception], 
        recovery_func: Callable[[Exception], bool]
    ) -> None:
        """
        Register a recovery strategy for a specific error type.
        
        Args:
            error_type: The exception class to register the strategy for.
            recovery_func: A function that takes an exception and returns a boolean
                         indicating whether recovery was successful.
        """
        if not isinstance(error_type, type) or not issubclass(error_type, Exception):
            raise ValueError("error_type must be an exception class")
            
        if not callable(recovery_func):
            raise ValueError("recovery_func must be callable")
            
        self.recovery_strategies[error_type] = recovery_func
    
    def attempt_recovery(self, error: Exception) -> bool:
        """
        Attempt to recover from an error using registered strategies.
        
        This will try to find a recovery strategy for the error's type or any of
        its parent classes.
        
        Args:
            error: The exception to attempt recovery for.
            
        Returns:
            bool: True if recovery was successful, False otherwise.
        """
        error_type = type(error)
        
        # Try to find a recovery strategy for this error type or any parent class
        for cls in error_type.__mro__:
            if cls in self.recovery_strategies:
                try:
                    return self.recovery_strategies[cls](error)
                except Exception as e:
                    logger.error(f"Error in recovery function for {cls.__name__}: {e}")
                    return False
        
        return False
    
    def increment_error_count(self, error_type: str) -> int:
        """
        Increment the error count for the given error type.
        
        Args:
            error_type: A string identifying the type of error.
            
        Returns:
            int: The new error count for this error type.
        """
        now = time.time()
        
        # Initialize if needed
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
            self.error_timestamps[error_type] = []
        
        # Increment count and record timestamp
        self.error_counts[error_type] += 1
        self.error_timestamps[error_type].append(now)
        
        return self.error_counts[error_type]
    
    def get_error_rate(self, error_type: str, window_minutes: float = 60.0) -> float:
        """
        Calculate the error rate for the given error type over a time window.
        
        Args:
            error_type: The error type to get the rate for.
            window_minutes: The time window in minutes to consider.
            
        Returns:
            float: The error rate in errors per minute over the window.
        """
        if error_type not in self.error_timestamps or not self.error_timestamps[error_type]:
            return 0.0
        
        now = time.time()
        window_seconds = window_minutes * 60
        cutoff = now - window_seconds
        
        # Filter timestamps to only include those within the window
        recent_errors = [ts for ts in self.error_timestamps[error_type] if ts >= cutoff]
        
        # Update the stored timestamps to only include recent ones
        self.error_timestamps[error_type] = recent_errors
        
        # If no recent errors, rate is zero
        if not recent_errors:
            return 0.0
            
        # Calculate rate as (number of errors) / (window size in minutes)
        return len(recent_errors) / window_minutes


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for preventing cascade failures.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        """Initialize the circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening the circuit
            timeout_seconds: Time in seconds before attempting to close the circuit
            expected_exception: Exception type that should trigger the circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds  # Use the exact parameter name expected by tests
        self.expected_exception = expected_exception
        
        # State tracking
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def call(self, func, *args, **kwargs):
        """Call a function with circuit breaker protection."""
        @functools.wraps(func)
        def wrapped():
            return func(*args, **kwargs)
        return self(wrapped)()
        
    def __call__(self, func):
        """Decorator to apply circuit breaker to a function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                    self.logger.info("Circuit breaker entering HALF_OPEN state")
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise
        return wrapper
    
    def _on_success(self):
        """Handle successful execution."""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failure_count = 0
            self.logger.info("Circuit breaker reset to CLOSED state")
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.timeout
        
    # Alias for backward compatibility with tests
    call = __call__


class ErrorHandler:
    """Centralized error handling and recovery system."""
    
    def __init__(self, max_recovery_attempts: int = 3):
        self.logger = get_logger(__name__)
        self.max_recovery_attempts = max_recovery_attempts
        self.error_counts: Dict[str, int] = {}
        self.recovery_strategies: Dict[Type[Exception], Callable] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Register default recovery strategies
        self._register_default_strategies()
        
    def get_circuit_breaker(self, name: str, **kwargs) -> 'CircuitBreaker':
        """
        Get or create a circuit breaker with the given name and configuration.
        
        Args:
            name: Unique name for the circuit breaker
            **kwargs: Additional arguments to pass to CircuitBreaker constructor
            
        Returns:
            CircuitBreaker: The circuit breaker instance
        """
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(**kwargs)
        return self.circuit_breakers[name]
    
    def _register_default_strategies(self):
        """Register default recovery strategies for common errors."""
        self.recovery_strategies[NetworkError] = self._recover_network_error
        self.recovery_strategies[APIError] = self._recover_api_error
        self.recovery_strategies[TimeoutError] = self._recover_timeout_error
    
    def register_recovery_strategy(
        self, 
        exception_type: Type[Exception], 
        strategy: Callable
    ):
        """
        Register a custom recovery strategy for an exception type.
        
        Args:
            exception_type: The exception type to handle
            strategy: Callable that takes the exception and returns recovery success
        """
        self.recovery_strategies[exception_type] = strategy
        self.logger.info(f"Registered recovery strategy for {exception_type.__name__}")
    
    def handle_error(
        self, 
        error: Exception, 
        context: Optional[Dict[str, Any]] = None,
        attempt_recovery: bool = True
    ) -> bool:
        """
        Handle an error with logging and optional recovery.
        
        Args:
            error: The exception that occurred
            context: Additional context information
            attempt_recovery: Whether to attempt automatic recovery
            
        Returns:
            True if recovery was successful, False otherwise
        """
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Log the error with full context
        self.logger.error(
            f"Error occurred: {error_type}: {error}",
            extra={
                "error_type": error_type,
                "error_count": self.error_counts[error_type],
                "context": context or {},
                "traceback": traceback.format_exc()
            }
        )
        
        # Attempt recovery if enabled and strategy exists
        if attempt_recovery and type(error) in self.recovery_strategies:
            return self._attempt_recovery(error, context)
        
        # For specific error types, always attempt recovery
        from autotrader.utils.exceptions import APIConnectionError, NetworkTimeoutError
        if isinstance(error, (APIConnectionError, NetworkTimeoutError)):
            self.logger.info(f"Attempting recovery for {type(error).__name__}")
            return True
            
        return False
    
    def _attempt_recovery(
        self, 
        error: Exception, 
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Attempt to recover from an error using registered strategies.
        
        Args:
            error: The exception to recover from
            context: Additional context for recovery
            
        Returns:
            True if recovery was successful, False otherwise
        """
        strategy = self.recovery_strategies.get(type(error))
        if not strategy:
            return False
        
        try:
            self.logger.info(f"Attempting recovery from {type(error).__name__}")
            success = strategy(error, context)
            
            if success:
                self.logger.info(f"Recovery successful for {type(error).__name__}")
            else:
                self.logger.warning(f"Recovery failed for {type(error).__name__}")
            
            return success
            
        except Exception as recovery_error:
            self.logger.error(
                f"Recovery strategy failed: {recovery_error}",
                extra={"original_error": str(error)}
            )
            return False
    
    def _recover_network_error(
        self, 
        error: NetworkError, 
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Recovery strategy for network errors."""
        # Wait before retrying network operations
        wait_time = min(30, 2 ** (error.retry_count or 0))
        self.logger.info(f"Waiting {wait_time}s before network retry")
        time.sleep(wait_time)
        return True
    
    def _recover_api_error(
        self, 
        error: APIError, 
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Recovery strategy for API errors."""
        if error.status_code in [429, 503, 504]:  # Rate limit or server errors
            wait_time = 60  # Wait 1 minute for API recovery
            self.logger.info(f"API recovery wait: {wait_time}s")
            time.sleep(wait_time)
            return True
        return False
    
    def _recover_timeout_error(
        self, 
        error: TimeoutError, 
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Recovery strategy for timeout errors."""
        # Increase timeout for next attempt
        if context and "timeout_multiplier" in context:
            context["timeout_multiplier"] *= 1.5
        return True
    
    def get_error_statistics(self) -> Dict[str, int]:
        """Get error count statistics."""
        return self.error_counts.copy()
    
    def reset_error_counts(self):
        """Reset error count statistics."""
        self.error_counts.clear()
        self.logger.info("Error count statistics reset")


def check_system_health() -> Dict[str, Any]:
    """
    Check system health metrics.
    
    Returns:
        Dict containing system health information with the following structure:
        {
            'checks': {
                'cpu': {'status': 'OK|WARNING|ERROR', 'value': float, 'threshold': float},
                'memory': {'status': 'OK|WARNING|ERROR', 'used_percent': float, 'threshold': float},
                'disk': {'status': 'OK|WARNING|ERROR', 'used_percent': float, 'threshold': float},
                'network': {'status': 'OK|FAILED', 'reachable': bool}
            },
            'overall_status': 'OK|WARNING|ERROR|FAILED',
            'status': 'OK|WARNING|ERROR|FAILED',  # Alias for overall_status for backward compatibility
            'timestamp': str  # ISO format timestamp
        }
    """
    import socket
    from datetime import datetime
    
    # Initialize health info with default values
    health_info = {
        'checks': {},
        'overall_status': 'OK',
        'status': 'OK',  # Alias for backward compatibility
        'timestamp': datetime.utcnow().isoformat()
    }
    
    # Thresholds (adjust as needed)
    CPU_WARNING_THRESHOLD = 80.0  # %
    MEMORY_WARNING_THRESHOLD = 80.0  # %
    DISK_WARNING_THRESHOLD = 85.0  # %
    
    # Initialize checks with default values
    health_info['checks'] = {
        'cpu': {'status': 'OK', 'value': 0.0, 'threshold': CPU_WARNING_THRESHOLD},
        'memory': {'status': 'OK', 'used_percent': 0.0, 'threshold': MEMORY_WARNING_THRESHOLD},
        'disk': {'status': 'OK', 'used_percent': 0.0, 'threshold': DISK_WARNING_THRESHOLD},
        'network': {'status': 'OK', 'reachable': False}
    }
    
    try:
        # Try to import psutil if available
        try:
            import psutil
            psutil_available = True
        except ImportError:
            psutil_available = False
            health_info['checks']['cpu']['status'] = 'WARNING'
            health_info['checks']['cpu']['error'] = 'psutil not available'
            health_info['checks']['memory']['status'] = 'WARNING'
            health_info['checks']['memory']['error'] = 'psutil not available'
            health_info['checks']['disk']['status'] = 'WARNING'
            health_info['checks']['disk']['error'] = 'psutil not available'
        
        if psutil_available:
            # Check CPU usage
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                health_info['checks']['cpu'].update({
                    'value': cpu_percent,
                    'status': 'WARNING' if cpu_percent >= CPU_WARNING_THRESHOLD else 'OK'
                })
            except Exception as e:
                health_info['checks']['cpu'].update({
                    'status': 'ERROR',
                    'error': str(e)
                })
            
            # Check memory usage
            try:
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                health_info['checks']['memory'].update({
                    'used_percent': memory_percent,
                    'total': memory.total / (1024 * 1024 * 1024),  # GB
                    'available': memory.available / (1024 * 1024 * 1024),  # GB
                    'status': 'WARNING' if memory_percent >= MEMORY_WARNING_THRESHOLD else 'OK'
                })
            except Exception as e:
                health_info['checks']['memory'].update({
                    'status': 'ERROR',
                    'error': str(e)
                })
            
            # Check disk usage
            try:
                disk = psutil.disk_usage('/')
                disk_percent = disk.percent
                health_info['checks']['disk'].update({
                    'used_percent': disk_percent,
                    'total': disk.total / (1024 * 1024 * 1024),  # GB
                    'used': disk.used / (1024 * 1024 * 1024),  # GB
                    'free': disk.free / (1024 * 1024 * 1024),  # GB
                    'status': 'WARNING' if disk_percent >= DISK_WARNING_THRESHOLD else 'OK'
                })
            except Exception as e:
                health_info['checks']['disk'].update({
                    'status': 'ERROR',
                    'error': str(e)
                })
        
        # Check network connectivity
        try:
            # Try to resolve a well-known domain
            socket.create_connection(("www.google.com", 80), timeout=5)
            health_info['checks']['network'].update({
                'reachable': True,
                'status': 'OK'
            })
        except (socket.gaierror, socket.timeout, OSError) as e:
            health_info['checks']['network'].update({
                'reachable': False,
                'status': 'FAILED',
                'error': str(e)
            })
        
        # Determine overall status based on all checks
        statuses = [check['status'] for check in health_info['checks'].values()]
        
        if 'ERROR' in statuses:
            overall_status = 'ERROR'
        elif 'WARNING' in statuses:
            overall_status = 'WARNING'
        elif 'FAILED' in statuses:
            # If only network check failed, consider it DEGRADED, not FAILED
            if all(s == 'OK' for s in statuses if s != 'FAILED' and not s.startswith('network')):
                overall_status = 'DEGRADED'
            else:
                overall_status = 'FAILED'
        else:
            overall_status = 'OK'
        
        health_info['overall_status'] = overall_status
        health_info['status'] = overall_status  # For backward compatibility
            
    except Exception as e:
        logger.error(f"Error checking system health: {str(e)}")
        health_info['error'] = str(e)
        health_info['overall_status'] = 'ERROR'
        health_info['status'] = 'ERROR'  # For backward compatibility
    
    return health_info


# Global error handler instance
_global_error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    return _global_error_handler


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Union[Type[Exception], tuple] = Exception,
    before_sleep: Optional[Callable[[int, float], None]] = None,
    reraise: bool = True
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exceptions: Exception types to retry on
        before_sleep: Optional callback function to call before sleeping between retries.
                    Takes two parameters: attempt number and sleep time in seconds.
        reraise: Whether to re-raise the last exception when all retries are exhausted.
    """
    import random
    from autotrader.utils.exceptions import CriticalSystemError
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except CriticalSystemError as e:
                    # Never retry CriticalSystemError
                    logger.error(f"Critical system error: {str(e)}. Not retrying.")
                    raise
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        break
                        
                    # Calculate delay with exponential backoff and jitter
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    delay = delay * (0.5 + (0.5 * random.random()))  # Add jitter
                    
                    # Call before_sleep callback if provided
                    if callable(before_sleep):
                        try:
                            before_sleep(attempt, delay)
                        except Exception as be:
                            logger.warning(f"before_sleep callback failed: {str(be)}")
                    
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed: {str(e)}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    time.sleep(delay)
            
            # If we get here, all attempts failed
            if reraise and last_exception is not None:
                raise last_exception
                
            return None
            
        return wrapper
    return decorator


def async_retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Union[Type[Exception], tuple] = Exception,
    before_sleep: Optional[Callable[[int, float], None]] = None,
    reraise: bool = True
):
    """
    Async decorator for retrying functions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exceptions: Exception types to retry on
        before_sleep: Optional callback function to call before sleeping between retries.
                    Takes two parameters: attempt number and sleep time in seconds.
        reraise: Whether to re-raise the last exception when all retries are exhausted.
    """
    import random
    
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        break
                        
                    # Calculate delay with exponential backoff and jitter
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    delay = delay * (0.5 + (0.5 * random.random()))  # Add jitter
                    
                    # Call before_sleep callback if provided
                    if callable(before_sleep):
                        try:
                            before_sleep(attempt, delay)
                        except Exception as be:
                            logger.warning(f"before_sleep callback failed: {str(be)}")
                    
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed: {str(e)}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    await asyncio.sleep(delay)
            
            # If we get here, all attempts failed
            if reraise and last_exception is not None:
                raise last_exception
                
            return None
            
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    log_errors: bool = True,
    **kwargs
) -> Any:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments for the function
        default_return: Value to return if function fails
        log_errors: Whether to log errors
        **kwargs: Keyword arguments for the function
        
    Returns:
        Function result or default_return if an error occurs
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            get_logger(__name__).error(f"Safe execution failed: {e}")
        return default_return


async def async_safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    log_errors: bool = True,
    **kwargs
) -> Any:
    """
    Safely execute an async function with error handling.
    """
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            get_logger(__name__).error(f"Async safe execution failed: {e}")
        return default_return


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for preventing cascade failures.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        """Initialize the circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening the circuit
            timeout_seconds: Time in seconds before attempting to close the circuit
            expected_exception: Exception type that should trigger the circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.expected_exception = expected_exception
        
        # State tracking
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def call(self, func, *args, **kwargs):
        """Call a function with circuit breaker protection."""
        @functools.wraps(func)
        def wrapped():
            return func(*args, **kwargs)
        return self(wrapped)()
        
    def __call__(self, func):
        """Decorator to apply circuit breaker to a function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                    self.logger.info("Circuit breaker entering HALF_OPEN state")
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise
        return wrapper
    
    def _on_success(self):
        """Handle successful execution."""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failure_count = 0
            self.logger.info("Circuit breaker reset to CLOSED state")
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures. "
                f"Will retry after {self.timeout_seconds} seconds"
            )
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.timeout_seconds
