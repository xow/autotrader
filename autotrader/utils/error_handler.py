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


class ErrorHandler:
    """
    Centralized error handling and recovery system.
    """
    
    def __init__(self, max_recovery_attempts: int = 3):
        self.logger = get_logger(__name__)
        self.max_recovery_attempts = max_recovery_attempts
        self.error_counts: Dict[str, int] = {}
        self.recovery_strategies: Dict[Type[Exception], Callable] = {}
        
        # Register default recovery strategies
        self._register_default_strategies()
    
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


# Global error handler instance
_global_error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    return _global_error_handler


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Union[Type[Exception], tuple] = Exception
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exceptions: Exception types to retry on
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=base_delay,
            max=max_delay
        ),
        retry=retry_if_exception_type(exceptions),
        before_sleep=before_sleep_log(get_logger(__name__), logging.WARNING)
    )


def async_retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Union[Type[Exception], tuple] = Exception
):
    """
    Async decorator for retrying functions with exponential backoff.
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise
                    
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    get_logger(__name__).warning(
                        f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
            
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
        timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        self.logger = get_logger(__name__)
    
    def __call__(self, func):
        """Decorator to apply circuit breaker to a function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                    self.logger.info("Circuit breaker entering HALF_OPEN state")
                else:
                    raise RecoveryError(
                        "Circuit breaker is OPEN",
                        recovery_attempts=self.failure_count
                    )
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.timeout
    
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
