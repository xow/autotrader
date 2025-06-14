"""
Test suite for the AutoTrader error handling framework.

This module tests all aspects of the error handling system including
exceptions, retry mechanisms, recovery strategies, and alerting.
"""

import pytest
import time
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autotrader.utils import (
    # Exceptions
    AutoTraderBaseException, APIConnectionError, APIRateLimitError,
    ModelTrainingError, InvalidPriceDataError, TradingException,
    CriticalSystemError, NetworkTimeoutError,
    
    # Error handling
    ErrorHandler, RetryConfig, with_retry, error_context,
    safe_execute, validate_data, CircuitBreaker, AlertingSystem,
    ErrorRecoveryManager, error_handler
)


class TestExceptions:
    """Test custom exception classes."""
    
    def test_base_exception_initialization(self):
        """Test AutoTraderBaseException initialization."""
        context = {"test": "data"}
        error = AutoTraderBaseException("Test message", "TEST_CODE", context)
        
        assert error.message == "Test message"
        assert error.error_code == "TEST_CODE"
        assert error.context == context
        assert error.timestamp is not None
        assert "Test message" in str(error)
    
    def test_exception_to_dict(self):
        """Test exception serialization to dictionary."""
        error = APIConnectionError("Connection failed", "http://api.test")
        error_dict = error.to_dict()
        
        assert error_dict["error_type"] == "APIConnectionError"
        assert error_dict["message"] == "Connection failed"
        assert error_dict["context"]["endpoint"] == "http://api.test"
        assert "timestamp" in error_dict
    
    def test_api_connection_error(self):
        """Test APIConnectionError with context."""
        error = APIConnectionError(
            "Failed to connect", 
            endpoint="https://api.test.com",
            status_code=500,
            response_text="Internal Server Error"
        )
        
        assert error.context["endpoint"] == "https://api.test.com"
        assert error.context["status_code"] == 500
        assert error.context["response_text"] == "Internal Server Error"
    
    def test_model_training_error(self):
        """Test ModelTrainingError with training context."""
        error = ModelTrainingError(
            "Training failed",
            training_samples=1000,
            epoch=5,
            loss=0.85
        )
        
        assert error.context["training_samples"] == 1000
        assert error.context["epoch"] == 5
        assert error.context["loss"] == 0.85
    
    def test_trading_exception_inheritance(self):
        """Test exception inheritance hierarchy."""
        error = TradingException("Trading error")
        assert isinstance(error, AutoTraderBaseException)
        assert isinstance(error, TradingException)


class TestRetryMechanism:
    """Test retry mechanisms and exponential backoff."""
    
    def test_retry_config(self):
        """Test RetryConfig initialization."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=2.0,
            exponential_base=3.0
        )
        
        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.exponential_base == 3.0
    
    def test_successful_retry_decorator(self):
        """Test retry decorator with successful operation."""
        call_count = 0
        
        @with_retry(RetryConfig(max_attempts=3))
        def test_function():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = test_function()
        assert result == "success"
        assert call_count == 1
    
    def test_retry_with_eventual_success(self):
        """Test retry decorator with eventual success."""
        call_count = 0
        
        @with_retry(RetryConfig(max_attempts=3, base_delay=0.1))
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise APIConnectionError("Connection failed")
            return "success"
        
        result = test_function()
        assert result == "success"
        assert call_count == 3
    
    def test_retry_exhaustion(self):
        """Test retry decorator when all attempts fail."""
        call_count = 0
        
        @with_retry(RetryConfig(max_attempts=2, base_delay=0.1))
        def test_function():
            nonlocal call_count
            call_count += 1
            raise APIConnectionError("Persistent failure")
        
        with pytest.raises(APIConnectionError):
            test_function()
        
        assert call_count == 2
    
    def test_retry_with_critical_error(self):
        """Test that critical errors don't get retried."""
        call_count = 0
        
        @with_retry(RetryConfig(max_attempts=3, base_delay=0.1))
        def test_function():
            nonlocal call_count
            call_count += 1
            raise CriticalSystemError("Critical failure")
        
        with pytest.raises(CriticalSystemError):
            test_function()
        
        assert call_count == 1  # Should not retry critical errors


class TestCircuitBreaker:
    """Test circuit breaker pattern implementation."""
    
    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state."""
        cb = CircuitBreaker(failure_threshold=3, timeout_seconds=1)
        
        def success_function():
            return "success"
        
        result = cb.call(success_function)
        assert result == "success"
        assert cb.state == "CLOSED"
    
    def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after threshold failures."""
        cb = CircuitBreaker(failure_threshold=2, timeout_seconds=1)
        
        def failing_function():
            raise Exception("Test failure")
        
        # First failure
        with pytest.raises(Exception):
            cb.call(failing_function)
        assert cb.state == "CLOSED"
        
        # Second failure - should open circuit
        with pytest.raises(Exception):
            cb.call(failing_function)
        assert cb.state == "OPEN"
        
        # Third call should fail immediately
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            cb.call(failing_function)
    
    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker recovery through half-open state."""
        cb = CircuitBreaker(failure_threshold=1, timeout_seconds=0.1)
        
        def failing_function():
            raise Exception("Test failure")
        
        def success_function():
            return "success"
        
        # Open the circuit
        with pytest.raises(Exception):
            cb.call(failing_function)
        assert cb.state == "OPEN"
        
        # Wait for timeout
        time.sleep(0.2)
        
        # Should allow one attempt (half-open)
        result = cb.call(success_function)
        assert result == "success"
        assert cb.state == "CLOSED"


class TestErrorRecoveryManager:
    """Test error recovery strategies."""
    
    def test_recovery_strategy_registration(self):
        """Test registering and using recovery strategies."""
        manager = ErrorRecoveryManager()
        
        def test_recovery(error):
            return True
        
        manager.register_recovery_strategy(APIConnectionError, test_recovery)
        
        error = APIConnectionError("Test error")
        result = manager.attempt_recovery(error)
        assert result is True
    
    def test_error_count_tracking(self):
        """Test error count tracking."""
        manager = ErrorRecoveryManager()
        
        count1 = manager.increment_error_count("TEST_ERROR")
        count2 = manager.increment_error_count("TEST_ERROR")
        
        assert count1 == 1
        assert count2 == 2
    
    def test_error_rate_calculation(self):
        """Test error rate calculation."""
        manager = ErrorRecoveryManager()
        
        # Simulate errors
        manager.increment_error_count("TEST_ERROR")
        manager.increment_error_count("TEST_ERROR")
        
        rate = manager.get_error_rate("TEST_ERROR", window_minutes=60)
        assert rate == 2.0 / 60  # 2 errors in 60 minutes


class TestAlertingSystem:
    """Test alerting and notification system."""
    
    def test_alert_creation(self):
        """Test alert creation and serialization."""
        from autotrader.utils.error_handler import ErrorAlert
        
        alert = ErrorAlert(
            error_type="TEST_ERROR",
            timestamp=datetime.now().isoformat(),
            message="Test alert message",
            severity="HIGH",
            context={"test": "data"}
        )
        
        assert alert.error_type == "TEST_ERROR"
        assert alert.severity == "HIGH"
        assert alert.context["test"] == "data"
    
    def test_alert_threshold_logic(self):
        """Test alert threshold logic."""
        alerting = AlertingSystem()
        
        # First few errors shouldn't trigger alert for medium severity
        assert not alerting.should_alert("TEST_ERROR", "MEDIUM")
        
        # Critical errors should always alert
        assert alerting.should_alert("CRITICAL_ERROR", "CRITICAL")
    
    @patch('smtplib.SMTP')
    def test_email_alert_sending(self, mock_smtp):
        """Test email alert sending functionality."""
        alerting = AlertingSystem()
        alerting.email_config = {
            'smtp_server': 'smtp.test.com',
            'smtp_port': 587,
            'username': 'test@test.com',
            'password': 'password',
            'recipient': 'admin@test.com'
        }
        
        from autotrader.utils.error_handler import ErrorAlert
        alert = ErrorAlert(
            error_type="TEST_ERROR",
            timestamp=datetime.now().isoformat(),
            message="Test alert",
            severity="CRITICAL",
            context={}
        )
        
        # Mock SMTP server
        mock_server = Mock()
        mock_smtp.return_value = mock_server
        
        alerting._send_email_alert(alert)
        
        # Verify SMTP methods were called
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once()
        mock_server.send_message.assert_called_once()
        mock_server.quit.assert_called_once()


class TestErrorContext:
    """Test error context management."""
    
    def test_error_context_success(self):
        """Test error context with successful operation."""
        with error_context("test_operation", {"param": "value"}):
            result = "success"
        
        assert result == "success"
    
    def test_error_context_with_exception(self):
        """Test error context with exception handling."""
        with pytest.raises(ValueError):
            with error_context("test_operation"):
                raise ValueError("Test error")


class TestDataValidation:
    """Test data validation utilities."""
    
    def test_successful_validation(self):
        """Test successful data validation."""
        data = {"price": 50000, "volume": 1.5}
        
        rules = {
            "price": lambda d: d.get("price", 0) > 0,
            "volume": lambda d: d.get("volume", 0) > 0
        }
        
        result = validate_data(data, rules)
        assert result is True
    
    def test_failed_validation(self):
        """Test failed data validation."""
        data = {"price": -100, "volume": 1.5}
        
        rules = {
            "price": lambda d: d.get("price", 0) > 0,
            "volume": lambda d: d.get("volume", 0) > 0
        }
        
        with pytest.raises(Exception):  # Should raise DataValidationException
            validate_data(data, rules)


class TestSafeExecution:
    """Test safe execution utilities."""
    
    def test_safe_execute_success(self):
        """Test safe execution with successful function."""
        def test_function():
            return "success"
        
        result = safe_execute(test_function)
        assert result == "success"
    
    def test_safe_execute_with_exception(self):
        """Test safe execution with exception."""
        def test_function():
            raise ValueError("Test error")
        
        result = safe_execute(test_function, default_return="default")
        assert result == "default"
    
    def test_safe_execute_no_default(self):
        """Test safe execution without default return."""
        def test_function():
            raise ValueError("Test error")
        
        result = safe_execute(test_function)
        assert result is None


class TestSystemHealthCheck:
    """Test system health monitoring."""
    
    @patch('autotrader.utils.error_handler.psutil')
    def test_health_check_with_psutil(self, mock_psutil):
        """Test health check when psutil is available."""
        # Mock memory and disk usage
        mock_memory = Mock()
        mock_memory.percent = 75
        mock_memory.available = 4 * 1024**3  # 4GB
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_disk = Mock()
        mock_disk.percent = 50
        mock_disk.free = 100 * 1024**3  # 100GB
        mock_psutil.disk_usage.return_value = mock_disk
        
        from autotrader.utils.error_handler import check_system_health
        health = check_system_health()
        
        assert health['overall_status'] in ['HEALTHY', 'DEGRADED']
        assert 'memory' in health['checks']
        assert 'disk' in health['checks']
        assert health['checks']['memory']['status'] == 'OK'
        assert health['checks']['disk']['status'] == 'OK'
    
    @patch('socket.create_connection')
    def test_network_health_check(self, mock_connection):
        """Test network connectivity health check."""
        from autotrader.utils.error_handler import check_system_health
        
        # Test successful connection
        mock_connection.return_value = None
        health = check_system_health()
        assert health['checks']['network']['status'] == 'OK'
        
        # Test failed connection
        mock_connection.side_effect = OSError("Connection failed")
        health = check_system_health()
        assert health['checks']['network']['status'] == 'FAILED'
        assert health['overall_status'] == 'DEGRADED'


class TestIntegration:
    """Integration tests for the complete error handling system."""
    
    def test_complete_error_handling_flow(self):
        """Test complete error handling flow from exception to recovery."""
        # Create a function that fails then succeeds
        call_count = 0
        
        @with_retry(RetryConfig(max_attempts=3, base_delay=0.1))
        def test_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise APIConnectionError("Temporary failure")
            return "success"
        
        # Should succeed after retry
        result = test_operation()
        assert result == "success"
        assert call_count == 2
    
    def test_error_handler_with_multiple_error_types(self):
        """Test error handler with different error types."""
        handler = ErrorHandler()
        
        # Test API error
        api_error = APIConnectionError("API down")
        recovery1 = handler.handle_error(api_error)
        assert recovery1 is True  # Should attempt recovery
        
        # Test critical error
        critical_error = CriticalSystemError("System compromised")
        recovery2 = handler.handle_error(critical_error)
        assert recovery2 is False  # Should not attempt recovery
    
    def test_circuit_breaker_integration(self):
        """Test circuit breaker integration with error handler."""
        handler = ErrorHandler()
        cb = handler.get_circuit_breaker("test_operation")
        
        def failing_operation():
            raise APIConnectionError("Failure")
        
        # Should fail but not open circuit immediately
        with pytest.raises(APIConnectionError):
            cb.call(failing_operation)
        
        assert cb.state == "CLOSED"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
