"""
Logging Helpers and Utilities for AutoTrader Bot

This module provides helper functions, decorators, and utilities to make
logging easier and more consistent throughout the application.
"""

import functools
import time
import psutil
import threading
from typing import Callable, Any, Dict, Optional
from datetime import datetime, timedelta
import logging

from .logging_config import get_logging_config, log_error, log_system_event


class PerformanceMonitor:
    """Monitor and log system performance metrics"""
    
    def __init__(self, interval_seconds: int = 60):
        self.interval_seconds = interval_seconds
        self.running = False
        self.thread = None
        
    def start(self):
        """Start performance monitoring in background thread"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        
        log_system_event("performance_monitor_started", 
                        "Performance monitoring started",
                        {"interval_seconds": self.interval_seconds})
    
    def stop(self):
        """Stop performance monitoring"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
            
        log_system_event("performance_monitor_stopped", 
                        "Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        config = get_logging_config()
        
        while self.running:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Log performance metrics
                config.log_performance_metrics(
                    cpu_usage=cpu_percent,
                    memory_usage=memory.used / (1024 * 1024),  # MB
                    disk_usage=disk.used / (1024 * 1024 * 1024)  # GB
                )
                
                # Sleep for remaining interval
                time.sleep(max(0, self.interval_seconds - 1))
                
            except Exception as e:
                log_error(e, "performance_monitoring")
                time.sleep(self.interval_seconds)


def log_execution_time(logger_category: str = 'performance'):
    """Decorator to log function execution time"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            function_name = f"{func.__module__}.{func.__name__}"
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                logger = get_logging_config().get_logger(logger_category)
                logger.debug(
                    f"Function {function_name} executed in {execution_time:.3f}s",
                    extra={
                        'performance_data': {
                            'function': function_name,
                            'execution_time': execution_time,
                            'success': True
                        }
                    }
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                logger = get_logging_config().get_logger(logger_category)
                logger.error(
                    f"Function {function_name} failed after {execution_time:.3f}s: {e}",
                    extra={
                        'performance_data': {
                            'function': function_name,
                            'execution_time': execution_time,
                            'success': False,
                            'error': str(e)
                        }
                    }
                )
                raise
                
        return wrapper
    return decorator


def log_api_calls(endpoint_name: str = None):
    """Decorator to log API calls with timing and error handling"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            endpoint = endpoint_name or func.__name__
            
            try:
                result = func(*args, **kwargs)
                response_time = time.time() - start_time
                
                config = get_logging_config()
                config.log_api_call(
                    endpoint=endpoint,
                    method="GET",  # Could be enhanced to detect method
                    response_time=response_time,
                    status_code=200  # Could be enhanced to get actual status
                )
                
                return result
                
            except Exception as e:
                response_time = time.time() - start_time
                
                config = get_logging_config()
                config.log_api_call(
                    endpoint=endpoint,
                    method="GET",
                    response_time=response_time,
                    error=str(e)
                )
                raise
                
        return wrapper
    return decorator


class TradingSessionLogger:
    """Context manager for logging trading sessions"""
    
    def __init__(self, session_id: str = None):
        self.session_id = session_id or f"session_{int(time.time())}"
        self.session_logger = None
        self.start_time = None
        self.trades_executed = 0
        self.total_profit_loss = 0.0
        
    def __enter__(self):
        self.start_time = datetime.now()
        
        config = get_logging_config()
        self.session_logger = config.create_session_logger(self.session_id)
        
        self.session_logger.info(f"Trading session {self.session_id} started")
        log_system_event("trading_session_started", 
                        f"Trading session started: {self.session_id}",
                        {"session_id": self.session_id})
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time
        
        session_summary = {
            'session_id': self.session_id,
            'duration_seconds': duration.total_seconds(),
            'trades_executed': self.trades_executed,
            'total_profit_loss': self.total_profit_loss,
            'success': exc_type is None
        }
        
        self.session_logger.info(
            f"Trading session {self.session_id} completed. "
            f"Duration: {duration}, Trades: {self.trades_executed}",
            extra={'context': session_summary}
        )
        
        log_system_event("trading_session_completed",
                        f"Trading session completed: {self.session_id}",
                        session_summary)
        
        if exc_type:
            log_error(exc_val, f"trading_session_{self.session_id}", session_summary)
    
    def log_trade(self, decision: str, price: float, confidence: float, 
                  indicators: Dict[str, float], balance: float, reasoning: str = ""):
        """Log a trade within this session"""
        self.trades_executed += 1
        
        trade_data = {
            'session_id': self.session_id,
            'trade_number': self.trades_executed,
            'decision': decision,
            'price': price,
            'confidence': confidence,
            'indicators': indicators,
            'balance': balance,
            'reasoning': reasoning
        }
        
        self.session_logger.info(
            f"Trade #{self.trades_executed}: {decision} at {price:.2f} AUD",
            extra={'trade_data': trade_data}
        )
        
        # Also log to main trading logger
        config = get_logging_config()
        config.log_trade_decision(decision, price, confidence, indicators, balance, reasoning)


class ModelTrainingLogger:
    """Context manager for logging model training sessions"""
    
    def __init__(self, model_name: str = "autotrader_model"):
        self.model_name = model_name
        self.training_start = None
        self.epoch_count = 0
        self.best_accuracy = 0.0
        
    def __enter__(self):
        self.training_start = datetime.now()
        
        logger = get_logging_config().get_logger('training')
        logger.info(f"Model training started: {self.model_name}")
        
        log_system_event("model_training_started",
                        f"Started training model: {self.model_name}",
                        {"model_name": self.model_name})
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.training_start
        
        training_summary = {
            'model_name': self.model_name,
            'total_epochs': self.epoch_count,
            'duration_seconds': duration.total_seconds(),
            'best_accuracy': self.best_accuracy,
            'success': exc_type is None
        }
        
        logger = get_logging_config().get_logger('training')
        logger.info(
            f"Model training completed: {self.model_name}. "
            f"Epochs: {self.epoch_count}, Best accuracy: {self.best_accuracy:.4f}",
            extra={'model_metrics': training_summary}
        )
        
        log_system_event("model_training_completed",
                        f"Completed training model: {self.model_name}",
                        training_summary)
        
        if exc_type:
            log_error(exc_val, f"model_training_{self.model_name}", training_summary)
    
    def log_epoch(self, epoch: int, loss: float, accuracy: float,
                  val_loss: float = None, val_accuracy: float = None,
                  data_samples: int = 0):
        """Log training epoch results"""
        self.epoch_count = max(self.epoch_count, epoch)
        self.best_accuracy = max(self.best_accuracy, accuracy)
        
        config = get_logging_config()
        config.log_training_progress(
            epoch=epoch,
            loss=loss,
            accuracy=accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            data_samples=data_samples,
            training_time=(datetime.now() - self.training_start).total_seconds()
        )


def setup_crash_handler():
    """Setup crash handler to log uncaught exceptions"""
    import sys
    
    def crash_handler(exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions"""
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't log keyboard interrupts as crashes
            log_system_event("application_shutdown",
                           "Application shutdown requested by user")
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        # Log the crash
        logger = get_logging_config().get_logger('error')
        logger.critical(
            f"Uncaught exception: {exc_type.__name__}: {exc_value}",
            exc_info=(exc_type, exc_value, exc_traceback),
            extra={
                'context': {
                    'crash': True,
                    'exception_type': exc_type.__name__,
                    'exception_message': str(exc_value)
                }
            }
        )
        
        log_system_event("application_crash",
                        f"Application crashed: {exc_type.__name__}: {exc_value}",
                        {"exception_type": exc_type.__name__})
        
        # Call the default handler
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    
    sys.excepthook = crash_handler


class LogRotationManager:
    """Manage log rotation and cleanup"""
    
    def __init__(self):
        self.config = get_logging_config()
    
    def rotate_logs(self):
        """Manually trigger log rotation"""
        log_system_event("log_rotation_started", "Manual log rotation triggered")
        
        try:
            # This would trigger rotation on all file handlers
            # Implementation depends on the specific logging setup
            for logger_name, logger in self.config.loggers.items():
                for handler in logger.handlers:
                    if hasattr(handler, 'doRollover'):
                        handler.doRollover()
            
            log_system_event("log_rotation_completed", "Log rotation completed successfully")
            
        except Exception as e:
            log_error(e, "log_rotation")
    
    def cleanup_old_logs(self, days: int = 30):
        """Clean up logs older than specified days"""
        try:
            self.config.cleanup_old_logs(days)
            log_system_event("log_cleanup_completed",
                           f"Log cleanup completed for files older than {days} days",
                           {"retention_days": days})
        except Exception as e:
            log_error(e, "log_cleanup")


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def start_performance_monitoring(interval_seconds: int = 60):
    """Start global performance monitoring"""
    global _performance_monitor
    
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor(interval_seconds)
    
    _performance_monitor.start()


def stop_performance_monitoring():
    """Stop global performance monitoring"""
    global _performance_monitor
    
    if _performance_monitor:
        _performance_monitor.stop()


def get_performance_monitor() -> Optional[PerformanceMonitor]:
    """Get the global performance monitor instance"""
    return _performance_monitor
