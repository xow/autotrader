"""
Example Usage of the AutoTrader Logging System

This module demonstrates how to use the comprehensive logging system
in the AutoTrader bot application.
"""

import time
import random
from datetime import datetime
from typing import Dict, Any

# Import our logging components
from .logging_config import setup_logging, get_logger, log_trade_decision, log_training_progress
from .logging_settings import setup_environment_logging
from .logging_helpers import (
    TradingSessionLogger, 
    ModelTrainingLogger, 
    log_execution_time,
    log_api_calls,
    start_performance_monitoring,
    stop_performance_monitoring,
    setup_crash_handler
)


@log_execution_time('system')
def simulate_data_collection():
    """Simulate data collection with execution time logging"""
    time.sleep(random.uniform(0.1, 0.5))  # Simulate API call time
    return {
        'price': random.uniform(50000, 70000),
        'volume': random.uniform(1000, 5000),
        'timestamp': datetime.now().isoformat()
    }


@log_api_calls('btc_markets_api')
def simulate_api_call():
    """Simulate API call with logging"""
    time.sleep(random.uniform(0.05, 0.2))  # Simulate network latency
    
    # Simulate occasional API failures
    if random.random() < 0.05:  # 5% failure rate
        raise Exception("API rate limit exceeded")
    
    return {
        'status': 'success',
        'data': simulate_data_collection()
    }


def example_basic_logging():
    """Example of basic logging usage"""
    print("=== Basic Logging Example ===")
    
    # Setup logging (this would typically be done once at startup)
    setup_environment_logging()
    
    # Get different category loggers
    system_logger = get_logger('system')
    trading_logger = get_logger('trading')
    training_logger = get_logger('training')
    
    # Basic logging
    system_logger.info("AutoTrader bot starting up")
    system_logger.debug("Debug information - only in debug mode")
    
    # Simulate some trading activity
    trading_logger.info("Market analysis completed")
    training_logger.info("Model training initiated")
    
    print("Basic logging messages sent to files and console")


def example_structured_logging():
    """Example of structured logging with context"""
    print("\n=== Structured Logging Example ===")
    
    # Use convenience functions for structured logging
    log_trade_decision(
        decision="BUY",
        price=65432.10,
        confidence=0.78,
        indicators={
            'rsi': 35.2,
            'macd': 250.5,
            'sma_20': 64800.0
        },
        balance=10000.0,
        reasoning="RSI indicates oversold condition, MACD shows bullish divergence"
    )
    
    log_training_progress(
        epoch=15,
        loss=0.0234,
        accuracy=0.847,
        val_loss=0.0287,
        val_accuracy=0.823,
        data_samples=1500,
        training_time=45.6
    )
    
    print("Structured logging with trade and training data completed")


def example_session_logging():
    """Example of session-based logging"""
    print("\n=== Session Logging Example ===")
    
    # Use trading session context manager
    with TradingSessionLogger("demo_session_001") as session:
        
        # Simulate some trades
        for i in range(3):
            price = random.uniform(60000, 70000)
            confidence = random.uniform(0.6, 0.9)
            decision = "BUY" if confidence > 0.75 else "HOLD"
            
            session.log_trade(
                decision=decision,
                price=price,
                confidence=confidence,
                indicators={'rsi': random.uniform(20, 80)},
                balance=10000.0 + (i * 100),
                reasoning=f"Automated decision based on confidence {confidence:.3f}"
            )
            
            time.sleep(0.1)  # Simulate time between trades
    
    print("Trading session logging completed")


def example_training_logging():
    """Example of model training logging"""
    print("\n=== Training Logging Example ===")
    
    # Use model training context manager
    with ModelTrainingLogger("lstm_v2") as training:
        
        # Simulate training epochs
        for epoch in range(1, 6):
            loss = 0.1 / epoch  # Decreasing loss
            accuracy = 0.7 + (epoch * 0.05)  # Increasing accuracy
            val_loss = loss * 1.1
            val_accuracy = accuracy * 0.95
            
            training.log_epoch(
                epoch=epoch,
                loss=loss,
                accuracy=accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                data_samples=1000 + (epoch * 100)
            )
            
            time.sleep(0.2)  # Simulate training time
    
    print("Model training logging completed")


def example_error_handling():
    """Example of error logging"""
    print("\n=== Error Handling Example ===")
    
    error_logger = get_logger('error')
    
    try:
        # Simulate an error
        result = 10 / 0
    except ZeroDivisionError as e:
        error_logger.error(
            "Division by zero error in calculation",
            exc_info=True,
            extra={
                'context': {
                    'operation': 'portfolio_calculation',
                    'values': {'numerator': 10, 'denominator': 0}
                }
            }
        )
        print("Error logged with full context and traceback")


def example_performance_monitoring():
    """Example of performance monitoring"""
    print("\n=== Performance Monitoring Example ===")
    
    # Start performance monitoring
    start_performance_monitoring(interval_seconds=5)  # Short interval for demo
    
    print("Performance monitoring started (check performance_metrics.log)")
    
    # Simulate some work
    for i in range(3):
        simulate_api_call()
        time.sleep(1)
    
    # Stop performance monitoring
    stop_performance_monitoring()
    print("Performance monitoring stopped")


def example_api_logging():
    """Example of API call logging"""
    print("\n=== API Call Logging Example ===")
    
    # Simulate multiple API calls
    for i in range(5):
        try:
            result = simulate_api_call()
            print(f"API call {i+1} successful")
        except Exception as e:
            print(f"API call {i+1} failed: {e}")
        
        time.sleep(0.1)
    
    print("API call logging completed (check api_calls.log)")


def run_full_example():
    """Run all logging examples"""
    print("AutoTrader Logging System - Full Example")
    print("=" * 50)
    
    # Setup crash handler
    setup_crash_handler()
    
    # Run all examples
    example_basic_logging()
    example_structured_logging()
    example_session_logging()
    example_training_logging()
    example_error_handling()
    example_performance_monitoring()
    example_api_logging()
    
    print("\n" + "=" * 50)
    print("All logging examples completed!")
    print("Check the 'logs' directory for generated log files:")
    print("  - autotrader.log (main log)")
    print("  - trading_decisions.log (trading events)")
    print("  - training_progress.log (model training)")
    print("  - system_events.log (system events)")
    print("  - api_calls.log (API interactions)")
    print("  - performance_metrics.log (system metrics)")
    print("  - errors.log (error tracking)")
    print("  - session_*.log (individual trading sessions)")


if __name__ == "__main__":
    run_full_example()
