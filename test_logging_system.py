#!/usr/bin/env python3
"""
Test script for the AutoTrader logging system

This script tests the logging configuration and verifies that
all components are working correctly.
"""

import os
import sys
import time
from pathlib import Path

# Add the autotrader module to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_logging_system():
    """Test the complete logging system"""
    try:
        print("Testing AutoTrader Logging System...")
        print("-" * 40)
        
        # Test imports
        print("1. Testing imports...")
        from autotrader.utils.logging_config import setup_logging, get_logger
        from autotrader.utils.logging_settings import LoggingSettings
        from autotrader.utils.logging_helpers import TradingSessionLogger
        print("   ✓ All imports successful")
        
        # Test basic configuration
        print("2. Testing basic configuration...")
        config = setup_logging(
            log_dir="test_logs",
            console_level="INFO",
            file_level="DEBUG",
            structured_logging=True
        )
        print("   ✓ Logging configuration created")
        
        # Test basic logging
        print("3. Testing basic logging...")
        logger = get_logger('system')
        logger.info("Test system message")
        logger.debug("Test debug message")
        logger.warning("Test warning message")
        print("   ✓ Basic logging working")
        
        # Test structured logging
        print("4. Testing structured logging...")
        config.log_trade_decision(
            decision="BUY",
            price=65000.0,
            confidence=0.85,
            indicators={'rsi': 40.0, 'macd': 150.0},
            balance=10000.0,
            reasoning="Test trade decision"
        )
        print("   ✓ Structured trade logging working")
        
        # Test training logging
        print("5. Testing training logging...")
        config.log_training_progress(
            epoch=1,
            loss=0.1,
            accuracy=0.8,
            val_loss=0.12,
            val_accuracy=0.78,
            data_samples=1000
        )
        print("   ✓ Training progress logging working")
        
        # Test session logging
        print("6. Testing session logging...")
        with TradingSessionLogger("test_session") as session:
            session.log_trade(
                decision="SELL",
                price=64500.0,
                confidence=0.72,
                indicators={'rsi': 65.0},
                balance=10100.0,
                reasoning="Test session trade"
            )
        print("   ✓ Session logging working")
        
        # Test error logging
        print("7. Testing error logging...")
        try:
            raise ValueError("Test error for logging")
        except ValueError as e:
            config.log_error(e, "test_context", {"test_data": True})
        print("   ✓ Error logging working")
        
        # Check if log files were created
        print("8. Checking log files...")
        log_dir = Path("test_logs")
        expected_files = [
            "autotrader.log",
            "trading_decisions.log",
            "training_progress.log",
            "system_events.log",
            "errors.log"
        ]
        
        for filename in expected_files:
            filepath = log_dir / filename
            if filepath.exists():
                print(f"   ✓ {filename} created")
            else:
                print(f"   ✗ {filename} missing")
        
        print("\n" + "=" * 40)
        print("Logging system test completed!")
        print("Check the 'test_logs' directory for generated files.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup_test_logs():
    """Clean up test log files"""
    import shutil
    
    log_dir = Path("test_logs")
    if log_dir.exists():
        shutil.rmtree(log_dir)
        print("Test logs cleaned up")


if __name__ == "__main__":
    success = test_logging_system()
    
    if success:
        print("\n✅ All tests passed!")
        
        # Ask if user wants to clean up
        response = input("\nClean up test logs? (y/n): ").lower().strip()
        if response == 'y':
            cleanup_test_logs()
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
