import requests
import signal
import sys
import threading
from datetime import datetime, timedelta, timezone
import tensorflow as tf
import numpy as np
import json
import time
import pickle
import os
from typing import Dict, List, Optional, Tuple, Any, Union, Deque
from collections import deque
import logging
import structlog
import os
import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory()
)
logger = structlog.get_logger()

# Constants
DEFAULT_CONFIG = {
    "confidence_threshold": 0.1,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "trade_amount": 0.001,
    "fee_rate": 0.001,
    "max_position_size": 0.1,
    "risk_per_trade": 0.02
}

# Try to import talib, with fallback for manual calculations
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    print("TA-Lib not available, using manual calculations")
    TALIB_AVAILABLE = False

# Configure logging for overnight operation
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see all messages
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autotrader.log'),
        logging.StreamHandler()
    ]
)

# Set log level for all loggers to DEBUG
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


# Import the Settings singleton
from autotrader.config.settings import get_settings

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory()
)
logger = structlog.get_logger()

# Configure logging for overnight operation
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see all messages
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autotrader.log'),
        logging.StreamHandler()
    ]
)

# Set log level for all loggers to DEBUG
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    import argparse
    from autotrader.core.continuous_autotrader import ContinuousAutoTrader # Import from new location
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='AutoTrader Bot - Continuous Cryptocurrency Trading')
    parser.add_argument('--limited-run', action='store_true', help='Run for a limited number of iterations')
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations to run in limited mode')
    parser.add_argument('--balance', type=float, default=70.0, help='Initial balance in AUD')
    
    args = parser.parse_args()

    # Instantiate and run the ContinuousAutoTrader
    trader = ContinuousAutoTrader(
        limited_run=args.limited_run,
        run_iterations=args.iterations
    )
    
    # Run the trading logic
    try:
        trader.run() # Call the main run loop
    except Exception as e:
        logger.error(f"Error during trading logic execution: {e}")
    finally:
        print("Trading completed.")

if __name__ == "__main__":
    main()
