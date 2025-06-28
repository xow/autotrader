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
from colorama import Fore, Style, init # Import colorama
import structlog
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Initialize colorama
init(autoreset=True)

# Custom processor for coloring log output
def add_coloring_processor(_, __, event_dict):
    level = event_dict.get("level")
    if level:
        if level == "info":
            event_dict["event"] = f"{Fore.GREEN}{event_dict['event']}{Style.RESET_ALL}"
        elif level == "warning":
            event_dict["event"] = f"{Fore.YELLOW}{event_dict['event']}{Style.RESET_ALL}"
        elif level == "error":
            event_dict["event"] = f"{Fore.RED}{event_dict['event']}{Style.RESET_ALL}"
        elif level == "debug":
            event_dict["event"] = f"{Fore.MAGENTA}{event_dict['event']}{Style.RESET_ALL}"
    return event_dict

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        add_coloring_processor, # Add the coloring processor here
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory()
)
logger = structlog.get_logger(__name__)

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

# Configure standard logging to file
logging.basicConfig(
    level=logging.INFO, # Set to INFO for general logging
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autotrader.log')
    ]
)

# Set log level for all loggers to INFO
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).setLevel(logging.INFO)

# Import the Settings singleton
from autotrader.config.settings import get_settings
from autotrader.core.continuous_autotrader import ContinuousAutoTrader # Import from new location

def main():
    import argparse
    
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
        logger.error(f"{Fore.RED}Error during trading logic execution: {e}{Style.RESET_ALL}")
    finally:
        print("\n" + Fore.CYAN + Style.BRIGHT + "Trading session completed." + Style.RESET_ALL)

if __name__ == "__main__":
    main()
