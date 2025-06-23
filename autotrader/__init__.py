"""
Autotrader Bot - Continuous Learning Cryptocurrency Trading System

A continuously learning trading system that utilizes machine learning 
to optimize cryptocurrency trading decisions using live data from BTCMarkets.
"""

__version__ = "0.1.0"
__author__ = "Autotrader Development Team"
__email__ = "autotrader@example.com"

# Core imports for easy access
from .core.engine import AutotraderEngine
from .core.config import Config
from .core.trader import ContinuousAutoTrader
from .utils.logging_config import setup_logging
from .utils.exceptions import AutotraderError

__all__ = [
    "AutotraderEngine",
    "Config",
    "ContinuousAutoTrader",
    "AutotraderError",
    "setup_logging",
    "__version__",
    "AutotraderEngine",
    "Config"
]
