"""
Main trader class for autotrader bot.

This is a refactored version that uses the modular package structure.
For now, it imports the original functionality until we complete the full refactoring.
"""

import sys
import os
import time

from .config import Config
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class ContinuousAutoTrader:
    """
    Main autotrader class - currently a wrapper around the original implementation.
    
    This will be fully refactored in subsequent iterations to use the new modular structure.
    """
    
    def __init__(self, config: Config = None, initial_balance: float = None):
        """
        Initialize the continuous autotrader.
        
        Args:
            config: Configuration instance (optional)
            initial_balance: Initial balance override (optional)
        """
        if config is None:
            config = Config.from_env()
        
        self.config = config
        
        # Use initial_balance parameter if provided, otherwise use config value
        balance = initial_balance if initial_balance is not None else config.initial_balance
        
        logger.info(f"Initializing ContinuousAutoTrader with balance: ${balance:.2f}")
        
        # Try to import and initialize the original trader class
        try:
            # Add the parent directory to the path so we can import the original autotrader
            parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            # Import the original autotrader.py as a module
            import importlib.util
            original_file = os.path.join(parent_dir, 'autotrader.py')
            
            if os.path.exists(original_file):
                spec = importlib.util.spec_from_file_location("original_autotrader", original_file)
                original_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(original_module)
                
                self._trader = original_module.ContinuousAutoTrader(initial_balance=balance)
                logger.info("Original trader initialized successfully")
            else:
                logger.warning("Original autotrader.py not found, using minimal implementation")
                self._create_minimal_trader(balance)
                
        except Exception as e:
            logger.warning(f"Could not initialize original trader: {e}, using minimal implementation")
            self._create_minimal_trader(balance)
    
    def _create_minimal_trader(self, balance: float):
        """Create a minimal trader implementation for testing."""
        self.balance = balance
        self.training_data = []
        logger.info("Minimal trader implementation created")
    
    def run_continuous_trading(self):
        """Run the continuous trading loop."""
        if hasattr(self, '_trader'):
            logger.info("Starting continuous trading with original implementation")
            self._trader.run_continuous_trading()
        else:
            logger.info("Starting minimal trading loop")
            self._run_minimal_loop()
    
    def _run_minimal_loop(self):
        """Minimal trading loop for testing."""
        import time
        iteration = 0
        
        try:
            while True:
                iteration += 1
                logger.info(f"Minimal trading iteration {iteration} - Balance: ${self.balance:.2f}")
                time.sleep(self.config.sleep_interval)
                
        except KeyboardInterrupt:
            logger.info("Trading loop interrupted by user")
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            raise
    
    def get_status(self) -> dict:
        """Get current trader status."""
        if hasattr(self, '_trader'):
            return {
                'balance': getattr(self._trader, 'balance', 0),
                'training_data_length': len(getattr(self._trader, 'training_data', [])),
                'scalers_fitted': getattr(self._trader, 'scalers_fitted', False)
            }
        else:
            return {
                'balance': self.balance,
                'training_data_length': len(self.training_data),
                'scalers_fitted': False
            }
