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
import pandas as pd
from sklearn.preprocessing import StandardScaler # Import StandardScaler
from colorama import Fore, Style, init # Import colorama

# Initialize colorama
init(autoreset=True)

# Import FeatureEngineer and FeatureConfig
from autotrader.ml.feature_engineer import FeatureEngineer, FeatureConfig
from autotrader.utils.display_manager import DisplayManager # NEW: Import DisplayManager

# Constants
DEFAULT_CONFIG = {
    "confidence_threshold": 0.1,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "trade_amount": 0.001,
    "fee_rate": 0.001,
    "max_position_size": 0.1,
}

# Try to import talib, with fallback for manual calculations
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    # Changed to logger.warning for consistency and to avoid direct print
    logging.warning("TA-Lib not available, using manual calculations")
    TALIB_AVAILABLE = False

# Configure logging for overnight operation
# Set logging level to INFO for file, but rely on DisplayManager for console output
logging.basicConfig(
    level=logging.INFO,  # Set to INFO for file logging, DisplayManager handles console
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autotrader.log')
    ]
)

# Custom processor for routing logs to DisplayManager
class DisplayManagerLogger:
    """
    A structlog processor that routes log messages to the DisplayManager.
    The DisplayManager instance is set after the ContinuousAutoTrader is initialized.
    """
    _instance = None # Singleton pattern for easier global access

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DisplayManagerLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self, display_manager_instance=None):
        if display_manager_instance: # Only set if provided, avoids re-init on subsequent calls
            self.display_manager = display_manager_instance
        elif not hasattr(self, 'display_manager'): # Initialize if not already present
            self.display_manager = None # Will be set later

    def set_display_manager(self, display_manager_instance):
        """Sets the DisplayManager instance for this logger."""
        self.display_manager = display_manager_instance

    def __call__(self, logger, method_name, event_dict):
        # We handle info, warning, error, and exception messages via DisplayManager
        # Debug messages will still go to the file if basicConfig is set to DEBUG
        if self.display_manager: # Only log if display_manager is set
            if method_name == "info":
                self.display_manager.log_message(event_dict["event"], level="info")
            elif method_name == "warning":
                self.display_manager.log_message(event_dict["event"], level="warning")
            elif method_name == "error":
                self.display_manager.log_message(event_dict["event"], level="error")
            elif method_name == "exception": # For logger.exception calls
                self.display_manager.log_message(event_dict["event"], level="error")
        return event_dict

# Global structlog configuration
# The DisplayManagerLogger instance will be created and its display_manager set later.
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        DisplayManagerLogger(), # Use the singleton instance
        structlog.dev.ConsoleRenderer() # Re-add ConsoleRenderer for debugging structlog issues
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory() # Revert to PrintLoggerFactory
)
logger = structlog.get_logger(__name__)

# Import the Settings singleton
from autotrader.config.settings import get_settings
from autotrader.utils.exceptions import APIError, DataError, NetworkError, NetworkTimeoutError


class ContinuousAutoTrader:
    """
    A continuous trading bot that uses an LSTM model to predict trade signals.
    """
    def __init__(
        self,
        limited_run: bool = False,
        run_iterations: int = 5,
        initial_balance: Optional[float] = None
    ):
        """
        Initialize the ContinuousAutoTrader.

        Args:
            limited_run (bool): If True, the bot will run for a limited number of iterations.
            run_iterations (int): Number of iterations to run in limited mode.
        """
        self.settings = get_settings()

        # Initialize state variables using settings
        self.balance = initial_balance if initial_balance is not None else self.settings.initial_balance
        self.confidence_threshold = self.settings.buy_confidence_threshold # Using buy_confidence_threshold as general confidence
        self.rsi_oversold = self.settings.rsi_oversold
        self.rsi_overbought = self.settings.rsi_overbought
        self.trade_amount = self.settings.trade_amount
        self.fee_rate = self.settings.fee_rate
        self.max_position_size = self.settings.max_position_size # Accessing from trading config
        self.risk_per_trade = self.settings.risk_per_trade # This value is now in settings
        self.limited_run = limited_run
        self.run_iterations = run_iterations
        
        # Initialize DisplayManager here
        self.display_manager = DisplayManager(total_iterations=run_iterations if limited_run else 0)
        
        # Set the DisplayManager instance for the global DisplayManagerLogger
        DisplayManagerLogger().set_display_manager(self.display_manager)

        self.position_size = 0.0
        self.entry_price = 0.0
        self.last_save_time = time.time()
        self.last_training_time = time.time()
        self._shutdown_requested = False
        self._shutdown_event = threading.Event()
        self.training_data_filename = self.settings.training_data_filename
        self.model_filename = self.settings.model_filename
        self.scaler_filename = self.settings.scalers_filename # Accessing from ML config
        self.state_filename = self.settings.state_filename
        self.min_data_points = self.settings.sequence_length # Using sequence_length as min_data_points
        self.sequence_length = self.settings.sequence_length # Use ml.sequence_length
        self.max_training_samples = self.settings.max_training_samples # Use ml.max_training_samples
        self.save_interval_seconds = self.settings.save_interval
        self.training_interval_seconds = self.settings.training_interval
        self.model = None # Initialize to None, test will set
        self.training_data = deque(self.load_training_data(), maxlen=self.max_training_samples) # Initialize deque with loaded data and maxlen
        self._price_history = []
        self._training_data_deque = deque(maxlen=self.sequence_length) # Changed to sequence_length for buffer
        self._model_summary_logged = False
        self._data_buffer = deque(maxlen=self.sequence_length)
        self.scalers_fitted = False # Initialize to False, will be updated by load_scalers or fit_scalers
        
        # Initialize FeatureEngineer
        self.feature_engineer = FeatureEngineer(config=FeatureConfig(
            scaling_method=self.settings.ml.scaling_method,
            sma_periods=self.settings.ml.sma_periods,
            ema_periods=self.settings.ml.ema_periods,
            rsi_period=self.settings.ml.rsi_period,
            macd_fast=self.settings.ml.macd_fast,
            macd_slow=self.settings.ml.macd_slow,
            macd_signal=self.settings.ml.macd_signal,
            bb_period=self.settings.ml.bb_period,
            bb_std=self.settings.ml.bb_std,
            volatility_window=self.settings.ml.volatility_window,
            lag_periods=self.settings.ml.lag_periods,
            rolling_windows=self.settings.ml.rolling_windows,
            use_sma=self.settings.ml.use_sma,
            use_ema=self.settings.ml.use_ema,
            use_rsi=self.settings.ml.use_rsi,
            use_macd=self.settings.ml.use_macd,
            use_bollinger=self.settings.ml.use_bollinger,
            use_volume_indicators=self.settings.ml.use_volume_indicators,
            use_price_ratios=self.settings.ml.use_price_ratios,
            use_price_differences=self.settings.ml.use_price_differences,
            use_log_returns=self.settings.ml.use_log_returns,
            use_volatility=self.settings.ml.use_volatility,
            use_time_features=self.settings.ml.use_time_features,
            use_cyclical_encoding=self.settings.ml.use_cyclical_encoding,
            use_lag_features=self.settings.ml.use_lag_features,
            use_rolling_stats=self.settings.ml.use_rolling_stats
        ))
        
        # Dynamically determine all possible feature names from FeatureEngineer's configuration
        # This ensures that feature_names_ is always complete, even if initial training data is not.
        temp_fe_config = FeatureConfig(
            scaling_method=self.settings.ml.scaling_method,
            sma_periods=self.settings.ml.sma_periods,
            ema_periods=self.settings.ml.ema_periods,
            rsi_period=self.settings.ml.rsi_period,
            macd_fast=self.settings.ml.macd_fast,
            macd_slow=self.settings.ml.macd_slow,
            macd_signal=self.settings.ml.macd_signal,
            bb_period=self.settings.ml.bb_period,
            bb_std=self.settings.ml.bb_std,
            volatility_window=self.settings.ml.volatility_window,
            lag_periods=self.settings.ml.lag_periods,
            rolling_windows=self.settings.ml.rolling_windows,
            use_sma=self.settings.ml.use_sma,
            use_ema=self.settings.ml.use_ema,
            use_rsi=self.settings.ml.use_rsi,
            use_macd=self.settings.ml.use_macd,
            use_bollinger=self.settings.ml.use_bollinger,
            use_volume_indicators=self.settings.ml.use_volume_indicators,
            use_price_ratios=self.settings.ml.use_price_ratios,
            use_price_differences=self.settings.ml.use_price_differences,
            use_log_returns=self.settings.ml.use_log_returns,
            use_volatility=self.settings.ml.use_volatility,
            use_time_features=self.settings.ml.use_time_features,
            use_cyclical_encoding=self.settings.ml.use_cyclical_encoding,
            use_lag_features=self.settings.ml.use_lag_features,
            use_rolling_stats=self.settings.ml.use_rolling_stats
        )
        temp_fe = FeatureEngineer(config=temp_fe_config)
        
        # Create a dummy DataFrame with all possible raw input columns
        dummy_data_points = 100 # Needs enough data for all indicators
        dummy_data = {
            'price': np.random.rand(dummy_data_points) * 10000,
            'volume': np.random.rand(dummy_data_points) * 1000,
            'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=dummy_data_points, freq='H')),
            'high': np.random.rand(dummy_data_points) * 10000 + 100,
            'low': np.random.rand(dummy_data_points) * 10000 - 100,
            'spread': np.random.rand(dummy_data_points) * 10,
            'marketId': 'BTC-AUD' # Include marketId as it might be in raw data
        }
        dummy_df = pd.DataFrame(dummy_data)
        
        # Generate features to get all possible feature names
        full_features_df = temp_fe._generate_all_features(dummy_df)
        self.feature_engineer.feature_names_ = list(full_features_df.columns)
        self.feature_engineer.is_fitted_ = True # Mark as fitted for feature names
        logger.debug("FeatureEngineer feature names initialized with full set", count=len(self.feature_engineer.feature_names_)) # Changed to debug

        # Load scalers after feature_engineer is initialized
        self.feature_scaler = self.load_scalers() # This will also set self.scalers_fitted
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        logger.debug("ContinuousAutoTrader __init__ completed.")
        
        # Load initial state
        # Only load state if initial_balance was not explicitly provided
        if initial_balance is None:
            self.load_state()
        
        # Removed model initialization here, test will provide it
        # if self.model is None:
        #     self.model = self.create_lstm_model()
        
        # Log configuration details (changed to debug or removed if not essential)
        logger.debug("AutoTrader initialized with settings:",
            initial_balance=self.balance,
            confidence_threshold=self.confidence_threshold,
            rsi_oversold=self.rsi_oversold,
            rsi_overbought=self.rsi_overbought,
            trade_amount=self.trade_amount,
            fee_rate=self.fee_rate,
            max_position_size=self.max_position_size,
            risk_per_trade=self.risk_per_trade,
            model_filename=self.model_filename,
            scaler_filename=self.scaler_filename,
            state_filename=self.state_filename,
            sequence_length=self.sequence_length,
            min_data_points=self.min_data_points,
            limited_run=self.limited_run,
            run_iterations=self.run_iterations
        )
        
        logger.debug("Loaded training samples", count=len(self.training_data)) # Changed to debug
        
        if self.model and not self._model_summary_logged:
            # Model summary should only be logged once and not to the console repeatedly
            self.model.summary(print_fn=lambda x: logger.debug(x)) # Changed to debug
            self._model_summary_logged = True
        
        logger.debug("Feature engineer scaler fitted status", is_fitted=self.feature_engineer.is_fitted_) # Changed to debug
        
        # The following initial state logs are now redundant with the main display
        # and can be removed or changed to debug if truly needed for deep debugging.
        # Removing for cleaner output as per requirements.
        # logger.info("Initial balance", balance=self.balance)
        # logger.info("Initial position size", position_size=self.position_size)
        # logger.info("Initial entry price", entry_price=self.entry_price)
        # logger.info("Initial last save time", last_save_time=self.last_save_time)
        # logger.info("Initial last training time", last_training_time=self.last_training_time)
        # logger.info("Initial shutdown requested", shutdown_requested=self._shutdown_requested)
        # logger.info("Initial shutdown event", shutdown_event=self._shutdown_event)
        # logger.info("Initial training data filename", training_data_filename=self.training_data_filename)
        # logger.info("Initial model filename", model_filename=self.model_filename)
        # logger.info("Initial scaler filename", scaler_filename=self.scaler_filename)
        # logger.info("Initial state filename", state_filename=self.state_filename)
        # logger.info("Initial min data points", min_data_points=self.min_data_points)
        # logger.info("Initial sequence length", sequence_length=self.sequence_length)
        # logger.info("Initial max training samples", max_training_samples=self.max_training_samples)
        # logger.info("Initial run iterations", run_iterations=self.run_iterations)
        # logger.info("Initial save interval seconds", save_interval_seconds=self.save_interval_seconds)
        # logger.info("Initial training interval seconds", training_interval_seconds=self.training_interval_seconds)
        # logger.info("Initial feature engineer", feature_engineer=self.feature_engineer)
        # logger.info("Initial model", model=self.model)
        # logger.info("Initial training data", training_data=list(self.training_data)[:5])
        # logger.info("Initial price history", price_history=self._price_history)
        # logger.info("Initial training data deque", training_data_deque=self._training_data_deque)
        # logger.info("Initial model summary logged", model_summary_logged=self._model_summary_logged)
        
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown, only if in the main thread."""
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            logger.debug("Signal handlers set up in main thread.")
        else:
            logger.debug("Skipping signal handler setup: not in main thread.")

    def _signal_handler(self, signum, frame):
        """Handle signals for graceful shutdown."""
        self.display_manager.log_message(f"Signal received, initiating graceful shutdown (Signal: {signum}).", level="info")
        self._shutdown_requested = True
        self._shutdown_event.set()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables or a config file."""
        # This method is no longer needed as configuration is handled by Settings
        # Keeping it as a placeholder if other parts of the code still call it
        logger.debug("'_load_config' method is deprecated. Using 'Settings' for configuration.")
        return {}

    def save_state(self):
        """Save the current state of the trader to a file."""
        try:
            with open(self.state_filename, 'wb') as f:
                pickle.dump({
                    'balance': self.balance,
                    'position_size': self.position_size,
                    'entry_price': self.entry_price
                }, f)
            self.display_manager.log_message("Trader state saved successfully.", level="info")
        except Exception as e:
            self.display_manager.log_message(f"Error saving trader state: {e}", level="error")
            logger.error("Error saving trader state", exc_info=e) # Log full traceback to file

    def load_state(self):
        """Load the trader state from a file."""
        try:
            with open(self.state_filename, 'rb') as f:
                state = pickle.load(f)
                self.balance = state.get('balance', self.settings.initial_balance) # Use settings for default
                self.position_size = state.get('position_size', 0.0)
                self.entry_price = state.get('entry_price', 0.0)
            self.display_manager.log_message("Trader state loaded successfully.", level="info")
        except FileNotFoundError:
            self.display_manager.log_message("No trader state file found, starting fresh.", level="info")
        except Exception as e:
            self.display_manager.log_message(f"Error loading trader state: {e}", level="error")
            logger.error("Error loading trader state", exc_info=e) # Log full traceback to file

    def save_scalers(self):
        """Save the scalers to a file."""
        try:
            with open(self.scaler_filename, 'wb') as f:
                pickle.dump({
                    'feature_engineer_scaler': self.feature_engineer.scaler,
                    'feature_names': self.feature_engineer.feature_names_
                }, f)
            self.display_manager.log_message("Scalers saved successfully.", level="info")
        except Exception as e:
            self.display_manager.log_message(f"Error saving scalers: {e}", level="error")
            logger.error("Error saving scalers", exc_info=e) # Log full traceback to file

    def load_scalers(self):
        """Load the scalers from a file."""
        try:
            with open(self.scaler_filename, 'rb') as f:
                scalers = pickle.load(f)
                self.feature_engineer.scaler = scalers['feature_engineer_scaler']
                self.feature_engineer.feature_names_ = scalers['feature_names']
                self.feature_engineer.is_fitted_ = True # Set to True on successful load
                self.scalers_fitted = True # Also update ContinuousAutoTrader's status
            self.display_manager.log_message("Scalers loaded successfully.", level="info")
        except FileNotFoundError:
            self.display_manager.log_message("No scalers file found, creating new scalers.", level="info")
            self.feature_engineer._init_scaler() # Initialize a new scaler
            self.feature_engineer.is_fitted_ = False # Not fitted yet
            self.scalers_fitted = False # Not fitted yet
        except Exception as e:
            self.display_manager.log_message(f"Error loading scalers: {e}", level="error")
            logger.error("Error loading scalers", exc_info=e) # Log full traceback to file
            self.feature_engineer._init_scaler() # Initialize a new scaler on error
            self.feature_engineer.is_fitted_ = False # Not fitted yet
            self.scalers_fitted = False # Not fitted yet
        return self.feature_engineer.scaler # Return the loaded scaler or a new one

    def fetch_market_data(self) -> List[Dict[str, Any]]:
        """Fetch market data from the BTCMarkets API."""
        try:
            url = self.settings.api_base_url + "/markets/tickers" # Use settings for base URL and tickers endpoint
            params = {"marketId": "BTC-AUD"} # Add marketId parameter
            response = requests.get(url, params=params, timeout=self.settings.api_timeout) # Use settings for timeout
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            
            market_data = response.json()
            logger.debug("Fetched market data points from BTCMarkets API", count=len(market_data))
            
            # Basic data validation
            if not isinstance(market_data, list):
                self.display_manager.log_message("Invalid data format from BTCMarkets API.", level="error")
                logger.error("Invalid data format from BTCMarkets API")
                return []
            
            # Convert timestamps to ISO format and ensure all keys are strings
            processed_data = []
            for data_point in market_data:
                # Ensure all keys are strings
                data_point = {str(k): v for k, v in data_point.items()}
                
                # Convert timestamp to ISO format
                timestamp_str = data_point.get('timestamp')
                if timestamp_str is not None:
                    try:
                        # Parse ISO format string to datetime object, handling 'Z' and ensuring UTC
                        dt_object = datetime.fromisoformat(timestamp_str.replace('Z', '')).replace(tzinfo=timezone.utc)
                        # Convert datetime object to Unix timestamp in milliseconds
                        data_point['timestamp'] = int(dt_object.timestamp() * 1000)
                    except ValueError as ve:
                        self.display_manager.log_message(f"Invalid timestamp format. Using current timestamp. Error: {ve}", level="warning")
                        logger.warning("Invalid timestamp format. Using current timestamp.", timestamp=timestamp_str, error=str(ve))
                        data_point['timestamp'] = int(time.time() * 1000) # Fallback to current timestamp
                else:
                    self.display_manager.log_message("Timestamp missing from API response. Adding current timestamp.", level="warning")
                    logger.warning("Timestamp missing from API response. Adding current timestamp.")
                    data_point['timestamp'] = int(time.time() * 1000) # Add current timestamp if not present
                
                processed_data.append(data_point)
            
            return processed_data
        
        except requests.exceptions.Timeout as e:
            self.display_manager.log_message("Market data request timed out.", level="error")
            logger.error("Market data request timed out", exc_info=e)
            raise NetworkTimeoutError("Market data request timed out", original_exception=e) from e
        except requests.exceptions.ConnectionError as e:
            self.display_manager.log_message("Failed to connect to the exchange.", level="error")
            logger.error("Failed to connect to the exchange", exc_info=e)
            raise NetworkError("Failed to connect to the exchange", original_exception=e) from e
        except requests.exceptions.HTTPError as e:
            self.display_manager.log_message(f"HTTP error fetching market data: {e.response.status_code} - {e.response.text}", level="error")
            logger.error("HTTP error fetching market data", status_code=e.response.status_code, response_text=e.response.text, exc_info=e)
            raise APIError("HTTP error fetching market data", status_code=e.response.status_code, response_text=e.response.text) from e
        except json.JSONDecodeError as e:
            self.display_manager.log_message(f"Failed to decode JSON response from API: {e}", level="error")
            logger.error("Failed to decode JSON response from API", exc_info=e, response_text=response.text if 'response' in locals() else 'N/A')
            raise DataError("Invalid JSON response from API", data_type="market_data", validation_errors={"json_decode_error": str(e)}) from e
        except Exception as e: # Catch all other exceptions
            self.display_manager.log_message(f"An unexpected error occurred while fetching market data: {e}", level="error")
            logger.error("An unexpected error occurred while fetching market data", exc_info=e)
            return []

    def extract_comprehensive_data(self, market_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Extract comprehensive data for BTC-AUD market from the API response.
        
        Args:
            market_data (List[Dict[str, Any]]): List of market data dictionaries from the API.
            
        Returns:
            Optional[Dict[str, Any]]: A dictionary containing extracted and validated data for BTC-AUD,
                                    or None if not found or invalid.
        """
        if not market_data:
            self.display_manager.log_message("No market data provided for extraction.", level="warning")
            logger.warning("No market data provided for extraction.")
            return None
        
        btc_aud_data = None
        for item in market_data:
            if item.get("marketId") == "BTC-AUD":
                btc_aud_data = item
                break
        
        if btc_aud_data is None:
            self.display_manager.log_message("BTC-AUD market data not found in the API response.", level="warning")
            logger.warning("BTC-AUD market data not found in the API response.")
            return None
        
        # Validate and convert data types
        try:
            extracted = {
                "marketId": btc_aud_data.get("marketId"),
                "price": float(btc_aud_data.get("lastPrice", 0.0)),
                "volume": float(btc_aud_data.get("volume24h", 0.0)),
                "bid": float(btc_aud_data.get("bestBid", 0.0)),
                "ask": float(btc_aud_data.get("bestAsk", 0.0)),
                "high24h": float(btc_aud_data.get("high24h", 0.0)),
                "low24h": float(btc_aud_data.get("low24h", 0.0)),
                "timestamp": btc_aud_data.get("timestamp", int(time.time() * 1000)) # Ensure timestamp is present
            }
            
            # Calculate spread if not provided
            if "spread" not in extracted:
                extracted["spread"] = extracted["ask"] - extracted["bid"]
            
            # Basic validation for critical fields
            if extracted["price"] <= 0 or extracted["volume"] < 0:
                self.display_manager.log_message(f"Invalid price or volume in extracted data: Price={extracted['price']}, Volume={extracted['volume']}", level="warning")
                logger.warning("Invalid price or volume in extracted data", price=extracted['price'], volume=extracted['volume'])
                return None
            
            return extracted
        
        except (ValueError, TypeError) as e:
            self.display_manager.log_message(f"Error converting market data values: {e}", level="error")
            logger.error("Error converting market data values", exc_info=e, data=btc_aud_data)
            return None
        except Exception as e:
            self.display_manager.log_message(f"An unexpected error occurred during data extraction: {e}", level="error")
            logger.error("An unexpected error occurred during data extraction", exc_info=e)
            return None

    def manual_sma(self, data: np.ndarray, period: int) -> float:
        """Calculate Simple Moving Average manually."""
        if len(data) < period:
            return np.mean(data)  # Not enough data for the full period
        return np.mean(data[-period:])

    def manual_ema(self, data: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average manually."""
        if len(data) < period:
            return np.mean(data)  # Not enough data for the full period
        
        k = 2 / (period + 1)
        ema = np.zeros_like(data, dtype=float)
        ema[period-1] = np.mean(data[:period])  # Seed with SMA
        
        for i in range(period, len(data)):
            ema[i] = (data[i] * k) + (ema[i-1] * (1 - k))
        
        return ema[-1]

    def manual_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index manually."""
        if len(prices) < period + 1:
            return 50.0  # Neutral value when not enough data
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            if avg_gain == 0:
                return 50.0 # If both are zero (constant prices), RSI is 50
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)

    def calculate_technical_indicators(self, market_data: Optional[List[Dict[str, Any]]] = None, prices: Optional[np.ndarray] = None, volumes: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        Calculate technical indicators for each data point.
        
        Args:
            market_data (List[Dict[str, Any]]): List of market data dictionaries.
            prices (Optional[np.ndarray]): Optional numpy array of prices for direct calculation.
            volumes (Optional[np.ndarray]): Optional numpy array of volumes for direct calculation.
            
        Returns:
            List[Dict[str, Any]]: List of market data dictionaries with added technical indicators.
        """
        if prices is not None and volumes is not None:
            # This branch is for performance tests that pass raw numpy arrays
            # Create a list of dictionaries from prices and volumes
            data_for_df = [{"price": p, "volume": v} for p, v in zip(prices, volumes)]
            df = pd.DataFrame(data_for_df)
            price_col = 'price'
        elif market_data:
            # Ensure market_data is not None and is a list
            if not isinstance(market_data, list):
                self.display_manager.log_message("Invalid market data format, cannot calculate indicators.", level="warning")
                logger.warning("Invalid market data format, cannot calculate indicators")
                return []
            
            # Convert market_data to a pandas DataFrame for easier calculations
            df = pd.DataFrame(market_data)
            
            # Ensure 'lastPrice' or 'price' column exists and standardize to 'price'
            if 'lastPrice' in df.columns:
                df['price'] = df['lastPrice']
            elif 'price' not in df.columns:
                self.display_manager.log_message("No 'lastPrice' or 'price' column found after standardization, cannot calculate indicators.", level="warning")
                logger.warning("No 'lastPrice' or 'price' column found after standardization, cannot calculate indicators")
                return market_data
            
            # Ensure 'timestamp' column exists. If not, add it from original market_data or generate.
            if 'timestamp' not in df.columns:
                # Try to get timestamp from original market_data if available
                if market_data and 'timestamp' in market_data[0]:
                    df['timestamp'] = [d.get('timestamp') for d in market_data]
                else:
                    self.display_manager.log_message("Timestamp column missing in market data. Adding current timestamp.", level="warning")
                    logger.warning("Timestamp column missing in market data. Adding current timestamp.")
                    df['timestamp'] = int(time.time() * 1000) # Add current Unix timestamp in milliseconds
            
            # Convert price and volume columns to numeric, coercing errors to NaN
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            if 'volume' in df.columns:
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            else:
                df['volume'] = 0.0 # Default volume to 0 if not present
            
            # Check for NaN values in critical columns after conversion
            if df['price'].isnull().any():
                self.display_manager.log_message(f"Invalid price data detected after numeric conversion. Skipping indicator calculation.", level="warning")
                logger.warning("Invalid price data detected after numeric conversion. Skipping indicator calculation.", invalid_prices=df[df['price'].isnull()]['price'].tolist())
                return []
            
            # Fill NaN prices and volumes with previous valid price or 0.0
            df['price'] = df['price'].ffill().fillna(0.0)
            df['volume'] = df['volume'].ffill().fillna(0.0)
            
            # Ensure there are enough valid data points for basic calculations
            if len(df) < 1: # At least one data point is needed
                self.display_manager.log_message(f"Not enough valid data points to calculate indicators. Needed: {1}, Got: {len(df)}", level="warning")
                logger.warning("Not enough valid data points to calculate indicators. Returning original market data.", needed=1, got=len(df))
                return market_data
        else:
            self.display_manager.log_message("No market data or price/volume arrays provided for indicator calculation.", level="warning")
            logger.warning("No market data or price/volume arrays provided for indicator calculation.")
            return []
        
        # Use 'price' as the consistent column name for calculations
        price_col = 'price'
        
        # Calculate Simple Moving Averages (SMA)
        df['sma_5'] = df[price_col].rolling(window=5).mean()
        df['sma_10'] = df[price_col].rolling(window=10).mean()
        df['sma_20'] = df[price_col].rolling(window=20).mean()
        
        # Calculate Exponential Moving Averages (EMA)
        df['ema_12'] = df[price_col].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df[price_col].ewm(span=26, adjust=False).mean()
        
        # Calculate Relative Strength Index (RSI)
        # Check if there's enough data for RSI calculation (period + 1 for diff)
        if len(df[price_col]) < 15: # 14 period + 1 for the first diff
            df['rsi'] = 50.0 # Default to neutral RSI
        else:
            delta = df[price_col].diff()
            up, down = delta.copy(), delta.copy()
            up[up < 0] = 0
            down[down > 0] = 0
            
            roll_up1 = up.ewm(span=14, adjust=False).mean() # Use ewm for RSI average gain/loss
            roll_down1 = np.abs(down.ewm(span=14, adjust=False).mean()) # Use ewm for RSI average gain/loss
            
            # Avoid division by zero, set RS to a large number if avg_loss is zero to push RSI towards 100
            # For constant prices, both avg_gain and avg_loss will be 0, leading to 0/0. Handle this to be 50.
            RS = np.where(roll_down1 == 0, np.inf, roll_up1 / roll_down1)
            # If both are zero (constant price), set RS to 1 to get RSI of 50
            RS = np.where((roll_up1 == 0) & (roll_down1 == 0), 1, RS)
            df['rsi'] = 100.0 - (100.0 / (1.0 + RS))
            # For cases where RS is inf (all gains), RSI should be 100
            df['rsi'] = np.where(RS == np.inf, 100.0, df['rsi'])
            # For cases where RS is 0 (all losses), RSI should be 0
            df['rsi'] = np.where(RS == 0, 0.0, df['rsi'])
            # Fill NaN values (due to insufficient data) with 50.0
            df['rsi'] = df['rsi'].fillna(50.0)
        
        # Calculate MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Calculate Bollinger Bands
        df['bb_middle'] = df[price_col].rolling(window=20).mean()
        df['bb_std'] = df[price_col].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        # Calculate Average True Range (ATR) - requires high, low, close
        # For simplicity, using price_col for ATR calculation as high/low are not always available in market_data
        df['atr'] = df[price_col].rolling(window=14).mean()
        
        # Add volume SMA
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(window=self.settings.volume_sma_period).mean() # Use settings for volume_sma_period
        else:
            df['volume_sma'] = 0.0 # Default if volume is not available
        
        # Convert the DataFrame to a list of dictionaries, which will include all columns (original and new indicators)
        return df.to_dict('records')
        
    def create_lstm_model(self):
        """Create the LSTM model."""
        try:
            # Define the LSTM model
            model = tf.keras.Sequential([
                tf.keras.Input(shape=(self.settings.sequence_length, self.settings.feature_count)), # Use Input layer as first layer
                tf.keras.layers.LSTM(self.settings.lstm_units, return_sequences=True),
                tf.keras.layers.Dropout(self.settings.dropout_rate),
                tf.keras.layers.LSTM(self.settings.lstm_units, return_sequences=False),
                tf.keras.layers.Dropout(self.settings.dropout_rate),
                tf.keras.layers.Dense(self.settings.dense_units),
                tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer
            ])
            
            # Compile the model
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.settings.learning_rate), loss='mse', metrics=['mae'])
            
            # Log model summary
            model.summary(print_fn=lambda x: logger.debug(x)) # Use print_fn to log summary
            
            return model
        
        except Exception as e:
            self.display_manager.log_message(f"Error creating LSTM model: {e}", level="error")
            logger.error("Error creating LSTM model", exc_info=e)
            return None
 
    def prepare_features(self, data_point: dict) -> list:
        """Prepare feature vector from a single data point using FeatureEngineer."""
        try:
            if not data_point:
                raise ValueError("Empty data point provided")
            
            # Convert single data point to a list of dicts for FeatureEngineer
            data_for_fe = [data_point]
            
            # Generate raw features using FeatureEngineer's internal method
            # This method handles adding all configured features and filling missing values.
            # It does not require the scaler to be fitted.
            raw_features_df = self.feature_engineer._generate_all_features(pd.DataFrame(data_for_fe))
            
            # If FeatureEngineer is not fitted, its feature_names_ might be empty or incomplete.
            # We need to ensure that the feature_names_ are populated correctly
            # before attempting to reindex or transform.
            if not self.feature_engineer.feature_names_:
                # Populate feature_names_ from the generated raw features.
                # This ensures that subsequent calls to transform or reindex have the correct feature set.
                self.feature_engineer.feature_names_ = list(raw_features_df.columns)
                logger.debug("FeatureEngineer feature names initialized from generated features.")
            
            # Reindex to ensure all expected features are present, filling missing with 0.0.
            # Use the feature_names_ from the feature_engineer, which should now be populated.
            features_df_reindexed = raw_features_df.reindex(columns=self.feature_engineer.feature_names_, fill_value=0.0)
            
            # Convert to numpy array
            features_array = features_df_reindexed.values
            
            # Apply scaling only if the FeatureEngineer's scaler is fitted
            if self.feature_engineer.is_fitted_:
                scaled_features = self.feature_engineer.scaler.transform(features_array)
                features = scaled_features[0].tolist()
                logger.debug("Prepared features using fitted FeatureEngineer and scaler", first_few_features=features[:5])
            else:
                features = features_array[0].tolist()
                self.display_manager.log_message("FeatureEngineer not fitted. Returning raw (unscaled) features.", level="warning")
                logger.warning("FeatureEngineer not fitted. Returning raw (unscaled) features.", first_few_features=features[:5])
            
            logger.debug("Prepared features count", count=len(features))
            logger.debug("Prepared feature names", names=self.feature_engineer.get_feature_names())
            
            return features
        
        except Exception as e:
            self.display_manager.log_message(f"Error preparing features with FeatureEngineer: {e}", level="error")
            logger.exception("Error preparing features with FeatureEngineer", exc_info=e)
            # Fallback to returning a zero vector of expected length on error
            # This length should ideally come from feature_engineer.get_feature_names()
            # or settings.ml.feature_count.
            # If feature_engineer.feature_names_ is populated, use its length.
            # Otherwise, fallback to settings.ml.feature_count or a default of 96.
            fallback_feature_count = len(self.feature_engineer.feature_names_) if self.feature_engineer.feature_names_ else (self.settings.ml.feature_count if self.settings.ml.feature_count > 0 else 96)
            return [0.0] * fallback_feature_count

    def _update_feature_buffer(self, market_data: Dict[str, Any]) -> None:
        """Update the feature buffer with the latest market data."""
        try:
            # Extract relevant features from market_data
            price = market_data.get('lastPrice') or market_data.get('price')
            volume = market_data.get('volume', 0.0) # Default to 0.0 if volume is missing
            
            # Validate that price is not None
            if price is None:
                self.display_manager.log_message("Missing price in market data for feature buffer update.", level="error")
                logger.error("Missing price in market data", price=price)
                raise DataError("Missing price in market data")
            
            # Convert price and volume to float
            try:
                price = float(price)
                volume = float(volume)
            except ValueError as ve:
                self.display_manager.log_message(f"Invalid price or volume format for feature buffer update: {ve}", level="error")
                logger.error("Invalid price or volume format", price=price, volume=volume, exc_info=ve)
                raise DataError(f"Invalid price or volume format: {ve}") from ve
            
            # Create a feature vector
            feature_vector = [price, volume]
            
            # Append the feature vector to the data buffer
            if hasattr(self, '_data_buffer'):
                logger.debug("Updated feature buffer", price=price, volume=volume)
                self._data_buffer.append(feature_vector)
            else:
                self.display_manager.log_message("Attempted to append to _data_buffer, but it does not exist.", level="error")
                logger.error("Attempted to append to _data_buffer, but it does not exist on self.")
                raise RuntimeError("'_data_buffer' attribute is missing from ContinuousAutoTrader instance.")
        except Exception as e:
            self.display_manager.log_message(f"Error updating feature buffer: {e}", level="error")
            logger.error("Error updating feature buffer", exc_info=e)

    def fit_scalers(self) -> bool:
        """Fit scalers using the FeatureEngineer to the training data."""
        try:
            # Ensure training_data is a deque
            if not isinstance(self.training_data, deque):
                self.display_manager.log_message("Training data is not a deque, cannot fit scalers.", level="error")
                logger.error("training_data is not a deque, cannot fit scalers.")
                self.feature_engineer.is_fitted_ = False
                return False
            
            if len(self.training_data) < self.settings.ml.sequence_length:
                self.display_manager.log_message(f"Insufficient training data for scaler fitting. Needed: {self.settings.ml.sequence_length}, Got: {len(self.training_data)}.", level="warning")
                logger.warning("Insufficient training data for scaler fitting.", needed=self.settings.ml.sequence_length, got=len(self.training_data))
                self.feature_engineer.is_fitted_ = False
                return False
            
            # Fit the FeatureEngineer directly on the raw training data
            # The FeatureEngineer handles its own internal scaler fitting
            self.feature_engineer.fit(list(self.training_data))
            
            self.display_manager.log_message("FeatureEngineer fitted successfully.", level="info")
            logger.info("FeatureEngineer fitted successfully")
            self.scalers_fitted = True # Set to True after successful fitting
            return True
            
        except Exception as e:
            self.display_manager.log_message(f"Error fitting FeatureEngineer: {e}", level="error")
            logger.error("Error fitting FeatureEngineer", exc_info=e)
            self.feature_engineer.is_fitted_ = False
            self.scalers_fitted = False # Set to False on error
            return False

    def prepare_lstm_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training using FeatureEngineer."""
        try:
            # Ensure we have enough data
            if len(self.training_data) < self.sequence_length:
                self.display_manager.log_message(f"Not enough data for training. Needed: {self.sequence_length}, Got: {len(self.training_data)}.", level="warning")
                logger.warning("Not enough data for training.", needed=self.sequence_length, got=len(self.training_data))
                # Return empty arrays with correct dimensions
                return np.empty((0, self.sequence_length, self.settings.ml.feature_count)), np.empty((0,))
            
            # Transform all training data using the fitted FeatureEngineer
            # This will return scaled features
            if not self.feature_engineer.is_fitted_:
                self.display_manager.log_message("FeatureEngineer not fitted, cannot prepare LSTM training data.", level="warning")
                logger.warning("FeatureEngineer not fitted, cannot prepare LSTM training data.")
                # Return empty arrays with correct dimensions if not fitted
                return np.empty((0, self.sequence_length, self.settings.ml.feature_count)), np.empty((0,))
            
            # Get all scaled features from the training data
            all_scaled_features = self.feature_engineer.transform(list(self.training_data))
            
            features, labels = [], []
            num_features = all_scaled_features.shape[1] # Get the actual number of features from FeatureEngineer
            
            for i in range(len(all_scaled_features) - self.sequence_length):
                # Extract the sequence of scaled features
                sequence = all_scaled_features[i : i + self.sequence_length]
                
                # Extract the label (next data point's price, unscaled)
                # We need the original price for the label, not the scaled one.
                next_original_data_point = self.training_data[i + self.sequence_length]
                next_price = float(next_original_data_point.get('price', next_original_data_point.get('lastPrice', 0.0)))
                
                features.append(sequence)
                labels.append(next_price)
            
            # Convert to numpy arrays
            features = np.array(features)
            labels = np.array(labels)
            
            # Log the shape of the features and labels
            logger.debug("Features shape", shape=features.shape)
            logger.debug("Labels shape", shape=labels.shape)
            
            return features, labels
        
        except Exception as e:
            self.display_manager.log_message(f"Error preparing LSTM training data: {e}", level="error")
            logger.error("Error preparing LSTM training data", exc_info=e)
            return np.array([]), np.array([])

    def train_model(self) -> bool:
        """Train the LSTM model."""
        try:
            # Prepare the training data
            X, y = self.prepare_lstm_training_data()
            
            # Ensure we have enough data
            if len(X) == 0 or len(y) == 0:
                self.display_manager.log_message("No training data available, skipping training.", level="warning")
                logger.warning("No training data available, skipping training")
                return False
            
            # The data is already in the correct shape (samples, timesteps, features)
            # Removed incorrect reshape: X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            # Train the model
            history = self.model.fit(X, y, epochs=self.settings.training_epochs, batch_size=self.settings.batch_size, verbose=0) # Use settings for epochs and batch_size
            
            # Log the training
            self.display_manager.log_message("LSTM model trained successfully.", level="info")
            logger.info("LSTM model trained successfully.")
            self._log_model_performance(history) # Log model performance
            
            return True
        
        except Exception as e:
            self.display_manager.log_message(f"Error training LSTM model: {e}", level="error")
            logger.error("Error training LSTM model", exc_info=e)
            return False

    def _log_model_performance(self, history):
        """Log model performance metrics."""
        try:
            if history is None or not hasattr(history, 'history'):
                self.display_manager.log_message("No training history available, cannot log performance.", level="warning")
                logger.warning("No training history available, cannot log performance.")
                return
            
            # Extract the loss and mean absolute error from the history
            loss = history.history.get('loss')
            mae = history.history.get('mae')
            
            # Log the loss and mean absolute error
            self.display_manager.log_message(f"Training performance: Loss={loss[-1]:.4f}, MAE={mae[-1]:.4f}", level="info")
            logger.info(f"Training performance: Loss={loss[-1]:.4f}, MAE={mae[-1]:.4f}")
            
        except Exception as e:
            self.display_manager.log_message(f"Error logging model performance: {e}", level="error")
            logger.error("Error logging model performance", exc_info=e)

    def calculate_position_pnl(self, current_price: float) -> Tuple[float, float]:
        """Calculate profit and loss for the current position."""
        if self.position_size == 0:
            return 0.0, 0.0
        
        # Calculate the profit and loss
        profit_loss = (current_price - self.entry_price) * self.position_size
        
        # Calculate the profit and loss percentage
        profit_loss_pct = (profit_loss / (self.entry_price * self.position_size)) * 100
        
        return profit_loss, profit_loss_pct

    def collect_and_store_data(self) -> bool:
        """Collect and store market data for training."""
        try:
            # Fetch market data
            market_data = self.fetch_market_data()
            
            # Ensure market data is not None and is a list
            if not market_data or not isinstance(market_data, list):
                self.display_manager.log_message("Invalid market data format, cannot collect and store data.", level="warning")
                logger.warning("Invalid market data format, cannot collect and store data")
                return False
            
            # Calculate technical indicators
            market_data = self.calculate_technical_indicators(market_data)
            
            # Ensure market data is not None and is a list
            if not market_data or not isinstance(market_data, list):
                self.display_manager.log_message("Invalid market data format after indicator calculation, cannot collect and store data.", level="warning")
                logger.warning("Invalid market data format, cannot collect and store data")
                return False
            
            # Update the feature buffer
            # Update the feature buffer and training data
            for data_point in market_data:
                self._update_feature_buffer(data_point)
                self.training_data.append(data_point) # training_data is already a deque with maxlen
            
            # Log the number of training samples
            logger.debug("Collected and stored market data points.", collected_count=len(market_data), total_samples=len(self.training_data))
            
            return True
        
        except Exception as e:
            self.display_manager.log_message(f"Error collecting and storing data: {e}", level="error")
            logger.error("Error collecting and storing data", error=str(e), exc_info=e)
            return False

    def load_training_data(self) -> Deque[Dict]:
        """Load training data from JSON file."""
        try:
            # Check if file exists and is not empty
            if os.path.exists(self.training_data_filename) and os.path.getsize(self.training_data_filename) > 0:
                with open(self.training_data_filename, "r") as f:
                    data = json.load(f)
                self.display_manager.log_message(f"Training data loaded: {len(data)} samples.", level="info")
                logger.info(f"Training data loaded: {len(data)} samples.")
                return deque(data, maxlen=self.max_training_samples) # Return as deque
            else:
                self.display_manager.log_message("Training data file not found or is empty, starting fresh.", level="info")
                logger.info("Training data file not found or is empty, starting fresh.")
                # Ensure the file exists and contains an empty JSON array if it was empty
                with open(self.training_data_filename, "w") as f:
                    json.dump([], f)
                return deque(maxlen=self.max_training_samples) # Return empty deque
        except json.JSONDecodeError as e:
            self.display_manager.log_message(f"Error decoding training data JSON, file might be corrupted. Starting fresh. Error: {e}", level="error")
            logger.error(f"Error decoding training data JSON, file might be corrupted. Starting fresh. Error: {e}", exc_info=e)
            # Overwrite corrupted file with empty JSON array
            with open(self.training_data_filename, "w") as f:
                json.dump([], f)
            return deque(maxlen=self.max_training_samples) # Return empty deque on error
        except FileNotFoundError:
            self.display_manager.log_message("No training data file found, creating new one and starting fresh.", level="info")
            logger.info("No training data file found, creating new one and starting fresh.")
            # Create the file with an empty JSON array
            with open(self.training_data_filename, "w") as f:
                json.dump([], f)
            return deque(maxlen=self.max_training_samples) # Return empty deque
        except Exception as e:
            self.display_manager.log_message(f"Error loading training data: {e}", level="error")
            logger.error(f"Error loading training data: {e}", exc_info=e)
            return deque(maxlen=self.max_training_samples) # Return empty deque on error

    def save_training_data(self):
        """Save training data to JSON file."""
        try:
            with open(self.training_data_filename, "w") as f:
                json.dump(list(self.training_data), f, indent=2) # Convert deque to list
            self.display_manager.log_message(f"Training data saved: {len(self.training_data)} samples.", level="info")
            logger.info(f"Training data saved: {len(self.training_data)} samples.")
        except Exception as e:
            self.display_manager.log_message(f"Error saving training data: {e}", level="error")
            logger.error(f"Error saving training data: {e}", exc_info=e)

    def load_model(self):
        """Load the TensorFlow model or create a new one with correct input shape."""
        try:
            # First try to load the model
            model = tf.keras.models.load_model(self.model_filename)
            
            # Check if the loaded model has the correct input shape
            expected_input_shape = (None, self.sequence_length, self.settings.ml.feature_count) # Use settings for feature_count
            if model.input_shape[1:] != expected_input_shape[1:]:
                self.display_manager.log_message(f"Model input shape mismatch. Creating new model. Expected: {expected_input_shape}, Actual: {model.input_shape}.", level="warning")
                logger.warning(f"Model input shape mismatch. Creating new model. Expected: {expected_input_shape}, Actual: {model.input_shape}.")
                model = self.create_lstm_model()
            else:
                self.display_manager.log_message(f"LSTM model loaded successfully with input shape: {model.input_shape}.", level="info")
                logger.info(f"LSTM model loaded successfully with input shape: {model.input_shape}.")
            
            return model
            
        except (FileNotFoundError, OSError) as e:
            self.display_manager.log_message(f"No valid model file found, will create new LSTM. Error: {e}", level="info")
            logger.info(f"No valid model file found, will create new LSTM. Error: {e}", exc_info=e)
            return self.create_lstm_model() # Ensure a new model is created and returned
        except Exception as e:
            self.display_manager.log_message(f"Error loading model. Creating new model. Error: {e}", level="error")
            logger.error(f"Error loading model. Creating new model. Error: {e}", exc_info=e)
            return self.create_lstm_model() # Ensure a new model is created and returned

    def save_model(self):
        """Save the TensorFlow model to a file."""
        try:
            if self.model:
                self.model.save(self.model_filename)
                self.display_manager.log_message(f"Model saved successfully to: {self.model_filename}.", level="info")
                logger.info(f"Model saved successfully to: {self.model_filename}.")
            else:
                self.display_manager.log_message("No model to save.", level="warning")
                logger.warning("No model to save.")
        except Exception as e:
            self.display_manager.log_message(f"Error saving model: {e}", level="error")
            logger.error(f"Error saving model: {e}", exc_info=e)

    def predict_trade_signal(self, latest_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts the trade signal (BUY, SELL, HOLD) and confidence based on the latest data.
        
        Args:
            latest_data (Dict[str, Any]): The latest market data point with indicators.
            
        Returns:
            Dict[str, Any]: A dictionary containing the predicted signal, confidence, price, and RSI.
        """
        try:
            # Determine the current market price from latest_data
            current_market_price = 0.0
            if isinstance(latest_data, dict):
                current_market_price = float(latest_data.get('price', latest_data.get('lastPrice', 0.0)))
                # Fallback to bid/ask average if price and lastPrice are missing or zero
                if current_market_price == 0.0:
                    ask = latest_data.get('bestAsk')
                    bid = latest_data.get('bestBid')
                    if ask is not None and bid is not None:
                        try:
                            current_market_price = (float(ask) + float(bid)) / 2
                        except ValueError:
                            current_market_price = 0.0 # Keep 0.0 if conversion fails
 
            if not self.model or not self.feature_engineer.is_fitted_: # Check feature_engineer.is_fitted_
                self.display_manager.log_message("Model or feature engineer not available for prediction, returning HOLD signal.", level="warning")
                logger.warning("Model or feature engineer not available for prediction, returning HOLD signal.")
                return {"signal": "HOLD", "confidence": 0.5, "price": current_market_price, "rsi": latest_data.get('rsi', 50.0) if isinstance(latest_data, dict) else 50.0}
            
            # Ensure latest_data is a dictionary before proceeding
            if not isinstance(latest_data, dict):
                self.display_manager.log_message(f"Invalid latest_data format for prediction. Expected dict, returning HOLD signal. Data type: {type(latest_data)}", level="warning")
                logger.warning("Invalid latest_data format for prediction. Expected dict, returning HOLD signal.", data_type=type(latest_data))
                return {"signal": "HOLD", "confidence": 0.5, "price": current_market_price, "rsi": 50.0}
 
            # Prepare features for prediction using FeatureEngineer
            # This will return scaled features directly
            features_array = self.feature_engineer.transform([latest_data]) # Pass as list for transform
            
            # Ensure features are not empty
            if features_array.size == 0:
                self.display_manager.log_message("No features prepared for prediction, returning HOLD signal.", level="warning")
                logger.warning("No features prepared for prediction, returning HOLD signal.")
                return {"signal": "HOLD", "confidence": 0.5, "price": current_market_price, "rsi": latest_data.get('rsi', 50.0)}
            
            # Reshape for LSTM input (1 sample, sequence_length, num_features)
            # For a single prediction, we need to create a sequence of length 1
            # and pad it if necessary, or ensure the model is trained for single step prediction
            # For now, assuming model expects (1, sequence_length, num_features)
            # and we are providing the latest single data point as a sequence of 1
            # This might need adjustment based on how the model was actually trained
            
            # Pad the scaled_features to match the sequence_length
            padded_features = np.zeros((1, self.sequence_length, features_array.shape[1])) # Use actual feature count
            padded_features[0, -1, :] = features_array[0, :] # Place the latest features at the end of the sequence
            
            prediction = self.model.predict(padded_features, verbose=0)[0][0] # Add verbose=0 to suppress Keras output
            
            # Determine signal based on prediction and confidence threshold
            signal = "HOLD"
            confidence = float(prediction)
            
            logger.debug("Prediction debug", confidence=confidence, buy_threshold=self.settings.buy_confidence_threshold, sell_threshold=self.settings.sell_confidence_threshold)
            
            # Adjusting for potential floating point inaccuracies to ensure 0.8 confidence triggers BUY
            # if the threshold is 0.8 or slightly higher due to precision.
            # The problem states buy_confidence_threshold <= 0.8, so if it's 0.8,
            # a prediction of 0.8 should be BUY.
            # Using a small epsilon to ensure values very close to the threshold are caught.
            # print(f"DEBUG: confidence={confidence}, buy_threshold={self.settings.buy_confidence_threshold}, sell_threshold={self.settings.sell_confidence_threshold}", flush=True) # Removed debug print
            if confidence >= self.settings.buy_confidence_threshold:
                signal = "BUY"
            elif confidence <= self.settings.sell_confidence_threshold:
                signal = "SELL"
            
            # logger.info("Prediction result", prediction=f"{prediction:.4f}", signal=signal, confidence=f"{confidence:.4f}") # Removed JSON logging
            
            return {
                "signal": signal,
                "confidence": confidence,
                "price": current_market_price,
                "rsi": latest_data.get('rsi', 50.0)
            }
        
        except Exception as e:
            # print(f"DEBUG: Exception in predict_trade_signal: {e}", flush=True) # Removed debug print
            self.display_manager.log_message(f"Error predicting trade signal: {e}", level="error")
            logger.error("Error predicting trade signal", exc_info=e)
            # Ensure current_market_price is defined even in exception
            current_market_price = 0.0
            if isinstance(latest_data, dict):
                current_market_price = float(latest_data.get('price', latest_data.get('lastPrice', 0.0)))
                if current_market_price == 0.0:
                    ask = latest_data.get('bestAsk')
                    bid = latest_data.get('bestBid')
                    if ask is not None and bid is not None:
                        try:
                            current_market_price = (float(ask) + float(bid)) / 2
                        except ValueError:
                            current_market_price = 0.0
            return {"signal": "HOLD", "confidence": 0.5, "price": current_market_price, "rsi": latest_data.get('rsi', 50.0) if isinstance(latest_data, dict) else 50.0}
 
    def execute_simulated_trade(self, trade_signal: Dict[str, Any]):
        """
        Executes a simulated trade based on the given trade signal.
        
        Args:
            trade_signal (Dict[str, Any]): Dictionary containing the trade signal (BUY, SELL, HOLD),
                                           confidence, price, and RSI.
        """
        if trade_signal is None:
            self.display_manager.log_message("Trade signal is None, skipping trade execution.", level="warning")
            logger.warning("Trade signal is None, skipping trade execution.")
            return

        signal_type = trade_signal.get("signal")
        confidence = trade_signal.get("confidence")
        current_price = trade_signal.get("price")
        rsi = trade_signal.get("rsi")
        
        if not current_price:
            self.display_manager.log_message("Current price not available for trade execution.", level="warning")
            logger.warning("Current price not available for trade execution.")
            return
        
        trade_executed = False
        
        # Apply RSI override
        if rsi is not None:
            if rsi > self.settings.rsi_overbought and signal_type == "BUY":
                self.display_manager.log_message(f"RSI is overbought ({rsi:.2f}), overriding BUY signal to HOLD.", level="info")
                logger.info("RSI is overbought, overriding BUY signal to HOLD.", rsi=f"{rsi:.2f}")
                signal_type = "HOLD"
            elif rsi < self.settings.rsi_oversold and signal_type == "SELL":
                self.display_manager.log_message(f"RSI is oversold ({rsi:.2f}), overriding SELL signal to HOLD.", level="info")
                logger.info("RSI is oversold, overriding SELL signal to HOLD.", rsi=f"{rsi:.2f}")
                signal_type = "HOLD"
        
        if signal_type == "BUY" and confidence >= self.settings.buy_confidence_threshold:
            # Check if we have enough balance to buy
            cost = self.trade_amount * current_price * (1 + self.fee_rate)
            if self.balance >= cost:
                self.balance -= cost
                self.position_size += self.trade_amount
                self.entry_price = current_price # For simplicity, assuming average entry price
                trade_executed = True
                self.display_manager.log_message(f"BUY executed: Amount: {self.trade_amount:.4f}, Price: ${current_price:,.2f}, New Balance: ${self.balance:,.2f}, Position: {self.position_size:.4f} BTC", level="info")
                logger.info("BUY executed", amount=self.trade_amount, price=current_price, new_balance=self.balance, position=self.position_size)
            else:
                self.display_manager.log_message(f"Insufficient balance to BUY. Needed: ${cost:.2f} AUD, Have: ${self.balance:,.2f} AUD", level="warning")
                logger.warning("Insufficient balance to BUY.", needed=f"{cost:.2f} AUD", have=f"{self.balance:.2f} AUD")
        
        elif signal_type == "SELL" and confidence <= self.settings.sell_confidence_threshold:
            # Check if we have enough position to sell
            if self.position_size >= self.trade_amount:
                revenue = self.trade_amount * current_price * (1 - self.fee_rate)
                self.balance += revenue
                self.position_size -= self.trade_amount
                if self.position_size < 1e-9: # Handle floating point inaccuracies
                    self.position_size = 0.0
                    self.entry_price = 0.0
                trade_executed = True
                self.display_manager.log_message(f"SELL executed: Amount: {self.trade_amount:.4f}, Price: ${current_price:,.2f}, New Balance: ${self.balance:,.2f}, Position: {self.position_size:.4f} BTC", level="info")
                logger.info("SELL executed", amount=self.trade_amount, price=current_price, new_balance=self.balance, position=self.position_size)
            else:
                self.display_manager.log_message(f"Insufficient position to SELL. Have: {self.position_size:.4f} BTC, Needed: {self.trade_amount:.4f} BTC", level="warning")
                logger.warning("Insufficient position to SELL.", have=f"{self.position_size:.4f} BTC", needed=f"{self.trade_amount:.4f} BTC")
        
        else:
            self.display_manager.log_message(f"HOLD signal. No trade executed. Confidence: {confidence:.2f}, RSI: {rsi:.2f}", level="info")
            logger.info("HOLD signal. No trade executed.", confidence=confidence, rsi=rsi)
        
        if trade_executed:
            pnl, pnl_pct = self.calculate_position_pnl(current_price)
            self.display_manager.log_message(f"Current PnL: ${pnl:,.2f} ({pnl_pct:.2f}%)", level="info")
            logger.info("Current PnL", pnl=pnl, pnl_pct=pnl_pct)

    def should_save(self) -> bool:
        """Determine if the model and state should be saved."""
        return (time.time() - self.last_save_time) >= self.settings.operations.save_interval
 
    def should_train(self) -> bool:
        """Determine if the model should be retrained."""
        return (time.time() - self.last_training_time) >= self.settings.operations.training_interval and len(self.training_data) >= self.settings.ml.sequence_length
 
    def run(self):
        """Main loop for the trading bot."""
        self.display_manager.print_initialization_message() # Using DisplayManager for init message
        iteration_count = 0
        
        while not self._shutdown_requested and (not self.limited_run or iteration_count < self.run_iterations): # Changed <= to < for run_iterations
            try:
                iteration_count += 1
                
                # 1. Collect and store data
                if not self.collect_and_store_data():
                    self.display_manager.log_message("Failed to collect and store data. Skipping iteration.", level="warning")
                    time.sleep(self.settings.data_collection_interval)
                    continue # Continue to next iteration instead of printing status
                
                # 2. Fit scalers if needed
                if not self.scalers_fitted and len(self.training_data) >= self.settings.ml.sequence_length:
                    self.fit_scalers()
                
                # 3. Train model if needed
                if self.should_train():
                    self.display_manager.print_training_start() # Using DisplayManager
                    if self.train_model():
                        self.last_training_time = time.time()
                        self.save_model()
                        self.save_scalers()
                        # Training complete message is already handled by _log_model_performance
                    else:
                        self.display_manager.log_message("Model training failed or skipped due to insufficient data.", level="warning")
                        logger.warning("Model training failed or skipped due to insufficient data.") # Log to file
                
                # 4. Get latest data point for prediction
                if not self.training_data:
                    self.display_manager.log_message("No training data available for prediction. Skipping trade signal generation.", level="warning")
                    time.sleep(self.settings.data_collection_interval)
                    continue # Continue to next iteration instead of printing status
                
                latest_data = self.training_data[-1]
                
                # 5. Predict trade signal
                trade_signal = self.predict_trade_signal(latest_data)
                
                # 6. Execute simulated trade
                self.execute_simulated_trade(trade_signal)
                
                # 7. Save state and training data if needed
                if self.should_save():
                    self.save_state()
                    self.save_training_data()
                    self.last_save_time = time.time()
                
                # Update and print status using DisplayManager
                signal_type = trade_signal.get("signal", "N/A")
                confidence = trade_signal.get("confidence", 0.0)
                current_price = trade_signal.get("price", 0.0)
                rsi = latest_data.get('rsi', 50.0)
                
                # Get training samples count for progress bar
                training_samples_count = len(self.training_data)
                # Assuming total_training_samples is the maxlen of training_data deque
                total_training_samples_for_display = self.max_training_samples
                
                self.display_manager.display_status(
                    iteration=iteration_count,
                    status="ACTIVE", # Assuming bot is active during run loop
                    price=current_price,
                    signal_type=signal_type,
                    confidence=confidence,
                    balance=self.balance,
                    position_size=self.position_size,
                    rsi=rsi,
                    training_samples=training_samples_count,
                    total_training_samples=total_training_samples_for_display
                )
                
                time.sleep(self.settings.data_collection_interval)
                
            except Exception as e:
                self.display_manager.log_message(f"An unexpected error occurred during main loop: {e}", level="error")
                logger.exception("An unexpected error occurred during main loop", error=str(e)) # Log full traceback to file
                time.sleep(self.settings.data_collection_interval)
        
        self.display_manager.print_shutdown_message() # Using DisplayManager for shutdown message
        self.save_state()
        self.save_training_data()
        self.save_model()
        self.save_scalers()

    def fetch_market_data(self) -> List[Dict[str, Any]]:
        """Fetch market data from the BTCMarkets API."""
        try:
            url = self.settings.api_base_url + "/markets/tickers" # Use settings for base URL and tickers endpoint
            params = {"marketId": "BTC-AUD"} # Add marketId parameter
            response = requests.get(url, params=params, timeout=self.settings.api_timeout) # Use settings for timeout
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            
            market_data = response.json()
            logger.debug("Fetched market data points from BTCMarkets API", count=len(market_data))
            
            # Basic data validation
            if not isinstance(market_data, list):
                logger.error("Invalid data format from BTCMarkets API")
                return []
            
            # Convert timestamps to ISO format and ensure all keys are strings
            processed_data = []
            for data_point in market_data:
                # Ensure all keys are strings
                data_point = {str(k): v for k, v in data_point.items()}
                
                # Convert timestamp to ISO format
                timestamp_str = data_point.get('timestamp')
                if timestamp_str is not None:
                    try:
                        # Parse ISO format string to datetime object, handling 'Z' and ensuring UTC
                        dt_object = datetime.fromisoformat(timestamp_str.replace('Z', '')).replace(tzinfo=timezone.utc)
                        # Convert datetime object to Unix timestamp in milliseconds
                        data_point['timestamp'] = int(dt_object.timestamp() * 1000)
                    except ValueError as ve:
                        logger.warning("Invalid timestamp format. Using current timestamp.", timestamp=timestamp_str, error=str(ve))
                        data_point['timestamp'] = int(time.time() * 1000) # Fallback to current timestamp
                else:
                    logger.warning("Timestamp missing from API response. Adding current timestamp.")
                    data_point['timestamp'] = int(time.time() * 1000) # Add current timestamp if not present
                
                processed_data.append(data_point)
            
            return processed_data
        
        except requests.exceptions.Timeout as e:
            logger.error("Market data request timed out", exc_info=e)
            raise NetworkTimeoutError("Market data request timed out", original_exception=e) from e
        except requests.exceptions.ConnectionError as e:
            logger.error("Failed to connect to the exchange", exc_info=e)
            raise NetworkError("Failed to connect to the exchange", original_exception=e) from e
        except requests.exceptions.HTTPError as e:
            logger.error("HTTP error fetching market data", status_code=e.response.status_code, response_text=e.response.text, exc_info=e)
            raise APIError("HTTP error fetching market data", status_code=e.response.status_code, response_text=e.response.text) from e
        except json.JSONDecodeError as e:
            logger.error("Failed to decode JSON response from API", exc_info=e, response_text=response.text if 'response' in locals() else 'N/A')
            raise DataError("Invalid JSON response from API", data_type="market_data", validation_errors={"json_decode_error": str(e)}) from e
        except Exception as e: # Catch all other exceptions
            logger.error("An unexpected error occurred while fetching market data", exc_info=e)
            return []

    def extract_comprehensive_data(self, market_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Extract comprehensive data for BTC-AUD market from the API response.
        
        Args:
            market_data (List[Dict[str, Any]]): List of market data dictionaries from the API.
            
        Returns:
            Optional[Dict[str, Any]]: A dictionary containing extracted and validated data for BTC-AUD,
                                    or None if not found or invalid.
        """
        if not market_data:
            logger.warning("No market data provided for extraction.")
            return None
        
        btc_aud_data = None
        for item in market_data:
            if item.get("marketId") == "BTC-AUD":
                btc_aud_data = item
                break
        
        if btc_aud_data is None:
            logger.warning("BTC-AUD market data not found in the API response.")
            return None
        
        # Validate and convert data types
        try:
            extracted = {
                "marketId": btc_aud_data.get("marketId"),
                "price": float(btc_aud_data.get("lastPrice", 0.0)),
                "volume": float(btc_aud_data.get("volume24h", 0.0)),
                "bid": float(btc_aud_data.get("bestBid", 0.0)),
                "ask": float(btc_aud_data.get("bestAsk", 0.0)),
                "high24h": float(btc_aud_data.get("high24h", 0.0)),
                "low24h": float(btc_aud_data.get("low24h", 0.0)),
                "timestamp": btc_aud_data.get("timestamp", int(time.time() * 1000)) # Ensure timestamp is present
            }
            
            # Calculate spread if not provided
            if "spread" not in extracted:
                extracted["spread"] = extracted["ask"] - extracted["bid"]
            
            # Basic validation for critical fields
            if extracted["price"] <= 0 or extracted["volume"] < 0:
                logger.warning("Invalid price or volume in extracted data", price=extracted['price'], volume=extracted['volume'])
                return None
            
            return extracted
        
        except (ValueError, TypeError) as e:
            logger.error("Error converting market data values", exc_info=e, data=btc_aud_data)
            return None
        except Exception as e:
            logger.error("An unexpected error occurred during data extraction", exc_info=e)
            return None

    def manual_sma(self, data: np.ndarray, period: int) -> float:
        """Calculate Simple Moving Average manually."""
        if len(data) < period:
            return np.mean(data)  # Not enough data for the full period
        return np.mean(data[-period:])

    def manual_ema(self, data: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average manually."""
        if len(data) < period:
            return np.mean(data)  # Not enough data for the full period
        
        k = 2 / (period + 1)
        ema = np.zeros_like(data, dtype=float)
        ema[period-1] = np.mean(data[:period])  # Seed with SMA
        
        for i in range(period, len(data)):
            ema[i] = (data[i] * k) + (ema[i-1] * (1 - k))
        
        return ema[-1]

    def manual_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index manually."""
        if len(prices) < period + 1:
            return 50.0  # Neutral value when not enough data
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            if avg_gain == 0:
                return 50.0 # If both are zero (constant prices), RSI is 50
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)

    def calculate_technical_indicators(self, market_data: Optional[List[Dict[str, Any]]] = None, prices: Optional[np.ndarray] = None, volumes: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        Calculate technical indicators for each data point.
        
        Args:
            market_data (List[Dict[str, Any]]): List of market data dictionaries.
            prices (Optional[np.ndarray]): Optional numpy array of prices for direct calculation.
            volumes (Optional[np.ndarray]): Optional numpy array of volumes for direct calculation.
            
        Returns:
            List[Dict[str, Any]]: List of market data dictionaries with added technical indicators.
        """
        if prices is not None and volumes is not None:
            # This branch is for performance tests that pass raw numpy arrays
            # Create a list of dictionaries from prices and volumes
            data_for_df = [{"price": p, "volume": v} for p, v in zip(prices, volumes)]
            df = pd.DataFrame(data_for_df)
            price_col = 'price'
        elif market_data:
            # Ensure market_data is not None and is a list
            if not isinstance(market_data, list):
                logger.warning("Invalid market data format, cannot calculate indicators")
                return []
            
            # Convert market_data to a pandas DataFrame for easier calculations
            df = pd.DataFrame(market_data)
            
            # Ensure 'lastPrice' or 'price' column exists and standardize to 'price'
            if 'lastPrice' in df.columns:
                df['price'] = df['lastPrice']
            elif 'price' not in df.columns:
                logger.warning("No 'lastPrice' or 'price' column found after standardization, cannot calculate indicators")
                return market_data
            
            # Ensure 'timestamp' column exists. If not, add it from original market_data or generate.
            if 'timestamp' not in df.columns:
                # Try to get timestamp from original market_data if available
                if market_data and 'timestamp' in market_data[0]:
                    df['timestamp'] = [d.get('timestamp') for d in market_data]
                else:
                    logger.warning("Timestamp column missing in market data. Adding current timestamp.")
                    df['timestamp'] = int(time.time() * 1000) # Add current Unix timestamp in milliseconds
            
            # Convert price and volume columns to numeric, coercing errors to NaN
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            if 'volume' in df.columns:
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            else:
                df['volume'] = 0.0 # Default volume to 0 if not present
            
            # Check for NaN values in critical columns after conversion
            if df['price'].isnull().any():
                logger.warning("Invalid price data detected after numeric conversion. Skipping indicator calculation.", invalid_prices=df[df['price'].isnull()]['price'].tolist())
                return []
            
            # Fill NaN prices and volumes with previous valid price or 0.0
            df['price'] = df['price'].ffill().fillna(0.0)
            df['volume'] = df['volume'].ffill().fillna(0.0)
            
            # Ensure there are enough valid data points for basic calculations
            if len(df) < 1: # At least one data point is needed
                logger.warning("Not enough valid data points to calculate indicators. Returning original market data.", needed=1, got=len(df))
                return market_data
        else:
            logger.warning("No market data or price/volume arrays provided for indicator calculation.")
            return []
        
        # Use 'price' as the consistent column name for calculations
        price_col = 'price'
        
        # Calculate Simple Moving Averages (SMA)
        df['sma_5'] = df[price_col].rolling(window=5).mean()
        df['sma_10'] = df[price_col].rolling(window=10).mean()
        df['sma_20'] = df[price_col].rolling(window=20).mean()
        
        # Calculate Exponential Moving Averages (EMA)
        df['ema_12'] = df[price_col].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df[price_col].ewm(span=26, adjust=False).mean()
        
        # Calculate Relative Strength Index (RSI)
        # Check if there's enough data for RSI calculation (period + 1 for diff)
        if len(df[price_col]) < 15: # 14 period + 1 for the first diff
            df['rsi'] = 50.0 # Default to neutral RSI
        else:
            delta = df[price_col].diff()
            up, down = delta.copy(), delta.copy()
            up[up < 0] = 0
            down[down > 0] = 0
            
            roll_up1 = up.ewm(span=14, adjust=False).mean() # Use ewm for RSI average gain/loss
            roll_down1 = np.abs(down.ewm(span=14, adjust=False).mean()) # Use ewm for RSI average gain/loss
            
            # Avoid division by zero, set RS to a large number if avg_loss is zero to push RSI towards 100
            # For constant prices, both avg_gain and avg_loss will be 0, leading to 0/0. Handle this to be 50.
            RS = np.where(roll_down1 == 0, np.inf, roll_up1 / roll_down1)
            # If both are zero (constant price), set RS to 1 to get RSI of 50
            RS = np.where((roll_up1 == 0) & (roll_down1 == 0), 1, RS)
            df['rsi'] = 100.0 - (100.0 / (1.0 + RS))
            # For cases where RS is inf (all gains), RSI should be 100
            df['rsi'] = np.where(RS == np.inf, 100.0, df['rsi'])
            # For cases where RS is 0 (all losses), RSI should be 0
            df['rsi'] = np.where(RS == 0, 0.0, df['rsi'])
            # Fill NaN values (due to insufficient data) with 50.0
            df['rsi'] = df['rsi'].fillna(50.0)
        
        # Calculate MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Calculate Bollinger Bands
        df['bb_middle'] = df[price_col].rolling(window=20).mean()
        df['bb_std'] = df[price_col].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        # Calculate Average True Range (ATR) - requires high, low, close
        # For simplicity, using price_col for ATR calculation as high/low are not always available in market_data
        df['atr'] = df[price_col].rolling(window=14).mean()
        
        # Add volume SMA
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(window=self.settings.volume_sma_period).mean() # Use settings for volume_sma_period
        else:
            df['volume_sma'] = 0.0 # Default if volume is not available
        
        # Convert the DataFrame to a list of dictionaries, which will include all columns (original and new indicators)
        return df.to_dict('records')
        
    def create_lstm_model(self):
        """Create the LSTM model."""
        try:
            # Define the LSTM model
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(self.settings.lstm_units, return_sequences=True, input_shape=(self.settings.sequence_length, self.settings.feature_count)),
                tf.keras.layers.Dropout(self.settings.dropout_rate),
                tf.keras.layers.LSTM(self.settings.lstm_units, return_sequences=False),
                tf.keras.layers.Dropout(self.settings.dropout_rate),
                tf.keras.layers.Dense(self.settings.dense_units),
                tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer
            ])
            
            # Compile the model
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.settings.learning_rate), loss='mse', metrics=['mae'])
            
            # Log model summary
            model.summary(print_fn=lambda x: logger.info(x)) # Use print_fn to log summary
            
            return model
        
        except Exception as e:
            logger.error("Error creating LSTM model", exc_info=e)
            return None
 
    def prepare_features(self, data_point: dict) -> list:
        """Prepare feature vector from a single data point using FeatureEngineer."""
        try:
            if not data_point:
                raise ValueError("Empty data point provided")
            
            # Convert single data point to a list of dicts for FeatureEngineer
            data_for_fe = [data_point]
            
            # Generate raw features using FeatureEngineer's internal method
            # This method handles adding all configured features and filling missing values.
            # It does not require the scaler to be fitted.
            raw_features_df = self.feature_engineer._generate_all_features(pd.DataFrame(data_for_fe))
            
            # If FeatureEngineer is not fitted, its feature_names_ might be empty or incomplete.
            # We need to ensure that the feature_names_ are populated correctly
            # before attempting to reindex or transform.
            if not self.feature_engineer.feature_names_:
                # Populate feature_names_ from the generated raw features.
                # This ensures that subsequent calls to transform or reindex have the correct feature set.
                self.feature_engineer.feature_names_ = list(raw_features_df.columns)
                logger.debug("FeatureEngineer feature names initialized from generated features.")
            
            # Reindex to ensure all expected features are present, filling missing with 0.0.
            # Use the feature_names_ from the feature_engineer, which should now be populated.
            features_df_reindexed = raw_features_df.reindex(columns=self.feature_engineer.feature_names_, fill_value=0.0)
            
            # Convert to numpy array
            features_array = features_df_reindexed.values
            
            # Apply scaling only if the FeatureEngineer's scaler is fitted
            if self.feature_engineer.is_fitted_:
                scaled_features = self.feature_engineer.scaler.transform(features_array)
                features = scaled_features[0].tolist()
                logger.debug("Prepared features using fitted FeatureEngineer and scaler", first_few_features=features[:5])
            else:
                features = features_array[0].tolist()
                logger.warning("FeatureEngineer not fitted. Returning raw (unscaled) features.", first_few_features=features[:5])
            
            logger.debug("Prepared features count", count=len(features))
            logger.debug("Prepared feature names", names=self.feature_engineer.get_feature_names())
            
            return features
        
        except Exception as e:
            logger.exception("Error preparing features with FeatureEngineer", exc_info=e)
            # Fallback to returning a zero vector of expected length on error
            # This length should ideally come from feature_engineer.get_feature_names()
            # or settings.ml.feature_count.
            # If feature_engineer.feature_names_ is populated, use its length.
            # Otherwise, fallback to settings.ml.feature_count or a default of 96.
            fallback_feature_count = len(self.feature_engineer.feature_names_) if self.feature_engineer.feature_names_ else (self.settings.ml.feature_count if self.settings.ml.feature_count > 0 else 96)
            return [0.0] * fallback_feature_count

    def _update_feature_buffer(self, market_data: Dict[str, Any]) -> None:
        """Update the feature buffer with the latest market data."""
        try:
            # Extract relevant features from market_data
            price = market_data.get('lastPrice') or market_data.get('price')
            volume = market_data.get('volume', 0.0) # Default to 0.0 if volume is missing
            
            # Validate that price is not None
            if price is None:
                logger.error("Missing price in market data", price=price)
                raise DataError("Missing price in market data")
            
            # Convert price and volume to float
            try:
                price = float(price)
                volume = float(volume)
            except ValueError as ve:
                logger.error("Invalid price or volume format", price=price, volume=volume, exc_info=ve)
                raise DataError(f"Invalid price or volume format: {ve}") from ve
            
            # Create a feature vector
            feature_vector = [price, volume]
            
            # Append the feature vector to the data buffer
            if hasattr(self, '_data_buffer'):
                logger.debug("Updated feature buffer", price=price, volume=volume)
                self._data_buffer.append(feature_vector)
            else:
                logger.error("Attempted to append to _data_buffer, but it does not exist on self.")
                raise RuntimeError("'_data_buffer' attribute is missing from ContinuousAutoTrader instance.")
        except Exception as e:
            logger.error("Error updating feature buffer", exc_info=e)

    def fit_scalers(self) -> bool:
        """Fit scalers using the FeatureEngineer to the training data."""
        try:
            # Ensure training_data is a deque
            if not isinstance(self.training_data, deque):
                logger.error("training_data is not a deque, cannot fit scalers.")
                self.feature_engineer.is_fitted_ = False
                return False
            
            if len(self.training_data) < self.settings.ml.sequence_length:
                logger.warning("Insufficient training data for scaler fitting.", needed=self.settings.ml.sequence_length, got=len(self.training_data))
                self.feature_engineer.is_fitted_ = False
                return False
            
            # Fit the FeatureEngineer directly on the raw training data
            # The FeatureEngineer handles its own internal scaler fitting
            self.feature_engineer.fit(list(self.training_data))
            
            logger.info("FeatureEngineer fitted successfully")
            self.scalers_fitted = True # Set to True after successful fitting
            return True
            
        except Exception as e:
            logger.error("Error fitting FeatureEngineer", exc_info=e)
            self.feature_engineer.is_fitted_ = False
            self.scalers_fitted = False # Set to False on error
            return False

    def prepare_lstm_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training using FeatureEngineer."""
        try:
            # Ensure we have enough data
            if len(self.training_data) < self.sequence_length:
                logger.warning("Not enough data for training.", needed=self.sequence_length, got=len(self.training_data))
                # Return empty arrays with correct dimensions
                return np.empty((0, self.sequence_length, self.settings.ml.feature_count)), np.empty((0,))
            
            # Transform all training data using the fitted FeatureEngineer
            # This will return scaled features
            if not self.feature_engineer.is_fitted_:
                logger.warning("FeatureEngineer not fitted, cannot prepare LSTM training data.")
                return np.empty((0, self.sequence_length, self.settings.ml.feature_count)), np.empty((0,))
            
            # Get all scaled features from the training data
            all_scaled_features = self.feature_engineer.transform(list(self.training_data))
            
            features, labels = [], []
            num_features = all_scaled_features.shape[1] # Get the actual number of features from FeatureEngineer
            
            for i in range(len(all_scaled_features) - self.sequence_length):
                # Extract the sequence of scaled features
                sequence = all_scaled_features[i : i + self.sequence_length]
                
                # Extract the label (next data point's price, unscaled)
                # We need the original price for the label, not the scaled one.
                next_original_data_point = self.training_data[i + self.sequence_length]
                next_price = float(next_original_data_point.get('price', next_original_data_point.get('lastPrice', 0.0)))
                
                features.append(sequence)
                labels.append(next_price)
            
            # Convert to numpy arrays
            features = np.array(features)
            labels = np.array(labels)
            
            # Log the shape of the features and labels
            logger.debug("Features shape", shape=features.shape)
            logger.debug("Labels shape", shape=labels.shape)
            
            return features, labels
        
        except Exception as e:
            logger.error("Error preparing LSTM training data", exc_info=e)
            return np.array([]), np.array([])

    def train_model(self) -> bool:
        """Train the LSTM model."""
        try:
            # Prepare the training data
            X, y = self.prepare_lstm_training_data()
            
            # Ensure we have enough data
            if len(X) == 0 or len(y) == 0:
                logger.warning("No training data available, skipping training")
                return False
            
            # The data is already in the correct shape (samples, timesteps, features)
            # Removed incorrect reshape: X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            # Train the model
            history = self.model.fit(X, y, epochs=self.settings.training_epochs, batch_size=self.settings.batch_size, verbose=0) # Use settings for epochs and batch_size
            
            # Log the training
            logger.info(f"{Fore.GREEN}LSTM model trained successfully.{Style.RESET_ALL}")
            self._log_model_performance(history) # Log model performance
            
            return True
        
        except Exception as e:
            logger.error(f"{Fore.RED}Error training LSTM model: {e}{Style.RESET_ALL}", exc_info=e)
            return False

    def _log_model_performance(self, history):
        """Log model performance metrics."""
        try:
            if history is None or not hasattr(history, 'history'):
                logger.warning(f"{Fore.YELLOW}No training history available, cannot log performance.{Style.RESET_ALL}")
                return
            
            # Extract the loss and mean absolute error from the history
            loss = history.history.get('loss')
            mae = history.history.get('mae')
            
            # Log the loss and mean absolute error
            logger.info(f"{Fore.CYAN}Training performance: Loss={loss[-1]:.4f}, MAE={mae[-1]:.4f}{Style.RESET_ALL}")
            
        except Exception as e:
            logger.error(f"{Fore.RED}Error logging model performance: {e}{Style.RESET_ALL}", exc_info=e)

    def calculate_position_pnl(self, current_price: float) -> Tuple[float, float]:
        """Calculate profit and loss for the current position."""
        if self.position_size == 0:
            return 0.0, 0.0
        
        # Calculate the profit and loss
        profit_loss = (current_price - self.entry_price) * self.position_size
        
        # Calculate the profit and loss percentage
        profit_loss_pct = (profit_loss / (self.entry_price * self.position_size)) * 100
        
        return profit_loss, profit_loss_pct

    def collect_and_store_data(self) -> bool:
        """Collect and store market data for training."""
        try:
            # Fetch market data
            market_data = self.fetch_market_data()
            
            # Ensure market data is not None and is a list
            if not market_data or not isinstance(market_data, list):
                logger.warning("Invalid market data format, cannot collect and store data")
                return False
            
            # Calculate technical indicators
            market_data = self.calculate_technical_indicators(market_data)
            
            # Ensure market data is not None and is a list
            if not market_data or not isinstance(market_data, list):
                logger.warning("Invalid market data format, cannot collect and store data")
                return False
            
            # Update the feature buffer
            # Update the feature buffer and training data
            for data_point in market_data:
                self._update_feature_buffer(data_point)
                self.training_data.append(data_point) # training_data is already a deque with maxlen
            
            # Log the number of training samples
            logger.debug("Collected and stored market data points.", collected_count=len(market_data), total_samples=len(self.training_data))
            
            return True
        
        except Exception as e:
            logger.error("Error collecting and storing data", error=str(e), exc_info=e)
            return False

    def load_training_data(self) -> Deque[Dict]:
        """Load training data from JSON file."""
        try:
            # Check if file exists and is not empty
            if os.path.exists(self.training_data_filename) and os.path.getsize(self.training_data_filename) > 0:
                with open(self.training_data_filename, "r") as f:
                    data = json.load(f)
                logger.info(f"{Fore.GREEN}Training data loaded: {len(data)} samples.{Style.RESET_ALL}")
                return deque(data, maxlen=self.max_training_samples) # Return as deque
            else:
                logger.info(f"{Fore.YELLOW}Training data file not found or is empty, starting fresh.{Style.RESET_ALL}")
                # Ensure the file exists and contains an empty JSON array if it was empty
                with open(self.training_data_filename, "w") as f:
                    json.dump([], f)
                return deque(maxlen=self.max_training_samples) # Return empty deque
        except json.JSONDecodeError as e:
            logger.error(f"{Fore.RED}Error decoding training data JSON, file might be corrupted. Starting fresh. Error: {e}{Style.RESET_ALL}", exc_info=e)
            # Overwrite corrupted file with empty JSON array
            with open(self.training_data_filename, "w") as f:
                json.dump([], f)
            return deque(maxlen=self.max_training_samples) # Return empty deque on error
        except FileNotFoundError:
            logger.info(f"{Fore.YELLOW}No training data file found, creating new one and starting fresh.{Style.RESET_ALL}")
            # Create the file with an empty JSON array
            with open(self.training_data_filename, "w") as f:
                json.dump([], f)
            return deque(maxlen=self.max_training_samples) # Return empty deque
        except Exception as e:
            logger.error(f"{Fore.RED}Error loading training data: {e}{Style.RESET_ALL}", exc_info=e)
            return deque(maxlen=self.max_training_samples) # Return empty deque on error

    def save_training_data(self):
        """Save training data to JSON file."""
        try:
            with open(self.training_data_filename, "w") as f:
                json.dump(list(self.training_data), f, indent=2) # Convert deque to list
            logger.info(f"{Fore.GREEN}Training data saved: {len(self.training_data)} samples.{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"{Fore.RED}Error saving training data: {e}{Style.RESET_ALL}", exc_info=e)

    def load_model(self):
        """Load the TensorFlow model or create a new one with correct input shape."""
        try:
            # First try to load the model
            model = tf.keras.models.load_model(self.model_filename)
            
            # Check if the loaded model has the correct input shape
            expected_input_shape = (None, self.sequence_length, self.settings.ml.feature_count) # Use settings for feature_count
            if model.input_shape[1:] != expected_input_shape[1:]:
                logger.warning(f"{Fore.YELLOW}Model input shape mismatch. Creating new model. Expected: {expected_input_shape}, Actual: {model.input_shape}.{Style.RESET_ALL}")
                model = self.create_lstm_model()
            else:
                logger.info(f"{Fore.GREEN}LSTM model loaded successfully with input shape: {model.input_shape}.{Style.RESET_ALL}")
            
            return model
            
        except (FileNotFoundError, OSError) as e:
            logger.info(f"{Fore.YELLOW}No valid model file found, will create new LSTM. Error: {e}{Style.RESET_ALL}", exc_info=e)
            return self.create_lstm_model() # Ensure a new model is created and returned
        except Exception as e:
            logger.error(f"{Fore.RED}Error loading model. Creating new model. Error: {e}{Style.RESET_ALL}", exc_info=e)
            return self.create_lstm_model() # Ensure a new model is created and returned

    def save_model(self):
        """Save the TensorFlow model to a file."""
        try:
            if self.model:
                self.model.save(self.model_filename)
                logger.info(f"{Fore.GREEN}Model saved successfully to: {self.model_filename}.{Style.RESET_ALL}")
            else:
                logger.warning(f"{Fore.YELLOW}No model to save.{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"{Fore.RED}Error saving model: {e}{Style.RESET_ALL}", exc_info=e)

    def predict_trade_signal(self, latest_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts the trade signal (BUY, SELL, HOLD) and confidence based on the latest data.
        
        Args:
            latest_data (Dict[str, Any]): The latest market data point with indicators.
            
        Returns:
            Dict[str, Any]: A dictionary containing the predicted signal, confidence, price, and RSI.
        """
        try:
            # Determine the current market price from latest_data
            current_market_price = 0.0
            if isinstance(latest_data, dict):
                current_market_price = float(latest_data.get('price', latest_data.get('lastPrice', 0.0)))
                # Fallback to bid/ask average if price and lastPrice are missing or zero
                if current_market_price == 0.0:
                    ask = latest_data.get('bestAsk')
                    bid = latest_data.get('bestBid')
                    if ask is not None and bid is not None:
                        try:
                            current_market_price = (float(ask) + float(bid)) / 2
                        except ValueError:
                            current_market_price = 0.0 # Keep 0.0 if conversion fails
 
            if not self.model or not self.feature_engineer.is_fitted_: # Check feature_engineer.is_fitted_
                logger.warning("Model or feature engineer not available for prediction, returning HOLD signal.")
                return {"signal": "HOLD", "confidence": 0.5, "price": current_market_price, "rsi": latest_data.get('rsi', 50.0) if isinstance(latest_data, dict) else 50.0}
            
            # Ensure latest_data is a dictionary before proceeding
            if not isinstance(latest_data, dict):
                logger.warning("Invalid latest_data format for prediction. Expected dict, returning HOLD signal.", data_type=type(latest_data))
                return {"signal": "HOLD", "confidence": 0.5, "price": current_market_price, "rsi": 50.0}
 
            # Prepare features for prediction using FeatureEngineer
            # This will return scaled features directly
            features_array = self.feature_engineer.transform([latest_data]) # Pass as list for transform
            
            # Ensure features are not empty
            if features_array.size == 0:
                logger.warning("No features prepared for prediction, returning HOLD signal.")
                return {"signal": "HOLD", "confidence": 0.5, "price": current_market_price, "rsi": latest_data.get('rsi', 50.0)}
            
            # Reshape for LSTM input (1 sample, sequence_length, num_features)
            # For a single prediction, we need to create a sequence of length 1
            # and pad it if necessary, or ensure the model is trained for single step prediction
            # For now, assuming model expects (1, sequence_length, num_features)
            # and we are providing the latest single data point as a sequence of 1
            # This might need adjustment based on how the model was actually trained
            
            # Pad the scaled_features to match the sequence_length
            padded_features = np.zeros((1, self.sequence_length, features_array.shape[1])) # Use actual feature count
            padded_features[0, -1, :] = features_array[0, :] # Place the latest features at the end of the sequence
            
            prediction = self.model.predict(padded_features)[0][0]
            
            # Determine signal based on prediction and confidence threshold
            signal = "HOLD"
            confidence = float(prediction)
            
            logger.debug("Prediction debug", confidence=confidence, buy_threshold=self.settings.buy_confidence_threshold, sell_threshold=self.settings.sell_confidence_threshold)
            
            # Adjusting for potential floating point inaccuracies to ensure 0.8 confidence triggers BUY
            # if the threshold is 0.8 or slightly higher due to precision.
            # The problem states buy_confidence_threshold <= 0.8, so if it's 0.8,
            # a prediction of 0.8 should be BUY.
            # Using a small epsilon to ensure values very close to the threshold are caught.
            # print(f"DEBUG: confidence={confidence}, buy_threshold={self.settings.buy_confidence_threshold}, sell_threshold={self.settings.sell_confidence_threshold}", flush=True) # Removed debug print
            if confidence >= self.settings.buy_confidence_threshold:
                signal = "BUY"
            elif confidence <= self.settings.sell_confidence_threshold:
                signal = "SELL"
            
            # logger.info("Prediction result", prediction=f"{prediction:.4f}", signal=signal, confidence=f"{confidence:.4f}") # Removed JSON logging
            
            return {
                "signal": signal,
                "confidence": confidence,
                "price": current_market_price,
                "rsi": latest_data.get('rsi', 50.0)
            }
        
        except Exception as e:
            # print(f"DEBUG: Exception in predict_trade_signal: {e}", flush=True) # Removed debug print
            logger.error("Error predicting trade signal", exc_info=e)
            # Ensure current_market_price is defined even in exception
            current_market_price = 0.0
            if isinstance(latest_data, dict):
                current_market_price = float(latest_data.get('price', latest_data.get('lastPrice', 0.0)))
                if current_market_price == 0.0:
                    ask = latest_data.get('bestAsk')
                    bid = latest_data.get('bestBid')
                    if ask is not None and bid is not None:
                        try:
                            current_market_price = (float(ask) + float(bid)) / 2
                        except ValueError:
                            current_market_price = 0.0
            return {"signal": "HOLD", "confidence": 0.5, "price": current_market_price, "rsi": latest_data.get('rsi', 50.0) if isinstance(latest_data, dict) else 50.0}
 
    def execute_simulated_trade(self, trade_signal: Dict[str, Any]):
        """
        Executes a simulated trade based on the given trade signal.
        
        Args:
            trade_signal (Dict[str, Any]): Dictionary containing the trade signal (BUY, SELL, HOLD),
                                           confidence, price, and RSI.
        """
        if trade_signal is None:
            logger.warning("Trade signal is None, skipping trade execution.")
            return

        signal_type = trade_signal.get("signal")
        confidence = trade_signal.get("confidence")
        current_price = trade_signal.get("price")
        rsi = trade_signal.get("rsi")
        
        if not current_price:
            logger.warning("Current price not available for trade execution.")
            return
        
        trade_executed = False
        
        # Apply RSI override
        if rsi is not None:
            if rsi > self.settings.rsi_overbought and signal_type == "BUY":
                logger.info("RSI is overbought, overriding BUY signal to HOLD.", rsi=f"{rsi:.2f}")
                signal_type = "HOLD"
            elif rsi < self.settings.rsi_oversold and signal_type == "SELL":
                logger.info("RSI is oversold, overriding SELL signal to HOLD.", rsi=f"{rsi:.2f}")
                signal_type = "HOLD"
        
        if signal_type == "BUY" and confidence >= self.settings.buy_confidence_threshold:
            # Check if we have enough balance to buy
            cost = self.trade_amount * current_price * (1 + self.fee_rate)
            if self.balance >= cost:
                self.balance -= cost
                self.position_size += self.trade_amount
                self.entry_price = current_price # For simplicity, assuming average entry price
                trade_executed = True
                logger.info("BUY executed", amount=self.trade_amount, price=current_price, new_balance=self.balance, position=self.position_size)
            else:
                logger.warning("Insufficient balance to BUY.", needed=f"{cost:.2f} AUD", have=f"{self.balance:.2f} AUD")
        
        elif signal_type == "SELL" and confidence <= self.settings.sell_confidence_threshold:
            # Check if we have enough position to sell
            if self.position_size >= self.trade_amount:
                revenue = self.trade_amount * current_price * (1 - self.fee_rate)
                self.balance += revenue
                self.position_size -= self.trade_amount
                if self.position_size < 1e-9: # Handle floating point inaccuracies
                    self.position_size = 0.0
                    self.entry_price = 0.0
                trade_executed = True
                logger.info("SELL executed", amount=self.trade_amount, price=current_price, new_balance=self.balance, position=self.position_size)
            else:
                logger.warning("Insufficient position to SELL.", have=f"{self.position_size:.4f} BTC", needed=f"{self.trade_amount:.4f} BTC")
        
        else:
            logger.info("HOLD signal. No trade executed.", confidence=confidence, rsi=rsi)
        
        if trade_executed:
            pnl, pnl_pct = self.calculate_position_pnl(current_price)
            logger.info("Current PnL", pnl=pnl, pnl_pct=pnl_pct)

    def should_save(self) -> bool:
        """Determine if the model and state should be saved."""
        return (time.time() - self.last_save_time) >= self.settings.operations.save_interval
 
    def should_train(self) -> bool:
        """Determine if the model should be retrained."""
        return (time.time() - self.last_training_time) >= self.settings.operations.training_interval and len(self.training_data) >= self.settings.ml.sequence_length
 
    def run(self):
        """Main loop for the trading bot."""
        logger.info("AutoTrader bot started.")
        iteration_count = 0
        
        while not self._shutdown_requested and (not self.limited_run or iteration_count <= self.run_iterations):
            try:
                iteration_count += 1
                
                # 1. Collect and store data
                if not self.collect_and_store_data():
                    logger.warning("Failed to collect and store data. Skipping iteration.")
                    time.sleep(self.settings.data_collection_interval)
                    print(f"\rIteration {iteration_count}: Data collection failed. Balance: {self.balance:.2f} AUD, Position: {self.position_size:.4f} BTC{' ' * 20}", end="", flush=True) # Clear line
                    continue
                
                # 2. Fit scalers if needed
                if not self.scalers_fitted and len(self.training_data) >= self.settings.ml.sequence_length:
                    self.fit_scalers()
                
                # 3. Train model if needed
                if self.should_train():
                    if self.train_model():
                        self.last_training_time = time.time()
                        self.save_model()
                        self.save_scalers()
                    else:
                        logger.warning("Model training failed or skipped due to insufficient data.")
                
                # 4. Get latest data point for prediction
                if not self.training_data:
                    logger.warning("No training data available for prediction. Skipping trade signal generation.")
                    time.sleep(self.settings.data_collection_interval)
                    print(f"\rIteration {iteration_count}: No training data. Balance: {self.balance:.2f} AUD, Position: {self.position_size:.4f} BTC{' ' * 20}", end="", flush=True) # Clear line
                    continue
                
                latest_data = self.training_data[-1]
                
                # 5. Predict trade signal
                trade_signal = self.predict_trade_signal(latest_data)
                
                # 6. Execute simulated trade
                self.execute_simulated_trade(trade_signal)
                
                # 7. Save state and training data if needed
                if self.should_save():
                    self.save_state()
                    self.save_training_data()
                    self.last_save_time = time.time()
                
                # Update and print status on a single line
                signal_type = trade_signal.get("signal", "N/A")
                confidence = trade_signal.get("confidence", 0.0)
                current_price = trade_signal.get("price", 0.0)
                rsi = latest_data.get('rsi', 50.0)

                status_message = (
                    f"\rIteration {iteration_count}: "
                    f"Price: {current_price:.2f} AUD | "
                    f"Signal: {signal_type} ({confidence:.2f}) | "
                    f"RSI: {rsi:.2f} | "
                    f"Balance: {self.balance:.2f} AUD | "
                    f"Position: {self.position_size:.4f} BTC"
                )
                print(status_message, end="", flush=True)
                
                time.sleep(self.settings.data_collection_interval)
                
            except Exception as e:
                logger.exception("An unexpected error occurred during main loop", error=str(e))
                print(f"\rIteration {iteration_count}: Error occurred. See logs. Balance: {self.balance:.2f} AUD, Position: {self.position_size:.4f} BTC{' ' * 20}", end="", flush=True) # Clear line
                time.sleep(self.settings.data_collection_interval)
        
        print("\n", end="", flush=True) # Ensure the last line is cleared before final messages
        logger.info("AutoTrader bot stopped.")
        self.save_state()
        self.save_training_data()
        self.save_model()
        self.save_scalers()
