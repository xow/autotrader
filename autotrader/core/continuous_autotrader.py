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
from autotrader.utils.exceptions import APIError, DataError, NetworkError, NetworkTimeoutError


class ContinuousAutoTrader:
    """
    A continuous trading bot that uses an LSTM model to predict trade signals.
    """
    def __init__(
        self,
        limited_run: bool = False,
        run_iterations: int = 5
    ):
        """
        Initialize the ContinuousAutoTrader.

        Args:
            limited_run (bool): If True, the bot will run for a limited number of iterations.
            run_iterations (int): Number of iterations to run in limited mode.
        """
        self.settings = get_settings()

        # Initialize state variables using settings
        self.balance = self.settings.initial_balance
        self.confidence_threshold = self.settings.buy_confidence_threshold # Using buy_confidence_threshold as general confidence
        self.rsi_oversold = self.settings.rsi_oversold
        self.rsi_overbought = self.settings.rsi_overbought
        self.trade_amount = self.settings.trade_amount
        self.fee_rate = self.settings.fee_rate
        self.max_position_size = self.settings.max_position_size # Accessing from trading config
        self.risk_per_trade = self.settings.risk_per_trade # This value is now in settings
        self.limited_run = limited_run
        self.run_iterations = run_iterations
        
        self.position_size = 0.0
        self.entry_price = 0.0
        self.last_save_time = time.time()
        self.last_training_time = time.time()
        self._shutdown_requested = False
        self._shutdown_event = threading.Event()
        self.training_data_filename = self.settings.training_data_filename
        self.model_filename = self.settings.model_filename
        self.scaler_filename = self.settings.config.ml.scalers_filename # Accessing from ML config
        self.state_filename = self.settings.state_filename
        self.min_data_points = self.settings.config.ml.sequence_length # Using sequence_length as min_data_points
        self.sequence_length = self.settings.config.ml.sequence_length # Use ml.sequence_length
        self.max_training_samples = self.settings.config.ml.max_training_samples # Use ml.max_training_samples
        self.save_interval_seconds = self.settings.config.operations.save_interval
        self.training_interval_seconds = self.settings.config.operations.training_interval
        self.feature_scaler = self.load_scalers()
        self.model = self.load_model()
        self.training_data = deque(self.load_training_data(), maxlen=self.max_training_samples) # Initialize deque with loaded data and maxlen
        self._price_history = []
        self._training_data_deque = deque(maxlen=self.sequence_length) # Changed to sequence_length for buffer
        self._model_summary_logged = False
        self._data_buffer = deque(maxlen=self.sequence_length)
        self.scalers_fitted = False # Add this attribute
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        logger.debug("ContinuousAutoTrader __init__ completed.")
        
        # Load initial state
        self.load_state()
        
        # Initialize the LSTM model
        if self.model is None:
            self.model = self.create_lstm_model()
        
        # Log configuration details
        logger.info("AutoTrader initialized", **{
            "initial_balance": self.balance,
            "confidence_threshold": self.confidence_threshold,
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought,
            "trade_amount": self.trade_amount,
            "fee_rate": self.fee_rate,
            "max_position_size": self.max_position_size,
            "risk_per_trade": self.risk_per_trade,
            "model_filename": self.model_filename,
            "scaler_filename": self.scaler_filename,
            "state_filename": self.state_filename,
            "sequence_length": self.sequence_length,
            "min_data_points": self.min_data_points,
            "limited_run": self.limited_run,
            "run_iterations": self.run_iterations
        })
        
        # Log the number of training samples
        logger.info(f"Loaded {len(self.training_data)} training samples")
        
        # Log the model summary
        if self.model and not self._model_summary_logged: # Check if model is not None before calling summary
            self.model.summary(print_fn=lambda x: logger.info(x)) # Use print_fn to log summary
            self._model_summary_logged = True
        
        # Log the feature scaler
        logger.info(f"Feature scaler: {self.feature_scaler}")
        
        # Log the initial balance
        logger.info(f"Initial balance: {self.balance}")
        
        # Log the initial position size
        logger.info(f"Initial position size: {self.position_size}")
        
        # Log the initial entry price
        logger.info(f"Initial entry price: {self.entry_price}")
        
        # Log the initial last save time
        logger.info(f"Initial last save time: {self.last_save_time}")
        
        # Log the initial last training time
        logger.info(f"Initial last training time: {self.last_training_time}")
        
        # Log the initial shutdown requested
        logger.info(f"Initial shutdown requested: {self._shutdown_requested}")
        
        # Log the initial shutdown event
        logger.info(f"Initial shutdown event: {self._shutdown_event}")
        
        # Log the initial training data filename
        logger.info(f"Initial training data filename: {self.training_data_filename}")
        
        # Log the initial model filename
        logger.info(f"Initial model filename: {self.model_filename}")
        
        # Log the initial scaler filename
        logger.info(f"Initial scaler filename: {self.scaler_filename}")
        
        # Log the initial state filename
        logger.info(f"Initial state filename: {self.state_filename}")
        
        # Log the initial min data points
        logger.info(f"Initial min data points: {self.min_data_points}")
        
        # Log the initial sequence length
        logger.info(f"Initial sequence length: {self.sequence_length}")
        
        # Log the initial max training samples
        logger.info(f"Initial max training samples: {self.max_training_samples}")
        
        # Log the initial run iterations
        logger.info(f"Initial run iterations: {self.run_iterations}")
        
        # Log the initial save interval seconds
        logger.info(f"Initial save interval seconds: {self.save_interval_seconds}")
        
        # Log the initial training interval seconds
        logger.info(f"Initial training interval seconds: {self.training_interval_seconds}")
        
        # Log the initial feature scaler
        logger.info(f"Initial feature scaler: {self.feature_scaler}")
        
        # Log the initial model
        logger.info(f"Initial model: {self.model}")
        
        # Log the initial training data
        logger.info(f"Initial training data: {self.training_data}")
        
        # Log the initial price history
        logger.info(f"Initial price history: {self._price_history}")
        
        # Log the initial training data deque
        logger.info(f"Initial training data deque: {self._training_data_deque}")
        
        # Log the initial model summary logged
        logger.info(f"Initial model summary logged: {self._model_summary_logged}")
        
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        logger.debug("Signal handlers set up.")

    def _signal_handler(self, signum, frame):
        """Handle signals for graceful shutdown."""
        logger.info(f"Signal {signum} received, initiating graceful shutdown.")
        self._shutdown_requested = True
        self._shutdown_event.set()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables or a config file."""
        # This method is no longer needed as configuration is handled by Settings
        # Keeping it as a placeholder if other parts of the code still call it
        logger.warning("'_load_config' method is deprecated. Use 'Settings' for configuration.")
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
            logger.info("Trader state saved successfully")
        except Exception as e:
            logger.error(f"Error saving trader state: {e}")

    def load_state(self):
        """Load the trader state from a file."""
        try:
            with open(self.state_filename, 'rb') as f:
                state = pickle.load(f)
                self.balance = state.get('balance', self.settings.initial_balance) # Use settings for default
                self.position_size = state.get('position_size', 0.0)
                self.entry_price = state.get('entry_price', 0.0)
            logger.info("Trader state loaded successfully")
        except FileNotFoundError:
            logger.info("No trader state file found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading trader state: {e}")

    def save_scalers(self):
        """Save the scalers to a file."""
        try:
            with open(self.scaler_filename, 'wb') as f:
                pickle.dump({
                    'feature_scaler': self.feature_scaler,
                }, f)
            logger.info("Scalers saved successfully")
        except Exception as e:
            logger.error(f"Error saving scalers: {e}")

    def load_scalers(self):
        """Load the scalers from a file."""
        try:
            with open(self.scaler_filename, 'rb') as f:
                scalers = pickle.load(f)
                self.feature_scaler = scalers['feature_scaler']
            logger.info("Scalers loaded successfully")
        except FileNotFoundError:
            logger.info("No scalers file found, creating new scalers")
            self.feature_scaler = StandardScaler() # Initialize a new scaler
        except Exception as e:
            logger.error(f"Error loading scalers: {e}")
            self.feature_scaler = StandardScaler() # Initialize a new scaler on error
        return self.feature_scaler # Return the loaded scaler or a new one

    def fetch_market_data(self) -> List[Dict[str, Any]]:
        """Fetch market data from the BTCMarkets API."""
        try:
            url = self.settings.config.api.base_url + "/markets/tickers" # Use settings for base URL and tickers endpoint
            params = {"marketId": "BTC-AUD"} # Add marketId parameter
            response = requests.get(url, params=params, timeout=self.settings.config.api.timeout) # Use settings for timeout
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            
            market_data = response.json()
            logger.debug(f"Fetched {len(market_data)} market data points from BTCMarkets API")
            
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
                        logger.warning(f"Invalid timestamp format: {timestamp_str}. Error: {ve}. Skipping timestamp conversion for this data point.")
                
                processed_data.append(data_point)
            
            return processed_data
        
        except requests.exceptions.Timeout as e:
            logger.error(f"Market data request timed out: {e}")
            raise NetworkTimeoutError(f"Market data request timed out: {e}") from e
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Failed to connect to the exchange: {e}")
            raise NetworkError(f"Failed to connect to the exchange: {e}") from e
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error fetching market data: {e.response.status_code} - {e.response.text}")
            raise APIError(f"HTTP error fetching market data: {e.response.status_code} - {e.response.text}", status_code=e.response.status_code) from e
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response from API: {e}. Response text: {response.text if 'response' in locals() else 'N/A'}")
            raise DataError("Invalid JSON response from API", data_type="market_data", validation_errors={"json_decode_error": str(e)}) from e
        except Exception as e:
            logger.error(f"An unexpected error occurred while fetching market data: {e}", exc_info=True)
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
                logger.warning(f"Invalid price or volume in extracted data: price={extracted['price']}, volume={extracted['volume']}")
                return None
            
            return extracted
        
        except (ValueError, TypeError) as e:
            logger.error(f"Error converting market data values: {e}. Data: {btc_aud_data}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during data extraction: {e}", exc_info=True)
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
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = 100 - 100 / (1 + rs)
        
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
            
            # Ensure 'lastPrice' or 'price' column exists
            if 'lastPrice' not in df.columns and 'price' not in df.columns:
                logger.warning("No 'lastPrice' or 'price' column in market data, cannot calculate indicators")
                return market_data
            
            # Use 'lastPrice' if available, otherwise use 'price'
            price_col = 'lastPrice' if 'lastPrice' in df.columns else 'price'
            
            # Convert price column to numeric, coercing errors to NaN
            df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
            
            # Fill NaN prices with previous valid price or 0.0
            df[price_col] = df[price_col].fillna(method='ffill').fillna(0.0)
            
            # Ensure there are enough valid data points for basic calculations
            if len(df) < 1: # At least one data point is needed
                logger.warning(f"Not enough valid data points to calculate indicators. Need at least 1, got {len(df)}. Returning original market data.")
                return market_data
        else:
            logger.warning("No market data or price/volume arrays provided for indicator calculation.")
            return []
        
        # Calculate Simple Moving Averages (SMA)
        df['sma_5'] = df[price_col].rolling(window=5).mean()
        df['sma_10'] = df[price_col].rolling(window=10).mean()
        df['sma_20'] = df[price_col].rolling(window=20).mean()
        
        # Calculate Exponential Moving Averages (EMA)
        df['ema_12'] = df[price_col].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df[price_col].ewm(span=26, adjust=False).mean()
        
        # Calculate Relative Strength Index (RSI)
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
            df['volume_sma'] = df['volume'].rolling(window=self.settings.config.ml.volume_sma_period).mean() # Use settings for volume_sma_period
        else:
            df['volume_sma'] = 0.0 # Default if volume is not available
        
        # Prepare the list of dictionaries to return
        # Ensure all original market_data fields are preserved and new indicators are added
        result_data = df.to_dict('records')
        
        # If original market_data was provided, merge the indicators back
        if market_data:
            # Ensure the length of indicators matches the length of market_data
            if len(result_data) != len(market_data):
                logger.warning(f"Length mismatch after indicator calculation: len(result_data)={len(result_data)}, len(market_data)={len(market_data)}. Returning original market_data.")
                return market_data # Return original if lengths don't match
            
            for i, indicator_data in enumerate(result_data):
                market_data[i].update(indicator_data)
            return market_data
        else:
            return result_data # Return the new list of dictionaries if only prices/volumes were provided

    def create_lstm_model(self):
        """Create the LSTM model."""
        try:
            # Define the LSTM model
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(self.settings.config.ml.lstm_units, return_sequences=True, input_shape=(self.settings.config.ml.sequence_length, self.settings.config.ml.feature_count)),
                tf.keras.layers.Dropout(self.settings.config.ml.dropout_rate),
                tf.keras.layers.LSTM(self.settings.config.ml.lstm_units, return_sequences=False),
                tf.keras.layers.Dropout(self.settings.config.ml.dropout_rate),
                tf.keras.layers.Dense(self.settings.config.ml.dense_units),
                tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer
            ])
            
            # Compile the model
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.settings.config.ml.learning_rate), loss='mse', metrics=['mae'])
            
            # Log model summary
            model.summary(print_fn=lambda x: logger.info(x)) # Use print_fn to log summary
            
            return model
        
        except Exception as e:
            logger.error(f"Error creating LSTM model: {e}", exc_info=True)
            return None

    def prepare_features(self, data_point: dict) -> list:
        """Prepare feature vector from a single data point.
        
        Args:
            data_point: Dictionary containing price and indicator data
            
        Returns:
            list: Feature vector for model input
        """
        try:
            if not data_point:
                raise ValueError("Empty data point provided")
            
            # Extract features in the exact order expected by the model
            features = [
                float(data_point.get('price', 0)),
                float(data_point.get('volume', 0)),
                float(data_point.get('spread', 0)),
                float(data_point.get('sma_5', 0)),
                float(data_point.get('sma_20', 0)),
                float(data_point.get('ema_12', 0)),
                float(data_point.get('ema_26', 0)),
                float(data_point.get('rsi', 0)),
                float(data_point.get('macd', 0)),
                float(data_point.get('macd_signal', 0)),
                float(data_point.get('bb_upper', 0)),
                float(data_point.get('bb_lower', 0))
            ]
            
            # Log the first few features for debugging (without sensitive data)
            if len(features) > 0:
                logger.debug(f"Prepared features: price={features[0]:.2f}, volume={features[1]:.2f}, "
                            f"spread={features[2]:.2f}, rsi={features[7]:.2f}")
            
            return features
        
        except Exception as e:
            logger.exception(f"Error preparing features: {e}")
            # Return a zero vector of expected length on error
            features = [0.0] * 12  # 12 features in total
            return features

    def _update_feature_buffer(self, market_data: Dict[str, Any]) -> None:
        """Update the feature buffer with the latest market data."""
        try:
            # Extract relevant features from market_data
            price = market_data.get('lastPrice') or market_data.get('price')
            volume = market_data.get('volume', 0.0) # Default to 0.0 if volume is missing
            
            # Validate that price is not None
            if price is None:
                logger.warning(f"Missing price in market data: price={price}")
                return
            
            # Convert price and volume to float
            try:
                price = float(price)
                volume = float(volume)
            except ValueError:
                logger.warning(f"Invalid price or volume format: {price}, volume={volume}")
                return
            
            # Create a feature vector
            feature_vector = [price, volume]
            
            # Append the feature vector to the data buffer
            if hasattr(self, '_data_buffer'):
                logger.debug(f"_data_buffer exists. Type: {type(self._data_buffer)}")
                self._data_buffer.append(feature_vector)
            else:
                logger.error("Attempted to append to _data_buffer, but it does not exist on self.")
                return
            
            logger.debug(f"Updated feature buffer with: price={price}, volume={volume}")
            
        except Exception as e:
            logger.error(f"Error updating feature buffer: {e}", exc_info=True)

    def fit_scalers(self) -> bool:
        """Fit scalers to the training data."""
        try:
            # Prepare the training data by extracting features
            # Ensure training_data is a list of dictionaries
            if not isinstance(self.training_data, deque):
                logger.error("training_data is not a deque, cannot fit scalers.")
                self.scalers_fitted = False
                return False

            processed_training_data = []
            for data_point in self.training_data:
                features = self.prepare_features(data_point)
                # Filter out data points with NaN values in features
                if not any(np.isnan(f) for f in features):
                    processed_training_data.append(features)
            
            # Ensure the processed training data is not empty
            if not processed_training_data:
                logger.warning("No valid processed training data available, skipping scaler fitting")
                self.scalers_fitted = False
                return False
            
            # Convert to numpy array
            processed_training_data_np = np.array(processed_training_data)
            
            # Initialize the feature scaler if not already initialized
            if self.feature_scaler is None:
                self.feature_scaler = StandardScaler()
            
            # Fit the feature scaler
            self.feature_scaler.fit(processed_training_data_np)
            self.scalers_fitted = True
            logger.info("Scalers fitted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error fitting scalers: {e}", exc_info=True)
            self.scalers_fitted = False
            return False

    def prepare_lstm_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training."""
        try:
            # Ensure we have enough data
            if len(self.training_data) < self.sequence_length:
                logger.warning(f"Not enough data for training. Need at least {self.sequence_length}, got {len(self.training_data)}")
                # Return empty arrays with correct dimensions
                return np.empty((0, self.sequence_length, self.settings.config.ml.feature_count)), np.empty((0,))
            
            # Extract features and labels
            features, labels = [], []
            for i in range(len(self.training_data) - self.sequence_length):
                # Extract the sequence and prepare features for each data point in the sequence
                sequence = [self.prepare_features(self.training_data[j]) for j in range(i, i + self.sequence_length)]
                
                # Extract the label (next data point) and prepare its features
                next_data_point = self.prepare_features(self.training_data[i + self.sequence_length])
                
                # Append the sequence and label
                features.append(sequence)
                labels.append(next_data_point[0]) # Assuming the label is the first feature (e.g., price)
            
            # Convert to numpy arrays
            features = np.array(features)
            labels = np.array(labels)
            
            # Log the shape of the features and labels
            logger.debug(f"Features shape: {features.shape}")
            logger.debug(f"Labels shape: {labels.shape}")
            
            return features, labels
        
        except Exception as e:
            logger.error(f"Error preparing LSTM training data: {e}", exc_info=True)
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
            # X = np.reshape(X, (X.shape[0], X.shape[1], 1)) # This line is incorrect and should be removed
            
            # Train the model
            history = self.model.fit(X, y, epochs=self.settings.config.ml.training_epochs, batch_size=self.settings.config.ml.batch_size, verbose=0) # Use settings for epochs and batch_size
            
            # Log the training
            logger.info("LSTM model trained successfully")
            self._log_model_performance(history) # Log model performance
            
            return True
        
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}", exc_info=True)
            return False

    def _log_model_performance(self, history):
        """Log model performance metrics."""
        try:
            if history is None or not hasattr(history, 'history'):
                logger.warning("No training history available, cannot log performance")
                return
            
            # Extract the loss and mean absolute error from the history
            loss = history.history.get('loss')
            mae = history.history.get('mae')
            
            # Log the loss and mean absolute error
            logger.info(f"Training Loss: {loss[-1]:.4f}, Mean Absolute Error: {mae[-1]:.4f}")
            
        except Exception as e:
            logger.error(f"Error logging model performance: {e}", exc_info=True)

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
            logger.debug(f"Collected and stored {len(market_data)} market data points. Total training samples: {len(self.training_data)}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error collecting and storing data: {e}", exc_info=True)
            return False

    def load_training_data(self) -> Deque[Dict]:
        """Load training data from JSON file."""
        try:
            with open(self.training_data_filename, "r") as f:
                data = json.load(f)
            logger.info(f"Training data loaded ({len(data)} samples)")
            return deque(data, maxlen=self.max_training_samples) # Return as deque
        except FileNotFoundError:
            logger.info("No training data file found, starting fresh")
            return deque(maxlen=self.max_training_samples) # Return empty deque
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return deque(maxlen=self.max_training_samples) # Return empty deque on error

    def save_training_data(self):
        """Save training data to JSON file."""
        try:
            with open(self.training_data_filename, "w") as f:
                json.dump(self.training_data, f, indent=2)
            logger.info(f"Training data saved ({len(self.training_data)} samples)")
        except Exception as e:
            logger.error(f"Error saving training data: {e}")

    def load_model(self):
        """Load the TensorFlow model or create a new one with correct input shape."""
        try:
            # First try to load the model
            model = tf.keras.models.load_model(self.model_filename)
            
            # Check if the loaded model has the correct input shape
            expected_input_shape = (None, self.sequence_length, self.settings.config.ml.feature_count) # Use settings for feature_count
            if model.input_shape[1:] != expected_input_shape[1:]:
                logger.warning(f"Model input shape mismatch. Expected {expected_input_shape}, got {model.input_shape}. Creating new model.")
                model = self.create_lstm_model()
            else:
                logger.info("LSTM model loaded successfully with input shape %s", model.input_shape)
            
            return model
            
        except (FileNotFoundError, OSError) as e:
            logger.info("No valid model file found, will create new LSTM: %s", str(e))
            return self.create_lstm_model() # Ensure a new model is created and returned
        except Exception as e:
            logger.error("Error loading model: %s. Creating new model.", str(e))
            return self.create_lstm_model() # Ensure a new model is created and returned

    def save_model(self):
        """Save the TensorFlow model to a file."""
        try:
            if self.model:
                self.model.save(self.model_filename)
                logger.info(f"Model saved successfully to {self.model_filename}")
            else:
                logger.warning("No model to save.")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def predict_trade_signal(self, latest_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts the trade signal (BUY, SELL, HOLD) and confidence based on the latest data.
        
        Args:
            latest_data (Dict[str, Any]): The latest market data point with indicators.
            
        Returns:
            Dict[str, Any]: A dictionary containing the predicted signal, confidence, price, and RSI.
        """
        try:
            if not self.model or not self.feature_scaler:
                logger.warning("Model or scaler not available for prediction, returning HOLD signal.")
                return {"signal": "HOLD", "confidence": 0.5, "price": latest_data.get('price', 0.0) if isinstance(latest_data, dict) else 0.0, "rsi": latest_data.get('rsi', 50.0) if isinstance(latest_data, dict) else 50.0}
            
            # Ensure latest_data is a dictionary before proceeding
            if not isinstance(latest_data, dict):
                logger.warning(f"Invalid latest_data format for prediction: {type(latest_data)}. Expected dict, returning HOLD signal.")
                return {"signal": "HOLD", "confidence": 0.5, "price": 0.0, "rsi": 50.0}

            # Prepare features for prediction
            features = self.prepare_features(latest_data)
            
            # Ensure features are not empty
            if not features:
                logger.warning("No features prepared for prediction, returning HOLD signal.")
                return {"signal": "HOLD", "confidence": 0.5, "price": latest_data.get('price', 0.0), "rsi": latest_data.get('rsi', 50.0)}
            
            # Scale the features
            scaled_features = self.feature_scaler.transform(np.array(features).reshape(1, -1))
            
            # Reshape for LSTM input (1 sample, sequence_length, num_features)
            # For a single prediction, we need to create a sequence of length 1
            # and pad it if necessary, or ensure the model is trained for single step prediction
            # For now, assuming model expects (1, sequence_length, num_features)
            # and we are providing the latest single data point as a sequence of 1
            # This might need adjustment based on how the model was actually trained
            
            # Pad the scaled_features to match the sequence_length
            padded_features = np.zeros((1, self.sequence_length, self.settings.config.ml.feature_count))
            padded_features[0, -1, :] = scaled_features # Place the latest features at the end of the sequence
            
            prediction = self.model.predict(padded_features)[0][0]
            
            # Determine signal based on prediction and confidence threshold
            signal = "HOLD"
            confidence = float(prediction)
            
            # Ensure price is not 0.0 if it's a valid data point
            price_for_prediction = latest_data.get('price', 0.0)
            if price_for_prediction == 0.0 and latest_data.get('lastPrice') is not None:
                price_for_prediction = float(latest_data['lastPrice'])

            if confidence > self.settings.config.trading.buy_confidence_threshold:
                signal = "BUY"
            elif confidence < self.settings.config.trading.sell_confidence_threshold:
                signal = "SELL"
            
            logger.info(f"Prediction: {prediction:.4f}, Signal: {signal}, Confidence: {confidence:.4f}")
            
            return {
                "signal": signal,
                "confidence": confidence,
                "price": latest_data.get('price', 0.0),
                "rsi": latest_data.get('rsi', 50.0)
            }
        
        except Exception as e:
            logger.error(f"Error predicting trade signal: {e}", exc_info=True)
            return {"signal": "HOLD", "confidence": 0.5, "price": latest_data.get('price', 0.0), "rsi": latest_data.get('rsi', 50.0)}

    def execute_simulated_trade(self, trade_signal: Dict[str, Any]):
        """
        Executes a simulated trade based on the given trade signal.
        
        Args:
            trade_signal (Dict[str, Any]): Dictionary containing the trade signal (BUY, SELL, HOLD),
                                           confidence, price, and RSI.
        """
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
            if rsi > self.settings.config.trading.rsi_overbought and signal_type == "BUY":
                logger.info(f"RSI ({rsi:.2f}) is overbought, overriding BUY signal to HOLD.")
                signal_type = "HOLD"
            elif rsi < self.settings.config.trading.rsi_oversold and signal_type == "SELL":
                logger.info(f"RSI ({rsi:.2f}) is oversold, overriding SELL signal to HOLD.")
                signal_type = "HOLD"
        
        if signal_type == "BUY" and confidence > self.settings.config.trading.buy_confidence_threshold:
            # Check if we have enough balance to buy
            cost = self.trade_amount * current_price * (1 + self.fee_rate)
            if self.balance >= cost:
                self.balance -= cost
                self.position_size += self.trade_amount
                self.entry_price = current_price # For simplicity, assuming average entry price
                trade_executed = True
                logger.info(f"BUY executed: {self.trade_amount:.4f} BTC at {current_price:.2f} AUD. New balance: {self.balance:.2f} AUD, Position: {self.position_size:.4f} BTC")
            else:
                logger.warning(f"Insufficient balance to BUY. Needed {cost:.2f} AUD, Have {self.balance:.2f} AUD.")
        
        elif signal_type == "SELL" and confidence < self.settings.config.trading.sell_confidence_threshold:
            # Check if we have enough position to sell
            if self.position_size >= self.trade_amount:
                revenue = self.trade_amount * current_price * (1 - self.fee_rate)
                self.balance += revenue
                self.position_size -= self.trade_amount
                if self.position_size < 1e-9: # Handle floating point inaccuracies
                    self.position_size = 0.0
                    self.entry_price = 0.0
                trade_executed = True
                logger.info(f"SELL executed: {self.trade_amount:.4f} BTC at {current_price:.2f} AUD. New balance: {self.balance:.2f} AUD, Position: {self.position_size:.4f} BTC")
            else:
                logger.warning(f"Insufficient position to SELL. Have {self.position_size:.4f} BTC, Need {self.trade_amount:.4f} BTC.")
        
        else:
            logger.info(f"HOLD signal: Confidence {confidence:.4f}, RSI {rsi:.2f}. No trade executed.")
        
        if trade_executed:
            pnl, pnl_pct = self.calculate_position_pnl(current_price)
            logger.info(f"Current PnL: {pnl:.2f} AUD ({pnl_pct:.2f}%)")

    def should_save(self) -> bool:
        """Determine if the model and state should be saved."""
        return (time.time() - self.last_save_time) >= self.settings.config.operations.save_interval

    def should_train(self) -> bool:
        """Determine if the model should be retrained."""
        return (time.time() - self.last_training_time) >= self.settings.config.operations.training_interval and len(self.training_data) >= self.settings.config.ml.sequence_length

    def run(self):
        """Main loop for the trading bot."""
        logger.info("AutoTrader bot started.")
        iteration_count = 0
        
        while not self._shutdown_requested and (not self.limited_run or iteration_count < self.run_iterations):
            try:
                logger.info(f"--- Iteration {iteration_count + 1} ---")
                
                # 1. Collect and store data
                if not self.collect_and_store_data():
                    logger.warning("Failed to collect and store data. Skipping iteration.")
                    time.sleep(self.settings.config.operations.data_collection_interval) # Wait before retrying
                    iteration_count += 1
                    continue
                
                # 2. Fit scalers if needed
                if self.feature_scaler is None and len(self.training_data) >= self.settings.config.ml.sequence_length:
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
                    time.sleep(self.settings.config.operations.data_collection_interval)
                    iteration_count += 1
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
                
                iteration_count += 1
                time.sleep(self.settings.config.operations.data_collection_interval) # Wait for next iteration
                
            except Exception as e:
                logger.exception(f"An unexpected error occurred during main loop: {e}")
                time.sleep(self.settings.config.operations.data_collection_interval) # Wait before retrying
        
        logger.info("AutoTrader bot stopped.")
        self.save_state()
        self.save_training_data()
        self.save_model()
        self.save_scalers()