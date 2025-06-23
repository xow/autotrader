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

class ContinuousAutoTrader:
    """
    A continuous trading bot that uses an LSTM model to predict trade signals.
    """
    def __init__(
        self,
        initial_balance: float = 1000.0,
        confidence_threshold: float = 0.1,
        rsi_oversold: int = 30,
        rsi_overbought: int = 70,
        trade_amount: float = 0.001,
        fee_rate: float = 0.001,
        max_position_size: float = 0.1,
        risk_per_trade: float = 0.02,
        limited_run: bool = False,
        run_iterations: int = 5
    ):
        """
        Initialize the ContinuousAutoTrader.

        Args:
            initial_balance (float): Initial balance in AUD.
            confidence_threshold (float): Confidence level required to make a trade.
            rsi_oversold (int): RSI value below which the asset is considered oversold.
            rsi_overbought (int): RSI value above which the asset is considered overbought.
            trade_amount (float): Amount of BTC to trade in each trade.
            fee_rate (float): Trading fee rate.
            max_position_size (float): Maximum position size in BTC.
            risk_per_trade (float): Maximum risk per trade as a fraction of the balance.
        """
        # Load configuration from environment variables or config file
        initial_config = {}
        self.config = self._load_config()
        self.initial_balance = initial_config.get("initial_balance", initial_balance)
        self.confidence_threshold = initial_config.get("confidence_threshold", confidence_threshold)
        self.rsi_oversold = initial_config.get("rsi_oversold", rsi_oversold)
        self.rsi_overbought = initial_config.get("rsi_overbought", rsi_overbought)
        self.trade_amount = initial_config.get("trade_amount", trade_amount)
        self.fee_rate = initial_config.get("fee_rate", fee_rate)
        self.max_position_size = initial_config.get("max_position_size", max_position_size)
        self.risk_per_trade = initial_config.get("risk_per_trade", risk_per_trade)
        self.limited_run = initial_config.get("limited_run", limited_run)
        self.run_iterations = initial_config.get("run_iterations", run_iterations)
        
        # Initialize state variables
        self.balance = self.initial_balance
        self.position_size = 0.0
        self.entry_price = 0.0
        self.last_save_time = time.time()
        self.last_training_time = time.time()
        self._shutdown_requested = False
        self._shutdown_event = threading.Event()
        self.training_data_filename = 'training_data.json'
        self.model_filename = 'autotrader_model.keras'
        self.scaler_filename = 'feature_scaler.pkl'
        self.state_filename = 'trader_state.pkl'
        self.min_data_points = 100  # Minimum data points for prediction
        self.sequence_length = 60  # LSTM sequence length
        self.max_training_samples = 1000  # Maximum number of training samples
        self.run_iterations = 5  # Number of iterations to run in limited mode
        self.save_interval_seconds = 3600  # Save every hour
        self.training_interval_seconds = 43200  # Train every 12 hours
        self.feature_scaler = self.load_scalers()
        self.model = self.load_model()
        self.training_data = self.load_training_data()
        self._price_history = []
        self._training_data_deque = deque(maxlen=self.max_training_samples)
        self._model_summary_logged = False
        self._data_buffer = deque(maxlen=self.sequence_length)
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        logger.debug("ContinuousAutoTrader __init__ completed.")
        
        # Load initial state
        self.load_state()
        
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            logger.info("Signal handlers set up for graceful shutdown.")
        except Exception as e:
            logger.error(f"Error setting up signal handlers: {e}")

    def _signal_handler(self, signum, frame):
        """Handle signals for graceful shutdown."""
        logger.info(f"Signal {signum} received, initiating graceful shutdown.")
        self._shutdown_requested = True
        self._shutdown_event.set()
        
        # Initialize the LSTM model
        if self.model is None:
            self.model = self.create_lstm_model()
        
        # Log configuration details
        logger.info("AutoTrader initialized",
                    initial_balance=self.initial_balance,
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
                    run_iterations=self.run_iterations)
        
        # Log the number of training samples
        logger.info(f"Loaded {len(self.training_data)} training samples")
        
        # Log the model summary
        logger.info(f"Model summary: {self.model.summary()}")
        
        # Log the feature scaler
        logger.info(f"Feature scaler: {self.feature_scaler}")
        
        # Log the initial balance
        logger.info(f"Initial balance: {self.initial_balance}")
        
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

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables or a config file."""
        config = DEFAULT_CONFIG.copy()
        
        # Load from environment variables
        config.update(os.environ)
        
        # Load from config file (if it exists)
        config_file_path = os.path.join(os.getcwd(), 'autotrader_config.json')
        if os.path.exists(config_file_path):
            try:
                with open(config_file_path, 'r') as f:
                    file_config = json.load(f)
                    config.update(file_config)
            except Exception as e:
                logger.error(f"Error loading config file: {e}")
        
        return config

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
                self.balance = state.get('balance', self.initial_balance)
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
        except Exception as e:
            logger.error(f"Error loading scalers: {e}")

    def fetch_market_data(self) -> List[Dict[str, Any]]:
        """Fetch market data from the BTCMarkets API."""
        try:
            url = "https://api.btcmarkets.net/v3/markets/BTC-AUD/trades"
            response = requests.get(url)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            
            trades = response.json()
            logger.debug(f"Fetched {len(trades)} trades from BTCMarkets API")
            
            # Basic data validation
            if not isinstance(trades, list):
                logger.error("Invalid data format from BTCMarkets API")
                return []
            
            # Convert timestamps to ISO format and ensure all keys are strings
            market_data = []
            for trade in trades:
                # Ensure all keys are strings
                trade = {str(k): v for k, v in trade.items()}
                
                # Convert timestamp to ISO format
                timestamp_ms = trade.get('timestamp')
                if timestamp_ms is not None:
                    try:
                        # Parse ISO format string to datetime object
                        # Parse ISO format string to datetime object, handling 'Z' and ensuring UTC
                        dt_object = datetime.fromisoformat(timestamp_ms.replace('Z', '')).replace(tzinfo=timezone.utc)
                        # Convert datetime object to Unix timestamp in milliseconds
                        timestamp = int(dt_object.timestamp() * 1000)
                        trade['timestamp'] = timestamp
                    except ValueError as ve:
                        logger.warning(f"Invalid timestamp format: {timestamp_ms}. Error: {ve}. Skipping timestamp conversion for this trade.")
                
                market_data.append(trade)
            
            return market_data
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching market data: {e}", exc_info=True)
            return []

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

    def calculate_technical_indicators(self, market_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate technical indicators for each data point."""
        # Ensure market_data is not None and is a list
        if not market_data or not isinstance(market_data, list):
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
        
        # Drop rows with NaN in the price column
        df = df.dropna(subset=[price_col])
        
        # Ensure there are enough valid data points
        if len(df) < 20:
            logger.warning("Not enough valid data points to calculate indicators")
            return market_data
        
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
        
        roll_up1 = up.rolling(window=14).mean()
        roll_down1 = np.abs(down.rolling(window=14).mean())
        
        RS = roll_up1 / roll_down1
        df['rsi'] = 100.0 - (100.0 / (1.0 + RS))
        
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
        df['atr'] = df[price_col].rolling(window=14).mean()  # Simplified
        
        # Merge indicators back into original market data
        indicators = df[['sma_5', 'sma_10', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_upper', 'bb_middle', 'bb_lower', 'atr']].to_dict('records')
        
        # Ensure the length of indicators matches the length of market_data
        if len(indicators) != len(market_data):
            logger.warning(f"Length mismatch: len(indicators)={len(indicators)}, len(market_data)={len(market_data)}")
            return market_data
        
        # Update each dictionary in market_data with corresponding indicators
        for i, indicator in enumerate(indicators):
            market_data[i].update(indicator)
        
        return market_data

    def create_lstm_model(self):
        """Create the LSTM model."""
        try:
            # Define the LSTM model
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 12)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(50, return_sequences=False),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(25),
                tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer
            ])
            
            # Compile the model
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            # Log model summary
            model.summary()
            
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
                logger.warning(f"Invalid price or volume format: price={price}, volume={volume}")
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

    def fit_scalers(self):
        """Fit scalers to the training data."""
        try:
            # Prepare the training data by extracting features
            processed_training_data = [self.prepare_features(data_point) for data_point in self.training_data]
            
            # Ensure the processed training data is not empty
            if not processed_training_data:
                logger.warning("No processed training data available, skipping scaler fitting")
                return
            
            # Convert to numpy array
            processed_training_data_np = np.array(processed_training_data)
            
            # Initialize the feature scaler if not already initialized
            if self.feature_scaler is None:
                self.feature_scaler = StandardScaler()
            
            # Fit the feature scaler
            self.feature_scaler.fit(processed_training_data_np)
            
            logger.info("Scalers fitted successfully")
            
        except Exception as e:
            logger.error(f"Error fitting scalers: {e}", exc_info=True)

    def prepare_lstm_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training."""
        try:
            # Ensure we have enough data
            if len(self.training_data) < self.sequence_length:
                logger.warning(f"Not enough data for training. Need at least {self.sequence_length}, got {len(self.training_data)}")
                return np.array([]), np.array([])
            
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
            self.model.fit(X, y, epochs=10, batch_size=32)
            
            # Log the training
            logger.info("LSTM model trained successfully")
            
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
            for data_point in market_data:
                self._update_feature_buffer(data_point)
            
            # Prepare the training data
            for data_point in market_data:
                self.training_data.append(data_point)
            
            # Limit the training data to the maximum number of samples
            self.training_data = self.training_data[-self.max_training_samples:]
            
            # Log the number of training samples
            logger.debug(f"Collected and stored {len(market_data)} market data points. Total training samples: {len(self.training_data)}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error collecting and storing data: {e}", exc_info=True)
            return False

    def load_training_data(self) -> List[Dict]:
        """Load training data from JSON file."""
        try:
            with open(self.training_data_filename, "r") as f:
                data = json.load(f)
            logger.info(f"Training data loaded ({len(data)} samples)")
            return data
        except FileNotFoundError:
            logger.info("No training data file found, starting fresh")
            return []
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return []

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
            expected_input_shape = (None, self.sequence_length, 12)
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
        initial_balance=args.balance,
        limited_run=args.limited_run,
        run_iterations=args.iterations
    )
    
    # Run the trading logic
    try:
        trader.collect_and_store_data()
        trader.fit_scalers()
        trader.train_model()
        trader.save_scalers()
        trader.save_model()
        trader.save_training_data()
        trader.save_state()
    except Exception as e:
        logger.error(f"Error during trading logic execution: {e}")
    finally:
        print("Trading completed.")

if __name__ == "__main__":
    main()
