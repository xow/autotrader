import requests
from datetime import datetime
import tensorflow as tf
import numpy as np
import json
import time
import pickle
import os
from typing import Dict, List, Optional, Tuple, Any, Union
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
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autotrader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContinuousAutoTrader:
    def __init__(self, initial_balance: float = 10000.0, test_mode: bool = False, test_iterations: int = 5):
        self.balance = initial_balance
        self.model_filename = "autotrader_model.keras"
        self.training_data_filename = "training_data.json"
        self.scalers_filename = "scalers.pkl"
        self.state_filename = "trader_state.pkl"
        self.save_interval_seconds = 1800  # Save every 30 minutes
        self.training_interval_seconds = 600  # Retrain every 10 minutes
        self.max_training_samples = 2000  # Increased for more data
        self.sequence_length = 20  # Number of time steps for LSTM
        self.min_data_points = 50  # Minimum data points before attempting predictions
        
        # Test mode settings
        self.test_mode = test_mode
        self.test_iterations = test_iterations
        
        # Position tracking
        self.position_size = 0.0  # Current position size in BTC
        self.entry_price = 0.0  # Average entry price of current position
        self.position_value = 0.0  # Current value of the position
        
        # Risk management parameters
        self.max_position_size = 0.1  # Maximum position size in BTC
        self.risk_per_trade = 0.02  # Risk 2% of balance per trade
        
        # Initialize scalers for adaptive normalization
        self.feature_scaler = StandardScaler()
        self.price_scaler = MinMaxScaler()
        self.scalers_fitted = False
        
        # Load persistent state
        self.load_state()
        
        # Initialize model
        self.model = self.load_model()
        if self.model is None:
            self.model = self.create_lstm_model()
        
        # Training data management
        self.training_data = self.load_training_data()
        self.feature_buffer = deque(maxlen=self.sequence_length)
        self.last_save_time = time.time()
        self.last_training_time = 0
        
        # Load scalers if they exist
        self.load_scalers()
        
        logger.info(f"AutoTrader initialized with balance: {self.balance:.2f} AUD")
        logger.info(f"Loaded {len(self.training_data)} historical data points")
        logger.info(f"Using sequence length: {self.sequence_length} for LSTM")

    def save_state(self):
        """Save the current state of the trader to a file."""
        try:
            state = {
                'balance': self.balance,
                'position_size': self.position_size,
                'entry_price': self.entry_price,
                'position_value': self.position_value,
                'last_save_time': self.last_save_time,
                'last_training_time': self.last_training_time,
                'scalers_fitted': self.scalers_fitted
            }
            with open(self.state_filename, 'wb') as f:
                pickle.dump(state, f)
            logger.info("Trader state saved successfully")
        except Exception as e:
            logger.error(f"Error saving trader state: {e}")

    def load_state(self):
        """Load the previous state of the trader."""
        try:
            with open(self.state_filename, 'rb') as f:
                state = pickle.load(f)
            self.balance = state.get('balance', self.balance)
            self.position_size = state.get('position_size', 0.0)
            self.entry_price = state.get('entry_price', 0.0)
            self.position_value = state.get('position_value', 0.0)
            self.last_save_time = state.get('last_save_time', time.time())
            self.last_training_time = state.get('last_training_time', 0)
            self.scalers_fitted = state.get('scalers_fitted', False)
            logger.info(f"Trader state loaded. Balance: {self.balance:.2f} AUD")
            if self.position_size != 0:
                logger.info(f"Loaded position: {self.position_size:.6f} BTC at ${self.entry_price:.2f}")
        except FileNotFoundError:
            logger.info("No previous state found. Starting fresh.")
        except Exception as e:
            logger.error(f"Error loading trader state: {e}")

    def save_scalers(self):
        """Save the fitted scalers."""
        try:
            scalers = {
                'feature_scaler': self.feature_scaler,
                'price_scaler': self.price_scaler,
                'scalers_fitted': self.scalers_fitted
            }
            with open(self.scalers_filename, 'wb') as f:
                pickle.dump(scalers, f)
            logger.info("Scalers saved successfully")
        except Exception as e:
            logger.error(f"Error saving scalers: {e}")

    def load_scalers(self):
        """Load the fitted scalers."""
        try:
            with open(self.scalers_filename, 'rb') as f:
                scalers = pickle.load(f)
            self.feature_scaler = scalers['feature_scaler']
            self.price_scaler = scalers['price_scaler']
            self.scalers_fitted = scalers['scalers_fitted']
            logger.info("Scalers loaded successfully")
        except FileNotFoundError:
            logger.info("No scalers file found, will create new ones")
        except Exception as e:
            logger.error(f"Error loading scalers: {e}")

    def fetch_market_data(self) -> Optional[List[Dict]]:
        """
        Fetch live market data with better error handling and performance optimizations.
        Uses session reuse and faster timeouts for better performance.
        """
        if not hasattr(self, '_session'):
            self._session = requests.Session()
            self._session.headers.update({
                'User-Agent': 'AutoTrader/1.0',
                'Accept': 'application/json'
            })
        
        url = 'https://api.btcmarkets.net/v3/markets/tickers?marketId=BTC-AUD'
        
        max_retries = 2  # Reduced from 3 to 2 for faster failure
        for attempt in range(max_retries):
            try:
                # Use a faster timeout for the request (5s instead of 10s)
                with self._session.get(url, timeout=(3.0, 5.0)) as response:
                    response.raise_for_status()
                    data = response.json()
                    if not data or not isinstance(data, list):
                        raise ValueError("Invalid response format")
                    return data
            except (requests.exceptions.RequestException, ValueError) as e:
                if attempt == max_retries - 1:  # Last attempt
                    logger.warning(f"Failed to fetch market data after {max_retries} attempts: {e}")
                    return None
                # Short delay before retry (reduced from exponential backoff)
                time.sleep(0.5 * (attempt + 1))

    def extract_comprehensive_data(self, data: List[Dict]) -> Optional[Dict]:
        """Extract comprehensive market data including OHLCV."""
        if not data or not isinstance(data, list):
            return None
        
        btc_aud_data = next((item for item in data if item.get('marketId') == 'BTC-AUD'), None)
        if not btc_aud_data:
            return None
        
        try:
            return {
                'price': float(btc_aud_data.get('lastPrice', 0)),
                'volume': float(btc_aud_data.get('volume24h', 0)),
                'bid': float(btc_aud_data.get('bestBid', 0)),
                'ask': float(btc_aud_data.get('bestAsk', 0)),
                'high24h': float(btc_aud_data.get('high24h', 0)),
                'low24h': float(btc_aud_data.get('low24h', 0))
            }
        except (ValueError, TypeError) as e:
            logger.warning(f"Error converting market data to float: {e}")
            return None

    def manual_sma(self, prices: np.ndarray, period: int) -> float:
        """Manual Simple Moving Average calculation."""
        if len(prices) < period:
            return prices[-1] if len(prices) > 0 else 0
        return np.mean(prices[-period:])

    def manual_ema(self, prices: np.ndarray, period: int) -> float:
        """Manual Exponential Moving Average calculation."""
        if len(prices) < period:
            return prices[-1] if len(prices) > 0 else 0
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema

    def manual_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Manual RSI calculation with improved handling of edge cases."""
        try:
            if len(prices) < period + 1:
                return 50.0  # Neutral RSI if not enough data
                
            deltas = np.diff(prices)
            
            # Handle case where there's no price movement
            if np.all(deltas == 0):
                return 50.0
                
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            # Calculate average gain and loss using exponential moving average
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
            for i in range(period, len(gains)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            # Avoid division by zero
            if avg_loss < 1e-10:
                return 100.0 if avg_gain > 1e-10 else 50.0
                
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Ensure RSI is within valid range
            return max(0.0, min(100.0, rsi))
            
        except Exception as e:
            logger.warning(f"Error in RSI calculation: {e}")
            return 50.0  # Return neutral RSI on error

    def calculate_technical_indicators(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, float]:
        """
        Calculate technical indicators using TA-Lib or manual calculations.
        Returns a dictionary of indicator values with proper error handling.
        """
        # Default values for indicators
        default_indicators = {
            'sma_5': 0.0, 'sma_20': 0.0, 'ema_12': 0.0, 'ema_26': 0.0,
            'rsi': 50.0, 'macd': 0.0, 'macd_signal': 0.0, 'bb_upper': 0.0,
            'bb_lower': 0.0, 'volume_sma': 0.0
        }
        
        # Return defaults if not enough data
        if len(prices) < 5:  # Minimum needed for any indicators
            logger.warning(f"Insufficient data points for indicators: {len(prices)} < 5")
            return default_indicators
        
        try:
            # Ensure inputs are numpy arrays
            prices = np.asarray(prices, dtype=np.float64)
            volumes = np.asarray(volumes, dtype=np.float64)
            
            # Calculate indicators
            indicators = {}
            
            # Simple Moving Averages
            indicators['sma_5'] = float(self.manual_sma(prices, 5) if len(prices) >= 5 else prices[-1])
            indicators['sma_20'] = float(self.manual_sma(prices, 20) if len(prices) >= 20 else prices[-1])
            
            # Exponential Moving Averages
            indicators['ema_12'] = float(self.manual_ema(prices, 12) if len(prices) >= 12 else prices[-1])
            indicators['ema_26'] = float(self.manual_ema(prices, 26) if len(prices) >= 26 else prices[-1])
            
            # RSI
            indicators['rsi'] = float(self.manual_rsi(prices, 14))
            
            # MACD
            macd_val = indicators['ema_12'] - indicators['ema_26']
            macd_signal = self.manual_ema(np.array([macd_val]), 9)
            indicators['macd'] = float(macd_val)
            indicators['macd_signal'] = float(macd_signal)
            
            # Bollinger Bands
            bb_middle = indicators['sma_20']
            bb_std = np.std(prices[-20:]) if len(prices) >= 20 else 0
            indicators['bb_upper'] = float(bb_middle + (2 * bb_std))
            indicators['bb_lower'] = float(bb_middle - (2 * bb_std))
            
            # Volume SMA
            indicators['volume_sma'] = float(self.manual_sma(volumes, 10) if len(volumes) >= 10 else volumes[-1])
            
            # Log indicator values for debugging
            logger.debug(
                f"Indicators - "
                f"Price: {prices[-1]:.2f}, "
                f"RSI: {indicators['rsi']:.2f}, "
                f"MACD: {indicators['macd']:.2f}, "
                f"Signal: {indicators['macd_signal']:.2f}, "
                f"BB: {indicators['bb_upper']:.2f}/{indicators['bb_lower']:.2f}"
            )
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}", exc_info=True)
            return default_indicators

    def collect_and_store_data(self) -> bool:
        """
        Collect current market data and add to training dataset.
        Optimized for performance by reducing unnecessary calculations.
        """
        try:
            market_data = self.fetch_market_data()
            if not market_data:
                logger.warning("No market data received from API")
                return False
                
            comprehensive_data = self.extract_comprehensive_data(market_data)
            if not comprehensive_data or comprehensive_data.get('price', 0) <= 0:
                logger.warning(f"Invalid comprehensive data: {comprehensive_data}")
                return False
                
            # Ensure we have all required fields
            required_fields = ['price', 'volume', 'bid', 'ask', 'high24h', 'low24h']
            if not all(field in comprehensive_data for field in required_fields):
                logger.warning(f"Missing required fields in market data: {comprehensive_data}")
                return False
            
            # Only calculate indicators if we have enough data
            indicators = {}
            if len(self.training_data) >= 20:
                try:
                    # Use most recent data points for indicators
                    recent_data = self.training_data[-20:]
                    recent_prices = np.array([dp['price'] for dp in recent_data if 'price' in dp])
                    recent_volumes = np.array([dp['volume'] for dp in recent_data if 'volume' in dp])
                    
                    if len(recent_prices) >= 20 and len(recent_volumes) >= 20:
                        indicators = self.calculate_technical_indicators(
                            recent_prices, 
                            recent_volumes
                        )
                except Exception as e:
                    logger.warning(f"Error calculating technical indicators: {e}")
            
            # Create and store data point
            data_point = {
                'timestamp': int(time.time()),
                **comprehensive_data,
                **indicators
            }
            
            # Add to training data (using deque for O(1) append/pop from both ends)
            if not hasattr(self, '_training_data_deque'):
                self._training_data_deque = deque(self.training_data, maxlen=self.max_training_samples)
            
            self._training_data_deque.append(data_point)
            self.training_data = list(self._training_data_deque)
            
            # Fit scalers on first run or if not fitted yet
            if len(self.training_data) >= self.min_data_points:
                if not self.scalers_fitted:
                    logger.info("Fitting scalers for the first time")
                    if not self.fit_scalers(self.training_data):
                        logger.error("Failed to fit scalers")
                        return False
                # Ensure feature buffer is populated for predictions
                if not self.feature_buffer:
                    self._update_feature_buffer()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in collect_and_store_data: {e}", exc_info=True)
            return False

    def create_lstm_model(self):
        """Create a new LSTM model for sequential data."""
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.sequence_length, 12), name='input_layer'),
            tf.keras.layers.LSTM(64, return_sequences=True, name='lstm_1'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3, name='dropout_1'),
            tf.keras.layers.LSTM(32, return_sequences=False, name='lstm_2'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3, name='dropout_2'),
            tf.keras.layers.Dense(16, activation='relu', name='dense_1'),
            tf.keras.layers.Dense(1, activation='sigmoid', name='output')
        ], name='lstm_model')
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        logger.info("New LSTM model created with input shape (None, %d, 12)", self.sequence_length)
        model.summary(print_fn=logger.info)
        return model

    def prepare_features(self, data_point: Dict) -> np.ndarray:
        """Prepare feature vector from a data point."""
        return np.array([
            data_point.get('price', 0),
            data_point.get('volume', 0),
            data_point.get('spread', 0),
            data_point.get('sma_5', 0),
            data_point.get('sma_20', 0),
            data_point.get('ema_12', 0),
            data_point.get('ema_26', 0),
            data_point.get('rsi', 50),
            data_point.get('macd', 0),
            data_point.get('macd_signal', 0),
            data_point.get('bb_upper', 0),
            data_point.get('bb_lower', 0)
        ])

    def _update_feature_buffer(self) -> None:
        """Update the feature buffer with the most recent data points."""
        if len(self.training_data) >= self.sequence_length:
            # Get the most recent sequence of data points
            recent_data = self.training_data[-self.sequence_length:]
            self.feature_buffer.clear()
            for data_point in recent_data:
                feature_vector = self.prepare_features(data_point)
                if feature_vector is not None:
                    self.feature_buffer.append(feature_vector)
            logger.debug(f"Updated feature buffer with {len(self.feature_buffer)} data points")
            
    def fit_scalers(self, training_data: List[Dict]) -> bool:
        """Fit scalers to the training data."""
        # Filter only valid dictionary entries
        valid_data = [dp for dp in training_data if isinstance(dp, dict) and 'price' in dp and dp['price'] > 0]
        
        if len(valid_data) < self.min_data_points:
            logger.warning(f"Not enough valid data to fit scalers properly (have {len(valid_data)}, need {self.min_data_points})")
            return False
        
        try:
            # Prepare feature matrix
            features = []
            prices = []
            
            for data_point in valid_data:
                feature_vector = self.prepare_features(data_point)
                features.append(feature_vector)
                prices.append([data_point.get('price', 0)])
            
            features = np.array(features)
            prices = np.array(prices)
            
            # Fit scalers
            self.feature_scaler.fit(features)
            self.price_scaler.fit(prices)
            self.scalers_fitted = True
            
            logger.info(f"Scalers fitted to {len(valid_data)} training data points")
            return True
            
        except Exception as e:
            logger.error(f"Error fitting scalers: {e}")
            return False

    def prepare_lstm_training_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare sequential training data for LSTM with proper future prediction labeling."""
        try:
            if len(self.training_data) <= self.sequence_length:
                logger.warning(f"Not enough data points. Have {len(self.training_data)}, need at least {self.sequence_length + 1}")
                return None, None
                
            sequences = []
            labels = []
            
            # Convert to list for easier slicing and filter valid data
            valid_data = [d for d in self.training_data if 'price' in d and 'volume' in d and d['price'] > 0]
            
            if len(valid_data) <= self.sequence_length:
                logger.warning(f"Not enough valid data points. Have {len(valid_data)}, need at least {self.sequence_length + 1}")
                return None, None
            
            logger.info(f"Preparing LSTM training data from {len(valid_data)} valid data points")
            
            for i in range(len(valid_data) - self.sequence_length):
                # Get sequence of data points
                sequence = valid_data[i:i + self.sequence_length]
                
                # Extract features for each point in the sequence
                sequence_features = []
                for j in range(len(sequence)):
                    feature_vector = self.prepare_features(sequence[j])
                    sequence_features.append(feature_vector)
                
                # The label is whether the price went up after the sequence
                current_price = valid_data[i + self.sequence_length - 1]['price']
                future_price = valid_data[i + self.sequence_length]['price']
                label = 1 if future_price > current_price else 0
                
                sequences.append(sequence_features)
                labels.append(label)
            
            if not sequences:
                logger.warning("No valid sequences created")
                return None, None
                
            # Convert to numpy arrays
            X = np.array(sequences, dtype=np.float32)
            y = np.array(labels, dtype=np.float32)
            
            logger.info(f"Created {len(X)} sequences with shape {X.shape} and {len(y)} labels")
            
            # Scale the features if we have enough data
            if len(X) > 0 and hasattr(self, 'feature_scaler') and self.scalers_fitted:
                try:
                    n_samples, seq_len, n_features = X.shape
                    X_reshaped = X.reshape(-1, n_features)
                    X_scaled = self.feature_scaler.transform(X_reshaped)
                    X = X_scaled.reshape(n_samples, seq_len, n_features)
                    logger.debug(f"Successfully scaled features to shape {X.shape}")
                except Exception as e:
                    logger.error(f"Error scaling features: {e}")
                    return None, None
            else:
                logger.warning("Skipping feature scaling - scaler not fitted")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error in prepare_lstm_training_data: {e}")
            return None, None

    def train_model(self):
        """Train the LSTM model with accumulated sequential data."""
        try:
            sequences, labels = self.prepare_lstm_training_data()
            if sequences is None or len(sequences) < 20:
                logger.warning("Not enough sequential data for LSTM training")
                return False
            
            # Train the model
            history = self.model.fit(
                sequences, labels,
                epochs=10,
                batch_size=16,
                validation_split=0.2,
                verbose=0,
                shuffle=False  # Don't shuffle time series data
            )
            
            loss = history.history['loss'][-1]
            accuracy = history.history.get('accuracy', [0])[-1]
            val_loss = history.history.get('val_loss', [loss])[-1]
            val_accuracy = history.history.get('val_accuracy', [accuracy])[-1]
            
            logger.info(f"LSTM trained - Loss: {loss:.4f}, Acc: {accuracy:.4f}, Val_Loss: {val_loss:.4f}, Val_Acc: {val_accuracy:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return False

    def calculate_position_pnl(self, current_price: float) -> Tuple[float, float]:
        """Calculate current profit/loss for the position."""
        if self.position_size <= 0 or self.entry_price <= 0:
            return 0.0, 0.0
            
        current_value = self.position_size * current_price
        cost_basis = self.position_size * self.entry_price
        pnl = current_value - cost_basis
        pnl_pct = (pnl / cost_basis) * 100 if cost_basis > 0 else 0
        return pnl, pnl_pct

    def _generate_test_data(self, num_points: int) -> None:
        """Generate test data points for testing purposes with all required fields."""
        if num_points <= 0:
            return
            
        logger.info(f"Generating {num_points} test data points...")
        last_price = 50000.0  # Starting price
        
        for i in range(num_points):
            # Simulate some price movement
            price_change = np.random.normal(0, 100)  # Random walk
            last_price = max(1000.0, last_price + price_change)  # Ensure positive price
            
            # Generate all required technical indicators with realistic relationships
            sma_5 = last_price * (1 + np.random.normal(0, 0.005))
            sma_20 = last_price * (1 + np.random.normal(0, 0.01))
            ema_12 = last_price * (1 + np.random.normal(0, 0.004))
            ema_26 = last_price * (1 + np.random.normal(0, 0.008))
            rsi = np.random.uniform(30, 70)
            macd = np.random.normal(0, 10)
            macd_signal = macd * 0.9 + np.random.normal(0, 2)
            bb_upper = last_price * 1.01 + np.random.normal(0, 50)
            bb_lower = last_price * 0.99 - np.random.normal(0, 50)
            
            data_point = {
                'timestamp': int(time.time()) + i,
                'lastPrice': last_price,  # Used by prediction
                'price': last_price,      # Used by feature preparation
                'volume': np.random.uniform(1, 10),
                'bid': last_price * 0.999,
                'ask': last_price * 1.001,
                'high24h': last_price * 1.01,
                'low24h': last_price * 0.99,
                'sma_5': sma_5,
                'sma_20': sma_20,
                'ema_12': ema_12,
                'ema_26': ema_26,
                'rsi': rsi,
                'macd': macd,
                'macd_signal': macd_signal,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'spread': (last_price * 1.001) - (last_price * 0.999)  # ask - bid
            }
            
            if not hasattr(self, '_training_data_deque'):
                self._training_data_deque = deque(maxlen=self.max_training_samples)
                
            self._training_data_deque.append(data_point)
            
        self.training_data = list(self._training_data_deque)
        logger.info(f"Generated {num_points} test data points. Total data points: {len(self.training_data)}")
    
    def predict_trade_signal(self, market_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict trading signal using LSTM model with sequential data."""
        logger.debug(f"predict_trade_signal called with {len(market_data) if market_data else 0} data points")
        
        if self.test_mode and len(self.training_data) >= self.min_data_points:
            # In test mode, use the training data for prediction if we don't have enough market data
            if not market_data or len(market_data) < self.min_data_points:
                logger.debug(f"Using {len(self.training_data)} training data points for prediction in test mode")
                market_data = self.training_data[-self.min_data_points:]
        
        if not market_data or len(market_data) < self.min_data_points:
            logger.warning(f"Not enough market data for prediction. Have {len(market_data) if market_data else 0}, need {self.min_data_points}")
            return {"signal": "HOLD", "confidence": 0.5, "price": 0.0, "rsi": 50.0, "valid": False}

        try:
            logger.debug(f"Making prediction with {len(market_data)} data points")
            # Get the latest data point for current price
            current_data = market_data[-1]
            
            # Try to get price from either 'lastPrice' or 'price' field
            current_price = None
            for price_field in ['lastPrice', 'price']:
                price = current_data.get(price_field)
                if price is not None:
                    try:
                        current_price = float(price)
                        if current_price > 0:
                            break
                    except (ValueError, TypeError):
                        continue
            
            if current_price is None or current_price <= 0:
                logger.warning(f"No valid price found in market data. Available keys: {list(current_data.keys())}")
                return {"signal": "HOLD", "confidence": 0.5, "price": 0.0, "rsi": 50.0, "valid": False}
            
            # Calculate RSI from recent price data
            price_data = []
            for m in market_data:
                for price_field in ['lastPrice', 'price']:
                    price = m.get(price_field)
                    if price is not None:
                        try:
                            price_float = float(price)
                            if price_float > 0:
                                price_data.append(price_float)
                                break
                        except (ValueError, TypeError):
                            continue
            
            if len(price_data) >= 15:  # Need at least 15 points for 14-period RSI
                price_array = np.array(price_data[-15:])  # Use last 15 points for 14-period RSI
                rsi = float(self.manual_rsi(price_array))
            else:
                rsi = 50.0  # Neutral RSI if not enough data
            
            # Prepare features for the model
            features = []
            for i in range(len(market_data) - self.sequence_length, len(market_data)):
                data_point = market_data[i]
                feature_vector = self.prepare_features(data_point)
                if feature_vector is not None:
                    features.append(feature_vector)
            
            if not features:
                return {"signal": "HOLD", "confidence": 0.5, "price": current_price, "rsi": rsi, "valid": False}
            
            features_array = np.array([features])  # Add batch dimension
            
            # Make prediction
            try:
                prediction = float(self.model.predict(features_array, verbose=0)[0][0])
                confidence = abs(prediction - 0.5) * 2  # Convert to 0-1 range
            except Exception as e:
                logger.error(f"Error in model prediction: {e}")
                return {"signal": "HOLD", "confidence": 0.5, "price": current_price, "rsi": rsi, "valid": False}
            
            # Adjust prediction based on RSI for extreme conditions
            if rsi > 80 and prediction > 0.7:  # Overbought
                prediction = max(0.5, prediction * 0.9)  # Reduce confidence in BUY signal
                logger.debug(f"Overbought market (RSI: {rsi:.1f}), reducing BUY confidence")
            elif rsi < 20 and prediction < 0.3:  # Oversold
                prediction = min(0.5, prediction * 1.1)  # Reduce confidence in SELL signal
                logger.debug(f"Oversold market (RSI: {rsi:.1f}), reducing SELL confidence")
            
            # Determine signal based on prediction
            if prediction > 0.6:  # More confident threshold for BUY
                signal = "BUY"
            elif prediction < 0.4:  # More confident threshold for SELL
                signal = "SELL"
            else:
                signal = "HOLD"
            
            # Check if we already have a position and modify signal accordingly
            if self.position_size > 0:
                try:
                    _, pnl_pct = self.calculate_position_pnl(current_price)
                    if signal == "BUY":  # Don't buy if we already have a position
                        signal = "HOLD"
                        logger.debug("Already in position, converting BUY to HOLD")
                    elif signal == "SELL" and pnl_pct < -5:  # Add stop loss check
                        logger.info(f"Stop loss triggered at {pnl_pct:.1f}% loss")
                    elif signal == "SELL" and pnl_pct > 10:  # Take profit at 10%
                        logger.info(f"Take profit triggered at {pnl_pct:.1f}% gain")
                except Exception as e:
                    logger.error(f"Error calculating position P&L: {e}")
            
            logger.debug(
                f"Prediction: {prediction:.3f}, Signal: {signal}, "
                f"Confidence: {confidence:.3f}, RSI: {rsi:.1f}"
            )
            
            return {
                "signal": signal,
                "confidence": confidence,
                "price": current_price,
                "rsi": rsi,
                "valid": True  # Mark as valid since we successfully generated a prediction
            }
            
        except Exception as e:
            logger.error(f"Error making LSTM prediction: {e}")
            return {"signal": "HOLD", "confidence": 0.5, "price": 0, "rsi": 50.0}

    def calculate_position_size(self, price: float, stop_loss_price: float = None) -> float:
        """Calculate position size based on risk management rules."""
        if stop_loss_price is None:
            stop_loss_pct = 0.02  # Default 2% stop loss if not specified
        else:
            stop_loss_pct = abs(price - stop_loss_price) / price
            
        if stop_loss_pct == 0:
            stop_loss_pct = 0.02  # Prevent division by zero
            
        risk_amount = self.balance * self.risk_per_trade
        position_size = risk_amount / (stop_loss_pct * price)
        
        # Cap position size to max allowed
        position_size = min(position_size, self.max_position_size)
        
        # Ensure we don't try to buy more than we can afford
        max_affordable = (self.balance * 0.99) / price  # Leave 1% buffer for fees
        position_size = min(position_size, max_affordable)
        
        return position_size

    def execute_simulated_trade(self, prediction: Dict[str, Any]) -> None:
        """
        Execute simulated trades based on LSTM predictions with enhanced position management.
        
        Args:
            prediction: Dictionary containing trade signal, confidence, price, and RSI
        """
        logger.debug(f"execute_simulated_trade called with prediction: {prediction}")
        
        if not prediction:
            logger.warning("No prediction data provided, no trade executed")
            return
            
        if prediction.get("price", 0) <= 0:
            logger.warning(f"Invalid price in prediction: {prediction.get('price')}")
            return
            
        if not prediction.get("valid", False):
            logger.warning("Prediction marked as invalid, no trade executed")
            return
        
        fee_rate = 0.001  # 0.1% trading fee
        price = float(prediction["price"])
        signal = prediction["signal"]
        confidence = float(prediction["confidence"])
        rsi = float(prediction.get("rsi", 50))
        
        # Log current position status
        if self.position_size > 0:
            pnl, pnl_pct = self.calculate_position_pnl(price)
            logger.info(
                f"Position: {self.position_size:.6f} BTC @ ${self.entry_price:.2f} | "
                f"Current: ${price:.2f} | P&L: ${pnl:.2f} ({pnl_pct:.2f}%)"
            )
        
        # Skip if confidence is too low
        min_confidence = 0.55  # Require at least 55% confidence to trade
        if confidence < min_confidence:
            logger.info(f"Confidence too low ({confidence:.3f} < {min_confidence:.2f}), holding position")
            return
        
        # Adjust signal based on RSI for better risk management
        rsi_overbought = 75
        rsi_oversold = 25
        
        if signal == "BUY" and rsi > rsi_overbought:
            logger.info(f"RSI high ({rsi:.1f} > {rsi_overbought}), avoiding BUY despite model confidence: {confidence:.3f}")
            signal = "HOLD"
        elif signal == "SELL" and rsi < rsi_oversold:
            logger.info(f"RSI low ({rsi:.1f} < {rsi_oversold}), avoiding SELL despite model confidence: {confidence:.3f}")
            signal = "HOLD"
        
        try:
            # Handle BUY signal
            if signal == "BUY" and self.position_size <= 0:  # Only buy if we don't already have a position
                # Calculate position size with 2% stop loss
                stop_loss_pct = 0.02
                stop_loss_price = price * (1 - stop_loss_pct)
                position_size = self.calculate_position_size(price, stop_loss_price)
                
                # Minimum trade size check
                min_trade_size = 0.0001  # Minimum 0.0001 BTC
                if position_size < min_trade_size:
                    logger.info(f"Position size too small: {position_size:.6f} BTC, minimum is {min_trade_size} BTC")
                    return
                
                # Calculate fees and total cost
                trade_value = price * position_size
                fee = trade_value * fee_rate
                total_cost = trade_value + fee
                
                if self.balance >= total_cost:
                    # Execute BUY order
                    self.balance -= total_cost
                    self.position_size = position_size
                    self.entry_price = price
                    self.position_value = trade_value
                    
                    logger.info(
                        f"✅ BUY executed: {position_size:.6f} BTC @ ${price:.2f} | "
                        f"Cost: ${total_cost:.2f} (Fee: ${fee:.2f}) | "
                        f"RSI: {rsi:.1f} | Confidence: {confidence:.3f}"
                    )
                    logger.info(f"💰 New position: {self.position_size:.6f} BTC @ ${self.entry_price:.2f} | Balance: ${self.balance:.2f}")
                    logger.info(f"Available balance: ${self.balance:.2f}")
                else:
                    logger.warning(
                        f"Insufficient balance for BUY. Needed: ${total_cost:.2f}, "
                        f"Available: ${self.balance:.2f}"
                    )
                    
            # Handle SELL signal
            elif signal == "SELL" and self.position_size > 0:  # Only sell if we have a position
                # Calculate position P&L
                pnl, pnl_pct = self.calculate_position_pnl(price)
                
                # Calculate trade value and fees
                trade_value = price * self.position_size
                fee = trade_value * fee_rate
                proceeds = trade_value - fee
                
                # Update balance and reset position
                self.balance += proceeds
                position_size = self.position_size  # Store for logging before reset
                self.position_size = 0.0
                self.position_value = 0.0
                
                # Determine if this was a profitable trade
                trade_result = "✅ PROFIT" if pnl >= 0 else "❌ LOSS"
                
                logger.info(
                    f"🔄 {trade_result} SELL executed: {position_size:.6f} BTC @ ${price:.2f} | "
                    f"P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%) | "
                    f"Fee: ${fee:.2f} | "
                    f"RSI: {rsi:.1f} | Confidence: {confidence:.3f}"
                )
                logger.info(f"💰 New balance: ${self.balance:.2f}")
                
                # Reset entry price after selling
                self.entry_price = 0.0
                
            # Handle HOLD signal or no action needed
            elif signal == "HOLD":
                if self.position_size > 0:
                    current_value = self.position_size * price
                    profit_loss = current_value - (self.entry_price * self.position_size)
                    profit_loss_pct = (profit_loss / (self.entry_price * self.position_size)) * 100 if self.entry_price > 0 else 0
                    
                    # Check for stop loss or take profit
                    if profit_loss_pct <= -5:  # 5% stop loss
                        logger.warning(f"⚠️  STOP LOSS triggered at {profit_loss_pct:.2f}% loss"
                                    f" (Entry: ${self.entry_price:.2f}, Current: ${price:.2f})")
                        signal = "SELL"  # Trigger sell on next iteration
                    elif profit_loss_pct >= 10:  # 10% take profit
                        logger.warning(f"🎯 TAKE PROFIT triggered at {profit_loss_pct:.2f}% profit"
                                    f" (Entry: ${self.entry_price:.2f}, Current: ${price:.2f})")
                        signal = "SELL"  # Trigger sell on next iteration
                    else:
                        logger.info(
                            f"📊 HOLDING - Position: {self.position_size:.6f} BTC @ ${self.entry_price:.2f} | "
                            f"Current: ${price:.2f} | "
                            f"P&L: ${profit_loss:+.2f} ({profit_loss_pct:+.2f}%) | "
                            f"RSI: {rsi:.1f}"
                        )
                else:
                    logger.info(f"⏸️  HOLD - No position. Price: ${price:.2f}, RSI: {rsi:.1f}")
            
            # Log current account status
            logger.info(
                f"💵 Account Summary | "
                f"Balance: ${self.balance:.2f} | "
                f"Position: {self.position_size:.6f} BTC | "
                f"Total Value: ${self.balance + (self.position_size * price):.2f}"
            )
            
            # Save state after each trade
            self.save_state()
            
        except Exception as e:
            logger.error(f"❌ Error executing trade: {e}")
            # Try to save state even if there was an error
            try:
                self.save_state()
            except Exception as save_error:
                logger.error(f"❌ Failed to save state after error: {save_error}")

    def save_training_data(self):
        """Save training data to JSON file."""
        try:
            with open(self.training_data_filename, "w") as f:
                json.dump(self.training_data, f, indent=2)
            logger.info(f"Training data saved ({len(self.training_data)} samples)")
        except Exception as e:
            logger.error(f"Error saving training data: {e}")

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

    def save_model(self):
        """Save the TensorFlow model."""
        try:
            self.model.save(self.model_filename)
            logger.info("LSTM model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

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
            return None
        except Exception as e:
            logger.error("Error loading model: %s. Creating new model.", str(e))
            return None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

    def should_save(self) -> bool:
        """Check if it's time to save data and model."""
        return time.time() - self.last_save_time >= self.save_interval_seconds

    def should_train(self) -> bool:
        """Check if it's time to retrain the model."""
        return time.time() - self.last_training_time >= self.training_interval_seconds

    def run_continuous_trading(self):
        """Main loop for continuous LSTM-based trading operation."""
        if self.test_mode:
            logger.info(f"Starting test mode with {self.test_iterations} iterations...")
            # In test mode, use existing data if we have enough
            if len(self.training_data) >= self.min_data_points:
                logger.info(f"Using existing {len(self.training_data)} data points for testing")
            else:
                # Generate test data if not enough
                self._generate_test_data(self.min_data_points - len(self.training_data))
                logger.info(f"Generated {len(self.training_data)} test data points")
        else:
            logger.info("Starting continuous LSTM trading operation...")
        
        iteration_count = 0
        last_train_count = 0
        last_save_count = 0
        
        # Initial data collection phase (only in live mode)
        if not self.test_mode:
            logger.info("Initial data collection phase...")
            while len(self.training_data) < self.min_data_points:
                if self.collect_and_store_data():
                    logger.info(f"Collected {len(self.training_data)}/{self.min_data_points} initial data points")
                time.sleep(5)  # Wait between data collection attempts
            
            logger.info(f"Initial data collection complete. Starting trading with {len(self.training_data)} data points")
        
        # Cache market data to avoid redundant API calls in test mode
        cached_market_data = None
        
        while True:
            try:
                iteration_count += 1
                start_time = time.time()
                
                # Only log every 10 iterations to reduce noise
                if iteration_count % 10 == 1:
                    logger.info(f"--- Iteration {iteration_count} ---")
                
                # In test mode, use existing data if we have enough
                if self.test_mode:
                    if len(self.training_data) >= self.min_data_points:
                        current_market_data = self.training_data[-self.min_data_points:]
                    else:
                        # Generate more test data if needed
                        needed = self.min_data_points - len(self.training_data)
                        self._generate_test_data(needed)
                        current_market_data = self.training_data[-self.min_data_points:]
                else:
                    # In live mode, fetch new market data
                    current_market_data = self.fetch_market_data()
                    if not current_market_data and cached_market_data:
                        logger.warning("Using cached market data")
                        current_market_data = cached_market_data
                    elif not current_market_data:
                        logger.warning("No market data available")
                        time.sleep(5)
                        continue
                    
                    # Store the new data point
                    if self.collect_and_store_data():
                        cached_market_data = current_market_data
                        logger.debug(f"Collected new market data. Total samples: {len(self.training_data)}")
                
                # Make trading prediction and execute if we have data
                if len(self.training_data) >= self.min_data_points:
                    # Use the most recent data points from training data for prediction
                    prediction_data = self.training_data[-self.min_data_points:]
                    prediction = self.predict_trade_signal(prediction_data)
                    
                    # Log prediction details for debugging
                    if prediction.get('valid', False):
                        logger.info(f"Prediction - Signal: {prediction['signal']}, "
                                    f"Confidence: {prediction['confidence']:.2f}, "
                                    f"Price: {prediction['price']:.2f}, "
                                    f"RSI: {prediction['rsi']:.1f}")
                    
                    self.execute_simulated_trade(prediction)
                else:
                    logger.warning(f"Not enough training data for prediction. Have {len(self.training_data)}, need {self.min_data_points}")
                
                # Retrain model less frequently in test mode
                train_interval = 5 if self.test_mode else 60  # Train every 5 iterations in test mode
                if (iteration_count - last_train_count >= train_interval and 
                    len(self.training_data) >= self.sequence_length + 20):
                    logger.info("Retraining LSTM model with new sequential data...")
                    if self.train_model():
                        self.last_training_time = time.time()
                        last_train_count = iteration_count
                
                # Save less frequently in test mode
                save_interval = 10 if self.test_mode else 600  # Save every 10 iterations in test mode
                if iteration_count - last_save_count >= save_interval:
                    logger.info("Saving data, model, and scalers...")
                    self.save_training_data()
                    self.save_model()
                    self.save_scalers()
                    self.save_state()
                    self.last_save_time = time.time()
                    last_save_count = iteration_count
                
                # Log status periodically
                if iteration_count % 20 == 0:
                    avg_rsi = np.mean([dp.get('rsi', 50) for dp in self.training_data[-10:] if 'rsi' in dp]) if len(self.training_data) >= 10 else 50
                    logger.info(
                        f"Status - Iteration: {iteration_count}, "
                        f"Balance: {self.balance:.2f} AUD, "
                        f"Samples: {len(self.training_data)}, "
                        f"Avg RSI: {avg_rsi:.1f}"
                    )
                
                # Check if we should exit in test mode
                if self.test_mode and iteration_count >= self.test_iterations:
                    logger.info(f"Test mode complete after {iteration_count} iterations")
                    # Force save before exiting
                    logger.info("Saving data before exit...")
                    self.save_training_data()
                    self.save_model()
                    self.save_scalers()
                    self.save_state()
                    break
                
                # Calculate time taken for this iteration
                iteration_time = time.time() - start_time
                
                # Adjust sleep time based on iteration time
                target_interval = 0.5 if self.test_mode else 5.0  # Aim for 2 iterations per second in test mode
                sleep_time = max(0.1, target_interval - iteration_time)
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                logger.info("Received shutdown signal, saving all data...")
                self.save_training_data()
                self.save_model()
                self.save_scalers()
                self.save_state()
                logger.info("Shutdown complete")
                break
            except Exception as e:
                logger.error(f"Unexpected error in trading loop: {e}")
                time.sleep(60)  # Wait before retrying


if __name__ == "__main__":
    import sys
    test_mode = '--test' in sys.argv
    trader = ContinuousAutoTrader(test_mode=test_mode)
    trader.run_continuous_trading()
