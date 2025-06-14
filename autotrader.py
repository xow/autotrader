import requests
from datetime import datetime
import tensorflow as tf
import numpy as np
import json
import os
import time
import logging
from typing import List, Dict, Optional, Tuple
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import deque
import pandas as pd

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
        
        # Test mode settings
        self.test_mode = test_mode
        self.test_iterations = test_iterations
        
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
        state = {
            'balance': self.balance,
            'last_save_time': self.last_save_time,
            'last_training_time': self.last_training_time,
            'training_data_length': len(self.training_data),
            'scalers_fitted': self.scalers_fitted
        }
        try:
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
            self.balance = state.get('balance', 10000.0)
            self.last_save_time = state.get('last_save_time', 0)
            self.last_training_time = state.get('last_training_time', 0)
            self.scalers_fitted = state.get('scalers_fitted', False)
            logger.info(f"Trader state loaded. Balance: {self.balance:.2f} AUD")
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
        """Fetch live market data with better error handling."""
        base_url = 'https://api.btcmarkets.net/v3'
        endpoint = '/markets/tickers?marketId=BTC-AUD'
        url = f"{base_url}{endpoint}"
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
                logger.debug("Market data fetched successfully")
                return data
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to fetch market data after {max_retries} attempts")
                    return None

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
        """Manual RSI calculation."""
        if len(prices) < period + 1:
            return 50  # Neutral RSI
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_technical_indicators(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, float]:
        """Calculate technical indicators using TA-Lib or manual calculations."""
        if len(prices) < 20:  # Need minimum data for indicators
            return {
                'sma_5': 0, 'sma_20': 0, 'ema_12': 0, 'ema_26': 0,
                'rsi': 50, 'macd': 0, 'macd_signal': 0, 'bb_upper': 0,
                'bb_lower': 0, 'volume_sma': 0
            }
        
        try:
            if TALIB_AVAILABLE:
                # Use TA-Lib if available
                sma_5 = talib.SMA(prices, timeperiod=5)[-1] if len(prices) >= 5 else prices[-1]
                sma_20 = talib.SMA(prices, timeperiod=20)[-1] if len(prices) >= 20 else prices[-1]
                ema_12 = talib.EMA(prices, timeperiod=12)[-1] if len(prices) >= 12 else prices[-1]
                ema_26 = talib.EMA(prices, timeperiod=26)[-1] if len(prices) >= 26 else prices[-1]
                rsi = talib.RSI(prices, timeperiod=14)[-1] if len(prices) >= 14 else 50
                
                macd, macd_signal, _ = talib.MACD(prices)
                macd_val = macd[-1] if len(macd) > 0 and not np.isnan(macd[-1]) else 0
                macd_signal_val = macd_signal[-1] if len(macd_signal) > 0 and not np.isnan(macd_signal[-1]) else 0
                
                bb_upper, bb_middle, bb_lower = talib.BBANDS(prices)
                bb_upper_val = bb_upper[-1] if len(bb_upper) > 0 and not np.isnan(bb_upper[-1]) else prices[-1]
                bb_lower_val = bb_lower[-1] if len(bb_lower) > 0 and not np.isnan(bb_lower[-1]) else prices[-1]
                
                volume_sma = talib.SMA(volumes, timeperiod=10)[-1] if len(volumes) >= 10 else volumes[-1]
            else:
                # Use manual calculations
                sma_5 = self.manual_sma(prices, 5)
                sma_20 = self.manual_sma(prices, 20)
                ema_12 = self.manual_ema(prices, 12)
                ema_26 = self.manual_ema(prices, 26)
                rsi = self.manual_rsi(prices, 14)
                
                # Simplified MACD
                macd_val = ema_12 - ema_26
                macd_signal_val = self.manual_ema(np.array([macd_val]), 9)
                
                # Simplified Bollinger Bands
                bb_middle = sma_20
                bb_std = np.std(prices[-20:]) if len(prices) >= 20 else 0
                bb_upper_val = bb_middle + (2 * bb_std)
                bb_lower_val = bb_middle - (2 * bb_std)
                
                volume_sma = self.manual_sma(volumes, 10)
            
            return {
                'sma_5': float(sma_5),
                'sma_20': float(sma_20),
                'ema_12': float(ema_12),
                'ema_26': float(ema_26),
                'rsi': float(rsi),
                'macd': float(macd_val),
                'macd_signal': float(macd_signal_val),
                'bb_upper': float(bb_upper_val),
                'bb_lower': float(bb_lower_val),
                'volume_sma': float(volume_sma)
            }
        except Exception as e:
            logger.warning(f"Error calculating technical indicators: {e}")
            return {
                'sma_5': 0, 'sma_20': 0, 'ema_12': 0, 'ema_26': 0,
                'rsi': 50, 'macd': 0, 'macd_signal': 0, 'bb_upper': 0,
                'bb_lower': 0, 'volume_sma': 0
            }

    def collect_and_store_data(self):
        """Collect current market data and add to training dataset."""
        try:
            market_data = self.fetch_market_data()
            if market_data:
                comprehensive_data = self.extract_comprehensive_data(market_data)
                if comprehensive_data and comprehensive_data['price'] > 0:
                    # Calculate technical indicators if we have enough historical data
                    indicators = {}
                    if len(self.training_data) >= 20:
                        try:
                            # Filter only valid dictionary entries and extract prices/volumes
                            valid_data = [dp for dp in self.training_data[-20:] if isinstance(dp, dict) and 'price' in dp and 'volume' in dp]
                            if len(valid_data) >= 20:
                                recent_prices = np.array([dp['price'] for dp in valid_data])
                                recent_volumes = np.array([dp['volume'] for dp in valid_data])
                                indicators = self.calculate_technical_indicators(recent_prices, recent_volumes)
                        except Exception as e:
                            logger.warning(f"Error calculating technical indicators: {e}")
                            indicators = {}
                    
                    # Store comprehensive data point with timestamp
                    data_point = {
                        'timestamp': datetime.now().isoformat(),
                        'price': comprehensive_data['price'],
                        'volume': comprehensive_data['volume'],
                        'bid': comprehensive_data['bid'],
                        'ask': comprehensive_data['ask'],
                        'high24h': comprehensive_data['high24h'],
                        'low24h': comprehensive_data['low24h'],
                        'spread': comprehensive_data['ask'] - comprehensive_data['bid'],
                        **indicators
                    }
                    
                    self.training_data.append(data_point)
                    
                    # Limit training data size to prevent memory issues
                    if len(self.training_data) > self.max_training_samples:
                        self.training_data = self.training_data[-self.max_training_samples:]
                    
                    logger.debug(f"Data collected: Price {comprehensive_data['price']:.2f} AUD, RSI {indicators.get('rsi', 0):.1f}")
                    return True
        except Exception as e:
            logger.error(f"Error collecting data: {e}")
        return False

    def create_lstm_model(self):
        """Create a new LSTM model for sequential data."""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 12)),  # 12 features
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(25, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("New LSTM model created")
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

    def fit_scalers(self, training_data: List[Dict]):
        """Fit scalers to the training data."""
        # Filter only valid dictionary entries
        valid_data = [dp for dp in training_data if isinstance(dp, dict) and 'price' in dp]
        
        if len(valid_data) < 50:
            logger.warning(f"Not enough valid data to fit scalers properly ({len(valid_data)} valid entries)")
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
        # Filter only valid dictionary entries
        valid_data = [dp for dp in self.training_data if isinstance(dp, dict) and 'price' in dp]
        
        if len(valid_data) < self.sequence_length + 10:
            logger.warning(f"Not enough valid data for LSTM training ({len(valid_data)} valid entries)")
            return None, None
        
        try:
            # Fit scalers if not already fitted
            if not self.scalers_fitted:
                if not self.fit_scalers(valid_data):
                    return None, None
            
            # Prepare sequences and labels
            sequences = []
            labels = []
            
            # Create sequences with proper future prediction labels
            for i in range(len(valid_data) - self.sequence_length - 1):  # -1 for future label
                # Create sequence of features
                sequence_features = []
                for j in range(i, i + self.sequence_length):
                    feature_vector = self.prepare_features(valid_data[j])
                    sequence_features.append(feature_vector)
                
                # Scale the sequence
                sequence_array = np.array(sequence_features)
                scaled_sequence = self.feature_scaler.transform(sequence_array)
                sequences.append(scaled_sequence)
                
                # Create label: will the price go up in the NEXT period?
                current_price = valid_data[i + self.sequence_length]['price']
                future_price = valid_data[i + self.sequence_length + 1]['price']
                label = 1 if future_price > current_price else 0
                labels.append(label)
            
            return np.array(sequences), np.array(labels)
            
        except Exception as e:
            logger.error(f"Error preparing LSTM training data: {e}")
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

    def predict_trade_signal(self, market_data: List[Dict]) -> Dict:
        """Predict trading signal using LSTM model with sequential data."""
        try:
            comprehensive_data = self.extract_comprehensive_data(market_data)
            if not comprehensive_data or comprehensive_data['price'] <= 0:
                return {"signal": "HOLD", "confidence": 0.5, "price": 0}
            
            # Need enough historical data for sequence
            if len(self.training_data) < self.sequence_length or not self.scalers_fitted:
                return {"signal": "HOLD", "confidence": 0.5, "price": comprehensive_data['price']}
            
            # Calculate current technical indicators
            recent_prices = np.array([dp['price'] for dp in self.training_data[-(self.sequence_length-1):]] + [comprehensive_data['price']])
            recent_volumes = np.array([dp['volume'] for dp in self.training_data[-(self.sequence_length-1):]] + [comprehensive_data['volume']])
            indicators = self.calculate_technical_indicators(recent_prices, recent_volumes)
            
            # Create current data point with indicators
            current_data = {**comprehensive_data, **indicators}
            
            # Prepare sequence for prediction
            sequence_data = self.training_data[-(self.sequence_length-1):] + [current_data]
            sequence_features = []
            
            for data_point in sequence_data:
                feature_vector = self.prepare_features(data_point)
                sequence_features.append(feature_vector)
            
            # Scale the sequence
            sequence_array = np.array(sequence_features)
            scaled_sequence = self.feature_scaler.transform(sequence_array)
            
            # Reshape for LSTM prediction
            prediction_input = scaled_sequence.reshape(1, self.sequence_length, 12)
            
            # Make prediction
            prediction = self.model.predict(prediction_input, verbose=0)[0][0]
            
            # Convert to trading signal with more conservative thresholds
            if prediction > 0.65:
                signal = "BUY"
            elif prediction < 0.35:
                signal = "SELL"
            else:
                signal = "HOLD"
            
            return {
                "signal": signal,
                "confidence": float(prediction),
                "price": comprehensive_data['price'],
                "rsi": indicators.get('rsi', 50),
                "macd": indicators.get('macd', 0)
            }
            
        except Exception as e:
            logger.error(f"Error making LSTM prediction: {e}")
            return {"signal": "HOLD", "confidence": 0.5, "price": 0}

    def execute_simulated_trade(self, prediction: Dict):
        """Execute simulated trades based on LSTM predictions."""
        if not prediction or prediction["price"] == 0:
            logger.warning("Invalid prediction data, no trade executed")
            return
        
        trade_amount = 0.01  # BTC
        fee_rate = 0.001  # 0.1%
        price = prediction["price"]
        signal = prediction["signal"]
        confidence = prediction["confidence"]
        rsi = prediction.get("rsi", 50)
        
        # More sophisticated trading logic
        # Don't trade if confidence is too neutral or if RSI indicates extreme conditions
        if abs(confidence - 0.5) < 0.15:  # Too uncertain
            logger.info(f"Confidence too neutral ({confidence:.3f}), holding position")
            return
        
        # Avoid buying when extremely overbought or selling when extremely oversold
        if signal == "BUY" and rsi > 80:
            logger.info(f"RSI too high ({rsi:.1f}), avoiding BUY signal")
            return
        if signal == "SELL" and rsi < 20:
            logger.info(f"RSI too low ({rsi:.1f}), avoiding SELL signal")
            return
        
        try:
            if signal == "BUY" and self.balance > price * trade_amount * (1 + fee_rate):
                cost = price * trade_amount * (1 + fee_rate)
                self.balance -= cost
                logger.info(f"BUY executed: {trade_amount} BTC at {price:.2f} AUD (Cost: {cost:.2f}, Confidence: {confidence:.3f}, RSI: {rsi:.1f})")
            
            elif signal == "SELL":
                revenue = price * trade_amount * (1 - fee_rate)
                self.balance += revenue
                logger.info(f"SELL executed: {trade_amount} BTC at {price:.2f} AUD (Revenue: {revenue:.2f}, Confidence: {confidence:.3f}, RSI: {rsi:.1f})")
            
            else:
                logger.info(f"HOLD - Price: {price:.2f} AUD, Confidence: {confidence:.3f}, RSI: {rsi:.1f}")
            
            logger.info(f"Current balance: {self.balance:.2f} AUD")
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")

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
        """Load the TensorFlow model."""
        try:
            model = tf.keras.models.load_model(self.model_filename)
            logger.info("LSTM model loaded successfully")
            return model
        except FileNotFoundError:
            logger.info("No model file found, will create new LSTM")
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
        else:
            logger.info("Starting continuous LSTM trading operation...")
        
        iteration_count = 0
        while True:
            try:
                iteration_count += 1
                logger.info(f"--- Iteration {iteration_count} ---")
                
                # Collect new market data
                if self.collect_and_store_data():
                    # Get current market data for prediction
                    current_market_data = self.fetch_market_data()
                    if current_market_data:
                        # Make trading prediction using LSTM
                        prediction = self.predict_trade_signal(current_market_data)
                        
                        # Execute trade based on prediction
                        self.execute_simulated_trade(prediction)
                
                # Retrain LSTM model if enough time has passed and we have sufficient data
                if (self.should_train() and 
                    len(self.training_data) >= self.sequence_length + 20):
                    logger.info("Retraining LSTM model with new sequential data...")
                    if self.train_model():
                        self.last_training_time = time.time()
                
                # Save everything periodically
                if self.should_save():
                    logger.info("Saving data, model, and scalers...")
                    self.save_training_data()
                    self.save_model()
                    self.save_scalers()
                    self.save_state()
                    self.last_save_time = time.time()
                
                # Log status every 10 iterations
                if iteration_count % 10 == 0:
                    avg_rsi = np.mean([dp.get('rsi', 50) for dp in self.training_data[-10:] if 'rsi' in dp]) if len(self.training_data) >= 10 else 50
                    logger.info(f"Status: Balance={self.balance:.2f} AUD, Training samples={len(self.training_data)}, Avg RSI={avg_rsi:.1f}")
                
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
                
                # Sleep before next iteration
                sleep_time = 5 if self.test_mode else 60  # Shorter sleep in test mode
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