import requests
from datetime import datetime
import tensorflow as tf
import numpy as np
import json
import os
import time
import logging
from typing import List, Dict, Optional
import pickle

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
    def __init__(self, initial_balance: float = 10000.0):
        self.balance = initial_balance
        self.model_filename = "autotrader_model.keras"
        self.training_data_filename = "training_data.json"
        self.state_filename = "trader_state.pkl"
        self.save_interval_seconds = 1800  # Save every 30 minutes for more frequent backups
        self.training_interval_seconds = 300  # Retrain every 5 minutes
        self.max_training_samples = 1000  # Limit training data to prevent memory issues
        
        # Load persistent state
        self.load_state()
        
        # Initialize model
        self.model = self.load_model()
        if self.model is None:
            self.model = self.create_new_model()
        
        # Training data management
        self.training_data = self.load_training_data()
        self.last_save_time = time.time()
        self.last_training_time = 0
        
        logger.info(f"AutoTrader initialized with balance: {self.balance:.2f} AUD")
        logger.info(f"Loaded {len(self.training_data)} historical data points")

    def save_state(self):
        """Save the current state of the trader to a file."""
        state = {
            'balance': self.balance,
            'last_save_time': self.last_save_time,
            'last_training_time': self.last_training_time,
            'training_data_length': len(self.training_data)
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
            logger.info(f"Trader state loaded. Balance: {self.balance:.2f} AUD")
        except FileNotFoundError:
            logger.info("No previous state found. Starting fresh.")
        except Exception as e:
            logger.error(f"Error loading trader state: {e}")

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

    def extract_price_from_data(self, data: List[Dict]) -> Optional[float]:
        """Extract BTC-AUD price from market data."""
        if not data or not isinstance(data, list):
            return None
        
        btc_aud_data = next((item for item in data if item.get('marketId') == 'BTC-AUD'), None)
        if btc_aud_data and 'lastPrice' in btc_aud_data:
            try:
                return float(btc_aud_data['lastPrice'])
            except (ValueError, TypeError):
                logger.warning("Could not convert 'lastPrice' to float")
        return None

    def collect_and_store_data(self):
        """Collect current market data and add to training dataset."""
        try:
            market_data = self.fetch_market_data()
            if market_data:
                price = self.extract_price_from_data(market_data)
                if price:
                    # Store data point with timestamp
                    data_point = {
                        'timestamp': datetime.now().isoformat(),
                        'price': price,
                        'volume': market_data[0].get('volume24h', 0) if market_data else 0
                    }
                    self.training_data.append(data_point)
                    
                    # Limit training data size to prevent memory issues
                    if len(self.training_data) > self.max_training_samples:
                        self.training_data = self.training_data[-self.max_training_samples:]
                    
                    logger.debug(f"Data collected: Price {price:.2f} AUD")
                    return True
        except Exception as e:
            logger.error(f"Error collecting data: {e}")
        return False

    def create_new_model(self):
        """Create a new neural network model."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(20, activation='relu', input_shape=(3,)),  # More inputs: price, volume, time
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        logger.info("New model created")
        return model

    def prepare_training_data(self):
        """Prepare training data with features and labels."""
        if len(self.training_data) < 10:
            return None, None
        
        try:
            # Extract features and create labels
            features = []
            labels = []
            
            for i in range(5, len(self.training_data)):  # Need at least 5 previous points
                current_price = self.training_data[i]['price']
                prev_prices = [self.training_data[j]['price'] for j in range(i-5, i)]
                volume = self.training_data[i].get('volume', 0)
                
                # Features: average of last 5 prices, current volume, price trend
                avg_price = np.mean(prev_prices)
                price_trend = (current_price - prev_prices[0]) / prev_prices[0] if prev_prices[0] > 0 else 0
                
                features.append([
                    current_price / 100000.0,  # Normalized price
                    volume / 1000000.0,        # Normalized volume
                    price_trend                # Price trend
                ])
                
                # Label: 1 if price goes up in next period, 0 otherwise
                if i < len(self.training_data) - 1:
                    next_price = self.training_data[i + 1]['price']
                    labels.append(1 if next_price > current_price else 0)
            
            return np.array(features), np.array(labels)
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None, None

    def train_model(self):
        """Train the model with accumulated data."""
        try:
            features, labels = self.prepare_training_data()
            if features is None or len(features) < 10:
                logger.warning("Not enough data for training")
                return False
            
            # Train the model
            history = self.model.fit(
                features, labels,
                epochs=5,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            loss = history.history['loss'][-1]
            accuracy = history.history.get('accuracy', [0])[-1]
            logger.info(f"Model trained - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False

    def predict_trade_signal(self, market_data: List[Dict]) -> Dict:
        """Predict trading signal based on current market data."""
        try:
            price = self.extract_price_from_data(market_data)
            if not price or len(self.training_data) < 5:
                return {"signal": "HOLD", "confidence": 0.5, "price": price or 0}
            
            # Prepare input features similar to training
            recent_prices = [dp['price'] for dp in self.training_data[-5:]]
            volume = market_data[0].get('volume24h', 0) if market_data else 0
            
            avg_price = np.mean(recent_prices)
            price_trend = (price - recent_prices[0]) / recent_prices[0] if recent_prices[0] > 0 else 0
            
            input_features = np.array([[
                price / 100000.0,
                volume / 1000000.0,
                price_trend
            ]])
            
            # Make prediction
            prediction = self.model.predict(input_features, verbose=0)[0][0]
            
            # Convert to trading signal
            if prediction > 0.6:
                signal = "BUY"
            elif prediction < 0.4:
                signal = "SELL"
            else:
                signal = "HOLD"
            
            return {
                "signal": signal,
                "confidence": float(prediction),
                "price": price
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {"signal": "HOLD", "confidence": 0.5, "price": 0}

    def execute_simulated_trade(self, prediction: Dict):
        """Execute simulated trades based on predictions."""
        if not prediction or prediction["price"] == 0:
            logger.warning("Invalid prediction data, no trade executed")
            return
        
        trade_amount = 0.01  # BTC
        fee_rate = 0.001  # 0.1%
        price = prediction["price"]
        signal = prediction["signal"]
        confidence = prediction["confidence"]
        
        # Only trade if confidence is high enough
        if abs(confidence - 0.5) < 0.1:  # Too uncertain
            logger.info(f"Confidence too low ({confidence:.3f}), holding position")
            return
        
        try:
            if signal == "BUY" and self.balance > price * trade_amount * (1 + fee_rate):
                cost = price * trade_amount * (1 + fee_rate)
                self.balance -= cost
                logger.info(f"BUY executed: {trade_amount} BTC at {price:.2f} AUD (Cost: {cost:.2f}, Confidence: {confidence:.3f})")
            
            elif signal == "SELL":
                revenue = price * trade_amount * (1 - fee_rate)
                self.balance += revenue
                logger.info(f"SELL executed: {trade_amount} BTC at {price:.2f} AUD (Revenue: {revenue:.2f}, Confidence: {confidence:.3f})")
            
            else:
                logger.info(f"HOLD - Price: {price:.2f} AUD, Confidence: {confidence:.3f}")
            
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
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def load_model(self):
        """Load the TensorFlow model."""
        try:
            model = tf.keras.models.load_model(self.model_filename)
            logger.info("Model loaded successfully")
            return model
        except FileNotFoundError:
            logger.info("No model file found, will create new one")
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
        """Main loop for continuous trading operation."""
        logger.info("Starting continuous trading operation...")
        
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
                        # Make trading prediction
                        prediction = self.predict_trade_signal(current_market_data)
                        
                        # Execute trade based on prediction
                        self.execute_simulated_trade(prediction)
                
                # Retrain model if enough time has passed
                if self.should_train() and len(self.training_data) >= 10:
                    logger.info("Retraining model with new data...")
                    if self.train_model():
                        self.last_training_time = time.time()
                
                # Save everything periodically
                if self.should_save():
                    logger.info("Saving data and model...")
                    self.save_training_data()
                    self.save_model()
                    self.save_state()
                    self.last_save_time = time.time()
                
                # Log status every 10 iterations
                if iteration_count % 10 == 0:
                    logger.info(f"Status: Balance={self.balance:.2f} AUD, Training samples={len(self.training_data)}")
                
                # Sleep before next iteration
                time.sleep(60)  # 1 minute intervals
                
            except KeyboardInterrupt:
                logger.info("Received shutdown signal, saving data...")
                self.save_training_data()
                self.save_model()
                self.save_state()
                logger.info("Shutdown complete")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                time.sleep(60)  # Wait before retrying

# Main execution
if __name__ == "__main__":
    try:
        trader = ContinuousAutoTrader(initial_balance=10000.0)
        trader.run_continuous_trading()
    except Exception as e:
        logger.error(f"Failed to start trader: {e}")