"""
Continuous learning and training for autotrader bot.

Implements online learning algorithms and training orchestration.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .models import LSTMModel
from ..core.config import Config

logger = logging.getLogger("autotrader.ml.training")


class ContinuousLearner:
    """Manages continuous learning and model training."""
    
    def __init__(self, config: Config):
        """
        Initialize continuous learner.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self.model = LSTMModel(config.sequence_length)
        self.feature_scaler = StandardScaler()
        self.price_scaler = MinMaxScaler()
        self.scalers_fitted = False
        self.training_history = []
    
    def prepare_features(self, data_point: Dict) -> np.ndarray:
        """
        Prepare feature vector from a data point.
        
        Args:
            data_point: Dictionary containing market data and indicators
        
        Returns:
            Feature vector as numpy array
        """
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
    
    def fit_scalers(self, training_data: List[Dict]) -> bool:
        """
        Fit scalers to training data.
        
        Args:
            training_data: List of training data points
        
        Returns:
            True if successful
        """
        if len(training_data) < 50:
            logger.warning("Not enough data to fit scalers properly")
            return False
        
        try:
            features = []
            prices = []
            
            for data_point in training_data:
                feature_vector = self.prepare_features(data_point)
                features.append(feature_vector)
                prices.append([data_point.get('price', 0)])
            
            features = np.array(features)
            prices = np.array(prices)
            
            self.feature_scaler.fit(features)
            self.price_scaler.fit(prices)
            self.scalers_fitted = True
            
            logger.info("Scalers fitted to training data")
            return True
            
        except Exception as e:
            logger.error(f"Error fitting scalers: {e}")
            return False
    
    def prepare_sequences(self, training_data: List[Dict]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Prepare sequential training data for LSTM.
        
        Args:
            training_data: List of training data points
        
        Returns:
            Tuple of (sequences, labels) or (None, None) if failed
        """
        if len(training_data) < self.config.sequence_length + 10:
            return None, None
        
        try:
            if not self.scalers_fitted:
                if not self.fit_scalers(training_data):
                    return None, None
            
            sequences = []
            labels = []
            
            for i in range(len(training_data) - self.config.sequence_length - 1):
                sequence_features = []
                for j in range(i, i + self.config.sequence_length):
                    feature_vector = self.prepare_features(training_data[j])
                    sequence_features.append(feature_vector)
                
                sequence_array = np.array(sequence_features)
                scaled_sequence = self.feature_scaler.transform(sequence_array)
                sequences.append(scaled_sequence)
                
                current_price = training_data[i + self.config.sequence_length]['price']
                future_price = training_data[i + self.config.sequence_length + 1]['price']
                label = 1 if future_price > current_price else 0
                labels.append(label)
            
            return np.array(sequences), np.array(labels)
            
        except Exception as e:
            logger.error(f"Error preparing sequences: {e}")
            return None, None
    
    def train_model(self, training_data: List[Dict]) -> bool:
        """
        Train the model with training data.
        
        Args:
            training_data: List of training data points
        
        Returns:
            True if training successful
        """
        try:
            sequences, labels = self.prepare_sequences(training_data)
            if sequences is None or len(sequences) < 20:
                logger.warning("Not enough sequential data for training")
                return False
            
            history = self.model.train(sequences, labels)
            
            # Log training results
            loss = history['loss'][-1]
            accuracy = history.get('accuracy', [0])[-1]
            val_loss = history.get('val_loss', [loss])[-1]
            val_accuracy = history.get('val_accuracy', [accuracy])[-1]
            
            logger.info(f"Training completed - Loss: {loss:.4f}, Acc: {accuracy:.4f}, Val_Loss: {val_loss:.4f}, Val_Acc: {val_accuracy:.4f}")
            
            # Store training history
            self.training_history.append({
                'timestamp': np.datetime64('now').item(),
                'loss': loss,
                'accuracy': accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'training_samples': len(sequences)
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def predict(self, recent_data: List[Dict], current_data: Dict) -> Optional[Dict]:
        """
        Make a prediction using the current model.
        
        Args:
            recent_data: Recent historical data points
            current_data: Current market data point
        
        Returns:
            Prediction dictionary or None if failed
        """
        try:
            if not self.scalers_fitted or len(recent_data) < self.config.sequence_length - 1:
                return None
            
            sequence_data = recent_data[-(self.config.sequence_length-1):] + [current_data]
            sequence_features = []
            
            for data_point in sequence_data:
                feature_vector = self.prepare_features(data_point)
                sequence_features.append(feature_vector)
            
            sequence_array = np.array(sequence_features)
            scaled_sequence = self.feature_scaler.transform(sequence_array)
            prediction_input = scaled_sequence.reshape(1, self.config.sequence_length, 12)
            
            prediction = self.model.predict(prediction_input)[0][0]
            
            return {
                'prediction': float(prediction),
                'confidence': float(prediction),
                'timestamp': np.datetime64('now').item()
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None
    
    def get_scalers_data(self) -> Dict:
        """Get scalers data for persistence."""
        return {
            'feature_scaler': self.feature_scaler,
            'price_scaler': self.price_scaler,
            'scalers_fitted': self.scalers_fitted
        }
    
    def set_scalers_data(self, scalers_data: Dict):
        """Set scalers data from persistence."""
        self.feature_scaler = scalers_data.get('feature_scaler', StandardScaler())
        self.price_scaler = scalers_data.get('price_scaler', MinMaxScaler())
        self.scalers_fitted = scalers_data.get('scalers_fitted', False)
