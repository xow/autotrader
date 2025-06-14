"""
Machine learning models for autotrader bot.

Contains LSTM and other neural network models for price prediction.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger("autotrader.ml.models")


class LSTMModel:
    """LSTM model for sequential price prediction."""
    
    def __init__(self, sequence_length: int = 20, num_features: int = 12):
        """
        Initialize LSTM model.
        
        Args:
            sequence_length: Length of input sequences
            num_features: Number of input features
        """
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.model = None
        self._create_model()
    
    def _create_model(self):
        """Create the LSTM model architecture."""
        self.model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(self.sequence_length, self.num_features)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(25, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("LSTM model created successfully")
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 16, validation_split: float = 0.2) -> dict:
        """
        Train the model.
        
        Args:
            X: Input sequences
            y: Target labels
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
        
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0,
            shuffle=False
        )
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input sequences
        
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        return self.model.predict(X, verbose=0)
    
    def save(self, filepath: str) -> bool:
        """
        Save the model.
        
        Args:
            filepath: Path to save the model
        
        Returns:
            True if successful
        """
        if self.model is None:
            return False
        
        try:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load(self, filepath: str) -> bool:
        """
        Load the model.
        
        Args:
            filepath: Path to load the model from
        
        Returns:
            True if successful
        """
        try:
            self.model = tf.keras.models.load_model(filepath)
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
