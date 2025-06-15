"""
Neural Network Architecture for time-series prediction in autotrader bot.

Implements configurable neural network models for cryptocurrency price prediction
with proper loss functions, optimizers, and validation metrics.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("autotrader.ml.neural_network")


class ModelType(Enum):
    """Supported neural network model types."""
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    CNN_LSTM = "cnn_lstm"


class LossType(Enum):
    """Supported loss functions."""
    BINARY_CROSSENTROPY = "binary_crossentropy"
    MEAN_SQUARED_ERROR = "mse"
    MEAN_ABSOLUTE_ERROR = "mae" 
    HUBER = "huber"


class OptimizerType(Enum):
    """Supported optimizers."""
    ADAM = "adam"
    RMSPROP = "rmsprop"
    SGD = "sgd"
    ADAMW = "adamw"


@dataclass
class ModelConfig:
    """Configuration for neural network models."""
    
    # Architecture parameters
    model_type: ModelType = ModelType.LSTM
    sequence_length: int = 20
    num_features: int = 13
    hidden_units: List[int] = None
    dropout_rate: float = 0.2
    recurrent_dropout: float = 0.1
    
    # Output configuration
    output_units: int = 1
    output_activation: str = "sigmoid"
    
    # Training parameters
    loss_function: LossType = LossType.BINARY_CROSSENTROPY
    optimizer_type: OptimizerType = OptimizerType.ADAM
    learning_rate: float = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-7
    
    # Regularization
    l1_reg: float = 0.0
    l2_reg: float = 0.0001
    batch_normalization: bool = True
    layer_normalization: bool = False
    
    # Architecture specific
    use_attention: bool = False
    attention_heads: int = 8
    cnn_filters: List[int] = None
    kernel_sizes: List[int] = None
    
    def __post_init__(self):
        if self.hidden_units is None:
            self.hidden_units = [64, 32, 16]
        if self.cnn_filters is None:
            self.cnn_filters = [32, 64, 32]
        if self.kernel_sizes is None:
            self.kernel_sizes = [3, 3, 3]


class NeuralNetworkArchitecture:
    """
    Configurable neural network architecture for time-series prediction.
    
    Supports multiple model types with proper regularization, normalization,
    and validation metrics for cryptocurrency price prediction.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize neural network architecture.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.model: Optional[tf.keras.Model] = None
        self.training_history: List[Dict] = []
        self.validation_metrics: Dict[str, float] = {}
        
        logger.info(f"Initializing {config.model_type.value} neural network")
    
    def build_model(self) -> tf.keras.Model:
        """
        Build the neural network model based on configuration.
        
        Returns:
            Compiled TensorFlow model
        """
        if self.config.model_type == ModelType.LSTM:
            model = self._build_lstm_model()
        elif self.config.model_type == ModelType.GRU:
            model = self._build_gru_model()
        elif self.config.model_type == ModelType.TRANSFORMER:
            model = self._build_transformer_model()
        elif self.config.model_type == ModelType.CNN_LSTM:
            model = self._build_cnn_lstm_model()
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        
        # Compile model
        model = self._compile_model(model)
        self.model = model
        
        logger.info(f"Built {self.config.model_type.value} model with {model.count_params()} parameters")
        return model
    
    def _build_lstm_model(self) -> tf.keras.Model:
        """Build LSTM-based model."""
        inputs = tf.keras.layers.Input(shape=(self.config.sequence_length, self.config.num_features))
        x = inputs
        
        # Add batch normalization if enabled
        if self.config.batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)
        
        # Build LSTM layers
        for i, units in enumerate(self.config.hidden_units):
            return_sequences = i < len(self.config.hidden_units) - 1
            
            x = tf.keras.layers.LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.recurrent_dropout,
                kernel_regularizer=self._get_regularizer(),
                name=f"lstm_{i+1}"
            )(x)
            
            # Add layer normalization if enabled
            if self.config.layer_normalization:
                x = tf.keras.layers.LayerNormalization()(x)
            
            # Add dropout
            x = tf.keras.layers.Dropout(self.config.dropout_rate)(x)
        
        # Add attention if enabled
        if self.config.use_attention and len(self.config.hidden_units) > 1:
            x = self._add_attention_layer(x)
        
        # Output layers
        x = tf.keras.layers.Dense(
            units=self.config.hidden_units[-1] // 2,
            activation='relu',
            kernel_regularizer=self._get_regularizer(),
            name="dense_intermediate"
        )(x)
        
        x = tf.keras.layers.Dropout(self.config.dropout_rate)(x)
        
        outputs = tf.keras.layers.Dense(
            units=self.config.output_units,
            activation=self.config.output_activation,
            name="output"
        )(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="lstm_model")
    
    def _build_gru_model(self) -> tf.keras.Model:
        """Build GRU-based model."""
        inputs = tf.keras.layers.Input(shape=(self.config.sequence_length, self.config.num_features))
        x = inputs
        
        if self.config.batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)
        
        for i, units in enumerate(self.config.hidden_units):
            return_sequences = i < len(self.config.hidden_units) - 1
            
            x = tf.keras.layers.GRU(
                units=units,
                return_sequences=return_sequences,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.recurrent_dropout,
                kernel_regularizer=self._get_regularizer(),
                name=f"gru_{i+1}"
            )(x)
            
            if self.config.layer_normalization:
                x = tf.keras.layers.LayerNormalization()(x)
            
            x = tf.keras.layers.Dropout(self.config.dropout_rate)(x)
        
        # Output layers
        x = tf.keras.layers.Dense(
            units=self.config.hidden_units[-1] // 2,
            activation='relu',
            kernel_regularizer=self._get_regularizer()
        )(x)
        
        x = tf.keras.layers.Dropout(self.config.dropout_rate)(x)
        
        outputs = tf.keras.layers.Dense(
            units=self.config.output_units,
            activation=self.config.output_activation
        )(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="gru_model")
    
    def _build_cnn_lstm_model(self) -> tf.keras.Model:
        """Build CNN-LSTM hybrid model."""
        inputs = tf.keras.layers.Input(shape=(self.config.sequence_length, self.config.num_features))
        x = inputs
        
        # CNN layers for feature extraction
        for i, (filters, kernel_size) in enumerate(zip(self.config.cnn_filters, self.config.kernel_sizes)):
            x = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                kernel_regularizer=self._get_regularizer(),
                name=f"conv1d_{i+1}"
            )(x)
            
            if self.config.batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x)
            
            x = tf.keras.layers.Dropout(self.config.dropout_rate)(x)
        
        # LSTM layers
        for i, units in enumerate(self.config.hidden_units):
            return_sequences = i < len(self.config.hidden_units) - 1
            
            x = tf.keras.layers.LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.recurrent_dropout,
                kernel_regularizer=self._get_regularizer(),
                name=f"lstm_{i+1}"
            )(x)
            
            x = tf.keras.layers.Dropout(self.config.dropout_rate)(x)
        
        # Output layers
        x = tf.keras.layers.Dense(
            units=self.config.hidden_units[-1] // 2,
            activation='relu',
            kernel_regularizer=self._get_regularizer()
        )(x)
        
        outputs = tf.keras.layers.Dense(
            units=self.config.output_units,
            activation=self.config.output_activation
        )(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_lstm_model")
    
    def _build_transformer_model(self) -> tf.keras.Model:
        """Build Transformer-based model."""
        inputs = tf.keras.layers.Input(shape=(self.config.sequence_length, self.config.num_features))
        x = inputs
        
        # Positional encoding
        x = self._add_positional_encoding(x)
        
        # Multi-head attention layers
        for i in range(len(self.config.hidden_units)):
            # Multi-head attention
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=self.config.attention_heads,
                key_dim=self.config.num_features,
                dropout=self.config.dropout_rate,
                name=f"multihead_attention_{i+1}"
            )(x, x)
            
            # Add & Norm
            x = tf.keras.layers.Add()([x, attention_output])
            x = tf.keras.layers.LayerNormalization()(x)
            
            # Feed forward
            ff_output = tf.keras.layers.Dense(
                self.config.hidden_units[i],
                activation='relu',
                kernel_regularizer=self._get_regularizer()
            )(x)
            ff_output = tf.keras.layers.Dropout(self.config.dropout_rate)(ff_output)
            ff_output = tf.keras.layers.Dense(
                self.config.num_features,
                kernel_regularizer=self._get_regularizer()
            )(ff_output)
            
            # Add & Norm
            x = tf.keras.layers.Add()([x, ff_output])
            x = tf.keras.layers.LayerNormalization()(x)
        
        # Global average pooling and output
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(
            self.config.hidden_units[-1],
            activation='relu',
            kernel_regularizer=self._get_regularizer()
        )(x) 
        x = tf.keras.layers.Dropout(self.config.dropout_rate)(x)
        
        outputs = tf.keras.layers.Dense(
            units=self.config.output_units,
            activation=self.config.output_activation
        )(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="transformer_model")
    
    def _add_attention_layer(self, x: tf.Tensor) -> tf.Tensor:
        """Add attention mechanism to LSTM output."""
        attention = tf.keras.layers.Dense(1, activation='tanh')(x)
        attention = tf.keras.layers.Dense(1, activation='softmax')(attention)
        context = tf.keras.layers.Multiply()([x, attention])
        return tf.keras.layers.Flatten()(context)
    
    def _add_positional_encoding(self, x: tf.Tensor) -> tf.Tensor:
        """Add positional encoding for transformer."""
        seq_len = self.config.sequence_length
        d_model = self.config.num_features
        
        # Simple learnable positional encoding
        pos_encoding = tf.Variable(
            tf.random.normal([seq_len, d_model], stddev=0.1),
            trainable=True,
            name="positional_encoding"
        )
        
        return x + pos_encoding
    
    def _get_regularizer(self) -> Optional[tf.keras.regularizers.Regularizer]:
        """Get regularizer based on configuration."""
        if self.config.l1_reg > 0 and self.config.l2_reg > 0:
            return tf.keras.regularizers.L1L2(l1=self.config.l1_reg, l2=self.config.l2_reg)
        elif self.config.l1_reg > 0:
            return tf.keras.regularizers.L1(self.config.l1_reg)
        elif self.config.l2_reg > 0:
            return tf.keras.regularizers.L2(self.config.l2_reg)
        return None
    
    def _compile_model(self, model: tf.keras.Model) -> tf.keras.Model:
        """Compile model with appropriate optimizer and loss function."""
        # Create optimizer
        if self.config.optimizer_type == OptimizerType.ADAM:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.config.learning_rate,
                beta_1=self.config.beta_1,
                beta_2=self.config.beta_2,
                epsilon=self.config.epsilon
            )
        elif self.config.optimizer_type == OptimizerType.ADAMW:
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=self.config.learning_rate,
                beta_1=self.config.beta_1,
                beta_2=self.config.beta_2,
                epsilon=self.config.epsilon
            )
        elif self.config.optimizer_type == OptimizerType.RMSPROP:
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.config.learning_rate)
        elif self.config.optimizer_type == OptimizerType.SGD:
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.config.learning_rate)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        
        # Define metrics
        metrics = ['accuracy']
        if self.config.loss_function in [LossType.MEAN_SQUARED_ERROR, LossType.MEAN_ABSOLUTE_ERROR, LossType.HUBER]:
            metrics = ['mae', 'mse']
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=self.config.loss_function.value,
            metrics=metrics
        )
        
        return model
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        callbacks: Optional[List] = None,
        verbose: int = 1
    ) -> Dict:
        """
        Train the neural network model.
        
        Args:
            X: Training features
            y: Training targets
            validation_data: Validation data tuple
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
            callbacks: Training callbacks
            verbose: Verbosity level
        
        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.build_model()
        
        # Default callbacks
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                )
            ]
        
        # Train model
        history = self.model.fit(
            X, y,
            validation_data=validation_data,
            validation_split=validation_split if validation_data is None else 0.0,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=False  # Important for time series
        )
        
        # Store training history
        self.training_history.append(history.history)
        
        # Calculate validation metrics
        self._calculate_validation_metrics(history.history)
        
        logger.info(f"Training completed - Final loss: {history.history['loss'][-1]:.4f}")
        
        return history.history
    
    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            batch_size: Batch size for prediction
        
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call build_model() and train() first.")
        
        return self.model.predict(X, batch_size=batch_size, verbose=0)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Test features
            y: Test targets  
            batch_size: Batch size
        
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        results = self.model.evaluate(X, y, batch_size=batch_size, verbose=0)
        metric_names = self.model.metrics_names
        
        return dict(zip(metric_names, results))
    
    def _calculate_validation_metrics(self, history: Dict) -> None:
        """Calculate and store validation metrics."""
        if 'val_loss' in history:
            self.validation_metrics = {
                'final_val_loss': history['val_loss'][-1],
                'best_val_loss': min(history['val_loss']),
                'final_loss': history['loss'][-1],
                'best_loss': min(history['loss'])
            }
            
            if 'val_accuracy' in history:
                self.validation_metrics.update({
                    'final_val_accuracy': history['val_accuracy'][-1],
                    'best_val_accuracy': max(history['val_accuracy']),
                    'final_accuracy': history['accuracy'][-1],
                    'best_accuracy': max(history['accuracy'])
                })
    
    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        if self.model is None:
            return "Model not built yet"
        
        import io
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        self.model.summary()
        
        sys.stdout = old_stdout
        summary = buffer.getvalue()
        
        return summary
    
    def save_model(self, filepath: str) -> bool:
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        
        Returns:
            True if successful
        """
        if self.model is None:
            logger.error("No model to save")
            return False
        
        try:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load a trained model.
        
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
