"""
Tests for neural network architecture and related components.
"""

import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
import os

from autotrader.ml.neural_network import (
    NeuralNetworkArchitecture, ModelConfig, ModelType, LossType, OptimizerType
)
from autotrader.ml.feature_engineer import FeatureEngineer, FeatureConfig
from autotrader.ml.model_manager import ModelManager, ModelMetadata


class TestNeuralNetworkArchitecture:
    """Test neural network architecture implementation."""
    
    def test_model_config_defaults(self):
        """Test default model configuration."""
        config = ModelConfig()
        
        assert config.model_type == ModelType.LSTM
        assert config.sequence_length == 20
        assert config.num_features == 13
        assert config.hidden_units == [64, 32, 16]
        assert config.dropout_rate == 0.2
        assert config.loss_function == LossType.BINARY_CROSSENTROPY
        assert config.optimizer_type == OptimizerType.ADAM
        assert config.learning_rate == 0.001
    
    def test_lstm_model_creation(self):
        """Test LSTM model creation and architecture."""
        config = ModelConfig(
            model_type=ModelType.LSTM,
            sequence_length=10,
            num_features=5,
            hidden_units=[32, 16]
        )
        
        nn = NeuralNetworkArchitecture(config)
        model = nn.build_model()
        
        assert model is not None
        assert model.input_shape == (None, 10, 5)
        assert len(model.layers) >= 4  # LSTM layers + dense layers
        
        # Check model is compiled
        assert model.optimizer is not None
        assert model.loss == 'binary_crossentropy'
    
    def test_gru_model_creation(self):
        """Test GRU model creation."""
        config = ModelConfig(
            model_type=ModelType.GRU,
            sequence_length=15,
            num_features=8,
            hidden_units=[64, 32]
        )
        
        nn = NeuralNetworkArchitecture(config)
        model = nn.build_model()
        
        assert model is not None
        assert model.input_shape == (None, 15, 8)
        
        # Check for GRU layers
        layer_types = [type(layer).__name__ for layer in model.layers]
        assert 'GRU' in layer_types
    
    def test_cnn_lstm_model_creation(self):
        """Test CNN-LSTM hybrid model creation."""
        config = ModelConfig(
            model_type=ModelType.CNN_LSTM,
            sequence_length=20,
            num_features=10,
            cnn_filters=[16, 32],
            kernel_sizes=[3, 3],
            hidden_units=[32, 16]
        )
        
        nn = NeuralNetworkArchitecture(config)
        model = nn.build_model()
        
        assert model is not None
        
        # Check for CNN and LSTM layers
        layer_types = [type(layer).__name__ for layer in model.layers]
        assert 'Conv1D' in layer_types
        assert 'LSTM' in layer_types
    
    def test_transformer_model_creation(self):
        """Test Transformer model creation."""
        config = ModelConfig(
            model_type=ModelType.TRANSFORMER,
            sequence_length=20,
            num_features=12,
            hidden_units=[64, 32],
            use_attention=True,
            attention_heads=4
        )
        
        nn = NeuralNetworkArchitecture(config)
        model = nn.build_model()
        
        assert model is not None
        
        # Check for attention layers
        layer_types = [type(layer).__name__ for layer in model.layers]
        assert 'MultiHeadAttention' in layer_types
    
    def test_model_training(self):
        """Test model training process."""
        config = ModelConfig(
            sequence_length=5,
            num_features=3,
            hidden_units=[8, 4]
        )
        
        nn = NeuralNetworkArchitecture(config)
        nn.build_model()
        
        # Create dummy training data
        X = np.random.random((20, 5, 3))
        y = np.random.randint(0, 2, (20, 1))
        
        # Train model
        history = nn.train(X, y, epochs=2, verbose=0)
        
        assert 'loss' in history
        assert 'accuracy' in history
        assert len(history['loss']) == 2
    
    def test_model_prediction(self):
        """Test model prediction."""
        config = ModelConfig(
            sequence_length=5,
            num_features=3,
            hidden_units=[8, 4]
        )
        
        nn = NeuralNetworkArchitecture(config)
        nn.build_model()
        
        # Create dummy data
        X = np.random.random((10, 5, 3))
        predictions = nn.predict(X)
        
        assert predictions.shape == (10, 1)
        assert np.all((predictions >= 0) & (predictions <= 1))  # Sigmoid output
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        config = ModelConfig(
            sequence_length=5,
            num_features=3,
            hidden_units=[8, 4]
        )
        
        nn = NeuralNetworkArchitecture(config)
        nn.build_model()
        
        # Create dummy data
        X = np.random.random((20, 5, 3))
        y = np.random.randint(0, 2, (20, 1))
        
        # Train briefly
        nn.train(X, y, epochs=1, verbose=0)
        
        # Evaluate
        metrics = nn.evaluate(X, y)
        
        assert 'loss' in metrics
        # The metric might be named differently based on TensorFlow version
        accuracy_key = 'accuracy' if 'accuracy' in metrics else [k for k in metrics.keys() if 'accuracy' in k.lower()][0] if any('accuracy' in k.lower() for k in metrics.keys()) else list(metrics.keys())[1] if len(metrics) > 1 else None
        
        assert isinstance(metrics['loss'], float)
        if accuracy_key:
            assert isinstance(metrics[accuracy_key], float)
    
    def test_different_optimizers(self):
        """Test different optimizer configurations."""
        optimizers = [
            OptimizerType.ADAM,
            OptimizerType.RMSPROP,
            OptimizerType.SGD,
            OptimizerType.ADAMW
        ]
        
        for optimizer in optimizers:
            config = ModelConfig(
                sequence_length=5,
                num_features=3,
                hidden_units=[8],
                optimizer_type=optimizer
            )
            
            nn = NeuralNetworkArchitecture(config)
            model = nn.build_model()
            
            assert model.optimizer is not None
    
    def test_different_loss_functions(self):
        """Test different loss function configurations."""
        loss_functions = [
            LossType.BINARY_CROSSENTROPY,
            LossType.MEAN_SQUARED_ERROR,
            LossType.MEAN_ABSOLUTE_ERROR,
            LossType.HUBER
        ]
        
        for loss_func in loss_functions:
            config = ModelConfig(
                sequence_length=5,
                num_features=3,
                hidden_units=[8],
                loss_function=loss_func,
                output_activation='linear' if loss_func != LossType.BINARY_CROSSENTROPY else 'sigmoid'
            )
            
            nn = NeuralNetworkArchitecture(config)
            model = nn.build_model()
            
            assert model.loss == loss_func.value
    
    def test_regularization_options(self):
        """Test regularization configurations."""
        config = ModelConfig(
            sequence_length=5,
            num_features=3,
            hidden_units=[16, 8],
            l1_reg=0.01,
            l2_reg=0.01,
            dropout_rate=0.3,
            batch_normalization=True,
            layer_normalization=True
        )
        
        nn = NeuralNetworkArchitecture(config)
        model = nn.build_model()
        
        # Check for regularization layers
        layer_types = [type(layer).__name__ for layer in model.layers]
        assert 'Dropout' in layer_types
        assert 'BatchNormalization' in layer_types or 'LayerNormalization' in layer_types
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        config = ModelConfig(
            sequence_length=5,
            num_features=3,
            hidden_units=[8, 4]
        )
        
        nn = NeuralNetworkArchitecture(config)
        nn.build_model()
        
        # Train briefly
        X = np.random.random((10, 5, 3))
        y = np.random.randint(0, 2, (10, 1))
        nn.train(X, y, epochs=1, verbose=0)
        
        # Test saving and loading
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.keras")
            
            # Save model
            success = nn.save_model(model_path)
            assert success
            assert os.path.exists(model_path)
            
            # Create new instance and load
            nn2 = NeuralNetworkArchitecture(config)
            success = nn2.load_model(model_path)
            assert success
            assert nn2.model is not None
            
            # Test predictions are similar
            pred1 = nn.predict(X)
            pred2 = nn2.predict(X)
            np.testing.assert_allclose(pred1, pred2, rtol=1e-5)


class TestFeatureEngineer:
    """Test feature engineering pipeline."""
    
    def test_feature_config_defaults(self):
        """Test default feature configuration."""
        config = FeatureConfig()
        
        assert config.use_sma is True
        assert config.use_ema is True
        assert config.use_rsi is True
        assert config.use_macd is True
        assert config.use_bollinger is True
        assert config.sma_periods == [5, 10, 20, 50]
        assert config.ema_periods == [12, 26, 50]
        assert config.scaling_method == "standard"
    
    def test_feature_engineer_initialization(self):
        """Test feature engineer initialization."""
        config = FeatureConfig()
        fe = FeatureEngineer(config)
        
        assert fe.config == config
        assert fe.indicators is not None
        assert fe.scaler is not None
        assert not fe.is_fitted_
    
    def test_feature_engineering_with_sample_data(self):
        """Test feature engineering with sample market data."""
        # Create sample data
        sample_data = []
        for i in range(100):
            sample_data.append({
                'price': 50000 + i * 10 + np.random.randn() * 100,
                'volume': 100 + np.random.randn() * 20,
                'timestamp': 1640000000 + i * 60  # Unix timestamp
            })
        
        fe = FeatureEngineer()
        
        # Test fit_transform
        features = fe.fit_transform(sample_data)
        
        assert features.shape[0] == len(sample_data)
        assert features.shape[1] > 10  # Should have many engineered features
        assert fe.is_fitted_
        
        # Test feature names
        feature_names = fe.get_feature_names()
        assert len(feature_names) > 0
        assert 'sma_5' in feature_names or any('sma' in name for name in feature_names)
    
    def test_different_scaling_methods(self):
        """Test different scaling methods."""
        sample_data = [
            {'price': 50000 + i, 'volume': 100, 'timestamp': 1640000000 + i * 60}
            for i in range(50)
        ]
        
        scaling_methods = ['standard', 'minmax', 'robust', 'quantile']
        
        for method in scaling_methods:
            config = FeatureConfig(scaling_method=method)
            fe = FeatureEngineer(config)
            
            features = fe.fit_transform(sample_data)
            assert features is not None
            assert not np.any(np.isnan(features))
    
    def test_sequence_preparation(self):
        """Test sequence preparation for time series models."""
        sample_data = [
            {'price': 50000 + i, 'volume': 100, 'timestamp': 1640000000 + i * 60}
            for i in range(30)
        ]
        
        fe = FeatureEngineer()
        fe.fit(sample_data)
        
        sequences, timestamps = fe.prepare_sequences(sample_data, sequence_length=10)
        
        assert len(sequences) == len(sample_data) - 10
        assert sequences.shape[1] == 10  # Sequence length
        assert len(timestamps) == len(sequences)
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        # Very little data
        sample_data = [
            {'price': 50000, 'volume': 100, 'timestamp': 1640000000}
        ]
        
        fe = FeatureEngineer()
        features = fe.fit_transform(sample_data)
        
        assert features is not None
        assert features.shape[0] == 1
        
        # Test sequence preparation with insufficient data
        sequences, timestamps = fe.prepare_sequences(sample_data, sequence_length=10)
        
        assert len(sequences) == 0
        assert len(timestamps) == 0


class TestModelManager:
    """Test model management system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_manager = ModelManager(models_dir=self.temp_dir, max_models=5)
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_model_manager_initialization(self):
        """Test model manager initialization."""
        self.setUp()
        
        assert os.path.exists(self.temp_dir)
        assert self.model_manager.max_models == 5
        assert len(self.model_manager.model_registry) == 0
        
        self.tearDown()
    
    def test_create_model(self):
        """Test model creation."""
        self.setUp()
        
        model_config = ModelConfig(
            sequence_length=10,
            num_features=5,
            hidden_units=[16, 8]
        )
        
        model_id = self.model_manager.create_model(model_config=model_config)
        
        assert model_id is not None
        assert self.model_manager.current_model is not None
        assert self.model_manager.current_feature_engineer is not None
        assert self.model_manager.current_metadata is not None
        assert self.model_manager.current_metadata.model_id == model_id
        
        self.tearDown()
    
    def test_model_training_and_validation(self):
        """Test model training and validation."""
        self.setUp()
        
        # Create model
        model_id = self.model_manager.create_model()
        
        # Create sample training data with enough samples for sequences
        sample_data = [
            {'price': 50000 + i, 'volume': 100 + i, 'timestamp': 1640000000 + i * 60}
            for i in range(100)  # Increased to ensure enough data for sequences
        ]
        y = np.random.randint(0, 2, 100)
        
        # Train model
        result = self.model_manager.train_model(
            sample_data, y, epochs=2, batch_size=16
        )
        
        assert 'history' in result
        assert 'training_time' in result
        assert result['training_time'] > 0
        
        # Validate model - use enough data for sequences
        val_metrics = self.model_manager.validate_model(sample_data[-50:], y[-50:])
        
        assert 'loss' in val_metrics
        # Check for any accuracy-related metric or compile_metrics
        has_accuracy = any('accuracy' in key.lower() for key in val_metrics.keys()) or 'compile_metrics' in val_metrics
        assert has_accuracy or len(val_metrics) >= 2
        
        self.tearDown()
    
    def test_model_prediction(self):
        """Test model prediction."""
        self.setUp()
        
        # Create and train model
        model_id = self.model_manager.create_model()
        
        sample_data = [
            {'price': 50000 + i, 'volume': 100 + i, 'timestamp': 1640000000 + i * 60}
            for i in range(100)
        ]
        y = np.random.randint(0, 2, 100)
        
        self.model_manager.train_model(sample_data, y, epochs=1)
        
        # Make predictions
        predictions = self.model_manager.predict(sample_data[-25:])
        
        assert predictions is not None
        # Note: predictions will be fewer than input due to sequence creation
        assert len(predictions) > 0  
        assert predictions.shape[1] == 1
        
        self.tearDown()
    
    def test_checkpoint_save_load(self):
        """Test model checkpoint saving and loading."""
        self.setUp()
        
        # Create and train model
        model_id = self.model_manager.create_model()
        
        sample_data = [
            {'price': 50000 + i, 'volume': 100 + i, 'timestamp': 1640000000 + i * 60}
            for i in range(100)
        ]
        y = np.random.randint(0, 2, 100)
        
        self.model_manager.train_model(sample_data, y, epochs=1)
        
        # Save checkpoint
        success = self.model_manager.save_checkpoint()
        assert success
        
        # Load checkpoint
        new_manager = ModelManager(models_dir=self.temp_dir)
        success = new_manager.load_checkpoint(model_id)
        assert success
        assert new_manager.current_model is not None
        assert new_manager.current_feature_engineer is not None
        
        self.tearDown()
    
    def test_model_listing_and_deletion(self):
        """Test model listing and deletion."""
        self.setUp()
        
        # Create multiple models
        model_ids = []
        sample_data = [
            {'price': 50000 + i, 'volume': 100 + i, 'timestamp': 1640000000 + i * 60}
            for i in range(100)
        ]
        y = np.random.randint(0, 2, 100)
        
        for i in range(3):
            model_id = self.model_manager.create_model(model_id=f"test_model_{i}")
            model_ids.append(model_id)
            # Need to train the model before saving
            self.model_manager.train_model(sample_data, y, epochs=1)
            self.model_manager.save_checkpoint()
        
        # List models
        models = self.model_manager.list_models()
        assert len(models) == 3
        
        # Delete a model
        success = self.model_manager.delete_model(model_ids[0])
        assert success
        
        # Verify deletion
        models = self.model_manager.list_models()
        assert len(models) == 2
        assert model_ids[0] not in self.model_manager.model_registry
        
        self.tearDown()
    
    def test_model_performance_tracking(self):
        """Test model performance tracking."""
        self.setUp()
        
        # Create and train model
        model_id = self.model_manager.create_model()
        
        sample_data = [
            {'price': 50000 + i, 'volume': 100 + i, 'timestamp': 1640000000 + i * 60}
            for i in range(100)
        ]
        y = np.random.randint(0, 2, 100)
        
        self.model_manager.train_model(sample_data, y, epochs=2)
        
        # Get performance metrics
        performance = self.model_manager.get_model_performance()
        
        assert 'model_id' in performance
        assert 'loss' in performance
        assert 'accuracy' in performance
        assert 'epochs_trained' in performance
        assert performance['epochs_trained'] == 2
        
        self.tearDown()


if __name__ == '__main__':
    pytest.main([__file__])
