"""
Model Management System for autotrader bot.

Handles model lifecycle including creation, training, validation, checkpointing,
and performance tracking for neural network models.
"""

import os
import json
import pickle
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path

from .neural_network import NeuralNetworkArchitecture, ModelConfig
from .feature_engineer import FeatureEngineer, FeatureConfig

logger = logging.getLogger("autotrader.ml.model_manager")


@dataclass
class ModelMetadata:
    """Metadata for model checkpoints."""
    
    model_id: str
    created_at: datetime
    updated_at: datetime
    model_type: str
    sequence_length: int
    num_features: int
    training_samples: int
    validation_samples: int
    
    # Performance metrics
    loss: float
    accuracy: float
    val_loss: float
    val_accuracy: float
    
    # Training info
    epochs_trained: int
    total_training_time: float
    learning_rate: float
    
    # File paths
    model_path: str
    feature_engineer_path: str
    config_path: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['created_at'] = self.created_at.isoformat()
        result['updated_at'] = self.updated_at.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelMetadata':
        """Create from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


class ModelManager:
    """
    Comprehensive model management system.
    
    Handles model creation, training, validation, checkpointing, and performance
    tracking with automatic cleanup and version management.
    """
    
    def __init__(
        self,
        models_dir: str = "models",
        max_models: int = 10,
        auto_cleanup: bool = True
    ):
        """
        Initialize model manager.
        
        Args:
            models_dir: Directory to store models
            max_models: Maximum number of models to keep
            auto_cleanup: Whether to auto-cleanup old models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.max_models = max_models
        self.auto_cleanup = auto_cleanup
        
        self.current_model: Optional[NeuralNetworkArchitecture] = None
        self.current_feature_engineer: Optional[FeatureEngineer] = None
        self.current_metadata: Optional[ModelMetadata] = None
        
        # Model registry
        self.model_registry: Dict[str, ModelMetadata] = {}
        self._load_registry()
        
        logger.info(f"Model manager initialized with {len(self.model_registry)} registered models")
    
    def create_model(
        self,
        model_config: Optional[ModelConfig] = None,
        feature_config: Optional[FeatureConfig] = None,
        model_id: Optional[str] = None
    ) -> str:
        """
        Create a new model with feature engineer.
        
        Args:
            model_config: Neural network configuration
            feature_config: Feature engineering configuration
            model_id: Custom model ID
        
        Returns:
            Model ID
        """
        # Generate model ID if not provided
        if model_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_id = f"model_{timestamp}"
        
        # Use default configs if not provided
        if feature_config is None:
            feature_config = FeatureConfig()
        
        # Create a feature engineer to determine the actual number of features
        temp_fe = FeatureEngineer(feature_config)
        # Create dummy data to fit and determine feature count
        dummy_data = [
            {'price': 50000, 'volume': 100, 'timestamp': 1640000000}
            for _ in range(50)
        ]
        temp_fe.fit(dummy_data)
        actual_num_features = len(temp_fe.get_feature_names())
        
        if model_config is None:
            model_config = ModelConfig(num_features=actual_num_features)
        else:
            # Update num_features to match actual features
            model_config.num_features = actual_num_features
        
        # Create model and feature engineer
        self.current_model = NeuralNetworkArchitecture(model_config)
        self.current_feature_engineer = FeatureEngineer(feature_config)
        
        # Create model metadata
        self.current_metadata = ModelMetadata(
            model_id=model_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            model_type=model_config.model_type.value,
            sequence_length=model_config.sequence_length,
            num_features=model_config.num_features,
            training_samples=0,
            validation_samples=0,
            loss=float('inf'),
            accuracy=0.0,
            val_loss=float('inf'),
            val_accuracy=0.0,
            epochs_trained=0,
            total_training_time=0.0,
            learning_rate=model_config.learning_rate,
            model_path=str(self.models_dir / f"{model_id}.keras"),
            feature_engineer_path=str(self.models_dir / f"{model_id}_features.pkl"),
            config_path=str(self.models_dir / f"{model_id}_config.json")
        )
        
        # Save configurations
        self._save_configs(model_config, feature_config)
        
        logger.info(f"Created new model: {model_id}")
        return model_id
    
    def train_model(
        self,
        X: Union[List[Dict], np.ndarray],
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        save_checkpoint: bool = True
    ) -> Dict[str, Any]:
        """
        Train the current model.
        
        Args:
            X: Training data
            y: Training targets
            validation_data: Validation data tuple
            epochs: Number of epochs
            batch_size: Batch size
            validation_split: Validation split ratio
            save_checkpoint: Whether to save checkpoint after training
        
        Returns:
            Training history and metrics
        """
        if self.current_model is None or self.current_feature_engineer is None:
            raise ValueError("No model created. Call create_model() first.")
        
        start_time = datetime.now()
        
        try:
            # Prepare features if X is raw data
            if isinstance(X, list):
                # Fit feature engineer on training data
                self.current_feature_engineer.fit(X)
                # Create sequences for time series models
                X_features, _ = self.current_feature_engineer.prepare_sequences(X, self.current_metadata.sequence_length)
                
                if len(X_features) == 0:
                    raise ValueError("Insufficient data to create sequences")
                
                # Adjust labels to match sequences
                if len(y) > len(X_features):
                    y = y[-len(X_features):]
                elif len(y) < len(X_features):
                    X_features = X_features[:len(y)]
            else:
                X_features = X
            
            # Build model if not already built
            if self.current_model.model is None:
                self.current_model.build_model()
            
            # Prepare validation data
            val_data = None
            if validation_data is not None:
                val_X, val_y = validation_data
                if isinstance(val_X, list):
                    val_X_seq, _ = self.current_feature_engineer.prepare_sequences(val_X, self.current_metadata.sequence_length)
                    if len(val_X_seq) > 0:
                        # Adjust validation labels to match sequences
                        if len(val_y) > len(val_X_seq):
                            val_y = val_y[-len(val_X_seq):]
                        elif len(val_y) < len(val_X_seq):
                            val_X_seq = val_X_seq[:len(val_y)]
                        val_data = (val_X_seq, val_y)
                else:
                    val_data = (val_X, val_y)
            
            # Train model
            history = self.current_model.train(
                X_features,
                y,
                validation_data=val_data,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split if val_data is None else 0.0,
                verbose=1
            )
            
            # Update metadata
            training_time = (datetime.now() - start_time).total_seconds()
            self._update_metadata_after_training(history, X_features, training_time, epochs)
            
            # Save checkpoint if requested
            if save_checkpoint:
                self.save_checkpoint()
            
            logger.info(f"Training completed in {training_time:.2f}s")
            
            return {
                'history': history,
                'training_time': training_time,
                'final_loss': history['loss'][-1],
                'final_accuracy': history.get('accuracy', [0])[-1],
                'val_loss': history.get('val_loss', [float('inf')])[-1],
                'val_accuracy': history.get('val_accuracy', [0])[-1]
            }
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def validate_model(
        self,
        X_val: Union[List[Dict], np.ndarray],
        y_val: np.ndarray,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Validate the current model on validation data.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
            batch_size: Batch size for validation
        
        Returns:
            Validation metrics
        """
        if self.current_model is None or self.current_feature_engineer is None:
            raise ValueError("No trained model available")
        
        try:
            # Prepare features
            if isinstance(X_val, list):
                X_features, _ = self.current_feature_engineer.prepare_sequences(X_val, self.current_metadata.sequence_length)
                
                if len(X_features) == 0:
                    raise ValueError("Insufficient validation data to create sequences")
                
                # Adjust labels to match sequences
                if len(y_val) > len(X_features):
                    y_val = y_val[-len(X_features):]
                elif len(y_val) < len(X_features):
                    X_features = X_features[:len(y_val)]
            else:
                X_features = X_val
            
            # Evaluate model
            metrics = self.current_model.evaluate(X_features, y_val, batch_size)
            
            logger.info(f"Validation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            raise
    
    def predict(
        self,
        X: Union[List[Dict], np.ndarray],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Make predictions with the current model.
        
        Args:
            X: Input data
            batch_size: Batch size for prediction
        
        Returns:
            Predictions array
        """
        if self.current_model is None or self.current_feature_engineer is None:
            raise ValueError("No trained model available")
        
        try:
            # Prepare features
            if isinstance(X, list):
                X_features, _ = self.current_feature_engineer.prepare_sequences(X, self.current_metadata.sequence_length)
                
                if len(X_features) == 0:
                    raise ValueError("Insufficient data to create sequences")
            else:
                X_features = X
            
            # Make predictions
            predictions = self.current_model.predict(X_features, batch_size)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def save_checkpoint(self, model_id: Optional[str] = None) -> bool:
        """
        Save model checkpoint with metadata.
        
        Args:
            model_id: Model ID (uses current if not provided)
        
        Returns:
            True if successful
        """
        if self.current_model is None or self.current_feature_engineer is None or self.current_metadata is None:
            logger.error("No model to save")
            return False
        
        if model_id:
            self.current_metadata.model_id = model_id
            # Update file paths
            self.current_metadata.model_path = str(self.models_dir / f"{model_id}.keras")
            self.current_metadata.feature_engineer_path = str(self.models_dir / f"{model_id}_features.pkl")
            self.current_metadata.config_path = str(self.models_dir / f"{model_id}_config.json")
        
        try:
            # Save model
            if not self.current_model.save_model(self.current_metadata.model_path):
                return False
            
            # Save feature engineer
            with open(self.current_metadata.feature_engineer_path, 'wb') as f:
                pickle.dump(self.current_feature_engineer, f)
            
            # Update metadata timestamp
            self.current_metadata.updated_at = datetime.now()
            
            # Register model
            self.model_registry[self.current_metadata.model_id] = self.current_metadata
            
            # Save registry
            self._save_registry()
            
            # Auto cleanup if enabled
            if self.auto_cleanup:
                self._cleanup_old_models()
            
            logger.info(f"Checkpoint saved: {self.current_metadata.model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            return False
    
    def load_checkpoint(self, model_id: str) -> bool:
        """
        Load model checkpoint.
        
        Args:
            model_id: Model ID to load
        
        Returns:
            True if successful
        """
        if model_id not in self.model_registry:
            logger.error(f"Model not found: {model_id}")
            return False
        
        metadata = self.model_registry[model_id]
        
        try:
            # Load configurations
            configs = self._load_configs(metadata.config_path)
            if not configs:
                return False
            
            model_config, feature_config = configs
            
            # Create architecture
            self.current_model = NeuralNetworkArchitecture(model_config)
            
            # Load trained model
            if not self.current_model.load_model(metadata.model_path):
                return False
            
            # Load feature engineer
            with open(metadata.feature_engineer_path, 'rb') as f:
                self.current_feature_engineer = pickle.load(f)
            
            # Set current metadata
            self.current_metadata = metadata
            
            logger.info(f"Checkpoint loaded: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return False
    
    def list_models(self) -> List[Dict]:
        """
        List all registered models with metadata.
        
        Returns:
            List of model metadata dictionaries
        """
        models = []
        for model_id, metadata in self.model_registry.items():
            model_info = metadata.to_dict()
            model_info['file_exists'] = os.path.exists(metadata.model_path)
            models.append(model_info)
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x['created_at'], reverse=True)
        
        return models
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model and its associated files.
        
        Args:
            model_id: Model ID to delete
        
        Returns:
            True if successful
        """
        if model_id not in self.model_registry:
            logger.error(f"Model not found: {model_id}")
            return False
        
        metadata = self.model_registry[model_id]
        
        try:
            # Delete files
            files_to_delete = [
                metadata.model_path,
                metadata.feature_engineer_path,
                metadata.config_path
            ]
            
            for file_path in files_to_delete:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Deleted: {file_path}")
            
            # Remove from registry
            del self.model_registry[model_id]
            
            # Save updated registry
            self._save_registry()
            
            # Clear current model if it was the deleted one
            if self.current_metadata and self.current_metadata.model_id == model_id:
                self.current_model = None
                self.current_feature_engineer = None
                self.current_metadata = None
            
            logger.info(f"Model deleted: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            return False
    
    def get_model_performance(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics for a model.
        
        Args:
            model_id: Model ID (uses current if not provided)
        
        Returns:
            Performance metrics dictionary
        """
        if model_id:
            if model_id not in self.model_registry:
                return {}
            metadata = self.model_registry[model_id]
        else:
            if self.current_metadata is None:
                return {}
            metadata = self.current_metadata
        
        return {
            'model_id': metadata.model_id,
            'model_type': metadata.model_type,
            'loss': metadata.loss,
            'accuracy': metadata.accuracy,
            'val_loss': metadata.val_loss,
            'val_accuracy': metadata.val_accuracy,
            'epochs_trained': metadata.epochs_trained,
            'training_samples': metadata.training_samples,
            'validation_samples': metadata.validation_samples,
            'training_time': metadata.total_training_time,
            'created_at': metadata.created_at.isoformat(),
            'updated_at': metadata.updated_at.isoformat()
        }
    
    def _update_metadata_after_training(
        self,
        history: Dict,
        X_features: np.ndarray,
        training_time: float,
        epochs: int
    ):
        """Update metadata after training."""
        if self.current_metadata is None:
            return
        
        # Update performance metrics
        self.current_metadata.loss = history['loss'][-1]
        self.current_metadata.accuracy = history.get('accuracy', [0])[-1]
        self.current_metadata.val_loss = history.get('val_loss', [float('inf')])[-1]
        self.current_metadata.val_accuracy = history.get('val_accuracy', [0])[-1]
        
        # Update training info
        self.current_metadata.epochs_trained += epochs
        self.current_metadata.total_training_time += training_time
        self.current_metadata.training_samples = len(X_features)
        self.current_metadata.updated_at = datetime.now()
    
    def _save_configs(self, model_config: ModelConfig, feature_config: FeatureConfig):
        """Save model and feature configurations."""
        if self.current_metadata is None:
            return
        
        configs = {
            'model_config': asdict(model_config),
            'feature_config': asdict(feature_config)
        }
        
        # Convert enums to strings for JSON serialization
        if 'model_type' in configs['model_config']:
            configs['model_config']['model_type'] = configs['model_config']['model_type'].value
        if 'loss_function' in configs['model_config']:
            configs['model_config']['loss_function'] = configs['model_config']['loss_function'].value
        if 'optimizer_type' in configs['model_config']:
            configs['model_config']['optimizer_type'] = configs['model_config']['optimizer_type'].value
        
        with open(self.current_metadata.config_path, 'w') as f:
            json.dump(configs, f, indent=2)
    
    def _load_configs(self, config_path: str) -> Optional[Tuple[ModelConfig, FeatureConfig]]:
        """Load model and feature configurations."""
        try:
            with open(config_path, 'r') as f:
                configs = json.load(f)
            
            # Convert string enums back
            from .neural_network import ModelType, LossType, OptimizerType
            
            model_config_dict = configs['model_config']
            if 'model_type' in model_config_dict:
                model_config_dict['model_type'] = ModelType(model_config_dict['model_type'])
            if 'loss_function' in model_config_dict:
                model_config_dict['loss_function'] = LossType(model_config_dict['loss_function'])
            if 'optimizer_type' in model_config_dict:
                model_config_dict['optimizer_type'] = OptimizerType(model_config_dict['optimizer_type'])
            
            model_config = ModelConfig(**model_config_dict)
            feature_config = FeatureConfig(**configs['feature_config'])
            
            return model_config, feature_config
            
        except Exception as e:
            logger.error(f"Error loading configs: {e}")
            return None
    
    def _save_registry(self):
        """Save model registry to disk."""
        registry_path = self.models_dir / "registry.json"
        registry_data = {
            model_id: metadata.to_dict()
            for model_id, metadata in self.model_registry.items()
        }
        
        with open(registry_path, 'w') as f:
            json.dump(registry_data, f, indent=2)
    
    def _load_registry(self):
        """Load model registry from disk."""
        registry_path = self.models_dir / "registry.json"
        
        if not registry_path.exists():
            return
        
        try:
            with open(registry_path, 'r') as f:
                registry_data = json.load(f)
            
            self.model_registry = {
                model_id: ModelMetadata.from_dict(data)
                for model_id, data in registry_data.items()
            }
            
        except Exception as e:
            logger.error(f"Error loading registry: {e}")
            self.model_registry = {}
    
    def _cleanup_old_models(self):
        """Remove old models to stay within max_models limit."""
        if len(self.model_registry) <= self.max_models:
            return
        
        # Sort by creation date (oldest first)
        sorted_models = sorted(
            self.model_registry.items(),
            key=lambda x: x[1].created_at
        )
        
        # Delete oldest models
        models_to_delete = len(self.model_registry) - self.max_models
        for i in range(models_to_delete):
            model_id, _ = sorted_models[i]
            logger.info(f"Auto-cleanup: deleting old model {model_id}")
            self.delete_model(model_id)
    
    def get_best_model(self, metric: str = 'val_accuracy') -> Optional[str]:
        """
        Get the best model based on a metric.
        
        Args:
            metric: Metric to use for comparison
        
        Returns:
            Best model ID or None
        """
        if not self.model_registry:
            return None
        
        best_model_id = None
        best_score = float('-inf') if 'accuracy' in metric else float('inf')
        
        for model_id, metadata in self.model_registry.items():
            score = getattr(metadata, metric, None)
            if score is None:
                continue
            
            if 'accuracy' in metric:
                if score > best_score:
                    best_score = score
                    best_model_id = model_id
            else:  # loss metrics
                if score < best_score:
                    best_score = score
                    best_model_id = model_id
        
        return best_model_id
