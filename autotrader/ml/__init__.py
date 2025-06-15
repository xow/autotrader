"""
Machine Learning components for the Autotrader Bot

Contains neural networks, continuous learning algorithms,
and model management functionality.
"""

from .continuous_learner import ContinuousLearner, LearningConfig
from .model_manager import ModelManager, ModelMetadata
from .feature_engineer import FeatureEngineer, FeatureConfig
from .neural_network import NeuralNetworkArchitecture, ModelConfig, ModelType, LossType, OptimizerType
from .indicators import TechnicalIndicators
from .models import LSTMModel
from .training import ContinuousLearner as TrainingContinuousLearner

__all__ = [
    "ContinuousLearner",
    "LearningConfig",
    "ModelManager",
    "ModelMetadata", 
    "FeatureEngineer",
    "FeatureConfig",
    "NeuralNetworkArchitecture",
    "ModelConfig",
    "ModelType",
    "LossType", 
    "OptimizerType",
    "TechnicalIndicators",
    "LSTMModel",
    "TrainingContinuousLearner"
]
