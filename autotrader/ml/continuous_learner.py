"""
Advanced Continuous Learning Engine for autotrader bot.

Implements sophisticated online learning algorithms with incremental training,
experience replay, adaptive learning rate scheduling, confidence scoring,
and comprehensive performance tracking.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import logging
from dataclasses import dataclass, field
from collections import deque
import pickle
import threading
import time
import json
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from enum import Enum
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from .model_manager import ModelManager
from .neural_network import ModelConfig
from .feature_engineer import FeatureEngineer, FeatureConfig

logger = logging.getLogger("autotrader.ml.continuous_learner")


class PerformanceMetric(Enum):
    """Performance metrics for adaptive learning."""
    ACCURACY = "accuracy"
    LOSS = "loss"
    VAL_ACCURACY = "val_accuracy"
    VAL_LOSS = "val_loss"
    F1_SCORE = "f1_score"
    PRECISION = "precision"
    RECALL = "recall"


class LearningRateSchedule(Enum):
    """Learning rate scheduling strategies."""
    COSINE = "cosine"
    EXPONENTIAL = "exponential"
    PLATEAU = "plateau"
    PERFORMANCE_BASED = "performance_based"
    ADAPTIVE = "adaptive"


@dataclass
class ConfidenceConfig:
    """Configuration for prediction confidence scoring."""
    
    # Ensemble settings
    enable_ensemble: bool = True
    ensemble_size: int = 5
    bootstrap_ratio: float = 0.8
    
    # Uncertainty quantification
    enable_monte_carlo_dropout: bool = True
    mc_dropout_samples: int = 10
    mc_dropout_rate: float = 0.1
    
    # Prediction variance
    enable_prediction_variance: bool = True
    variance_window: int = 100
    
    # Performance-based confidence
    performance_weight: float = 0.4
    consistency_weight: float = 0.3
    uncertainty_weight: float = 0.3
    
    # Confidence thresholds
    high_confidence_threshold: float = 0.8
    low_confidence_threshold: float = 0.3


@dataclass
class OnlineLearningConfig:
    """Configuration for online learning algorithms."""
    
    # SGD configurations
    momentum: float = 0.9
    nesterov: bool = True
    weight_decay: float = 1e-4
    
    # Adaptive learning rate
    lr_schedule: LearningRateSchedule = LearningRateSchedule.ADAPTIVE
    warmup_epochs: int = 10
    cooldown_epochs: int = 5
    patience: int = 10
    
    # Performance tracking
    tracking_metrics: List[PerformanceMetric] = field(default_factory=lambda: [
        PerformanceMetric.ACCURACY, PerformanceMetric.LOSS, 
        PerformanceMetric.VAL_ACCURACY, PerformanceMetric.VAL_LOSS
    ])
    
    # Memory management
    gradient_accumulation_steps: int = 1
    max_gradient_norm: float = 1.0
    enable_gradient_checkpointing: bool = False


@dataclass
class ExperienceReplayConfig:
    """Configuration for experience replay buffer."""
    
    # Buffer management
    buffer_size: int = 50000
    prioritized_replay: bool = True
    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    priority_beta_annealing: float = 0.001
    
    # Sampling strategy
    uniform_ratio: float = 0.1  # Ratio of uniform vs prioritized sampling
    recent_ratio: float = 0.3   # Ratio of recent vs all experiences
    
    # Experience importance
    temporal_decay: float = 0.99
    error_clip: float = 1.0
    min_priority: float = 1e-6


@dataclass
class LearningConfig:
    """Enhanced configuration for continuous learning."""
    
    # Buffer settings
    buffer_size: int = 10000
    min_buffer_size: int = 100
    
    # Training settings
    batch_size: int = 32
    epochs_per_update: int = 1
    validation_split: float = 0.2
    
    # Learning schedule
    update_frequency: int = 10  # Update every N new samples
    full_retrain_frequency: int = 1000  # Full retrain every N samples
    
    # Performance monitoring
    performance_window: int = 100
    min_accuracy_threshold: float = 0.55
    max_loss_threshold: float = 1.0
    
    # Adaptive learning
    adaptive_learning_rate: bool = True
    learning_rate_decay: float = 0.95
    min_learning_rate: float = 1e-6
    
    # Experience replay
    use_experience_replay: bool = True
    replay_ratio: float = 0.3  # Ratio of old samples in each batch
    
    # Model management
    auto_save_frequency: int = 100  # Save every N updates
    keep_best_model: bool = True
    
    # Enhanced configurations
    online_config: OnlineLearningConfig = field(default_factory=OnlineLearningConfig)
    replay_config: ExperienceReplayConfig = field(default_factory=ExperienceReplayConfig)
    confidence_config: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    
    # Advanced learning features
    enable_meta_learning: bool = True
    catastrophic_forgetting_prevention: bool = True
    ewc_lambda: float = 400.0  # Elastic Weight Consolidation
    
    # Performance tracking and logging
    detailed_logging: bool = True
    performance_log_frequency: int = 50
    save_training_history: bool = True


class PerformanceTracker:
    """Advanced performance tracking for continuous learning."""
    
    def __init__(self, config: LearningConfig):
        self.config = config
        self.metrics_history: Dict[str, deque] = {}
        self.session_metrics: Dict[str, List] = {}
        self.benchmark_scores: Dict[str, float] = {}
        self._lock = threading.Lock()
        
        # Initialize metric tracking
        for metric in config.online_config.tracking_metrics:
            self.metrics_history[metric.value] = deque(maxlen=config.performance_window)
            self.session_metrics[metric.value] = []
    
    def update_metrics(self, metrics: Dict[str, float], training_time: float = 0):
        """Update performance metrics."""
        with self._lock:
            timestamp = datetime.now()
            
            for metric_name, value in metrics.items():
                if metric_name in self.metrics_history:
                    self.metrics_history[metric_name].append({
                        'value': value,
                        'timestamp': timestamp,
                        'training_time': training_time
                    })
                    self.session_metrics[metric_name].append(value)
    
    def get_recent_performance(self, metric: str, window: int = 10) -> Dict[str, float]:
        """Get recent performance statistics."""
        with self._lock:
            if metric not in self.metrics_history:
                return {}
            
            recent_values = list(self.metrics_history[metric])[-window:]
            if not recent_values:
                return {}
            
            values = [entry['value'] for entry in recent_values]
            return {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'trend': self._calculate_trend(values),
                'stability': 1.0 / (1.0 + np.std(values))  # Higher is more stable
            }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction (-1 to 1)."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        return np.clip(coeffs[0] * 10, -1.0, 1.0)  # Scale and clip
    
    def should_adjust_learning(self) -> Tuple[bool, str]:
        """Determine if learning parameters should be adjusted."""
        with self._lock:
            # Check validation accuracy trend
            val_acc_stats = self.get_recent_performance('val_accuracy', 20)
            if val_acc_stats and val_acc_stats['trend'] < -0.3:
                return True, "declining_performance"
            
            # Check loss trend
            loss_stats = self.get_recent_performance('loss', 20)
            if loss_stats and loss_stats['trend'] > 0.3:
                return True, "increasing_loss"
            
            # Check stability
            if val_acc_stats and val_acc_stats['stability'] < 0.3:
                return True, "unstable_training"
            
            return False, "stable"
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics for analysis."""
        with self._lock:
            return {
                'metrics_history': {
                    name: list(history) for name, history in self.metrics_history.items()
                },
                'session_summary': {
                    name: {
                        'count': len(values),
                        'mean': np.mean(values) if values else 0,
                        'std': np.std(values) if values else 0,
                        'best': max(values) if values else 0
                    } for name, values in self.session_metrics.items()
                },
                'benchmark_scores': self.benchmark_scores.copy()
            }


class ExperienceBuffer:
    """Advanced experience replay buffer with prioritized sampling."""
    
    def __init__(self, config: ExperienceReplayConfig):
        """
        Initialize experience buffer.
        
        Args:
            config: Experience replay configuration
        """
        self.config = config
        self.max_size = config.buffer_size
        self.buffer = deque(maxlen=self.max_size)
        self.priorities = deque(maxlen=self.max_size)
        self.importance_weights = deque(maxlen=self.max_size)
        self.timestamps = deque(maxlen=self.max_size)
        self._lock = threading.Lock()
        self._sample_count = 0
        
    def add(self, experience: Dict, priority: Optional[float] = None):
        """
        Add experience to buffer with computed priority.
        
        Args:
            experience: Experience dictionary
            priority: Priority value (computed if None)
        """
        with self._lock:
            # Compute priority if not provided
            if priority is None:
                priority = self._compute_priority(experience)
            
            # Apply temporal decay to existing priorities
            if self.config.temporal_decay < 1.0:
                for i in range(len(self.priorities)):
                    self.priorities[i] *= self.config.temporal_decay
            
            # Add new experience
            self.buffer.append(experience)
            self.priorities.append(max(priority, self.config.min_priority))
            self.timestamps.append(datetime.now())
            
            # Update importance weights
            self._update_importance_weights()
    
    def _compute_priority(self, experience: Dict) -> float:
        """Compute priority for experience based on prediction error."""
        if 'prediction' not in experience or 'actual_outcome' not in experience:
            return 1.0
        
        prediction = experience.get('prediction')
        actual = experience.get('actual_outcome')
        
        if prediction is None or actual is None:
            return 1.0
        
        # TD error as priority
        error = abs(float(prediction) - float(actual))
        priority = (error + self.config.min_priority) ** self.config.priority_alpha
        
        return min(priority, self.config.error_clip)
    
    def _update_importance_weights(self):
        """Update importance weights for bias correction."""
        if not self.config.prioritized_replay:
            return
        
        beta = min(1.0, self.config.priority_beta + 
                  self._sample_count * self.config.priority_beta_annealing)
        
        if len(self.priorities) == 0:
            return
        
        max_priority = max(self.priorities)
        self.importance_weights.clear()
        
        for priority in self.priorities:
            weight = (priority / max_priority) ** (-beta)
            self.importance_weights.append(weight)
    
    def sample(self, batch_size: int, use_priority: bool = None) -> Tuple[List[Dict], List[int], List[float]]:
        """
        Sample experiences from buffer.
        
        Args:
            batch_size: Number of experiences to sample
            use_priority: Whether to use priority sampling (uses config if None)
        
        Returns:
            Tuple of (experiences, indices, importance_weights)
        """
        with self._lock:
            if len(self.buffer) == 0:
                return [], [], []
            
            batch_size = min(batch_size, len(self.buffer))
            use_priority = use_priority if use_priority is not None else self.config.prioritized_replay
            
            # Determine sampling strategy
            uniform_samples = int(batch_size * self.config.uniform_ratio)
            recent_samples = int(batch_size * self.config.recent_ratio)
            priority_samples = batch_size - uniform_samples - recent_samples
            
            all_indices = []
            
            # Uniform random sampling
            if uniform_samples > 0:
                uniform_indices = np.random.choice(
                    len(self.buffer), size=uniform_samples, replace=False
                )
                all_indices.extend(uniform_indices)
            
            # Recent sampling
            if recent_samples > 0:
                recent_start = max(0, len(self.buffer) - recent_samples * 2)
                recent_indices = np.random.choice(
                    range(recent_start, len(self.buffer)),
                    size=min(recent_samples, len(self.buffer) - recent_start),
                    replace=False
                )
                all_indices.extend(recent_indices)
            
            # Priority sampling
            if priority_samples > 0 and use_priority:
                remaining_indices = list(set(range(len(self.buffer))) - set(all_indices))
                if remaining_indices:
                    remaining_priorities = [self.priorities[i] for i in remaining_indices]
                    probabilities = np.array(remaining_priorities)
                    probabilities = probabilities / probabilities.sum()
                    
                    priority_indices = np.random.choice(
                        remaining_indices,
                        size=min(priority_samples, len(remaining_indices)),
                        replace=False,
                        p=probabilities
                    )
                    all_indices.extend(priority_indices)
            elif priority_samples > 0:
                # Fallback to uniform if priority disabled
                remaining_indices = list(set(range(len(self.buffer))) - set(all_indices))
                if remaining_indices:
                    priority_indices = np.random.choice(
                        remaining_indices,
                        size=min(priority_samples, len(remaining_indices)),
                        replace=False
                    )
                    all_indices.extend(priority_indices)
            
            # Get experiences and weights
            experiences = [self.buffer[i] for i in all_indices]
            weights = [self.importance_weights[i] if i < len(self.importance_weights) else 1.0 
                      for i in all_indices]
            
            self._sample_count += 1
            
            return experiences, all_indices, weights
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for sampled experiences."""
        with self._lock:
            for idx, priority in zip(indices, priorities):
                if 0 <= idx < len(self.priorities):
                    self.priorities[idx] = max(priority, self.config.min_priority)
            
            self._update_importance_weights()
    
    def get_recent(self, n: int) -> List[Dict]:
        """Get n most recent experiences."""
        with self._lock:
            return list(self.buffer)[-n:] if n <= len(self.buffer) else list(self.buffer)
    
    def size(self) -> int:
        """Get buffer size."""
        return len(self.buffer)
    
    def clear(self):
        """Clear buffer."""
        with self._lock:
            self.buffer.clear()
            self.priorities.clear()
            self.importance_weights.clear()
            self.timestamps.clear()
            self._sample_count = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self._lock:
            if not self.buffer:
                return {}
            
            priorities_array = np.array(list(self.priorities))
            return {
                'size': len(self.buffer),
                'max_size': self.max_size,
                'utilization': len(self.buffer) / self.max_size,
                'priority_stats': {
                    'mean': np.mean(priorities_array),
                    'std': np.std(priorities_array),
                    'min': np.min(priorities_array),
                    'max': np.max(priorities_array)
                },
                'sample_count': self._sample_count,
                'oldest_experience': min(self.timestamps).isoformat() if self.timestamps else None,
                'newest_experience': max(self.timestamps).isoformat() if self.timestamps else None
            }


class AdvancedLearningRateScheduler:
    """Advanced learning rate scheduler with multiple strategies."""
    
    def __init__(self, config: OnlineLearningConfig, initial_lr: float):
        self.config = config
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.step_count = 0
        self.best_metric = None
        self.patience_count = 0
        self.warmup_steps = 0
        
    def get_learning_rate(self, performance_metrics: Dict[str, float]) -> float:
        """Get current learning rate based on performance."""
        self.step_count += 1
        
        if self.config.lr_schedule == LearningRateSchedule.COSINE:
            return self._cosine_schedule()
        elif self.config.lr_schedule == LearningRateSchedule.EXPONENTIAL:
            return self._exponential_schedule()
        elif self.config.lr_schedule == LearningRateSchedule.PLATEAU:
            return self._plateau_schedule(performance_metrics)
        elif self.config.lr_schedule == LearningRateSchedule.PERFORMANCE_BASED:
            return self._performance_based_schedule(performance_metrics)
        else:  # ADAPTIVE
            return self._adaptive_schedule(performance_metrics)
    
    def _cosine_schedule(self) -> float:
        """Cosine annealing schedule."""
        return self.initial_lr * 0.5 * (1 + np.cos(np.pi * self.step_count / 1000))
    
    def _exponential_schedule(self) -> float:
        """Exponential decay schedule."""
        decay_rate = 0.95
        return self.initial_lr * (decay_rate ** (self.step_count / 100))
    
    def _plateau_schedule(self, metrics: Dict[str, float]) -> float:
        """Reduce on plateau schedule."""
        current_metric = metrics.get('val_accuracy', 0)
        
        if self.best_metric is None or current_metric > self.best_metric:
            self.best_metric = current_metric
            self.patience_count = 0
        else:
            self.patience_count += 1
        
        if self.patience_count >= self.config.patience:
            self.current_lr *= 0.5
            self.patience_count = 0
        
        return self.current_lr
    
    def _performance_based_schedule(self, metrics: Dict[str, float]) -> float:
        """Performance-based adaptive learning rate."""
        val_accuracy = metrics.get('val_accuracy', 0)
        loss = metrics.get('loss', 1.0)
        
        # Increase LR if performing well, decrease if struggling
        if val_accuracy > 0.8:
            return min(self.current_lr * 1.05, self.initial_lr * 2)
        elif val_accuracy < 0.6 or loss > 1.0:
            return max(self.current_lr * 0.95, self.initial_lr * 0.1)
        
        return self.current_lr
    
    def _adaptive_schedule(self, metrics: Dict[str, float]) -> float:
        """Adaptive schedule combining multiple strategies."""
        # Warmup phase
        if self.step_count < self.config.warmup_epochs:
            warmup_lr = self.initial_lr * (self.step_count / self.config.warmup_epochs)
            return warmup_lr
        
        # Combine plateau and performance-based
        plateau_lr = self._plateau_schedule(metrics)
        performance_lr = self._performance_based_schedule(metrics)
        
        # Take weighted average
        self.current_lr = 0.7 * plateau_lr + 0.3 * performance_lr
        return self.current_lr


class ConfidenceEstimator:
    """Advanced confidence estimation for predictions."""
    
    def __init__(self, config: ConfidenceConfig):
        self.config = config
        self.prediction_history = deque(maxlen=config.variance_window)
        self.error_history = deque(maxlen=config.variance_window)
        
    def estimate_confidence(
        self,
        model,
        input_data: np.ndarray,
        recent_performance: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Estimate prediction confidence using multiple methods.
        
        Args:
            model: Trained model
            input_data: Input data for prediction
            recent_performance: Recent performance metrics
        
        Returns:
            Confidence scores and uncertainty measures
        """
        confidence_scores = {}
        
        # Monte Carlo Dropout for uncertainty estimation
        if self.config.enable_monte_carlo_dropout:
            mc_confidence = self._monte_carlo_dropout_confidence(model, input_data)
            confidence_scores['mc_dropout'] = mc_confidence
        
        # Prediction variance
        if self.config.enable_prediction_variance:
            variance_confidence = self._prediction_variance_confidence(model, input_data)
            confidence_scores['variance'] = variance_confidence
        
        # Performance-based confidence
        performance_confidence = self._performance_based_confidence(recent_performance)
        confidence_scores['performance'] = performance_confidence
        
        # Ensemble confidence (if enabled)
        if self.config.enable_ensemble:
            ensemble_confidence = self._ensemble_confidence(model, input_data)
            confidence_scores['ensemble'] = ensemble_confidence
        
        # Weighted combination
        final_confidence = self._combine_confidence_scores(confidence_scores)
        
        return {
            'confidence': final_confidence,
            'uncertainty': 1.0 - final_confidence,
            'components': confidence_scores
        }
    
    def _monte_carlo_dropout_confidence(self, model, input_data: np.ndarray) -> float:
        """Monte Carlo Dropout for uncertainty estimation."""
        if not hasattr(model, 'model') or model.model is None:
            return 0.5
        
        predictions = []
        
        # Enable training mode for dropout
        for _ in range(self.config.mc_dropout_samples):
            pred = model.model(input_data, training=True)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        
        # Calculate confidence from prediction variance
        prediction_variance = np.var(predictions, axis=0)
        confidence = 1.0 / (1.0 + prediction_variance.mean())
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _prediction_variance_confidence(self, model, input_data: np.ndarray) -> float:
        """Confidence based on prediction variance over time."""
        if len(self.prediction_history) < 5:
            return 0.5
        
        recent_predictions = np.array(list(self.prediction_history)[-10:])
        variance = np.var(recent_predictions)
        
        # Lower variance = higher confidence
        confidence = 1.0 / (1.0 + variance * 10)
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _performance_based_confidence(self, recent_performance: Dict[str, float]) -> float:
        """Confidence based on recent model performance."""
        accuracy = recent_performance.get('mean', 0.5)
        stability = recent_performance.get('stability', 0.5)
        
        # Combine accuracy and stability
        confidence = (accuracy * 0.7) + (stability * 0.3)
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _ensemble_confidence(self, model, input_data: np.ndarray) -> float:
        """Ensemble-based confidence (simplified implementation)."""
        # This would require multiple models in practice
        # For now, return moderate confidence
        return 0.7
    
    def _combine_confidence_scores(self, scores: Dict[str, float]) -> float:
        """Combine multiple confidence scores using weighted average."""
        if not scores:
            return 0.5
        
        weights = {
            'performance': self.config.performance_weight,
            'variance': self.config.consistency_weight,
            'mc_dropout': self.config.uncertainty_weight,
            'ensemble': 0.2
        }
        
        total_weight = 0
        weighted_sum = 0
        
        for score_type, score in scores.items():
            weight = weights.get(score_type, 0.1)
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        return weighted_sum / total_weight
    
    def update_prediction_history(self, prediction: float, actual: Optional[float] = None):
        """Update prediction history for variance calculation."""
        self.prediction_history.append(prediction)
        
        if actual is not None:
            error = abs(prediction - actual)
            self.error_history.append(error)


class ContinuousLearner:
    """
    Advanced continuous learning engine with sophisticated online training,
    experience replay, adaptive learning rate scheduling, confidence scoring,
    and comprehensive performance tracking.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        learning_config: Optional[LearningConfig] = None,
        model_config: Optional[ModelConfig] = None,
        feature_config: Optional[FeatureConfig] = None
    ):
        """
        Initialize advanced continuous learner.
        
        Args:
            model_manager: Model management system
            learning_config: Enhanced learning configuration
            model_config: Neural network configuration  
            feature_config: Feature engineering configuration
        """
        self.model_manager = model_manager
        self.config = learning_config or LearningConfig()
        self.model_config = model_config or ModelConfig()
        self.feature_config = feature_config or FeatureConfig()
        
        # Advanced components
        self.experience_buffer = ExperienceBuffer(self.config.replay_config)
        self.performance_tracker = PerformanceTracker(self.config)
        self.lr_scheduler = AdvancedLearningRateScheduler(
            self.config.online_config, 
            self.model_config.learning_rate
        )
        self.confidence_estimator = ConfidenceEstimator(self.config.confidence_config)
        
        # Learning state
        self.samples_processed = 0
        self.updates_performed = 0
        self.last_full_retrain = 0
        self.training_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Performance tracking
        self.performance_history = deque(maxlen=self.config.performance_window)
        self.current_learning_rate = self.model_config.learning_rate
        
        # Threading for background updates
        self.update_thread = None
        self.stop_learning = False
        self._learning_lock = threading.Lock()
        
        # Elastic Weight Consolidation for catastrophic forgetting prevention
        self.ewc_fisher_matrix = None
        self.ewc_optimal_params = None
        
        # Initialize checkpointing if model manager supports it
        if hasattr(self.model_manager, 'checkpoint_manager'):
            self.checkpoint_manager = self.model_manager.checkpoint_manager
        else:
            self.checkpoint_manager = None
        
        # Initialize model if not exists
        if not self.model_manager.current_model:
            model_id = self.model_manager.create_model(
                self.model_config,
                self.feature_config
            )
            logger.info(f"Created initial model: {model_id}")
        
        # Setup performance monitoring
        self._setup_performance_monitoring()
        
        logger.info(f"Advanced continuous learner initialized (session: {self.training_session_id})")
    
    def add_experience(
        self,
        market_data: Dict,
        prediction: Optional[float] = None,
        actual_outcome: Optional[float] = None,
        reward: Optional[float] = None,
        confidence: Optional[float] = None
    ):
        """
        Add new market experience to learning buffer with enhanced tracking.
        
        Args:
            market_data: Market data point
            prediction: Model prediction (if available)
            actual_outcome: Actual market outcome
            reward: Trading reward/loss
            confidence: Prediction confidence score
        """
        experience = {
            'timestamp': datetime.now(),
            'market_data': market_data,
            'prediction': prediction,
            'actual_outcome': actual_outcome,
            'reward': reward,
            'confidence': confidence,
            'session_id': self.training_session_id
        }
        
        # Add to experience buffer (priority computed automatically)
        self.experience_buffer.add(experience)
        self.samples_processed += 1
        
        # Update confidence estimator history
        if prediction is not None:
            self.confidence_estimator.update_prediction_history(prediction, actual_outcome)
        
        # Log detailed experience if enabled
        if self.config.detailed_logging and self.samples_processed % self.config.performance_log_frequency == 0:
            logger.debug(f"Experience added: samples={self.samples_processed}, "
                        f"prediction={prediction}, actual={actual_outcome}, "
                        f"confidence={confidence}")
        
        # Trigger learning update if needed
        if self._should_update():
            self._schedule_update()
    
    def _should_update(self) -> bool:
        """Check if model should be updated."""
        # Check buffer size
        if self.experience_buffer.size() < self.config.min_buffer_size:
            return False
        
        # Check update frequency
        if self.samples_processed % self.config.update_frequency != 0:
            return False
        
        return True
    
    def _schedule_update(self):
        """Schedule background model update."""
        if self.update_thread is None or not self.update_thread.is_alive():
            self.update_thread = threading.Thread(target=self._perform_update)
            self.update_thread.daemon = True
            self.update_thread.start()
    
    def _perform_update(self):
        """Perform incremental model update."""
        with self._learning_lock:
            try:
                start_time = time.time()
                
                # Check if full retrain is needed
                if (self.samples_processed - self.last_full_retrain) >= self.config.full_retrain_frequency:
                    self._perform_full_retrain()
                else:
                    self._perform_incremental_update()
                
                update_time = time.time() - start_time
                self.updates_performed += 1
                
                logger.debug(f"Model update completed in {update_time:.2f}s")
                
                # Auto-save if configured
                if self.updates_performed % self.config.auto_save_frequency == 0:
                    self.model_manager.save_checkpoint()
                
            except Exception as e:
                logger.error(f"Error during model update: {e}")
    
    def _perform_incremental_update(self):
        """Perform advanced incremental model update with prioritized experience replay."""
        try:
            # Get experiences using prioritized sampling
            experiences, indices, importance_weights = self.experience_buffer.sample(
                self.config.batch_size,
                use_priority=self.config.replay_config.prioritized_replay
            )
            
            if len(experiences) < self.config.batch_size // 2:
                logger.debug("Insufficient experiences for incremental update")
                return
            
            # Prepare training data
            X, y = self._prepare_training_data(experiences)
            
            if len(X) == 0:
                return
            
            # Apply importance sampling weights
            sample_weights = np.array(importance_weights) if self.config.replay_config.prioritized_replay else None
            
            # Get current performance metrics for adaptive learning
            recent_performance = {}
            if len(self.performance_history) > 0:
                recent_perf = self.performance_history[-1]
                recent_performance = {
                    'val_accuracy': recent_perf.get('val_accuracy', 0.5),
                    'loss': recent_perf.get('loss', 1.0)
                }
            
            # Update learning rate
            new_lr = self.lr_scheduler.get_learning_rate(recent_performance)
            if abs(new_lr - self.current_learning_rate) > 1e-6:
                self.current_learning_rate = new_lr
                if self.model_manager.current_model and self.model_manager.current_model.model:
                    self.model_manager.current_model.model.optimizer.learning_rate.assign(new_lr)
                logger.debug(f"Updated learning rate to {new_lr}")
            
            # Perform training with gradient clipping
            if hasattr(self.model_manager.current_model.model, 'optimizer'):
                # Gradient clipping
                optimizer = self.model_manager.current_model.model.optimizer
                if hasattr(optimizer, 'clipnorm'):
                    optimizer.clipnorm = self.config.online_config.max_gradient_norm
            
            # Train with importance sampling weights
            history = self.model_manager.current_model.train(
                X, y,
                epochs=self.config.epochs_per_update,
                batch_size=min(self.config.batch_size, len(X)),
                validation_split=self.config.validation_split,
                sample_weight=sample_weights,
                verbose=0
            )
            
            # Calculate training errors for priority updates
            if self.config.replay_config.prioritized_replay and len(indices) > 0:
                predictions = self.model_manager.current_model.predict(X, batch_size=len(X))
                errors = np.abs(predictions.flatten() - y)
                
                # Update priorities in experience buffer
                new_priorities = [(error + self.config.replay_config.min_priority) ** self.config.replay_config.priority_alpha 
                                for error in errors]
                self.experience_buffer.update_priorities(indices, new_priorities)
            
            # Enhanced performance tracking
            training_time = 0  # Would be calculated from actual training
            self.performance_tracker.update_metrics({
                'accuracy': history.get('accuracy', [0])[-1],
                'loss': history.get('loss', [1.0])[-1],
                'val_accuracy': history.get('val_accuracy', [0])[-1],
                'val_loss': history.get('val_loss', [1.0])[-1]
            }, training_time)
            
            # Update legacy performance tracking for compatibility
            self._update_performance_tracking(history)
            
            # Check for learning adjustments
            should_adjust, reason = self.performance_tracker.should_adjust_learning()
            if should_adjust:
                logger.info(f"Learning adjustment needed: {reason}")
                self._apply_learning_adjustments(reason)
            
        except Exception as e:
            logger.error(f"Error in incremental training: {e}")
    
    def _apply_learning_adjustments(self, reason: str):
        """Apply learning adjustments based on performance analysis."""
        if reason == "declining_performance":
            # Reduce learning rate more aggressively
            self.current_learning_rate *= 0.5
            logger.info(f"Reducing learning rate to {self.current_learning_rate} due to declining performance")
        elif reason == "increasing_loss":
            # Reset to previous best checkpoint if available
            if self.checkpoint_manager:
                best_checkpoint = self.checkpoint_manager.get_best_checkpoint(
                    self.model_manager.current_metadata.model_id
                )
                if best_checkpoint:
                    logger.info(f"Rolling back to best checkpoint: {best_checkpoint}")
                    self.checkpoint_manager.rollback_to_checkpoint(best_checkpoint, self.model_manager)
        elif reason == "unstable_training":
            # Reduce batch size and learning rate
            self.config.batch_size = max(16, self.config.batch_size // 2)
            self.current_learning_rate *= 0.8
            logger.info(f"Adjusting training parameters for stability: batch_size={self.config.batch_size}, lr={self.current_learning_rate}")
    
    def _perform_full_retrain(self):
        """Perform full model retraining."""
        logger.info("Performing full model retrain")
        
        # Get all experiences
        all_experiences = list(self.experience_buffer.buffer)
        
        if len(all_experiences) < self.config.min_buffer_size:
            logger.warning("Insufficient data for full retrain")
            return
        
        # Prepare full training dataset
        X, y = self._prepare_training_data(all_experiences)
        
        if len(X) == 0:
            return
        
        # Create new model (or reset current one)
        current_model_id = None
        if self.model_manager.current_metadata:
            current_model_id = self.model_manager.current_metadata.model_id
        
        try:
            # Train with more epochs for full retrain
            history = self.model_manager.train_model(
                X, y,
                epochs=max(10, self.config.epochs_per_update * 5),
                batch_size=self.config.batch_size,
                validation_split=self.config.validation_split
            )
            
            self.last_full_retrain = self.samples_processed
            
            # Update performance tracking
            self._update_performance_tracking(history['history'])
            
            logger.info("Full retrain completed successfully")
            
        except Exception as e:
            logger.error(f"Error in full retrain: {e}")
    
    def _prepare_training_data(self, experiences: List[Dict]) -> Tuple[List[Dict], np.ndarray]:
        """
        Prepare training data from experiences.
        
        Args:
            experiences: List of experience dictionaries
        
        Returns:
            Tuple of (features, labels)
        """
        market_data = []
        labels = []
        
        for exp in experiences:
            if 'market_data' not in exp or 'actual_outcome' not in exp:
                continue
            
            if exp['actual_outcome'] is None:
                continue
            
            market_data.append(exp['market_data'])
            
            # Convert actual outcome to binary label (price went up/down)
            current_price = exp['market_data'].get('price', 0)
            if current_price > 0:
                label = 1 if exp['actual_outcome'] > current_price else 0
            else:
                label = 0
            
            labels.append(label)
        
        return market_data, np.array(labels)
    
    def _update_performance_tracking(self, history: Dict):
        """Update performance tracking metrics."""
        if 'loss' in history and len(history['loss']) > 0:
            final_loss = history['loss'][-1]
            final_accuracy = history.get('accuracy', [0.5])[-1]
            
            performance = {
                'timestamp': datetime.now(),
                'loss': final_loss,
                'accuracy': final_accuracy,
                'val_loss': history.get('val_loss', [final_loss])[-1],
                'val_accuracy': history.get('val_accuracy', [final_accuracy])[-1],
                'samples_processed': self.samples_processed,
                'learning_rate': self.current_learning_rate
            }
            
            self.performance_history.append(performance)
    
    def _adjust_learning_rate(self, history: Dict):
        """Adjust learning rate based on performance."""
        if len(self.performance_history) < 2:
            return
        
        current_perf = self.performance_history[-1]
        previous_perf = self.performance_history[-2]
        
        # Check if performance is degrading
        if current_perf['val_loss'] > previous_perf['val_loss']:
            # Reduce learning rate
            self.current_learning_rate *= self.config.learning_rate_decay
            self.current_learning_rate = max(
                self.current_learning_rate,
                self.config.min_learning_rate
            )
            
            # Update model's learning rate
            if self.model_manager.current_model and self.model_manager.current_model.model:
                self.model_manager.current_model.model.optimizer.learning_rate.assign(
                    self.current_learning_rate
                )
                
                logger.debug(f"Reduced learning rate to {self.current_learning_rate}")
    
    def predict(self, market_data: Dict) -> Dict[str, Any]:
        """
        Make advanced prediction with comprehensive confidence estimation.
        
        Args:
            market_data: Current market data
        
        Returns:
            Enhanced prediction dictionary with confidence and uncertainty
        """
        try:
            # Prepare input data
            X_features, _ = self.model_manager.current_feature_engineer.prepare_sequences(
                [market_data], 
                self.model_manager.current_metadata.sequence_length
            )
            
            if len(X_features) == 0:
                return self._get_default_prediction()
            
            # Make base prediction
            predictions = self.model_manager.current_model.predict(X_features, batch_size=1)
            
            if len(predictions) > 0:
                prediction = float(predictions[0][0])
                
                # Advanced confidence estimation
                recent_performance = self.performance_tracker.get_recent_performance('val_accuracy', 10)
                
                confidence_results = self.confidence_estimator.estimate_confidence(
                    self.model_manager.current_model,
                    X_features,
                    recent_performance
                )
                
                # Enhanced trading signal generation
                signal_info = self._get_enhanced_trading_signal(
                    prediction, 
                    confidence_results['confidence'],
                    confidence_results['uncertainty']
                )
                
                # Performance metrics
                performance_summary = self._get_performance_summary()
                
                return {
                    'prediction': prediction,
                    'confidence': confidence_results['confidence'],
                    'uncertainty': confidence_results['uncertainty'],
                    'confidence_components': confidence_results['components'],
                    'signal': signal_info['signal'],
                    'signal_strength': signal_info['strength'],
                    'signal_reasoning': signal_info['reasoning'],
                    'model_performance': performance_summary,
                    'timestamp': datetime.now(),
                    'session_id': self.training_session_id
                }
            else:
                return self._get_default_prediction()
                
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return self._get_default_prediction()
    
    def _calculate_confidence(self, prediction: float) -> float:
        """Calculate prediction confidence based on recent performance."""
        if len(self.performance_history) == 0:
            return 0.5  # Neutral confidence
        
        # Use recent accuracy as base confidence
        recent_accuracy = np.mean([
            p['val_accuracy'] for p in list(self.performance_history)[-10:]
        ])
        
        # Adjust based on prediction extremity (more confident for extreme predictions)
        extremity_bonus = abs(prediction - 0.5) * 0.2
        
        confidence = min(recent_accuracy + extremity_bonus, 1.0)
        
        return float(confidence)
    
    def _get_trading_signal(self, prediction: float, confidence: float) -> str:
        """Convert prediction and confidence to trading signal."""
        # Require high confidence for trading signals
        if confidence < 0.6:
            return "HOLD"
        
        if prediction > 0.7:
            return "BUY"
        elif prediction < 0.3:
            return "SELL"
        else:
            return "HOLD"
    
    def _setup_performance_monitoring(self):
        """Setup performance monitoring and logging."""
        if self.config.detailed_logging:
            logger.info(f"Performance monitoring enabled for session {self.training_session_id}")
            logger.info(f"Tracking metrics: {[m.value for m in self.config.online_config.tracking_metrics]}")
    
    def _get_enhanced_trading_signal(
        self, 
        prediction: float, 
        confidence: float,
        uncertainty: float
    ) -> Dict[str, Any]:
        """Generate enhanced trading signal with reasoning."""
        
        # Base signal logic
        if confidence < self.config.confidence_config.low_confidence_threshold:
            signal = "HOLD"
            strength = 0.1
            reasoning = f"Low confidence ({confidence:.3f}) - insufficient certainty for trading"
        elif confidence > self.config.confidence_config.high_confidence_threshold:
            if prediction > 0.7:
                signal = "BUY"
                strength = min((prediction - 0.7) / 0.3 * confidence, 1.0)
                reasoning = f"High confidence ({confidence:.3f}) buy signal - prediction {prediction:.3f}"
            elif prediction < 0.3:
                signal = "SELL"
                strength = min((0.3 - prediction) / 0.3 * confidence, 1.0)
                reasoning = f"High confidence ({confidence:.3f}) sell signal - prediction {prediction:.3f}"
            else:
                signal = "HOLD"
                strength = 0.3
                reasoning = f"Neutral prediction ({prediction:.3f}) despite high confidence"
        else:
            # Medium confidence - more conservative
            if prediction > 0.8:
                signal = "BUY"
                strength = (prediction - 0.8) / 0.2 * confidence * 0.7
                reasoning = f"Medium confidence ({confidence:.3f}) buy signal"
            elif prediction < 0.2:
                signal = "SELL"
                strength = (0.2 - prediction) / 0.2 * confidence * 0.7
                reasoning = f"Medium confidence ({confidence:.3f}) sell signal"
            else:
                signal = "HOLD"
                strength = 0.2
                reasoning = f"Medium confidence ({confidence:.3f}) - holding position"
        
        # Adjust for uncertainty
        if uncertainty > 0.7:
            strength *= 0.5
            reasoning += f" - reduced strength due to high uncertainty ({uncertainty:.3f})"
        
        return {
            'signal': signal,
            'strength': max(0.0, min(1.0, strength)),
            'reasoning': reasoning
        }
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary."""
        return {
            'samples_processed': self.samples_processed,
            'updates_performed': self.updates_performed,
            'current_learning_rate': self.current_learning_rate,
            'buffer_utilization': self.experience_buffer.size() / self.experience_buffer.max_size,
            'recent_accuracy': self.performance_tracker.get_recent_performance('val_accuracy', 10),
            'recent_loss': self.performance_tracker.get_recent_performance('loss', 10)
        }
    
    def _get_default_prediction(self) -> Dict[str, Any]:
        """Get default prediction when model fails."""
        return {
            'prediction': 0.5,
            'confidence': 0.1,
            'uncertainty': 0.9,
            'confidence_components': {},
            'signal': 'HOLD',
            'signal_strength': 0.0,
            'signal_reasoning': 'Model unavailable - default neutral prediction',
            'model_performance': self._get_performance_summary(),
            'timestamp': datetime.now(),
            'session_id': self.training_session_id
        }
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics."""
        # Basic stats
        basic_stats = {
            'session_id': self.training_session_id,
            'samples_processed': self.samples_processed,
            'updates_performed': self.updates_performed,
            'buffer_size': self.experience_buffer.size(),
            'current_learning_rate': self.current_learning_rate,
            'last_full_retrain': self.last_full_retrain,
            'model_id': self.model_manager.current_metadata.model_id if self.model_manager.current_metadata else None
        }
        
        # Enhanced performance metrics
        performance_stats = self.performance_tracker.export_metrics()
        
        # Experience buffer statistics
        buffer_stats = self.experience_buffer.get_statistics()
        
        # Learning rate scheduler info
        lr_stats = {
            'lr_schedule_type': self.config.online_config.lr_schedule.value,
            'step_count': self.lr_scheduler.step_count,
            'patience_count': getattr(self.lr_scheduler, 'patience_count', 0),
            'best_metric': getattr(self.lr_scheduler, 'best_metric', None)
        }
        
        # Confidence estimation stats
        confidence_stats = {
            'prediction_history_size': len(self.confidence_estimator.prediction_history),
            'error_history_size': len(self.confidence_estimator.error_history),
            'recent_error_mean': np.mean(list(self.confidence_estimator.error_history)[-10:]) if len(self.confidence_estimator.error_history) > 0 else 0
        }
        
        # Adaptive learning indicators
        should_adjust, reason = self.performance_tracker.should_adjust_learning()
        adaptive_stats = {
            'should_adjust_learning': should_adjust,
            'adjustment_reason': reason,
            'training_stability': self._calculate_training_stability()
        }
        
        return {
            'basic': basic_stats,
            'performance': performance_stats,
            'buffer': buffer_stats,
            'learning_rate': lr_stats,
            'confidence': confidence_stats,
            'adaptive': adaptive_stats,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_training_stability(self) -> float:
        """Calculate training stability score."""
        if len(self.performance_history) < 5:
            return 0.5
        
        recent_losses = [p.get('val_loss', 1.0) for p in list(self.performance_history)[-10:]]
        stability = 1.0 / (1.0 + np.std(recent_losses))
        return min(stability, 1.0)
    
    def save_state(self, filepath: str) -> bool:
        """
        Save learning state to file.
        
        Args:
            filepath: Path to save state
        
        Returns:
            True if successful
        """
        try:
            state = {
                'samples_processed': self.samples_processed,
                'updates_performed': self.updates_performed,
                'last_full_retrain': self.last_full_retrain,
                'current_learning_rate': self.current_learning_rate,
                'performance_history': list(self.performance_history),
                'config': self.config,
                'experience_buffer': list(self.experience_buffer.buffer)
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            
            logger.info(f"Learning state saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving learning state: {e}")
            return False
    
    def load_state(self, filepath: str) -> bool:
        """
        Load learning state from file.
        
        Args:
            filepath: Path to load state from
        
        Returns:
            True if successful
        """
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.samples_processed = state.get('samples_processed', 0)
            self.updates_performed = state.get('updates_performed', 0)
            self.last_full_retrain = state.get('last_full_retrain', 0)
            self.current_learning_rate = state.get('current_learning_rate', self.model_config.learning_rate)
            
            # Restore performance history
            if 'performance_history' in state:
                self.performance_history.clear()
                self.performance_history.extend(state['performance_history'])
            
            # Restore experience buffer
            if 'experience_buffer' in state:
                self.experience_buffer.clear()
                for experience in state['experience_buffer']:
                    self.experience_buffer.add(experience)
            
            logger.info(f"Learning state loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading learning state: {e}")
            return False
    
    def stop(self):
        """Stop continuous learning."""
        self.stop_learning = True
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5.0)
        
        logger.info("Continuous learning stopped")
