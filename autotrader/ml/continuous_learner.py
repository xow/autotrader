"""
Continuous Learning Engine for autotrader bot.

Implements online learning algorithms with incremental training,
experience replay, and adaptive learning capabilities.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
from collections import deque
import pickle
import threading
import time
from datetime import datetime, timedelta

from .model_manager import ModelManager
from .neural_network import ModelConfig
from .feature_engineer import FeatureEngineer, FeatureConfig

logger = logging.getLogger("autotrader.ml.continuous_learner")


@dataclass
class LearningConfig:
    """Configuration for continuous learning."""
    
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


class ExperienceBuffer:
    """Experience replay buffer for continuous learning."""
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize experience buffer.
        
        Args:
            max_size: Maximum buffer size
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self._lock = threading.Lock()
    
    def add(self, experience: Dict, priority: float = 1.0):
        """
        Add experience to buffer.
        
        Args:
            experience: Experience dictionary
            priority: Experience priority for sampling
        """
        with self._lock:
            self.buffer.append(experience)
            self.priorities.append(max(priority, 0.01))  # Minimum priority
    
    def sample(self, batch_size: int, use_priority: bool = True) -> List[Dict]:
        """
        Sample experiences from buffer.
        
        Args:
            batch_size: Number of experiences to sample
            use_priority: Whether to use priority sampling
        
        Returns:
            List of sampled experiences
        """
        with self._lock:
            if len(self.buffer) == 0:
                return []
            
            batch_size = min(batch_size, len(self.buffer))
            
            if use_priority and len(self.priorities) == len(self.buffer):
                # Prioritized sampling
                priorities = np.array(list(self.priorities))
                probabilities = priorities / priorities.sum()
                indices = np.random.choice(
                    len(self.buffer),
                    size=batch_size,
                    replace=False,
                    p=probabilities
                )
            else:
                # Uniform random sampling
                indices = np.random.choice(
                    len(self.buffer),
                    size=batch_size,
                    replace=False
                )
            
            return [self.buffer[i] for i in indices]
    
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


class ContinuousLearner:
    """
    Continuous learning engine with online training and experience replay.
    
    Manages incremental model updates, performance monitoring, and adaptive
    learning strategies for real-time cryptocurrency trading.
    """
    
    def __init__(
        self,
        model_manager: ModelManager,
        learning_config: Optional[LearningConfig] = None,
        model_config: Optional[ModelConfig] = None,
        feature_config: Optional[FeatureConfig] = None
    ):
        """
        Initialize continuous learner.
        
        Args:
            model_manager: Model management system
            learning_config: Learning configuration
            model_config: Neural network configuration
            feature_config: Feature engineering configuration
        """
        self.model_manager = model_manager
        self.config = learning_config or LearningConfig()
        self.model_config = model_config or ModelConfig()
        self.feature_config = feature_config or FeatureConfig()
        
        # Experience buffer
        self.experience_buffer = ExperienceBuffer(self.config.buffer_size)
        
        # Learning state
        self.samples_processed = 0
        self.updates_performed = 0
        self.last_full_retrain = 0
        
        # Performance tracking
        self.performance_history = deque(maxlen=self.config.performance_window)
        self.current_learning_rate = self.model_config.learning_rate
        
        # Threading for background updates
        self.update_thread = None
        self.stop_learning = False
        self._learning_lock = threading.Lock()
        
        # Initialize model if not exists
        if not self.model_manager.current_model:
            model_id = self.model_manager.create_model(
                self.model_config,
                self.feature_config
            )
            logger.info(f"Created initial model: {model_id}")
        
        logger.info("Continuous learner initialized")
    
    def add_experience(
        self,
        market_data: Dict,
        prediction: Optional[float] = None,
        actual_outcome: Optional[float] = None,
        reward: Optional[float] = None
    ):
        """
        Add new market experience to learning buffer.
        
        Args:
            market_data: Market data point
            prediction: Model prediction (if available)
            actual_outcome: Actual market outcome
            reward: Trading reward/loss
        """
        experience = {
            'timestamp': datetime.now(),
            'market_data': market_data,
            'prediction': prediction,
            'actual_outcome': actual_outcome,
            'reward': reward
        }
        
        # Calculate priority based on prediction error
        priority = 1.0
        if prediction is not None and actual_outcome is not None:
            error = abs(prediction - actual_outcome)
            priority = min(error * 10, 5.0)  # Scale and cap priority
        
        self.experience_buffer.add(experience, priority)
        self.samples_processed += 1
        
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
        """Perform incremental model update with experience replay."""
        # Get recent experiences
        recent_batch_size = int(self.config.batch_size * (1 - self.config.replay_ratio))
        recent_experiences = self.experience_buffer.get_recent(recent_batch_size)
        
        # Get replay experiences if configured
        replay_experiences = []
        if self.config.use_experience_replay:
            replay_batch_size = self.config.batch_size - len(recent_experiences)
            if replay_batch_size > 0:
                replay_experiences = self.experience_buffer.sample(replay_batch_size)
        
        # Combine experiences
        all_experiences = recent_experiences + replay_experiences
        
        if len(all_experiences) < self.config.batch_size:
            logger.debug("Insufficient experiences for incremental update")
            return
        
        # Prepare training data
        X, y = self._prepare_training_data(all_experiences)
        
        if len(X) == 0:
            return
        
        # Perform training
        try:
            history = self.model_manager.current_model.train(
                X, y,
                epochs=self.config.epochs_per_update,
                batch_size=self.config.batch_size,
                validation_split=self.config.validation_split,
                verbose=0
            )
            
            # Update performance tracking
            self._update_performance_tracking(history)
            
            # Adaptive learning rate
            if self.config.adaptive_learning_rate:
                self._adjust_learning_rate(history)
            
        except Exception as e:
            logger.error(f"Error in incremental training: {e}")
    
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
        Make prediction with confidence estimation.
        
        Args:
            market_data: Current market data
        
        Returns:
            Prediction dictionary with confidence
        """
        try:
            # Use model manager for prediction
            predictions = self.model_manager.predict([market_data])
            
            if len(predictions) > 0:
                prediction = float(predictions[0][0])
                
                # Calculate confidence based on recent performance
                confidence = self._calculate_confidence(prediction)
                
                return {
                    'prediction': prediction,
                    'confidence': confidence,
                    'signal': self._get_trading_signal(prediction, confidence),
                    'timestamp': datetime.now()
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
    
    def _get_default_prediction(self) -> Dict[str, Any]:
        """Get default prediction when model fails."""
        return {
            'prediction': 0.5,
            'confidence': 0.5,
            'signal': 'HOLD',
            'timestamp': datetime.now()
        }
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        recent_performance = {}
        if len(self.performance_history) > 0:
            recent_perf = list(self.performance_history)[-10:]
            recent_performance = {
                'avg_accuracy': np.mean([p['accuracy'] for p in recent_perf]),
                'avg_val_accuracy': np.mean([p['val_accuracy'] for p in recent_perf]),
                'avg_loss': np.mean([p['loss'] for p in recent_perf]),
                'avg_val_loss': np.mean([p['val_loss'] for p in recent_perf])
            }
        
        return {
            'samples_processed': self.samples_processed,
            'updates_performed': self.updates_performed,
            'buffer_size': self.experience_buffer.size(),
            'current_learning_rate': self.current_learning_rate,
            'last_full_retrain': self.last_full_retrain,
            'recent_performance': recent_performance,
            'model_id': self.model_manager.current_metadata.model_id if self.model_manager.current_metadata else None
        }
    
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
