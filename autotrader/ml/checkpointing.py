"""
Advanced Model Checkpointing System for autotrader bot.

Implements comprehensive checkpointing with versioning, metadata storage,
integrity checking, rollback functionality, and model comparison tools.
"""

import os
import json
import shutil
import hashlib
import threading
import tempfile
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import logging
import pickle
import numpy as np
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor, as_completed

from .model_manager import ModelManager, ModelMetadata
from .neural_network import ModelConfig
from .feature_engineer import FeatureConfig

logger = logging.getLogger("autotrader.ml.checkpointing")


class CheckpointStatus(Enum):
    """Checkpoint status enumeration."""
    CREATING = "creating"
    COMPLETE = "complete"
    CORRUPTED = "corrupted"
    VALIDATING = "validating"
    ROLLING_BACK = "rolling_back"


class CheckpointType(Enum):
    """Types of checkpoints."""
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    MILESTONE = "milestone"
    BACKUP = "backup"
    ROLLBACK = "rollback"


@dataclass
class CheckpointMetadata:
    """Enhanced metadata for model checkpoints."""
    
    # Core identification
    checkpoint_id: str
    model_id: str
    version: str
    checkpoint_type: CheckpointType
    
    # Timestamps
    created_at: datetime
    completed_at: Optional[datetime] = None
    validated_at: Optional[datetime] = None
    
    # Status and health
    status: CheckpointStatus = CheckpointStatus.CREATING
    integrity_hash: Optional[str] = None
    validation_passed: bool = False
    
    # Performance metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    training_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Model architecture info
    model_config: Optional[Dict] = None
    feature_config: Optional[Dict] = None
    model_summary: Optional[str] = None
    
    # File paths and sizes
    model_path: str = ""
    feature_engineer_path: str = ""
    metadata_path: str = ""
    config_path: str = ""
    
    # File integrity
    file_sizes: Dict[str, int] = field(default_factory=dict)
    file_hashes: Dict[str, str] = field(default_factory=dict)
    
    # Training context
    training_samples: int = 0
    validation_samples: int = 0
    epochs_trained: int = 0
    training_time: float = 0.0
    
    # Comparative metrics
    improvement_over_previous: Optional[float] = None
    rank_among_models: Optional[int] = None
    
    # Additional metadata
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    parent_checkpoint: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['created_at'] = self.created_at.isoformat()
        result['completed_at'] = self.completed_at.isoformat() if self.completed_at else None
        result['validated_at'] = self.validated_at.isoformat() if self.validated_at else None
        result['checkpoint_type'] = self.checkpoint_type.value
        result['status'] = self.status.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CheckpointMetadata':
        """Create from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('completed_at'):
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        if data.get('validated_at'):
            data['validated_at'] = datetime.fromisoformat(data['validated_at'])
        data['checkpoint_type'] = CheckpointType(data['checkpoint_type'])
        data['status'] = CheckpointStatus(data['status'])
        return cls(**data)


@dataclass
class CheckpointConfig:
    """Configuration for checkpointing system."""
    
    # Storage settings
    checkpoint_dir: str = "checkpoints"
    max_checkpoints: int = 20
    auto_cleanup: bool = True
    compression_enabled: bool = True
    
    # Validation settings
    enable_integrity_checking: bool = True
    validate_on_creation: bool = True
    validation_timeout: int = 300  # seconds
    
    # Backup settings
    create_backup_copies: bool = True
    backup_dir: str = "backups"
    remote_backup: bool = False
    remote_backup_path: Optional[str] = None
    
    # Automatic checkpointing
    auto_checkpoint_interval: int = 100  # training steps
    milestone_intervals: List[int] = field(default_factory=lambda: [100, 500, 1000, 5000])
    performance_based_checkpoints: bool = True
    
    # Rollback settings
    enable_rollback: bool = True
    max_rollback_history: int = 5
    rollback_validation: bool = True
    
    # Parallel processing
    max_workers: int = 3
    parallel_validation: bool = True
    
    # Retention policy
    retention_days: int = 30
    keep_milestone_checkpoints: bool = True
    keep_best_checkpoints: int = 5


class CheckpointManager:
    """
    Advanced checkpoint management system with versioning, validation,
    and rollback capabilities.
    """
    
    def __init__(self, config: Optional[CheckpointConfig] = None):
        """Initialize checkpoint manager."""
        self.config = config or CheckpointConfig()
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.backup_dir = Path(self.config.backup_dir)
        
        # Create directories
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)
        
        # Registry for all checkpoints
        self.checkpoint_registry: Dict[str, CheckpointMetadata] = {}
        self.version_history: Dict[str, List[str]] = {}  # model_id -> checkpoint_ids
        
        # Threading
        self._lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # Load existing registry
        self._load_registry()
        
        # Validate existing checkpoints
        if self.config.enable_integrity_checking:
            self._validate_existing_checkpoints()
        
        logger.info(f"Checkpoint manager initialized with {len(self.checkpoint_registry)} checkpoints")
    
    def create_checkpoint(
        self,
        model_manager: ModelManager,
        checkpoint_type: CheckpointType = CheckpointType.AUTOMATIC,
        tags: Optional[List[str]] = None,
        notes: str = "",
        force: bool = False
    ) -> Optional[str]:
        """
        Create a new checkpoint.
        
        Args:
            model_manager: Model manager instance
            checkpoint_type: Type of checkpoint
            tags: Optional tags for the checkpoint
            notes: Optional notes
            force: Force creation even if model hasn't changed
        
        Returns:
            Checkpoint ID if successful
        """
        if not model_manager.current_model or not model_manager.current_metadata:
            logger.error("No model to checkpoint")
            return None
        
        # Generate checkpoint ID and version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = model_manager.current_metadata.model_id
        version = self._generate_version(model_id)
        checkpoint_id = f"{model_id}_v{version}_{timestamp}"
        
        # Create checkpoint metadata
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            model_id=model_id,
            version=version,
            checkpoint_type=checkpoint_type,
            created_at=datetime.now(),
            tags=tags or [],
            notes=notes
        )
        
        # Set file paths
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        checkpoint_path.mkdir(exist_ok=True)
        
        metadata.model_path = str(checkpoint_path / "model.keras")
        metadata.feature_engineer_path = str(checkpoint_path / "feature_engineer.pkl")
        metadata.metadata_path = str(checkpoint_path / "metadata.json")
        metadata.config_path = str(checkpoint_path / "config.json")
        
        try:
            # Extract performance metrics
            metadata.performance_metrics = model_manager.get_model_performance(model_id)
            
            # Store configurations
            if hasattr(model_manager.current_model, 'config'):
                metadata.model_config = asdict(model_manager.current_model.config)
            if hasattr(model_manager.current_feature_engineer, 'config'):
                metadata.feature_config = asdict(model_manager.current_feature_engineer.config)
            
            # Store model summary
            if model_manager.current_model.model:
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                    model_manager.current_model.model.summary(print_fn=lambda x: f.write(x + '\n'))
                    f.flush()
                    with open(f.name, 'r') as summary_file:
                        metadata.model_summary = summary_file.read()
                    os.unlink(f.name)
            
            # Save model files
            metadata.status = CheckpointStatus.CREATING
            self._save_checkpoint_files(model_manager, metadata)
            
            # Calculate file hashes and sizes
            self._compute_file_integrity(metadata)
            
            # Create integrity hash
            metadata.integrity_hash = self._create_integrity_hash(metadata)
            
            # Mark as complete
            metadata.completed_at = datetime.now()
            metadata.status = CheckpointStatus.COMPLETE
            
            # Validate if enabled
            if self.config.validate_on_creation:
                metadata.status = CheckpointStatus.VALIDATING
                validation_result = self._validate_checkpoint(metadata)
                metadata.validation_passed = validation_result
                metadata.validated_at = datetime.now()
                metadata.status = CheckpointStatus.COMPLETE if validation_result else CheckpointStatus.CORRUPTED
            
            # Save metadata
            self._save_metadata(metadata)
            
            # Register checkpoint
            with self._lock:
                self.checkpoint_registry[checkpoint_id] = metadata
                if model_id not in self.version_history:
                    self.version_history[model_id] = []
                self.version_history[model_id].append(checkpoint_id)
                
                # Update comparative metrics
                self._update_comparative_metrics(metadata)
            
            # Save registry
            self._save_registry()
            
            # Create backup if enabled
            if self.config.create_backup_copies:
                self._create_backup(metadata)
            
            # Auto cleanup
            if self.config.auto_cleanup:
                self._cleanup_old_checkpoints()
            
            logger.info(f"Checkpoint created: {checkpoint_id}")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Error creating checkpoint: {e}")
            # Cleanup failed checkpoint
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path, ignore_errors=True)
            return None
    
    def load_checkpoint(
        self,
        checkpoint_id: str,
        model_manager: ModelManager,
        validate: bool = True
    ) -> bool:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID to load
            model_manager: Model manager instance
            validate: Whether to validate before loading
        
        Returns:
            True if successful
        """
        with self._lock:
            if checkpoint_id not in self.checkpoint_registry:
                logger.error(f"Checkpoint not found: {checkpoint_id}")
                return False
            
            metadata = self.checkpoint_registry[checkpoint_id]
        
        try:
            # Validate checkpoint if requested
            if validate and not self._validate_checkpoint(metadata):
                logger.error(f"Checkpoint validation failed: {checkpoint_id}")
                return False
            
            # Load configurations
            configs = self._load_checkpoint_configs(metadata)
            if not configs:
                return False
            
            model_config, feature_config = configs
            
            # Create and load model
            model_manager.current_model = model_manager._create_architecture(model_config)
            if not model_manager.current_model.load_model(metadata.model_path):
                return False
            
            # Load feature engineer
            with open(metadata.feature_engineer_path, 'rb') as f:
                model_manager.current_feature_engineer = pickle.load(f)
            
            # Update model manager metadata
            model_manager.current_metadata = model_manager._create_metadata_from_checkpoint(metadata)
            
            logger.info(f"Checkpoint loaded: {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return False
    
    def rollback_to_checkpoint(
        self,
        checkpoint_id: str,
        model_manager: ModelManager,
        create_rollback_checkpoint: bool = True
    ) -> bool:
        """
        Rollback to a previous checkpoint.
        
        Args:
            checkpoint_id: Target checkpoint ID
            model_manager: Model manager instance  
            create_rollback_checkpoint: Whether to create a rollback point
        
        Returns:
            True if successful
        """
        if not self.config.enable_rollback:
            logger.error("Rollback is disabled")
            return False
        
        # Create rollback checkpoint of current state
        if create_rollback_checkpoint and model_manager.current_model:
            rollback_id = self.create_checkpoint(
                model_manager,
                CheckpointType.ROLLBACK,
                tags=["rollback_point"],
                notes=f"Rollback point before reverting to {checkpoint_id}"
            )
            if rollback_id:
                logger.info(f"Created rollback checkpoint: {rollback_id}")
        
        # Load target checkpoint
        success = self.load_checkpoint(checkpoint_id, model_manager, validate=self.config.rollback_validation)
        
        if success:
            logger.info(f"Successfully rolled back to checkpoint: {checkpoint_id}")
        else:
            logger.error(f"Rollback failed for checkpoint: {checkpoint_id}")
        
        return success
    
    def compare_checkpoints(
        self,
        checkpoint_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple checkpoints.
        
        Args:
            checkpoint_ids: List of checkpoint IDs to compare
            metrics: Specific metrics to compare
        
        Returns:
            Comparison results
        """
        if not checkpoint_ids:
            return {}
        
        comparison = {
            'checkpoints': {},
            'metrics_comparison': {},
            'ranking': {},
            'recommendations': []
        }
        
        # Get checkpoint data
        valid_checkpoints = []
        for cp_id in checkpoint_ids:
            if cp_id in self.checkpoint_registry:
                metadata = self.checkpoint_registry[cp_id]
                comparison['checkpoints'][cp_id] = {
                    'version': metadata.version,
                    'created_at': metadata.created_at.isoformat(),
                    'performance_metrics': metadata.performance_metrics,
                    'training_metrics': metadata.training_metrics,
                    'status': metadata.status.value,
                    'validation_passed': metadata.validation_passed
                }
                valid_checkpoints.append((cp_id, metadata))
        
        if not valid_checkpoints:
            return comparison
        
        # Compare metrics
        default_metrics = ['accuracy', 'val_accuracy', 'loss', 'val_loss']
        compare_metrics = metrics or default_metrics
        
        for metric in compare_metrics:
            metric_values = {}
            for cp_id, metadata in valid_checkpoints:
                value = metadata.performance_metrics.get(metric)
                if value is not None:
                    metric_values[cp_id] = value
            
            if metric_values:
                comparison['metrics_comparison'][metric] = metric_values
                
                # Rank by metric (higher is better for accuracy, lower for loss)
                ascending = 'loss' in metric.lower()
                sorted_checkpoints = sorted(
                    metric_values.items(),
                    key=lambda x: x[1],
                    reverse=not ascending
                )
                
                comparison['ranking'][metric] = [
                    {'checkpoint_id': cp_id, 'value': value, 'rank': i + 1}
                    for i, (cp_id, value) in enumerate(sorted_checkpoints)
                ]
        
        # Generate recommendations
        comparison['recommendations'] = self._generate_checkpoint_recommendations(valid_checkpoints)
        
        return comparison
    
    def get_best_checkpoint(
        self,
        model_id: Optional[str] = None,
        metric: str = 'val_accuracy',
        validated_only: bool = True
    ) -> Optional[str]:
        """
        Get the best checkpoint based on a metric.
        
        Args:
            model_id: Specific model ID (all models if None)
            metric: Metric to optimize
            validated_only: Only consider validated checkpoints
        
        Returns:
            Best checkpoint ID
        """
        candidates = []
        
        with self._lock:
            for cp_id, metadata in self.checkpoint_registry.items():
                # Filter by model ID
                if model_id and metadata.model_id != model_id:
                    continue
                
                # Filter by validation status
                if validated_only and not metadata.validation_passed:
                    continue
                
                # Check if metric exists
                if metric not in metadata.performance_metrics:
                    continue
                
                candidates.append((cp_id, metadata.performance_metrics[metric]))
        
        if not candidates:
            return None
        
        # Find best (higher is better for accuracy, lower for loss)
        ascending = 'loss' in metric.lower()
        best_checkpoint = min(candidates, key=lambda x: x[1]) if ascending else max(candidates, key=lambda x: x[1])
        
        return best_checkpoint[0]
    
    def cleanup_corrupted_checkpoints(self) -> int:
        """Clean up corrupted checkpoints."""
        corrupted_count = 0
        
        with self._lock:
            corrupted_checkpoints = [
                cp_id for cp_id, metadata in self.checkpoint_registry.items()
                if metadata.status == CheckpointStatus.CORRUPTED
            ]
        
        for cp_id in corrupted_checkpoints:
            if self.delete_checkpoint(cp_id):
                corrupted_count += 1
        
        logger.info(f"Cleaned up {corrupted_count} corrupted checkpoints")
        return corrupted_count
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint and its files.
        
        Args:
            checkpoint_id: Checkpoint ID to delete
        
        Returns:
            True if successful
        """
        with self._lock:
            if checkpoint_id not in self.checkpoint_registry:
                logger.error(f"Checkpoint not found: {checkpoint_id}")
                return False
            
            metadata = self.checkpoint_registry[checkpoint_id]
        
        try:
            # Delete checkpoint directory
            checkpoint_path = Path(metadata.model_path).parent
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path, ignore_errors=True)
            
            # Delete backup if exists
            backup_path = self.backup_dir / checkpoint_id
            if backup_path.exists():
                shutil.rmtree(backup_path, ignore_errors=True)
            
            # Remove from registry
            with self._lock:
                del self.checkpoint_registry[checkpoint_id]
                
                # Update version history
                model_id = metadata.model_id
                if model_id in self.version_history:
                    if checkpoint_id in self.version_history[model_id]:
                        self.version_history[model_id].remove(checkpoint_id)
                    if not self.version_history[model_id]:
                        del self.version_history[model_id]
            
            # Save updated registry
            self._save_registry()
            
            logger.info(f"Checkpoint deleted: {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting checkpoint: {e}")
            return False
    
    def list_checkpoints(
        self,
        model_id: Optional[str] = None,
        checkpoint_type: Optional[CheckpointType] = None,
        validated_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        List checkpoints with optional filtering.
        
        Args:
            model_id: Filter by model ID
            checkpoint_type: Filter by checkpoint type
            validated_only: Only return validated checkpoints
        
        Returns:
            List of checkpoint information
        """
        checkpoints = []
        
        with self._lock:
            for cp_id, metadata in self.checkpoint_registry.items():
                # Apply filters
                if model_id and metadata.model_id != model_id:
                    continue
                if checkpoint_type and metadata.checkpoint_type != checkpoint_type:
                    continue
                if validated_only and not metadata.validation_passed:
                    continue
                
                checkpoint_info = {
                    'checkpoint_id': cp_id,
                    'model_id': metadata.model_id,
                    'version': metadata.version,
                    'type': metadata.checkpoint_type.value,
                    'created_at': metadata.created_at.isoformat(),
                    'status': metadata.status.value,
                    'validated': metadata.validation_passed,
                    'performance_metrics': metadata.performance_metrics,
                    'tags': metadata.tags,
                    'notes': metadata.notes,
                    'file_exists': os.path.exists(metadata.model_path)
                }
                checkpoints.append(checkpoint_info)
        
        # Sort by creation date (newest first)
        checkpoints.sort(key=lambda x: x['created_at'], reverse=True)
        
        return checkpoints
    
    def get_checkpoint_statistics(self) -> Dict[str, Any]:
        """Get comprehensive checkpoint statistics."""
        with self._lock:
            total_checkpoints = len(self.checkpoint_registry)
            if total_checkpoints == 0:
                return {}
            
            # Status distribution
            status_counts = {}
            type_counts = {}
            model_counts = {}
            
            total_size = 0
            validated_count = 0
            
            for metadata in self.checkpoint_registry.values():
                # Count by status
                status = metadata.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
                
                # Count by type
                cp_type = metadata.checkpoint_type.value
                type_counts[cp_type] = type_counts.get(cp_type, 0) + 1
                
                # Count by model
                model_id = metadata.model_id
                model_counts[model_id] = model_counts.get(model_id, 0) + 1
                
                # Size and validation
                total_size += sum(metadata.file_sizes.values())
                if metadata.validation_passed:
                    validated_count += 1
            
            return {
                'total_checkpoints': total_checkpoints,
                'validated_checkpoints': validated_count,
                'validation_rate': validated_count / total_checkpoints,
                'total_size_mb': total_size / (1024 * 1024),
                'average_size_mb': total_size / (1024 * 1024) / total_checkpoints,
                'status_distribution': status_counts,
                'type_distribution': type_counts,
                'model_distribution': model_counts,
                'unique_models': len(model_counts),
                'oldest_checkpoint': min(
                    (cp.created_at for cp in self.checkpoint_registry.values())
                ).isoformat(),
                'newest_checkpoint': max(
                    (cp.created_at for cp in self.checkpoint_registry.values())
                ).isoformat()
            }
    
    # Private methods for internal operations
    
    def _generate_version(self, model_id: str) -> str:
        """Generate version number for model."""
        with self._lock:
            if model_id not in self.version_history:
                return "1.0.0"
            
            versions = []
            for cp_id in self.version_history[model_id]:
                if cp_id in self.checkpoint_registry:
                    version = self.checkpoint_registry[cp_id].version
                    versions.append(version)
            
            if not versions:
                return "1.0.0"
            
            # Simple version increment (major.minor.patch)
            latest_version = max(versions, key=lambda v: [int(x) for x in v.split('.')])
            major, minor, patch = map(int, latest_version.split('.'))
            
            return f"{major}.{minor}.{patch + 1}"
    
    def _save_checkpoint_files(self, model_manager: ModelManager, metadata: CheckpointMetadata):
        """Save model files for checkpoint."""
        # Save model
        if not model_manager.current_model.save_model(metadata.model_path):
            raise Exception("Failed to save model")
        
        # Save feature engineer
        with open(metadata.feature_engineer_path, 'wb') as f:
            pickle.dump(model_manager.current_feature_engineer, f)
        
        # Save configurations
        configs = {
            'model_config': metadata.model_config,
            'feature_config': metadata.feature_config
        }
        
        with open(metadata.config_path, 'w') as f:
            json.dump(configs, f, indent=2)
    
    def _compute_file_integrity(self, metadata: CheckpointMetadata):
        """Compute file hashes and sizes."""
        files_to_check = [
            metadata.model_path,
            metadata.feature_engineer_path,
            metadata.config_path
        ]
        
        for file_path in files_to_check:
            if os.path.exists(file_path):
                # File size
                metadata.file_sizes[file_path] = os.path.getsize(file_path)
                
                # File hash
                hasher = hashlib.sha256()
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)
                metadata.file_hashes[file_path] = hasher.hexdigest()
    
    def _create_integrity_hash(self, metadata: CheckpointMetadata) -> str:
        """Create overall integrity hash for checkpoint."""
        hasher = hashlib.sha256()
        
        # Include all file hashes
        for file_path in sorted(metadata.file_hashes.keys()):
            hasher.update(metadata.file_hashes[file_path].encode())
        
        # Include metadata
        hasher.update(metadata.checkpoint_id.encode())
        hasher.update(metadata.version.encode())
        
        return hasher.hexdigest()
    
    def _validate_checkpoint(self, metadata: CheckpointMetadata) -> bool:
        """Validate checkpoint integrity."""
        try:
            # Check all files exist
            files_to_check = [
                metadata.model_path,
                metadata.feature_engineer_path,
                metadata.config_path
            ]
            
            for file_path in files_to_check:
                if not os.path.exists(file_path):
                    logger.error(f"Missing file: {file_path}")
                    return False
            
            # Verify file hashes
            for file_path, expected_hash in metadata.file_hashes.items():
                if os.path.exists(file_path):
                    hasher = hashlib.sha256()
                    with open(file_path, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hasher.update(chunk)
                    actual_hash = hasher.hexdigest()
                    
                    if actual_hash != expected_hash:
                        logger.error(f"Hash mismatch for {file_path}")
                        return False
            
            # Try loading the model
            try:
                tf.keras.models.load_model(metadata.model_path)
            except Exception as e:
                logger.error(f"Model loading failed: {e}")
                return False
            
            # Try loading feature engineer
            try:
                with open(metadata.feature_engineer_path, 'rb') as f:
                    pickle.load(f)
            except Exception as e:
                logger.error(f"Feature engineer loading failed: {e}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    def _validate_existing_checkpoints(self):
        """Validate all existing checkpoints in background."""
        def validate_checkpoints():
            corrupted = []
            for cp_id, metadata in list(self.checkpoint_registry.items()):
                if not self._validate_checkpoint(metadata):
                    metadata.status = CheckpointStatus.CORRUPTED
                    metadata.validation_passed = False
                    corrupted.append(cp_id)
                else:
                    metadata.validation_passed = True
            
            if corrupted:
                logger.warning(f"Found {len(corrupted)} corrupted checkpoints")
                self._save_registry()
        
        # Run validation in background
        self.executor.submit(validate_checkpoints)
    
    def _save_metadata(self, metadata: CheckpointMetadata):
        """Save checkpoint metadata."""
        with open(metadata.metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
    
    def _load_checkpoint_configs(self, metadata: CheckpointMetadata) -> Optional[Tuple[ModelConfig, FeatureConfig]]:
        """Load configurations from checkpoint."""
        try:
            with open(metadata.config_path, 'r') as f:
                configs = json.load(f)
            
            # Reconstruct configs (simplified - would need proper enum handling)
            model_config = ModelConfig(**configs['model_config'])
            feature_config = FeatureConfig(**configs['feature_config'])
            
            return model_config, feature_config
            
        except Exception as e:
            logger.error(f"Error loading checkpoint configs: {e}")
            return None
    
    def _create_backup(self, metadata: CheckpointMetadata):
        """Create backup copy of checkpoint."""
        backup_path = self.backup_dir / metadata.checkpoint_id
        checkpoint_path = Path(metadata.model_path).parent
        
        try:
            shutil.copytree(checkpoint_path, backup_path, dirs_exist_ok=True)
            logger.debug(f"Backup created: {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")
    
    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints based on retention policy."""
        with self._lock:
            # Get checkpoints sorted by age
            checkpoints_by_age = sorted(
                self.checkpoint_registry.items(),
                key=lambda x: x[1].created_at
            )
            
            # Keep milestone and best checkpoints
            protected_checkpoints = set()
            
            if self.config.keep_milestone_checkpoints:
                for cp_id, metadata in checkpoints_by_age:
                    if metadata.checkpoint_type == CheckpointType.MILESTONE:
                        protected_checkpoints.add(cp_id)
            
            # Keep best checkpoints by performance
            if self.config.keep_best_checkpoints > 0:
                best_checkpoints = sorted(
                    checkpoints_by_age,
                    key=lambda x: x[1].performance_metrics.get('val_accuracy', 0),
                    reverse=True
                )[:self.config.keep_best_checkpoints]
                
                for cp_id, _ in best_checkpoints:
                    protected_checkpoints.add(cp_id)
            
            # Delete old checkpoints
            cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
            deleted_count = 0
            
            for cp_id, metadata in checkpoints_by_age:
                if (len(self.checkpoint_registry) - deleted_count <= self.config.max_checkpoints):
                    break
                
                if (cp_id not in protected_checkpoints and 
                    metadata.created_at < cutoff_date):
                    
                    if self.delete_checkpoint(cp_id):
                        deleted_count += 1
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old checkpoints")
    
    def _update_comparative_metrics(self, metadata: CheckpointMetadata):
        """Update comparative metrics for checkpoint."""
        model_id = metadata.model_id
        
        # Get all checkpoints for this model
        model_checkpoints = [
            cp for cp in self.checkpoint_registry.values()
            if cp.model_id == model_id and cp.checkpoint_id != metadata.checkpoint_id
        ]
        
        if model_checkpoints:
            # Find improvement over previous best
            val_accuracies = [
                cp.performance_metrics.get('val_accuracy', 0)
                for cp in model_checkpoints
            ]
            
            if val_accuracies:
                best_previous = max(val_accuracies)
                current_accuracy = metadata.performance_metrics.get('val_accuracy', 0)
                metadata.improvement_over_previous = current_accuracy - best_previous
        
        # Calculate rank among all checkpoints
        all_accuracies = [
            (cp.checkpoint_id, cp.performance_metrics.get('val_accuracy', 0))
            for cp in self.checkpoint_registry.values()
        ]
        
        sorted_by_accuracy = sorted(all_accuracies, key=lambda x: x[1], reverse=True)
        for rank, (cp_id, _) in enumerate(sorted_by_accuracy, 1):
            if cp_id == metadata.checkpoint_id:
                metadata.rank_among_models = rank
                break
    
    def _generate_checkpoint_recommendations(self, checkpoints: List[Tuple[str, CheckpointMetadata]]) -> List[str]:
        """Generate recommendations based on checkpoint comparison."""
        recommendations = []
        
        if len(checkpoints) < 2:
            return recommendations
        
        # Find best performing checkpoint
        best_checkpoint = max(
            checkpoints,
            key=lambda x: x[1].performance_metrics.get('val_accuracy', 0)
        )
        
        recommendations.append(
            f"Best performing checkpoint: {best_checkpoint[0]} "
            f"(accuracy: {best_checkpoint[1].performance_metrics.get('val_accuracy', 0):.3f})"
        )
        
        # Find most recent checkpoint
        most_recent = max(checkpoints, key=lambda x: x[1].created_at)
        if most_recent[0] != best_checkpoint[0]:
            recommendations.append(
                f"Most recent checkpoint: {most_recent[0]} "
                f"(created: {most_recent[1].created_at.strftime('%Y-%m-%d %H:%M')})"
            )
        
        # Check for performance degradation
        sorted_by_time = sorted(checkpoints, key=lambda x: x[1].created_at)
        if len(sorted_by_time) >= 3:
            recent_accuracies = [
                cp[1].performance_metrics.get('val_accuracy', 0)
                for cp in sorted_by_time[-3:]
            ]
            
            if recent_accuracies[0] > recent_accuracies[-1]:
                recommendations.append(
                    "Warning: Recent performance decline detected. "
                    "Consider rolling back to an earlier checkpoint."
                )
        
        return recommendations
    
    def _save_registry(self):
        """Save checkpoint registry to disk."""
        registry_path = self.checkpoint_dir / "checkpoint_registry.json"
        
        registry_data = {
            'checkpoints': {
                cp_id: metadata.to_dict()
                for cp_id, metadata in self.checkpoint_registry.items()
            },
            'version_history': self.version_history,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(registry_path, 'w') as f:
            json.dump(registry_data, f, indent=2)
    
    def _load_registry(self):
        """Load checkpoint registry from disk."""
        registry_path = self.checkpoint_dir / "checkpoint_registry.json"
        
        if not registry_path.exists():
            return
        
        try:
            with open(registry_path, 'r') as f:
                registry_data = json.load(f)
            
            # Load checkpoints
            self.checkpoint_registry = {
                cp_id: CheckpointMetadata.from_dict(data)
                for cp_id, data in registry_data.get('checkpoints', {}).items()
            }
            
            # Load version history
            self.version_history = registry_data.get('version_history', {})
            
            logger.info(f"Loaded {len(self.checkpoint_registry)} checkpoints from registry")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint registry: {e}")
            self.checkpoint_registry = {}
            self.version_history = {}
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
