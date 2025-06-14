"""
State management for the Autotrader Bot

Handles persistence of training states, model checkpoints,
and session continuity for autonomous operation.
"""

import asyncio
import pickle
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .config import Config
from ..utils.logging_config import get_logger
from ..utils.exceptions import StateError


class StateManager:
    """
    Manages persistence and recovery of training states.
    
    Ensures complete state continuity across shutdowns and restarts
    for seamless continuous learning operation.
    """
    
    def __init__(self, config: Config):
        """
        Initialize state manager.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # State storage paths
        self.state_dir = Path("state")
        self.state_dir.mkdir(exist_ok=True)
        
        self.training_state_file = self.state_dir / "training_state.pkl"
        self.session_metadata_file = self.state_dir / "session_metadata.json"
        self.checkpoint_dir = self.state_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Current state
        self.current_state = {
            "session_id": None,
            "training_iteration": 0,
            "last_update": None,
            "model_version": 0,
            "performance_metrics": {},
            "trading_stats": {},
            "data_stats": {}
        }
        
        self.logger.info("State manager initialized")
    
    def has_saved_state(self) -> bool:
        """
        Check if saved state exists.
        
        Returns:
            True if saved state is available
        """
        return (
            self.training_state_file.exists() and 
            self.session_metadata_file.exists()
        )
    
    async def load_state(self) -> Dict[str, Any]:
        """
        Load previous training state.
        
        Returns:
            Loaded state dictionary
            
        Raises:
            StateError: If state loading fails
        """
        if not self.has_saved_state():
            raise StateError("No saved state available to load")
        
        try:
            self.logger.info("Loading training state...")
            
            # Load training state
            with open(self.training_state_file, 'rb') as f:
                self.current_state = pickle.load(f)
            
            # Load session metadata
            with open(self.session_metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Validate state integrity
            self._validate_state(self.current_state, metadata)
            
            self.logger.info(
                f"State loaded successfully - Session: {self.current_state['session_id']}, "
                f"Iteration: {self.current_state['training_iteration']}"
            )
            
            return self.current_state
            
        except Exception as e:
            raise StateError(f"Failed to load state: {e}")
    
    async def save_state(self, force: bool = False) -> None:
        """
        Save current training state.
        
        Args:
            force: Force save even if no changes detected
            
        Raises:
            StateError: If state saving fails
        """
        try:
            # Update timestamp
            self.current_state["last_update"] = datetime.now().isoformat()
            
            # Save training state
            temp_state_file = self.training_state_file.with_suffix('.tmp')
            with open(temp_state_file, 'wb') as f:
                pickle.dump(self.current_state, f)
            
            # Atomic rename
            temp_state_file.rename(self.training_state_file)
            
            # Save session metadata
            metadata = {
                "session_id": self.current_state["session_id"],
                "created": self.current_state.get("created"),
                "last_update": self.current_state["last_update"],
                "training_iteration": self.current_state["training_iteration"],
                "model_version": self.current_state["model_version"],
                "state_file_size": self.training_state_file.stat().st_size,
                "checkpoint_count": len(list(self.checkpoint_dir.glob("*.ckpt")))
            }
            
            temp_metadata_file = self.session_metadata_file.with_suffix('.tmp')
            with open(temp_metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Atomic rename
            temp_metadata_file.rename(self.session_metadata_file)
            
            self.logger.debug(
                f"State saved - Iteration: {self.current_state['training_iteration']}"
            )
            
        except Exception as e:
            raise StateError(f"Failed to save state: {e}")
    
    def _validate_state(self, state: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        """
        Validate loaded state integrity.
        
        Args:
            state: Loaded training state
            metadata: Loaded session metadata
            
        Raises:
            StateError: If validation fails
        """
        required_fields = [
            "session_id", "training_iteration", "last_update",
            "model_version", "performance_metrics"
        ]
        
        for field in required_fields:
            if field not in state:
                raise StateError(f"Missing required field in state: {field}")
        
        # Validate session ID consistency
        if state["session_id"] != metadata.get("session_id"):
            raise StateError("Session ID mismatch between state and metadata")
        
        # Validate timestamps
        try:
            datetime.fromisoformat(state["last_update"])
        except ValueError:
            raise StateError("Invalid timestamp format in state")
        
        self.logger.info("State validation passed")
    
    def create_new_session(self) -> str:
        """
        Create a new training session.
        
        Returns:
            New session ID
        """
        session_id = f"session_{int(time.time())}"
        
        self.current_state = {
            "session_id": session_id,
            "created": datetime.now().isoformat(),
            "training_iteration": 0,
            "last_update": datetime.now().isoformat(),
            "model_version": 0,
            "performance_metrics": {
                "accuracy": 0.0,
                "loss": float('inf'),
                "precision": 0.0,
                "recall": 0.0
            },
            "trading_stats": {
                "total_trades": 0,
                "winning_trades": 0,
                "total_profit": 0.0,
                "max_drawdown": 0.0
            },
            "data_stats": {
                "total_data_points": 0,
                "last_data_timestamp": None,
                "data_quality_score": 0.0
            }
        }
        
        self.logger.info(f"New training session created: {session_id}")
        return session_id
    
    def update_training_progress(
        self,
        iteration: int,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Update training progress in current state.
        
        Args:
            iteration: Current training iteration
            metrics: Performance metrics to update
        """
        self.current_state["training_iteration"] = iteration
        self.current_state["last_update"] = datetime.now().isoformat()
        
        if metrics:
            self.current_state["performance_metrics"].update(metrics)
        
        self.logger.debug(f"Training progress updated - Iteration: {iteration}")
    
    def update_trading_stats(self, stats: Dict[str, Union[int, float]]) -> None:
        """
        Update trading statistics in current state.
        
        Args:
            stats: Trading statistics to update
        """
        self.current_state["trading_stats"].update(stats)
        self.logger.debug("Trading statistics updated")
    
    def update_data_stats(self, stats: Dict[str, Union[int, float, str]]) -> None:
        """
        Update data statistics in current state.
        
        Args:
            stats: Data statistics to update
        """
        self.current_state["data_stats"].update(stats)
        self.logger.debug("Data statistics updated")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get summary of current session.
        
        Returns:
            Session summary dictionary
        """
        if not self.current_state["session_id"]:
            return {"error": "No active session"}
        
        return {
            "session_id": self.current_state["session_id"],
            "created": self.current_state.get("created"),
            "duration_hours": self._calculate_session_duration(),
            "training_iteration": self.current_state["training_iteration"],
            "model_version": self.current_state["model_version"],
            "latest_metrics": self.current_state["performance_metrics"],
            "trading_summary": self.current_state["trading_stats"],
            "data_summary": self.current_state["data_stats"]
        }
    
    def _calculate_session_duration(self) -> float:
        """Calculate session duration in hours."""
        if not self.current_state.get("created"):
            return 0.0
        
        try:
            created = datetime.fromisoformat(self.current_state["created"])
            now = datetime.now()
            duration = (now - created).total_seconds() / 3600
            return round(duration, 2)
        except (ValueError, TypeError):
            return 0.0
    
    def cleanup_old_checkpoints(self, keep_count: int = 10) -> None:
        """
        Clean up old checkpoint files to manage disk space.
        
        Args:
            keep_count: Number of most recent checkpoints to keep
        """
        try:
            checkpoint_files = sorted(
                self.checkpoint_dir.glob("*.ckpt"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            files_to_remove = checkpoint_files[keep_count:]
            
            for file_path in files_to_remove:
                file_path.unlink()
                self.logger.debug(f"Removed old checkpoint: {file_path.name}")
            
            if files_to_remove:
                self.logger.info(f"Cleaned up {len(files_to_remove)} old checkpoints")
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup checkpoints: {e}")
    
    def export_session_data(self, export_path: Optional[Path] = None) -> Path:
        """
        Export session data for analysis.
        
        Args:
            export_path: Path to save exported data
            
        Returns:
            Path to exported file
        """
        if export_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = Path(f"session_export_{timestamp}.json")
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "session_summary": self.get_session_summary(),
            "full_state": self.current_state
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Session data exported to: {export_path}")
        return export_path
