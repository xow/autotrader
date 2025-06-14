"""
Main engine for the Autotrader Bot

Orchestrates all components for continuous learning and autonomous operation.
"""

import asyncio
import signal
import sys
from typing import Optional
from pathlib import Path

from .config import Config
from .state_manager import StateManager
from ..utils.logging_config import setup_logging, get_logger
from ..utils.exceptions import AutotraderError


class AutotraderEngine:
    """
    Main engine that orchestrates all autotrader components.
    
    Manages the continuous learning loop, state persistence,
    and autonomous operation lifecycle.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the autotrader engine.
        
        Args:
            config: Configuration instance, creates default if None
        """
        self.config = config or Config()
        self.logger = None
        self.state_manager = None
        self.running = False
        self._shutdown_event = asyncio.Event()
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize core components
        self._initialize_components()
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        setup_logging(
            log_level=self.config.system.log_level,
            log_dir=self.config.system.log_dir,
            enable_console=True,
            enable_structlog=True
        )
        self.logger = get_logger(__name__)
        self.logger.info("Autotrader Engine initializing...")
    
    def _initialize_components(self):
        """Initialize all engine components."""
        try:
            # Initialize state manager
            self.state_manager = StateManager(self.config)
            
            # TODO: Initialize other components
            # - ML components
            # - Data stream processor
            # - Trading simulator
            # - API clients
            
            self.logger.info("Engine components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise AutotraderError(f"Engine initialization failed: {e}")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def start(self):
        """
        Start the autotrader engine.
        
        This method blocks until the engine is stopped.
        """
        if self.running:
            self.logger.warning("Engine is already running")
            return
        
        self.logger.info("Starting Autotrader Engine...")
        
        try:
            # Run async main loop
            asyncio.run(self._run_async())
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Engine error: {e}")
            raise
        finally:
            self.logger.info("Autotrader Engine stopped")
    
    async def _run_async(self):
        """Main async execution loop."""
        self.running = True
        
        try:
            # Load previous state if available
            await self._load_state()
            
            # Start main processing loop
            await self._main_loop()
            
        except Exception as e:
            self.logger.error(f"Async run error: {e}")
            raise
        finally:
            await self._cleanup()
            self.running = False
    
    async def _load_state(self):
        """Load previous training state if available."""
        try:
            if self.state_manager.has_saved_state():
                self.logger.info("Loading previous training state...")
                await self.state_manager.load_state()
                self.logger.info("Previous state loaded successfully")
            else:
                self.logger.info("No previous state found, starting fresh")
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            # Don't fail on state loading errors, just log and continue
    
    async def _main_loop(self):
        """Main processing loop for continuous learning."""
        self.logger.info("Starting main processing loop...")
        
        iteration = 0
        
        while self.running and not self._shutdown_event.is_set():
            try:
                # TODO: Implement main processing logic
                # 1. Receive new market data
                # 2. Process data and update ML model
                # 3. Generate trading predictions
                # 4. Execute trading decisions (simulation)
                # 5. Save state checkpoint
                
                iteration += 1
                self.logger.debug(f"Processing iteration {iteration}")
                
                # Placeholder: sleep for demo
                await asyncio.sleep(1.0)
                
                # Check for shutdown signal
                if self._shutdown_event.is_set():
                    break
                
            except Exception as e:
                self.logger.error(f"Error in main loop iteration {iteration}: {e}")
                # Continue processing unless it's a critical error
                await asyncio.sleep(5.0)  # Brief pause before retry
    
    async def _cleanup(self):
        """Cleanup resources and save final state."""
        self.logger.info("Performing cleanup...")
        
        try:
            # Save current state
            if self.state_manager:
                await self.state_manager.save_state()
                self.logger.info("Final state saved successfully")
            
            # TODO: Cleanup other resources
            # - Close database connections
            # - Close websocket connections
            # - Save final model checkpoints
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def stop(self):
        """Stop the autotrader engine gracefully."""
        if not self.running:
            self.logger.warning("Engine is not running")
            return
        
        self.logger.info("Stopping Autotrader Engine...")
        self.running = False
        self._shutdown_event.set()
    
    def get_status(self) -> dict:
        """
        Get current engine status.
        
        Returns:
            Dictionary with engine status information
        """
        return {
            "running": self.running,
            "config_loaded": self.config is not None,
            "state_manager_ready": self.state_manager is not None,
            "has_saved_state": self.state_manager.has_saved_state() if self.state_manager else False,
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.running:
            self.stop()
