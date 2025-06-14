"""
Command-line interface for autotrader bot.

Provides CLI commands for running the bot, managing configuration, and monitoring.
"""

import argparse
import sys
import signal
import os
from typing import Optional

from .core.config import Config
from .core.trader import ContinuousAutoTrader
from .utils.logging_config import setup_logging


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print(f"\nReceived signal {signum}, shutting down gracefully...")
    sys.exit(0)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Continuous Learning Cryptocurrency Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  autotrader run                    # Run with default settings
  autotrader run --balance 5000     # Run with custom initial balance
  autotrader run --config-file config.yaml  # Run with config file
  autotrader validate-config        # Validate configuration
  autotrader status                 # Check bot status
        """
    )
    
    # Global options
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set logging level'
    )
    
    parser.add_argument(
        '--log-file',
        default='autotrader.log',
        help='Log file path'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run the autotrader bot')
    run_parser.add_argument(
        '--balance',
        type=float,
        help='Initial balance in AUD'
    )
    run_parser.add_argument(
        '--config-file',
        help='Path to configuration file'
    )
    run_parser.add_argument(
        '--no-console',
        action='store_true',
        help='Disable console logging (file only)'
    )
    
    # Config validation command
    config_parser = subparsers.add_parser('validate-config', help='Validate configuration')
    config_parser.add_argument(
        '--config-file',
        help='Path to configuration file to validate'
    )
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check bot status')
    status_parser.add_argument(
        '--state-file',
        default='trader_state.pkl',
        help='Path to state file'
    )
    
    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean old files and logs')
    clean_parser.add_argument(
        '--max-age-hours',
        type=int,
        default=168,  # 1 week
        help='Maximum age of files to keep (hours)'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(
        log_filename=args.log_file,
        log_level=args.log_level,
        console_output=not getattr(args, 'no_console', False)
    )
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Execute command
    if args.command == 'run' or args.command is None:
        run_command(args, logger)
    elif args.command == 'validate-config':
        validate_config_command(args, logger)
    elif args.command == 'status':
        status_command(args, logger)
    elif args.command == 'clean':
        clean_command(args, logger)
    else:
        parser.print_help()
        sys.exit(1)


def run_command(args, logger):
    """Execute the run command."""
    try:
        logger.info("Starting Autotrader Bot...")
        
        # Load configuration
        config = Config.from_env()
        
        # Override with command line arguments
        if args.balance:
            config.initial_balance = args.balance
            logger.info(f"Using custom initial balance: ${args.balance:.2f}")
        
        # Validate configuration
        try:
            config.validate()
            logger.info("Configuration validated successfully")
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            sys.exit(1)
        
        # Create and run trader
        trader = ContinuousAutoTrader(config=config)
        logger.info("AutoTrader initialized, starting continuous operation...")
        
        # Run the trading loop
        trader.run_continuous_trading()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


def validate_config_command(args, logger):
    """Execute the validate-config command."""
    try:
        config = Config.from_env()
        config.validate()
        
        logger.info("Configuration validation successful")
        print("✓ Configuration is valid")
        
        # Print configuration summary
        print("\nConfiguration Summary:")
        config_dict = config.to_dict()
        for key, value in config_dict.items():
            print(f"  {key}: {value}")
        
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        print(f"✗ Configuration validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error validating configuration: {e}")
        print(f"✗ Error validating configuration: {e}")
        sys.exit(1)


def status_command(args, logger):
    """Execute the status command."""
    try:
        from .utils.helpers import safe_pickle_load
        
        # Try to load state file
        state = safe_pickle_load(args.state_file)
        
        if state is None:
            print("✗ No state file found or unable to read state")
            print(f"  Looking for: {args.state_file}")
            sys.exit(1)
        
        print("✓ Bot state file found")
        print("\nCurrent Status:")
        print(f"  Balance: ${state.get('balance', 0):.2f} AUD")
        print(f"  Training Data Points: {state.get('training_data_length', 0)}")
        print(f"  Scalers Fitted: {state.get('scalers_fitted', False)}")
        
        if 'last_save_time' in state:
            import datetime
            last_save = datetime.datetime.fromtimestamp(state['last_save_time'])
            print(f"  Last Save: {last_save.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if 'last_training_time' in state:
            import datetime
            if state['last_training_time'] > 0:
                last_training = datetime.datetime.fromtimestamp(state['last_training_time'])
                print(f"  Last Training: {last_training.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print("  Last Training: Never")
        
        # Check for other files
        config = Config.from_env()
        files_to_check = [
            ('Model File', config.model_filename),
            ('Training Data', config.training_data_filename),
            ('Scalers File', config.scalers_filename)
        ]
        
        print("\nFile Status:")
        for name, filename in files_to_check:
            if os.path.exists(filename):
                size = os.path.getsize(filename)
                print(f"  ✓ {name}: {filename} ({size:,} bytes)")
            else:
                print(f"  ✗ {name}: {filename} (not found)")
        
    except Exception as e:
        logger.error(f"Error checking status: {e}")
        print(f"✗ Error checking status: {e}")
        sys.exit(1)


def clean_command(args, logger):
    """Execute the clean command."""
    try:
        from .utils.helpers import cleanup_old_files
        
        # Clean log files
        log_dir = os.path.dirname(args.log_file) or "."
        deleted_logs = cleanup_old_files(log_dir, args.max_age_hours)
        
        print(f"Cleaned {deleted_logs} old log files")
        logger.info(f"Cleanup completed: {deleted_logs} files removed")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        print(f"✗ Error during cleanup: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
