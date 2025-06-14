#!/usr/bin/env python3
"""
Configuration management CLI for AutoTrader Bot.

This script provides command-line tools for managing configuration files,
validating settings, and setting up environments.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional

from .config import Config, Environment, load_config
from .settings import Settings, validate_environment, setup_logging


def create_default_configs():
    """Create default configuration files for all environments"""
    config_dir = Path(__file__).parent
    
    for env in Environment:
        config = Config()
        config.environment = env
        
        # Environment-specific adjustments
        if env == Environment.PRODUCTION:
            config.operations.log_level = "WARNING"
            config.operations.enable_detailed_logging = False
            config.trading.buy_confidence_threshold = 0.7
            config.trading.sell_confidence_threshold = 0.3
        elif env == Environment.DEVELOPMENT:
            config.operations.log_level = "DEBUG"
            config.operations.enable_detailed_logging = True
            config.trading.initial_balance = 1000.0
            config.trading.trade_amount = 0.005
        elif env == Environment.STAGING:
            config.operations.log_level = "INFO" 
            config.trading.initial_balance = 5000.0
            config.trading.trade_amount = 0.008
        
        config_path = config_dir / f"config_{env.value}.json"
        config.to_file(config_path)
        print(f"✓ Created {env.value} config: {config_path}")


def validate_config(config_path: Optional[str] = None):
    """Validate configuration file or current environment"""
    try:
        if config_path:
            config = Config.from_file(config_path)
            print(f"✓ Configuration file '{config_path}' is valid")
        else:
            settings = Settings()
            validate_environment()
            print(f"✓ Current environment configuration is valid")
            print(f"  Environment: {settings.environment.value}")
            print(f"  API credentials: {'✓' if settings.has_api_credentials else '✗'}")
            
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        return False
    
    return True


def show_config(config_path: Optional[str] = None, show_sensitive: bool = False):
    """Display configuration values"""
    try:
        if config_path:
            config = Config.from_file(config_path)
        else:
            settings = Settings()
            config = settings.config
        
        config_dict = config.to_dict()
        
        # Hide sensitive information unless requested
        if not show_sensitive:
            if 'api' in config_dict:
                config_dict['api']['api_key'] = '***HIDDEN***' if config_dict['api']['api_key'] else None
                config_dict['api']['api_secret'] = '***HIDDEN***' if config_dict['api']['api_secret'] else None
        
        print(json.dumps(config_dict, indent=2))
        
    except Exception as e:
        print(f"✗ Error displaying configuration: {e}")
        return False
    
    return True


def set_environment(env_name: str):
    """Set the environment for configuration"""
    try:
        env = Environment(env_name.lower())
        
        # Create .env file with environment setting
        env_file = Path.cwd() / '.env'
        
        # Read existing .env if it exists
        env_content = []
        if env_file.exists():
            with open(env_file, 'r') as f:
                env_content = f.readlines()
        
        # Update or add AUTOTRADER_ENV
        updated = False
        for i, line in enumerate(env_content):
            if line.startswith('AUTOTRADER_ENV='):
                env_content[i] = f'AUTOTRADER_ENV={env.value}\n'
                updated = True
                break
        
        if not updated:
            env_content.append(f'AUTOTRADER_ENV={env.value}\n')
        
        # Write back to .env
        with open(env_file, 'w') as f:
            f.writelines(env_content)
        
        print(f"✓ Environment set to '{env.value}' in {env_file}")
        
    except ValueError:
        print(f"✗ Invalid environment: {env_name}")
        print(f"  Valid environments: {', '.join([e.value for e in Environment])}")
        return False
    except Exception as e:
        print(f"✗ Error setting environment: {e}")
        return False
    
    return True


def check_api_credentials():
    """Check if API credentials are properly configured"""
    settings = Settings()
    api_key, api_secret = settings.get_api_credentials()
    
    print("API Credentials Status:")
    print(f"  Environment: {settings.environment.value}")
    print(f"  API Key: {'✓ Set' if api_key else '✗ Missing'}")
    print(f"  API Secret: {'✓ Set' if api_secret else '✗ Missing'}")
    print(f"  Valid credentials: {'✓ Yes' if settings.has_api_credentials else '✗ No'}")
    
    if not settings.has_api_credentials:
        print("\nTo set API credentials:")
        print("1. Copy .env.template to .env")
        print("2. Edit .env and set BTCMARKETS_API_KEY and BTCMARKETS_API_SECRET")
        print("3. Or set environment variables directly")
        return False
    
    return True


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="AutoTrader Configuration Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s create-defaults        Create default config files
  %(prog)s validate              Validate current configuration
  %(prog)s validate config.json  Validate specific config file
  %(prog)s show                  Show current configuration
  %(prog)s show --sensitive      Show config with API credentials
  %(prog)s set-env production    Set environment to production
  %(prog)s check-api             Check API credential status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create defaults command
    subparsers.add_parser(
        'create-defaults',
        help='Create default configuration files for all environments'
    )
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration')
    validate_parser.add_argument(
        'config_file', 
        nargs='?', 
        help='Configuration file to validate (default: current environment)'
    )
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Display configuration')
    show_parser.add_argument(
        'config_file',
        nargs='?',
        help='Configuration file to show (default: current environment)'
    )
    show_parser.add_argument(
        '--sensitive',
        action='store_true',
        help='Show sensitive information like API keys'
    )
    
    # Set environment command
    set_env_parser = subparsers.add_parser('set-env', help='Set environment')
    set_env_parser.add_argument(
        'environment',
        choices=[e.value for e in Environment],
        help='Environment to set'
    )
    
    # Check API credentials command
    subparsers.add_parser('check-api', help='Check API credential status')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute commands
    success = True
    
    try:
        if args.command == 'create-defaults':
            create_default_configs()
        
        elif args.command == 'validate':
            success = validate_config(args.config_file)
        
        elif args.command == 'show':
            success = show_config(args.config_file, args.sensitive)
        
        elif args.command == 'set-env':
            success = set_environment(args.environment)
        
        elif args.command == 'check-api':
            success = check_api_credentials()
    
    except KeyboardInterrupt:
        print("\n✗ Operation cancelled")
        return 1
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return 1
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
