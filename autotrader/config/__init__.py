"""
Configuration management for the Autotrader Bot
"""

from .default_config import get_default_config
from .config_templates import create_config_template

__all__ = [
    "get_default_config",
    "create_config_template"
]
