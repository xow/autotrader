"""
Helper utilities for autotrader bot.

Common utility functions used across the application.
"""

import json
import pickle
import numpy as np
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import os


def validate_data(data: Any, expected_type: type = None, required_keys: List[str] = None) -> bool:
    """
    Validate data structure and content.
    
    Args:
        data: Data to validate
        expected_type: Expected data type
        required_keys: Required keys for dictionary data
    
    Returns:
        True if data is valid, False otherwise
    """
    try:
        # Type validation
        if expected_type is not None and not isinstance(data, expected_type):
            return False
        
        # Dictionary key validation
        if required_keys and isinstance(data, dict):
            for key in required_keys:
                if key not in data:
                    return False
        
        # Additional validations for specific types
        if isinstance(data, (int, float)):
            if np.isnan(data) or np.isinf(data):
                return False
        
        if isinstance(data, dict):
            # Check for valid numeric values in price data
            for key, value in data.items():
                if 'price' in key.lower() and isinstance(value, (int, float)):
                    if value <= 0 or np.isnan(value) or np.isinf(value):
                        return False
        
        return True
        
    except Exception:
        return False


def format_currency(amount: float, currency: str = "AUD", decimal_places: int = 2) -> str:
    """
    Format currency amounts for display.
    
    Args:
        amount: Amount to format
        currency: Currency code
        decimal_places: Number of decimal places
    
    Returns:
        Formatted currency string
    """
    try:
        if currency.upper() == "BTC":
            return f"{amount:.8f} BTC"
        else:
            return f"${amount:,.{decimal_places}f} {currency}"
    except (ValueError, TypeError):
        return f"${0:.{decimal_places}f} {currency}"


def safe_json_dump(data: Any, filename: str, indent: int = 2) -> bool:
    """
    Safely dump data to JSON file with error handling.
    
    Args:
        data: Data to dump
        filename: Output filename
        indent: JSON indentation
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True) if os.path.dirname(filename) else None
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, default=str)
        return True
    except Exception:
        return False


def safe_json_load(filename: str) -> Optional[Any]:
    """
    Safely load data from JSON file with error handling.
    
    Args:
        filename: Input filename
    
    Returns:
        Loaded data or None if failed
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, Exception):
        return None


def safe_pickle_dump(data: Any, filename: str) -> bool:
    """
    Safely dump data to pickle file with error handling.
    
    Args:
        data: Data to dump
        filename: Output filename
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True) if os.path.dirname(filename) else None
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception:
        return False


def safe_pickle_load(filename: str) -> Optional[Any]:
    """
    Safely load data from pickle file with error handling.
    
    Args:
        filename: Input filename
    
    Returns:
        Loaded data or None if failed
    """
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, pickle.PickleError, Exception):
        return None


def timestamp_to_string(timestamp: Union[datetime, float, int] = None) -> str:
    """
    Convert timestamp to ISO format string.
    
    Args:
        timestamp: Timestamp to convert (defaults to now)
    
    Returns:
        ISO format timestamp string
    """
    if timestamp is None:
        timestamp = datetime.now()
    elif isinstance(timestamp, (int, float)):
        timestamp = datetime.fromtimestamp(timestamp)
    
    return timestamp.isoformat()


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Original value
        new_value: New value
    
    Returns:
        Percentage change
    """
    try:
        if old_value == 0:
            return 0.0
        return ((new_value - old_value) / old_value) * 100
    except (ValueError, TypeError, ZeroDivisionError):
        return 0.0


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp a value between minimum and maximum bounds.
    
    Args:
        value: Value to clamp
        min_value: Minimum allowed value
        max_value: Maximum allowed value
    
    Returns:
        Clamped value
    """
    return max(min_value, min(value, max_value))


def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """
    Calculate exponential backoff delay.
    
    Args:
        attempt: Attempt number (0-based)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
    
    Returns:
        Delay in seconds
    """
    delay = base_delay * (2 ** attempt)
    return min(delay, max_delay)


def flatten_dict(d: dict, parent_key: str = '', sep: str = '_') -> dict:
    """
    Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for keys
    
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def ensure_directory_exists(file_path: str) -> bool:
    """
    Ensure directory exists for a given file path.
    
    Args:
        file_path: Path to file
    
    Returns:
        True if directory exists or was created successfully
    """
    try:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        return True
    except Exception:
        return False


def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to file
    
    Returns:
        File size in bytes, or 0 if file doesn't exist
    """
    try:
        return os.path.getsize(file_path)
    except (OSError, FileNotFoundError):
        return 0


def cleanup_old_files(directory: str, max_age_hours: int = 168) -> int:
    """
    Clean up old files in a directory.
    
    Args:
        directory: Directory to clean
        max_age_hours: Maximum age of files in hours (default: 1 week)
    
    Returns:
        Number of files deleted
    """
    deleted_count = 0
    try:
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > max_age_seconds:
                    os.remove(file_path)
                    deleted_count += 1
    except Exception:
        pass
    
    return deleted_count
