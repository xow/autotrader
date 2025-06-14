"""
API integration components for the Autotrader Bot

Handles BTCMarkets API communication, real-time data streaming,
and external service integrations.
"""

from .btcmarkets_client import BTCMarketsClient
from .websocket_handler import WebSocketHandler
from .rate_limiter import RateLimiter

__all__ = [
    "BTCMarketsClient",
    "WebSocketHandler",
    "RateLimiter"
]
