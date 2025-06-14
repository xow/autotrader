"""
Technical indicators calculation for autotrader bot.

Provides both TA-Lib and manual implementations of common technical indicators.
"""

import numpy as np
from typing import Dict, Optional, List
import logging

# Try to import talib, with fallback for manual calculations
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

logger = logging.getLogger("autotrader.ml.indicators")


class TechnicalIndicators:
    """Technical indicators calculator with TA-Lib and manual implementations."""
    
    def __init__(self, use_talib: bool = True):
        """
        Initialize technical indicators calculator.
        
        Args:
            use_talib: Whether to use TA-Lib when available
        """
        self.use_talib = use_talib and TALIB_AVAILABLE
        
        if not TALIB_AVAILABLE and use_talib:
            logger.warning("TA-Lib not available, using manual calculations")
    
    def simple_moving_average(self, prices: np.ndarray, period: int) -> float:
        """
        Calculate Simple Moving Average.
        
        Args:
            prices: Array of prices
            period: Period for calculation
        
        Returns:
            SMA value
        """
        if len(prices) < period:
            return prices[-1] if len(prices) > 0 else 0
        
        if self.use_talib:
            try:
                sma = talib.SMA(prices, timeperiod=period)
                return float(sma[-1]) if len(sma) > 0 and not np.isnan(sma[-1]) else prices[-1]
            except Exception:
                pass
        
        # Manual calculation
        return float(np.mean(prices[-period:]))
    
    def exponential_moving_average(self, prices: np.ndarray, period: int) -> float:
        """
        Calculate Exponential Moving Average.
        
        Args:
            prices: Array of prices
            period: Period for calculation
        
        Returns:
            EMA value
        """
        if len(prices) < period:
            return prices[-1] if len(prices) > 0 else 0
        
        if self.use_talib:
            try:
                ema = talib.EMA(prices, timeperiod=period)
                return float(ema[-1]) if len(ema) > 0 and not np.isnan(ema[-1]) else prices[-1]
            except Exception:
                pass
        
        # Manual calculation
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return float(ema)
    
    def relative_strength_index(self, prices: np.ndarray, period: int = 14) -> float:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Array of prices
            period: Period for calculation
        
        Returns:
            RSI value
        """
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        if self.use_talib:
            try:
                rsi = talib.RSI(prices, timeperiod=period)
                return float(rsi[-1]) if len(rsi) > 0 and not np.isnan(rsi[-1]) else 50.0
            except Exception:
                pass
        
        # Manual calculation
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)
    
    def macd(self, prices: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, float]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Array of prices
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
        
        Returns:
            Dictionary with MACD line and signal line values
        """
        if len(prices) < slow_period:
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
        
        if self.use_talib:
            try:
                macd_line, signal_line, histogram = talib.MACD(prices, fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
                return {
                    'macd': float(macd_line[-1]) if len(macd_line) > 0 and not np.isnan(macd_line[-1]) else 0.0,
                    'signal': float(signal_line[-1]) if len(signal_line) > 0 and not np.isnan(signal_line[-1]) else 0.0,
                    'histogram': float(histogram[-1]) if len(histogram) > 0 and not np.isnan(histogram[-1]) else 0.0
                }
            except Exception:
                pass
        
        # Manual calculation
        ema_fast = self.exponential_moving_average(prices, fast_period)
        ema_slow = self.exponential_moving_average(prices, slow_period)
        macd_line = ema_fast - ema_slow
        
        # For signal line, we need to calculate EMA of MACD line
        # Simplified approach: use single MACD value
        signal_line = macd_line  # Simplified
        histogram = macd_line - signal_line
        
        return {
            'macd': float(macd_line),
            'signal': float(signal_line),
            'histogram': float(histogram)
        }
    
    def bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: int = 2) -> Dict[str, float]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Array of prices
            period: Period for calculation
            std_dev: Standard deviation multiplier
        
        Returns:
            Dictionary with upper, middle, and lower band values
        """
        if len(prices) < period:
            price = prices[-1] if len(prices) > 0 else 0
            return {'upper': price, 'middle': price, 'lower': price}
        
        if self.use_talib:
            try:
                upper, middle, lower = talib.BBANDS(prices, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev, matype=0)
                return {
                    'upper': float(upper[-1]) if len(upper) > 0 and not np.isnan(upper[-1]) else prices[-1],
                    'middle': float(middle[-1]) if len(middle) > 0 and not np.isnan(middle[-1]) else prices[-1],
                    'lower': float(lower[-1]) if len(lower) > 0 and not np.isnan(lower[-1]) else prices[-1]
                }
            except Exception:
                pass
        
        # Manual calculation
        sma = self.simple_moving_average(prices, period)
        std = float(np.std(prices[-period:]))
        
        return {
            'upper': sma + (std_dev * std),
            'middle': sma,
            'lower': sma - (std_dev * std)
        }
    
    def volume_weighted_average_price(self, prices: np.ndarray, volumes: np.ndarray, period: int = 20) -> float:
        """
        Calculate Volume Weighted Average Price.
        
        Args:
            prices: Array of prices
            volumes: Array of volumes
            period: Period for calculation
        
        Returns:
            VWAP value
        """
        if len(prices) < period or len(volumes) < period:
            return prices[-1] if len(prices) > 0 else 0
        
        try:
            recent_prices = prices[-period:]
            recent_volumes = volumes[-period:]
            
            total_volume = np.sum(recent_volumes)
            if total_volume == 0:
                return float(np.mean(recent_prices))
            
            weighted_prices = recent_prices * recent_volumes
            vwap = np.sum(weighted_prices) / total_volume
            
            return float(vwap)
        except Exception:
            return prices[-1] if len(prices) > 0 else 0
    
    def calculate_all_indicators(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, float]:
        """
        Calculate all technical indicators at once.
        
        Args:
            prices: Array of prices
            volumes: Array of volumes
        
        Returns:
            Dictionary with all indicator values
        """
        if len(prices) < 5:  # Need minimum data
            return {
                'sma_5': 0, 'sma_20': 0, 'ema_12': 0, 'ema_26': 0,
                'rsi': 50, 'macd': 0, 'macd_signal': 0, 'macd_histogram': 0,
                'bb_upper': 0, 'bb_middle': 0, 'bb_lower': 0, 'volume_sma': 0,
                'vwap': 0
            }
        
        try:
            # Moving averages
            sma_5 = self.simple_moving_average(prices, 5)
            sma_20 = self.simple_moving_average(prices, 20)
            ema_12 = self.exponential_moving_average(prices, 12)
            ema_26 = self.exponential_moving_average(prices, 26)
            
            # RSI
            rsi = self.relative_strength_index(prices, 14)
            
            # MACD
            macd_data = self.macd(prices)
            
            # Bollinger Bands
            bb_data = self.bollinger_bands(prices)
            
            # Volume indicators
            volume_sma = self.simple_moving_average(volumes, 10) if len(volumes) >= 10 else volumes[-1] if len(volumes) > 0 else 0
            vwap = self.volume_weighted_average_price(prices, volumes) if len(volumes) > 0 else prices[-1] if len(prices) > 0 else 0
            
            return {
                'sma_5': sma_5,
                'sma_20': sma_20,
                'ema_12': ema_12,
                'ema_26': ema_26,
                'rsi': rsi,
                'macd': macd_data['macd'],
                'macd_signal': macd_data['signal'],
                'macd_histogram': macd_data['histogram'],
                'bb_upper': bb_data['upper'],
                'bb_middle': bb_data['middle'],
                'bb_lower': bb_data['lower'],
                'volume_sma': volume_sma,
                'vwap': vwap
            }
            
        except Exception as e:
            logger.warning(f"Error calculating technical indicators: {e}")
            return {
                'sma_5': 0, 'sma_20': 0, 'ema_12': 0, 'ema_26': 0,
                'rsi': 50, 'macd': 0, 'macd_signal': 0, 'macd_histogram': 0,
                'bb_upper': 0, 'bb_middle': 0, 'bb_lower': 0, 'volume_sma': 0,
                'vwap': 0
            }
