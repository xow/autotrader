"""
Trading signal generation for autotrader bot.

Converts ML predictions into trading signals with risk management.
"""

from typing import Dict, Optional
import logging

from ..core.config import Config

logger = logging.getLogger("autotrader.trading.signals")


class SignalGenerator:
    """Generates trading signals from ML predictions."""
    
    def __init__(self, config: Config):
        """
        Initialize signal generator.
        
        Args:
            config: Configuration instance
        """
        self.config = config
    
    def generate_signal(self, prediction: Dict, market_data: Dict) -> Dict:
        """
        Generate trading signal from prediction and market data.
        
        Args:
            prediction: ML prediction dictionary
            market_data: Current market data
        
        Returns:
            Trading signal dictionary
        """
        confidence = prediction.get('confidence', 0.5)
        price = market_data.get('price', 0)
        rsi = market_data.get('rsi', 50)
        
        # Generate base signal from confidence
        if confidence > self.config.buy_threshold:
            signal = "BUY"
        elif confidence < self.config.sell_threshold:
            signal = "SELL"
        else:
            signal = "HOLD"
        
        # Apply risk management rules
        signal = self._apply_risk_management(signal, confidence, rsi, market_data)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'price': price,
            'rsi': rsi,
            'timestamp': prediction.get('timestamp'),
            'reasoning': self._get_signal_reasoning(signal, confidence, rsi)
        }
    
    def _apply_risk_management(self, signal: str, confidence: float, rsi: float, market_data: Dict) -> str:
        """Apply risk management rules to modify signals."""
        # Check confidence threshold
        if abs(confidence - 0.5) < self.config.confidence_threshold:
            return "HOLD"
        
        # RSI-based risk management
        if signal == "BUY" and rsi > self.config.rsi_overbought:
            return "HOLD"
        
        if signal == "SELL" and rsi < self.config.rsi_oversold:
            return "HOLD"
        
        return signal
    
    def _get_signal_reasoning(self, signal: str, confidence: float, rsi: float) -> str:
        """Get human-readable reasoning for the signal."""
        if signal == "BUY":
            return f"BUY signal: High confidence ({confidence:.3f}), RSI {rsi:.1f}"
        elif signal == "SELL":
            return f"SELL signal: Low confidence ({confidence:.3f}), RSI {rsi:.1f}"
        else:
            return f"HOLD signal: Neutral confidence ({confidence:.3f}), RSI {rsi:.1f}"
