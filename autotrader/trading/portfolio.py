"""
Portfolio management for autotrader bot.

Tracks portfolio state, positions, and performance metrics.
"""

from typing import Dict, List
from datetime import datetime
import logging

logger = logging.getLogger("autotrader.trading.portfolio")


class Portfolio:
    """Manages trading portfolio state and metrics."""
    
    def __init__(self, initial_balance: float = 10000.0):
        """
        Initialize portfolio.
        
        Args:
            initial_balance: Initial cash balance
        """
        self.initial_balance = initial_balance
        self.cash_balance = initial_balance
        self.positions = {}  # Symbol -> quantity
        self.trade_history = []
        self.created_at = datetime.now()
    
    def get_total_value(self, current_prices: Dict[str, float] = None) -> float:
        """
        Calculate total portfolio value.
        
        Args:
            current_prices: Dictionary of current asset prices
        
        Returns:
            Total portfolio value
        """
        total_value = self.cash_balance
        
        if current_prices:
            for symbol, quantity in self.positions.items():
                if symbol in current_prices:
                    total_value += quantity * current_prices[symbol]
        
        return total_value
    
    def add_trade(self, trade: Dict):
        """Add a trade to the portfolio history."""
        self.trade_history.append(trade)
    
    def get_performance_summary(self, current_prices: Dict[str, float] = None) -> Dict:
        """Get portfolio performance summary."""
        current_value = self.get_total_value(current_prices)
        total_return = current_value - self.initial_balance
        return_percentage = (total_return / self.initial_balance) * 100
        
        return {
            'initial_balance': self.initial_balance,
            'current_value': current_value,
            'cash_balance': self.cash_balance,
            'total_return': total_return,
            'return_percentage': return_percentage,
            'positions': self.positions.copy(),
            'total_trades': len(self.trade_history)
        }
