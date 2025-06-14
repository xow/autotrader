"""
Trading simulation for autotrader bot.

Simulates trading operations with realistic constraints and fees.
"""

import logging
from typing import Dict, Optional
from datetime import datetime

from ..core.config import Config

logger = logging.getLogger("autotrader.trading.simulator")


class TradingSimulator:
    """Simulates cryptocurrency trading operations."""
    
    def __init__(self, config: Config, initial_balance: float = None):
        """
        Initialize trading simulator.
        
        Args:
            config: Configuration instance
            initial_balance: Initial balance override
        """
        self.config = config
        self.balance = initial_balance or config.initial_balance
        self.initial_balance = self.balance
        self.trade_history = []
    
    def execute_trade(self, signal: str, price: float, confidence: float, additional_data: Dict = None) -> Dict:
        """
        Execute a simulated trade.
        
        Args:
            signal: Trading signal (BUY, SELL, HOLD)
            price: Current price
            confidence: Prediction confidence
            additional_data: Additional trade data
        
        Returns:
            Trade execution result
        """
        additional_data = additional_data or {}
        trade_result = {
            'timestamp': datetime.now().isoformat(),
            'signal': signal,
            'price': price,
            'confidence': confidence,
            'balance_before': self.balance,
            'balance_after': self.balance,
            'amount': 0,
            'cost': 0,
            'executed': False,
            'reason': 'No action taken',
            **additional_data
        }
        
        try:
            if signal == 'BUY':
                trade_result = self._execute_buy(price, confidence, trade_result, additional_data)
            elif signal == 'SELL':
                trade_result = self._execute_sell(price, confidence, trade_result, additional_data)
            else:
                trade_result['reason'] = 'HOLD signal - no trade executed'
            
            self.trade_history.append(trade_result)
            return trade_result
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            trade_result['reason'] = f'Error: {str(e)}'
            return trade_result
    
    def _execute_buy(self, price: float, confidence: float, trade_result: Dict, additional_data: Dict) -> Dict:
        """Execute a buy trade."""
        rsi = additional_data.get('rsi', 50)
        
        # Check confidence threshold
        if abs(confidence - 0.5) < self.config.confidence_threshold:
            trade_result['reason'] = f'Confidence too neutral ({confidence:.3f})'
            return trade_result
        
        # Check RSI
        if rsi > self.config.rsi_overbought:
            trade_result['reason'] = f'RSI too high ({rsi:.1f}), avoiding BUY'
            return trade_result
        
        # Calculate trade amount and cost
        amount = self.config.trade_amount
        cost = price * amount * (1 + self.config.fee_rate)
        
        # Check if we have enough balance
        if self.balance < cost:
            trade_result['reason'] = f'Insufficient balance: ${self.balance:.2f} < ${cost:.2f}'
            return trade_result
        
        # Execute trade
        self.balance -= cost
        trade_result.update({
            'balance_after': self.balance,
            'amount': amount,
            'cost': cost,
            'executed': True,
            'reason': f'BUY executed: {amount} BTC at ${price:.2f}'
        })
        
        logger.info(f"BUY: {amount} BTC at ${price:.2f} (Cost: ${cost:.2f}, Confidence: {confidence:.3f}, RSI: {rsi:.1f})")
        return trade_result
    
    def _execute_sell(self, price: float, confidence: float, trade_result: Dict, additional_data: Dict) -> Dict:
        """Execute a sell trade."""
        rsi = additional_data.get('rsi', 50)
        
        # Check confidence threshold
        if abs(confidence - 0.5) < self.config.confidence_threshold:
            trade_result['reason'] = f'Confidence too neutral ({confidence:.3f})'
            return trade_result
        
        # Check RSI
        if rsi < self.config.rsi_oversold:
            trade_result['reason'] = f'RSI too low ({rsi:.1f}), avoiding SELL'
            return trade_result
        
        # Calculate trade amount and revenue
        amount = self.config.trade_amount
        revenue = price * amount * (1 - self.config.fee_rate)
        
        # Execute trade
        self.balance += revenue
        trade_result.update({
            'balance_after': self.balance,
            'amount': amount,
            'cost': -revenue,  # Negative for sell
            'executed': True,
            'reason': f'SELL executed: {amount} BTC at ${price:.2f}'
        })
        
        logger.info(f"SELL: {amount} BTC at ${price:.2f} (Revenue: ${revenue:.2f}, Confidence: {confidence:.3f}, RSI: {rsi:.1f})")
        return trade_result
    
    def get_performance_metrics(self) -> Dict:
        """Get trading performance metrics."""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'profitable_trades': 0,
                'total_return': 0,
                'total_return_percentage': 0,
                'win_rate': 0
            }
        
        executed_trades = [t for t in self.trade_history if t['executed']]
        buy_trades = [t for t in executed_trades if t['signal'] == 'BUY']
        sell_trades = [t for t in executed_trades if t['signal'] == 'SELL']
        
        total_return = self.balance - self.initial_balance
        total_return_percentage = (total_return / self.initial_balance) * 100
        
        return {
            'total_trades': len(executed_trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'total_return': total_return,
            'total_return_percentage': total_return_percentage,
            'current_balance': self.balance,
            'initial_balance': self.initial_balance
        }
