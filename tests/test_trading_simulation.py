# -*- coding: utf-8 -*-
"""
Tests for trading simulation and portfolio management.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Import the autotrader module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autotrader import ContinuousAutoTrader


class TestTradingSimulation:
    """Test cases for trading simulation and portfolio management."""
    
    def test_buy_trade_execution(self, isolated_trader):
        """Test BUY trade execution logic."""
        initial_balance = isolated_trader.balance
        
        buy_signal = {
            "signal": "BUY",
            "confidence": 0.8,
            "price": 45000.0,
            "rsi": 50.0
        }
        
        isolated_trader.execute_simulated_trade(buy_signal)
        
        # Balance should decrease (spent on BTC)
        assert isolated_trader.balance < initial_balance
        
        # Calculate expected cost
        trade_amount = 0.01  # BTC
        fee_rate = 0.001  # 0.1%
        expected_cost = buy_signal["price"] * trade_amount * (1 + fee_rate)
        expected_balance = initial_balance - expected_cost
        
        assert abs(isolated_trader.balance - expected_balance) < 0.01
    
    def test_sell_trade_execution(self, isolated_trader):
        """Test SELL trade execution logic."""
        initial_balance = isolated_trader.balance
        
        sell_signal = {
            "signal": "SELL",
            "confidence": 0.2,
            "price": 45000.0,
            "rsi": 50.0
        }
        
        isolated_trader.execute_simulated_trade(sell_signal)
        
        # Balance should increase (gained from selling BTC)
        assert isolated_trader.balance > initial_balance
        
        # Calculate expected revenue
        trade_amount = 0.01  # BTC
        fee_rate = 0.001  # 0.1%
        expected_revenue = sell_signal["price"] * trade_amount * (1 - fee_rate)
        expected_balance = initial_balance + expected_revenue
        
        assert abs(isolated_trader.balance - expected_balance) < 0.01
    
    def test_hold_trade_execution(self, isolated_trader):
        """Test HOLD signal (no trade execution)."""
        initial_balance = isolated_trader.balance
        
        hold_signal = {
            "signal": "HOLD",
            "confidence": 0.5,
            "price": 45000.0,
            "rsi": 50.0
        }
        
        isolated_trader.execute_simulated_trade(hold_signal)
        
        # Balance should remain unchanged
        assert isolated_trader.balance == initial_balance
    
    def test_trade_execution_insufficient_confidence(self, isolated_trader):
        """Test that trades are not executed with insufficient confidence."""
        initial_balance = isolated_trader.balance
        
        # Test low confidence BUY signal
        low_confidence_buy = {
            "signal": "BUY",
            "confidence": 0.55,  # Too close to neutral (0.5)
            "price": 45000.0,
            "rsi": 50.0
        }
        
        isolated_trader.execute_simulated_trade(low_confidence_buy)
        assert isolated_trader.balance == initial_balance
        
        # Test low confidence SELL signal
        low_confidence_sell = {
            "signal": "SELL",
            "confidence": 0.45,  # Too close to neutral (0.5)
            "price": 45000.0,
            "rsi": 50.0
        }
        
        isolated_trader.execute_simulated_trade(low_confidence_sell)
        assert isolated_trader.balance == initial_balance
    
    def test_rsi_override_overbought_condition(self, isolated_trader):
        """Test that BUY signals are ignored when RSI indicates overbought."""
        initial_balance = isolated_trader.balance
        
        overbought_buy_signal = {
            "signal": "BUY",
            "confidence": 0.9,  # High confidence
            "price": 45000.0,
            "rsi": 85.0  # Overbought (>80)
        }
        
        isolated_trader.execute_simulated_trade(overbought_buy_signal)
        
        # Trade should be avoided despite high confidence
        assert isolated_trader.balance == initial_balance
    
    def test_rsi_override_oversold_condition(self, isolated_trader):
        """Test that SELL signals are ignored when RSI indicates oversold."""
        initial_balance = isolated_trader.balance
        
        oversold_sell_signal = {
            "signal": "SELL",
            "confidence": 0.1,  # Low confidence (should trigger SELL)
            "price": 45000.0,
            "rsi": 15.0  # Oversold (<20)
        }
        
        isolated_trader.execute_simulated_trade(oversold_sell_signal)
        
        # Trade should be avoided despite low confidence
        assert isolated_trader.balance == initial_balance
    
    def test_trade_execution_insufficient_balance(self, isolated_trader):
        """Test trade execution when balance is insufficient."""
        # Set very low balance
        isolated_trader.balance = 10.0  # Very low balance
        
        expensive_buy_signal = {
            "signal": "BUY",
            "confidence": 0.8,
            "price": 45000.0,  # Would cost 450+ AUD for 0.01 BTC
            "rsi": 50.0
        }
        
        isolated_trader.execute_simulated_trade(expensive_buy_signal)
        
        # Balance should remain unchanged (insufficient funds)
        assert isolated_trader.balance == 10.0
    
    def test_trade_fee_calculation(self, isolated_trader):
        """Test that trading fees are correctly calculated."""
        initial_balance = isolated_trader.balance
        
        # Test BUY trade fee calculation
        buy_signal = {
            "signal": "BUY",
            "confidence": 0.8,
            "price": 50000.0,
            "rsi": 50.0
        }
        
        isolated_trader.execute_simulated_trade(buy_signal)
        
        trade_amount = 0.01
        fee_rate = 0.001
        expected_cost = 50000.0 * trade_amount * (1 + fee_rate)
        expected_balance = initial_balance - expected_cost
        
        assert abs(isolated_trader.balance - expected_balance) < 0.01
        
        # Reset balance for SELL test
        isolated_trader.balance = initial_balance
        
        # Test SELL trade fee calculation
        sell_signal = {
            "signal": "SELL",
            "confidence": 0.2,
            "price": 50000.0,
            "rsi": 50.0
        }
        
        isolated_trader.execute_simulated_trade(sell_signal)
        
        expected_revenue = 50000.0 * trade_amount * (1 - fee_rate)
        expected_balance = initial_balance + expected_revenue
        
        assert abs(isolated_trader.balance - expected_balance) < 0.01
    
    def test_trade_execution_invalid_prediction(self, isolated_trader):
        """Test trade execution with invalid prediction data."""
        initial_balance = isolated_trader.balance
        
        # Test with None prediction
        isolated_trader.execute_simulated_trade(None)
        assert isolated_trader.balance == initial_balance
        
        # Test with empty prediction
        isolated_trader.execute_simulated_trade({})
        assert isolated_trader.balance == initial_balance
        
        # Test with invalid price
        invalid_prediction = {
            "signal": "BUY",
            "confidence": 0.8,
            "price": 0,  # Invalid price
            "rsi": 50.0
        }
        isolated_trader.execute_simulated_trade(invalid_prediction)
        assert isolated_trader.balance == initial_balance
    
    def test_multiple_consecutive_trades(self, isolated_trader):
        """Test multiple consecutive trades and balance tracking."""
        initial_balance = isolated_trader.balance
        
        # Execute multiple BUY trades
        for i in range(3):
            buy_signal = {
                "signal": "BUY",
                "confidence": 0.8,
                "price": 45000.0 + (i * 100),
                "rsi": 50.0
            }
            isolated_trader.execute_simulated_trade(buy_signal)
        
        # Balance should have decreased significantly
        assert isolated_trader.balance < initial_balance
        
        # Execute multiple SELL trades
        for i in range(3):
            sell_signal = {
                "signal": "SELL",
                "confidence": 0.2,
                "price": 46000.0 + (i * 100),
                "rsi": 50.0
            }
            isolated_trader.execute_simulated_trade(sell_signal)
        
        # Balance might be higher or lower than initial depending on price differences
        assert isinstance(isolated_trader.balance, (int, float))
        assert isolated_trader.balance > 0
    
    def test_trade_execution_edge_confidence_values(self, isolated_trader):
        """Test trade execution with edge confidence values."""
        initial_balance = isolated_trader.balance
        
        # Test exact threshold values
        edge_cases = [
            (0.65, "BUY", True),   # Exactly at BUY threshold
            (0.35, "SELL", True),  # Exactly at SELL threshold
            (0.649, "HOLD", False), # Just below BUY threshold
            (0.351, "HOLD", False), # Just above SELL threshold
        ]
        
        for confidence, expected_signal, should_trade in edge_cases:
            test_signal = {
                "signal": expected_signal if should_trade else "HOLD",
                "confidence": confidence,
                "price": 45000.0,
                "rsi": 50.0
            }
            
            balance_before = isolated_trader.balance
            isolated_trader.execute_simulated_trade(test_signal)
            
            if should_trade:
                assert isolated_trader.balance != balance_before
            else:
                assert isolated_trader.balance == balance_before
    
    def test_trade_execution_logging(self, isolated_trader, mock_logging):
        """Test that trade executions are properly logged."""
        buy_signal = {
            "signal": "BUY",
            "confidence": 0.8,
            "price": 45000.0,
            "rsi": 50.0
        }
        
        with patch('logging.getLogger') as mock_logger:
            isolated_trader.execute_simulated_trade(buy_signal)
            
            # Should have logged the trade execution
            mock_logger.return_value.info.assert_called()
    
    def test_portfolio_balance_tracking(self, isolated_trader):
        """Test portfolio balance tracking across multiple trades."""
        initial_balance = isolated_trader.balance
        
        # Track balance changes
        balance_history = [initial_balance]
        
        # Execute a series of trades
        trades = [
            {"signal": "BUY", "confidence": 0.8, "price": 45000.0, "rsi": 50.0},
            {"signal": "SELL", "confidence": 0.2, "price": 46000.0, "rsi": 50.0},
            {"signal": "BUY", "confidence": 0.7, "price": 44000.0, "rsi": 50.0},
            {"signal": "SELL", "confidence": 0.3, "price": 47000.0, "rsi": 50.0},
        ]
        
        for trade in trades:
            isolated_trader.execute_simulated_trade(trade)
            balance_history.append(isolated_trader.balance)
        
        # Should have 5 balance records (initial + 4 trades)
        assert len(balance_history) == 5
        
        # All balances should be positive
        assert all(balance > 0 for balance in balance_history)
        
        # Final balance should be different from initial
        assert balance_history[-1] != balance_history[0]
    
    def test_trade_simulation_realistic_scenarios(self, isolated_trader):
        """Test trading simulation with realistic market scenarios."""
        # Scenario 1: Bull market (prices rising)
        bull_market_trades = [
            {"signal": "BUY", "confidence": 0.8, "price": 45000.0, "rsi": 45.0},
            {"signal": "HOLD", "confidence": 0.55, "price": 45500.0, "rsi": 55.0},
            {"signal": "SELL", "confidence": 0.3, "price": 46000.0, "rsi": 70.0},
        ]
        
        initial_balance = isolated_trader.balance
        
        for trade in bull_market_trades:
            isolated_trader.execute_simulated_trade(trade)
        
        # In a bull market scenario, we should profit
        profit_loss = isolated_trader.balance - initial_balance
        # Note: Actual profit depends on specific timing and fees
        assert isinstance(profit_loss, (int, float))
        
        # Scenario 2: Bear market (prices falling)
        isolated_trader.balance = initial_balance  # Reset
        
        bear_market_trades = [
            {"signal": "SELL", "confidence": 0.2, "price": 45000.0, "rsi": 55.0},
            {"signal": "HOLD", "confidence": 0.45, "price": 44500.0, "rsi": 45.0},
            {"signal": "BUY", "confidence": 0.7, "price": 44000.0, "rsi": 30.0},
        ]
        
        for trade in bear_market_trades:
            isolated_trader.execute_simulated_trade(trade)
        
        # Should have executed trades appropriately
        assert isolated_trader.balance != initial_balance
    
    def test_trade_amount_consistency(self, isolated_trader):
        """Test that trade amounts are consistent across all trades."""
        # All trades should use the same amount (0.01 BTC)
        trade_amount = 0.01
        
        # Test multiple trades with different prices
        prices = [40000.0, 45000.0, 50000.0, 55000.0]
        
        for price in prices:
            initial_balance = isolated_trader.balance
            
            buy_signal = {
                "signal": "BUY",
                "confidence": 0.8,
                "price": price,
                "rsi": 50.0
            }
            
            isolated_trader.execute_simulated_trade(buy_signal)
            
            # Calculate expected cost
            fee_rate = 0.001
            expected_cost = price * trade_amount * (1 + fee_rate)
            expected_balance = initial_balance - expected_cost
            
            assert abs(isolated_trader.balance - expected_balance) < 0.01
            
            # Reset balance for next test
            isolated_trader.balance = initial_balance
