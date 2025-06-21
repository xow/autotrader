"""
Default configuration settings for the Autotrader Bot.
"""

def get_default_config():
    """
    Return the default configuration as a dictionary.
    
    Returns:
        dict: Default configuration dictionary
    """
    return {
        "general": {
            "environment": "development",
            "log_level": "INFO",
            "data_dir": "data",
            "models_dir": "models",
            "checkpoints_dir": "checkpoints"
        },
        "trading": {
            "initial_balance": 10000.0,
            "trade_fee": 0.001,  # 0.1% fee per trade
            "max_open_trades": 5,
            "risk_per_trade": 0.02,  # 2% risk per trade
            "stop_loss_pct": 0.05,  # 5% stop loss
            "take_profit_pct": 0.10  # 10% take profit
        },
        "data": {
            "candlestick_interval": "1h",
            "historical_days": 30,
            "data_refresh_interval": 300,  # 5 minutes
            "features": [
                "close",
                "volume",
                "sma_20",
                "sma_50",
                "rsi_14",
                "macd"
            ]
        },
        "model": {
            "name": "lstm_predictor",
            "input_sequence_length": 24,  # hours
            "output_sequence_length": 1,  # predict next candle
            "hidden_layers": [64, 32],
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 50,
            "validation_split": 0.2,
            "early_stopping_patience": 5
        },
        "api": {
            "exchange": "btcmarkets",
            "api_key": "",
            "api_secret": "",
            "testnet": True,
            "request_timeout": 30
        },
        "monitoring": {
            "enable_metrics": True,
            "metrics_port": 8000,
            "enable_health_checks": True,
            "health_check_interval": 300  # 5 minutes
        }
    }
