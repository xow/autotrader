# Autotrader Bot Project

A continuously learning cryptocurrency trading bot that uses machine learning to optimize trading decisions with live data from BTCMarkets.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Development](#development)
- [Testing](#testing)
- [Architecture](#architecture)
- [API Integration](#api-integration)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements an autonomous trading bot that:
- **Continuously learns** from live market data using machine learning
- **Operates 24/7** with automatic recovery and state persistence
- **Simulates trading** decisions based on neural network predictions
- **Preserves all progress** across shutdowns and restarts
- **Improves over time** through incremental learning algorithms

The bot is designed for long-term autonomous operation, capable of running overnight while continuously improving its predictive capabilities.

## Features

### 🧠 Continuous Learning
- **Real-time adaptation** to market conditions
- **Incremental learning** with each new data point
- **Progressive improvement** in prediction accuracy
- **Memory-efficient** online learning algorithms

### 🔄 Autonomous Operation
- **24/7 operation** without human intervention
- **Automatic recovery** from network/system failures
- **Graceful shutdown** with complete state preservation
- **Seamless resume** from exact stopping point

### 💾 Data Persistence
- **Zero data loss** protection
- **Instant state saving** after each training iteration
- **Incremental checkpointing** of model weights and optimizer states
- **Session continuity** across multiple runs

### 📊 Trading Simulation
- **Real-time portfolio tracking**
- **Performance analytics** and metrics
- **Confidence-based** trading decisions
- **Comprehensive trade logging**

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/xow/autotrader.git
   cd autotrader
   ```

2. **Set up the environment:**
   ```bash
   make setup
   ```

3. **Run the bot:**
   ```bash
   make run
   ```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Automatic Setup
```bash
make setup
```

### Manual Setup
1. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install optional dependencies (recommended):**
   ```bash
   pip install talib-binary  # For technical indicators
   ```

## Configuration

### Environment Variables
Create a `.env` file in the project root:

```env
# BTCMarkets API Configuration
BTCMARKETS_API_KEY=your_api_key_here
BTCMARKETS_API_SECRET=your_api_secret_here

# Trading Configuration
INITIAL_BALANCE=10000.0
TRADING_PAIR=BTC-AUD
RISK_TOLERANCE=0.02

# Model Configuration
SEQUENCE_LENGTH=20
MAX_TRAINING_SAMPLES=2000
LEARNING_RATE=0.001
```

### Configuration Files
- `config.json` - Main application configuration
- `model_config.json` - Neural network architecture settings
- `trading_config.json` - Trading parameters and risk management

## Usage

### Basic Operation
```bash
# Start the bot
python autotrader.py

# Run with specific configuration
python autotrader.py --config custom_config.json

# Run in background
nohup python autotrader.py &
```

### Development Commands
```bash
# Run tests
make test

# Check code quality
make lint

# Format code
make format

# Generate documentation
make docs

# Clean build artifacts
make clean
```

### Monitoring
- **Logs:** Check `autotrader.log` for detailed operation logs
- **State:** Monitor `trader_state.pkl` for current bot state
- **Performance:** View training progress in real-time output

## Development

### Project Structure
```
autotrader/
├── autotrader.py          # Main application entry point
├── src/                   # Source code modules
│   ├── models/           # Machine learning models
│   ├── data/             # Data processing and storage
│   ├── trading/          # Trading logic and simulation
│   └── utils/            # Utility functions
├── tests/                # Test suites
├── docs/                 # Documentation
├── config/               # Configuration files
└── requirements.txt      # Python dependencies
```

### Development Setup
1. **Install development dependencies:**
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Set up pre-commit hooks:**
   ```bash
   pre-commit install
   ```

3. **Run tests:**
   ```bash
   pytest tests/
   ```

### Code Quality Standards
- **Type hints** for all function signatures
- **Docstrings** for all public methods
- **Unit tests** with >90% coverage
- **PEP 8** compliance with black formatting
- **Import sorting** with isort

## Testing

### Running Tests
```bash
# Run all tests
make test

# Run specific test module
pytest tests/test_models.py

# Run with coverage
pytest --cov=src tests/

# Run integration tests
pytest tests/integration/
```

### Test Categories
- **Unit Tests:** Individual component testing
- **Integration Tests:** API and system integration
- **Performance Tests:** Memory usage and speed benchmarks
- **Stress Tests:** Long-term operation simulation

## Architecture

### Core Components

#### 1. Continuous Learning Engine
- **Online Learning:** Incremental model updates with each data point
- **Neural Network:** TensorFlow-based LSTM for time series prediction
- **Feature Engineering:** Technical indicators and market data preprocessing
- **Performance Tracking:** Real-time accuracy monitoring

#### 2. Data Management System
- **Live Data Stream:** Real-time BTCMarkets API integration
- **Persistence Layer:** SQLite database for historical data
- **State Management:** Automatic checkpointing and recovery
- **Data Validation:** Integrity checks and error handling

#### 3. Trading Simulation Engine
- **Portfolio Management:** Virtual balance and position tracking
- **Decision Engine:** ML-based buy/sell signal generation
- **Risk Management:** Position sizing and stop-loss logic
- **Performance Analytics:** Profit/loss tracking and reporting

#### 4. Autonomous Operation System
- **Session Orchestrator:** Long-term operation management
- **Health Monitoring:** System resource and performance tracking
- **Error Recovery:** Automatic restart and state restoration
- **Logging System:** Comprehensive operation logging

### Technology Stack
- **Python 3.8+** - Core programming language
- **TensorFlow 2.x** - Machine learning framework
- **pandas/numpy** - Data processing and analysis
- **asyncio** - Asynchronous programming
- **SQLite** - Local data storage
- **requests** - HTTP API client

## API Integration

### BTCMarkets API
The bot integrates with BTCMarkets API for:
- **Real-time market data** streaming
- **Historical price data** retrieval
- **Order book information**
- **Account balance** monitoring (simulation only)

### Rate Limiting
- Automatic rate limit handling
- Exponential backoff for API failures
- Connection pooling for efficiency
- Health monitoring and alerting

## Contributing

### Development Workflow
1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Coding Standards
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

### Issue Reporting
Please use GitHub Issues to report bugs or request features. Include:
- **Environment details** (OS, Python version)
- **Steps to reproduce** the issue
- **Expected vs actual** behavior
- **Log files** if applicable

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation:** [Project Wiki](https://github.com/xow/autotrader/wiki)
- **Issues:** [GitHub Issues](https://github.com/xow/autotrader/issues)
- **Discussions:** [GitHub Discussions](https://github.com/xow/autotrader/discussions)

---

**⚠️ Disclaimer:** This bot is for educational and simulation purposes only. It does not execute real trades. Always conduct thorough testing before using any trading algorithm with real money.
