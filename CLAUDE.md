# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CryptoBandit3 is an automated cryptocurrency trading bot for Binance that uses RSI (Relative Strength Index) and EMA (Exponential Moving Average) technical indicators to execute buy/sell orders across multiple cryptocurrencies simultaneously.

### Core Architecture

The system consists of three main components:

1. **Main Trading Bot** (`cryptoBandit3.py`): Real-time trading execution with state persistence
2. **Backtesting Scripts** (`cryptoBacktest-*.py`): Historical strategy validation 
3. **State Management**: File-based persistence using JSON and text files

### Trading Strategy

**Buy Logic**: `percent_change <= -buy_threshold and (rsi < 30) and (ema1 > ema200)`
- RSI below 30 (oversold condition)
- EMA1 above EMA200 (upward trend confirmation)
- Price drop exceeding buy threshold

**Sell Logic**: Profit target or stop-loss based on configurable thresholds

## Development Setup

### Environment Setup
```bash
python3 -m venv venv
source ./venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### API Configuration
Create `_secrets/__init__.py` with your Binance API credentials:
```python
api_key = 'your_binance_api_key'
secret_key = 'your_binance_secret_key'
```

### Running the Bot
```bash
python3 cryptoBandit3.py
```

### Backtesting
```bash
python3 cryptoBacktest-EMA-RSI-exitPerPercent.py
python3 cryptoBacktest-EMA-exitPerEmaCross.py
```

## Key Constants and Configuration

Located in `cryptoBandit3.py:86-92`:
- `usd_amount = 100`: USD amount per trade
- `buy_threshold = 0.01`: 1% price drop trigger
- `sell_threshold = 0.02`: 2% profit target
- `stop_loss_threshold = 0.8`: 80% stop-loss
- `kline_interval = Client.KLINE_INTERVAL_1MINUTE`: Candle timeframe

## State Management System

The bot maintains persistent state across restarts:

### Per-Symbol Files
- `order_id_{SYMBOL}.txt`: Stores buy price and quantity as comma-separated values
- `output_{SYMBOL}.txt`: Trading logs and outputs

### Global State
- `status.json`: Overall profit/loss tracking per symbol
- Automatic state restoration on startup from existing files

## Monitored Cryptocurrencies

Default symbols in `cryptoBandit3.py:22-36`:
`BTCUSDC`, `ETHUSDC`, `BNBUSDC`, `ADAUSDC`, `XRPUSDC`, `DOGEUSDC`, `SOLUSDC`, `PNUTUSDC`, `PEPEUSDC`, `SHIBUSDC`, `XLMUSDC`, `LINKUSDC`, `IOTAUSDC`

## Technical Indicators

### RSI Calculation (`calculate_rsi()`)
- 14-period RSI using 50-minute historical data
- Returns integer value for buy signal evaluation

### EMA Calculation (`calculate_emas()`)
- EMA1, EMA9, EMA26, EMA200 using 300-minute historical data
- Used for trend confirmation in buy logic

## Emergency Controls

- **Graceful Shutdown**: CTRL+C for graceful exit
- **Panic Sell**: Type 'x' + ENTER during execution
- Signal handling for SIGINT and SIGTERM

## Switching to Different Timeframes

For hourly candles:
1. Change `kline_interval = Client.KLINE_INTERVAL_1HOUR`
2. Update historical data windows in both `calculate_rsi()` and `calculate_emas()` from "50 minutes ago UTC" and "300 minutes ago UTC" to "400 hours ago UTC"

## File Structure Notes

- Main bot logic and trading functions in `cryptoBandit3.py`
- Backtesting variants test different exit strategies
- `_secrets/` directory contains API credentials (gitignored)
- State files are created automatically during trading
- Virtual environment in `venv/` directory