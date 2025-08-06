# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

cryptoBandit3 is an automated cryptocurrency trading bot for Binance that uses RSI (Relative Strength Index) and EMA (Exponential Moving Average) strategies. The system tracks multiple cryptocurrencies simultaneously and executes buy/sell orders based on defined technical indicators and thresholds.

## Environment Setup

This project uses Python 3.13 with pipenv for dependency management:

```bash
# Install dependencies
pipenv install

# Activate virtual environment
pipenv shell

# Run the main trading bot
python cryptoBandit3.py

# Run backtesting scripts
python cryptoBacktest-EMA-RSI-exitPerPercent.py
python cryptoBacktest-EMA-exitPerEmaCross.py

# Download historical data
python binance_data_download.py
```

Alternative using requirements.txt:
```bash
pip install -r requirements.txt
```

## Core Architecture

### Main Components

1. **cryptoBandit3.py** - Main trading bot with live trading logic
   - Multi-symbol tracking (BTC, ETH, BNB, ADA, XRP, DOGE, SOL, etc.)
   - RSI-based buy signals (RSI < 30) with EMA trend confirmation
   - Real-time order management and position tracking
   - Graceful shutdown handling (Ctrl+C or 'x' + ENTER for panic sell)

2. **Backtesting Scripts**
   - **cryptoBacktest-EMA-RSI-exitPerPercent.py** - Percentage-based exit strategy
   - **cryptoBacktest-EMA-exitPerEmaCross.py** - EMA crossover exit strategy

3. **binance_data_download.py** - Historical data fetcher for backtesting
   - Downloads OHLCV data from Binance API
   - Supports multiple timeframes (1m, 1h, 4h, 1d)
   - Handles data cleaning and CSV export

### Trading Logic

**Buy Conditions:**
```python
if percent_change <= -buy_threshold and (rsi < 30) and (ema1 > ema200):
```

**Key Functions:**
- `calculate_rsi(symbol, interval, client)` - RSI calculation (cryptoBandit3.py:105)
- `calculate_emas(symbol, interval, client)` - EMA calculation (cryptoBandit3.py:143)

### Configuration

Trading parameters are configured as constants at the top of each script:
- `usd_amount = 50` - Trade amount per position
- `buy_threshold = 0.01` - 1% price drop threshold
- `sell_threshold = 0.01` - 1% profit target
- `stop_loss_threshold = 0.8` - 80% stop loss
- `kline_interval` - Timeframe for analysis (1m, 1h, etc.)

### API Keys

Binance API credentials must be stored in `_secrets/__init__.py`:
```python
api_key = 'your_binance_api_key'
secret_key = 'your_binance_secret_key'
```

### State Management

The bot maintains persistent state using local text files:
- `order_id_{SYMBOL}.txt` - Active order IDs
- `last_buy_price_{SYMBOL}.txt` - Last purchase prices
- Position tracking across bot restarts

### Data Files

Historical price data is stored in CSV format with naming convention:
- `btc_usdt_{timeframe}_{date_range}_{timestamp}.csv`
- Contains OHLCV data for backtesting and analysis

## Switching Timeframes

To change from minute-based to hourly analysis:
1. Update `kline_interval = Client.KLINE_INTERVAL_1HOUR`
2. Adjust lookback windows in `calculate_rsi` and `calculate_emas` from "15 minutes ago UTC" to "400 hours ago UTC"

## Safety Features

- Graceful shutdown on SIGINT/SIGTERM
- Panic sell option ('x' + ENTER)
- Position tracking to prevent duplicate orders
- Retry mechanisms using tenacity library
- Real-time status logging with termcolor