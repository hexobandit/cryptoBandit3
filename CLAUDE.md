# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based cryptocurrency trading bot for Binance that uses RSI and EMA technical indicators to automate buy/sell decisions across multiple cryptocurrency pairs. The bot runs continuously, monitoring market conditions and executing trades based on configurable thresholds.

## Setup and Installation

```bash
# Create virtual environment
python3 -m venv venv
source ./venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
# Create _secrets/__init__.py with:
# api_key = 'your_binance_api_key'
# secret_key = 'your_binance_secret_key'

# Run the main trading bot
python3 cryptoBandit3.py
```

## Core Architecture

### Main Trading Bot (`cryptoBandit3.py`)
- **Trading Logic**: RSI < 30 + EMA trend analysis + price drop threshold triggers buy signals
- **Multi-coin Support**: Tracks 14 cryptocurrency pairs simultaneously (BTCUSDC, ETHUSDC, etc.)
- **State Management**: Persists position data in individual text files (`order_id_{SYMBOL}.txt`)
- **Profit Tracking**: Maintains overall P&L in `status.json`
- **Manual Controls**: Press 'x' + ENTER to trigger emergency sell-all

### Key Trading Parameters (in `cryptoBandit3.py`)
```python
usd_amount = 100                # Trade size per position
buy_threshold = 0.01            # 1% price drop required
sell_threshold = 0.02           # 2% profit target
stop_loss_threshold = 0.8       # 80% stop loss
reset_initial_price = 0.003     # 0.3% price reset threshold
kline_interval = Client.KLINE_INTERVAL_1MINUTE  # Candle timeframe
```

### Buy Signal Logic
```python
if percent_change <= -buy_threshold and (rsi < 30) and (ema1 > ema200):
```

### Backtesting Scripts
- `cryptoBacktest-EMA-RSI-exitPerPercent.py`: Backtest with percentage-based exits
- `cryptoBacktest-EMA-exitPerEmaCross.py`: Backtest with EMA crossover exits

## File Structure

- **Trading State Files**: `order_id_{SYMBOL}.txt` - Contains buy price and quantity for open positions
- **Output Files**: `output_{SYMBOL}.txt` - Price history logs for each symbol
- **Status Tracking**: `status.json` - Overall profit/loss tracking across all symbols
- **Configuration**: Trading parameters are hardcoded in the main script

## Technical Indicators

### RSI Calculation
- 14-period RSI using 50-minute lookback window
- Buy trigger: RSI < 30 (oversold condition)

### EMA Analysis
- EMA1, EMA9, EMA26, EMA200 calculated from 300-minute window
- Trend confirmation: EMA1 > EMA200 for bullish bias

## Switching to Different Timeframes

To switch from 1-minute to hourly candles:
1. Change `kline_interval = Client.KLINE_INTERVAL_1HOUR`
2. Update lookback windows in `calculate_rsi()` and `calculate_emas()`:
   - Replace `"50 minutes ago UTC"` with `"400 hours ago UTC"`
   - Replace `"300 minutes ago UTC"` with `"400 hours ago UTC"`

## Dependencies

Core libraries from `requirements.txt`:
- `python-binance` - Binance API client
- `pandas` - Data analysis for technical indicators
- `tenacity` - Retry mechanisms for API calls
- `termcolor` - Colored terminal output
- `slack_sdk` - Slack notifications (TODO feature)

## Security Notes

- API keys must be stored in `_secrets/__init__.py` (excluded from git)
- The bot uses market orders for immediate execution
- Position sizing is controlled by `usd_amount` constant
- Emergency sell-all function available via 'x' command