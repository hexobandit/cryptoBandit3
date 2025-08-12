# cryptoBandit3
**Advanced Cryptocurrency Trading Bot Suite with Multiple Strategies**

Automated cryptocurrency trading on Binance using RSI/EMA analysis and candlestick pattern recognition. Multiple trading strategies with comprehensive backtesting and live trading capabilities.

## 🚀 Trading Strategies

### 1. RSI + EMA Strategy (`cryptoBandit3.py`)
Original strategy using Relative Strength Index and Exponential Moving Averages.

**Buy Logic:** `percent_change <= -buy_threshold and (rsi < 30) and (ema1 > ema200)`

**Key Features:**
- RSI-based buy signals (RSI < 30)
- EMA trend confirmation (EMA1 > EMA200)
- Custom price drop thresholds
- Real-time performance tracking
- Emergency sell command: "Type 'x' + ENTER"

### 2. Candlestick Pattern Strategy (`cryptoBanditCandles.py`, `cBc-live.py`)
Advanced pattern recognition trading using Japanese candlestick formations.

**Supported Patterns:**
- **Bullish:** Hammer, Bullish Engulfing, Morning Star, Doji
- **Bearish:** Shooting Star, Bearish Engulfing, Evening Star

**Live Trading:** `cBc-live.py`
- 1-minute candle analysis
- 13 cryptocurrency pairs
- 1% take profit, 10% stop loss
- Real-time pattern detection

## 📊 Backtesting Suite

### Comprehensive Strategy Testing (`backtesting/backtest_candles.py`)
- **Historical Data:** 1 year of market data
- **Multiple Timeframes:** 1m, 5m, 15m, 30m, 1h, 4h, 12h, 1d, 3d
- **Performance Metrics:** Win rate, profit/loss, trade frequency
- **Fee Calculation:** 0.1% per trade (0.2% total)

**Recent Backtest Results:**
```
📊 FINAL BACKTEST SUMMARY
================================================================================
4h     | Trades: 125  | Win Rate: 72.8% | P&L: +149.16 USDT (BEST)
1h     | Trades: 158  | Win Rate: 71.5% | P&L: +135.01 USDT  
1d     | Trades: 72   | Win Rate: 68.1% | P&L: +78.39 USDT
--------------------------------------------------------------------------------
TOTAL  | Trades: 355  | Win Rate: 71.3% | P&L: +362.56 USDT
```

## 🔧 Setup & Installation

### Prerequisites
```bash
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

### API Configuration
Create `_secrets/__init__.py`:
```python
api_key = 'your_binance_api_key'
secret_key = 'your_binance_secret_key'
```

### Required Dependencies
- `python-binance` - Binance API client
- `pandas` - Data analysis and technical indicators
- `tenacity` - Retry mechanisms for API calls
- `termcolor` - Colored terminal output
- `requests` - HTTP requests

## 🎮 Usage

### RSI/EMA Strategy
```bash
python3 cryptoBandit3.py
```

### Candlestick Pattern Analysis (Signal Detection Only)
```bash
python3 cryptoBanditCandles.py
```

### Live Candlestick Trading
```bash
python3 cBc-live.py
```

### Run Backtesting
```bash
cd backtesting
python3 backtest_candles.py
```

## 📁 Project Structure

```
cryptoBandit3/
├── cryptoBandit3.py              # RSI/EMA live trading bot
├── cryptoBanditCandles.py        # Candlestick pattern detector
├── cBc-live.py                   # Live candlestick trading bot
├── backtesting/
│   └── backtest_candles.py       # Comprehensive backtesting suite
├── orders/                       # RSI strategy order files
├── outputs/                      # RSI strategy logs
├── orders_candles/               # Candlestick detector orders
├── outputs_candles/              # Candlestick detector logs
├── orders_candles_1m/            # Live trading orders (1m)
├── outputs_candles_1m/           # Live trading logs (1m)
├── status.json                   # RSI strategy P&L tracking
├── status_candles.json           # Candlestick detector P&L
├── status_candles_1m.json        # Live trading P&L
└── CLAUDE.md                     # Development guidance
```

## ⚙️ Configuration

### RSI/EMA Strategy Constants
```python
usd_amount = 100                  # USDT per trade
buy_threshold = 0.01              # 1% price drop required
sell_threshold = 0.02             # 2% profit target  
stop_loss_threshold = 0.8         # 80% stop loss
kline_interval = KLINE_INTERVAL_1MINUTE
```

### Candlestick Strategy Constants
```python
usd_amount = 20                   # USDT per trade
take_profit_percent = 0.01        # 1% take profit
stop_loss_percent = 0.10          # 10% stop loss
kline_interval = KLINE_INTERVAL_1MINUTE
check_interval_minutes = 1        # Check every minute
```

## 📈 Supported Cryptocurrencies

**13 Trading Pairs:**
- BTCUSDC, ETHUSDC, BNBUSDC
- ADAUSDC, XRPUSDC, DOGEUSDC  
- SOLUSDC, PNUTUSDC, PEPEUSDC
- SHIBUSDC, XLMUSDC, LINKUSDC
- IOTAUSDC

## 🛡️ Safety Features

- **Emergency Sell:** Press 'x' + ENTER to sell all positions
- **Graceful Shutdown:** Ctrl+C handling
- **Persistent State:** Survives restarts and crashes
- **Fee Calculation:** All P&L includes trading fees
- **Position Limits:** One position per symbol
- **API Retry Logic:** Handles network issues

## 🎯 Trading Performance

The candlestick pattern strategy has demonstrated:
- **71.3% Win Rate** across all timeframes
- **+362.56 USDT Profit** in backtesting (1 year)
- **4-Hour Timeframe** shows best performance
- **Consistent Profitability** across multiple crypto pairs

## ⚠️ Risk Disclaimer

This software is for educational purposes. Cryptocurrency trading involves substantial risk of loss. Never invest more than you can afford to lose. Past performance does not guarantee future results.

## 📞 Support

For issues and feedback, check the CLAUDE.md file for development guidance and common troubleshooting steps.