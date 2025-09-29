# CryptoBandit3 - Advanced Cryptocurrency Trading Bot Suite

Automated trading system for Binance with multiple strategies, professional risk management, and comprehensive backtesting capabilities.

## 🤖 Trading Bots

### RSI/EMA Strategy
- **`cryptoBandit3.py`** - Original RSI + EMA strategy (RSI < 30, EMA1 > EMA200)
- Trades 14 pairs with 1-minute candles
- P&L tracking in `status.json`

### Candlestick Pattern Bots
- **`cryptoBanditCandles.py`** - Pattern detection only (analysis mode)
- **`cBc-live.py`** - Basic live trading with patterns
- **`cBc-live-advanced.py`** - Graduated exits, stop-loss waits
- **`cBc-live-pro.py`** - **NEW: DRY_RUN mode**, BTC correlation filter
- **`cBc-live-ema-4h.py`** - 4-hour EMA-enhanced patterns
- **`cBc-trader-pro.py`** - **Most Advanced**: Trend lines, S/R levels, progressive trailing stops, **DRY_RUN mode**
- **`cBc-channel-trader.py`** - Channel-based strategy

### Key Features
- **DRY_RUN Mode**: Test strategies without real money (in pro versions)
- **Pattern Recognition**: Hammer, Engulfing, Morning/Evening Star, Doji, Shooting Star
- **Risk Management**: 1% take profit, 10% stop loss
- **Smart Exits**: Progressive trailing stops, partial profit-taking
- **BTC Correlation**: Alt coins only trade when BTC is stable

## 📊 Backtesting

### Tools
- **`backtesting/backtest_candles.py`** - Full candlestick pattern backtesting
- **`backtesting/backtest_candles_optimized.py`** - Optimized version with caching
- **`backtesting/create_dashboard.py`** - Generate interactive HTML reports
- **`cryptoBacktest-EMA-RSI-exitPerPercent.py`** - RSI/EMA with percentage exits
- **`cryptoBacktest-EMA-exitPerEmaCross.py`** - EMA crossover exits

### Latest Results
- **Best Timeframe**: 4-hour candles (72.8% win rate)
- **Total Profit**: $362.56 USDT (355 trades)
- **13 Pairs**: BTC, ETH, BNB, ADA, XRP, DOGE, SOL, PNUT, PEPE, SHIB, XLM, LINK, IOTA

## 🚀 Quick Start

### Setup
```bash
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

### Configuration
Create `_secrets/__init__.py`:
```python
api_key = 'your_binance_api_key'
secret_key = 'your_binance_secret_key'
```

### Running Bots

**Test Mode (No Real Money):**
```bash
# Set DRY_RUN = True in the bot file, then:
python3 cBc-trader-pro.py
python3 cBc-live-pro.py
```

**Live Trading:**
```bash
# Set DRY_RUN = False (WARNING: Real money!)
python3 cBc-trader-pro.py
```

**Check P&L:**
```bash
python3 cBc-trader-pro.py --check-pnl
python3 cBc-trader-pro.py --reset-pnl  # Reset if needed
```

### Emergency Controls
- **Manual Sell**: Type 'x' + ENTER → confirm with 'YES'
- **Graceful Shutdown**: Ctrl+C

## 📁 File Structure
```
├── Trading Bots (*.py)
├── backtesting/          # Backtesting tools & results
├── orders*/              # Position files per strategy
├── outputs*/             # Trading logs
├── trader_pro/           # Professional trader data
├── status*.json          # P&L tracking files
└── _secrets/             # API credentials (git-ignored)
```

## ⚠️ Risk Warning
- Start with DRY_RUN mode to understand the strategy
- Test with small amounts first
- Markets are volatile - losses are possible
- Never invest more than you can afford to lose

## 📈 Performance Tracking
Each bot variant maintains separate P&L tracking:
- Dry mode: `*_dry.json` files
- Live mode: `*_live.json` files

Position state persists across restarts - the bot will resume managing open positions.

---
*Built with Python, Binance API, and battle-tested trading strategies*