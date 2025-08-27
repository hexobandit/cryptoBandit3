# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an advanced cryptocurrency trading bot suite with multiple strategies: RSI/EMA technical analysis and candlestick pattern recognition. The system includes live trading bots, comprehensive backtesting, and interactive performance dashboards.

## Trading Strategies

### 1. RSI + EMA Strategy (`cryptoBandit3.py`)
- **Logic**: `percent_change <= -buy_threshold and (rsi < 30) and (ema1 > ema200)`
- **Files**: `orders/`, `outputs/`, `status.json`
- **Timeframe**: 1-minute candles with 10-minute check intervals

### 2. Candlestick Pattern Strategy
- **Signal Detection**: `cryptoBanditCandles.py` (analysis only)
- **Live Trading**: `cBc-live.py` (1-minute candles, real trades)
- **Patterns**: Hammer, Bullish/Bearish Engulfing, Morning/Evening Star, Doji, Shooting Star
- **Files**: `orders_candles_1m/`, `outputs_candles_1m/`, `status_candles_1m.json`
- **Exit Logic**: 1% take profit OR 10% stop loss OR bearish patterns

## Setup and Installation

```bash
# Environment setup
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt

# API credentials in _secrets/__init__.py:
api_key = 'your_binance_api_key'
secret_key = 'your_binance_secret_key'
```

## Common Commands

```bash
# RSI/EMA live trading
python3 cryptoBandit3.py

# Candlestick pattern detection (signals only)
python3 cryptoBanditCandles.py

# Candlestick live trading (real money)
python3 cBc-live.py

# Run comprehensive backtesting
cd backtesting && python3 backtest_candles.py

# Generate performance dashboards
cd backtesting && python3 create_dashboard.py
```

## Backtesting Architecture

### Core Backtesting (`backtesting/backtest_candles.py`)
- **Historical Data**: 365 days across 9 timeframes (1m to 3d)
- **Symbols**: 13 cryptocurrency pairs
- **Strategy**: Long-only candlestick pattern recognition
- **Risk Management**: 1% take profit, 10% stop loss
- **Fee Calculation**: 0.1% per trade (0.2% total per round trip)

### Exit Conditions (When Backtesting Sells)
1. **Take Profit**: +1% price increase from entry
2. **Stop Loss**: -10% price decrease from entry  
3. **End of Data**: Close remaining positions at final candle

### Dashboard Generation (`backtesting/create_dashboard.py`)
- **Dependencies**: `pip install plotly seaborn matplotlib`
- **Output**: 6 interactive HTML dashboards + comprehensive report
- **Data Source**: `backtest_results.json`

## File Structure

```
├── cryptoBandit3.py              # RSI/EMA strategy
├── cryptoBanditCandles.py        # Pattern detection only
├── cBc-live.py                   # Live candlestick trading
├── backtesting/
│   ├── backtest_candles.py       # Comprehensive backtesting
│   ├── create_dashboard.py       # Dashboard generator
│   └── *.html                    # Generated dashboards
├── orders/                       # RSI strategy positions
├── orders_candles_1m/            # Live trading positions
├── _secrets/__init__.py          # API credentials (git-ignored)
└── status*.json                  # P&L tracking files
```

## Pattern Detection Logic

### Bullish Patterns (Buy Signals)
- **Hammer**: `lower_shadow >= 2 * body AND upper_shadow <= body * 0.5`
- **Bullish Engulfing**: Current green candle engulfs previous red candle
- **Morning Star**: 3-candle bullish reversal pattern
- **Doji**: `body <= range * 0.1` (indecision candle)

### Bearish Patterns (Sell Signals)
- **Shooting Star**: `upper_shadow >= 2 * body AND lower_shadow <= body * 0.5`
- **Bearish Engulfing**: Current red candle engulfs previous green candle
- **Evening Star**: 3-candle bearish reversal pattern

## Live Trading Configuration

### Candlestick Live Trading (`cBc-live.py`)
```python
usd_amount = 150                    # USDT per trade
take_profit_percent = 0.01          # 1% profit target
stop_loss_percent = 0.10            # 10% stop loss
check_interval_minutes = 1          # Check every minute
symbols = [13 cryptocurrency pairs] # All backtest-validated pairs
```

### Emergency Controls
- **Manual Sell**: Type 'x' + ENTER → confirm with 'YES'
- **Graceful Shutdown**: Ctrl+C
- **State Persistence**: Positions survive bot restarts

## Performance Metrics

Recent backtesting results show:
- **Total Profit**: $2,908.37 USDT (10,995 trades)
- **Win Rate**: 66.8% across all timeframes
- **Best Timeframe**: 3-day candles
- **Fee-Adjusted Returns**: All P&L includes 0.2% trading fees

## Key Architecture Concepts

### Position Management
- **State Files**: CSV format with `buy_price,quantity,entry_date,pattern`
- **One Position Per Symbol**: Maximum one open position per cryptocurrency
- **Fee Integration**: All profit calculations include Binance trading fees
- **Pattern Tracking**: Entry patterns recorded for performance analysis

### Data Pipeline
1. **Live Trading**: Binance WebSocket → Pattern Analysis → Trade Execution
2. **Backtesting**: Historical Klines → Pattern Detection → Simulated Trading
3. **Dashboards**: JSON Results → Plotly Visualizations → HTML Reports

### Risk Management
- Stop losses prevent catastrophic losses
- Take profits lock in gains systematically  
- Position limits prevent overexposure
- Fee calculations ensure realistic backtesting

## Development Notes

When modifying trading parameters:
- Test changes in backtesting first
- Update corresponding live trading constants
- Verify fee calculations remain accurate
- Regenerate dashboards to validate performance impact