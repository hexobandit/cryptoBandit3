# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Advanced cryptocurrency trading bot suite for Binance with multiple evolution stages: RSI/EMA strategies, candlestick pattern recognition, and professional-grade trend analysis. The codebase includes 7+ bot variants, comprehensive backtesting, and sophisticated P&L tracking with separate dry/live modes.

## Bot Architecture Evolution

### Basic → Advanced → Professional
1. **Basic**: `cryptoBandit3.py` (RSI/EMA), `cBc-live.py` (patterns)
2. **Advanced**: `cBc-live-advanced.py` (graduated exits), `cBc-live-pro.py` (BTC correlation)
3. **Professional**: `cBc-trader-pro.py` (trend lines, S/R levels, trailing stops)

### Key Architectural Decisions
- **State Persistence**: All positions stored in JSON/CSV files, survive restarts
- **P&L Tracking**: Separate files for dry (`*_dry.json`) vs live (`*_live.json`) modes
- **One Position Rule**: Maximum one open position per symbol
- **Fee Integration**: All P&L calculations include 0.1% Binance trading fees

## Common Commands

```bash
# Professional trader with DRY_RUN mode
python3 cBc-trader-pro.py              # Check DRY_RUN setting in file
python3 cBc-trader-pro.py --check-pnl  # View cumulative P&L breakdown
python3 cBc-trader-pro.py --reset-pnl  # Reset P&L tracking

# Live trading bots
python3 cBc-live-pro.py     # Enhanced with BTC correlation filter
python3 cryptoBandit3.py    # Original RSI/EMA strategy

# Backtesting
cd backtesting && python3 backtest_candles_optimized.py  # With caching
cd backtesting && python3 create_dashboard.py            # Generate HTML reports
```

## Critical Code Patterns

### DRY_RUN Mode Implementation
```python
# At file top (lines 20-26)
DRY_RUN = True  # Set to False for LIVE trading with real money

# In buy/sell functions
if DRY_RUN:
    # Simulate with current market price
    ticker = client.get_symbol_ticker(symbol=symbol)
    # Create mock order response
else:
    # Execute real Binance API call
    order = client.order_market_buy(...)
```

### P&L Tracking Architecture

#### Simple Approach (`cBc-live-pro.py`)
- Single cumulative counter: `overall_status[symbol] += profit_or_loss`
- Stored in `status_pro_dry.json` or `status_pro_live.json`

#### Advanced Approach (`cBc-trader-pro.py`)
- Separated tracking: `total_realized_pnl` (cumulative) vs `realized_profit` (current position)
- Position files: `trader_pro/position_{symbol}_{mode}.json`
- Session file: `trader_pro/session_pnl_{mode}.json`

### Position State Management
```python
# Standard position state structure
{
    "position_is_open": bool,
    "entry_price": float,
    "quantity": float,
    "remaining_quantity": float,  # For partial exits
    "entry_date": str,
    "entry_pattern": str,          # Strategy that triggered entry
    "stop_loss": float,
    "take_profit_1": float,
    "realized_profit": float,      # Current position realized P&L
    "total_realized_pnl": float    # Cumulative all-time P&L
}
```

## Trading Strategy Parameters

### Entry Strategies (cBc-trader-pro.py)
1. **Support Bounce**: Price within 0.3% of support with ≥2 touches
2. **Trend Breakout**: Break above trend line +0.2% with ≥3 touches  
3. **RSI Oversold**: RSI < 25 with price stabilization
4. **Volume Spike**: 2.5x average volume on green candle
5. **Liquidity Hunt**: Stop hunt recovery pattern

### Exit Management
- **TP1**: 1% profit → Sell 50%, move stop to breakeven
- **Progressive Trail**: <1%: 0.8% | 1-2%: 0.5% | >2%: 0.3%
- **BTC Filter**: No alt trades when BTC dumps >1.5%/hr or RSI<30

## File Organization

### Position & P&L Files by Bot
```
cBc-live.py         → orders_candles_1m/, status_candles_1m.json
cBc-live-pro.py     → orders_pro/*_{dry|live}.txt, status_pro_{dry|live}.json
cBc-trader-pro.py   → trader_pro/*_{dry|live}.json
cryptoBandit3.py    → orders/, status.json
```

### Backtesting Data Flow
1. **Data Cache**: `backtesting/data_cache/{symbol}_{timeframe}_data.json`
2. **Results**: `backtesting/backtest_results_{strategy}.json`
3. **Dashboards**: `backtesting/*.html` (6 interactive visualizations)

## Emergency Procedures

### Manual Position Management
- **Emergency Sell**: Type 'x' + ENTER → confirm 'YES'
- **Graceful Shutdown**: Ctrl+C (saves state before exit)
- **Check Positions**: Review JSON/CSV files in respective orders directories

### P&L Discrepancy Resolution
If P&L seems incorrect:
1. Check cumulative: `python3 {bot_name}.py --check-pnl`
2. Compare position files vs trade records
3. Reset if needed: `python3 {bot_name}.py --reset-pnl`

## Development Workflow

### Testing New Strategies
1. Set `DRY_RUN = True` in bot file
2. Run backtesting first: `cd backtesting && python3 backtest_candles_optimized.py`
3. Monitor dry run for 24-48 hours
4. Review P&L and win rates
5. Only then set `DRY_RUN = False` for live trading

### Modifying Entry/Exit Logic
- Entry score threshold: `min_entry_score = 70` (0-100 scale)
- Risk/Reward minimum: `min_rr_ratio = 2.0`
- Partial exit percentages in `graduated_exits` or `TP1_SELL_PERCENT`
- Always preserve fee calculations: `fees = (buy_price * qty + sell_price * qty) * 0.001`

## Symbol Configuration
All bots trade these 14 pairs:
```python
symbols = ["BTCUSDC", "ETHUSDC", "BNBUSDC", "ADAUSDC", "XRPUSDC", 
          "DOGEUSDC", "SOLUSDC", "PNUTUSDC", "PEPEUSDC", "SHIBUSDC", 
          "XLMUSDC", "LINKUSDC", "IOTAUSDC", "ENAUSDC"]
```