# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Advanced cryptocurrency trading bot suite for Binance with multiple evolution stages: RSI/EMA strategies, candlestick pattern recognition, VWAP-based trading, and professional-grade trend analysis. The codebase includes 10+ bot variants with comprehensive P&L tracking and separate dry/live modes.

## Bot Architecture Evolution

### Strategy Progression
1. **Basic RSI/EMA**: `cryptoBandit3.py` - RSI < 30 + EMA crossover
2. **Pattern Recognition**: `cBc-live.py` - Candlestick patterns (hammer, engulfing, doji)
3. **Advanced Exits**: `cBc-live-advanced.py` - Graduated exits, stop-loss waits
4. **BTC Correlation**: `cBc-live-pro.py` - Alt trading filtered by BTC stability
5. **Professional Trading**: `cBc-trader-pro.py` - Trend lines, S/R levels, progressive trailing stops
6. **VWAP Strategy**: `cBc-vwap-trader.py` - VWAP crosses with ADX regime detection
7. **Channel Trading**: `cBc-channel-trader.py` - Price channel breakout strategy

### Key Architectural Patterns
- **State Persistence**: JSON files survive restarts (`position_{symbol}_{mode}.json`)
- **P&L Tracking**: Separate dry (`*_dry.json`) vs live (`*_live.json`) modes
- **One Position Rule**: Maximum one open position per symbol
- **Fee Integration**: All calculations include 0.1% Binance fees
- **Graduated Exits**: TP1 (50% at 1%), TP2 (remaining at 2-3%)

## Common Commands

### Running Trading Bots
```bash
# Activate virtual environment
source ./venv/bin/activate

# Professional trader (check DRY_RUN setting in file)
python3 cBc-trader-pro.py              # Main execution
python3 cBc-trader-pro.py --check-pnl  # View cumulative P&L breakdown
python3 cBc-trader-pro.py --reset-pnl  # Reset P&L tracking

# VWAP trader
python3 cBc-vwap-trader.py

# Live pattern trading
python3 cBc-live-pro.py     # BTC correlation filter
python3 cBc-live.py          # Basic patterns
python3 cryptoBandit3.py    # Original RSI/EMA
```

### Analysis and Reporting
```bash
# Generate HTML performance reports
python3 analyze_trades.py --mode live      # For cBc-trader-pro
python3 analyze_trades.py --mode dry       # Dry run analysis
python3 analyze_vwap_trades.py --mode live # For VWAP trader
```

### Emergency Controls
```bash
# Manual sell: Type 'x' + ENTER → confirm 'YES' (in running bot)
# Graceful shutdown: Ctrl+C (saves state before exit)
```

## Critical Code Patterns

### DRY_RUN Mode Implementation
```python
# At file top (lines 20-26 in most bots)
DRY_RUN = True  # Set to False for LIVE trading with real money

# In buy/sell functions
if DRY_RUN:
    ticker = client.get_symbol_ticker(symbol=symbol)
    # Simulate order with current price
else:
    order = client.order_market_buy(...)  # Real API call
```

### Position State Structure
```python
{
    "position_is_open": bool,
    "entry_price": float,
    "quantity": float,
    "remaining_quantity": float,    # For partial exits
    "entry_time": str,
    "entry_pattern": str,           # Strategy trigger
    "entry_reason": str,            # Detailed entry logic
    "stop_loss": float,
    "take_profit_1": float,
    "take_profit_2": float,
    "trailing_stop_price": float,   # Dynamic trailing
    "realized_profit": float,       # Position P&L
    "total_realized_pnl": float     # All-time cumulative
}
```

### P&L Tracking Architecture

#### Simple Tracking (`cBc-live-pro.py`)
```python
# Single cumulative counter per symbol
overall_status[symbol] += profit_loss
# Stored in status_pro_{dry|live}.json
```

#### Advanced Tracking (`cBc-trader-pro.py`)
```python
# Separated: current vs cumulative
position["realized_profit"]     # Current position P&L
position["total_realized_pnl"]  # All-time P&L
# Files: trader_pro/position_{symbol}_{mode}.json
#        trader_pro/session_pnl_{mode}.json
#        trader_pro/trades_{symbol}_{mode}.json
```

## Trading Strategy Parameters

### cBc-trader-pro.py Entry Strategies
1. **Support Bounce**: Price within 0.3% of support, ≥2 touches
2. **Trend Breakout**: Break above trend +0.2%, ≥3 touches
3. **RSI Oversold**: RSI < 25 with price stabilization
4. **Volume Spike**: 2.5x average volume on green candle
5. **Liquidity Hunt**: Stop hunt recovery pattern

### cBc-vwap-trader.py Strategy
- **Entry**: Price crosses above VWAP with volume confirmation
- **ADX Filter**: ADX < 25 = ranging (tighter stops), > 30 = trending
- **Adaptive Stops**: 0.5% ranging, 1% trending
- **Progressive Trail**: <1%: 0.8% | 1-2%: 0.5% | >2%: 0.3%

### Exit Management (Common)
- **TP1**: 1-1.5% profit → Sell 50%, move stop to breakeven
- **TP2**: 2-3% profit → Sell remaining
- **Trailing Stop**: Activates at 1% profit
- **BTC Filter**: No alt trades when BTC dumps >1.5%/hr

## File Organization

### Position & Trade Files
```
# Per-bot organization
cBc-live.py         → orders_candles_1m/, status_candles_1m.json
cBc-live-pro.py     → orders_pro/*_{dry|live}.txt, status_pro_{dry|live}.json
cBc-trader-pro.py   → trader_pro/*_{dry|live}.json
cBc-vwap-trader.py  → trader_vwap/*_{dry|live}.json
cryptoBandit3.py    → orders/, status.json
```

### Log Files
```
outputs*/output_{symbol}_{mode}.txt  # Real-time trading logs
trader_pro/trades_{symbol}_{mode}.json  # Trade history
```

## Development Workflow

### Testing New Strategies
1. **Always start with DRY_RUN = True**
2. Run for 24-48 hours minimum
3. Generate analysis report: `python3 analyze_trades.py --mode dry`
4. Review win rate, max drawdown, profit factor
5. Only then consider live trading with small amounts

### Modifying Trading Logic
- **Entry threshold**: `min_entry_score = 70` (0-100 scale)
- **Risk/Reward**: `min_rr_ratio = 2.0`
- **Partial exits**: `TP1_SELL_PERCENT = 0.5`
- **Fee calculation**: Always `(buy + sell) * quantity * 0.001`

## Symbol Configuration
Standard 14-pair portfolio:
```python
symbols = ["BTCUSDC", "ETHUSDC", "BNBUSDC", "ADAUSDC", "XRPUSDC",
          "DOGEUSDC", "SOLUSDC", "PNUTUSDC", "PEPEUSDC", "SHIBUSDC",
          "XLMUSDC", "LINKUSDC", "IOTAUSDC", "ENAUSDC"]
```

## API Configuration
Create `_secrets/__init__.py`:
```python
api_key = 'your_binance_api_key'
secret_key = 'your_binance_secret_key'
```

⚠️ **Security Note**: Use environment variables in production. Never commit API keys.