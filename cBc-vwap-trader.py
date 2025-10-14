#!/usr/bin/env python3
"""
VWAP Trading Bot with Market Regime Detection
==============================================
Long-only strategy based on VWAP crosses with ADX filtering for sideways markets.
Features adaptive stops and volume confirmation for improved performance.
"""

import os
import json
import time
import math
import hmac
import hashlib
import datetime
import signal
import sys
import threading
import statistics
from binance.client import Client
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
from termcolor import colored
sys.path.append('../')
from _secrets import api_key, secret_key

# ==================== CONFIGURATION ====================
# TRADING MODE - IMPORTANT: Set to True to prevent real money losses!
DRY_RUN = False  # Set to False for LIVE trading with real money

# WARNING: With DRY_RUN = False, this bot will execute REAL trades!
# Only set to False when you're confident in the strategy
# ========================================================

# Trading symbols
symbols = [
    "BTCUSDC", "ETHUSDC", "BNBUSDC", "ADAUSDC", "XRPUSDC", 
    "DOGEUSDC", "SOLUSDC", "PNUTUSDC", "PEPEUSDC", "SHIBUSDC", 
    "XLMUSDC", "LINKUSDC", "IOTAUSDC", "ENAUSDC"
]

# Data Classes
@dataclass
class MarketRegime:
    """Market regime classification"""
    adx: float
    trend: str  # 'trending', 'ranging', 'transitioning'
    volatility: str  # 'high', 'normal', 'low'
    strength: float  # 0-100 trend strength

@dataclass
class VWAPData:
    """VWAP calculation data"""
    vwap: float
    upper_band: float  # VWAP + 1 std dev
    lower_band: float  # VWAP - 1 std dev
    cumulative_volume: float
    cumulative_pv: float  # Price * Volume cumulative

# Trading parameters
BASE_POSITION_SIZE = 100  # USDT per trade
MIN_ADX_FOR_TREND = 25    # Below this = ranging market
STRONG_TREND_ADX = 30      # Above this = strong trend
VOLUME_MULTIPLIER = 2.0    # Minimum volume spike for entry
CHECK_INTERVAL_MINUTES = 15

# Risk management - Adaptive based on market regime
STOP_LOSS_TRENDING = 0.01      # 1% stop loss in trending markets
STOP_LOSS_RANGING = 0.005       # 0.5% tighter stop in ranging markets
TAKE_PROFIT_1 = 0.015           # 1.5% first target (sell 50%)
TAKE_PROFIT_2 = 0.03            # 3% second target
TP1_SELL_PERCENT = 0.5          # Sell 50% at TP1

# Trailing stop configuration
TRAILING_ACTIVATION = 0.01      # Activate after 1% profit
TRAILING_DISTANCE_INITIAL = 0.008  # 0.8% initial trailing
TRAILING_DISTANCE_PROFIT = 0.005   # 0.5% when in good profit
TRAILING_DISTANCE_STRONG = 0.003   # 0.3% when >2% profit

# VWAP parameters
VWAP_LOOKBACK = 96    # 24 hours for 15-minute candles
ADX_PERIOD = 14       # Standard ADX period
VOLUME_MA_PERIOD = 20 # Volume moving average period

# File management
mode_suffix = "_dry" if DRY_RUN else "_live"

# Initialize Binance client
client = Client(api_key, secret_key)

# Global shutdown flag
shutdown = False

def handle_exit(sig, frame):
    """Handle graceful shutdown"""
    global shutdown
    print("\n🔻 Graceful shutdown requested...")
    shutdown = True

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

# Trader state management
trader_state = {
    symbol: {
        # Position tracking
        "position_is_open": False,
        "entry_price": None,
        "quantity": None,
        "entry_time": None,
        "stop_loss": None,
        "take_profit_1": None,
        "take_profit_2": None,
        "entry_reason": None,
        "market_regime": None,  # Market regime at entry
        
        # Trailing stop management
        "tp1_hit": False,
        "highest_price": None,
        "trailing_stop": None,
        "trailing_distance": None,
        "remaining_quantity": None,
        "realized_profit": 0,
        
        # VWAP data
        "vwap_data": None,
        "last_vwap_cross": None,  # 'above' or 'below'
        "cross_timestamp": None,
        
        # Market analysis
        "current_adx": 0,
        "volume_ma": 0,
        "last_volume": 0,
        
        # P&L tracking
        "total_realized_pnl": 0,
        
        # File paths
        "filename_position": f"trader_vwap/position_{symbol}{mode_suffix}.json",
        "filename_trades": f"trader_vwap/trades_{symbol}{mode_suffix}.json",
    }
    for symbol in symbols
}

# Create directories
os.makedirs("trader_vwap", exist_ok=True)

def calculate_vwap(candles: list) -> VWAPData:
    """
    Calculate VWAP with bands
    
    VWAP = Σ(Price × Volume) / Σ(Volume)
    """
    if not candles:
        return None
        
    cumulative_pv = 0
    cumulative_volume = 0
    prices = []
    
    for candle in candles:
        # Typical price = (High + Low + Close) / 3
        high = float(candle[2])
        low = float(candle[3])
        close = float(candle[4])
        volume = float(candle[5])
        
        typical_price = (high + low + close) / 3
        prices.append(typical_price)
        
        cumulative_pv += typical_price * volume
        cumulative_volume += volume
    
    if cumulative_volume == 0:
        return None
    
    vwap = cumulative_pv / cumulative_volume
    
    # Calculate standard deviation for bands
    squared_diffs = [(price - vwap) ** 2 for price in prices]
    std_dev = math.sqrt(sum(squared_diffs) / len(squared_diffs))
    
    return VWAPData(
        vwap=vwap,
        upper_band=vwap + std_dev,
        lower_band=vwap - std_dev,
        cumulative_volume=cumulative_volume,
        cumulative_pv=cumulative_pv
    )

def calculate_adx(candles: list, period: int = 14) -> float:
    """
    Calculate Average Directional Index (ADX)
    ADX > 25: Trending market
    ADX < 25: Ranging/sideways market
    """
    if len(candles) < period * 2:
        return 0
    
    highs = [float(c[2]) for c in candles]
    lows = [float(c[3]) for c in candles]
    closes = [float(c[4]) for c in candles]
    
    # Calculate True Range
    tr_list = []
    for i in range(1, len(candles)):
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i-1])
        low_close = abs(lows[i] - closes[i-1])
        tr = max(high_low, high_close, low_close)
        tr_list.append(tr)
    
    # Calculate directional movement
    plus_dm = []
    minus_dm = []
    for i in range(1, len(candles)):
        high_diff = highs[i] - highs[i-1]
        low_diff = lows[i-1] - lows[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm.append(high_diff)
        else:
            plus_dm.append(0)
            
        if low_diff > high_diff and low_diff > 0:
            minus_dm.append(low_diff)
        else:
            minus_dm.append(0)
    
    # Smooth the values
    atr = sum(tr_list[-period:]) / period
    plus_di = 100 * sum(plus_dm[-period:]) / (atr * period) if atr > 0 else 0
    minus_di = 100 * sum(minus_dm[-period:]) / (atr * period) if atr > 0 else 0
    
    # Calculate ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
    
    return dx

def get_market_regime(symbol: str, candles: list) -> MarketRegime:
    """Determine current market regime using ADX and volatility"""
    adx = calculate_adx(candles)
    
    # Calculate recent volatility
    recent_candles = candles[-20:] if len(candles) >= 20 else candles
    returns = []
    for i in range(1, len(recent_candles)):
        close_prev = float(recent_candles[i-1][4])
        close_curr = float(recent_candles[i][4])
        returns.append((close_curr - close_prev) / close_prev)
    
    volatility = statistics.stdev(returns) * math.sqrt(96) * 100 if len(returns) > 1 else 0  # Annualized vol
    
    # Classify market regime
    if adx >= STRONG_TREND_ADX:
        trend = 'trending'
        strength = min(100, adx * 2)
    elif adx >= MIN_ADX_FOR_TREND:
        trend = 'transitioning'
        strength = adx * 2
    else:
        trend = 'ranging'
        strength = max(0, 50 - adx)
    
    # Classify volatility
    if volatility > 50:
        vol_class = 'high'
    elif volatility > 30:
        vol_class = 'normal'
    else:
        vol_class = 'low'
    
    return MarketRegime(
        adx=adx,
        trend=trend,
        volatility=vol_class,
        strength=strength
    )

def check_vwap_cross(symbol: str, current_price: float, vwap_data: VWAPData) -> Optional[str]:
    """
    Check for VWAP crossover
    Returns: 'bullish' for upward cross, None otherwise
    """
    state = trader_state[symbol]
    
    # Determine current position relative to VWAP
    current_position = 'above' if current_price > vwap_data.vwap else 'below'
    
    # Check for bullish crossover
    if state["last_vwap_cross"] == 'below' and current_position == 'above':
        # Price crossed above VWAP
        return 'bullish'
    
    # Update state for next check
    state["last_vwap_cross"] = current_position
    
    return None

def calculate_position_size(market_regime: MarketRegime) -> float:
    """
    Adjust position size based on market regime
    Smaller positions in ranging markets to reduce risk
    """
    if market_regime.trend == 'trending':
        return BASE_POSITION_SIZE
    elif market_regime.trend == 'transitioning':
        return BASE_POSITION_SIZE * 0.75
    else:  # ranging
        return BASE_POSITION_SIZE * 0.5

def get_adaptive_stop_loss(market_regime: MarketRegime) -> float:
    """Get stop loss percentage based on market regime"""
    if market_regime.trend == 'ranging':
        return STOP_LOSS_RANGING
    else:
        return STOP_LOSS_TRENDING

def analyze_symbol(symbol: str):
    """
    Main analysis function for each symbol
    Checks VWAP cross, ADX filter, and volume confirmation
    """
    state = trader_state[symbol]
    
    try:
        # Get 15-minute candles for VWAP calculation
        candles = client.get_klines(
            symbol=symbol,
            interval=Client.KLINE_INTERVAL_15MINUTE,
            limit=VWAP_LOOKBACK
        )
        
        if not candles:
            return None
        
        # Current price and volume
        current_candle = candles[-1]
        current_price = float(current_candle[4])
        current_volume = float(current_candle[5])
        
        # Calculate indicators
        vwap_data = calculate_vwap(candles)
        if not vwap_data:
            return None
            
        market_regime = get_market_regime(symbol, candles)
        
        # Calculate volume MA
        volumes = [float(c[5]) for c in candles[-VOLUME_MA_PERIOD:]]
        volume_ma = sum(volumes) / len(volumes) if volumes else 0
        
        # Update state
        state["vwap_data"] = vwap_data
        state["current_adx"] = market_regime.adx
        state["volume_ma"] = volume_ma
        state["last_volume"] = current_volume
        
        # Display current status
        vwap_position = "Above" if current_price > vwap_data.vwap else "Below"
        # Color code price position relative to VWAP
        vwap_color = "\033[92m" if current_price > vwap_data.vwap else "\033[91m"
        regime_color = "\033[96m" if market_regime.trend == "trending" else "\033[93m" if market_regime.trend == "transitioning" else "\033[95m"
        volume_color = "\033[92m" if current_volume > volume_ma * 1.5 else "\033[93m" if current_volume > volume_ma else "\033[91m"
        
        print(f"\n📊 [{symbol}] Price: {vwap_color}{current_price:.6f}\033[0m | VWAP: {vwap_data.vwap:.6f} ({vwap_position})")
        print(f" - ADX: {market_regime.adx:.1f} ({market_regime.trend})")
        print(f" - Volume: {current_volume/volume_ma:.1f}x average")
        
        # Check for position management first
        if state["position_is_open"]:
            manage_position(symbol, current_price)
            return None
        
        # Entry logic
        cross = check_vwap_cross(symbol, current_price, vwap_data)
        
        if cross == 'bullish':
            # Check filters
            volume_spike = current_volume > (volume_ma * VOLUME_MULTIPLIER)
            adx_filter = market_regime.adx >= MIN_ADX_FOR_TREND
            
            # Distance from VWAP filter (not too far from VWAP)
            distance_from_vwap = abs(current_price - vwap_data.vwap) / vwap_data.vwap
            distance_filter = distance_from_vwap < 0.01  # Within 1% of VWAP
            
            print(f" - VWAP Cross: BULLISH \033[92m✓\033[0m")
            print(f" - Volume Filter: {'\033[92m✓\033[0m' if volume_spike else '\033[91m✗\033[0m'} ({current_volume/volume_ma:.1f}x)")
            print(f" - ADX Filter: {'\033[92m✓\033[0m' if adx_filter else '\033[91m✗\033[0m'} (ADX: {market_regime.adx:.1f})")
            print(f" - Distance Filter: {'\033[92m✓\033[0m' if distance_filter else '\033[91m✗\033[0m'} ({distance_from_vwap*100:.2f}%)")
            
            if volume_spike and adx_filter and distance_filter:
                # All conditions met - execute entry
                position_size = calculate_position_size(market_regime)
                stop_loss_pct = get_adaptive_stop_loss(market_regime)
                
                reason = f"VWAP cross with {current_volume/volume_ma:.1f}x volume, ADX {market_regime.adx:.1f}"
                
                execute_buy(symbol, current_price, position_size, stop_loss_pct, reason, market_regime)
        
    except Exception as e:
        print(f"{colored('⚠️ Error analyzing', 'red', attrs=['bold'])} {colored(symbol, 'white')}: {colored(str(e), 'red')}")

def execute_buy(symbol: str, price: float, position_size: float, stop_loss_pct: float, 
                reason: str, market_regime: MarketRegime):
    """Execute buy order with proper risk management"""
    state = trader_state[symbol]
    
    print(f"\n{colored('🔸 [DRY RUN]', 'yellow', attrs=['bold']) if DRY_RUN else ''} {colored('📈 EXECUTING BUY', 'green', attrs=['bold'])} for {symbol}")
    print(f" - Reason: {reason}")
    print(f" - Market: {market_regime.trend} (ADX: {market_regime.adx:.1f})")
    print(f" - Entry: {colored(f'{price:.6f}', 'green', attrs=['bold'])}")
    print(f" - Size: ${position_size:.2f}")
    print(f" - Stop Loss: {stop_loss_pct*100:.1f}%") 
    
    try:
        if DRY_RUN:
            # Simulate order
            quantity = position_size / price
            order = {
                'orderId': f'DRY_{int(time.time())}',
                'executedQty': quantity,
                'fills': [{'price': price, 'qty': quantity}]
            }
        else:
            # Real order
            order = client.order_market_buy(
                symbol=symbol,
                quoteOrderQty=position_size
            )
            quantity = float(order['executedQty'])
            price = float(order['fills'][0]['price'])
        
        # Update state
        state["position_is_open"] = True
        state["entry_price"] = price
        state["quantity"] = quantity
        state["remaining_quantity"] = quantity
        state["entry_time"] = datetime.datetime.now().isoformat()
        state["entry_reason"] = reason
        state["market_regime"] = market_regime
        
        # Set stops and targets
        state["stop_loss"] = price * (1 - stop_loss_pct)
        state["take_profit_1"] = price * (1 + TAKE_PROFIT_1)
        state["take_profit_2"] = price * (1 + TAKE_PROFIT_2)
        state["highest_price"] = price
        
        # Save position
        save_position_state(symbol)
        
        print(f" {colored('✅ Position opened:', 'green', attrs=['bold'])} {colored(f'{quantity:.8f}', 'white')} @ {colored(f'{price:.6f}', 'green', attrs=['bold'])}")
        
    except Exception as e:
        print(f" {colored('❌ Buy failed:', 'red', attrs=['bold'])} {colored(str(e), 'red')}")

def manage_position(symbol: str, current_price: float):
    """Manage open position with trailing stops"""
    state = trader_state[symbol]
    
    if not state["position_is_open"]:
        return
    
    entry_price = state["entry_price"]
    stop_loss = state["stop_loss"]
    profit_pct = (current_price - entry_price) / entry_price
    
    pnl_color = '\033[92m' if profit_pct > 0 else '\033[91m'
    print(f" - Position: OPEN | P&L: {pnl_color}{profit_pct*100:+.2f}%\033[0m")
    print(f" - Stop: {stop_loss:.6f} | TP1: {state['take_profit_1']:.6f}")
    
    # Update highest price
    if current_price > state["highest_price"]:
        state["highest_price"] = current_price
    
    # Check stop loss
    if current_price <= stop_loss:
        execute_sell(symbol, current_price, state["remaining_quantity"], "Stop Loss")
        return
    
    # Check TP1 (sell 50%)
    if not state["tp1_hit"] and current_price >= state["take_profit_1"]:
        sell_quantity = state["quantity"] * TP1_SELL_PERCENT
        execute_sell(symbol, current_price, sell_quantity, "TP1 - 50% Exit")
        state["tp1_hit"] = True
        state["stop_loss"] = entry_price  # Move stop to breakeven
        print(f" {colored('🎯 TP1 Hit! Moved stop to breakeven', 'cyan', attrs=['bold'])}")
        # Save updated state immediately after TP1 hit
        save_position_state(symbol)
    
    # Implement trailing stop
    if profit_pct >= TRAILING_ACTIVATION:
        # Determine trailing distance based on profit level
        if profit_pct >= 0.02:
            trail_distance = TRAILING_DISTANCE_STRONG
        elif profit_pct >= TAKE_PROFIT_1:
            trail_distance = TRAILING_DISTANCE_PROFIT
        else:
            trail_distance = TRAILING_DISTANCE_INITIAL
        
        new_trailing_stop = state["highest_price"] * (1 - trail_distance)
        
        if new_trailing_stop > state["stop_loss"]:
            state["stop_loss"] = new_trailing_stop
            state["trailing_distance"] = trail_distance
            locked_profit = (new_trailing_stop - entry_price) / entry_price
            print(f" {colored(f'📈 Trailing stop updated: {new_trailing_stop:.6f}', 'yellow', attrs=['bold'])} (Locked: {colored(f'{locked_profit*100:.2f}%', 'green')})")
    
    # Check TP2 (close remaining)
    if current_price >= state["take_profit_2"]:
        execute_sell(symbol, current_price, state["remaining_quantity"], "TP2 - Full Exit")

def execute_sell(symbol: str, price: float, quantity: float, reason: str):
    """Execute sell order"""
    state = trader_state[symbol]
    
    if quantity <= 0:
        return
    
    sell_color = 'yellow' if 'TP' in reason else 'red'
    print(f"\n{colored('🔸 [DRY RUN]', 'yellow', attrs=['bold']) if DRY_RUN else ''} {colored('📉 EXECUTING SELL', sell_color, attrs=['bold'])} for {colored(symbol, 'cyan', attrs=['bold'])}")
    print(f" 🎯 Reason: {colored(reason, 'white', attrs=['bold'])}")
    print(f" 💰 Price: {colored(f'{price:.6f}', sell_color, attrs=['bold'])}")
    print(f" 📊 Quantity: {colored(f'{quantity:.8f}', 'white')}")
    
    try:
        if DRY_RUN:
            # Simulate order
            order = {
                'orderId': f'DRY_SELL_{int(time.time())}',
                'executedQty': quantity,
                'cummulativeQuoteQty': quantity * price
            }
        else:
            # Real order - need to adjust quantity for Binance precision
            exchange_info = client.get_symbol_info(symbol)
            lot_size_filter = next(f for f in exchange_info['filters'] if f['filterType'] == 'LOT_SIZE')
            step_size = float(lot_size_filter['stepSize'])
            precision = int(round(-math.log(step_size, 10), 0))
            quantity = round(quantity, precision)
            
            order = client.order_market_sell(
                symbol=symbol,
                quantity=quantity
            )
        
        executed_qty = float(order['executedQty'])
        revenue = float(order.get('cummulativeQuoteQty', executed_qty * price))
        
        # Calculate P&L
        cost = state["entry_price"] * executed_qty
        gross_profit = revenue - cost
        fees = (cost + revenue) * 0.001  # 0.1% fees
        net_profit = gross_profit - fees
        
        # Update state
        state["remaining_quantity"] -= executed_qty
        state["realized_profit"] += net_profit
        state["total_realized_pnl"] += net_profit
        
        # Save trade
        save_trade(symbol, state["entry_price"], price, executed_qty, net_profit, reason)
        
        pnl_color = 'green' if net_profit > 0 else 'red'
        print(f" {colored('✅ Sold', pnl_color, attrs=['bold'])} {colored(f'{executed_qty:.8f}', 'white')} | P&L: {colored(f'{net_profit:+.2f} USDT', pnl_color, attrs=['bold'])}")
        
        # Close position if fully sold
        if state["remaining_quantity"] <= 0:
            pnl_color = 'green' if state['realized_profit'] > 0 else 'red'
            print(f" {colored('🏁 Position closed', pnl_color, attrs=['bold'])} | Total P&L: {colored(f"{state['realized_profit']:+.2f} USDT", pnl_color, attrs=['bold'])}")
            reset_position_state(symbol)
        
        # ALWAYS save position state after any sell (partial or full)
        save_position_state(symbol)
        
    except Exception as e:
        print(f" {colored('❌ Sell failed:', 'red', attrs=['bold'])} {colored(str(e), 'red')}")

def save_position_state(symbol: str):
    """Save position state to file"""
    state = trader_state[symbol]
    position_data = {
        "position_is_open": state["position_is_open"],
        "entry_price": state["entry_price"],
        "quantity": state["quantity"],
        "remaining_quantity": state["remaining_quantity"],
        "entry_time": state["entry_time"],
        "stop_loss": state["stop_loss"],
        "take_profit_1": state["take_profit_1"],
        "take_profit_2": state["take_profit_2"],
        "tp1_hit": state["tp1_hit"],
        "highest_price": state["highest_price"],
        "trailing_distance": state["trailing_distance"],
        "realized_profit": state["realized_profit"],
        "total_realized_pnl": state["total_realized_pnl"],
        "entry_reason": state["entry_reason"],
        "market_regime": {
            "adx": state["market_regime"].adx if state["market_regime"] else None,
            "trend": state["market_regime"].trend if state["market_regime"] else None
        } if state["market_regime"] else None
    }
    
    with open(state["filename_position"], 'w') as f:
        json.dump(position_data, f, indent=2)

def save_trade(symbol: str, entry_price: float, exit_price: float, quantity: float, 
               profit_loss: float, reason: str):
    """Save trade record"""
    state = trader_state[symbol]
    
    trades = []
    if os.path.exists(state["filename_trades"]):
        with open(state["filename_trades"], 'r') as f:
            trades = json.load(f)
    
    trade_record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "symbol": symbol,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "quantity": quantity,
        "profit_loss": profit_loss,
        "reason": reason,
        "market_regime": state["market_regime"].trend if state["market_regime"] else None,
        "adx": state["market_regime"].adx if state["market_regime"] else None
    }
    
    trades.append(trade_record)
    
    with open(state["filename_trades"], 'w') as f:
        json.dump(trades, f, indent=2)

def reset_position_state(symbol: str):
    """Reset position state after closing"""
    state = trader_state[symbol]
    state["position_is_open"] = False
    state["entry_price"] = None
    state["quantity"] = None
    state["remaining_quantity"] = None
    state["entry_time"] = None
    state["stop_loss"] = None
    state["take_profit_1"] = None
    state["take_profit_2"] = None
    state["tp1_hit"] = False
    state["highest_price"] = None
    state["trailing_distance"] = None
    state["realized_profit"] = 0
    state["entry_reason"] = None
    state["market_regime"] = None

def load_saved_positions():
    """Load saved positions on startup"""
    for symbol in symbols:
        state = trader_state[symbol]
        if os.path.exists(state["filename_position"]):
            try:
                with open(state["filename_position"], 'r') as f:
                    saved_data = json.load(f)
                    if saved_data.get("position_is_open"):
                        state["position_is_open"] = saved_data["position_is_open"]
                        state["entry_price"] = saved_data["entry_price"]
                        state["quantity"] = saved_data["quantity"]
                        state["remaining_quantity"] = saved_data["remaining_quantity"]
                        state["entry_time"] = saved_data["entry_time"]
                        state["stop_loss"] = saved_data["stop_loss"]
                        state["take_profit_1"] = saved_data["take_profit_1"]
                        state["take_profit_2"] = saved_data["take_profit_2"]
                        state["tp1_hit"] = saved_data.get("tp1_hit", False)
                        state["highest_price"] = saved_data.get("highest_price")
                        state["realized_profit"] = saved_data.get("realized_profit", 0)
                        state["total_realized_pnl"] = saved_data.get("total_realized_pnl", 0)
                        print(f"{colored('📁 Loaded position for', 'cyan')} {colored(symbol, 'white', attrs=['bold'])}")
            except Exception as e:
                print(f"{colored('⚠️ Error loading position for', 'yellow')} {colored(symbol, 'white')}: {colored(str(e), 'red')}")

def display_portfolio_summary():
    """Display portfolio P&L summary"""
    print("\n" + "="*80)
    mode_text = " (DRY RUN)" if DRY_RUN else " (LIVE)"
    print(f"💼 VWAP TRADER PORTFOLIO{mode_text}")
    print("="*80)
    
    total_realized = 0
    total_unrealized = 0
    open_positions = 0
    
    for symbol in symbols:
        state = trader_state[symbol]
        
        # Add cumulative realized P&L
        total_realized += state["total_realized_pnl"]
        
        if state["position_is_open"]:
            open_positions += 1
            # Get current price for unrealized P&L
            try:
                ticker = client.get_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])
                
                unrealized = (current_price - state["entry_price"]) * state["remaining_quantity"]
                unrealized -= unrealized * 0.001  # Est. fees
                total_unrealized += unrealized
                
                pnl_pct = (current_price - state["entry_price"]) / state["entry_price"] * 100
                
                # Color code individual P&L
                unrealized_color = '\033[92m' if unrealized > 0 else '\033[91m'
                total_color = '\033[92m' if state['total_realized_pnl'] > 0 else '\033[91m'
                pct_color = '\033[92m' if pnl_pct > 0 else '\033[91m'
                
                print(f"  {colored(symbol, 'white', attrs=['bold']):14} | Unrealized: {unrealized_color}{unrealized:+7.2f}\033[0m | Total: {total_color}{state['total_realized_pnl']:+7.2f}\033[0m | ({pct_color}{pnl_pct:+.2f}%\033[0m)")
            except:
                pass
    
    print(colored("-"*80, 'cyan'))
    print(f"{colored('📊 Open Positions:', 'cyan')} {colored(f'{open_positions} / {len(symbols)}', 'white', attrs=['bold'])}")
    
    # Color code realized P&L
    realized_color = 'green' if total_realized > 0 else 'red'
    print(f"{colored('💰 Total Realized:', 'cyan')}   {colored(f'${total_realized:+.2f}', realized_color, attrs=['bold'])}")
    
    # Color code unrealized P&L
    unrealized_color = 'green' if total_unrealized > 0 else 'red'
    print(f"{colored('📈 Total Unrealized:', 'cyan')} {colored(f'${total_unrealized:+.2f}', unrealized_color, attrs=['bold'])}")
    print(colored("-"*80, 'cyan'))
    
    # Color code total P&L
    total_pnl = total_realized + total_unrealized
    total_color = 'green' if total_pnl > 0 else 'red'
    print(f"{colored('💎 TOTAL P&L:', 'cyan', attrs=['bold'])} {colored(f'${total_pnl:+.2f}', total_color, attrs=['bold'])}")
    print(colored("="*80, 'cyan'))

def main_loop():
    """Main trading loop"""
    print(colored("="*80, 'cyan'))
    print(colored("🚀 VWAP TRADER WITH MARKET REGIME DETECTION", 'cyan', attrs=['bold']))
    print(colored("="*80, 'cyan'))
    mode_color = "yellow" if DRY_RUN else "red"
    mode_text = "DRY RUN MODE - NO REAL TRADES" if DRY_RUN else "LIVE TRADING - REAL MONEY"
    print(f"{'🔸' if DRY_RUN else '⚠️'} {colored(mode_text, mode_color, attrs=['bold'])}")
    print(colored("="*80, 'cyan'))
    print(f"{colored('Strategy:', 'cyan')} VWAP Cross with ADX Filter")
    print(f"{colored('ADX Threshold:', 'cyan')} {colored(f'{MIN_ADX_FOR_TREND}', 'white', attrs=['bold'])} (Skip trades below)")
    print(f"{colored('Volume Filter:', 'cyan')} {colored(f'{VOLUME_MULTIPLIER}x', 'white', attrs=['bold'])} average required")
    print(f"{colored('Adaptive Stops:', 'cyan')} {colored(f'{STOP_LOSS_RANGING*100}%', 'red')} (ranging) / {colored(f'{STOP_LOSS_TRENDING*100}%', 'red')} (trending)")
    print(f"{colored('Check Interval:', 'cyan')} {colored(f'{CHECK_INTERVAL_MINUTES} minutes', 'white', attrs=['bold'])}")
    print("="*80)
    
    # Load saved positions
    load_saved_positions()
    
    while not shutdown:
        try:
            print(f"\n{colored('⏰ Analysis at', 'cyan')} {colored(datetime.datetime.now().strftime('%H:%M:%S'), 'white', attrs=['bold'])}")
            print(colored("-"*80, 'cyan'))
            
            # Analyze all symbols
            for symbol in symbols:
                if shutdown:
                    break
                analyze_symbol(symbol)
            
            # Display portfolio summary
            display_portfolio_summary()
            
            # Wait for next check
            if not shutdown:
                next_check = datetime.datetime.now() + datetime.timedelta(minutes=CHECK_INTERVAL_MINUTES)
                print(f"\n{colored('⏰ Next analysis at', 'cyan')} {colored(next_check.strftime('%H:%M:%S'), 'white', attrs=['bold'])}")
                
                for _ in range(CHECK_INTERVAL_MINUTES * 60):
                    if shutdown:
                        break
                    time.sleep(1)
                    
        except Exception as e:
            print(f"{colored('⚠️ Error in main loop:', 'red', attrs=['bold'])} {colored(str(e), 'red')}")
            time.sleep(60)
    
    print("\n" + colored("="*80, 'cyan'))
    print(colored("🛑 VWAP Trader shutdown complete", 'cyan', attrs=['bold']))
    print(colored("="*80, 'cyan'))

if __name__ == "__main__":
    # Check API connection
    try:
        account = client.get_account()
        balance = next((b for b in account['balances'] if b['asset'] == 'USDC'), None)
        if balance:
            print(f"{colored('✅ API connected', 'green', attrs=['bold'])} | USDC Balance: {colored(balance['free'], 'green', attrs=['bold'])}")
        else:
            print(colored('⚠️ USDC balance not found', 'yellow', attrs=['bold']))
    except Exception as e:
        print(f"{colored('❌ API connection failed:', 'red', attrs=['bold'])} {colored(str(e), 'red')}")
        sys.exit(1)
    
    # Start trading
    main_loop()