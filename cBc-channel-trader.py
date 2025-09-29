#!/usr/bin/env python3
"""
Channel Breakout Trading Bot
Pure channel-based trading with parallel trend lines
Supports both LIVE and DRY RUN modes
"""

import os
import json
import datetime
import time
import math
import threading
import signal
import sys
from termcolor import colored
import pandas as pd
import numpy as np
from binance.client import Client
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
sys.path.append('../')
from _secrets import api_key, secret_key

# ==================== CONFIGURATION ====================
# TRADING MODE
DRY_RUN = True  # Set to False for LIVE trading with real money

# TIMEFRAME SELECTION
TIMEFRAME_MINUTES = 15  # Options: 1, 3, 5, 15, 30, 60, 240, 1440 (24h)
# 1min = Scalping (very fast, lots of signals)
# 5min = Active trading (good balance)  
# 15min = Swing trading (fewer, higher quality signals)
# 60min+ = Position trading (long-term channels)

# Position sizing
MIN_POSITION_VALUE = 50   # Minimum $50 position
MAX_POSITION_VALUE = 100  # Maximum $100 position

# Channel parameters (auto-adjusted for timeframe)
if TIMEFRAME_MINUTES <= 5:
    # Scalping - more lenient, faster signals
    MIN_CHANNEL_TOUCHES = 3
    CHANNEL_LOOKBACK = 200        # More history for 5min
    BREAKOUT_CONFIRM_PERCENT = 0.2  # Lower threshold for faster entry
    VOLUME_SPIKE_MULTIPLIER = 1.3   # Lower volume requirement
    MIN_CHANNEL_WIDTH_PERCENT = 0.3 # Narrower channels OK
elif TIMEFRAME_MINUTES <= 15:
    # Active trading - balanced
    MIN_CHANNEL_TOUCHES = 3
    CHANNEL_LOOKBACK = 100
    BREAKOUT_CONFIRM_PERCENT = 0.3
    VOLUME_SPIKE_MULTIPLIER = 1.5
    MIN_CHANNEL_WIDTH_PERCENT = 0.5
else:
    # Swing/Position trading - more conservative  
    MIN_CHANNEL_TOUCHES = 4        # Higher quality channels
    CHANNEL_LOOKBACK = 80          # Less history needed
    BREAKOUT_CONFIRM_PERCENT = 0.4  # Higher confirmation
    VOLUME_SPIKE_MULTIPLIER = 2.0   # Strong volume required
    MIN_CHANNEL_WIDTH_PERCENT = 0.8 # Wider channels only

PARALLEL_TOLERANCE = 0.15      # Max slope difference for parallel lines (15%)

# Entry filters
RSI_OVERBOUGHT = 70            # Don't buy if RSI > 70
RSI_OVERSOLD = 30              # Extra confidence if bouncing from oversold

# Risk Management
STOP_LOSS_BUFFER = 0.003       # 0.3% buffer below support
TP1_CHANNEL_RATIO = 1.0        # Take profit 1 at 1x channel width
TP2_CHANNEL_RATIO = 1.618      # Take profit 2 at 1.618x channel width (Fibonacci)
MIN_RISK_REWARD = 2.0          # Minimum risk/reward ratio

# Partial exit strategy
TP1_SELL_PERCENT = 0.5         # Sell 50% at TP1

# Trailing stop parameters (after TP1)
TRAIL_INITIAL = 0.8   # 0.8% trailing stop initially
TRAIL_MEDIUM = 0.5    # 0.5% after 2% profit
TRAIL_TIGHT = 0.3     # 0.3% after 3% profit

# Timeframe mapping
TIMEFRAME_MAP = {
    1: Client.KLINE_INTERVAL_1MINUTE,
    3: Client.KLINE_INTERVAL_3MINUTE,
    5: Client.KLINE_INTERVAL_5MINUTE,
    15: Client.KLINE_INTERVAL_15MINUTE,
    30: Client.KLINE_INTERVAL_30MINUTE,
    60: Client.KLINE_INTERVAL_1HOUR,
    240: Client.KLINE_INTERVAL_4HOUR,
    1440: Client.KLINE_INTERVAL_1DAY
}

# Validate and set timeframe
if TIMEFRAME_MINUTES not in TIMEFRAME_MAP:
    print(f"❌ Invalid timeframe: {TIMEFRAME_MINUTES}. Using 15 minutes as default.")
    TIMEFRAME_MINUTES = 15

TIMEFRAME = TIMEFRAME_MAP[TIMEFRAME_MINUTES]
CHECK_INTERVAL = TIMEFRAME_MINUTES  # Check every timeframe period

# ==================== END CONFIGURATION ====================

# Initialize Binance client
client = Client(api_key, secret_key)

# Trading symbols
symbols = [
    "BTCUSDC", "ETHUSDC", "BNBUSDC", "SOLUSDC", "ADAUSDC",
    "XRPUSDC", "DOGEUSDC", "LINKUSDC", "MATICUSDC", "UNIUSDC",
    "ATOMUSDC", "LTCUSDC", "AVAXUSDC"
]

# Global state
shutdown = False
trader_state = {}

@dataclass
class Channel:
    """Represents a price channel with parallel trend lines"""
    upper_line_points: List[Tuple[int, float]]  # (candle_index, price)
    lower_line_points: List[Tuple[int, float]]
    upper_slope: float
    lower_slope: float
    upper_intercept: float
    lower_intercept: float
    channel_width: float
    midline: float
    touches_upper: int
    touches_lower: int
    is_valid: bool
    trend: str  # 'bullish', 'bearish', 'sideways'

class ChannelTrader:
    def __init__(self):
        self.initialize_state()
        self.load_all_positions()
        
    def initialize_state(self):
        """Initialize trader state for all symbols"""
        for symbol in symbols:
            state_file = f"trader_channel/positions/position_{symbol}.json"
            os.makedirs("trader_channel/positions", exist_ok=True)
            
            trader_state[symbol] = {
                "position_is_open": False,
                "entry_price": None,
                "quantity": None,
                "entry_time": None,
                "stop_loss": None,
                "take_profit_1": None,
                "take_profit_2": None,
                "risk_reward_ratio": None,
                "entry_reason": None,
                "entry_type": None,
                "tp1_hit": False,
                "highest_price": None,
                "trailing_stop": None,
                "trailing_distance": None,
                "original_quantity": None,
                "remaining_quantity": None,
                "realized_profit": 0,
                "tp1_exit_price": None,
                "channel_width": None,
                "breakout_level": None
            }
    
    def load_all_positions(self):
        """Load saved positions for all symbols"""
        for symbol in symbols:
            self.load_position_state(symbol)
    
    def load_position_state(self, symbol: str):
        """Load position state from file"""
        state_file = f"trader_channel/positions/position_{symbol}.json"
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    saved_state = json.load(f)
                    trader_state[symbol].update(saved_state)
                    if trader_state[symbol]["position_is_open"]:
                        mode_prefix = "🔸 [DRY RUN]" if DRY_RUN else ""
                        print(f"{mode_prefix} Loaded open position for {symbol}")
            except Exception as e:
                print(f"Error loading state for {symbol}: {e}")
    
    def save_position_state(self, symbol: str):
        """Save position state to file"""
        state_file = f"trader_channel/positions/position_{symbol}.json"
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        with open(state_file, 'w') as f:
            json.dump(trader_state[symbol], f, indent=2, default=str)
    
    def get_candle_data(self, symbol: str, limit: int = CHANNEL_LOOKBACK) -> pd.DataFrame:
        """Get historical candle data"""
        try:
            klines = client.get_klines(symbol=symbol, interval=TIMEFRAME, limit=limit)
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
            
        except Exception as e:
            print(f"Error getting candle data for {symbol}: {e}")
            return pd.DataFrame()
    
    def find_pivot_points(self, df: pd.DataFrame, window: int = 5) -> Tuple[List, List]:
        """Find pivot highs and lows in price data"""
        highs = []
        lows = []
        
        for i in range(window, len(df) - window):
            # Check for pivot high
            is_pivot_high = True
            for j in range(i - window, i + window + 1):
                if j != i and df['high'].iloc[j] >= df['high'].iloc[i]:
                    is_pivot_high = False
                    break
            
            if is_pivot_high:
                highs.append((i, df['high'].iloc[i]))
            
            # Check for pivot low
            is_pivot_low = True
            for j in range(i - window, i + window + 1):
                if j != i and df['low'].iloc[j] <= df['low'].iloc[i]:
                    is_pivot_low = False
                    break
            
            if is_pivot_low:
                lows.append((i, df['low'].iloc[i]))
        
        return highs, lows
    
    def fit_trend_line(self, points: List[Tuple[int, float]]) -> Tuple[float, float]:
        """Fit a trend line to points using linear regression"""
        if len(points) < 2:
            return 0, 0
        
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        
        # Linear regression: y = mx + b
        n = len(points)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
        intercept = (np.sum(y) - slope * np.sum(x)) / n
        
        return slope, intercept
    
    def count_line_touches(self, df: pd.DataFrame, slope: float, intercept: float, 
                          is_upper: bool, tolerance_percent: float = 0.3) -> int:
        """Count how many candles touch or come close to a trend line"""
        touches = 0
        
        for i in range(len(df)):
            line_price = slope * i + intercept
            tolerance = line_price * tolerance_percent / 100
            
            if is_upper:
                # Check if high is near upper line
                if abs(df['high'].iloc[i] - line_price) <= tolerance:
                    touches += 1
            else:
                # Check if low is near lower line
                if abs(df['low'].iloc[i] - line_price) <= tolerance:
                    touches += 1
        
        return touches
    
    def detect_channel(self, df: pd.DataFrame) -> Optional[Channel]:
        """Detect price channel from candle data"""
        if len(df) < 20:
            return None
        
        # Find pivot points
        pivot_highs, pivot_lows = self.find_pivot_points(df)
        
        if len(pivot_highs) < 2 or len(pivot_lows) < 2:
            return None
        
        # Try different combinations of pivot points to find best channel
        best_channel = None
        max_score = 0
        
        # Use recent pivot points for trend lines
        recent_highs = pivot_highs[-5:] if len(pivot_highs) > 5 else pivot_highs
        recent_lows = pivot_lows[-5:] if len(pivot_lows) > 5 else pivot_lows
        
        # Fit trend lines
        upper_slope, upper_intercept = self.fit_trend_line(recent_highs)
        lower_slope, lower_intercept = self.fit_trend_line(recent_lows)
        
        # Check if lines are roughly parallel
        slope_diff = abs(upper_slope - lower_slope)
        avg_slope = (abs(upper_slope) + abs(lower_slope)) / 2
        
        if avg_slope > 0:
            parallelism = slope_diff / avg_slope
        else:
            parallelism = 0
        
        if parallelism > PARALLEL_TOLERANCE:
            return None  # Lines not parallel enough
        
        # Count touches on each line
        upper_touches = self.count_line_touches(df, upper_slope, upper_intercept, True)
        lower_touches = self.count_line_touches(df, lower_slope, lower_intercept, False)
        
        # Calculate channel width at current candle
        current_idx = len(df) - 1
        upper_current = upper_slope * current_idx + upper_intercept
        lower_current = lower_slope * current_idx + lower_intercept
        channel_width = upper_current - lower_current
        
        # Check minimum channel width
        current_price = df['close'].iloc[-1]
        width_percent = (channel_width / current_price) * 100
        
        if width_percent < MIN_CHANNEL_WIDTH_PERCENT:
            return None  # Channel too narrow
        
        # Determine trend based on average slope
        avg_slope = (upper_slope + lower_slope) / 2
        if avg_slope > current_price * 0.0001:  # Bullish if rising more than 0.01% per candle
            trend = "bullish"
        elif avg_slope < -current_price * 0.0001:  # Bearish if falling
            trend = "bearish"
        else:
            trend = "sideways"
        
        # Calculate midline
        midline = (upper_current + lower_current) / 2
        
        # Create channel object
        channel = Channel(
            upper_line_points=recent_highs,
            lower_line_points=recent_lows,
            upper_slope=upper_slope,
            lower_slope=lower_slope,
            upper_intercept=upper_intercept,
            lower_intercept=lower_intercept,
            channel_width=channel_width,
            midline=midline,
            touches_upper=upper_touches,
            touches_lower=lower_touches,
            is_valid=(upper_touches >= MIN_CHANNEL_TOUCHES and lower_touches >= MIN_CHANNEL_TOUCHES),
            trend=trend
        )
        
        return channel if channel.is_valid else None
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def find_channel_opportunity(self, symbol: str, df: pd.DataFrame, channel: Channel) -> Optional[dict]:
        """Find trading opportunity based on channel analysis"""
        if len(df) < 20 or not channel:
            return None
        
        current_price = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].iloc[-20:-1].mean()
        
        # Calculate current channel boundaries
        current_idx = len(df) - 1
        upper_channel = channel.upper_slope * current_idx + channel.upper_intercept
        lower_channel = channel.lower_slope * current_idx + channel.lower_intercept
        midline = (upper_channel + lower_channel) / 2
        
        # Calculate RSI
        rsi = self.calculate_rsi(df['close'], 14)
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        
        # Strategy 1: Channel Breakout (Upward)
        breakout_level = upper_channel * (1 + BREAKOUT_CONFIRM_PERCENT / 100)
        if current_price > breakout_level and prev_close <= upper_channel:
            # Confirm with volume
            if current_volume > avg_volume * VOLUME_SPIKE_MULTIPLIER:
                # Check RSI not overbought
                if current_rsi < RSI_OVERBOUGHT:
                    stop_loss = upper_channel * (1 - STOP_LOSS_BUFFER)
                    take_profit_1 = current_price + channel.channel_width * TP1_CHANNEL_RATIO
                    take_profit_2 = current_price + channel.channel_width * TP2_CHANNEL_RATIO
                    
                    risk = current_price - stop_loss
                    reward = take_profit_2 - current_price
                    risk_reward = reward / risk if risk > 0 else 0
                    
                    if risk_reward >= MIN_RISK_REWARD:
                        return {
                            "type": "channel_breakout",
                            "direction": "long",
                            "entry_price": current_price,
                            "stop_loss": stop_loss,
                            "take_profit_1": take_profit_1,
                            "take_profit_2": take_profit_2,
                            "risk_reward": risk_reward,
                            "channel_width": channel.channel_width,
                            "breakout_level": upper_channel,
                            "reason": f"Channel breakout with {current_volume/avg_volume:.1f}x volume",
                            "confidence": min(80 + (current_volume/avg_volume - 1.5) * 10, 95)
                        }
        
        # Strategy 2: Lower Channel Bounce
        if channel.trend in ["bullish", "sideways"]:
            # Check if price bounced from lower channel
            if df['low'].iloc[-1] <= lower_channel * (1 + STOP_LOSS_BUFFER):
                if current_price > lower_channel and current_price < midline:
                    # Extra confidence if oversold
                    confidence_boost = 10 if current_rsi < RSI_OVERSOLD else 0
                    
                    stop_loss = lower_channel * (1 - STOP_LOSS_BUFFER)
                    take_profit_1 = midline + channel.channel_width * 0.25  # Conservative TP1
                    take_profit_2 = upper_channel  # Full channel width
                    
                    risk = current_price - stop_loss
                    reward = take_profit_2 - current_price
                    risk_reward = reward / risk if risk > 0 else 0
                    
                    if risk_reward >= MIN_RISK_REWARD:
                        return {
                            "type": "channel_bounce",
                            "direction": "long",
                            "entry_price": current_price,
                            "stop_loss": stop_loss,
                            "take_profit_1": take_profit_1,
                            "take_profit_2": take_profit_2,
                            "risk_reward": risk_reward,
                            "channel_width": channel.channel_width,
                            "breakout_level": lower_channel,
                            "reason": f"Bounce from lower channel (RSI: {current_rsi:.0f})",
                            "confidence": 70 + confidence_boost
                        }
        
        # Strategy 3: Midline Bounce (in strong uptrend)
        if channel.trend == "bullish" and channel.touches_upper >= 4:
            # Check if price bounced from midline
            if df['low'].iloc[-1] <= midline * (1 + 0.002):  # Within 0.2% of midline
                if current_price > midline and current_price < upper_channel * 0.95:
                    stop_loss = midline * (1 - STOP_LOSS_BUFFER)
                    take_profit_1 = upper_channel
                    take_profit_2 = upper_channel + channel.channel_width * 0.618  # Fibonacci extension
                    
                    risk = current_price - stop_loss
                    reward = take_profit_2 - current_price
                    risk_reward = reward / risk if risk > 0 else 0
                    
                    if risk_reward >= MIN_RISK_REWARD:
                        return {
                            "type": "midline_bounce",
                            "direction": "long",
                            "entry_price": current_price,
                            "stop_loss": stop_loss,
                            "take_profit_1": take_profit_1,
                            "take_profit_2": take_profit_2,
                            "risk_reward": risk_reward,
                            "channel_width": channel.channel_width,
                            "breakout_level": midline,
                            "reason": f"Midline bounce in uptrend channel",
                            "confidence": 75
                        }
        
        return None
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk management"""
        if entry_price <= 0 or stop_loss <= 0:
            return 0
        
        # For channel trading, we use fixed position sizing
        position_value = MAX_POSITION_VALUE
        
        # Ensure minimum position
        if position_value < MIN_POSITION_VALUE:
            position_value = MIN_POSITION_VALUE
        
        position_size = position_value / entry_price
        return position_size
    
    def execute_trade(self, symbol: str, opportunity: dict) -> bool:
        """Execute trade (live or dry run)"""
        mode_prefix = "🔸 [DRY RUN]" if DRY_RUN else ""
        
        try:
            if opportunity["direction"] == "long":
                position_size = self.calculate_position_size(
                    opportunity["entry_price"], 
                    opportunity["stop_loss"]
                )
                
                if position_size <= 0:
                    return False
                
                # Display trade setup
                strategy_emojis = {
                    "channel_breakout": "📈",
                    "channel_bounce": "🔄",
                    "midline_bounce": "➖"
                }
                
                emoji = strategy_emojis.get(opportunity["type"], "📊")
                strategy_name = opportunity["type"].replace("_", " ").title()
                
                print(f"\n{colored(f'{mode_prefix} {emoji} {strategy_name.upper()} SIGNAL', 'green', attrs=['bold'])} for {symbol}")
                print(f"  Entry: {opportunity['entry_price']:.6f}")
                print(f"  Stop Loss: {opportunity['stop_loss']:.6f} (-{((opportunity['entry_price']-opportunity['stop_loss'])/opportunity['entry_price']*100):.1f}%)")
                print(f"  TP1: {opportunity['take_profit_1']:.6f} (+{((opportunity['take_profit_1']-opportunity['entry_price'])/opportunity['entry_price']*100):.1f}%)")
                print(f"  TP2: {opportunity['take_profit_2']:.6f} (+{((opportunity['take_profit_2']-opportunity['entry_price'])/opportunity['entry_price']*100):.1f}%)")
                print(f"  Risk/Reward: {opportunity['risk_reward']:.2f}")
                print(f"  Channel Width: {opportunity['channel_width']:.6f}")
                print(f"  Confidence: {opportunity['confidence']:.0f}%")
                print(f"  Reason: {opportunity['reason']}")
                
                if DRY_RUN:
                    # Simulate trade execution
                    quote_quantity = position_size * opportunity["entry_price"]
                    quote_quantity = round(quote_quantity, 2)
                    if quote_quantity < 10:
                        quote_quantity = 10
                    
                    print(f"  {colored(f'{mode_prefix} Order Size: ${quote_quantity:.2f}', 'yellow')}")
                    
                    # Simulate fill
                    executed_qty = quote_quantity / opportunity["entry_price"]
                    avg_price = opportunity["entry_price"]
                    
                    print(f"{colored(f'  {mode_prefix} ✅ POSITION OPENED (SIMULATED)', 'green')} - {executed_qty:.8f} {symbol[:-4]}")
                    
                else:
                    # LIVE TRADING
                    quote_quantity = position_size * opportunity["entry_price"]
                    quote_quantity = round(quote_quantity, 2)
                    if quote_quantity < 10:
                        quote_quantity = 10
                    
                    print(f"  Order Size: ${quote_quantity:.2f}")
                    
                    # Execute real market buy
                    order = client.order_market_buy(
                        symbol=symbol,
                        quoteOrderQty=quote_quantity
                    )
                    
                    if 'orderId' not in order:
                        print(f"  {colored('❌ Order failed', 'red')}")
                        return False
                    
                    executed_qty = sum(float(fill['qty']) for fill in order['fills'])
                    avg_price = float(order['fills'][0]['price'])
                    
                    print(f"{colored(f'  ✅ POSITION OPENED (LIVE)', 'green')} - {executed_qty:.8f} {symbol[:-4]}")
                
                # Update state (same for dry run and live)
                trader_state[symbol].update({
                    "position_is_open": True,
                    "entry_price": avg_price,
                    "quantity": executed_qty,
                    "entry_time": datetime.datetime.now().isoformat(),
                    "stop_loss": opportunity["stop_loss"],
                    "take_profit_1": opportunity["take_profit_1"],
                    "take_profit_2": opportunity["take_profit_2"],
                    "risk_reward_ratio": opportunity["risk_reward"],
                    "entry_reason": opportunity["reason"],
                    "entry_type": opportunity["type"],
                    "tp1_hit": False,
                    "highest_price": avg_price,
                    "trailing_stop": None,
                    "trailing_distance": None,
                    "original_quantity": executed_qty,
                    "remaining_quantity": executed_qty,
                    "realized_profit": 0,
                    "tp1_exit_price": None,
                    "channel_width": opportunity.get("channel_width"),
                    "breakout_level": opportunity.get("breakout_level")
                })
                
                self.save_position_state(symbol)
                
                # Log to P&L file
                self.log_trade_open(symbol, avg_price, executed_qty, opportunity)
                
                return True
                
        except Exception as e:
            print(f"{colored(f'  ❌ TRADE ERROR: {e}', 'red')}")
            return False
    
    def execute_partial_sell(self, symbol: str, sell_percent: float, reason: str) -> bool:
        """Execute partial sell (live or dry run)"""
        mode_prefix = "🔸 [DRY RUN]" if DRY_RUN else ""
        
        try:
            state = trader_state[symbol]
            current_quantity = state.get("remaining_quantity") or state["quantity"]
            
            if current_quantity <= 0:
                return False
            
            sell_quantity = current_quantity * sell_percent
            
            if DRY_RUN:
                # Simulate partial sell
                ticker = client.get_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])
                
                # Simulate execution
                executed_qty = sell_quantity
                exit_price = current_price
                
                print(f"  {mode_prefix} 💰 Partial sell ({sell_percent*100:.0f}%) executed at {exit_price:.6f}")
                
            else:
                # LIVE partial sell
                # Get precision for the symbol
                exchange_info = client.get_symbol_info(symbol)
                lot_size_filter = next(f for f in exchange_info['filters'] if f['filterType'] == 'LOT_SIZE')
                step_size = float(lot_size_filter['stepSize'])
                precision = int(round(-math.log(step_size, 10), 0))
                
                # Round quantity
                sell_quantity = round(sell_quantity, precision)
                if sell_quantity <= 0:
                    return False
                
                print(f"  💰 Executing partial sell ({sell_percent*100:.0f}%) for {symbol}")
                
                order = client.order_market_sell(
                    symbol=symbol,
                    quantity=sell_quantity
                )
                
                executed_qty = float(order.get('executedQty', 0))
                cummulative_quote_qty = float(order.get('cummulativeQuoteQty', 0))
                
                if executed_qty <= 0:
                    return False
                
                exit_price = cummulative_quote_qty / executed_qty
            
            # Update state (same for dry run and live)
            entry_price = state["entry_price"]
            
            # Calculate profit
            gross_profit = (exit_price - entry_price) * executed_qty
            fees = (entry_price * executed_qty + exit_price * executed_qty) * 0.001
            net_profit = gross_profit - fees
            
            # Update state
            state["remaining_quantity"] = current_quantity - executed_qty
            state["realized_profit"] = state.get("realized_profit", 0) + net_profit
            
            # If this was TP1, record the exit price
            if reason == "TP1":
                state["tp1_exit_price"] = exit_price
            
            self.save_position_state(symbol)
            
            profit_pct = ((exit_price - entry_price) / entry_price) * 100
            color = 'green' if net_profit > 0 else 'red'
            print(f"  {colored(f'Profit: ${net_profit:.2f} ({profit_pct:+.1f}%)', color)}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Partial sell error: {e}")
            return False
    
    def check_exit_conditions(self, symbol: str, current_price: float) -> bool:
        """Check exit conditions for open positions"""
        state = trader_state[symbol]
        if not state["position_is_open"]:
            return False
        
        mode_prefix = "🔸 [DRY RUN]" if DRY_RUN else ""
        
        entry_price = state["entry_price"]
        stop_loss = state["stop_loss"]
        tp1 = state["take_profit_1"]
        
        # Update highest price
        if state["highest_price"] is None or current_price > state["highest_price"]:
            state["highest_price"] = current_price
        
        # Check for TP1 hit
        if current_price >= tp1 and not state.get("tp1_hit", False):
            print(f"\n{colored(f'{mode_prefix} 🎯 TAKE PROFIT 1 HIT', 'green')} for {symbol}")
            
            if self.execute_partial_sell(symbol, TP1_SELL_PERCENT, "TP1"):
                state["tp1_hit"] = True
                state["trailing_stop"] = entry_price  # Move stop to breakeven
                state["stop_loss"] = entry_price
                print(f"  🔒 Stop moved to breakeven: {entry_price:.6f}")
                self.save_position_state(symbol)
        
        # Progressive trailing stop after TP1
        if state.get("tp1_hit", False):
            profit_percent = ((current_price - entry_price) / entry_price) * 100
            
            # Determine trailing distance
            if profit_percent < 2.0:
                trailing_percent = TRAIL_INITIAL
            elif profit_percent < 3.0:
                trailing_percent = TRAIL_MEDIUM
            else:
                trailing_percent = TRAIL_TIGHT
            
            # Calculate new trailing stop
            new_trailing_stop = state["highest_price"] * (1 - trailing_percent / 100)
            
            # Update if higher than current stop
            if new_trailing_stop > state["stop_loss"]:
                state["stop_loss"] = new_trailing_stop
                state["trailing_distance"] = trailing_percent
                self.save_position_state(symbol)
        
        # Check for stop loss hit
        if current_price <= stop_loss:
            remaining_qty = state.get("remaining_quantity") or state["quantity"]
            
            if remaining_qty > 0:
                reason = "Stop Loss"
                if state.get("tp1_hit", False):
                    profit_pct = ((current_price - entry_price) / entry_price) * 100
                    reason = f"Trailing Stop ({profit_pct:.1f}% profit)"
                
                print(f"\n{colored(f'{mode_prefix} 🛑 {reason.upper()}', 'red' if not state.get('tp1_hit') else 'yellow')} for {symbol}")
                
                # Close position
                if self.close_position(symbol, current_price, reason):
                    return True
        
        return False
    
    def close_position(self, symbol: str, exit_price: float, reason: str) -> bool:
        """Close entire position"""
        mode_prefix = "🔸 [DRY RUN]" if DRY_RUN else ""
        
        try:
            state = trader_state[symbol]
            remaining_qty = state.get("remaining_quantity") or state["quantity"]
            
            if DRY_RUN:
                # Simulate close
                executed_qty = remaining_qty
                
            else:
                # LIVE close
                # Get precision
                exchange_info = client.get_symbol_info(symbol)
                lot_size_filter = next(f for f in exchange_info['filters'] if f['filterType'] == 'LOT_SIZE')
                step_size = float(lot_size_filter['stepSize'])
                precision = int(round(-math.log(step_size, 10), 0))
                
                # Round quantity
                sell_quantity = round(remaining_qty, precision)
                
                if sell_quantity <= 0:
                    return False
                
                order = client.order_market_sell(
                    symbol=symbol,
                    quantity=sell_quantity
                )
                
                executed_qty = float(order.get('executedQty', 0))
                cummulative_quote_qty = float(order.get('cummulativeQuoteQty', 0))
                
                if executed_qty > 0:
                    exit_price = cummulative_quote_qty / executed_qty
            
            # Calculate final P&L
            entry_price = state["entry_price"]
            gross_profit = (exit_price - entry_price) * executed_qty
            fees = (entry_price * executed_qty + exit_price * executed_qty) * 0.001
            net_profit = gross_profit - fees
            total_profit = state.get("realized_profit", 0) + net_profit
            
            # Log the trade
            self.log_trade_close(symbol, exit_price, total_profit, reason)
            
            # Reset state
            trader_state[symbol] = {
                "position_is_open": False,
                "entry_price": None,
                "quantity": None,
                "entry_time": None,
                "stop_loss": None,
                "take_profit_1": None,
                "take_profit_2": None,
                "risk_reward_ratio": None,
                "entry_reason": None,
                "entry_type": None,
                "tp1_hit": False,
                "highest_price": None,
                "trailing_stop": None,
                "trailing_distance": None,
                "original_quantity": None,
                "remaining_quantity": None,
                "realized_profit": 0,
                "tp1_exit_price": None,
                "channel_width": None,
                "breakout_level": None
            }
            
            self.save_position_state(symbol)
            
            # Display result
            profit_pct = ((exit_price - entry_price) / entry_price) * 100
            color = 'green' if total_profit > 0 else 'red'
            
            print(f"  {mode_prefix} Position closed at {exit_price:.6f}")
            print(f"  {colored(f'Total P&L: ${total_profit:.2f} ({profit_pct:+.1f}%)', color)}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Close position error: {e}")
            return False
    
    def log_trade_open(self, symbol: str, entry_price: float, quantity: float, opportunity: dict):
        """Log trade opening to session file"""
        session_file = "trader_channel/session_pnl_dry.json" if DRY_RUN else "trader_channel/session_pnl.json"
        
        # Implementation continues in next part...
    
    def log_trade_close(self, symbol: str, exit_price: float, total_profit: float, reason: str):
        """Log trade closing to session file"""
        session_file = "trader_channel/session_pnl_dry.json" if DRY_RUN else "trader_channel/session_pnl.json"
        
        try:
            # Load existing data
            if os.path.exists(session_file):
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
            else:
                session_data = {
                    "trades": [],
                    "total_trades": 0,
                    "profitable_trades": 0,
                    "total_realized": 0,
                    "last_update": None
                }
            
            # Add trade record
            trade_record = {
                "timestamp": datetime.datetime.now().isoformat(),
                "symbol": symbol,
                "entry_price": trader_state[symbol]["entry_price"],
                "exit_price": exit_price,
                "quantity": trader_state[symbol]["original_quantity"],
                "profit_loss": total_profit,
                "reason": reason,
                "entry_reason": trader_state[symbol]["entry_reason"],
                "entry_type": trader_state[symbol]["entry_type"],
                "tp1_hit": trader_state[symbol].get("tp1_hit", False),
                "highest_price": trader_state[symbol].get("highest_price")
            }
            
            session_data["trades"].append(trade_record)
            session_data["total_trades"] += 1
            if total_profit > 0:
                session_data["profitable_trades"] += 1
            session_data["total_realized"] += total_profit
            session_data["last_update"] = datetime.datetime.now().isoformat()
            
            # Save
            os.makedirs(os.path.dirname(session_file), exist_ok=True)
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Error logging trade: {e}")
    
    def explain_channel_analysis(self, symbol: str, current_price: float, upper: float, lower: float, midline: float, channel) -> None:
        """Provide detailed explanation of channel analysis"""
        width_percent = (channel.channel_width / current_price) * 100
        
        # Determine price position in channel
        if current_price > upper:
            position = "ABOVE"
            position_color = "red"
            distance_from_level = ((current_price - upper) / upper) * 100
            explanation = f"Price is {distance_from_level:.1f}% above upper channel"
            
            if channel.trend == "bullish":
                explanation += " - Potential bullish breakout occurring!"
            elif channel.trend == "bearish":
                explanation += " - May reject back into channel"
            else:
                explanation += " - Range breakout, needs volume confirmation"
                
        elif current_price < lower:
            position = "BELOW"
            position_color = "red"
            distance_from_level = ((lower - current_price) / lower) * 100
            explanation = f"Price is {distance_from_level:.1f}% below lower channel"
            
            if channel.trend == "bullish":
                explanation += " - Potential bounce opportunity"
            elif channel.trend == "bearish":
                explanation += " - Bearish breakdown continuing"
            else:
                explanation += " - Range breakdown, needs volume confirmation"
                
        elif current_price > midline:
            position = "UPPER HALF"
            position_color = "yellow"
            distance_to_upper = ((upper - current_price) / current_price) * 100
            explanation = f"Price in upper half, {distance_to_upper:.1f}% from resistance"
            
            if channel.trend == "bullish":
                explanation += " - Good position for continuation"
            else:
                explanation += " - May face resistance at upper channel"
                
        else:
            position = "LOWER HALF"
            position_color = "green"
            distance_to_lower = ((current_price - lower) / current_price) * 100
            explanation = f"Price in lower half, {distance_to_lower:.1f}% from support"
            
            if channel.trend == "bullish":
                explanation += " - Potential bounce zone"
            else:
                explanation += " - Weak position, watch for breakdown"
        
        print(f"  📍 Position: {colored(position, position_color)} ({width_percent:.1f}% channel width)")
        print(f"  💡 Analysis: {explanation}")
        
        # Add signal expectations
        if channel.trend == "bullish":
            print(f"  🎯 Looking for: Bounce from {lower:.6f} or breakout above {upper * 1.003:.6f}")
        elif channel.trend == "bearish":
            print(f"  🎯 Looking for: Rejection from {upper:.6f} (SHORT signals disabled)")
        else:
            print(f"  🎯 Looking for: Breakout above {upper * 1.003:.6f} or bounce from {lower:.6f}")
    
    def explain_no_channel(self, symbol: str) -> None:
        """Explain why no valid channel was detected"""
        try:
            df = self.get_candle_data(symbol)
            if len(df) < 20:
                print(f"  💡 Reason: Not enough price history (need {CHANNEL_LOOKBACK} candles)")
                return
            
            # Find pivot points to analyze the issue
            pivot_highs, pivot_lows = self.find_pivot_points(df)
            
            if len(pivot_highs) < 2:
                print(f"  💡 Reason: Not enough swing highs (found {len(pivot_highs)}, need 2+)")
            elif len(pivot_lows) < 2:
                print(f"  💡 Reason: Not enough swing lows (found {len(pivot_lows)}, need 2+)")
            else:
                # Check if we can create lines
                recent_highs = pivot_highs[-5:] if len(pivot_highs) > 5 else pivot_highs
                recent_lows = pivot_lows[-5:] if len(pivot_lows) > 5 else pivot_lows
                
                upper_slope, upper_intercept = self.fit_trend_line(recent_highs)
                lower_slope, lower_intercept = self.fit_trend_line(recent_lows)
                
                # Check parallelism
                slope_diff = abs(upper_slope - lower_slope)
                avg_slope = (abs(upper_slope) + abs(lower_slope)) / 2
                
                if avg_slope > 0:
                    parallelism = slope_diff / avg_slope
                    if parallelism > PARALLEL_TOLERANCE:
                        print(f"  💡 Reason: Lines not parallel enough ({parallelism:.1%} > {PARALLEL_TOLERANCE:.1%})")
                        return
                
                # Check touches
                upper_touches = self.count_line_touches(df, upper_slope, upper_intercept, True)
                lower_touches = self.count_line_touches(df, lower_slope, lower_intercept, False)
                
                if upper_touches < MIN_CHANNEL_TOUCHES:
                    print(f"  💡 Reason: Upper line weak ({upper_touches} touches < {MIN_CHANNEL_TOUCHES} required)")
                elif lower_touches < MIN_CHANNEL_TOUCHES:
                    print(f"  💡 Reason: Lower line weak ({lower_touches} touches < {MIN_CHANNEL_TOUCHES} required)")
                else:
                    # Check channel width
                    current_idx = len(df) - 1
                    upper_current = upper_slope * current_idx + upper_intercept
                    lower_current = lower_slope * current_idx + lower_intercept
                    channel_width = upper_current - lower_current
                    current_price = df['close'].iloc[-1]
                    width_percent = (channel_width / current_price) * 100
                    
                    if width_percent < MIN_CHANNEL_WIDTH_PERCENT:
                        print(f"  💡 Reason: Channel too narrow ({width_percent:.1f}% < {MIN_CHANNEL_WIDTH_PERCENT}%)")
                    else:
                        print(f"  💡 Reason: Channel validation failed (unknown technical issue)")
        
        except Exception as e:
            print(f"  💡 Reason: Analysis error - price action too volatile")

    def display_portfolio_summary(self):
        """Display P&L summary"""
        mode_prefix = "🔸 [DRY RUN]" if DRY_RUN else ""
        session_file = "trader_channel/session_pnl_dry.json" if DRY_RUN else "trader_channel/session_pnl.json"
        
        print("")
        print("=" * 80)
        print(colored(f"💼 {mode_prefix} PORTFOLIO SUMMARY", "cyan", attrs=["bold"]))
        print("=" * 80)
        
        # Count open positions
        open_positions = sum(1 for s in trader_state.values() if s["position_is_open"])
        
        # Load session data
        total_realized = 0
        total_trades = 0
        profitable_trades = 0
        
        if os.path.exists(session_file):
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                    total_realized = session_data.get("total_realized", 0)
                    total_trades = session_data.get("total_trades", 0)
                    profitable_trades = session_data.get("profitable_trades", 0)
            except:
                pass
        
        # Calculate unrealized P&L
        total_unrealized = 0
        for symbol, state in trader_state.items():
            if state["position_is_open"]:
                try:
                    ticker = client.get_symbol_ticker(symbol=symbol)
                    current_price = float(ticker['price'])
                    remaining_qty = state.get("remaining_quantity") or state["quantity"]
                    entry_price = state["entry_price"]
                    
                    unrealized = (current_price - entry_price) * remaining_qty
                    fees = current_price * remaining_qty * 0.001
                    unrealized -= fees
                    total_unrealized += unrealized
                except:
                    pass
        
        # Display summary
        print(f"Open Positions: {open_positions}")
        print(f"Total Trades: {total_trades}")
        if total_trades > 0:
            win_rate = (profitable_trades / total_trades) * 100
            print(f"Win Rate: {win_rate:.1f}% ({profitable_trades}/{total_trades})")
        
        print(f"Realized P&L: {colored(f'${total_realized:+.2f}', 'green' if total_realized > 0 else 'red')}")
        print(f"Unrealized P&L: {colored(f'${total_unrealized:+.2f}', 'green' if total_unrealized > 0 else 'red')}")
        
        total_pnl = total_realized + total_unrealized
        print(f"Total P&L: {colored(f'${total_pnl:+.2f}', 'green' if total_pnl > 0 else 'red', attrs=['bold'])}")
        print("=" * 80)
    
    def wait_for_next_candle(self):
        """Wait until next candle close + 30 seconds buffer"""
        now = datetime.datetime.now()
        minutes = now.minute
        seconds = now.second
        
        # Calculate minutes until next timeframe mark
        mins_to_wait = TIMEFRAME_MINUTES - (minutes % TIMEFRAME_MINUTES)
        if mins_to_wait == TIMEFRAME_MINUTES and seconds < 30:
            mins_to_wait = 0
        
        # Total seconds to wait (including 30 second buffer)
        total_seconds = mins_to_wait * 60 - seconds + 30
        
        if total_seconds > 0:
            timeframe_text = f"{TIMEFRAME_MINUTES}min" if TIMEFRAME_MINUTES < 60 else f"{TIMEFRAME_MINUTES//60}h"
            print(f"⏰ Waiting {total_seconds} seconds for next {timeframe_text} candle...")
            time.sleep(total_seconds)
    
    def run(self):
        """Main trading loop"""
        mode_text = "DRY RUN MODE - No real trades" if DRY_RUN else "LIVE TRADING MODE"
        mode_color = "yellow" if DRY_RUN else "red"
        timeframe_text = f"{TIMEFRAME_MINUTES}min" if TIMEFRAME_MINUTES < 60 else f"{TIMEFRAME_MINUTES//60}h"
        
        print("=" * 80)
        print(colored(f"📊 CHANNEL BREAKOUT TRADER ({timeframe_text.upper()})", "cyan", attrs=["bold"]))
        print(colored(f"🔸 {mode_text}", mode_color, attrs=["bold"]))
        print("=" * 80)
        
        print("\n" + colored("STRATEGY CONFIGURATION", "yellow", attrs=["bold"]))
        print("─" * 80)
        print(f"⏰ Timeframe: {colored(timeframe_text.upper(), 'cyan', attrs=['bold'])} candles")
        print(f"📈 Channel Detection: {CHANNEL_LOOKBACK} candles, {MIN_CHANNEL_TOUCHES} touches minimum")
        print(f"🎯 Entry Signals: Breakout, Lower Bounce, Midline Bounce")
        print(f"💰 Position Size: ${MIN_POSITION_VALUE}-${MAX_POSITION_VALUE}")
        print(f"📊 Risk/Reward: Minimum {MIN_RISK_REWARD}:1")
        print(f"⚡ Volume Filter: {VOLUME_SPIKE_MULTIPLIER}x for breakouts")
        print(f"🎯 Targets: TP1={TP1_CHANNEL_RATIO}x width, TP2={TP2_CHANNEL_RATIO}x width")
        
        # Timeframe-specific info
        if TIMEFRAME_MINUTES <= 5:
            print(f"⚡ {colored('SCALPING MODE', 'yellow')}: Fast signals, tight stops, quick exits")
        elif TIMEFRAME_MINUTES <= 15:
            print(f"📊 {colored('ACTIVE TRADING', 'green')}: Balanced approach, good risk/reward")
        elif TIMEFRAME_MINUTES <= 60:
            print(f"📈 {colored('SWING TRADING', 'blue')}: Higher quality signals, longer holds")
        else:
            print(f"🏔️ {colored('POSITION TRADING', 'purple')}: Strong trends, multi-day holds")
            
        print("=" * 80)
        
        # Initial wait
        self.wait_for_next_candle()
        
        while not shutdown:
            current_time = datetime.datetime.now()
            mode_prefix = "🔸 [DRY RUN]" if DRY_RUN else ""
            
            timeframe_text = f"{TIMEFRAME_MINUTES}min" if TIMEFRAME_MINUTES < 60 else f"{TIMEFRAME_MINUTES//60}h"
            print(f"\n=== {mode_prefix} {timeframe_text.upper()} Channel Analysis: {current_time.strftime('%Y-%m-%d %H:%M:%S')} ===")
            
            for symbol in symbols:
                try:
                    # Get current price
                    ticker = client.get_symbol_ticker(symbol=symbol)
                    current_price = float(ticker['price'])
                    
                    # Check existing position
                    if trader_state[symbol]["position_is_open"]:
                        # Display position status
                        entry_price = trader_state[symbol]["entry_price"]
                        pnl_percent = ((current_price - entry_price) / entry_price) * 100
                        
                        print(f"\n[{symbol}] Price: {colored(f'{current_price:.6f}', 'cyan')}")
                        
                        strategy_emojis = {
                            "channel_breakout": "📈",
                            "channel_bounce": "🔄",
                            "midline_bounce": "➖"
                        }
                        emoji = strategy_emojis.get(trader_state[symbol].get("entry_type"), "📊")
                        
                        print(f"  Position: {colored('OPEN', 'yellow')} {emoji} (Entry: {entry_price:.6f})")
                        print(f"  P&L: {colored(f'{pnl_percent:+.2f}%', 'green' if pnl_percent > 0 else 'red')}")
                        
                        if trader_state[symbol].get("tp1_hit"):
                            print(f"  Status: {colored('🚀 TRAILING', 'cyan')}")
                        
                        # Check exit conditions
                        self.check_exit_conditions(symbol, current_price)
                        
                    else:
                        # Look for new opportunities
                        df = self.get_candle_data(symbol)
                        if len(df) > 0:
                            channel = self.detect_channel(df)
                            
                            if channel:
                                # Display channel info
                                current_idx = len(df) - 1
                                upper = channel.upper_slope * current_idx + channel.upper_intercept
                                lower = channel.lower_slope * current_idx + channel.lower_intercept
                                midline = (upper + lower) / 2
                                
                                print(f"\n[{symbol}] Price: {colored(f'{current_price:.6f}', 'cyan')}")
                                print(f"  📊 Channel: Upper={upper:.6f} | Lower={lower:.6f} | Width={channel.channel_width:.6f}")
                                print(f"  Touches: Upper={channel.touches_upper} | Lower={channel.touches_lower}")
                                print(f"  Trend: {colored(channel.trend.upper(), 'green' if channel.trend == 'bullish' else 'red' if channel.trend == 'bearish' else 'yellow')}")
                                
                                # Add detailed explanation
                                self.explain_channel_analysis(symbol, current_price, upper, lower, midline, channel)
                                
                                # Check for opportunity
                                opportunity = self.find_channel_opportunity(symbol, df, channel)
                                
                                if opportunity and opportunity['confidence'] >= 70:
                                    self.execute_trade(symbol, opportunity)
                                else:
                                    if opportunity:
                                        print(f"  ⚠️ Signal found but confidence too low ({opportunity['confidence']:.0f}% < 70%)")
                                    
                            else:
                                print(f"\n[{symbol}] No valid channel detected")
                                # Explain why no channel was detected
                                self.explain_no_channel(symbol)
                
                except Exception as e:
                    print(f"\n[{symbol}] Error: {colored(str(e), 'red')}")
            
            # Display portfolio summary
            self.display_portfolio_summary()
            
            # Wait for next candle
            self.wait_for_next_candle()
        
        print(f"\n{colored('📊 Channel Trader stopped', 'yellow')}")

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown
    print(f"\n{colored('Shutting down...', 'yellow')}")
    shutdown = True
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check Binance connection
    try:
        account = client.get_account()
        balance = next((b for b in account['balances'] if b['asset'] == 'USDC'), None)
        
        mode_text = "DRY RUN MODE" if DRY_RUN else "LIVE TRADING"
        
        if balance:
            print(colored(f"✅ Binance connected - {mode_text}", "green"))
            print(f"USDC Balance: ${float(balance['free']):.2f}")
        else:
            print("❌ USDC balance not found")
            
    except Exception as e:
        print(f"❌ Binance connection failed: {e}")
        sys.exit(1)
    
    # Start trader
    trader = ChannelTrader()
    trader.run()