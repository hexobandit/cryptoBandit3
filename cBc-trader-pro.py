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
import tenacity
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
sys.path.append('../')
from _secrets import api_key, secret_key

# ==================== CONFIGURATION ====================
# TRADING MODE - IMPORTANT: Set to True to prevent real money losses!
DRY_RUN = False  # Set to False for LIVE trading with real money

# WARNING: With DRY_RUN = False, this bot will execute REAL trades!
# Only set to False when you're confident in the strategy
# ========================================================

# Professional Trader Data Classes
@dataclass
class TrendLine:
    """Represents a trend line with two points"""
    start_price: float
    start_time: datetime.datetime
    end_price: float
    end_time: datetime.datetime
    slope: float
    strength: int  # How many touches
    
    def get_price_at_time(self, target_time: datetime.datetime) -> float:
        """Calculate expected price at given time"""
        time_diff = (target_time - self.start_time).total_seconds()
        start_time_diff = (self.end_time - self.start_time).total_seconds()
        if start_time_diff == 0:
            return self.start_price
        ratio = time_diff / start_time_diff
        return self.start_price + (self.end_price - self.start_price) * ratio

@dataclass
class KeyLevel:
    """Represents a support/resistance level"""
    price: float
    level_type: str  # 'support' or 'resistance'
    strength: int  # Number of touches
    last_touch: datetime.datetime
    origin: str  # 'swing_high', 'swing_low', 'previous_high', etc.

@dataclass
class MarketStructure:
    """Current market structure analysis"""
    trend: str  # 'uptrend', 'downtrend', 'ranging'
    last_higher_high: Optional[float] = None
    last_higher_low: Optional[float] = None
    last_lower_high: Optional[float] = None
    last_lower_low: Optional[float] = None
    structure_break: bool = False
    break_direction: Optional[str] = None

# Trading symbols
symbols = [
    "BTCUSDC", "ETHUSDC", "BNBUSDC", "ADAUSDC", "XRPUSDC", 
    "DOGEUSDC", "SOLUSDC", "PNUTUSDC", "PEPEUSDC", "SHIBUSDC", 
    "XLMUSDC", "LINKUSDC", "IOTAUSDC", "ENAUSDC"
]

shutdown = False

def handle_exit(sig, frame):
    global shutdown
    print("\\n🔻 Professional Trader shutting down...")
    shutdown = True

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

# Professional Trader State
# Determine file suffix based on mode
mode_suffix = "_dry" if DRY_RUN else "_live"

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
        "risk_reward_ratio": None,
        "entry_reason": None,
        
        # Progressive trailing stop
        "tp1_hit": False,
        "highest_price": None,
        "trailing_stop": None,
        "trailing_distance": None,  # Current trailing distance percentage
        
        # Partial position management
        "original_quantity": None,  # Initial position size
        "tp1_executed_quantity": None,  # Amount sold at TP1
        "remaining_quantity": None,  # Current position size
        "realized_profit": 0,  # Profit from partial sells
        "tp1_exit_price": None,  # Actual TP1 exit price
        
        # Technical analysis
        "swing_highs": deque(maxlen=20),
        "swing_lows": deque(maxlen=20),
        "trend_lines": [],
        "key_levels": [],
        "market_structure": MarketStructure(trend="ranging"),
        "current_trend": "neutral",
        
        # Risk management
        "position_size_usd": 0,
        "risk_per_trade": 100,  # Max risk in USD
        "min_rr_ratio": 2.0,   # Minimum risk/reward ratio
        
        # Cumulative tracking (never reset)
        "total_realized_pnl": 0,  # Cumulative P&L from all closed trades
        
        # Files - separate for dry and live mode
        "filename_position": f"trader_pro/position_{symbol}{mode_suffix}.json",
        "filename_analysis": f"trader_pro/analysis_{symbol}{mode_suffix}.json",
        "filename_trades": f"trader_pro/trades_{symbol}{mode_suffix}.json",
    }
    for symbol in symbols
}

# Create directories
os.makedirs("trader_pro", exist_ok=True)

# Trading constants
TIMEFRAME = Client.KLINE_INTERVAL_15MINUTE  # 15-minute for professional analysis
LOOKBACK_CANDLES = 100  # 25 hours of 15m data
CHECK_INTERVAL = 15  # Check every 15 minutes
MIN_POSITION_VALUE = 50  # Minimum $50 position
MAX_POSITION_VALUE = 100  # Maximum $100 position per trade

# Technical analysis parameters
SWING_LOOKBACK = 5  # Candles to look back/forward for swing points
TRENDLINE_MIN_TOUCHES = 2  # Minimum touches to confirm trend line
LEVEL_PROXIMITY_PERCENT = 0.3  # 0.3% proximity to consider level touched
BREAKOUT_CONFIRMATION_PERCENT = 0.2  # 0.2% break beyond level for confirmation

# Progressive trailing stop parameters
TRAIL_INITIAL = 0.8  # 0.8% trail when profit < 1%
TRAIL_MEDIUM = 0.5   # 0.5% trail when profit 1-2%
TRAIL_TIGHT = 0.3    # 0.3% trail when profit > 2%

# Partial exit parameters
TP1_SELL_PERCENT = 0.5  # Sell 50% at TP1
TP1_PROFIT_TARGET = 0.01  # 1% profit for TP1
TP2_OPTIONAL_TARGET = 0.03  # Optional 3% for another partial (not fixed exit)

# Set up Binance client
client = Client(api_key, secret_key)

class ProfessionalTrader:
    def __init__(self):
        self.load_states()
    
    def load_states(self):
        """Load previous trading states"""
        for symbol in symbols:
            try:
                # Load position state
                if os.path.exists(trader_state[symbol]["filename_position"]):
                    with open(trader_state[symbol]["filename_position"], 'r') as f:
                        position_data = json.load(f)
                        if position_data:
                            trader_state[symbol].update(position_data)
            except Exception as e:
                print(f"Warning: Could not load state for {symbol}: {e}")
    
    def save_position_state(self, symbol):
        """Save current position state"""
        state_to_save = {
            "position_is_open": trader_state[symbol]["position_is_open"],
            "entry_price": trader_state[symbol]["entry_price"],
            "quantity": trader_state[symbol]["quantity"],
            "entry_time": trader_state[symbol]["entry_time"],
            "stop_loss": trader_state[symbol]["stop_loss"],
            "take_profit_1": trader_state[symbol]["take_profit_1"],
            "take_profit_2": trader_state[symbol]["take_profit_2"],
            "risk_reward_ratio": trader_state[symbol]["risk_reward_ratio"],
            "entry_reason": trader_state[symbol]["entry_reason"],
            "tp1_hit": trader_state[symbol].get("tp1_hit", False),
            "highest_price": trader_state[symbol].get("highest_price"),
            "trailing_stop": trader_state[symbol].get("trailing_stop"),
            "trailing_distance": trader_state[symbol].get("trailing_distance"),
            "original_quantity": trader_state[symbol].get("original_quantity"),
            "tp1_executed_quantity": trader_state[symbol].get("tp1_executed_quantity"),
            "remaining_quantity": trader_state[symbol].get("remaining_quantity"),
            "realized_profit": trader_state[symbol].get("realized_profit", 0),
            "tp1_exit_price": trader_state[symbol].get("tp1_exit_price"),
            "total_realized_pnl": trader_state[symbol].get("total_realized_pnl", 0),
            "entry_type": trader_state[symbol].get("entry_type"),
        }
        
        with open(trader_state[symbol]["filename_position"], 'w') as f:
            json.dump(state_to_save, f, indent=2)
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_ema(self, prices: pd.Series, span: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=span, adjust=False).mean()
    
    def check_btc_market_condition(self) -> dict:
        """Check BTC market condition to filter alt trades"""
        try:
            # Get BTC 15-minute data
            btc_df = self.get_candle_data("BTCUSDC", limit=100)
            if len(btc_df) < 50:
                return {"safe_for_alts": True, "btc_trend": "unknown", "btc_1h_change": 0}
            
            # Calculate BTC EMAs
            btc_ema20 = self.calculate_ema(btc_df['close'], 20)
            btc_ema50 = self.calculate_ema(btc_df['close'], 50)
            
            # BTC current price
            btc_current = btc_df['close'].iloc[-1]
            
            # BTC 1 hour change (4 candles ago)
            if len(btc_df) >= 4:
                btc_1h_ago = btc_df['close'].iloc[-5]  # 1 hour ago (4 * 15min)
                btc_1h_change = ((btc_current - btc_1h_ago) / btc_1h_ago) * 100
            else:
                btc_1h_change = 0
            
            # BTC 4 hour trend
            if len(btc_df) >= 16:
                btc_4h_ago = btc_df['close'].iloc[-17]  # 4 hours ago
                btc_4h_change = ((btc_current - btc_4h_ago) / btc_4h_ago) * 100
            else:
                btc_4h_change = 0
            
            # Calculate BTC RSI
            btc_rsi = self.calculate_rsi(btc_df['close'], 14)
            btc_current_rsi = btc_rsi.iloc[-1] if len(btc_rsi) > 0 else 50
            
            # Determine if safe for alt trading
            safe_for_alts = True
            btc_status = "neutral"
            
            # RED FLAGS - Don't trade alts when:
            if btc_1h_change < -1.5:  # BTC dumping >1.5% in 1 hour
                safe_for_alts = False
                btc_status = "dumping"
            elif btc_ema20.iloc[-1] < btc_ema50.iloc[-1] and btc_current < btc_ema20.iloc[-1]:
                # BTC in downtrend AND below both EMAs
                safe_for_alts = False
                btc_status = "bearish"
            elif btc_current_rsi < 30:  # BTC oversold - wait for bounce
                safe_for_alts = False
                btc_status = "oversold"
            elif btc_4h_change < -3:  # BTC down >3% in 4 hours
                safe_for_alts = False
                btc_status = "declining"
            
            # GREEN FLAGS - Good for alts
            if btc_ema20.iloc[-1] > btc_ema50.iloc[-1]:
                btc_status = "bullish"
            
            return {
                "safe_for_alts": safe_for_alts,
                "btc_trend": btc_status,
                "btc_1h_change": btc_1h_change,
                "btc_4h_change": btc_4h_change,
                "btc_rsi": btc_current_rsi,
                "btc_price": btc_current
            }
            
        except Exception as e:
            print(f"Warning: Could not check BTC condition: {e}")
            return {"safe_for_alts": True, "btc_trend": "unknown", "btc_1h_change": 0}
    
    def get_candle_data(self, symbol: str, limit: int = LOOKBACK_CANDLES) -> pd.DataFrame:
        """Fetch professional-grade candle data"""
        try:
            klines = client.get_historical_klines(
                symbol, TIMEFRAME, f"{limit * 15} minutes ago UTC"
            )
            
            df = pd.DataFrame(klines, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])
            
            # Convert to proper types
            df["open"] = df["open"].astype(float)
            df["high"] = df["high"].astype(float)
            df["low"] = df["low"].astype(float)
            df["close"] = df["close"].astype(float)
            df["volume"] = df["volume"].astype(float)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def identify_swing_points(self, df: pd.DataFrame) -> Tuple[List[Tuple], List[Tuple]]:
        """Identify swing highs and lows like a professional trader"""
        swing_highs = []
        swing_lows = []
        
        if len(df) < SWING_LOOKBACK * 2 + 1:
            return swing_highs, swing_lows
        
        for i in range(SWING_LOOKBACK, len(df) - SWING_LOOKBACK):
            # Check for swing high
            is_swing_high = True
            current_high = df.iloc[i]['high']
            
            # Must be higher than surrounding candles
            for j in range(i - SWING_LOOKBACK, i + SWING_LOOKBACK + 1):
                if j != i and df.iloc[j]['high'] >= current_high:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing_highs.append((df.iloc[i]['timestamp'], current_high, i))
            
            # Check for swing low
            is_swing_low = True
            current_low = df.iloc[i]['low']
            
            # Must be lower than surrounding candles
            for j in range(i - SWING_LOOKBACK, i + SWING_LOOKBACK + 1):
                if j != i and df.iloc[j]['low'] <= current_low:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_lows.append((df.iloc[i]['timestamp'], current_low, i))
        
        return swing_highs[-10:], swing_lows[-10:]  # Keep last 10 of each
    
    def draw_trend_lines(self, swing_points: List[Tuple]) -> List[TrendLine]:
        """Draw trend lines like a professional trader"""
        trend_lines = []
        
        if len(swing_points) < 2:
            return trend_lines
        
        # Try to connect swing points to form trend lines
        for i in range(len(swing_points) - 1):
            for j in range(i + 1, len(swing_points)):
                point1 = swing_points[i]
                point2 = swing_points[j]
                
                # Calculate slope
                time_diff = (point2[0] - point1[0]).total_seconds()
                if time_diff <= 0:
                    continue
                
                price_diff = point2[1] - point1[1]
                slope = price_diff / time_diff
                
                # Count how many other points are near this line
                touches = 2  # The two points used to create the line
                tolerance = abs(point2[1] - point1[1]) * 0.002  # 0.2% tolerance
                
                for k, point in enumerate(swing_points):
                    if k == i or k == j:
                        continue
                    
                    # Calculate expected price on line at this time
                    time_on_line = (point[0] - point1[0]).total_seconds()
                    expected_price = point1[1] + slope * time_on_line
                    
                    if abs(point[1] - expected_price) <= tolerance:
                        touches += 1
                
                if touches >= TRENDLINE_MIN_TOUCHES:
                    trend_line = TrendLine(
                        start_price=point1[1],
                        start_time=point1[0],
                        end_price=point2[1],
                        end_time=point2[0],
                        slope=slope,
                        strength=touches
                    )
                    trend_lines.append(trend_line)
        
        # Sort by strength and return top lines
        trend_lines.sort(key=lambda x: x.strength, reverse=True)
        return trend_lines[:3]  # Keep top 3 strongest lines
    
    def identify_key_levels(self, df: pd.DataFrame, swing_highs: List[Tuple], swing_lows: List[Tuple]) -> List[KeyLevel]:
        """Identify key support/resistance levels"""
        key_levels = []
        current_price = df['close'].iloc[-1]
        current_time = df['timestamp'].iloc[-1]
        
        # Process swing highs as potential resistance
        for timestamp, price, index in swing_highs:
            # Count how many times this level was tested
            touches = 1
            for _, other_price, _ in swing_highs:
                if abs(other_price - price) / price < LEVEL_PROXIMITY_PERCENT / 100:
                    touches += 1
            
            level_type = "resistance" if price > current_price else "support"
            key_levels.append(KeyLevel(
                price=price,
                level_type=level_type,
                strength=touches,
                last_touch=timestamp,
                origin="swing_high"
            ))
        
        # Process swing lows as potential support
        for timestamp, price, index in swing_lows:
            touches = 1
            for _, other_price, _ in swing_lows:
                if abs(other_price - price) / price < LEVEL_PROXIMITY_PERCENT / 100:
                    touches += 1
            
            level_type = "support" if price < current_price else "resistance"
            key_levels.append(KeyLevel(
                price=price,
                level_type=level_type,
                strength=touches,
                last_touch=timestamp,
                origin="swing_low"
            ))
        
        # Add previous day high/low as key levels
        if len(df) >= 96:  # 24 hours of 15m candles
            day_ago_start = len(df) - 96
            prev_day_high = df['high'].iloc[day_ago_start:day_ago_start+96].max()
            prev_day_low = df['low'].iloc[day_ago_start:day_ago_start+96].min()
            
            key_levels.extend([
                KeyLevel(prev_day_high, "resistance" if prev_day_high > current_price else "support", 
                        3, current_time, "previous_day_high"),
                KeyLevel(prev_day_low, "support" if prev_day_low < current_price else "resistance", 
                        3, current_time, "previous_day_low")
            ])
        
        # Remove duplicates and sort by strength
        unique_levels = []
        for level in key_levels:
            is_duplicate = False
            for unique_level in unique_levels:
                if abs(level.price - unique_level.price) / level.price < LEVEL_PROXIMITY_PERCENT / 100:
                    is_duplicate = True
                    if level.strength > unique_level.strength:
                        unique_levels.remove(unique_level)
                        unique_levels.append(level)
                    break
            if not is_duplicate:
                unique_levels.append(level)
        
        # Sort by strength and proximity to current price
        unique_levels.sort(key=lambda x: (x.strength, -abs(x.price - current_price) / current_price), reverse=True)
        return unique_levels[:5]  # Keep top 5 levels
    
    def analyze_market_structure(self, swing_highs: List[Tuple], swing_lows: List[Tuple]) -> MarketStructure:
        """Analyze market structure like a professional trader"""
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return MarketStructure(trend="ranging")
        
        # Sort by time
        highs_sorted = sorted(swing_highs, key=lambda x: x[0])
        lows_sorted = sorted(swing_lows, key=lambda x: x[0])
        
        structure = MarketStructure(trend="ranging")
        
        # Analyze recent highs
        if len(highs_sorted) >= 2:
            recent_high = highs_sorted[-1][1]
            previous_high = highs_sorted[-2][1]
            
            if recent_high > previous_high:
                structure.last_higher_high = recent_high
            else:
                structure.last_lower_high = recent_high
        
        # Analyze recent lows
        if len(lows_sorted) >= 2:
            recent_low = lows_sorted[-1][1]
            previous_low = lows_sorted[-2][1]
            
            if recent_low > previous_low:
                structure.last_higher_low = recent_low
            else:
                structure.last_lower_low = recent_low
        
        # Determine trend
        if structure.last_higher_high and structure.last_higher_low:
            structure.trend = "uptrend"
        elif structure.last_lower_high and structure.last_lower_low:
            structure.trend = "downtrend"
        else:
            structure.trend = "ranging"
        
        return structure
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, risk_amount: float) -> float:
        """Calculate position size based on risk management"""
        if entry_price <= 0 or stop_loss <= 0 or risk_amount <= 0:
            return 0
        
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit == 0:
            return 0
        
        position_size = risk_amount / risk_per_unit
        
        # Ensure position value is within limits
        position_value = position_size * entry_price
        if position_value < MIN_POSITION_VALUE:
            position_size = MIN_POSITION_VALUE / entry_price
        elif position_value > MAX_POSITION_VALUE:
            position_size = MAX_POSITION_VALUE / entry_price
        
        return position_size
    
    def find_trading_opportunity(self, symbol: str, df: pd.DataFrame) -> Optional[dict]:
        """Find trading opportunities like a professional trader"""
        if len(df) < 20:
            return None
        
        # Use PREVIOUS completed candle for analysis
        analysis_price = df['close'].iloc[-2]  # Previous candle close
        analysis_time = df['timestamp'].iloc[-2]  # Previous candle time
        
        # But use CURRENT price for actual trading decisions
        current_price = df['close'].iloc[-1]  # Current price for entry
        
        # Check BTC market condition for alt coins
        is_altcoin = symbol != "BTCUSDC"
        btc_condition = None
        
        if is_altcoin:
            btc_condition = self.check_btc_market_condition()
            
            # Skip alt trading if BTC is not favorable
            if not btc_condition["safe_for_alts"]:
                # Log why we're skipping
                if btc_condition["btc_trend"] == "dumping":
                    print(f"   ⚠️ BTC dumping ({btc_condition['btc_1h_change']:.1f}% 1H) - Skipping alt trades")
                elif btc_condition["btc_trend"] == "bearish":
                    print(f"   ⚠️ BTC bearish trend - Skipping alt trades")
                elif btc_condition["btc_trend"] == "oversold":
                    print(f"   ⚠️ BTC oversold (RSI: {btc_condition['btc_rsi']:.0f}) - Waiting for bounce")
                else:
                    print(f"   ⚠️ BTC declining - Skipping alt trades")
                return None
        
        # Get technical analysis (using all candles including previous)
        swing_highs, swing_lows = self.identify_swing_points(df)
        trend_lines = self.draw_trend_lines(swing_highs + swing_lows)
        key_levels = self.identify_key_levels(df, swing_highs, swing_lows)
        market_structure = self.analyze_market_structure(swing_highs, swing_lows)
        
        # Store analysis
        trader_state[symbol]["swing_highs"] = swing_highs
        trader_state[symbol]["swing_lows"] = swing_lows
        trader_state[symbol]["trend_lines"] = trend_lines
        trader_state[symbol]["key_levels"] = key_levels
        trader_state[symbol]["market_structure"] = market_structure
        
        opportunities = []
        
        # Strategy 1: Trend Line Breakout (check on PREVIOUS candle)
        for trend_line in trend_lines:
            if trend_line.strength >= 3:  # Strong trend line
                # Get trend line price at PREVIOUS candle time
                trend_line_at_analysis = trend_line.get_price_at_time(analysis_time)
                breakout_threshold = trend_line_at_analysis * (1 + BREAKOUT_CONFIRMATION_PERCENT / 100)
                breakdown_threshold = trend_line_at_analysis * (1 - BREAKOUT_CONFIRMATION_PERCENT / 100)
                
                # Check if PREVIOUS candle closed above trend line (confirmed breakout)
                if analysis_price > breakout_threshold and trend_line.slope > 0:
                    # Find stop loss below trend line (use current position)
                    current_line_price = trend_line.get_price_at_time(df['timestamp'].iloc[-1])
                    stop_loss = current_line_price * 0.99  # 1.0% below line
                    
                    # Find resistance for take profit
                    resistance_levels = [level for level in key_levels if level.level_type == "resistance" and level.price > current_price]
                    if resistance_levels:
                        take_profit = resistance_levels[0].price * 0.995
                        risk_reward = (take_profit - current_price) / (current_price - stop_loss)
                        
                        if risk_reward >= trader_state[symbol]["min_rr_ratio"]:
                            opportunities.append({
                                "type": "trend_line_breakout",
                                "direction": "long",
                                "entry_price": current_price,
                                "stop_loss": stop_loss,
                                "take_profit_1": current_price + (take_profit - current_price) * 0.5,
                                "take_profit_2": take_profit,
                                "risk_reward": risk_reward,
                                "reason": f"Bullish breakout above trend line (strength: {trend_line.strength})",
                                "confidence": min(trend_line.strength * 20, 100)
                            })
        
        # Strategy 2: Support/Resistance Bounce (check on PREVIOUS candle)
        for level in key_levels:
            # Check proximity of PREVIOUS candle to level
            proximity = abs(analysis_price - level.price) / analysis_price
            
            if proximity < LEVEL_PROXIMITY_PERCENT / 100:  # Previous candle was near key level
                if level.level_type == "support" and market_structure.trend in ["uptrend", "ranging"]:
                    # Long from support
                    stop_loss = level.price * 0.988  # 1.2% below support
                    
                    # Find next resistance
                    resistance_levels = [l for l in key_levels if l.level_type == "resistance" and l.price > current_price]
                    if resistance_levels:
                        take_profit = resistance_levels[0].price * 0.995
                        risk_reward = (take_profit - current_price) / (current_price - stop_loss)
                        
                        if risk_reward >= trader_state[symbol]["min_rr_ratio"]:
                            opportunities.append({
                                "type": "support_bounce",
                                "direction": "long",
                                "entry_price": current_price,
                                "stop_loss": stop_loss,
                                "take_profit_1": current_price + (take_profit - current_price) * 0.4,
                                "take_profit_2": take_profit,
                                "risk_reward": risk_reward,
                                "reason": f"Bounce from {level.origin} support (strength: {level.strength})",
                                "confidence": level.strength * 25
                            })
                
        
        # Strategy 3: RSI Oversold Bounce
        if len(df) >= 14:  # Need enough data for RSI
            rsi = self.calculate_rsi(df['close'], 14)
            current_rsi = rsi.iloc[-1]
            prev_rsi = rsi.iloc[-2]
            
            # Extreme oversold condition on PREVIOUS candle
            if prev_rsi < 25 and analysis_price > df['low'].iloc[-3] * 0.998:
                # Price stopped falling (not making significant new lows)
                stop_loss = df['low'].iloc[-2] * 0.985  # 1.5% below previous candle low
                
                # Conservative target for oversold bounces
                take_profit = current_price * 1.018  # 1.8% target
                risk_reward = (take_profit - current_price) / (current_price - stop_loss)
                
                if risk_reward >= 2.0:
                    opportunities.append({
                        "type": "rsi_oversold_bounce",
                        "direction": "long",
                        "entry_price": current_price,
                        "stop_loss": stop_loss,
                        "take_profit_1": current_price * 1.01,  # TP1 at 1%
                        "take_profit_2": take_profit,
                        "risk_reward": risk_reward,
                        "reason": f"Extreme oversold bounce (RSI: {prev_rsi:.1f})",
                        "confidence": min(90, 100 - prev_rsi * 2)  # Lower RSI = higher confidence
                    })
        
        # Strategy 4: Volume Spike Momentum
        if len(df) >= 20:
            avg_volume = df['volume'].iloc[-20:-2].mean()  # 20-candle average volume
            prev_volume = df['volume'].iloc[-2]
            
            # Check for massive volume spike on previous candle
            if prev_volume > avg_volume * 2.5:  # 2.5x average volume
                if analysis_price > df['open'].iloc[-2]:  # Green candle
                    # Big players entered - follow them
                    stop_loss = df['low'].iloc[-2] * 0.99  # 1.0% below the volume candle
                    take_profit = current_price * 1.025  # 2.5% target for momentum
                    risk_reward = (take_profit - current_price) / (current_price - stop_loss)
                    
                    if risk_reward >= 2.0:
                        opportunities.append({
                            "type": "volume_breakout",
                            "direction": "long",
                            "entry_price": current_price,
                            "stop_loss": stop_loss,
                            "take_profit_1": current_price * 1.012,  # TP1 at 1.2%
                            "take_profit_2": take_profit,
                            "risk_reward": risk_reward,
                            "reason": f"Volume spike {prev_volume/avg_volume:.1f}x average",
                            "confidence": min(85, 50 + prev_volume/avg_volume * 10)
                        })
        
        # Strategy 5: Liquidity Hunt Recovery (Stop Hunt)
        if len(df) >= 3:
            two_candles_ago_low = df['low'].iloc[-3]
            prev_candle_low = df['low'].iloc[-2]
            prev_candle_close = df['close'].iloc[-2]
            prev_candle_open = df['open'].iloc[-2]
            
            # Check for stop hunt pattern
            if prev_candle_low < two_candles_ago_low * 0.997:  # Wick below support
                if prev_candle_close > two_candles_ago_low:  # Recovered above
                    wick_size = prev_candle_close - prev_candle_low
                    body_size = abs(prev_candle_close - prev_candle_open)
                    
                    if wick_size > body_size * 2:  # Long lower wick (2x body)
                        stop_loss = prev_candle_low * 0.992  # 0.8% below the hunt low
                        take_profit = current_price * 1.02  # 2% target
                        risk_reward = (take_profit - current_price) / (current_price - stop_loss)
                        
                        if risk_reward >= 2.0:
                            opportunities.append({
                                "type": "liquidity_hunt",
                                "direction": "long",
                                "entry_price": current_price,
                                "stop_loss": stop_loss,
                                "take_profit_1": current_price * 1.01,
                                "take_profit_2": take_profit,
                                "risk_reward": risk_reward,
                                "reason": "Stop hunt recovery - whale accumulation",
                                "confidence": 80
                            })
        
        
        # Return best opportunity
        if opportunities:
            best_opportunity = max(opportunities, key=lambda x: x["confidence"] * x["risk_reward"])
            return best_opportunity if best_opportunity["confidence"] > 60 else None
        
        return None
    
    def execute_trade(self, symbol: str, opportunity: dict) -> bool:
        """Execute trade like a professional trader"""
        try:
            if opportunity["direction"] == "long":
                # Calculate position size
                risk_amount = trader_state[symbol]["risk_per_trade"]
                position_size = self.calculate_position_size(
                    opportunity["entry_price"], 
                    opportunity["stop_loss"], 
                    risk_amount
                )
                
                if position_size <= 0:
                    return False
                
                # Display strategy type with emoji
                strategy_emojis = {
                    "support_bounce": "🔄",
                    "trend_line_breakout": "📈",
                    "rsi_oversold_bounce": "📉",
                    "volume_breakout": "📊",
                    "liquidity_hunt": "🎣"
                }
                strategy_emoji = strategy_emojis.get(opportunity["type"], "🎯")
                strategy_name = opportunity["type"].replace("_", " ").title()
                
                mode_prefix = "🔸 [DRY RUN]" if DRY_RUN else ""
                print(f"{colored(f'{mode_prefix} {strategy_emoji} EXECUTING {strategy_name.upper()}', 'green', attrs=['bold'])} for {symbol}")
                print(f" - Strategy: {colored(strategy_name, 'cyan', attrs=['bold'])}")
                print(f" - Entry: {opportunity['entry_price']:.6f}")
                print(f" - Stop Loss: {opportunity['stop_loss']:.6f}")
                print(f" - Take Profit 1: {opportunity['take_profit_1']:.6f}")
                print(f" - Take Profit 2: {opportunity['take_profit_2']:.6f}")
                print(f" - Risk/Reward: {opportunity['risk_reward']:.2f}")
                print(f" - Reason: {opportunity['reason']}")
                
                # Calculate order size
                quote_quantity = position_size * opportunity["entry_price"]
                quote_quantity = round(quote_quantity, 2)
                if quote_quantity < 10:
                    quote_quantity = 10
                
                print(f" - Order Size: ${quote_quantity:.2f}")
                
                if DRY_RUN:
                    # SIMULATE trade execution
                    print(colored(f" 🔸 [DRY RUN] Simulating market buy...", "yellow"))
                    executed_qty = quote_quantity / opportunity["entry_price"]
                    avg_price = opportunity["entry_price"]
                    order_success = True
                else:
                    # REAL trade execution
                    order = client.order_market_buy(
                        symbol=symbol,
                        quoteOrderQty=quote_quantity
                    )
                    order_success = 'orderId' in order
                    if order_success:
                        executed_qty = sum(float(fill['qty']) for fill in order['fills'])
                        avg_price = float(order['fills'][0]['price'])
                
                if order_success:
                    
                    # Update state
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
                        "entry_type": opportunity["type"],  # Save strategy type
                        "tp1_hit": False,
                        "highest_price": avg_price,
                        "trailing_stop": None,
                        "trailing_distance": None,
                        "original_quantity": executed_qty,
                        "remaining_quantity": executed_qty,
                        "realized_profit": 0,
                    })
                    
                    self.save_position_state(symbol)
                    
                    status_text = ' ✅ POSITION OPENED (SIMULATED)' if DRY_RUN else ' ✅ POSITION OPENED'
                    print(f"{colored(status_text, 'green')} - {executed_qty:.8f} {symbol[:-4]}")
                    return True
                
        except Exception as e:
            print(f"{colored(f' ❌ TRADE ERROR for {symbol}: {e}', 'red')}")
        
        return False
    
    def calculate_trailing_stop(self, entry_price: float, current_price: float, highest_price: float) -> Tuple[float, float]:
        """Calculate progressive trailing stop distance based on profit level"""
        profit_percent = ((current_price - entry_price) / entry_price) * 100
        
        # Determine trailing distance based on profit level
        if profit_percent < 1.0:
            trailing_percent = TRAIL_INITIAL  # 0.8% trail
        elif profit_percent < 2.0:
            trailing_percent = TRAIL_MEDIUM   # 0.5% trail
        else:
            trailing_percent = TRAIL_TIGHT    # 0.3% trail
        
        # Calculate new trailing stop
        new_trailing_stop = highest_price * (1 - trailing_percent / 100)
        
        return new_trailing_stop, trailing_percent
    
    def execute_partial_sell(self, symbol: str, sell_percent: float, reason: str) -> bool:
        """Execute a partial sell of position"""
        try:
            state = trader_state[symbol]
            current_quantity = state.get("remaining_quantity") or state["quantity"]
            
            if current_quantity <= 0:
                return False
            
            # Calculate sell quantity
            sell_quantity = current_quantity * sell_percent
            
            mode_prefix = "🔸 [DRY RUN]" if DRY_RUN else ""
            print(f" {mode_prefix} 💰 Executing partial sell ({sell_percent*100:.0f}%) for {symbol}")
            
            if DRY_RUN:
                # SIMULATE partial sell
                ticker = client.get_symbol_ticker(symbol=symbol)
                actual_exit_price = float(ticker['price'])
                executed_qty = sell_quantity
                print(colored(f" 🔸 [DRY RUN] Simulated sell at {actual_exit_price:.6f}", "yellow"))
            else:
                # REAL partial sell
                # Get precision for the symbol
                exchange_info = client.get_symbol_info(symbol)
                lot_size_filter = next(f for f in exchange_info['filters'] if f['filterType'] == 'LOT_SIZE')
                step_size = float(lot_size_filter['stepSize'])
                precision = int(round(-math.log(step_size, 10), 0))
                
                # Round quantity to match Binance precision
                sell_quantity = round(sell_quantity, precision)
                
                if sell_quantity <= 0:
                    return False
                
                order = client.order_market_sell(
                    symbol=symbol,
                    quantity=sell_quantity
                )
                
                executed_qty = float(order.get('executedQty', 0))
                cummulative_quote_qty = float(order.get('cummulativeQuoteQty', 0))
                
                if executed_qty <= 0:
                    return False
                    
                actual_exit_price = cummulative_quote_qty / executed_qty
            
            if executed_qty > 0:
                entry_price = state["entry_price"]
                
                # Calculate profit from this partial sell
                gross_profit = (actual_exit_price - entry_price) * executed_qty
                fees = (entry_price * executed_qty + cummulative_quote_qty) * 0.001
                net_profit = gross_profit - fees
                
                # Update state
                state["remaining_quantity"] = current_quantity - executed_qty
                state["realized_profit"] = state.get("realized_profit", 0) + net_profit
                
                # Update cumulative total (never gets reset)
                state["total_realized_pnl"] = state.get("total_realized_pnl", 0) + net_profit
                
                print(f" ✅ Partial sell executed:")
                print(f"    - Sold: {executed_qty:.8f} {symbol[:-4]}")
                print(f"    - Price: {actual_exit_price:.6f}")
                print(f"    - Profit: {colored(f'+{net_profit:.2f} USDT', 'green')}")
                print(f"    - Remaining: {state['remaining_quantity']:.8f} {symbol[:-4]}")
                
                return True
                
        except Exception as e:
            print(f" ❌ Partial sell error for {symbol}: {e}")
        
        return False
    
    def check_exit_conditions(self, symbol: str, current_price: float) -> bool:
        """Check professional exit conditions with partial exits and progressive trailing"""
        state = trader_state[symbol]
        if not state["position_is_open"]:
            return False
        
        entry_price = state["entry_price"]
        stop_loss = state["stop_loss"]
        tp1 = state["take_profit_1"]
        
        # Initialize position tracking
        if state.get("original_quantity") is None:
            state["original_quantity"] = state["quantity"]
            state["remaining_quantity"] = state["quantity"]
        
        # Initialize highest price tracking
        if state["highest_price"] is None:
            state["highest_price"] = current_price
        
        # Check for TP1 hit first time - PARTIAL SELL
        if current_price >= tp1 and not state.get("tp1_hit", False):
            print(f"{colored(' 🎯 TAKE PROFIT 1 HIT', 'green')} for {symbol}")
            
            # Execute partial sell (50% by default)
            if self.execute_partial_sell(symbol, TP1_SELL_PERCENT, "TP1 Partial Exit"):
                state["tp1_hit"] = True
                state["tp1_exit_price"] = current_price
                state["highest_price"] = max(current_price, state["highest_price"])
                
                # Initialize trailing stop at breakeven for remaining position
                state["trailing_stop"] = entry_price
                state["stop_loss"] = entry_price
                print(f" 🔒 Trailing stop at breakeven for remaining position: {entry_price:.6f}")
                
                self.save_position_state(symbol)
            else:
                print(f" ⚠️ Failed to execute partial sell, keeping full position")
        
        # If TP1 was hit, use progressive trailing stop
        if state.get("tp1_hit", False):
            # Update highest price
            if current_price > state["highest_price"]:
                state["highest_price"] = current_price
                
                # Calculate new trailing stop
                new_trailing_stop, trailing_percent = self.calculate_trailing_stop(
                    entry_price, current_price, state["highest_price"]
                )
                
                # Only update if new stop is higher than current stop
                if new_trailing_stop > state["stop_loss"]:
                    state["stop_loss"] = new_trailing_stop
                    state["trailing_stop"] = new_trailing_stop
                    state["trailing_distance"] = trailing_percent
                    
                    profit_locked = ((new_trailing_stop - entry_price) / entry_price) * 100
                    print(f" 📈 Trailing stop updated for {symbol}")
                    print(f"    - New stop: {new_trailing_stop:.6f}")
                    print(f"    - Locked profit: {profit_locked:.2f}%")
                    print(f"    - Trail distance: {trailing_percent}%")
        
        # Check stop loss (now might be trailing stop)
        if current_price <= stop_loss:
            if state.get("tp1_hit", False):
                profit_percent = ((stop_loss - entry_price) / entry_price) * 100
                total_realized = state.get("realized_profit", 0)
                print(f"{colored(' 🔒 TRAILING STOP HIT', 'yellow')} for {symbol}")
                print(f"    - Locked profit on remaining: {profit_percent:.2f}%")
                print(f"    - Total realized so far: ${total_realized:.2f}")
                return self.close_position(symbol, current_price, f"Trailing Stop ({profit_percent:.2f}% profit)")
            else:
                print(f"{colored(' 🛑 STOP LOSS HIT', 'red')} for {symbol}")
                return self.close_position(symbol, current_price, "Stop Loss")
        
        # NO MORE FIXED TP2! Let winners run with trailing stop
        # Optional: Could add dynamic targets for additional partial sells
        
        return False
    
    def close_position(self, symbol: str, exit_price: float, reason: str) -> bool:
        """Close remaining position professionally"""
        try:
            state = trader_state[symbol]
            # Use remaining quantity if available, otherwise use original quantity
            quantity = state.get("remaining_quantity") or state["quantity"]
            entry_price = state["entry_price"]
            
            print(f"{colored(' 🔄 CLOSING REMAINING POSITION', 'cyan')} for {symbol}")
            print(f" - Reason: {reason}")
            print(f" - Exit Price: {exit_price:.6f}")
            
            # Get precision for the symbol
            exchange_info = client.get_symbol_info(symbol)
            lot_size_filter = next(f for f in exchange_info['filters'] if f['filterType'] == 'LOT_SIZE')
            step_size = float(lot_size_filter['stepSize'])
            precision = int(round(-math.log(step_size, 10), 0))
            
            # Round quantity to match Binance precision
            quantity = round(quantity, precision)
            
            if quantity <= 0:
                print(f" ⚠️ No remaining position to close")
                # Still reset state
                trader_state[symbol].update({
                    "position_is_open": False,
                    "entry_price": None,
                    "quantity": None,
                    "remaining_quantity": None,
                    "original_quantity": None,
                })
                self.save_position_state(symbol)
                return True
            
            if DRY_RUN:
                # SIMULATE closing position
                ticker = client.get_symbol_ticker(symbol=symbol)
                actual_exit_price = float(ticker['price'])
                executed_qty = quantity
                print(colored(f" 🔸 [DRY RUN] Simulated close at {actual_exit_price:.6f}", "yellow"))
            else:
                # REAL closing position
                order = client.order_market_sell(
                    symbol=symbol,
                    quantity=quantity
                )
                
                executed_qty = float(order.get('executedQty', 0))
                cummulative_quote_qty = float(order.get('cummulativeQuoteQty', 0))
                
                if executed_qty <= 0:
                    return False
                    
                actual_exit_price = cummulative_quote_qty / executed_qty
            
            if executed_qty > 0:
                
                # Calculate P&L for this exit
                gross_profit = (actual_exit_price - entry_price) * executed_qty
                fees = (entry_price * executed_qty + cummulative_quote_qty) * 0.001  # 0.1% fee
                net_profit = gross_profit - fees
                
                # Add to realized profit
                total_realized = state.get("realized_profit", 0) + net_profit
                
                # Update cumulative total (never gets reset)
                state["total_realized_pnl"] = state.get("total_realized_pnl", 0) + net_profit
                
                status_text = ' ✅ REMAINING POSITION CLOSED (SIMULATED)' if DRY_RUN else ' ✅ REMAINING POSITION CLOSED'
                print(f"{colored(status_text, 'cyan')}")
                print(f" - This exit P&L: {colored(f'{net_profit:+.2f} USDT', 'green' if net_profit > 0 else 'red')}")
                print(f" - Total realized P&L: {colored(f'{total_realized:+.2f} USDT', 'green' if total_realized > 0 else 'red')}")
                
                # Log the trade
                trade_record = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "symbol": symbol,
                    "entry_price": entry_price,
                    "exit_price": actual_exit_price,
                    "quantity": executed_qty,
                    "profit_loss": net_profit,
                    "total_realized_profit": total_realized,
                    "partial_exits": state.get("tp1_hit", False),
                    "tp1_exit_price": state.get("tp1_exit_price"),
                    "original_quantity": state.get("original_quantity"),
                    "highest_price_reached": state.get("highest_price"),
                    "reason": reason,
                    "entry_reason": state["entry_reason"],
                    "risk_reward_planned": state["risk_reward_ratio"]
                }
                
                # Also update session summary with this trade
                self.update_session_trade_summary(symbol, trade_record, state)
                
                # Save trade record
                trades_file = state["filename_trades"]
                trades = []
                if os.path.exists(trades_file):
                    with open(trades_file, 'r') as f:
                        trades = json.load(f)
                trades.append(trade_record)
                with open(trades_file, 'w') as f:
                    json.dump(trades, f, indent=2)
                
                # Reset state
                trader_state[symbol].update({
                    "position_is_open": False,
                    "entry_price": None,
                    "quantity": None,
                    "entry_time": None,
                    "stop_loss": None,
                    "take_profit_1": None,
                    "take_profit_2": None,
                    "risk_reward_ratio": None,
                    "entry_reason": None,
                    "tp1_hit": False,
                    "highest_price": None,
                    "trailing_stop": None,
                    "trailing_distance": None,
                    "original_quantity": None,
                    "tp1_executed_quantity": None,
                    "remaining_quantity": None,
                    "realized_profit": 0,
                    "tp1_exit_price": None,
                    "entry_type": None,
                })
                
                self.save_position_state(symbol)
                return True
                
        except Exception as e:
            print(f"{colored(f' ❌ CLOSE ERROR for {symbol}: {e}', 'red')}")
        
        return False
    
    def update_session_trade_summary(self, symbol: str, trade_record: dict, state: dict):
        """Update session file with detailed trade information"""
        session_file = "trader_pro/session_pnl_dry.json" if DRY_RUN else "trader_pro/session_pnl_live.json"
        session_data = {}
        
        if os.path.exists(session_file):
            with open(session_file, 'r') as f:
                session_data = json.load(f)
        
        # Initialize structure if needed
        if "trades" not in session_data:
            session_data["trades"] = []
        if "trade_summaries" not in session_data:
            session_data["trade_summaries"] = {}
        
        # Create comprehensive trade summary
        if trade_record.get("partial_exits", False):
            # This is a complete trade with partial exits
            trade_summary = {
                "symbol": symbol,
                "entry_timestamp": state.get("entry_time"),
                "exit_timestamp": trade_record["timestamp"],
                "entry_price": trade_record["entry_price"],
                "tp1_exit_price": trade_record.get("tp1_exit_price"),
                "final_exit_price": trade_record["exit_price"],
                "highest_price": trade_record.get("highest_price_reached"),
                "original_quantity": trade_record.get("original_quantity"),
                "tp1_quantity_sold": trade_record.get("original_quantity", 0) * TP1_SELL_PERCENT,
                "final_quantity_sold": trade_record["quantity"],
                "total_profit": trade_record["total_realized_profit"],
                "strategy": "Professional Partial Exit",
                "entry_reason": trade_record["entry_reason"],
                "exit_reason": trade_record["reason"],
                "performance_metrics": {
                    "entry_to_tp1_percent": ((trade_record.get("tp1_exit_price", 0) - trade_record["entry_price"]) / trade_record["entry_price"]) * 100 if trade_record.get("tp1_exit_price") else 0,
                    "entry_to_final_percent": ((trade_record["exit_price"] - trade_record["entry_price"]) / trade_record["entry_price"]) * 100,
                    "highest_gain_percent": ((trade_record.get("highest_price_reached", 0) - trade_record["entry_price"]) / trade_record["entry_price"]) * 100 if trade_record.get("highest_price_reached") else 0,
                }
            }
            
            session_data["trade_summaries"][f"{symbol}_{trade_record['timestamp']}"] = trade_summary
        
        # Add to trades list
        session_data["trades"].append(trade_record)
        
        # Update summary stats
        session_data["total_trades"] = len(session_data["trades"])
        session_data["profitable_trades"] = len([t for t in session_data["trades"] if t["profit_loss"] > 0])
        session_data["total_pnl"] = sum(t["profit_loss"] for t in session_data["trades"])
        session_data["last_trade"] = trade_record["timestamp"]
        
        # Save updated session data
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
    
    def display_portfolio_summary(self):
        """Display comprehensive P&L summary across all positions"""
        print("")
        print("=" * 80)
        mode_text = " (DRY RUN)" if DRY_RUN else " (LIVE)"
        print(colored(f"💼 PORTFOLIO SUMMARY{mode_text}", "cyan", attrs=["bold"]))
        print("=" * 80)
        
        total_realized = 0
        total_unrealized = 0
        open_positions = 0
        positions_in_profit = 0
        
        # Load session P&L tracking - use separate files for dry/live mode
        session_file = "trader_pro/session_pnl_dry.json" if DRY_RUN else "trader_pro/session_pnl_live.json"
        session_data = {}
        if os.path.exists(session_file):
            with open(session_file, 'r') as f:
                session_data = json.load(f)
        
        # Collect data for all symbols
        position_details = []
        
        # Track cumulative P&L from ALL symbols (both open and closed)
        for symbol in symbols:
            state = trader_state[symbol]
            
            # Add cumulative realized P&L for this symbol (tracks all historical trades)
            cumulative_pnl = state.get("total_realized_pnl", 0)
            total_realized += cumulative_pnl
            
            if state["position_is_open"]:
                open_positions += 1
                
                # Get current price
                try:
                    ticker = client.get_symbol_ticker(symbol=symbol)
                    current_price = float(ticker['price'])
                    
                    entry_price = state["entry_price"]
                    remaining_qty = state.get("remaining_quantity") or state["quantity"]
                    realized_profit = state.get("realized_profit", 0)
                    
                    # Calculate unrealized P&L on remaining position
                    unrealized_value = (current_price - entry_price) * remaining_qty
                    unrealized_fees = current_price * remaining_qty * 0.001  # Estimated exit fee
                    unrealized_profit = unrealized_value - unrealized_fees
                    
                    total_unrealized += unrealized_profit
                    
                    if unrealized_profit > 0:
                        positions_in_profit += 1
                    
                    # Store position details
                    position_details.append({
                        "symbol": symbol,
                        "unrealized": unrealized_profit,
                        "realized": realized_profit,
                        "total": unrealized_profit + realized_profit,
                        "pnl_percent": ((current_price - entry_price) / entry_price) * 100,
                        "is_trailing": state.get("tp1_hit", False)
                    })
                except:
                    pass
        
        # Display summary
        total_pnl = total_realized + total_unrealized
        
        print(f"📊 Open Positions: {colored(str(open_positions), 'yellow')} / {len(symbols)}")
        print(f"✅ Positions in Profit: {colored(str(positions_in_profit), 'green')}")
        print("")
        
        # Show individual positions if any are open
        if position_details:
            print("Position Details:")
            for pos in sorted(position_details, key=lambda x: x['total'], reverse=True):
                symbol_name = pos['symbol'][:-4]  # Remove USDC
                status_icon = "🚀" if pos['is_trailing'] else "📈"
                
                print(f"  {status_icon} {symbol_name:8} | "
                      f"Unrealized: {colored(f'{pos['unrealized']:+.2f}', 'green' if pos['unrealized'] > 0 else 'red'):12} | "
                      f"Realized: {colored(f'{pos['realized']:+.2f}', 'green' if pos['realized'] > 0 else 'yellow'):10} | "
                      f"Total: {colored(f'{pos['total']:+.2f}', 'green' if pos['total'] > 0 else 'red'):10} | "
                      f"({pos['pnl_percent']:+.2f}%)")
            print("")
        
        # Display totals
        print("─" * 80)
        print(f"💰 Total Realized P&L:   {colored(f'${total_realized:+.2f}', 'green' if total_realized > 0 else 'red')}")
        print(f"📈 Total Unrealized P&L: {colored(f'${total_unrealized:+.2f}', 'green' if total_unrealized > 0 else 'red')}")
        print("─" * 80)
        print(f"💎 TOTAL PORTFOLIO P&L:  {colored(f'${total_pnl:+.2f}', 'green' if total_pnl > 0 else 'red', attrs=['bold'])}")
        
        # Save session P&L for tracking
        session_data["last_update"] = datetime.datetime.now().isoformat()
        session_data["total_realized"] = total_realized
        session_data["total_unrealized"] = total_unrealized
        session_data["total_pnl"] = total_pnl
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print("=" * 80)
        print("")
    
    def wait_for_next_candle_close(self):
        """Wait until 30 seconds after the next 15-minute candle close"""
        now = datetime.datetime.now()
        
        # Calculate next 15-minute mark
        minutes = now.minute
        seconds = now.second
        
        # Find how many minutes until next 15-minute mark (0, 15, 30, 45)
        minutes_to_next = 15 - (minutes % 15)
        if minutes_to_next == 15:  # We're exactly on a 15-minute mark
            minutes_to_next = 0
        
        # Calculate total seconds to wait (next 15-min mark + 30 seconds)
        seconds_to_wait = (minutes_to_next * 60) - seconds + 30
        
        # If we're within the first 30 seconds after a 15-minute mark, wait for the next one
        if seconds_to_wait <= 0:
            seconds_to_wait += 15 * 60  # Add 15 minutes
        
        next_run = now + datetime.timedelta(seconds=seconds_to_wait)
        print(f"⏰ Next analysis at {next_run.strftime('%H:%M:%S')} ({int(seconds_to_wait/60)}m {seconds_to_wait%60}s)...")
        
        # Wait with interrupt checking
        for _ in range(seconds_to_wait):
            if shutdown:
                break
            time.sleep(1)
    
    def run(self):
        """Main professional trading loop"""
        mode_text = "DRY RUN MODE - NO REAL TRADES" if DRY_RUN else "LIVE TRADING - REAL MONEY"
        mode_color = "yellow" if DRY_RUN else "red"
        
        print("=" * 80)
        print(colored("📈 PROFESSIONAL TRADER BOT ACTIVE", "green", attrs=["bold"]))
        print(colored(f"🔸 {mode_text}", mode_color, attrs=["bold"]))
        print("=" * 80)
        
        # Display account balance
        #try:
        #    account = client.get_account()
        #    usdc_balance = next((float(b['free']) for b in account['balances'] if b['asset'] == 'USDC'), 0)
        #    print(colored("💰 ACCOUNT BALANCE", "yellow", attrs=["bold"]))
        #    print(f"   USDC Available: {colored(f'${usdc_balance:.2f}', 'green', attrs=['bold'])}")
        #    print("─" * 80)
        #except Exception as e:
        #    print(f"⚠️ Could not fetch balance: {e}")
        
        # Strategy Summary
        print(colored("STRATEGY OVERVIEW", "yellow", attrs=["bold"]))
        print("─" * 80)
        print("📍 " + colored("ENTRY CONDITIONS (5 STRATEGIES - LONG ONLY):", "cyan"))
        print("   1. " + colored("Support Bounce:", "green") + " Price within 0.3% of strong support (≥2 touches)")
        print("   2. " + colored("Trend Breakout:", "green") + " Price breaks above trend line +0.2% (≥3 touches)")
        print("   3. " + colored("RSI Oversold:", "yellow") + " RSI < 25 with price stabilization")
        print("   4. " + colored("Volume Spike:", "yellow") + " 2.5x average volume on green candle")
        print("   5. " + colored("Liquidity Hunt:", "magenta") + " Stop hunt wick with recovery above support")
        print("")
        print("🛡️ " + colored("BTC FILTER (ALT PROTECTION):", "red"))
        print("   • " + colored("NO trades when:", "red") + " BTC dumps >1.5%/hr, RSI<30, or bearish trend")
        print("   • " + colored("Higher confidence:", "green") + " When BTC is bullish or rising")
        print("")
        print("🎯 " + colored("EXIT STRATEGY:", "cyan"))
        print("   • " + colored("TP1 (1% profit):", "yellow") + " Sell 50%, move stop to breakeven")
        print("   • " + colored("Progressive Trail:", "yellow") + " <1%: 0.8% | 1-2%: 0.5% | >2%: 0.3%")
        print("   • " + colored("No fixed TP2:", "yellow") + " Let winners run with trailing stop")
        print("")
        print("⚙️ " + colored("RISK PARAMETERS:", "cyan"))
        print("   • Position Size: $50-$100 | Risk/Reward: ≥2.0")
        print("   • Confidence: >70% | Stop Loss: 0.8% below support")
        print("   • Timeframe: 15-minute candles (analyzed 30s after close)")
        print("=" * 80)
        print("")
        
        # Display current positions and P&L at startup (BEFORE waiting)
        print("=" * 80)
        print(colored("📊 STARTUP STATUS CHECK", "cyan", attrs=["bold"]))
        print("=" * 80)
        
        open_positions_count = 0
        for symbol in symbols:
            try:
                # Get current price
                ticker = client.get_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])
                
                # Check existing position
                if trader_state[symbol]["position_is_open"]:
                    open_positions_count += 1
                    entry_price = trader_state[symbol]["entry_price"]
                    pnl_percent = ((current_price - entry_price) / entry_price) * 100
                    
                    original_qty = trader_state[symbol].get('original_quantity') or trader_state[symbol]['quantity']
                    remaining_qty = trader_state[symbol].get('remaining_quantity') or trader_state[symbol]['quantity']
                    realized_pnl = trader_state[symbol].get('realized_profit', 0)
                    
                    # Get strategy info for display
                    entry_type = trader_state[symbol].get('entry_type', 'unknown')
                    strategy_emojis = {
                        "support_bounce": "🔄",
                        "trend_line_breakout": "📈",
                        "rsi_oversold_bounce": "📉",
                        "volume_breakout": "📊",
                        "liquidity_hunt": "🎣"
                    }
                    strategy_emoji = strategy_emojis.get(entry_type, "🎯")
                    strategy_display = entry_type.replace("_", " ").title()
                    
                    print(f"[{symbol}] Price: {colored(f'{current_price:.6f}', 'cyan')} USDT")
                    print(f" - Position: {colored('OPEN', 'yellow')} {strategy_emoji} {strategy_display} (Entry: {entry_price:.6f})")
                    print(f" - Remaining: {remaining_qty:.8f}/{original_qty:.8f} {symbol[:-4]}")
                    print(f" - Unrealized P&L: {colored(f'{pnl_percent:+.2f}%', 'green' if pnl_percent > 0 else 'red')}")
                    if realized_pnl > 0:
                        print(f" - Realized P&L: {colored(f'+${realized_pnl:.2f}', 'green')}")
                    print(f" - Stop Loss: {trader_state[symbol]['stop_loss']:.6f}")
                    
                    # Show trailing info if TP1 was hit
                    if trader_state[symbol].get('tp1_hit', False):
                        locked_profit = ((trader_state[symbol]['stop_loss'] - entry_price) / entry_price) * 100
                        trail_dist = trader_state[symbol].get('trailing_distance', 0.0)
                        if trail_dist is None:
                            trail_dist = 0.0
                        print(f" - Status: {colored('🚀 TRAILING - 50% SOLD', 'cyan', attrs=['bold'])}")
                        print(f" - Locked profit: {locked_profit:.2f}% (Trail: {trail_dist:.1f}%)")
                        if trader_state[symbol].get('highest_price'):
                            high_pnl = ((trader_state[symbol]['highest_price'] - entry_price) / entry_price) * 100
                            print(f" - Highest reached: {trader_state[symbol]['highest_price']:.6f} ({high_pnl:.2f}%)")
                    print("")
            except Exception as e:
                print(f"[{symbol}] Error: {colored(str(e), 'red')}")
        
        if open_positions_count == 0:
            print(colored("💰 No open positions - Ready to trade", "green"))
            print("")
        
        # Display portfolio summary at startup
        self.display_portfolio_summary()
        
        # Initial sync to market time
        print("")
        print(colored("🔄 Syncing to market time...", "yellow"))
        print(colored("⏰ Waiting for next 15-minute candle to analyze...", "yellow"))
        print("=" * 80)
        self.wait_for_next_candle_close()
        
        while not shutdown:
            current_time = datetime.datetime.now()
            prev_candle_time = current_time - datetime.timedelta(seconds=30)
            candle_start = prev_candle_time.replace(minute=(prev_candle_time.minute // 15) * 15, second=0, microsecond=0)
            
            mode_prefix = "🔸 [DRY RUN]" if DRY_RUN else ""
            print(f"=== {mode_prefix} Professional Analysis: {current_time.strftime('%Y-%m-%d %H:%M:%S')} ===")
            print(f"    Analyzing candle: {candle_start.strftime('%H:%M')} - {(candle_start + datetime.timedelta(minutes=15)).strftime('%H:%M')} (completed)")
            print(f"")
            
            for symbol in symbols:
                try:
                    # Get current price
                    ticker = client.get_symbol_ticker(symbol=symbol)
                    current_price = float(ticker['price'])
                    
                    print(f"")
                    print(f"[{symbol}] Price: {colored(f'{current_price:.6f}', 'cyan')} USDT")
                    
                    # Check existing position
                    if trader_state[symbol]["position_is_open"]:
                        entry_price = trader_state[symbol]["entry_price"]
                        pnl_percent = ((current_price - entry_price) / entry_price) * 100
                        
                        original_qty = trader_state[symbol].get('original_quantity') or trader_state[symbol]['quantity']
                        remaining_qty = trader_state[symbol].get('remaining_quantity') or trader_state[symbol]['quantity']
                        realized_pnl = trader_state[symbol].get('realized_profit', 0)
                        
                        # Get strategy info for display
                        entry_type = trader_state[symbol].get('entry_type', 'unknown')
                        strategy_emojis = {
                            "support_bounce": "🔄",
                            "trend_line_breakout": "📈",
                            "rsi_oversold_bounce": "📉",
                            "volume_breakout": "📊",
                            "liquidity_hunt": "🎣"
                        }
                        strategy_emoji = strategy_emojis.get(entry_type, "🎯")
                        strategy_display = entry_type.replace("_", " ").title()
                        
                        print(f" - Position: {colored('OPEN', 'yellow')} {strategy_emoji} {strategy_display} (Entry: {entry_price:.6f})")
                        print(f" - Remaining: {remaining_qty:.8f}/{original_qty:.8f} {symbol[:-4]}")
                        print(f" - Unrealized P&L: {colored(f'{pnl_percent:+.2f}%', 'green' if pnl_percent > 0 else 'red')}")
                        if realized_pnl > 0:
                            print(f" - Realized P&L: {colored(f'+${realized_pnl:.2f}', 'green')}")
                        print(f" - Stop Loss: {trader_state[symbol]['stop_loss']:.6f}")
                        
                        # Show trailing info if TP1 was hit
                        if trader_state[symbol].get('tp1_hit', False):
                            locked_profit = ((trader_state[symbol]['stop_loss'] - entry_price) / entry_price) * 100
                            trail_dist = trader_state[symbol].get('trailing_distance', 0.0)
                            if trail_dist is None:
                                trail_dist = 0.0
                            print(f" - Status: {colored('🚀 TRAILING - 50% SOLD', 'cyan', attrs=['bold'])}")
                            print(f" - Locked profit: {locked_profit:.2f}% (Trail: {trail_dist:.1f}%)")
                            if trader_state[symbol].get('highest_price'):
                                high_pnl = ((trader_state[symbol]['highest_price'] - entry_price) / entry_price) * 100
                                print(f" - Highest reached: {trader_state[symbol]['highest_price']:.6f} ({high_pnl:.2f}%)")
                        
                        # Check exit conditions
                        if self.check_exit_conditions(symbol, current_price):
                            continue
                    
                    else:
                        # Look for new opportunities
                        df = self.get_candle_data(symbol)
                        if len(df) > 0:
                            opportunity = self.find_trading_opportunity(symbol, df)
                            
                            if opportunity:
                                print(f" - {colored('OPPORTUNITY FOUND', 'green', attrs=['bold'])}")
                                print(f" - Type: {opportunity['type']}")
                                print(f" - Direction: {opportunity['direction'].upper()}")
                                print(f" - R/R: {opportunity['risk_reward']:.2f}")
                                print(f" - Confidence: {opportunity['confidence']}%")
                                print(f" - Reason: {opportunity['reason']}")
                                
                                # Execute if confidence is high enough
                                if opportunity['confidence'] > 70:
                                    self.execute_trade(symbol, opportunity)
                                else:
                                    print(f" - {colored('Skipped (low confidence)', 'yellow')}")
                            else:
                                print(f" - Status: {colored('No opportunities', 'grey')}")
                
                except Exception as e:
                    print(f" - Error: {colored(str(e), 'red')}")
            
            # Calculate and display overall P&L summary
            self.display_portfolio_summary()
            
            # Wait for next candle close (synced to market time)
            self.wait_for_next_candle_close()
            print(f"")
        
        print(f"\\n{colored('📈 Professional Trader stopped', 'yellow')}")

def emergency_sell():
    """Emergency sell all positions"""
    print(colored("\\n🔴 EMERGENCY SELL TRIGGERED", "red", attrs=['bold']))
    confirm = input("Type 'YES' to sell all positions: ").strip()
    if confirm != "YES":
        print("❌ Cancelled")
        return
    
    trader = ProfessionalTrader()
    for symbol in symbols:
        if trader_state[symbol]["position_is_open"]:
            ticker = client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
            trader.close_position(symbol, current_price, "Emergency Exit")

def listen_for_commands():
    """Listen for emergency commands"""
    while not shutdown:
        try:
            key = input().strip().lower()
            if key == "x":
                emergency_sell()
        except:
            break

if __name__ == "__main__":
    # Check for command-line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--check-pnl":
            print("=" * 80)
            print(colored("📊 CHECKING CUMULATIVE P&L", "cyan", attrs=['bold']))
            print("=" * 80)
            mode_suffix = "_dry" if DRY_RUN else "_live"
            total_cumulative = 0
            for symbol in symbols:
                position_file = f"trader_pro/position_{symbol}{mode_suffix}.json"
                if os.path.exists(position_file):
                    with open(position_file, 'r') as f:
                        data = json.load(f)
                        pnl = data.get('total_realized_pnl', 0)
                        if pnl != 0:
                            print(f"{symbol}: {colored(f'${pnl:+.2f}', 'green' if pnl > 0 else 'red')}")
                            total_cumulative += pnl
            print("─" * 80)
            print(f"TOTAL: {colored(f'${total_cumulative:+.2f}', 'green' if total_cumulative > 0 else 'red', attrs=['bold'])}")
            print("=" * 80)
            sys.exit(0)
        elif sys.argv[1] == "--reset-pnl":
            print(colored("⚠️ WARNING: This will reset all cumulative P&L tracking!", "yellow", attrs=['bold']))
            confirm = input("Type 'RESET' to confirm: ").strip()
            if confirm == "RESET":
                mode_suffix = "_dry" if DRY_RUN else "_live"
                for symbol in symbols:
                    position_file = f"trader_pro/position_{symbol}{mode_suffix}.json"
                    if os.path.exists(position_file):
                        with open(position_file, 'r') as f:
                            data = json.load(f)
                        data['total_realized_pnl'] = 0
                        data['realized_profit'] = 0
                        with open(position_file, 'w') as f:
                            json.dump(data, f, indent=2)
                print(colored("✅ Cumulative P&L has been reset", "green"))
            else:
                print("❌ Reset cancelled")
            sys.exit(0)
    
    # Check Binance connection
    try:
        account = client.get_account()
        balance = next((b for b in account['balances'] if b['asset'] == 'USDC'), None)
        if balance:
            print("=" * 80)
            print(colored("✅ Binance connection successful", "green"))
            print(f"USDC Balance: {balance['free']}")
        else:
            print("❌ USDC balance not found")
    except Exception as e:
        print(f"❌ Binance connection failed: {e}")
        sys.exit(1)
    
    # Start command listener
    threading.Thread(target=listen_for_commands, daemon=True).start()
    
    # Start professional trader
    trader = ProfessionalTrader()
    trader.run()