import os
import requests
import json
import hmac
import hashlib
import datetime
import time
import math
from termcolor import colored
import signal
import sys
import pandas as pd
from binance.client import Client
import tenacity
import threading
import numpy as np
from collections import deque
sys.path.append('../')
from _secrets import api_key, secret_key

# ==================== CONFIGURATION ====================
# TRADING MODE - IMPORTANT: Set to True to prevent real money losses!
DRY_RUN = True  # Set to False for LIVE trading with real money

# WARNING: With DRY_RUN = False, this bot will execute REAL trades!
# Only set to False when you're confident in the strategy
# ========================================================

# List of coins to track (all symbols from backtest)
symbols = [
    "BTCUSDC", "ETHUSDC", "BNBUSDC", "ADAUSDC", "XRPUSDC", 
    "DOGEUSDC", "SOLUSDC", "PNUTUSDC", "PEPEUSDC", "SHIBUSDC", 
    "XLMUSDC", "LINKUSDC", "IOTAUSDC", "ENAUSDC"
]

shutdown = False
btc_trend = {"15m": None, "1h": None, "4h": None, "last_update": None}

# Determine file suffix based on mode
mode_suffix = "_dry" if DRY_RUN else "_live"

def handle_exit(sig, frame):
    global shutdown
    print("\n🔻 Graceful shutdown requested. Exiting...")
    shutdown = True

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

coins = {
    symbol: {
        "filename_order_id": f"orders_pro/order_id_{symbol}{mode_suffix}.txt",
        "filename_output": f"outputs_pro/output_{symbol}{mode_suffix}.txt",
        "filename_stop_loss_wait": f"orders_pro/stop_loss_wait_{symbol}{mode_suffix}.txt",
        "filename_partial_exits": f"orders_pro/partial_exits_{symbol}{mode_suffix}.txt",
        "buy_price": None,
        "sell_price": None,
        "bought_quantity": None,
        "remaining_quantity": None,
        "position_is_open": False,
        "entry_date": None,
        "entry_pattern": None,
        "stop_loss_wait_until": None,
        "highest_price": None,
        "trailing_stop_price": None,
        "partial_exits": [],
        "swing_highs": deque(maxlen=20),  # Increased for better S/R detection
        "swing_lows": deque(maxlen=20),
        "support_levels": [],
        "resistance_levels": [],
        "last_rsi": None,
        "last_macd_histogram": deque(maxlen=5),
        "entry_volume": None,
        "volume_ma": None,
        "pattern_success_rate": {},  # Track pattern performance
        "hourly_performance": {},  # Track performance by hour
    }
    for symbol in symbols
}

# Create directories if they don't exist
os.makedirs("orders_pro", exist_ok=True)
os.makedirs("outputs_pro", exist_ok=True)

# Restore state from files
for symbol in symbols:
    # Restore position state
    try:
        with open(coins[symbol]["filename_order_id"], "r") as f:
            data_content = f.read().strip()
            if data_content:
                parts = data_content.split(",")
                if len(parts) >= 4:  # buy_price,quantity,entry_date,pattern
                    buy_price, quantity, entry_date, pattern = parts[0], parts[1], parts[2], parts[3]
                    coins[symbol]["buy_price"] = float(buy_price)
                    coins[symbol]["bought_quantity"] = float(quantity)
                    coins[symbol]["remaining_quantity"] = float(quantity)
                    coins[symbol]["position_is_open"] = True
                    coins[symbol]["entry_date"] = entry_date
                    coins[symbol]["entry_pattern"] = pattern
                    coins[symbol]["highest_price"] = float(buy_price)
    except FileNotFoundError:
        pass
    
    # Restore partial exits
    try:
        with open(coins[symbol]["filename_partial_exits"], "r") as f:
            exits_data = f.read().strip()
            if exits_data:
                coins[symbol]["partial_exits"] = json.loads(exits_data)
                # Calculate remaining quantity
                total_sold = sum(exit['quantity'] for exit in coins[symbol]["partial_exits"])
                if coins[symbol]["bought_quantity"]:
                    coins[symbol]["remaining_quantity"] = coins[symbol]["bought_quantity"] - total_sold
    except FileNotFoundError:
        pass
    
    # Restore stop loss wait state
    try:
        with open(coins[symbol]["filename_stop_loss_wait"], "r") as f:
            wait_until_str = f.read().strip()
            if wait_until_str:
                coins[symbol]["stop_loss_wait_until"] = datetime.datetime.fromisoformat(wait_until_str)
    except FileNotFoundError:
        pass

# Load profit/loss tracking - use separate files for dry/live mode
status_file = f"status_pro{mode_suffix}.json"
overall_status = {}
if os.path.exists(status_file):
    with open(status_file, "r") as f:
        overall_status = json.load(f)

# Ensure all tracked symbols are present
for symbol in symbols:
    if symbol not in overall_status:
        overall_status[symbol] = 0

# Global market analysis tracking
market_analysis = {
    "total_positions": 0,
    "correlation_limit": 3,  # Max correlated positions
    "last_btc_check": None,
    "market_sentiment": "neutral",
}

# Trading Constants - Enhanced with smart entry parameters
usd_amount = 100  # USDT per trade
kline_interval = Client.KLINE_INTERVAL_5MINUTE  # 5-minute for pattern detection (changed from 1m)
kline_interval_5m = Client.KLINE_INTERVAL_5MINUTE  # 5-minute for pattern detection
kline_interval_15m = Client.KLINE_INTERVAL_15MINUTE  # 15-minute for Bitcoin check and medium trend
kline_interval_1h = Client.KLINE_INTERVAL_1HOUR  # 1-hour for indicators
kline_interval_4h = Client.KLINE_INTERVAL_4HOUR  # 4-hour for major trend
candles_lookback_1m = 100  # Lookback for 1-minute pattern analysis (deprecated)
candles_lookback_5m = 60   # Lookback for 5-minute pattern analysis (5 hours of data)
candles_lookback_15m = 30  # Lookback for 15-minute Bitcoin analysis
candles_lookback_1h = 50   # Lookback for 1-hour indicators
candles_lookback_4h = 100  # Lookback for 4-hour EMA calculation
base_take_profit_percent = 0.015  # 1.5% base take profit
base_stop_loss_percent = 0.10   # 10% base stop loss
check_interval_minutes = 5  # Check positions every 5 minutes (aligned with candle timeframe)
trade_fee_percent = 0.001  # 0.1% fee per trade
stop_loss_wait_hours = 12  # Wait 12 hours after stop loss
time_stop_hours = 72  # Exit after 72 hours if no profit

# Enhanced entry requirements
min_entry_score = 70  # Minimum score required for entry (out of 100)
sr_distance_threshold = 0.005  # 0.5% distance from S/R levels
volume_profile_periods = 20  # Periods for volume profile analysis
adx_trend_threshold = 25  # Minimum ADX for trending market
min_volume_ratio = 0.8  # Minimum volume vs average for entry

# Graduated exit percentages
graduated_exits = [
    {"trigger": 0.005, "sell_percent": 0.25},  # Sell 25% at 0.5% profit
    {"trigger": 0.01, "sell_percent": 0.25},   # Sell 25% at 1% profit
    {"trigger": 0.02, "sell_percent": 0.25},   # Sell 25% at 2% profit
    {"trigger": 0.03, "sell_percent": 0.25},   # Sell remaining 25% at 3% profit
]

# EMA configuration
ema_short_period = 9    # EMA9 for short trend
ema_medium_period = 21  # EMA21 for medium trend
ema_long_period = 55    # EMA55 for major trend

# Technical indicator thresholds
rsi_overbought = 70
rsi_oversold = 30
rsi_divergence_lookback = 10
volume_spike_multiplier = 2.0
atr_multiplier = 2.5

# Set up the Binance API client
client = Client(api_key, secret_key)

# Support and Resistance Detection
def detect_support_resistance(highs, lows, current_price, lookback=20):
    """Detect support and resistance levels using swing points"""
    if len(highs) < lookback or len(lows) < lookback:
        return [], [], None
    
    recent_highs = list(highs)[-lookback:]
    recent_lows = list(lows)[-lookback:]
    
    # Find significant levels using clustering
    all_levels = recent_highs + recent_lows
    if len(all_levels) < 5:
        return [], [], None
    
    # Cluster levels within 0.5% of each other
    clustered_levels = []
    used = set()
    
    for i, level in enumerate(all_levels):
        if i in used:
            continue
        cluster = [level]
        used.add(i)
        for j, other_level in enumerate(all_levels):
            if j != i and j not in used:
                if abs(level - other_level) / level < 0.005:  # Within 0.5%
                    cluster.append(other_level)
                    used.add(j)
        
        if len(cluster) >= 2:  # Significant if touched multiple times
            clustered_levels.append(np.mean(cluster))
    
    # Separate into support and resistance
    support_levels = [level for level in clustered_levels if level < current_price]
    resistance_levels = [level for level in clustered_levels if level > current_price]
    
    # Sort by distance from current price
    support_levels.sort(reverse=True)  # Nearest first
    resistance_levels.sort()  # Nearest first
    
    # Determine if price is near S/R
    near_level = None
    if support_levels and (current_price - support_levels[0]) / current_price < sr_distance_threshold:
        near_level = "support"
    elif resistance_levels and (resistance_levels[0] - current_price) / current_price < sr_distance_threshold:
        near_level = "resistance"
    
    return support_levels[:3], resistance_levels[:3], near_level

def calculate_volume_profile(df, periods=20):
    """Calculate volume profile and identify high volume nodes"""
    if len(df) < periods:
        return None, None
    
    recent_df = df.tail(periods)
    
    # Create price bins
    price_range = recent_df['high'].max() - recent_df['low'].min()
    if price_range <= 0:
        return None, None
    
    num_bins = min(10, len(recent_df))
    bins = np.linspace(recent_df['low'].min(), recent_df['high'].max(), num_bins)
    
    # Calculate volume at each price level
    volume_profile = {}
    for _, row in recent_df.iterrows():
        # Distribute volume across the candle's range
        candle_bins = bins[(bins >= row['low']) & (bins <= row['high'])]
        if len(candle_bins) > 0:
            volume_per_bin = row['volume'] / len(candle_bins)
            for price_bin in candle_bins:
                if price_bin not in volume_profile:
                    volume_profile[price_bin] = 0
                volume_profile[price_bin] += volume_per_bin
    
    if not volume_profile:
        return None, None
    
    # Find Point of Control (highest volume node)
    poc = max(volume_profile, key=volume_profile.get)
    
    # Calculate average volume
    avg_volume = np.mean(list(volume_profile.values()))
    
    return poc, avg_volume

def get_bitcoin_trend():
    """Get Bitcoin's trend on multiple timeframes"""
    global btc_trend
    
    try:
        # Only update every 5 minutes
        if btc_trend["last_update"] and (datetime.datetime.now() - btc_trend["last_update"]).seconds < 300:
            return btc_trend
        
        # Get Bitcoin 15m data
        klines_15m = client.get_historical_klines(
            "BTCUSDC", 
            Client.KLINE_INTERVAL_15MINUTE, 
            f"{candles_lookback_15m * 15} minutes ago UTC"
        )
        
        if len(klines_15m) >= 20:
            df_15m = pd.DataFrame(klines_15m, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])
            closes_15m = df_15m['close'].astype(float)
            
            # Calculate EMAs
            ema9 = closes_15m.ewm(span=9, adjust=False).mean().iloc[-1]
            ema21 = closes_15m.ewm(span=21, adjust=False).mean().iloc[-1]
            current_price = closes_15m.iloc[-1]
            
            # Determine trend
            if current_price > ema9 > ema21:
                btc_trend["15m"] = "bullish"
            elif current_price < ema9 < ema21:
                btc_trend["15m"] = "bearish"
            else:
                btc_trend["15m"] = "neutral"
            
            # Calculate trend strength (percentage above/below EMA21)
            btc_trend["strength"] = ((current_price - ema21) / ema21) * 100
        
        # Get 1h trend
        klines_1h = client.get_historical_klines(
            "BTCUSDC", 
            Client.KLINE_INTERVAL_1HOUR, 
            "24 hours ago UTC"
        )
        
        if len(klines_1h) >= 10:
            df_1h = pd.DataFrame(klines_1h, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])
            closes_1h = df_1h['close'].astype(float)
            ema9_1h = closes_1h.ewm(span=9, adjust=False).mean().iloc[-1]
            current_1h = closes_1h.iloc[-1]
            btc_trend["1h"] = "bullish" if current_1h > ema9_1h else "bearish"
        
        # Get 4h trend
        klines_4h = client.get_historical_klines(
            "BTCUSDC", 
            Client.KLINE_INTERVAL_4HOUR, 
            "4 days ago UTC"
        )
        
        if len(klines_4h) >= 10:
            df_4h = pd.DataFrame(klines_4h, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])
            closes_4h = df_4h['close'].astype(float)
            ema9_4h = closes_4h.ewm(span=9, adjust=False).mean().iloc[-1]
            current_4h = closes_4h.iloc[-1]
            btc_trend["4h"] = "bullish" if current_4h > ema9_4h else "bearish"
        
        btc_trend["last_update"] = datetime.datetime.now()
        
        return btc_trend
        
    except Exception as e:
        print(f"Error getting Bitcoin trend: {e}")
        return btc_trend

# Technical Indicator Calculations
def calculate_ema(prices, period):
    """Calculate Exponential Moving Average"""
    if len(prices) < period:
        return None
    
    prices = np.array(prices)
    alpha = 2 / (period + 1)
    ema = np.zeros(len(prices))
    ema[0] = prices[0]
    
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    
    return ema[-1]

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    if len(prices) < period + 1:
        return None
    
    prices = np.array(prices)
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    
    if down == 0:
        return 100
    
    rs = up / down
    rsi = 100 - (100 / (1 + rs))
    
    for delta in deltas[period:]:
        if delta > 0:
            upval = delta
            downval = 0
        else:
            upval = 0
            downval = -delta
        
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        
        if down == 0:
            return 100
        
        rs = up / down
        rsi = 100 - (100 / (1 + rs))
    
    return rsi

def detect_rsi_divergence(prices, rsi_values, lookback=10):
    """Detect RSI divergence (bearish or bullish)"""
    if len(prices) < lookback or len(rsi_values) < lookback:
        return None
    
    prices = np.array(prices[-lookback:])
    rsi = np.array(rsi_values[-lookback:])
    
    # Find price peaks and troughs
    price_high_idx = np.argmax(prices)
    price_low_idx = np.argmin(prices)
    
    # Bearish divergence: price makes higher high, RSI makes lower high
    if price_high_idx > lookback // 2:  # Recent high
        earlier_high_idx = np.argmax(prices[:price_high_idx-1])
        if prices[price_high_idx] > prices[earlier_high_idx] and rsi[price_high_idx] < rsi[earlier_high_idx]:
            return "bearish"
    
    # Bullish divergence: price makes lower low, RSI makes higher low
    if price_low_idx > lookback // 2:  # Recent low
        earlier_low_idx = np.argmin(prices[:price_low_idx-1])
        if prices[price_low_idx] < prices[earlier_low_idx] and rsi[price_low_idx] > rsi[earlier_low_idx]:
            return "bullish"
    
    return None

def detect_volume_divergence(prices, volumes, lookback=10):
    """Detect volume divergence"""
    if len(prices) < lookback or len(volumes) < lookback:
        return False
    
    prices = np.array(prices[-lookback:])
    volumes = np.array(volumes[-lookback:])
    
    # Find recent high
    recent_high_idx = np.argmax(prices[-5:]) + (len(prices) - 5)
    
    # Check if making new high with decreasing volume
    if recent_high_idx == len(prices) - 1:  # Latest candle is highest
        avg_volume = np.mean(volumes[:-1])
        if volumes[-1] < avg_volume * 0.7:  # Volume is 30% below average
            return True
    
    return False

def calculate_obv(close, volume):
    """Calculate On-Balance Volume"""
    if len(close) < 2:
        return None
    
    close = np.array(close)
    volume = np.array(volume)
    
    obv = np.zeros(len(close))
    obv[0] = volume[0]
    
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    
    return obv

def identify_market_structure(highs, lows, lookback=20):
    """Identify market structure (trending up, down, or ranging)"""
    if len(highs) < lookback or len(lows) < lookback:
        return "unknown"
    
    recent_highs = highs[-lookback:]
    recent_lows = lows[-lookback:]
    
    # Check for higher highs and higher lows (uptrend)
    higher_highs = 0
    higher_lows = 0
    lower_highs = 0
    lower_lows = 0
    
    for i in range(1, len(recent_highs)):
        if recent_highs[i] > recent_highs[i-1]:
            higher_highs += 1
        else:
            lower_highs += 1
        
        if recent_lows[i] > recent_lows[i-1]:
            higher_lows += 1
        else:
            lower_lows += 1
    
    if higher_highs > lower_highs and higher_lows > lower_lows:
        return "uptrend"
    elif lower_highs > higher_highs and lower_lows > higher_lows:
        return "downtrend"
    else:
        return "ranging"

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    if len(high) < period + 1:
        return None, None, None
    
    high = np.array(high)
    low = np.array(low)
    close = np.array(close)
    
    # Calculate True Range
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    
    for i in range(1, len(high)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    # Calculate directional movements
    plus_dm = np.zeros(len(high))
    minus_dm = np.zeros(len(high))
    
    for i in range(1, len(high)):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth the TR and DMs
    atr = pd.Series(tr).rolling(window=period).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean().values / atr
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
    adx = pd.Series(dx).rolling(window=period).mean().values
    
    return adx[-1], plus_di[-1], minus_di[-1]

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD and Signal line"""
    if len(prices) < slow + signal:
        return None, None, None
    
    prices = np.array(prices)
    
    # Calculate EMAs
    ema_fast = pd.Series(prices).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(prices).ewm(span=slow, adjust=False).mean().values
    
    # MACD line
    macd_line = ema_fast - ema_slow
    
    # Signal line
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    
    # Histogram
    histogram = macd_line - signal_line
    
    return macd_line[-1], signal_line[-1], histogram[-1]

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    if len(high) < period + 1:
        return None
    
    high = np.array(high)
    low = np.array(low)
    close = np.array(close)
    
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    
    for i in range(1, len(high)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr = pd.Series(tr).rolling(window=period).mean().values
    return atr[-1]

def get_multi_timeframe_alignment(symbol):
    """Check trend alignment across multiple timeframes"""
    try:
        alignment_score = 0
        timeframe_trends = {}
        
        # 15-minute trend (changed from 5m analysis since 5m is now for patterns)
        klines_15m_trend = client.get_historical_klines(
            symbol, 
            Client.KLINE_INTERVAL_15MINUTE, 
            f"{candles_lookback_15m * 15} minutes ago UTC"
        )
        
        if len(klines_15m_trend) >= 20:
            df_15m = pd.DataFrame(klines_15m_trend, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])
            closes_15m = df_15m['close'].astype(float).tolist()
            ema9_15m = calculate_ema(closes_15m, 9)
            ema21_15m = calculate_ema(closes_15m, 21)
            
            if ema9_15m and ema21_15m:
                if closes_15m[-1] > ema9_15m > ema21_15m:
                    timeframe_trends["15m"] = "bullish"
                    alignment_score += 20
                elif closes_15m[-1] < ema9_15m < ema21_15m:
                    timeframe_trends["15m"] = "bearish"
                    alignment_score -= 20
                else:
                    timeframe_trends["15m"] = "neutral"
        
        # 1-hour trend
        klines_1h = client.get_historical_klines(
            symbol, 
            Client.KLINE_INTERVAL_1HOUR, 
            f"{candles_lookback_1h} hours ago UTC"
        )
        
        if len(klines_1h) >= 20:
            df_1h = pd.DataFrame(klines_1h, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])
            closes_1h = df_1h['close'].astype(float).tolist()
            highs_1h = df_1h['high'].astype(float).tolist()
            lows_1h = df_1h['low'].astype(float).tolist()
            
            # Calculate ADX for trend strength
            adx, plus_di, minus_di = calculate_adx(highs_1h, lows_1h, closes_1h)
            
            ema9_1h = calculate_ema(closes_1h, 9)
            ema21_1h = calculate_ema(closes_1h, 21)
            
            if ema9_1h and ema21_1h:
                if closes_1h[-1] > ema9_1h > ema21_1h:
                    timeframe_trends["1h"] = "bullish"
                    alignment_score += 30
                    if adx and adx > adx_trend_threshold:
                        alignment_score += 10  # Bonus for strong trend
                elif closes_1h[-1] < ema9_1h < ema21_1h:
                    timeframe_trends["1h"] = "bearish"
                    alignment_score -= 30
                else:
                    timeframe_trends["1h"] = "neutral"
            
            timeframe_trends["adx"] = adx
            timeframe_trends["trend_strength"] = "strong" if adx and adx > adx_trend_threshold else "weak"
        
        # 4-hour trend
        klines_4h = client.get_historical_klines(
            symbol, 
            Client.KLINE_INTERVAL_4HOUR, 
            f"{candles_lookback_4h * 4} hours ago UTC"
        )
        
        if len(klines_4h) >= 20:
            df_4h = pd.DataFrame(klines_4h, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])
            closes_4h = df_4h['close'].astype(float).tolist()
            ema9_4h = calculate_ema(closes_4h, 9)
            ema21_4h = calculate_ema(closes_4h, 21)
            ema55_4h = calculate_ema(closes_4h, 55)
            
            if ema9_4h and ema21_4h and ema55_4h:
                if closes_4h[-1] > ema9_4h > ema21_4h > ema55_4h:
                    timeframe_trends["4h"] = "strong_bullish"
                    alignment_score += 40
                elif closes_4h[-1] > ema21_4h:
                    timeframe_trends["4h"] = "bullish"
                    alignment_score += 20
                elif closes_4h[-1] < ema9_4h < ema21_4h < ema55_4h:
                    timeframe_trends["4h"] = "strong_bearish"
                    alignment_score -= 40
                elif closes_4h[-1] < ema21_4h:
                    timeframe_trends["4h"] = "bearish"
                    alignment_score -= 20
                else:
                    timeframe_trends["4h"] = "neutral"
        
        return alignment_score, timeframe_trends
        
    except Exception as e:
        print(f"Error getting multi-timeframe alignment for {symbol}: {e}")
        return 0, {}

def calculate_entry_score(symbol, pattern_text, current_price, df_5m, indicators):
    """Calculate comprehensive entry score with all filters"""
    entry_score = 0
    entry_reasons = []
    entry_warnings = []
    
    # 1. Support/Resistance Analysis (30 points max)
    highs = df_5m['high'].tolist()
    lows = df_5m['low'].tolist()
    support_levels, resistance_levels, near_level = detect_support_resistance(
        coins[symbol]["swing_highs"], 
        coins[symbol]["swing_lows"], 
        current_price
    )
    
    if near_level == "support":
        entry_score += 30
        entry_reasons.append("Near support level")
    elif near_level == "resistance":
        entry_score -= 20
        entry_warnings.append("Near resistance (avoid)")
    else:
        entry_score += 10  # Neutral zone
    
    # 2. Bitcoin Correlation Check (20 points max)
    btc = get_bitcoin_trend()
    if btc["15m"] == "bullish":
        entry_score += 20
        entry_reasons.append("BTC 15m bullish")
    elif btc["15m"] == "bearish":
        if btc["strength"] and btc["strength"] < -1:  # Strong bearish
            entry_score -= 30
            entry_warnings.append(f"BTC strongly bearish ({btc['strength']:.1f}%)")
        else:
            entry_score -= 10
            entry_warnings.append("BTC 15m bearish")
    
    # Check longer BTC timeframes
    if btc["1h"] == "bearish" and btc["4h"] == "bearish":
        entry_score -= 20
        entry_warnings.append("BTC bearish on higher timeframes")
    
    # 3. Multi-Timeframe Alignment (40 points max)
    alignment_score, timeframe_trends = get_multi_timeframe_alignment(symbol)
    entry_score += alignment_score
    
    if alignment_score >= 30:
        entry_reasons.append(f"Timeframes aligned ({alignment_score})")
    elif alignment_score <= -20:
        entry_warnings.append(f"Timeframe conflict ({alignment_score})")
    
    # ADX trend strength bonus
    if timeframe_trends.get("adx") and timeframe_trends["adx"] > adx_trend_threshold:
        entry_score += 10
        entry_reasons.append(f"ADX strong trend ({timeframe_trends['adx']:.1f})")
    
    # 4. Volume Profile Analysis (20 points max)
    poc, avg_volume = calculate_volume_profile(df_5m)
    current_volume = df_5m['volume'].iloc[-1]
    avg_recent_volume = df_5m['volume'].tail(20).mean()
    
    if current_volume > avg_recent_volume * 1.5:
        entry_score += 15
        entry_reasons.append("Volume surge")
    elif current_volume < avg_recent_volume * min_volume_ratio:
        entry_score -= 10
        entry_warnings.append("Low volume")
    
    # Check if price is near Point of Control
    if poc and abs(current_price - poc) / current_price < 0.01:
        entry_score += 5
        entry_reasons.append("At volume POC")
    
    # 5. RSI Analysis (15 points max)
    rsi = indicators.get('rsi')
    if rsi:
        if 35 <= rsi <= 65:
            entry_score += 15
            entry_reasons.append(f"RSI neutral zone ({rsi:.1f})")
        elif rsi < 35:
            entry_score += 10
            entry_reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 70:
            entry_score -= 15
            entry_warnings.append(f"RSI overbought ({rsi:.1f})")
    
    # 6. Pattern Quality (15 points max based on pattern strength)
    pattern_quality_score = 0
    if "Morning Star" in pattern_text or "Bullish Engulfing" in pattern_text:
        pattern_quality_score = 15
    elif "Hammer" in pattern_text:
        pattern_quality_score = 12
    elif "Doji" in pattern_text:
        pattern_quality_score = 8
    
    entry_score += pattern_quality_score
    if pattern_quality_score > 10:
        entry_reasons.append(f"Strong pattern ({pattern_text})")
    
    # 7. Market Conditions Check (10 points max)
    current_hour = datetime.datetime.now().hour
    
    # Avoid low liquidity hours (2-6 AM UTC)
    if 2 <= current_hour <= 6:
        entry_score -= 5
        entry_warnings.append("Low liquidity hour")
    
    # Check total open positions
    open_positions = sum(1 for coin in coins.values() if coin["position_is_open"])
    if open_positions >= market_analysis["correlation_limit"]:
        entry_score -= 20
        entry_warnings.append(f"Too many positions ({open_positions})")
    
    # 8. MACD Momentum (10 points max)
    macd = indicators.get('macd')
    macd_signal = indicators.get('macd_signal')
    if macd and macd_signal:
        if macd > macd_signal and macd > 0:
            entry_score += 10
            entry_reasons.append("MACD bullish")
        elif macd < macd_signal:
            entry_score -= 5
            entry_warnings.append("MACD bearish")
    
    # Cap the score at 100
    entry_score = min(100, max(0, entry_score))
    
    return entry_score, entry_reasons, entry_warnings

def get_comprehensive_indicators(symbol, client):
    """Get all technical indicators for a symbol"""
    try:
        indicators = {}
        
        # Fetch 1-hour data for RSI, MACD, Volume analysis
        klines_1h = client.get_historical_klines(
            symbol, 
            Client.KLINE_INTERVAL_1HOUR, 
            f"{candles_lookback_1h} hours ago UTC"
        )
        
        if len(klines_1h) >= 30:
            df_1h = pd.DataFrame(klines_1h, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])
            
            closes_1h = df_1h['close'].astype(float).tolist()
            highs_1h = df_1h['high'].astype(float).tolist()
            lows_1h = df_1h['low'].astype(float).tolist()
            volumes_1h = df_1h['volume'].astype(float).tolist()
            
            # RSI
            indicators['rsi'] = calculate_rsi(closes_1h)
            
            # MACD
            macd, signal, histogram = calculate_macd(closes_1h)
            indicators['macd'] = macd
            indicators['macd_signal'] = signal
            indicators['macd_histogram'] = histogram
            
            # ATR for dynamic stop loss
            indicators['atr'] = calculate_atr(highs_1h, lows_1h, closes_1h)
            
            # ADX for trend strength
            adx, plus_di, minus_di = calculate_adx(highs_1h, lows_1h, closes_1h)
            indicators['adx'] = adx
            indicators['plus_di'] = plus_di
            indicators['minus_di'] = minus_di
            
            # Volume analysis
            indicators['current_volume'] = volumes_1h[-1]
            indicators['avg_volume'] = np.mean(volumes_1h[:-1])
            indicators['volume_spike'] = volumes_1h[-1] > (indicators['avg_volume'] * volume_spike_multiplier)
            
            # RSI Divergence
            if len(closes_1h) >= rsi_divergence_lookback:
                rsi_values = [calculate_rsi(closes_1h[:i+15]) for i in range(len(closes_1h)-14)]
                rsi_values = [r for r in rsi_values if r is not None]
                indicators['rsi_divergence'] = detect_rsi_divergence(closes_1h, rsi_values, rsi_divergence_lookback)
            
            # Volume Divergence
            indicators['volume_divergence'] = detect_volume_divergence(closes_1h, volumes_1h)
            
            # OBV
            obv_values = calculate_obv(closes_1h, volumes_1h)
            indicators['obv'] = obv_values[-1] if obv_values is not None else None
            indicators['obv_trend'] = "up" if obv_values is not None and len(obv_values) > 1 and obv_values[-1] > obv_values[-2] else "down"
            
            # Market Structure
            indicators['market_structure'] = identify_market_structure(highs_1h, lows_1h)
        
        # Fetch 4-hour data for EMA trend
        klines_4h = client.get_historical_klines(
            symbol, 
            Client.KLINE_INTERVAL_4HOUR, 
            f"{candles_lookback_4h * 4} hours ago UTC"
        )
        
        if len(klines_4h) >= ema_long_period:
            df_4h = pd.DataFrame(klines_4h, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])
            
            closes_4h = df_4h['close'].astype(float).tolist()
            
            # Triple EMA system
            indicators['ema1_4h'] = calculate_ema(closes_4h, ema_short_period)
            indicators['ema21_4h'] = calculate_ema(closes_4h, ema_medium_period)
            indicators['ema99_4h'] = calculate_ema(closes_4h, ema_long_period)
            
            # EMA trend strength
            if indicators['ema1_4h'] and indicators['ema21_4h'] and indicators['ema99_4h']:
                if indicators['ema1_4h'] > indicators['ema21_4h'] > indicators['ema99_4h']:
                    indicators['ema_trend'] = "strong_bullish"
                elif indicators['ema1_4h'] > indicators['ema99_4h']:
                    indicators['ema_trend'] = "bullish"
                elif indicators['ema1_4h'] < indicators['ema21_4h'] < indicators['ema99_4h']:
                    indicators['ema_trend'] = "strong_bearish"
                elif indicators['ema1_4h'] < indicators['ema99_4h']:
                    indicators['ema_trend'] = "bearish"
                else:
                    indicators['ema_trend'] = "neutral"
                
                # EMA slope (rate of change)
                if len(closes_4h) >= 5:
                    ema1_prev = calculate_ema(closes_4h[:-1], ema_short_period)
                    if ema1_prev:
                        indicators['ema_slope'] = (indicators['ema1_4h'] - ema1_prev) / ema1_prev
        
        return indicators
        
    except Exception as e:
        print(f"Error calculating indicators for {symbol}: {e}")
        return {}

# Candlestick Pattern Detection Functions
@tenacity.retry(wait=tenacity.wait_fixed(10), stop=tenacity.stop_after_delay(300))
def get_candle_data(symbol, interval, client, limit=100):
    """Fetch recent candlestick data for pattern analysis"""
    try:
        if interval == Client.KLINE_INTERVAL_5MINUTE:
            klines = client.get_historical_klines(
                symbol, interval, f"{limit * 5} minutes ago UTC"
            )
        elif interval == Client.KLINE_INTERVAL_15MINUTE:
            klines = client.get_historical_klines(
                symbol, interval, f"{limit * 15} minutes ago UTC"
            )
        elif interval == Client.KLINE_INTERVAL_1HOUR:
            klines = client.get_historical_klines(
                symbol, interval, f"{limit} hours ago UTC"
            )
        else:  # 4-hour interval
            klines = client.get_historical_klines(
                symbol, interval, f"{limit * 4} hours ago UTC"
            )
            
        df = pd.DataFrame(
            klines,
            columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
            ],
        )
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        return df
    except Exception as e:
        print(f"Error fetching candle data for {symbol}: {e}")
        raise tenacity.TryAgain

def is_bullish_candle(open_price, close_price):
    return close_price > open_price

def is_bearish_candle(open_price, close_price):
    return close_price < open_price

def candle_body_size(open_price, close_price):
    return abs(close_price - open_price)

def candle_range(high_price, low_price):
    return high_price - low_price

def upper_shadow(high_price, open_price, close_price):
    return high_price - max(open_price, close_price)

def lower_shadow(low_price, open_price, close_price):
    return min(open_price, close_price) - low_price

# Pattern Detection Functions
def is_hammer(row):
    """Detect Hammer pattern (bullish reversal)"""
    open_price, high_price, low_price, close_price = row['open'], row['high'], row['low'], row['close']
    body = candle_body_size(open_price, close_price)
    range_candle = candle_range(high_price, low_price)
    lower_shad = lower_shadow(low_price, open_price, close_price)
    upper_shad = upper_shadow(high_price, open_price, close_price)
    
    if range_candle > 0 and body > 0:
        return (lower_shad >= 2 * body and 
                upper_shad <= body * 0.5 and
                body <= range_candle * 0.3)
    return False

def is_bullish_engulfing(prev_row, curr_row):
    """Detect Bullish Engulfing pattern"""
    if not is_bearish_candle(prev_row['open'], prev_row['close']):
        return False
    
    return (is_bullish_candle(curr_row['open'], curr_row['close']) and
            curr_row['open'] < prev_row['close'] and
            curr_row['close'] > prev_row['open'])

def is_doji(row):
    """Detect Doji pattern (indecision)"""
    open_price, close_price = row['open'], row['close']
    high_price, low_price = row['high'], row['low']
    body = candle_body_size(open_price, close_price)
    range_candle = candle_range(high_price, low_price)
    
    if range_candle > 0:
        return body <= range_candle * 0.1
    return False

def is_morning_star(df, idx):
    """Detect Morning Star pattern (3-candle bullish reversal)"""
    if idx < 2:
        return False
    
    candle1 = df.iloc[idx-2]
    candle2 = df.iloc[idx-1]
    candle3 = df.iloc[idx]
    
    if not is_bearish_candle(candle1['open'], candle1['close']):
        return False
    
    middle_body = candle_body_size(candle2['open'], candle2['close'])
    first_body = candle_body_size(candle1['open'], candle1['close'])
    if middle_body >= first_body * 0.5:
        return False
    
    return (is_bullish_candle(candle3['open'], candle3['close']) and
            candle3['close'] > (candle1['open'] + candle1['close']) / 2)

def is_shooting_star(row):
    """Detect Shooting Star pattern (bearish reversal)"""
    open_price, high_price, low_price, close_price = row['open'], row['high'], row['low'], row['close']
    body = candle_body_size(open_price, close_price)
    range_candle = candle_range(high_price, low_price)
    lower_shad = lower_shadow(low_price, open_price, close_price)
    upper_shad = upper_shadow(high_price, open_price, close_price)
    
    if range_candle > 0 and body > 0:
        return (upper_shad >= 2 * body and 
                lower_shad <= body * 0.5 and
                body <= range_candle * 0.3)
    return False

def is_bearish_engulfing(prev_row, curr_row):
    """Detect Bearish Engulfing pattern"""
    if not is_bullish_candle(prev_row['open'], prev_row['close']):
        return False
    
    return (is_bearish_candle(curr_row['open'], curr_row['close']) and
            curr_row['open'] > prev_row['close'] and
            curr_row['close'] < prev_row['open'])

def is_evening_star(df, idx):
    """Detect Evening Star pattern (3-candle bearish reversal)"""
    if idx < 2:
        return False
    
    candle1 = df.iloc[idx-2]
    candle2 = df.iloc[idx-1]
    candle3 = df.iloc[idx]
    
    if not is_bullish_candle(candle1['open'], candle1['close']):
        return False
    
    middle_body = candle_body_size(candle2['open'], candle2['close'])
    first_body = candle_body_size(candle1['open'], candle1['close'])
    if middle_body >= first_body * 0.5:
        return False
    
    return (is_bearish_candle(candle3['open'], candle3['close']) and
            candle3['close'] < (candle1['open'] + candle1['close']) / 2)

def analyze_candlestick_patterns(df):
    """Analyze candlestick patterns and return buy/sell signals"""
    if len(df) < 3:
        return None, None, None
    
    latest_idx = len(df) - 1
    latest_candle = df.iloc[latest_idx]
    
    buy_signals = []
    sell_signals = []
    
    # Single candle patterns
    if is_hammer(latest_candle):
        buy_signals.append("Hammer")
    
    if is_shooting_star(latest_candle):
        sell_signals.append("Shooting Star")
    
    if is_doji(latest_candle):
        buy_signals.append("Doji")
    
    # Two candle patterns
    if latest_idx >= 1:
        prev_candle = df.iloc[latest_idx - 1]
        
        if is_bullish_engulfing(prev_candle, latest_candle):
            buy_signals.append("Bullish Engulfing")
        
        if is_bearish_engulfing(prev_candle, latest_candle):
            sell_signals.append("Bearish Engulfing")
    
    # Three candle patterns
    if is_morning_star(df, latest_idx):
        buy_signals.append("Morning Star")
    
    if is_evening_star(df, latest_idx):
        sell_signals.append("Evening Star")
    
    return buy_signals, sell_signals, latest_candle['close']

# Trading Functions (simplified versions for exit logic)
def buy(symbol, usd_amount, pattern, entry_score, entry_reasons):
    """Execute a real buy order with enhanced logging"""
    try:
        mode_prefix = "🔸 [DRY RUN]" if DRY_RUN else ""
        print(f"{colored(f'{mode_prefix} 🟢 EXECUTING BUY ORDER', 'green', attrs=['bold'])} for {symbol}")
        print(f" - Pattern: {colored(pattern, 'yellow')}")
        print(f" - Entry Score: {colored(f'{entry_score}/100', 'green')}")
        print(f" - Confirmations: {', '.join(entry_reasons[:3])}")
        print(f" - Amount: {usd_amount} USDT")
        
        if DRY_RUN:
            # SIMULATE buy order
            ticker = client.get_symbol_ticker(symbol=symbol)
            buy_price = float(ticker['price'])
            quantity = usd_amount / buy_price
            print(colored(f" 🔸 [DRY RUN] Simulating market buy at {buy_price:.6f}", "yellow"))
            order = {'orderId': 'DRY_RUN_' + str(int(time.time())), 
                     'fills': [{'price': str(buy_price), 'qty': str(quantity)}]}
        else:
            # REAL buy order
            order = client.order_market_buy(
                symbol=symbol,
                quoteOrderQty=usd_amount
            )
        
        if 'orderId' in order:
            quantity = sum(float(fill['qty']) for fill in order['fills'])
            buy_price = float(order['fills'][0]['price'])
            
            coins[symbol]["bought_quantity"] = quantity
            coins[symbol]["remaining_quantity"] = quantity
            coins[symbol]["buy_price"] = buy_price
            coins[symbol]["position_is_open"] = True
            coins[symbol]["entry_date"] = datetime.datetime.now().isoformat()
            coins[symbol]["entry_pattern"] = pattern
            coins[symbol]["highest_price"] = buy_price
            coins[symbol]["partial_exits"] = []
            
            # Track pattern performance
            if pattern not in coins[symbol]["pattern_success_rate"]:
                coins[symbol]["pattern_success_rate"][pattern] = {"wins": 0, "losses": 0}
            
            # Save to file
            with open(coins[symbol]["filename_order_id"], "w") as f:
                f.write(f"{buy_price},{quantity},{coins[symbol]['entry_date']},{pattern}")
            
            status_text = ' ✅ BUY SUCCESS (SIMULATED)' if DRY_RUN else ' ✅ BUY SUCCESS'
            print(f"{colored(status_text, 'green')} | Order ID: {order['orderId']}")
            print(f" - Quantity: {quantity:.8f} {symbol[:-4]}")
            print(f" - Price: {buy_price:.4f} USDT")
            
            # Update market analysis
            market_analysis["total_positions"] = sum(1 for coin in coins.values() if coin["position_is_open"])
            
            return True
        else:
            print(f"{colored(' ❌ BUY FAILED', 'red')} - No order ID returned")
            return False
            
    except Exception as e:
        print(f"{colored(f' ❌ BUY ERROR for {symbol}: {e}', 'red')}")
        return False

def calculate_exit_score(symbol, current_price, indicators):
    """Calculate a comprehensive exit score based on multiple indicators"""
    if not coins[symbol]["position_is_open"]:
        return 0, []
    
    exit_score = 0
    exit_reasons = []
    
    buy_price = coins[symbol]["buy_price"]
    price_change = (current_price - buy_price) / buy_price
    
    # 1. Volume divergence (high weight)
    if indicators.get('volume_divergence'):
        exit_score += 30
        exit_reasons.append("Volume divergence detected")
    
    # 2. RSI conditions
    rsi = indicators.get('rsi')
    if rsi:
        if rsi > rsi_overbought:
            exit_score += 25
            exit_reasons.append(f"RSI overbought ({rsi:.1f})")
        elif rsi < rsi_oversold and price_change < 0:
            exit_score += 20
            exit_reasons.append(f"RSI oversold with loss ({rsi:.1f})")
        
        # RSI divergence
        if indicators.get('rsi_divergence') == 'bearish':
            exit_score += 35
            exit_reasons.append("Bearish RSI divergence")
    
    # 3. MACD histogram declining
    macd_hist = indicators.get('macd_histogram')
    if macd_hist is not None:
        coins[symbol]["last_macd_histogram"].append(macd_hist)
        if len(coins[symbol]["last_macd_histogram"]) >= 3:
            hist_values = list(coins[symbol]["last_macd_histogram"])
            if all(hist_values[i] > hist_values[i+1] for i in range(len(hist_values)-1)):
                exit_score += 25
                exit_reasons.append("MACD histogram declining")
    
    # 4. EMA trend analysis
    ema_trend = indicators.get('ema_trend')
    if ema_trend == 'strong_bearish':
        exit_score += 40
        exit_reasons.append("Strong bearish EMA trend")
    elif ema_trend == 'bearish' and price_change < 0:
        exit_score += 30
        exit_reasons.append("Bearish EMA trend with loss")
    
    # EMA slope turning negative
    ema_slope = indicators.get('ema_slope')
    if ema_slope is not None and ema_slope < -0.001:
        exit_score += 20
        exit_reasons.append("EMA momentum declining")
    
    # 5. Market structure breakdown
    if indicators.get('market_structure') == 'downtrend':
        exit_score += 25
        exit_reasons.append("Market structure breakdown")
    
    # 6. OBV trend
    if indicators.get('obv_trend') == 'down' and price_change < 0:
        exit_score += 20
        exit_reasons.append("OBV declining with price")
    
    # 7. Time-based exit
    if coins[symbol]["entry_date"]:
        entry_time = datetime.datetime.fromisoformat(coins[symbol]["entry_date"])
        hours_held = (datetime.datetime.now() - entry_time).total_seconds() / 3600
        if hours_held > time_stop_hours and price_change < 0.005:
            exit_score += 30
            exit_reasons.append(f"Time stop ({hours_held:.1f}h)")
    
    # 8. ATR-based trailing stop
    atr = indicators.get('atr')
    if atr and coins[symbol]["highest_price"]:
        trailing_stop = coins[symbol]["highest_price"] - (atr * atr_multiplier)
        if current_price < trailing_stop:
            exit_score += 50
            exit_reasons.append(f"ATR trailing stop hit")
    
    return exit_score, exit_reasons

def sell_partial(symbol, sell_percent, reason="Partial Exit"):
    """Execute a partial sell order"""
    try:
        if not coins[symbol]["position_is_open"]:
            print(f" - No open position for {symbol}")
            return None
            
        if coins[symbol]["remaining_quantity"] is None or coins[symbol]["remaining_quantity"] <= 0:
            print(f" - No remaining quantity to sell for {symbol}")
            # Clean up the position state
            coins[symbol]["position_is_open"] = False
            coins[symbol]["buy_price"] = None
            coins[symbol]["bought_quantity"] = None
            coins[symbol]["remaining_quantity"] = None
            coins[symbol]["entry_date"] = None
            coins[symbol]["entry_pattern"] = None
            coins[symbol]["highest_price"] = None
            with open(coins[symbol]["filename_order_id"], "w") as f:
                f.write("")
            return None
        
        total_quantity = coins[symbol]["bought_quantity"]
        remaining_quantity = coins[symbol]["remaining_quantity"]
        quantity_to_sell = total_quantity * sell_percent
        
        # Don't sell more than we have
        quantity_to_sell = min(quantity_to_sell, remaining_quantity)
        
        mode_prefix = "🔸 [DRY RUN]" if DRY_RUN else ""
        print(f"{colored(f'{mode_prefix} 🔶 PARTIAL SELL ({sell_percent*100:.0f}%)', 'yellow', attrs=['bold'])} for {symbol}")
        print(f" - Reason: {colored(reason, 'cyan')}")
        print(f" - Quantity: {quantity_to_sell:.8f}")
        
        # Get symbol info for precision and minimum notional
        exchange_info = client.get_symbol_info(symbol)
        lot_size_filter = next(f for f in exchange_info['filters'] if f['filterType'] == 'LOT_SIZE')
        step_size = float(lot_size_filter['stepSize'])
        precision = int(round(-math.log(step_size, 10), 0))
        
        # Get minimum notional value
        min_notional_filter = next((f for f in exchange_info['filters'] if f['filterType'] == 'NOTIONAL'), None)
        min_notional = float(min_notional_filter['minNotional']) if min_notional_filter else 10.0
        
        # Check if the order value meets minimum notional requirement
        ticker = client.get_symbol_ticker(symbol=symbol)
        current_price = float(ticker['price'])
        order_value = quantity_to_sell * current_price
        
        if order_value < min_notional and not DRY_RUN:
            print(f" - Order value (${order_value:.2f}) below minimum (${min_notional:.2f})")
            print(f" - Selling full remaining position instead")
            # Sell everything if partial amount is below minimum
            quantity_to_sell = remaining_quantity
            order_value = quantity_to_sell * current_price
            if order_value < min_notional:
                print(f" - Full position still below minimum, cannot sell")
                return None
        
        # Adjust quantity to match Binance precision
        quantity_to_sell = round(quantity_to_sell, precision)
        
        if quantity_to_sell <= 0 and not DRY_RUN:
            print(f" - Quantity too small to sell")
            return None
        
        if DRY_RUN:
            # SIMULATE sell order
            print(colored(f" 🔸 [DRY RUN] Simulating partial sell at {current_price:.6f}", "yellow"))
            order = {'orderId': 'DRY_RUN_SELL_' + str(int(time.time())),
                     'executedQty': str(quantity_to_sell),
                     'cummulativeQuoteQty': str(quantity_to_sell * current_price)}
        else:
            # REAL sell order
            order = client.order_market_sell(
                symbol=symbol,
                quantity=quantity_to_sell
            )
        
        executed_qty = float(order.get('executedQty', 0))
        cummulative_quote_qty = float(order.get('cummulativeQuoteQty', 0))
        
        if executed_qty > 0:
            sell_price = cummulative_quote_qty / executed_qty
            buy_price = coins[symbol]["buy_price"]
            
            # Calculate P&L for this partial exit
            gross_profit = cummulative_quote_qty - (buy_price * executed_qty)
            sell_fee = cummulative_quote_qty * trade_fee_percent
            buy_fee = (buy_price * executed_qty) * trade_fee_percent
            profit_or_loss = gross_profit - sell_fee - buy_fee
            
            # Update remaining quantity
            coins[symbol]["remaining_quantity"] -= executed_qty
            
            # Record partial exit
            exit_record = {
                "timestamp": datetime.datetime.now().isoformat(),
                "quantity": executed_qty,
                "sell_price": sell_price,
                "profit_loss": profit_or_loss,
                "reason": reason
            }
            coins[symbol]["partial_exits"].append(exit_record)
            
            # Save partial exits to file
            with open(coins[symbol]["filename_partial_exits"], "w") as f:
                json.dump(coins[symbol]["partial_exits"], f)
            
            print(f"{colored(' ✅ PARTIAL SELL SUCCESS', 'cyan')} | Order ID: {order['orderId']}")
            print(f" - Sell Price: {sell_price:.4f} USDT")
            print(f" - Partial P&L: {colored(f'{profit_or_loss:+.2f} USDT', 'green' if profit_or_loss > 0 else 'red')}")
            print(f" - Remaining: {coins[symbol]['remaining_quantity']:.8f} {symbol[:-4]}")
            
            # Update overall status
            overall_status[symbol] += profit_or_loss
            
            # If we've sold everything, close the position
            if coins[symbol]["remaining_quantity"] <= 0:
                print(f" - Position fully exited, closing position for {symbol}")
                coins[symbol]["position_is_open"] = False
                coins[symbol]["buy_price"] = None
                coins[symbol]["bought_quantity"] = None
                coins[symbol]["remaining_quantity"] = None
                coins[symbol]["entry_date"] = None
                coins[symbol]["entry_pattern"] = None
                coins[symbol]["highest_price"] = None
                coins[symbol]["partial_exits"] = []
                with open(coins[symbol]["filename_order_id"], "w") as f:
                    f.write("")
                with open(coins[symbol]["filename_partial_exits"], "w") as f:
                    f.write("[]")
            
            return {
                'sell_price': sell_price,
                'profit_loss': profit_or_loss,
                'executed_qty': executed_qty
            }
        else:
            print(f"{colored(' ❌ PARTIAL SELL FAILED', 'red')} - Zero quantity executed")
            return None
            
    except Exception as e:
        print(f"{colored(f' ❌ PARTIAL SELL ERROR for {symbol}: {e}', 'red')}")
        return None

def sell(symbol, reason="Manual"):
    """Execute a full sell order (simplified for space)"""
    try:
        if not coins[symbol]["position_is_open"]:
            return None
        
        if coins[symbol]["remaining_quantity"] is None or coins[symbol]["remaining_quantity"] <= 0:
            # Clean up the position state
            coins[symbol]["position_is_open"] = False
            return None
        
        quantity = coins[symbol]["remaining_quantity"]
        buy_price = coins[symbol]["buy_price"]
        
        mode_prefix = "🔸 [DRY RUN]" if DRY_RUN else ""
        print(f"{colored(f'{mode_prefix} 🔴 EXECUTING SELL', 'red')} for {symbol} - Reason: {reason}")
        
        # Get symbol info for precision
        exchange_info = client.get_symbol_info(symbol)
        lot_size_filter = next(f for f in exchange_info['filters'] if f['filterType'] == 'LOT_SIZE')
        step_size = float(lot_size_filter['stepSize'])
        precision = int(round(-math.log(step_size, 10), 0))
        
        # Adjust quantity to match Binance precision
        quantity = round(quantity, precision)
        
        if DRY_RUN:
            # SIMULATE sell order
            ticker = client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
            print(colored(f" 🔸 [DRY RUN] Simulating market sell at {current_price:.6f}", "yellow"))
            order = {'orderId': 'DRY_RUN_SELL_' + str(int(time.time())),
                     'executedQty': str(quantity),
                     'cummulativeQuoteQty': str(quantity * current_price)}
        else:
            # REAL sell order
            order = client.order_market_sell(
                symbol=symbol,
                quantity=quantity
            )
        
        executed_qty = float(order.get('executedQty', 0))
        cummulative_quote_qty = float(order.get('cummulativeQuoteQty', 0))
        
        if executed_qty > 0:
            sell_price = cummulative_quote_qty / executed_qty
            
            # Calculate P&L
            gross_profit = cummulative_quote_qty - (buy_price * executed_qty)
            total_fees = (buy_price * executed_qty + cummulative_quote_qty) * trade_fee_percent
            profit_or_loss = gross_profit - total_fees
            
            # Update pattern success rate
            pattern = coins[symbol]["entry_pattern"]
            if pattern in coins[symbol]["pattern_success_rate"]:
                if profit_or_loss > 0:
                    coins[symbol]["pattern_success_rate"][pattern]["wins"] += 1
                else:
                    coins[symbol]["pattern_success_rate"][pattern]["losses"] += 1
            
            # Update overall status
            overall_status[symbol] += profit_or_loss
            
            # Reset position
            coins[symbol]["position_is_open"] = False
            coins[symbol]["buy_price"] = None
            coins[symbol]["bought_quantity"] = None
            coins[symbol]["remaining_quantity"] = None
            coins[symbol]["entry_date"] = None
            coins[symbol]["entry_pattern"] = None
            coins[symbol]["highest_price"] = None
            
            # Clear files
            with open(coins[symbol]["filename_order_id"], "w") as f:
                f.write("")
            
            # Update market analysis
            market_analysis["total_positions"] = sum(1 for coin in coins.values() if coin["position_is_open"])
            
            return {
                'sell_price': sell_price,
                'profit_loss': profit_or_loss,
                'executed_qty': executed_qty
            }
            
    except Exception as e:
        print(f"{colored(f' ❌ SELL ERROR for {symbol}: {e}', 'red')}")
        return None

def check_graduated_exits(symbol, current_price):
    """Check and execute graduated exits based on profit levels"""
    if not coins[symbol]["position_is_open"]:
        return False
    
    buy_price = coins[symbol]["buy_price"]
    price_change = (current_price - buy_price) / buy_price
    
    for exit_level in graduated_exits:
        # Check if we've hit this exit trigger and haven't already executed it
        if price_change >= exit_level["trigger"]:
            # Check if we've already done this partial exit
            exit_reason = f"Graduated Exit {exit_level['trigger']*100:.1f}%"
            already_executed = any(
                exit['reason'] == exit_reason 
                for exit in coins[symbol]["partial_exits"]
            )
            
            if not already_executed and coins[symbol]["remaining_quantity"] > 0:
                print(f"{colored(f' 🎯 GRADUATED EXIT TRIGGERED', 'green')} for {symbol}")
                print(f" - Price change: +{price_change*100:.2f}%")
                print(f" - Selling {exit_level['sell_percent']*100:.0f}% of original position")
                
                sell_partial(symbol, exit_level["sell_percent"], exit_reason)
                return True
    
    return False

def check_advanced_exit_conditions(symbol, current_price, indicators):
    """Advanced exit conditions with multiple confirmation signals"""
    if not coins[symbol]["position_is_open"]:
        return False
    
    buy_price = coins[symbol]["buy_price"]
    price_change = (current_price - buy_price) / buy_price
    
    # Update highest price for trailing stop
    if current_price > coins[symbol]["highest_price"]:
        coins[symbol]["highest_price"] = current_price
    
    # Calculate comprehensive exit score
    exit_score, exit_reasons = calculate_exit_score(symbol, current_price, indicators)
    
    # Check graduated exits first (for profits)
    if check_graduated_exits(symbol, current_price):
        return True
    
    # Exit thresholds based on position status
    if price_change > 0:  # In profit
        exit_threshold = 60  # Need strong signals to exit profitable position
    else:  # In loss
        exit_threshold = 50  # Lower threshold for losing positions
    
    # Execute exit if score exceeds threshold
    if exit_score >= exit_threshold:
        # Check if we actually have remaining quantity to sell
        if coins[symbol]["remaining_quantity"] <= 0:
            print(f" - Position already fully exited (0 remaining)")
            # Clean up the position state
            coins[symbol]["position_is_open"] = False
            coins[symbol]["buy_price"] = None
            coins[symbol]["bought_quantity"] = None
            coins[symbol]["remaining_quantity"] = None
            coins[symbol]["entry_date"] = None
            coins[symbol]["entry_pattern"] = None
            coins[symbol]["highest_price"] = None
            with open(coins[symbol]["filename_order_id"], "w") as f:
                f.write("")
            return False
        
        print(f"{colored(f' ⚠️  ADVANCED EXIT TRIGGERED', 'red')} for {symbol}")
        print(f" - Exit Score: {exit_score}/100")
        print(f" - Reasons: {', '.join(exit_reasons)}")
        print(f" - Price change: {price_change*100:.2f}%")
        
        # Determine exit percentage based on score
        if exit_score >= 80:  # Very strong signal
            sell(symbol, f"Strong Exit Signal ({exit_score})")
        elif exit_score >= 65:  # Strong signal
            sell_partial(symbol, 0.75, f"Exit Signal ({exit_score})")
        else:  # Moderate signal
            sell_partial(symbol, 0.5, f"Moderate Exit Signal ({exit_score})")
        
        return True
    
    # Hard stop loss (ATR-based)
    atr = indicators.get('atr')
    if atr:
        stop_loss_price = buy_price - (atr * atr_multiplier)
        if current_price <= stop_loss_price:
            print(f"{colored(f' 🛑 ATR STOP LOSS TRIGGERED', 'red')} for {symbol}")
            print(f" - Current: {current_price:.4f}, Stop: {stop_loss_price:.4f}")
            sell(symbol, "ATR Stop Loss")
            return True
    
    # Fallback fixed stop loss
    if price_change <= -base_stop_loss_percent:
        print(f"{colored(f' 🛑 FIXED STOP LOSS TRIGGERED', 'red')} for {symbol}")
        print(f" - Price change: {price_change*100:.2f}%")
        sell(symbol, "Fixed Stop Loss")
        return True
    
    return False

# Terminal colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
GREY = "\033[90m"
END = "\033[0m"

# Check connection and print USDT balance
try:
    account = client.get_account()
    
    for balance in account['balances']:
        if balance['asset'] == 'USDC':
            print("")
            print("=" * 80)
            mode_text = " [DRY RUN MODE - NO REAL TRADES]" if DRY_RUN else " [LIVE TRADING - REAL MONEY]"
            mode_color = YELLOW if DRY_RUN else RED
            print("🚀 CryptoBandit PRO Trading Bot (Enhanced Entry System)")
            print("=" * 80)
            print(f"{mode_color}{'🔸' if DRY_RUN else '⚠️'} MODE: {mode_text}{END}")
            print("=" * 80)
            print(f"Starting USDC balance:           {GREEN}{balance['free']}{END}")
            print(f"Pattern Detection:               {CYAN}5-Minute Candles{END}")
            print(f"Entry System:                    {CYAN}Multi-Factor Scoring{END}")
            print(f"Key Features:                    {YELLOW}S/R Detection, BTC Correlation{END}")
            print(f"                                 {YELLOW}Multi-Timeframe, Volume Profile{END}")
            print(f"                                 {YELLOW}ADX Trend Filter{END}")
            print(f"Minimum Entry Score:             {GREEN}{min_entry_score}/100{END}")
            print(f"Symbols:                         {CYAN}{len(symbols)} pairs{END}")
            print(f"Trade Amount:                    {YELLOW}{usd_amount} USDT{END}")
            print(f"Max Concurrent Positions:        {YELLOW}{market_analysis['correlation_limit']}{END}")
            print(f"Take Profit:                     {GREEN}+{base_take_profit_percent*100}%{END}")
            print(f"Stop Loss:                       {RED}-{base_stop_loss_percent*100}%{END}")
            print(f"Trading Fees:                    {YELLOW}0.1% per trade{END}")
            print("=" * 80)
            print("")
            break
    else:
        print("❌ USDC balance not found")

except Exception as e:
    print("❌ Binance API connection failed:", e)
    sys.exit(1)

def sell_all_positions():
    """Emergency sell all positions"""
    print(colored("\n🔴 EMERGENCY SELL-ALL TRIGGERED", "red", attrs=['bold']))
    confirm = input("Type 'YES' to confirm selling all open positions: ").strip()
    if confirm != "YES":
        print("❌ Cancelled.")
        return
    
    for symbol in coins:
        if coins[symbol]["position_is_open"]:
            # Check position value first
            if coins[symbol]["remaining_quantity"] and coins[symbol]["remaining_quantity"] > 0:
                try:
                    ticker = client.get_symbol_ticker(symbol=symbol)
                    current_price = float(ticker['price'])
                    position_value = coins[symbol]["remaining_quantity"] * current_price
                    
                    if position_value < 10:  # Below minimum
                        print(f"\n⚠️ {symbol} position too small (${position_value:.2f}) - marking as closed")
                        # Just mark it as closed without selling
                        coins[symbol]["position_is_open"] = False
                        coins[symbol]["buy_price"] = None
                        coins[symbol]["bought_quantity"] = None
                        coins[symbol]["remaining_quantity"] = None
                        coins[symbol]["entry_date"] = None
                        coins[symbol]["entry_pattern"] = None
                        coins[symbol]["highest_price"] = None
                        with open(coins[symbol]["filename_order_id"], "w") as f:
                            f.write("")
                        continue
                except:
                    pass
            
            print(f"\n🔄 Selling {symbol}...")
            result = sell(symbol, "Emergency Sell")
            if result:
                print(f"✅ {symbol} sold successfully")
            else:
                print(f"❌ Failed to sell {symbol}")

def listen_for_manual_sell():
    """Listen for manual emergency sell command"""
    while True:
        try:
            key = input().strip().lower()
            if key == "x":
                sell_all_positions()
        except:
            break

# Start the keyboard listener for emergency commands
threading.Thread(target=listen_for_manual_sell, daemon=True).start()

print("=" * 80)
print(colored("🎯 PROFESSIONAL ENTRY SYSTEM ACTIVE", "green", attrs=["bold"]))
print("-" * 80)
print("Entry Requirements:")
print(f" ✓ Pattern detected + Entry score ≥ {min_entry_score}")
print(f" ✓ Not near resistance level")
print(f" ✓ Bitcoin not in strong downtrend")
print(f" ✓ Multiple timeframes aligned")
print(f" ✓ ADX > {adx_trend_threshold} (trending market)")
print("-" * 80)
print("Exit Strategy:")
print(f" ✓ Graduated Exits: 25% at 0.5%, 1%, 2%, 3% profit")
print(f" ✓ Exit Score System: 50-60/100 threshold (dynamic)")
print(f" ✓ ATR Trailing Stop: {atr_multiplier}x ATR from highest price")
print(f" ✓ Fixed Stop Loss: -{base_stop_loss_percent*100:.0f}% (fallback)")
print(f" ✓ Time Stop: Exit after {time_stop_hours} hours if < 0.5% profit")
print(f" ✓ Partial exits based on signal strength")
print("-" * 80)
print("Commands:")
print(colored(" - Type 'x' + ENTER", "yellow", attrs=["bold"]) + " → Emergency sell all positions")
print("-" * 80)
print("")

# Main trading loop
while not shutdown:
    print(f"\n=== Trading Cycle: {datetime.datetime.now()} ===")
    
    # Update Bitcoin trend
    btc = get_bitcoin_trend()
    if btc["15m"]:
        btc_color = 'green' if btc["15m"] == "bullish" else 'red' if btc["15m"] == "bearish" else 'yellow'
        print(f"Bitcoin Status: {colored(btc['15m'].upper(), btc_color)} " +
              f"(15m: {btc.get('strength', 0):.1f}% | 1h: {btc.get('1h', 'N/A')} | 4h: {btc.get('4h', 'N/A')})")
    
    print(f"Open Positions: {market_analysis['total_positions']}/{market_analysis['correlation_limit']}")
    
    for symbol, data in coins.items():
        try:
            # Get current price and data
            ticker = client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
            indicators = get_comprehensive_indicators(symbol, client)
            df_5m = get_candle_data(symbol, kline_interval, client, candles_lookback_5m)
            
            # Update swing points for S/R detection
            if len(df_5m) > 0:
                coins[symbol]["swing_highs"].extend(df_5m['high'].tolist()[-5:])
                coins[symbol]["swing_lows"].extend(df_5m['low'].tolist()[-5:])
            
            print(f"\n[{symbol}]")
            print(f" - Current Price: {colored(f'{current_price:.6f}', 'cyan')} USDT")
            
            # Check exit conditions if position is open
            if data["position_is_open"]:
                # First check if remaining quantity is valid
                if data["remaining_quantity"] is None or data["remaining_quantity"] <= 0:
                    print(f" - Cleaning up empty position for {symbol}")
                    # Clean up the position
                    data["position_is_open"] = False
                    data["buy_price"] = None
                    data["bought_quantity"] = None
                    data["remaining_quantity"] = None
                    data["entry_date"] = None
                    data["entry_pattern"] = None
                    data["highest_price"] = None
                    data["partial_exits"] = []
                    with open(data["filename_order_id"], "w") as f:
                        f.write("")
                    with open(data["filename_partial_exits"], "w") as f:
                        f.write("[]")
                    continue
                
                buy_price = data["buy_price"]
                price_change = (current_price - buy_price) / buy_price
                
                print(f" - Position: {colored('OPEN', 'yellow')} (Entry: {buy_price:.6f})")
                print(f" - P&L: {colored(f'{price_change*100:+.2f}%', 'green' if price_change > 0 else 'red')}")
                print(f" - Pattern: {data['entry_pattern']}")
                print(f" - Remaining: {data['remaining_quantity']:.8f} {symbol[:-4]}")
                
                # Calculate and display exit score
                exit_score, exit_reasons = calculate_exit_score(symbol, current_price, indicators)
                if exit_score > 30:
                    print(f" - Exit Score: {colored(f'{exit_score}/100', 'yellow' if exit_score < 50 else 'red')}")
                    if exit_reasons:
                        print(f" - Exit Signals: {', '.join(exit_reasons[:3])}")
                
                # Check advanced exit conditions
                if check_advanced_exit_conditions(symbol, current_price, indicators):
                    continue
            
            else:
                # Look for entry signals on 5-minute candles
                buy_signals, sell_signals, _ = analyze_candlestick_patterns(df_5m)
                
                if buy_signals:
                    pattern_text = ', '.join(buy_signals)
                    print(f" 🔍 Pattern (5m): {colored(pattern_text, 'yellow')}")
                    
                    # Calculate comprehensive entry score
                    entry_score, entry_reasons, entry_warnings = calculate_entry_score(
                        symbol, pattern_text, current_price, df_5m, indicators
                    )
                    
                    # Display entry analysis
                    score_color = 'green' if entry_score >= min_entry_score else 'yellow' if entry_score >= 50 else 'red'
                    print(f" - Entry Score: {colored(f'{entry_score}/100', score_color)}")
                    
                    if entry_reasons:
                        print(f" - ✓ {', '.join(entry_reasons[:3])}")
                    if entry_warnings:
                        print(f" - ⚠ {', '.join(entry_warnings[:2])}")
                    
                    # Execute trade if score meets threshold
                    if entry_score >= min_entry_score:
                        print(f"{colored(' ✅ ENTRY APPROVED', 'green', attrs=['bold'])}")
                        success = buy(symbol, usd_amount, pattern_text, entry_score, entry_reasons)
                        if success:
                            print(f"{colored(' 🎯 POSITION OPENED', 'green')} for {symbol}")
                    else:
                        print(f" - {colored('Entry rejected (score too low)', 'red')}")
                else:
                    print(f" - Status: {colored('No patterns', 'grey')}")
            
            # Log to output file
            with open(data["filename_output"], "a") as f:
                status = "OPEN" if data["position_is_open"] else "CLOSED"
                f.write(f"{datetime.datetime.now()}: {symbol} - {current_price:.6f} USDT - {status}\n")
            
        except Exception as e:
            print(f"{colored(f' - Error processing {symbol}: {e}', 'red')}")
    
    # Save overall status
    with open(status_file, "w") as f:
        json.dump(overall_status, f, indent=2)
    
    # Print P&L summary if any
    if any(pnl != 0 for pnl in overall_status.values()):
        print(f"\n{colored('📊 P&L Status', 'cyan')}")
        total_pnl = sum(overall_status.values())
        for symbol, pnl in overall_status.items():
            if pnl != 0:
                print(f" - {symbol}: {colored(f'{pnl:+.2f} USDT', 'green' if pnl > 0 else 'red')}")
        print(f" - TOTAL: {colored(f'{total_pnl:+.2f} USDT', 'green' if total_pnl > 0 else 'red', attrs=['bold'])}")
    
    # Wait for next check
    print(f"\n⏰ Next check in {check_interval_minutes} minutes...")
    for _ in range(check_interval_minutes * 60):
        if shutdown:
            break
        time.sleep(1)

print(f"\n{colored('🔻 Trading bot stopped gracefully', 'yellow')}")