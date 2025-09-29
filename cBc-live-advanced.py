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

# List of coins to track (all symbols from backtest)
symbols = [
    "BTCUSDC", "ETHUSDC", "BNBUSDC", "ADAUSDC", "XRPUSDC", 
    "DOGEUSDC", "SOLUSDC", "PNUTUSDC", "PEPEUSDC", "SHIBUSDC", 
    "XLMUSDC", "LINKUSDC", "IOTAUSDC", "ENAUSDC"
]

shutdown = False

def handle_exit(sig, frame):
    global shutdown
    print("\n🔻 Graceful shutdown requested. Exiting...")
    shutdown = True

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

coins = {
    symbol: {
        "filename_order_id": f"orders_advanced/order_id_{symbol}.txt",
        "filename_output": f"outputs_advanced/output_{symbol}.txt",
        "filename_stop_loss_wait": f"orders_advanced/stop_loss_wait_{symbol}.txt",
        "filename_partial_exits": f"orders_advanced/partial_exits_{symbol}.txt",
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
        "swing_highs": deque(maxlen=10),
        "swing_lows": deque(maxlen=10),
        "last_rsi": None,
        "last_macd_histogram": deque(maxlen=5),
        "entry_volume": None,
        "volume_ma": None,
    }
    for symbol in symbols
}

# Create directories if they don't exist
os.makedirs("orders_advanced", exist_ok=True)
os.makedirs("outputs_advanced", exist_ok=True)

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

# Load profit/loss tracking
status_file = "status_advanced.json"
overall_status = {}
if os.path.exists(status_file):
    with open(status_file, "r") as f:
        overall_status = json.load(f)

# Ensure all tracked symbols are present
for symbol in symbols:
    if symbol not in overall_status:
        overall_status[symbol] = 0


# Trading Constants - Enhanced with dynamic parameters
usd_amount = 50  # USDT per trade
kline_interval = Client.KLINE_INTERVAL_1MINUTE  # 1-minute for pattern detection
kline_interval_4h = Client.KLINE_INTERVAL_4HOUR  # 4-hour for EMA trend
kline_interval_1h = Client.KLINE_INTERVAL_1HOUR  # 1-hour for RSI/MACD
candles_lookback_1m = 100  # Lookback for 1-minute pattern analysis
candles_lookback_4h = 100  # Lookback for 4-hour EMA calculation
candles_lookback_1h = 50   # Lookback for 1-hour indicators
base_take_profit_percent = 0.015  # 1.5% base take profit (dynamic adjustment)
base_stop_loss_percent = 0.10   # 10% base stop loss (will be replaced by ATR)
check_interval_minutes = 1  # Check positions every minute
trade_fee_percent = 0.001  # 0.1% fee per trade
stop_loss_wait_hours = 12  # Wait 12 hours after stop loss
time_stop_hours = 72  # Exit after 72 hours if no profit

# Graduated exit percentages
graduated_exits = [
    {"trigger": 0.005, "sell_percent": 0.25},  # Sell 25% at 0.5% profit
    {"trigger": 0.01, "sell_percent": 0.25},   # Sell 25% at 1% profit
    {"trigger": 0.02, "sell_percent": 0.25},   # Sell 25% at 2% profit
    {"trigger": 0.03, "sell_percent": 0.25},   # Sell remaining 25% at 3% profit
]

# EMA configuration
ema_short_period = 1   # EMA1 for 4h trend detection
ema_medium_period = 21  # EMA21 for medium trend
ema_long_period = 99   # EMA99 for major trend

# Technical indicator thresholds
rsi_overbought = 70
rsi_oversold = 30
rsi_divergence_lookback = 10  # Candles to look back for divergence
volume_spike_multiplier = 2.0  # Volume 2x average = spike
atr_multiplier = 2.5  # ATR multiplier for dynamic stop loss

# Set up the Binance API client
client = Client(api_key, secret_key)

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

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    if len(prices) < period:
        return None, None, None
    
    prices = np.array(prices)
    sma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    return upper_band, sma, lower_band

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
            
            # Volume analysis
            indicators['current_volume'] = volumes_1h[-1]
            indicators['avg_volume'] = np.mean(volumes_1h[:-1])
            indicators['volume_spike'] = volumes_1h[-1] > (indicators['avg_volume'] * volume_spike_multiplier)
            
            # OBV
            obv_values = calculate_obv(closes_1h, volumes_1h)
            indicators['obv'] = obv_values[-1] if obv_values is not None else None
            indicators['obv_trend'] = "up" if obv_values is not None and len(obv_values) > 1 and obv_values[-1] > obv_values[-2] else "down"
            
            # RSI Divergence
            if len(closes_1h) >= rsi_divergence_lookback:
                rsi_values = [calculate_rsi(closes_1h[:i+15]) for i in range(len(closes_1h)-14)]
                rsi_values = [r for r in rsi_values if r is not None]
                indicators['rsi_divergence'] = detect_rsi_divergence(closes_1h, rsi_values, rsi_divergence_lookback)
            
            # Volume Divergence
            indicators['volume_divergence'] = detect_volume_divergence(closes_1h, volumes_1h)
            
            # Market Structure
            indicators['market_structure'] = identify_market_structure(highs_1h, lows_1h)
            
            # Bollinger Bands
            upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(closes_1h)
            indicators['bb_upper'] = upper_bb
            indicators['bb_middle'] = middle_bb
            indicators['bb_lower'] = lower_bb
            indicators['bb_position'] = "above" if closes_1h[-1] > upper_bb else "below" if closes_1h[-1] < lower_bb else "inside"
        
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

def get_4h_ema_trend(symbol):
    """Get EMA trend from 4-hour candle data"""
    try:
        indicators = get_comprehensive_indicators(symbol, client)
        return (
            indicators.get('ema1_4h'),
            indicators.get('ema21_4h'),
            indicators.get('ema99_4h'),
            indicators.get('ema_trend', 'unknown')
        )
    except Exception as e:
        print(f"Error fetching 4h EMA data for {symbol}: {e}")
        return None, None, None, "unknown"

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
    
    # 6. Bollinger Band position
    if indicators.get('bb_position') == 'above' and price_change > 0.01:
        exit_score += 15
        exit_reasons.append("Above Bollinger Band")
    
    # 7. OBV trend
    if indicators.get('obv_trend') == 'down' and price_change < 0:
        exit_score += 20
        exit_reasons.append("OBV declining with price")
    
    # 8. Time-based exit
    if coins[symbol]["entry_date"]:
        entry_time = datetime.datetime.fromisoformat(coins[symbol]["entry_date"])
        hours_held = (datetime.datetime.now() - entry_time).total_seconds() / 3600
        if hours_held > time_stop_hours and price_change < 0.005:
            exit_score += 30
            exit_reasons.append(f"Time stop ({hours_held:.1f}h)")
    
    # 9. ATR-based trailing stop
    atr = indicators.get('atr')
    if atr and coins[symbol]["highest_price"]:
        trailing_stop = coins[symbol]["highest_price"] - (atr * atr_multiplier)
        if current_price < trailing_stop:
            exit_score += 50
            exit_reasons.append(f"ATR trailing stop hit")
    
    return exit_score, exit_reasons

# Candlestick Pattern Detection Functions (same as before)
@tenacity.retry(wait=tenacity.wait_fixed(10), stop=tenacity.stop_after_delay(300))
def get_candle_data(symbol, interval, client, limit=100):
    """Fetch recent candlestick data for pattern analysis"""
    try:
        if interval == Client.KLINE_INTERVAL_1MINUTE:
            klines = client.get_historical_klines(
                symbol, interval, f"{limit} minutes ago UTC"
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

# Bullish Pattern Detection
def is_hammer(row, volume_confirmation=None):
    """Detect Hammer pattern with optional volume confirmation"""
    open_price, high_price, low_price, close_price = row['open'], row['high'], row['low'], row['close']
    body = candle_body_size(open_price, close_price)
    range_candle = candle_range(high_price, low_price)
    lower_shad = lower_shadow(low_price, open_price, close_price)
    upper_shad = upper_shadow(high_price, open_price, close_price)
    
    if range_candle > 0 and body > 0:
        pattern_detected = (lower_shad >= 2 * body and 
                          upper_shad <= body * 0.5 and
                          body <= range_candle * 0.3)
        
        # Volume confirmation makes pattern stronger
        if pattern_detected and volume_confirmation:
            return row['volume'] > volume_confirmation
        return pattern_detected
    return False

def is_bullish_engulfing(prev_row, curr_row, volume_confirmation=None):
    """Detect Bullish Engulfing pattern with volume confirmation"""
    if not is_bearish_candle(prev_row['open'], prev_row['close']):
        return False
    
    pattern_detected = (is_bullish_candle(curr_row['open'], curr_row['close']) and
                       curr_row['open'] < prev_row['close'] and
                       curr_row['close'] > prev_row['open'])
    
    if pattern_detected and volume_confirmation:
        return curr_row['volume'] > volume_confirmation
    return pattern_detected

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

# Bearish Pattern Detection  
def is_shooting_star(row, volume_confirmation=None):
    """Detect Shooting Star pattern with volume confirmation"""
    open_price, high_price, low_price, close_price = row['open'], row['high'], row['low'], row['close']
    body = candle_body_size(open_price, close_price)
    range_candle = candle_range(high_price, low_price)
    lower_shad = lower_shadow(low_price, open_price, close_price)
    upper_shad = upper_shadow(high_price, open_price, close_price)
    
    if range_candle > 0 and body > 0:
        pattern_detected = (upper_shad >= 2 * body and 
                          lower_shad <= body * 0.5 and
                          body <= range_candle * 0.3)
        
        if pattern_detected and volume_confirmation:
            return row['volume'] > volume_confirmation
        return pattern_detected
    return False

def is_bearish_engulfing(prev_row, curr_row, volume_confirmation=None):
    """Detect Bearish Engulfing pattern with volume confirmation"""
    if not is_bullish_candle(prev_row['open'], prev_row['close']):
        return False
    
    pattern_detected = (is_bearish_candle(curr_row['open'], curr_row['close']) and
                       curr_row['open'] > prev_row['close'] and
                       curr_row['close'] < prev_row['open'])
    
    if pattern_detected and volume_confirmation:
        return curr_row['volume'] > volume_confirmation
    return pattern_detected

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

def analyze_candlestick_patterns(df, avg_volume=None):
    """Analyze candlestick patterns with volume weighting"""
    if len(df) < 3:
        return None, None, None, 0
    
    latest_idx = len(df) - 1
    latest_candle = df.iloc[latest_idx]
    
    buy_signals = []
    sell_signals = []
    pattern_strength = 0
    
    # Volume confirmation threshold
    volume_threshold = avg_volume * 1.5 if avg_volume else None
    
    # Single candle patterns with volume weighting
    if is_hammer(latest_candle, volume_threshold):
        buy_signals.append("Hammer")
        pattern_strength += 2 if latest_candle['volume'] > volume_threshold else 1
    
    if is_shooting_star(latest_candle, volume_threshold):
        sell_signals.append("Shooting Star")
        pattern_strength += 2 if latest_candle['volume'] > volume_threshold else 1
    
    if is_doji(latest_candle):
        buy_signals.append("Doji")
        pattern_strength += 1
    
    # Two candle patterns
    if latest_idx >= 1:
        prev_candle = df.iloc[latest_idx - 1]
        
        if is_bullish_engulfing(prev_candle, latest_candle, volume_threshold):
            buy_signals.append("Bullish Engulfing")
            pattern_strength += 3 if latest_candle['volume'] > volume_threshold else 2
        
        if is_bearish_engulfing(prev_candle, latest_candle, volume_threshold):
            sell_signals.append("Bearish Engulfing")
            pattern_strength += 3 if latest_candle['volume'] > volume_threshold else 2
    
    # Three candle patterns (strongest signals)
    if is_morning_star(df, latest_idx):
        buy_signals.append("Morning Star")
        pattern_strength += 4
    
    if is_evening_star(df, latest_idx):
        sell_signals.append("Evening Star")
        pattern_strength += 4
    
    return buy_signals, sell_signals, latest_candle['close'], pattern_strength

# Trading Functions
def buy(symbol, usd_amount, pattern, indicators):
    """Execute a real buy order with enhanced entry logic"""
    try:
        print(f"{colored(' 🟢 EXECUTING BUY ORDER', 'green', attrs=['bold'])} for {symbol}")
        print(f" - Pattern: {colored(pattern, 'yellow')}")
        print(f" - Amount: {usd_amount} USDT")
        
        # Add indicator confirmations to entry log
        if indicators.get('rsi'):
            print(f" - RSI: {indicators['rsi']:.1f}")
        if indicators.get('volume_spike'):
            print(f" - Volume spike detected")
        
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
            coins[symbol]["entry_volume"] = indicators.get('current_volume')
            
            # Save to file
            with open(coins[symbol]["filename_order_id"], "w") as f:
                f.write(f"{buy_price},{quantity},{coins[symbol]['entry_date']},{pattern}")
            
            print(f"{colored(' ✅ BUY SUCCESS', 'green')} | Order ID: {order['orderId']}")
            print(f" - Quantity: {quantity:.8f} {symbol[:-4]}")
            print(f" - Price: {buy_price:.4f} USDT")
            return True
        else:
            print(f"{colored(' ❌ BUY FAILED', 'red')} - No order ID returned")
            return False
            
    except Exception as e:
        print(f"{colored(f' ❌ BUY ERROR for {symbol}: {e}', 'red')}")
        return False

def set_stop_loss_wait(symbol):
    """Set a wait period after stop loss"""
    wait_until = datetime.datetime.now() + datetime.timedelta(hours=stop_loss_wait_hours)
    coins[symbol]["stop_loss_wait_until"] = wait_until
    
    # Save to file for persistence
    with open(coins[symbol]["filename_stop_loss_wait"], "w") as f:
        f.write(wait_until.isoformat())
    
    print(f"  - Stop loss wait set until: {colored(wait_until.strftime('%Y-%m-%d %H:%M:%S'), 'yellow')}")

def is_in_stop_loss_wait(symbol):
    """Check if symbol is still in stop loss wait period"""
    if coins[symbol]["stop_loss_wait_until"] is None:
        return False
    
    current_time = datetime.datetime.now()
    if current_time < coins[symbol]["stop_loss_wait_until"]:
        return True
    else:
        # Wait period has expired, clear it
        coins[symbol]["stop_loss_wait_until"] = None
        try:
            os.remove(coins[symbol]["filename_stop_loss_wait"])
        except FileNotFoundError:
            pass
        return False

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
        
        print(f"{colored(f' 🔶 PARTIAL SELL ({sell_percent*100:.0f}%)', 'yellow', attrs=['bold'])} for {symbol}")
        print(f" - Reason: {colored(reason, 'cyan')}")
        print(f" - Quantity: {quantity_to_sell:.8f}")
        
        # Get symbol info for precision
        exchange_info = client.get_symbol_info(symbol)
        lot_size_filter = next(f for f in exchange_info['filters'] if f['filterType'] == 'LOT_SIZE')
        step_size = float(lot_size_filter['stepSize'])
        precision = int(round(-math.log(step_size, 10), 0))
        
        # Adjust quantity to match Binance precision
        quantity_to_sell = round(quantity_to_sell, precision)
        
        if quantity_to_sell <= 0:
            print(f" - Quantity too small to sell")
            return None
        
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
                coins[symbol]["position_is_open"] = False
                coins[symbol]["buy_price"] = None
                coins[symbol]["bought_quantity"] = None
                coins[symbol]["entry_date"] = None
                coins[symbol]["entry_pattern"] = None
                coins[symbol]["highest_price"] = None
                with open(coins[symbol]["filename_order_id"], "w") as f:
                    f.write("")
            
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
    """Execute a full sell order"""
    try:
        if not coins[symbol]["position_is_open"]:
            print(f"No open position for {symbol}")
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
            coins[symbol]["partial_exits"] = []
            with open(coins[symbol]["filename_order_id"], "w") as f:
                f.write("")
            with open(coins[symbol]["filename_partial_exits"], "w") as f:
                f.write("[]")
            return None
        
        quantity = coins[symbol]["remaining_quantity"]
        buy_price = coins[symbol]["buy_price"]
        
        print(f"{colored(' 🔴 EXECUTING FULL SELL ORDER', 'red', attrs=['bold'])} for {symbol}")
        print(f" - Reason: {colored(reason, 'yellow')}")
        print(f" - Quantity: {quantity:.8f}")
        
        # Get symbol info for precision
        exchange_info = client.get_symbol_info(symbol)
        lot_size_filter = next(f for f in exchange_info['filters'] if f['filterType'] == 'LOT_SIZE')
        step_size = float(lot_size_filter['stepSize'])
        precision = int(round(-math.log(step_size, 10), 0))
        
        # Adjust quantity to match Binance precision
        quantity = round(quantity, precision)
        
        order = client.order_market_sell(
            symbol=symbol,
            quantity=quantity
        )
        
        executed_qty = float(order.get('executedQty', 0))
        cummulative_quote_qty = float(order.get('cummulativeQuoteQty', 0))
        
        if executed_qty > 0:
            sell_price = cummulative_quote_qty / executed_qty
            
            # Calculate total P&L including all partial exits
            total_sold_qty = executed_qty + sum(exit['quantity'] for exit in coins[symbol]["partial_exits"])
            total_revenue = cummulative_quote_qty + sum(exit['quantity'] * exit['sell_price'] for exit in coins[symbol]["partial_exits"])
            total_cost = buy_price * coins[symbol]["bought_quantity"]
            
            gross_profit = total_revenue - total_cost
            total_fees = (total_cost + total_revenue) * trade_fee_percent
            profit_or_loss = gross_profit - total_fees
            
            print(f"{colored(' ✅ SELL SUCCESS', 'cyan')} | Order ID: {order['orderId']}")
            print(f" - Sell Price: {sell_price:.4f} USDT")
            print(f" - Trading Fees: {total_fees:.4f} USDT")
            
            if profit_or_loss > 0:
                print(f" - {colored(f'NET PROFIT: +{profit_or_loss:.2f} USDT 💰', 'green')}")
            else:
                print(f" - {colored(f'NET LOSS: {profit_or_loss:.2f} USDT', 'red')}")
            
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
            coins[symbol]["partial_exits"] = []
            
            # Clear the order files
            with open(coins[symbol]["filename_order_id"], "w") as f:
                f.write("")
            with open(coins[symbol]["filename_partial_exits"], "w") as f:
                f.write("[]")
            
            # Set stop loss wait period if this was a stop loss
            if "Stop" in reason or "ATR" in reason:
                set_stop_loss_wait(symbol)
            
            return {
                'sell_price': sell_price,
                'profit_loss': profit_or_loss,
                'executed_qty': executed_qty
            }
        else:
            print(f"{colored(' ❌ SELL FAILED', 'red')} - Zero quantity executed")
            return None
            
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

def sell_all_positions():
    """Emergency sell all positions"""
    print(colored("\n🔴 EMERGENCY SELL-ALL TRIGGERED", "red", attrs=['bold']))
    confirm = input("Type 'YES' to confirm selling all open positions: ").strip()
    if confirm != "YES":
        print("❌ Cancelled.")
        return
    
    for symbol in coins:
        if coins[symbol]["position_is_open"]:
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
            print("=" * 70)
            print("🚀 CryptoBandit ADVANCED Trading Bot (Multi-Indicator Exit)")
            print("=" * 70)
            print(f"Starting USDC balance:           {GREEN}{balance['free']}{END}")
            print(f"Pattern Detection:               {YELLOW}1 Minute Candles{END}")
            print(f"Exit Strategy:                   {CYAN}Advanced Multi-Indicator{END}")
            print(f"Indicators:                      {CYAN}RSI, MACD, ATR, Volume, EMA{END}")
            print(f"Symbols:                         {CYAN}{len(symbols)} pairs{END}")
            print(f"Trade Amount:                    {YELLOW}{usd_amount} USDT{END}")
            print(f"Graduated Exits:                 {GREEN}25% @ 0.5%, 1%, 2%, 3%{END}")
            print(f"Dynamic Stop Loss:               {RED}ATR-based (2.5x ATR){END}")
            print(f"Time Stop:                       {YELLOW}{time_stop_hours} hours{END}")
            print(f"Exit Score Threshold:            {YELLOW}50-60/100{END}")
            print(f"Trading Fees:                    {YELLOW}0.1% per trade{END}")
            print("=" * 70)
            print("")
            break
    else:
        print("❌ USDC balance not found")

except Exception as e:
    print("❌ Binance API connection failed:", e)
    sys.exit(1)

# Start the keyboard listener for emergency commands
threading.Thread(target=listen_for_manual_sell, daemon=True).start()

print("=" * 70)
print(colored("🎯 ADVANCED MULTI-INDICATOR TRADING ACTIVE", "green", attrs=["bold"]))
print("-" * 70)
print("Commands:")
print(colored(" - Type 'x' + ENTER", "yellow", attrs=["bold"]) + " → Emergency sell all positions")
print("-" * 70)
print("")

# Main trading loop
while not shutdown:
    print(f"\n=== Trading Cycle: {datetime.datetime.now()} ===")
    
    for symbol, data in coins.items():
        try:
            # Get current price and comprehensive indicators
            ticker = client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
            indicators = get_comprehensive_indicators(symbol, client)
            df_1m = get_candle_data(symbol, kline_interval, client, candles_lookback_1m)
            
            print(f"\n[{symbol}]")
            print(f" - Current Price: {colored(current_price, 'cyan')} USDT")
            
            # Display key indicators
            if indicators.get('rsi'):
                rsi_color = 'red' if indicators['rsi'] > rsi_overbought else 'green' if indicators['rsi'] < rsi_oversold else 'yellow'
                print(f" - RSI: {colored(f'{indicators['rsi']:.1f}', rsi_color)}")
            
            if indicators.get('ema_trend'):
                trend_color = 'green' if 'bullish' in indicators['ema_trend'] else 'red' if 'bearish' in indicators['ema_trend'] else 'yellow'
                print(f" - EMA Trend: {colored(indicators['ema_trend'], trend_color)}")
            
            if indicators.get('volume_spike'):
                print(f" - {colored('Volume Spike Detected', 'yellow', attrs=['bold'])}")
            
            # Check exit conditions first if we have a position
            if data["position_is_open"]:
                buy_price = data["buy_price"]
                price_change = (current_price - buy_price) / buy_price
                entry_pattern = data["entry_pattern"]
                
                # Calculate position age
                entry_time = datetime.datetime.fromisoformat(data["entry_date"])
                position_age = (datetime.datetime.now() - entry_time).total_seconds() / 3600
                
                print(f" - Position: {colored('OPEN', 'yellow')} (Entry: {buy_price:.4f})")
                print(f" - P&L: {colored(f'{price_change*100:+.2f}%', 'green' if price_change > 0 else 'red')}")
                print(f" - Entry Pattern: {colored(entry_pattern, 'yellow')}")
                print(f" - Position Age: {position_age:.1f} hours")
                print(f" - Remaining: {data['remaining_quantity']:.8f} {symbol[:-4]}")
                
                # Calculate and display exit score
                exit_score, exit_reasons = calculate_exit_score(symbol, current_price, indicators)
                if exit_score > 30:
                    print(f" - Exit Score: {colored(f'{exit_score}/100', 'yellow' if exit_score < 50 else 'red')}")
                    if exit_reasons:
                        print(f" - Exit Signals: {', '.join(exit_reasons[:3])}")
                
                # Check advanced exit conditions
                if check_advanced_exit_conditions(symbol, current_price, indicators):
                    continue  # Position was modified, move to next symbol
            
            else:
                # Check if we're in stop loss wait period
                if is_in_stop_loss_wait(symbol):
                    remaining_time = coins[symbol]["stop_loss_wait_until"] - datetime.datetime.now()
                    hours_left = remaining_time.total_seconds() / 3600
                    print(f" - Status: {colored(f'Stop Loss Wait ({hours_left:.1f}h remaining)', 'yellow')}")
                else:
                    # Look for entry signals with enhanced filtering
                    avg_volume = indicators.get('avg_volume')
                    buy_signals, sell_signals, _, pattern_strength = analyze_candlestick_patterns(df_1m, avg_volume)
                    
                    if buy_signals and pattern_strength >= 2:  # Require minimum pattern strength
                        pattern_text = ', '.join(buy_signals)
                        print(f"{colored(' 🟢 PATTERN DETECTED', 'green')} - {colored(pattern_text, 'yellow')} (Strength: {pattern_strength})")
                        
                        # Enhanced entry confirmation
                        entry_score = 0
                        entry_confirmations = []
                        
                        # Check 4H EMA trend
                        if indicators.get('ema_trend') in ['bullish', 'strong_bullish']:
                            entry_score += 30
                            entry_confirmations.append("EMA trend bullish")
                        
                        # Check RSI not overbought
                        if indicators.get('rsi') and indicators['rsi'] < rsi_overbought:
                            entry_score += 20
                            entry_confirmations.append(f"RSI favorable ({indicators['rsi']:.1f})")
                        
                        # Check volume confirmation
                        if indicators.get('volume_spike'):
                            entry_score += 25
                            entry_confirmations.append("Volume spike")
                        
                        # Check market structure
                        if indicators.get('market_structure') != 'downtrend':
                            entry_score += 15
                            entry_confirmations.append("Structure intact")
                        
                        # Check OBV trend
                        if indicators.get('obv_trend') == 'up':
                            entry_score += 10
                            entry_confirmations.append("OBV rising")
                        
                        print(f" - Entry Score: {colored(f'{entry_score}/100', 'green' if entry_score >= 50 else 'yellow')}")
                        
                        if entry_score >= 50:  # Minimum score for entry
                            print(f"{colored(' ✅ ENTRY CONFIRMED', 'green')} - {', '.join(entry_confirmations)}")
                            
                            # Execute buy order
                            success = buy(symbol, usd_amount, pattern_text, indicators)
                            if success:
                                print(f"{colored(' 🎯 POSITION OPENED', 'green')} for {symbol}")
                        else:
                            print(f"{colored(' ❌ ENTRY REJECTED', 'red')} - Insufficient confirmation (Score: {entry_score})")
                    else:
                        print(f" - Status: {colored('No signals - Waiting', 'grey')}")
            
            # Log to output file
            with open(data["filename_output"], "a") as f:
                status = "OPEN" if data["position_is_open"] else "CLOSED"
                indicator_info = f"RSI:{indicators.get('rsi', 0):.1f} | Trend:{indicators.get('ema_trend', 'unknown')}"
                f.write(f"{datetime.datetime.now()}: {symbol} - {current_price:.4f} USDT - {status} | {indicator_info}\n")
            
        except Exception as e:
            print(f"{colored(f' - Error processing {symbol}: {e}', 'red')}")
    
    # Save overall status
    with open(status_file, "w") as f:
        json.dump(overall_status, f, indent=2)
    
    # Print overall status
    print(f"\n{colored('📊 Overall P&L Status', 'cyan', attrs=['bold'])}")
    total_pnl = 0
    for symbol, pnl in overall_status.items():
        if pnl != 0:
            color = 'green' if pnl > 0 else 'red'
            print(f" - {symbol}: {colored(f'{pnl:+.2f} USDT', color)}")
            total_pnl += pnl
    
    if total_pnl != 0:
        total_color = 'green' if total_pnl > 0 else 'red'
        print(f" - {colored('TOTAL:', 'yellow')} {colored(f'{total_pnl:+.2f} USDT', total_color, attrs=['bold'])}")
    
    # Wait for next check
    print(f"\n⏰ Next check in {check_interval_minutes} minute{'s' if check_interval_minutes != 1 else ''}...")
    for _ in range(check_interval_minutes * 60):
        if shutdown:
            break
        time.sleep(1)

print(f"\n{colored('🔻 Trading bot stopped gracefully', 'yellow')}")