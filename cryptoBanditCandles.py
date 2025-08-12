import os
import requests
import json
import hmac
import hashlib
import datetime
import time
from termcolor import colored
import signal
import sys
import pandas as pd
from binance.client import Client
import tenacity
import slack_sdk
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import threading
sys.path.append('../')  # Adjust path to include the parent directory
from _secrets import api_key, secret_key

# List of coins to track
symbols = ["BTCUSDC", 
           "ETHUSDC", 
           "BNBUSDC", 
           "ADAUSDC", 
           "XRPUSDC", 
           "DOGEUSDC", 
           "SOLUSDC", 
           "PNUTUSDC", 
           "PEPEUSDC", 
           "SHIBUSDC", 
           "XLMUSDC", 
           "LINKUSDC", 
           "SHIBUSDC",
           "IOTAUSDC"
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
        "filename_order_id": f"orders_candles/order_id_{symbol}.txt",
        "filename_output": f"outputs_candles/output_{symbol}.txt",
        "buy_price": None,
        "sell_price": None,
        "bought_quantity": None,  # Track exact quantity bought
        "position_is_open": False,
        "initial_price": None,  # To track initial price before buying
    }
    for symbol in symbols
}

# Restore state from files
for symbol in symbols:
    try:
        with open(coins[symbol]["filename_order_id"], "r") as f:
            data_content = f.read().strip()
            if data_content:
                buy_price, quantity = map(float, data_content.split(","))
                coins[symbol]["buy_price"] = buy_price
                coins[symbol]["bought_quantity"] = quantity
                coins[symbol]["position_is_open"] = True
    except FileNotFoundError:
        continue

# Load profit/loss tracking
status_file = "status_candles.json"
overall_status = {}
if os.path.exists(status_file):
    with open(status_file, "r") as f:
        overall_status = json.load(f)

# Ensure all tracked symbols are present
for symbol in symbols:
    if symbol not in overall_status:
        overall_status[symbol] = 0

# Constants
usd_amount = 100  
kline_interval = Client.KLINE_INTERVAL_1MINUTE
candles_lookback = 20  # Number of candles to analyze for patterns

# Set up the Binance API client
client = Client(api_key, secret_key)

# Candlestick Pattern Detection Functions
@tenacity.retry(wait=tenacity.wait_fixed(10), stop=tenacity.stop_after_delay(300))
def get_candle_data(symbol, interval, client, limit=20):
    """Fetch recent candlestick data for pattern analysis"""
    try:
        # Fetch historical klines
        klines = client.get_historical_klines(
            symbol, interval, f"{limit} minutes ago UTC"
        )
        # Create a DataFrame
        df = pd.DataFrame(
            klines,
            columns=[
                "timestamp",
                "open",
                "high", 
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
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
    """Check if candle is bullish (close > open)"""
    return close_price > open_price

def is_bearish_candle(open_price, close_price):
    """Check if candle is bearish (close < open)"""
    return close_price < open_price

def candle_body_size(open_price, close_price):
    """Calculate the size of candle body"""
    return abs(close_price - open_price)

def candle_range(high_price, low_price):
    """Calculate the full range of the candle"""
    return high_price - low_price

def upper_shadow(high_price, open_price, close_price):
    """Calculate upper shadow length"""
    return high_price - max(open_price, close_price)

def lower_shadow(low_price, open_price, close_price):
    """Calculate lower shadow length"""
    return min(open_price, close_price) - low_price

# Bullish Pattern Detection
def is_hammer(row):
    """Detect Hammer pattern (bullish reversal)"""
    open_price, high_price, low_price, close_price = row['open'], row['high'], row['low'], row['close']
    body = candle_body_size(open_price, close_price)
    range_candle = candle_range(high_price, low_price)
    lower_shad = lower_shadow(low_price, open_price, close_price)
    upper_shad = upper_shadow(high_price, open_price, close_price)
    
    # Hammer criteria: small body, long lower shadow, short upper shadow
    if range_candle > 0 and body > 0:
        return (lower_shad >= 2 * body and 
                upper_shad <= body * 0.5 and
                body <= range_candle * 0.3)
    return False

def is_bullish_engulfing(prev_row, curr_row):
    """Detect Bullish Engulfing pattern"""
    # Previous candle should be bearish
    if not is_bearish_candle(prev_row['open'], prev_row['close']):
        return False
    
    # Current candle should be bullish and engulf previous candle
    return (is_bullish_candle(curr_row['open'], curr_row['close']) and
            curr_row['open'] < prev_row['close'] and
            curr_row['close'] > prev_row['open'])

def is_doji(row):
    """Detect Doji pattern (indecision)"""
    open_price, close_price = row['open'], row['close']
    high_price, low_price = row['high'], row['low']
    body = candle_body_size(open_price, close_price)
    range_candle = candle_range(high_price, low_price)
    
    # Doji: very small body compared to range
    if range_candle > 0:
        return body <= range_candle * 0.1
    return False

def is_morning_star(df, idx):
    """Detect Morning Star pattern (3-candle bullish reversal)"""
    if idx < 2:
        return False
    
    candle1 = df.iloc[idx-2]  # First candle (bearish)
    candle2 = df.iloc[idx-1]  # Middle candle (small body)
    candle3 = df.iloc[idx]    # Third candle (bullish)
    
    # First candle should be bearish with good body
    if not is_bearish_candle(candle1['open'], candle1['close']):
        return False
    
    # Middle candle should have small body (gap down)
    middle_body = candle_body_size(candle2['open'], candle2['close'])
    first_body = candle_body_size(candle1['open'], candle1['close'])
    if middle_body >= first_body * 0.5:
        return False
    
    # Third candle should be bullish and close well into first candle's body
    return (is_bullish_candle(candle3['open'], candle3['close']) and
            candle3['close'] > (candle1['open'] + candle1['close']) / 2)

# Bearish Pattern Detection
def is_shooting_star(row):
    """Detect Shooting Star pattern (bearish reversal)"""
    open_price, high_price, low_price, close_price = row['open'], row['high'], row['low'], row['close']
    body = candle_body_size(open_price, close_price)
    range_candle = candle_range(high_price, low_price)
    lower_shad = lower_shadow(low_price, open_price, close_price)
    upper_shad = upper_shadow(high_price, open_price, close_price)
    
    # Shooting star criteria: small body, long upper shadow, short lower shadow
    if range_candle > 0 and body > 0:
        return (upper_shad >= 2 * body and 
                lower_shad <= body * 0.5 and
                body <= range_candle * 0.3)
    return False

def is_bearish_engulfing(prev_row, curr_row):
    """Detect Bearish Engulfing pattern"""
    # Previous candle should be bullish
    if not is_bullish_candle(prev_row['open'], prev_row['close']):
        return False
    
    # Current candle should be bearish and engulf previous candle
    return (is_bearish_candle(curr_row['open'], curr_row['close']) and
            curr_row['open'] > prev_row['close'] and
            curr_row['close'] < prev_row['open'])

def is_evening_star(df, idx):
    """Detect Evening Star pattern (3-candle bearish reversal)"""
    if idx < 2:
        return False
    
    candle1 = df.iloc[idx-2]  # First candle (bullish)
    candle2 = df.iloc[idx-1]  # Middle candle (small body)
    candle3 = df.iloc[idx]    # Third candle (bearish)
    
    # First candle should be bullish with good body
    if not is_bullish_candle(candle1['open'], candle1['close']):
        return False
    
    # Middle candle should have small body (gap up)
    middle_body = candle_body_size(candle2['open'], candle2['close'])
    first_body = candle_body_size(candle1['open'], candle1['close'])
    if middle_body >= first_body * 0.5:
        return False
    
    # Third candle should be bearish and close well into first candle's body
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
        buy_signals.append("Doji (Potential Reversal)")
    
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

# Terminal colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
GREY = "\033[90m"
END = "\033[0m"

# Check connection and print USDT balance
try:
    endpoint = "https://api.binance.com/api/v3/account"
    timestamp = int(time.time() * 1000)
    params = {'timestamp': timestamp}
    query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
    signature = hmac.new(secret_key.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    params['signature'] = signature
    headers = {'X-MBX-APIKEY': api_key}

    response = requests.get(endpoint, params=params, headers=headers)
    data = response.json()

    if 'balances' in data:
        for balance in data['balances']:
            if balance['asset'] == 'USDC':
                print("")
                print("=" * 60)
                print("CryptoBandit Candlestick Pattern Bot")
                print("=" * 60)
                print(f"Starting USDC balance:           {GREEN}{balance['free']}{END}")
                print("=" * 60)
                print("")
                break
    else:
        print("Failed to fetch balance:", data)

except Exception as e:
    print("❌ Binance API balance check failed:", e)
    sys.exit(1)

print("=" * 60)
print(colored("CANDLESTICK PATTERN ANALYSIS MODE", "cyan", attrs=["bold"]))
print("-" * 60)
print("This bot will detect candlestick patterns and print signals.")
print(colored("NO ACTUAL TRADING WILL OCCUR - SIGNAL DETECTION ONLY", "yellow", attrs=["bold"]))
print("-" * 60)
print("")

# Main loop to monitor all coins
while not shutdown:
    print(f"\n=== Pattern Analysis Cycle: {datetime.datetime.now()} ===")
    for symbol, data in coins.items():
        try:
            # Fetch candlestick data
            df = get_candle_data(symbol, kline_interval, client, candles_lookback)
            
            # Analyze patterns
            buy_signals, sell_signals, current_price = analyze_candlestick_patterns(df)
            
            # Progress Update
            print(f"\n[{symbol}]")
            print(f" - Current Price: {colored(current_price, 'cyan')} USDT")
            
            # Print buy signals
            if buy_signals:
                for pattern in buy_signals:
                    print(f"{colored(' 🟢 BUY SIGNAL', 'green')} - Pattern: {colored(pattern, 'yellow')}")
            
            # Print sell signals  
            if sell_signals:
                for pattern in sell_signals:
                    print(f"{colored(' 🔴 SELL SIGNAL', 'red')} - Pattern: {colored(pattern, 'yellow')}")
            
            # If no signals, show status
            if not buy_signals and not sell_signals:
                print(f" - Status: {colored('No pattern signals detected', 'grey')}")
            
            # Save updated data to files
            with open(data["filename_output"], "a") as f:
                signals_text = f"BUY: {', '.join(buy_signals) if buy_signals else 'None'} | SELL: {', '.join(sell_signals) if sell_signals else 'None'}"
                f.write(f"{datetime.datetime.now()}: {symbol} - {current_price} USDT - {signals_text}\n")

        except Exception as e:
            print(f"{colored(f' - Error processing {symbol}: {e}', 'red')}")

    # Save status
    with open(status_file, "w") as f:
        json.dump(overall_status, f, indent=2)

    # Wait for next cycle (10 minutes)
    for _ in range(60 * 1):
        if shutdown:
            break
        time.sleep(1)