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
sys.path.append('../')
from _secrets import api_key, secret_key

# List of coins to track (all symbols from backtest)
symbols = [
    "BTCUSDC", "ETHUSDC", "BNBUSDC", "ADAUSDC", "XRPUSDC", 
    "DOGEUSDC", "SOLUSDC", "PNUTUSDC", "PEPEUSDC", "SHIBUSDC", 
    "XLMUSDC", "LINKUSDC", "IOTAUSDC"
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
        "filename_order_id": f"orders_candles_1m/order_id_{symbol}.txt",
        "filename_output": f"outputs_candles_1m/output_{symbol}.txt",
        "buy_price": None,
        "sell_price": None,
        "bought_quantity": None,
        "position_is_open": False,
        "entry_date": None,
        "entry_pattern": None,
    }
    for symbol in symbols
}

# Create directories if they don't exist
os.makedirs("orders_candles_1m", exist_ok=True)
os.makedirs("outputs_candles_1m", exist_ok=True)

# Restore state from files
for symbol in symbols:
    try:
        with open(coins[symbol]["filename_order_id"], "r") as f:
            data_content = f.read().strip()
            if data_content:
                parts = data_content.split(",")
                if len(parts) >= 4:  # buy_price,quantity,entry_date,pattern
                    buy_price, quantity, entry_date, pattern = parts[0], parts[1], parts[2], parts[3]
                    coins[symbol]["buy_price"] = float(buy_price)
                    coins[symbol]["bought_quantity"] = float(quantity)
                    coins[symbol]["position_is_open"] = True
                    coins[symbol]["entry_date"] = entry_date
                    coins[symbol]["entry_pattern"] = pattern
    except FileNotFoundError:
        continue

# Load profit/loss tracking
status_file = "status_candles_1m.json"
overall_status = {}
if os.path.exists(status_file):
    with open(status_file, "r") as f:
        overall_status = json.load(f)

# Ensure all tracked symbols are present
for symbol in symbols:
    if symbol not in overall_status:
        overall_status[symbol] = 0

# Trading Constants - Based on successful backtest results
usd_amount = 20  # USDT per trade
kline_interval = Client.KLINE_INTERVAL_1MINUTE  
candles_lookback = 20  # Number of candles to analyze for patterns
take_profit_percent = 0.01  # 5% take profit (same as backtest)
stop_loss_percent = 0.10   # 10% stop loss (same as backtest)
check_interval_minutes = 1  # Check positions every minute for 1m candles
trade_fee_percent = 0.001  # 0.1% fee per trade (buy + sell = 0.2% total)

# Set up the Binance API client
client = Client(api_key, secret_key)

# Candlestick Pattern Detection Functions (same as backtesting)
@tenacity.retry(wait=tenacity.wait_fixed(10), stop=tenacity.stop_after_delay(300))
def get_candle_data(symbol, interval, client, limit=20):
    """Fetch recent candlestick data for pattern analysis"""
    try:
        klines = client.get_historical_klines(
            symbol, interval, f"{limit} minutes ago UTC"  # Adjusted for 1m candles
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

# Bearish Pattern Detection  
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

# Trading Functions
def buy(symbol, usd_amount, pattern):
    """Execute a real buy order"""
    try:
        print(f"{colored(' 🟢 EXECUTING BUY ORDER', 'green', attrs=['bold'])} for {symbol}")
        print(f" - Pattern: {colored(pattern, 'yellow')}")
        print(f" - Amount: {usd_amount} USDT")
        
        order = client.order_market_buy(
            symbol=symbol,
            quoteOrderQty=usd_amount
        )
        
        if 'orderId' in order:
            quantity = sum(float(fill['qty']) for fill in order['fills'])
            buy_price = float(order['fills'][0]['price'])
            
            coins[symbol]["bought_quantity"] = quantity
            coins[symbol]["buy_price"] = buy_price
            coins[symbol]["position_is_open"] = True
            coins[symbol]["entry_date"] = datetime.datetime.now().isoformat()
            coins[symbol]["entry_pattern"] = pattern
            
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

def sell(symbol, reason="Manual"):
    """Execute a real sell order"""
    try:
        if not coins[symbol]["position_is_open"]:
            print(f"No open position for {symbol}")
            return None
        
        quantity = coins[symbol]["bought_quantity"]
        buy_price = coins[symbol]["buy_price"]
        
        print(f"{colored(' 🔴 EXECUTING SELL ORDER', 'red', attrs=['bold'])} for {symbol}")
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
            
            # Calculate P&L including trading fees
            gross_profit = cummulative_quote_qty - (buy_price * executed_qty)
            buy_fee = (buy_price * executed_qty) * trade_fee_percent
            sell_fee = cummulative_quote_qty * trade_fee_percent
            total_fees = buy_fee + sell_fee
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
            coins[symbol]["entry_date"] = None
            coins[symbol]["entry_pattern"] = None
            
            # Clear the order file
            with open(coins[symbol]["filename_order_id"], "w") as f:
                f.write("")
            
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

def check_exit_conditions(symbol, current_price):
    """Check if position should be closed due to stop loss or take profit"""
    if not coins[symbol]["position_is_open"]:
        return False
    
    buy_price = coins[symbol]["buy_price"]
    price_change = (current_price - buy_price) / buy_price
    
    # Check take profit
    if price_change >= take_profit_percent:
        print(f"{colored(f' 🎯 TAKE PROFIT TRIGGERED for {symbol}', 'green')}")
        print(f" - Price change: +{price_change*100:.2f}%")
        sell(symbol, "Take Profit")
        return True
    
    # Check stop loss
    elif price_change <= -stop_loss_percent:
        print(f"{colored(f' 🛑 STOP LOSS TRIGGERED for {symbol}', 'red')}")
        print(f" - Price change: {price_change*100:.2f}%")
        sell(symbol, "Stop Loss")
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
            print("🚀 CryptoBandit Candlestick LIVE Trading Bot")
            print("=" * 70)
            print(f"Starting USDC balance:           {GREEN}{balance['free']}{END}")
            print(f"Timeframe:                       {YELLOW}1 Minute Candles{END}")
            print(f"Strategy:                        {CYAN}Candlestick Patterns{END}")
            print(f"Symbols:                         {CYAN}{len(symbols)} pairs{END}")
            print(f"Trade Amount:                    {YELLOW}{usd_amount} USDT{END}")
            print(f"Take Profit:                     {GREEN}+{take_profit_percent*100}%{END}")
            print(f"Stop Loss:                       {RED}-{stop_loss_percent*100}%{END}")
            print(f"Trading Fees:                    {YELLOW}0.1% per trade (0.2% total){END}")
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
print(colored("🎯 LIVE TRADING ACTIVE", "green", attrs=["bold"]))
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
            # Get current price
            ticker = client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
            
            print(f"\n[{symbol}]")
            print(f" - Current Price: {colored(current_price, 'cyan')} USDT")
            
            # Check exit conditions first if we have a position
            if data["position_is_open"]:
                buy_price = data["buy_price"]
                price_change = (current_price - buy_price) / buy_price
                entry_pattern = data["entry_pattern"]
                
                print(f" - Position: {colored('OPEN', 'yellow')} (Entry: {buy_price:.4f})")
                print(f" - P&L: {colored(f'{price_change*100:+.2f}%', 'green' if price_change > 0 else 'red')}")
                print(f" - Entry Pattern: {colored(entry_pattern, 'yellow')}")
                
                # Check exit conditions
                if check_exit_conditions(symbol, current_price):
                    continue  # Position was closed, move to next symbol
            
            else:
                # Look for entry signals
                df = get_candle_data(symbol, kline_interval, client, candles_lookback)
                buy_signals, sell_signals, _ = analyze_candlestick_patterns(df)
                
                if buy_signals:
                    pattern_text = ', '.join(buy_signals)
                    print(f"{colored(' 🟢 BUY SIGNAL DETECTED', 'green')} - {colored(pattern_text, 'yellow')}")
                    
                    # Execute buy order
                    success = buy(symbol, usd_amount, pattern_text)
                    if success:
                        print(f"{colored(' 🎯 POSITION OPENED', 'green')} for {symbol}")
                else:
                    print(f" - Status: {colored('No signals - Waiting', 'grey')}")
            
            # Log to output file
            with open(data["filename_output"], "a") as f:
                status = "OPEN" if data["position_is_open"] else "CLOSED"
                f.write(f"{datetime.datetime.now()}: {symbol} - {current_price:.4f} USDT - {status}\n")
            
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
    
    # Wait for next check (1 minute intervals for 1m candles)
    print(f"\n⏰ Next check in {check_interval_minutes} minute{'s' if check_interval_minutes != 1 else ''}...")
    for _ in range(check_interval_minutes * 60):
        if shutdown:
            break
        time.sleep(1)

print(f"\n{colored('🔻 Trading bot stopped gracefully', 'yellow')}")