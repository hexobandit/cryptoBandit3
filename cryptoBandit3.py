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
        "filename_order_id": f"order_id_{symbol}.txt",
        "filename_output": f"output_{symbol}.txt",
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
status_file = "status.json"
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
buy_threshold = 0.01           # 0.01 = 1% + RSI < 30
sell_threshold = 0.01           # 0.01 = 1%
stop_loss_threshold = 0.8       # 0.80 = 80%
reset_initial_price = 0.005     # 0.005 = 0.5%
kline_interval = Client.KLINE_INTERVAL_1MINUTE

#   1       = 100%
#   0.1     = 10%
#   0.01    = 1%
#   0.001   = 0.1%
#   0.0001  = 0.01%

# Set up the Binance API client
client = Client(api_key, secret_key)

# RSI Calculation Function
@tenacity.retry(wait=tenacity.wait_fixed(10), stop=tenacity.stop_after_delay(300))
def calculate_rsi(symbol, interval, client):
    try:
        # Fetch historical klines
        klines = client.get_historical_klines(
            symbol, interval, "50 minutes ago UTC"
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
        df["close"] = df["close"].astype(float)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return int(rsi.iloc[-1]) #return rsi.iloc[-1]
    except Exception as e:
        print(f"Error calculating RSI for {symbol}: {e}")
        raise tenacity.TryAgain

# EMA Cross Calculation Function
def calculate_emas(symbol, interval, client):
    try:
        # Fetch historical klines
        klines = client.get_historical_klines(symbol, interval, "300 minutes ago UTC")
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        df["close"] = df["close"].astype(float)

        # Calculate EMA 9 and EMA 26
        df["ema1"] = df["close"].ewm(span=1, adjust=False).mean()
        df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
        df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
        df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

        # Return only the current values
        return df["ema1"].iloc[-1], df["ema9"].iloc[-1], df["ema26"].iloc[-1], df["ema200"].iloc[-1]

    except Exception as e:
        print(f"Error calculating EMAs for {symbol}: {e}")
        return None, None

def buy(symbol, usd_amount):
    endpoint = 'https://api.binance.com/api/v3/order'
    params = {
        'symbol': symbol,
        'side': 'BUY',
        'type': 'MARKET',
        'quoteOrderQty': usd_amount
    }
    timestamp = int(time.time() * 1000)
    params['timestamp'] = timestamp
    query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
    signature = hmac.new(secret_key.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
    params['signature'] = signature
    headers = {'X-MBX-APIKEY': api_key}
    response = requests.post(endpoint, params=params, headers=headers)
    data = response.json()

    if 'orderId' in data:
        quantity = sum(float(fill['qty']) for fill in data['fills'])  # Total bought quantity
        buy_price = float(data['fills'][0]['price'])  # Price from the first fill
        coins[symbol]["bought_quantity"] = quantity  # Save in memory
        coins[symbol]["buy_price"] = buy_price  # Save buy price

        # Save to text files
        with open(coins[symbol]["filename_order_id"], "w") as f:
            f.write(f"{buy_price},{quantity}")

        print(f" - Bought {symbol}: Order ID {data['orderId']} | Quantity: {quantity} | Price: {buy_price}")
        return True  # Indicate buy was successful
    else:
        print(f"Error buying {symbol}: {data}")
        return False  # Indicate buy failed

def sell(symbol):
    try:
        # Load buy_price and quantity from file
        try:
            with open(coins[symbol]["filename_order_id"], "r") as f:
                data_content = f.read().strip()
                if not data_content:
                    raise ValueError("Order file is empty.")
                parts = data_content.split(",")
                if len(parts) != 2:
                    raise ValueError(f"Order file contents invalid: '{data_content}'")
                buy_price, quantity = map(float, parts)

        except (FileNotFoundError, ValueError) as e:
            print(f"Error reading order file for {symbol}: {e}")
            return None  # Cannot proceed with sell

        # Fetch trading rules to match Binance LOT_SIZE precision
        exchange_info = client.get_symbol_info(symbol)
        lot_size_filter = next(f for f in exchange_info['filters'] if f['filterType'] == 'LOT_SIZE')
        step_size = float(lot_size_filter['stepSize'])
        precision = int(round(-math.log(step_size, 10), 0))

        # Adjust quantity to match Binance precision
        quantity = round(quantity, precision)

        if quantity <= 0:
            raise ValueError("Quantity to sell is zero or negative.")

        # Create the sell order
        order = client.order_market_sell(
            symbol=symbol,
            quantity=quantity
        )

        # Extract relevant details from the sell order
        executed_qty = float(order.get('executedQty', 0))
        cummulative_quote_qty = float(order.get('cummulativeQuoteQty', 0))

        if executed_qty == 0:
            raise ValueError("Sell order executed quantity is zero.")

        sell_price = cummulative_quote_qty / executed_qty  # Average sell price per unit
        profit_or_loss = cummulative_quote_qty - (buy_price * executed_qty)  # Profit or loss calculation

        # Determine if it was profitable
        if profit_or_loss > 0:
            result = f" - Profit: {profit_or_loss:.2f} USDT 💪😎"
            color = "green"
        else:
            result = f" - Loss: {profit_or_loss:.2f} USDT"
            color = "red"

        print(f"{colored(' - Sell successful!', 'cyan')} {symbol} at {sell_price:.2f} USDT")
        print(colored(result, color))

        # Reset state in memory and clear the file
        coins[symbol]["position_is_open"] = False
        coins[symbol]["buy_price"] = None
        coins[symbol]["bought_quantity"] = None
        with open(coins[symbol]["filename_order_id"], "w") as f:
            f.write("")  # Clear the file

        # Return necessary details for profit calculation
        return {
            'cummulativeQuoteQty': cummulative_quote_qty,
            'executedQty': executed_qty,
            'buy_price': buy_price
        }

    except Exception as e:
        print(f"Error selling {symbol}: {e}")
        return None

def sell_all_positions():
    print(colored("\n🔴 Sell-All Triggered", "red"))
    confirm = input("Type 'YES' to confirm selling all open positions: ").strip()
    if confirm != "YES":
        print("❌ Cancelled.")
        return
    for symbol in coins:
        if coins[symbol]["position_is_open"]:
            print(f" - Attempting to sell {symbol}...")
            result = sell(symbol)
            if result:
                profit_or_loss = (
                    float(result['cummulativeQuoteQty']) -
                    (float(result['buy_price']) * float(result['executedQty']))
                )
                overall_status[symbol] += profit_or_loss
                print(f" - Sold {symbol} ✅  ({colored(f'{profit_or_loss:.2f} USDT', 'green' if profit_or_loss >= 0 else 'red')})")
            else:
                print(f" - Failed to sell {symbol} ❌")
    # Save to file
    with open(status_file, "w") as f:
        json.dump(overall_status, f, indent=2)

# Terminal colors
GREEN = "\033[92m"
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
                print("=" * 47)
                print("Hello World... ")
                print("=" * 47)
                print(f"Starting USDC balance:           {GREEN}{balance['free']}{END}")
                print("=" * 47)
                print("")
                break
    else:
        print("Failed to fetch balance:", data)

except Exception as e:
    print("❌ Binance API balance check failed:", e)
    sys.exit(1)


import math  # Added import for math module

def listen_for_manual_sell():
    while True:
        key = input().strip().lower()
        if key == "x":
            sell_all_positions()

# Start the keyboard listener
threading.Thread(target=listen_for_manual_sell, daemon=True).start()

print("=" * 47)
print(colored("MANUAL COMMANDS", "cyan", attrs=["bold"]))
print("-" * 47)
print("To trigger manual sell of ALL open positions:")
print(colored(" - Type 'x' + ENTER", "yellow", attrs=["bold"]))
print(" - Then confirm with" + " " + colored("'YES'", "green", attrs=["bold"]) + " to execute.")
print("-" * 47)
print("")

# Main loop to monitor all coins
while not shutdown:
    print(f"\n=== Monitoring Cycle: {datetime.datetime.now()} ===")
    for symbol, data in coins.items():
        try:
            # Fetch the current price
            params = {"symbol": symbol}
            response = requests.get(
                "https://api.binance.com/api/v3/ticker/price", params=params
            )
            response_data = response.json()
            price = float(response_data["price"])

            # Calculate RSI & EMA
            rsi = calculate_rsi(symbol, kline_interval, client)
            ema1, ema9, ema26, ema200 = calculate_emas(symbol, kline_interval, client)
            trend_ema26 = "above26" if ema9 > ema26 else "below26"
            trend_ema200 = "above200" if ema1 > ema200 else "below200"
            rsi_trend = "rsiBellow30" if rsi < 30 else "rsiAbove30" 

            # Progress Update
            print(f"\n[{symbol}]")
            print(f" - Current Price: {colored(price, 'cyan')} USDT")
            print(f" - RSI: {colored(rsi, 'yellow')} : {colored(rsi_trend, 'green' if rsi_trend == 'rsiAbove30' else 'red' if rsi_trend == 'rsiBellow30' else 'yellow')}")

            #print(f" - EMA(9): {ema9:.4f}, EMA(26): {ema26:.4f}, Trend: {colored(trend_ema26, 'green' if trend_ema26 == 'above' else 'red' if trend_ema26 == 'below' else 'yellow')}")
            
            print(f" - EMA26 Trend: {colored(trend_ema26, 'green' if trend_ema26 == 'above26' else 'red' if trend_ema26 == 'below26' else 'yellow')}")
            print(f" - EMA200 Trend: {colored(trend_ema200, 'green' if trend_ema200 == 'above200' else 'red' if trend_ema200 == 'below200' else 'yellow')}")

            # Buy logic
            if not data["position_is_open"]:
                if data["initial_price"] is None:
                    data["initial_price"] = price
                    print(f" - Initial price set: {colored(price, 'green')} USDT")
                    continue

                percent_change = (price - data["initial_price"]) / data["initial_price"]
                print(f" - Price change: {percent_change * 100:.2f}% vs reset threshold {reset_initial_price * 100:.2f}%")

                if percent_change > reset_initial_price:
                    print(f" - Price change > reset threshold ({reset_initial_price*100:.2f}%), resetting initial price.")
                    data["initial_price"] = price
                    continue

                #if percent_change <= -buy_threshold and (rsi < 30) and (ema9 > ema26) and (ema1 > ema200):  <<< EMA TREND RISING <<<
                #if percent_change <= -buy_threshold and rsi < 30: <<< ORIGINAL

                if percent_change <= -buy_threshold and (rsi < 30) and (ema1 > ema200): 
                    print(f"{colored(' - BUY SIGNAL', 'green')} for {symbol} at {price} USDT (RSI: {rsi})")
                    success = buy(symbol, usd_amount)
                    if success:
                        data["position_is_open"] = True
                        data["initial_price"] = None
                    else:
                        print(f"{colored(f' - Buy failed for {symbol}.', 'red')}")

            # Sell logic
            elif data["position_is_open"]:
                if data["buy_price"] is None:
                    try:
                        with open(data["filename_order_id"], "r") as f:
                            data_content = f.read().strip()
                            if not data_content:
                                raise ValueError("Order file is empty.")
                            buy_price, _ = map(float, data_content.split(","))
                            data["buy_price"] = buy_price
                    except (FileNotFoundError, ValueError) as e:
                        print(f"{colored(' - Error:', 'red')} buy_price not found for {symbol}. Skipping.")
                        continue


                # Use buy_price for tracking P/L
                buy_price = data["buy_price"]
                if buy_price is None:
                    print(f" - Skipping {symbol}: buy_price is None")
                    continue

                percent_change = (price - buy_price) / buy_price
                print(f" - Price change since buy: {percent_change * 100:.2f}%")


                if percent_change >= sell_threshold or percent_change <= -stop_loss_threshold:
                    print(f"{colored(' - SELL SIGNAL', 'red')} for {symbol} at {price} USDT (RSI: {rsi})")
                    sell_order = sell(symbol)
                    if sell_order is None:
                        print(f"{colored(' - Sell failed for {symbol}.', 'red')}")
                        continue  # Skip to the next symbol

                    # Update overall profit or loss
                    profit_or_loss = (
                        float(sell_order['cummulativeQuoteQty']) -
                        (float(sell_order['buy_price']) * float(sell_order['executedQty']))
                    )
                    overall_status[symbol] += profit_or_loss

                    data["position_is_open"] = False
                    data["buy_price"] = None
                    data["bought_quantity"] = None  # Reset after selling

            # Save updated data to files
            with open(data["filename_output"], "a") as f:
                f.write(f"{datetime.datetime.now()}: {symbol} - {price} USDT\n")

        except Exception as e:
            print(f"{colored(f' - Error processing {symbol}: {e}', 'red')}")

    with open(status_file, "w") as f:
        json.dump(overall_status, f, indent=2)

    # Print overall status for all coins
    print("\n=== Overall Status ===")
    for symbol, status in overall_status.items():
        status_color = "green" if status > 0 else "red"
        print(f"{symbol}: {colored(f'{status:.2f} USDT', status_color)}")

    for _ in range(60 * 1):
        if shutdown:
            break
        time.sleep(1)