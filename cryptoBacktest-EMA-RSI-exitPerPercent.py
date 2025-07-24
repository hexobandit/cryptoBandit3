import pandas as pd
from binance.client import Client
import time
import sys

# 🔐 Load secrets
sys.path.append('../')
from _secrets import api_key, secret_key

client = Client(api_key, secret_key)

# 🎯 Config
interval = Client.KLINE_INTERVAL_1MINUTE  # Change to 1MINUTE, 1HOUR, etc.
max_candles = 50000  # Number of candles to fetch for backtesting
trade_amount = 100  # 💵 amount of USDC per trade
trade_fee_percent = 0.1  # 💸 Fee per trade (0.1% typical for Binance without BNB)
take_profit_percent = 0.01  # ✅ Take profit at 5% = 0.05
start_date = "2022-01-01" # 📅 Start date for backtesting (YYYY-MM-DD) up to now

symbols = [
    "BTCUSDC", "ETHUSDC", "BNBUSDC", "ADAUSDC", "XRPUSDC",
    "DOGEUSDC", "SOLUSDC", "PNUTUSDC", "PEPEUSDC", "SHIBUSDC",
    "XLMUSDC", "LINKUSDC", "SHIBUSDC", "IOTAUSDC"
]

# 📊 Available Binance KLINE_INTERVAL options:
# Minute-based:
#   Client.KLINE_INTERVAL_1MINUTE     # 1-minute candles
#   Client.KLINE_INTERVAL_3MINUTE     # 3-minute candles
#   Client.KLINE_INTERVAL_5MINUTE     # 5-minute candles
#   Client.KLINE_INTERVAL_15MINUTE    # 15-minute candles
#   Client.KLINE_INTERVAL_30MINUTE    # 30-minute candles
#
# Hour-based:
#   Client.KLINE_INTERVAL_1HOUR       # 1-hour candles
#   Client.KLINE_INTERVAL_2HOUR       # 2-hour candles
#   Client.KLINE_INTERVAL_4HOUR       # 4-hour candles
#   Client.KLINE_INTERVAL_6HOUR       # 6-hour candles
#   Client.KLINE_INTERVAL_8HOUR       # 8-hour candles
#   Client.KLINE_INTERVAL_12HOUR      # 12-hour candles
#
# Day/Week/Month:
#   Client.KLINE_INTERVAL_1DAY        # Daily candles
#   Client.KLINE_INTERVAL_3DAY        # 3-day candles
#   Client.KLINE_INTERVAL_1WEEK       # Weekly candles
#   Client.KLINE_INTERVAL_1MONTH      # Monthly candles


# 📈 Bitcoin Market Phases (Approximate)
# --------------------------------------
# 2020-10-01  Start of bull run
# 2021-04-14  End of bull run (first peak ~64k)
# 2021-05-01  Start of decline
# 2021-07-20  Temporary bottom (~29k)
# 2021-10-10  Second bull peak phase (~69k)
# 2021-11-10  End of bull run (final peak)
# 2022-01-01  Start of bear market
# 2022-11-21  Cycle bottom (~15.5k)
# 2023-01-01  Stagnation / slow accumulation
# 2024-01-01  Start of bull run (ETF + halving rally)
# 2025-07-01  Ongoing bull run


GREY = "\033[90m"
RESET = "\033[0m"


print("")
print("==================================================================")
print("📊 CRYPTOBANDIT - Crypto Backtest")
print("📊 EMA & RSI Strategy with Exit at Profit Percent\n")


def get_klines(symbol, interval, start_date=start_date, max_candles=max_candles):
    all_klines = []
    limit = 1000
    start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
    end_time = int(time.time() * 1000)

    while True:
        print(f"{GREY}⏳ Requesting 1k candles for {symbol} from {pd.to_datetime(start_ts, unit='ms')}...{RESET}")

        new_klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=start_ts,
            endTime=end_time,
            limit=limit
        )

        if not new_klines:
            print(f"⚠️ No more data returned for {symbol} at {pd.to_datetime(start_ts, unit='ms')}")
            break

        all_klines += new_klines
        if len(all_klines) >= max_candles:
            all_klines = all_klines[:max_candles]  # trim if too long
            break

        # update to next starting point
        last_ts = new_klines[-1][0]
        start_ts = last_ts + 1  # avoid overlap

        time.sleep(0.5)  # avoid hammering API

    # convert to DataFrame
    df = pd.DataFrame(all_klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df["close"] = df["close"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

def calculate_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df["rsi"] = rsi
    return df

def backtest(df):
    df["ema1"] = df["close"].ewm(span=1, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    df = calculate_rsi(df, period=14)

    in_position = False
    trades = []
    quantity = 0
    buy_price = 0
    total_profit = 0
    fee_buy = 0

    for i in range(200, len(df)):
        price = df["close"].iloc[i]
        ema1 = df["ema1"].iloc[i]
        ema200 = df["ema200"].iloc[i]
        rsi = df["rsi"].iloc[i]

        # Check for buy signal / BUY condition HERE 
        if not in_position and ema1 > ema200 and rsi < 30:
            buy_price = price
            quantity = trade_amount / price
            fee_buy = trade_amount * (trade_fee_percent / 100)
            in_position = True
            trades.append((
                "BUY",
                df["timestamp"].iloc[i],
                price,
                trade_amount,
                fee_buy,
                None  # No profit yet
            ))

        elif in_position:
            sell_price = price
            gain_percent = (sell_price - buy_price) / buy_price

            if gain_percent >= take_profit_percent:  # ✅ 5% profit
                proceeds = sell_price * quantity
                fee_sell = proceeds * (trade_fee_percent / 100)
                profit = proceeds - trade_amount - fee_buy - fee_sell
                total_profit += profit
                in_position = False
                trades.append((
                    "SELL",
                    df["timestamp"].iloc[i],
                    sell_price,
                    proceeds,
                    fee_sell,
                    profit
                ))
                #print(f"🚀 5% gain reached → SELL at {sell_price:.2f} ({gain_percent*100:.2f}%)")

    return trades, total_profit

def run_all_backtests():
    results = []
    for symbol in symbols:
        print(f"\n🔍 Backtesting {symbol} with ${trade_amount} per trade...")
        try:
            df = get_klines(symbol, interval, start_date, max_candles)
            #print(f"🗓️  {symbol} - Showing data from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
            print(f"📊 {symbol} - Total candles fetched: {len(df)}")
            print(df[['timestamp', 'close']].iloc[[0, -1]])
            print(f"📈 {symbol} - Starting backtest...")
            trades, profit = backtest(df)
            for t in trades:
                side, ts, price, amount, fee, profit_or_none = t
                if side == "BUY":
                    print(f"🟢 BUY  @ {price:.4f} on {ts} | Spent: ${amount:.2f} + Fee: ${fee:.4f}")
                else:
                    print(f"🔴 SELL @ {price:.4f} on {ts} | Received: ${amount:.2f} - Fee: ${fee:.4f} → PnL: {profit_or_none:+.2f}")
            results.append((symbol, profit, len(trades)//2))
            print(f"✅ {symbol}: Net profit = {profit:.2f} USDC | Trades = {len(trades)//2}")
        except Exception as e:
            print(f"❌ Error with {symbol}: {e}")
    return results

if __name__ == "__main__":
    summary = run_all_backtests()
    print("\n📊 Final Summary")
    for symbol, profit, trade_count in summary:
        color = "\033[92m" if profit >= 0 else "\033[91m"
        print(f"{symbol}: {color}{profit:.2f} USDC ({trade_count} trades)\033[0m")