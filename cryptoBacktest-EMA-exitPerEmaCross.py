import pandas as pd
from binance.client import Client
import time
import sys

# 🔐 Load secrets
sys.path.append('../')
from _secrets import api_key, secret_key

client = Client(api_key, secret_key)

# 🎯 Config
interval = Client.KLINE_INTERVAL_1DAY  # Change to 1MINUTE, 1HOUR, etc.
candles_back = 1000  # Number of candles to fetch for backtesting
trade_amount = 100  # 💵 amount of USDC per trade
trade_fee_percent = 0.1  # 💸 Fee per trade (0.1% typical for Binance without BNB)

symbols = [
    "BTCUSDC", "ETHUSDC", "BNBUSDC", "ADAUSDC", "XRPUSDC",
    "DOGEUSDC", "SOLUSDC", "PNUTUSDC", "PEPEUSDC", "SHIBUSDC",
    "XLMUSDC", "LINKUSDC", "SHIBUSDC", "IOTAUSDC"
]

def get_klines(symbol, interval, candles=candles_back):
    all_klines = []
    limit = 1000
    end_time = int(time.time() * 1000)  # current time in ms

    while len(all_klines) < candles:
        new_klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            endTime=end_time,
            limit=limit
        )

        if not new_klines:
            break

        all_klines = new_klines + all_klines  # prepend to get oldest first
        end_time = new_klines[0][0]  # move back in time

        if len(all_klines) >= candles:
            break

        time.sleep(0.25)  # avoid hitting API too 

    df = pd.DataFrame(all_klines[-candles:], columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])

    df["close"] = df["close"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

def backtest(df):
    df["ema1"] = df["close"].ewm(span=1, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

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

        if not in_position and ema1 > ema200:
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

        elif in_position and ema1 < ema200:
            sell_price = price
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

    return trades, total_profit

def run_all_backtests():
    results = []
    for symbol in symbols:
        print(f"\n🔍 Backtesting {symbol} with ${trade_amount} per trade...")
        try:
            df = get_klines(symbol, interval)
            #print(f"🗓️  {symbol} - Showing data from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
            print(df[['timestamp', 'close']].iloc[[0, -1]])
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