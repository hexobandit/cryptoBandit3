# cryptoBandit3
Binance API Buy Low, Sell High, Multiple Cryptos at the same time (RSI &lt; 30 and EMA trend rising for buy)

## Description:
A Python bot for automated cryptocurrency trading on Binance using the Relative Strength Index (RSI) strategy. The bot tracks multiple coins, executes buy/sell orders based on defined thresholds, and provides real-time status updates. Added EMA trend logic.

### Buy Logic
```if percent_change <= -buy_threshold and (rsi < 30) and (ema1 > ema200):```

### Key features include:

- Tracks multiple coins like BTCUSDT, ETHUSDT, etc.
- RSI-based buy signals (RSI < 30)
- EMA monitoring
- Custom thresholds for buy, sell, and stop-loss
- Real-time performance tracking
- Slack notifications for critical events (TODO)
- Persistent state tracking using local files
- Panic sell option: "Type 'x' + ENTER"

### Technologies:
- Binance API for trading and market data.
- Slack SDK for notifications. (TODO)
- Pandas for RSI calculation and data handling.
- Tenacity for retry mechanisms.

### Usage:
- Adjust tracked symbols and thresholds as needed.
- Requires a Binance account and API keys stored in ```_secrets/__init__py```
   - (``api_key = 'xxxxxxx'``
   - ``secret_key= 'xxxxxxx'``)


```
    python3 -m venv venv
    source ./venv/bin/activate
    pip install -r requirements.txt
    python3 cryptoBandit3.py
```

### Constants:
```
    usd_amount = 50  
    buy_threshold = 0.01  # 1%
    sell_threshold = 0.01 
    stop_loss_threshold = 0.8  # 80% 
    reset_initial_price = 0.01  
    kline_interval = Client.KLINE_INTERVAL_1MINUTE
```

### Loop Time
Loop time is defined at the end of the script, in bellow example it runs every 10 minutes:

```
    for _ in range(60 * 10):
        if shutdown:
            break
        time.sleep(1)
```


### Switch to Hourly Candles

1. Change the interval:

```
    kline_interval = Client.KLINE_INTERVAL_1HOUR
```

2. Adjust lookback window:
In both **calculate_rsi** and ***calculate_emas***, replace ```"15 minutes ago UTC"``` and ```"300 minutes ago UTC"``` with:

``` 
    "400 hours ago UTC" 
```


## Example Run:
<img width="641" alt="image" src="https://github.com/user-attachments/assets/93ddc8f6-015e-405d-adfc-d6910ac8e259">
