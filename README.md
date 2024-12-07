# cryptoBandit3
Binance API Buy Low, Sell High, 10 Cryptos at the same time (RSI &lt; 30 for buy)

## Description:
A Python bot for automated cryptocurrency trading on Binance using the Relative Strength Index (RSI) strategy. The bot tracks multiple coins, executes buy/sell orders based on defined thresholds, and provides real-time status updates. 

### Key features include:

- Tracks multiple coins like BTCUSDT, ETHUSDT, etc.
- RSI-based buy/sell signals.
- Custom thresholds for buy, sell, and stop-loss.
- Real-time performance tracking.
- Slack notifications for critical events (TODO)
- Persistent state tracking using local files.

### Technologies:
- Binance API for trading and market data.
- Slack SDK for notifications.
- Pandas for RSI calculation and data handling.
- Tenacity for retry mechanisms.

### Usage:
- Adjust tracked symbols and thresholds as needed.
- Requires a Binance account and API keys stored securely in _secrets.py.

## Example Run:
<img width="641" alt="image" src="https://github.com/user-attachments/assets/93ddc8f6-015e-405d-adfc-d6910ac8e259">


