import pandas as pd
from binance.client import Client
import time
import sys
import json
from datetime import datetime, timedelta
from termcolor import colored

# Load secrets
import os
sys.path.append('../')  # Go up one level to reach the parent directory

# Debug: print current directory and path
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path[-1]}")

try:
    from _secrets import api_key, secret_key
    print("✅ Successfully imported API credentials")
except ImportError as e:
    print(f"❌ Failed to import credentials: {e}")
    print("Please ensure _secrets/__init__.py exists in the parent directory")
    sys.exit(1)

client = Client(api_key, secret_key)

# Configuration
# Start with fewer symbols for faster testing
symbols = [
    "BTCUSDC", "ETHUSDC", "BNBUSDC", "ADAUSDC", "XRPUSDC", "DOGEUSDC", "SOLUSDC", "PNUTUSDC", "PEPEUSDC", "SHIBUSDC", "XLMUSDC", "LINKUSDC", "IOTAUSDC"
]

# Available timeframes to test (start with fewer for faster testing)
timeframes = {
    "1h": Client.KLINE_INTERVAL_1HOUR,
    "4h": Client.KLINE_INTERVAL_4HOUR,
    "1d": Client.KLINE_INTERVAL_1DAY,
    "1m": Client.KLINE_INTERVAL_1MINUTE,
    "5m": Client.KLINE_INTERVAL_5MINUTE, "15m": Client.KLINE_INTERVAL_15MINUTE,
    "30m": Client.KLINE_INTERVAL_30MINUTE, "12h": Client.KLINE_INTERVAL_12HOUR,
    "3d": Client.KLINE_INTERVAL_3DAY
}

# Trading parameters
trade_amount = 100  # USDT per trade
trade_fee_percent = 0.001  # 0.1% fee
stop_loss_percent = -0.10  # 10% stop loss
take_profit_percent = 0.05  # 5% take profit

def get_historical_data(symbol, interval, days_back=365):
    """Fetch historical data for backtesting"""
    print(f"Fetching {days_back} days of {interval} data for {symbol}...")
    
    # Calculate start time
    start_time = datetime.now() - timedelta(days=days_back)
    start_str = start_time.strftime("%Y-%m-%d")
    
    try:
        klines = client.get_historical_klines(
            symbol, interval, start_str
        )
        
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        
        # Convert to proper data types
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        
        print(f"Loaded {len(df)} candles for {symbol}")
        return df
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

# Candlestick Pattern Detection Functions (same as cryptoBanditCandles.py)
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

def is_bullish_engulfing(prev_row, curr_row):
    """Detect Bullish Engulfing pattern"""
    if not is_bearish_candle(prev_row['open'], prev_row['close']):
        return False
    
    return (is_bullish_candle(curr_row['open'], curr_row['close']) and
            curr_row['open'] < prev_row['close'] and
            curr_row['close'] > prev_row['open'])

def is_bearish_engulfing(prev_row, curr_row):
    """Detect Bearish Engulfing pattern"""
    if not is_bullish_candle(prev_row['open'], prev_row['close']):
        return False
    
    return (is_bearish_candle(curr_row['open'], curr_row['close']) and
            curr_row['open'] > prev_row['close'] and
            curr_row['close'] < prev_row['open'])

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

def detect_patterns(df, idx):
    """Detect all patterns for a given candle index"""
    if idx < 2:
        return [], []
    
    current_candle = df.iloc[idx]
    buy_signals = []
    sell_signals = []
    
    # Single candle patterns
    if is_hammer(current_candle):
        buy_signals.append("Hammer")
    
    if is_shooting_star(current_candle):
        sell_signals.append("Shooting Star")
    
    if is_doji(current_candle):
        buy_signals.append("Doji")
    
    # Two candle patterns
    if idx >= 1:
        prev_candle = df.iloc[idx - 1]
        
        if is_bullish_engulfing(prev_candle, current_candle):
            buy_signals.append("Bullish Engulfing")
        
        if is_bearish_engulfing(prev_candle, current_candle):
            sell_signals.append("Bearish Engulfing")
    
    # Three candle patterns
    if is_morning_star(df, idx):
        buy_signals.append("Morning Star")
    
    if is_evening_star(df, idx):
        sell_signals.append("Evening Star")
    
    return buy_signals, sell_signals

class Position:
    def __init__(self, symbol, entry_price, quantity, entry_date, pattern):
        self.symbol = symbol
        self.entry_price = entry_price
        self.quantity = quantity
        self.entry_date = entry_date
        self.pattern = pattern
        self.exit_price = None
        self.exit_date = None
        self.profit_loss = 0
        self.exit_reason = None

def backtest_strategy(df, symbol, timeframe):
    """Backtest the candlestick pattern strategy"""
    positions = []
    current_position = None
    
    total_trades = 0
    profitable_trades = 0
    losing_trades = 0
    total_profit = 0
    
    for i in range(2, len(df)):
        current_candle = df.iloc[i]
        current_price = current_candle['close']
        current_date = current_candle['timestamp']
        
        # Check for exit conditions if we have a position
        if current_position is not None:
            entry_price = current_position.entry_price
            price_change = (current_price - entry_price) / entry_price
            
            # Check stop loss
            if price_change <= stop_loss_percent:
                current_position.exit_price = current_price
                current_position.exit_date = current_date
                current_position.exit_reason = "Stop Loss"
                
                # Calculate P&L including fees
                gross_profit = (current_price - entry_price) * current_position.quantity
                fees = (entry_price * current_position.quantity * trade_fee_percent) + (current_price * current_position.quantity * trade_fee_percent)
                net_profit = gross_profit - fees
                
                current_position.profit_loss = net_profit
                total_profit += net_profit
                total_trades += 1
                
                if net_profit > 0:
                    profitable_trades += 1
                else:
                    losing_trades += 1
                
                positions.append(current_position)
                current_position = None
                continue
            
            # Check take profit
            elif price_change >= take_profit_percent:
                current_position.exit_price = current_price
                current_position.exit_date = current_date
                current_position.exit_reason = "Take Profit"
                
                # Calculate P&L including fees
                gross_profit = (current_price - entry_price) * current_position.quantity
                fees = (entry_price * current_position.quantity * trade_fee_percent) + (current_price * current_position.quantity * trade_fee_percent)
                net_profit = gross_profit - fees
                
                current_position.profit_loss = net_profit
                total_profit += net_profit
                total_trades += 1
                
                if net_profit > 0:
                    profitable_trades += 1
                else:
                    losing_trades += 1
                
                positions.append(current_position)
                current_position = None
                continue
        
        # Look for entry signals if no position
        if current_position is None:
            buy_signals, sell_signals = detect_patterns(df, i)
            
            # Only enter on buy signals (long-only strategy)
            if buy_signals:
                quantity = trade_amount / current_price
                current_position = Position(
                    symbol=symbol,
                    entry_price=current_price,
                    quantity=quantity,
                    entry_date=current_date,
                    pattern=', '.join(buy_signals)
                )
    
    # Close any remaining position at the end
    if current_position is not None:
        final_price = df.iloc[-1]['close']
        final_date = df.iloc[-1]['timestamp']
        
        current_position.exit_price = final_price
        current_position.exit_date = final_date
        current_position.exit_reason = "End of Data"
        
        gross_profit = (final_price - current_position.entry_price) * current_position.quantity
        fees = (current_position.entry_price * current_position.quantity * trade_fee_percent) + (final_price * current_position.quantity * trade_fee_percent)
        net_profit = gross_profit - fees
        
        current_position.profit_loss = net_profit
        total_profit += net_profit
        total_trades += 1
        
        if net_profit > 0:
            profitable_trades += 1
        else:
            losing_trades += 1
        
        positions.append(current_position)
    
    return {
        'symbol': symbol,
        'timeframe': timeframe,
        'total_trades': total_trades,
        'profitable_trades': profitable_trades,
        'losing_trades': losing_trades,
        'win_rate': (profitable_trades / total_trades * 100) if total_trades > 0 else 0,
        'total_profit': total_profit,
        'avg_profit_per_trade': total_profit / total_trades if total_trades > 0 else 0,
        'positions': positions
    }

def run_comprehensive_backtest():
    """Run backtest across all symbols and timeframes"""
    results = {}
    
    print(colored("🚀 Starting Comprehensive Candlestick Pattern Backtest", "cyan", attrs=["bold"]))
    print("=" * 80)
    
    for timeframe_name, timeframe_interval in timeframes.items():
        print(f"\n{colored(f'Testing Timeframe: {timeframe_name}', 'yellow', attrs=['bold'])}")
        print("-" * 50)
        
        timeframe_results = {}
        
        for symbol in symbols:
            try:
                # Get historical data
                df = get_historical_data(symbol, timeframe_interval, days_back=365)
                
                if df is None or len(df) < 100:
                    print(f"Insufficient data for {symbol}, skipping...")
                    continue
                
                # Run backtest
                result = backtest_strategy(df, symbol, timeframe_name)
                timeframe_results[symbol] = result
                
                # Print individual result
                print(f"{symbol:<12} | Trades: {result['total_trades']:<3} | Win Rate: {result['win_rate']:<5.1f}% | P&L: {result['total_profit']:<8.2f} USDT")
                
                # Small delay to avoid API limits
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error backtesting {symbol}: {e}")
                continue
        
        results[timeframe_name] = timeframe_results
        
        # Print timeframe summary
        if timeframe_results:
            total_profit_tf = sum([r['total_profit'] for r in timeframe_results.values()])
            total_trades_tf = sum([r['total_trades'] for r in timeframe_results.values()])
            profitable_trades_tf = sum([r['profitable_trades'] for r in timeframe_results.values()])
            
            avg_win_rate_tf = (profitable_trades_tf / total_trades_tf * 100) if total_trades_tf > 0 else 0
            
            print(f"\n{colored(f'{timeframe_name} SUMMARY:', 'green', attrs=['bold'])}")
            print(f"Total Trades: {total_trades_tf}")
            print(f"Overall Win Rate: {avg_win_rate_tf:.1f}%")
            print(f"Total P&L: {total_profit_tf:.2f} USDT")
    
    return results

def print_final_summary(results):
    """Print comprehensive summary of all backtests"""
    print(f"\n{colored('📊 FINAL BACKTEST SUMMARY', 'cyan', attrs=['bold'])}")
    print("=" * 80)
    
    # Overall statistics
    grand_total_profit = 0
    grand_total_trades = 0
    grand_profitable_trades = 0
    
    best_timeframe = None
    best_profit = float('-inf')
    
    for timeframe_name, timeframe_results in results.items():
        if not timeframe_results:
            continue
            
        tf_profit = sum([r['total_profit'] for r in timeframe_results.values()])
        tf_trades = sum([r['total_trades'] for r in timeframe_results.values()])
        tf_profitable = sum([r['profitable_trades'] for r in timeframe_results.values()])
        
        grand_total_profit += tf_profit
        grand_total_trades += tf_trades
        grand_profitable_trades += tf_profitable
        
        if tf_profit > best_profit:
            best_profit = tf_profit
            best_timeframe = timeframe_name
        
        win_rate = (tf_profitable / tf_trades * 100) if tf_trades > 0 else 0
        
        print(f"{timeframe_name:<6} | Trades: {tf_trades:<4} | Win Rate: {win_rate:<5.1f}% | P&L: {tf_profit:<10.2f} USDT")
    
    print("-" * 80)
    overall_win_rate = (grand_profitable_trades / grand_total_trades * 100) if grand_total_trades > 0 else 0
    
    print(f"{'TOTAL':<6} | Trades: {grand_total_trades:<4} | Win Rate: {overall_win_rate:<5.1f}% | P&L: {colored(f'{grand_total_profit:.2f}', 'green' if grand_total_profit > 0 else 'red')} USDT")
    print(f"\nBest Performing Timeframe: {colored(best_timeframe, 'yellow')} ({best_profit:.2f} USDT)")
    
    # Save results to JSON
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'grand_total_profit': grand_total_profit,
        'grand_total_trades': grand_total_trades,
        'grand_profitable_trades': grand_profitable_trades,
        'overall_win_rate': overall_win_rate,
        'best_timeframe': best_timeframe,
        'best_profit': best_profit,
        'detailed_results': {}
    }
    
    # Convert Position objects to dictionaries for JSON serialization
    for tf_name, tf_results in results.items():
        results_summary['detailed_results'][tf_name] = {}
        for symbol, result in tf_results.items():
            result_copy = result.copy()
            result_copy['positions'] = [
                {
                    'symbol': p.symbol,
                    'entry_price': p.entry_price,
                    'exit_price': p.exit_price,
                    'quantity': p.quantity,
                    'entry_date': p.entry_date.isoformat() if p.entry_date else None,
                    'exit_date': p.exit_date.isoformat() if p.exit_date else None,
                    'pattern': p.pattern,
                    'profit_loss': p.profit_loss,
                    'exit_reason': p.exit_reason
                }
                for p in result['positions']
            ]
            results_summary['detailed_results'][tf_name][symbol] = result_copy
    
    with open('backtest_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nDetailed results saved to: {colored('backtest_results.json', 'yellow')}")

if __name__ == "__main__":
    results = run_comprehensive_backtest()
    print_final_summary(results)