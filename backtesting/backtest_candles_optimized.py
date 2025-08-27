import pandas as pd
from binance.client import Client
import time
import sys
import json
import os
from datetime import datetime, timedelta
from termcolor import colored

# Load secrets
import os
sys.path.append('../')  # Go up one level to reach the parent directory

try:
    from _secrets import api_key, secret_key
    print("✅ Successfully imported API credentials")
except ImportError as e:
    print(f"❌ Failed to import credentials: {e}")
    print("Please ensure _secrets/__init__.py exists in the parent directory")
    sys.exit(1)

client = Client(api_key, secret_key)

# Configuration
symbols = [
    "BTCUSDC", "ETHUSDC", "BNBUSDC", "ADAUSDC", "XRPUSDC", "DOGEUSDC", 
    "SOLUSDC", "PEPEUSDC", "SHIBUSDC", "XLMUSDC", "LINKUSDC", "IOTAUSDC"
]

# Available timeframes to test (reduced for demo)
timeframes = {
    "1h": Client.KLINE_INTERVAL_1HOUR,
    "4h": Client.KLINE_INTERVAL_4HOUR,
    "1d": Client.KLINE_INTERVAL_1DAY,
    "1m": Client.KLINE_INTERVAL_1MINUTE,
    "5m": Client.KLINE_INTERVAL_5MINUTE, 
    "15m": Client.KLINE_INTERVAL_15MINUTE,
    "30m": Client.KLINE_INTERVAL_30MINUTE, 
    "12h": Client.KLINE_INTERVAL_12HOUR,
    "3d": Client.KLINE_INTERVAL_3DAY
}

# Trading parameters
trade_amount = 100  # USDT per trade
trade_fee_percent = 0.001  # 0.1% fee
stop_loss_percent = -0.02  # 10% stop loss
take_profit_percent = 0.04  # 1% take profit

# Data cache configuration
CACHE_DIR = "data_cache"
METADATA_FILE = os.path.join(CACHE_DIR, "cache_metadata.json")
DAYS_BACK = 365  # Default lookback period

def ensure_cache_directory():
    """Create cache directory if it doesn't exist"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        print(f"✅ Created cache directory: {CACHE_DIR}")

def get_cache_filename(symbol, timeframe):
    """Generate cache filename for symbol and timeframe"""
    return os.path.join(CACHE_DIR, f"{symbol}_{timeframe}_data.json")

def load_cache_metadata():
    """Load cache metadata to track last update times"""
    ensure_cache_directory()
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    return {}

def save_cache_metadata(metadata):
    """Save cache metadata"""
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

def load_cached_data(symbol, timeframe):
    """Load cached data for symbol and timeframe"""
    cache_file = get_cache_filename(symbol, timeframe)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                df = pd.DataFrame(data['klines'])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
                return df, data['last_update']
        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            return None, None
    return None, None

def save_cached_data(symbol, timeframe, df, last_update):
    """Save data to cache"""
    cache_file = get_cache_filename(symbol, timeframe)
    
    # Convert DataFrame to serializable format
    df_copy = df.copy()
    df_copy["timestamp"] = df_copy["timestamp"].astype(int) // 10**6  # Convert to milliseconds
    
    cache_data = {
        'klines': df_copy.to_dict('records'),
        'last_update': last_update,
        'cached_at': datetime.now().isoformat()
    }
    
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f)

def get_historical_data_optimized(symbol, interval, days_back=DAYS_BACK, force_refresh=False):
    """
    Optimized data fetching with incremental updates
    
    Args:
        symbol: Trading pair symbol
        interval: Binance interval constant
        days_back: Number of days to look back
        force_refresh: If True, ignore cache and fetch all data fresh
    """
    interval_name = next(name for name, const in timeframes.items() if const == interval)
    
    print(f"📊 Getting data for {symbol} ({interval_name})...", end=" ")
    
    # Load cache metadata
    metadata = load_cache_metadata()
    cache_key = f"{symbol}_{interval_name}"
    
    # Calculate target start time
    target_start = datetime.now() - timedelta(days=days_back)
    
    if not force_refresh:
        # Try to load cached data
        cached_df, last_update = load_cached_data(symbol, interval_name)
        
        if cached_df is not None and last_update:
            last_update_dt = datetime.fromisoformat(last_update)
            
            # Check if we need to fetch new data
            time_since_update = datetime.now() - last_update_dt
            
            # Only fetch delta if cache is recent (less than 1 day old for efficiency)
            if time_since_update < timedelta(days=1):
                print("Using cached data with delta update...", end=" ")
                
                # Get the latest timestamp from cached data
                latest_cached_time = cached_df['timestamp'].max()
                
                # Fetch only new data since last update
                try:
                    new_klines = client.get_historical_klines(
                        symbol, 
                        interval, 
                        str(int(latest_cached_time.timestamp() * 1000)),  # Start from last cached time
                        limit=1000
                    )
                    
                    if new_klines and len(new_klines) > 1:  # Skip if only current candle
                        # Convert new data to DataFrame
                        new_df = pd.DataFrame(new_klines[1:], columns=[  # Skip first candle (duplicate)
                            "timestamp", "open", "high", "low", "close", "volume",
                            "close_time", "quote_asset_volume", "number_of_trades",
                            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
                        ])
                        
                        # Convert data types
                        new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], unit='ms')
                        new_df["open"] = new_df["open"].astype(float)
                        new_df["high"] = new_df["high"].astype(float)
                        new_df["low"] = new_df["low"].astype(float)
                        new_df["close"] = new_df["close"].astype(float)
                        new_df["volume"] = new_df["volume"].astype(float)
                        
                        # Merge with cached data
                        combined_df = pd.concat([cached_df, new_df], ignore_index=True)
                        combined_df = combined_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                        
                        # Trim to required period
                        combined_df = combined_df[combined_df['timestamp'] >= target_start]
                        
                        # Save updated cache
                        save_cached_data(symbol, interval_name, combined_df, datetime.now().isoformat())
                        
                        # Update metadata
                        metadata[cache_key] = {
                            'last_update': datetime.now().isoformat(),
                            'records_count': len(combined_df),
                            'start_date': combined_df['timestamp'].min().isoformat(),
                            'end_date': combined_df['timestamp'].max().isoformat()
                        }
                        save_cache_metadata(metadata)
                        
                        print(f"✅ Updated with {len(new_df)} new records (Total: {len(combined_df)})")
                        return combined_df
                        
                except Exception as e:
                    print(f"⚠️ Delta update failed: {e}. Using cached data.")
                
                # If delta update failed, use existing cache if it's sufficient
                if len(cached_df) >= 100:  # Minimum data requirement
                    trimmed_df = cached_df[cached_df['timestamp'] >= target_start]
                    if len(trimmed_df) >= 100:
                        print(f"✅ Using cached data ({len(trimmed_df)} records)")
                        return trimmed_df
    
    # Full refresh needed
    print("Fetching fresh data...", end=" ")
    
    try:
        # Calculate start time in milliseconds
        start_ms = int(target_start.timestamp() * 1000)
        
        # Fetch data in chunks to overcome 1000 candle limit
        all_klines = []
        current_start = start_ms
        limit = 1000
        
        while len(all_klines) < 50000:  # Safety limit to prevent infinite loops
            chunk_klines = client.get_historical_klines(
                symbol, 
                interval, 
                start_str=current_start,
                limit=limit
            )
            
            if not chunk_klines or len(chunk_klines) < 2:
                break
            
            # Add chunk to results (avoid duplicates)
            if not all_klines:
                all_klines.extend(chunk_klines)
            else:
                # Skip first candle to avoid duplicate from previous chunk
                all_klines.extend(chunk_klines[1:])
            
            # Update start time for next chunk (use last candle's timestamp)
            current_start = chunk_klines[-1][6] + 1  # close_time + 1ms
            
            # Break if we got less than requested (reached current time)
            if len(chunk_klines) < limit:
                break
            
            # Small delay to avoid API rate limits
            time.sleep(0.1)
        
        print(f"fetched {len(all_klines)} candles...", end=" ")
        
        df = pd.DataFrame(all_klines, columns=[
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
        
        # Save to cache
        save_cached_data(symbol, interval_name, df, datetime.now().isoformat())
        
        # Update metadata
        metadata[cache_key] = {
            'last_update': datetime.now().isoformat(),
            'records_count': len(df),
            'start_date': df['timestamp'].min().isoformat() if len(df) > 0 else None,
            'end_date': df['timestamp'].max().isoformat() if len(df) > 0 else None
        }
        save_cache_metadata(metadata)
        
        print(f"✅ Fetched {len(df)} records")
        return df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def clear_cache():
    """Clear all cached data"""
    if os.path.exists(CACHE_DIR):
        import shutil
        shutil.rmtree(CACHE_DIR)
        print("🗑️ Cache cleared")
        ensure_cache_directory()

def show_cache_info():
    """Display cache information"""
    metadata = load_cache_metadata()
    
    if not metadata:
        print("📂 No cached data found")
        return
    
    print(f"\n📊 Cache Information ({len(metadata)} datasets)")
    print("=" * 80)
    
    total_records = 0
    timeframe_stats = {}
    
    for cache_key, info in metadata.items():
        symbol, timeframe = cache_key.split('_', 1)
        records = info.get('records_count', 0)
        total_records += records
        last_update = datetime.fromisoformat(info['last_update']).strftime('%Y-%m-%d %H:%M')
        
        # Calculate actual time coverage
        if info.get('start_date') and info.get('end_date') and info['start_date'] != 'NaT' and info['end_date'] != 'NaT':
            try:
                start_dt = datetime.fromisoformat(info['start_date'].replace('Z', '+00:00').replace('+00:00', ''))
                end_dt = datetime.fromisoformat(info['end_date'].replace('Z', '+00:00').replace('+00:00', ''))
                days_coverage = (end_dt - start_dt).days
            except (ValueError, TypeError):
                days_coverage = 0
        else:
            days_coverage = 0
        
        print(f"{symbol:<10} {timeframe:<4} | {records:>6} records | {days_coverage:>3} days | Updated: {last_update}")
        
        # Track timeframe stats
        if timeframe not in timeframe_stats:
            timeframe_stats[timeframe] = {'records': 0, 'days': 0, 'symbols': 0}
        timeframe_stats[timeframe]['records'] += records
        timeframe_stats[timeframe]['days'] = max(timeframe_stats[timeframe]['days'], days_coverage)
        timeframe_stats[timeframe]['symbols'] += 1
    
    print("=" * 80)
    print(f"Total cached records: {total_records:,}")
    
    # Show timeframe summary
    print("\n📊 Timeframe Coverage Summary:")
    print("-" * 50)
    for tf in sorted(timeframe_stats.keys()):
        stats = timeframe_stats[tf]
        avg_records = stats['records'] // stats['symbols'] if stats['symbols'] > 0 else 0
        print(f"{tf:<4} | Avg: {avg_records:>5} records | Max Coverage: {stats['days']:>3} days | {stats['symbols']} symbols")
    
    # Calculate cache size
    cache_size_mb = 0
    if os.path.exists(CACHE_DIR):
        for filename in os.listdir(CACHE_DIR):
            if filename.endswith('.json'):
                file_path = os.path.join(CACHE_DIR, filename)
                cache_size_mb += os.path.getsize(file_path) / (1024 * 1024)
    
    print(f"\nCache size: {cache_size_mb:.1f} MB")

# Candlestick Pattern Detection Functions (same as original)
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

def run_comprehensive_backtest(force_refresh=False):
    """Run backtest across all symbols and timeframes"""
    results = {}
    
    print(colored("🚀 Starting Optimized Candlestick Pattern Backtest", "cyan", attrs=["bold"]))
    print("=" * 80)
    
    # Show cache info before starting
    show_cache_info()
    print()
    
    total_datasets = len(symbols) * len(timeframes)
    current_dataset = 0
    
    for timeframe_name, timeframe_interval in timeframes.items():
        print(f"\n{colored(f'Testing Timeframe: {timeframe_name}', 'yellow', attrs=['bold'])}")
        print("-" * 50)
        
        timeframe_results = {}
        
        for symbol in symbols:
            current_dataset += 1
            try:
                print(f"[{current_dataset}/{total_datasets}] ", end="")
                
                # Get optimized historical data
                df = get_historical_data_optimized(symbol, timeframe_interval, 
                                                 days_back=DAYS_BACK, 
                                                 force_refresh=force_refresh)
                
                if df is None or len(df) < 100:
                    print(f"⚠️ Insufficient data for {symbol}, skipping...")
                    continue
                
                # Run backtest
                result = backtest_strategy(df, symbol, timeframe_name)
                timeframe_results[symbol] = result
                
                # Calculate HODL performance for comparison
                first_price = df.iloc[0]['close']
                last_price = df.iloc[-1]['close']
                hodl_return = ((last_price - first_price) / first_price) * 100
                hodl_profit = ((last_price - first_price) / first_price) * trade_amount
                
                # Store HODL data in result for later use
                result['hodl_return_pct'] = hodl_return
                result['hodl_profit'] = hodl_profit
                result['first_price'] = first_price
                result['last_price'] = last_price
                
                # Print individual result with HODL comparison
                strategy_color = 'green' if result['total_profit'] > 0 else 'red'
                hodl_color = 'green' if hodl_profit > 0 else 'red'
                
                print(f"{symbol:<10} | T:{result['total_trades']:<3} | W:{result['win_rate']:<5.1f}% | " + 
                      f"Strategy: {colored(f'{result['total_profit']:+8.2f}', strategy_color)} | " +
                      f"HODL: {colored(f'{hodl_profit:+8.2f}', hodl_color)} ({hodl_return:+5.1f}%)")
                
                # Small delay to avoid API limits (reduced since we're using cache)
                time.sleep(0.05)
                
            except Exception as e:
                print(f"❌ Error backtesting {symbol}: {e}")
                continue
        
        results[timeframe_name] = timeframe_results
        
        # Print timeframe summary
        if timeframe_results:
            total_profit_tf = sum([r['total_profit'] for r in timeframe_results.values()])
            total_hodl_profit_tf = sum([r['hodl_profit'] for r in timeframe_results.values()])
            total_trades_tf = sum([r['total_trades'] for r in timeframe_results.values()])
            profitable_trades_tf = sum([r['profitable_trades'] for r in timeframe_results.values()])
            
            avg_win_rate_tf = (profitable_trades_tf / total_trades_tf * 100) if total_trades_tf > 0 else 0
            
            print(f"\n{colored(f'{timeframe_name} SUMMARY:', 'green', attrs=['bold'])}")
            print(f"Strategy P&L: {colored(f'{total_profit_tf:+.2f}', 'green' if total_profit_tf > 0 else 'red')} USDT | " +
                  f"HODL P&L: {colored(f'{total_hodl_profit_tf:+.2f}', 'green' if total_hodl_profit_tf > 0 else 'red')} USDT")
            print(f"Trades: {total_trades_tf} | Win Rate: {avg_win_rate_tf:.1f}% | " +
                  f"Strategy vs HODL: {colored(f'{total_profit_tf - total_hodl_profit_tf:+.2f}', 'green' if total_profit_tf > total_hodl_profit_tf else 'red')} USDT")
            
            # Show per-coin breakdown
            print(f"\n{colored('Per-Coin Performance:', 'cyan')}")
            print(f"{'Symbol':<12} {'Strategy P&L':<15} {'HODL P&L':<15} {'Difference':<12} {'Winner'}")
            print("-" * 65)
            
            for symbol, result in timeframe_results.items():
                strategy_pnl = result['total_profit']
                hodl_pnl = result['hodl_profit']
                difference = strategy_pnl - hodl_pnl
                winner = "Strategy" if difference > 0 else "HODL" if difference < 0 else "Tie"
                
                strategy_color = 'green' if strategy_pnl > 0 else 'red'
                hodl_color = 'green' if hodl_pnl > 0 else 'red'
                diff_color = 'green' if difference > 0 else 'red' if difference < 0 else 'yellow'
                winner_color = 'green' if winner == "Strategy" else 'red' if winner == "HODL" else 'yellow'
                
                print(f"{symbol:<12} " +
                      f"{colored(f'{strategy_pnl:+8.2f}', strategy_color):<24} " +
                      f"{colored(f'{hodl_pnl:+8.2f}', hodl_color):<24} " +
                      f"{colored(f'{difference:+8.2f}', diff_color):<21} " +
                      f"{colored(winner, winner_color)}")
            
            strategy_wins = sum(1 for r in timeframe_results.values() if r['total_profit'] > r['hodl_profit'])
            total_symbols = len(timeframe_results)
            print("-" * 65)
            print(f"Strategy wins: {strategy_wins}/{total_symbols} symbols ({strategy_wins/total_symbols*100:.1f}%)")
    
    return results

def print_final_summary(results):
    """Print comprehensive summary of all backtests"""
    print(f"\n{colored('📊 FINAL BACKTEST SUMMARY', 'cyan', attrs=['bold'])}")
    print("=" * 100)
    
    # Overall statistics
    grand_total_profit = 0
    grand_total_hodl_profit = 0
    grand_total_trades = 0
    grand_profitable_trades = 0
    
    best_timeframe = None
    best_profit = float('-inf')
    best_vs_hodl_timeframe = None
    best_vs_hodl_diff = float('-inf')
    
    print(f"{'Timeframe':<10} {'Trades':<8} {'Win Rate':<10} {'Strategy P&L':<15} {'HODL P&L':<15} {'Difference':<12} {'Winner'}")
    print("-" * 100)
    
    for timeframe_name, timeframe_results in results.items():
        if not timeframe_results:
            continue
            
        tf_profit = sum([r['total_profit'] for r in timeframe_results.values()])
        tf_hodl_profit = sum([r['hodl_profit'] for r in timeframe_results.values()])
        tf_trades = sum([r['total_trades'] for r in timeframe_results.values()])
        tf_profitable = sum([r['profitable_trades'] for r in timeframe_results.values()])
        
        grand_total_profit += tf_profit
        grand_total_hodl_profit += tf_hodl_profit
        grand_total_trades += tf_trades
        grand_profitable_trades += tf_profitable
        
        if tf_profit > best_profit:
            best_profit = tf_profit
            best_timeframe = timeframe_name
        
        tf_diff = tf_profit - tf_hodl_profit
        if tf_diff > best_vs_hodl_diff:
            best_vs_hodl_diff = tf_diff
            best_vs_hodl_timeframe = timeframe_name
        
        win_rate = (tf_profitable / tf_trades * 100) if tf_trades > 0 else 0
        winner = "Strategy" if tf_diff > 0 else "HODL" if tf_diff < 0 else "Tie"
        
        strategy_color = 'green' if tf_profit > 0 else 'red'
        hodl_color = 'green' if tf_hodl_profit > 0 else 'red' 
        diff_color = 'green' if tf_diff > 0 else 'red' if tf_diff < 0 else 'yellow'
        winner_color = 'green' if winner == "Strategy" else 'red' if winner == "HODL" else 'yellow'
        
        print(f"{timeframe_name:<10} {tf_trades:<8} {win_rate:<9.1f}% " +
              f"{colored(f'{tf_profit:+8.2f}', strategy_color):<24} " +
              f"{colored(f'{tf_hodl_profit:+8.2f}', hodl_color):<24} " +
              f"{colored(f'{tf_diff:+8.2f}', diff_color):<21} " +
              f"{colored(winner, winner_color)}")
    
    print("-" * 100)
    overall_win_rate = (grand_profitable_trades / grand_total_trades * 100) if grand_total_trades > 0 else 0
    grand_diff = grand_total_profit - grand_total_hodl_profit
    overall_winner = "Strategy" if grand_diff > 0 else "HODL" if grand_diff < 0 else "Tie"
    
    strategy_color = 'green' if grand_total_profit > 0 else 'red'
    hodl_color = 'green' if grand_total_hodl_profit > 0 else 'red'
    diff_color = 'green' if grand_diff > 0 else 'red' if grand_diff < 0 else 'yellow'
    winner_color = 'green' if overall_winner == "Strategy" else 'red' if overall_winner == "HODL" else 'yellow'
    
    print(f"{'TOTAL':<10} {grand_total_trades:<8} {overall_win_rate:<9.1f}% " +
          f"{colored(f'{grand_total_profit:+8.2f}', strategy_color):<24} " +
          f"{colored(f'{grand_total_hodl_profit:+8.2f}', hodl_color):<24} " +
          f"{colored(f'{grand_diff:+8.2f}', diff_color):<21} " +
          f"{colored(overall_winner, winner_color)}")
    
    print(f"\n{colored('🏆 Performance Summary:', 'yellow', attrs=['bold'])}")
    print(f"• Best Strategy Timeframe: {colored(best_timeframe, 'green')} ({best_profit:+.2f} USDT)")
    print(f"• Best vs HODL Timeframe: {colored(best_vs_hodl_timeframe, 'green')} ({best_vs_hodl_diff:+.2f} USDT advantage)")
    
    # Count timeframes where strategy beats HODL
    strategy_winning_timeframes = sum(1 for tf_name, tf_results in results.items() 
                                    if tf_results and sum(r['total_profit'] for r in tf_results.values()) > 
                                    sum(r['hodl_profit'] for r in tf_results.values()))
    total_timeframes = len([tf for tf in results.values() if tf])
    
    print(f"• Strategy beats HODL in: {strategy_winning_timeframes}/{total_timeframes} timeframes " +
          f"({strategy_winning_timeframes/total_timeframes*100:.1f}%)")
    
    hodl_return_pct = (grand_total_hodl_profit / (len(symbols) * trade_amount)) * 100
    strategy_return_pct = (grand_total_profit / (len(symbols) * trade_amount * len(timeframes))) * 100
    
    print(f"• Average HODL Return: {hodl_return_pct:+.2f}% per symbol")
    print(f"• Average Strategy Return: {strategy_return_pct:+.2f}% per symbol per timeframe")
    
    # Print comprehensive strategy summary
    print(f"\n{colored('📋 STRATEGY SUMMARY', 'cyan', attrs=['bold'])}")
    print("=" * 80)
    print(f"Strategy Type:           {colored('Long-Only Candlestick Pattern Trading', 'yellow')}")
    print(f"Trade Amount:            {colored(f'{trade_amount} USDT', 'white')} per position")
    print(f"Take Profit:             {colored(f'{take_profit_percent*100:+.1f}%', 'green')}")
    print(f"Stop Loss:               {colored(f'{stop_loss_percent*100:+.1f}%', 'red')}")
    print(f"Trading Fees:            {colored(f'{trade_fee_percent*100:.1f}% per trade', 'yellow')} ({trade_fee_percent*2*100:.1f}% round trip)")
    print(f"Backtest Period:         {colored(f'{DAYS_BACK} days', 'white')} ({datetime.now() - timedelta(days=DAYS_BACK):%Y-%m-%d} to {datetime.now():%Y-%m-%d})")
    print(f"Symbols Tested:          {colored(f'{len(symbols)} pairs', 'white')} - {', '.join(symbols)}")
    print(f"Timeframes Tested:       {colored(f'{len(timeframes)} timeframes', 'white')} - {', '.join(timeframes.keys())}")
    
    print(f"\n{colored('📊 Pattern Detection Logic:', 'cyan')}")
    print("• Bullish Patterns (Buy Signals):")
    print("  - Hammer: Small body + long lower shadow (bullish reversal)")
    print("  - Bullish Engulfing: Bullish candle engulfs previous bearish candle")  
    print("  - Doji: Very small body indicating indecision/potential reversal")
    print("  - Morning Star: 3-candle bullish reversal pattern")
    print("• Bearish Patterns (Sell Signals - for reference only in long-only strategy):")
    print("  - Shooting Star: Small body + long upper shadow (bearish reversal)")
    print("  - Bearish Engulfing: Bearish candle engulfs previous bullish candle")
    print("  - Evening Star: 3-candle bearish reversal pattern")
    
    print(f"\n{colored('⚙️ Trading Rules:', 'cyan')}")
    print("• Entry: Buy when bullish candlestick pattern is detected")
    print("• Exit Conditions:")
    print(f"  1. Take Profit: +{take_profit_percent*100:.1f}% gain")
    print(f"  2. Stop Loss: -{abs(stop_loss_percent)*100:.1f}% loss") 
    print("  3. End of backtest period (forced exit)")
    print("• Position Management: Only one position per symbol at a time")
    print("• Strategy Type: Long-only (no short selling)")
    
    print(f"\n{colored('🔧 Technical Implementation:', 'cyan')}")
    print("• Data Source: Binance API with intelligent caching")
    print("• Pattern Recognition: Custom algorithms for 7 candlestick patterns")
    print("• Fee Calculation: Included in all P&L calculations")
    print("• HODL Comparison: Buy-and-hold from first to last price in dataset")
    print("• Performance Metrics: Win rate, total P&L, average per trade, vs HODL comparison")
    
    # Save results to JSON
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'grand_total_profit': grand_total_profit,
        'grand_total_hodl_profit': grand_total_hodl_profit,
        'grand_total_trades': grand_total_trades,
        'grand_profitable_trades': grand_profitable_trades,
        'overall_win_rate': overall_win_rate,
        'best_timeframe': best_timeframe,
        'best_profit': best_profit,
        'best_vs_hodl_timeframe': best_vs_hodl_timeframe,
        'best_vs_hodl_diff': best_vs_hodl_diff,
        'strategy_vs_hodl_winner': overall_winner,
        'strategy_summary': {
            'strategy_type': 'Long-Only Candlestick Pattern Trading',
            'trade_amount': trade_amount,
            'take_profit_percent': take_profit_percent,
            'stop_loss_percent': stop_loss_percent,
            'trading_fee_percent': trade_fee_percent,
            'backtest_days': DAYS_BACK,
            'backtest_start_date': (datetime.now() - timedelta(days=DAYS_BACK)).strftime('%Y-%m-%d'),
            'backtest_end_date': datetime.now().strftime('%Y-%m-%d'),
            'symbols_tested': symbols,
            'timeframes_tested': list(timeframes.keys()),
            'patterns_detected': [
                'Hammer', 'Bullish Engulfing', 'Doji', 'Morning Star',
                'Shooting Star', 'Bearish Engulfing', 'Evening Star'
            ],
            'trading_rules': {
                'entry': 'Buy on bullish candlestick patterns',
                'exit_conditions': [
                    f'Take Profit: +{take_profit_percent*100:.1f}%',
                    f'Stop Loss: -{abs(stop_loss_percent)*100:.1f}%',
                    'End of backtest period'
                ],
                'position_management': 'One position per symbol maximum',
                'strategy_direction': 'Long-only (no short selling)'
            }
        },
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
            # Include HODL comparison data in JSON
            result_copy['hodl_return_pct'] = result.get('hodl_return_pct', 0)
            result_copy['hodl_profit'] = result.get('hodl_profit', 0)
            result_copy['first_price'] = result.get('first_price', 0)
            result_copy['last_price'] = result.get('last_price', 0)
            result_copy['strategy_vs_hodl_diff'] = result['total_profit'] - result.get('hodl_profit', 0)
            result_copy['strategy_beats_hodl'] = bool(result['total_profit'] > result.get('hodl_profit', 0))
            
            results_summary['detailed_results'][tf_name][symbol] = result_copy
    
    with open('backtest_results_optimized.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nDetailed results saved to: {colored('backtest_results_optimized.json', 'yellow')}")

def main():
    """Main function with command line options"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized Candlestick Pattern Backtesting')
    parser.add_argument('--refresh', action='store_true', help='Force refresh all cached data')
    parser.add_argument('--clear-cache', action='store_true', help='Clear all cached data')
    parser.add_argument('--cache-info', action='store_true', help='Show cache information only')
    parser.add_argument('--days', type=int, default=365, help='Number of days to look back (default: 365)')
    
    args = parser.parse_args()
    
    global DAYS_BACK
    DAYS_BACK = args.days
    
    if args.clear_cache:
        clear_cache()
        return
    
    if args.cache_info:
        show_cache_info()
        return
    
    # Ensure cache directory exists
    ensure_cache_directory()
    
    # Run backtest
    results = run_comprehensive_backtest(force_refresh=args.refresh)
    print_final_summary(results)
    
    # Show final cache info
    print("\n" + "="*80)
    show_cache_info()

if __name__ == "__main__":
    main()