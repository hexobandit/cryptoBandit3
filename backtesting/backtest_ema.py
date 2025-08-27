#!/usr/bin/env python3
"""
EMA Trading Strategy Backtesting
================================
Backtests the EMA-enhanced candlestick pattern strategy from cBc-live-ema.py
across multiple timeframes with 100 days of historical data.

Strategy:
- Entry: Bullish candlestick patterns (Hammer, Bullish Engulfing, Morning Star, Doji)
- Exit: Enhanced EMA logic - cut losses on EMA7 < EMA99, let winners run while EMA7 > EMA99
- Take Profit: 1% (but only if EMA7 <= EMA99)
- Stop Loss: 10% (standard)
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from binance.client import Client
import time

# Add parent directory to path for secrets
sys.path.append('../')
from _secrets import api_key, secret_key

# Initialize Binance client
client = Client(api_key, secret_key)

# Trading symbols (same as live trading)
SYMBOLS = [
    "BTCUSDC", "ETHUSDC", "BNBUSDC", "ADAUSDC", "XRPUSDC", 
    "DOGEUSDC", "SOLUSDC", "PNUTUSDC", "PEPEUSDC", "SHIBUSDC", 
    "XLMUSDC", "LINKUSDC", "IOTAUSDC"
]

# Timeframes to test
TIMEFRAMES = [
    ('1m', Client.KLINE_INTERVAL_1MINUTE),
    ('5m', Client.KLINE_INTERVAL_5MINUTE),
    ('15m', Client.KLINE_INTERVAL_15MINUTE),
    ('30m', Client.KLINE_INTERVAL_30MINUTE),
    ('1h', Client.KLINE_INTERVAL_1HOUR),
    ('4h', Client.KLINE_INTERVAL_4HOUR),
    ('1d', Client.KLINE_INTERVAL_1DAY)
]

# Strategy parameters
INITIAL_BALANCE = 10000  # $10,000 starting balance
TRADE_AMOUNT = 150       # $150 per trade (same as live)
TAKE_PROFIT_PERCENT = 0.01   # 1%
STOP_LOSS_PERCENT = 0.10     # 10%
TRADING_FEE_PERCENT = 0.001  # 0.1% per trade (0.2% round trip)

# EMA parameters
EMA_SHORT_PERIOD = 7   # EMA7
EMA_LONG_PERIOD = 99   # EMA99

def calculate_ema(prices, period):
    """Calculate Exponential Moving Average"""
    if len(prices) < period:
        return [None] * len(prices)
    
    prices = np.array(prices)
    alpha = 2 / (period + 1)
    ema = np.zeros(len(prices))
    
    # Initialize first EMA value with first price
    ema[0] = prices[0]
    
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    
    return ema

def get_historical_data(symbol, interval, days_back=100):
    """Fetch historical kline data"""
    print(f"  📥 Fetching {days_back} days of {interval} data for {symbol}...")
    
    try:
        # Calculate start time
        start_time = datetime.now() - timedelta(days=days_back)
        start_str = start_time.strftime('%d %b %Y')
        
        klines = client.get_historical_klines(
            symbol, 
            interval, 
            start_str
        )
        
        if not klines:
            print(f"  ❌ No data returned for {symbol}")
            return None
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert to numeric types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        print(f"  ✅ Retrieved {len(df)} candles for {symbol} ({interval})")
        return df.reset_index(drop=True)
        
    except Exception as e:
        print(f"  ❌ Error fetching data for {symbol}: {e}")
        return None

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

def analyze_candlestick_patterns(df, idx):
    """Analyze candlestick patterns at specific index"""
    if idx < 0 or idx >= len(df):
        return []
    
    buy_signals = []
    current_candle = df.iloc[idx]
    
    # Single candle patterns
    if is_hammer(current_candle):
        buy_signals.append("Hammer")
    
    if is_doji(current_candle):
        buy_signals.append("Doji")
    
    # Two candle patterns
    if idx >= 1:
        prev_candle = df.iloc[idx - 1]
        
        if is_bullish_engulfing(prev_candle, current_candle):
            buy_signals.append("Bullish Engulfing")
    
    # Three candle patterns
    if is_morning_star(df, idx):
        buy_signals.append("Morning Star")
    
    return buy_signals

def backtest_symbol_timeframe(symbol, interval_name, interval_code, days_back=100):
    """Backtest a single symbol on a single timeframe"""
    print(f"\n🔄 Backtesting {symbol} on {interval_name} timeframe...")
    
    # Get historical data
    df = get_historical_data(symbol, interval_code, days_back)
    if df is None or len(df) < max(EMA_SHORT_PERIOD, EMA_LONG_PERIOD) + 10:
        print(f"  ❌ Insufficient data for {symbol} ({interval_name})")
        return None
    
    # Calculate EMAs
    closes = df['close'].values
    ema7_values = calculate_ema(closes, EMA_SHORT_PERIOD)
    ema99_values = calculate_ema(closes, EMA_LONG_PERIOD)
    
    df['ema7'] = ema7_values
    df['ema99'] = ema99_values
    
    # Initialize tracking variables
    balance = INITIAL_BALANCE
    position = None  # {'entry_price': float, 'quantity': float, 'entry_idx': int, 'pattern': str}
    trades = []
    
    # Start backtesting from a point where we have sufficient EMA data
    start_idx = EMA_LONG_PERIOD + 10
    
    for i in range(start_idx, len(df)):
        current_candle = df.iloc[i]
        current_price = current_candle['close']
        current_ema7 = current_candle['ema7']
        current_ema99 = current_candle['ema99']
        
        if position is None:
            # Look for entry signals
            buy_signals = analyze_candlestick_patterns(df, i)
            
            if buy_signals and balance >= TRADE_AMOUNT:
                # Execute buy
                entry_price = current_price
                quantity = TRADE_AMOUNT / entry_price
                buy_fee = TRADE_AMOUNT * TRADING_FEE_PERCENT
                balance -= TRADE_AMOUNT + buy_fee
                
                position = {
                    'entry_price': entry_price,
                    'quantity': quantity,
                    'entry_idx': i,
                    'pattern': ', '.join(buy_signals),
                    'entry_date': current_candle['timestamp']
                }
                
                print(f"  🟢 BUY: {symbol} at ${entry_price:.4f} | Pattern: {position['pattern']} | Balance: ${balance:.2f}")
        
        else:
            # Check exit conditions
            entry_price = position['entry_price']
            price_change = (current_price - entry_price) / entry_price
            
            sell_reason = None
            
            # Enhanced exit logic (same as live trading)
            # 1. If position is at a loss AND EMA7 crosses below EMA99 = SELL
            prev_ema7 = df.iloc[i-1]['ema7'] if i > 0 else None
            prev_ema99 = df.iloc[i-1]['ema99'] if i > 0 else None
            
            ema_crossover = False
            if None not in [prev_ema7, prev_ema99, current_ema7, current_ema99]:
                ema_crossover = (prev_ema7 > prev_ema99) and (current_ema7 < current_ema99)
            
            if price_change < 0 and ema_crossover:
                sell_reason = "EMA Crossover Loss"
            
            # 2. If profit is above take profit BUT EMA7 is still above EMA99, keep position open
            elif price_change >= TAKE_PROFIT_PERCENT:
                if None not in [current_ema7, current_ema99] and current_ema7 > current_ema99:
                    # Keep position open - trend still bullish
                    pass
                else:
                    sell_reason = "Take Profit"
            
            # 3. Standard stop loss
            elif price_change <= -STOP_LOSS_PERCENT:
                sell_reason = "Stop Loss"
            
            # Execute sell if conditions met
            if sell_reason:
                sell_value = position['quantity'] * current_price
                sell_fee = sell_value * TRADING_FEE_PERCENT
                balance += sell_value - sell_fee
                
                profit_loss = sell_value - TRADE_AMOUNT - (TRADE_AMOUNT * TRADING_FEE_PERCENT) - sell_fee
                
                trade_record = {
                    'symbol': symbol,
                    'timeframe': interval_name,
                    'entry_date': position['entry_date'],
                    'exit_date': current_candle['timestamp'],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'quantity': position['quantity'],
                    'entry_pattern': position['pattern'],
                    'exit_reason': sell_reason,
                    'price_change_percent': price_change * 100,
                    'profit_loss': profit_loss,
                    'balance_after': balance
                }
                
                trades.append(trade_record)
                
                status = "💰" if profit_loss > 0 else "📉"
                print(f"  🔴 SELL: {symbol} at ${current_price:.4f} | {sell_reason} | P&L: ${profit_loss:.2f} {status} | Balance: ${balance:.2f}")
                
                position = None
    
    # Close any remaining position at the end
    if position is not None:
        current_candle = df.iloc[-1]
        current_price = current_candle['close']
        sell_value = position['quantity'] * current_price
        sell_fee = sell_value * TRADING_FEE_PERCENT
        balance += sell_value - sell_fee
        
        profit_loss = sell_value - TRADE_AMOUNT - (TRADE_AMOUNT * TRADING_FEE_PERCENT) - sell_fee
        
        trade_record = {
            'symbol': symbol,
            'timeframe': interval_name,
            'entry_date': position['entry_date'],
            'exit_date': current_candle['timestamp'],
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'quantity': position['quantity'],
            'entry_pattern': position['pattern'],
            'exit_reason': "End of Data",
            'price_change_percent': ((current_price - position['entry_price']) / position['entry_price']) * 100,
            'profit_loss': profit_loss,
            'balance_after': balance
        }
        
        trades.append(trade_record)
        print(f"  🔴 FINAL SELL: {symbol} at ${current_price:.4f} | End of Data | P&L: ${profit_loss:.2f} | Balance: ${balance:.2f}")
    
    # Calculate statistics
    if trades:
        total_profit_loss = sum(trade['profit_loss'] for trade in trades)
        winning_trades = [t for t in trades if t['profit_loss'] > 0]
        losing_trades = [t for t in trades if t['profit_loss'] <= 0]
        
        win_rate = len(winning_trades) / len(trades) * 100
        avg_win = np.mean([t['profit_loss'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['profit_loss'] for t in losing_trades]) if losing_trades else 0
        
        result = {
            'symbol': symbol,
            'timeframe': interval_name,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_profit_loss': total_profit_loss,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'final_balance': balance,
            'return_percent': ((balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100,
            'trades': trades
        }
        
        print(f"  ✅ {symbol} ({interval_name}): {len(trades)} trades | Win Rate: {win_rate:.1f}% | P&L: ${total_profit_loss:.2f}")
        return result
    
    else:
        print(f"  ⚠️  {symbol} ({interval_name}): No trades executed")
        return None

def main():
    """Main backtesting function"""
    print("=" * 80)
    print("🚀 EMA-Enhanced Candlestick Strategy Backtesting")
    print("=" * 80)
    print(f"Strategy: Bullish patterns + EMA exit logic")
    print(f"Symbols: {len(SYMBOLS)} pairs")
    print(f"Timeframes: {len(TIMEFRAMES)} intervals")
    print(f"Lookback: 100 days")
    print(f"Initial Balance: ${INITIAL_BALANCE:,}")
    print(f"Trade Size: ${TRADE_AMOUNT}")
    print(f"Take Profit: {TAKE_PROFIT_PERCENT*100}% (if EMA7 <= EMA99)")
    print(f"Stop Loss: {STOP_LOSS_PERCENT*100}%")
    print(f"EMA Periods: {EMA_SHORT_PERIOD}, {EMA_LONG_PERIOD}")
    print("=" * 80)
    
    all_results = []
    
    # Backtest each symbol on each timeframe
    for symbol in SYMBOLS:
        for interval_name, interval_code in TIMEFRAMES:
            try:
                result = backtest_symbol_timeframe(symbol, interval_name, interval_code)
                if result:
                    all_results.append(result)
                
                # Small delay to avoid rate limits
                time.sleep(0.1)
                
            except Exception as e:
                print(f"  ❌ Error backtesting {symbol} ({interval_name}): {e}")
                continue
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'backtest_ema_results_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Generate summary
    if all_results:
        print("\n" + "=" * 80)
        print("📊 BACKTESTING SUMMARY")
        print("=" * 80)
        
        # Overall statistics
        total_trades = sum(r['total_trades'] for r in all_results)
        total_profit_loss = sum(r['total_profit_loss'] for r in all_results)
        winning_results = [r for r in all_results if r['total_profit_loss'] > 0]
        
        print(f"Total Results: {len(all_results)} symbol-timeframe combinations")
        print(f"Total Trades: {total_trades:,}")
        print(f"Total P&L: ${total_profit_loss:,.2f}")
        print(f"Profitable Combinations: {len(winning_results)}/{len(all_results)} ({len(winning_results)/len(all_results)*100:.1f}%)")
        
        # Best performers by timeframe
        print(f"\n📈 Best Performing Timeframes:")
        timeframe_summary = {}
        for result in all_results:
            tf = result['timeframe']
            if tf not in timeframe_summary:
                timeframe_summary[tf] = {'profit': 0, 'trades': 0, 'count': 0}
            timeframe_summary[tf]['profit'] += result['total_profit_loss']
            timeframe_summary[tf]['trades'] += result['total_trades']
            timeframe_summary[tf]['count'] += 1
        
        sorted_timeframes = sorted(timeframe_summary.items(), key=lambda x: x[1]['profit'], reverse=True)
        for tf, stats in sorted_timeframes[:5]:
            avg_profit = stats['profit'] / stats['count'] if stats['count'] > 0 else 0
            print(f"  {tf:>3s}: ${stats['profit']:>8.2f} total | ${avg_profit:>6.2f} avg | {stats['trades']:>4d} trades | {stats['count']} symbols")
        
        # Top performing symbols
        print(f"\n🏆 Top 10 Symbol-Timeframe Combinations:")
        sorted_results = sorted(all_results, key=lambda x: x['total_profit_loss'], reverse=True)
        for i, result in enumerate(sorted_results[:10]):
            print(f"  {i+1:2d}. {result['symbol']} ({result['timeframe']}): ${result['total_profit_loss']:>8.2f} | "
                  f"{result['total_trades']:>3d} trades | {result['win_rate']:>5.1f}% win rate")
        
        print("=" * 80)
        print("✅ Backtesting completed successfully!")
        print("=" * 80)
    
    else:
        print("❌ No successful backtesting results generated")

if __name__ == "__main__":
    main()