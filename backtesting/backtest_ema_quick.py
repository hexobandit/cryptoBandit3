#!/usr/bin/env python3
"""
Quick EMA Trading Strategy Backtesting
=====================================
Faster version focusing on higher timeframes (15m+) for initial analysis.
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

# Trading symbols (reduced for quick testing)
SYMBOLS = ["BTCUSDC", "ETHUSDC", "BNBUSDC", "ADAUSDC", "SOLUSDC"]

# Focus on higher timeframes for speed
TIMEFRAMES = [
    ('15m', Client.KLINE_INTERVAL_15MINUTE),
    ('30m', Client.KLINE_INTERVAL_30MINUTE),
    ('1h', Client.KLINE_INTERVAL_1HOUR),
    ('4h', Client.KLINE_INTERVAL_4HOUR),
    ('1d', Client.KLINE_INTERVAL_1DAY)
]

# Strategy parameters
INITIAL_BALANCE = 10000
TRADE_AMOUNT = 150
TAKE_PROFIT_PERCENT = 0.01
STOP_LOSS_PERCENT = 0.10
TRADING_FEE_PERCENT = 0.001

# EMA parameters
EMA_SHORT_PERIOD = 7
EMA_LONG_PERIOD = 99

def calculate_ema(prices, period):
    """Calculate Exponential Moving Average"""
    if len(prices) < period:
        return [None] * len(prices)
    
    prices = np.array(prices)
    alpha = 2 / (period + 1)
    ema = np.zeros(len(prices))
    ema[0] = prices[0]
    
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    
    return ema

def get_historical_data(symbol, interval, days_back=100):
    """Fetch historical kline data"""
    print(f"  📥 Fetching {days_back} days of {interval} data for {symbol}...")
    
    try:
        start_time = datetime.now() - timedelta(days=days_back)
        start_str = start_time.strftime('%d %b %Y')
        
        klines = client.get_historical_klines(symbol, interval, start_str)
        
        if not klines:
            return None
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        print(f"  ✅ Retrieved {len(df)} candles for {symbol}")
        return df.reset_index(drop=True)
        
    except Exception as e:
        print(f"  ❌ Error fetching data for {symbol}: {e}")
        return None

# Simplified pattern detection for speed
def detect_bullish_pattern(df, idx):
    """Quick bullish pattern detection"""
    if idx < 2:
        return None
    
    current = df.iloc[idx]
    prev = df.iloc[idx-1]
    
    # Simple bullish conditions
    body_size = abs(current['close'] - current['open'])
    range_size = current['high'] - current['low']
    
    # Hammer-like pattern
    if range_size > 0:
        lower_shadow = min(current['open'], current['close']) - current['low']
        if lower_shadow >= 2 * body_size:
            return "Hammer"
    
    # Bullish engulfing
    if (prev['close'] < prev['open'] and 
        current['close'] > current['open'] and
        current['close'] > prev['open']):
        return "Bullish Engulfing"
    
    # Doji
    if range_size > 0 and body_size <= range_size * 0.1:
        return "Doji"
    
    return None

def backtest_symbol_timeframe(symbol, interval_name, interval_code, days_back=100):
    """Quick backtest implementation"""
    print(f"\n🔄 Quick backtest: {symbol} on {interval_name}...")
    
    df = get_historical_data(symbol, interval_code, days_back)
    if df is None or len(df) < EMA_LONG_PERIOD + 10:
        return None
    
    # Calculate EMAs
    closes = df['close'].values
    ema7_values = calculate_ema(closes, EMA_SHORT_PERIOD)
    ema99_values = calculate_ema(closes, EMA_LONG_PERIOD)
    
    df['ema7'] = ema7_values
    df['ema99'] = ema99_values
    
    balance = INITIAL_BALANCE
    position = None
    trades = []
    
    start_idx = EMA_LONG_PERIOD + 10
    
    for i in range(start_idx, len(df)):
        current = df.iloc[i]
        current_price = current['close']
        
        if position is None:
            # Look for entry
            pattern = detect_bullish_pattern(df, i)
            
            if pattern and balance >= TRADE_AMOUNT:
                entry_price = current_price
                quantity = TRADE_AMOUNT / entry_price
                buy_fee = TRADE_AMOUNT * TRADING_FEE_PERCENT
                balance -= TRADE_AMOUNT + buy_fee
                
                position = {
                    'entry_price': entry_price,
                    'quantity': quantity,
                    'entry_idx': i,
                    'pattern': pattern
                }
        
        else:
            # Check exit conditions
            entry_price = position['entry_price']
            price_change = (current_price - entry_price) / entry_price
            
            sell_reason = None
            
            # EMA crossover logic
            if i > start_idx:
                prev_ema7 = df.iloc[i-1]['ema7']
                prev_ema99 = df.iloc[i-1]['ema99']
                current_ema7 = current['ema7']
                current_ema99 = current['ema99']
                
                # Bearish crossover + loss
                if (price_change < 0 and 
                    prev_ema7 > prev_ema99 and current_ema7 < current_ema99):
                    sell_reason = "EMA Crossover Loss"
                
                # Take profit (only if EMA7 <= EMA99)
                elif price_change >= TAKE_PROFIT_PERCENT:
                    if current_ema7 <= current_ema99:
                        sell_reason = "Take Profit"
                
                # Stop loss
                elif price_change <= -STOP_LOSS_PERCENT:
                    sell_reason = "Stop Loss"
            
            if sell_reason:
                sell_value = position['quantity'] * current_price
                sell_fee = sell_value * TRADING_FEE_PERCENT
                balance += sell_value - sell_fee
                
                profit_loss = sell_value - TRADE_AMOUNT - (TRADE_AMOUNT * TRADING_FEE_PERCENT) - sell_fee
                
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pattern': position['pattern'],
                    'exit_reason': sell_reason,
                    'profit_loss': profit_loss,
                    'price_change_percent': price_change * 100
                })
                
                position = None
    
    # Close final position
    if position:
        current_price = df.iloc[-1]['close']
        sell_value = position['quantity'] * current_price
        sell_fee = sell_value * TRADING_FEE_PERCENT
        balance += sell_value - sell_fee
        
        profit_loss = sell_value - TRADE_AMOUNT - (TRADE_AMOUNT * TRADING_FEE_PERCENT) - sell_fee
        trades.append({
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'pattern': position['pattern'],
            'exit_reason': "End of Data",
            'profit_loss': profit_loss,
            'price_change_percent': ((current_price - position['entry_price']) / position['entry_price']) * 100
        })
    
    if trades:
        total_profit_loss = sum(t['profit_loss'] for t in trades)
        winning_trades = len([t for t in trades if t['profit_loss'] > 0])
        win_rate = winning_trades / len(trades) * 100
        
        print(f"  ✅ {len(trades)} trades | Win: {win_rate:.1f}% | P&L: ${total_profit_loss:.2f}")
        
        return {
            'symbol': symbol,
            'timeframe': interval_name,
            'total_trades': len(trades),
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_profit_loss': total_profit_loss,
            'final_balance': balance,
            'return_percent': ((balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
        }
    
    return None

def main():
    """Quick backtesting main function"""
    print("=" * 60)
    print("⚡ Quick EMA Strategy Backtest")
    print("=" * 60)
    print(f"Symbols: {len(SYMBOLS)} pairs")
    print(f"Timeframes: {len(TIMEFRAMES)} intervals (15m+)")
    print(f"Lookback: 100 days")
    print("=" * 60)
    
    all_results = []
    
    for symbol in SYMBOLS:
        for interval_name, interval_code in TIMEFRAMES:
            try:
                result = backtest_symbol_timeframe(symbol, interval_name, interval_code)
                if result:
                    all_results.append(result)
                time.sleep(0.1)
            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'backtest_ema_quick_results_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Summary
    if all_results:
        print(f"\n" + "=" * 60)
        print("📊 QUICK RESULTS SUMMARY")
        print("=" * 60)
        
        total_trades = sum(r['total_trades'] for r in all_results)
        total_profit_loss = sum(r['total_profit_loss'] for r in all_results)
        profitable_combos = len([r for r in all_results if r['total_profit_loss'] > 0])
        
        print(f"Total combinations tested: {len(all_results)}")
        print(f"Total trades executed: {total_trades:,}")
        print(f"Total P&L: ${total_profit_loss:,.2f}")
        print(f"Profitable combinations: {profitable_combos}/{len(all_results)} ({profitable_combos/len(all_results)*100:.1f}%)")
        
        # Best performers
        print(f"\n🏆 Top 5 Performers:")
        sorted_results = sorted(all_results, key=lambda x: x['total_profit_loss'], reverse=True)
        for i, result in enumerate(sorted_results[:5]):
            print(f"  {i+1}. {result['symbol']} ({result['timeframe']}): ${result['total_profit_loss']:>8.2f} | "
                  f"{result['total_trades']:>3d} trades | {result['win_rate']:>5.1f}% win")
        
        print("=" * 60)
        print(f"💾 Results saved to: {results_file}")
    
    else:
        print("❌ No results generated")

if __name__ == "__main__":
    main()