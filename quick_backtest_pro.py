#!/usr/bin/env python3
"""
Quick Backtest for Professional Trader Strategies
Focused analysis on top symbols and timeframes
"""

import os
import json
import time
import datetime
import pandas as pd
import numpy as np
from binance.client import Client
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional
import sys
sys.path.append('../')
from _secrets import api_key, secret_key

# Focus on top trading pairs only
symbols = ["BTCUSDC", "ETHUSDC", "BNBUSDC", "SOLUSDC"]

# Test key timeframes
TIMEFRAMES = {
    '15m': Client.KLINE_INTERVAL_15MINUTE,
    '1h': Client.KLINE_INTERVAL_1HOUR,
    '4h': Client.KLINE_INTERVAL_4HOUR,
}

# Strategy parameters
POSITION_SIZE = 100
TAKE_PROFIT_1 = 0.01
TAKE_PROFIT_2 = 0.02
STOP_LOSS = 0.02
TP1_SELL_PERCENT = 0.5
TRADING_FEE = 0.001
MIN_RR_RATIO = 2.0

client = Client(api_key, secret_key)

@dataclass
class QuickBacktestResult:
    timeframe: str
    total_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl_per_trade: float
    strategy_performance: Dict

class QuickBacktester:
    def __init__(self):
        os.makedirs("backtest_cache", exist_ok=True)
        
    def fetch_data(self, symbol: str, interval: str, days: int = 30) -> pd.DataFrame:
        """Fetch recent historical data"""
        cache_file = f"backtest_cache/quick_{symbol}_{interval}.json"
        
        # Use cache if available and fresh (< 4 hours old)
        if os.path.exists(cache_file):
            mod_time = os.path.getmtime(cache_file)
            if time.time() - mod_time < 14400:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
        
        print(f"  Fetching {symbol} {interval} data...")
        klines = client.get_historical_klines(
            symbol, interval, f"{days} days ago UTC"
        )
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # Cache
        cache_data = df.copy()
        cache_data['timestamp'] = cache_data['timestamp'].astype(str)
        with open(cache_file, 'w') as f:
            json.dump(cache_data.to_dict('records'), f)
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Quick RSI calculation"""
        if len(prices) < period:
            return 50
        
        deltas = prices.diff()
        gain = deltas.where(deltas > 0, 0).rolling(window=period).mean()
        loss = -deltas.where(deltas < 0, 0).rolling(window=period).mean()
        
        if loss.iloc[-1] == 0:
            return 100
        
        rs = gain.iloc[-1] / loss.iloc[-1]
        return 100 - (100 / (1 + rs))
    
    def test_strategies(self, df: pd.DataFrame) -> Dict:
        """Test all 5 strategies on the data"""
        results = {
            'Support Bounce': {'trades': 0, 'wins': 0, 'pnl': 0},
            'Trend Breakout': {'trades': 0, 'wins': 0, 'pnl': 0},
            'RSI Oversold': {'trades': 0, 'wins': 0, 'pnl': 0},
            'Volume Spike': {'trades': 0, 'wins': 0, 'pnl': 0},
            'Liquidity Hunt': {'trades': 0, 'wins': 0, 'pnl': 0},
        }
        
        position_open = False
        
        for i in range(20, len(df) - 5):
            if position_open:
                continue
            
            current_price = df['close'].iloc[i]
            opportunities = []
            
            # Strategy 1: Support Bounce
            recent_lows = df['low'].iloc[i-20:i]
            support = recent_lows.min()
            if abs(current_price - support) / support <= 0.003:
                opportunities.append(('Support Bounce', current_price))
            
            # Strategy 2: Trend Breakout (simplified)
            if i >= 10:
                ma_10 = df['close'].iloc[i-10:i].mean()
                if current_price > ma_10 * 1.002:
                    opportunities.append(('Trend Breakout', current_price))
            
            # Strategy 3: RSI Oversold
            rsi = self.calculate_rsi(df['close'][:i+1])
            if rsi < 25:
                opportunities.append(('RSI Oversold', current_price))
            
            # Strategy 4: Volume Spike
            avg_volume = df['volume'].iloc[i-20:i].mean()
            if df['volume'].iloc[i] > avg_volume * 2.5:
                if df['close'].iloc[i] > df['open'].iloc[i]:
                    opportunities.append(('Volume Spike', current_price))
            
            # Strategy 5: Liquidity Hunt
            if i >= 3:
                if df['low'].iloc[i-1] < df['low'].iloc[i-2] * 0.995:
                    if current_price > df['low'].iloc[i-2]:
                        opportunities.append(('Liquidity Hunt', current_price))
            
            # Take first opportunity
            if opportunities:
                strategy, entry_price = opportunities[0]
                position_open = True
                
                # Simulate trade outcome
                exit_found = False
                for j in range(i+1, min(i+20, len(df))):
                    # Check stop loss
                    if df['low'].iloc[j] <= entry_price * (1 - STOP_LOSS):
                        pnl = -POSITION_SIZE * STOP_LOSS
                        pnl -= POSITION_SIZE * TRADING_FEE * 2
                        results[strategy]['trades'] += 1
                        results[strategy]['pnl'] += pnl
                        exit_found = True
                        break
                    
                    # Check take profit
                    if df['high'].iloc[j] >= entry_price * (1 + TAKE_PROFIT_2):
                        pnl = POSITION_SIZE * TAKE_PROFIT_2
                        pnl -= POSITION_SIZE * TRADING_FEE * 2
                        results[strategy]['trades'] += 1
                        results[strategy]['wins'] += 1
                        results[strategy]['pnl'] += pnl
                        exit_found = True
                        break
                
                if not exit_found:
                    # Exit at current price
                    exit_price = df['close'].iloc[min(i+20, len(df)-1)]
                    pnl = POSITION_SIZE * ((exit_price - entry_price) / entry_price)
                    pnl -= POSITION_SIZE * TRADING_FEE * 2
                    results[strategy]['trades'] += 1
                    if pnl > 0:
                        results[strategy]['wins'] += 1
                    results[strategy]['pnl'] += pnl
                
                position_open = False
        
        return results
    
    def run_quick_test(self):
        """Run quick backtest on all timeframes"""
        print("\n" + "="*80)
        print("QUICK PROFESSIONAL TRADER BACKTEST")
        print("30 Days | 4 Top Symbols | 3 Timeframes")
        print("="*80)
        
        all_results = {}
        
        for timeframe, interval in TIMEFRAMES.items():
            print(f"\nTesting {timeframe} timeframe...")
            
            combined_results = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0})
            
            for symbol in symbols:
                df = self.fetch_data(symbol, interval, days=30)
                results = self.test_strategies(df)
                
                for strategy, stats in results.items():
                    combined_results[strategy]['trades'] += stats['trades']
                    combined_results[strategy]['wins'] += stats['wins']
                    combined_results[strategy]['pnl'] += stats['pnl']
            
            # Calculate metrics
            total_trades = sum(s['trades'] for s in combined_results.values())
            total_wins = sum(s['wins'] for s in combined_results.values())
            total_pnl = sum(s['pnl'] for s in combined_results.values())
            
            all_results[timeframe] = QuickBacktestResult(
                timeframe=timeframe,
                total_trades=total_trades,
                win_rate=(total_wins / total_trades * 100) if total_trades > 0 else 0,
                total_pnl=total_pnl,
                avg_pnl_per_trade=total_pnl / total_trades if total_trades > 0 else 0,
                strategy_performance=dict(combined_results)
            )
            
            print(f"  Trades: {total_trades} | Win Rate: {all_results[timeframe].win_rate:.1f}%")
            print(f"  P&L: ${total_pnl:.2f}")
        
        return all_results
    
    def print_summary(self, results: Dict):
        """Print detailed summary"""
        print("\n" + "="*80)
        print("DETAILED RESULTS BY TIMEFRAME")
        print("="*80)
        
        for timeframe, result in results.items():
            print(f"\n📊 {timeframe.upper()} TIMEFRAME")
            print(f"  Total Trades: {result.total_trades}")
            print(f"  Win Rate: {result.win_rate:.1f}%")
            print(f"  Total P&L: ${result.total_pnl:.2f}")
            print(f"  Avg P&L per Trade: ${result.avg_pnl_per_trade:.2f}")
            
            print("\n  Strategy Breakdown:")
            for strategy, stats in result.strategy_performance.items():
                if stats['trades'] > 0:
                    win_rate = (stats['wins'] / stats['trades']) * 100
                    avg_pnl = stats['pnl'] / stats['trades']
                    print(f"    {strategy:20} - Trades: {stats['trades']:3} | Win: {win_rate:.1f}% | P&L: ${stats['pnl']:7.2f} | Avg: ${avg_pnl:.2f}")
        
        # Find best configuration
        best_timeframe = max(results.keys(), key=lambda x: results[x].total_pnl)
        best_result = results[best_timeframe]
        
        print("\n" + "="*80)
        print("🏆 OPTIMAL CONFIGURATION")
        print("="*80)
        print(f"  Timeframe: {best_timeframe.upper()}")
        print(f"  Total P&L: ${best_result.total_pnl:.2f}")
        print(f"  Win Rate: {best_result.win_rate:.1f}%")
        print(f"  Avg per Trade: ${best_result.avg_pnl_per_trade:.2f}")
        
        # Find best strategy
        all_strategies = defaultdict(lambda: {'pnl': 0, 'trades': 0})
        for result in results.values():
            for strategy, stats in result.strategy_performance.items():
                all_strategies[strategy]['pnl'] += stats['pnl']
                all_strategies[strategy]['trades'] += stats['trades']
        
        best_strategy = max(all_strategies.keys(), key=lambda x: all_strategies[x]['pnl'])
        
        print(f"\n  Best Strategy: {best_strategy}")
        print(f"    Total P&L: ${all_strategies[best_strategy]['pnl']:.2f}")
        print(f"    Total Trades: {all_strategies[best_strategy]['trades']}")
        
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        
        if best_result.win_rate > 45:
            print(f"✅ Use {best_timeframe.upper()} timeframe for optimal results")
        
        if all_strategies[best_strategy]['pnl'] > 0:
            print(f"✅ Focus on {best_strategy} strategy")
        
        # Strategy specific recommendations
        profitable_strategies = [s for s in all_strategies.keys() 
                               if all_strategies[s]['pnl'] > 0]
        
        if profitable_strategies:
            print(f"✅ Profitable strategies: {', '.join(profitable_strategies)}")
        
        print("\nNote: This is a quick 30-day backtest. Run full 180-day analysis for comprehensive results.")

def main():
    backtester = QuickBacktester()
    results = backtester.run_quick_test()
    backtester.print_summary(results)

if __name__ == "__main__":
    main()