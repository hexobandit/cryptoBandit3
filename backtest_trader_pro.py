#!/usr/bin/env python3
"""
Professional Backtester for cBc-trader-pro.py strategies
Tests multiple timeframes and strategies over 180 days of historical data
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
from typing import List, Tuple, Optional, Dict
import statistics
import sys
sys.path.append('../')
from _secrets import api_key, secret_key

# ==================== CONFIGURATION ====================
# Trading symbols
symbols = [
    "BTCUSDC", "ETHUSDC", "BNBUSDC", "ADAUSDC", "XRPUSDC", 
    "DOGEUSDC", "SOLUSDC", "PNUTUSDC", "PEPEUSDC", "SHIBUSDC", 
    "XLMUSDC", "LINKUSDC", "IOTAUSDC", "ENAUSDC"
]

# Timeframes to test
TIMEFRAMES = {
    '15m': Client.KLINE_INTERVAL_15MINUTE,
    '30m': Client.KLINE_INTERVAL_30MINUTE,
    '1h': Client.KLINE_INTERVAL_1HOUR,
    '4h': Client.KLINE_INTERVAL_4HOUR,
}

# Strategy parameters (from cBc-trader-pro.py)
POSITION_SIZE = 100  # USD per trade
TAKE_PROFIT_1 = 0.01  # 1%
TAKE_PROFIT_2 = 0.02  # 2%
STOP_LOSS = 0.02  # 2%
TP1_SELL_PERCENT = 0.5  # Sell 50% at TP1
TRADING_FEE = 0.001  # 0.1% Binance fee

# Trailing stop configuration
TRAIL_INITIAL = 0.008  # 0.8% trail when profit < 1%
TRAIL_MEDIUM = 0.005   # 0.5% trail when 1% < profit < 2%
TRAIL_TIGHT = 0.003    # 0.3% trail when profit > 2%

# BTC correlation filter
BTC_FILTER_ENABLED = True
BTC_DUMP_THRESHOLD = -0.015  # -1.5% in 1 hour

# Minimum risk/reward ratio
MIN_RR_RATIO = 2.0

# Initialize Binance client
client = Client(api_key, secret_key)

@dataclass
class BacktestTrade:
    """Record of a single backtest trade"""
    symbol: str
    strategy: str
    entry_time: datetime.datetime
    exit_time: datetime.datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_percent: float
    exit_reason: str
    timeframe: str
    
@dataclass
class BacktestResult:
    """Results for a single backtest configuration"""
    timeframe: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_percent: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    strategy_breakdown: Dict
    trades: List[BacktestTrade]

class ProfessionalBacktester:
    def __init__(self):
        self.data_cache = {}
        os.makedirs("backtest_cache", exist_ok=True)
        
    def fetch_historical_data(self, symbol: str, interval: str, days: int = 180) -> pd.DataFrame:
        """Fetch historical data with caching"""
        cache_file = f"backtest_cache/{symbol}_{interval}_{days}days.json"
        
        # Check cache (valid for 24 hours)
        if os.path.exists(cache_file):
            mod_time = os.path.getmtime(cache_file)
            if time.time() - mod_time < 86400:  # 24 hours
                try:
                    print(f"  Loading cached data for {symbol} {interval}")
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                    df = pd.DataFrame(data)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    return df
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    print(f"  Cache file corrupted for {symbol}, fetching fresh data...")
                    os.remove(cache_file)
        
        # Fetch fresh data
        print(f"  Fetching fresh data for {symbol} {interval} (180 days)...")
        end_time = datetime.datetime.now()
        start_time = end_time - datetime.timedelta(days=days)
        
        klines = []
        current_start = int(start_time.timestamp() * 1000)
        
        while current_start < int(end_time.timestamp() * 1000):
            try:
                batch = client.get_historical_klines(
                    symbol,
                    interval,
                    current_start,
                    min(current_start + 30 * 24 * 60 * 60 * 1000, int(end_time.timestamp() * 1000))  # 30 days at a time
                )
                klines.extend(batch)
                if not batch:
                    break
                current_start = batch[-1][0] + 1
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                print(f"    Error fetching data: {e}")
                time.sleep(1)
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # Cache data
        cache_data = df.copy()
        cache_data['timestamp'] = cache_data['timestamp'].astype(str)
        with open(cache_file, 'w') as f:
            json.dump(cache_data.to_dict('records'), f)
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period:
            return 50
        
        deltas = prices.diff()
        gain = deltas.where(deltas > 0, 0).rolling(window=period).mean()
        loss = -deltas.where(deltas < 0, 0).rolling(window=period).mean()
        
        if loss.iloc[-1] == 0:
            return 100
        
        rs = gain.iloc[-1] / loss.iloc[-1]
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def find_support_resistance(self, df: pd.DataFrame, lookback: int = 50) -> Tuple[List[float], List[float]]:
        """Find support and resistance levels"""
        if len(df) < lookback:
            return [], []
        
        recent_df = df.iloc[-lookback:]
        
        # Find swing highs and lows
        highs = []
        lows = []
        
        for i in range(2, len(recent_df) - 2):
            # Swing high
            if (recent_df['high'].iloc[i] > recent_df['high'].iloc[i-1] and
                recent_df['high'].iloc[i] > recent_df['high'].iloc[i-2] and
                recent_df['high'].iloc[i] > recent_df['high'].iloc[i+1] and
                recent_df['high'].iloc[i] > recent_df['high'].iloc[i+2]):
                highs.append(recent_df['high'].iloc[i])
            
            # Swing low
            if (recent_df['low'].iloc[i] < recent_df['low'].iloc[i-1] and
                recent_df['low'].iloc[i] < recent_df['low'].iloc[i-2] and
                recent_df['low'].iloc[i] < recent_df['low'].iloc[i+1] and
                recent_df['low'].iloc[i] < recent_df['low'].iloc[i+2]):
                lows.append(recent_df['low'].iloc[i])
        
        # Cluster nearby levels
        support_levels = []
        resistance_levels = []
        
        # Cluster lows for support
        for low in lows:
            added = False
            for i, support in enumerate(support_levels):
                if abs(low - support) / support < 0.005:  # Within 0.5%
                    support_levels[i] = (support + low) / 2  # Average
                    added = True
                    break
            if not added:
                support_levels.append(low)
        
        # Cluster highs for resistance
        for high in highs:
            added = False
            for i, resistance in enumerate(resistance_levels):
                if abs(high - resistance) / resistance < 0.005:  # Within 0.5%
                    resistance_levels[i] = (resistance + high) / 2  # Average
                    added = True
                    break
            if not added:
                resistance_levels.append(high)
        
        return support_levels, resistance_levels
    
    def check_btc_filter(self, btc_df: pd.DataFrame, current_idx: int) -> bool:
        """Check if BTC is dumping (would block alt trades)"""
        if not BTC_FILTER_ENABLED:
            return True
        
        if current_idx < 4:  # Need 1 hour of data (4x15min)
            return True
        
        # Check 1-hour BTC performance
        hour_ago_price = btc_df['close'].iloc[current_idx - 4]
        current_price = btc_df['close'].iloc[current_idx]
        btc_change = (current_price - hour_ago_price) / hour_ago_price
        
        # Block if BTC dumped more than threshold
        if btc_change < BTC_DUMP_THRESHOLD:
            return False
        
        # Check RSI
        btc_rsi = self.calculate_rsi(btc_df['close'][:current_idx+1])
        if btc_rsi < 30:
            return False
        
        return True
    
    def evaluate_entry_strategies(self, df: pd.DataFrame, idx: int, symbol: str) -> List[Dict]:
        """Evaluate all 5 entry strategies at given index"""
        if idx < 50:  # Need historical data
            return []
        
        opportunities = []
        current_price = df['close'].iloc[idx]
        
        # Get support/resistance levels
        support_levels, resistance_levels = self.find_support_resistance(df[:idx+1])
        
        # Strategy 1: Support Bounce
        for support in support_levels:
            distance = abs(current_price - support) / support
            if distance <= 0.003:  # Within 0.3% of support
                stop_loss = support * 0.98  # 2% below support
                take_profit_1 = current_price * (1 + TAKE_PROFIT_1)
                take_profit_2 = current_price * (1 + TAKE_PROFIT_2)
                
                rr_ratio = (take_profit_2 - current_price) / (current_price - stop_loss)
                
                if rr_ratio >= MIN_RR_RATIO:
                    opportunities.append({
                        'type': 'Support Bounce',
                        'score': 75,
                        'reason': f'Support bounce at {support:.2f}',
                        'entry': current_price,
                        'stop_loss': stop_loss,
                        'take_profit_1': take_profit_1,
                        'take_profit_2': take_profit_2,
                        'risk_reward': rr_ratio
                    })
        
        # Strategy 2: Trend Breakout
        if idx >= 20:
            # Simple trend line using linear regression on recent highs
            recent_highs = df['high'].iloc[idx-20:idx]
            x = np.arange(len(recent_highs))
            if len(recent_highs) > 0:
                z = np.polyfit(x, recent_highs, 1)
                trend_line_current = z[0] * (len(recent_highs) - 1) + z[1]
                
                if current_price > trend_line_current * 1.002:  # Break above +0.2%
                    stop_loss = trend_line_current * 0.98
                    take_profit_1 = current_price * (1 + TAKE_PROFIT_1)
                    take_profit_2 = current_price * (1 + TAKE_PROFIT_2)
                    
                    rr_ratio = (take_profit_2 - current_price) / (current_price - stop_loss)
                    
                    if rr_ratio >= MIN_RR_RATIO:
                        opportunities.append({
                            'type': 'Trend Breakout',
                            'score': 70,
                            'reason': f'Trend breakout above {trend_line_current:.2f}',
                            'entry': current_price,
                            'stop_loss': stop_loss,
                            'take_profit_1': take_profit_1,
                            'take_profit_2': take_profit_2,
                            'risk_reward': rr_ratio
                        })
        
        # Strategy 3: RSI Oversold Bounce
        if idx >= 14:
            rsi = self.calculate_rsi(df['close'][:idx+1])
            
            if rsi < 25:
                # Check for price stabilization
                recent_volatility = df['close'].iloc[idx-3:idx+1].std() / df['close'].iloc[idx]
                
                if recent_volatility < 0.002:  # Low volatility = stabilization
                    recent_low = df['low'].iloc[idx-10:idx+1].min()
                    stop_loss = recent_low * 0.99
                    take_profit_1 = current_price * (1 + TAKE_PROFIT_1)
                    take_profit_2 = current_price * (1 + TAKE_PROFIT_2)
                    
                    rr_ratio = (take_profit_2 - current_price) / (current_price - stop_loss)
                    
                    if rr_ratio >= MIN_RR_RATIO:
                        opportunities.append({
                            'type': 'RSI Oversold',
                            'score': 65,
                            'reason': f'RSI oversold at {rsi:.1f}',
                            'entry': current_price,
                            'stop_loss': stop_loss,
                            'take_profit_1': take_profit_1,
                            'take_profit_2': take_profit_2,
                            'risk_reward': rr_ratio
                        })
        
        # Strategy 4: Volume Spike Momentum
        if idx >= 20:
            avg_volume = df['volume'].iloc[idx-20:idx].mean()
            current_volume = df['volume'].iloc[idx]
            
            if current_volume > avg_volume * 2.5:  # 2.5x volume spike
                # Check if green candle
                if df['close'].iloc[idx] > df['open'].iloc[idx]:
                    candle_low = df['low'].iloc[idx]
                    stop_loss = candle_low * 0.99
                    take_profit_1 = current_price * (1 + TAKE_PROFIT_1)
                    take_profit_2 = current_price * (1 + TAKE_PROFIT_2)
                    
                    rr_ratio = (take_profit_2 - current_price) / (current_price - stop_loss)
                    
                    if rr_ratio >= MIN_RR_RATIO:
                        opportunities.append({
                            'type': 'Volume Spike',
                            'score': 70,
                            'reason': f'Volume spike {current_volume/avg_volume:.1f}x average',
                            'entry': current_price,
                            'stop_loss': stop_loss,
                            'take_profit_1': take_profit_1,
                            'take_profit_2': take_profit_2,
                            'risk_reward': rr_ratio
                        })
        
        # Strategy 5: Liquidity Hunt Recovery
        if idx >= 3:
            two_candles_ago_low = df['low'].iloc[idx-2]
            prev_low = df['low'].iloc[idx-1]
            current_low = df['low'].iloc[idx]
            
            # Check for stop hunt pattern
            if prev_low < two_candles_ago_low * 0.995:  # Wick below
                if current_price > two_candles_ago_low:  # Recovery
                    stop_loss = prev_low * 0.995
                    take_profit_1 = current_price * (1 + TAKE_PROFIT_1)
                    take_profit_2 = current_price * (1 + TAKE_PROFIT_2)
                    
                    rr_ratio = (take_profit_2 - current_price) / (current_price - stop_loss)
                    
                    if rr_ratio >= MIN_RR_RATIO:
                        opportunities.append({
                            'type': 'Liquidity Hunt',
                            'score': 60,
                            'reason': 'Stop hunt recovery pattern',
                            'entry': current_price,
                            'stop_loss': stop_loss,
                            'take_profit_1': take_profit_1,
                            'take_profit_2': take_profit_2,
                            'risk_reward': rr_ratio
                        })
        
        # Sort by score and return best opportunity
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        return opportunities[:1] if opportunities else []
    
    def simulate_trade(self, df: pd.DataFrame, entry_idx: int, opportunity: Dict, 
                      symbol: str, timeframe: str) -> Optional[BacktestTrade]:
        """Simulate a trade from entry to exit"""
        entry_price = opportunity['entry']
        stop_loss = opportunity['stop_loss']
        tp1 = opportunity['take_profit_1']
        tp2 = opportunity['take_profit_2']
        
        quantity = POSITION_SIZE / entry_price
        
        # Track position state
        remaining_quantity = quantity
        realized_pnl = 0
        tp1_hit = False
        highest_price = entry_price
        
        # Simulate from entry to exit
        for i in range(entry_idx + 1, len(df)):
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            current_close = df['close'].iloc[i]
            
            # Update highest price for trailing stop
            if current_high > highest_price:
                highest_price = current_high
            
            # Check stop loss
            if current_low <= stop_loss:
                # Position stopped out
                exit_price = stop_loss
                pnl = (exit_price - entry_price) * remaining_quantity
                fees = (entry_price * quantity + exit_price * remaining_quantity) * TRADING_FEE
                net_pnl = pnl - fees
                
                return BacktestTrade(
                    symbol=symbol,
                    strategy=opportunity['type'],
                    entry_time=df['timestamp'].iloc[entry_idx],
                    exit_time=df['timestamp'].iloc[i],
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=quantity,
                    pnl=net_pnl,
                    pnl_percent=(net_pnl / POSITION_SIZE) * 100,
                    exit_reason='Stop Loss',
                    timeframe=timeframe
                )
            
            # Check TP1 (sell 50%)
            if not tp1_hit and current_high >= tp1:
                tp1_hit = True
                sell_quantity = quantity * TP1_SELL_PERCENT
                remaining_quantity -= sell_quantity
                partial_pnl = (tp1 - entry_price) * sell_quantity
                realized_pnl += partial_pnl
                
                # Move stop to breakeven
                stop_loss = entry_price
            
            # Check TP2 (close position)
            if current_high >= tp2:
                exit_price = tp2
                final_pnl = (exit_price - entry_price) * remaining_quantity
                total_pnl = realized_pnl + final_pnl
                fees = (entry_price * quantity + tp1 * quantity * TP1_SELL_PERCENT + 
                       exit_price * remaining_quantity) * TRADING_FEE
                net_pnl = total_pnl - fees
                
                return BacktestTrade(
                    symbol=symbol,
                    strategy=opportunity['type'],
                    entry_time=df['timestamp'].iloc[entry_idx],
                    exit_time=df['timestamp'].iloc[i],
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=quantity,
                    pnl=net_pnl,
                    pnl_percent=(net_pnl / POSITION_SIZE) * 100,
                    exit_reason='Take Profit 2',
                    timeframe=timeframe
                )
            
            # Implement trailing stop after TP1
            if tp1_hit and highest_price > entry_price * 1.01:
                profit_percent = ((highest_price - entry_price) / entry_price) * 100
                
                if profit_percent < 1.0:
                    trail_percent = TRAIL_INITIAL
                elif profit_percent < 2.0:
                    trail_percent = TRAIL_MEDIUM
                else:
                    trail_percent = TRAIL_TIGHT
                
                new_stop = highest_price * (1 - trail_percent)
                if new_stop > stop_loss:
                    stop_loss = new_stop
        
        # Position still open at end of data - close at market
        exit_price = df['close'].iloc[-1]
        if tp1_hit:
            final_pnl = realized_pnl + (exit_price - entry_price) * remaining_quantity
            fees = (entry_price * quantity + tp1 * quantity * TP1_SELL_PERCENT + 
                   exit_price * remaining_quantity) * TRADING_FEE
        else:
            final_pnl = (exit_price - entry_price) * quantity
            fees = (entry_price * quantity + exit_price * quantity) * TRADING_FEE
        
        net_pnl = final_pnl - fees
        
        return BacktestTrade(
            symbol=symbol,
            strategy=opportunity['type'],
            entry_time=df['timestamp'].iloc[entry_idx],
            exit_time=df['timestamp'].iloc[-1],
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl=net_pnl,
            pnl_percent=(net_pnl / POSITION_SIZE) * 100,
            exit_reason='End of Data',
            timeframe=timeframe
        )
    
    def backtest_timeframe(self, timeframe: str, interval: str) -> BacktestResult:
        """Run backtest for a specific timeframe"""
        print(f"\n{'='*80}")
        print(f"BACKTESTING {timeframe.upper()} TIMEFRAME")
        print(f"{'='*80}")
        
        all_trades = []
        positions = {}  # Track open positions per symbol
        
        # Fetch BTC data for filter
        btc_df = self.fetch_historical_data("BTCUSDC", interval)
        
        # Process each symbol
        for symbol in symbols:
            print(f"\nProcessing {symbol}...")
            df = self.fetch_historical_data(symbol, interval)
            
            if len(df) < 50:
                print(f"  Insufficient data for {symbol}")
                continue
            
            # Simulate trading
            for i in range(50, len(df)):
                # Skip if position already open for this symbol
                if symbol in positions:
                    continue
                
                # Check BTC filter for alts
                if symbol != "BTCUSDC":
                    if not self.check_btc_filter(btc_df, i):
                        continue
                
                # Evaluate entry opportunities
                opportunities = self.evaluate_entry_strategies(df, i, symbol)
                
                if opportunities:
                    # Take the best opportunity
                    opportunity = opportunities[0]
                    
                    # Simulate the trade
                    trade = self.simulate_trade(df, i, opportunity, symbol, timeframe)
                    
                    if trade:
                        all_trades.append(trade)
                        positions[symbol] = i  # Mark position as open
                        
                        # Clear position after exit
                        exit_idx = df[df['timestamp'] == trade.exit_time].index[0] if not df[df['timestamp'] == trade.exit_time].empty else len(df) - 1
                        if exit_idx < len(df) - 1:
                            del positions[symbol]
        
        # Calculate statistics
        if not all_trades:
            return BacktestResult(
                timeframe=timeframe,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                total_pnl=0,
                total_pnl_percent=0,
                max_drawdown=0,
                sharpe_ratio=0,
                profit_factor=0,
                avg_win=0,
                avg_loss=0,
                best_trade=0,
                worst_trade=0,
                strategy_breakdown={},
                trades=[]
            )
        
        # Calculate metrics
        winning_trades = [t for t in all_trades if t.pnl > 0]
        losing_trades = [t for t in all_trades if t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in all_trades)
        total_pnl_percent = (total_pnl / (POSITION_SIZE * len(all_trades))) * 100 if all_trades else 0
        
        # Calculate drawdown
        cumulative_pnl = []
        running_total = 0
        peak = 0
        max_drawdown = 0
        
        for trade in sorted(all_trades, key=lambda x: x.exit_time):
            running_total += trade.pnl
            cumulative_pnl.append(running_total)
            if running_total > peak:
                peak = running_total
            drawdown = (peak - running_total) / POSITION_SIZE if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate Sharpe ratio
        returns = [t.pnl_percent for t in all_trades]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
        
        # Calculate profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Strategy breakdown
        strategy_breakdown = defaultdict(lambda: {
            'trades': 0, 'wins': 0, 'pnl': 0, 'win_rate': 0
        })
        
        for trade in all_trades:
            strategy_breakdown[trade.strategy]['trades'] += 1
            if trade.pnl > 0:
                strategy_breakdown[trade.strategy]['wins'] += 1
            strategy_breakdown[trade.strategy]['pnl'] += trade.pnl
        
        for strategy in strategy_breakdown:
            if strategy_breakdown[strategy]['trades'] > 0:
                strategy_breakdown[strategy]['win_rate'] = (
                    strategy_breakdown[strategy]['wins'] / 
                    strategy_breakdown[strategy]['trades']
                ) * 100
        
        return BacktestResult(
            timeframe=timeframe,
            total_trades=len(all_trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=(len(winning_trades) / len(all_trades) * 100) if all_trades else 0,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            max_drawdown=max_drawdown * 100,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            avg_win=np.mean([t.pnl for t in winning_trades]) if winning_trades else 0,
            avg_loss=np.mean([t.pnl for t in losing_trades]) if losing_trades else 0,
            best_trade=max(t.pnl for t in all_trades) if all_trades else 0,
            worst_trade=min(t.pnl for t in all_trades) if all_trades else 0,
            strategy_breakdown=dict(strategy_breakdown),
            trades=all_trades
        )
    
    def run_full_backtest(self):
        """Run backtest for all timeframes"""
        results = {}
        
        print("\n" + "="*80)
        print("PROFESSIONAL TRADER BACKTESTER")
        print("Testing 180 days of historical data")
        print("Strategies: Support Bounce, Trend Breakout, RSI Oversold, Volume Spike, Liquidity Hunt")
        print("="*80)
        
        for timeframe, interval in TIMEFRAMES.items():
            result = self.backtest_timeframe(timeframe, interval)
            results[timeframe] = result
            
            # Print summary
            print(f"\n{timeframe.upper()} RESULTS:")
            print(f"  Total Trades: {result.total_trades}")
            print(f"  Win Rate: {result.win_rate:.1f}%")
            print(f"  Total P&L: ${result.total_pnl:.2f} ({result.total_pnl_percent:.1f}%)")
            print(f"  Max Drawdown: {result.max_drawdown:.1f}%")
            print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print(f"  Profit Factor: {result.profit_factor:.2f}")
        
        return results
    
    def generate_report(self, results: Dict[str, BacktestResult]):
        """Generate detailed HTML report"""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Professional Trader Backtest Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 10px; }
        .summary { background: white; padding: 20px; margin: 20px 0; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .timeframe-section { background: white; padding: 20px; margin: 20px 0; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .metric { padding: 15px; background: #f8f9fa; border-radius: 5px; }
        .metric-value { font-size: 24px; font-weight: bold; }
        .metric-label { color: #666; font-size: 12px; margin-top: 5px; }
        .positive { color: #27ae60; }
        .negative { color: #e74c3c; }
        .chart { height: 400px; margin: 20px 0; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #34495e; color: white; }
        .best { background: #d4edda; }
        .worst { background: #f8d7da; }
        h2 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Professional Trader Backtest Report</h1>
        <p>180 Days Historical Analysis | Multiple Timeframes & Strategies</p>
        <p>Generated: """ + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    </div>
"""
        
        # Find best timeframe
        best_timeframe = max(results.keys(), key=lambda x: results[x].total_pnl)
        best_result = results[best_timeframe]
        
        # Overall summary
        html += f"""
    <div class="summary">
        <h2>📊 Overall Summary</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value {'positive' if best_result.total_pnl > 0 else 'negative'}">
                    {best_timeframe.upper()}
                </div>
                <div class="metric-label">BEST TIMEFRAME</div>
            </div>
            <div class="metric">
                <div class="metric-value {'positive' if best_result.total_pnl > 0 else 'negative'}">
                    ${best_result.total_pnl:.2f}
                </div>
                <div class="metric-label">BEST TOTAL P&L</div>
            </div>
            <div class="metric">
                <div class="metric-value">
                    {best_result.win_rate:.1f}%
                </div>
                <div class="metric-label">BEST WIN RATE</div>
            </div>
            <div class="metric">
                <div class="metric-value">
                    {best_result.profit_factor:.2f}
                </div>
                <div class="metric-label">BEST PROFIT FACTOR</div>
            </div>
        </div>
    </div>
"""
        
        # Timeframe comparison chart
        timeframes_list = list(results.keys())
        pnls = [results[tf].total_pnl for tf in timeframes_list]
        win_rates = [results[tf].win_rate for tf in timeframes_list]
        trade_counts = [results[tf].total_trades for tf in timeframes_list]
        
        html += """
    <div class="summary">
        <h2>📈 Timeframe Comparison</h2>
        <div id="timeframeChart"></div>
        <script>
            var trace1 = {
                x: """ + str(timeframes_list) + """,
                y: """ + str(pnls) + """,
                name: 'Total P&L ($)',
                type: 'bar',
                marker: { color: 'rgb(52, 152, 219)' }
            };
            
            var trace2 = {
                x: """ + str(timeframes_list) + """,
                y: """ + str(win_rates) + """,
                name: 'Win Rate (%)',
                type: 'bar',
                yaxis: 'y2',
                marker: { color: 'rgb(46, 204, 113)' }
            };
            
            var layout = {
                title: 'Performance by Timeframe',
                yaxis: { title: 'P&L ($)' },
                yaxis2: {
                    title: 'Win Rate (%)',
                    overlaying: 'y',
                    side: 'right'
                }
            };
            
            Plotly.newPlot('timeframeChart', [trace1, trace2], layout);
        </script>
    </div>
"""
        
        # Detailed results for each timeframe
        for timeframe, result in results.items():
            pnl_class = 'positive' if result.total_pnl > 0 else 'negative'
            
            html += f"""
    <div class="timeframe-section">
        <h2>⏰ {timeframe.upper()} Timeframe Results</h2>
        
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{result.total_trades}</div>
                <div class="metric-label">TOTAL TRADES</div>
            </div>
            <div class="metric">
                <div class="metric-value">{result.win_rate:.1f}%</div>
                <div class="metric-label">WIN RATE</div>
            </div>
            <div class="metric">
                <div class="metric-value {pnl_class}">${result.total_pnl:.2f}</div>
                <div class="metric-label">TOTAL P&L</div>
            </div>
            <div class="metric">
                <div class="metric-value">{result.max_drawdown:.1f}%</div>
                <div class="metric-label">MAX DRAWDOWN</div>
            </div>
            <div class="metric">
                <div class="metric-value">{result.sharpe_ratio:.2f}</div>
                <div class="metric-label">SHARPE RATIO</div>
            </div>
            <div class="metric">
                <div class="metric-value">{result.profit_factor:.2f}</div>
                <div class="metric-label">PROFIT FACTOR</div>
            </div>
        </div>
        
        <h3>Strategy Breakdown</h3>
        <table>
            <tr>
                <th>Strategy</th>
                <th>Trades</th>
                <th>Wins</th>
                <th>Win Rate</th>
                <th>Total P&L</th>
            </tr>
"""
            
            for strategy, stats in result.strategy_breakdown.items():
                html += f"""
            <tr>
                <td>{strategy}</td>
                <td>{stats['trades']}</td>
                <td>{stats['wins']}</td>
                <td>{stats['win_rate']:.1f}%</td>
                <td class="{'positive' if stats['pnl'] > 0 else 'negative'}">${stats['pnl']:.2f}</td>
            </tr>
"""
            
            html += """
        </table>
    </div>
"""
        
        # Strategy performance across all timeframes
        all_strategies = set()
        for result in results.values():
            all_strategies.update(result.strategy_breakdown.keys())
        
        html += """
    <div class="summary">
        <h2>🎯 Strategy Performance Summary</h2>
        <table>
            <tr>
                <th>Strategy</th>
"""
        for tf in timeframes_list:
            html += f"<th>{tf.upper()}</th>"
        html += "<th>Total P&L</th></tr>"
        
        for strategy in all_strategies:
            html += f"<tr><td><strong>{strategy}</strong></td>"
            total_strategy_pnl = 0
            for tf in timeframes_list:
                if strategy in results[tf].strategy_breakdown:
                    pnl = results[tf].strategy_breakdown[strategy]['pnl']
                    total_strategy_pnl += pnl
                    html += f"<td class='{'positive' if pnl > 0 else 'negative'}'>${pnl:.2f}</td>"
                else:
                    html += "<td>-</td>"
            html += f"<td class='{'positive' if total_strategy_pnl > 0 else 'negative'}'><strong>${total_strategy_pnl:.2f}</strong></td></tr>"
        
        html += """
        </table>
    </div>
    
    <div class="summary">
        <h2>📋 Recommendations</h2>
        <ul>
"""
        
        # Generate recommendations
        if best_result.win_rate > 50:
            html += f"<li>✅ <strong>{best_timeframe.upper()}</strong> timeframe shows the best performance with {best_result.win_rate:.1f}% win rate</li>"
        
        # Find best strategy
        best_strategy = None
        best_strategy_pnl = -float('inf')
        for result in results.values():
            for strategy, stats in result.strategy_breakdown.items():
                if stats['pnl'] > best_strategy_pnl:
                    best_strategy = strategy
                    best_strategy_pnl = stats['pnl']
        
        if best_strategy:
            html += f"<li>🎯 <strong>{best_strategy}</strong> is the most profitable strategy with ${best_strategy_pnl:.2f} total P&L</li>"
        
        if best_result.sharpe_ratio > 1:
            html += f"<li>📊 Sharpe ratio of {best_result.sharpe_ratio:.2f} indicates good risk-adjusted returns</li>"
        
        if best_result.max_drawdown < 5:
            html += f"<li>🛡️ Low maximum drawdown of {best_result.max_drawdown:.1f}% shows good risk management</li>"
        
        html += """
        </ul>
    </div>
</body>
</html>
"""
        
        # Save report
        report_file = f"backtest_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_file, 'w') as f:
            f.write(html)
        
        print(f"\n📊 Report saved to: {report_file}")
        return report_file

def main():
    """Run the backtest"""
    backtester = ProfessionalBacktester()
    results = backtester.run_full_backtest()
    
    # Generate detailed report
    report_file = backtester.generate_report(results)
    
    # Print final summary
    print("\n" + "="*80)
    print("BACKTEST COMPLETE")
    print("="*80)
    
    # Find and display best configuration
    best_timeframe = None
    best_pnl = -float('inf')
    
    for timeframe, result in results.items():
        if result.total_pnl > best_pnl:
            best_pnl = result.total_pnl
            best_timeframe = timeframe
    
    if best_timeframe:
        best = results[best_timeframe]
        print(f"\n🏆 BEST CONFIGURATION: {best_timeframe.upper()}")
        print(f"   Total P&L: ${best.total_pnl:.2f}")
        print(f"   Win Rate: {best.win_rate:.1f}%")
        print(f"   Total Trades: {best.total_trades}")
        print(f"   Sharpe Ratio: {best.sharpe_ratio:.2f}")
        print(f"   Profit Factor: {best.profit_factor:.2f}")
        print(f"   Max Drawdown: {best.max_drawdown:.1f}%")
        
        print("\n📊 Strategy Performance:")
        for strategy, stats in best.strategy_breakdown.items():
            print(f"   {strategy}: ${stats['pnl']:.2f} ({stats['trades']} trades, {stats['win_rate']:.1f}% win rate)")
    
    print(f"\n📈 Open the report for detailed analysis: {report_file}")
    
    return results

if __name__ == "__main__":
    main()