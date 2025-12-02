import os
import json
import datetime
import pandas as pd
import numpy as np
from binance.client import Client
from termcolor import colored
from typing import Dict, List
import sys
sys.path.append('../')
from _secrets import api_key, secret_key

# ==================== MINIMAL CONFIGURATION ====================
EMA_PERIOD = 99
LOOKBACK_DAYS = 999  # Approx. 10 years of daily data
INITIAL_CAPITAL = 100000.0
FIXED_POSITION_SIZE = 1000.0  # Fixed $1000 per trade
TRADING_FEE = 0.001  # 0.1% Binance fee

# Top cryptocurrencies by market cap (USDC pairs)
symbols = [
    # Current list
    "BTCUSDC", "ETHUSDC", "BNBUSDC", "ADAUSDC", "XRPUSDC", 
    "DOGEUSDC", "SOLUSDC", "PNUTUSDC", "PEPEUSDC", "SHIBUSDC", 
    "XLMUSDC", "LINKUSDC", "IOTAUSDC", "ENAUSDC",
    
    # Additional 20 by market cap
    "AVAXUSDC", "TRXUSDC", "DOTUSDC", "TONUSDC", "MATICUSDC",
    "ICPUSDC", "SEIUSDC", "NEARUSDC", "APTUSDC", "UNIUSDC",
    "LTCUSDC", "FILUSDC", "ARBUSDC", "OPUSDC", "INJUSDC",
    "SUIUSDC", "RNDRUSDC", "ATOMUSDC", "WIFUSDC", "FETUSDC"
]

class SimpleEMABacktest:
    def __init__(self):
        self.client = Client(api_key, secret_key)
        self.cache_dir = "simple_backtest_cache"
        self.ensure_cache_dir()
        self.results = {"trades": []}
        
    def ensure_cache_dir(self):
        """Create cache directory if it doesn't exist"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def get_historical_data(self, symbol: str) -> pd.DataFrame:
        """Get daily candles - simplified"""
        cache_file = f"{self.cache_dir}/{symbol}_daily.csv"
        
        # Check cache first
        if os.path.exists(cache_file):
            file_time = datetime.datetime.fromtimestamp(os.path.getmtime(cache_file))
            if (datetime.datetime.now() - file_time).days < 1:
                print(colored(f"📁 Loading {symbol} from cache", "cyan"))
                df = pd.read_csv(cache_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
        
        # Fetch from Binance
        print(colored(f"📊 Fetching {symbol} data...", "yellow"))
        
        try:
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=LOOKBACK_DAYS)
            print(f"Fetching from {start_date.date()} to {end_date.date()}")
            
            # Get all data in chunks
            all_klines = []
            current_start = int(start_date.timestamp() * 1000)
            
            while current_start < int(end_date.timestamp() * 1000):
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=Client.KLINE_INTERVAL_1DAY,
                    startTime=current_start,
                    limit=1000
                )
                
                if not klines:
                    break
                
                all_klines.extend(klines)
                current_start = klines[-1][0] + 86400000  # Next day
            
            if not all_klines:
                print(colored(f"❌ No data for {symbol}", "red"))
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(all_klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close'] = df['close'].astype(float)
            
            # Calculate EMA
            df[f'ema_{EMA_PERIOD}'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
            
            # Save to cache
            df.to_csv(cache_file, index=False)
            print(colored(f"✅ Fetched {len(df)} candles for {symbol}", "green"))
            
            return df
            
        except Exception as e:
            print(colored(f"❌ Error fetching {symbol}: {e}", "red"))
            return pd.DataFrame()
    
    def find_signals(self, df: pd.DataFrame) -> List[int]:
        """Find EMA crossover signals - simplified"""
        if len(df) < EMA_PERIOD + 1:
            return []
        
        signals = []
        
        for i in range(EMA_PERIOD, len(df)):
            prev_close = df['close'].iloc[i-1]
            curr_close = df['close'].iloc[i]
            prev_ema = df[f'ema_{EMA_PERIOD}'].iloc[i-1]
            curr_ema = df[f'ema_{EMA_PERIOD}'].iloc[i]
            
            # Simple bullish crossover: price crosses above EMA
            if prev_close <= prev_ema and curr_close > curr_ema:
                signals.append(i)
        
        return signals
    
    def run_backtest(self):
        """Run simple backtest"""
        print(colored("\\n🚀 Starting Simple EMA Backtest", "cyan"))
        
        capital = INITIAL_CAPITAL
        all_trades = []
        
        # Load data for all symbols
        print(colored("="*60, "blue"))
        data = {}
        for symbol in symbols:
            df = self.get_historical_data(symbol)
            if not df.empty:
                data[symbol] = df
                oldest = df['timestamp'].min().strftime('%Y-%m-%d')
                newest = df['timestamp'].max().strftime('%Y-%m-%d')
                print(colored(f"{symbol:10} | {oldest} to {newest} | {len(df)} days", "white"))
        print(colored("="*60, "blue"))
        
        if not data:
            print(colored("❌ No data available", "red"))
            return
        
        # Process each symbol separately
        for symbol in symbols:
            if symbol not in data:
                continue
                
            df = data[symbol]
            signals = self.find_signals(df)
            
            print(colored(f"\\n📊 Processing {symbol}: Found {len(signals)} signals", "yellow"))
            
            position = None  # Track current position
            
            for signal_idx in signals:
                signal_date = df['timestamp'].iloc[signal_idx]
                entry_price = df['close'].iloc[signal_idx]
                
                # Skip if already in position
                if position is not None:
                    continue
                
                # Skip if not enough capital
                if capital < FIXED_POSITION_SIZE:
                    continue
                
                # Open position
                quantity = FIXED_POSITION_SIZE / entry_price
                capital -= FIXED_POSITION_SIZE
                
                position = {
                    'entry_date': signal_date,
                    'entry_price': entry_price,
                    'quantity': quantity,
                    'entry_idx': signal_idx
                }
                
                print(f"  🟢 BUY {symbol} at {entry_price:.4f} on {signal_date.date()}")
                
                # Look for exit: next crossover below EMA or end of data
                exit_idx = None
                exit_reason = "end_of_data"
                
                for i in range(signal_idx + 1, len(df)):
                    curr_close = df['close'].iloc[i]
                    curr_ema = df[f'ema_{EMA_PERIOD}'].iloc[i]
                    
                    # Exit when price closes below EMA
                    if curr_close < curr_ema:
                        exit_idx = i
                        exit_reason = "ema_cross_down"
                        break
                
                # Use last available data if no exit signal
                if exit_idx is None:
                    exit_idx = len(df) - 1
                
                # Execute exit
                exit_date = df['timestamp'].iloc[exit_idx]
                exit_price = df['close'].iloc[exit_idx]
                
                # Calculate P&L
                gross_pnl = (exit_price - entry_price) * quantity
                fees = (entry_price * quantity + exit_price * quantity) * TRADING_FEE
                net_pnl = gross_pnl - fees
                
                # Return capital
                capital += FIXED_POSITION_SIZE + net_pnl
                
                # Record trade
                trade = {
                    'symbol': symbol,
                    'entry_date': signal_date,
                    'exit_date': exit_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'quantity': quantity,
                    'gross_pnl': gross_pnl,
                    'fees': fees,
                    'net_pnl': net_pnl,
                    'pnl_percent': ((exit_price - entry_price) / entry_price) * 100,
                    'days_held': (exit_date - signal_date).days,
                    'exit_reason': exit_reason
                }
                all_trades.append(trade)
                
                print(f"  🔴 SELL {symbol} at {exit_price:.4f} on {exit_date.date()} | P&L: ${net_pnl:.2f} ({trade['pnl_percent']:.2f}%)")
                
                # Clear position
                position = None
        
        self.results['trades'] = all_trades
        self.print_results(capital)
    
    def print_results(self, final_capital):
        """Print simple results and save to file"""
        trades = self.results['trades']
        
        # Create output content
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_lines = []
        
        output_lines.append("="*80)
        output_lines.append("📊 SIMPLE EMA BACKTEST RESULTS")
        output_lines.append("="*80)
        
        print(colored("\\n" + "="*80, "blue"))
        print(colored("📊 SIMPLE EMA BACKTEST RESULTS", "cyan"))
        print(colored("="*80, "blue"))
        
        strategy_line = f"Strategy: EMA{EMA_PERIOD} Crossover (Close above/below EMA)"
        initial_line = f"Initial Capital: ${INITIAL_CAPITAL:,.2f}"
        final_line = f"Final Capital: ${final_capital:,.2f}"
        total_return = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        return_line = f"Total Return: {total_return:.2f}%"
        
        output_lines.extend([strategy_line, initial_line, final_line, return_line])
        
        print(colored(strategy_line, "yellow"))
        print(initial_line)
        print(final_line)
        color = "green" if total_return > 0 else "red"
        print(colored(return_line, color))
        
        if not trades:
            no_trades_line = "\\nNo trades executed!"
            output_lines.append(no_trades_line)
            print(colored(no_trades_line, "yellow"))
            self.save_to_file(output_lines, timestamp)
            return
        
        # Stats
        winning_trades = [t for t in trades if t['net_pnl'] > 0]
        losing_trades = [t for t in trades if t['net_pnl'] <= 0]
        
        stats_lines = [
            "",
            f"Total Trades: {len(trades)}",
            f"Winning Trades: {len(winning_trades)}",
            f"Losing Trades: {len(losing_trades)}"
        ]
        
        if trades:
            win_rate = (len(winning_trades) / len(trades)) * 100
            win_rate_line = f"Win Rate: {win_rate:.1f}%"
            stats_lines.append(win_rate_line)
        
        total_pnl = sum(t['net_pnl'] for t in trades)
        total_pnl_line = f"Total P&L: ${total_pnl:.2f}"
        stats_lines.append(total_pnl_line)
        
        if trades:
            avg_pnl = total_pnl / len(trades)
            avg_pnl_line = f"Average P&L per trade: ${avg_pnl:.2f}"
            stats_lines.append(avg_pnl_line)
        
        output_lines.extend(stats_lines)
        
        print(f"\\nTotal Trades: {len(trades)}")
        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Losing Trades: {len(losing_trades)}")
        if trades:
            win_rate = (len(winning_trades) / len(trades)) * 100
            print(colored(f"Win Rate: {win_rate:.1f}%", "green" if win_rate > 50 else "red"))
        
        print(f"Total P&L: ${total_pnl:.2f}")
        
        if trades:
            avg_pnl = total_pnl / len(trades)
            print(f"Average P&L per trade: ${avg_pnl:.2f}")
        
        # Trade details
        trade_header = "\\nTRADE DETAILS:"
        trade_columns = f"{'#':>3} | {'Symbol':^8} | {'Entry Date':^12} | {'Exit Date':^12} | {'Days':^4} | {'Entry $':^8} | {'Exit $':^8} | {'P&L $':^8} | {'P&L %':^7} | {'Reason':^12}"
        trade_separator = "-" * 120
        
        output_lines.extend([trade_header, trade_columns, trade_separator])
        
        print(colored(trade_header, "yellow"))
        print(trade_columns)
        print(trade_separator)
        
        for i, trade in enumerate(trades, 1):
            entry_date = trade['entry_date'].strftime('%Y-%m-%d')
            exit_date = trade['exit_date'].strftime('%Y-%m-%d')
            color = "green" if trade['net_pnl'] > 0 else "red"
            
            trade_line = (f"{i:3d} | {trade['symbol']:^8} | {entry_date:^12} | {exit_date:^12} | {trade['days_held']:^4d} | "
                         f"{trade['entry_price']:>8.2f} | {trade['exit_price']:>8.2f} | {trade['net_pnl']:>8.2f} | "
                         f"{trade['pnl_percent']:>7.2f} | {trade['exit_reason']:^12}")
            
            output_lines.append(trade_line)
            
            print(colored(trade_line, color))
        
        # Symbol performance summary
        symbol_header = "\\nSYMBOL PERFORMANCE SUMMARY:"
        symbol_columns = f"{'Symbol':^10} | {'Trades':^7} | {'Wins':^5} | {'Losses':^6} | {'Win %':^6} | {'Total P&L':^10} | {'Avg P&L':^8} | {'Best':^8} | {'Worst':^8}"
        symbol_separator = "-" * 90
        
        output_lines.extend([symbol_header, symbol_columns, symbol_separator])
        
        print(colored(symbol_header, "yellow"))
        print(symbol_columns)
        print(symbol_separator)
        
        symbol_stats = {}
        for trade in trades:
            symbol = trade['symbol']
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pnl': 0,
                    'best_trade': float('-inf'),
                    'worst_trade': float('inf')
                }
            
            stats = symbol_stats[symbol]
            stats['trades'] += 1
            stats['total_pnl'] += trade['net_pnl']
            
            if trade['net_pnl'] > 0:
                stats['wins'] += 1
            else:
                stats['losses'] += 1
            
            stats['best_trade'] = max(stats['best_trade'], trade['net_pnl'])
            stats['worst_trade'] = min(stats['worst_trade'], trade['net_pnl'])
        
        # Sort by total P&L
        sorted_symbols = sorted(symbol_stats.items(), key=lambda x: x[1]['total_pnl'], reverse=True)
        
        for symbol, stats in sorted_symbols:
            win_rate = (stats['wins'] / stats['trades']) * 100 if stats['trades'] > 0 else 0
            avg_pnl = stats['total_pnl'] / stats['trades'] if stats['trades'] > 0 else 0
            color = "green" if stats['total_pnl'] > 0 else "red"
            
            symbol_line = (f"{symbol:^10} | {stats['trades']:^7d} | {stats['wins']:^5d} | {stats['losses']:^6d} | {win_rate:>6.1f} | "
                          f"{stats['total_pnl']:>10.2f} | {avg_pnl:>8.2f} | {stats['best_trade']:>8.2f} | {stats['worst_trade']:>8.2f}")
            
            output_lines.append(symbol_line)
            
            print(colored(symbol_line, color))
        
        final_separator = "="*80
        output_lines.append(final_separator)
        
        print(colored("\\n" + final_separator, "blue"))
        
        # Save to file
        self.save_to_file(output_lines, timestamp)
    
    def save_to_file(self, output_lines, timestamp):
        """Save results to text file"""
        filename = f"ema_backtest_results_{timestamp}.txt"
        
        try:
            with open(filename, 'w') as f:
                for line in output_lines:
                    f.write(line + "\\n")
            
            print(colored(f"\\n📄 Results saved to {filename}", "green"))
        except Exception as e:
            print(colored(f"❌ Error saving file: {e}", "red"))

def main():
    backtest = SimpleEMABacktest()
    backtest.run_backtest()

if __name__ == "__main__":
    main()