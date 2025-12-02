import os
import json
import datetime
import pandas as pd
import numpy as np
from binance.client import Client
from termcolor import colored
# import matplotlib.pyplot as plt  # Removed - no graphs needed
from typing import Dict, List, Tuple
import sys
sys.path.append('../')
from _secrets import api_key, secret_key

# ==================== BACKTEST CONFIGURATION ====================
# Strategy Parameters (same as live bot)
EMA_PERIOD = 99
STOP_LOSS_PERCENT = 3.0
TRAILING_STOP_ACTIVATION = 5.0
TRAILING_STOP_DISTANCE = 3.0  # Trail by 3% from highest

# Backtest Settings
LOOKBACK_DAYS = 1000  # 2 years +
INITIAL_CAPITAL = 10000.0
FIXED_POSITION_SIZE = 1000.0  # Fixed $1000 per trade
MAX_POSITIONS = 3
MIN_VOLUME_RATIO = 1.2
BTC_FILTER = True
TRADING_FEE = 0.001  # 0.1% Binance fee

# Symbols to test
symbols = [
    "BTCUSDC", "ETHUSDC", "BNBUSDC", "ADAUSDC", "XRPUSDC", 
    "DOGEUSDC", "SOLUSDC", "PEPEUSDC", "SHIBUSDC", 
    "XLMUSDC", "LINKUSDC", "IOTAUSDC"
]

class DailyEMABacktest:
    def __init__(self):
        self.client = Client(api_key, secret_key)
        self.cache_dir = "backtest_cache"
        self.ensure_cache_dir()
        self.results = {
            "trades": [],
            "equity_curve": [],
            "statistics": {},
            "symbol_performance": {}
        }
        
    def ensure_cache_dir(self):
        """Create cache directory if it doesn't exist"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def get_historical_data(self, symbol: str) -> pd.DataFrame:
        """Fetch or load historical daily data"""
        cache_file = f"{self.cache_dir}/{symbol}_daily_2y.csv"
        
        # Check cache
        if os.path.exists(cache_file):
            # Check if cache is recent (less than 1 day old)
            file_time = datetime.datetime.fromtimestamp(os.path.getmtime(cache_file))
            if (datetime.datetime.now() - file_time).days < 1:
                print(colored(f"📁 Loading {symbol} from cache", "cyan"))
                df = pd.read_csv(cache_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
        
        # Fetch from Binance
        print(colored(f"📊 Fetching {symbol} data from Binance...", "yellow"))
        
        try:
            # Calculate start date
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=LOOKBACK_DAYS)
            print(f"DEBUG: Fetching {symbol} from {start_date.date()} to {end_date.date()}")
            
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
                current_start = klines[-1][0] + 86400000  # Next day in ms
            
            if not all_klines:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(all_klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Save to cache
            df.to_csv(cache_file, index=False)
            
            return df
            
        except Exception as e:
            print(colored(f"❌ Error fetching {symbol}: {e}", "red"))
            return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        if df.empty:
            return df
        
        # EMA
        df[f'ema_{EMA_PERIOD}'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
        
        # Volume average
        df['volume_avg'] = df['volume'].rolling(window=20).mean()
        
        # Signal generation
        df['signal'] = 0
        
        # Find crossovers
        for i in range(1, len(df)):
            if i < EMA_PERIOD:
                continue
                
            prev_close = df['close'].iloc[i-1]
            curr_close = df['close'].iloc[i]
            prev_ema = df[f'ema_{EMA_PERIOD}'].iloc[i-1]
            curr_ema = df[f'ema_{EMA_PERIOD}'].iloc[i]
            curr_volume = df['volume'].iloc[i]
            volume_avg = df['volume_avg'].iloc[i]
            
            # Bullish crossover with volume confirmation
            if (prev_close <= prev_ema and curr_close > curr_ema and 
                curr_volume > volume_avg * MIN_VOLUME_RATIO):
                df.loc[df.index[i], 'signal'] = 1
        
        return df
    
    def simulate_trading(self, btc_df: pd.DataFrame = None):
        """Run the backtest simulation"""
        capital = INITIAL_CAPITAL
        positions = {}  # symbol -> position dict
        equity_curve = []
        all_trades = []
        
        # Get all data first
        print(colored("\\n📊 Loading historical data...", "cyan"))
        print(colored("="*80, "blue"))
        data = {}
        for symbol in symbols:
            df = self.get_historical_data(symbol)
            if not df.empty:
                df = self.calculate_indicators(df)
                data[symbol] = df
                # Print data range for each symbol
                oldest_date = df['timestamp'].min().strftime('%Y-%m-%d')
                newest_date = df['timestamp'].max().strftime('%Y-%m-%d')
                total_days = (df['timestamp'].max() - df['timestamp'].min()).days
                print(colored(f"{symbol:12} | From: {oldest_date} | To: {newest_date} | Days: {total_days:4d} | Candles: {len(df):4d}", "white"))
            else:
                print(colored(f"{symbol:12} | ❌ No data available", "yellow"))
        print(colored("="*80, "blue"))
        
        # Get BTC data for filtering
        if BTC_FILTER and "BTCUSDC" in data:
            btc_df = data["BTCUSDC"]
        
        # Find common date range
        min_date = max(df['timestamp'].min() for df in data.values() if not df.empty)
        max_date = min(df['timestamp'].max() for df in data.values() if not df.empty)
        
        print(colored(f"\\n🔄 Simulating from {min_date.date()} to {max_date.date()}", "green"))
        print(colored(f"DEBUG: Date range check - min_date type: {type(min_date)}, max_date type: {type(max_date)}", "yellow"))
        
        # Simulate each day
        current_date = min_date
        while current_date <= max_date:
            daily_equity = capital
            
            # Calculate current equity
            for symbol, pos in positions.items():
                if pos["is_open"]:
                    df = data[symbol]
                    current_row = df[df['timestamp'].dt.date == current_date.date()]
                    if not current_row.empty:
                        current_price = current_row['close'].iloc[0]
                        pos["current_price"] = current_price
                        pos["highest_price"] = max(pos["highest_price"], current_price)
                        unrealized_pnl = (current_price - pos["entry_price"]) * pos["quantity"]
                        daily_equity += unrealized_pnl
            
            equity_curve.append({
                "date": current_date,
                "equity": daily_equity,
                "capital": capital,
                "positions": len([p for p in positions.values() if p.get("is_open", False)])
            })
            
            # Check for exits first
            for symbol in list(positions.keys()):
                if not positions[symbol]["is_open"]:
                    continue
                
                df = data[symbol]
                current_row = df[df['timestamp'].dt.date == current_date.date()]
                if current_row.empty:
                    continue
                
                current_price = current_row['close'].iloc[0]
                high_price = current_row['high'].iloc[0]
                low_price = current_row['low'].iloc[0]
                pos = positions[symbol]
                
                exit_price = None
                exit_reason = None
                
                # Check stop loss (use low price for more realistic simulation)
                if low_price <= pos["stop_loss"]:
                    exit_price = pos["stop_loss"]
                    exit_reason = "stop_loss"
                
                # Check trailing stop
                elif pos.get("trailing_stop") and low_price <= pos["trailing_stop"]:
                    exit_price = pos["trailing_stop"]
                    exit_reason = "trailing_stop"
                
                # Update highest price and trailing stop
                if current_price > pos.get("highest_price", pos["entry_price"]):
                    pos["highest_price"] = current_price
                    profit_percent = ((current_price - pos["entry_price"]) / pos["entry_price"]) * 100
                    if profit_percent >= TRAILING_STOP_ACTIVATION:
                        new_trailing = current_price * (1 - TRAILING_STOP_DISTANCE / 100)
                        if not pos.get("trailing_stop") or new_trailing > pos["trailing_stop"]:
                            pos["trailing_stop"] = new_trailing
                
                # Execute exit if needed (full exit only with trailing strategy)
                if exit_price and exit_reason:
                    sell_quantity = pos["quantity"]
                    gross_pnl = (exit_price - pos["entry_price"]) * sell_quantity
                    fees = (pos["entry_price"] * sell_quantity + exit_price * sell_quantity) * TRADING_FEE
                    net_pnl = gross_pnl - fees
                    
                    # Return the original position value plus/minus the P&L
                    capital += FIXED_POSITION_SIZE + net_pnl
                    
                    # Debug print
                    print(f"DEBUG: {symbol} | Entry: {pos['entry_price']:.4f} | Exit: {exit_price:.4f} | Qty: {sell_quantity:.6f} | P&L: ${net_pnl:.2f}")
                    
                    # Record max profit that was available
                    max_profit_percent = ((pos.get("highest_price", pos["entry_price"]) - pos["entry_price"]) / pos["entry_price"]) * 100
                    
                    trade_record = {
                        "symbol": symbol,
                        "entry_date": pos["entry_date"],
                        "exit_date": current_date,
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "quantity": sell_quantity,
                        "exit_reason": exit_reason,
                        "gross_pnl": gross_pnl,
                        "fees": fees,
                        "net_pnl": net_pnl,
                        "pnl_percent": ((exit_price - pos["entry_price"]) / pos["entry_price"]) * 100,
                        "holding_days": (current_date - pos["entry_date"]).days,
                        "max_profit_percent": max_profit_percent,
                        "position_value": FIXED_POSITION_SIZE
                    }
                    all_trades.append(trade_record)
                    
                    pos["is_open"] = False
            
            # Check for new entries
            open_positions = len([p for p in positions.values() if p.get("is_open", False)])
            
            if open_positions < MAX_POSITIONS:
                for symbol in symbols:
                    if symbol in positions and positions[symbol].get("is_open", False):
                        continue
                    
                    if open_positions >= MAX_POSITIONS:
                        break
                    
                    df = data[symbol]
                    current_row = df[df['timestamp'].dt.date == current_date.date()]
                    if current_row.empty or current_row['signal'].iloc[0] != 1:
                        continue
                    
                    # Check BTC filter
                    if BTC_FILTER and symbol != "BTCUSDC" and btc_df is not None:
                        btc_row = btc_df[btc_df['timestamp'].dt.date == current_date.date()]
                        if not btc_row.empty:
                            btc_price = btc_row['close'].iloc[0]
                            btc_ema = btc_row[f'ema_{EMA_PERIOD}'].iloc[0]
                            if btc_price < btc_ema:
                                continue
                    
                    # Open position
                    entry_price = current_row['close'].iloc[0]
                    position_value = FIXED_POSITION_SIZE
                    
                    if capital >= position_value:
                        quantity = position_value / entry_price
                        capital -= position_value
                        
                        positions[symbol] = {
                            "is_open": True,
                            "entry_date": current_date,
                            "entry_price": entry_price,
                            "quantity": quantity,
                            "stop_loss": entry_price * (1 - STOP_LOSS_PERCENT / 100),
                            "highest_price": entry_price,
                            "trailing_stop": None,
                            "current_price": entry_price
                        }
                        
                        open_positions += 1
            
            current_date += datetime.timedelta(days=1)
        
        # Close any remaining positions at market price
        for symbol, pos in positions.items():
            if pos.get("is_open", False):
                df = data[symbol]
                final_price = df['close'].iloc[-1]
                sell_quantity = pos["quantity"]
                gross_pnl = (final_price - pos["entry_price"]) * sell_quantity
                fees = (pos["entry_price"] * sell_quantity + final_price * sell_quantity) * TRADING_FEE
                net_pnl = gross_pnl - fees
                
                trade_record = {
                    "symbol": symbol,
                    "entry_date": pos["entry_date"],
                    "exit_date": df['timestamp'].iloc[-1],
                    "entry_price": pos["entry_price"],
                    "exit_price": final_price,
                    "quantity": sell_quantity,
                    "exit_reason": "end_of_backtest",
                    "gross_pnl": gross_pnl,
                    "fees": fees,
                    "net_pnl": net_pnl,
                    "pnl_percent": ((final_price - pos["entry_price"]) / pos["entry_price"]) * 100,
                    "holding_days": (df['timestamp'].iloc[-1] - pos["entry_date"]).days,
                    "max_profit_percent": ((pos.get("highest_price", pos["entry_price"]) - pos["entry_price"]) / pos["entry_price"]) * 100,
                    "position_value": FIXED_POSITION_SIZE
                }
                all_trades.append(trade_record)
        
        self.results["trades"] = all_trades
        self.results["equity_curve"] = equity_curve
        
        # Calculate statistics
        self.calculate_statistics()
    
    def calculate_statistics(self):
        """Calculate performance statistics"""
        trades = self.results["trades"]
        equity_curve = self.results["equity_curve"]
        
        if not trades:
            print(colored("⚠️  No trades executed", "yellow"))
            return
        
        # Overall statistics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t["net_pnl"] > 0]
        losing_trades = [t for t in trades if t["net_pnl"] <= 0]
        
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        
        total_pnl = sum(t["net_pnl"] for t in trades)
        total_fees = sum(t["fees"] for t in trades)
        
        avg_win = np.mean([t["net_pnl"] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t["net_pnl"] for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(sum(t["net_pnl"] for t in winning_trades) / sum(t["net_pnl"] for t in losing_trades)) if losing_trades else float('inf')
        
        # Calculate max drawdown
        equity_values = [e["equity"] for e in equity_curve]
        peak = equity_values[0]
        max_drawdown = 0
        
        for value in equity_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate Sharpe ratio (assuming daily returns)
        if len(equity_values) > 1:
            returns = pd.Series(equity_values).pct_change().dropna()
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(365) if returns.std() != 0 else 0
        else:
            sharpe_ratio = 0
        
        # Per-symbol statistics
        symbol_stats = {}
        for symbol in symbols:
            symbol_trades = [t for t in trades if t["symbol"] == symbol]
            if symbol_trades:
                symbol_stats[symbol] = {
                    "trades": len(symbol_trades),
                    "win_rate": (len([t for t in symbol_trades if t["net_pnl"] > 0]) / len(symbol_trades)) * 100,
                    "total_pnl": sum(t["net_pnl"] for t in symbol_trades),
                    "avg_pnl": np.mean([t["net_pnl"] for t in symbol_trades]),
                    "best_trade": max(t["net_pnl"] for t in symbol_trades),
                    "worst_trade": min(t["net_pnl"] for t in symbol_trades),
                    "avg_holding_days": np.mean([t["holding_days"] for t in symbol_trades])
                }
        
        self.results["statistics"] = {
            "initial_capital": INITIAL_CAPITAL,
            "final_capital": equity_values[-1] if equity_values else INITIAL_CAPITAL,
            "total_return": ((equity_values[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100 if equity_values else 0,
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "total_fees": total_fees,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "avg_holding_days": np.mean([t["holding_days"] for t in trades]) if trades else 0
        }
        
        self.results["symbol_performance"] = symbol_stats
    
    def print_results(self):
        """Print backtest results"""
        stats = self.results["statistics"]
        trades = self.results["trades"]
        
        print(colored("\\n" + "="*80, "blue"))
        print(colored("📊 BACKTEST RESULTS - Daily EMA Crossover Strategy", "cyan", attrs=['bold']))
        print(colored("="*80, "blue"))
        
        print(colored(f"\\nStrategy Parameters:", "yellow"))
        print(f"  EMA Period: {EMA_PERIOD}")
        print(f"  Stop Loss: {STOP_LOSS_PERCENT}%")
        print(f"  Trailing Stop: Activate at {TRAILING_STOP_ACTIVATION}%, trail by {TRAILING_STOP_DISTANCE}%")
        print(f"  Position Size: Fixed ${FIXED_POSITION_SIZE:.2f} per trade")
        print(f"  Max Positions: {MAX_POSITIONS}")
        
        print(colored(f"\\nPerformance Summary:", "yellow"))
        print(f"  Initial Capital: ${stats['initial_capital']:,.2f}")
        print(f"  Final Capital: ${stats['final_capital']:,.2f}")
        color = "green" if stats['total_return'] > 0 else "red"
        print(colored(f"  Total Return: {stats['total_return']:.2f}%", color))
        print(f"  Total P&L: ${stats['total_pnl']:,.2f}")
        print(f"  Total Fees Paid: ${stats['total_fees']:,.2f}")
        
        print(colored(f"\\nTrading Statistics:", "yellow"))
        print(f"  Total Trades: {stats['total_trades']}")
        print(f"  Winning Trades: {stats['winning_trades']}")
        print(f"  Losing Trades: {stats['losing_trades']}")
        print(colored(f"  Win Rate: {stats['win_rate']:.2f}%", "green" if stats['win_rate'] > 50 else "red"))
        print(f"  Average Win: ${stats['avg_win']:,.2f}")
        print(f"  Average Loss: ${stats['avg_loss']:,.2f}")
        print(f"  Profit Factor: {stats['profit_factor']:.2f}")
        print(f"  Max Drawdown: {stats['max_drawdown']:.2f}%")
        print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        print(f"  Avg Holding Days: {stats['avg_holding_days']:.1f}")
        
        # Detailed trade-by-trade report
        print(colored("\\n" + "="*80, "blue"))
        print(colored("DETAILED TRADE REPORT", "cyan", attrs=['bold']))
        print(colored("="*80, "blue"))
        print(colored(f"{'#':>3} | {'Symbol':^10} | {'Entry Date':^12} | {'Exit Date':^12} | {'Days':^5} | {'Entry $':^8} | {'Exit $':^8} | {'P&L $':^8} | {'P&L %':^7} | {'Max %':^7} | {'Reason':^12}", "white"))
        print(colored("-"*140, "blue"))
        
        for i, trade in enumerate(trades, 1):
            entry_date = trade['entry_date'].strftime('%Y-%m-%d') if hasattr(trade['entry_date'], 'strftime') else str(trade['entry_date'])[:10]
            exit_date = trade['exit_date'].strftime('%Y-%m-%d') if hasattr(trade['exit_date'], 'strftime') else str(trade['exit_date'])[:10]
            
            color = "green" if trade["net_pnl"] > 0 else "red"
            max_profit = trade.get('max_profit_percent', 0)
            
            print(colored(
                f"{i:3d} | {trade['symbol']:^10} | {entry_date:^12} | {exit_date:^12} | {trade['holding_days']:^5d} | "
                f"{trade['entry_price']:>8.2f} | {trade['exit_price']:>8.2f} | {trade['net_pnl']:>8.2f} | "
                f"{trade['pnl_percent']:>7.2f} | {max_profit:>7.2f} | {trade['exit_reason']:^12}",
                color
            ))
        
        # Symbol summary
        print(colored("\\n" + "="*80, "blue"))
        print(colored("SYMBOL PERFORMANCE SUMMARY", "cyan", attrs=['bold']))
        print(colored("="*80, "blue"))
        symbol_perf = self.results["symbol_performance"]
        sorted_symbols = sorted(symbol_perf.items(), key=lambda x: x[1]["total_pnl"], reverse=True)
        
        print(colored(f"{'Symbol':^10} | {'Trades':^7} | {'Wins':^5} | {'Losses':^6} | {'Win %':^6} | {'Total P&L':^10} | {'Avg P&L':^8} | {'Best':^8} | {'Worst':^8}", "white"))
        print(colored("-"*90, "blue"))
        
        for symbol, perf in sorted_symbols:
            if perf['trades'] > 0:
                wins = int(perf['trades'] * perf['win_rate'] / 100)
                losses = perf['trades'] - wins
                color = "green" if perf["total_pnl"] > 0 else "red"
                print(colored(
                    f"{symbol:^10} | {perf['trades']:^7d} | {wins:^5d} | {losses:^6d} | {perf['win_rate']:>6.1f} | "
                    f"{perf['total_pnl']:>10.2f} | {perf['avg_pnl']:>8.2f} | {perf['best_trade']:>8.2f} | {perf['worst_trade']:>8.2f}",
                    color
                ))
        
        print(colored("\\n" + "="*80, "blue"))
    
    def save_results(self):
        """Save results to JSON file"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_results_daily_ema_{timestamp}.json"
        
        # Convert datetime objects to strings
        results_copy = self.results.copy()
        for trade in results_copy["trades"]:
            trade["entry_date"] = trade["entry_date"].isoformat() if isinstance(trade["entry_date"], datetime.datetime) else str(trade["entry_date"])
            trade["exit_date"] = trade["exit_date"].isoformat() if isinstance(trade["exit_date"], datetime.datetime) else str(trade["exit_date"])
        
        for equity in results_copy["equity_curve"]:
            equity["date"] = equity["date"].isoformat() if isinstance(equity["date"], datetime.datetime) else str(equity["date"])
        
        with open(filename, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        print(colored(f"\\n📁 Results saved to {filename}", "green"))
        
        return filename
    
    # Removed plot_results function - no graphs needed

def main():
    print(colored("\\n🚀 Starting Daily EMA Crossover Backtest", "cyan", attrs=['bold']))
    print(colored(f"Testing {len(symbols)} symbols over {LOOKBACK_DAYS} days", "white"))
    
    backtest = DailyEMABacktest()
    backtest.simulate_trading()
    backtest.print_results()
    # backtest.save_results()  # Optional - uncomment if you want JSON file
    
    print(colored("\\n✅ Backtest completed!", "green", attrs=['bold']))

if __name__ == "__main__":
    main()