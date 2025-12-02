import os
import json
import datetime
import time
import signal
import sys
from termcolor import colored
import pandas as pd
import numpy as np
from binance.client import Client
from typing import Dict, Optional, Tuple
sys.path.append('../')
from _secrets import api_key, secret_key

# ==================== CONFIGURATION ====================
# TRADING MODE - IMPORTANT: Set to True to prevent real money losses!
DRY_RUN = True  # Set to False for LIVE trading with real money

# EXIT STRATEGY PARAMETERS (Trailing stop only strategy)
STOP_LOSS_PERCENT = 3.0         # Initial stop loss at -3%
TRAILING_STOP_ACTIVATION = 5.0  # Activate trailing stop after 5% profit
TRAILING_STOP_DISTANCE = 3.0    # Trail by 3% from highest (good for daily swings)

# ENTRY STRATEGY PARAMETERS
EMA_PERIOD = 99                 # Long-term EMA for trend
WAIT_AFTER_CLOSE = 60           # Seconds to wait after daily candle close
MIN_VOLUME_RATIO = 1.2          # Minimum volume vs 20-day average
BTC_FILTER = True               # Check BTC trend before alt entries

# RISK MANAGEMENT
MAX_POSITIONS = 3               # Maximum concurrent positions
FIXED_POSITION_SIZE = 1000.0    # Fixed $1000 per trade
MIN_USDC_BALANCE = 100.0        # Keep minimum balance

# ========================================================

# Trading symbols
symbols = [
    "BTCUSDC", "ETHUSDC", "BNBUSDC", "ADAUSDC", "XRPUSDC", 
    "DOGEUSDC", "SOLUSDC", "PNUTUSDC", "PEPEUSDC", "SHIBUSDC", 
    "XLMUSDC", "LINKUSDC", "IOTAUSDC", "ENAUSDC"
]

shutdown = False

def handle_exit(sig, frame):
    global shutdown
    print("\\n🔻 Daily EMA Trader shutting down...")
    shutdown = True

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

class DailyEMATrader:
    def __init__(self):
        self.client = Client(api_key, secret_key)
        self.mode_suffix = "_dry" if DRY_RUN else "_live"
        self.positions_dir = "daily_ema_positions"
        self.ensure_directories()
        self.positions = self.load_all_positions()
        self.last_check_time = {}
        self.btc_ema = None
        self.btc_price = None
        
    def ensure_directories(self):
        """Create necessary directories"""
        if not os.path.exists(self.positions_dir):
            os.makedirs(self.positions_dir)
            
    def load_all_positions(self) -> Dict:
        """Load all position states from files"""
        positions = {}
        for symbol in symbols:
            position_file = f"{self.positions_dir}/{symbol}{self.mode_suffix}.json"
            if os.path.exists(position_file):
                with open(position_file, 'r') as f:
                    positions[symbol] = json.load(f)
            else:
                positions[symbol] = self.create_empty_position()
        return positions
    
    def create_empty_position(self) -> Dict:
        """Create empty position structure"""
        return {
            "position_is_open": False,
            "entry_price": None,
            "quantity": None,
            "entry_time": None,
            "stop_loss": None,
            "highest_price": None,
            "trailing_stop": None,
            "total_realized_pnl": 0.0,
            "last_checked": None
        }
    
    def save_position(self, symbol: str):
        """Save position state to file"""
        position_file = f"{self.positions_dir}/{symbol}{self.mode_suffix}.json"
        with open(position_file, 'w') as f:
            json.dump(self.positions[symbol], f, indent=2, default=str)
    
    def get_daily_candles(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Get daily candles for analysis"""
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=Client.KLINE_INTERVAL_1DAY,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['close'] = df['close'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['open'] = df['open'].astype(float)
            df['volume'] = df['volume'].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Calculate EMA
            df[f'ema_{EMA_PERIOD}'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
            
            # Calculate volume average
            df['volume_avg'] = df['volume'].rolling(window=20).mean()
            
            return df
        except Exception as e:
            print(colored(f"❌ Error getting candles for {symbol}: {e}", "red"))
            return pd.DataFrame()
    
    def check_ema_crossover(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check for EMA crossover signal"""
        if len(df) < EMA_PERIOD + 1:
            return False, ""
        
        current_close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        current_ema = df[f'ema_{EMA_PERIOD}'].iloc[-1]
        prev_ema = df[f'ema_{EMA_PERIOD}'].iloc[-2]
        current_volume = df['volume'].iloc[-1]
        volume_avg = df['volume_avg'].iloc[-1]
        
        # Bullish crossover: price crosses above EMA
        if prev_close <= prev_ema and current_close > current_ema:
            # Check volume confirmation
            if current_volume > volume_avg * MIN_VOLUME_RATIO:
                return True, "bullish_crossover"
        
        return False, ""
    
    def check_btc_trend(self) -> bool:
        """Check if BTC is above its EMA (for alt trading)"""
        if not BTC_FILTER:
            return True
            
        try:
            df_btc = self.get_daily_candles("BTCUSDC", limit=100)
            if df_btc.empty:
                return True  # Allow trading if can't check
            
            self.btc_price = df_btc['close'].iloc[-1]
            self.btc_ema = df_btc[f'ema_{EMA_PERIOD}'].iloc[-1]
            
            return self.btc_price > self.btc_ema
        except:
            return True
    
    def calculate_position_size(self, symbol: str, entry_price: float) -> float:
        """Calculate position size for fixed $1000 investment"""
        try:
            # Fixed position size in USDC
            quantity = FIXED_POSITION_SIZE / entry_price
            
            # Get symbol info for precision
            info = self.client.get_symbol_info(symbol)
            for filter in info['filters']:
                if filter['filterType'] == 'LOT_SIZE':
                    step_size = float(filter['stepSize'])
                    precision = len(str(step_size).split('.')[-1].rstrip('0'))
                    quantity = round(quantity - (quantity % step_size), precision)
                    break
            
            return quantity
        except Exception as e:
            print(colored(f"❌ Error calculating position size: {e}", "red"))
            return 0
    
    def open_position(self, symbol: str, signal_type: str) -> bool:
        """Open a new position"""
        try:
            # Check if we already have a position
            if self.positions[symbol]["position_is_open"]:
                return False
            
            # Check total open positions
            open_count = sum(1 for s in symbols if self.positions[s]["position_is_open"])
            if open_count >= MAX_POSITIONS:
                print(colored(f"⚠️  Max positions ({MAX_POSITIONS}) reached", "yellow"))
                return False
            
            # Get current price
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            entry_price = float(ticker['price'])
            
            # Calculate position size
            quantity = self.calculate_position_size(symbol, entry_price)
            if quantity == 0:
                return False
            
            # Execute buy order
            if DRY_RUN:
                print(colored(f"🧪 DRY RUN: Would buy {quantity:.4f} {symbol} at {entry_price:.4f}", "cyan"))
                order_id = f"DRY_{int(time.time())}"
                executed_qty = quantity
                executed_price = entry_price
            else:
                order = self.client.order_market_buy(symbol=symbol, quantity=quantity)
                order_id = order['orderId']
                executed_qty = float(order['executedQty'])
                executed_price = float(order['fills'][0]['price']) if order['fills'] else entry_price
            
            # Calculate exit levels
            stop_loss = executed_price * (1 - STOP_LOSS_PERCENT / 100)
            
            # Update position
            self.positions[symbol] = {
                "position_is_open": True,
                "entry_price": executed_price,
                "quantity": executed_qty,
                "entry_time": datetime.datetime.now().isoformat(),
                "stop_loss": stop_loss,
                "highest_price": executed_price,
                "trailing_stop": None,
                "signal_type": signal_type,
                "order_id": order_id,
                "total_realized_pnl": self.positions[symbol].get("total_realized_pnl", 0.0),
                "last_checked": datetime.datetime.now().isoformat()
            }
            
            self.save_position(symbol)
            
            print(colored(f"✅ Opened {symbol} position:", "green"))
            print(colored(f"   Entry: {executed_price:.4f} | Qty: {executed_qty:.4f} | Value: ${executed_price * executed_qty:.2f}", "white"))
            print(colored(f"   SL: {stop_loss:.4f} | Trailing activates at +{TRAILING_STOP_ACTIVATION}%", "white"))
            
            return True
            
        except Exception as e:
            print(colored(f"❌ Error opening position for {symbol}: {e}", "red"))
            return False
    
    def check_exit_conditions(self, symbol: str) -> Tuple[bool, str]:
        """Check if position should be exited"""
        pos = self.positions[symbol]
        if not pos["position_is_open"]:
            return False, ""
        
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
            entry_price = pos["entry_price"]
            
            # Update highest price
            if current_price > pos["highest_price"]:
                pos["highest_price"] = current_price
                
                # Check trailing stop activation
                profit_percent = ((current_price - entry_price) / entry_price) * 100
                if profit_percent >= TRAILING_STOP_ACTIVATION:
                    new_trailing = current_price * (1 - TRAILING_STOP_DISTANCE / 100)
                    if not pos["trailing_stop"] or new_trailing > pos["trailing_stop"]:
                        pos["trailing_stop"] = new_trailing
                        print(colored(f"🎯 {symbol}: Trailing stop activated at {pos['trailing_stop']:.4f}", "cyan"))
            
            # Update trailing stop if already active
            elif pos["trailing_stop"]:
                new_trailing = pos["highest_price"] * (1 - TRAILING_STOP_DISTANCE / 100)
                if new_trailing > pos["trailing_stop"]:
                    pos["trailing_stop"] = new_trailing
            
            # Check stop loss
            if current_price <= pos["stop_loss"]:
                return True, "stop_loss"
            
            # Check trailing stop
            if pos["trailing_stop"] and current_price <= pos["trailing_stop"]:
                return True, "trailing_stop"
            
            return False, ""
            
        except Exception as e:
            print(colored(f"❌ Error checking exit for {symbol}: {e}", "red"))
            return False, ""
    
    def close_position(self, symbol: str, reason: str) -> bool:
        """Close position (full exit only with trailing strategy)"""
        pos = self.positions[symbol]
        if not pos["position_is_open"]:
            return False
        
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            exit_price = float(ticker['price'])
            
            # Sell full position
            sell_qty = pos["quantity"]
            
            # Get symbol precision
            info = self.client.get_symbol_info(symbol)
            for filter in info['filters']:
                if filter['filterType'] == 'LOT_SIZE':
                    step_size = float(filter['stepSize'])
                    precision = len(str(step_size).split('.')[-1].rstrip('0'))
                    sell_qty = round(sell_qty - (sell_qty % step_size), precision)
                    break
            
            # Execute sell order
            if DRY_RUN:
                print(colored(f"🧪 DRY RUN: Would sell {sell_qty:.4f} {symbol} at {exit_price:.4f}", "cyan"))
                executed_qty = sell_qty
                executed_price = exit_price
            else:
                order = self.client.order_market_sell(symbol=symbol, quantity=sell_qty)
                executed_qty = float(order['executedQty'])
                executed_price = float(order['fills'][0]['price']) if order['fills'] else exit_price
            
            # Calculate P&L
            entry_price = pos["entry_price"]
            gross_pnl = (executed_price - entry_price) * executed_qty
            fees = (entry_price * executed_qty + executed_price * executed_qty) * 0.001
            net_pnl = gross_pnl - fees
            pnl_percent = ((executed_price - entry_price) / entry_price) * 100
            
            # Update position
            pos["position_is_open"] = False
            pos["total_realized_pnl"] += net_pnl
            pos["last_checked"] = datetime.datetime.now().isoformat()
            
            self.save_position(symbol)
            
            print(colored(f"💰 Exit {symbol} ({reason}):", "green" if net_pnl > 0 else "red"))
            print(colored(f"   Exit: {executed_price:.4f} | P&L: ${net_pnl:.2f} ({pnl_percent:.2f}%) | Position was ${FIXED_POSITION_SIZE:.0f}", 
                         "green" if net_pnl > 0 else "red"))
            
            # Show max profit that was available
            if pos["highest_price"] and pos["highest_price"] > entry_price:
                max_profit_percent = ((pos["highest_price"] - entry_price) / entry_price) * 100
                print(colored(f"   Max profit reached: {max_profit_percent:.2f}%", "yellow"))
            
            return True
            
        except Exception as e:
            print(colored(f"❌ Error closing position for {symbol}: {e}", "red"))
            return False
    
    def is_new_daily_candle(self) -> bool:
        """Check if a new daily candle has just closed"""
        now = datetime.datetime.utcnow()
        
        # Daily candles close at 00:00 UTC
        # Check if we're within the first 5 minutes of a new day
        if now.hour == 0 and now.minute < 5:
            today = now.date()
            
            # Check if we haven't checked today yet
            if not hasattr(self, 'last_daily_check') or self.last_daily_check != today:
                # Wait specified seconds after candle close
                if now.minute * 60 + now.second >= WAIT_AFTER_CLOSE:
                    self.last_daily_check = today
                    return True
        
        return False
    
    def scan_for_entries(self):
        """Scan all symbols for entry signals"""
        if not self.is_new_daily_candle():
            return
        
        print(colored(f"\\n🔍 Scanning for daily EMA crossovers at {datetime.datetime.utcnow()}", "cyan"))
        
        # Check BTC trend first
        btc_trend_ok = self.check_btc_trend()
        if not btc_trend_ok and BTC_FILTER:
            print(colored(f"⚠️  BTC below EMA{EMA_PERIOD}, skipping alt entries", "yellow"))
        
        for symbol in symbols:
            if self.positions[symbol]["position_is_open"]:
                continue
            
            # Skip alts if BTC filter active and BTC is below EMA
            if symbol != "BTCUSDC" and not btc_trend_ok and BTC_FILTER:
                continue
            
            # Get daily candles
            df = self.get_daily_candles(symbol)
            if df.empty:
                continue
            
            # Check for crossover
            has_signal, signal_type = self.check_ema_crossover(df)
            
            if has_signal:
                print(colored(f"🎯 {symbol}: {signal_type} detected!", "green"))
                self.open_position(symbol, signal_type)
    
    def monitor_positions(self):
        """Monitor open positions for exit conditions"""
        for symbol in symbols:
            if not self.positions[symbol]["position_is_open"]:
                continue
            
            should_exit, reason = self.check_exit_conditions(symbol)
            if should_exit:
                self.close_position(symbol, reason)
    
    def print_portfolio_status(self):
        """Print current portfolio status"""
        open_positions = []
        total_pnl = 0
        
        for symbol in symbols:
            pos = self.positions[symbol]
            if pos["position_is_open"]:
                try:
                    ticker = self.client.get_symbol_ticker(symbol=symbol)
                    current_price = float(ticker['price'])
                    entry_price = pos["entry_price"]
                    pnl_percent = ((current_price - entry_price) / entry_price) * 100
                    open_positions.append({
                        "symbol": symbol,
                        "entry": entry_price,
                        "current": current_price,
                        "pnl_percent": pnl_percent
                    })
                except:
                    pass
            
            total_pnl += pos.get("total_realized_pnl", 0)
        
        # Print status
        print(colored(f"\\n{'='*60}", "blue"))
        print(colored(f"📈 Daily EMA Trader Status - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "cyan"))
        print(colored(f"Mode: {'DRY RUN' if DRY_RUN else 'LIVE TRADING'}", "yellow" if DRY_RUN else "red"))
        
        if open_positions:
            print(colored(f"\\nOpen Positions ({len(open_positions)}/{MAX_POSITIONS}):", "white"))
            for pos in open_positions:
                color = "green" if pos["pnl_percent"] > 0 else "red"
                print(colored(f"  {pos['symbol']}: Entry={pos['entry']:.4f} Current={pos['current']:.4f} P&L={pos['pnl_percent']:.2f}%", color))
        else:
            print(colored(f"\\nNo open positions", "white"))
        
        print(colored(f"\\nTotal Realized P&L: ${total_pnl:.2f}", "green" if total_pnl > 0 else "red"))
        
        if self.btc_price and self.btc_ema:
            btc_above = self.btc_price > self.btc_ema
            print(colored(f"BTC Status: ${self.btc_price:.2f} {'above' if btc_above else 'below'} EMA{EMA_PERIOD} (${self.btc_ema:.2f})", 
                         "green" if btc_above else "yellow"))
        
        print(colored(f"{'='*60}\\n", "blue"))
    
    def run(self):
        """Main trading loop"""
        print(colored(f"🚀 Daily EMA Trader Started - {'DRY RUN' if DRY_RUN else 'LIVE'} Mode", "cyan"))
        print(colored(f"Strategy: EMA{EMA_PERIOD} Crossover on Daily Candles", "white"))
        print(colored(f"Position Size: Fixed ${FIXED_POSITION_SIZE:.2f} per trade", "white"))
        print(colored(f"Exit: SL={STOP_LOSS_PERCENT}% | Trailing Stop: Activate at +{TRAILING_STOP_ACTIVATION}%, Trail by {TRAILING_STOP_DISTANCE}%", "white"))
        
        last_status_print = time.time()
        
        while not shutdown:
            try:
                # Scan for new entries (only at daily close)
                self.scan_for_entries()
                
                # Monitor open positions
                self.monitor_positions()
                
                # Print status every 30 minutes
                if time.time() - last_status_print > 1800:
                    self.print_portfolio_status()
                    last_status_print = time.time()
                
                # Sleep for 1 minute
                time.sleep(60)
                
            except Exception as e:
                print(colored(f"❌ Main loop error: {e}", "red"))
                time.sleep(60)
        
        print(colored("\\n✅ Daily EMA Trader stopped gracefully", "green"))
        self.print_portfolio_status()

def main():
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--check-pnl':
            trader = DailyEMATrader()
            total_pnl = sum(trader.positions[s].get("total_realized_pnl", 0) for s in symbols)
            print(colored(f"Total Realized P&L: ${total_pnl:.2f}", "green" if total_pnl > 0 else "red"))
            return
        elif sys.argv[1] == '--reset-pnl':
            confirm = input("Are you sure you want to reset all P&L tracking? (yes/no): ")
            if confirm.lower() == 'yes':
                trader = DailyEMATrader()
                for symbol in symbols:
                    trader.positions[symbol]["total_realized_pnl"] = 0
                    trader.save_position(symbol)
                print(colored("P&L tracking reset", "yellow"))
            return
    
    # Run the trader
    trader = DailyEMATrader()
    trader.run()

if __name__ == "__main__":
    main()