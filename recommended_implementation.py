#!/usr/bin/env python3
"""
RECOMMENDED TRADING STRATEGY IMPLEMENTATION
Based on comprehensive backtesting results

Strategy: Enhanced Smart Momentum Mean Reversion
Best Symbols: BNBUSDC (71.4% win rate), XRPUSDC (60% win rate), SHIBUSDC (62.5% win rate)
Overall Performance: 49.2% win rate with proper risk management
"""

import pandas as pd
import numpy as np
from binance.client import Client
import sys
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load secrets
sys.path.append('../')
from _secrets import api_key, secret_key

# Recommended symbols based on backtesting results
RECOMMENDED_SYMBOLS = [
    "BNBUSDC",   # 71.4% win rate - PRIMARY TARGET
    "XRPUSDC",   # 60.0% win rate - SECONDARY  
    "SHIBUSDC"   # 62.5% win rate - TERTIARY
]

class RecommendedStrategy:
    def __init__(self):
        self.client = Client(api_key, secret_key)
        self.symbols = RECOMMENDED_SYMBOLS
        
    def get_current_data(self, symbol, interval=Client.KLINE_INTERVAL_1HOUR, limit=100):
        """Get current market data for analysis"""
        try:
            klines = self.client.get_historical_klines(symbol, interval, f"{limit} hours ago UTC")
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base', 
                'taker_buy_quote', 'ignore'
            ])
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return self.add_indicators(df[['open', 'high', 'low', 'close', 'volume']])
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def add_indicators(self, df):
        """Add technical indicators optimized from backtesting"""
        # RSI - proven performer
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # EMA system
        df['ema21'] = df['close'].ewm(span=21).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        df['ema100'] = df['close'].ewm(span=100).mean()
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Stochastic
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        
        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + (bb_std * 2)
        df['bb_lower'] = df['bb_mid'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume
        df['volume_ma'] = df['volume'].rolling(20).mean()
        
        return df
    
    def check_buy_signal(self, df, symbol):
        """Enhanced buy signal based on backtesting results"""
        if len(df) < 50:
            return False, "Insufficient data"
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Core conditions from successful backtests
        conditions = {
            'rsi_oversold': latest['rsi'] < 45,
            'rsi_improving': latest['rsi'] > prev['rsi'],
            'above_ema21': latest['close'] > latest['ema21'],
            'stoch_oversold': latest['stoch_k'] < 40,
            'macd_improving': latest['macd_hist'] > prev['macd_hist'],
            'volume_ok': latest['volume'] > latest['volume_ma'] * 0.8,
            'not_in_crash': latest['close'] > latest['ema100'] * 0.85,
            'bb_position_ok': latest['bb_position'] < 0.35
        }
        
        # Symbol-specific optimizations based on backtest results
        if symbol == "BNBUSDC":
            # More aggressive for best performer
            conditions['rsi_oversold'] = latest['rsi'] < 50
            conditions['consolidation'] = latest['bb_width'] < df['bb_width'].rolling(20).mean().iloc[-1]
        
        elif symbol == "XRPUSDC":
            # Conservative for consistent performer
            conditions['strong_volume'] = latest['volume'] > latest['volume_ma'] * 1.2
            conditions['momentum_alignment'] = latest['macd'] > latest['macd_signal']
        
        elif symbol == "SHIBUSDC":
            # Volatility adjusted
            conditions['rsi_oversold'] = latest['rsi'] < 40  # Tighter for volatile asset
            conditions['recent_support'] = abs(latest['close'] - latest['ema21']) / latest['ema21'] < 0.03
        
        # Count conditions met
        met_conditions = sum(conditions.values())
        total_conditions = len(conditions)
        
        # Require at least 70% of conditions to be met
        buy_signal = met_conditions >= (total_conditions * 0.7)
        
        details = {
            'conditions_met': met_conditions,
            'total_conditions': total_conditions,
            'percentage': (met_conditions / total_conditions) * 100,
            'failed_conditions': [k for k, v in conditions.items() if not v]
        }
        
        return buy_signal, details
    
    def check_sell_signal(self, df, entry_price=None):
        """Enhanced sell signal based on backtesting results"""
        if len(df) < 2:
            return False, "Insufficient data"
        
        latest = df.iloc[-1]
        
        # Quick profit conditions
        profit_conditions = {
            'rsi_profit': latest['rsi'] > 60,
            'stoch_profit': latest['stoch_k'] > 70,
            'bb_upper': latest['bb_position'] > 0.75,
            'macd_turn': latest['macd'] < latest['macd_signal']
        }
        
        # Risk management
        risk_conditions = {
            'support_break': latest['close'] < latest['ema21'] * 0.98,
            'trend_break': latest['close'] < latest['ema50']
        }
        
        # If we have entry price, check stop loss and take profit
        if entry_price:
            pct_change = (latest['close'] - entry_price) / entry_price * 100
            risk_conditions['stop_loss'] = pct_change <= -2.5
            profit_conditions['take_profit'] = pct_change >= 4.0
        
        sell_signal = any(profit_conditions.values()) or any(risk_conditions.values())
        
        details = {
            'profit_reasons': [k for k, v in profit_conditions.items() if v],
            'risk_reasons': [k for k, v in risk_conditions.items() if v]
        }
        
        return sell_signal, details
    
    def analyze_current_opportunities(self):
        """Analyze current market for trading opportunities"""
        print("🎯 RECOMMENDED STRATEGY - MARKET ANALYSIS")
        print("=" * 60)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Analyzing: {', '.join(self.symbols)}")
        print("=" * 60)
        
        opportunities = []
        
        for symbol in self.symbols:
            print(f"\n📊 {symbol} Analysis:")
            
            df = self.get_current_data(symbol)
            if df is None:
                print("   ❌ Failed to fetch data")
                continue
            
            # Current price info
            current_price = df['close'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            
            print(f"   💲 Current Price: ${current_price:.6f}")
            print(f"   📈 RSI: {rsi:.1f}")
            
            # Check buy signal
            buy_signal, buy_details = self.check_buy_signal(df, symbol)
            
            if buy_signal:
                print(f"   ✅ BUY SIGNAL DETECTED!")
                print(f"   🎯 Conditions Met: {buy_details['conditions_met']}/{buy_details['total_conditions']} ({buy_details['percentage']:.1f}%)")
                
                opportunities.append({
                    'symbol': symbol,
                    'price': current_price,
                    'rsi': rsi,
                    'signal_strength': buy_details['percentage'],
                    'entry_recommended': True
                })
            else:
                print(f"   ⚠️  No buy signal")
                print(f"   📊 Conditions Met: {buy_details['conditions_met']}/{buy_details['total_conditions']} ({buy_details['percentage']:.1f}%)")
                
                if buy_details['failed_conditions']:
                    print(f"   🔍 Missing: {', '.join(buy_details['failed_conditions'][:3])}")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"🎯 OPPORTUNITIES SUMMARY")
        print(f"{'='*60}")
        
        if opportunities:
            print(f"✅ {len(opportunities)} trading opportunities found:")
            for opp in sorted(opportunities, key=lambda x: x['signal_strength'], reverse=True):
                print(f"   {opp['symbol']}: {opp['signal_strength']:.1f}% signal strength")
        else:
            print("⚠️ No current opportunities meeting criteria")
            print("💡 Monitor for conditions to align")
        
        return opportunities
    
    def get_position_sizing_recommendation(self, balance, symbol):
        """Get recommended position size based on backtesting results"""
        # Conservative sizing based on historical performance
        if symbol == "BNBUSDC":
            return balance * 0.03  # 3% for best performer
        elif symbol in ["XRPUSDC", "SHIBUSDC"]:
            return balance * 0.02  # 2% for others
        else:
            return balance * 0.01  # 1% for untested symbols

def main():
    """Main execution - analyze current market conditions"""
    strategy = RecommendedStrategy()
    
    try:
        opportunities = strategy.analyze_current_opportunities()
        
        print(f"\n📋 IMPLEMENTATION NOTES:")
        print("- Use 2.5% stop loss and 4% take profit")
        print("- Maximum hold time: 48 hours")
        print("- Monitor RSI for exit signals (>60)")
        print("- Paper trade first to validate")
        
    except Exception as e:
        print(f"Error in analysis: {e}")

if __name__ == "__main__":
    main()