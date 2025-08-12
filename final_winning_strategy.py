import pandas as pd
import numpy as np
from binance.client import Client
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Load secrets
sys.path.append('../')
from _secrets import api_key, secret_key

client = Client(api_key, secret_key)

SYMBOLS = ["BTCUSDC", "ETHUSDC", "BNBUSDC", "SOLUSDC", "XRPUSDC", "DOGEUSDC", "ADAUSDC", "SHIBUSDC", "LINKUSDC"]

class FinalWinningStrategy:
    def __init__(self, symbols):
        self.symbols = symbols
        self.client = client
        
    def get_data(self, symbol, days_back=90):
        """Get 1-hour data for 3 months - more granular"""
        try:
            start_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
            klines = self.client.get_historical_klines(
                symbol, Client.KLINE_INTERVAL_1HOUR, start_time
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base', 
                'taker_buy_quote', 'ignore'
            ])
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            print(f"Error with {symbol}: {e}")
            return None
    
    def add_indicators(self, df):
        """Add comprehensive indicators"""
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Multiple EMAs
        df['ema9'] = df['close'].ewm(span=9).mean()
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
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Williams %R
        df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14)
        
        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + (bb_std * 2)
        df['bb_lower'] = df['bb_mid'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Price action
        df['body'] = abs(df['close'] - df['open'])
        df['range'] = df['high'] - df['low']
        df['upper_wick'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_wick'] = np.minimum(df['open'], df['close']) - df['low']
        
        # Patterns
        df['green'] = df['close'] > df['open']
        df['red'] = df['close'] < df['open']
        df['doji'] = df['body'] < df['range'] * 0.1
        df['hammer'] = (df['lower_wick'] > df['body'] * 2) & (df['upper_wick'] < df['body'] * 0.3)
        df['shooting_star'] = (df['upper_wick'] > df['body'] * 2) & (df['lower_wick'] < df['body'] * 0.3)
        
        return df
    
    def winning_strategy(self, df):
        """The final winning strategy - Smart Momentum Mean Reversion"""
        signals = pd.DataFrame(index=df.index, columns=['signal'])
        signals['signal'] = 0
        
        # Core setup: Price in consolidation near key support
        consolidation = df['bb_width'] < df['bb_width'].rolling(50).quantile(0.4)
        
        # Multiple oversold confirmations (more lenient)
        rsi_oversold = df['rsi'] < 45  # More lenient RSI
        stoch_oversold = df['stoch_k'] < 40  # More lenient Stochastic
        williams_oversold = df['williams_r'] < -60  # More lenient Williams
        
        # Support confluence
        near_ema21 = abs(df['close'] - df['ema21']) / df['ema21'] < 0.03
        near_bb_lower = df['bb_position'] < 0.35
        
        # Momentum divergence (price making lower lows, indicators improving)
        price_declining = df['close'] < df['close'].rolling(5).mean()
        rsi_improving = df['rsi'] > df['rsi'].shift(1)
        macd_improving = df['macd_hist'] > df['macd_hist'].shift(1)
        
        # Volume confirmation (not too strict)
        volume_ok = df['volume_ratio'] > 0.8  # Just above average volume
        
        # Pattern confirmation
        bullish_pattern = df['hammer'] | df['doji'] | (df['green'] & (df['body'] > df['range'] * 0.6))
        
        # Long-term trend filter (very lenient)
        not_in_crash = df['close'] > df['ema100'] * 0.85  # Allow even in mild downtrend
        
        # BUY CONDITIONS (Multiple combinations for higher probability)
        buy_setup_1 = (
            consolidation & 
            rsi_oversold & 
            near_ema21 & 
            rsi_improving & 
            volume_ok &
            not_in_crash
        )
        
        buy_setup_2 = (
            stoch_oversold & 
            williams_oversold & 
            near_bb_lower & 
            macd_improving & 
            bullish_pattern &
            not_in_crash
        )
        
        buy_setup_3 = (
            price_declining & 
            (df['rsi'] < 40) & 
            (df['stoch_k'] < 35) &
            rsi_improving & 
            macd_improving &
            (df['close'] > df['ema50']) &  # Still in uptrend
            volume_ok
        )
        
        # Combine all setups
        buy_condition = buy_setup_1 | buy_setup_2 | buy_setup_3
        
        # SELL CONDITIONS (Take profits at reasonable levels)
        sell_condition = (
            (df['rsi'] > 60) |  # Not too greedy
            (df['stoch_k'] > 70) |
            (df['bb_position'] > 0.75) |
            (df['macd'] < df['macd_signal']) |  # Momentum turning down
            (df['close'] < df['ema21'] * 0.98)  # Support break
        )
        
        signals.loc[buy_condition, 'signal'] = 1
        signals.loc[sell_condition, 'signal'] = -1
        
        return signals
    
    def backtest_realistic(self, df, signals, symbol):
        """Realistic backtesting with tight risk management"""
        balance = 1000
        position = 0
        entry_price = 0
        trades = []
        max_hold_periods = 48  # Maximum 48 hours (2 days)
        periods_in_trade = 0
        
        for i in range(1, len(signals)):
            current_price = df['close'].iloc[i]
            
            # Risk management if in position
            if position > 0:
                periods_in_trade += 1
                pct_change = (current_price - entry_price) / entry_price * 100
                
                # Tight stop loss
                if pct_change <= -2.5:
                    balance = position * current_price * 0.9995  # Include slippage
                    trades.append(pct_change)
                    position = 0
                    periods_in_trade = 0
                    continue
                
                # Quick take profit
                elif pct_change >= 4:
                    balance = position * current_price * 0.9995
                    trades.append(pct_change)
                    position = 0
                    periods_in_trade = 0
                    continue
                
                # Maximum hold time
                elif periods_in_trade >= max_hold_periods:
                    balance = position * current_price * 0.9995
                    trades.append(pct_change)
                    position = 0
                    periods_in_trade = 0
                    continue
            
            signal = signals['signal'].iloc[i]
            
            # Buy signal
            if signal == 1 and position == 0:
                position = balance * 0.9995 / current_price  # 0.05% fee
                entry_price = current_price
                periods_in_trade = 0
                
            # Sell signal
            elif signal == -1 and position > 0:
                balance = position * current_price * 0.9995
                pct_change = (current_price - entry_price) / entry_price * 100
                trades.append(pct_change)
                position = 0
                periods_in_trade = 0
        
        # Close final position if any
        if position > 0:
            balance = position * df['close'].iloc[-1] * 0.9995
            pct_change = (df['close'].iloc[-1] - entry_price) / entry_price * 100
            trades.append(pct_change)
        
        if not trades:
            return None
        
        wins = len([t for t in trades if t > 0])
        losses = len([t for t in trades if t <= 0])
        win_rate = wins / len(trades) * 100
        
        avg_win = np.mean([t for t in trades if t > 0]) if wins > 0 else 0
        avg_loss = np.mean([t for t in trades if t <= 0]) if losses > 0 else 0
        
        profit_factor = abs(avg_win * wins / (avg_loss * losses)) if avg_loss != 0 else float('inf')
        
        return {
            'symbol': symbol,
            'trades': len(trades),
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_return': (balance - 1000) / 1000 * 100,
            'best_trade': max(trades),
            'worst_trade': min(trades),
            'all_trades': trades
        }
    
    def run_final_test(self):
        """Run the final comprehensive test"""
        print("🎯 FINAL WINNING STRATEGY TEST")
        print("=" * 50)
        print("Strategy: Smart Momentum Mean Reversion")
        print("Risk Management: 2.5% stop loss, 4% take profit, 48h max hold")
        print(f"Testing {len(self.symbols)} symbols with 1-hour data")
        print("=" * 50)
        
        results = []
        
        for symbol in self.symbols:
            print(f"\n📊 {symbol}")
            
            df = self.get_data(symbol)
            if df is None or len(df) < 200:
                print("   ❌ Insufficient data")
                continue
            
            df = self.add_indicators(df)
            signals = self.winning_strategy(df)
            result = self.backtest_realistic(df, signals, symbol)
            
            if result and result['trades'] >= 5:
                results.append(result)
                print(f"   ✅ {result['trades']} trades")
                print(f"   🎯 Win Rate: {result['win_rate']:.1f}%")
                print(f"   💰 Total Return: {result['total_return']:.1f}%")
                print(f"   🔥 Profit Factor: {result['profit_factor']:.2f}")
                print(f"   📈 Best: +{result['best_trade']:.1f}% | Worst: {result['worst_trade']:.1f}%")
            else:
                print("   ⚠️ Insufficient trades generated")
        
        if not results:
            print("\n❌ No viable results generated")
            return None
        
        # Calculate overall performance
        total_trades = sum(r['trades'] for r in results)
        total_wins = sum(r['wins'] for r in results)
        overall_win_rate = total_wins / total_trades * 100
        
        avg_return = np.mean([r['total_return'] for r in results])
        avg_profit_factor = np.mean([r['profit_factor'] for r in results if r['profit_factor'] != float('inf')])
        
        print(f"\n{'='*50}")
        print(f"🏆 OVERALL RESULTS")
        print(f"{'='*50}")
        print(f"📊 Total Trades: {total_trades}")
        print(f"🎯 Overall Win Rate: {overall_win_rate:.1f}%")
        print(f"💰 Average Return per Symbol: {avg_return:.1f}%")
        print(f"🔥 Average Profit Factor: {avg_profit_factor:.2f}")
        
        if overall_win_rate >= 70:
            print(f"\n🎉 SUCCESS! Win rate: {overall_win_rate:.1f}% (Target: 70%+)")
            
            # Save detailed results
            import json
            final_results = {
                'strategy_name': 'Smart Momentum Mean Reversion',
                'overall_win_rate': overall_win_rate,
                'average_return': avg_return,
                'average_profit_factor': avg_profit_factor,
                'total_trades': total_trades,
                'individual_results': results
            }
            
            with open('WINNING_STRATEGY_FINAL.json', 'w') as f:
                json.dump(final_results, f, indent=2, default=str)
            
            print(f"💾 Results saved to WINNING_STRATEGY_FINAL.json")
            return final_results
        else:
            print(f"\n⚠️ Win rate {overall_win_rate:.1f}% below 70% target")
            
        return {
            'overall_win_rate': overall_win_rate,
            'results': results
        }

def main():
    strategy_tester = FinalWinningStrategy(SYMBOLS)
    result = strategy_tester.run_final_test()
    
    if result and result.get('overall_win_rate', 0) >= 70:
        print(f"\n🌟 WINNING STRATEGY FOUND!")
        print(f"📈 Implementation ready for live trading")
    else:
        print(f"\n🔄 Strategy optimization needed")

if __name__ == "__main__":
    main()