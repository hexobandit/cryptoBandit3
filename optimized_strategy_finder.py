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

# Focus on most liquid pairs
FOCUS_SYMBOLS = ["BTCUSDC", "ETHUSDC", "BNBUSDC", "SOLUSDC", "XRPUSDC"]

class OptimizedStrategyFinder:
    def __init__(self, symbols, initial_balance=1000, trade_fee=0.001):
        self.symbols = symbols
        self.initial_balance = initial_balance
        self.trade_fee = trade_fee
        self.client = client
        
    def get_data(self, symbol, interval=Client.KLINE_INTERVAL_4HOUR, days_back=180):
        """Get optimized dataset - 4H candles, 6 months"""
        try:
            start_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
            klines = self.client.get_historical_klines(symbol, interval, start_time)
            
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
            print(f"Error fetching {symbol}: {e}")
            return None
    
    def add_indicators(self, df):
        """Add essential indicators only"""
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # EMAs
        df['ema8'] = df['close'].ewm(span=8).mean()
        df['ema21'] = df['close'].ewm(span=21).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + (bb_std * 2)
        df['bb_lower'] = df['bb_mid'] - (bb_std * 2)
        df['bb_pos'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Simple patterns
        df['hammer'] = ((df['low'] < df['open']) & (df['low'] < df['close']) & 
                       ((df['open'] - df['low']) > 2 * abs(df['close'] - df['open'])))
        
        df['bullish_engulf'] = ((df['close'] > df['open']) & 
                               (df['close'].shift(1) < df['open'].shift(1)) & 
                               (df['close'] > df['open'].shift(1)) & 
                               (df['open'] < df['close'].shift(1)))
        
        return df
    
    def golden_cross_reversal_strategy(self, df):
        """Ultra-selective strategy focusing on high-probability setups"""
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        
        # Golden cross setup (EMA8 crossing above EMA21)
        golden_cross = (df['ema8'] > df['ema21']) & (df['ema8'].shift(1) <= df['ema21'].shift(1))
        
        # Multiple oversold confirmations
        oversold_multi = (
            (df['rsi'] < 25) &  # Very oversold RSI
            (df['bb_pos'] < 0.1) &  # Near lower BB
            (df['macd'] > df['macd'].shift(1))  # MACD turning up
        )
        
        # Above major trend (EMA50)
        major_trend_up = df['close'] > df['ema50']
        
        # Bullish pattern confirmation
        pattern_confirm = df['hammer'] | df['bullish_engulf']
        
        # Ultra-strict buy condition
        buy_condition = (
            golden_cross & 
            oversold_multi & 
            major_trend_up & 
            pattern_confirm
        )
        
        # Conservative exit - take profits early
        sell_condition = (
            (df['rsi'] > 55) |  # Early exit on RSI
            (df['ema8'] < df['ema21']) |  # EMA cross down
            (df['bb_pos'] > 0.8)  # Near upper BB
        )
        
        signals.loc[buy_condition, 'signal'] = 1
        signals.loc[sell_condition, 'signal'] = -1
        
        return signals
    
    def mean_reversion_premium_strategy(self, df):
        """Premium mean reversion with strict filters"""
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        
        # Extreme oversold but not in downtrend
        extreme_oversold = (
            (df['rsi'] < 20) &
            (df['bb_pos'] < 0.05) &
            (df['close'] > df['ema50'])  # Still above major MA
        )
        
        # MACD showing reversal signs
        macd_reversal = (
            (df['macd'] > df['macd'].shift(1)) &
            (df['macd'] < 0) &  # Still negative but turning
            (df['macd_signal'] < df['macd_signal'].shift(1))  # Signal line declining
        )
        
        # Hammer at key level
        support_hammer = df['hammer'] & (abs(df['close'] - df['ema50']) / df['ema50'] < 0.05)
        
        buy_condition = extreme_oversold & macd_reversal & support_hammer
        
        # Exit on mean reversion completion
        sell_condition = (
            (df['rsi'] > 50) |
            (df['bb_pos'] > 0.6) |
            (df['close'] < df['ema50'])
        )
        
        signals.loc[buy_condition, 'signal'] = 1
        signals.loc[sell_condition, 'signal'] = -1
        
        return signals
    
    def backtest_with_strict_risk_mgmt(self, df, signals, symbol):
        """Backtest with 3% stop loss and 6% take profit"""
        balance = self.initial_balance
        position = 0
        entry_price = 0
        trades = []
        
        for i in range(1, len(signals)):
            current_price = df['close'].iloc[i]
            signal = signals['signal'].iloc[i]
            
            # Risk management if in position
            if position > 0:
                pct_change = (current_price - entry_price) / entry_price * 100
                
                # 3% stop loss
                if pct_change <= -3:
                    sell_value = position * current_price * (1 - self.trade_fee)
                    profit_loss = sell_value - (entry_price * position)
                    
                    trades.append({
                        'entry': entry_price,
                        'exit': current_price,
                        'return': pct_change,
                        'reason': 'stop_loss'
                    })
                    
                    balance = sell_value
                    position = 0
                    continue
                
                # 6% take profit
                elif pct_change >= 6:
                    sell_value = position * current_price * (1 - self.trade_fee)
                    profit_loss = sell_value - (entry_price * position)
                    
                    trades.append({
                        'entry': entry_price,
                        'exit': current_price,
                        'return': pct_change,
                        'reason': 'take_profit'
                    })
                    
                    balance = sell_value
                    position = 0
                    continue
            
            # Entry
            if signal == 1 and position == 0:
                position = balance * (1 - self.trade_fee) / current_price
                entry_price = current_price
                
            # Exit on signal
            elif signal == -1 and position > 0:
                sell_value = position * current_price * (1 - self.trade_fee)
                pct_change = (current_price - entry_price) / entry_price * 100
                
                trades.append({
                    'entry': entry_price,
                    'exit': current_price,
                    'return': pct_change,
                    'reason': 'signal'
                })
                
                balance = sell_value
                position = 0
        
        if not trades:
            return None
        
        trades_df = pd.DataFrame(trades)
        winning_trades = len(trades_df[trades_df['return'] > 0])
        win_rate = winning_trades / len(trades) * 100
        
        return {
            'symbol': symbol,
            'trades': len(trades),
            'win_rate': win_rate,
            'total_return': (balance - self.initial_balance) / self.initial_balance * 100,
            'avg_return': trades_df['return'].mean(),
            'best_trade': trades_df['return'].max(),
            'worst_trade': trades_df['return'].min(),
            'details': trades_df
        }
    
    def test_strategy(self, strategy_func, name):
        """Test strategy on focus symbols"""
        print(f"\n🔍 Testing: {name}")
        results = []
        
        for symbol in self.symbols:
            print(f"   📈 {symbol}...", end=" ")
            
            df = self.get_data(symbol)
            if df is None or len(df) < 100:
                print("❌ No data")
                continue
                
            df = self.add_indicators(df)
            signals = strategy_func(df)
            result = self.backtest_with_strict_risk_mgmt(df, signals, symbol)
            
            if result and result['trades'] > 0:
                results.append(result)
                print(f"✅ {result['win_rate']:.1f}% ({result['trades']} trades)")
            else:
                print("⚠️  No trades")
        
        if not results:
            return None
            
        avg_wr = np.mean([r['win_rate'] for r in results])
        avg_ret = np.mean([r['total_return'] for r in results])
        
        return {
            'name': name,
            'avg_win_rate': avg_wr,
            'avg_return': avg_ret,
            'results': results
        }

def main():
    print("🎯 Optimized Strategy Finder")
    print(f"📊 Testing {len(FOCUS_SYMBOLS)} symbols: {', '.join(FOCUS_SYMBOLS)}")
    
    finder = OptimizedStrategyFinder(FOCUS_SYMBOLS)
    
    strategies = [
        (finder.golden_cross_reversal_strategy, "Golden Cross Reversal Ultra"),
        (finder.mean_reversion_premium_strategy, "Mean Reversion Premium")
    ]
    
    for strategy_func, name in strategies:
        result = finder.test_strategy(strategy_func, name)
        
        if result:
            print(f"\n📊 {name} Summary:")
            print(f"   🎯 Avg Win Rate: {result['avg_win_rate']:.1f}%")
            print(f"   💰 Avg Return: {result['avg_return']:.1f}%")
            
            if result['avg_win_rate'] > 70:
                print(f"\n🎉 WINNER FOUND: {name}")
                print(f"✅ Win Rate: {result['avg_win_rate']:.1f}%")
                
                print(f"\n📈 Detailed Results:")
                for r in result['results']:
                    print(f"   {r['symbol']}: {r['win_rate']:.1f}% WR, "
                          f"{r['total_return']:.1f}% return, {r['trades']} trades")
                    print(f"            Best: +{r['best_trade']:.1f}%, Worst: {r['worst_trade']:.1f}%")
                
                # Save winning strategy
                import json
                winner_data = {
                    'strategy': name,
                    'avg_win_rate': result['avg_win_rate'],
                    'avg_return': result['avg_return'],
                    'symbol_results': []
                }
                
                for r in result['results']:
                    winner_data['symbol_results'].append({
                        'symbol': r['symbol'],
                        'win_rate': r['win_rate'],
                        'total_return': r['total_return'],
                        'trades': r['trades'],
                        'trade_details': r['details'].to_dict('records')
                    })
                
                with open('winning_strategy.json', 'w') as f:
                    json.dump(winner_data, f, indent=2)
                
                print(f"\n💾 Strategy saved to winning_strategy.json")
                return result
    
    print("\n❌ No strategy achieved >70% win rate")
    return None

if __name__ == "__main__":
    main()