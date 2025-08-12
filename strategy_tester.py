import pandas as pd
import numpy as np
from binance.client import Client
import time
import sys
import math
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Load secrets
sys.path.append('../')
from _secrets import api_key, secret_key

client = Client(api_key, secret_key)

# Top 10 cryptocurrencies by market cap (as of 2024) with USDC pairs
TOP_10_SYMBOLS = [
    "BTCUSDC",   # Bitcoin
    "ETHUSDC",   # Ethereum  
    "BNBUSDC",   # BNB
    "SOLUSDC",   # Solana
    "XRPUSDC",   # XRP
    "DOGEUSDC",  # Dogecoin
    "ADAUSDC",   # Cardano
    "SHIBUSDC",  # Shiba Inu
    "AVAXUSDC",  # Avalanche
    "LINKUSDC"   # Chainlink
]

class StrategyTester:
    def __init__(self, symbols, initial_balance=1000, trade_fee=0.001):
        self.symbols = symbols
        self.initial_balance = initial_balance
        self.trade_fee = trade_fee
        self.client = client
        
    def get_historical_data(self, symbol, interval=Client.KLINE_INTERVAL_1HOUR, days_back=365):
        """Fetch historical kline data"""
        try:
            start_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
            klines = self.client.get_historical_klines(
                symbol, interval, start_time
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base', 
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert to proper data types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, df):
        """Calculate various technical indicators"""
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # EMAs
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['ema_200'] = df['close'].ewm(span=200).mean()
        
        # Stochastic
        df['stoch_k'] = ((df['close'] - df['low'].rolling(14).min()) / 
                        (df['high'].rolling(14).max() - df['low'].rolling(14).min())) * 100
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Williams %R
        df['williams_r'] = ((df['high'].rolling(14).max() - df['close']) / 
                           (df['high'].rolling(14).max() - df['low'].rolling(14).min())) * -100
        
        return df
    
    def detect_candlestick_patterns(self, df):
        """Detect various candlestick patterns"""
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['total_range'] = df['high'] - df['low']
        
        # Doji
        df['doji'] = (df['body_size'] / df['total_range'] < 0.1) & (df['total_range'] > 0)
        
        # Hammer
        df['hammer'] = ((df['lower_shadow'] > 2 * df['body_size']) & 
                       (df['upper_shadow'] < 0.1 * df['body_size']) & 
                       (df['close'] < df['open']))
        
        # Shooting Star
        df['shooting_star'] = ((df['upper_shadow'] > 2 * df['body_size']) & 
                              (df['lower_shadow'] < 0.1 * df['body_size']) & 
                              (df['close'] < df['open']))
        
        # Engulfing patterns
        df['bullish_engulfing'] = ((df['close'] > df['open']) & 
                                  (df['close'].shift(1) < df['open'].shift(1)) & 
                                  (df['close'] > df['open'].shift(1)) & 
                                  (df['open'] < df['close'].shift(1)))
        
        df['bearish_engulfing'] = ((df['close'] < df['open']) & 
                                  (df['close'].shift(1) > df['open'].shift(1)) & 
                                  (df['close'] < df['open'].shift(1)) & 
                                  (df['open'] > df['close'].shift(1)))
        
        return df

    def strategy_1_rsi_oversold_bounce(self, df):
        """Strategy 1: RSI Oversold Bounce with EMA Trend"""
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        
        # Buy: RSI < 30, price above EMA200, bullish hammer/doji
        buy_condition = (
            (df['rsi'] < 30) & 
            (df['close'] > df['ema_200']) &
            (df['hammer'] | df['doji'])
        )
        
        # Sell: RSI > 70 or price below EMA21
        sell_condition = (df['rsi'] > 70) | (df['close'] < df['ema_21'])
        
        signals.loc[buy_condition, 'signal'] = 1
        signals.loc[sell_condition, 'signal'] = -1
        
        return signals
    
    def strategy_2_macd_bollinger(self, df):
        """Strategy 2: MACD Crossover with Bollinger Bands"""
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        
        # Buy: MACD crosses above signal, price near lower BB, bullish engulfing
        buy_condition = (
            (df['macd'] > df['macd_signal']) & 
            (df['macd'].shift(1) <= df['macd_signal'].shift(1)) &
            (df['bb_position'] < 0.2) &
            (df['bullish_engulfing'])
        )
        
        # Sell: MACD crosses below signal or price near upper BB
        sell_condition = (
            ((df['macd'] < df['macd_signal']) & 
             (df['macd'].shift(1) >= df['macd_signal'].shift(1))) |
            (df['bb_position'] > 0.8)
        )
        
        signals.loc[buy_condition, 'signal'] = 1
        signals.loc[sell_condition, 'signal'] = -1
        
        return signals
    
    def strategy_3_multi_ema_stoch(self, df):
        """Strategy 3: Multi-EMA with Stochastic"""
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        
        # Buy: EMA9 > EMA21 > EMA50, Stochastic oversold, no bearish patterns
        buy_condition = (
            (df['ema_9'] > df['ema_21']) & 
            (df['ema_21'] > df['ema_50']) &
            (df['stoch_k'] < 20) &
            (~df['bearish_engulfing']) &
            (~df['shooting_star'])
        )
        
        # Sell: EMA9 < EMA21 or Stochastic overbought
        sell_condition = (
            (df['ema_9'] < df['ema_21']) | 
            (df['stoch_k'] > 80)
        )
        
        signals.loc[buy_condition, 'signal'] = 1
        signals.loc[sell_condition, 'signal'] = -1
        
        return signals
    
    def strategy_4_williams_pattern_combo(self, df):
        """Strategy 4: Williams %R with Candlestick Patterns"""
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        
        # Buy: Williams %R < -80, bullish patterns, above EMA50
        buy_condition = (
            (df['williams_r'] < -80) &
            (df['close'] > df['ema_50']) &
            (df['bullish_engulfing'] | df['hammer'])
        )
        
        # Sell: Williams %R > -20 or bearish patterns
        sell_condition = (
            (df['williams_r'] > -20) |
            (df['bearish_engulfing'] | df['shooting_star'])
        )
        
        signals.loc[buy_condition, 'signal'] = 1
        signals.loc[sell_condition, 'signal'] = -1
        
        return signals
    
    def strategy_5_trend_momentum_combo(self, df):
        """Strategy 5: Trend + Momentum + Pattern Combination"""
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        
        # Strong trend confirmation
        strong_uptrend = (
            (df['ema_9'] > df['ema_21']) & 
            (df['ema_21'] > df['ema_50']) & 
            (df['ema_50'] > df['ema_200'])
        )
        
        # Buy: Strong uptrend + momentum oversold + bullish pattern
        buy_condition = (
            strong_uptrend &
            (df['rsi'] < 40) &
            (df['stoch_k'] < 30) &
            (df['williams_r'] < -60) &
            (df['bullish_engulfing'] | df['hammer'] | df['doji'])
        )
        
        # Sell: Trend breaks or momentum overbought
        sell_condition = (
            (df['ema_9'] < df['ema_21']) |
            (df['rsi'] > 75) |
            (df['bearish_engulfing'] | df['shooting_star'])
        )
        
        signals.loc[buy_condition, 'signal'] = 1
        signals.loc[sell_condition, 'signal'] = -1
        
        return signals
    
    def backtest_strategy(self, df, signals, symbol):
        """Backtest a strategy and return performance metrics"""
        balance = self.initial_balance
        position = 0
        entry_price = 0
        trades = []
        
        for i in range(1, len(signals)):
            current_price = df['close'].iloc[i]
            signal = signals['signal'].iloc[i]
            
            # Buy signal
            if signal == 1 and position == 0:
                position = (balance * 0.95) / current_price  # 5% reserved for fees
                entry_price = current_price
                balance *= 0.05  # Keep 5% as cash
                
            # Sell signal
            elif signal == -1 and position > 0:
                sell_value = position * current_price
                fee = sell_value * self.trade_fee
                profit_loss = sell_value - fee - (entry_price * position)
                
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'profit_loss': profit_loss,
                    'return_pct': (current_price - entry_price) / entry_price * 100
                })
                
                balance += sell_value - fee
                position = 0
        
        # Close any open position at the end
        if position > 0:
            final_price = df['close'].iloc[-1]
            sell_value = position * final_price
            fee = sell_value * self.trade_fee
            profit_loss = sell_value - fee - (entry_price * position)
            
            trades.append({
                'entry_price': entry_price,
                'exit_price': final_price,
                'profit_loss': profit_loss,
                'return_pct': (final_price - entry_price) / entry_price * 100
            })
            
            balance += sell_value - fee
        
        if not trades:
            return None
        
        trades_df = pd.DataFrame(trades)
        
        # Calculate metrics
        total_return = (balance - self.initial_balance) / self.initial_balance * 100
        num_trades = len(trades)
        winning_trades = len(trades_df[trades_df['profit_loss'] > 0])
        win_rate = winning_trades / num_trades * 100
        avg_return = trades_df['return_pct'].mean()
        max_loss = trades_df['return_pct'].min()
        max_gain = trades_df['return_pct'].max()
        
        return {
            'symbol': symbol,
            'total_return': total_return,
            'final_balance': balance,
            'num_trades': num_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'avg_return_per_trade': avg_return,
            'max_loss': max_loss,
            'max_gain': max_gain,
            'trades': trades_df
        }
    
    def test_strategy(self, strategy_func, strategy_name):
        """Test a strategy across all symbols"""
        print(f"\n🧪 Testing {strategy_name}...")
        results = []
        
        for symbol in self.symbols:
            print(f"   📈 Processing {symbol}...")
            try:
                df = self.get_historical_data(symbol)
                if df is None or len(df) < 300:
                    print(f"   ❌ Insufficient data for {symbol}")
                    continue
                
                df = self.calculate_technical_indicators(df)
                df = self.detect_candlestick_patterns(df)
                
                signals = strategy_func(df)
                result = self.backtest_strategy(df, signals, symbol)
                
                if result:
                    results.append(result)
                    print(f"   ✅ {symbol}: {result['win_rate']:.1f}% win rate, {result['total_return']:.2f}% return")
                else:
                    print(f"   ⚠️  {symbol}: No trades generated")
                    
            except Exception as e:
                print(f"   ❌ Error with {symbol}: {e}")
                continue
        
        if not results:
            return None
        
        # Calculate overall statistics
        avg_win_rate = np.mean([r['win_rate'] for r in results])
        avg_return = np.mean([r['total_return'] for r in results])
        total_trades = sum([r['num_trades'] for r in results])
        
        return {
            'strategy_name': strategy_name,
            'avg_win_rate': avg_win_rate,
            'avg_return': avg_return,
            'total_trades': total_trades,
            'results': results
        }

def main():
    """Main function to test strategies sequentially"""
    print("🚀 Initializing Strategy Tester...")
    tester = StrategyTester(TOP_10_SYMBOLS)
    print(f"✅ Testing {len(TOP_10_SYMBOLS)} symbols: {', '.join(TOP_10_SYMBOLS)}")
    
    strategies = [
        (tester.strategy_1_rsi_oversold_bounce, "RSI Oversold Bounce + EMA + Patterns"),
        (tester.strategy_2_macd_bollinger, "MACD + Bollinger Bands + Engulfing"),
        (tester.strategy_3_multi_ema_stoch, "Multi-EMA + Stochastic + Patterns"),
        (tester.strategy_4_williams_pattern_combo, "Williams %R + Pattern Combo"),
        (tester.strategy_5_trend_momentum_combo, "Trend + Momentum + Pattern Combo")
    ]
    
    winning_strategy = None
    
    for strategy_func, strategy_name in strategies:
        result = tester.test_strategy(strategy_func, strategy_name)
        
        if result:
            print(f"\n📊 {strategy_name} Results:")
            print(f"   🎯 Average Win Rate: {result['avg_win_rate']:.2f}%")
            print(f"   💰 Average Return: {result['avg_return']:.2f}%")
            print(f"   📈 Total Trades: {result['total_trades']}")
            
            if result['avg_win_rate'] > 70:
                print(f"🎉 FOUND WINNING STRATEGY: {strategy_name}")
                print(f"✅ Win Rate: {result['avg_win_rate']:.2f}%")
                winning_strategy = result
                break
        else:
            print(f"❌ {strategy_name} failed to generate results")
    
    if winning_strategy:
        print(f"\n🏆 FINAL WINNING STRATEGY: {winning_strategy['strategy_name']}")
        print(f"📊 Detailed Results:")
        for r in winning_strategy['results']:
            print(f"   {r['symbol']}: {r['win_rate']:.1f}% win rate, "
                  f"{r['total_return']:.2f}% return, {r['num_trades']} trades")
    else:
        print("\n❌ No strategy achieved >70% win rate")

if __name__ == "__main__":
    main()