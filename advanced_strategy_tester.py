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

class AdvancedStrategyTester:
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
    
    def calculate_advanced_indicators(self, df):
        """Calculate advanced technical indicators"""
        # Basic indicators
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
        
        # Multiple EMAs
        df['ema_8'] = df['close'].ewm(span=8).mean()
        df['ema_13'] = df['close'].ewm(span=13).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['ema_55'] = df['close'].ewm(span=55).mean()
        df['ema_200'] = df['close'].ewm(span=200).mean()
        
        # Bollinger Bands with multiple periods
        for period in [20, 50]:
            bb_middle = df['close'].rolling(window=period).mean()
            bb_std = df['close'].rolling(window=period).std()
            df[f'bb_{period}_upper'] = bb_middle + (bb_std * 2)
            df[f'bb_{period}_lower'] = bb_middle - (bb_std * 2)
            df[f'bb_{period}_position'] = (df['close'] - df[f'bb_{period}_lower']) / (df[f'bb_{period}_upper'] - df[f'bb_{period}_lower'])
        
        # Advanced momentum indicators
        df['stoch_k'] = ((df['close'] - df['low'].rolling(14).min()) / 
                        (df['high'].rolling(14).max() - df['low'].rolling(14).min())) * 100
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Williams %R with different periods
        for period in [14, 21]:
            df[f'williams_r_{period}'] = ((df['high'].rolling(period).max() - df['close']) / 
                                         (df['high'].rolling(period).max() - df['low'].rolling(period).min())) * -100
        
        # ATR for volatility
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # Volume indicators
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Price action
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['total_range'] = df['high'] - df['low']
        
        return df
    
    def detect_advanced_patterns(self, df):
        """Detect advanced candlestick and chart patterns"""
        # Basic patterns
        df['doji'] = (df['body_size'] / df['total_range'] < 0.1) & (df['total_range'] > 0)
        df['hammer'] = ((df['lower_shadow'] > 2 * df['body_size']) & 
                       (df['upper_shadow'] < 0.1 * df['total_range']) & 
                       (df['close'] != df['open']))
        
        # Engulfing patterns with volume confirmation
        df['bullish_engulfing'] = ((df['close'] > df['open']) & 
                                  (df['close'].shift(1) < df['open'].shift(1)) & 
                                  (df['close'] > df['open'].shift(1)) & 
                                  (df['open'] < df['close'].shift(1)) &
                                  (df['volume'] > df['volume'].shift(1)))
        
        # Support and resistance levels
        df['local_high'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
        df['local_low'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))
        
        # Trend strength
        df['trend_strength'] = (df['ema_8'] - df['ema_200']) / df['ema_200']
        
        # Volatility breakout
        df['volatility_breakout'] = df['atr'] > df['atr'].rolling(20).mean() * 1.5
        
        return df
    
    def strategy_6_high_probability_reversal(self, df):
        """Strategy 6: High Probability Reversal with Multiple Confirmations"""
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        
        # Strong oversold conditions with multiple confirmations
        oversold_condition = (
            (df['rsi'] < 25) & 
            (df['stoch_k'] < 15) & 
            (df['williams_r_14'] < -85) &
            (df['bb_20_position'] < 0.1)
        )
        
        # Trend and pattern confirmations
        trend_ok = (df['close'] > df['ema_200']) | (df['trend_strength'] > -0.2)
        pattern_confirm = df['hammer'] | df['bullish_engulfing'] | df['doji']
        volume_confirm = df['volume_ratio'] > 1.2
        
        buy_condition = oversold_condition & trend_ok & pattern_confirm & volume_confirm
        
        # Conservative exit conditions
        sell_condition = (
            (df['rsi'] > 65) | 
            (df['stoch_k'] > 75) |
            (df['close'] < df['ema_13']) |
            (df['bb_20_position'] > 0.85)
        )
        
        signals.loc[buy_condition, 'signal'] = 1
        signals.loc[sell_condition, 'signal'] = -1
        
        return signals
    
    def strategy_7_breakout_momentum(self, df):
        """Strategy 7: Volatility Breakout with Momentum"""
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        
        # Breakout conditions
        breakout_up = (
            (df['close'] > df['bb_20_upper']) &
            (df['volume_ratio'] > 1.5) &
            (df['volatility_breakout']) &
            (df['macd'] > df['macd_signal']) &
            (df['rsi'] > 50) & (df['rsi'] < 80)
        )
        
        # Trend alignment
        trend_alignment = (
            (df['ema_8'] > df['ema_13']) &
            (df['ema_13'] > df['ema_21']) &
            (df['ema_21'] > df['ema_55'])
        )
        
        buy_condition = breakout_up & trend_alignment
        
        # Exit on momentum loss or reversal
        sell_condition = (
            (df['macd'] < df['macd_signal']) |
            (df['rsi'] > 80) |
            (df['close'] < df['ema_21'])
        )
        
        signals.loc[buy_condition, 'signal'] = 1
        signals.loc[sell_condition, 'signal'] = -1
        
        return signals
    
    def strategy_8_mean_reversion_premium(self, df):
        """Strategy 8: Mean Reversion with Premium Entry Conditions"""
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        
        # Extreme oversold with quality setup
        extreme_oversold = (
            (df['rsi'] < 20) & 
            (df['williams_r_21'] < -90) &
            (df['bb_50_position'] < 0.05)
        )
        
        # Price near significant support (EMA 200 or previous low)
        support_test = (
            (abs(df['close'] - df['ema_200']) / df['ema_200'] < 0.02) |
            (df['local_low'])
        )
        
        # Bullish divergence approximation
        price_momentum = df['close'].rolling(10).mean() / df['close'].rolling(10).mean().shift(10)
        rsi_momentum = df['rsi'].rolling(10).mean() / df['rsi'].rolling(10).mean().shift(10)
        bullish_divergence = (price_momentum < 1) & (rsi_momentum > 1)
        
        buy_condition = extreme_oversold & support_test & (bullish_divergence | df['hammer'])
        
        # Exit on mean reversion completion
        sell_condition = (
            (df['rsi'] > 55) |
            (df['bb_20_position'] > 0.7) |
            (df['close'] < df['ema_55'])
        )
        
        signals.loc[buy_condition, 'signal'] = 1
        signals.loc[sell_condition, 'signal'] = -1
        
        return signals
    
    def strategy_9_multi_timeframe_confluence(self, df):
        """Strategy 9: Multi-timeframe Confluence (simulated)"""
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        
        # Simulate higher timeframe trend (using longer EMAs)
        htf_uptrend = (df['ema_55'] > df['ema_200']) & (df['ema_55'].diff() > 0)
        
        # Lower timeframe setup
        ltf_oversold = (
            (df['rsi'] < 35) &
            (df['stoch_k'] < 25) &
            (df['close'] < df['ema_21'])
        )
        
        # Confluence factors
        volume_spike = df['volume_ratio'] > 1.3
        pattern_support = df['hammer'] | df['bullish_engulfing']
        
        # Price approaching key level
        key_level_test = (
            (abs(df['close'] - df['ema_55']) / df['ema_55'] < 0.03) |
            (abs(df['close'] - df['bb_50_lower']) / df['bb_50_lower'] < 0.02)
        )
        
        buy_condition = htf_uptrend & ltf_oversold & volume_spike & pattern_support & key_level_test
        
        # Exit conditions
        sell_condition = (
            (df['rsi'] > 70) |
            (df['close'] < df['ema_55']) |
            (~htf_uptrend)
        )
        
        signals.loc[buy_condition, 'signal'] = 1
        signals.loc[sell_condition, 'signal'] = -1
        
        return signals
    
    def strategy_10_adaptive_volatility(self, df):
        """Strategy 10: Adaptive Volatility-Based Strategy"""
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        
        # Volatility-adjusted RSI thresholds
        vol_percentile = df['atr'].rolling(50).rank(pct=True)
        dynamic_rsi_low = 30 - (vol_percentile * 10)  # Lower threshold in high vol
        dynamic_rsi_high = 70 + (vol_percentile * 10)  # Higher threshold in high vol
        
        # Adaptive entry
        adaptive_oversold = df['rsi'] < dynamic_rsi_low
        volatility_contraction = df['atr'] < df['atr'].rolling(20).mean() * 0.8
        
        # Quality setups
        trend_filter = df['close'] > df['ema_200']
        momentum_alignment = (df['macd'] > df['macd'].shift(1)) & (df['macd_signal'] > df['macd_signal'].shift(1))
        
        buy_condition = (
            adaptive_oversold & 
            volatility_contraction & 
            trend_filter & 
            momentum_alignment &
            (df['hammer'] | df['bullish_engulfing'])
        )
        
        # Adaptive exit
        sell_condition = (
            (df['rsi'] > dynamic_rsi_high) |
            (df['close'] < df['ema_21']) |
            (df['atr'] > df['atr'].rolling(10).mean() * 1.8)  # Exit on vol expansion
        )
        
        signals.loc[buy_condition, 'signal'] = 1
        signals.loc[sell_condition, 'signal'] = -1
        
        return signals
    
    def backtest_strategy_with_stops(self, df, signals, symbol, stop_loss_pct=5, take_profit_pct=10):
        """Enhanced backtesting with stop loss and take profit"""
        balance = self.initial_balance
        position = 0
        entry_price = 0
        trades = []
        
        for i in range(1, len(signals)):
            current_price = df['close'].iloc[i]
            signal = signals['signal'].iloc[i]
            
            # Check stop loss and take profit if in position
            if position > 0:
                pct_change = (current_price - entry_price) / entry_price * 100
                
                # Stop loss hit
                if pct_change <= -stop_loss_pct:
                    sell_value = position * current_price
                    fee = sell_value * self.trade_fee
                    profit_loss = sell_value - fee - (entry_price * position)
                    
                    trades.append({
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'profit_loss': profit_loss,
                        'return_pct': pct_change,
                        'exit_reason': 'stop_loss'
                    })
                    
                    balance += sell_value - fee
                    position = 0
                    continue
                
                # Take profit hit
                elif pct_change >= take_profit_pct:
                    sell_value = position * current_price
                    fee = sell_value * self.trade_fee
                    profit_loss = sell_value - fee - (entry_price * position)
                    
                    trades.append({
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'profit_loss': profit_loss,
                        'return_pct': pct_change,
                        'exit_reason': 'take_profit'
                    })
                    
                    balance += sell_value - fee
                    position = 0
                    continue
            
            # Buy signal
            if signal == 1 and position == 0:
                position = (balance * 0.95) / current_price
                entry_price = current_price
                balance *= 0.05
                
            # Sell signal
            elif signal == -1 and position > 0:
                sell_value = position * current_price
                fee = sell_value * self.trade_fee
                profit_loss = sell_value - fee - (entry_price * position)
                pct_change = (current_price - entry_price) / entry_price * 100
                
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'profit_loss': profit_loss,
                    'return_pct': pct_change,
                    'exit_reason': 'signal'
                })
                
                balance += sell_value - fee
                position = 0
        
        # Close any open position at the end
        if position > 0:
            final_price = df['close'].iloc[-1]
            sell_value = position * final_price
            fee = sell_value * self.trade_fee
            profit_loss = sell_value - fee - (entry_price * position)
            pct_change = (final_price - entry_price) / entry_price * 100
            
            trades.append({
                'entry_price': entry_price,
                'exit_price': final_price,
                'profit_loss': profit_loss,
                'return_pct': pct_change,
                'exit_reason': 'end'
            })
            
            balance += sell_value - fee
        
        if not trades:
            return None
        
        trades_df = pd.DataFrame(trades)
        
        # Calculate enhanced metrics
        total_return = (balance - self.initial_balance) / self.initial_balance * 100
        num_trades = len(trades)
        winning_trades = len(trades_df[trades_df['profit_loss'] > 0])
        win_rate = winning_trades / num_trades * 100
        avg_return = trades_df['return_pct'].mean()
        avg_win = trades_df[trades_df['return_pct'] > 0]['return_pct'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['return_pct'] < 0]['return_pct'].mean() if num_trades - winning_trades > 0 else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * (num_trades - winning_trades))) if avg_loss != 0 else float('inf')
        
        return {
            'symbol': symbol,
            'total_return': total_return,
            'final_balance': balance,
            'num_trades': num_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'avg_return_per_trade': avg_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_loss': trades_df['return_pct'].min(),
            'max_gain': trades_df['return_pct'].max(),
            'trades': trades_df
        }
    
    def test_advanced_strategy(self, strategy_func, strategy_name):
        """Test an advanced strategy across all symbols"""
        print(f"\n🔬 Testing {strategy_name}...")
        results = []
        
        for symbol in self.symbols:
            print(f"   📊 Processing {symbol}...")
            try:
                df = self.get_historical_data(symbol, days_back=730)  # 2 years of data
                if df is None or len(df) < 500:
                    print(f"   ❌ Insufficient data for {symbol}")
                    continue
                
                df = self.calculate_advanced_indicators(df)
                df = self.detect_advanced_patterns(df)
                
                signals = strategy_func(df)
                result = self.backtest_strategy_with_stops(df, signals, symbol)
                
                if result:
                    results.append(result)
                    print(f"   ✅ {symbol}: {result['win_rate']:.1f}% win rate, "
                          f"{result['total_return']:.2f}% return, PF: {result['profit_factor']:.2f}")
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
        avg_profit_factor = np.mean([r['profit_factor'] for r in results if r['profit_factor'] != float('inf')])
        
        return {
            'strategy_name': strategy_name,
            'avg_win_rate': avg_win_rate,
            'avg_return': avg_return,
            'total_trades': total_trades,
            'avg_profit_factor': avg_profit_factor,
            'results': results
        }

def main():
    """Main function to test advanced strategies"""
    print("🚀 Advanced Strategy Tester Starting...")
    tester = AdvancedStrategyTester(TOP_10_SYMBOLS)
    print(f"✅ Testing {len(TOP_10_SYMBOLS)} symbols with advanced strategies")
    
    strategies = [
        (tester.strategy_6_high_probability_reversal, "High Probability Reversal"),
        (tester.strategy_7_breakout_momentum, "Breakout Momentum"),
        (tester.strategy_8_mean_reversion_premium, "Mean Reversion Premium"),
        (tester.strategy_9_multi_timeframe_confluence, "Multi-timeframe Confluence"),
        (tester.strategy_10_adaptive_volatility, "Adaptive Volatility")
    ]
    
    winning_strategy = None
    
    for strategy_func, strategy_name in strategies:
        result = tester.test_advanced_strategy(strategy_func, strategy_name)
        
        if result:
            print(f"\n📈 {strategy_name} Results:")
            print(f"   🎯 Average Win Rate: {result['avg_win_rate']:.2f}%")
            print(f"   💰 Average Return: {result['avg_return']:.2f}%")
            print(f"   📊 Total Trades: {result['total_trades']}")
            print(f"   🔥 Avg Profit Factor: {result['avg_profit_factor']:.2f}")
            
            if result['avg_win_rate'] > 70:
                print(f"🎉 FOUND WINNING STRATEGY: {strategy_name}")
                print(f"✅ Win Rate: {result['avg_win_rate']:.2f}%")
                winning_strategy = result
                break
        else:
            print(f"❌ {strategy_name} failed to generate results")
    
    if winning_strategy:
        print(f"\n🏆 WINNING STRATEGY FOUND: {winning_strategy['strategy_name']}")
        print(f"📊 Detailed Performance:")
        for r in winning_strategy['results']:
            print(f"   {r['symbol']}: {r['win_rate']:.1f}% win rate, "
                  f"{r['total_return']:.2f}% return, {r['num_trades']} trades, PF: {r['profit_factor']:.2f}")
        
        # Save detailed results
        import json
        with open(f"winning_strategy_results.json", "w") as f:
            # Convert DataFrames to dict for JSON serialization
            results_for_json = []
            for r in winning_strategy['results']:
                r_copy = r.copy()
                r_copy['trades'] = r_copy['trades'].to_dict('records')
                results_for_json.append(r_copy)
            
            winning_strategy['results'] = results_for_json
            json.dump(winning_strategy, f, indent=2)
        
        print(f"💾 Results saved to winning_strategy_results.json")
        
    else:
        print("\n🔍 No strategy achieved >70% win rate. Continuing search...")

if __name__ == "__main__":
    main()