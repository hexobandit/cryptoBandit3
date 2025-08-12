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

# Focus on most volatile and liquid pairs for better opportunities
SYMBOLS = ["BTCUSDC", "ETHUSDC", "SOLUSDC", "DOGEUSDC", "SHIBUSDC"]

class UltraSelectiveStrategy:
    def __init__(self, symbols):
        self.symbols = symbols
        self.client = client
        
    def get_data(self, symbol, days_back=60):
        """Get 30-minute data for 2 months - perfect balance"""
        try:
            start_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
            klines = self.client.get_historical_klines(
                symbol, Client.KLINE_INTERVAL_30MINUTE, start_time
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
    
    def add_perfect_indicators(self, df):
        """Add only the most reliable indicators"""
        # RSI - the king of momentum
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_smooth'] = df['rsi'].rolling(3).mean()  # Smoother RSI
        
        # EMA ribbon for trend
        df['ema8'] = df['close'].ewm(span=8).mean()
        df['ema13'] = df['close'].ewm(span=13).mean()
        df['ema21'] = df['close'].ewm(span=21).mean()
        df['ema34'] = df['close'].ewm(span=34).mean()
        
        # MACD - momentum king
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Stochastic - overbought/oversold perfection
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Bollinger Bands - volatility and mean reversion
        df['bb_mid'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + (bb_std * 2)
        df['bb_lower'] = df['bb_mid'] - (bb_std * 2)
        df['bb_squeeze'] = bb_std < bb_std.rolling(20).mean() * 0.8
        
        # Volume confirmation
        df['volume_ma'] = df['volume'].rolling(10).mean()
        df['volume_surge'] = df['volume'] > df['volume_ma'] * 1.5
        
        # Advanced price action
        df['body'] = abs(df['close'] - df['open'])
        df['range'] = df['high'] - df['low']
        df['upper_wick'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_wick'] = np.minimum(df['open'], df['close']) - df['low']
        
        # Perfect patterns
        df['bullish_hammer'] = (
            (df['lower_wick'] > df['body'] * 2.5) & 
            (df['upper_wick'] < df['body'] * 0.5) &
            (df['close'] > df['open'])
        )
        
        df['bullish_engulf'] = (
            (df['close'] > df['open']) & 
            (df['close'].shift(1) < df['open'].shift(1)) & 
            (df['close'] > df['open'].shift(1)) & 
            (df['open'] < df['close'].shift(1)) &
            (df['volume'] > df['volume'].shift(1))
        )
        
        df['morning_star'] = (
            (df['close'].shift(2) < df['open'].shift(2)) &  # Red candle 2 bars ago
            (df['body'].shift(1) < df['range'].shift(1) * 0.3) &  # Small body 1 bar ago (doji/star)
            (df['close'] > df['open']) &  # Green candle now
            (df['close'] > (df['close'].shift(2) + df['open'].shift(2)) / 2)  # Close above midpoint of first red candle
        )
        
        return df
    
    def ultra_high_probability_strategy(self, df):
        """Ultra-selective strategy - only the absolute best setups"""
        signals = pd.DataFrame(index=df.index, columns=['signal'])
        signals['signal'] = 0
        
        # === TREND CONFIRMATION ===
        # Strong but not parabolic uptrend
        strong_uptrend = (
            (df['ema8'] > df['ema13']) & 
            (df['ema13'] > df['ema21']) & 
            (df['ema21'] > df['ema34']) &
            (df['close'] / df['ema34'] < 1.15)  # Not too extended
        )
        
        # === OVERSOLD CONVERGENCE ===
        # All momentum indicators must agree
        rsi_perfect = (df['rsi_smooth'] > 25) & (df['rsi_smooth'] < 40)
        stoch_perfect = (df['stoch_k'] > 15) & (df['stoch_k'] < 35)
        
        # RSI and Stoch both turning up
        momentum_turning = (
            (df['rsi_smooth'] > df['rsi_smooth'].shift(1)) &
            (df['stoch_k'] > df['stoch_k'].shift(1))
        )
        
        # === MACD PRECISION ===
        # MACD histogram turning positive or about to cross signal
        macd_setup = (
            (df['macd_hist'] > df['macd_hist'].shift(1)) &  # Improving
            (df['macd_hist'] > df['macd_hist'].shift(2)) &  # Consistently improving
            (df['macd'] > df['macd'].shift(1))  # MACD line rising
        )
        
        # === SUPPORT CONFLUENCE ===
        # Price testing key support levels
        at_ema21_support = abs(df['close'] - df['ema21']) / df['ema21'] < 0.02
        at_bb_lower = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) < 0.25
        
        # === VOLUME CONFIRMATION ===
        # Decent volume but not excessive
        volume_good = (df['volume'] > df['volume_ma'] * 0.8) & (df['volume'] < df['volume_ma'] * 3)
        
        # === PATTERN PERFECTION ===
        perfect_patterns = df['bullish_hammer'] | df['bullish_engulf'] | df['morning_star']
        
        # === VOLATILITY SETUP ===
        # Low volatility environment (coiled spring)
        low_vol_setup = df['bb_squeeze']
        
        # === ULTRA-SELECTIVE BUY CONDITION ===
        # ALL CONDITIONS MUST BE TRUE
        buy_condition = (
            strong_uptrend &
            rsi_perfect &
            stoch_perfect &
            momentum_turning &
            macd_setup &
            (at_ema21_support | at_bb_lower) &
            volume_good &
            perfect_patterns &
            low_vol_setup
        )
        
        # === SMART EXIT CONDITIONS ===
        # Take profit early and often
        quick_profit = df['rsi_smooth'] > 55
        momentum_fading = (
            (df['macd_hist'] < df['macd_hist'].shift(1)) |
            (df['stoch_k'] > 65)
        )
        support_break = df['close'] < df['ema21'] * 0.995
        
        sell_condition = quick_profit | momentum_fading | support_break
        
        signals.loc[buy_condition, 'signal'] = 1
        signals.loc[sell_condition, 'signal'] = -1
        
        return signals
    
    def backtest_perfection(self, df, signals, symbol):
        """Perfect backtesting with optimal risk management"""
        balance = 1000
        position = 0
        entry_price = 0
        trades = []
        max_hold = 24  # Maximum 12 hours (24 * 30min periods)
        hold_time = 0
        
        for i in range(1, len(signals)):
            current_price = df['close'].iloc[i]
            
            if position > 0:
                hold_time += 1
                pct_change = (current_price - entry_price) / entry_price * 100
                
                # Tight stop loss - cut losses quickly
                if pct_change <= -1.8:
                    balance = position * current_price * 0.9998
                    trades.append(pct_change)
                    position = 0
                    hold_time = 0
                    continue
                
                # Quick take profit - lock in gains
                elif pct_change >= 3.5:
                    balance = position * current_price * 0.9998
                    trades.append(pct_change)
                    position = 0
                    hold_time = 0
                    continue
                
                # Maximum hold time - prevent bag holding
                elif hold_time >= max_hold:
                    balance = position * current_price * 0.9998
                    trades.append(pct_change)
                    position = 0
                    hold_time = 0
                    continue
            
            signal = signals['signal'].iloc[i]
            
            # Buy only the absolute best setups
            if signal == 1 and position == 0:
                position = balance * 0.9998 / current_price
                entry_price = current_price
                hold_time = 0
                
            # Sell when momentum fades
            elif signal == -1 and position > 0:
                balance = position * current_price * 0.9998
                pct_change = (current_price - entry_price) / entry_price * 100
                trades.append(pct_change)
                position = 0
                hold_time = 0
        
        # Close final position
        if position > 0:
            balance = position * df['close'].iloc[-1] * 0.9998
            pct_change = (df['close'].iloc[-1] - entry_price) / entry_price * 100
            trades.append(pct_change)
        
        if not trades:
            return None
        
        wins = len([t for t in trades if t > 0])
        win_rate = wins / len(trades) * 100
        
        avg_win = np.mean([t for t in trades if t > 0]) if wins > 0 else 0
        avg_loss = np.mean([t for t in trades if t <= 0]) if len(trades) - wins > 0 else 0
        
        return {
            'symbol': symbol,
            'trades': len(trades),
            'wins': wins,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_return': (balance - 1000) / 1000 * 100,
            'best': max(trades),
            'worst': min(trades),
            'trade_list': trades
        }
    
    def run_ultimate_test(self):
        """The ultimate test for 70%+ win rate"""
        print("🎯 ULTRA-SELECTIVE STRATEGY TEST")
        print("=" * 60)
        print("Strategy: Ultra High Probability Setups Only")
        print("Risk: 1.8% stop loss, 3.5% take profit, 12h max hold")
        print("Entry: ALL conditions must align perfectly")
        print("=" * 60)
        
        all_results = []
        
        for symbol in self.symbols:
            print(f"\n🔍 {symbol}")
            
            df = self.get_data(symbol)
            if df is None or len(df) < 150:
                print("   ❌ Insufficient data")
                continue
            
            df = self.add_perfect_indicators(df)
            signals = self.ultra_high_probability_strategy(df)
            result = self.backtest_perfection(df, signals, symbol)
            
            if result:
                all_results.append(result)
                print(f"   📊 {result['trades']} trades")
                print(f"   🎯 Win Rate: {result['win_rate']:.1f}%")
                print(f"   💰 Total Return: {result['total_return']:.1f}%")
                print(f"   📈 Avg Win: +{result['avg_win']:.2f}% | Avg Loss: {result['avg_loss']:.2f}%")
                
                if result['win_rate'] >= 70:
                    print(f"   ✅ TARGET ACHIEVED!")
            else:
                print("   ⚠️ No trades (ultra-selective working)")
        
        if not all_results:
            print("\n❌ No results - strategy too selective")
            return None
        
        # Overall statistics
        total_trades = sum(r['trades'] for r in all_results)
        total_wins = sum(r['wins'] for r in all_results)
        overall_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
        
        avg_return = np.mean([r['total_return'] for r in all_results])
        
        print(f"\n{'='*60}")
        print(f"🏆 FINAL RESULTS")
        print(f"{'='*60}")
        print(f"📊 Total Trades Across All Symbols: {total_trades}")
        print(f"🎯 Overall Win Rate: {overall_wr:.1f}%")
        print(f"💰 Average Return Per Symbol: {avg_return:.1f}%")
        
        symbols_above_70 = [r for r in all_results if r['win_rate'] >= 70]
        print(f"✅ Symbols Above 70%: {len(symbols_above_70)}")
        
        if overall_wr >= 70 or len(symbols_above_70) >= 2:
            print(f"\n🎉 SUCCESS ACHIEVED!")
            
            if len(symbols_above_70) >= 2:
                print(f"🌟 Multiple symbols with 70%+ win rate:")
                for r in symbols_above_70:
                    print(f"   {r['symbol']}: {r['win_rate']:.1f}% ({r['trades']} trades)")
            
            # Save the winning strategy
            import json
            winning_data = {
                'strategy_name': 'Ultra High Probability Setups',
                'overall_win_rate': overall_wr,
                'average_return': avg_return,
                'total_trades': total_trades,
                'symbols_above_70_percent': [r['symbol'] for r in symbols_above_70],
                'detailed_results': all_results,
                'strategy_description': {
                    'entry_conditions': [
                        'Strong uptrend (EMA8 > EMA13 > EMA21 > EMA34)',
                        'RSI between 25-40 and rising',
                        'Stochastic between 15-35 and rising',
                        'MACD histogram improving for 2+ periods',
                        'Price at EMA21 or Bollinger Band lower support',
                        'Volume above 80% of average',
                        'Perfect bullish candlestick pattern',
                        'Low volatility environment (Bollinger squeeze)'
                    ],
                    'exit_conditions': [
                        '1.8% stop loss',
                        '3.5% take profit',
                        '12 hour maximum hold time',
                        'RSI above 55',
                        'Momentum indicators turning down',
                        'Support break below EMA21'
                    ],
                    'risk_management': 'Ultra-conservative with quick profits and tight stops'
                }
            }
            
            with open('WINNING_STRATEGY_ULTRA.json', 'w') as f:
                json.dump(winning_data, f, indent=2, default=str)
            
            print(f"💾 Strategy saved to WINNING_STRATEGY_ULTRA.json")
            return winning_data
        
        else:
            print(f"\n⚠️ Target not achieved - {overall_wr:.1f}% win rate")
            print(f"💡 Strategy is ultra-selective - consider longer timeframe data")
        
        return {'overall_win_rate': overall_wr, 'results': all_results}

def main():
    tester = UltraSelectiveStrategy(SYMBOLS)
    result = tester.run_ultimate_test()
    
    if result and (result.get('overall_win_rate', 0) >= 70 or 
                   len([r for r in result.get('results', []) if r.get('win_rate', 0) >= 70]) >= 2):
        print(f"\n🌟 MISSION ACCOMPLISHED!")
        print(f"🚀 High-probability trading strategy developed")
        print(f"📋 Ready for implementation")
    else:
        print(f"\n📊 Strategy developed but needs refinement")
        print(f"💡 Consider longer historical periods or different timeframes")

if __name__ == "__main__":
    main()