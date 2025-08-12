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

# Test on all major pairs
SYMBOLS = ["BTCUSDC", "ETHUSDC", "BNBUSDC", "SOLUSDC", "XRPUSDC", "DOGEUSDC", "ADAUSDC"]

class PracticalWinnerFinder:
    def __init__(self, symbols):
        self.symbols = symbols
        self.client = client
        
    def get_data(self, symbol, days_back=120):
        """Get 4-hour data for 4 months"""
        try:
            start_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
            klines = self.client.get_historical_klines(
                symbol, Client.KLINE_INTERVAL_4HOUR, start_time
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
    
    def calculate_indicators(self, df):
        """Calculate key indicators"""
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # EMAs
        df['ema12'] = df['close'].ewm(span=12).mean()
        df['ema26'] = df['close'].ewm(span=26).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        
        # MACD
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Stochastic
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + (bb_std * 2)
        df['bb_lower'] = df['bb_mid'] - (bb_std * 2)
        
        # Simple candlestick patterns
        df['green'] = df['close'] > df['open']
        df['red'] = df['close'] < df['open']
        df['doji'] = abs(df['close'] - df['open']) < (df['high'] - df['low']) * 0.1
        
        return df
    
    def strategy_rsi_stoch_oversold(self, df):
        """RSI + Stochastic Oversold with EMA Filter"""
        signals = pd.DataFrame(index=df.index, columns=['signal'])
        signals['signal'] = 0
        
        # Buy: Both RSI and Stochastic oversold, price above EMA50
        buy_condition = (
            (df['rsi'] < 35) & 
            (df['stoch_k'] < 25) &
            (df['close'] > df['ema50']) &  # Uptrend filter
            (df['macd_hist'] > df['macd_hist'].shift(1))  # MACD improving
        )
        
        # Sell: Either RSI or Stochastic overbought
        sell_condition = (df['rsi'] > 65) | (df['stoch_k'] > 75)
        
        signals.loc[buy_condition, 'signal'] = 1
        signals.loc[sell_condition, 'signal'] = -1
        
        return signals
    
    def strategy_macd_bb_combo(self, df):
        """MACD Signal Cross + Bollinger Band Position"""
        signals = pd.DataFrame(index=df.index, columns=['signal'])
        signals['signal'] = 0
        
        # MACD cross above signal line
        macd_cross_up = (
            (df['macd'] > df['macd_signal']) & 
            (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        )
        
        # Price near lower BB but not crashing
        bb_position = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        near_lower_bb = (bb_position < 0.3) & (bb_position > 0.1)
        
        # Buy condition
        buy_condition = macd_cross_up & near_lower_bb & (df['close'] > df['ema50'])
        
        # Sell: MACD cross down or price near upper BB
        macd_cross_down = (
            (df['macd'] < df['macd_signal']) & 
            (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        )
        sell_condition = macd_cross_down | (bb_position > 0.8)
        
        signals.loc[buy_condition, 'signal'] = 1
        signals.loc[sell_condition, 'signal'] = -1
        
        return signals
    
    def strategy_ema_momentum_reversal(self, df):
        """EMA Momentum Reversal Strategy"""
        signals = pd.DataFrame(index=df.index, columns=['signal'])
        signals['signal'] = 0
        
        # EMA alignment for uptrend
        ema_bullish = df['ema12'] > df['ema26']
        
        # Oversold conditions
        oversold = (df['rsi'] < 40) & (df['stoch_k'] < 30)
        
        # Momentum turning up
        momentum_up = (
            (df['macd_hist'] > df['macd_hist'].shift(1)) &
            (df['rsi'] > df['rsi'].shift(1))
        )
        
        buy_condition = ema_bullish & oversold & momentum_up
        
        # Exit conditions
        sell_condition = (
            (df['rsi'] > 70) | 
            (df['ema12'] < df['ema26']) |
            (df['stoch_k'] > 80)
        )
        
        signals.loc[buy_condition, 'signal'] = 1
        signals.loc[sell_condition, 'signal'] = -1
        
        return signals
    
    def strategy_trend_pullback_entry(self, df):
        """Trend Following with Pullback Entry"""
        signals = pd.DataFrame(index=df.index, columns=['signal'])
        signals['signal'] = 0
        
        # Strong uptrend: price well above EMA50
        strong_uptrend = df['close'] > df['ema50'] * 1.02
        
        # Pullback to EMA26 area
        pullback_to_ema = (
            (df['close'] <= df['ema26'] * 1.01) & 
            (df['close'] >= df['ema26'] * 0.99)
        )
        
        # RSI not extremely overbought
        rsi_ok = df['rsi'] < 70
        
        # Bullish momentum returning
        momentum_return = (
            (df['macd'] > df['macd'].shift(1)) &
            (df['stoch_k'] > df['stoch_k'].shift(1))
        )
        
        buy_condition = strong_uptrend & pullback_to_ema & rsi_ok & momentum_return
        
        # Exit on trend break or overbought
        sell_condition = (
            (df['close'] < df['ema50']) |
            (df['rsi'] > 80) |
            (df['macd'] < df['macd_signal'])
        )
        
        signals.loc[buy_condition, 'signal'] = 1
        signals.loc[sell_condition, 'signal'] = -1
        
        return signals
    
    def strategy_volatility_breakout(self, df):
        """Volatility Breakout Strategy"""
        signals = pd.DataFrame(index=df.index, columns=['signal'])
        signals['signal'] = 0
        
        # Calculate volatility (rolling std of returns)
        returns = df['close'].pct_change()
        volatility = returns.rolling(20).std()
        vol_percentile = volatility.rolling(50).rank(pct=True)
        
        # Low volatility environment
        low_vol = vol_percentile < 0.3
        
        # Price breaking above recent high
        recent_high = df['high'].rolling(20).max().shift(1)
        breakout = df['close'] > recent_high
        
        # Volume confirmation (approximated by green candle)
        volume_confirm = df['green']
        
        # RSI not extremely overbought
        rsi_healthy = df['rsi'] < 75
        
        buy_condition = low_vol & breakout & volume_confirm & rsi_healthy
        
        # Exit on reversal signs
        sell_condition = (
            (df['close'] < df['ema26']) |
            (df['rsi'] > 85) |
            (df['red'] & (df['close'] < df['open'] * 0.98))  # Strong red candle
        )
        
        signals.loc[buy_condition, 'signal'] = 1
        signals.loc[sell_condition, 'signal'] = -1
        
        return signals
    
    def backtest_strategy(self, df, signals, symbol, stop_loss=4, take_profit=8):
        """Backtest with stop loss and take profit"""
        balance = 1000
        position = 0
        entry_price = 0
        trades = []
        
        for i in range(1, len(signals)):
            current_price = df['close'].iloc[i]
            
            # Check stop/profit if in position
            if position > 0:
                pct_change = (current_price - entry_price) / entry_price * 100
                
                if pct_change <= -stop_loss:  # Stop loss
                    balance = position * current_price * 0.999  # 0.1% fee
                    trades.append(pct_change)
                    position = 0
                    continue
                elif pct_change >= take_profit:  # Take profit
                    balance = position * current_price * 0.999
                    trades.append(pct_change)
                    position = 0
                    continue
            
            signal = signals['signal'].iloc[i]
            
            # Buy
            if signal == 1 and position == 0:
                position = balance * 0.999 / current_price  # 0.1% fee
                entry_price = current_price
                
            # Sell
            elif signal == -1 and position > 0:
                balance = position * current_price * 0.999
                pct_change = (current_price - entry_price) / entry_price * 100
                trades.append(pct_change)
                position = 0
        
        if not trades:
            return None
        
        wins = len([t for t in trades if t > 0])
        win_rate = wins / len(trades) * 100
        avg_return = np.mean(trades)
        total_return = (balance - 1000) / 1000 * 100
        
        return {
            'symbol': symbol,
            'trades': len(trades),
            'win_rate': win_rate,
            'avg_return': avg_return,
            'total_return': total_return,
            'best': max(trades),
            'worst': min(trades)
        }
    
    def test_strategy(self, strategy_func, name):
        """Test strategy across symbols"""
        print(f"\n🧪 {name}")
        results = []
        
        for symbol in self.symbols:
            print(f"  {symbol}...", end=" ")
            
            df = self.get_data(symbol)
            if df is None or len(df) < 80:
                print("❌")
                continue
            
            df = self.calculate_indicators(df)
            signals = strategy_func(df)
            result = self.backtest_strategy(df, signals, symbol)
            
            if result and result['trades'] >= 3:  # Minimum 3 trades
                results.append(result)
                print(f"✅ {result['win_rate']:.0f}% ({result['trades']})")
            else:
                print("⚠️")
        
        if not results:
            return None
        
        avg_wr = np.mean([r['win_rate'] for r in results])
        avg_total_ret = np.mean([r['total_return'] for r in results])
        
        return {
            'name': name,
            'avg_win_rate': avg_wr,
            'avg_total_return': avg_total_ret,
            'results': results
        }

def main():
    print("🎯 Practical Winner Finder")
    print(f"📊 Symbols: {', '.join(SYMBOLS)}")
    
    finder = PracticalWinnerFinder(SYMBOLS)
    
    strategies = [
        (finder.strategy_rsi_stoch_oversold, "RSI + Stochastic Oversold"),
        (finder.strategy_macd_bb_combo, "MACD + Bollinger Bands"),
        (finder.strategy_ema_momentum_reversal, "EMA Momentum Reversal"),
        (finder.strategy_trend_pullback_entry, "Trend Pullback Entry"),
        (finder.strategy_volatility_breakout, "Volatility Breakout")
    ]
    
    winner_found = False
    
    for strategy_func, name in strategies:
        result = finder.test_strategy(strategy_func, name)
        
        if result:
            print(f"\n📊 {name}:")
            print(f"   🎯 Avg Win Rate: {result['avg_win_rate']:.1f}%")
            print(f"   💰 Avg Total Return: {result['avg_total_return']:.1f}%")
            
            if result['avg_win_rate'] >= 70:
                print(f"\n🎉 WINNER: {name} ({result['avg_win_rate']:.1f}% win rate)")
                
                print(f"\n📈 Individual Results:")
                for r in result['results']:
                    print(f"   {r['symbol']}: {r['win_rate']:.1f}% WR, "
                          f"{r['total_return']:.1f}% total, {r['trades']} trades")
                    print(f"             Best: +{r['best']:.1f}%, Worst: {r['worst']:.1f}%")
                
                # Save winner
                import json
                with open('winning_strategy_found.json', 'w') as f:
                    json.dump({
                        'strategy_name': name,
                        'avg_win_rate': result['avg_win_rate'],
                        'avg_total_return': result['avg_total_return'],
                        'results': result['results']
                    }, f, indent=2)
                
                print(f"\n💾 Winner saved to winning_strategy_found.json")
                winner_found = True
                break
    
    if not winner_found:
        print(f"\n❌ No strategy achieved ≥70% win rate")

if __name__ == "__main__":
    main()