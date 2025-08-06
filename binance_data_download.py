#!/usr/bin/env python3
"""
Enhanced script to download BTC/USDT data from Binance API for testing purposes.
Uses pandas, requests, and python-binance for better data handling.
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time

try:
    from binance.client import Client as BinanceClient
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    print("Warning: python-binance not available, using requests fallback")

class BinanceDataDownloader:
    def __init__(self):
        if BINANCE_AVAILABLE:
            self.client = BinanceClient()  # Public client - no API keys needed for market data
            self.use_binance_client = True
        else:
            self.client = None
            self.use_binance_client = False
        
    def get_klines_df(self, symbol='BTCUSDT', interval='1h', limit=1000, start_time=None, end_time=None):
        """
        Download kline/candlestick data and return as pandas DataFrame
        
        Args:
            symbol: Trading pair (default: BTCUSDT)
            interval: Kline interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            limit: Number of records to fetch (max 1000, default 1000)
            start_time: Start time as datetime object or timestamp string
            end_time: End time as datetime object or timestamp string
        
        Returns:
            pandas.DataFrame: OHLCV data with proper datetime index
        """
        try:
            # Use requests API directly (more reliable)
            base_url = 'https://api.binance.com/api/v3/klines'
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            # Add time range if specified
            if start_time:
                if isinstance(start_time, datetime):
                    start_ms = int(start_time.timestamp() * 1000)
                elif isinstance(start_time, str):
                    start_dt = pd.to_datetime(start_time)
                    start_ms = int(start_dt.timestamp() * 1000)
                else:
                    start_ms = int(start_time)
                params['startTime'] = start_ms
                
            if end_time:
                if isinstance(end_time, datetime):
                    end_ms = int(end_time.timestamp() * 1000)
                elif isinstance(end_time, str):
                    end_dt = pd.to_datetime(end_time)
                    end_ms = int(end_dt.timestamp() * 1000)
                else:
                    end_ms = int(end_time)
                params['endTime'] = end_ms
            
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            klines = response.json()
            
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert timestamps to datetime
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            # Convert price and volume columns to float
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                              'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            df['number_of_trades'] = df['number_of_trades'].astype(int)
            
            # Set datetime index
            df.set_index('open_time', inplace=True)
            
            # Drop the ignore column
            df.drop('ignore', axis=1, inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error fetching klines data: {e}")
            return None
    
    def get_historical_data_bulk(self, symbol='BTCUSDT', interval='1d', start_date='2020-01-01'):
        """
        Download large amounts of historical data by making multiple API calls
        
        Args:
            symbol: Trading pair (default: BTCUSDT)
            interval: Kline interval (1h, 4h, 1d, etc.)
            start_date: Start date as string (YYYY-MM-DD)
        
        Returns:
            pandas.DataFrame: Complete historical data
        """
        print(f"Downloading {symbol} {interval} data since {start_date}...")
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.Timestamp.now()
        
        all_data = []
        current_start = start_dt
        
        # Calculate chunk size based on interval
        if interval == '1m':
            chunk_days = 1  # 1440 minutes per day, close to 1000 limit
        elif interval in ['3m', '5m']:
            chunk_days = 2
        elif interval == '15m':
            chunk_days = 10
        elif interval == '30m':
            chunk_days = 20
        elif interval == '1h':
            chunk_days = 41  # ~1000 hours
        elif interval == '4h':
            chunk_days = 166  # ~1000 4h periods
        elif interval == '1d':
            chunk_days = 1000  # 1000 days
        else:
            chunk_days = 100  # Conservative default
        
        chunk_count = 0
        while current_start < end_dt:
            chunk_end = min(current_start + timedelta(days=chunk_days), end_dt)
            
            print(f"Fetching chunk {chunk_count + 1}: {current_start.date()} to {chunk_end.date()}")
            
            df_chunk = self.get_klines_df(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=chunk_end,
                limit=1000
            )
            
            if df_chunk is not None and len(df_chunk) > 0:
                all_data.append(df_chunk)
                # Update start time to last candle's close time + 1 interval
                last_time = df_chunk.index[-1]
                if interval == '1h':
                    current_start = last_time + timedelta(hours=1)
                elif interval == '4h':
                    current_start = last_time + timedelta(hours=4)
                elif interval == '1d':
                    current_start = last_time + timedelta(days=1)
                else:
                    current_start = chunk_end
            else:
                # If no data, move forward by chunk size
                current_start = chunk_end
                
            chunk_count += 1
            
            # Add small delay to be nice to the API
            time.sleep(0.1)
            
            # Safety break
            if chunk_count > 100:
                print("Warning: Stopped after 100 chunks to avoid excessive API calls")
                break
        
        if all_data:
            print(f"Combining {len(all_data)} chunks...")
            combined_df = pd.concat(all_data)
            
            # Remove duplicates and sort
            combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
            combined_df = combined_df.sort_index()
            
            print(f"Final dataset: {len(combined_df)} records from {combined_df.index[0]} to {combined_df.index[-1]}")
            return combined_df
        else:
            print("No data retrieved")
            return None
    
    def get_current_price(self, symbol='BTCUSDT'):
        """Get current price for a symbol"""
        try:
            url = f'https://api.binance.com/api/v3/ticker/price?symbol={symbol}'
            response = requests.get(url)
            response.raise_for_status()
            ticker = response.json()
            return float(ticker['price'])
        except Exception as e:
            print(f"Error fetching current price: {e}")
            return None
    
    def get_24hr_stats(self, symbol='BTCUSDT'):
        """Get 24hr ticker statistics"""
        try:
            url = f'https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}'
            response = requests.get(url)
            response.raise_for_status()
            stats = response.json()
            return {
                'price_change': float(stats['priceChange']),
                'price_change_percent': float(stats['priceChangePercent']),
                'high_price': float(stats['highPrice']),
                'low_price': float(stats['lowPrice']),
                'volume': float(stats['volume']),
                'count': int(stats['count'])
            }
        except Exception as e:
            print(f"Error fetching 24hr stats: {e}")
            return None
    
    def get_multiple_timeframes(self, symbol='BTCUSDT', timeframes=['1h', '4h', '1d'], limit=100):
        """Get data for multiple timeframes"""
        data = {}
        for tf in timeframes:
            print(f"Fetching {tf} data...")
            df = self.get_klines_df(symbol=symbol, interval=tf, limit=limit)
            if df is not None:
                data[tf] = df
        return data

def main():
    print("Enhanced Binance Data Downloader - Historical Data Since 2020")
    print("=" * 60)
    
    downloader = BinanceDataDownloader()
    
    # Get current price and 24hr stats
    current_price = downloader.get_current_price('BTCUSDT')
    stats_24hr = downloader.get_24hr_stats('BTCUSDT')
    
    if current_price:
        print(f"Current BTC/USDT price: ${current_price:,.2f}")
    
    if stats_24hr:
        print(f"24hr Change: {stats_24hr['price_change_percent']:+.2f}% (${stats_24hr['price_change']:+,.2f})")
        print(f"24hr High: ${stats_24hr['high_price']:,.2f}")
        print(f"24hr Low: ${stats_24hr['low_price']:,.2f}")
        print(f"24hr Volume: {stats_24hr['volume']:,.2f} BTC")
    
    print("\nDownloading hourly data since 2020...")
    print("This will take several minutes due to large dataset size and API rate limits...")
    print("Hourly data since 2020 = ~45,000+ records across ~50+ API calls")
    
    # Download hourly data since 2020 
    print("\n" + "="*50)
    print("HOURLY DATA SINCE 2020")
    print("="*50)
    
    df_hourly = downloader.get_historical_data_bulk(
        symbol='BTCUSDT', 
        interval='1h', 
        start_date='2020-01-01'
    )
    
    if df_hourly is not None:
        # Calculate comprehensive statistics
        print(f"\nHourly Data Statistics:")
        print(f"Records: {len(df_hourly)}")
        print(f"Date range: {df_hourly.index[0]} to {df_hourly.index[-1]}")
        print(f"All-time High: ${df_hourly['high'].max():,.2f}")
        print(f"All-time Low: ${df_hourly['low'].min():,.2f}")
        print(f"Average Close: ${df_hourly['close'].mean():,.2f}")
        print(f"Total Volume: {df_hourly['volume'].sum():,.0f} BTC")
        
        # Calculate returns and volatility (hourly basis)
        df_hourly['hourly_returns'] = df_hourly['close'].pct_change()
        annual_return = df_hourly['hourly_returns'].mean() * 24 * 365 * 100
        annual_volatility = df_hourly['hourly_returns'].std() * np.sqrt(24 * 365) * 100
        
        print(f"Annualized Return: {annual_return:.2f}%")
        print(f"Annualized Volatility: {annual_volatility:.2f}%")
        if annual_volatility > 0:
            print(f"Sharpe Ratio: {annual_return/annual_volatility:.2f}")
        
        # Max drawdown calculation
        df_hourly['cumulative'] = (1 + df_hourly['hourly_returns'].fillna(0)).cumprod()
        df_hourly['rolling_max'] = df_hourly['cumulative'].expanding().max()
        df_hourly['drawdown'] = df_hourly['cumulative'] / df_hourly['rolling_max'] - 1
        max_drawdown = df_hourly['drawdown'].min() * 100
        
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        
        # Show some key historical data
        print(f"\nFirst 5 records (2020):")
        early_data = df_hourly.head()[['open', 'high', 'low', 'close', 'volume']]
        for idx, row in early_data.iterrows():
            print(f"{idx} | O: ${row['open']:,.2f} | H: ${row['high']:,.2f} | L: ${row['low']:,.2f} | C: ${row['close']:,.2f} | V: {row['volume']:,.0f}")
        
        print(f"\nRecent 5 records:")
        recent_data = df_hourly.tail()[['open', 'high', 'low', 'close', 'volume']]
        for idx, row in recent_data.iterrows():
            print(f"{idx} | O: ${row['open']:,.2f} | H: ${row['high']:,.2f} | L: ${row['low']:,.2f} | C: ${row['close']:,.2f} | V: {row['volume']:,.0f}")
        
        # Save comprehensive hourly data
        filename_hourly = f"btc_usdt_hourly_2020_to_now_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_hourly.to_csv(filename_hourly)
        print(f"\nHourly data saved to: {filename_hourly}")
        print(f"File size: ~{len(df_hourly) * 0.001:.1f}K records")
        
        # Also save a clean version with just OHLCV for backtesting
        df_clean = df_hourly[['open', 'high', 'low', 'close', 'volume']].copy()
        filename_clean = f"btc_usdt_hourly_clean_2020_to_now_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_clean.to_csv(filename_clean)
        print(f"Clean OHLCV hourly data saved to: {filename_clean}")
        
        # Create additional daily summary from hourly data
        df_daily_from_hourly = df_hourly.resample('D').agg({
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        filename_daily_summary = f"btc_usdt_daily_from_hourly_2020_to_now_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_daily_from_hourly.to_csv(filename_daily_summary)
        print(f"Daily summary from hourly data saved to: {filename_daily_summary}")
    
    # Optional: Download recent high-frequency data for detailed analysis
    print("\n" + "="*50)
    print("RECENT 15-MINUTE DATA (Last 30 Days)")
    print("="*50)
    
    df_15m = downloader.get_historical_data_bulk(
        symbol='BTCUSDT', 
        interval='15m', 
        start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    )
    
    if df_15m is not None:
        print(f"\n15M Data Statistics:")
        print(f"Records: {len(df_15m)}")
        print(f"Date range: {df_15m.index[0]} to {df_15m.index[-1]}")
        
        filename_15m = f"btc_usdt_15m_last30days_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_15m.to_csv(filename_15m)
        print(f"15M data saved to: {filename_15m}")
    
    print(f"\n" + "="*60)
    print("HOURLY DATA DOWNLOAD COMPLETE!")
    print("="*60)
    print("Files ready for high-frequency backtesting and analysis.")
    if df_hourly is not None:
        print(f"✓ Complete hourly dataset: {len(df_hourly)} records since 2020")
        print(f"✓ Covers {(df_hourly.index[-1] - df_hourly.index[0]).days} days of trading history")
        print(f"✓ Perfect for intraday strategy testing and analysis")

def demo_advanced_usage():
    """Demonstrate advanced usage patterns for testing"""
    print("\nAdvanced Usage Examples:")
    print("=" * 30)
    
    downloader = BinanceDataDownloader()
    
    # Get specific date range
    start_date = datetime.now() - timedelta(days=7)
    end_date = datetime.now() - timedelta(days=1)
    
    print(f"Getting data from {start_date.date()} to {end_date.date()}")
    df_range = downloader.get_klines_df(
        symbol='BTCUSDT',
        interval='1h', 
        start_time=start_date,
        end_time=end_date
    )
    
    if df_range is not None:
        print(f"Retrieved {len(df_range)} hourly records for date range")
        print(f"Price range: ${df_range['low'].min():.2f} - ${df_range['high'].max():.2f}")

if __name__ == "__main__":
    main()
    
    # Uncomment to see advanced examples
    # demo_advanced_usage()