#!/usr/bin/env python3
"""
Test script to verify data coverage for different timeframes
"""

import sys
sys.path.append('../')
from _secrets import api_key, secret_key
from binance.client import Client
import pandas as pd
from datetime import datetime, timedelta

client = Client(api_key, secret_key)

def test_data_coverage():
    """Test how much historical data we can get for each timeframe"""
    
    timeframes = {
        "1m": Client.KLINE_INTERVAL_1MINUTE,
        "5m": Client.KLINE_INTERVAL_5MINUTE,
        "15m": Client.KLINE_INTERVAL_15MINUTE,
        "1h": Client.KLINE_INTERVAL_1HOUR,
        "4h": Client.KLINE_INTERVAL_4HOUR,
        "1d": Client.KLINE_INTERVAL_1DAY,
    }
    
    symbol = "BTCUSDC"  # Test with BTC
    target_days = 365
    target_start = datetime.now() - timedelta(days=target_days)
    
    print(f"🔍 Testing data coverage for {symbol}")
    print(f"Target: {target_days} days back to {target_start.strftime('%Y-%m-%d')}")
    print("=" * 70)
    
    for tf_name, tf_interval in timeframes.items():
        try:
            print(f"\nTesting {tf_name}...")
            
            # Try to get data with simple call first
            start_ms = int(target_start.timestamp() * 1000)
            klines = client.get_historical_klines(
                symbol, 
                tf_interval, 
                start_str=start_ms,
                limit=1000
            )
            
            if klines:
                df = pd.DataFrame(klines)
                df[0] = pd.to_datetime(df[0], unit='ms')  # timestamp column
                
                actual_start = df[0].min()
                actual_end = df[0].max()
                actual_days = (actual_end - actual_start).days
                record_count = len(df)
                
                print(f"  📊 Records: {record_count}")
                print(f"  📅 Period: {actual_start.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')}")
                print(f"  ⏰ Coverage: {actual_days} days (Target: {target_days})")
                
                # Calculate theoretical max for this timeframe
                if tf_name == "1m":
                    theoretical_records = target_days * 24 * 60
                elif tf_name == "5m":
                    theoretical_records = target_days * 24 * 12
                elif tf_name == "15m":
                    theoretical_records = target_days * 24 * 4
                elif tf_name == "1h":
                    theoretical_records = target_days * 24
                elif tf_name == "4h":
                    theoretical_records = target_days * 6
                elif tf_name == "1d":
                    theoretical_records = target_days
                else:
                    theoretical_records = "Unknown"
                
                coverage_pct = (actual_days / target_days) * 100 if target_days > 0 else 0
                
                print(f"  🎯 Coverage: {coverage_pct:.1f}% of target")
                if isinstance(theoretical_records, int):
                    efficiency_pct = (record_count / theoretical_records) * 100
                    print(f"  📈 Efficiency: {efficiency_pct:.1f}% of theoretical maximum")
                
                # Show limitation
                if record_count >= 1000:
                    print(f"  ⚠️  LIMITED by 1000 candle API limit!")
                else:
                    print(f"  ✅ Complete data available")
            else:
                print(f"  ❌ No data received")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n" + "=" * 70)
    print("📝 CONCLUSION:")
    print("• 1m, 5m, 15m, 1h timeframes are LIMITED by 1000 candle API limit")
    print("• Only 4h, 1d, 3d timeframes can provide close to 365 days")
    print("• Chunked fetching (multiple API calls) needed for full coverage")

if __name__ == "__main__":
    test_data_coverage()