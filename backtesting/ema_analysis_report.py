#!/usr/bin/env python3
"""
EMA Strategy Analysis Report Generator
=====================================
Analyzes the backtest results and generates comprehensive performance reports.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

def load_results(filename):
    """Load backtest results from JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Results file {filename} not found")
        return None

def generate_comprehensive_analysis(results):
    """Generate comprehensive analysis of backtest results"""
    if not results:
        print("❌ No results to analyze")
        return
    
    print("=" * 80)
    print("📊 EMA-ENHANCED CANDLESTICK STRATEGY - COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Symbol-Timeframe Combinations: {len(results)}")
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Overall Performance Metrics
    total_trades = df['total_trades'].sum()
    total_profit_loss = df['total_profit_loss'].sum()
    winning_combinations = len(df[df['total_profit_loss'] > 0])
    losing_combinations = len(df[df['total_profit_loss'] <= 0])
    
    print(f"\n🎯 OVERALL PERFORMANCE")
    print("-" * 40)
    print(f"Total Trades Executed: {total_trades:,}")
    print(f"Total Net P&L: ${total_profit_loss:,.2f}")
    print(f"Profitable Combinations: {winning_combinations}/{len(df)} ({winning_combinations/len(df)*100:.1f}%)")
    print(f"Average P&L per Combination: ${total_profit_loss/len(df):.2f}")
    
    if total_trades > 0:
        avg_profit_per_trade = total_profit_loss / total_trades
        print(f"Average P&L per Trade: ${avg_profit_per_trade:.2f}")
    
    # Performance by Timeframe
    print(f"\n📈 PERFORMANCE BY TIMEFRAME")
    print("-" * 40)
    timeframe_analysis = df.groupby('timeframe').agg({
        'total_trades': 'sum',
        'total_profit_loss': ['sum', 'mean'],
        'win_rate': 'mean',
        'return_percent': 'mean'
    }).round(2)
    
    timeframe_analysis.columns = ['Total_Trades', 'Total_PnL', 'Avg_PnL', 'Avg_Win_Rate', 'Avg_Return_%']
    timeframe_analysis = timeframe_analysis.sort_values('Total_PnL', ascending=False)
    
    print(timeframe_analysis.to_string())
    
    # Performance by Symbol
    print(f"\n🪙 PERFORMANCE BY SYMBOL")
    print("-" * 40)
    symbol_analysis = df.groupby('symbol').agg({
        'total_trades': 'sum',
        'total_profit_loss': ['sum', 'mean'],
        'win_rate': 'mean',
        'return_percent': 'mean'
    }).round(2)
    
    symbol_analysis.columns = ['Total_Trades', 'Total_PnL', 'Avg_PnL', 'Avg_Win_Rate', 'Avg_Return_%']
    symbol_analysis = symbol_analysis.sort_values('Total_PnL', ascending=False)
    
    print(symbol_analysis.to_string())
    
    # Top Performers
    print(f"\n🏆 TOP 10 PERFORMING COMBINATIONS")
    print("-" * 40)
    top_performers = df.nlargest(10, 'total_profit_loss')[['symbol', 'timeframe', 'total_trades', 'win_rate', 'total_profit_loss', 'return_percent']]
    for idx, row in top_performers.iterrows():
        print(f"{row['symbol']:>8s} ({row['timeframe']:>3s}): ${row['total_profit_loss']:>8.2f} | "
              f"{row['total_trades']:>3d} trades | {row['win_rate']:>5.1f}% win | {row['return_percent']:>6.2f}% return")
    
    # Worst Performers
    worst_performers = df.nsmallest(5, 'total_profit_loss')[['symbol', 'timeframe', 'total_trades', 'win_rate', 'total_profit_loss', 'return_percent']]
    if len(worst_performers) > 0:
        print(f"\n🔻 WORST 5 PERFORMING COMBINATIONS")
        print("-" * 40)
        for idx, row in worst_performers.iterrows():
            print(f"{row['symbol']:>8s} ({row['timeframe']:>3s}): ${row['total_profit_loss']:>8.2f} | "
                  f"{row['total_trades']:>3d} trades | {row['win_rate']:>5.1f}% win | {row['return_percent']:>6.2f}% return")
    
    # Risk Analysis
    print(f"\n⚠️  RISK ANALYSIS")
    print("-" * 40)
    print(f"Best Single Performance: ${df['total_profit_loss'].max():.2f}")
    print(f"Worst Single Performance: ${df['total_profit_loss'].min():.2f}")
    print(f"Performance Volatility (Std Dev): ${df['total_profit_loss'].std():.2f}")
    print(f"Win Rate Range: {df['win_rate'].min():.1f}% - {df['win_rate'].max():.1f}%")
    
    # Statistical Summary
    print(f"\n📊 STATISTICAL SUMMARY")
    print("-" * 40)
    print(f"Median P&L: ${df['total_profit_loss'].median():.2f}")
    print(f"75th Percentile P&L: ${df['total_profit_loss'].quantile(0.75):.2f}")
    print(f"25th Percentile P&L: ${df['total_profit_loss'].quantile(0.25):.2f}")
    print(f"Average Win Rate: {df['win_rate'].mean():.1f}%")
    print(f"Median Win Rate: {df['win_rate'].median():.1f}%")
    
    # Trading Frequency Analysis
    print(f"\n🔄 TRADING FREQUENCY ANALYSIS")
    print("-" * 40)
    print(f"Most Active Combination: {df.loc[df['total_trades'].idxmax(), 'symbol']} ({df.loc[df['total_trades'].idxmax(), 'timeframe']}) - {df['total_trades'].max()} trades")
    print(f"Least Active Combination: {df.loc[df['total_trades'].idxmin(), 'symbol']} ({df.loc[df['total_trades'].idxmin(), 'timeframe']}) - {df['total_trades'].min()} trades")
    print(f"Average Trades per Combination: {df['total_trades'].mean():.1f}")
    print(f"Total Trading Days Simulated: 100 days per combination")
    print(f"Trades per Day Average: {total_trades/(len(df)*100):.2f}")
    
    # Strategy Effectiveness
    print(f"\n✨ STRATEGY EFFECTIVENESS")
    print("-" * 40)
    profitable_pct = winning_combinations / len(df) * 100
    if profitable_pct >= 80:
        effectiveness = "🔥 EXCELLENT"
    elif profitable_pct >= 60:
        effectiveness = "✅ GOOD"
    elif profitable_pct >= 40:
        effectiveness = "⚠️  MODERATE"
    else:
        effectiveness = "❌ POOR"
    
    print(f"Overall Strategy Rating: {effectiveness}")
    print(f"Consistency Score: {profitable_pct:.1f}% profitable combinations")
    
    if total_profit_loss > 0:
        avg_daily_return = (total_profit_loss / (len(df) * 100)) 
        annual_return_estimate = avg_daily_return * 365
        print(f"Estimated Daily P&L: ${avg_daily_return:.2f}")
        print(f"Estimated Annual P&L: ${annual_return_estimate:,.2f}")
    
    # Recommendations
    print(f"\n💡 STRATEGIC RECOMMENDATIONS")
    print("-" * 40)
    
    best_timeframe = timeframe_analysis.index[0]
    best_symbol = symbol_analysis.index[0]
    
    print(f"1. 🎯 Best Timeframe: {best_timeframe} (${timeframe_analysis.loc[best_timeframe, 'Total_PnL']:.2f} total P&L)")
    print(f"2. 🪙 Best Symbol: {best_symbol} (${symbol_analysis.loc[best_symbol, 'Total_PnL']:.2f} total P&L)")
    print(f"3. 🚀 Optimal Combination: {top_performers.iloc[0]['symbol']} on {top_performers.iloc[0]['timeframe']} timeframe")
    
    if profitable_pct > 70:
        print(f"4. ✅ Strategy shows strong consistency - consider live implementation")
    else:
        print(f"4. ⚠️  Strategy shows mixed results - further optimization needed")
    
    print(f"5. 📊 Focus on timeframes with >60% win rate for better reliability")
    
    print("\n" + "=" * 80)
    print("📈 ANALYSIS COMPLETE")
    print("=" * 80)
    
    return df

def generate_summary_csv(df, filename_prefix):
    """Generate CSV summary files"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Overall results
    summary_file = f"{filename_prefix}_summary_{timestamp}.csv"
    df.to_csv(summary_file, index=False)
    print(f"💾 Summary saved to: {summary_file}")
    
    # Timeframe summary
    timeframe_summary = df.groupby('timeframe').agg({
        'total_trades': 'sum',
        'winning_trades': 'sum', 
        'total_profit_loss': ['sum', 'mean'],
        'win_rate': 'mean',
        'return_percent': 'mean'
    }).round(2)
    
    timeframe_file = f"{filename_prefix}_by_timeframe_{timestamp}.csv"
    timeframe_summary.to_csv(timeframe_file)
    print(f"💾 Timeframe analysis saved to: {timeframe_file}")
    
    return summary_file, timeframe_file

def main():
    """Main analysis function"""
    # Look for the most recent results file
    import glob
    import os
    
    results_files = glob.glob("backtest_ema_*_results_*.json")
    if not results_files:
        print("❌ No EMA backtest results files found")
        return
    
    # Use the most recent file
    latest_file = max(results_files, key=os.path.getctime)
    print(f"📂 Analyzing results from: {latest_file}")
    
    results = load_results(latest_file)
    if results:
        df = generate_comprehensive_analysis(results)
        
        # Generate CSV summaries
        filename_prefix = latest_file.replace('.json', '')
        generate_summary_csv(df, filename_prefix)
        
        print(f"\n🎉 Analysis complete! Check the generated CSV files for detailed data.")
    
if __name__ == "__main__":
    main()