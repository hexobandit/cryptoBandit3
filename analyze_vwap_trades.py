#!/usr/bin/env python3
"""
Trading Performance Analyzer for cBc-vwap-trader
Generates comprehensive HTML report with interactive charts and metrics
"""

import json
import os
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

def load_trade_data(mode='live'):
    """Load all trade data from VWAP trader JSON files"""
    trader_dir = f'trader_vwap{f"_{mode}" if mode != "live" else ""}'
    trades_data = {}
    position_data = {}
    
    suffix = f"_{mode}" if mode != "live" else ""
    
    # Get all symbols
    symbols = ['BTCUSDC', 'ETHUSDC', 'BNBUSDC', 'ADAUSDC', 'XRPUSDC', 
               'DOGEUSDC', 'SOLUSDC', 'PNUTUSDC', 'PEPEUSDC', 'SHIBUSDC',
               'XLMUSDC', 'LINKUSDC', 'IOTAUSDC', 'ENAUSDC']
    
    for symbol in symbols:
        # Load trades
        trades_file = os.path.join(trader_dir, f'trades_{symbol}{suffix}.json')
        if os.path.exists(trades_file):
            try:
                with open(trades_file, 'r') as f:
                    trades_data[symbol] = json.load(f)
            except:
                trades_data[symbol] = []
        else:
            trades_data[symbol] = []
            
        # Load current position
        position_file = os.path.join(trader_dir, f'position_{symbol}{suffix}.json')
        if os.path.exists(position_file):
            try:
                with open(position_file, 'r') as f:
                    position_data[symbol] = json.load(f)
            except:
                position_data[symbol] = {}
        else:
            position_data[symbol] = {}
    
    return trades_data, position_data

def calculate_metrics(trades_data, position_data):
    """Calculate comprehensive trading metrics for VWAP strategy"""
    metrics = {
        'overall': {},
        'by_symbol': {},
        'by_market_regime': defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0, 'trades': 0}),
        'by_exit_type': defaultdict(lambda: {'count': 0, 'total_pnl': 0, 'avg_profit_pct': []}),
        'timeline': [],
        'vwap_performance': {'above_vwap': 0, 'below_vwap': 0, 'crossover_success': 0}
    }
    
    # Aggregate all trades
    all_trades = []
    for symbol, trades in trades_data.items():
        for trade in trades:
            trade['symbol'] = symbol
            all_trades.append(trade)
    
    # Sort by timestamp
    all_trades.sort(key=lambda x: x['timestamp'])
    
    # Calculate overall metrics
    total_trades = len(all_trades)
    wins = [t for t in all_trades if t.get('profit_loss', 0) > 0]
    losses = [t for t in all_trades if t.get('profit_loss', 0) <= 0]
    
    win_count = len(wins)
    loss_count = len(losses)
    
    if wins:
        avg_win = sum(t['profit_loss'] for t in wins) / len(wins)
        max_win = max(t['profit_loss'] for t in wins)
    else:
        avg_win = 0
        max_win = 0
    
    if losses:
        avg_loss = sum(t['profit_loss'] for t in losses) / len(losses)
        max_loss = min(t['profit_loss'] for t in losses)
    else:
        avg_loss = 0
        max_loss = 0
    
    # Calculate total P&L from trades (for reference)
    trades_total_pnl = sum(t.get('profit_loss', 0) for t in all_trades)
    
    # Win rate
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    
    # Profit factor
    gross_profit = sum(t['profit_loss'] for t in wins) if wins else 0
    gross_loss = abs(sum(t['profit_loss'] for t in losses)) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    # Average R:R (if available in VWAP trades)
    rr_values = [t.get('risk_reward', 0) for t in all_trades if t.get('risk_reward')]
    avg_rr = statistics.mean(rr_values) if rr_values else 0
    
    # CORRECT P&L: Use cumulative P&L from position files (this is what the bot shows)
    total_pnl = sum(p.get('total_realized_pnl', 0) for p in position_data.values())
    
    metrics['overall'] = {
        'total_trades': total_trades,
        'wins': win_count,
        'losses': loss_count,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_win': max_win,
        'max_loss': max_loss,
        'total_pnl': total_pnl,
        'profit_factor': profit_factor,
        'avg_risk_reward': avg_rr,
        'current_balance': total_pnl
    }
    
    # Calculate metrics by symbol
    for symbol, trades in trades_data.items():
        symbol_wins = [t for t in trades if t.get('profit_loss', 0) > 0] if trades else []
        symbol_losses = [t for t in trades if t.get('profit_loss', 0) <= 0] if trades else []
        
        # Use cumulative P&L from position file (the real P&L)
        symbol_cumulative_pnl = position_data.get(symbol, {}).get('total_realized_pnl', 0)
        
        metrics['by_symbol'][symbol] = {
            'total_trades': len(trades) if trades else 0,
            'wins': len(symbol_wins),
            'losses': len(symbol_losses),
            'win_rate': len(symbol_wins) / len(trades) * 100 if trades else 0,
            'total_pnl': symbol_cumulative_pnl,
            'avg_pnl': statistics.mean([t.get('profit_loss', 0) for t in trades]) if trades else 0,
            'current_pnl': symbol_cumulative_pnl
        }
    
    # Analyze by market regime (VWAP specific)
    for trade in all_trades:
        market_regime = trade.get('market_regime', 'Unknown')
        if not market_regime or market_regime == 'Unknown':
            # Try to extract from reason
            reason = trade.get('reason', '')
            if 'trending' in reason.lower():
                market_regime = 'Trending'
            elif 'ranging' in reason.lower():
                market_regime = 'Ranging'
            elif 'transitioning' in reason.lower():
                market_regime = 'Transitioning'
            else:
                market_regime = 'Unknown'
        
        metrics['by_market_regime'][market_regime]['trades'] += 1
        metrics['by_market_regime'][market_regime]['total_pnl'] += trade.get('profit_loss', 0)
        
        if trade.get('profit_loss', 0) > 0:
            metrics['by_market_regime'][market_regime]['wins'] += 1
        else:
            metrics['by_market_regime'][market_regime]['losses'] += 1
    
    # Analyze by exit type
    for trade in all_trades:
        exit_reason = trade.get('reason', 'Unknown')
        
        if 'Stop Loss' in exit_reason:
            exit_type = 'Stop Loss'
        elif 'Trailing Stop' in exit_reason:
            exit_type = 'Trailing Stop'
            # Extract profit percentage if available
            try:
                import re
                match = re.search(r'\(([\d.]+)%', exit_reason)
                if match:
                    profit_pct = float(match.group(1))
                    metrics['by_exit_type'][exit_type]['avg_profit_pct'].append(profit_pct)
            except:
                pass
        elif 'TP' in exit_reason or 'Take Profit' in exit_reason:
            exit_type = 'Take Profit'
        else:
            exit_type = 'Other'
        
        metrics['by_exit_type'][exit_type]['count'] += 1
        metrics['by_exit_type'][exit_type]['total_pnl'] += trade.get('profit_loss', 0)
    
    # VWAP specific analysis
    for trade in all_trades:
        reason = trade.get('entry_reason', '')
        if 'VWAP' in reason:
            metrics['vwap_performance']['crossover_success'] += 1 if trade.get('profit_loss', 0) > 0 else 0
        
        # Check if entry was above or below VWAP (from reason or other fields)
        if 'above VWAP' in reason or 'bullish' in reason.lower():
            metrics['vwap_performance']['above_vwap'] += 1
        elif 'below VWAP' in reason:
            metrics['vwap_performance']['below_vwap'] += 1
    
    # Create timeline data
    cumulative_pnl = 0
    for trade in all_trades:
        cumulative_pnl += trade.get('profit_loss', 0)
        metrics['timeline'].append({
            'timestamp': trade['timestamp'],
            'symbol': trade['symbol'],
            'pnl': trade.get('profit_loss', 0),
            'cumulative_pnl': cumulative_pnl,
            'entry_price': trade.get('entry_price', 0),
            'exit_price': trade.get('exit_price', 0),
            'reason': trade.get('reason', ''),
            'market_regime': trade.get('market_regime', 'Unknown')
        })
    
    return metrics, all_trades

def generate_html_report(metrics, all_trades, mode='live'):
    """Generate interactive HTML report for VWAP trading"""
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VWAP Trading Performance - {mode.upper()} Mode</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/date-fns@2.29.3/index.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        h1 {{
            color: white;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .subtitle {{
            color: rgba(255,255,255,0.95);
            text-align: center;
            margin-bottom: 10px;
            font-size: 1.2em;
        }}
        
        .timestamp {{
            color: rgba(255,255,255,0.9);
            text-align: center;
            margin-bottom: 20px;
            font-size: 0.9em;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.2);
        }}
        
        .stat-label {{
            font-size: 0.85em;
            color: #666;
            margin-bottom: 5px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .stat-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
        }}
        
        .positive {{
            color: #10b981;
        }}
        
        .negative {{
            color: #ef4444;
        }}
        
        .chart-container {{
            background: white;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        
        .chart-title {{
            font-size: 1.3em;
            margin-bottom: 20px;
            color: #333;
            font-weight: 600;
        }}
        
        canvas {{
            max-height: 400px;
        }}
        
        .table-container {{
            background: white;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            overflow-x: auto;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th {{
            background: #f3f4f6;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: #374151;
            border-bottom: 2px solid #e5e7eb;
        }}
        
        td {{
            padding: 12px;
            border-bottom: 1px solid #e5e7eb;
        }}
        
        tr:hover {{
            background: #f9fafb;
        }}
        
        .filter-container {{
            margin-bottom: 20px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        
        .filter-btn {{
            padding: 8px 16px;
            border: 1px solid #d1d5db;
            border-radius: 6px;
            background: white;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .filter-btn:hover {{
            background: #f3f4f6;
        }}
        
        .filter-btn.active {{
            background: #667eea;
            color: white;
            border-color: #667eea;
        }}
        
        .symbol-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        
        .symbol-card {{
            background: white;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        
        .symbol-name {{
            font-weight: bold;
            color: #4b5563;
            margin-bottom: 8px;
        }}
        
        .symbol-pnl {{
            font-size: 1.2em;
            font-weight: bold;
        }}
        
        .symbol-winrate {{
            font-size: 0.85em;
            color: #6b7280;
            margin-top: 5px;
        }}
        
        .regime-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .regime-card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        
        .regime-title {{
            font-weight: bold;
            color: #374151;
            margin-bottom: 15px;
            font-size: 1.1em;
        }}
        
        .regime-stats {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }}
        
        .regime-stat-label {{
            color: #6b7280;
            font-size: 0.9em;
        }}
        
        .regime-stat-value {{
            font-weight: bold;
            color: #111827;
        }}
        
        .vwap-indicator {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: bold;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 8px;
            background: #e5e7eb;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #10b981 0%, #059669 100%);
            transition: width 0.5s ease;
        }}
        
        @media (max-width: 768px) {{
            .stats-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
            
            .symbol-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 VWAP Trading Performance Analysis</h1>
        <div class="subtitle">Volume Weighted Average Price Strategy with ADX Filter</div>
        <div class="timestamp">Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Mode: {mode.upper()}</div>
        
        <!-- Overall Statistics -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Trades</div>
                <div class="stat-value">{metrics['overall']['total_trades']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Win Rate</div>
                <div class="stat-value {'positive' if metrics['overall']['win_rate'] >= 50 else 'negative'}">{metrics['overall']['win_rate']:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total P&L</div>
                <div class="stat-value {'positive' if metrics['overall']['total_pnl'] > 0 else 'negative'}">${metrics['overall']['total_pnl']:.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Profit Factor</div>
                <div class="stat-value {'positive' if metrics['overall']['profit_factor'] > 1 else 'negative'}">{metrics['overall']['profit_factor']:.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Average Win</div>
                <div class="stat-value positive">${metrics['overall']['avg_win']:.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Average Loss</div>
                <div class="stat-value negative">${metrics['overall']['avg_loss']:.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Max Win</div>
                <div class="stat-value positive">${metrics['overall']['max_win']:.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Max Loss</div>
                <div class="stat-value negative">${metrics['overall']['max_loss']:.2f}</div>
            </div>
        </div>
        
        <!-- P&L Curve Chart -->
        <div class="chart-container">
            <div class="chart-title">📈 Cumulative P&L Timeline</div>
            <canvas id="pnlChart"></canvas>
        </div>
        
        <!-- Market Regime Performance -->
        <div class="chart-container">
            <div class="chart-title">🎯 Market Regime Performance</div>
            <div class="regime-grid">
                {generate_regime_cards(metrics['by_market_regime'])}
            </div>
        </div>
        
        <!-- Performance by Symbol -->
        <div class="chart-container">
            <div class="chart-title">💰 Performance by Symbol</div>
            <div class="symbol-grid">
                {generate_symbol_cards(metrics['by_symbol'])}
            </div>
        </div>
        
        <!-- Win/Loss Distribution -->
        <div class="chart-container">
            <div class="chart-title">📊 Win/Loss Distribution</div>
            <canvas id="winLossChart"></canvas>
        </div>
        
        <!-- Exit Type Analysis -->
        <div class="chart-container">
            <div class="chart-title">🚪 Exit Type Analysis</div>
            <canvas id="exitTypeChart"></canvas>
        </div>
        
        <!-- Trade Details Table -->
        <div class="table-container">
            <div class="chart-title">📋 Trade History</div>
            <div class="filter-container">
                <button class="filter-btn active" onclick="filterTrades('all')">All Trades</button>
                <button class="filter-btn" onclick="filterTrades('wins')">Winners</button>
                <button class="filter-btn" onclick="filterTrades('losses')">Losers</button>
                <button class="filter-btn" onclick="sortTable('timestamp')">Sort by Date</button>
                <button class="filter-btn" onclick="sortTable('profit_loss')">Sort by P&L</button>
                <button class="filter-btn" onclick="sortTable('symbol')">Sort by Symbol</button>
            </div>
            <table id="tradesTable">
                <thead>
                    <tr>
                        <th>Date/Time</th>
                        <th>Symbol</th>
                        <th>Entry</th>
                        <th>Exit</th>
                        <th>Quantity</th>
                        <th>P&L</th>
                        <th>%</th>
                        <th>Regime</th>
                        <th>Exit Type</th>
                    </tr>
                </thead>
                <tbody id="tradesTableBody">
                    {generate_trades_table(all_trades)}
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        // P&L Timeline Chart
        const pnlCtx = document.getElementById('pnlChart').getContext('2d');
        const pnlData = {json.dumps(metrics['timeline'])};
        
        new Chart(pnlCtx, {{
            type: 'line',
            data: {{
                labels: pnlData.map(d => d.timestamp.split('T')[0]),
                datasets: [{{
                    label: 'Cumulative P&L',
                    data: pnlData.map(d => d.cumulative_pnl),
                    borderColor: pnlData[pnlData.length - 1]?.cumulative_pnl > 0 ? '#10b981' : '#ef4444',
                    backgroundColor: pnlData[pnlData.length - 1]?.cumulative_pnl > 0 ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                    tension: 0.4,
                    fill: true
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{
                            label: (context) => `P&L: ${{context.parsed.y.toFixed(2)}}`
                        }}
                    }}
                }},
                scales: {{
                    y: {{
                        ticks: {{
                            callback: (value) => '$' + value.toFixed(0)
                        }}
                    }}
                }}
            }}
        }});
        
        // Win/Loss Distribution Chart
        const winLossCtx = document.getElementById('winLossChart').getContext('2d');
        new Chart(winLossCtx, {{
            type: 'doughnut',
            data: {{
                labels: ['Wins', 'Losses'],
                datasets: [{{
                    data: [{metrics['overall']['wins']}, {metrics['overall']['losses']}],
                    backgroundColor: ['#10b981', '#ef4444'],
                    borderWidth: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'right'
                    }}
                }}
            }}
        }});
        
        // Exit Type Chart
        const exitTypeCtx = document.getElementById('exitTypeChart').getContext('2d');
        const exitTypeData = {json.dumps({k: v['count'] for k, v in metrics['by_exit_type'].items()})};
        
        new Chart(exitTypeCtx, {{
            type: 'bar',
            data: {{
                labels: Object.keys(exitTypeData),
                datasets: [{{
                    label: 'Exit Count',
                    data: Object.values(exitTypeData),
                    backgroundColor: '#667eea'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }}
            }}
        }});
        
        // Trade filtering and sorting
        let allTrades = {json.dumps(all_trades)};
        let currentFilter = 'all';
        
        function filterTrades(type) {{
            currentFilter = type;
            const buttons = document.querySelectorAll('.filter-btn');
            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            let filteredTrades = allTrades;
            
            if (type === 'wins') {{
                filteredTrades = allTrades.filter(t => t.profit_loss > 0);
            }} else if (type === 'losses') {{
                filteredTrades = allTrades.filter(t => t.profit_loss <= 0);
            }}
            
            updateTable(filteredTrades);
        }}
        
        function sortTable(column) {{
            let filteredTrades = allTrades;
            
            if (currentFilter === 'wins') {{
                filteredTrades = allTrades.filter(t => t.profit_loss > 0);
            }} else if (currentFilter === 'losses') {{
                filteredTrades = allTrades.filter(t => t.profit_loss <= 0);
            }}
            
            filteredTrades.sort((a, b) => {{
                if (column === 'timestamp') {{
                    return new Date(b.timestamp) - new Date(a.timestamp);
                }} else if (column === 'profit_loss') {{
                    return b.profit_loss - a.profit_loss;
                }} else if (column === 'symbol') {{
                    return a.symbol.localeCompare(b.symbol);
                }}
            }});
            
            updateTable(filteredTrades);
        }}
        
        function updateTable(trades) {{
            const tbody = document.getElementById('tradesTableBody');
            tbody.innerHTML = trades.map(trade => {{
                const profit_pct = ((trade.exit_price - trade.entry_price) / trade.entry_price * 100).toFixed(2);
                const pnlClass = trade.profit_loss > 0 ? 'positive' : 'negative';
                const regime = trade.market_regime || 'Unknown';
                
                return `
                    <tr>
                        <td>${{trade.timestamp.replace('T', ' ').substring(0, 19)}}</td>
                        <td>${{trade.symbol}}</td>
                        <td>${{trade.entry_price.toFixed(6)}}</td>
                        <td>${{trade.exit_price.toFixed(6)}}</td>
                        <td>${{trade.quantity.toFixed(2)}}</td>
                        <td class="${{pnlClass}}">${{trade.profit_loss > 0 ? '+' : ''}}${{trade.profit_loss.toFixed(2)}}</td>
                        <td class="${{pnlClass}}">${{profit_pct}}%</td>
                        <td><span class="vwap-indicator">${{regime}}</span></td>
                        <td>${{trade.reason || '-'}}</td>
                    </tr>
                `;
            }}).join('');
        }}
    </script>
</body>
</html>
"""
    
    return html_content

def generate_symbol_cards(symbol_data):
    """Generate HTML for symbol performance cards"""
    cards = []
    for symbol, data in sorted(symbol_data.items(), key=lambda x: x[1]['total_pnl'], reverse=True):
        pnl_class = 'positive' if data['total_pnl'] > 0 else 'negative'
        winrate_color = '#10b981' if data['win_rate'] >= 50 else '#ef4444'
        
        card = f"""
        <div class="symbol-card">
            <div class="symbol-name">{symbol}</div>
            <div class="symbol-pnl {pnl_class}">${data['total_pnl']:.2f}</div>
            <div class="symbol-winrate">Win Rate: <span style="color: {winrate_color}">{data['win_rate']:.1f}%</span></div>
            <div class="symbol-winrate">{data['wins']}/{data['total_trades']} trades won</div>
        </div>
        """
        cards.append(card)
    
    return ''.join(cards)

def generate_regime_cards(regime_data):
    """Generate HTML for market regime performance cards"""
    cards = []
    for regime, data in sorted(regime_data.items(), key=lambda x: x[1]['trades'], reverse=True):
        if data['trades'] == 0:
            continue
            
        win_rate = (data['wins'] / data['trades'] * 100) if data['trades'] > 0 else 0
        avg_pnl = data['total_pnl'] / data['trades'] if data['trades'] > 0 else 0
        
        card = f"""
        <div class="regime-card">
            <div class="regime-title">{regime} Market</div>
            <div class="regime-stats">
                <span class="regime-stat-label">Total Trades:</span>
                <span class="regime-stat-value">{data['trades']}</span>
            </div>
            <div class="regime-stats">
                <span class="regime-stat-label">Win Rate:</span>
                <span class="regime-stat-value {'positive' if win_rate >= 50 else 'negative'}">{win_rate:.1f}%</span>
            </div>
            <div class="regime-stats">
                <span class="regime-stat-label">Total P&L:</span>
                <span class="regime-stat-value {'positive' if data['total_pnl'] > 0 else 'negative'}">${data['total_pnl']:.2f}</span>
            </div>
            <div class="regime-stats">
                <span class="regime-stat-label">Avg P&L:</span>
                <span class="regime-stat-value {'positive' if avg_pnl > 0 else 'negative'}">${avg_pnl:.2f}</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {win_rate}%; background: {'linear-gradient(90deg, #10b981 0%, #059669 100%)' if win_rate >= 50 else 'linear-gradient(90deg, #ef4444 0%, #dc2626 100%)'}"></div>
            </div>
        </div>
        """
        cards.append(card)
    
    return ''.join(cards)

def generate_trades_table(trades):
    """Generate HTML for trades table"""
    rows = []
    for trade in sorted(trades, key=lambda x: x['timestamp'], reverse=True)[:50]:  # Show last 50 trades
        profit_pct = ((trade['exit_price'] - trade['entry_price']) / trade['entry_price'] * 100)
        pnl_class = 'positive' if trade.get('profit_loss', 0) > 0 else 'negative'
        regime = trade.get('market_regime', 'Unknown')
        
        row = f"""
        <tr>
            <td>{trade['timestamp'].replace('T', ' ')[:19]}</td>
            <td>{trade['symbol']}</td>
            <td>{trade['entry_price']:.6f}</td>
            <td>{trade['exit_price']:.6f}</td>
            <td>{trade['quantity']:.2f}</td>
            <td class="{pnl_class}">{'+ ' if trade.get('profit_loss', 0) > 0 else ''}{trade.get('profit_loss', 0):.2f}</td>
            <td class="{pnl_class}">{profit_pct:.2f}%</td>
            <td><span class="vwap-indicator">{regime}</span></td>
            <td>{trade.get('reason', '-')}</td>
        </tr>
        """
        rows.append(row)
    
    return ''.join(rows)

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze VWAP trading performance')
    parser.add_argument('--mode', default='live', choices=['live', 'dry'], help='Analysis mode')
    parser.add_argument('--output', default='vwap_analysis.html', help='Output HTML file')
    
    args = parser.parse_args()
    
    print(f"📊 Loading VWAP trade data for {args.mode} mode...")
    trades_data, position_data = load_trade_data(args.mode)
    
    print("📈 Calculating VWAP metrics...")
    metrics, all_trades = calculate_metrics(trades_data, position_data)
    
    print("🎨 Generating VWAP HTML report...")
    html_content = generate_html_report(metrics, all_trades, args.mode)
    
    # Save HTML file
    with open(args.output, 'w') as f:
        f.write(html_content)
    
    print(f"✅ VWAP Analysis complete! Report saved to {args.output}")
    print(f"\n📊 Quick Summary:")
    print(f"   Total Trades: {metrics['overall']['total_trades']}")
    print(f"   Win Rate: {metrics['overall']['win_rate']:.1f}%")
    print(f"   Total P&L: ${metrics['overall']['total_pnl']:.2f}")
    print(f"   Profit Factor: {metrics['overall']['profit_factor']:.2f}")
    
    # Show market regime breakdown if available
    if metrics['by_market_regime']:
        print(f"\n📈 Market Regime Performance:")
        for regime, data in metrics['by_market_regime'].items():
            if data['trades'] > 0:
                win_rate = (data['wins'] / data['trades'] * 100)
                print(f"   {regime}: {data['trades']} trades, {win_rate:.1f}% win rate, ${data['total_pnl']:.2f} P&L")

if __name__ == "__main__":
    main()