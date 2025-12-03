"""
Generate daily performance report.

Usage:
    python scripts/generate_daily_report.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json


def main():
    print("=" * 60)
    print(f"DAILY REPORT - {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 60)
    
    # Load recent trades
    trades_path = Path('data/trades/recent_trades.csv')
    if not trades_path.exists():
        print("\n⚠️  No recent trades found")
        return
    
    trades_df = pd.read_csv(trades_path)
    trades_df['date'] = pd.to_datetime(trades_df['date'])
    
    # Filter last 30 days
    cutoff_date = datetime.now() - timedelta(days=30)
    recent_trades = trades_df[trades_df['date'] >= cutoff_date]
    
    if len(recent_trades) == 0:
        print("\n⚠️  No trades in last 30 days")
        return
    
    # Calculate metrics
    print(f"\n📊 PERFORMANCE METRICS (Last 30 days)")
    print("=" * 60)
    
    total_profit = recent_trades['profit'].sum()
    total_stake = recent_trades['stake'].sum()
    roi = (total_profit / total_stake) * 100 if total_stake > 0 else 0
    
    wins = len(recent_trades[recent_trades['result'] == 'WIN'])
    losses = len(recent_trades[recent_trades['result'] == 'LOSS'])
    win_rate = (wins / len(recent_trades)) * 100
    
    avg_odds = recent_trades['odds'].mean()
    avg_stake = recent_trades['stake'].mean()
    
    print(f"Total Bets: {len(recent_trades)}")
    print(f"Wins: {wins} | Losses: {losses}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Total Profit: ${total_profit:+.2f}")
    print(f"ROI: {roi:+.2f}%")
    print(f"Avg Odds: {avg_odds:.2f}")
    print(f"Avg Stake: ${avg_stake:.2f}")
    
    # Sharpe ratio (simplified)
    if len(recent_trades) > 1:
        returns = recent_trades['profit'] / recent_trades['stake']
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        print(f"Sharpe Ratio: {sharpe:.2f}")
    
    # Daily breakdown
    print(f"\n📅 DAILY BREAKDOWN (Last 7 days)")
    print("=" * 60)
    
    last_7_days = datetime.now() - timedelta(days=7)
    recent_7d = recent_trades[recent_trades['date'] >= last_7_days]
    
    daily_stats = recent_7d.groupby(recent_7d['date'].dt.date).agg({
        'profit': 'sum',
        'stake': 'sum',
        'result': lambda x: (x == 'WIN').sum()
    })
    
    for date, row in daily_stats.iterrows():
        daily_roi = (row['profit'] / row['stake']) * 100 if row['stake'] > 0 else 0
        print(f"{date}: {row['result']} bets, ${row['profit']:+.2f} ({daily_roi:+.1f}%)")
    
    # Alerts
    print(f"\n🚨 ALERTS")
    print("=" * 60)
    
    alerts = []
    
    if roi < -5:
        alerts.append(f"⚠️  CRITICAL: ROI below -5% ({roi:.1f}%)")
    
    if win_rate < 35:
        alerts.append(f"⚠️  WARNING: Win rate below 35% ({win_rate:.1f}%)")
    
    if len(recent_7d) < 5:
        alerts.append(f"⚠️  INFO: Low bet volume in last 7 days ({len(recent_7d)} bets)")
    
    if alerts:
        for alert in alerts:
            print(alert)
    else:
        print("✅ No alerts")
    
    # Save report
    report_dir = Path('reports/daily')
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_data = {
        'date': datetime.now().isoformat(),
        'total_bets': len(recent_trades),
        'wins': int(wins),
        'losses': int(losses),
        'win_rate': float(win_rate),
        'total_profit': float(total_profit),
        'roi': float(roi),
        'avg_odds': float(avg_odds),
        'alerts': alerts
    }
    
    report_path = report_dir / f"report_{datetime.now().strftime('%Y%m%d')}.json"
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n✓ Report saved to {report_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
