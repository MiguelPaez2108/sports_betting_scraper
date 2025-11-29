"""
Backtest with Correct Features

Calculate features for specific 2024 matches and run backtest.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.backtesting.domain.backtester import Backtester
from src.ml_models.domain.models.xgboost_model import XGBoostModel
from src.feature_engineering.application.feature_pipeline import FeaturePipeline

console = Console()

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'betting_db',
    'user': 'betting_user',
    'password': 'betting_pass_change_in_prod'
}


def run_correct_backtest():
    """Run backtest with correctly calculated features"""
    
    console.print(Panel.fit(
        "[bold cyan]Backtesting with Real Bet365 Odds[/bold cyan]\n"
        "[yellow]Calculating Correct Features for 2024[/yellow]",
        border_style="cyan"
    ))
    
    # Load trained model
    console.print("\n[bold yellow]Step 1: Loading Trained Model[/bold yellow]")
    model = XGBoostModel.load("models/xgboost_full.json")
    console.print("[OK] Model loaded")
    
    # Initialize backtester
    backtester = Backtester(model, DB_CONFIG)
    
    # Load test data with odds from 2024
    console.print("\n[bold yellow]Step 2: Loading 2024 Matches with Odds[/bold yellow]")
    test_data = backtester.load_test_data_with_odds(
        start_date='2024-01-01',
        end_date='2024-12-31',
        leagues=['E0', 'SP1']  # Premier League + La Liga
    )
    console.print(f"[OK] Loaded {len(test_data)} matches")
    
    # Calculate features for these specific matches
    console.print("\n[bold yellow]Step 3: Calculating Features for 2024 Matches[/bold yellow]")
    console.print("[dim]This will take a few minutes...[/dim]")
    
    # Initialize feature pipeline
    pipeline = FeaturePipeline(DB_CONFIG)
    pipeline.connect()
    
    features_list = []
    
    console.print(f"Calculating features for {len(test_data)} matches...")
    
    for idx, row in test_data.iterrows():
        if idx % 50 == 0:
            console.print(f"  Progress: {idx}/{len(test_data)} matches...")
        
        try:
            # Calculate features with correct signature
            features = pipeline.calculate_features(
                home_team=row['HomeTeam'],
                away_team=row['AwayTeam'],
                match_date=row['Date']
            )
            features_list.append(features)
            
        except Exception as e:
            console.print(f"[red]Error for {row['HomeTeam']} vs {row['AwayTeam']}: {str(e)[:100]}[/red]")
            features_list.append(None)
    
    pipeline.disconnect()
    
    # Convert to DataFrame and filter valid features
    features_df = pd.DataFrame(features_list)
    valid_mask = features_df.notna().all(axis=1)
    features_df = features_df[valid_mask]
    test_data = test_data[valid_mask].reset_index(drop=True)
    features_df = features_df.reset_index(drop=True)
    
    console.print(f"[OK] Calculated features for {len(features_df)} matches")
    
    # Run backtests with different strategies
    console.print("\n[bold yellow]Step 4: Running Backtests[/bold yellow]")
    
    strategies = [
        ('fixed', {'fixed_stake': 100.0, 'min_confidence': 0.4}),
        ('kelly', {'min_confidence': 0.4}),
        ('value', {'min_edge': 0.05, 'min_confidence': 0.4})
    ]
    
    results = {}
    
    for strategy_name, params in strategies:
        console.print(f"\n  Testing {strategy_name.upper()} strategy...")
        
        metrics = backtester.run_backtest(
            test_data=test_data,
            features_df=features_df,
            strategy=strategy_name,
            initial_bankroll=10000.0,
            **params
        )
        
        results[strategy_name] = metrics
        
        console.print(f"    Bets: {metrics['total_bets']}, Win Rate: {metrics['win_rate']}%, ROI: {metrics['roi']}%")
    
    # Display comparison
    console.print("\n[bold yellow]Step 5: Results Comparison[/bold yellow]\n")
    
    comparison_table = Table(title=f"Backtest Results - 2024 ({len(test_data)} matches)")
    comparison_table.add_column("Metric", style="cyan")
    comparison_table.add_column("Fixed Stake", style="yellow")
    comparison_table.add_column("Kelly Criterion", style="green")
    comparison_table.add_column("Value Betting", style="magenta")
    
    metrics_to_show = [
        ('total_bets', 'Total Bets'),
        ('wins', 'Wins'),
        ('win_rate', 'Win Rate (%)'),
        ('total_staked', 'Total Staked ($)'),
        ('net_profit', 'Net Profit ($)'),
        ('roi', 'ROI (%)'),
        ('sharpe_ratio', 'Sharpe Ratio'),
        ('max_drawdown', 'Max Drawdown (%)'),
        ('longest_win_streak', 'Longest Win Streak'),
        ('longest_loss_streak', 'Longest Loss Streak'),
        ('avg_odds', 'Average Odds')
    ]
    
    for metric_key, metric_name in metrics_to_show:
        if metric_key in results['fixed']:
            comparison_table.add_row(
                metric_name,
                str(results['fixed'][metric_key]),
                str(results['kelly'][metric_key]),
                str(results['value'][metric_key])
            )
    
    console.print(comparison_table)
    
    # Determine best strategy
    best_strategy = max(results.items(), key=lambda x: x[1].get('roi', -999))
    
    console.print(f"\n[bold green]Best Strategy: {best_strategy[0].upper()}[/bold green]")
    console.print(f"  ROI: {best_strategy[1]['roi']}%")
    console.print(f"  Sharpe Ratio: {best_strategy[1]['sharpe_ratio']}")
    console.print(f"  Win Rate: {best_strategy[1]['win_rate']}%")
    
    # Summary
    console.print(f"\n[bold cyan]BACKTEST COMPLETE![/bold cyan]")
    console.print(f"[dim]Tested with real Bet365 historical odds from 2024[/dim]")
    console.print(f"[dim]Features calculated specifically for test matches[/dim]\n")


if __name__ == "__main__":
    run_correct_backtest()
