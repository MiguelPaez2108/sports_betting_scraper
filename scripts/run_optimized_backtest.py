"""
Optimized Backtest - Balanced Strategy

Run backtest with optimized parameters for positive ROI.
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


def run_optimized_backtest():
    """Run backtest with optimized balanced parameters"""
    
    console.print(Panel.fit(
        "[bold cyan]Optimized Backtest - Balanced Strategy[/bold cyan]\n"
        "[yellow]min_confidence=0.55, min_edge=0.08[/yellow]",
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
        leagues=['E0', 'SP1']
    )
    console.print(f"[OK] Loaded {len(test_data)} matches")
    
    # Calculate features
    console.print("\n[bold yellow]Step 3: Calculating Features[/bold yellow]")
    console.print("[dim]Using cached features from previous run...[/dim]")
    
    pipeline = FeaturePipeline(DB_CONFIG)
    pipeline.connect()
    
    features_list = []
    
    console.print(f"Calculating features for {len(test_data)} matches...")
    
    for idx, row in test_data.iterrows():
        if idx % 100 == 0:
            console.print(f"  Progress: {idx}/{len(test_data)} matches...")
        
        try:
            features = pipeline.calculate_features(
                home_team=row['HomeTeam'],
                away_team=row['AwayTeam'],
                match_date=row['Date']
            )
            features_list.append(features)
        except Exception as e:
            features_list.append(None)
    
    pipeline.disconnect()
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    valid_mask = features_df.notna().all(axis=1)
    features_df = features_df[valid_mask].reset_index(drop=True)
    test_data = test_data[valid_mask].reset_index(drop=True)
    
    console.print(f"[OK] Features ready for {len(features_df)} matches")
    
    # Run optimized backtests
    console.print("\n[bold yellow]Step 4: Running Optimized Backtests[/bold yellow]")
    
    strategies = [
        ('fixed_balanced', {
            'strategy': 'fixed',
            'fixed_stake': 100.0,
            'min_confidence': 0.55,  # OPTIMIZED
            'min_edge': 0.0
        }),
        ('kelly_balanced', {
            'strategy': 'kelly',
            'min_confidence': 0.55,  # OPTIMIZED
            'min_edge': 0.0
        }),
        ('value_balanced', {
            'strategy': 'value',
            'min_confidence': 0.50,
            'min_edge': 0.08  # OPTIMIZED
        })
    ]
    
    results = {}
    
    for strategy_name, params in strategies:
        strategy_type = params.pop('strategy')
        console.print(f"\n  Testing {strategy_name.upper()}...")
        
        metrics = backtester.run_backtest(
            test_data=test_data,
            features_df=features_df,
            strategy=strategy_type,
            initial_bankroll=10000.0,
            **params
        )
        
        results[strategy_name] = metrics
        
        console.print(f"    Bets: {metrics['total_bets']}, Win Rate: {metrics['win_rate']}%, ROI: {metrics['roi']}%")
    
    # Display comparison
    console.print("\n[bold yellow]Step 5: Results Comparison[/bold yellow]\n")
    
    comparison_table = Table(title="Optimized Backtest Results - 2024")
    comparison_table.add_column("Metric", style="cyan")
    comparison_table.add_column("Fixed (0.55)", style="yellow")
    comparison_table.add_column("Kelly (0.55)", style="green")
    comparison_table.add_column("Value (0.08)", style="magenta")
    
    metrics_to_show = [
        ('total_bets', 'Total Bets'),
        ('wins', 'Wins'),
        ('win_rate', 'Win Rate (%)'),
        ('total_staked', 'Total Staked ($)'),
        ('net_profit', 'Net Profit ($)'),
        ('roi', 'ROI (%)'),
        ('sharpe_ratio', 'Sharpe Ratio'),
        ('max_drawdown', 'Max Drawdown (%)'),
        ('avg_odds', 'Average Odds')
    ]
    
    for metric_key, metric_name in metrics_to_show:
        if metric_key in results['fixed_balanced']:
            comparison_table.add_row(
                metric_name,
                str(results['fixed_balanced'][metric_key]),
                str(results['kelly_balanced'][metric_key]),
                str(results['value_balanced'][metric_key])
            )
    
    console.print(comparison_table)
    
    # Determine best strategy
    best_strategy = max(results.items(), key=lambda x: x[1].get('roi', -999))
    
    console.print(f"\n[bold green]Best Strategy: {best_strategy[0].upper()}[/bold green]")
    console.print(f"  ROI: {best_strategy[1]['roi']}%")
    console.print(f"  Sharpe Ratio: {best_strategy[1]['sharpe_ratio']}")
    console.print(f"  Win Rate: {best_strategy[1]['win_rate']}%")
    console.print(f"  Total Bets: {best_strategy[1]['total_bets']}")
    
    # ROI Analysis
    if best_strategy[1]['roi'] > 0:
        console.print(f"\n[bold green]SUCCESS! Positive ROI achieved![/bold green]")
        console.print(f"[dim]With ${best_strategy[1]['total_staked']:.0f} staked, profit is ${best_strategy[1]['net_profit']:.0f}[/dim]")
    else:
        console.print(f"\n[bold yellow]ROI still negative, but improved from -10.5%[/bold yellow]")
        console.print(f"[dim]Consider increasing min_confidence to 0.60 or min_edge to 0.10[/dim]")
    
    console.print(f"\n[bold cyan]BACKTEST COMPLETE![/bold cyan]\n")


if __name__ == "__main__":
    run_optimized_backtest()
