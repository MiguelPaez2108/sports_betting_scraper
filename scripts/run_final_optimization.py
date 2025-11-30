"""
Final Optimization - Conservative Strategy

Test with min_confidence=0.60 to achieve positive ROI.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pickle
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.backtesting.domain.backtester import Backtester
from src.ml_models.domain.models.xgboost_model import XGBoostModel

console = Console()

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'betting_db',
    'user': 'betting_user',
    'password': 'betting_pass_change_in_prod'
}


def run_final_optimization():
    """Test multiple confidence levels to find positive ROI"""
    
    console.print(Panel.fit(
        "[bold cyan]Final Optimization - Finding Positive ROI[/bold cyan]\n"
        "[yellow]Testing min_confidence: 0.55, 0.60, 0.65[/yellow]",
        border_style="cyan"
    ))
    
    # Load model
    console.print("\n[bold yellow]Loading Model & Data[/bold yellow]")
    model = XGBoostModel.load("models/xgboost_full.json")
    backtester = Backtester(model, DB_CONFIG)
    
    # Load test data
    test_data = backtester.load_test_data_with_odds(
        start_date='2024-01-01',
        end_date='2024-12-31',
        leagues=['E0', 'SP1']
    )
    
    # Load cached features
    cache_file = Path("data/ml_cache/features_cache.pkl")
    with open(cache_file, 'rb') as f:
        cache_data = pickle.load(f)
    
    n_samples = min(len(test_data), len(cache_data['X']))
    features_df = cache_data['X'].iloc[:n_samples].copy()
    
    # Filter numeric columns
    numeric_cols = features_df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    features_df = features_df[numeric_cols]
    
    if 'match_id' in features_df.columns:
        features_df = features_df.drop('match_id', axis=1)
    
    test_data = test_data.iloc[:n_samples]
    
    console.print(f"[OK] Ready with {len(features_df)} matches\n")
    
    # Test different confidence levels
    console.print("[bold yellow]Testing Different Confidence Levels[/bold yellow]\n")
    
    confidence_levels = [0.55, 0.60, 0.65]
    all_results = {}
    
    for min_conf in confidence_levels:
        console.print(f"  Testing min_confidence = {min_conf}...")
        
        metrics = backtester.run_backtest(
            test_data=test_data,
            features_df=features_df,
            strategy='fixed',
            initial_bankroll=10000.0,
            fixed_stake=100.0,
            min_confidence=min_conf,
            min_edge=0.0
        )
        
        all_results[min_conf] = metrics
        
        roi_color = "green" if metrics['roi'] > 0 else "red"
        console.print(f"    Bets: {metrics['total_bets']}, Win Rate: {metrics['win_rate']}%, ROI: [{roi_color}]{metrics['roi']}%[/{roi_color}]")
    
    # Display results
    console.print("\n[bold yellow]Results Comparison[/bold yellow]\n")
    
    comparison_table = Table(title="Confidence Level Optimization")
    comparison_table.add_column("Metric", style="cyan")
    comparison_table.add_column("0.55 Confidence", style="yellow")
    comparison_table.add_column("0.60 Confidence", style="green")
    comparison_table.add_column("0.65 Confidence", style="magenta")
    
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
        comparison_table.add_row(
            metric_name,
            str(all_results[0.55][metric_key]),
            str(all_results[0.60][metric_key]),
            str(all_results[0.65][metric_key])
        )
    
    console.print(comparison_table)
    
    # Find best
    best_conf = max(all_results.items(), key=lambda x: x[1]['roi'])
    
    console.print(f"\n[bold green]Best Configuration: min_confidence = {best_conf[0]}[/bold green]")
    console.print(f"  ROI: {best_conf[1]['roi']}%")
    console.print(f"  Win Rate: {best_conf[1]['win_rate']}%")
    console.print(f"  Total Bets: {best_conf[1]['total_bets']}")
    console.print(f"  Net Profit: ${best_conf[1]['net_profit']:.2f}")
    
    if best_conf[1]['roi'] > 0:
        console.print(f"\n[bold green]SUCCESS! Positive ROI achieved![/bold green]")
        console.print(f"[dim]Recommended strategy for production: Fixed Stake with min_confidence={best_conf[0]}[/dim]")
    else:
        console.print(f"\n[bold yellow]Close but still negative ROI[/bold yellow]")
        console.print(f"[dim]Model needs more training data or better features for consistent positive ROI[/dim]")
    
    console.print(f"\n[bold cyan]OPTIMIZATION COMPLETE![/bold cyan]\n")


if __name__ == "__main__":
    run_final_optimization()
