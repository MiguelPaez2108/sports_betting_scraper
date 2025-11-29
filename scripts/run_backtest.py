"""
Run Backtest

Execute backtesting with trained XGBoost model and real Bet365 odds.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.backtesting.domain.backtester import Backtester
from src.ml_models.domain.models.xgboost_model import XGBoostModel
from src.ml_models.infrastructure.data_loader import MLDataLoader

console = Console()

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'betting_db',
    'user': 'betting_user',
    'password': 'betting_pass_change_in_prod'
}


def run_backtest():
    """Run backtest on 2024 data"""
    
    console.print(Panel.fit(
        "[bold cyan]Backtesting with Real Bet365 Odds[/bold cyan]\n"
        "[yellow]Testing Period: 2024[/yellow]",
        border_style="cyan"
    ))
    
    # Load trained model
    console.print("\n[bold yellow]Step 1: Loading Trained Model[/bold yellow]")
    model = XGBoostModel.load("models/xgboost_full.json")
    console.print("[OK] Model loaded")
    
    # Initialize backtester
    backtester = Backtester(model, DB_CONFIG)
    
    # Load test data with odds
    console.print("\n[bold yellow]Step 2: Loading 2024 Matches with Odds[/bold yellow]")
    test_data = backtester.load_test_data_with_odds(
        start_date='2024-01-01',
        end_date='2024-12-31',
        leagues=['E0', 'SP1', 'I1', 'D1', 'F1']
    )
    console.print(f"[OK] Loaded {len(test_data)} matches")
    
    # Calculate features for test data
    console.print("\n[bold yellow]Step 3: Calculating Features[/bold yellow]")
    console.print("[dim]This may take a few minutes...[/dim]")
    
    with MLDataLoader(DB_CONFIG) as loader:
        # We need to calculate features for these specific matches
        # For now, use a simplified approach - load from cache if available
        data = loader.prepare_training_data(
            start_date='2024-01-01',
            end_date='2024-12-31',
            test_size=0.0,  # Use all as test
            val_size=0.0,
            use_cache=True,
            limit=len(test_data)
        )
        features_df = data['X_train']
    
    console.print(f"[OK] Features calculated: {len(features_df)} samples")
    
    # Run backtests with different strategies
    strategies = [
        ('fixed', {'fixed_stake': 100.0}),
        ('kelly', {}),
        ('value', {'min_edge': 0.05})
    ]
    
    results = {}
    
    for strategy_name, params in strategies:
        console.print(f"\n[bold yellow]Running {strategy_name.upper()} Strategy[/bold yellow]")
        
        metrics = backtester.run_backtest(
            test_data=test_data,
            features_df=features_df,
            strategy=strategy_name,
            initial_bankroll=10000.0,
            **params
        )
        
        results[strategy_name] = metrics
        
        console.print(f"Bets placed: {metrics['total_bets']}")
        console.print(f"Win rate: {metrics['win_rate']}%")
        console.print(f"ROI: {metrics['roi']}%")
    
    # Display comparison
    console.print("\n[bold yellow]Strategy Comparison[/bold yellow]")
    
    comparison_table = Table(title="Backtest Results - 2024")
    comparison_table.add_column("Metric", style="cyan")
    comparison_table.add_column("Fixed Stake", style="yellow")
    comparison_table.add_column("Kelly Criterion", style="green")
    comparison_table.add_column("Value Betting", style="magenta")
    
    metrics_to_show = [
        'total_bets', 'wins', 'win_rate', 'total_staked',
        'net_profit', 'roi', 'sharpe_ratio', 'max_drawdown',
        'longest_win_streak', 'longest_loss_streak'
    ]
    
    for metric in metrics_to_show:
        if metric in results['fixed']:
            comparison_table.add_row(
                metric.replace('_', ' ').title(),
                str(results['fixed'][metric]),
                str(results['kelly'][metric]),
                str(results['value'][metric])
            )
    
    console.print(comparison_table)
    
    # Determine best strategy
    best_strategy = max(results.items(), key=lambda x: x[1]['roi'])
    
    console.print(f"\n[bold green]Best Strategy: {best_strategy[0].upper()}[/bold green]")
    console.print(f"ROI: {best_strategy[1]['roi']}%")
    console.print(f"Sharpe Ratio: {best_strategy[1]['sharpe_ratio']}")
    
    console.print(f"\n[bold green]SUCCESS![/bold green]")
    console.print(f"[dim]Backtesting complete with real historical odds[/dim]\n")


if __name__ == "__main__":
    run_backtest()
