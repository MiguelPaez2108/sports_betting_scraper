"""
Train with Complete Dataset (60K matches)

Train models with all available data including Champions League.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.ml_models.infrastructure.data_loader import MLDataLoader
from src.ml_models.domain.models.logistic_regression import LogisticRegressionModel
from src.ml_models.domain.models.xgboost_model import XGBoostModel

console = Console()

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'betting_db',
    'user': 'betting_user',
    'password': 'betting_pass_change_in_prod'
}


def train_complete_dataset():
    """Train models with complete 60K dataset"""
    
    console.print(Panel.fit(
        "[bold cyan]Training ML Models - COMPLETE DATASET[/bold cyan]\n"
        "[yellow]~60,000 matches (1993-2025) + Champions League[/yellow]",
        border_style="cyan"
    ))
    
    # Load complete data
    console.print("\n[bold yellow]Step 1: Loading Complete Dataset[/bold yellow]")
    console.print("[dim]This will take 1-2 hours for feature calculation...[/dim]\n")
    
    with MLDataLoader(DB_CONFIG) as loader:
        data = loader.prepare_training_data(
            start_date='1993-01-01',  # From 1993
            end_date='2025-12-31',    # To current season
            leagues=['E0', 'SP1', 'I1', 'D1', 'F1', 'CL'],  # Include Champions League
            test_size=0.15,
            val_size=0.15,
            batch_size=100,
            use_cache=True,
            limit=None  # NO LIMIT - use ALL data
        )
    
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    console.print(f"\n[bold green]Data loaded successfully![/bold green]")
    console.print(f"Train: {len(X_train)} samples")
    console.print(f"Val: {len(X_val)} samples")
    console.print(f"Test: {len(X_test)} samples")
    console.print(f"Features: {len(data['feature_names'])}")
    
    # Target distribution
    console.print(f"\n[bold]Target Distribution (Test Set):[/bold]")
    target_counts = y_test.value_counts()
    target_names = {2: 'Home Win', 1: 'Draw', 0: 'Away Win'}
    
    dist_table = Table()
    dist_table.add_column("Outcome", style="cyan")
    dist_table.add_column("Count", style="green")
    dist_table.add_column("Percentage", style="yellow")
    
    for target_val in [2, 1, 0]:
        if target_val in target_counts.index:
            count = target_counts[target_val]
            pct = count / len(y_test) * 100
            dist_table.add_row(
                target_names[target_val],
                str(count),
                f"{pct:.1f}%"
            )
    
    console.print(dist_table)
    
    # Train Logistic Regression
    console.print("\n[bold yellow]Step 2: Training Logistic Regression[/bold yellow]")
    
    lr_model = LogisticRegressionModel(C=1.0, max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    lr_metrics = lr_model.evaluate(X_test, y_test)
    
    console.print(f"Test Accuracy: {lr_metrics['accuracy']:.4f}")
    console.print(f"Test Log Loss: {lr_metrics['log_loss']:.4f}")
    
    # Train XGBoost
    console.print("\n[bold yellow]Step 3: Training XGBoost[/bold yellow]")
    
    xgb_model = XGBoostModel(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8
    )
    xgb_model.fit(X_train, y_train, verbose=False)
    
    xgb_metrics = xgb_model.evaluate(X_test, y_test)
    
    console.print(f"\nTest Accuracy: {xgb_metrics['accuracy']:.4f}")
    console.print(f"Test Log Loss: {xgb_metrics['log_loss']:.4f}")
    
    # Compare models
    console.print("\n[bold yellow]Step 4: Model Comparison[/bold yellow]")
    
    comparison_table = Table(title="Model Performance (60K Dataset)")
    comparison_table.add_column("Metric", style="cyan")
    comparison_table.add_column("Logistic Regression", style="yellow")
    comparison_table.add_column("XGBoost", style="green")
    comparison_table.add_column("Winner", style="magenta")
    
    metrics_to_compare = ['accuracy', 'log_loss', 'f1_home', 'f1_draw', 'f1_away']
    
    for metric in metrics_to_compare:
        if metric in lr_metrics and metric in xgb_metrics:
            lr_val = lr_metrics[metric]
            xgb_val = xgb_metrics[metric]
            
            if metric == 'log_loss':
                winner = "LR" if lr_val < xgb_val else "XGB"
            else:
                winner = "LR" if lr_val > xgb_val else "XGB"
            
            comparison_table.add_row(
                metric.replace('_', ' ').title(),
                f"{lr_val:.4f}",
                f"{xgb_val:.4f}",
                winner
            )
    
    console.print(comparison_table)
    
    # Feature importance
    console.print("\n[bold yellow]Step 5: Top 15 Features (XGBoost)[/bold yellow]")
    
    feature_importance = xgb_model.get_feature_importance()
    
    importance_table = Table()
    importance_table.add_column("Rank", style="cyan")
    importance_table.add_column("Feature", style="yellow")
    importance_table.add_column("Importance", style="green")
    
    for idx, (feature, importance) in enumerate(list(feature_importance.items())[:15], 1):
        importance_table.add_row(
            str(idx),
            feature,
            f"{importance:.2f}"
        )
    
    console.print(importance_table)
    
    # Save models
    console.print("\n[bold yellow]Step 6: Saving Models[/bold yellow]")
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    lr_model.save(str(models_dir / "logistic_regression_60k.pkl"))
    xgb_model.save(str(models_dir / "xgboost_60k.json"))
    
    console.print(f"Models saved to {models_dir}/")
    
    console.print(f"\n[bold green]SUCCESS![/bold green]")
    console.print(f"[bold]Models trained on {len(X_train) + len(X_val) + len(X_test)} total matches[/bold]")
    console.print(f"[dim]Ready for backtesting with improved accuracy![/dim]\n")


if __name__ == "__main__":
    train_complete_dataset()
