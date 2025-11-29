"""
Train Baseline Models

Train and evaluate Logistic Regression and XGBoost models.
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


def train_models():
    """Train and compare baseline models"""
    
    console.print(Panel.fit(
        "[bold cyan]Training ML Models[/bold cyan]\n"
        "[yellow]Logistic Regression + XGBoost[/yellow]",
        border_style="cyan"
    ))
    
    # Load data
    console.print("\n[bold yellow]Step 1: Loading Data[/bold yellow]")
    
    with MLDataLoader(DB_CONFIG) as loader:
        data = loader.prepare_training_data(
            start_date='2020-01-01',  # Use recent data
            end_date='2024-12-31',
            leagues=['E0', 'SP1'],  # Premier League + La Liga
            test_size=0.15,
            val_size=0.15,
            batch_size=50,
            use_cache=True,
            limit=500  # Limit for quick testing
        )
    
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    console.print(f"Train: {len(X_train)} samples")
    console.print(f"Val: {len(X_val)} samples")
    console.print(f"Test: {len(X_test)} samples")
    console.print(f"Features: {len(data['feature_names'])}")
    
    # Train Logistic Regression
    console.print("\n[bold yellow]Step 2: Training Logistic Regression[/bold yellow]")
    
    lr_model = LogisticRegressionModel(C=1.0, max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    lr_metrics = lr_model.evaluate(X_test, y_test)
    
    console.print(f"Accuracy: {lr_metrics['accuracy']:.4f}")
    console.print(f"Log Loss: {lr_metrics['log_loss']:.4f}")
    
    # Train XGBoost
    console.print("\n[bold yellow]Step 3: Training XGBoost[/bold yellow]")
    
    xgb_model = XGBoostModel(
        max_depth=5,
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.8
    )
    xgb_model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=20, verbose=False)
    
    xgb_metrics = xgb_model.evaluate(X_test, y_test)
    
    console.print(f"Accuracy: {xgb_metrics['accuracy']:.4f}")
    console.print(f"Log Loss: {xgb_metrics['log_loss']:.4f}")
    
    # Compare models
    console.print("\n[bold yellow]Step 4: Model Comparison[/bold yellow]")
    
    comparison_table = Table(title="Model Performance on Test Set")
    comparison_table.add_column("Metric", style="cyan")
    comparison_table.add_column("Logistic Regression", style="yellow")
    comparison_table.add_column("XGBoost", style="green")
    comparison_table.add_column("Winner", style="magenta")
    
    metrics_to_compare = ['accuracy', 'log_loss', 'f1_home', 'f1_draw', 'f1_away']
    
    for metric in metrics_to_compare:
        if metric in lr_metrics and metric in xgb_metrics:
            lr_val = lr_metrics[metric]
            xgb_val = xgb_metrics[metric]
            
            # Determine winner (lower is better for log_loss)
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
    
    # Feature importance (XGBoost)
    console.print("\n[bold yellow]Step 5: Top 10 Features (XGBoost)[/bold yellow]")
    
    feature_importance = xgb_model.get_feature_importance()
    
    importance_table = Table()
    importance_table.add_column("Rank", style="cyan")
    importance_table.add_column("Feature", style="yellow")
    importance_table.add_column("Importance", style="green")
    
    for idx, (feature, importance) in enumerate(list(feature_importance.items())[:10], 1):
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
    
    lr_model.save(str(models_dir / "logistic_regression.pkl"))
    xgb_model.save(str(models_dir / "xgboost.json"))
    
    console.print(f"Models saved to {models_dir}/")
    
    console.print(f"\n[bold green]SUCCESS![/bold green]")
    console.print(f"[dim]Models trained and ready for predictions[/dim]\n")


if __name__ == "__main__":
    train_models()
