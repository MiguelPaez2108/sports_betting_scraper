"""
Evaluate Trained Model

Load the trained XGBoost model and evaluate it on test data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from src.ml_models.infrastructure.data_loader import MLDataLoader
from src.ml_models.domain.models.xgboost_model import XGBoostModel

console = Console()

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'betting_db',
    'user': 'betting_user',
    'password': 'betting_pass_change_in_prod'
}

def evaluate_model():
    """Evaluate the trained model"""
    
    console.print("[bold cyan]Evaluating Trained XGBoost Model (60K Dataset)[/bold cyan]\n")
    
    # Load test data
    console.print("Loading test data...")
    with MLDataLoader(DB_CONFIG) as loader:
        data = loader.prepare_training_data(
            start_date='1993-01-01',
            end_date='2025-12-31',
            leagues=['E0', 'SP1', 'I1', 'D1', 'F1', 'CL'],
            test_size=0.15,
            val_size=0.15,
            batch_size=100,
            use_cache=True,
            limit=None
        )
    
    X_test = data['X_test']
    y_test = data['y_test']
    
    console.print(f"Test set: {len(X_test)} matches\n")
    
    # Load model
    console.print("Loading trained model...")
    model = XGBoostModel()
    model.load("models/xgboost_60k.json")
    
    # Evaluate
    console.print("Evaluating model...\n")
    metrics = model.evaluate(X_test, y_test)
    
    # Display results
    table = Table(title="Model Performance on Test Set")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Accuracy", f"{metrics['accuracy']:.2%}")
    table.add_row("Log Loss", f"{metrics['log_loss']:.4f}")
    table.add_row("F1 (Home Win)", f"{metrics['f1_home']:.4f}")
    table.add_row("F1 (Draw)", f"{metrics['f1_draw']:.4f}")
    table.add_row("F1 (Away Win)", f"{metrics['f1_away']:.4f}")
    
    console.print(table)
    
    # Comparison with previous model
    console.print("\n[bold]Comparison with Previous Model:[/bold]")
    console.print(f"Previous (18K dataset): ~39%")
    console.print(f"Current (60K dataset): {metrics['accuracy']:.2%}")
    
    improvement = (metrics['accuracy'] - 0.39) / 0.39 * 100
    if improvement > 0:
        console.print(f"[green]Improvement: +{improvement:.1f}%[/green]")
    else:
        console.print(f"[red]Change: {improvement:.1f}%[/red]")

if __name__ == "__main__":
    evaluate_model()
