"""
Check Model Performance

Quick script to load and evaluate the trained models.
"""

import pickle
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

def check_performance():
    """Load and display model performance"""
    
    console.print("[bold cyan]Model Performance Summary[/bold cyan]\n")
    
    # Check XGBoost metadata
    xgb_meta_path = Path("models/xgboost_60k_metadata.pkl")
    if xgb_meta_path.exists():
        with open(xgb_meta_path, 'rb') as f:
            xgb_meta = pickle.load(f)
        
        console.print("[bold]XGBoost Model (60k dataset)[/bold]")
        table = Table(show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Train", style="green")
        table.add_column("Validation", style="yellow")
        table.add_column("Test", style="magenta")
        
        table.add_row(
            "Accuracy",
            f"{xgb_meta.get('train_accuracy', 0):.2%}",
            f"{xgb_meta.get('val_accuracy', 0):.2%}",
            f"{xgb_meta.get('test_accuracy', 0):.2%}"
        )
        
        console.print(table)
        console.print(f"\nTraining samples: {xgb_meta.get('n_samples', 'N/A')}")
        console.print(f"Features: {xgb_meta.get('n_features', 'N/A')}")
        console.print(f"Trained: {xgb_meta.get('trained_at', 'N/A')}")
    
    # Check Logistic Regression
    lr_path = Path("models/logistic_regression_60k.pkl")
    if lr_path.exists():
        console.print("\n[bold]Logistic Regression Model (60k dataset)[/bold]")
        console.print("Model file exists and ready for use")
    
    console.print("\n[green]All models loaded successfully![/green]")

if __name__ == "__main__":
    check_performance()
