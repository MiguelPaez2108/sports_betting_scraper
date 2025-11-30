"""
Quick Model Evaluation - Fixed

Load trained model directly with XGBoost and evaluate.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, f1_score
from rich.console import Console
from rich.table import Table

console = Console()

def quick_eval():
    """Quick evaluation using cached data"""
    
    console.print("[bold cyan]Evaluating Trained Model (53K Dataset)[/bold cyan]\n")
    
    # Load cached data
    console.print("Loading cached features...")
    with open('data/ml_cache/features_cache.pkl', 'rb') as f:
        cached = pickle.load(f)
    
    X = cached['X']
    y = cached['y']
    
    console.print(f"Total samples: {len(X)}")
    
    # Split (same as training: 70% train, 15% val, 15% test)
    n = len(X)
    test_idx = int(n * 0.85)
    
    X_test = X.iloc[test_idx:]
    y_test = y.iloc[test_idx:]
    
    # Remove metadata columns
    metadata_cols = ['match_id', 'match_date', 'league_code']
    X_test_clean = X_test.drop(columns=metadata_cols, errors='ignore')
    
    console.print(f"Test set: {len(X_test_clean)} samples\n")
    
    # Load model directly with XGBoost
    console.print("Loading trained model...")
    model = xgb.XGBClassifier()
    model.load_model("models/xgboost_60k.json")
    
    # Predict
    console.print("Evaluating...\n")
    y_pred = model.predict(X_test_clean)
    y_pred_proba = model.predict_proba(X_test_clean)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)
    
    f1_scores = f1_score(y_test, y_pred, average=None)
    
    # Display results
    table = Table(title="Model Performance (53K Dataset)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Accuracy", f"{accuracy:.2%}")
    table.add_row("Log Loss", f"{logloss:.4f}")
    table.add_row("F1 (Away Win)", f"{f1_scores[0]:.4f}")
    table.add_row("F1 (Draw)", f"{f1_scores[1]:.4f}")
    table.add_row("F1 (Home Win)", f"{f1_scores[2]:.4f}")
    
    console.print(table)
    
    # Comparison
    console.print("\n[bold]Comparison:[/bold]")
    console.print(f"Previous (18K): ~39.0%")
    console.print(f"Current (53K): {accuracy:.2%}")
    
    improvement = (accuracy - 0.39) / 0.39 * 100
    if improvement > 0:
        console.print(f"[green]Improvement: +{improvement:.1f}%[/green]")
    else:
        console.print(f"[red]Change: {improvement:.1f}%[/red]")
    
    # ROI prediction
    console.print("\n[bold]ROI Prediction:[/bold]")
    if accuracy >= 0.45:
        console.print("[green]Accuracy >= 45% - Positive ROI likely possible![/green]")
    elif accuracy >= 0.42:
        console.print("[yellow]Accuracy 42-45% - Marginal ROI possible with optimization[/yellow]")
    else:
        console.print("[red]Accuracy < 42% - Positive ROI unlikely[/red]")
    
    return accuracy

if __name__ == "__main__":
    quick_eval()
