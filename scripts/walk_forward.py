"""
Walk-Forward Validation: Test model on rolling temporal windows.

This ensures the model generalizes across time and detects overfitting.

Usage:
    python scripts/walk_forward.py --data data/processed/features.parquet --n-folds 5
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, f1_score
from xgboost import XGBClassifier

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.calibrator_manager import CalibratorManager


def create_temporal_splits(df, n_folds=5, min_train_size=0.4):
    """
    Create temporal train/test splits.
    
    Args:
        df: DataFrame with 'date' column
        n_folds: Number of folds
        min_train_size: Minimum fraction of data for initial training
        
    Returns:
        List of (train_idx, test_idx) tuples
    """
    df = df.sort_values('date').reset_index(drop=True)
    n_samples = len(df)
    
    # Initial training size
    initial_train = int(n_samples * min_train_size)
    
    # Calculate fold size for remaining data
    remaining = n_samples - initial_train
    fold_size = remaining // n_folds
    
    splits = []
    for i in range(n_folds):
        train_end = initial_train + (i * fold_size)
        test_start = train_end
        test_end = min(test_start + fold_size, n_samples)
        
        train_idx = list(range(train_end))
        test_idx = list(range(test_start, test_end))
        
        if len(test_idx) > 0:
            splits.append((train_idx, test_idx))
    
    return splits


def train_and_evaluate_fold(
    X_train, y_train, X_test, y_test,
    fold_num, output_dir, leagues_train=None, leagues_test=None
):
    """Train model and calibrator on one fold, evaluate on test."""
    print(f"\n{'='*60}")
    print(f"FOLD {fold_num}")
    print(f"{'='*60}")
    print(f"Train: {len(X_train)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # Train model
    print("\n1. Training XGBoost...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Split train into train/val for calibration
    val_size = int(len(X_train) * 0.2)
    X_val = X_train.iloc[-val_size:]
    y_val = y_train[-val_size:]
    leagues_val = leagues_train[-val_size:] if leagues_train is not None else None
    
    # Calibrate
    print("\n2. Calibrating...")
    calibrator_dir = output_dir / f'fold_{fold_num}' / 'calibrators'
    manager = CalibratorManager(model, calibrator_dir)
    manager.fit_global(X_val, y_val)
    
    if leagues_val is not None:
        unique_leagues = np.unique(leagues_val)
        for league in unique_leagues:
            mask = (leagues_val == league)
            manager.fit_by_league(league, X_val[mask], y_val[mask], min_samples=100)
    
    # Predict on test
    print("\n3. Evaluating...")
    if leagues_test is not None:
        y_pred_proba = manager.predict_proba_batch(X_test, leagues_test)
    else:
        y_pred_proba = manager.predict_proba(X_test)
    
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    metrics = {
        'fold': fold_num,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'accuracy': accuracy_score(y_test, y_pred),
        'log_loss': log_loss(y_test, y_pred_proba),
        'brier_score': brier_score_loss(
            y_test, 
            y_pred_proba,
            pos_label=list(range(3))
        ),
        'f1_home': f1_score(y_test, y_pred, labels=[2], average='macro'),
        'f1_draw': f1_score(y_test, y_pred, labels=[1], average='macro'),
        'f1_away': f1_score(y_test, y_pred, labels=[0], average='macro')
    }
    
    print(f"\nMetrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Log Loss: {metrics['log_loss']:.4f}")
    print(f"  Brier Score: {metrics['brier_score']:.4f}")
    print(f"  F1 (H/D/A): {metrics['f1_home']:.3f} / {metrics['f1_draw']:.3f} / {metrics['f1_away']:.3f}")
    
    # Save artifacts
    fold_dir = output_dir / f'fold_{fold_num}'
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    model.save_model(str(fold_dir / 'model.json'))
    manager.save(fold_dir / 'calibrator_state.pkl')
    joblib.dump(metrics, fold_dir / 'metrics.pkl')
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Walk-forward validation')
    parser.add_argument('--data', type=str, required=True, help='Path to features parquet')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--output', type=str, default='analysis/walk_forward', help='Output directory')
    args = parser.parse_args()
    
    print("=" * 60)
    print("WALK-FORWARD VALIDATION")
    print("=" * 60)
    
    # Load data
    print(f"\n1. Loading data from {args.data}...")
    df = pd.read_parquet(args.data)
    
    # Ensure date column
    if 'date' not in df.columns:
        raise ValueError("Data must have 'date' column for temporal splits")
    
    df['date'] = pd.to_datetime(df['date'])
    
    # Separate features and target
    feature_cols = [c for c in df.columns if c not in ['FTR', 'date', 'match_id', 'league']]
    X = df[feature_cols]
    y = df['FTR'].map({'H': 2, 'D': 1, 'A': 0}).values
    leagues = df['league'].values if 'league' in df.columns else None
    
    print(f"✓ Loaded {len(df)} samples with {len(feature_cols)} features")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Create splits
    print(f"\n2. Creating {args.n_folds} temporal splits...")
    splits = create_temporal_splits(df, n_folds=args.n_folds)
    print(f"✓ Created {len(splits)} splits")
    
    # Run validation
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_metrics = []
    
    for fold_num, (train_idx, test_idx) in enumerate(splits, 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        leagues_train = leagues[train_idx] if leagues is not None else None
        leagues_test = leagues[test_idx] if leagues is not None else None
        
        metrics = train_and_evaluate_fold(
            X_train, y_train, X_test, y_test,
            fold_num, output_dir,
            leagues_train, leagues_test
        )
        all_metrics.append(metrics)
    
    # Aggregate results
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    
    results_df = pd.DataFrame(all_metrics)
    
    print("\nPer-Fold Metrics:")
    print(results_df[['fold', 'accuracy', 'log_loss', 'brier_score']].to_string(index=False))
    
    print("\nAggregate Statistics:")
    print(f"  Accuracy: {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
    print(f"  Log Loss: {results_df['log_loss'].mean():.4f} ± {results_df['log_loss'].std():.4f}")
    print(f"  Brier Score: {results_df['brier_score'].mean():.4f} ± {results_df['brier_score'].std():.4f}")
    
    # Calculate stability (coefficient of variation)
    cv_accuracy = results_df['accuracy'].std() / results_df['accuracy'].mean()
    print(f"\nStability (CV of Accuracy): {cv_accuracy:.4f}")
    
    if cv_accuracy < 0.05:
        print("✓ Model is STABLE across time")
    elif cv_accuracy < 0.10:
        print("⚠ Model shows MODERATE variance")
    else:
        print("✗ Model shows HIGH variance - investigate!")
    
    # Save results
    results_path = output_dir / 'walkforward_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to {results_path}")
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
