"""
Calibrate trained model using validation data.

Usage:
    python scripts/calibrate_model.py --model models/xgboost_balanced.json
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.metrics import brier_score_loss, log_loss
from xgboost import XGBClassifier

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.calibrator_manager import CalibratorManager


def load_validation_data():
    """Load validation data from training artifacts."""
    data_dir = Path('data/processed')
    
    # Load validation split
    X_val = pd.read_parquet(data_dir / 'X_val.parquet')
    y_val = pd.read_parquet(data_dir / 'y_val.parquet')['FTR'].values
    
    # Load league codes if available
    if (data_dir / 'leagues_val.parquet').exists():
        leagues_val = pd.read_parquet(data_dir / 'leagues_val.parquet')['league'].values
    else:
        leagues_val = None
        
    return X_val, y_val, leagues_val


def evaluate_calibration(y_true, y_pred_proba, name='Model'):
    """Calculate calibration metrics."""
    # Brier score (lower is better)
    brier = brier_score_loss(
        y_true, 
        y_pred_proba, 
        pos_label=list(range(y_pred_proba.shape[1]))
    )
    
    # Log loss (lower is better)
    logloss = log_loss(y_true, y_pred_proba)
    
    print(f"\n{name} Metrics:")
    print(f"  Brier Score: {brier:.4f}")
    print(f"  Log Loss: {logloss:.4f}")
    
    return {'brier': brier, 'logloss': logloss}


def main():
    parser = argparse.ArgumentParser(description='Calibrate betting model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output', type=str, default='models/calibrators', help='Output directory')
    args = parser.parse_args()
    
    print("=" * 60)
    print("MODEL CALIBRATION")
    print("=" * 60)
    
    # Load model
    print(f"\n1. Loading model from {args.model}...")
    model = XGBClassifier()
    model.load_model(args.model)
    print("✓ Model loaded")
    
    # Load validation data
    print("\n2. Loading validation data...")
    X_val, y_val, leagues_val = load_validation_data()
    print(f"✓ Loaded {len(X_val)} validation samples")
    
    # Baseline predictions (uncalibrated)
    print("\n3. Evaluating uncalibrated model...")
    y_pred_uncalib = model.predict_proba(X_val)
    metrics_uncalib = evaluate_calibration(y_val, y_pred_uncalib, 'Uncalibrated')
    
    # Initialize CalibratorManager
    print(f"\n4. Fitting calibrators...")
    output_dir = Path(args.output)
    manager = CalibratorManager(model, output_dir)
    
    # Fit global calibrator
    manager.fit_global(X_val, y_val)
    
    # Fit league-specific calibrators if league data available
    if leagues_val is not None:
        unique_leagues = np.unique(leagues_val)
        print(f"\nFitting {len(unique_leagues)} league-specific calibrators...")
        
        for league in unique_leagues:
            mask = (leagues_val == league)
            X_league = X_val[mask]
            y_league = y_val[mask]
            
            manager.fit_by_league(league, X_league, y_league, min_samples=300)
    
    # Evaluate calibrated predictions
    print("\n5. Evaluating calibrated model...")
    if leagues_val is not None:
        y_pred_calib = manager.predict_proba_batch(X_val, leagues_val)
    else:
        y_pred_calib = manager.predict_proba(X_val)
        
    metrics_calib = evaluate_calibration(y_val, y_pred_calib, 'Calibrated')
    
    # Calculate improvements
    print("\n" + "=" * 60)
    print("IMPROVEMENT SUMMARY")
    print("=" * 60)
    
    brier_improvement = (metrics_uncalib['brier'] - metrics_calib['brier']) / metrics_uncalib['brier'] * 100
    logloss_improvement = (metrics_uncalib['logloss'] - metrics_calib['logloss']) / metrics_uncalib['logloss'] * 100
    
    print(f"Brier Score: {metrics_uncalib['brier']:.4f} → {metrics_calib['brier']:.4f} ({brier_improvement:+.2f}%)")
    print(f"Log Loss: {metrics_uncalib['logloss']:.4f} → {metrics_calib['logloss']:.4f} ({logloss_improvement:+.2f}%)")
    
    # Save manager state
    manager_path = output_dir / 'manager_state.pkl'
    manager.save(manager_path)
    print(f"\n✓ Calibrator manager saved to {manager_path}")
    
    # Save metrics report
    report = {
        'uncalibrated': metrics_uncalib,
        'calibrated': metrics_calib,
        'improvement': {
            'brier_pct': brier_improvement,
            'logloss_pct': logloss_improvement
        }
    }
    
    report_path = output_dir / 'calibration_report.pkl'
    joblib.dump(report, report_path)
    print(f"✓ Report saved to {report_path}")
    
    print("\n" + "=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
