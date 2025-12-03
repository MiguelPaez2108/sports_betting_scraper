"""
Script to train ensemble model.

Usage:
    python scripts/train_ensemble.py --data data/processed/features.parquet
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.ensemble import train_ensemble


def main():
    parser = argparse.ArgumentParser(description='Train stacking ensemble')
    parser.add_argument('--data', type=str, required=True, help='Path to features parquet')
    parser.add_argument('--output', type=str, default='models/ensemble', help='Output directory')
    parser.add_argument('--val-size', type=float, default=0.2, help='Validation split size')
    args = parser.parse_args()
    
    print("=" * 60)
    print("ENSEMBLE MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    print(f"\n1. Loading data from {args.data}...")
    df = pd.read_parquet(args.data)
    
    # Separate features and target
    feature_cols = [c for c in df.columns if c not in ['FTR', 'date', 'match_id', 'league']]
    X = df[feature_cols]
    y = df['FTR'].map({'H': 2, 'D': 1, 'A': 0}).values
    
    print(f"✓ Loaded {len(df)} samples with {len(feature_cols)} features")
    
    # Train/val split (temporal)
    split_idx = int(len(df) * (1 - args.val_size))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"\nTrain: {len(X_train)} samples")
    print(f"Val: {len(X_val)} samples")
    
    # Train ensemble
    output_dir = Path(args.output)
    ensemble = train_ensemble(X_train, y_train, X_val, y_val, output_dir)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Model saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
