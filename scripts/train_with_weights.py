"""
Train model with sample weights for class balancing.

Usage:
    python scripts/train_with_weights.py --data data/processed/features.parquet
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier

import sys
sys.path.append(str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description='Train model with sample weights')
    parser.add_argument('--data', type=str, required=True, help='Path to features parquet')
    parser.add_argument('--output', type=str, default='models/weighted', help='Output directory')
    args = parser.parse_args()
    
    print("=" * 60)
    print("TRAINING WITH SAMPLE WEIGHTS")
    print("=" * 60)
    
    # Load data
    print(f"\n1. Loading data from {args.data}...")
    df = pd.read_parquet(args.data)
    
    # Separate features and target
    feature_cols = [c for c in df.columns if c not in ['FTR', 'date', 'match_id', 'league']]
    X = df[feature_cols]
    y = df['FTR'].map({'H': 2, 'D': 1, 'A': 0}).values
    
    print(f"✓ Loaded {len(df)} samples with {len(feature_cols)} features")
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nClass distribution:")
    for cls, count in zip(unique, counts):
        cls_name = ['Away', 'Draw', 'Home'][cls]
        print(f"  {cls_name}: {count} ({count/len(y)*100:.1f}%)")
    
    # Compute sample weights
    print(f"\n2. Computing sample weights...")
    class_weights = compute_class_weight('balanced', classes=unique, y=y)
    sample_weights = np.array([class_weights[label] for label in y])
    
    print(f"Class weights:")
    for cls, weight in zip(unique, class_weights):
        cls_name = ['Away', 'Draw', 'Home'][cls]
        print(f"  {cls_name}: {weight:.3f}")
    
    # Train/val split
    split_idx = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    weights_train = sample_weights[:split_idx]
    
    # Train without weights (baseline)
    print(f"\n3. Training baseline model (no weights)...")
    model_baseline = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    model_baseline.fit(X_train, y_train)
    
    y_pred_baseline = model_baseline.predict(X_val)
    f1_baseline = {
        'home': f1_score(y_val, y_pred_baseline, labels=[2], average='macro'),
        'draw': f1_score(y_val, y_pred_baseline, labels=[1], average='macro'),
        'away': f1_score(y_val, y_pred_baseline, labels=[0], average='macro')
    }
    
    print(f"\nBaseline F1 scores:")
    print(f"  Home: {f1_baseline['home']:.3f}")
    print(f"  Draw: {f1_baseline['draw']:.3f}")
    print(f"  Away: {f1_baseline['away']:.3f}")
    
    # Train with weights
    print(f"\n4. Training weighted model...")
    model_weighted = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    model_weighted.fit(X_train, y_train, sample_weight=weights_train)
    
    y_pred_weighted = model_weighted.predict(X_val)
    f1_weighted = {
        'home': f1_score(y_val, y_pred_weighted, labels=[2], average='macro'),
        'draw': f1_score(y_val, y_pred_weighted, labels=[1], average='macro'),
        'away': f1_score(y_val, y_pred_weighted, labels=[0], average='macro')
    }
    
    print(f"\nWeighted F1 scores:")
    print(f"  Home: {f1_weighted['home']:.3f}")
    print(f"  Draw: {f1_weighted['draw']:.3f}")
    print(f"  Away: {f1_weighted['away']:.3f}")
    
    # Compare improvements
    print(f"\n" + "=" * 60)
    print("IMPROVEMENT SUMMARY")
    print("=" * 60)
    
    for cls in ['home', 'draw', 'away']:
        improvement = (f1_weighted[cls] - f1_baseline[cls]) / f1_baseline[cls] * 100
        print(f"{cls.capitalize()}: {f1_baseline[cls]:.3f} → {f1_weighted[cls]:.3f} ({improvement:+.1f}%)")
    
    # Save weighted model
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_weighted.save_model(str(output_dir / 'model_weighted.json'))
    print(f"\n✓ Weighted model saved to {output_dir}")
    
    # Full classification report
    print(f"\n" + "=" * 60)
    print("CLASSIFICATION REPORT (Weighted Model)")
    print("=" * 60)
    print(classification_report(y_val, y_pred_weighted, target_names=['Away', 'Draw', 'Home']))


if __name__ == '__main__':
    main()
