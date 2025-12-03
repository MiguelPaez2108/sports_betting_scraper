"""
Detect data drift in production.

Usage:
    python scripts/detect_drift.py --reference data/train --current data/recent
"""
import argparse
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.monitoring.drift_detector import DriftDetector


def main():
    parser = argparse.ArgumentParser(description='Detect data drift')
    parser.add_argument('--reference', type=str, required=True, help='Reference data path')
    parser.add_argument('--current', type=str, required=True, help='Current data path')
    parser.add_argument('--threshold', type=float, default=0.1, help='Drift threshold')
    parser.add_argument('--output', type=str, default='analysis/drift', help='Output directory')
    args = parser.parse_args()
    
    print("=" * 60)
    print("DRIFT DETECTION")
    print("=" * 60)
    
    # Load data
    print(f"\n1. Loading data...")
    ref_df = pd.read_parquet(args.reference)
    cur_df = pd.read_parquet(args.current)
    print(f"✓ Reference: {len(ref_df)} samples")
    print(f"✓ Current: {len(cur_df)} samples")
    
    # Get feature columns (exclude metadata)
    features = [c for c in ref_df.columns 
                if c not in ['match_id', 'date', 'FTR', 'league', 'home_team', 'away_team']]
    
    print(f"\n2. Checking {len(features)} features for drift...")
    
    # Detect drift
    detector = DriftDetector(threshold=args.threshold)
    results = detector.detect_drift(ref_df, cur_df, features)
    
    # Print summary
    detector.print_summary(results)
    
    # Save report
    output_dir = Path(args.output)
    detector.save_report(results, output_dir / 'drift_report.json')
    
    # Exit with error code if drift detected
    if results['features_with_drift'] > 0:
        print("\n⚠️  WARNING: Drift detected - consider retraining model")
        sys.exit(1)
    else:
        print("\n✅ No significant drift detected")
        sys.exit(0)


if __name__ == '__main__':
    main()
