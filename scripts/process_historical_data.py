"""
Process historical data for validation pipeline.

Consolidates all CSV files, cleans data, adds features, and prepares for:
- Walk-forward validation
- Calibration
- Backtesting
- EV analysis

Usage:
    python scripts/process_historical_data.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.feature_engineering.elo_calculator import EloCalculator
from src.feature_engineering.form_calculator import FormCalculator
from src.feature_engineering.h2h_calculator import H2HCalculator
from src.feature_engineering.poisson_calculator import PoissonCalculator


def load_all_historical_data(data_dir: Path) -> pd.DataFrame:
    """Load and consolidate all historical CSV files."""
    print("=" * 60)
    print("LOADING HISTORICAL DATA")
    print("=" * 60)
    
    all_data = []
    csv_files = list(data_dir.glob("*.csv"))
    
    print(f"\nFound {len(csv_files)} CSV files")
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Extract league from filename (e.g., E0, D1, SP1)
            league = csv_file.stem.split('_')[0]
            df['league'] = league
            df['source_file'] = csv_file.name
            
            all_data.append(df)
            print(f"  OK {csv_file.name}: {len(df)} matches")
            
        except Exception as e:
            print(f"  ERROR {csv_file.name}: {e}")
    
    # Concatenate all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nOK Total matches loaded: {len(combined_df):,}")
    
    return combined_df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize data."""
    print("\n" + "=" * 60)
    print("CLEANING DATA")
    print("=" * 60)
    
    initial_count = len(df)
    
    # Rename columns to standard format
    column_mapping = {
        'HomeTeam': 'home_team',
        'AwayTeam': 'away_team',
        'FTHG': 'FTHG',
        'FTAG': 'FTAG',
        'FTR': 'FTR',
        'Date': 'date',
        'B365H': 'odds_home',
        'B365D': 'odds_draw',
        'B365A': 'odds_away'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
    
    # Remove rows with missing critical data
    df = df.dropna(subset=['home_team', 'away_team', 'FTR', 'date'])
    
    # Remove rows with missing odds
    df = df.dropna(subset=['odds_home', 'odds_draw', 'odds_away'])
    
    # Remove invalid odds (< 1.01)
    df = df[(df['odds_home'] > 1.01) & (df['odds_draw'] > 1.01) & (df['odds_away'] > 1.01)]
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Add match_id
    df['match_id'] = range(len(df))
    
    print(f"  Initial matches: {initial_count:,}")
    print(f"  After cleaning: {len(df):,}")
    print(f"  Removed: {initial_count - len(df):,} ({(initial_count - len(df))/initial_count*100:.1f}%)")
    
    # Date range
    print(f"\n  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    # Leagues
    print(f"\n  Leagues: {df['league'].nunique()}")
    for league in df['league'].value_counts().head(10).items():
        print(f"    {league[0]}: {league[1]:,} matches")
    
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all features using calculators."""
    print("\n" + "=" * 60)
    print("ADDING FEATURES")
    print("=" * 60)
    
    # Initialize calculators
    elo_calc = EloCalculator(k_factor=20.0, home_advantage=100.0)
    form_calc = FormCalculator(window=5)
    h2h_calc = H2HCalculator(window=5)
    poisson_calc = PoissonCalculator(window=10)
    
    # Compute features
    print("\n  Computing ELO ratings...")
    elo_features = elo_calc.compute_batch(df)
    
    print("  Computing form statistics...")
    form_features = form_calc.compute_batch(df)
    
    print("  Computing H2H statistics...")
    h2h_features = h2h_calc.compute_batch(df)
    
    print("  Computing Poisson probabilities...")
    poisson_features = poisson_calc.compute_batch(df)
    
    # Merge all features
    print("\n  Merging features...")
    df = df.merge(elo_features, on='match_id', how='left')
    df = df.merge(form_features, on='match_id', how='left')
    df = df.merge(h2h_features, on='match_id', how='left')
    df = df.merge(poisson_features, on='match_id', how='left')
    
    # Fill NaN values (for early matches without history)
    df = df.fillna(0)
    
    print(f"\nOK Total features: {len(df.columns)}")
    
    return df


def split_data(df: pd.DataFrame, output_dir: Path) -> None:
    """Split data into train/val/test sets (temporal)."""
    print("\n" + "=" * 60)
    print("SPLITTING DATA")
    print("=" * 60)
    
    # Temporal splits
    total_matches = len(df)
    train_end_idx = int(total_matches * 0.7)
    val_end_idx = int(total_matches * 0.85)
    
    train_df = df.iloc[:train_end_idx]
    val_df = df.iloc[train_end_idx:val_end_idx]
    test_df = df.iloc[val_end_idx:]
    
    print(f"\n  Train: {len(train_df):,} matches ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"  Val:   {len(val_df):,} matches ({val_df['date'].min().date()} to {val_df['date'].max().date()})")
    print(f"  Test:  {len(test_df):,} matches ({test_df['date'].min().date()} to {test_df['date'].max().date()})")
    
    # Save splits
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert object columns to string to avoid Arrow errors
    for col in train_df.select_dtypes(include=['object']).columns:
        train_df[col] = train_df[col].astype(str)
        val_df[col] = val_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)
        df[col] = df[col].astype(str)
    
    train_df.to_parquet(output_dir / 'train.parquet', index=False)
    val_df.to_parquet(output_dir / 'val.parquet', index=False)
    test_df.to_parquet(output_dir / 'test.parquet', index=False)
    
    # Save full dataset
    df.to_parquet(output_dir / 'full_dataset.parquet', index=False)
    
    print(f"\nOK Data saved to {output_dir}")


def main():
    print("=" * 60)
    print("HISTORICAL DATA PROCESSING PIPELINE")
    print("=" * 60)
    
    # Paths
    data_dir = Path('data/historical')
    output_dir = Path('data/processed')
    
    # 1. Load data
    df = load_all_historical_data(data_dir)
    
    # 2. Clean data
    df = clean_data(df)
    
    # 3. Add features
    df = add_features(df)
    
    # 4. Split data
    split_data(df, output_dir)
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nReady for validation pipeline!")
    print(f"  Total matches: {len(df):,}")
    print(f"  Total features: {len(df.columns)}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"\nNext steps:")
    print(f"  1. python scripts/walk_forward.py --data data/processed/full_dataset.parquet")
    print(f"  2. python scripts/calibrate_model.py --model models/xgboost_balanced.json")
    print(f"  3. python scripts/analyze_ev_bins.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
