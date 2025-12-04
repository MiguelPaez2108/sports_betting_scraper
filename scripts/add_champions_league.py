"""
Add Champions League data to processed dataset.

Combines scores and odds files for Champions League.
"""
import pandas as pd
from pathlib import Path


def process_champions_league():
    """Process Champions League data and add to dataset."""
    print("=" * 60)
    print("PROCESSING CHAMPIONS LEAGUE DATA")
    print("=" * 60)
    
    # Load BOTH score files (Current Season + Long Season)
    print("\n1. Loading Champions League scores...")
    scores_cs = pd.read_csv('data/historical/Database - Scores - Europe Champions League - CS.csv')
    scores_ls = pd.read_csv('data/historical/Database - Scores - Europe Champions League - LS.csv')
    print(f"  CS (Current Season): {len(scores_cs)} matches")
    print(f"  LS (Long Season): {len(scores_ls)} matches")
    
    # Combine scores
    scores_df = pd.concat([scores_cs, scores_ls], ignore_index=True)
    print(f"  Total scores: {len(scores_df)} matches")
    
    # Load BOTH odds files
    print("\n2. Loading Champions League odds...")
    odds_cs = pd.read_csv('data/historical/Database - Odds - Europe Champions League - CS.csv')
    odds_ls = pd.read_csv('data/historical/Database - Odds - Europe Champions League - LS.csv')
    print(f"  CS (Current Season): {len(odds_cs)} odds")
    print(f"  LS (Long Season): {len(odds_ls)} odds")
    
    # Combine odds
    odds_df = pd.concat([odds_cs, odds_ls], ignore_index=True)
    print(f"  Total odds: {len(odds_df)} records")
    
    # Merge scores and odds
    print("\n3. Merging scores and odds...")
    cl_df = scores_df.merge(odds_df[['id', 'H', 'D', 'A']], on='id', how='inner')
    print(f"  Merged: {len(cl_df)} matches with complete data")
    
    # Rename columns to match standard format
    cl_df = cl_df.rename(columns={
        'homeTeam': 'home_team',
        'awayTeam': 'away_team',
        'matchDate': 'date',
        'H': 'odds_home',
        'D': 'odds_draw',
        'A': 'odds_away'
    })
    
    # Add league identifier
    cl_df['league'] = 'CL'  # Champions League
    cl_df['source_file'] = 'Champions_League'
    
    # Convert date
    cl_df['date'] = pd.to_datetime(cl_df['date'], format='%d-%m-%y %H:%M', errors='coerce')
    
    # Clean data
    cl_df = cl_df.dropna(subset=['home_team', 'away_team', 'FTR', 'date'])
    cl_df = cl_df.dropna(subset=['odds_home', 'odds_draw', 'odds_away'])
    cl_df = cl_df[(cl_df['odds_home'] > 1.01) & (cl_df['odds_draw'] > 1.01) & (cl_df['odds_away'] > 1.01)]
    
    print(f"\n4. After cleaning: {len(cl_df)} Champions League matches")
    print(f"  Date range: {cl_df['date'].min()} to {cl_df['date'].max()}")
    
    # Load existing processed data
    print("\n5. Loading existing processed data...")
    existing_df = pd.read_parquet('data/processed/full_dataset.parquet')
    print(f"  Existing matches: {len(existing_df)}")
    
    # Select only columns that exist in both
    common_cols = list(set(cl_df.columns) & set(existing_df.columns))
    cl_df = cl_df[common_cols]
    existing_df = existing_df[common_cols]
    
    # Combine datasets
    print("\n6. Combining datasets...")
    combined_df = pd.concat([existing_df, cl_df], ignore_index=True)
    
    # Sort by date
    combined_df = combined_df.sort_values('date').reset_index(drop=True)
    
    # Add match_id
    combined_df['match_id'] = range(len(combined_df))
    
    print(f"\n  Total matches: {len(combined_df)}")
    print(f"  Leagues: {combined_df['league'].value_counts().to_dict()}")
    
    # Recompute features for ALL data (including Champions)
    print("\n7. Recomputing features for complete dataset...")
    print("  (This will take a few minutes...)")
    
    from src.feature_engineering.elo_calculator import EloCalculator
    from src.feature_engineering.form_calculator import FormCalculator
    from src.feature_engineering.h2h_calculator import H2HCalculator
    from src.feature_engineering.poisson_calculator import PoissonCalculator
    
    elo_calc = EloCalculator(k_factor=20.0, home_advantage=100.0)
    form_calc = FormCalculator(window=5)
    h2h_calc = H2HCalculator(window=5)
    poisson_calc = PoissonCalculator(window=10)
    
    print("  Computing ELO...")
    elo_features = elo_calc.compute_batch(combined_df)
    
    print("  Computing Form...")
    form_features = form_calc.compute_batch(combined_df)
    
    print("  Computing H2H...")
    h2h_features = h2h_calc.compute_batch(combined_df)
    
    print("  Computing Poisson...")
    poisson_features = poisson_calc.compute_batch(combined_df)
    
    # Merge features
    combined_df = combined_df.merge(elo_features, on='match_id', how='left')
    combined_df = combined_df.merge(form_features, on='match_id', how='left')
    combined_df = combined_df.merge(h2h_features, on='match_id', how='left')
    combined_df = combined_df.merge(poisson_features, on='match_id', how='left')
    
    combined_df = combined_df.fillna(0)
    
    print(f"\n  Total features: {len(combined_df.columns)}")
    
    # Re-split data
    print("\n8. Re-splitting data (70/15/15)...")
    total_matches = len(combined_df)
    train_end_idx = int(total_matches * 0.7)
    val_end_idx = int(total_matches * 0.85)
    
    train_df = combined_df.iloc[:train_end_idx]
    val_df = combined_df.iloc[train_end_idx:val_end_idx]
    test_df = combined_df.iloc[val_end_idx:]
    
    print(f"  Train: {len(train_df)} matches")
    print(f"  Val:   {len(val_df)} matches")
    print(f"  Test:  {len(test_df)} matches")
    
    # Convert object columns to string
    for col in combined_df.select_dtypes(include=['object']).columns:
        train_df[col] = train_df[col].astype(str)
        val_df[col] = val_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)
        combined_df[col] = combined_df[col].astype(str)
    
    # Save updated datasets
    print("\n9. Saving updated datasets...")
    output_dir = Path('data/processed')
    
    train_df.to_parquet(output_dir / 'train.parquet', index=False)
    val_df.to_parquet(output_dir / 'val.parquet', index=False)
    test_df.to_parquet(output_dir / 'test.parquet', index=False)
    combined_df.to_parquet(output_dir / 'full_dataset.parquet', index=False)
    
    print("\n" + "=" * 60)
    print("CHAMPIONS LEAGUE ADDED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nFinal dataset:")
    print(f"  Total matches: {len(combined_df):,}")
    print(f"  Total features: {len(combined_df.columns)}")
    print(f"  Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    print(f"\nLeagues:")
    for league, count in combined_df['league'].value_counts().items():
        print(f"  {league}: {count:,} matches")
    print("=" * 60)


if __name__ == '__main__':
    import sys
    sys.path.append('.')
    process_champions_league()
