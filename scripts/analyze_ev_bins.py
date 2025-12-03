"""
Analyze Expected Value (EV) by probability bins.

This script groups predictions by probability bins and calculates:
- Actual win rate
- Average implied probability (from odds)
- Edge (actual - implied)
- ROI per bin

Usage:
    python scripts/analyze_ev_bins.py --predictions data/predictions.parquet --odds data/odds.parquet
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_implied_prob(odds):
    """Convert decimal odds to implied probability."""
    return 1.0 / odds


def analyze_ev_by_bin(predictions_df, odds_df, bin_size=0.05):
    """
    Analyze EV by probability bins.
    
    Args:
        predictions_df: DataFrame with columns [match_id, pred_home, pred_draw, pred_away, actual_result]
        odds_df: DataFrame with columns [match_id, odds_home, odds_draw, odds_away]
        bin_size: Size of probability bins (default 5%)
        
    Returns:
        DataFrame with EV analysis per bin
    """
    # Merge predictions with odds
    df = predictions_df.merge(odds_df, on='match_id', how='inner')
    
    # Calculate implied probabilities
    df['imp_home'] = calculate_implied_prob(df['odds_home'])
    df['imp_draw'] = calculate_implied_prob(df['odds_draw'])
    df['imp_away'] = calculate_implied_prob(df['odds_away'])
    
    # Normalize implied probs
    total_imp = df['imp_home'] + df['imp_draw'] + df['imp_away']
    df['imp_home_norm'] = df['imp_home'] / total_imp
    df['imp_draw_norm'] = df['imp_draw'] / total_imp
    df['imp_away_norm'] = df['imp_away'] / total_imp
    
    # Create bins
    bins = np.arange(0, 1 + bin_size, bin_size)
    bin_labels = [f"{b:.2f}-{b+bin_size:.2f}" for b in bins[:-1]]
    
    results = []
    
    # Analyze each outcome type
    for outcome, pred_col, imp_col in [
        ('Home', 'pred_home', 'imp_home_norm'),
        ('Draw', 'pred_draw', 'imp_draw_norm'),
        ('Away', 'pred_away', 'imp_away_norm')
    ]:
        # Bin predictions
        df['bin'] = pd.cut(df[pred_col], bins=bins, labels=bin_labels, include_lowest=True)
        
        # Group by bin
        for bin_label in bin_labels:
            bin_data = df[df['bin'] == bin_label]
            
            if len(bin_data) == 0:
                continue
                
            # Calculate metrics
            actual_wins = (bin_data['actual_result'] == outcome).sum()
            total_bets = len(bin_data)
            win_rate = actual_wins / total_bets if total_bets > 0 else 0
            
            avg_pred_prob = bin_data[pred_col].mean()
            avg_imp_prob = bin_data[imp_col].mean()
            
            edge = win_rate - avg_imp_prob
            
            # Calculate ROI (simplified: assumes flat stake)
            # ROI = (wins * avg_odds - total_bets) / total_bets
            avg_odds = 1 / avg_imp_prob if avg_imp_prob > 0 else 0
            roi = (actual_wins * avg_odds - total_bets) / total_bets if total_bets > 0 else 0
            
            results.append({
                'outcome': outcome,
                'bin': bin_label,
                'count': total_bets,
                'win_rate': win_rate,
                'avg_pred_prob': avg_pred_prob,
                'avg_imp_prob': avg_imp_prob,
                'edge': edge,
                'roi': roi
            })
    
    return pd.DataFrame(results)


def plot_ev_heatmap(ev_df, output_path):
    """Create heatmap of ROI by outcome and probability bin."""
    # Pivot for heatmap
    pivot = ev_df.pivot(index='outcome', columns='bin', values='roi')
    
    plt.figure(figsize=(14, 6))
    sns.heatmap(
        pivot, 
        annot=True, 
        fmt='.2%', 
        cmap='RdYlGn',
        center=0,
        cbar_kws={'label': 'ROI'}
    )
    plt.title('ROI by Probability Bin and Outcome')
    plt.xlabel('Predicted Probability Bin')
    plt.ylabel('Outcome')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Heatmap saved to {output_path}")


def plot_edge_by_bin(ev_df, output_path):
    """Plot edge (win_rate - implied_prob) by bin."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, outcome in enumerate(['Home', 'Draw', 'Away']):
        data = ev_df[ev_df['outcome'] == outcome].sort_values('avg_pred_prob')
        
        ax = axes[idx]
        ax.bar(range(len(data)), data['edge'], color='steelblue', alpha=0.7)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(data['bin'], rotation=45, ha='right')
        ax.set_title(f'{outcome} - Edge by Bin')
        ax.set_ylabel('Edge (Win Rate - Implied Prob)')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Edge plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze EV by probability bins')
    parser.add_argument('--predictions', type=str, required=True, help='Path to predictions parquet')
    parser.add_argument('--odds', type=str, required=True, help='Path to odds parquet')
    parser.add_argument('--output', type=str, default='analysis', help='Output directory')
    parser.add_argument('--bin-size', type=float, default=0.05, help='Bin size (default 0.05 = 5%)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("EV-BY-BIN ANALYSIS")
    print("=" * 60)
    
    # Load data
    print(f"\n1. Loading data...")
    predictions_df = pd.read_parquet(args.predictions)
    odds_df = pd.read_parquet(args.odds)
    print(f"✓ Loaded {len(predictions_df)} predictions")
    
    # Analyze
    print(f"\n2. Analyzing EV by {args.bin_size:.0%} bins...")
    ev_df = analyze_ev_by_bin(predictions_df, odds_df, bin_size=args.bin_size)
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / 'ev_by_bin.csv'
    ev_df.to_csv(csv_path, index=False)
    print(f"✓ Results saved to {csv_path}")
    
    # Generate plots
    print(f"\n3. Generating visualizations...")
    plot_ev_heatmap(ev_df, output_dir / 'ev_heatmap.png')
    plot_edge_by_bin(ev_df, output_dir / 'edge_by_bin.png')
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY - BEST PERFORMING BINS")
    print("=" * 60)
    
    # Top 5 bins by ROI
    top_bins = ev_df.nlargest(5, 'roi')[['outcome', 'bin', 'count', 'win_rate', 'edge', 'roi']]
    print("\nTop 5 Bins by ROI:")
    print(top_bins.to_string(index=False))
    
    # Bins with positive edge and sufficient volume
    profitable = ev_df[(ev_df['edge'] > 0.02) & (ev_df['count'] >= 20)]
    print(f"\nProfitable bins (edge > 2%, count >= 20): {len(profitable)}")
    if len(profitable) > 0:
        print(profitable[['outcome', 'bin', 'count', 'win_rate', 'edge', 'roi']].to_string(index=False))
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
