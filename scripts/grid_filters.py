"""
Grid Search for Betting Filter Thresholds.

Systematically tests combinations of:
- min_confidence: Minimum predicted probability
- min_edge: Minimum edge (prob - implied_prob)
- odds_range: (min_odds, max_odds)

Goal: Find sweet spot with 40-60 bets and optimal ROI.

Usage:
    python scripts/grid_filters.py --predictions data/predictions.parquet --odds data/odds.parquet
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.backtesting.backtester import EnhancedBacktester, BettingFilters
from src.backtesting.stake_strategies import FractionalKelly


def run_backtest_with_filters(
    matches_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    filters: BettingFilters,
    initial_bankroll: float = 1000.0
) -> dict:
    """Run backtest with specific filter configuration."""
    backtester = EnhancedBacktester(
        initial_bankroll=initial_bankroll,
        stake_strategy=FractionalKelly(fraction=0.25),
        filters=filters,
        slippage_pct=0.01
    )
    
    metrics_df, trades_df = backtester.run(matches_df, predictions_df)
    
    if metrics_df.empty:
        return {
            'num_bets': 0,
            'roi': 0,
            'win_rate': 0,
            'sharpe': 0,
            'final_bankroll': initial_bankroll
        }
    
    return {
        'num_bets': int(metrics_df['num_bets'].iloc[0]),
        'roi': float(metrics_df['roi'].iloc[0]),
        'win_rate': float(metrics_df['win_rate'].iloc[0]),
        'sharpe': float(metrics_df['sharpe_ratio'].iloc[0]),
        'final_bankroll': float(metrics_df['final_bankroll'].iloc[0])
    }


def grid_search(
    matches_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    min_conf_range: list,
    min_edge_range: list,
    odds_ranges: list,
    target_bets: tuple = (40, 60)
) -> pd.DataFrame:
    """
    Perform grid search over filter parameters.
    
    Args:
        matches_df: Historical matches with odds
        predictions_df: Model predictions
        min_conf_range: List of min_confidence values to test
        min_edge_range: List of min_edge values to test
        odds_ranges: List of (min_odds, max_odds) tuples
        target_bets: (min, max) acceptable bet count
        
    Returns:
        DataFrame with results for each configuration
    """
    results = []
    
    # Generate all combinations
    combinations = list(product(min_conf_range, min_edge_range, odds_ranges))
    
    print(f"Testing {len(combinations)} filter combinations...")
    
    for min_conf, min_edge, (min_odds, max_odds) in tqdm(combinations):
        filters = BettingFilters(
            min_confidence=min_conf,
            min_edge=min_edge,
            min_odds=min_odds,
            max_odds=max_odds,
            max_book_margin=0.15
        )
        
        metrics = run_backtest_with_filters(matches_df, predictions_df, filters)
        
        # Check if meets target bet count
        in_target = target_bets[0] <= metrics['num_bets'] <= target_bets[1]
        
        results.append({
            'min_conf': min_conf,
            'min_edge': min_edge,
            'min_odds': min_odds,
            'max_odds': max_odds,
            'num_bets': metrics['num_bets'],
            'roi': metrics['roi'],
            'win_rate': metrics['win_rate'],
            'sharpe': metrics['sharpe'],
            'final_bankroll': metrics['final_bankroll'],
            'in_target': in_target
        })
    
    return pd.DataFrame(results)


def plot_heatmap(results_df: pd.DataFrame, output_path: Path):
    """Create heatmap of ROI by min_conf and min_edge."""
    # Filter to configurations in target bet range
    target_results = results_df[results_df['in_target']]
    
    if len(target_results) == 0:
        print("⚠ No configurations in target bet range")
        return
    
    # Pivot for heatmap
    pivot = target_results.pivot_table(
        index='min_edge',
        columns='min_conf',
        values='roi',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.2%',
        cmap='RdYlGn',
        center=0,
        cbar_kws={'label': 'ROI'}
    )
    plt.title('ROI by Filter Configuration (Target Bet Range Only)')
    plt.xlabel('Min Confidence')
    plt.ylabel('Min Edge')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Heatmap saved to {output_path}")


def plot_pareto_frontier(results_df: pd.DataFrame, output_path: Path):
    """Plot ROI vs Bet Count to visualize trade-offs."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Scatter plot
    scatter = ax.scatter(
        results_df['num_bets'],
        results_df['roi'],
        c=results_df['sharpe'],
        s=100,
        alpha=0.6,
        cmap='viridis'
    )
    
    # Highlight target range
    target_mask = results_df['in_target']
    ax.scatter(
        results_df[target_mask]['num_bets'],
        results_df[target_mask]['roi'],
        edgecolors='red',
        facecolors='none',
        s=150,
        linewidths=2,
        label='Target Range (40-60 bets)'
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sharpe Ratio')
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Number of Bets')
    ax.set_ylabel('ROI')
    ax.set_title('ROI vs Bet Volume - Pareto Frontier')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Pareto plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Grid search for optimal filters')
    parser.add_argument('--predictions', type=str, required=True, help='Predictions parquet')
    parser.add_argument('--odds', type=str, required=True, help='Odds parquet')
    parser.add_argument('--output', type=str, default='analysis/grid_search', help='Output directory')
    parser.add_argument('--target-min', type=int, default=40, help='Min target bets')
    parser.add_argument('--target-max', type=int, default=60, help='Max target bets')
    args = parser.parse_args()
    
    print("=" * 60)
    print("GRID SEARCH FOR OPTIMAL FILTERS")
    print("=" * 60)
    
    # Load data
    print(f"\n1. Loading data...")
    predictions_df = pd.read_parquet(args.predictions)
    matches_df = pd.read_parquet(args.odds)
    print(f"✓ Loaded {len(predictions_df)} predictions")
    
    # Define search space
    print(f"\n2. Defining search space...")
    min_conf_range = [0.40, 0.45, 0.48, 0.50, 0.52, 0.55]
    min_edge_range = [0.00, 0.01, 0.02, 0.03, 0.05]
    odds_ranges = [
        (1.20, 6.00),
        (1.30, 5.00),
        (1.40, 4.00),
        (1.50, 3.50)
    ]
    
    total_combinations = len(min_conf_range) * len(min_edge_range) * len(odds_ranges)
    print(f"✓ Testing {total_combinations} combinations")
    print(f"  Min Confidence: {min_conf_range}")
    print(f"  Min Edge: {min_edge_range}")
    print(f"  Odds Ranges: {odds_ranges}")
    
    # Run grid search
    print(f"\n3. Running grid search...")
    results_df = grid_search(
        matches_df,
        predictions_df,
        min_conf_range,
        min_edge_range,
        odds_ranges,
        target_bets=(args.target_min, args.target_max)
    )
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / 'grid_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to {csv_path}")
    
    # Generate visualizations
    print(f"\n4. Generating visualizations...")
    plot_heatmap(results_df, output_dir / 'roi_heatmap.png')
    plot_pareto_frontier(results_df, output_dir / 'pareto_frontier.png')
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Best overall ROI
    best_roi = results_df.nlargest(1, 'roi').iloc[0]
    print(f"\nBest ROI Configuration:")
    print(f"  Min Conf: {best_roi['min_conf']:.2f}")
    print(f"  Min Edge: {best_roi['min_edge']:.2f}")
    print(f"  Odds Range: ({best_roi['min_odds']:.2f}, {best_roi['max_odds']:.2f})")
    print(f"  Bets: {best_roi['num_bets']}")
    print(f"  ROI: {best_roi['roi']:.2%}")
    print(f"  Win Rate: {best_roi['win_rate']:.2%}")
    print(f"  Sharpe: {best_roi['sharpe']:.2f}")
    
    # Best in target range
    target_results = results_df[results_df['in_target']]
    if len(target_results) > 0:
        best_target = target_results.nlargest(1, 'roi').iloc[0]
        print(f"\nBest in Target Range ({args.target_min}-{args.target_max} bets):")
        print(f"  Min Conf: {best_target['min_conf']:.2f}")
        print(f"  Min Edge: {best_target['min_edge']:.2f}")
        print(f"  Odds Range: ({best_target['min_odds']:.2f}, {best_target['max_odds']:.2f})")
        print(f"  Bets: {best_target['num_bets']}")
        print(f"  ROI: {best_target['roi']:.2%}")
        print(f"  Win Rate: {best_target['win_rate']:.2%}")
        print(f"  Sharpe: {best_target['sharpe']:.2f}")
    else:
        print(f"\n⚠ No configurations found in target range")
        print(f"  Try adjusting target or expanding search space")
    
    # Distribution summary
    print(f"\nDistribution:")
    print(f"  Configurations in target: {len(target_results)} / {len(results_df)}")
    print(f"  Positive ROI: {len(results_df[results_df['roi'] > 0])}")
    print(f"  Sharpe > 1.0: {len(results_df[results_df['sharpe'] > 1.0])}")
    
    print("\n" + "=" * 60)
    print("GRID SEARCH COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
