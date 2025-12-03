"""
Run Monte Carlo simulation on backtest results.

Usage:
    python scripts/run_monte_carlo.py --trades analysis/backtest/trades.csv
"""
import argparse
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.risk.monte_carlo import MonteCarloSimulator


def main():
    parser = argparse.ArgumentParser(description='Run Monte Carlo risk simulation')
    parser.add_argument('--trades', type=str, required=True, help='Path to trades CSV')
    parser.add_argument('--n-sims', type=int, default=10000, help='Number of simulations')
    parser.add_argument('--initial-bankroll', type=float, default=1000.0, help='Initial bankroll')
    parser.add_argument('--output', type=str, default='analysis/monte_carlo', help='Output directory')
    args = parser.parse_args()
    
    print("=" * 60)
    print("MONTE CARLO RISK SIMULATION")
    print("=" * 60)
    
    # Load trades
    print(f"\n1. Loading trades from {args.trades}...")
    trades_df = pd.read_csv(args.trades)
    print(f"✓ Loaded {len(trades_df)} trades")
    
    # Run simulation
    print(f"\n2. Running {args.n_sims} simulations...")
    simulator = MonteCarloSimulator(n_simulations=args.n_sims)
    result = simulator.simulate(trades_df, initial_bankroll=args.initial_bankroll)
    
    # Print summary
    simulator.print_summary(result)
    
    # Generate plots
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n3. Generating visualizations...")
    simulator.plot_results(result, str(output_dir / 'simulation_results.png'))
    
    # Save detailed results
    results_df = pd.DataFrame({
        'final_bankroll': result.final_bankrolls,
        'max_drawdown': result.max_drawdowns
    })
    results_df.to_csv(output_dir / 'simulation_data.csv', index=False)
    print(f"✓ Detailed results saved to {output_dir / 'simulation_data.csv'}")
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
