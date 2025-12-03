"""
Monte Carlo Simulation for Risk Analysis.

Simulates thousands of betting scenarios to estimate:
- Maximum drawdown probability
- Ruin probability
- Expected bankroll growth
- Variance of returns
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class SimulationResult:
    """Results from Monte Carlo simulation."""
    final_bankrolls: np.ndarray
    max_drawdowns: np.ndarray
    ruin_probability: float
    median_final_bankroll: float
    expected_growth: float
    var_95: float  # Value at Risk (95th percentile loss)
    cvar_95: float  # Conditional VaR (expected loss beyond VaR)


class MonteCarloSimulator:
    """
    Monte Carlo simulator for betting risk analysis.
    
    Simulates multiple scenarios by:
    1. Shuffling bet sequence (breaks temporal dependencies)
    2. Simulating bankroll evolution
    3. Calculating risk metrics
    """
    
    def __init__(self, n_simulations: int = 1000, random_state: int = 42):
        """
        Args:
            n_simulations: Number of Monte Carlo runs
            random_state: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.random_state = random_state
        np.random.seed(random_state)
    
    def simulate(
        self,
        trades_df: pd.DataFrame,
        initial_bankroll: float = 1000.0
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation.
        
        Args:
            trades_df: Historical trades with 'profit' column
            initial_bankroll: Starting capital
            
        Returns:
            SimulationResult with risk metrics
        """
        profits = trades_df['profit'].values
        n_bets = len(profits)
        
        print(f"Running {self.n_simulations} simulations on {n_bets} bets...")
        
        final_bankrolls = np.zeros(self.n_simulations)
        max_drawdowns = np.zeros(self.n_simulations)
        ruins = 0
        
        for i in range(self.n_simulations):
            # Shuffle bet sequence
            shuffled_profits = np.random.permutation(profits)
            
            # Simulate bankroll evolution
            bankroll_curve = initial_bankroll + np.cumsum(shuffled_profits)
            
            # Check for ruin
            if np.any(bankroll_curve <= 0):
                ruins += 1
                final_bankrolls[i] = 0
                max_drawdowns[i] = 1.0  # 100% drawdown
            else:
                final_bankrolls[i] = bankroll_curve[-1]
                
                # Calculate max drawdown
                peak = np.maximum.accumulate(np.insert(bankroll_curve, 0, initial_bankroll))
                peak = peak[1:]  # Remove first element
                drawdowns = (peak - bankroll_curve) / peak
                max_drawdowns[i] = np.max(drawdowns)
        
        # Calculate risk metrics
        ruin_prob = ruins / self.n_simulations
        median_final = np.median(final_bankrolls)
        expected_growth = (np.mean(final_bankrolls) - initial_bankroll) / initial_bankroll
        
        # Value at Risk (VaR) and Conditional VaR (CVaR)
        returns = (final_bankrolls - initial_bankroll) / initial_bankroll
        var_95 = np.percentile(returns, 5)  # 5th percentile (worst 5%)
        cvar_95 = np.mean(returns[returns <= var_95])  # Expected loss beyond VaR
        
        return SimulationResult(
            final_bankrolls=final_bankrolls,
            max_drawdowns=max_drawdowns,
            ruin_probability=ruin_prob,
            median_final_bankroll=median_final,
            expected_growth=expected_growth,
            var_95=var_95,
            cvar_95=cvar_95
        )
    
    def plot_results(self, result: SimulationResult, output_path: str) -> None:
        """Generate visualization of simulation results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Final bankroll distribution
        ax = axes[0, 0]
        ax.hist(result.final_bankrolls, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(result.median_final_bankroll, color='red', linestyle='--', 
                   label=f'Median: ${result.median_final_bankroll:.0f}')
        ax.set_xlabel('Final Bankroll ($)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Final Bankroll')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. Max drawdown distribution
        ax = axes[0, 1]
        ax.hist(result.max_drawdowns, bins=50, alpha=0.7, edgecolor='black', color='orange')
        ax.axvline(np.median(result.max_drawdowns), color='red', linestyle='--',
                   label=f'Median: {np.median(result.max_drawdowns):.1%}')
        ax.set_xlabel('Max Drawdown')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Maximum Drawdown')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 3. Growth distribution
        ax = axes[1, 0]
        returns = (result.final_bankrolls - 1000) / 1000  # Assuming 1000 initial
        ax.hist(returns, bins=50, alpha=0.7, edgecolor='black', color='green')
        ax.axvline(result.expected_growth, color='red', linestyle='--',
                   label=f'Expected: {result.expected_growth:.1%}')
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Return (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Returns')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 4. Risk metrics summary
        ax = axes[1, 1]
        ax.axis('off')
        
        metrics_text = f"""
        RISK METRICS SUMMARY
        {'='*40}
        
        Ruin Probability: {result.ruin_probability:.2%}
        
        Expected Growth: {result.expected_growth:.2%}
        Median Final: ${result.median_final_bankroll:.0f}
        
        Max Drawdown (Median): {np.median(result.max_drawdowns):.1%}
        Max Drawdown (95th %ile): {np.percentile(result.max_drawdowns, 95):.1%}
        
        VaR (95%): {result.var_95:.2%}
        CVaR (95%): {result.cvar_95:.2%}
        
        Simulations: {self.n_simulations:,}
        """
        
        ax.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Simulation plot saved to {output_path}")
    
    def print_summary(self, result: SimulationResult) -> None:
        """Print summary of simulation results."""
        print("\n" + "=" * 60)
        print("MONTE CARLO SIMULATION RESULTS")
        print("=" * 60)
        
        print(f"\nRuin Probability: {result.ruin_probability:.2%}")
        
        if result.ruin_probability > 0.10:
            print("  ⚠️  WARNING: High ruin risk!")
        elif result.ruin_probability > 0.05:
            print("  ⚠️  CAUTION: Moderate ruin risk")
        else:
            print("  ✅ Low ruin risk")
        
        print(f"\nExpected Growth: {result.expected_growth:.2%}")
        print(f"Median Final Bankroll: ${result.median_final_bankroll:.2f}")
        
        print(f"\nDrawdown Risk:")
        print(f"  Median Max DD: {np.median(result.max_drawdowns):.1%}")
        print(f"  95th %ile DD: {np.percentile(result.max_drawdowns, 95):.1%}")
        print(f"  99th %ile DD: {np.percentile(result.max_drawdowns, 99):.1%}")
        
        print(f"\nValue at Risk (95%):")
        print(f"  VaR: {result.var_95:.2%}")
        print(f"  CVaR: {result.cvar_95:.2%}")
        
        print("\n" + "=" * 60)
