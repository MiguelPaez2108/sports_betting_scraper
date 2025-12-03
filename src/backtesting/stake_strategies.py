"""
Stake Strategies for bankroll management.

Implements various staking methods:
- Fixed stake
- Kelly Criterion
- Fractional Kelly
- Limited Kelly (with volatility cap)
- Percentage of bankroll
"""
from abc import ABC, abstractmethod
import numpy as np


class StakeStrategy(ABC):
    """Base class for staking strategies."""
    
    @abstractmethod
    def compute_stake(self, bankroll: float, edge: float, odds: float, **kwargs) -> float:
        """
        Compute stake amount.
        
        Args:
            bankroll: Current bankroll
            edge: Expected edge (prob - implied_prob)
            odds: Decimal odds
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            Stake amount
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return strategy name."""
        pass


class FixedStake(StakeStrategy):
    """Fixed stake amount regardless of edge or bankroll."""
    
    def __init__(self, stake: float = 10.0):
        self.stake = stake
    
    def compute_stake(self, bankroll: float, edge: float, odds: float, **kwargs) -> float:
        return min(self.stake, bankroll)
    
    def get_name(self) -> str:
        return f"Fixed({self.stake})"


class KellyStake(StakeStrategy):
    """Full Kelly Criterion: f = (bp - q) / b where b = odds - 1."""
    
    def compute_stake(self, bankroll: float, edge: float, odds: float, **kwargs) -> float:
        prob = kwargs.get('prob', 0.5)  # Win probability
        
        b = odds - 1  # Net odds
        p = prob
        q = 1 - p
        
        # Kelly formula
        kelly_fraction = (b * p - q) / b
        
        # Only bet if positive edge
        if kelly_fraction <= 0:
            return 0.0
        
        stake = bankroll * kelly_fraction
        return max(0, min(stake, bankroll))
    
    def get_name(self) -> str:
        return "Kelly"


class FractionalKelly(StakeStrategy):
    """Fractional Kelly: reduces variance by betting fraction of Kelly."""
    
    def __init__(self, fraction: float = 0.25):
        """
        Args:
            fraction: Fraction of Kelly to bet (e.g., 0.25 = Quarter Kelly)
        """
        if not (0 < fraction <= 1):
            raise ValueError("Fraction must be between 0 and 1")
        self.fraction = fraction
        self.kelly = KellyStake()
    
    def compute_stake(self, bankroll: float, edge: float, odds: float, **kwargs) -> float:
        full_kelly = self.kelly.compute_stake(bankroll, edge, odds, **kwargs)
        return full_kelly * self.fraction
    
    def get_name(self) -> str:
        return f"FracKelly({self.fraction})"


class LimitedKelly(StakeStrategy):
    """
    Kelly with hard limits and volatility cap.
    
    Implements:
    - Max stake as % of bankroll
    - Absolute max stake
    - Volatility-based reduction
    """
    
    def __init__(
        self,
        fraction: float = 0.25,
        max_stake_pct: float = 0.02,  # Max 2% of bankroll
        max_stake_abs: float = 100.0,  # Absolute max
        volatility_cap: bool = True
    ):
        self.fraction = fraction
        self.max_stake_pct = max_stake_pct
        self.max_stake_abs = max_stake_abs
        self.volatility_cap = volatility_cap
        self.kelly = FractionalKelly(fraction)
    
    def compute_stake(self, bankroll: float, edge: float, odds: float, **kwargs) -> float:
        # Base Kelly stake
        stake = self.kelly.compute_stake(bankroll, edge, odds, **kwargs)
        
        # Apply percentage limit
        max_by_pct = bankroll * self.max_stake_pct
        stake = min(stake, max_by_pct)
        
        # Apply absolute limit
        stake = min(stake, self.max_stake_abs)
        
        # Apply volatility cap if enabled
        if self.volatility_cap:
            volatility = kwargs.get('volatility', 0.0)
            if volatility > 0.15:  # High volatility threshold
                reduction_factor = 0.5
                stake *= reduction_factor
        
        return max(0, min(stake, bankroll))
    
    def get_name(self) -> str:
        return f"LimitedKelly({self.fraction},{self.max_stake_pct:.1%})"


class PercentageBankroll(StakeStrategy):
    """Fixed percentage of current bankroll."""
    
    def __init__(self, percentage: float = 0.01):
        """
        Args:
            percentage: Percentage of bankroll to bet (e.g., 0.01 = 1%)
        """
        if not (0 < percentage <= 0.1):
            raise ValueError("Percentage should be between 0 and 10%")
        self.percentage = percentage
    
    def compute_stake(self, bankroll: float, edge: float, odds: float, **kwargs) -> float:
        return bankroll * self.percentage
    
    def get_name(self) -> str:
        return f"Percentage({self.percentage:.1%})"


def create_stake_strategy(strategy_name: str, **params) -> StakeStrategy:
    """
    Factory function to create stake strategies.
    
    Args:
        strategy_name: Name of strategy ('fixed', 'kelly', 'frac_kelly', 'limited_kelly', 'percentage')
        **params: Strategy-specific parameters
        
    Returns:
        StakeStrategy instance
    """
    strategies = {
        'fixed': FixedStake,
        'kelly': KellyStake,
        'frac_kelly': FractionalKelly,
        'limited_kelly': LimitedKelly,
        'percentage': PercentageBankroll
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Choose from {list(strategies.keys())}")
    
    return strategies[strategy_name](**params)
