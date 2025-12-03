"""
ELO Rating Calculator with incremental updates.

Features:
- Incremental ELO per match
- Home advantage parameter
- League-specific K factors
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional


class EloCalculator:
    """
    Calculate ELO ratings for teams.
    """
    
    def __init__(self, k_factor: float = 20.0, home_advantage: float = 100.0):
        """
        Args:
            k_factor: ELO update rate (higher = more volatile)
            home_advantage: Points added to home team
        """
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.team_ratings: Dict[str, float] = {}
        
    def _get_rating(self, team: str, default: float = 1500.0) -> float:
        """Get current ELO rating for team."""
        return self.team_ratings.get(team, default)
    
    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for team A."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def _update_rating(self, rating: float, actual: float, expected: float) -> float:
        """Update ELO rating based on match result."""
        return rating + self.k_factor * (actual - expected)
    
    def compute_match(
        self,
        home_team: str,
        away_team: str,
        result: str  # 'H', 'D', 'A'
    ) -> tuple[float, float, float, float]:
        """
        Compute ELO for a single match and update ratings.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            result: Match result
            
        Returns:
            (elo_home_before, elo_away_before, elo_home_after, elo_away_after)
        """
        # Get current ratings
        elo_home = self._get_rating(home_team)
        elo_away = self._get_rating(away_team)
        
        # Apply home advantage
        elo_home_adj = elo_home + self.home_advantage
        
        # Expected scores
        expected_home = self._expected_score(elo_home_adj, elo_away)
        expected_away = 1 - expected_home
        
        # Actual scores
        if result == 'H':
            actual_home, actual_away = 1.0, 0.0
        elif result == 'D':
            actual_home, actual_away = 0.5, 0.5
        else:  # 'A'
            actual_home, actual_away = 0.0, 1.0
        
        # Update ratings (without home advantage for storage)
        new_elo_home = self._update_rating(elo_home, actual_home, expected_home)
        new_elo_away = self._update_rating(elo_away, actual_away, expected_away)
        
        # Store updated ratings
        self.team_ratings[home_team] = new_elo_home
        self.team_ratings[away_team] = new_elo_away
        
        return elo_home, elo_away, new_elo_home, new_elo_away
    
    def compute_batch(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute ELO for all matches in chronological order.
        
        Args:
            matches_df: DataFrame with columns [home_team, away_team, FTR, date]
            
        Returns:
            DataFrame with ELO features
        """
        # Ensure sorted by date
        df = matches_df.sort_values('date').copy()
        
        elo_home_list = []
        elo_away_list = []
        elo_diff_list = []
        
        for _, row in df.iterrows():
            elo_h, elo_a, _, _ = self.compute_match(
                row['home_team'],
                row['away_team'],
                row['FTR']
            )
            
            elo_home_list.append(elo_h)
            elo_away_list.append(elo_a)
            elo_diff_list.append(elo_h - elo_a)
        
        return pd.DataFrame({
            'match_id': df['match_id'].values,
            'elo_home': elo_home_list,
            'elo_away': elo_away_list,
            'elo_diff': elo_diff_list
        })
