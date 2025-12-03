"""
Poisson Calculator - Probabilistic baseline features.

Estimates attack/defense strengths and calculates Poisson probabilities.
"""
import pandas as pd
import numpy as np
from scipy.stats import poisson
from typing import Dict


class PoissonCalculator:
    """
    Calculate Poisson-based probability features.
    """
    
    def __init__(self, window: int = 10):
        """
        Args:
            window: Number of recent matches for lambda estimation
        """
        self.window = window
    
    def compute_batch(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Poisson features for all matches.
        
        Args:
            matches_df: DataFrame with match data
            
        Returns:
            DataFrame with Poisson probability features
        """
        df = matches_df.sort_values('date').copy()
        
        # Track team stats
        team_stats: Dict[str, Dict] = {}
        
        features = []
        
        for _, row in df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            
            # Get lambdas
            lambda_home, lambda_away = self._estimate_lambdas(
                team_stats, home_team, away_team
            )
            
            # Calculate Poisson probabilities
            probs = self._poisson_probabilities(lambda_home, lambda_away)
            
            features.append({
                'match_id': row['match_id'],
                'lambda_home': lambda_home,
                'lambda_away': lambda_away,
                'poisson_prob_home': probs['home'],
                'poisson_prob_draw': probs['draw'],
                'poisson_prob_away': probs['away']
            })
            
            # Update team stats
            home_goals = row.get('FTHG', 0)
            away_goals = row.get('FTAG', 0)
            
            if home_team not in team_stats:
                team_stats[home_team] = {'gf': [], 'ga': []}
            if away_team not in team_stats:
                team_stats[away_team] = {'gf': [], 'ga': []}
            
            team_stats[home_team]['gf'].append(home_goals)
            team_stats[home_team]['ga'].append(away_goals)
            team_stats[away_team]['gf'].append(away_goals)
            team_stats[away_team]['ga'].append(home_goals)
            
            # Keep only last N
            team_stats[home_team]['gf'] = team_stats[home_team]['gf'][-self.window:]
            team_stats[home_team]['ga'] = team_stats[home_team]['ga'][-self.window:]
            team_stats[away_team]['gf'] = team_stats[away_team]['gf'][-self.window:]
            team_stats[away_team]['ga'] = team_stats[away_team]['ga'][-self.window:]
        
        return pd.DataFrame(features)
    
    def _estimate_lambdas(
        self,
        team_stats: Dict,
        home_team: str,
        away_team: str
    ) -> tuple[float, float]:
        """Estimate expected goals for each team."""
        # Default league average
        default_lambda = 1.5
        
        # Home team attack strength
        if home_team in team_stats and team_stats[home_team]['gf']:
            home_attack = np.mean(team_stats[home_team]['gf'])
        else:
            home_attack = default_lambda
        
        # Away team defense strength
        if away_team in team_stats and team_stats[away_team]['ga']:
            away_defense = np.mean(team_stats[away_team]['ga'])
        else:
            away_defense = default_lambda
        
        # Away team attack strength
        if away_team in team_stats and team_stats[away_team]['gf']:
            away_attack = np.mean(team_stats[away_team]['gf'])
        else:
            away_attack = default_lambda
        
        # Home team defense strength
        if home_team in team_stats and team_stats[home_team]['ga']:
            home_defense = np.mean(team_stats[home_team]['ga'])
        else:
            home_defense = default_lambda
        
        # Combine attack and defense
        lambda_home = (home_attack + away_defense) / 2
        lambda_away = (away_attack + home_defense) / 2
        
        # Apply home advantage (10% boost)
        lambda_home *= 1.1
        
        return lambda_home, lambda_away
    
    def _poisson_probabilities(
        self,
        lambda_home: float,
        lambda_away: float,
        max_goals: int = 5
    ) -> Dict[str, float]:
        """Calculate match outcome probabilities using Poisson."""
        prob_home = 0.0
        prob_draw = 0.0
        prob_away = 0.0
        
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                prob = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
                
                if i > j:
                    prob_home += prob
                elif i == j:
                    prob_draw += prob
                else:
                    prob_away += prob
        
        # Normalize
        total = prob_home + prob_draw + prob_away
        
        return {
            'home': prob_home / total,
            'draw': prob_draw / total,
            'away': prob_away / total
        }
