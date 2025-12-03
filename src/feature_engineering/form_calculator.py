"""
Form Calculator - Rolling window statistics.

Calculates momentum features:
- Points in last N matches
- Goals for/against in last N matches
- Separate home/away form
"""
import pandas as pd
import numpy as np
from typing import Dict, List


class FormCalculator:
    """
    Calculate rolling form statistics.
    """
    
    def __init__(self, window: int = 5):
        """
        Args:
            window: Number of recent matches to consider
        """
        self.window = window
        
    def _calculate_points(self, result: str, is_home: bool) -> int:
        """Calculate points from result."""
        if result == 'H':
            return 3 if is_home else 0
        elif result == 'D':
            return 1
        else:  # 'A'
            return 0 if is_home else 3
    
    def compute_batch(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute form features for all matches.
        
        Args:
            matches_df: DataFrame with match data
            
        Returns:
            DataFrame with form features
        """
        df = matches_df.sort_values('date').copy()
        
        # Initialize storage
        team_history: Dict[str, List[Dict]] = {}
        
        features = []
        
        for _, row in df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            result = row['FTR']
            
            # Get recent form for both teams
            home_form = self._get_team_form(team_history.get(home_team, []), is_home=True)
            away_form = self._get_team_form(team_history.get(away_team, []), is_home=False)
            
            features.append({
                'match_id': row['match_id'],
                **home_form,
                **away_form
            })
            
            # Update history
            home_goals = row.get('FTHG', 0)
            away_goals = row.get('FTAG', 0)
            
            if home_team not in team_history:
                team_history[home_team] = []
            if away_team not in team_history:
                team_history[away_team] = []
            
            team_history[home_team].append({
                'is_home': True,
                'result': result,
                'gf': home_goals,
                'ga': away_goals,
                'points': self._calculate_points(result, True)
            })
            
            team_history[away_team].append({
                'is_home': False,
                'result': result,
                'gf': away_goals,
                'ga': home_goals,
                'points': self._calculate_points(result, False)
            })
            
            # Keep only last N matches
            team_history[home_team] = team_history[home_team][-self.window:]
            team_history[away_team] = team_history[away_team][-self.window:]
        
        return pd.DataFrame(features)
    
    def _get_team_form(self, history: List[Dict], is_home: bool) -> Dict:
        """Extract form features from team history."""
        prefix = 'home' if is_home else 'away'
        
        if not history:
            return {
                f'pts_last{self.window}_{prefix}': 0,
                f'gf_last{self.window}_{prefix}': 0,
                f'ga_last{self.window}_{prefix}': 0,
                f'gd_last{self.window}_{prefix}': 0
            }
        
        pts = sum(m['points'] for m in history)
        gf = sum(m['gf'] for m in history)
        ga = sum(m['ga'] for m in history)
        
        return {
            f'pts_last{self.window}_{prefix}': pts,
            f'gf_last{self.window}_{prefix}': gf,
            f'ga_last{self.window}_{prefix}': ga,
            f'gd_last{self.window}_{prefix}': gf - ga
        }
