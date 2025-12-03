"""
Head-to-Head Calculator.

Calculates H2H statistics between teams:
- Wins/Draws in last N H2H matches
- Average goals
- BTTS percentage
- Over 2.5 goals percentage
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class H2HCalculator:
    """
    Calculate head-to-head statistics.
    """
    
    def __init__(self, window: int = 5):
        """
        Args:
            window: Number of recent H2H matches to consider
        """
        self.window = window
    
    def compute_batch(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute H2H features for all matches.
        
        Args:
            matches_df: DataFrame with match data
            
        Returns:
            DataFrame with H2H features
        """
        df = matches_df.sort_values('date').copy()
        
        # Store H2H history per team pair
        h2h_history: Dict[Tuple[str, str], List[Dict]] = {}
        
        features = []
        
        for _, row in df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            
            # Get H2H stats
            h2h_stats = self._get_h2h_stats(h2h_history, home_team, away_team)
            
            features.append({
                'match_id': row['match_id'],
                **h2h_stats
            })
            
            # Update H2H history
            key = tuple(sorted([home_team, away_team]))
            if key not in h2h_history:
                h2h_history[key] = []
            
            home_goals = row.get('FTHG', 0)
            away_goals = row.get('FTAG', 0)
            
            h2h_history[key].append({
                'home_team': home_team,
                'away_team': away_team,
                'result': row['FTR'],
                'home_goals': home_goals,
                'away_goals': away_goals,
                'total_goals': home_goals + away_goals,
                'btts': (home_goals > 0 and away_goals > 0)
            })
            
            # Keep only last N
            h2h_history[key] = h2h_history[key][-self.window:]
        
        return pd.DataFrame(features)
    
    def _get_h2h_stats(
        self,
        h2h_history: Dict,
        home_team: str,
        away_team: str
    ) -> Dict:
        """Extract H2H statistics."""
        key = tuple(sorted([home_team, away_team]))
        history = h2h_history.get(key, [])
        
        if not history:
            return {
                'h2h_wins_home': 0,
                'h2h_draws': 0,
                'h2h_wins_away': 0,
                'h2h_avg_goals': 0,
                'h2h_btts_pct': 0,
                'h2h_over25_pct': 0
            }
        
        # Count results from home team perspective
        wins_home = sum(
            1 for m in history
            if (m['home_team'] == home_team and m['result'] == 'H') or
               (m['away_team'] == home_team and m['result'] == 'A')
        )
        
        draws = sum(1 for m in history if m['result'] == 'D')
        wins_away = len(history) - wins_home - draws
        
        avg_goals = np.mean([m['total_goals'] for m in history])
        btts_pct = np.mean([m['btts'] for m in history])
        over25_pct = np.mean([m['total_goals'] > 2.5 for m in history])
        
        return {
            'h2h_wins_home': wins_home,
            'h2h_draws': draws,
            'h2h_wins_away': wins_away,
            'h2h_avg_goals': avg_goals,
            'h2h_btts_pct': btts_pct,
            'h2h_over25_pct': over25_pct
        }
