"""
Head-to-Head (H2H) Features Calculator

Calculates historical performance between two specific teams.
Includes H2H wins, goals, over/under, and BTTS statistics.
"""

from datetime import datetime
from typing import Dict, List
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

from src.shared.logging.logger import get_logger

logger = get_logger(__name__)


class H2HFeaturesCalculator:
    """
    Calculate head-to-head features between two teams
    
    Features:
    - H2H wins/draws/losses
    - H2H goals averages
    - Over/Under statistics
    - Both teams to score (BTTS)
    """
    
    def __init__(self, db_config: Dict):
        """
        Initialize calculator
        
        Args:
            db_config: Database connection configuration
        """
        self.db_config = db_config
        self.conn = None
    
    def connect(self):
        """Establish database connection"""
        self.conn = psycopg2.connect(**self.db_config)
        logger.info("Connected to PostgreSQL for H2H features")
    
    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Disconnected from PostgreSQL")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
    
    def _get_h2h_matches(
        self,
        team1: str,
        team2: str,
        before_date: datetime,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get head-to-head matches between two teams
        
        Args:
            team1: First team name
            team2: Second team name
            before_date: Get matches before this date
            limit: Maximum number of matches
        
        Returns:
            List of H2H match records
        """
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT 
                match_date, home_team, away_team,
                fthg, ftag, ftr
            FROM historical_matches
            WHERE (
                (home_team = %s AND away_team = %s) OR
                (home_team = %s AND away_team = %s)
            )
            AND match_date < %s
            ORDER BY match_date DESC
            LIMIT %s
        """
        
        cursor.execute(query, [team1, team2, team2, team1, before_date, limit])
        matches = cursor.fetchall()
        cursor.close()
        
        return matches
    
    def calculate_h2h_record(
        self,
        home_team: str,
        away_team: str,
        before_date: datetime,
        num_matches: int = 10
    ) -> Dict[str, int]:
        """
        Calculate head-to-head record
        
        Args:
            home_team: Home team name
            away_team: Away team name
            before_date: Calculate before this date
            num_matches: Number of H2H matches to consider
        
        Returns:
            Dictionary with wins/draws/losses from home team perspective
        """
        matches = self._get_h2h_matches(home_team, away_team, before_date, num_matches)
        
        if not matches:
            return {
                'h2h_home_wins': 0,
                'h2h_draws': 0,
                'h2h_away_wins': 0,
                'h2h_total_matches': 0
            }
        
        home_wins = 0
        draws = 0
        away_wins = 0
        
        for match in matches:
            # Determine which team was home in this H2H match
            if match['home_team'] == home_team:
                # Current home team was home in H2H
                if match['ftr'] == 'H':
                    home_wins += 1
                elif match['ftr'] == 'D':
                    draws += 1
                else:
                    away_wins += 1
            else:
                # Current home team was away in H2H
                if match['ftr'] == 'A':
                    home_wins += 1
                elif match['ftr'] == 'D':
                    draws += 1
                else:
                    away_wins += 1
        
        return {
            'h2h_home_wins': home_wins,
            'h2h_draws': draws,
            'h2h_away_wins': away_wins,
            'h2h_total_matches': len(matches)
        }
    
    def calculate_h2h_goals(
        self,
        home_team: str,
        away_team: str,
        before_date: datetime,
        num_matches: int = 10
    ) -> Dict[str, float]:
        """
        Calculate H2H goals statistics
        
        Returns:
            Dictionary with average goals scored by each team in H2H
        """
        matches = self._get_h2h_matches(home_team, away_team, before_date, num_matches)
        
        if not matches:
            return {
                'h2h_home_goals_avg': 1.5,
                'h2h_away_goals_avg': 1.5,
                'h2h_total_goals_avg': 3.0
            }
        
        home_goals = []
        away_goals = []
        
        for match in matches:
            if match['home_team'] == home_team:
                # Current home team was home in H2H
                home_goals.append(match['fthg'] or 0)
                away_goals.append(match['ftag'] or 0)
            else:
                # Current home team was away in H2H
                home_goals.append(match['ftag'] or 0)
                away_goals.append(match['fthg'] or 0)
        
        return {
            'h2h_home_goals_avg': round(np.mean(home_goals), 2),
            'h2h_away_goals_avg': round(np.mean(away_goals), 2),
            'h2h_total_goals_avg': round(np.mean(home_goals) + np.mean(away_goals), 2)
        }
    
    def calculate_h2h_over_under(
        self,
        home_team: str,
        away_team: str,
        before_date: datetime,
        threshold: float = 2.5,
        num_matches: int = 10
    ) -> Dict[str, float]:
        """
        Calculate over/under statistics in H2H matches
        
        Args:
            home_team: Home team name
            away_team: Away team name
            before_date: Calculate before this date
            threshold: Goals threshold (default 2.5)
            num_matches: Number of H2H matches
        
        Returns:
            Dictionary with over/under percentages
        """
        matches = self._get_h2h_matches(home_team, away_team, before_date, num_matches)
        
        if not matches:
            return {
                'h2h_over_2_5_pct': 0.5,
                'h2h_under_2_5_pct': 0.5
            }
        
        over_count = 0
        
        for match in matches:
            total_goals = (match['fthg'] or 0) + (match['ftag'] or 0)
            if total_goals > threshold:
                over_count += 1
        
        over_pct = over_count / len(matches)
        
        return {
            'h2h_over_2_5_pct': round(over_pct, 2),
            'h2h_under_2_5_pct': round(1 - over_pct, 2)
        }
    
    def calculate_h2h_btts(
        self,
        home_team: str,
        away_team: str,
        before_date: datetime,
        num_matches: int = 10
    ) -> Dict[str, float]:
        """
        Calculate both teams to score (BTTS) statistics in H2H
        
        Returns:
            Dictionary with BTTS percentages
        """
        matches = self._get_h2h_matches(home_team, away_team, before_date, num_matches)
        
        if not matches:
            return {
                'h2h_btts_yes_pct': 0.5,
                'h2h_btts_no_pct': 0.5
            }
        
        btts_count = 0
        
        for match in matches:
            home_goals = match['fthg'] or 0
            away_goals = match['ftag'] or 0
            
            if home_goals > 0 and away_goals > 0:
                btts_count += 1
        
        btts_pct = btts_count / len(matches)
        
        return {
            'h2h_btts_yes_pct': round(btts_pct, 2),
            'h2h_btts_no_pct': round(1 - btts_pct, 2)
        }
    
    def calculate_all_h2h_features(
        self,
        home_team: str,
        away_team: str,
        match_date: datetime
    ) -> Dict[str, float]:
        """
        Calculate all H2H features for a match
        
        Args:
            home_team: Home team name
            away_team: Away team name
            match_date: Match date
        
        Returns:
            Dictionary with all H2H features
        """
        features = {}
        
        # H2H record
        record = self.calculate_h2h_record(home_team, away_team, match_date)
        features.update(record)
        
        # H2H goals
        goals = self.calculate_h2h_goals(home_team, away_team, match_date)
        features.update(goals)
        
        # Over/Under
        over_under = self.calculate_h2h_over_under(home_team, away_team, match_date)
        features.update(over_under)
        
        # BTTS
        btts = self.calculate_h2h_btts(home_team, away_team, match_date)
        features.update(btts)
        
        logger.info(f"Calculated {len(features)} H2H features for {home_team} vs {away_team}")
        
        return features
