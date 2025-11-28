"""
Basic Features Calculator

Calculates fundamental team performance metrics from historical match data.
Includes form, goals, win percentages, and match statistics.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

from src.shared.logging.logger import get_logger

logger = get_logger(__name__)


class BasicFeaturesCalculator:
    """
    Calculate basic features for match prediction
    
    Features:
    - Recent form (last 5, 10 matches)
    - Goals statistics (scored/conceded averages)
    - Win percentages
    - Match statistics (shots, corners, cards)
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
        logger.info("Connected to PostgreSQL for feature calculation")
    
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
    
    def _get_team_matches(
        self,
        team_name: str,
        before_date: datetime,
        limit: int = 20,
        home_only: bool = False,
        away_only: bool = False
    ) -> List[Dict]:
        """
        Get historical matches for a team
        
        Args:
            team_name: Team name
            before_date: Get matches before this date
            limit: Maximum number of matches
            home_only: Only home matches
            away_only: Only away matches
        
        Returns:
            List of match records
        """
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        
        # Build query based on filters
        where_clauses = ["match_date < %s"]
        params = [before_date]
        
        if home_only:
            where_clauses.append("home_team = %s")
            params.append(team_name)
        elif away_only:
            where_clauses.append("away_team = %s")
            params.append(team_name)
        else:
            where_clauses.append("(home_team = %s OR away_team = %s)")
            params.extend([team_name, team_name])
        
        query = f"""
            SELECT 
                match_date, home_team, away_team,
                fthg, ftag, ftr,
                hthg, htag, htr,
                hs, as_, hst, ast,
                hc, ac, hf, af,
                hy, ay, hr, ar
            FROM historical_matches
            WHERE {' AND '.join(where_clauses)}
            ORDER BY match_date DESC
            LIMIT %s
        """
        params.append(limit)
        
        cursor.execute(query, params)
        matches = cursor.fetchall()
        cursor.close()
        
        return matches
    
    def calculate_form(
        self,
        team_name: str,
        before_date: datetime,
        num_matches: int = 5,
        home_only: bool = False,
        away_only: bool = False
    ) -> float:
        """
        Calculate recent form (points from last N matches)
        
        Uses exponential moving average for weighting:
        - Most recent match: highest weight
        - Older matches: lower weight
        
        Args:
            team_name: Team name
            before_date: Calculate form before this date
            num_matches: Number of matches to consider
            home_only: Only home matches
            away_only: Only away matches
        
        Returns:
            Form score (0-3 scale, weighted average)
        """
        matches = self._get_team_matches(
            team_name, before_date, num_matches, home_only, away_only
        )
        
        if not matches:
            return 1.0  # Neutral form
        
        points = []
        for match in matches:
            # Determine if team was home or away
            is_home = match['home_team'] == team_name
            
            # Get result
            if match['ftr'] == 'D':
                pts = 1
            elif (match['ftr'] == 'H' and is_home) or (match['ftr'] == 'A' and not is_home):
                pts = 3
            else:
                pts = 0
            
            points.append(pts)
        
        # Calculate EMA (exponential moving average)
        # More recent matches have higher weight
        weights = np.exp(np.linspace(-1, 0, len(points)))
        weights /= weights.sum()
        
        form_score = np.average(points, weights=weights)
        
        return round(form_score, 2)
    
    def calculate_goals_stats(
        self,
        team_name: str,
        before_date: datetime,
        num_matches: int = 10,
        home_only: bool = False,
        away_only: bool = False
    ) -> Dict[str, float]:
        """
        Calculate goals statistics
        
        Returns:
            Dictionary with:
            - avg_goals_scored
            - avg_goals_conceded
            - goal_difference
        """
        matches = self._get_team_matches(
            team_name, before_date, num_matches, home_only, away_only
        )
        
        if not matches:
            return {
                'avg_goals_scored': 1.0,
                'avg_goals_conceded': 1.0,
                'goal_difference': 0.0
            }
        
        goals_scored = []
        goals_conceded = []
        
        for match in matches:
            is_home = match['home_team'] == team_name
            
            if is_home:
                goals_scored.append(match['fthg'] or 0)
                goals_conceded.append(match['ftag'] or 0)
            else:
                goals_scored.append(match['ftag'] or 0)
                goals_conceded.append(match['fthg'] or 0)
        
        return {
            'avg_goals_scored': round(np.mean(goals_scored), 2),
            'avg_goals_conceded': round(np.mean(goals_conceded), 2),
            'goal_difference': round(np.mean(goals_scored) - np.mean(goals_conceded), 2)
        }
    
    def calculate_win_percentages(
        self,
        team_name: str,
        before_date: datetime,
        num_matches: int = 20,
        home_only: bool = False,
        away_only: bool = False
    ) -> Dict[str, float]:
        """
        Calculate win/draw/loss percentages
        
        Returns:
            Dictionary with:
            - win_pct
            - draw_pct
            - loss_pct
            - clean_sheet_pct
        """
        matches = self._get_team_matches(
            team_name, before_date, num_matches, home_only, away_only
        )
        
        if not matches:
            return {
                'win_pct': 0.33,
                'draw_pct': 0.33,
                'loss_pct': 0.33,
                'clean_sheet_pct': 0.25
            }
        
        wins = 0
        draws = 0
        losses = 0
        clean_sheets = 0
        
        for match in matches:
            is_home = match['home_team'] == team_name
            result = match['ftr']
            
            # Count result
            if result == 'D':
                draws += 1
            elif (result == 'H' and is_home) or (result == 'A' and not is_home):
                wins += 1
            else:
                losses += 1
            
            # Count clean sheets
            goals_conceded = match['ftag'] if is_home else match['fthg']
            if goals_conceded == 0:
                clean_sheets += 1
        
        total = len(matches)
        
        return {
            'win_pct': round(wins / total, 2),
            'draw_pct': round(draws / total, 2),
            'loss_pct': round(losses / total, 2),
            'clean_sheet_pct': round(clean_sheets / total, 2)
        }
    
    def calculate_match_stats(
        self,
        team_name: str,
        before_date: datetime,
        num_matches: int = 10,
        home_only: bool = False,
        away_only: bool = False
    ) -> Dict[str, float]:
        """
        Calculate average match statistics
        
        Returns:
            Dictionary with averages for:
            - shots, shots_on_target
            - corners, fouls
            - yellow_cards, red_cards
        """
        matches = self._get_team_matches(
            team_name, before_date, num_matches, home_only, away_only
        )
        
        if not matches:
            return {
                'avg_shots': 10.0,
                'avg_shots_on_target': 4.0,
                'avg_corners': 5.0,
                'avg_fouls': 12.0,
                'avg_yellow_cards': 2.0,
                'avg_red_cards': 0.1
            }
        
        shots = []
        shots_on_target = []
        corners = []
        fouls = []
        yellow_cards = []
        red_cards = []
        
        for match in matches:
            is_home = match['home_team'] == team_name
            
            if is_home:
                shots.append(match['hs'] or 0)
                shots_on_target.append(match['hst'] or 0)
                corners.append(match['hc'] or 0)
                fouls.append(match['hf'] or 0)
                yellow_cards.append(match['hy'] or 0)
                red_cards.append(match['hr'] or 0)
            else:
                shots.append(match['as_'] or 0)
                shots_on_target.append(match['ast'] or 0)
                corners.append(match['ac'] or 0)
                fouls.append(match['af'] or 0)
                yellow_cards.append(match['ay'] or 0)
                red_cards.append(match['ar'] or 0)
        
        return {
            'avg_shots': round(np.mean(shots), 2),
            'avg_shots_on_target': round(np.mean(shots_on_target), 2),
            'avg_corners': round(np.mean(corners), 2),
            'avg_fouls': round(np.mean(fouls), 2),
            'avg_yellow_cards': round(np.mean(yellow_cards), 2),
            'avg_red_cards': round(np.mean(red_cards), 2)
        }
    
    def calculate_all_basic_features(
        self,
        home_team: str,
        away_team: str,
        match_date: datetime
    ) -> Dict[str, float]:
        """
        Calculate all basic features for a match
        
        Args:
            home_team: Home team name
            away_team: Away team name
            match_date: Match date
        
        Returns:
            Dictionary with all basic features
        """
        features = {}
        
        # Home team features
        features['home_form_last_5'] = self.calculate_form(home_team, match_date, 5, home_only=True)
        features['home_form_last_10'] = self.calculate_form(home_team, match_date, 10)
        
        home_goals = self.calculate_goals_stats(home_team, match_date, 10, home_only=True)
        features['home_avg_goals_scored'] = home_goals['avg_goals_scored']
        features['home_avg_goals_conceded'] = home_goals['avg_goals_conceded']
        features['home_goal_difference'] = home_goals['goal_difference']
        
        home_win_pct = self.calculate_win_percentages(home_team, match_date, 20, home_only=True)
        features['home_win_pct'] = home_win_pct['win_pct']
        features['home_draw_pct'] = home_win_pct['draw_pct']
        features['home_clean_sheet_pct'] = home_win_pct['clean_sheet_pct']
        
        home_stats = self.calculate_match_stats(home_team, match_date, 10, home_only=True)
        features['home_avg_shots'] = home_stats['avg_shots']
        features['home_avg_shots_on_target'] = home_stats['avg_shots_on_target']
        features['home_avg_corners'] = home_stats['avg_corners']
        
        # Away team features
        features['away_form_last_5'] = self.calculate_form(away_team, match_date, 5, away_only=True)
        features['away_form_last_10'] = self.calculate_form(away_team, match_date, 10)
        
        away_goals = self.calculate_goals_stats(away_team, match_date, 10, away_only=True)
        features['away_avg_goals_scored'] = away_goals['avg_goals_scored']
        features['away_avg_goals_conceded'] = away_goals['avg_goals_conceded']
        features['away_goal_difference'] = away_goals['goal_difference']
        
        away_win_pct = self.calculate_win_percentages(away_team, match_date, 20, away_only=True)
        features['away_win_pct'] = away_win_pct['win_pct']
        features['away_draw_pct'] = away_win_pct['draw_pct']
        features['away_clean_sheet_pct'] = away_win_pct['clean_sheet_pct']
        
        away_stats = self.calculate_match_stats(away_team, match_date, 10, away_only=True)
        features['away_avg_shots'] = away_stats['avg_shots']
        features['away_avg_shots_on_target'] = away_stats['avg_shots_on_target']
        features['away_avg_corners'] = away_stats['avg_corners']
        
        logger.info(f"Calculated {len(features)} basic features for {home_team} vs {away_team}")
        
        return features
