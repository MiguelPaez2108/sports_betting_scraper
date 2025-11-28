"""
Advanced Features Calculator

Calculates sophisticated statistical features for match prediction.
Includes Elo ratings, xG, Poisson parameters, momentum, and fatigue.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from scipy.stats import poisson

from src.shared.logging.logger import get_logger

logger = get_logger(__name__)


class AdvancedFeaturesCalculator:
    """
    Calculate advanced statistical features for match prediction
    
    Features:
    - Elo rating system (dynamic team strength)
    - Expected goals (xG)
    - Poisson distribution parameters
    - Momentum and streaks
    - Fatigue indicators
    - Attack/defense strength
    """
    
    # Elo rating constants
    K_FACTOR = 32  # Sensitivity to new results
    HOME_ADVANTAGE = 100  # Elo points for home advantage
    INITIAL_ELO = 1500  # Starting Elo for new teams
    
    def __init__(self, db_config: Dict):
        """
        Initialize calculator
        
        Args:
            db_config: Database connection configuration
        """
        self.db_config = db_config
        self.conn = None
        self.elo_ratings = {}  # Cache for Elo ratings
    
    def connect(self):
        """Establish database connection"""
        self.conn = psycopg2.connect(**self.db_config)
        logger.info("Connected to PostgreSQL for advanced features")
    
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
        limit: int = 50
    ) -> List[Dict]:
        """Get historical matches for a team"""
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT 
                match_date, home_team, away_team,
                fthg, ftag, ftr,
                hs, as_, hst, ast,
                hc, ac
            FROM historical_matches
            WHERE (home_team = %s OR away_team = %s)
                AND match_date < %s
            ORDER BY match_date DESC
            LIMIT %s
        """
        
        cursor.execute(query, [team_name, team_name, before_date, limit])
        matches = cursor.fetchall()
        cursor.close()
        
        return matches
    
    def calculate_elo_rating(
        self,
        team_name: str,
        before_date: datetime,
        league_code: str = None
    ) -> float:
        """
        Calculate Elo rating for a team
        
        Elo rating is a dynamic measure of team strength that updates
        after each match based on expected vs actual result.
        
        Args:
            team_name: Team name
            before_date: Calculate Elo before this date
            league_code: Optional league filter
        
        Returns:
            Current Elo rating
        """
        # Check cache
        cache_key = f"{team_name}_{before_date.date()}"
        if cache_key in self.elo_ratings:
            return self.elo_ratings[cache_key]
        
        # Get all matches up to this date
        matches = self._get_team_matches(team_name, before_date, limit=100)
        
        if not matches:
            return self.INITIAL_ELO
        
        # Start with initial Elo
        elo = self.INITIAL_ELO
        
        # Process matches chronologically (reverse order)
        for match in reversed(matches):
            is_home = match['home_team'] == team_name
            
            # Determine opponent and result
            if is_home:
                opponent_goals = match['ftag'] or 0
                team_goals = match['fthg'] or 0
            else:
                opponent_goals = match['fthg'] or 0
                team_goals = match['ftag'] or 0
            
            # Actual result (1 = win, 0.5 = draw, 0 = loss)
            if team_goals > opponent_goals:
                actual_score = 1.0
            elif team_goals == opponent_goals:
                actual_score = 0.5
            else:
                actual_score = 0.0
            
            # Expected score based on Elo difference
            # Assume opponent has average Elo (1500)
            # In production, you'd track opponent Elo too
            opponent_elo = self.INITIAL_ELO
            
            if is_home:
                elo_diff = elo + self.HOME_ADVANTAGE - opponent_elo
            else:
                elo_diff = elo - (opponent_elo + self.HOME_ADVANTAGE)
            
            expected_score = 1 / (1 + 10 ** (-elo_diff / 400))
            
            # Update Elo
            elo += self.K_FACTOR * (actual_score - expected_score)
        
        # Cache result
        self.elo_ratings[cache_key] = round(elo, 1)
        
        return round(elo, 1)
    
    def calculate_expected_goals(
        self,
        team_name: str,
        before_date: datetime,
        is_home: bool = True,
        num_matches: int = 10
    ) -> float:
        """
        Calculate expected goals (xG) for a team
        
        xG is calculated based on:
        - Shots on target conversion rate
        - Historical goals per shot
        - Quality of chances (approximated by shots on target ratio)
        
        Args:
            team_name: Team name
            before_date: Calculate xG before this date
            is_home: Whether team is playing at home
            num_matches: Number of matches to consider
        
        Returns:
            Expected goals per match
        """
        matches = self._get_team_matches(team_name, before_date, num_matches)
        
        if not matches:
            return 1.5  # League average
        
        total_goals = 0
        total_shots = 0
        total_shots_on_target = 0
        
        for match in matches:
            team_is_home = match['home_team'] == team_name
            
            # Only consider matches where team played in same venue
            if (is_home and not team_is_home) or (not is_home and team_is_home):
                continue
            
            if team_is_home:
                goals = match['fthg'] or 0
                shots = match['hs'] or 0
                shots_on_target = match['hst'] or 0
            else:
                goals = match['ftag'] or 0
                shots = match['as_'] or 0
                shots_on_target = match['ast'] or 0
            
            total_goals += goals
            total_shots += shots
            total_shots_on_target += shots_on_target
        
        if total_shots == 0:
            return 1.5
        
        # Calculate conversion rate
        conversion_rate = total_goals / total_shots if total_shots > 0 else 0.1
        
        # Calculate shot quality (shots on target ratio)
        shot_quality = total_shots_on_target / total_shots if total_shots > 0 else 0.3
        
        # Average shots per match
        avg_shots = total_shots / len(matches) if matches else 10
        
        # xG = shots * quality * conversion
        xg = avg_shots * shot_quality * conversion_rate
        
        return round(xg, 2)
    
    def calculate_poisson_parameters(
        self,
        home_team: str,
        away_team: str,
        before_date: datetime
    ) -> Dict[str, float]:
        """
        Calculate Poisson distribution parameters for goal prediction
        
        Poisson distribution is commonly used to model goal scoring in football.
        Lambda (λ) represents the average goals per match.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            before_date: Calculate before this date
        
        Returns:
            Dictionary with lambda values and probabilities
        """
        # Get attack and defense strengths
        home_attack = self._calculate_attack_strength(home_team, before_date, is_home=True)
        home_defense = self._calculate_defense_strength(home_team, before_date, is_home=True)
        away_attack = self._calculate_attack_strength(away_team, before_date, is_home=False)
        away_defense = self._calculate_defense_strength(away_team, before_date, is_home=False)
        
        # League average (approximate)
        league_avg_goals = 1.4
        
        # Calculate expected goals using Poisson
        lambda_home = home_attack * away_defense * league_avg_goals
        lambda_away = away_attack * home_defense * league_avg_goals
        
        # Calculate probabilities
        prob_home_win = sum(
            poisson.pmf(i, lambda_home) * poisson.cdf(i - 1, lambda_away)
            for i in range(0, 10)
        )
        
        prob_away_win = sum(
            poisson.pmf(i, lambda_away) * poisson.cdf(i - 1, lambda_home)
            for i in range(0, 10)
        )
        
        prob_draw = 1 - prob_home_win - prob_away_win
        
        return {
            'lambda_home': round(lambda_home, 2),
            'lambda_away': round(lambda_away, 2),
            'prob_home_win': round(prob_home_win, 3),
            'prob_draw': round(prob_draw, 3),
            'prob_away_win': round(prob_away_win, 3)
        }
    
    def _calculate_attack_strength(
        self,
        team_name: str,
        before_date: datetime,
        is_home: bool = True
    ) -> float:
        """Calculate attack strength relative to league average"""
        matches = self._get_team_matches(team_name, before_date, 10)
        
        if not matches:
            return 1.0
        
        goals_scored = []
        for match in matches:
            team_is_home = match['home_team'] == team_name
            
            if (is_home and team_is_home) or (not is_home and not team_is_home):
                goals = match['fthg'] if team_is_home else match['ftag']
                goals_scored.append(goals or 0)
        
        if not goals_scored:
            return 1.0
        
        avg_goals = np.mean(goals_scored)
        league_avg = 1.4  # Approximate league average
        
        return avg_goals / league_avg if league_avg > 0 else 1.0
    
    def _calculate_defense_strength(
        self,
        team_name: str,
        before_date: datetime,
        is_home: bool = True
    ) -> float:
        """Calculate defense strength relative to league average"""
        matches = self._get_team_matches(team_name, before_date, 10)
        
        if not matches:
            return 1.0
        
        goals_conceded = []
        for match in matches:
            team_is_home = match['home_team'] == team_name
            
            if (is_home and team_is_home) or (not is_home and not team_is_home):
                goals = match['ftag'] if team_is_home else match['fthg']
                goals_conceded.append(goals or 0)
        
        if not goals_conceded:
            return 1.0
        
        avg_conceded = np.mean(goals_conceded)
        league_avg = 1.4
        
        return avg_conceded / league_avg if league_avg > 0 else 1.0
    
    def calculate_momentum(
        self,
        team_name: str,
        before_date: datetime,
        num_matches: int = 5
    ) -> Dict[str, float]:
        """
        Calculate momentum metrics
        
        Momentum captures recent performance trends and streaks.
        
        Returns:
            Dictionary with momentum indicators
        """
        matches = self._get_team_matches(team_name, before_date, num_matches)
        
        if not matches:
            return {
                'current_streak': 0,
                'momentum_score': 0.0,
                'points_trend': 0.0
            }
        
        # Calculate current streak
        streak = 0
        last_result = None
        
        for match in matches:
            is_home = match['home_team'] == team_name
            result = match['ftr']
            
            # Determine if won, drew, or lost
            if result == 'D':
                current_result = 'D'
            elif (result == 'H' and is_home) or (result == 'A' and not is_home):
                current_result = 'W'
            else:
                current_result = 'L'
            
            if last_result is None:
                last_result = current_result
                streak = 1
            elif current_result == last_result:
                streak += 1
            else:
                break
        
        # Assign sign to streak (positive for wins, negative for losses)
        if last_result == 'W':
            streak = abs(streak)
        elif last_result == 'L':
            streak = -abs(streak)
        else:
            streak = 0
        
        # Calculate momentum score (weighted recent points)
        points = []
        for match in matches:
            is_home = match['home_team'] == team_name
            result = match['ftr']
            
            if result == 'D':
                pts = 1
            elif (result == 'H' and is_home) or (result == 'A' and not is_home):
                pts = 3
            else:
                pts = 0
            
            points.append(pts)
        
        # Exponential weighting (recent matches more important)
        weights = np.exp(np.linspace(-1, 0, len(points)))
        weights /= weights.sum()
        momentum_score = np.average(points, weights=weights)
        
        # Calculate trend (are points improving?)
        if len(points) >= 3:
            recent_avg = np.mean(points[:3])
            older_avg = np.mean(points[3:]) if len(points) > 3 else np.mean(points)
            points_trend = recent_avg - older_avg
        else:
            points_trend = 0.0
        
        return {
            'current_streak': streak,
            'momentum_score': round(momentum_score, 2),
            'points_trend': round(points_trend, 2)
        }
    
    def calculate_fatigue(
        self,
        team_name: str,
        match_date: datetime
    ) -> Dict[str, float]:
        """
        Calculate fatigue indicators
        
        Fatigue is based on:
        - Days since last match
        - Number of matches in last 14 days
        - Match congestion
        
        Returns:
            Dictionary with fatigue metrics
        """
        # Get matches in last 30 days
        start_date = match_date - timedelta(days=30)
        
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        query = """
            SELECT match_date
            FROM historical_matches
            WHERE (home_team = %s OR away_team = %s)
                AND match_date >= %s
                AND match_date < %s
            ORDER BY match_date DESC
        """
        cursor.execute(query, [team_name, team_name, start_date, match_date])
        recent_matches = cursor.fetchall()
        cursor.close()
        
        if not recent_matches:
            return {
                'days_since_last_match': 7,
                'matches_last_14_days': 0,
                'fatigue_index': 0.0
            }
        
        # Days since last match
        last_match_date = recent_matches[0]['match_date']
        days_rest = (match_date.date() - last_match_date).days
        
        # Matches in last 14 days
        fourteen_days_ago = match_date - timedelta(days=14)
        matches_14_days = sum(
            1 for m in recent_matches 
            if m['match_date'] >= fourteen_days_ago.date()
        )
        
        # Fatigue index (0 = well rested, 1 = very fatigued)
        # Based on: few rest days + many recent matches = high fatigue
        rest_factor = max(0, 1 - days_rest / 7)  # 0 if 7+ days rest
        congestion_factor = min(1, matches_14_days / 5)  # 1 if 5+ matches
        
        fatigue_index = (rest_factor + congestion_factor) / 2
        
        return {
            'days_since_last_match': days_rest,
            'matches_last_14_days': matches_14_days,
            'fatigue_index': round(fatigue_index, 2)
        }
    
    def calculate_all_advanced_features(
        self,
        home_team: str,
        away_team: str,
        match_date: datetime
    ) -> Dict[str, float]:
        """
        Calculate all advanced features for a match
        
        Args:
            home_team: Home team name
            away_team: Away team name
            match_date: Match date
        
        Returns:
            Dictionary with all advanced features
        """
        features = {}
        
        # Elo ratings
        features['home_elo'] = self.calculate_elo_rating(home_team, match_date)
        features['away_elo'] = self.calculate_elo_rating(away_team, match_date)
        features['elo_difference'] = features['home_elo'] - features['away_elo']
        
        # Expected goals
        features['home_xg'] = self.calculate_expected_goals(home_team, match_date, is_home=True)
        features['away_xg'] = self.calculate_expected_goals(away_team, match_date, is_home=False)
        features['xg_difference'] = features['home_xg'] - features['away_xg']
        
        # Poisson parameters
        poisson_params = self.calculate_poisson_parameters(home_team, away_team, match_date)
        features['poisson_lambda_home'] = poisson_params['lambda_home']
        features['poisson_lambda_away'] = poisson_params['lambda_away']
        features['poisson_prob_home_win'] = poisson_params['prob_home_win']
        features['poisson_prob_draw'] = poisson_params['prob_draw']
        features['poisson_prob_away_win'] = poisson_params['prob_away_win']
        
        # Momentum
        home_momentum = self.calculate_momentum(home_team, match_date)
        features['home_streak'] = home_momentum['current_streak']
        features['home_momentum'] = home_momentum['momentum_score']
        features['home_points_trend'] = home_momentum['points_trend']
        
        away_momentum = self.calculate_momentum(away_team, match_date)
        features['away_streak'] = away_momentum['current_streak']
        features['away_momentum'] = away_momentum['momentum_score']
        features['away_points_trend'] = away_momentum['points_trend']
        
        # Fatigue
        home_fatigue = self.calculate_fatigue(home_team, match_date)
        features['home_days_rest'] = home_fatigue['days_since_last_match']
        features['home_matches_14d'] = home_fatigue['matches_last_14_days']
        features['home_fatigue'] = home_fatigue['fatigue_index']
        
        away_fatigue = self.calculate_fatigue(away_team, match_date)
        features['away_days_rest'] = away_fatigue['days_since_last_match']
        features['away_matches_14d'] = away_fatigue['matches_last_14_days']
        features['away_fatigue'] = away_fatigue['fatigue_index']
        
        logger.info(f"Calculated {len(features)} advanced features for {home_team} vs {away_team}")
        
        return features
