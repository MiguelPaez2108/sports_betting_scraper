"""
Feature Pipeline Orchestrator

Combines all feature calculators into a single pipeline.
Provides unified interface for calculating complete feature vectors.
"""

from datetime import datetime
from typing import Dict, Optional
import time

from src.feature_engineering.domain.calculators.basic_features import BasicFeaturesCalculator
from src.feature_engineering.domain.calculators.advanced_features import AdvancedFeaturesCalculator
from src.feature_engineering.domain.calculators.h2h_features import H2HFeaturesCalculator
from src.shared.logging.logger import get_logger

logger = get_logger(__name__)


class FeaturePipeline:
    """
    Orchestrates feature calculation from all calculators
    
    Combines:
    - Basic features (22)
    - Advanced features (23)
    - H2H features (11)
    
    Total: 56+ features
    """
    
    def __init__(self, db_config: Dict):
        """
        Initialize feature pipeline
        
        Args:
            db_config: Database connection configuration
        """
        self.db_config = db_config
        self.basic_calculator = None
        self.advanced_calculator = None
        self.h2h_calculator = None
    
    def connect(self):
        """Initialize all calculators"""
        self.basic_calculator = BasicFeaturesCalculator(self.db_config)
        self.basic_calculator.connect()
        
        self.advanced_calculator = AdvancedFeaturesCalculator(self.db_config)
        self.advanced_calculator.connect()
        
        self.h2h_calculator = H2HFeaturesCalculator(self.db_config)
        self.h2h_calculator.connect()
        
        logger.info("Feature pipeline initialized")
    
    def disconnect(self):
        """Close all calculator connections"""
        if self.basic_calculator:
            self.basic_calculator.disconnect()
        
        if self.advanced_calculator:
            self.advanced_calculator.disconnect()
        
        if self.h2h_calculator:
            self.h2h_calculator.disconnect()
        
        logger.info("Feature pipeline disconnected")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
    
    def calculate_features(
        self,
        home_team: str,
        away_team: str,
        match_date: datetime,
        include_basic: bool = True,
        include_advanced: bool = True,
        include_h2h: bool = True
    ) -> Dict[str, float]:
        """
        Calculate complete feature vector for a match
        
        Args:
            home_team: Home team name
            away_team: Away team name
            match_date: Match date
            include_basic: Include basic features
            include_advanced: Include advanced features
            include_h2h: Include H2H features
        
        Returns:
            Dictionary with all features
        """
        start_time = time.time()
        
        features = {}
        
        # Calculate basic features
        if include_basic and self.basic_calculator:
            try:
                basic_features = self.basic_calculator.calculate_all_basic_features(
                    home_team, away_team, match_date
                )
                features.update(basic_features)
                logger.debug(f"Calculated {len(basic_features)} basic features")
            except Exception as e:
                logger.error(f"Error calculating basic features: {e}")
        
        # Calculate advanced features
        if include_advanced and self.advanced_calculator:
            try:
                advanced_features = self.advanced_calculator.calculate_all_advanced_features(
                    home_team, away_team, match_date
                )
                features.update(advanced_features)
                logger.debug(f"Calculated {len(advanced_features)} advanced features")
            except Exception as e:
                logger.error(f"Error calculating advanced features: {e}")
        
        # Calculate H2H features
        if include_h2h and self.h2h_calculator:
            try:
                h2h_features = self.h2h_calculator.calculate_all_h2h_features(
                    home_team, away_team, match_date
                )
                features.update(h2h_features)
                logger.debug(f"Calculated {len(h2h_features)} H2H features")
            except Exception as e:
                logger.error(f"Error calculating H2H features: {e}")
        
        elapsed = time.time() - start_time
        
        logger.info(
            f"Calculated {len(features)} total features for {home_team} vs {away_team} "
            f"in {elapsed*1000:.0f}ms"
        )
        
        return features
    
    def validate_features(self, features: Dict[str, float]) -> Dict[str, any]:
        """
        Validate feature vector quality
        
        Checks for:
        - Missing values (NaN)
        - Outliers
        - Expected ranges
        
        Args:
            features: Feature dictionary
        
        Returns:
            Validation results
        """
        validation = {
            'is_valid': True,
            'missing_count': 0,
            'outlier_count': 0,
            'warnings': []
        }
        
        # Check for missing values
        missing_features = [k for k, v in features.items() if v is None or (isinstance(v, float) and v != v)]
        validation['missing_count'] = len(missing_features)
        
        if missing_features:
            validation['is_valid'] = False
            validation['warnings'].append(f"Missing values in: {', '.join(missing_features[:5])}")
        
        # Check for extreme outliers
        for key, value in features.items():
            if isinstance(value, (int, float)) and value == value:  # Not NaN
                # Check for unreasonable values
                if abs(value) > 1000:  # Arbitrary threshold
                    validation['outlier_count'] += 1
                    validation['warnings'].append(f"Outlier detected: {key}={value}")
        
        return validation
    
    def get_feature_names(self) -> list:
        """
        Get list of all feature names
        
        Returns:
            List of feature names
        """
        # This would ideally be generated from the calculators
        # For now, return a static list
        return [
            # Basic features (22)
            'home_form_last_5', 'home_form_last_10',
            'home_avg_goals_scored', 'home_avg_goals_conceded', 'home_goal_difference',
            'home_win_pct', 'home_draw_pct', 'home_clean_sheet_pct',
            'home_avg_shots', 'home_avg_shots_on_target', 'home_avg_corners',
            'away_form_last_5', 'away_form_last_10',
            'away_avg_goals_scored', 'away_avg_goals_conceded', 'away_goal_difference',
            'away_win_pct', 'away_draw_pct', 'away_clean_sheet_pct',
            'away_avg_shots', 'away_avg_shots_on_target', 'away_avg_corners',
            
            # Advanced features (23)
            'home_elo', 'away_elo', 'elo_difference',
            'home_xg', 'away_xg', 'xg_difference',
            'poisson_lambda_home', 'poisson_lambda_away',
            'poisson_prob_home_win', 'poisson_prob_draw', 'poisson_prob_away_win',
            'home_streak', 'home_momentum', 'home_points_trend',
            'away_streak', 'away_momentum', 'away_points_trend',
            'home_days_rest', 'home_matches_14d', 'home_fatigue',
            'away_days_rest', 'away_matches_14d', 'away_fatigue',
            
            # H2H features (11)
            'h2h_home_wins', 'h2h_draws', 'h2h_away_wins', 'h2h_total_matches',
            'h2h_home_goals_avg', 'h2h_away_goals_avg', 'h2h_total_goals_avg',
            'h2h_over_2_5_pct', 'h2h_under_2_5_pct',
            'h2h_btts_yes_pct', 'h2h_btts_no_pct'
        ]
    
    def get_feature_importance_groups(self) -> Dict[str, list]:
        """
        Group features by category for analysis
        
        Returns:
            Dictionary mapping category to feature names
        """
        return {
            'form': [
                'home_form_last_5', 'home_form_last_10',
                'away_form_last_5', 'away_form_last_10',
                'home_momentum', 'away_momentum'
            ],
            'goals': [
                'home_avg_goals_scored', 'home_avg_goals_conceded',
                'away_avg_goals_scored', 'away_avg_goals_conceded',
                'home_xg', 'away_xg', 'h2h_total_goals_avg'
            ],
            'strength': [
                'home_elo', 'away_elo', 'elo_difference',
                'poisson_lambda_home', 'poisson_lambda_away'
            ],
            'h2h': [
                'h2h_home_wins', 'h2h_draws', 'h2h_away_wins',
                'h2h_over_2_5_pct', 'h2h_btts_yes_pct'
            ],
            'fatigue': [
                'home_days_rest', 'home_matches_14d', 'home_fatigue',
                'away_days_rest', 'away_matches_14d', 'away_fatigue'
            ]
        }
