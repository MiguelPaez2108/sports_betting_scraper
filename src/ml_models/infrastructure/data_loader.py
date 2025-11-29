"""
Data Loader for ML Training

Loads historical matches and calculates features for model training.
Handles batch processing, caching, and train/test splitting.
"""

from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import pickle
from pathlib import Path

from src.feature_engineering.application.feature_pipeline import FeaturePipeline
from src.shared.logging.logger import get_logger

logger = get_logger(__name__)


class MLDataLoader:
    """
    Load and prepare data for ML model training
    
    Features:
    - Batch feature calculation
    - Progress tracking
    - Feature caching
    - Temporal train/test split
    """
    
    def __init__(self, db_config: Dict):
        """
        Initialize data loader
        
        Args:
            db_config: Database connection configuration
        """
        self.db_config = db_config
        self.conn = None
        self.feature_pipeline = None
        self.cache_dir = Path("data/ml_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def connect(self):
        """Establish database connection"""
        self.conn = psycopg2.connect(**self.db_config)
        self.feature_pipeline = FeaturePipeline(self.db_config)
        self.feature_pipeline.connect()
        logger.info("Data loader connected to database")
    
    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
        if self.feature_pipeline:
            self.feature_pipeline.disconnect()
        logger.info("Data loader disconnected")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
    
    def load_historical_matches(
        self,
        start_date: str = '2015-01-01',
        end_date: str = '2024-12-31',
        leagues: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load historical matches from database
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            leagues: List of league codes (e.g., ['E0', 'SP1'])
            limit: Maximum number of matches (for testing)
        
        Returns:
            DataFrame with match data
        """
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        
        # Build query
        where_clauses = [
            "match_date >= %s",
            "match_date <= %s"
        ]
        params = [start_date, end_date]
        
        if leagues:
            placeholders = ','.join(['%s'] * len(leagues))
            where_clauses.append(f"league_code IN ({placeholders})")
            params.extend(leagues)
        
        query = f"""
            SELECT 
                id, match_date, league_code,
                home_team, away_team,
                fthg, ftag, ftr
            FROM historical_matches
            WHERE {' AND '.join(where_clauses)}
            ORDER BY match_date ASC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query, params)
        matches = cursor.fetchall()
        cursor.close()
        
        df = pd.DataFrame(matches)
        logger.info(f"Loaded {len(df)} historical matches")
        
        return df
    
    def calculate_features_batch(
        self,
        matches_df: pd.DataFrame,
        batch_size: int = 100,
        use_cache: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Calculate features for all matches in batches
        
        Args:
            matches_df: DataFrame with match data
            batch_size: Number of matches per batch
            use_cache: Whether to use cached features
        
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        cache_file = self.cache_dir / "features_cache.pkl"
        
        # Check cache
        if use_cache and cache_file.exists():
            logger.info("Loading features from cache...")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            logger.info(f"Loaded {len(cached_data['X'])} cached features")
            return cached_data['X'], cached_data['y']
        
        # Calculate features
        logger.info(f"Calculating features for {len(matches_df)} matches...")
        
        all_features = []
        all_targets = []
        
        total_batches = (len(matches_df) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(matches_df))
            batch = matches_df.iloc[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch)} matches)")
            
            for idx, match in batch.iterrows():
                try:
                    # Calculate features
                    features = self.feature_pipeline.calculate_features(
                        home_team=match['home_team'],
                        away_team=match['away_team'],
                        match_date=pd.to_datetime(match['match_date'])
                    )
                    
                    # Add match metadata
                    features['match_id'] = match['id']
                    features['match_date'] = match['match_date']
                    features['league_code'] = match['league_code']
                    
                    all_features.append(features)
                    
                    # Target: match result
                    # H = 2 (home win), D = 1 (draw), A = 0 (away win)
                    target_map = {'H': 2, 'D': 1, 'A': 0}
                    all_targets.append(target_map.get(match['ftr'], 1))
                    
                except Exception as e:
                    logger.error(f"Error calculating features for match {match['id']}: {e}")
                    continue
            
            # Progress update
            progress = (batch_idx + 1) / total_batches * 100
            logger.info(f"Progress: {progress:.1f}% ({len(all_features)} features calculated)")
        
        # Convert to DataFrame
        X = pd.DataFrame(all_features)
        y = pd.Series(all_targets, name='target')
        
        logger.info(f"Feature calculation complete: {len(X)} samples, {len(X.columns)} features")
        
        # Cache results
        if use_cache:
            logger.info("Caching features...")
            with open(cache_file, 'wb') as f:
                pickle.dump({'X': X, 'y': y}, f)
            logger.info(f"Features cached to {cache_file}")
        
        return X, y
    
    def split_data_temporal(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.15,
        val_size: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data temporally (no shuffling to prevent data leakage)
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion for test set
            val_size: Proportion for validation set
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Sort by date
        X_sorted = X.sort_values('match_date').reset_index(drop=True)
        y_sorted = y.loc[X_sorted.index].reset_index(drop=True)
        
        n_samples = len(X_sorted)
        
        # Calculate split indices
        test_idx = int(n_samples * (1 - test_size))
        val_idx = int(test_idx * (1 - val_size / (1 - test_size)))
        
        # Split
        X_train = X_sorted.iloc[:val_idx]
        X_val = X_sorted.iloc[val_idx:test_idx]
        X_test = X_sorted.iloc[test_idx:]
        
        y_train = y_sorted.iloc[:val_idx]
        y_val = y_sorted.iloc[val_idx:test_idx]
        y_test = y_sorted.iloc[test_idx:]
        
        logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def prepare_training_data(
        self,
        start_date: str = '2015-01-01',
        end_date: str = '2024-12-31',
        leagues: Optional[List[str]] = None,
        test_size: float = 0.15,
        val_size: float = 0.15,
        batch_size: int = 100,
        use_cache: bool = True,
        limit: Optional[int] = None
    ) -> Dict:
        """
        Complete pipeline: load, calculate features, split
        
        Returns:
            Dictionary with train/val/test splits
        """
        # Load matches
        matches_df = self.load_historical_matches(
            start_date, end_date, leagues, limit
        )
        
        # Calculate features
        X, y = self.calculate_features_batch(
            matches_df, batch_size, use_cache
        )
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data_temporal(
            X, y, test_size, val_size
        )
        
        # Remove metadata columns for training
        metadata_cols = ['match_id', 'match_date', 'league_code']
        X_train_clean = X_train.drop(columns=metadata_cols, errors='ignore')
        X_val_clean = X_val.drop(columns=metadata_cols, errors='ignore')
        X_test_clean = X_test.drop(columns=metadata_cols, errors='ignore')
        
        return {
            'X_train': X_train_clean,
            'X_val': X_val_clean,
            'X_test': X_test_clean,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': list(X_train_clean.columns)
        }
