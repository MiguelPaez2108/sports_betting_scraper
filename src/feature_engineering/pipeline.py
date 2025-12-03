"""
Optimized Feature Pipeline with caching and vectorization.

Manages feature calculation workflow:
- ELO ratings (incremental updates)
- Rolling form statistics
- Head-to-head records
- Poisson probabilities
- Market features (odds-derived, inference only)

Performance target: < 5 minutes for nightly batch
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import joblib
from dataclasses import dataclass

from ..feature_engineering.elo_calculator import EloCalculator
from ..feature_engineering.form_calculator import FormCalculator
from ..feature_engineering.h2h_calculator import H2HCalculator
from ..feature_engineering.poisson_calculator import PoissonCalculator


@dataclass
class PipelineConfig:
    """Configuration for feature pipeline."""
    cache_dir: Path = Path('data/cache')
    elo_k_factor: float = 20.0
    elo_home_advantage: float = 100.0
    form_window: int = 5
    h2h_window: int = 5
    enable_cache: bool = True


class FeaturePipeline:
    """
    Orchestrates feature calculation with caching and optimization.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize calculators
        self.elo_calc = EloCalculator(
            k_factor=self.config.elo_k_factor,
            home_advantage=self.config.elo_home_advantage
        )
        self.form_calc = FormCalculator(window=self.config.form_window)
        self.h2h_calc = H2HCalculator(window=self.config.h2h_window)
        self.poisson_calc = PoissonCalculator()
        
        self._cache = {}
        
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path for cached features."""
        return self.config.cache_dir / f"{cache_key}.parquet"
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load features from cache if available."""
        if not self.config.enable_cache:
            return None
            
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        return None
    
    def _save_to_cache(self, cache_key: str, df: pd.DataFrame) -> None:
        """Save features to cache."""
        if self.config.enable_cache:
            cache_path = self._get_cache_path(cache_key)
            df.to_parquet(cache_path, index=False)
    
    def compute_elo_features(self, matches_df: pd.DataFrame, cache_key: str = 'elo') -> pd.DataFrame:
        """
        Compute ELO ratings for all teams.
        
        Args:
            matches_df: Historical matches sorted by date
            cache_key: Cache identifier
            
        Returns:
            DataFrame with elo_home, elo_away, elo_diff columns
        """
        # Try cache
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            print(f"✓ Loaded ELO features from cache ({len(cached)} rows)")
            return cached
        
        print("Computing ELO features...")
        start_time = datetime.now()
        
        # Ensure sorted by date
        df = matches_df.sort_values('date').copy()
        
        # Compute ELO incrementally
        elo_features = self.elo_calc.compute_batch(df)
        
        # Save to cache
        self._save_to_cache(cache_key, elo_features)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"✓ ELO features computed in {elapsed:.2f}s")
        
        return elo_features
    
    def compute_form_features(self, matches_df: pd.DataFrame, cache_key: str = 'form') -> pd.DataFrame:
        """
        Compute rolling form statistics.
        
        Returns:
            DataFrame with pts_last5_home, gf_last5_home, etc.
        """
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            print(f"✓ Loaded form features from cache ({len(cached)} rows)")
            return cached
        
        print("Computing form features...")
        start_time = datetime.now()
        
        df = matches_df.sort_values('date').copy()
        form_features = self.form_calc.compute_batch(df)
        
        self._save_to_cache(cache_key, form_features)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"✓ Form features computed in {elapsed:.2f}s")
        
        return form_features
    
    def compute_h2h_features(self, matches_df: pd.DataFrame, cache_key: str = 'h2h') -> pd.DataFrame:
        """
        Compute head-to-head statistics.
        
        Returns:
            DataFrame with h2h_wins_home, h2h_draws, etc.
        """
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            print(f"✓ Loaded H2H features from cache ({len(cached)} rows)")
            return cached
        
        print("Computing H2H features...")
        start_time = datetime.now()
        
        df = matches_df.sort_values('date').copy()
        h2h_features = self.h2h_calc.compute_batch(df)
        
        self._save_to_cache(cache_key, h2h_features)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"✓ H2H features computed in {elapsed:.2f}s")
        
        return h2h_features
    
    def compute_poisson_features(self, matches_df: pd.DataFrame, cache_key: str = 'poisson') -> pd.DataFrame:
        """
        Compute Poisson-based probability features.
        
        Returns:
            DataFrame with poisson_prob_home, poisson_prob_draw, poisson_prob_away
        """
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            print(f"✓ Loaded Poisson features from cache ({len(cached)} rows)")
            return cached
        
        print("Computing Poisson features...")
        start_time = datetime.now()
        
        df = matches_df.sort_values('date').copy()
        poisson_features = self.poisson_calc.compute_batch(df)
        
        self._save_to_cache(cache_key, poisson_features)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"✓ Poisson features computed in {elapsed:.2f}s")
        
        return poisson_features
    
    def compute_odds_features(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute odds-derived features (INFERENCE ONLY - not for training).
        
        Features:
        - Implied probabilities (normalized)
        - Book margin
        - Odds ratios
        
        WARNING: These features should NEVER be used in training to avoid leakage.
        """
        df = matches_df.copy()
        
        # Calculate implied probabilities
        df['imp_home'] = 1 / df['odds_home']
        df['imp_draw'] = 1 / df['odds_draw']
        df['imp_away'] = 1 / df['odds_away']
        
        # Normalize
        total_imp = df['imp_home'] + df['imp_draw'] + df['imp_away']
        df['imp_home_norm'] = df['imp_home'] / total_imp
        df['imp_draw_norm'] = df['imp_draw'] / total_imp
        df['imp_away_norm'] = df['imp_away'] / total_imp
        
        # Book margin (overround)
        df['book_margin'] = total_imp - 1.0
        
        # Odds ratios (home vs away)
        df['odds_ratio_ha'] = df['odds_home'] / df['odds_away']
        
        return df[['match_id', 'imp_home_norm', 'imp_draw_norm', 'imp_away_norm', 
                   'book_margin', 'odds_ratio_ha']]
    
    def transform(
        self,
        matches_df: pd.DataFrame,
        include_odds_features: bool = False,
        cache_prefix: str = 'default'
    ) -> pd.DataFrame:
        """
        Transform matches into feature matrix.
        
        Args:
            matches_df: Raw match data
            include_odds_features: Whether to include odds features (only for inference)
            cache_prefix: Prefix for cache keys
            
        Returns:
            Complete feature matrix
        """
        print("=" * 60)
        print("FEATURE PIPELINE TRANSFORM")
        print("=" * 60)
        
        overall_start = datetime.now()
        
        # Ensure match_id exists
        if 'match_id' not in matches_df.columns:
            matches_df['match_id'] = range(len(matches_df))
        
        # Compute all feature groups
        elo_features = self.compute_elo_features(matches_df, f'{cache_prefix}_elo')
        form_features = self.compute_form_features(matches_df, f'{cache_prefix}_form')
        h2h_features = self.compute_h2h_features(matches_df, f'{cache_prefix}_h2h')
        poisson_features = self.compute_poisson_features(matches_df, f'{cache_prefix}_poisson')
        
        # Merge all features
        print("\nMerging feature groups...")
        features = matches_df[['match_id']].copy()
        
        for feat_df in [elo_features, form_features, h2h_features, poisson_features]:
            features = features.merge(feat_df, on='match_id', how='left')
        
        # Add odds features if requested (inference only)
        if include_odds_features:
            if all(col in matches_df.columns for col in ['odds_home', 'odds_draw', 'odds_away']):
                print("Adding odds features (INFERENCE MODE)...")
                odds_features = self.compute_odds_features(matches_df)
                features = features.merge(odds_features, on='match_id', how='left')
            else:
                print("⚠ Odds columns not found, skipping odds features")
        
        # Fill NaN values (for early matches without history)
        features = features.fillna(0)
        
        total_elapsed = (datetime.now() - overall_start).total_seconds()
        
        print("\n" + "=" * 60)
        print(f"PIPELINE COMPLETE")
        print(f"  Total time: {total_elapsed:.2f}s")
        print(f"  Features: {len(features.columns) - 1}")  # Exclude match_id
        print(f"  Samples: {len(features)}")
        print("=" * 60)
        
        return features
    
    def clear_cache(self) -> None:
        """Clear all cached features."""
        import shutil
        if self.config.cache_dir.exists():
            shutil.rmtree(self.config.cache_dir)
            self.config.cache_dir.mkdir(parents=True)
        print("✓ Cache cleared")
