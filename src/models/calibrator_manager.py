"""
CalibratorManager: Manages global and league-specific probability calibration.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Union
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator
import joblib


class CalibratorManager:
    """
    Manages probability calibration for betting models.
    
    Features:
    - Global calibrator for all matches
    - League-specific calibrators (min 300 samples)
    - Lazy loading from disk
    - Meta-calibration (weighted ensemble)
    """
    
    def __init__(self, base_model: BaseEstimator, calibrator_dir: Union[str, Path]):
        """
        Initialize CalibratorManager.
        
        Args:
            base_model: Pre-trained classifier (e.g., XGBoost)
            calibrator_dir: Directory to save/load calibrators
        """
        self.base_model = base_model
        self.calibrator_dir = Path(calibrator_dir)
        self.calibrator_dir.mkdir(parents=True, exist_ok=True)
        
        self.global_calibrator: Optional[CalibratedClassifierCV] = None
        self.league_calibrators: Dict[str, CalibratedClassifierCV] = {}
        self._loaded_leagues = set()
        
    def fit_global(self, X_val: pd.DataFrame, y_val: np.ndarray) -> None:
        """
        Fit global calibrator on validation data.
        
        Args:
            X_val: Validation features
            y_val: Validation labels (0=Away, 1=Draw, 2=Home)
        """
        print(f"Fitting global calibrator on {len(X_val)} samples...")
        self.global_calibrator = CalibratedClassifierCV(
            self.base_model,
            method='isotonic',
            cv='prefit'
        )
        self.global_calibrator.fit(X_val, y_val)
        
        # Save to disk
        save_path = self.calibrator_dir / 'global.pkl'
        joblib.dump(self.global_calibrator, save_path)
        print(f"✓ Global calibrator saved to {save_path}")
        
    def fit_by_league(
        self, 
        league_code: str, 
        X_val: pd.DataFrame, 
        y_val: np.ndarray,
        min_samples: int = 300
    ) -> bool:
        """
        Fit league-specific calibrator if sufficient data.
        
        Args:
            league_code: League identifier (e.g., 'E0', 'CL')
            X_val: Validation features for this league
            y_val: Validation labels
            min_samples: Minimum samples required
            
        Returns:
            True if calibrator was fitted, False if insufficient data
        """
        if len(X_val) < min_samples:
            print(f"⚠ League {league_code}: {len(X_val)} samples < {min_samples}, skipping")
            return False
            
        print(f"Fitting calibrator for {league_code} ({len(X_val)} samples)...")
        calibrator = CalibratedClassifierCV(
            self.base_model,
            method='isotonic',
            cv='prefit'
        )
        calibrator.fit(X_val, y_val)
        
        self.league_calibrators[league_code] = calibrator
        self._loaded_leagues.add(league_code)
        
        # Save to disk
        save_path = self.calibrator_dir / f'{league_code}.pkl'
        joblib.dump(calibrator, save_path)
        print(f"✓ {league_code} calibrator saved to {save_path}")
        return True
        
    def _try_load_league_calibrator(self, league_code: str) -> bool:
        """Lazy load league calibrator from disk if exists."""
        if league_code in self._loaded_leagues:
            return True
            
        calibrator_path = self.calibrator_dir / f'{league_code}.pkl'
        if calibrator_path.exists():
            self.league_calibrators[league_code] = joblib.load(calibrator_path)
            self._loaded_leagues.add(league_code)
            return True
        return False
        
    def predict_proba(
        self, 
        X: pd.DataFrame, 
        league_code: Optional[str] = None
    ) -> np.ndarray:
        """
        Predict calibrated probabilities.
        
        Args:
            X: Features
            league_code: Optional league code for league-specific calibration
            
        Returns:
            Array of shape (n_samples, 3) with probabilities [Away, Draw, Home]
        """
        if self.global_calibrator is None:
            raise ValueError("Global calibrator not fitted. Call fit_global() first.")
            
        # Try league-specific calibrator
        if league_code and self._try_load_league_calibrator(league_code):
            return self.league_calibrators[league_code].predict_proba(X)
            
        # Fallback to global
        return self.global_calibrator.predict_proba(X)
        
    def predict_proba_batch(
        self, 
        X: pd.DataFrame, 
        league_codes: np.ndarray
    ) -> np.ndarray:
        """
        Optimized batch prediction for mixed leagues.
        
        Args:
            X: Features
            league_codes: Array of league codes corresponding to X
            
        Returns:
            Array of shape (n_samples, 3) with probabilities
        """
        if len(X) != len(league_codes):
            raise ValueError("X and league_codes must have same length")
            
        if self.global_calibrator is None:
            raise ValueError("Global calibrator not fitted")
            
        # Initialize result array
        n_samples = len(X)
        n_classes = 3
        final_probs = np.zeros((n_samples, n_classes))
        
        # Process each unique league
        unique_leagues = np.unique(league_codes)
        
        for league in unique_leagues:
            mask = (league_codes == league)
            
            if self._try_load_league_calibrator(league):
                # Use league-specific calibrator
                final_probs[mask] = self.league_calibrators[league].predict_proba(X[mask])
            else:
                # Fallback to global
                final_probs[mask] = self.global_calibrator.predict_proba(X[mask])
                
        return final_probs
        
    def predict_proba_ensemble(
        self,
        X: pd.DataFrame,
        league_code: str,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Meta-calibration: weighted ensemble of global and league calibrators.
        
        Formula: p_final = α * p_global + (1-α) * p_league
        
        Args:
            X: Features
            league_code: League identifier
            alpha: Weight for global model (0.0 to 1.0)
            
        Returns:
            Ensemble probabilities
        """
        if not (0 <= alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1")
            
        if self.global_calibrator is None:
            raise ValueError("Global calibrator not fitted")
            
        p_global = self.global_calibrator.predict_proba(X)
        
        if self._try_load_league_calibrator(league_code):
            p_league = self.league_calibrators[league_code].predict_proba(X)
            return alpha * p_global + (1 - alpha) * p_league
        else:
            # No league calibrator, return global
            return p_global
            
    def save(self, path: Union[str, Path]) -> None:
        """Save manager state (excluding calibrators, which are saved individually)."""
        state = {
            'calibrator_dir': str(self.calibrator_dir),
            'loaded_leagues': list(self._loaded_leagues)
        }
        joblib.dump(state, path)
        
    @classmethod
    def load(cls, path: Union[str, Path], base_model: BaseEstimator) -> 'CalibratorManager':
        """Load manager state."""
        state = joblib.load(path)
        manager = cls(base_model, state['calibrator_dir'])
        
        # Load global calibrator
        global_path = manager.calibrator_dir / 'global.pkl'
        if global_path.exists():
            manager.global_calibrator = joblib.load(global_path)
            
        return manager
