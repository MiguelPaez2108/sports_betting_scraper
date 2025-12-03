"""
Two-Stage Model for improved Draw/Away predictions.

Stage 1: Home vs NotHome (binary)
Stage 2: Draw vs Away (for NotHome predictions)
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier
import joblib
from pathlib import Path


class TwoStageModel(BaseEstimator, ClassifierMixin):
    """
    Two-stage prediction model.
    
    Stage 1: Predict if Home wins or not
    Stage 2: If not Home, predict Draw vs Away
    """
    
    def __init__(
        self,
        stage1_params: dict = None,
        stage2_params: dict = None
    ):
        """
        Args:
            stage1_params: Parameters for stage 1 model
            stage2_params: Parameters for stage 2 model
        """
        self.stage1_params = stage1_params or {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        }
        
        self.stage2_params = stage2_params or {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        }
        
        self.stage1_model = None
        self.stage2_model = None
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'TwoStageModel':
        """
        Fit two-stage model.
        
        Args:
            X: Features
            y: Labels (0=Away, 1=Draw, 2=Home)
        """
        print("Training Two-Stage Model...")
        
        # Stage 1: Home vs NotHome
        print("  Stage 1: Home vs NotHome...")
        y_stage1 = (y == 2).astype(int)  # 1 if Home, 0 if NotHome
        
        self.stage1_model = XGBClassifier(**self.stage1_params)
        self.stage1_model.fit(X, y_stage1)
        
        # Stage 2: Draw vs Away (only for NotHome samples)
        print("  Stage 2: Draw vs Away...")
        not_home_mask = (y != 2)
        X_stage2 = X[not_home_mask]
        y_stage2 = y[not_home_mask]  # 0=Away, 1=Draw
        
        self.stage2_model = XGBClassifier(**self.stage2_params)
        self.stage2_model.fit(X_stage2, y_stage2)
        
        print("✓ Two-stage model trained")
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Returns:
            Array of shape (n_samples, 3) with probabilities [Away, Draw, Home]
        """
        n_samples = len(X)
        probs = np.zeros((n_samples, 3))
        
        # Stage 1: Probability of Home
        prob_home = self.stage1_model.predict_proba(X)[:, 1]
        probs[:, 2] = prob_home
        
        # Stage 2: For NotHome, split between Draw and Away
        prob_not_home = 1 - prob_home
        prob_draw_given_not_home = self.stage2_model.predict_proba(X)[:, 1]
        
        probs[:, 1] = prob_not_home * prob_draw_given_not_home
        probs[:, 0] = prob_not_home * (1 - prob_draw_given_not_home)
        
        return probs
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'stage1_model': self.stage1_model,
            'stage2_model': self.stage2_model,
            'stage1_params': self.stage1_params,
            'stage2_params': self.stage2_params
        }
        
        joblib.dump(state, path)
        print(f"✓ Two-stage model saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'TwoStageModel':
        """Load model from disk."""
        state = joblib.load(path)
        
        model = cls(
            stage1_params=state['stage1_params'],
            stage2_params=state['stage2_params']
        )
        
        model.stage1_model = state['stage1_model']
        model.stage2_model = state['stage2_model']
        
        return model
