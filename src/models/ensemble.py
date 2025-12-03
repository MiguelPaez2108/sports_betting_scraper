"""
Ensemble Models with Stacking.

Combines multiple base models using a meta-learner:
- XGBoost
- LightGBM  
- CatBoost
- Meta-learner: Logistic Regression on out-of-fold predictions

This reduces overfitting and improves generalization.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import joblib
from pathlib import Path

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Stacking ensemble with out-of-fold predictions.
    
    Process:
    1. Train base models on K-fold CV, generating OOF predictions
    2. Train meta-learner on OOF predictions
    3. Retrain base models on full dataset for final predictions
    """
    
    def __init__(
        self,
        base_models: Optional[Dict[str, BaseEstimator]] = None,
        meta_learner: Optional[BaseEstimator] = None,
        n_folds: int = 5,
        random_state: int = 42
    ):
        """
        Args:
            base_models: Dictionary of {name: model} for base learners
            meta_learner: Model for combining base predictions
            n_folds: Number of CV folds for OOF predictions
            random_state: Random seed
        """
        self.base_models = base_models or self._default_base_models()
        self.meta_learner = meta_learner or LogisticRegression(
            C=1.0,
            solver='lbfgs',
            max_iter=1000,
            random_state=random_state
        )
        self.n_folds = n_folds
        self.random_state = random_state
        
        self.fitted_base_models_: Dict[str, BaseEstimator] = {}
        self.meta_learner_: Optional[BaseEstimator] = None
        
    def _default_base_models(self) -> Dict[str, BaseEstimator]:
        """Create default base models."""
        return {
            'xgb': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'lgbm': LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            ),
            'catboost': CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                verbose=False
            )
        }
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'StackingEnsemble':
        """
        Fit ensemble using stacking.
        
        Args:
            X: Training features
            y: Training labels (0=Away, 1=Draw, 2=Home)
        """
        print("=" * 60)
        print("STACKING ENSEMBLE TRAINING")
        print("=" * 60)
        
        X = X.reset_index(drop=True) if isinstance(X, pd.DataFrame) else X
        
        # Step 1: Generate OOF predictions
        print(f"\n1. Generating out-of-fold predictions ({self.n_folds} folds)...")
        oof_predictions = self._generate_oof_predictions(X, y)
        
        # Step 2: Train meta-learner on OOF predictions
        print("\n2. Training meta-learner...")
        self.meta_learner_ = clone(self.meta_learner)
        self.meta_learner_.fit(oof_predictions, y)
        print(f"✓ Meta-learner trained")
        
        # Step 3: Retrain base models on full dataset
        print("\n3. Retraining base models on full dataset...")
        for name, model in self.base_models.items():
            print(f"  Training {name}...")
            fitted_model = clone(model)
            fitted_model.fit(X, y)
            self.fitted_base_models_[name] = fitted_model
        
        print("\n" + "=" * 60)
        print("ENSEMBLE TRAINING COMPLETE")
        print("=" * 60)
        
        return self
    
    def _generate_oof_predictions(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """Generate out-of-fold predictions for meta-learner training."""
        n_samples = len(X)
        n_classes = len(np.unique(y))
        n_models = len(self.base_models)
        
        # Initialize OOF prediction matrix: [n_samples, n_models * n_classes]
        oof_preds = np.zeros((n_samples, n_models * n_classes))
        
        # K-Fold CV
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"  Fold {fold}/{self.n_folds}...")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y[train_idx]
            
            for i, (name, model) in enumerate(self.base_models.items()):
                # Clone and fit
                clf = clone(model)
                clf.fit(X_train, y_train)
                
                # Predict on validation fold
                probs = clf.predict_proba(X_val)
                
                # Store in OOF matrix
                start_col = i * n_classes
                oof_preds[val_idx, start_col:start_col + n_classes] = probs
        
        return oof_preds
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features
            
        Returns:
            Array of shape (n_samples, n_classes) with probabilities
        """
        if not self.fitted_base_models_:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        # Get predictions from all base models
        base_preds = []
        for name, model in self.fitted_base_models_.items():
            base_preds.append(model.predict_proba(X))
        
        # Concatenate: [n_samples, n_models * n_classes]
        meta_features = np.hstack(base_preds)
        
        # Predict with meta-learner
        return self.meta_learner_.predict_proba(meta_features)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
    def save(self, path: Path) -> None:
        """Save ensemble to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'fitted_base_models': self.fitted_base_models_,
            'meta_learner': self.meta_learner_,
            'n_folds': self.n_folds,
            'random_state': self.random_state
        }
        
        joblib.dump(state, path)
        print(f"✓ Ensemble saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'StackingEnsemble':
        """Load ensemble from disk."""
        state = joblib.load(path)
        
        ensemble = cls(
            base_models={},  # Will be populated from state
            meta_learner=state['meta_learner'],
            n_folds=state['n_folds'],
            random_state=state['random_state']
        )
        
        ensemble.fitted_base_models_ = state['fitted_base_models']
        ensemble.meta_learner_ = state['meta_learner']
        
        return ensemble


def train_ensemble(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    output_dir: Path
) -> StackingEnsemble:
    """
    Train and evaluate stacking ensemble.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        output_dir: Directory to save model
        
    Returns:
        Trained ensemble
    """
    from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
    
    # Train ensemble
    ensemble = StackingEnsemble(n_folds=5, random_state=42)
    ensemble.fit(X_train, y_train)
    
    # Evaluate on validation
    print("\n" + "=" * 60)
    print("VALIDATION METRICS")
    print("=" * 60)
    
    y_pred_proba = ensemble.predict_proba(X_val)
    y_pred = ensemble.predict(X_val)
    
    accuracy = accuracy_score(y_val, y_pred)
    logloss = log_loss(y_val, y_pred_proba)
    brier = brier_score_loss(y_val, y_pred_proba, pos_label=list(range(3)))
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    print(f"Brier Score: {brier:.4f}")
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    ensemble.save(output_dir / 'ensemble.pkl')
    
    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'log_loss': logloss,
        'brier_score': brier
    }
    joblib.dump(metrics, output_dir / 'metrics.pkl')
    
    return ensemble
