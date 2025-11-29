"""
XGBoost Model

Advanced gradient boosting model for match outcome prediction.
"""

from typing import Dict, Optional
import xgboost as xgb
import joblib

from src.shared.logging.logger import get_logger

logger = get_logger(__name__)


class XGBoostModel:
    """
    XGBoost classifier for match prediction
    
    Target classes:
    - 0: Away win
    - 1: Draw  
    - 2: Home win
    """
    
    def __init__(
        self,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        n_estimators: int = 200,
        min_child_weight: int = 3,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        gamma: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize XGBoost model
        
        Args:
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            n_estimators: Number of boosting rounds
            min_child_weight: Minimum sum of instance weight
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            gamma: Minimum loss reduction for split
            random_state: Random seed
        """
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.random_state = random_state
        
        self.model = xgb.XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            random_state=random_state,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",  # Faster training
            enable_categorical=False,
        )
        
        self.is_fitted = False
        self.feature_names: Optional[list[str]] = None
    
    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        early_stopping_rounds: int = 50,
        verbose: bool = True
    ):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            early_stopping_rounds: Early stopping patience
            verbose: Print training progress
        """
        logger.info("Training XGBoost model...")
        
        fit_kwargs = {"verbose": verbose}

        # Use early stopping only if validation set is provided
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
        
        self.model.fit(X_train, y_train, **fit_kwargs)
        
        self.is_fitted = True
        self.feature_names = list(X_train.columns) if hasattr(X_train, "columns") else None
        
        return self
    
    def predict(self, X):
        """
        Predict class labels
        
        Args:
            X: Features
        
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X: Features
        
        Returns:
            Probability matrix (n_samples, 3)
            Columns: [P(away_win), P(draw), P(home_win)]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X, y) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X: Features
            y: True labels
        
        Returns:
            Dictionary with metrics
        """
        from sklearn.metrics import accuracy_score, log_loss, classification_report
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "log_loss": log_loss(y, y_proba),
        }
        
        # Class-wise metrics
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        
        for class_idx, class_name in [(2, "home"), (1, "draw"), (0, "away")]:
            if str(class_idx) in report:
                metrics[f"precision_{class_name}"] = report[str(class_idx)]["precision"]
                metrics[f"recall_{class_name}"] = report[str(class_idx)]["recall"]
                metrics[f"f1_{class_name}"] = report[str(class_idx)]["f1-score"]
        
        return metrics
    
    def get_feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        """
        Get feature importance
        
        Args:
            importance_type: 'gain', 'weight', or 'cover'
        
        Returns:
            Dictionary mapping features to importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        importance = self.model.get_booster().get_score(importance_type=importance_type)
        
        # Map feature indices to names if available
        if self.feature_names:
            mapped_importance: Dict[str, float] = {}
            for k, v in importance.items():
                # k is like 'f0', 'f1', etc.
                if k.startswith("f") and k[1:].isdigit():
                    idx = int(k[1:])
                    if 0 <= idx < len(self.feature_names):
                        mapped_importance[self.feature_names[idx]] = v
                else:
                    mapped_importance[k] = v
            importance = mapped_importance
        
        # Sort by importance
        sorted_features = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return dict(sorted_features)
    
    def save(self, filepath: str):
        """Save model to disk"""
        self.model.save_model(filepath)
        
        # Save metadata
        metadata_path = filepath.replace(".json", "_metadata.pkl")
        joblib.dump(
            {
                "feature_names": self.feature_names,
                "params": {
                    "max_depth": self.max_depth,
                    "learning_rate": self.learning_rate,
                    "n_estimators": self.n_estimators,
                    "min_child_weight": self.min_child_weight,
                    "subsample": self.subsample,
                    "colsample_bytree": self.colsample_bytree,
                    "gamma": self.gamma,
                    "random_state": self.random_state,
                },
            },
            metadata_path,
        )
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load model from disk"""
        # Load metadata
        metadata_path = filepath.replace(".json", "_metadata.pkl")
        metadata = joblib.load(metadata_path)
        
        # Create instance with same params
        instance = cls(**metadata["params"])
        
        # Load model weights into already configured XGBClassifier
        instance.model.load_model(filepath)
        instance.is_fitted = True
        instance.feature_names = metadata["feature_names"]
        
        logger.info(f"Model loaded from {filepath}")
        return instance
