"""
Baseline Logistic Regression Model

Simple baseline model for match outcome prediction.
Uses multinomial logistic regression with L2 regularization.
"""

from typing import Dict, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

from src.shared.logging.logger import get_logger

logger = get_logger(__name__)


class LogisticRegressionModel:
    """
    Baseline logistic regression model
    
    Target classes:
    - 0: Away win
    - 1: Draw
    - 2: Home win
    """
    
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42
    ):
        """
        Initialize model
        
        Args:
            C: Inverse of regularization strength
            max_iter: Maximum iterations
            random_state: Random seed
        """
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            multi_class='multinomial',
            solver='lbfgs',
            class_weight='balanced'  # Handle class imbalance
        )
        
        self.is_fitted = False
    
    def fit(self, X_train, y_train):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info("Training Logistic Regression model...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_scaled, y_train)
        
        self.is_fitted = True
        
        # Log training accuracy
        train_acc = self.model.score(X_scaled, y_train)
        logger.info(f"Training accuracy: {train_acc:.4f}")
        
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
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
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
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X, y):
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
            'accuracy': accuracy_score(y, y_pred),
            'log_loss': log_loss(y, y_proba),
        }
        
        # Class-wise metrics
        report = classification_report(y, y_pred, output_dict=True)
        metrics['precision_home'] = report['2']['precision']
        metrics['recall_home'] = report['2']['recall']
        metrics['f1_home'] = report['2']['f1-score']
        
        metrics['precision_draw'] = report['1']['precision']
        metrics['recall_draw'] = report['1']['recall']
        metrics['f1_draw'] = report['1']['f1-score']
        
        metrics['precision_away'] = report['0']['precision']
        metrics['recall_away'] = report['0']['recall']
        metrics['f1_away'] = report['0']['f1-score']
        
        return metrics
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance (coefficients)
        
        Args:
            feature_names: List of feature names
        
        Returns:
            Dictionary mapping features to importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Average absolute coefficients across classes
        importance = np.abs(self.model.coef_).mean(axis=0)
        
        feature_importance = dict(zip(feature_names, importance))
        
        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return dict(sorted_features)
    
    def save(self, filepath: str):
        """Save model to disk"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'params': {
                'C': self.C,
                'max_iter': self.max_iter,
                'random_state': self.random_state
            }
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load model from disk"""
        data = joblib.load(filepath)
        
        instance = cls(**data['params'])
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
        return instance
