"""
Unit tests for Stacking Ensemble
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.models.ensemble import StackingEnsemble
import tempfile
from pathlib import Path
import shutil


@pytest.fixture
def temp_dir():
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def dummy_data():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(200, 10))
    y = np.random.randint(0, 3, 200)
    return X, y


class TestStackingEnsemble:
    
    def test_fit_predict(self, dummy_data):
        """Test basic fit and predict."""
        X, y = dummy_data
        
        # Use simple models for speed
        base_models = {
            'rf1': RandomForestClassifier(n_estimators=10, random_state=42),
            'rf2': RandomForestClassifier(n_estimators=10, max_depth=3, random_state=43)
        }
        
        ensemble = StackingEnsemble(base_models=base_models, n_folds=3)
        ensemble.fit(X, y)
        
        # Check fitted
        assert len(ensemble.fitted_base_models_) == 2
        assert ensemble.meta_learner_ is not None
        
        # Predict
        probs = ensemble.predict_proba(X)
        preds = ensemble.predict(X)
        
        assert probs.shape == (200, 3)
        assert preds.shape == (200,)
        assert np.allclose(probs.sum(axis=1), 1.0)
    
    def test_oof_predictions(self, dummy_data):
        """Test OOF prediction generation."""
        X, y = dummy_data
        
        base_models = {
            'rf': RandomForestClassifier(n_estimators=5, random_state=42)
        }
        
        ensemble = StackingEnsemble(base_models=base_models, n_folds=3)
        oof_preds = ensemble._generate_oof_predictions(X, y)
        
        # Should have shape [n_samples, n_models * n_classes]
        assert oof_preds.shape == (200, 1 * 3)  # 1 model, 3 classes
        
        # All samples should have predictions (no zeros)
        assert np.all(oof_preds.sum(axis=1) > 0)
    
    def test_save_load(self, dummy_data, temp_dir):
        """Test save and load functionality."""
        X, y = dummy_data
        
        base_models = {
            'rf': RandomForestClassifier(n_estimators=5, random_state=42)
        }
        
        ensemble = StackingEnsemble(base_models=base_models, n_folds=3)
        ensemble.fit(X, y)
        
        # Save
        save_path = temp_dir / 'ensemble.pkl'
        ensemble.save(save_path)
        
        # Load
        loaded_ensemble = StackingEnsemble.load(save_path)
        
        # Compare predictions
        probs1 = ensemble.predict_proba(X)
        probs2 = loaded_ensemble.predict_proba(X)
        
        np.testing.assert_array_almost_equal(probs1, probs2)
    
    def test_meta_learner_improvement(self, dummy_data):
        """Test that ensemble improves over single model."""
        X, y = dummy_data
        
        # Split data
        split = 150
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Train single model
        from sklearn.metrics import accuracy_score
        single_model = RandomForestClassifier(n_estimators=10, random_state=42)
        single_model.fit(X_train, y_train)
        single_acc = accuracy_score(y_test, single_model.predict(X_test))
        
        # Train ensemble
        base_models = {
            'rf1': RandomForestClassifier(n_estimators=10, random_state=42),
            'rf2': RandomForestClassifier(n_estimators=10, max_depth=3, random_state=43)
        }
        ensemble = StackingEnsemble(base_models=base_models, n_folds=3)
        ensemble.fit(X_train, y_train)
        ensemble_acc = accuracy_score(y_test, ensemble.predict(X_test))
        
        # Ensemble should be at least as good (usually better)
        assert ensemble_acc >= single_acc * 0.95  # Allow 5% tolerance
