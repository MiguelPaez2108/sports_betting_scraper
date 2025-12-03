"""
Unit tests for CalibratorManager
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from sklearn.ensemble import RandomForestClassifier
from src.models.calibrator_manager import CalibratorManager


@pytest.fixture
def temp_dir():
    """Create temporary directory for calibrators."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def base_model():
    """Create and fit a simple base model."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 3, 100)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def validation_data():
    """Generate validation data."""
    np.random.seed(123)
    X_val = pd.DataFrame(np.random.randn(50, 5))
    y_val = np.random.randint(0, 3, 50)
    return X_val, y_val


class TestCalibratorManager:
    
    def test_init(self, base_model, temp_dir):
        """Test initialization."""
        manager = CalibratorManager(base_model, temp_dir)
        assert manager.calibrator_dir == temp_dir
        assert manager.global_calibrator is None
        assert len(manager.league_calibrators) == 0
        
    def test_fit_global(self, base_model, temp_dir, validation_data):
        """Test global calibrator fitting."""
        X_val, y_val = validation_data
        manager = CalibratorManager(base_model, temp_dir)
        
        manager.fit_global(X_val, y_val)
        
        assert manager.global_calibrator is not None
        assert (temp_dir / 'global.pkl').exists()
        
    def test_predict_proba_global(self, base_model, temp_dir, validation_data):
        """Test prediction with global calibrator."""
        X_val, y_val = validation_data
        manager = CalibratorManager(base_model, temp_dir)
        manager.fit_global(X_val, y_val)
        
        X_test = pd.DataFrame(np.random.randn(10, 5))
        probs = manager.predict_proba(X_test)
        
        assert probs.shape == (10, 3)
        assert np.allclose(probs.sum(axis=1), 1.0)
        
    def test_fit_by_league_sufficient_data(self, base_model, temp_dir, validation_data):
        """Test league calibrator with sufficient data."""
        X_val, y_val = validation_data
        manager = CalibratorManager(base_model, temp_dir)
        manager.fit_global(X_val, y_val)
        
        # Fit league calibrator (50 samples > 30 min)
        result = manager.fit_by_league('E0', X_val, y_val, min_samples=30)
        
        assert result is True
        assert 'E0' in manager.league_calibrators
        assert (temp_dir / 'E0.pkl').exists()
        
    def test_fit_by_league_insufficient_data(self, base_model, temp_dir):
        """Test league calibrator with insufficient data."""
        manager = CalibratorManager(base_model, temp_dir)
        
        # Only 20 samples, min_samples=300
        X_small = pd.DataFrame(np.random.randn(20, 5))
        y_small = np.random.randint(0, 3, 20)
        
        result = manager.fit_by_league('SP1', X_small, y_small, min_samples=300)
        
        assert result is False
        assert 'SP1' not in manager.league_calibrators
        
    def test_predict_proba_with_league(self, base_model, temp_dir, validation_data):
        """Test prediction with league-specific calibrator."""
        X_val, y_val = validation_data
        manager = CalibratorManager(base_model, temp_dir)
        manager.fit_global(X_val, y_val)
        manager.fit_by_league('E0', X_val, y_val, min_samples=30)
        
        X_test = pd.DataFrame(np.random.randn(10, 5))
        probs = manager.predict_proba(X_test, league_code='E0')
        
        assert probs.shape == (10, 3)
        assert np.allclose(probs.sum(axis=1), 1.0)
        
    def test_fallback_to_global(self, base_model, temp_dir, validation_data):
        """Test fallback to global when league calibrator doesn't exist."""
        X_val, y_val = validation_data
        manager = CalibratorManager(base_model, temp_dir)
        manager.fit_global(X_val, y_val)
        
        X_test = pd.DataFrame(np.random.randn(10, 5))
        probs = manager.predict_proba(X_test, league_code='NONEXISTENT')
        
        # Should use global calibrator
        assert probs.shape == (10, 3)
        
    def test_lazy_loading(self, base_model, temp_dir, validation_data):
        """Test lazy loading of league calibrators."""
        X_val, y_val = validation_data
        
        # First manager: fit and save
        manager1 = CalibratorManager(base_model, temp_dir)
        manager1.fit_global(X_val, y_val)
        manager1.fit_by_league('E0', X_val, y_val, min_samples=30)
        
        # Second manager: should lazy load
        manager2 = CalibratorManager(base_model, temp_dir)
        manager2.fit_global(X_val, y_val)
        
        assert 'E0' not in manager2.league_calibrators  # Not loaded yet
        
        X_test = pd.DataFrame(np.random.randn(10, 5))
        manager2.predict_proba(X_test, league_code='E0')
        
        assert 'E0' in manager2.league_calibrators  # Now loaded
        
    def test_batch_prediction(self, base_model, temp_dir, validation_data):
        """Test batch prediction with mixed leagues."""
        X_val, y_val = validation_data
        manager = CalibratorManager(base_model, temp_dir)
        manager.fit_global(X_val, y_val)
        manager.fit_by_league('E0', X_val, y_val, min_samples=30)
        
        # Mixed batch
        X_test = pd.DataFrame(np.random.randn(20, 5))
        leagues = np.array(['E0'] * 10 + ['SP1'] * 10)
        
        probs = manager.predict_proba_batch(X_test, leagues)
        
        assert probs.shape == (20, 3)
        assert np.allclose(probs.sum(axis=1), 1.0)
        
    def test_ensemble_prediction(self, base_model, temp_dir, validation_data):
        """Test meta-calibration ensemble."""
        X_val, y_val = validation_data
        manager = CalibratorManager(base_model, temp_dir)
        manager.fit_global(X_val, y_val)
        manager.fit_by_league('E0', X_val, y_val, min_samples=30)
        
        X_test = pd.DataFrame(np.random.randn(10, 5))
        
        # Test different alpha values
        probs_global = manager.predict_proba_ensemble(X_test, 'E0', alpha=1.0)
        probs_league = manager.predict_proba_ensemble(X_test, 'E0', alpha=0.0)
        probs_mixed = manager.predict_proba_ensemble(X_test, 'E0', alpha=0.5)
        
        assert probs_global.shape == (10, 3)
        assert probs_league.shape == (10, 3)
        assert probs_mixed.shape == (10, 3)
        
        # Mixed should be between global and league
        assert not np.allclose(probs_mixed, probs_global)
        assert not np.allclose(probs_mixed, probs_league)
        
    def test_save_load(self, base_model, temp_dir, validation_data):
        """Test save and load functionality."""
        X_val, y_val = validation_data
        manager1 = CalibratorManager(base_model, temp_dir)
        manager1.fit_global(X_val, y_val)
        
        save_path = temp_dir / 'manager_state.pkl'
        manager1.save(save_path)
        
        manager2 = CalibratorManager.load(save_path, base_model)
        
        assert manager2.global_calibrator is not None
        assert manager2.calibrator_dir == temp_dir
