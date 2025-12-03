"""
Drift Detection - Monitor data distribution changes.

Uses KL divergence and Wasserstein distance to detect drift.
"""
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.special import kl_div
from typing import Dict, List
import json
from pathlib import Path


class DriftDetector:
    """
    Detect data drift between reference and current distributions.
    """
    
    def __init__(self, threshold: float = 0.1):
        """
        Args:
            threshold: Drift threshold (0.0 to 1.0)
        """
        self.threshold = threshold
    
    def detect_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        features: List[str]
    ) -> Dict:
        """
        Detect drift in specified features.
        
        Args:
            reference_df: Reference (training) data
            current_df: Current (production) data
            features: List of feature names to check
            
        Returns:
            Dictionary with drift metrics per feature
        """
        drift_results = {}
        alerts = []
        
        for feature in features:
            if feature not in reference_df.columns or feature not in current_df.columns:
                continue
            
            ref_values = reference_df[feature].dropna().values
            cur_values = current_df[feature].dropna().values
            
            if len(ref_values) == 0 or len(cur_values) == 0:
                continue
            
            # Calculate Wasserstein distance
            wasserstein = wasserstein_distance(ref_values, cur_values)
            
            # Calculate KL divergence (for binned distributions)
            kl = self._calculate_kl_divergence(ref_values, cur_values)
            
            # Detect drift
            drift_detected = (wasserstein > self.threshold) or (kl > self.threshold)
            
            drift_results[feature] = {
                'wasserstein_distance': float(wasserstein),
                'kl_divergence': float(kl),
                'drift_detected': drift_detected,
                'ref_mean': float(np.mean(ref_values)),
                'cur_mean': float(np.mean(cur_values)),
                'ref_std': float(np.std(ref_values)),
                'cur_std': float(np.std(cur_values))
            }
            
            if drift_detected:
                alerts.append(f"⚠️  DRIFT DETECTED in {feature}: "
                            f"Wasserstein={wasserstein:.4f}, KL={kl:.4f}")
        
        return {
            'drift_results': drift_results,
            'alerts': alerts,
            'total_features_checked': len(features),
            'features_with_drift': sum(1 for r in drift_results.values() if r['drift_detected'])
        }
    
    def _calculate_kl_divergence(
        self,
        ref_values: np.ndarray,
        cur_values: np.ndarray,
        n_bins: int = 20
    ) -> float:
        """Calculate KL divergence between two distributions."""
        # Create bins based on reference data
        bins = np.histogram_bin_edges(ref_values, bins=n_bins)
        
        # Get histograms
        ref_hist, _ = np.histogram(ref_values, bins=bins, density=True)
        cur_hist, _ = np.histogram(cur_values, bins=bins, density=True)
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        ref_hist = ref_hist + epsilon
        cur_hist = cur_hist + epsilon
        
        # Normalize
        ref_hist = ref_hist / ref_hist.sum()
        cur_hist = cur_hist / cur_hist.sum()
        
        # Calculate KL divergence
        kl = np.sum(kl_div(cur_hist, ref_hist))
        
        return float(kl)
    
    def save_report(self, results: Dict, output_path: Path) -> None:
        """Save drift detection report."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Drift report saved to {output_path}")
    
    def print_summary(self, results: Dict) -> None:
        """Print drift detection summary."""
        print("\n" + "=" * 60)
        print("DRIFT DETECTION SUMMARY")
        print("=" * 60)
        
        print(f"\nFeatures checked: {results['total_features_checked']}")
        print(f"Features with drift: {results['features_with_drift']}")
        
        if results['alerts']:
            print("\n🚨 ALERTS:")
            for alert in results['alerts']:
                print(f"  {alert}")
        else:
            print("\n✅ No drift detected")
        
        print("\n" + "=" * 60)
