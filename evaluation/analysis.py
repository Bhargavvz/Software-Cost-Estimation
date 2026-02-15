"""
Data Quality Impact Analysis & Ablation Studies
Measures performance deltas across cleaning stages and synthetic data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
import json
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR
from evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class AblationStudy:
    """
    Run ablation experiments to quantify the impact of each
    data quality stage and synthetic data augmentation.
    """

    def __init__(self):
        self.results: List[Dict] = []

    def add_result(self, experiment_name: str,
                   y_true: np.ndarray,
                   y_pred: np.ndarray,
                   description: str = ""):
        """Record results from one ablation experiment."""
        metrics = compute_all_metrics(y_true, y_pred)
        entry = {
            "experiment": experiment_name,
            "description": description,
            "n_samples": len(y_true),
            **metrics,
        }
        self.results.append(entry)
        logger.info(f"Ablation [{experiment_name}]: "
                    f"MAE={metrics['MAE']:.4f}, R²={metrics['R2']:.4f}")

    def get_summary(self) -> pd.DataFrame:
        """Return all ablation results as a DataFrame."""
        return pd.DataFrame(self.results)

    def save(self, path: Optional[str] = None):
        """Save ablation results to JSON."""
        path = path or os.path.join(OUTPUT_DIR, "ablation_results.json")
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Ablation results saved: {path}")


class SensitivityAnalyzer:
    """
    Analyze model sensitivity to input noise at various levels.
    """

    def __init__(self):
        self.results: List[Dict] = []

    def analyze(self, model_predict_fn,
                X_test: np.ndarray,
                y_test: np.ndarray,
                noise_levels: List[float] = None):
        """
        Test model robustness by adding increasing noise to inputs.

        Args:
            model_predict_fn: callable(X) → predictions
            X_test: clean test features
            y_test: ground truth
            noise_levels: list of noise scale multipliers
        """
        noise_levels = noise_levels or [0.0, 0.05, 0.10, 0.20, 0.50]

        for noise_scale in noise_levels:
            if noise_scale > 0:
                noise = np.random.randn(*X_test.shape) * noise_scale
                X_noisy = X_test + noise * np.std(X_test, axis=0, keepdims=True)
            else:
                X_noisy = X_test

            preds = model_predict_fn(X_noisy)
            metrics = compute_all_metrics(y_test, preds)

            entry = {
                "noise_level": noise_scale,
                **metrics,
            }
            self.results.append(entry)
            logger.info(f"Noise {noise_scale:.0%}: "
                        f"MAE={metrics['MAE']:.4f}, R²={metrics['R2']:.4f}")

    def get_summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

    def save(self, path: Optional[str] = None):
        path = path or os.path.join(OUTPUT_DIR, "sensitivity_results.json")
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
