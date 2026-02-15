"""
Evaluation Metrics: MAE, RMSE, MAPE, R²
Per-sprint / per-project aggregation and comparisons.
"""

import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray,
         epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (%)."""
    denom = np.abs(y_true) + epsilon
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R²)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-8:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_all_metrics(y_true: np.ndarray,
                        y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all standard regression metrics."""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def print_metrics(metrics: Dict[str, float], label: str = ""):
    """Pretty-print metrics."""
    header = f"  [{label}]" if label else ""
    logger.info(f"Metrics{header}:")
    for k, v in metrics.items():
        logger.info(f"  {k:>8s}: {v:.4f}")


def compare_data_quality_impact(
        metrics_before: Dict[str, float],
        metrics_after: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """
    Compare model performance before and after data quality processing.
    Returns improvement for each metric.
    """
    comparison = {}
    for metric in metrics_before:
        before = metrics_before[metric]
        after = metrics_after[metric]
        # For MAE/RMSE/MAPE: lower is better → positive improvement
        # For R²: higher is better → positive improvement
        if metric == "R2":
            improvement = after - before
        else:
            improvement = before - after
        comparison[metric] = {
            "before_cleaning": before,
            "after_cleaning": after,
            "improvement": improvement,
            "improvement_pct": (improvement / max(abs(before), 1e-8)) * 100,
        }
    return comparison
