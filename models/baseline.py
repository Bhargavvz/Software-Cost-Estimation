"""
Baseline ML Models: XGBoost and Random Forest wrappers.
Flatten sprint sequences into aggregate features for traditional ML.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
import logging

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CANONICAL_FEATURES, TARGET_EFFORT

logger = logging.getLogger(__name__)


def _flatten_project_features(df: pd.DataFrame,
                              feature_cols: List[str],
                              target_col: str = TARGET_EFFORT) \
        -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Aggregate sprint-level features into project-level summaries.
    For each feature, compute: mean, std, min, max, last, trend (slope).
    """
    flat_features = []
    flat_targets = []
    flat_names = []

    first = True
    for pid, grp in df.groupby("project_id"):
        grp = grp.sort_values("sprint_id")
        row = []

        for col in feature_cols:
            if col not in grp.columns:
                vals = np.zeros(len(grp))
            else:
                vals = grp[col].values.astype(float)

            row.extend([
                np.nanmean(vals),
                np.nanstd(vals),
                np.nanmin(vals),
                np.nanmax(vals),
                vals[-1],                         # last sprint
                np.polyfit(np.arange(len(vals)), vals, 1)[0]
                if len(vals) > 1 else 0.0,        # trend slope
            ])

            if first:
                for suffix in ["_mean", "_std", "_min", "_max",
                               "_last", "_trend"]:
                    flat_names.append(col + suffix)

        # Add sequence length
        row.append(len(grp))
        if first:
            flat_names.append("n_sprints")

        first = False
        flat_features.append(row)
        flat_targets.append(grp[target_col].values[-1])

    X = np.array(flat_features, dtype=np.float64)
    y = np.array(flat_targets, dtype=np.float64)

    # Replace NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, flat_names


class XGBoostEstimator:
    """XGBoost regression wrapper for effort estimation."""

    def __init__(self, **kwargs):
        try:
            from xgboost import XGBRegressor
            self.model = XGBRegressor(
                n_estimators=kwargs.get("n_estimators", 500),
                max_depth=kwargs.get("max_depth", 8),
                learning_rate=kwargs.get("learning_rate", 0.05),
                subsample=kwargs.get("subsample", 0.8),
                colsample_bytree=kwargs.get("colsample_bytree", 0.8),
                reg_alpha=kwargs.get("reg_alpha", 0.1),
                reg_lambda=kwargs.get("reg_lambda", 1.0),
                random_state=42,
                n_jobs=-1,
                tree_method="hist",
            )
        except ImportError:
            logger.warning("XGBoost not installed; using RandomForest fallback")
            self.model = RandomForestRegressor(
                n_estimators=500, max_depth=8, random_state=42, n_jobs=-1)

        self.feature_names: List[str] = []

    def fit(self, df: pd.DataFrame,
            feature_cols: List[str],
            target_col: str = TARGET_EFFORT):
        X, y, self.feature_names = _flatten_project_features(
            df, feature_cols, target_col)
        self.model.fit(X, y)
        logger.info(f"XGBoost fitted on {X.shape[0]} projects, "
                    f"{X.shape[1]} features")

    def predict(self, df: pd.DataFrame,
                feature_cols: List[str],
                target_col: str = TARGET_EFFORT) -> np.ndarray:
        X, _, _ = _flatten_project_features(df, feature_cols, target_col)
        return self.model.predict(X)

    def feature_importance(self) -> Dict[str, float]:
        imp = self.model.feature_importances_
        return dict(zip(self.feature_names, imp))


class RandomForestEstimator:
    """Random Forest regression wrapper for effort estimation."""

    def __init__(self, **kwargs):
        self.model = RandomForestRegressor(
            n_estimators=kwargs.get("n_estimators", 500),
            max_depth=kwargs.get("max_depth", 12),
            min_samples_leaf=kwargs.get("min_samples_leaf", 5),
            random_state=42,
            n_jobs=-1,
        )
        self.feature_names: List[str] = []

    def fit(self, df: pd.DataFrame,
            feature_cols: List[str],
            target_col: str = TARGET_EFFORT):
        X, y, self.feature_names = _flatten_project_features(
            df, feature_cols, target_col)
        self.model.fit(X, y)
        logger.info(f"RandomForest fitted on {X.shape[0]} projects, "
                    f"{X.shape[1]} features")

    def predict(self, df: pd.DataFrame,
                feature_cols: List[str],
                target_col: str = TARGET_EFFORT) -> np.ndarray:
        X, _, _ = _flatten_project_features(df, feature_cols, target_col)
        return self.model.predict(X)

    def feature_importance(self) -> Dict[str, float]:
        imp = self.model.feature_importances_
        return dict(zip(self.feature_names, imp))
