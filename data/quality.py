"""
Data Quality Intelligence Pipeline
8-stage data-centric processing: imputation → outlier detection → agile
consistency → temporal smoothing → feature validation → label QA →
versioning & audit.
"""

import numpy as np
import pandas as pd
import hashlib
import json
import os
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_regression

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (CANONICAL_FEATURES, TARGET_EFFORT, TARGET_COST,
                    dq_cfg, DATA_QUALITY_DIR, VERSIONED_DATA_DIR)

logger = logging.getLogger(__name__)

NUMERIC_FEATURES = CANONICAL_FEATURES + [TARGET_EFFORT, TARGET_COST]


# ──────────────────────────────────────────────────────────────────────
# Quality Report
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QualityReport:
    """Tracks every quality action for auditability."""
    stage: str = ""
    records_before: int = 0
    records_after: int = 0
    actions: List[str] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════
#  STAGE 1: Missing Data Intelligence
# ══════════════════════════════════════════════════════════════════════
class MissingDataHandler:
    """Context-aware imputation with reliability tracking."""

    def __init__(self, cfg=None):
        self.cfg = cfg or dq_cfg
        self.missingness_report: Dict[str, float] = {}
        self.feature_reliability: Dict[str, float] = {}

    def analyze_missingness(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute per-feature missing ratio."""
        ratios = {}
        for col in NUMERIC_FEATURES:
            if col in df.columns:
                ratios[col] = df[col].isna().mean()
        self.missingness_report = ratios
        return ratios

    def impute(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, QualityReport]:
        report = QualityReport(stage="missing_data", records_before=len(df))
        df = df.copy()

        self.analyze_missingness(df)
        dropped_features = []

        for col in NUMERIC_FEATURES:
            if col not in df.columns:
                continue

            miss_ratio = self.missingness_report.get(col, 0)

            # Drop feature entirely if too much missing
            if miss_ratio > self.cfg.max_missing_ratio:
                dropped_features.append(col)
                report.actions.append(
                    f"Dropped '{col}' (missing {miss_ratio:.1%})")
                continue

            if miss_ratio == 0:
                self.feature_reliability[col] = 1.0
                continue

            # Strategy 1: time-series interpolation within each project
            if col in ["velocity", "effort_hours", "completed_story_points"]:
                df[col] = df.groupby("project_id")[col].transform(
                    lambda s: s.interpolate(
                        method=self.cfg.interpolation_method).bfill().ffill()
                )
                report.actions.append(
                    f"Time-series interpolation on '{col}'")
            # Strategy 2: team-wise median
            elif col in ["team_size", "team_experience_months"]:
                df[col] = df.groupby("project_id")[col].transform(
                    lambda s: s.fillna(s.median())
                )
                df[col].fillna(df[col].median(), inplace=True)
                report.actions.append(
                    f"Team-wise median imputation on '{col}'")
            # Strategy 3: global median fallback
            else:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                report.actions.append(
                    f"Global median imputation on '{col}' (val={median_val:.2f})")

            self.feature_reliability[col] = 1.0 - miss_ratio

        # Drop heavily-missing features
        df.drop(columns=[c for c in dropped_features if c in df.columns],
                inplace=True, errors="ignore")

        report.records_after = len(df)
        report.stats = {
            "missingness": self.missingness_report,
            "reliability": self.feature_reliability,
            "dropped_features": dropped_features,
        }
        return df, report


# ══════════════════════════════════════════════════════════════════════
#  STAGE 2: Noise & Outlier Detection
# ══════════════════════════════════════════════════════════════════════
class OutlierDetector:
    """IQR + Isolation Forest + temporal anomaly detection."""

    def __init__(self, cfg=None):
        self.cfg = cfg or dq_cfg

    def _iqr_flags(self, df: pd.DataFrame) -> pd.Series:
        """Flag rows with any feature outside IQR bounds."""
        outlier_flag = pd.Series(False, index=df.index)
        for col in NUMERIC_FEATURES:
            if col not in df.columns:
                continue
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lo = q1 - self.cfg.iqr_multiplier * iqr
            hi = q3 + self.cfg.iqr_multiplier * iqr
            outlier_flag |= (df[col] < lo) | (df[col] > hi)
        return outlier_flag

    def _isolation_forest_flags(self, df: pd.DataFrame) -> pd.Series:
        """Isolation Forest anomaly detection on numeric features."""
        feat_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
        if not feat_cols:
            return pd.Series(False, index=df.index)

        X = df[feat_cols].fillna(0).values
        iso = IsolationForest(
            contamination=self.cfg.isolation_forest_contamination,
            random_state=42, n_jobs=-1)
        preds = iso.fit_predict(X)
        return pd.Series(preds == -1, index=df.index)

    def _temporal_anomaly_flags(self, df: pd.DataFrame) -> pd.Series:
        """Detect spikes in effort/velocity across adjacent sprints."""
        flag = pd.Series(False, index=df.index)
        for target in [TARGET_EFFORT, "velocity"]:
            if target not in df.columns:
                continue
            window = self.cfg.temporal_anomaly_window
            rolling_mean = df.groupby("project_id")[target].transform(
                lambda s: s.rolling(window, min_periods=1, center=True).mean()
            )
            rolling_std = df.groupby("project_id")[target].transform(
                lambda s: s.rolling(window, min_periods=1, center=True).std()
            ).fillna(1)
            z = (df[target] - rolling_mean).abs() / rolling_std.clip(lower=1e-6)
            flag |= (z > 3.0)
        return flag

    def detect_and_handle(self, df: pd.DataFrame) \
            -> Tuple[pd.DataFrame, QualityReport]:
        report = QualityReport(stage="outlier_detection",
                               records_before=len(df))
        df = df.copy()

        iqr_flags = self._iqr_flags(df)
        iso_flags = self._isolation_forest_flags(df)
        temp_flags = self._temporal_anomaly_flags(df)

        combined = iqr_flags | iso_flags | temp_flags
        n_outliers = combined.sum()

        # Down-weight outliers instead of removing them
        if "quality_score" not in df.columns:
            df["quality_score"] = 1.0
        df.loc[combined, "quality_score"] *= 0.3

        # Winsorize extreme values for numeric features
        for col in NUMERIC_FEATURES:
            if col in df.columns:
                lo = df[col].quantile(0.01)
                hi = df[col].quantile(0.99)
                df[col] = df[col].clip(lo, hi)

        report.records_after = len(df)
        report.actions = [
            f"IQR outliers: {iqr_flags.sum()}",
            f"Isolation Forest outliers: {iso_flags.sum()}",
            f"Temporal anomalies: {temp_flags.sum()}",
            f"Total flagged (down-weighted): {n_outliers}",
            "Applied winsorization at 1st/99th percentile",
        ]
        report.stats = {"n_outliers": int(n_outliers),
                        "outlier_pct": float(n_outliers / max(len(df), 1))}
        return df, report


# ══════════════════════════════════════════════════════════════════════
#  STAGE 3: Agile Consistency Rules
# ══════════════════════════════════════════════════════════════════════
class AgileConsistencyEngine:
    """Enforce domain constraints on Agile sprint data."""

    def __init__(self, cfg=None):
        self.cfg = cfg or dq_cfg

    def enforce(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, QualityReport]:
        report = QualityReport(stage="agile_consistency",
                               records_before=len(df))
        df = df.copy()
        violations = 0

        # Rule 1: Velocity cannot jump more than max_velocity_jump_ratio
        if "velocity" in df.columns:
            for pid, grp in df.groupby("project_id"):
                vel = grp["velocity"].values
                for i in range(1, len(vel)):
                    if vel[i-1] > 0:
                        ratio = vel[i] / vel[i-1]
                        if ratio > self.cfg.max_velocity_jump_ratio:
                            vel[i] = vel[i-1] * self.cfg.max_velocity_jump_ratio
                            violations += 1
                        elif ratio < 1.0 / self.cfg.max_velocity_jump_ratio:
                            vel[i] = vel[i-1] / self.cfg.max_velocity_jump_ratio
                            violations += 1
                df.loc[grp.index, "velocity"] = vel

        # Rule 2: Effort per story point within bounds
        if TARGET_EFFORT in df.columns and "completed_story_points" in df.columns:
            eff_per_pt = (df[TARGET_EFFORT] /
                          df["completed_story_points"].clip(lower=1))
            too_low = eff_per_pt < self.cfg.min_effort_per_point
            too_high = eff_per_pt > self.cfg.max_effort_per_point
            corrections = too_low | too_high
            violations += corrections.sum()

            median_eff = eff_per_pt[~corrections].median()
            if pd.notna(median_eff):
                df.loc[corrections, TARGET_EFFORT] = (
                    df.loc[corrections, "completed_story_points"] * median_eff)

        # Rule 3: Completed ≤ Planned story points
        if ("completed_story_points" in df.columns and
                "planned_story_points" in df.columns):
            exceed = df["completed_story_points"] > df["planned_story_points"] * 1.1
            violations += exceed.sum()
            df.loc[exceed, "completed_story_points"] = \
                df.loc[exceed, "planned_story_points"]

        report.records_after = len(df)
        report.actions.append(f"Total consistency violations fixed: {violations}")
        report.stats = {"violations_fixed": int(violations)}
        return df, report


# ══════════════════════════════════════════════════════════════════════
#  STAGE 4: Temporal Smoothing & Denoising
# ══════════════════════════════════════════════════════════════════════
class TemporalSmoother:
    """Sprint-level denoising: moving average + Kalman filter."""

    def __init__(self, cfg=None):
        self.cfg = cfg or dq_cfg

    def _moving_average(self, series: pd.Series) -> pd.Series:
        return series.rolling(
            window=self.cfg.moving_avg_window,
            min_periods=1, center=True).mean()

    def _kalman_smooth(self, values: np.ndarray) -> np.ndarray:
        """1D Kalman smoother for sprint time-series."""
        try:
            from filterpy.kalman import KalmanFilter
            kf = KalmanFilter(dim_x=2, dim_z=1)
            kf.x = np.array([[values[0]], [0.0]])
            kf.F = np.array([[1, 1], [0, 1]])
            kf.H = np.array([[1, 0]])
            kf.P *= 10
            kf.R = np.array([[self.cfg.kalman_measurement_noise]])
            kf.Q = np.eye(2) * self.cfg.kalman_process_noise

            smoothed = np.zeros(len(values))
            for i, z in enumerate(values):
                kf.predict()
                kf.update(np.array([[z]]))
                smoothed[i] = kf.x[0, 0]
            return smoothed
        except ImportError:
            logger.warning("filterpy not available; falling back to MA")
            return pd.Series(values).rolling(
                self.cfg.moving_avg_window, min_periods=1, center=True
            ).mean().values

    def smooth(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, QualityReport]:
        report = QualityReport(stage="temporal_smoothing",
                               records_before=len(df))
        df = df.copy()
        smoothed_cols = []

        targets = ["velocity", TARGET_EFFORT, "completed_story_points"]
        for col in targets:
            if col not in df.columns:
                continue

            # Apply Kalman smoothing per project
            df[col] = df.groupby("project_id")[col].transform(
                lambda s: self._kalman_smooth(s.values)
            )
            smoothed_cols.append(col)

        report.records_after = len(df)
        report.actions.append(
            f"Kalman-smoothed columns: {smoothed_cols}")
        return df, report


# ══════════════════════════════════════════════════════════════════════
#  STAGE 5: Feature Validation & Selection
# ══════════════════════════════════════════════════════════════════════
class FeatureValidator:
    """Correlation analysis, mutual information, stability tests."""

    def __init__(self, cfg=None):
        self.cfg = cfg or dq_cfg

    def validate_and_select(self, df: pd.DataFrame) \
            -> Tuple[pd.DataFrame, QualityReport, List[str]]:
        report = QualityReport(stage="feature_validation",
                               records_before=len(df))
        df = df.copy()
        available = [c for c in CANONICAL_FEATURES if c in df.columns]

        removed = []

        # 1. Remove high-correlation pairs
        if len(available) > 1:
            corr = df[available].corr().abs()
            upper = corr.where(
                np.triu(np.ones(corr.shape), k=1).astype(bool))
            to_drop_corr = [col for col in upper.columns
                            if any(upper[col] > self.cfg.max_correlation)]
            removed.extend(to_drop_corr)
            report.actions.append(
                f"Removed high-correlation features: {to_drop_corr}")

        # 2. Mutual information with target
        remaining = [c for c in available if c not in removed]
        if remaining and TARGET_EFFORT in df.columns:
            X = df[remaining].fillna(0)
            y = df[TARGET_EFFORT].fillna(0)
            mi = mutual_info_regression(X, y, random_state=42)
            mi_scores = dict(zip(remaining, mi))

            low_mi = [c for c, s in mi_scores.items()
                      if s < self.cfg.min_mutual_info]
            removed.extend(low_mi)
            report.actions.append(
                f"Low mutual-info features: {low_mi}")
            report.stats["mutual_info_scores"] = {
                k: round(v, 4) for k, v in mi_scores.items()}

        # 3. Feature stability (coefficient of variation across projects)
        remaining = [c for c in available if c not in removed]
        for col in remaining:
            project_means = df.groupby("project_id")[col].mean()
            cv = project_means.std() / max(project_means.mean(), 1e-6)
            if cv > 10:  # extremely unstable
                removed.append(col)
                report.actions.append(
                    f"Unstable feature '{col}' (CV={cv:.2f})")

        # Unique removals
        removed = list(set(removed))
        selected = [c for c in available if c not in removed]

        report.records_after = len(df)
        report.stats["selected_features"] = selected
        report.stats["removed_features"] = removed
        return df, report, selected


# ══════════════════════════════════════════════════════════════════════
#  STAGE 6: Label Quality Assurance
# ══════════════════════════════════════════════════════════════════════
class LabelQualityAssurer:
    """Check, smooth, and score effort/cost labels."""

    def __init__(self, cfg=None):
        self.cfg = cfg or dq_cfg

    def assure(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, QualityReport]:
        report = QualityReport(stage="label_quality",
                               records_before=len(df))
        df = df.copy()
        fixes = 0

        for target in [TARGET_EFFORT, TARGET_COST]:
            if target not in df.columns:
                continue

            # Cross-sprint consistency: smooth outlier labels
            project_means = df.groupby("project_id")[target].transform("mean")
            project_stds = df.groupby("project_id")[target].transform("std")
            project_stds = project_stds.clip(lower=1)

            z_scores = (df[target] - project_means).abs() / project_stds
            anomalous = z_scores > 3

            # Replace anomalous labels with project rolling mean
            if anomalous.any():
                rolling = df.groupby("project_id")[target].transform(
                    lambda s: s.rolling(3, min_periods=1, center=True).mean())
                df.loc[anomalous, target] = rolling.loc[anomalous]
                fixes += anomalous.sum()

        # Confidence scores based on quality
        if "quality_score" not in df.columns:
            df["quality_score"] = 1.0

        # Lower confidence for records that were heavily corrected
        df["label_confidence"] = df["quality_score"].copy()

        # Sample weight = label_confidence (used during training)
        df["sample_weight"] = df["label_confidence"].clip(0.1, 1.0)

        report.records_after = len(df)
        report.actions.append(f"Label anomalies smoothed: {fixes}")
        report.stats = {"labels_smoothed": int(fixes)}
        return df, report


# ══════════════════════════════════════════════════════════════════════
#  STAGE 7: Data Versioning & Auditability
# ══════════════════════════════════════════════════════════════════════
class DataVersionManager:
    """SHA-based versioning with full audit trail."""

    def __init__(self, base_dir: str = VERSIONED_DATA_DIR):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _compute_hash(self, df: pd.DataFrame) -> str:
        content = pd.util.hash_pandas_object(df).values.tobytes()
        return hashlib.sha256(content).hexdigest()[:12]

    def save_version(self, df: pd.DataFrame, stage: str,
                     report: QualityReport) -> str:
        """Save a versioned snapshot with metadata."""
        version_hash = self._compute_hash(df)
        version_dir = os.path.join(
            self.base_dir, f"{stage}_{version_hash}")
        os.makedirs(version_dir, exist_ok=True)

        # Save data
        data_path = os.path.join(version_dir, "data.csv")
        df.to_csv(data_path, index=False)

        # Save report
        report_path = os.path.join(version_dir, "quality_report.json")
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

        # Save metadata
        meta = {
            "stage": stage,
            "hash": version_hash,
            "timestamp": datetime.now().isoformat(),
            "n_records": len(df),
            "n_features": len(df.columns),
            "columns": list(df.columns),
        }
        meta_path = os.path.join(version_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved version: {stage}_{version_hash} "
                    f"({len(df)} records)")
        return version_hash


# ══════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR: Full Data Quality Pipeline
# ══════════════════════════════════════════════════════════════════════
class DataQualityPipeline:
    """
    End-to-end data quality pipeline executing all 7 stages
    in sequence, with versioning after each stage.
    """

    def __init__(self, cfg=None, versioning: bool = True):
        self.cfg = cfg or dq_cfg
        self.versioning = versioning
        self.version_mgr = DataVersionManager() if versioning else None

        self.missing_handler = MissingDataHandler(self.cfg)
        self.outlier_detector = OutlierDetector(self.cfg)
        self.consistency_engine = AgileConsistencyEngine(self.cfg)
        self.smoother = TemporalSmoother(self.cfg)
        self.feature_validator = FeatureValidator(self.cfg)
        self.label_assurer = LabelQualityAssurer(self.cfg)

        self.reports: List[QualityReport] = []
        self.selected_features: List[str] = []

    def run(self, df: pd.DataFrame) \
            -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Execute the full data-quality pipeline.

        Returns:
            (cleaned_df, list_of_stage_reports)
        """
        logger.info(f"▶ Data Quality Pipeline: {len(df)} records input")

        # Stage 1: Missing data
        df, r = self.missing_handler.impute(df)
        self.reports.append(r)
        if self.version_mgr:
            self.version_mgr.save_version(df, "01_imputed", r)
        logger.info(f"  ✓ Stage 1 (Imputation): {len(r.actions)} actions")

        # Stage 2: Outlier detection
        df, r = self.outlier_detector.detect_and_handle(df)
        self.reports.append(r)
        if self.version_mgr:
            self.version_mgr.save_version(df, "02_outliers", r)
        logger.info(f"  ✓ Stage 2 (Outliers): {r.stats.get('n_outliers', 0)} flagged")

        # Stage 3: Agile consistency
        df, r = self.consistency_engine.enforce(df)
        self.reports.append(r)
        if self.version_mgr:
            self.version_mgr.save_version(df, "03_consistency", r)
        logger.info(f"  ✓ Stage 3 (Consistency): "
                    f"{r.stats.get('violations_fixed', 0)} fixes")

        # Stage 4: Temporal smoothing
        df, r = self.smoother.smooth(df)
        self.reports.append(r)
        if self.version_mgr:
            self.version_mgr.save_version(df, "04_smoothed", r)
        logger.info(f"  ✓ Stage 4 (Smoothing): done")

        # Stage 5: Feature validation
        df, r, selected = self.feature_validator.validate_and_select(df)
        self.selected_features = selected
        self.reports.append(r)
        if self.version_mgr:
            self.version_mgr.save_version(df, "05_features", r)
        logger.info(f"  ✓ Stage 5 (Features): {len(selected)} selected")

        # Stage 6: Label quality
        df, r = self.label_assurer.assure(df)
        self.reports.append(r)
        if self.version_mgr:
            self.version_mgr.save_version(df, "06_labels", r)
        logger.info(f"  ✓ Stage 6 (Labels): {r.stats.get('labels_smoothed', 0)} fixed")

        # Save full quality report
        if self.versioning:
            full_report = [rp.to_dict() for rp in self.reports]
            report_path = os.path.join(DATA_QUALITY_DIR,
                                       "full_quality_report.json")
            with open(report_path, "w") as f:
                json.dump(full_report, f, indent=2, default=str)
            logger.info(f"  ✓ Quality report saved: {report_path}")

        logger.info(f"▶ Pipeline complete: {len(df)} clean records")
        return df, [rp.to_dict() for rp in self.reports]
