"""
Canonical Agile Schema – Schema Harmonization Module
Maps heterogeneous datasets (ISBSG, PROMISE, NASA, raw Agile CSVs)
into a single unified representation with consistent naming, units, and scales.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import logging

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CANONICAL_FEATURES, TARGET_EFFORT, TARGET_COST

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Canonical Sprint Record
# ──────────────────────────────────────────────────────────────────────
@dataclass
class SprintRecord:
    """Single canonical sprint observation."""
    project_id: str
    sprint_id: int
    team_size: float
    sprint_duration_days: float
    planned_story_points: float
    completed_story_points: float
    velocity: float
    num_stories: int
    avg_story_complexity: float
    defect_count: int
    scope_change_ratio: float
    team_experience_months: float
    tech_debt_hours: float
    meeting_overhead_hours: float
    effort_hours: float              # primary target
    cost_usd: float                  # secondary target
    quality_score: float = 1.0       # confidence in this record
    data_source: str = "unknown"


# ──────────────────────────────────────────────────────────────────────
# Column Mapping Adapters
# ──────────────────────────────────────────────────────────────────────

# Maps from source column name → canonical column name
ISBSG_COLUMN_MAP: Dict[str, str] = {
    "Project ID":                   "project_id",
    "Team Size":                    "team_size",
    "Project Elapsed Time":         "sprint_duration_days",
    "Functional Size":              "planned_story_points",
    "Adjusted Function Points":     "completed_story_points",
    "Speed of Delivery":            "velocity",
    "Defect Count":                 "defect_count",
    "Summary Work Effort":          "effort_hours",
    "Normalised Work Effort":       "cost_usd",
}

PROMISE_COLUMN_MAP: Dict[str, str] = {
    "ProjectID":         "project_id",
    "TeamSize":          "team_size",
    "Duration":          "sprint_duration_days",
    "StoryPoints":       "planned_story_points",
    "CompletedPoints":   "completed_story_points",
    "Velocity":          "velocity",
    "Defects":           "defect_count",
    "Effort":            "effort_hours",
    "Cost":              "cost_usd",
}

NASA_COLUMN_MAP: Dict[str, str] = {
    "project_id":        "project_id",
    "TEAM_SIZE":         "team_size",
    "DURATION":          "sprint_duration_days",
    "SLOC":              "planned_story_points",
    "COMPLETED_SLOC":    "completed_story_points",
    "PRODUCTIVITY":      "velocity",
    "DEFECTS":           "defect_count",
    "EFFORT":            "effort_hours",
    "COST":              "cost_usd",
}

AGILE_COLUMN_MAP: Dict[str, str] = {
    # identity map – raw Agile CSVs are expected in canonical names
    col: col for col in CANONICAL_FEATURES + [TARGET_EFFORT, TARGET_COST,
                                               "project_id", "sprint_id"]
}


def _apply_column_map(df: pd.DataFrame,
                      col_map: Dict[str, str],
                      source_name: str) -> pd.DataFrame:
    """Rename columns and keep only canonical ones."""
    rename = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=rename)
    # Keep only canonical columns that exist
    all_canonical = (["project_id", "sprint_id"] +
                     CANONICAL_FEATURES +
                     [TARGET_EFFORT, TARGET_COST])
    keep = [c for c in all_canonical if c in df.columns]
    df = df[keep].copy()
    df["data_source"] = source_name
    return df


# ──────────────────────────────────────────────────────────────────────
# Unit & Scale Normalization
# ──────────────────────────────────────────────────────────────────────

def _normalize_units(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent units across all datasets."""
    # Convert effort from person-months → hours (if flagged)
    if "effort_unit" in df.columns:
        mask = df["effort_unit"].str.lower() == "person-months"
        df.loc[mask, "effort_hours"] = df.loc[mask, "effort_hours"] * 160
        df.drop(columns=["effort_unit"], inplace=True, errors="ignore")

    # Clip negative values to zero for non-negative fields
    non_neg = ["team_size", "sprint_duration_days", "planned_story_points",
               "completed_story_points", "velocity", "num_stories",
               "defect_count", "effort_hours", "cost_usd"]
    for col in non_neg:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    return df


def _assign_sprint_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Assign monotonic sprint indices per project for temporal alignment."""
    if "sprint_id" not in df.columns:
        df["sprint_id"] = df.groupby("project_id").cumcount()
    else:
        # Re-index to ensure contiguous 0, 1, 2, …
        df["sprint_id"] = df.groupby("project_id").cumcount()
    df = df.sort_values(["project_id", "sprint_id"]).reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def harmonize_isbsg(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonize an ISBSG-format DataFrame into canonical schema."""
    return _assign_sprint_indices(
        _normalize_units(_apply_column_map(df, ISBSG_COLUMN_MAP, "ISBSG")))


def harmonize_promise(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonize a PROMISE-format DataFrame into canonical schema."""
    return _assign_sprint_indices(
        _normalize_units(_apply_column_map(df, PROMISE_COLUMN_MAP, "PROMISE")))


def harmonize_nasa(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonize a NASA-format DataFrame into canonical schema."""
    return _assign_sprint_indices(
        _normalize_units(_apply_column_map(df, NASA_COLUMN_MAP, "NASA")))


def harmonize_agile(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonize raw Agile CSV into canonical schema."""
    return _assign_sprint_indices(
        _normalize_units(_apply_column_map(df, AGILE_COLUMN_MAP, "Agile")))


def merge_datasets(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate multiple harmonized datasets into one canonical DataFrame.
    Ensures unique project IDs across sources by prefixing with source name.
    """
    parts = []
    for df in dataframes:
        df = df.copy()
        if "data_source" in df.columns:
            df["project_id"] = (df["data_source"].astype(str) + "_" +
                                df["project_id"].astype(str))
        parts.append(df)

    merged = pd.concat(parts, ignore_index=True)
    merged = _assign_sprint_indices(merged)
    logger.info(f"Merged dataset: {len(merged)} records, "
                f"{merged['project_id'].nunique()} projects")
    return merged


def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that a DataFrame conforms to the canonical schema.
    Returns (is_valid, list_of_issues).
    """
    issues = []
    required = ["project_id", "sprint_id"] + [TARGET_EFFORT]
    for col in required:
        if col not in df.columns:
            issues.append(f"Missing required column: {col}")

    for col in CANONICAL_FEATURES:
        if col in df.columns and df[col].dtype == object:
            issues.append(f"Feature '{col}' has non-numeric dtype: {df[col].dtype}")

    if TARGET_EFFORT in df.columns and (df[TARGET_EFFORT] < 0).any():
        issues.append(f"Negative values in '{TARGET_EFFORT}'")

    return len(issues) == 0, issues
