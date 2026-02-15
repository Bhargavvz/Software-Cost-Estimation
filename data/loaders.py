"""
Data Loaders & PyTorch Dataset
Handles dataset loading, temporal-aware splitting, and PyTorch integration.
"""

import numpy as np
import pandas as pd
import os
import logging
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (CANONICAL_FEATURES, TARGET_EFFORT, TARGET_COST,
                    DATA_DIR, hw)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# CSV / Parquet Loaders
# ──────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file with basic type handling."""
    df = pd.read_csv(path)
    logger.info(f"Loaded {path}: {df.shape}")
    return df


def load_parquet(path: str) -> pd.DataFrame:
    """Load a Parquet file."""
    df = pd.read_parquet(path)
    logger.info(f"Loaded {path}: {df.shape}")
    return df


def load_all_datasets(data_dir: str = DATA_DIR) -> List[pd.DataFrame]:
    """Load all CSV/Parquet files from the data directory."""
    dfs = []
    if not os.path.exists(data_dir):
        logger.warning(f"Data directory not found: {data_dir}")
        return dfs

    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        if fname.endswith(".csv"):
            dfs.append(load_csv(fpath))
        elif fname.endswith(".parquet"):
            dfs.append(load_parquet(fpath))
    return dfs


# ──────────────────────────────────────────────────────────────────────
# Temporal-Aware Splitting
# ──────────────────────────────────────────────────────────────────────

def temporal_split(df: pd.DataFrame,
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15,
                   test_ratio: float = 0.15) \
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by project — each project's sprints stay together.
    Projects are randomly assigned to splits (not chronological across
    projects, but chronological within each project is preserved).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    project_ids = df["project_id"].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(project_ids)

    n = len(project_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_pids = set(project_ids[:n_train])
    val_pids = set(project_ids[n_train:n_train + n_val])
    test_pids = set(project_ids[n_train + n_val:])

    train_df = df[df["project_id"].isin(train_pids)].reset_index(drop=True)
    val_df = df[df["project_id"].isin(val_pids)].reset_index(drop=True)
    test_df = df[df["project_id"].isin(test_pids)].reset_index(drop=True)

    logger.info(f"Split: train={len(train_df)}, val={len(val_df)}, "
                f"test={len(test_df)}")
    return train_df, val_df, test_df


# ──────────────────────────────────────────────────────────────────────
# Feature Normalization
# ──────────────────────────────────────────────────────────────────────

class FeatureNormalizer:
    """Z-score normalization fitted on training data."""

    def __init__(self):
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.fitted = False

    def fit(self, df: pd.DataFrame, feature_cols: List[str]):
        for col in feature_cols:
            if col in df.columns:
                self.means[col] = df[col].mean()
                self.stds[col] = df[col].std()
                if self.stds[col] < 1e-8:
                    self.stds[col] = 1.0
        self.fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.means:
            if col in df.columns:
                df[col] = (df[col] - self.means[col]) / self.stds[col]
        return df

    def fit_transform(self, df: pd.DataFrame,
                      feature_cols: List[str]) -> pd.DataFrame:
        self.fit(df, feature_cols)
        return self.transform(df)

    def inverse_transform_target(self, values: np.ndarray,
                                 target: str = TARGET_EFFORT) -> np.ndarray:
        """Un-normalize target for evaluation."""
        if target in self.means:
            return values * self.stds[target] + self.means[target]
        return values


# ──────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────

class SprintDataset(Dataset):
    """
    PyTorch dataset that groups sprints by project into sequences.
    Each item is (feature_sequence, target, sample_weight, seq_len).
    """

    def __init__(self, df: pd.DataFrame,
                 feature_cols: List[str],
                 target_col: str = TARGET_EFFORT,
                 max_seq_len: int = 200):
        self.max_seq_len = max_seq_len
        self.target_col = target_col
        self.feature_cols = feature_cols

        # Group by project
        self.sequences = []
        self.targets = []
        self.weights = []

        for pid, grp in df.groupby("project_id"):
            grp = grp.sort_values("sprint_id")
            feats = grp[feature_cols].values.astype(np.float32)
            target = grp[target_col].values[-1].astype(np.float32)

            weight = 1.0
            if "sample_weight" in grp.columns:
                weight = grp["sample_weight"].mean()

            self.sequences.append(feats)
            self.targets.append(target)
            self.weights.append(float(weight))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        target = self.targets[idx]
        weight = self.weights[idx]

        seq_len = min(len(seq), self.max_seq_len)

        # Pad or truncate
        if len(seq) >= self.max_seq_len:
            seq = seq[:self.max_seq_len]
        else:
            pad = np.zeros((self.max_seq_len - len(seq), seq.shape[1]),
                           dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)

        return (torch.tensor(seq, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32),
                torch.tensor(weight, dtype=torch.float32),
                torch.tensor(seq_len, dtype=torch.long))


def collate_sprints(batch):
    """Custom collation for SprintDataset."""
    seqs, targets, weights, lengths = zip(*batch)
    return (torch.stack(seqs),
            torch.stack(targets),
            torch.stack(weights),
            torch.stack(lengths))


def create_dataloaders(train_df: pd.DataFrame,
                       val_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       feature_cols: List[str],
                       target_col: str = TARGET_EFFORT,
                       batch_size: int = None,
                       max_seq_len: int = 200) \
        -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train / val / test DataLoaders."""
    bs = batch_size or hw.batch_size

    train_ds = SprintDataset(train_df, feature_cols, target_col, max_seq_len)
    val_ds = SprintDataset(val_df, feature_cols, target_col, max_seq_len)
    test_ds = SprintDataset(test_df, feature_cols, target_col, max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              collate_fn=collate_sprints,
                              num_workers=hw.num_workers,
                              pin_memory=hw.pin_memory, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            collate_fn=collate_sprints,
                            num_workers=hw.num_workers,
                            pin_memory=hw.pin_memory)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                             collate_fn=collate_sprints,
                             num_workers=hw.num_workers,
                             pin_memory=hw.pin_memory)

    logger.info(f"DataLoaders: train={len(train_ds)}, val={len(val_ds)}, "
                f"test={len(test_ds)}, batch_size={bs}")
    return train_loader, val_loader, test_loader
