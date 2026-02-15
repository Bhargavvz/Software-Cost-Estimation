"""
Curriculum Learning Scheduler
Progressively introduces harder examples during training,
starting with clean high-confidence data and gradually adding
noisy/synthetic/complex samples.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
import logging

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import curriculum_cfg

logger = logging.getLogger(__name__)


class CurriculumScheduler:
    """
    Data difficulty scoring + epoch-based sample filtering.

    Curriculum phases:
        1. Clean: only high-confidence real data
        2. Augmented: add synthetic data, lower quality threshold
        3. Full: all data including noisy / edge-case samples
    """

    def __init__(self, config=None):
        self.cfg = config or curriculum_cfg
        self.current_phase_idx = 0

    def _get_phase(self, epoch: int) -> dict:
        """Get the curriculum phase for the given epoch."""
        for phase in self.cfg.phases:
            if phase["epoch_start"] <= epoch < phase["epoch_end"]:
                return phase
        # Default to last phase
        return self.cfg.phases[-1]

    def score_difficulty(self, df: pd.DataFrame) -> pd.Series:
        """
        Score each sample's difficulty based on:
        - Data quality score (from quality pipeline)
        - Whether it's synthetic
        - Label variance within project
        """
        scores = pd.Series(0.0, index=df.index)

        # Quality score (higher = easier)
        if "quality_score" in df.columns:
            scores += (1.0 - df["quality_score"]) * 0.4

        # Synthetic data is slightly harder
        if "data_source" in df.columns:
            is_synthetic = df["data_source"].str.contains("synthetic", na=False)
            scores += is_synthetic.astype(float) * 0.2

            # Rare events are hardest
            is_rare = df["data_source"] == "synthetic_rare"
            scores += is_rare.astype(float) * 0.3

        # High label variance within project = harder
        if "effort_hours" in df.columns:
            project_cv = df.groupby("project_id")["effort_hours"].transform(
                lambda s: s.std() / max(s.mean(), 1))
            cv_normalized = (project_cv - project_cv.min()) / \
                            max(project_cv.max() - project_cv.min(), 1e-6)
            scores += cv_normalized * 0.1

        return scores.clip(0, 1)

    def filter_for_epoch(self, df: pd.DataFrame,
                         epoch: int) -> pd.DataFrame:
        """
        Filter dataset for the current curriculum phase.

        Returns:
            Filtered DataFrame appropriate for this training epoch.
        """
        if not self.cfg.enabled:
            return df

        phase = self._get_phase(epoch)
        difficulty = self.score_difficulty(df)

        # Filter by quality threshold
        quality_mask = difficulty <= (1.0 - phase["quality_threshold"])

        # Filter synthetic data
        if not phase["include_synthetic"]:
            if "data_source" in df.columns:
                synthetic_mask = ~df["data_source"].str.contains(
                    "synthetic", na=False)
                quality_mask &= synthetic_mask

        filtered = df[quality_mask].copy()

        if len(filtered) < 10:  # safety: always have some data
            filtered = df.copy()

        logger.debug(f"Epoch {epoch} [{phase['name']}]: "
                     f"{len(filtered)}/{len(df)} samples")
        return filtered

    def get_phase_name(self, epoch: int) -> str:
        return self._get_phase(epoch)["name"]
