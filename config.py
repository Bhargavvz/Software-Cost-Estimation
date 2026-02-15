"""
Central configuration for the Ultra-Scale Software Cost Estimation system.
Designed for NVIDIA H200 SXM (141 GB HBM3e) with BF16 mixed-precision.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

# ──────────────────────────────────────────────────────────────────────
# Path configuration
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
DATA_QUALITY_DIR = os.path.join(OUTPUT_DIR, "data_quality")
VERSIONED_DATA_DIR = os.path.join(OUTPUT_DIR, "versioned_data")

for _d in [DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR, PLOTS_DIR,
           DATA_QUALITY_DIR, VERSIONED_DATA_DIR]:
    os.makedirs(_d, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# Canonical Agile Feature Schema
# ──────────────────────────────────────────────────────────────────────
CANONICAL_FEATURES = [
    "team_size",
    "sprint_duration_days",
    "planned_story_points",
    "completed_story_points",
    "velocity",
    "num_stories",
    "avg_story_complexity",
    "defect_count",
    "scope_change_ratio",
    "team_experience_months",
    "tech_debt_hours",
    "meeting_overhead_hours",
]

TARGET_EFFORT = "effort_hours"
TARGET_COST = "cost_usd"


# ──────────────────────────────────────────────────────────────────────
# Hardware-Aware Training (H200 SXM)
# ──────────────────────────────────────────────────────────────────────
@dataclass
class H200Config:
    """Settings tuned for NVIDIA H200 SXM with 141 GB HBM3e."""
    device: str = "cuda"
    precision: str = "bf16"             # bf16 | fp16 | fp32
    batch_size: int = 512               # large-batch on H200
    gradient_accumulation_steps: int = 4
    effective_batch_size: int = 2048     # batch_size * grad_accum
    max_epochs: int = 200
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 10
    patience: int = 25                  # early stopping
    num_workers: int = 8
    pin_memory: bool = True
    compile_model: bool = True          # torch.compile for H200
    gradient_clip_norm: float = 1.0


# ──────────────────────────────────────────────────────────────────────
# Model Hyper-parameters
# ──────────────────────────────────────────────────────────────────────
@dataclass
class LSTMConfig:
    input_dim: int = len(CANONICAL_FEATURES)
    hidden_dim: int = 512
    num_layers: int = 4
    dropout: float = 0.3
    bidirectional: bool = True


@dataclass
class TransformerConfig:
    input_dim: int = len(CANONICAL_FEATURES)
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 200              # max sprint sequence length


@dataclass
class HybridCNNTransformerConfig:
    input_dim: int = len(CANONICAL_FEATURES)
    cnn_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    cnn_kernel_size: int = 3
    d_model: int = 256
    nhead: int = 8
    num_transformer_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.15
    max_seq_len: int = 200


# ──────────────────────────────────────────────────────────────────────
# Data Quality Thresholds
# ──────────────────────────────────────────────────────────────────────
@dataclass
class DataQualityConfig:
    # Missing data
    max_missing_ratio: float = 0.40     # drop feature if > 40% missing
    interpolation_method: str = "linear"

    # Outlier detection
    iqr_multiplier: float = 1.5
    isolation_forest_contamination: float = 0.05
    temporal_anomaly_window: int = 3

    # Agile consistency
    max_velocity_jump_ratio: float = 2.0  # velocity can't 2× in one sprint
    min_effort_per_point: float = 0.5
    max_effort_per_point: float = 40.0

    # Temporal smoothing
    moving_avg_window: int = 3
    kalman_process_noise: float = 0.01
    kalman_measurement_noise: float = 0.1

    # Feature selection
    min_mutual_info: float = 0.01
    max_correlation: float = 0.95       # remove one of highly-corr pair


# ──────────────────────────────────────────────────────────────────────
# Synthetic Data Generation
# ──────────────────────────────────────────────────────────────────────
@dataclass
class SyntheticConfig:
    n_projects: int = 500
    sprints_per_project_range: tuple = (8, 40)
    team_size_range: tuple = (3, 15)
    base_velocity_range: tuple = (20, 80)
    scope_creep_probability: float = 0.15
    failure_probability: float = 0.05
    noise_scale: float = 0.10
    seed: int = 42


# ──────────────────────────────────────────────────────────────────────
# Curriculum Learning
# ──────────────────────────────────────────────────────────────────────
@dataclass
class CurriculumConfig:
    enabled: bool = True
    phases: List[dict] = field(default_factory=lambda: [
        {"name": "clean",    "epoch_start": 0,   "epoch_end": 50,
         "quality_threshold": 0.9,  "include_synthetic": False},
        {"name": "augmented", "epoch_start": 50,  "epoch_end": 120,
         "quality_threshold": 0.6,  "include_synthetic": True},
        {"name": "full",      "epoch_start": 120, "epoch_end": 200,
         "quality_threshold": 0.0,  "include_synthetic": True},
    ])


# ──────────────────────────────────────────────────────────────────────
# Convenience singleton
# ──────────────────────────────────────────────────────────────────────
hw = H200Config()
lstm_cfg = LSTMConfig()
transformer_cfg = TransformerConfig()
hybrid_cfg = HybridCNNTransformerConfig()
dq_cfg = DataQualityConfig()
syn_cfg = SyntheticConfig()
curriculum_cfg = CurriculumConfig()
