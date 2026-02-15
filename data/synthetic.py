"""
Synthetic Agile Sprint Generator
Produces realistic sprint sequences modeling velocity drift, scope creep,
team learning curves, and rare failure scenarios.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
import logging

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CANONICAL_FEATURES, TARGET_EFFORT, TARGET_COST, syn_cfg

logger = logging.getLogger(__name__)


class AgileSynthesizer:
    """
    Generates synthetic Agile sprint data with realistic dynamics:
    - Team learning curves (logarithmic productivity ramp)
    - Velocity drift (random walk with mean-reversion)
    - Scope creep injection (sudden story-point inflation)
    - Rare failures (sprint collapse, team departure, critical bugs)
    """

    def __init__(self, config=None):
        self.cfg = config or syn_cfg
        self.rng = np.random.default_rng(self.cfg.seed)

    # ──────────────────────────────────────────────────────────────
    # Team learning curve
    # ──────────────────────────────────────────────────────────────
    def _learning_curve(self, n_sprints: int,
                        maturity_sprint: int = 10) -> np.ndarray:
        """
        Logarithmic learning curve: productivity ramps from ~0.5 to ~1.0
        over `maturity_sprint` sprints, then plateaus.
        """
        t = np.arange(n_sprints, dtype=np.float64)
        curve = 0.5 + 0.5 * np.log1p(t) / np.log1p(maturity_sprint)
        return np.clip(curve, 0.5, 1.05)

    # ──────────────────────────────────────────────────────────────
    # Velocity drift (mean-reverting random walk)
    # ──────────────────────────────────────────────────────────────
    def _velocity_drift(self, n_sprints: int,
                        base_velocity: float,
                        reversion_rate: float = 0.15) -> np.ndarray:
        """Ornstein-Uhlenbeck-like velocity process."""
        vel = np.zeros(n_sprints)
        vel[0] = base_velocity
        for t in range(1, n_sprints):
            noise = self.rng.normal(0, base_velocity * self.cfg.noise_scale)
            vel[t] = (vel[t-1]
                      + reversion_rate * (base_velocity - vel[t-1])
                      + noise)
        return np.clip(vel, base_velocity * 0.3, base_velocity * 2.0)

    # ──────────────────────────────────────────────────────────────
    # Scope creep
    # ──────────────────────────────────────────────────────────────
    def _inject_scope_creep(self, planned_points: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray]:
        """
        Randomly inflate planned story points mid-sprint.
        Returns (modified_planned, scope_change_ratio).
        """
        n = len(planned_points)
        creep_mask = self.rng.random(n) < self.cfg.scope_creep_probability
        creep_factor = 1.0 + self.rng.uniform(0.1, 0.5, n) * creep_mask
        new_planned = planned_points * creep_factor
        scope_change = (new_planned - planned_points) / np.maximum(planned_points, 1)
        return new_planned, scope_change

    # ──────────────────────────────────────────────────────────────
    # Rare failure events
    # ──────────────────────────────────────────────────────────────
    def _inject_failures(self, velocity: np.ndarray,
                         defects: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate sprint collapses: velocity drops sharply, defects spike.
        """
        n = len(velocity)
        fail_mask = self.rng.random(n) < self.cfg.failure_probability
        velocity_out = velocity.copy()
        defects_out = defects.copy()
        velocity_out[fail_mask] *= self.rng.uniform(0.1, 0.4,
                                                     fail_mask.sum())
        defects_out[fail_mask] = (defects_out[fail_mask] *
                                  self.rng.uniform(2, 5, fail_mask.sum()))
        return velocity_out, defects_out.astype(int)

    # ──────────────────────────────────────────────────────────────
    # Generate a single project
    # ──────────────────────────────────────────────────────────────
    def _generate_project(self, project_id: str,
                          n_sprints: int) -> pd.DataFrame:
        """Generate a single project's sprint sequence."""
        team_size = self.rng.integers(*self.cfg.team_size_range)
        base_vel = self.rng.uniform(*self.cfg.base_velocity_range)
        sprint_duration = self.rng.choice([7, 10, 14, 21])
        team_exp = self.rng.uniform(6, 120)  # months

        # Learning curve & velocity
        learn = self._learning_curve(n_sprints)
        vel_raw = self._velocity_drift(n_sprints, base_vel)
        velocity = vel_raw * learn

        # Story points planning
        planned_sp = velocity + self.rng.normal(0, base_vel * 0.05, n_sprints)
        planned_sp = np.clip(planned_sp, 5, 500)
        planned_sp, scope_change = self._inject_scope_creep(planned_sp)

        # Completion ratio (affected by team learning)
        completion_ratio = np.clip(
            learn * self.rng.normal(0.85, 0.08, n_sprints), 0.3, 1.0)
        completed_sp = planned_sp * completion_ratio

        # Num stories
        avg_sp_per_story = self.rng.uniform(2, 8)
        num_stories = np.maximum(1, (planned_sp / avg_sp_per_story)).astype(int)

        # Complexity (1-5 scale)
        avg_complexity = np.clip(
            self.rng.normal(3.0, 0.8, n_sprints), 1, 5)

        # Defects
        defect_base = (planned_sp * 0.05 *
                       self.rng.uniform(0.5, 1.5, n_sprints))
        defects = np.maximum(0, defect_base).astype(int)

        # Inject failures
        velocity, defects = self._inject_failures(velocity, defects)

        # Tech debt & meeting overhead
        tech_debt = self.rng.uniform(2, 20, n_sprints) * (team_size / 5)
        meeting_hours = team_size * sprint_duration * self.rng.uniform(
            0.02, 0.08, n_sprints)

        # Effort (target) = f(story_points, complexity, team_size, duration)
        base_effort = (completed_sp * avg_complexity *
                       self.rng.uniform(1.5, 4.0, n_sprints))
        effort_hours = base_effort + tech_debt + meeting_hours
        effort_hours *= self.rng.normal(1.0, self.cfg.noise_scale, n_sprints)
        effort_hours = np.clip(effort_hours, 10, 50000)

        # Cost (target) = effort * blended hourly rate
        hourly_rate = self.rng.uniform(40, 150)
        cost_usd = effort_hours * hourly_rate * self.rng.normal(
            1.0, 0.05, n_sprints)
        cost_usd = np.clip(cost_usd, 100, 5_000_000)

        # Team experience grows each sprint
        team_exp_array = team_exp + np.arange(n_sprints) * (sprint_duration / 30)

        return pd.DataFrame({
            "project_id":             project_id,
            "sprint_id":              np.arange(n_sprints),
            "team_size":              float(team_size),
            "sprint_duration_days":   float(sprint_duration),
            "planned_story_points":   planned_sp,
            "completed_story_points": completed_sp,
            "velocity":               velocity,
            "num_stories":            num_stories,
            "avg_story_complexity":   avg_complexity,
            "defect_count":           defects,
            "scope_change_ratio":     scope_change,
            "team_experience_months": team_exp_array,
            "tech_debt_hours":        tech_debt,
            "meeting_overhead_hours": meeting_hours,
            TARGET_EFFORT:            effort_hours,
            TARGET_COST:              cost_usd,
            "data_source":            "synthetic",
            "quality_score":          1.0,
        })

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────
    def generate(self, n_projects: Optional[int] = None,
                 sprints_per_project: Optional[int] = None) -> pd.DataFrame:
        """
        Generate synthetic Agile sprint dataset.

        Args:
            n_projects: Number of projects (default from config)
            sprints_per_project: Fixed sprint count, or None for random
                range from config.

        Returns:
            DataFrame in canonical schema.
        """
        n_projects = n_projects or self.cfg.n_projects

        projects = []
        for i in range(n_projects):
            pid = f"SYN_{i:05d}"
            if sprints_per_project is not None:
                n_sp = sprints_per_project
            else:
                lo, hi = self.cfg.sprints_per_project_range
                n_sp = self.rng.integers(lo, hi + 1)

            projects.append(self._generate_project(pid, n_sp))

        df = pd.concat(projects, ignore_index=True)
        logger.info(f"Generated {len(df)} synthetic sprint records "
                    f"across {n_projects} projects")
        return df

    def generate_rare_events(self, n_events: int = 100) -> pd.DataFrame:
        """
        Generate edge-case sprint sequences with high failure rates
        and extreme scope creep for stress-testing models.
        """
        orig_fail = self.cfg.failure_probability
        orig_creep = self.cfg.scope_creep_probability

        self.cfg.failure_probability = 0.30
        self.cfg.scope_creep_probability = 0.40

        df = self.generate(n_projects=n_events, sprints_per_project=12)
        df["data_source"] = "synthetic_rare"

        # Restore
        self.cfg.failure_probability = orig_fail
        self.cfg.scope_creep_probability = orig_creep

        logger.info(f"Generated {len(df)} rare-event records")
        return df
