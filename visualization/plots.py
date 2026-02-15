"""
Visualization Module
Publication-quality plots for results, data quality, and model analysis.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from typing import Dict, List, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PLOTS_DIR

logger = logging.getLogger(__name__)

# Global style
plt.rcParams.update({
    "figure.figsize": (12, 8),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
})


def _save_fig(fig, name: str, plots_dir: str = PLOTS_DIR):
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Plot saved: {path}")
    return path


# ──────────────────────────────────────────────────────────────────────
# 1. Actual vs Predicted
# ──────────────────────────────────────────────────────────────────────
def plot_actual_vs_predicted(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             title: str = "Actual vs Predicted Effort",
                             model_name: str = "") -> str:
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(y_true, y_pred, alpha=0.5, s=20, color="#2196F3",
               edgecolor="white", linewidth=0.3)

    # Perfect prediction line
    lims = [min(y_true.min(), y_pred.min()),
            max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=2, label="Perfect Prediction")

    # Regression line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax.plot(sorted(y_true), p(sorted(y_true)), "g-", linewidth=1.5,
            label=f"Fit: y={z[0]:.2f}x+{z[1]:.1f}")

    ax.set_xlabel("Actual Effort (hours)")
    ax.set_ylabel("Predicted Effort (hours)")
    ax.set_title(f"{title}\n{model_name}" if model_name else title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return _save_fig(fig, f"actual_vs_predicted_{model_name or 'model'}")


# ──────────────────────────────────────────────────────────────────────
# 2. Sprint-wise Error Heatmap
# ──────────────────────────────────────────────────────────────────────
def plot_sprint_error_heatmap(errors_by_project: Dict[str, np.ndarray],
                              max_projects: int = 30,
                              title: str = "Sprint-wise Prediction Error") -> str:
    # Limit to top projects by max error
    projects = list(errors_by_project.keys())[:max_projects]
    max_sprints = max(len(v) for v in errors_by_project.values())

    matrix = np.full((len(projects), max_sprints), np.nan)
    for i, pid in enumerate(projects):
        arr = errors_by_project[pid]
        matrix[i, :len(arr)] = arr

    fig, ax = plt.subplots(figsize=(14, max(6, len(projects) * 0.3)))
    sns.heatmap(matrix, ax=ax, cmap="YlOrRd", annot=False,
                xticklabels=[f"S{i}" for i in range(max_sprints)],
                yticklabels=projects,
                cbar_kws={"label": "Absolute Error (hours)"})
    ax.set_xlabel("Sprint Index")
    ax.set_ylabel("Project ID")
    ax.set_title(title)

    return _save_fig(fig, "sprint_error_heatmap")


# ──────────────────────────────────────────────────────────────────────
# 3. Data Quality Dashboard
# ──────────────────────────────────────────────────────────────────────
def plot_data_quality_dashboard(reports: List[Dict],
                                title: str = "Data Quality Pipeline Dashboard") -> str:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 3a. Records per stage
    stages = [r.get("stage", f"S{i}") for i, r in enumerate(reports)]
    records = [r.get("records_after", r.get("records_before", 0))
               for r in reports]
    axes[0, 0].bar(stages, records, color="#4CAF50", alpha=0.8)
    axes[0, 0].set_title("Records After Each Stage")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # 3b. Missingness heatmap
    miss_data = {}
    for r in reports:
        if "missingness" in r.get("stats", {}):
            miss_data = r["stats"]["missingness"]
            break

    if miss_data:
        features = list(miss_data.keys())[:15]
        values = [miss_data[f] for f in features]
        axes[0, 1].barh(features, values, color="#FF9800", alpha=0.8)
        axes[0, 1].set_title("Feature Missingness Ratio")
        axes[0, 1].set_xlabel("Missing Ratio")
    else:
        axes[0, 1].text(0.5, 0.5, "No missingness data",
                        ha="center", va="center", fontsize=14)
        axes[0, 1].set_title("Feature Missingness Ratio")

    # 3c. Outlier counts
    outlier_data = {}
    for r in reports:
        if "n_outliers" in r.get("stats", {}):
            outlier_data = r["stats"]
            break

    if outlier_data:
        labels = ["Outliers", "Clean"]
        n_outliers = outlier_data.get("n_outliers", 0)
        # Find the records_after from the outlier detection stage
        n_total = 100
        for r in reports:
            if r.get("stage") == "outlier_detection":
                n_total = r.get("records_after", 100)
                break
        sizes = [n_outliers, max(n_total - n_outliers, 0)]

        axes[1, 0].pie(sizes, labels=labels, autopct="%1.1f%%",
                       colors=["#F44336", "#4CAF50"])
        axes[1, 0].set_title("Outlier Distribution")
    else:
        axes[1, 0].text(0.5, 0.5, "No outlier data",
                        ha="center", va="center", fontsize=14)
        axes[1, 0].set_title("Outlier Distribution")

    # 3d. Actions summary
    all_actions = []
    for r in reports:
        for a in r.get("actions", []):
            all_actions.append(f"[{r.get('stage', '?')}] {a}")

    action_text = "\n".join(all_actions[:15])
    axes[1, 1].text(0.05, 0.95, action_text, transform=axes[1, 1].transAxes,
                    fontsize=8, verticalalignment="top",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    axes[1, 1].set_title("Quality Actions Log")
    axes[1, 1].axis("off")

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    return _save_fig(fig, "data_quality_dashboard")


# ──────────────────────────────────────────────────────────────────────
# 4. Feature Importance
# ──────────────────────────────────────────────────────────────────────
def plot_feature_importance(importance: Dict[str, float],
                            title: str = "Feature Importance",
                            top_k: int = 20) -> str:
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    sorted_imp = sorted_imp[:top_k]

    features = [f for f, _ in sorted_imp]
    values = [v for _, v in sorted_imp]

    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.35)))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
    ax.barh(features[::-1], values[::-1], color=colors)
    ax.set_xlabel("Importance Score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="x")

    return _save_fig(fig, "feature_importance")


# ──────────────────────────────────────────────────────────────────────
# 5. Training Loss Curves
# ──────────────────────────────────────────────────────────────────────
def plot_training_curves(history: Dict[str, List[float]],
                         title: str = "Training Progress") -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    ax1.plot(epochs, history["train_loss"], "b-", linewidth=2,
             label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", linewidth=2,
             label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Learning rate
    if "lr" in history:
        ax2.plot(epochs, history["lr"], "g-", linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("Learning Rate Schedule")
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale("log")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    return _save_fig(fig, "training_curves")


# ──────────────────────────────────────────────────────────────────────
# 6. Model Comparison
# ──────────────────────────────────────────────────────────────────────
def plot_model_comparison(results: Dict[str, Dict[str, float]],
                          title: str = "Model Performance Comparison") -> str:
    """
    Results format: {"ModelName": {"MAE": x, "RMSE": x, "MAPE": x, "R2": x}}
    """
    models = list(results.keys())
    metrics = ["MAE", "RMSE", "MAPE", "R2"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for i, metric in enumerate(metrics):
        values = [results[m].get(metric, 0) for m in models]
        bars = axes[i].bar(models, values, color=colors)
        axes[i].set_title(metric, fontsize=14, fontweight="bold")
        axes[i].tick_params(axis="x", rotation=45)
        axes[i].grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar, val in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    return _save_fig(fig, "model_comparison")


# ──────────────────────────────────────────────────────────────────────
# 7. Attention Heatmap Plot
# ──────────────────────────────────────────────────────────────────────
def plot_attention_heatmap(attention: np.ndarray,
                           sample_idx: int = 0,
                           title: str = "Transformer Attention Heatmap") -> str:
    if attention.ndim == 3:
        attn = attention[sample_idx]
    else:
        attn = attention

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(attn, ax=ax, cmap="Blues", square=True,
                cbar_kws={"label": "Attention Weight"})
    ax.set_xlabel("Key Sprint")
    ax.set_ylabel("Query Sprint")
    ax.set_title(title)

    return _save_fig(fig, "attention_heatmap")


# ──────────────────────────────────────────────────────────────────────
# 8. Data Quality Impact (Before vs After)
# ──────────────────────────────────────────────────────────────────────
def plot_quality_impact(comparison: Dict[str, Dict[str, float]],
                        title: str = "Data Quality Impact on Performance") -> str:
    metrics = list(comparison.keys())
    before = [comparison[m]["before_cleaning"] for m in metrics]
    after = [comparison[m]["after_cleaning"] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    bars1 = ax.bar(x - width/2, before, width, label="Before Cleaning",
                   color="#FF7043", alpha=0.8)
    bars2 = ax.bar(x + width/2, after, width, label="After Cleaning",
                   color="#66BB6A", alpha=0.8)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Improvement annotations
    for i, metric in enumerate(metrics):
        imp = comparison[metric]["improvement_pct"]
        color = "green" if imp > 0 else "red"
        ax.annotate(f"{imp:+.1f}%",
                    xy=(i, max(before[i], after[i])),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold",
                    color=color)

    plt.tight_layout()
    return _save_fig(fig, "quality_impact")
