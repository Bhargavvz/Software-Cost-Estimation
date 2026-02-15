"""
run_pipeline.py — End-to-End Orchestrator
Data generation → Quality pipeline → Training → Evaluation → Visualization

Usage:
    python run_pipeline.py                           # full run on GPU
    python run_pipeline.py --mode dry_run --device cpu --epochs 2
    python run_pipeline.py --model transformer --epochs 100
"""

import argparse
import logging
import os
import sys
import json
import time
import numpy as np
import pandas as pd

# ── Project imports ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (CANONICAL_FEATURES, TARGET_EFFORT, TARGET_COST,
                    OUTPUT_DIR, PLOTS_DIR, DATA_QUALITY_DIR,
                    hw, lstm_cfg, transformer_cfg, hybrid_cfg,
                    dq_cfg, syn_cfg, curriculum_cfg)
from data.synthetic import AgileSynthesizer
from data.quality import DataQualityPipeline
from data.loaders import (temporal_split, FeatureNormalizer,
                           SprintDataset, create_dataloaders)
from evaluation.metrics import compute_all_metrics, print_metrics
from evaluation.analysis import AblationStudy
from visualization.plots import (plot_actual_vs_predicted,
                                  plot_data_quality_dashboard,
                                  plot_feature_importance,
                                  plot_training_curves,
                                  plot_model_comparison,
                                  plot_quality_impact)

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-20s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ══════════════════════════════════════════════════════════════════════
#  Stage runners
# ══════════════════════════════════════════════════════════════════════

def stage_generate_data(args) -> pd.DataFrame:
    """Generate synthetic Agile sprint data."""
    logger.info("=" * 70)
    logger.info("STAGE 1: Synthetic Data Generation")
    logger.info("=" * 70)

    synth = AgileSynthesizer()
    df = synth.generate(n_projects=args.n_projects,
                        sprints_per_project=args.n_sprints
                        if args.n_sprints else None)

    # Add rare events
    rare = synth.generate_rare_events(n_events=max(10, args.n_projects // 10))
    df = pd.concat([df, rare], ignore_index=True)

    logger.info(f"Generated {len(df)} total sprint records "
                f"({df['project_id'].nunique()} projects)")
    return df


def stage_data_quality(df: pd.DataFrame, args) -> tuple:
    """Run the full data quality pipeline."""
    logger.info("=" * 70)
    logger.info("STAGE 2: Data Quality Pipeline")
    logger.info("=" * 70)

    pipeline = DataQualityPipeline(versioning=not args.no_versioning)
    clean_df, reports = pipeline.run(df)

    logger.info(f"Quality pipeline: {len(df)} → {len(clean_df)} records")
    return clean_df, reports, pipeline.selected_features


def stage_train_baselines(train_df, val_df, test_df,
                          feature_cols, args) -> dict:
    """Train and evaluate baseline ML models."""
    logger.info("=" * 70)
    logger.info("STAGE 3a: Baseline Models (XGBoost, Random Forest)")
    logger.info("=" * 70)

    from models.baseline import XGBoostEstimator, RandomForestEstimator

    results = {}

    for name, ModelClass in [("XGBoost", XGBoostEstimator),
                              ("RandomForest", RandomForestEstimator)]:
        logger.info(f"\nTraining {name}...")
        model = ModelClass()
        model.fit(train_df, feature_cols)

        # Evaluate on test
        preds = model.predict(test_df, feature_cols)

        # Get actual targets
        actuals = []
        for pid, grp in test_df.groupby("project_id"):
            actuals.append(grp[TARGET_EFFORT].values[-1])
        actuals = np.array(actuals)

        metrics = compute_all_metrics(actuals, preds)
        print_metrics(metrics, name)
        results[name] = {
            "metrics": metrics,
            "predictions": preds,
            "actuals": actuals,
            "importance": model.feature_importance(),
        }

    return results


def stage_train_deep_model(model_name, ModelClass, train_df, val_df,
                           test_df, feature_cols, args) -> dict:
    """Train and evaluate a deep learning model."""
    import torch
    from training.trainer import H200Trainer
    from training.curriculum import CurriculumScheduler

    logger.info(f"\nTraining {model_name}...")

    # Curriculum-aware data
    curriculum = CurriculumScheduler()

    # Create model
    model = ModelClass(input_dim=len(feature_cols))

    # Configure trainer
    train_cfg = type(hw)()
    train_cfg.device = args.device
    train_cfg.max_epochs = args.epochs
    # Batch size should be relative to project count, not row count
    n_train_projects = train_df["project_id"].nunique()
    train_cfg.batch_size = min(hw.batch_size, max(4, n_train_projects // 4))
    if args.device == "cpu":
        train_cfg.precision = "fp32"
        train_cfg.compile_model = False
        train_cfg.num_workers = 0

    # Create data loaders (batch_size will be further capped in create_dataloaders)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, feature_cols,
        batch_size=train_cfg.batch_size,
        max_seq_len=200)

    # Train
    trainer = H200Trainer(model, config=train_cfg)
    history = trainer.train(train_loader, val_loader,
                            max_epochs=args.epochs)

    # Evaluate
    preds, actuals = trainer.predict(test_loader)
    metrics = compute_all_metrics(actuals, preds)
    print_metrics(metrics, model_name)

    return {
        "metrics": metrics,
        "predictions": preds,
        "actuals": actuals,
        "history": history,
        "model": model,
    }


def stage_train_deep_models(train_df, val_df, test_df,
                            feature_cols, args) -> dict:
    """Train all deep learning models."""
    logger.info("=" * 70)
    logger.info("STAGE 3b: Deep Learning Models")
    logger.info("=" * 70)

    from models.lstm import DeepLSTMStack
    from models.transformer import SprintTransformer
    from models.hybrid import CNNTransformerHybrid

    deep_models = {
        "LSTM": DeepLSTMStack,
        "Transformer": SprintTransformer,
        "CNN-Transformer": CNNTransformerHybrid,
    }

    # Filter by args
    if args.model != "all":
        model_map = {
            "lstm": "LSTM",
            "transformer": "Transformer",
            "hybrid": "CNN-Transformer",
        }
        selected = model_map.get(args.model.lower())
        if selected and selected in deep_models:
            deep_models = {selected: deep_models[selected]}

    results = {}
    for name, ModelClass in deep_models.items():
        result = stage_train_deep_model(
            name, ModelClass, train_df, val_df, test_df,
            feature_cols, args)
        results[name] = result

    return results


def stage_visualize(baseline_results, deep_results,
                    quality_reports, args) -> None:
    """Generate all visualizations."""
    logger.info("=" * 70)
    logger.info("STAGE 4: Visualization & Analysis")
    logger.info("=" * 70)

    # Combine all results
    all_results = {}
    for name, r in {**baseline_results, **deep_results}.items():
        all_results[name] = r["metrics"]

        # Actual vs predicted
        if "predictions" in r and "actuals" in r:
            plot_actual_vs_predicted(
                r["actuals"], r["predictions"], model_name=name)

        # Feature importance (baselines)
        if "importance" in r:
            plot_feature_importance(r["importance"],
                                   title=f"Feature Importance — {name}")

        # Training curves (deep models)
        if "history" in r:
            plot_training_curves(r["history"],
                                title=f"Training Progress — {name}")

    # Model comparison
    if len(all_results) > 1:
        plot_model_comparison(all_results)

    # Data quality dashboard
    if quality_reports:
        plot_data_quality_dashboard(quality_reports)

    logger.info(f"All plots saved to: {PLOTS_DIR}")


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Ultra-Scale Software Cost Estimation Pipeline")
    parser.add_argument("--mode", choices=["full", "dry_run"],
                        default="full",
                        help="Run mode (dry_run = small data, few epochs)")
    parser.add_argument("--device", default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--model", default="all",
                        choices=["all", "lstm", "transformer", "hybrid"],
                        help="Which deep model(s) to train")
    parser.add_argument("--n_projects", type=int, default=500,
                        help="Number of synthetic projects")
    parser.add_argument("--n_sprints", type=int, default=None,
                        help="Fixed sprints per project (None = random)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Training epochs")
    parser.add_argument("--no_versioning", action="store_true",
                        help="Disable data versioning")

    args = parser.parse_args()

    # Dry-run overrides
    if args.mode == "dry_run":
        args.n_projects = min(args.n_projects, 20)
        args.n_sprints = args.n_sprints or 8
        args.epochs = min(args.epochs, 3)
        args.device = "cpu"
        logger.info("DRY RUN MODE: small data, CPU, 3 epochs")

    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║  Ultra-Scale Software Cost Estimation Pipeline      ║")
    logger.info("║  Data-Centric Deep Learning for Agile Methodology   ║")
    logger.info("╚══════════════════════════════════════════════════════╝")

    t_start = time.time()

    # 1. Generate data
    raw_df = stage_generate_data(args)

    # 2. Data quality pipeline
    clean_df, quality_reports, selected_features = \
        stage_data_quality(raw_df, args)

    # Use selected features (fallback to canonical if empty)
    feature_cols = (selected_features
                    if selected_features else CANONICAL_FEATURES)
    feature_cols = [c for c in feature_cols if c in clean_df.columns]
    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")

    # Normalize features
    normalizer = FeatureNormalizer()
    clean_df = normalizer.fit_transform(clean_df, feature_cols +
                                        [TARGET_EFFORT])

    # 3. Split
    train_df, val_df, test_df = temporal_split(clean_df)

    # 4. Train baselines
    baseline_results = stage_train_baselines(
        train_df, val_df, test_df, feature_cols, args)

    # 5. Train deep models
    deep_results = stage_train_deep_models(
        train_df, val_df, test_df, feature_cols, args)

    # 6. Visualize
    stage_visualize(baseline_results, deep_results,
                    quality_reports, args)

    # 7. Save final results
    final_results = {}
    for name, r in {**baseline_results, **deep_results}.items():
        final_results[name] = r["metrics"]

    results_path = os.path.join(OUTPUT_DIR, "final_results.json")
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)

    elapsed = time.time() - t_start
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    logger.info(f"Results: {results_path}")
    logger.info(f"Plots:   {PLOTS_DIR}")
    logger.info(f"{'=' * 70}")

    # Print summary
    logger.info("\n  MODEL COMPARISON SUMMARY")
    logger.info(f"  {'Model':<20s} {'MAE':>10s} {'RMSE':>10s} "
                f"{'MAPE':>10s} {'R²':>10s}")
    logger.info("  " + "─" * 62)
    for name, m in final_results.items():
        logger.info(f"  {name:<20s} {m['MAE']:>10.2f} {m['RMSE']:>10.2f} "
                    f"{m['MAPE']:>10.2f} {m['R2']:>10.4f}")


if __name__ == "__main__":
    main()
