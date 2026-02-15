# Ultra-Scale Software Cost Estimation in Agile Methodology  
### Using Deep Learning with Data Quality Intelligence

> A research-grade, data-centric deep learning system that predicts Agile software development effort and cost by modeling sprint-level temporal behavior, learning team and process dynamics, and actively improving data quality. Designed for **NVIDIA H200 SXM** (141 GB HBM3e) with BF16 mixed precision.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                     DATA GENERATION LAYER                        │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐     │
│  │ ISBSG/PROMISE │  │  NASA Data   │  │ Synthetic Agile    │     │
│  │   Adapters    │  │   Adapter    │  │ Sprint Simulator   │     │
│  └──────┬───────┘  └──────┬───────┘  └────────┬───────────┘     │
│         └──────────────────┼──────────────────┘                  │
│                  ┌─────────▼──────────┐                          │
│                  │ Schema Harmonizer  │                          │
│                  │ (Canonical Schema) │                          │
│                  └─────────┬──────────┘                          │
├────────────────────────────┼─────────────────────────────────────┤
│                     DATA QUALITY PIPELINE (7 Stages)             │
│  ┌─────────┐ ┌─────────┐ ┌──────────┐ ┌─────────┐ ┌─────────┐  │
│  │ Missing │→│ Outlier │→│  Agile   │→│Temporal │→│Feature  │  │
│  │  Data   │ │Detection│ │Consistency│ │Smoothing│ │Selection│  │
│  └─────────┘ └─────────┘ └──────────┘ └─────────┘ └────┬────┘  │
│                                                    ┌────▼────┐  │
│                     Data Versioning & Audit ◄──────│ Label   │  │
│                                                    │   QA    │  │
│                                                    └─────────┘  │
├──────────────────────────────────────────────────────────────────┤
│                     MODEL ARCHITECTURES                          │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌────────────────┐  │
│  │ XGBoost  │  │ Deep     │  │   Large   │  │  CNN +         │  │
│  │ / RF     │  │ BiLSTM   │  │Transformer│  │  Transformer   │  │
│  │(Baseline)│  │ + Attn   │  │ Encoder   │  │  Hybrid        │  │
│  └──────────┘  └──────────┘  └───────────┘  └────────────────┘  │
├──────────────────────────────────────────────────────────────────┤
│                     H200-AWARE TRAINING                          │
│  BF16 Mixed Precision │ Gradient Accumulation │ Curriculum      │
│  OneCycleLR + Warmup  │ Gradient Clipping     │ Early Stopping  │
├──────────────────────────────────────────────────────────────────┤
│                     EVALUATION & EXPLAINABILITY                  │
│  MAE/RMSE/MAPE/R² │ Attention Heatmaps │ SHAP │ Ablation Study │
└──────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
varshiniiiiiiiiii/
├── config.py                    # Central configuration (H200, models, DQ thresholds)
├── run_pipeline.py              # End-to-end CLI orchestrator
├── requirements.txt             # Python dependencies
│
├── data/
│   ├── schema.py                # Canonical Agile schema + dataset adapters
│   ├── quality.py               # 7-stage data quality pipeline
│   ├── synthetic.py             # Agile sprint simulator
│   └── loaders.py               # DataLoaders, normalization, splitting
│
├── models/
│   ├── common.py                # Positional encoding, regression heads, utilities
│   ├── baseline.py              # XGBoost + Random Forest wrappers
│   ├── lstm.py                  # Deep bidirectional LSTM with attention pooling
│   ├── transformer.py           # Large Transformer encoder with CLS pooling
│   └── hybrid.py                # CNN + Transformer hybrid architecture
│
├── training/
│   ├── trainer.py               # H200Trainer (BF16, grad accum, early stopping)
│   └── curriculum.py            # Curriculum learning scheduler
│
├── evaluation/
│   ├── metrics.py               # MAE, RMSE, MAPE, R² computation
│   └── analysis.py              # Ablation studies, sensitivity analysis
│
├── visualization/
│   ├── explainability.py        # Attention heatmaps, SHAP, error attribution
│   └── plots.py                 # 8 publication-quality plot generators
│
└── outputs/                     # Generated outputs
    ├── plots/                   # Visualization PNGs
    ├── checkpoints/             # Model checkpoints
    ├── data_quality/            # Quality reports
    └── versioned_data/          # SHA-versioned dataset snapshots
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Dry Run (CPU, small data, 3 epochs)
```bash
python run_pipeline.py --mode dry_run --device cpu --n_projects 20 --epochs 3
```

### 3. Full Run (H200 GPU)
```bash
python run_pipeline.py --mode full --device cuda --n_projects 500 --epochs 200
```

### 4. Train a Specific Model
```bash
python run_pipeline.py --model transformer --epochs 100
python run_pipeline.py --model lstm --epochs 100
python run_pipeline.py --model hybrid --epochs 100
```

---

## Data-Centric Pipeline

### 7-Stage Data Quality Process

| Stage | Module | Signal Improvement |
|-------|--------|--------------------|
| 1. Missing Data | Context-aware imputation (interpolation, team-wise median) | Removes data gaps |
| 2. Outlier Detection | IQR + Isolation Forest + temporal anomaly | Down-weights noise |
| 3. Agile Consistency | Velocity bounds, effort-scope correlation | Enforces domain rules |
| 4. Temporal Smoothing | Kalman filtering per project | Removes reporting noise |
| 5. Feature Selection | Mutual information + correlation analysis | Removes redundancy |
| 6. Label QA | Cross-sprint consistency, confidence scoring | Stabilizes targets |
| 7. Versioning | SHA-based snapshots after every stage | Full audit trail |

### Synthetic Data Generator

Simulates realistic Agile sprint dynamics:
- **Team learning curves**: logarithmic productivity ramp-up
- **Velocity drift**: Ornstein-Uhlenbeck mean-reverting process  
- **Scope creep**: random story-point inflation mid-sprint
- **Failure injection**: sprint collapses and defect spikes

---

## Model Architectures

| Model | Type | Key Feature |
|-------|------|-------------|
| **XGBoost / RF** | Baseline | Flattened project-level features (mean, std, min, max, last, trend) |
| **DeepLSTMStack** | Deep | 4-layer bidirectional LSTM with learned attention pooling |
| **SprintTransformer** | Deep | 6-layer Transformer encoder with CLS token, extractable attention |
| **CNNTransformerHybrid** | Deep | 1D CNN (local patterns) → Transformer (global dependencies) |

---

## H200-Aware Training

- **BF16 Mixed Precision** via `torch.amp` for 2× memory efficiency
- **Gradient Accumulation** (4 steps → 2048 effective batch size)
- **OneCycleLR** with cosine annealing and warmup
- **Curriculum Learning**: clean data → augmented → full (noisy/rare)
- **Gradient Clipping** at norm 1.0
- **torch.compile()** for H200 kernel optimization

---

## Evaluation

### Metrics
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (Coefficient of Determination)

### Data Quality Impact Analysis
- Performance comparison before vs. after cleaning
- Ablation studies toggling each quality stage
- Noise sensitivity analysis

---

## Explainability

- **Attention Heatmaps**: sprint-level importance visualization
- **SHAP Values**: feature-level attribution for any model
- **Error Attribution**: identify which sprints/features cause largest errors

---

## Outputs

After a full run, the `outputs/` directory contains:

| Output | Description |
|--------|-------------|
| `final_results.json` | All model metrics |
| `plots/actual_vs_predicted_*.png` | Prediction scatter plots |
| `plots/training_curves.png` | Loss and LR curves |
| `plots/model_comparison.png` | Side-by-side metric comparison |
| `plots/feature_importance.png` | Feature ranking |
| `plots/data_quality_dashboard.png` | Quality pipeline summary |
| `data_quality/full_quality_report.json` | Detailed quality audit |
| `versioned_data/` | SHA-versioned dataset snapshots |
| `checkpoints/` | Best and final model weights |

---

## License

Research use only. See project requirements for details.
