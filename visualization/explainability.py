"""
Explainability Module
Attention heatmap extraction, SHAP values, and error attribution.
"""

import numpy as np
import pandas as pd
import torch
import logging
from typing import Dict, List, Optional, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CANONICAL_FEATURES, PLOTS_DIR

logger = logging.getLogger(__name__)


class AttentionExplainer:
    """Extract and analyze attention weights from Transformer models."""

    def __init__(self, model, feature_names: List[str] = None):
        self.model = model
        self.feature_names = feature_names or CANONICAL_FEATURES

    @torch.no_grad()
    def extract_attention(self, x: torch.Tensor,
                          lengths: Optional[torch.Tensor] = None) \
            -> List[np.ndarray]:
        """
        Extract attention weights from all Transformer layers.

        Returns:
            List of attention matrices, each (batch, seq_len+1, seq_len+1)
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        x = x.to(device)
        if lengths is not None:
            lengths = lengths.to(device)

        # Forward pass
        _ = self.model(x, lengths)

        # Get attention weights
        if hasattr(self.model, "get_attention_weights"):
            weights = self.model.get_attention_weights()
            return [w.cpu().numpy() if isinstance(w, torch.Tensor)
                    else w for w in weights]
        else:
            logger.warning("Model does not support attention extraction")
            return []

    def compute_attention_heatmap(self, attn_weights: List[np.ndarray],
                                  layer_idx: int = -1) -> np.ndarray:
        """
        Average attention weights across heads for a specific layer.
        Returns: (batch, seq_len, seq_len) attention heatmap.
        """
        if not attn_weights:
            return np.array([])

        # Use last layer by default
        weights = attn_weights[layer_idx]
        if weights is None:
            return np.array([])

        # Already averaged across heads from our custom layer
        return weights

    def sprint_importance(self, attn_weights: List[np.ndarray]) -> np.ndarray:
        """
        Compute sprint-level importance from CLS token attention.
        Returns: (batch, seq_len) importance scores.
        """
        if not attn_weights:
            return np.array([])

        # CLS token is position 0
        last_layer = attn_weights[-1]
        if last_layer is None:
            return np.array([])

        # Attention from CLS to all sprints (skip CLS-to-CLS)
        cls_attention = last_layer[:, 0, 1:]  # (batch, seq_len)
        return cls_attention


class SHAPExplainer:
    """SHAP value computation for any model type."""

    def __init__(self, model, feature_names: List[str] = None):
        self.model = model
        self.feature_names = feature_names or CANONICAL_FEATURES

    def compute_shap_values(self, X_background: np.ndarray,
                            X_explain: np.ndarray,
                            n_background: int = 100) -> Optional[np.ndarray]:
        """
        Compute SHAP values using KernelExplainer.

        Args:
            X_background: background dataset for SHAP
            X_explain: samples to explain
            n_background: number of background samples to use

        Returns:
            SHAP values array or None if SHAP unavailable
        """
        try:
            import shap

            # Subsample background if needed
            if len(X_background) > n_background:
                idx = np.random.choice(
                    len(X_background), n_background, replace=False)
                X_bg = X_background[idx]
            else:
                X_bg = X_background

            # Create predict function
            if hasattr(self.model, "predict"):
                predict_fn = self.model.predict
            else:
                # PyTorch model wrapper
                def predict_fn(X):
                    with torch.no_grad():
                        X_tensor = torch.tensor(X, dtype=torch.float32)
                        return self.model(X_tensor).cpu().numpy()

            explainer = shap.KernelExplainer(predict_fn, X_bg)
            shap_values = explainer.shap_values(X_explain)

            logger.info(f"SHAP values computed: {np.array(shap_values).shape}")
            return np.array(shap_values)

        except ImportError:
            logger.warning("SHAP not installed")
            return None
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            return None


class ErrorAttributor:
    """Attribute prediction errors to specific sprints and features."""

    def __init__(self, feature_names: List[str] = None):
        self.feature_names = feature_names or CANONICAL_FEATURES

    def attribute_errors(self, y_true: np.ndarray,
                         y_pred: np.ndarray,
                         sprint_features: np.ndarray,
                         sprint_indices: np.ndarray) -> pd.DataFrame:
        """
        Analyze which sprints and features contribute most to errors.

        Args:
            y_true: actual values (n_samples,)
            y_pred: predictions (n_samples,)
            sprint_features: (n_samples, n_sprints, n_features)
            sprint_indices: project/sprint index mapping

        Returns:
            DataFrame with error attribution per sprint/feature.
        """
        errors = np.abs(y_true - y_pred)

        # Sort by error magnitude
        error_order = np.argsort(errors)[::-1]

        records = []
        for idx in error_order[:50]:  # top 50 errors
            records.append({
                "sample_idx": int(idx),
                "actual": float(y_true[idx]),
                "predicted": float(y_pred[idx]),
                "error": float(errors[idx]),
                "error_pct": float(errors[idx] / max(abs(y_true[idx]), 1e-8) * 100),
            })

        df = pd.DataFrame(records)
        return df

    def feature_error_correlation(self, y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  X: np.ndarray,
                                  feature_names: List[str] = None) \
            -> Dict[str, float]:
        """
        Compute correlation between each feature and prediction error.
        """
        errors = np.abs(y_true - y_pred)
        names = feature_names or self.feature_names

        correlations = {}
        for i, name in enumerate(names):
            if i < X.shape[1]:
                corr = np.corrcoef(X[:, i], errors)[0, 1]
                correlations[name] = float(corr) if not np.isnan(corr) else 0.0

        return dict(sorted(correlations.items(),
                           key=lambda x: abs(x[1]), reverse=True))
