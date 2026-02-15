"""
H200-Aware Trainer
BF16 mixed-precision, gradient accumulation, curriculum learning,
early stopping, and checkpointing. Designed for NVIDIA H200 SXM.
"""

import os
import time
import logging
import numpy as np
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import hw, CHECKPOINT_DIR

logger = logging.getLogger(__name__)


class H200Trainer:
    """
    GPU-aware training loop for H200 SXM with:
    - BF16/FP16 mixed precision via torch.amp
    - Large batch training with gradient accumulation
    - Curriculum learning integration
    - OneCycleLR scheduling with warmup
    - Gradient clipping
    - Checkpointing and early stopping
    - Weighted loss using sample quality scores
    """

    def __init__(self, model: nn.Module,
                 config=None,
                 checkpoint_dir: str = CHECKPOINT_DIR):
        self.cfg = config or hw
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Device setup
        self.device = torch.device(
            self.cfg.device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # Optionally compile for H200 performance
        if (self.cfg.compile_model and torch.cuda.is_available()
                and hasattr(torch, "compile")):
            try:
                self.model = torch.compile(self.model)
                logger.info("Model compiled with torch.compile()")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")

        # Mixed precision
        self.use_amp = self.cfg.precision in ("bf16", "fp16")
        self.amp_dtype = (torch.bfloat16 if self.cfg.precision == "bf16"
                          else torch.float16)
        self.scaler = GradScaler("cuda", enabled=(self.cfg.precision == "fp16"))

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        # Loss
        self.criterion = nn.SmoothL1Loss(reduction="none")

        # Training state
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "lr": []}

    def _train_epoch(self, loader: DataLoader,
                     epoch: int) -> float:
        """Train for one epoch with gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        self.optimizer.zero_grad()

        for step, (seqs, targets, weights, lengths) in enumerate(loader):
            seqs = seqs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            weights = weights.to(self.device, non_blocking=True)
            lengths = lengths.to(self.device, non_blocking=True)

            with autocast("cuda", dtype=self.amp_dtype,
                          enabled=self.use_amp):
                preds = self.model(seqs, lengths)
                per_sample_loss = self.criterion(preds, targets)
                # Apply sample weights
                weighted_loss = (per_sample_loss * weights).mean()
                loss = weighted_loss / self.cfg.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.cfg.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += weighted_loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for seqs, targets, weights, lengths in loader:
            seqs = seqs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            weights = weights.to(self.device, non_blocking=True)
            lengths = lengths.to(self.device, non_blocking=True)

            with autocast("cuda", dtype=self.amp_dtype,
                          enabled=self.use_amp):
                preds = self.model(seqs, lengths)
                loss = (self.criterion(preds, targets) * weights).mean()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _save_checkpoint(self, epoch: int, val_loss: float,
                         tag: str = "best"):
        """Save model checkpoint."""
        path = os.path.join(
            self.checkpoint_dir, f"model_{tag}.pt")
        state = {
            "epoch": epoch,
            "model_state_dict": (self.model._orig_mod.state_dict()
                                 if hasattr(self.model, "_orig_mod")
                                 else self.model.state_dict()),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": vars(self.cfg),
        }
        torch.save(state, path)
        logger.info(f"Checkpoint saved: {path} (val_loss={val_loss:.6f})")

    def train(self, train_loader: DataLoader,
              val_loader: DataLoader,
              max_epochs: int = None) -> Dict[str, List[float]]:
        """
        Full training loop with early stopping.

        Returns:
            Training history dict with loss curves.
        """
        max_epochs = max_epochs or self.cfg.max_epochs

        # Learning rate scheduler
        pct_start = min(self.cfg.warmup_epochs / max(max_epochs, 1), 0.3)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.cfg.learning_rate,
            epochs=max_epochs,
            steps_per_epoch=max(len(train_loader), 1),
            pct_start=pct_start,
            anneal_strategy="cos",
        )

        logger.info(f"Training on {self.device} | "
                    f"Precision: {self.cfg.precision} | "
                    f"Batch: {self.cfg.batch_size} × "
                    f"{self.cfg.gradient_accumulation_steps} "
                    f"= {self.cfg.effective_batch_size}")

        for epoch in range(max_epochs):
            t0 = time.time()

            train_loss = self._train_epoch(train_loader, epoch)
            val_loss = self._validate(val_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(current_lr)

            elapsed = time.time() - t0

            # Logging
            if epoch % 5 == 0 or epoch == max_epochs - 1:
                logger.info(
                    f"Epoch {epoch:4d}/{max_epochs} | "
                    f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                    f"LR: {current_lr:.2e} | {elapsed:.1f}s")

            # Step scheduler
            try:
                scheduler.step()
            except Exception:
                pass

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_loss, "best")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.cfg.patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(patience={self.cfg.patience})")
                    break

        # Save final checkpoint
        self._save_checkpoint(epoch, val_loss, "final")
        return self.history

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions.

        Returns:
            (predictions, actuals) as numpy arrays.
        """
        self.model.eval()
        all_preds, all_targets = [], []

        for seqs, targets, weights, lengths in loader:
            seqs = seqs.to(self.device, non_blocking=True)
            lengths = lengths.to(self.device, non_blocking=True)

            with autocast("cuda", dtype=self.amp_dtype,
                          enabled=self.use_amp):
                preds = self.model(seqs, lengths)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())

        return (np.concatenate(all_preds),
                np.concatenate(all_targets))
