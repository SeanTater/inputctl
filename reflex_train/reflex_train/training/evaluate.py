"""Core evaluation logic for reflex model."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.intent import INTENTS
from ..data.keys import IDX_TO_KEY, NUM_KEYS
from ..models.reflex_net import ReflexNet


@dataclass
class KeyMetrics:
    """Per-key and aggregate metrics for multi-label key prediction."""

    f1_micro: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    precision_micro: float = 0.0
    precision_macro: float = 0.0
    recall_micro: float = 0.0
    recall_macro: float = 0.0
    per_key: dict = field(default_factory=dict)
    # Aggregated counts
    total_samples: int = 0
    total_positives: int = 0


@dataclass
class IntentMetrics:
    """Multi-class intent classification metrics."""

    accuracy: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.zeros((7, 7)))
    confusion_matrix_normalized: np.ndarray = field(
        default_factory=lambda: np.zeros((7, 7))
    )
    per_class: dict = field(default_factory=dict)
    total_samples: int = 0


@dataclass
class ValueMetrics:
    """Regression metrics for value head."""

    mse: float = 0.0
    mae: float = 0.0
    pearson_r: float = 0.0
    pearson_p: float = 1.0
    spearman_r: float = 0.0
    spearman_p: float = 1.0
    total_samples: int = 0


@dataclass
class InvDynMetrics:
    """Inverse dynamics head metrics."""

    f1_micro: float = 0.0
    f1_macro: float = 0.0
    exact_match_accuracy: float = 0.0
    total_samples: int = 0


@dataclass
class EvaluationResults:
    """Complete evaluation results."""

    keys: KeyMetrics
    intent: IntentMetrics
    value: ValueMetrics
    inv_dyn: Optional[InvDynMetrics]
    config: dict = field(default_factory=dict)


class EvalAccumulator:
    """Accumulates predictions and targets during evaluation."""

    def __init__(self, has_inv_dyn: bool = False):
        self.has_inv_dyn = has_inv_dyn
        self.reset()

    def reset(self):
        self.key_preds = []
        self.key_targets = []
        self.intent_preds = []
        self.intent_targets = []
        self.value_preds = []
        self.value_targets = []
        self.inv_dyn_preds = []
        self.inv_dyn_targets = []

    def update(
        self,
        key_logits: torch.Tensor,
        key_targets: torch.Tensor,
        intent_logits: torch.Tensor,
        intent_targets: torch.Tensor,
        value_preds: torch.Tensor,
        value_targets: torch.Tensor,
        inv_dyn_logits: Optional[torch.Tensor] = None,
        inv_dyn_targets: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ):
        # Keys: multi-label
        key_probs = torch.sigmoid(key_logits)
        key_binary = (key_probs > threshold).cpu().numpy()
        self.key_preds.append(key_binary)
        self.key_targets.append((key_targets > 0.5).cpu().numpy())

        # Intent: argmax
        intent_pred = torch.argmax(intent_logits, dim=1).cpu().numpy()
        self.intent_preds.append(intent_pred)
        self.intent_targets.append(intent_targets.cpu().numpy())

        # Value
        self.value_preds.append(value_preds.cpu().numpy())
        self.value_targets.append(value_targets.cpu().numpy())

        # Inverse dynamics
        if inv_dyn_logits is not None and self.has_inv_dyn:
            inv_probs = torch.sigmoid(inv_dyn_logits)
            inv_binary = (inv_probs > threshold).cpu().numpy()
            self.inv_dyn_preds.append(inv_binary)
            if inv_dyn_targets is not None:
                self.inv_dyn_targets.append((inv_dyn_targets > 0.5).cpu().numpy())

    def compute_metrics(self) -> EvaluationResults:
        """Compute all metrics from accumulated predictions."""
        keys = self._compute_key_metrics()
        intent = self._compute_intent_metrics()
        value = self._compute_value_metrics()
        inv_dyn = self._compute_inv_dyn_metrics() if self.has_inv_dyn else None

        return EvaluationResults(
            keys=keys, intent=intent, value=value, inv_dyn=inv_dyn
        )

    def _compute_key_metrics(self) -> KeyMetrics:
        y_pred = np.vstack(self.key_preds)
        y_true = np.vstack(self.key_targets)

        metrics = KeyMetrics()
        metrics.total_samples = y_true.shape[0]
        metrics.total_positives = int(y_true.sum())

        # Aggregate metrics
        metrics.f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
        metrics.f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        metrics.f1_weighted = f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        metrics.precision_micro = precision_score(
            y_true, y_pred, average="micro", zero_division=0
        )
        metrics.precision_macro = precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        metrics.recall_micro = recall_score(
            y_true, y_pred, average="micro", zero_division=0
        )
        metrics.recall_macro = recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )

        # Per-key metrics
        for key_idx in range(NUM_KEYS):
            key_name = IDX_TO_KEY.get(key_idx, f"KEY_{key_idx}")
            y_true_k = y_true[:, key_idx]
            y_pred_k = y_pred[:, key_idx]
            support = int(y_true_k.sum())

            if support > 0:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true_k, y_pred_k, average="binary", zero_division=0
                )
                metrics.per_key[key_name] = {
                    "f1": float(f1),
                    "precision": float(precision),
                    "recall": float(recall),
                    "support": support,
                    "support_pct": support / metrics.total_samples * 100,
                }

        return metrics

    def _compute_intent_metrics(self) -> IntentMetrics:
        y_pred = np.concatenate(self.intent_preds)
        y_true = np.concatenate(self.intent_targets)

        metrics = IntentMetrics()
        metrics.total_samples = len(y_true)

        # Accuracy
        metrics.accuracy = float((y_pred == y_true).mean())

        # F1 scores
        metrics.f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        metrics.f1_weighted = f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(INTENTS))))
        metrics.confusion_matrix = cm

        # Row-normalized confusion matrix
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_norm = np.where(row_sums > 0, cm / row_sums, 0)
        metrics.confusion_matrix_normalized = cm_norm

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(INTENTS))), zero_division=0
        )
        for i, intent_name in enumerate(INTENTS):
            metrics.per_class[intent_name] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }

        return metrics

    def _compute_value_metrics(self) -> ValueMetrics:
        y_pred = np.concatenate(self.value_preds)
        y_true = np.concatenate(self.value_targets)

        metrics = ValueMetrics()
        metrics.total_samples = len(y_true)

        # MSE and MAE
        metrics.mse = float(np.mean((y_pred - y_true) ** 2))
        metrics.mae = float(np.mean(np.abs(y_pred - y_true)))

        # Correlation (if variance exists)
        if np.var(y_true) > 1e-10 and np.var(y_pred) > 1e-10:
            r, p = pearsonr(y_pred, y_true)
            metrics.pearson_r = float(r)
            metrics.pearson_p = float(p)

            rho, p_s = spearmanr(y_pred, y_true)
            metrics.spearman_r = float(rho)
            metrics.spearman_p = float(p_s)

        return metrics

    def _compute_inv_dyn_metrics(self) -> Optional[InvDynMetrics]:
        if not self.inv_dyn_preds or not self.inv_dyn_targets:
            return None

        y_pred = np.vstack(self.inv_dyn_preds)
        y_true = np.vstack(self.inv_dyn_targets)

        metrics = InvDynMetrics()
        metrics.total_samples = y_true.shape[0]

        # F1 scores
        metrics.f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
        metrics.f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # Exact match: all keys correct in a sample
        exact_matches = (y_pred == y_true).all(axis=1)
        metrics.exact_match_accuracy = float(exact_matches.mean())

        return metrics


def load_checkpoint(checkpoint_path: str, device: torch.device, cfg: dict) -> ReflexNet:
    """Load a model checkpoint."""
    model = ReflexNet(
        context_frames=cfg.get("context_frames", 3),
        goal_dim=len(INTENTS),
        num_keys=NUM_KEYS,
        inv_dynamics=cfg.get("inv_dyn_enabled", True),
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def evaluate_model(
    model: ReflexNet,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    verbose: bool = True,
) -> EvaluationResults:
    """Run evaluation on a dataloader and compute metrics."""
    has_inv_dyn = model.inv_dynamics_enabled and model.head_inv_dynamics is not None
    accumulator = EvalAccumulator(has_inv_dyn=has_inv_dyn)

    model.eval()
    progress = tqdm(dataloader, desc="Evaluating", disable=not verbose)

    with torch.no_grad():
        for batch in progress:
            pixels = batch["pixels"].to(device, non_blocking=True)
            goals = batch["goal"].to(device, non_blocking=True)
            target_keys = batch["label_keys"].to(device, non_blocking=True)
            target_intent = batch["label_intent"].to(device, non_blocking=True)
            target_returns = batch["return"].to(device, non_blocking=True)
            next_pixels = batch["next_pixels"].to(device, non_blocking=True)
            current_keys = batch["current_keys"].to(device, non_blocking=True)

            keys_logit, _, inv_dyn_logits, intent_logits, value = model(
                pixels, goals, next_pixels
            )

            # Use current_keys as inv_dyn target (keys held during frame transition)
            inv_dyn_target = current_keys if has_inv_dyn else None

            accumulator.update(
                key_logits=keys_logit,
                key_targets=target_keys,
                intent_logits=intent_logits,
                intent_targets=target_intent,
                value_preds=value,
                value_targets=target_returns,
                inv_dyn_logits=inv_dyn_logits,
                inv_dyn_targets=inv_dyn_target,
                threshold=threshold,
            )

    return accumulator.compute_metrics()


def results_to_dict(results: EvaluationResults) -> dict:
    """Convert EvaluationResults to a JSON-serializable dict."""
    out = {
        "keys": {
            "f1_micro": results.keys.f1_micro,
            "f1_macro": results.keys.f1_macro,
            "f1_weighted": results.keys.f1_weighted,
            "precision_micro": results.keys.precision_micro,
            "precision_macro": results.keys.precision_macro,
            "recall_micro": results.keys.recall_micro,
            "recall_macro": results.keys.recall_macro,
            "total_samples": results.keys.total_samples,
            "total_positives": results.keys.total_positives,
            "per_key": results.keys.per_key,
        },
        "intent": {
            "accuracy": results.intent.accuracy,
            "f1_macro": results.intent.f1_macro,
            "f1_weighted": results.intent.f1_weighted,
            "confusion_matrix": results.intent.confusion_matrix.tolist(),
            "confusion_matrix_normalized": results.intent.confusion_matrix_normalized.tolist(),
            "per_class": results.intent.per_class,
            "total_samples": results.intent.total_samples,
            "intents": INTENTS,
        },
        "value": {
            "mse": results.value.mse,
            "mae": results.value.mae,
            "pearson_r": results.value.pearson_r,
            "pearson_p": results.value.pearson_p,
            "spearman_r": results.value.spearman_r,
            "spearman_p": results.value.spearman_p,
            "total_samples": results.value.total_samples,
        },
        "config": results.config,
    }

    if results.inv_dyn is not None:
        out["inv_dyn"] = {
            "f1_micro": results.inv_dyn.f1_micro,
            "f1_macro": results.inv_dyn.f1_macro,
            "exact_match_accuracy": results.inv_dyn.exact_match_accuracy,
            "total_samples": results.inv_dyn.total_samples,
        }

    return out
