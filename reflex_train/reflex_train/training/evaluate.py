"""Core evaluation logic for reflex model."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

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
class TemporalMetrics:
    """Temporal coherence metrics measuring prediction stability."""

    key_chatter_rate: float = 0.0  # Flip-flops per frame pair
    key_chatter_per_key: dict = field(default_factory=dict)  # {key_name: rate}


@dataclass
class CalibrationMetrics:
    """Calibration metrics checking if probabilities match frequencies."""

    ece: float = 0.0  # Expected Calibration Error
    bins: list = field(default_factory=list)  # [(bin_center, pred_prob, actual_freq, count)]
    per_key_ece: dict = field(default_factory=dict)  # {key_name: ece} for top keys


@dataclass
class ConditionalMetrics:
    """Conditional performance metrics stratified by various factors."""

    key_f1_by_rarity: dict = field(default_factory=dict)  # {bucket: f1}
    f1_near_episode_boundary: float = 0.0
    f1_away_from_boundary: float = 0.0


@dataclass
class EvaluationResults:
    """Complete evaluation results."""

    keys: KeyMetrics
    value: ValueMetrics
    inv_dyn: Optional[InvDynMetrics]
    temporal: Optional[TemporalMetrics] = None
    calibration: Optional[CalibrationMetrics] = None
    conditional: Optional[ConditionalMetrics] = None
    config: dict = field(default_factory=dict)


class EvalAccumulator:
    """Accumulates predictions and targets during evaluation."""

    def __init__(self, has_inv_dyn: bool = False):
        self.has_inv_dyn = has_inv_dyn
        self.reset()

    def reset(self):
        self.key_preds = []
        self.key_targets = []
        self.key_probs = []  # Raw probabilities for calibration
        self.value_preds = []
        self.value_targets = []
        self.inv_dyn_preds = []
        self.inv_dyn_targets = []
        # For temporal/conditional metrics
        self.session_indices = []
        self.sample_indices = []
        self.done_flags = []

    def update(
        self,
        key_logits: torch.Tensor,
        key_targets: torch.Tensor,
        value_preds: torch.Tensor,
        value_targets: torch.Tensor,
        inv_dyn_logits: Optional[torch.Tensor] = None,
        inv_dyn_targets: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        session_idx: Optional[torch.Tensor] = None,
        sample_idx: Optional[torch.Tensor] = None,
        done: Optional[torch.Tensor] = None,
    ):
        # Keys: multi-label
        key_probs_tensor = torch.sigmoid(key_logits)
        key_binary = (key_probs_tensor > threshold).cpu().numpy()
        self.key_preds.append(key_binary)
        self.key_targets.append((key_targets > 0.5).cpu().numpy())
        self.key_probs.append(key_probs_tensor.cpu().numpy())

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

        # Temporal/conditional metadata
        if session_idx is not None:
            self.session_indices.append(session_idx.cpu().numpy())
        if sample_idx is not None:
            self.sample_indices.append(sample_idx.cpu().numpy())
        if done is not None:
            self.done_flags.append(done.cpu().numpy())

    def compute_metrics(self) -> EvaluationResults:
        """Compute all metrics from accumulated predictions."""
        keys = self._compute_key_metrics()
        value = self._compute_value_metrics()
        inv_dyn = self._compute_inv_dyn_metrics() if self.has_inv_dyn else None
        temporal = self._compute_temporal_metrics()
        calibration = self._compute_calibration_metrics(keys.per_key)
        conditional = self._compute_conditional_metrics(keys.per_key)

        return EvaluationResults(
            keys=keys,
            value=value,
            inv_dyn=inv_dyn,
            temporal=temporal,
            calibration=calibration,
            conditional=conditional,
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

    def _compute_temporal_metrics(self) -> Optional[TemporalMetrics]:
        """Compute temporal coherence metrics (chatter, stability)."""
        if not self.session_indices or not self.sample_indices:
            return None

        session_arr = np.concatenate(self.session_indices)
        sample_arr = np.concatenate(self.sample_indices)
        key_preds = np.vstack(self.key_preds)
        key_targets = np.vstack(self.key_targets)

        # Sort by (session, sample) to ensure temporal ordering
        order = np.lexsort((sample_arr, session_arr))
        session_arr = session_arr[order]
        sample_arr = sample_arr[order]
        key_preds = key_preds[order]
        key_targets = key_targets[order]

        metrics = TemporalMetrics()
        num_keys = key_preds.shape[1]

        # Compute chatter: flips between adjacent frames within same session
        total_pairs = 0
        total_flips = 0
        per_key_flips = np.zeros(num_keys)
        per_key_pairs = np.zeros(num_keys)

        for i in range(1, len(session_arr)):
            if session_arr[i] == session_arr[i - 1]:
                # Same session, consecutive frames
                total_pairs += 1
                flips = key_preds[i] != key_preds[i - 1]
                total_flips += flips.sum()
                per_key_flips += flips.astype(int)
                per_key_pairs += 1

        if total_pairs > 0:
            metrics.key_chatter_rate = float(total_flips / (total_pairs * num_keys))

            for key_idx in range(num_keys):
                key_name = IDX_TO_KEY.get(key_idx, f"KEY_{key_idx}")
                if per_key_pairs[key_idx] > 0:
                    metrics.key_chatter_per_key[key_name] = float(
                        per_key_flips[key_idx] / per_key_pairs[key_idx]
                    )

        return metrics

    def _compute_calibration_metrics(
        self, per_key_support: dict
    ) -> Optional[CalibrationMetrics]:
        """Compute calibration metrics (ECE, reliability bins)."""
        if not self.key_probs:
            return None

        probs = np.vstack(self.key_probs)
        targets = np.vstack(self.key_targets)

        metrics = CalibrationMetrics()
        num_bins = 10
        bin_boundaries = np.linspace(0, 1, num_bins + 1)

        # Aggregate calibration across all keys
        all_probs = probs.flatten()
        all_targets = targets.flatten()

        bin_data = []
        weighted_error = 0.0
        total_samples = len(all_probs)

        for i in range(num_bins):
            low, high = bin_boundaries[i], bin_boundaries[i + 1]
            if i == num_bins - 1:
                mask = (all_probs >= low) & (all_probs <= high)
            else:
                mask = (all_probs >= low) & (all_probs < high)

            count = mask.sum()
            if count > 0:
                bin_probs = all_probs[mask]
                bin_targets = all_targets[mask]
                mean_pred = float(np.mean(bin_probs))
                actual_freq = float(np.mean(bin_targets))
                bin_center = (low + high) / 2
                bin_data.append((bin_center, mean_pred, actual_freq, int(count)))
                weighted_error += count * abs(mean_pred - actual_freq)

        metrics.bins = bin_data
        metrics.ece = weighted_error / total_samples if total_samples > 0 else 0.0

        # Per-key ECE for top 10 keys by support
        top_keys = sorted(
            per_key_support.items(), key=lambda x: x[1].get("support", 0), reverse=True
        )[:10]
        for key_name, key_info in top_keys:
            key_idx = None
            for idx, name in IDX_TO_KEY.items():
                if name == key_name:
                    key_idx = idx
                    break
            if key_idx is None:
                continue

            key_probs_col = probs[:, key_idx]
            key_targets_col = targets[:, key_idx]

            key_ece = 0.0
            key_total = len(key_probs_col)
            for j in range(num_bins):
                low, high = bin_boundaries[j], bin_boundaries[j + 1]
                if j == num_bins - 1:
                    mask = (key_probs_col >= low) & (key_probs_col <= high)
                else:
                    mask = (key_probs_col >= low) & (key_probs_col < high)
                count = mask.sum()
                if count > 0:
                    mean_pred = np.mean(key_probs_col[mask])
                    actual_freq = np.mean(key_targets_col[mask])
                    key_ece += count * abs(mean_pred - actual_freq)
            metrics.per_key_ece[key_name] = float(key_ece / key_total) if key_total else 0.0

        return metrics

    def _compute_conditional_metrics(
        self, per_key_support: dict
    ) -> Optional[ConditionalMetrics]:
        """Compute conditional performance metrics."""
        key_preds = np.vstack(self.key_preds)
        key_targets = np.vstack(self.key_targets)

        metrics = ConditionalMetrics()

        # Key F1 by rarity buckets
        # Categorize keys: rare (<1%), medium (1-10%), common (>10%)
        total_samples = key_targets.shape[0]
        rare_mask = np.zeros(key_targets.shape[1], dtype=bool)
        medium_mask = np.zeros(key_targets.shape[1], dtype=bool)
        common_mask = np.zeros(key_targets.shape[1], dtype=bool)

        for key_name, info in per_key_support.items():
            pct = info.get("support_pct", 0)
            key_idx = None
            for idx, name in IDX_TO_KEY.items():
                if name == key_name:
                    key_idx = idx
                    break
            if key_idx is None:
                continue
            if pct < 1:
                rare_mask[key_idx] = True
            elif pct < 10:
                medium_mask[key_idx] = True
            else:
                common_mask[key_idx] = True

        for bucket_name, bucket_mask in [
            ("rare (<1%)", rare_mask),
            ("medium (1-10%)", medium_mask),
            ("common (>10%)", common_mask),
        ]:
            if bucket_mask.sum() == 0:
                continue
            bucket_preds = key_preds[:, bucket_mask]
            bucket_targets = key_targets[:, bucket_mask]
            f1 = f1_score(bucket_targets, bucket_preds, average="micro", zero_division=0)
            metrics.key_f1_by_rarity[bucket_name] = float(f1)

        # Episode boundary performance
        if self.done_flags and self.session_indices and self.sample_indices:
            done_arr = np.concatenate(self.done_flags)
            session_arr = np.concatenate(self.session_indices)
            sample_arr = np.concatenate(self.sample_indices)

            # Sort for temporal ordering
            order = np.lexsort((sample_arr, session_arr))
            done_arr = done_arr[order]
            key_preds_ordered = key_preds[order]
            key_targets_ordered = key_targets[order]

            # Find frames within ±5 of episode boundaries
            boundary_frames = set()
            for i, d in enumerate(done_arr):
                if d > 0.5:
                    for offset in range(-5, 6):
                        boundary_frames.add(i + offset)

            near_mask = np.array([i in boundary_frames for i in range(len(done_arr))])
            near_mask = near_mask & (np.arange(len(done_arr)) < len(done_arr))

            if near_mask.sum() > 10:
                metrics.f1_near_episode_boundary = float(
                    f1_score(
                        key_targets_ordered[near_mask],
                        key_preds_ordered[near_mask],
                        average="micro",
                        zero_division=0,
                    )
                )
            away_mask = ~near_mask
            if away_mask.sum() > 10:
                metrics.f1_away_from_boundary = float(
                    f1_score(
                        key_targets_ordered[away_mask],
                        key_preds_ordered[away_mask],
                        average="micro",
                        zero_division=0,
                    )
                )

        return metrics


def load_checkpoint(checkpoint_path: str, device: torch.device, cfg: dict) -> ReflexNet:
    """Load a model checkpoint."""
    model = ReflexNet(
        context_frames=cfg.get("context_frames", 3),
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
            target_keys = batch["label_keys"].to(device, non_blocking=True)
            target_returns = batch["return"].to(device, non_blocking=True)
            next_pixels = batch["next_pixels"].to(device, non_blocking=True)
            current_keys = batch["current_keys"].to(device, non_blocking=True)

            # Optional metadata for temporal/conditional metrics
            session_idx = batch.get("session_idx")
            sample_idx = batch.get("sample_idx")
            done = batch.get("done")

            keys_logit, _, inv_dyn_logits, value = model(pixels, next_pixels)

            # Use current_keys as inv_dyn target (keys held during frame transition)
            inv_dyn_target = current_keys if has_inv_dyn else None

            accumulator.update(
                key_logits=keys_logit,
                key_targets=target_keys,
                value_preds=value,
                value_targets=target_returns,
                inv_dyn_logits=inv_dyn_logits,
                inv_dyn_targets=inv_dyn_target,
                threshold=threshold,
                session_idx=session_idx,
                sample_idx=sample_idx,
                done=done,
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

    if results.temporal is not None:
        out["temporal"] = {
            "key_chatter_rate": results.temporal.key_chatter_rate,
            "key_chatter_per_key": results.temporal.key_chatter_per_key,
        }

    if results.calibration is not None:
        out["calibration"] = {
            "ece": results.calibration.ece,
            "bins": [
                {
                    "bin_center": b[0],
                    "mean_pred": b[1],
                    "actual_freq": b[2],
                    "count": b[3],
                }
                for b in results.calibration.bins
            ],
            "per_key_ece": results.calibration.per_key_ece,
        }

    if results.conditional is not None:
        out["conditional"] = {
            "key_f1_by_rarity": results.conditional.key_f1_by_rarity,
            "f1_near_episode_boundary": results.conditional.f1_near_episode_boundary,
            "f1_away_from_boundary": results.conditional.f1_away_from_boundary,
        }

    return out
