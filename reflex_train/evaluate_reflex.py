#!/usr/bin/env python3
"""
Evaluate a trained Reflex model on held-out data.

Usage:
    python evaluate_reflex.py --checkpoint checkpoints/reflex_epoch_9.pth --data-dir dataset

Outputs:
    - Console summary of metrics
    - JSON file with full results (if --output specified)
    - HTML report with interactive charts (if --html specified)
"""

import json
import os
from pathlib import Path

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch.utils.data import DataLoader, Subset

from reflex_train.data.dataset import MultiStreamDataset
from reflex_train.data.keys import NUM_KEYS
from reflex_train.models.reflex_net import ReflexNet
from reflex_train.training.evaluate import (
    evaluate_model,
    results_to_dict,
)
from reflex_train.training.report import generate_html_report


class EvalConfig(BaseSettings):
    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_kebab_case=True,
        env_prefix="REFLEX_EVAL_",
    )

    checkpoint: str
    data_dir: str
    output: str = ""
    html: str = ""
    batch_size: int = 32
    workers: int = 4
    context_frames: int = 3
    threshold: float = 0.5
    test_split: float = 0.1
    seed: int = 1337


def find_run_dirs(data_dir: str) -> list[str]:
    if not os.path.exists(data_dir):
        return []
    return [
        os.path.join(data_dir, d)
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]


def load_config_from_checkpoint(checkpoint_path: str) -> dict:
    """Try to load config.json from checkpoint directory."""
    checkpoint_dir = Path(checkpoint_path).parent
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def main():
    cfg = EvalConfig()

    # Try to load training config
    train_cfg = load_config_from_checkpoint(cfg.checkpoint)
    context_frames = train_cfg.get("context_frames", cfg.context_frames)
    inv_dyn_enabled = train_cfg.get("inv_dyn_enabled", True)

    print(f"Loading checkpoint: {cfg.checkpoint}")
    print(f"Data directory: {cfg.data_dir}")
    print(f"Context frames: {context_frames}")
    print(f"Inverse dynamics: {inv_dyn_enabled}")
    print()

    # Load dataset
    run_dirs = find_run_dirs(cfg.data_dir)
    if not run_dirs:
        print(f"No run directories found in {cfg.data_dir}")
        return

    print(f"Found {len(run_dirs)} sessions")

    dataset = MultiStreamDataset(
        run_dirs=run_dirs,
        context_frames=context_frames,
        transform=None,
        action_horizon=train_cfg.get("action_horizon", 2),
    )

    # Deterministic test split: last N% of samples
    total = len(dataset)
    test_size = int(total * cfg.test_split)
    test_indices = list(range(total - test_size, total))
    test_dataset = Subset(dataset, test_indices)

    print(f"Total samples: {total:,}")
    print(f"Test samples: {len(test_dataset):,} ({cfg.test_split*100:.0f}%)")
    print()

    # DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
        persistent_workers=cfg.workers > 0,
    )

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ReflexNet(
        context_frames=context_frames,
        num_keys=NUM_KEYS,
        inv_dynamics=inv_dyn_enabled,
    ).to(device)

    state_dict = torch.load(cfg.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    print("Model loaded successfully")
    print()

    # Evaluate
    results = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        threshold=cfg.threshold,
        verbose=True,
    )

    # Add config to results
    results.config = {
        "checkpoint": cfg.checkpoint,
        "data_dir": cfg.data_dir,
        "threshold": cfg.threshold,
        "test_split": cfg.test_split,
        "context_frames": context_frames,
        "inv_dyn_enabled": inv_dyn_enabled,
    }

    # Print summary
    print()
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print()

    print("Keys Head (Multi-Label):")
    print(f"  F1 Micro:     {results.keys.f1_micro:.4f}")
    print(f"  F1 Macro:     {results.keys.f1_macro:.4f}")
    print(f"  Precision:    {results.keys.precision_micro:.4f}")
    print(f"  Recall:       {results.keys.recall_micro:.4f}")
    print()

    print("Value Head (Regression):")
    print(f"  MSE:          {results.value.mse:.6f}")
    print(f"  MAE:          {results.value.mae:.6f}")
    print(f"  Pearson r:    {results.value.pearson_r:.4f}")
    print(f"  Spearman rho: {results.value.spearman_r:.4f}")
    print()

    if results.inv_dyn is not None:
        print("Inverse Dynamics Head:")
        print(f"  F1 Micro:     {results.inv_dyn.f1_micro:.4f}")
        print(f"  F1 Macro:     {results.inv_dyn.f1_macro:.4f}")
        print(f"  Exact Match:  {results.inv_dyn.exact_match_accuracy:.4f}")
        print()

    # Top/bottom keys
    per_key = results.keys.per_key
    if per_key:
        sorted_keys = sorted(per_key.items(), key=lambda x: x[1]["f1"], reverse=True)
        print("Top 5 Keys by F1:")
        for k, v in sorted_keys[:5]:
            print(f"  {k:15s}: F1={v['f1']:.3f}, support={v['support']}")
        print()

        print("Bottom 5 Keys by F1 (with support > 10):")
        bottom = [kv for kv in sorted_keys if kv[1]["support"] > 10][-5:]
        for k, v in bottom:
            print(f"  {k:15s}: F1={v['f1']:.3f}, support={v['support']}")
        print()

    # Save outputs
    results_dict = results_to_dict(results)

    if cfg.output:
        Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.output, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2)
        print(f"Results saved to: {cfg.output}")

    if cfg.html:
        Path(cfg.html).parent.mkdir(parents=True, exist_ok=True)
        generate_html_report(results_dict, cfg.html, cfg.checkpoint)
        print(f"HTML report saved to: {cfg.html}")


if __name__ == "__main__":
    main()
