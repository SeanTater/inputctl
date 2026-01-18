import json
import os
import random
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from reflex_train.data.dataset import MultiStreamDataset, StreamingDataset
from reflex_train.data.keys import NUM_KEYS
from reflex_train.data.intent import INTENTS
from reflex_train.models.reflex_net import ReflexNet
from .metrics import RunningMetrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_run_dirs(data_dir):
    if not os.path.exists(data_dir):
        return []
    return [
        os.path.join(data_dir, d)
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]


def has_intent_logs(run_dirs):
    return any(os.path.exists(os.path.join(d, "intent.jsonl")) for d in run_dirs)


def write_config_snapshot(cfg, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    cfg_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(cfg_path):
        return
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg.model_dump(), f, indent=2, sort_keys=True)


def train(cfg):
    set_seed(cfg.seed)

    run_dirs = find_run_dirs(cfg.data_dir)
    if not run_dirs:
        print("No run directories found.")
        return

    if (
        cfg.goal_intent == "INFER"
        and cfg.require_intent_labels
        and not has_intent_logs(run_dirs)
    ):
        raise RuntimeError(
            "No intent.jsonl found; run the weak-label precompute step first."
        )

    dataset = MultiStreamDataset(
        run_dirs=run_dirs,
        context_frames=cfg.context_frames,
        transform=None,
        goal_intent=None if cfg.goal_intent == "INFER" else cfg.goal_intent,
        action_horizon=cfg.action_horizon,
        intent_labeler=None,
    )

    val_size = int(len(dataset) * cfg.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    common_loader_kwargs = {
        "num_workers": cfg.workers,
        "pin_memory": True,
        "persistent_workers": cfg.workers > 0,
    }
    if cfg.workers > 0:
        common_loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        StreamingDataset(train_dataset, seed=cfg.seed),
        batch_size=cfg.batch_size,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        **common_loader_kwargs,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ReflexNet(
        context_frames=cfg.context_frames,
        goal_dim=len(INTENTS),
        num_keys=NUM_KEYS,
    ).to(device)

    if cfg.compile_model:
        model = torch.compile(model, mode="reduce-overhead")

    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)

    criterion_keys = nn.BCEWithLogitsLoss()
    criterion_mouse = nn.MSELoss()
    criterion_intent = nn.CrossEntropyLoss()

    write_config_snapshot(cfg, cfg.checkpoint_dir)

    for epoch in range(cfg.epochs):
        model.train()
        train_metrics = RunningMetrics()
        start_time = time.time()

        step_times = []
        total_steps = math.ceil(train_size / cfg.batch_size) if cfg.batch_size else None
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
            unit="batch",
            total=total_steps,
            leave=False,
            smoothing=0.0,
        )

        for batch_idx, batch in enumerate(progress):
            step_start = time.time()
            pixels = batch["pixels"].to(device, non_blocking=True)
            goals = batch["goal"].to(device, non_blocking=True)
            target_keys = batch["label_keys"].to(device, non_blocking=True)
            target_mouse = batch["label_mouse"].to(device, non_blocking=True)
            target_intent = batch["label_intent"].to(device, non_blocking=True)

            keys_logit, mouse_out, _, intent_logits = model(pixels, goals)

            loss_k = criterion_keys(keys_logit, target_keys)
            loss_m = criterion_mouse(mouse_out, target_mouse)
            loss_i = criterion_intent(intent_logits, target_intent)

            loss = loss_k + loss_m + cfg.intent_weight * loss_i

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_metrics.update_loss(loss.item())
            train_metrics.update_keys(
                keys_logit.detach(), target_keys, cfg.key_threshold
            )
            train_metrics.update_intent(intent_logits.detach(), target_intent)

            step_times.append(time.time() - step_start)

            if batch_idx % cfg.log_interval == 0:
                summary = train_metrics.summary()
                avg_step = sum(step_times) / len(step_times)
                progress.set_postfix(
                    {
                        "loss": f"{summary['loss']:.4f}",
                        "k_f1": f"{summary['key_f1']:.3f}",
                        "i_acc": f"{summary['intent_acc']:.3f}",
                        "step_s": f"{avg_step:.2f}",
                    }
                )

        val_summary = validate(
            model,
            val_loader,
            device,
            criterion_keys,
            criterion_mouse,
            criterion_intent,
            cfg,
        )

        elapsed = time.time() - start_time
        avg_step = sum(step_times) / len(step_times) if step_times else 0.0
        print(
            f"Epoch {epoch} Done. "
            f"Train Loss: {train_metrics.summary()['loss']:.4f}, "
            f"Val Loss: {val_summary['loss']:.4f}, "
            f"Val K-F1: {val_summary['key_f1']:.3f}, "
            f"Val I-Acc: {val_summary['intent_acc']:.3f}. "
            f"Step: {avg_step:.2f}s. "
            f"Time: {elapsed:.1f}s"
        )

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(cfg.checkpoint_dir, f"reflex_epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)


def validate(model, loader, device, crit_k, crit_m, crit_i, cfg):
    model.eval()
    metrics = RunningMetrics()
    with torch.no_grad():
        for batch in loader:
            pixels = batch["pixels"].to(device, non_blocking=True)
            goals = batch["goal"].to(device, non_blocking=True)
            target_keys = batch["label_keys"].to(device, non_blocking=True)
            target_mouse = batch["label_mouse"].to(device, non_blocking=True)
            target_intent = batch["label_intent"].to(device, non_blocking=True)

            k, m, _, i = model(pixels, goals)

            loss = (
                crit_k(k, target_keys)
                + crit_m(m, target_mouse)
                + cfg.intent_weight * crit_i(i, target_intent)
            )
            metrics.update_loss(loss.item())
            metrics.update_keys(k, target_keys, cfg.key_threshold)
            metrics.update_intent(i, target_intent)

    return metrics.summary()
