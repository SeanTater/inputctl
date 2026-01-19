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
from torchvision import transforms
from tqdm import tqdm
from reflex_train.data.dataset import MultiStreamDataset, StreamingDataset
from reflex_train.data.keys import NUM_KEYS
from reflex_train.data.intent import INTENTS
from reflex_train.models.reflex_net import ReflexNet


def get_train_transform(image_size: int = 224):
    """Transform for training: resize and normalize."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def expectile_loss(pred, target, expectile):
    diff = target - pred
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return (weight * diff.pow(2)).mean()


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

    transform = get_train_transform(image_size=224)

    dataset = MultiStreamDataset(
        run_dirs=run_dirs,
        context_frames=cfg.context_frames,
        transform=transform,
        goal_intent=None if cfg.goal_intent == "INFER" else cfg.goal_intent,
        action_horizon=cfg.action_horizon,
        intent_labeler=None,
    )

    val_size = int(len(dataset) * cfg.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        StreamingDataset(train_dataset, seed=cfg.seed),
        batch_size=cfg.batch_size,
    )
    val_loader = DataLoader(
        StreamingDataset(val_dataset, seed=cfg.seed + 1),
        batch_size=cfg.batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ReflexNet(
        context_frames=cfg.context_frames,
        goal_dim=len(INTENTS),
        num_keys=NUM_KEYS,
        inv_dynamics=True,
    ).to(device)
    model = torch.compile(model, mode="reduce-overhead")

    # SGD with foreach=True for fused optimizer operations
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.learning_rate,
        momentum=cfg.momentum,
        foreach=True,  # Fused multi-tensor operations
    )

    # Loss functions (IQL always uses unreduced for advantage weighting)
    criterion_keys = nn.BCEWithLogitsLoss(reduction="none")
    criterion_intent = nn.CrossEntropyLoss(reduction="none")
    criterion_mouse = nn.MSELoss()
    criterion_inv_dyn = nn.BCEWithLogitsLoss()

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
            if cfg.max_steps_per_epoch and batch_idx >= cfg.max_steps_per_epoch:
                break
            step_start = time.time()
            pixels = batch["pixels"].to(device, non_blocking=True)
            goals = batch["goal"].to(device, non_blocking=True)
            target_keys = batch["label_keys"].to(device, non_blocking=True)
            target_mouse = batch["label_mouse"].to(device, non_blocking=True)
            target_intent = batch["label_intent"].to(device, non_blocking=True)
            target_returns = batch["return"].to(device, non_blocking=True)
            current_keys = batch["current_keys"].to(device, non_blocking=True)
            next_pixels = batch["next_pixels"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                keys_logit, mouse_out, inv_dyn_logits, intent_logits, value = model(
                    pixels, goals, next_pixels
                )

                # IQL advantage weighting (always enabled)
                loss_v = expectile_loss(value, target_returns, cfg.iql_expectile)
                with torch.no_grad():
                    advantage = target_returns - value
                    weights = torch.exp(advantage / cfg.iql_adv_temperature)
                    weights = weights.clamp(max=cfg.advantage_clip)
                    weights = weights / weights.mean()

                loss_k_per_sample = criterion_keys(
                    keys_logit, target_keys
                ).mean(dim=1)
                loss_k = (loss_k_per_sample * weights).mean()

                loss_i_per_sample = criterion_intent(
                    intent_logits, target_intent
                )
                loss_i = (loss_i_per_sample * weights).mean()
                loss_m = criterion_mouse(mouse_out, target_mouse)

                # Inverse dynamics (always enabled)
                inv_target = (
                    target_keys if cfg.inv_dyn_use_action_horizon else current_keys
                )
                loss_inv = criterion_inv_dyn(inv_dyn_logits, inv_target)

                loss = (
                    loss_k
                    + loss_m
                    + cfg.intent_weight * loss_i
                    + cfg.value_weight * loss_v
                    + cfg.inv_dyn_weight * loss_inv
                )

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
            criterion_inv_dyn,
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


def validate(model, loader, device, crit_k, crit_m, crit_i, crit_inv, cfg):
    model.eval()
    metrics = RunningMetrics()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch_idx, batch in enumerate(loader):
            if cfg.max_steps_per_epoch and batch_idx >= cfg.max_steps_per_epoch:
                break
            pixels = batch["pixels"].to(device, non_blocking=True)
            goals = batch["goal"].to(device, non_blocking=True)
            target_keys = batch["label_keys"].to(device, non_blocking=True)
            target_mouse = batch["label_mouse"].to(device, non_blocking=True)
            target_intent = batch["label_intent"].to(device, non_blocking=True)
            target_returns = batch["return"].to(device, non_blocking=True)
            current_keys = batch["current_keys"].to(device, non_blocking=True)

            k, m, inv_dyn_logits, i, v = model(
                pixels, goals, batch["next_pixels"].to(device, non_blocking=True)
            )

            # IQL advantage weighting (always enabled)
            loss_v = expectile_loss(v, target_returns, cfg.iql_expectile)
            advantage = target_returns - v
            weights = torch.exp(advantage / cfg.iql_adv_temperature)
            weights = weights.clamp(max=cfg.advantage_clip)
            weights = weights / weights.mean()

            loss_k = (crit_k(k, target_keys).mean(dim=1) * weights).mean()
            loss_i = (crit_i(i, target_intent) * weights).mean()
            loss_m = crit_m(m, target_mouse)

            # Inverse dynamics (always enabled)
            inv_target = (
                target_keys if cfg.inv_dyn_use_action_horizon else current_keys
            )
            loss_inv = crit_inv(inv_dyn_logits, inv_target)

            loss = (
                loss_k
                + loss_m
                + cfg.intent_weight * loss_i
                + cfg.value_weight * loss_v
                + cfg.inv_dyn_weight * loss_inv
            )
            metrics.update_loss(loss.item())
            metrics.update_keys(k, target_keys, cfg.key_threshold)
            metrics.update_intent(i, target_intent)

    return metrics.summary()
