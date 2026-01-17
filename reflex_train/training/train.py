import json
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from reflex_train.data.dataset import MultiStreamDataset
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

    if cfg.goal_intent == "INFER" and cfg.require_intent_labels and not has_intent_logs(run_dirs):
        raise RuntimeError("No intent.jsonl found; run the weak-label precompute step first.")

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ReflexNet(
        context_frames=cfg.context_frames,
        goal_dim=len(INTENTS),
        num_keys=NUM_KEYS,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)

    criterion_keys = nn.BCEWithLogitsLoss()
    criterion_mouse = nn.MSELoss()
    criterion_intent = nn.CrossEntropyLoss()

    write_config_snapshot(cfg, cfg.checkpoint_dir)

    for epoch in range(cfg.epochs):
        model.train()
        train_metrics = RunningMetrics()
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            pixels = batch["pixels"].to(device)
            goals = batch["goal"].to(device)
            target_keys = batch["label_keys"].to(device)
            target_mouse = batch["label_mouse"].to(device)
            target_intent = batch["label_intent"].to(device)

            keys_logit, mouse_out, _, intent_logits = model(pixels, goals)

            loss_k = criterion_keys(keys_logit, target_keys)
            loss_m = criterion_mouse(mouse_out, target_mouse)
            loss_i = criterion_intent(intent_logits, target_intent)

            loss = loss_k + loss_m + cfg.intent_weight * loss_i

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_metrics.update_loss(loss.item())
            train_metrics.update_keys(keys_logit.detach(), target_keys, cfg.key_threshold)
            train_metrics.update_intent(intent_logits.detach(), target_intent)

            if batch_idx % cfg.log_interval == 0:
                summary = train_metrics.summary()
                print(
                    f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {summary['loss']:.4f} "
                    f"K-F1: {summary['key_f1']:.3f} "
                    f"I-Acc: {summary['intent_acc']:.3f}"
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
        print(
            f"Epoch {epoch} Done. "
            f"Train Loss: {train_metrics.summary()['loss']:.4f}, "
            f"Val Loss: {val_summary['loss']:.4f}, "
            f"Val K-F1: {val_summary['key_f1']:.3f}, "
            f"Val I-Acc: {val_summary['intent_acc']:.3f}. "
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
            pixels = batch["pixels"].to(device)
            goals = batch["goal"].to(device)
            target_keys = batch["label_keys"].to(device)
            target_mouse = batch["label_mouse"].to(device)
            target_intent = batch["label_intent"].to(device)

            k, m, _, i = model(pixels, goals)

            loss = crit_k(k, target_keys) + crit_m(m, target_mouse) + cfg.intent_weight * crit_i(i, target_intent)
            metrics.update_loss(loss.item())
            metrics.update_keys(k, target_keys, cfg.key_threshold)
            metrics.update_intent(i, target_intent)

    return metrics.summary()
