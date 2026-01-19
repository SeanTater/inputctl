import os

import polars as pl
from reflex_train.data.logs import load_key_sets, load_frame_logs
from reflex_train.weak_labels import KeyWindowIntentLabeler, SuperTuxIntentLabeler
from .config import LabelingConfig
from .events import EventDetector
from .episodes import segment_episodes, compute_returns


def find_run_dirs(data_dir):
    if not os.path.exists(data_dir):
        return []
    return [
        os.path.join(data_dir, d)
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]


def write_intents(path, frame_indices, frame_timestamps, intents, overwrite=False):
    if os.path.exists(path) and not overwrite:
        return False
    data = {"frame_idx": frame_indices, "intent": intents}
    if frame_timestamps:
        data["timestamp"] = frame_timestamps
    df = pl.DataFrame(data)
    df.write_parquet(path)
    return True


def write_events(path, events, overwrite=False):
    """Write detected events to parquet file."""
    if os.path.exists(path) and not overwrite:
        return False
    df = pl.DataFrame({
        "frame_idx": [e.frame_idx for e in events],
        "event": [e.event for e in events],
        "confidence": [e.confidence for e in events],
    })
    df.write_parquet(path)
    return True


def write_episodes(path, episodes, overwrite=False):
    """Write episode segmentation to parquet file."""
    if os.path.exists(path) and not overwrite:
        return False
    df = pl.DataFrame([ep.to_dict() for ep in episodes])
    df.write_parquet(path)
    return True


def write_returns(path, returns, overwrite=False):
    """Write per-frame returns to parquet file."""
    if os.path.exists(path) and not overwrite:
        return False
    sorted_items = sorted(returns.items())
    df = pl.DataFrame({
        "frame_idx": [k for k, _ in sorted_items],
        "return": [v for _, v in sorted_items],
    })
    df.write_parquet(path)
    return True


def precompute(cfg: LabelingConfig):
    run_dirs = find_run_dirs(cfg.data_dir)
    if not run_dirs:
        print("No run directories found.")
        return

    # Intent labeler
    if cfg.labeler == "supertux":
        labeler = SuperTuxIntentLabeler(
            base_dir=cfg.base_dir,
            intent_horizon=cfg.intent_horizon,
            sprite_scale=cfg.sprite_scale,
            sprite_threshold=cfg.sprite_threshold,
            proximity_px=cfg.sprite_proximity,
            frame_stride=cfg.intent_stride,
        )
    elif cfg.labeler == "keys":
        labeler = KeyWindowIntentLabeler(intent_horizon=cfg.intent_horizon)
    else:
        raise ValueError("labeler must be 'supertux' or 'keys'")

    # Event detector for RL
    event_detector = None
    if cfg.detect_events:
        event_detector = EventDetector(
            base_dir=cfg.base_dir,
            sprite_scale=cfg.sprite_scale,
            death_threshold=cfg.death_threshold,
            win_proximity_px=cfg.win_proximity_px,
            sparkle_threshold=cfg.sparkle_threshold,
            win_min_frames=cfg.win_min_frames,
            frame_stride=cfg.event_stride,
            win_llm_gate=cfg.win_llm_gate,
            win_llm_sample_stride=cfg.win_llm_sample_stride,
            win_llm_prompt=cfg.win_llm_prompt,
            win_llm_timeout_s=cfg.win_llm_timeout_s,
            win_llm_model=cfg.win_llm_model,
            win_llm_url=cfg.win_llm_url,
        )

    for i, run_dir in enumerate(run_dirs):
        print(f"\n[{i + 1}/{len(run_dirs)}] Processing {os.path.basename(run_dir)}")
        frames_log = os.path.join(run_dir, "frames.parquet")
        inputs_log = os.path.join(run_dir, "inputs.parquet")
        video_path = os.path.join(run_dir, "recording.mp4")
        if not os.path.exists(frames_log) or not os.path.exists(video_path):
            continue

        frame_indices, frame_timestamps = load_frame_logs(frames_log)
        _, key_sets = load_key_sets(frames_log, inputs_log)
        if not frame_indices or not key_sets:
            continue

        # Write intent labels
        intents = labeler.label_intents(video_path, frame_indices, key_sets)
        intent_path = os.path.join(run_dir, "intent.parquet")
        wrote = write_intents(
            intent_path,
            frame_indices,
            frame_timestamps,
            intents,
            overwrite=cfg.overwrite,
        )
        status = "wrote" if wrote else "skipped"
        print(f"{status} {intent_path}")

        # Write event/episode/return labels for RL
        if event_detector:
            total_frames = max(frame_indices) + 1 if frame_indices else 0

            # Detect events
            events_path = os.path.join(run_dir, "events.parquet")
            events = event_detector.detect_events(video_path)
            if events is None:
                print(f"skipped {events_path} (video corrupted)")
                continue
            wrote = write_events(events_path, events, overwrite=cfg.overwrite)
            status = "wrote" if wrote else "skipped"
            n_deaths = sum(1 for e in events if e.event == "DEATH")
            n_wins = sum(1 for e in events if e.event == "WIN")
            n_attacks = sum(1 for e in events if e.event == "ATTACK")
            print(f"{status} {events_path} ({n_deaths} deaths, {n_wins} wins, {n_attacks} attacks)")

            # Segment into episodes
            episodes_path = os.path.join(run_dir, "episodes.parquet")
            episodes = segment_episodes(
                events,
                total_frames,
                respawn_gap=cfg.respawn_gap,
                death_reward=cfg.death_reward,
                win_reward=cfg.win_reward,
            )
            wrote = write_episodes(episodes_path, episodes, overwrite=cfg.overwrite)
            status = "wrote" if wrote else "skipped"
            print(f"{status} {episodes_path} ({len(episodes)} episodes)")

            # Compute returns (including attack bonuses)
            returns_path = os.path.join(run_dir, "returns.parquet")
            returns = compute_returns(
                episodes,
                events=events,
                gamma=cfg.gamma,
                survival_bonus=cfg.survival_bonus,
                attack_reward=cfg.attack_reward,
            )
            wrote = write_returns(returns_path, returns, overwrite=cfg.overwrite)
            status = "wrote" if wrote else "skipped"
            print(f"{status} {returns_path}")


def main():
    cfg = LabelingConfig()
    precompute(cfg)


if __name__ == "__main__":
    main()
