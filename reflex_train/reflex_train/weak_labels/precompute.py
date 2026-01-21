import os

import polars as pl
from reflex_train.data.logs import load_frame_logs
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

    # Event detector for RL
    event_detector = None
    if cfg.detect_events:
        event_detector = EventDetector(
            base_dir=cfg.base_dir,
            death_threshold=cfg.death_threshold,
            attack_threshold=cfg.attack_threshold,
            win_key=cfg.win_key,
            win_key_min_presses=cfg.win_key_min_presses,
            win_key_window_s=cfg.win_key_window_s,
            win_key_cooldown_s=cfg.win_key_cooldown_s,
            frame_stride=cfg.event_stride,
            blank_frame_mean_threshold=cfg.blank_frame_mean_threshold,
            blank_frame_std_threshold=cfg.blank_frame_std_threshold,
            attack_min_gap=cfg.attack_min_gap,
            death_min_gap=cfg.death_min_gap,
        )

    for i, run_dir in enumerate(run_dirs):
        print(f"\n[{i + 1}/{len(run_dirs)}] Processing {os.path.basename(run_dir)}")
        frames_log = os.path.join(run_dir, "frames.parquet")
        video_path = os.path.join(run_dir, "recording.mp4")
        if not os.path.exists(frames_log) or not os.path.exists(video_path):
            continue

        frame_indices, _frame_timestamps = load_frame_logs(frames_log)
        if not frame_indices:
            continue

        # Write event/episode/return labels for RL
        if event_detector:
            total_frames = max(frame_indices) + 1 if frame_indices else 0

            # Detect events
            events_path = os.path.join(run_dir, "events.parquet")
            inputs_log = os.path.join(run_dir, "inputs.parquet")
            events = event_detector.detect_events(video_path, inputs_log=inputs_log)
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
