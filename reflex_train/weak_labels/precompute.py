import json
import os
from reflex_train.data.logs import load_key_sets, load_frame_logs
from reflex_train.weak_labels import KeyWindowIntentLabeler, SuperTuxIntentLabeler
from .config import LabelingConfig


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
    with open(path, "w", encoding="utf-8") as f:
        for pos, (idx, intent) in enumerate(zip(frame_indices, intents)):
            entry = {"frame_idx": idx, "intent": intent}
            if frame_timestamps:
                entry["timestamp"] = frame_timestamps[pos]
            f.write(json.dumps(entry) + "\n")
    return True


def precompute(cfg: LabelingConfig):
    run_dirs = find_run_dirs(cfg.data_dir)
    if not run_dirs:
        print("No run directories found.")
        return

    if cfg.labeler == "supertux":
        labeler = SuperTuxIntentLabeler(
            base_dir=cfg.base_dir,
            intent_horizon=cfg.intent_horizon,
            sprite_scale=cfg.sprite_scale,
            sprite_threshold=cfg.sprite_threshold,
            proximity_px=cfg.sprite_proximity,
        )
    elif cfg.labeler == "keys":
        labeler = KeyWindowIntentLabeler(intent_horizon=cfg.intent_horizon)
    else:
        raise ValueError("labeler must be 'supertux' or 'keys'")

    for run_dir in run_dirs:
        frames_log = os.path.join(run_dir, "frames.jsonl")
        inputs_log = os.path.join(run_dir, "inputs.jsonl")
        video_path = os.path.join(run_dir, "recording.mp4")
        if not os.path.exists(frames_log) or not os.path.exists(video_path):
            continue

        frame_indices, frame_timestamps = load_frame_logs(frames_log)
        _, key_sets = load_key_sets(frames_log, inputs_log)
        if not frame_indices or not key_sets:
            continue

        intents = labeler.label_intents(video_path, frame_indices, key_sets)
        intent_path = os.path.join(run_dir, "intent.jsonl")
        wrote = write_intents(intent_path, frame_indices, frame_timestamps, intents, overwrite=cfg.overwrite)
        status = "wrote" if wrote else "skipped"
        print(f"{status} {intent_path}")


def main():
    cfg = LabelingConfig()
    precompute(cfg)


if __name__ == "__main__":
    main()
