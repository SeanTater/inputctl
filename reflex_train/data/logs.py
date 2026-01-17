import os
import polars as pl


def load_frame_logs(frames_log: str):
    if not os.path.exists(frames_log):
        return [], []
    df_frames = pl.read_ndjson(frames_log)
    frame_timestamps = df_frames["timestamp"].to_list()
    frame_indices = df_frames["frame_idx"].to_list()
    return frame_indices, frame_timestamps


def load_key_sets(frames_log: str, inputs_log: str):
    frame_indices, frame_timestamps = load_frame_logs(frames_log)
    if not frame_indices:
        return frame_indices, []

    key_events = []
    if os.path.exists(inputs_log):
        df_inputs = pl.read_ndjson(inputs_log)
        df_keys = df_inputs.filter(pl.col("event_type") == "key").sort("timestamp")
        if not df_keys.is_empty():
            key_events = df_keys.to_dicts()

    key_idx = 0
    current_keys = set()
    key_set_by_frame = []

    for f_ts in frame_timestamps:
        while key_idx < len(key_events):
            evt = key_events[key_idx]
            if evt["timestamp"] > f_ts:
                break

            k = evt["key_name"]
            if "Key(" in k:
                k = k.replace("Key(", "").replace(")", "")

            state = evt["state"]
            if state == "down":
                current_keys.add(k)
            elif state == "up":
                current_keys.discard(k)

            key_idx += 1

        key_set_by_frame.append(set(current_keys))

    return frame_indices, key_set_by_frame
