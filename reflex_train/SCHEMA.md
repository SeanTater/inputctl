# Dataset Schema Documentation

This document describes the Parquet schema for all dataset files in the reflex_train pipeline.

## File Inventory

| File | Source | Description |
|------|--------|-------------|
| `frames.parquet` | `inputctl-record` | Frame timing metadata from recorder |
| `inputs.parquet` | `inputctl-record` | Keyboard events during recording |
| `events.parquet` | `precompute_labels.py` | Death/win/attack events |
| `episodes.parquet` | `precompute_labels.py` | Episode boundaries |
| `returns.parquet` | `precompute_labels.py` | Per-frame discounted returns |
| `recording.mp4` | `inputctl-record` | Video frames (not Parquet) |

## Schema Definitions

### frames.parquet

Frame timing from the recorder.

| Column | Type | Description |
|--------|------|-------------|
| `frame_idx` | Int64 | Zero-based frame index |
| `timestamp` | Int64 | Milliseconds since Unix epoch |

**Written by:** `inputctl-capture/src/recorder.rs`
**Read by:** `reflex_train/data/logs.py`, `inputctl-reflex/src/main.rs`

### inputs.parquet

Keyboard events captured during recording.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `timestamp` | Int64 | No | Milliseconds since Unix epoch |
| `event_type` | Utf8 | No | Always "key" |
| `key_code` | Int32 | Yes | Linux evdev key code |
| `key_name` | Utf8 | Yes | Key name (e.g., "KEY_W") |
| `state` | Utf8 | Yes | "down", "up", or "repeat" |
| `x` | Int32 | Yes | Mouse X (unused) |
| `y` | Int32 | Yes | Mouse Y (unused) |

**Written by:** `inputctl-capture/src/recorder.rs`
**Read by:** `reflex_train/data/logs.py`

### events.parquet

Terminal and attack events detected in gameplay (wins from key presses).

| Column | Type | Description |
|--------|------|-------------|
| `frame_idx` | Int64 | Frame where event occurred |
| `event` | Utf8 | "DEATH", "WIN", or "ATTACK" (wins from key presses) |
| `confidence` | Float64 | Detection confidence [0, 1] (wins are 1.0 from key presses) |

**Written by:** `reflex_train/weak_labels/precompute.py` (via `precompute_labels.py`)
**Read by:** `reflex_train/data/dataset.py`

### episodes.parquet

Episode boundaries for RL training.

| Column | Type | Description |
|--------|------|-------------|
| `episode_id` | Int64 | Unique episode identifier |
| `start_frame` | Int64 | First frame of episode |
| `end_frame` | Int64 | Last frame of episode |
| `outcome` | Utf8 | "DEATH", "WIN", or "INCOMPLETE" |
| `reward` | Float64 | Terminal reward (-1, +1, or 0) |
| `length` | Int64 | Number of frames in episode |

**Written by:** `reflex_train/weak_labels/precompute.py` (via `precompute_labels.py`)
**Read by:** `reflex_train/data/dataset.py`

### returns.parquet

Per-frame discounted returns for value function training.

| Column | Type | Description |
|--------|------|-------------|
| `frame_idx` | Int64 | Frame index |
| `return` | Float64 | Discounted return from this frame |

**Written by:** `reflex_train/weak_labels/precompute.py` (via `precompute_labels.py`)
**Read by:** `reflex_train/data/dataset.py`

## Generation Pipeline

```
1. inputctl-record
   └── frames.parquet, inputs.parquet, recording.mp4

2. precompute_labels.py (weak labeling)
   ├── reads: frames.parquet, inputs.parquet, recording.mp4
   └── writes: events.parquet, episodes.parquet, returns.parquet

3. train.py (training)
   └── reads: all parquet files + recording.mp4

4. inputctl-reflex (inference)
   └── reads: frames.parquet
```

## Update Checklist

When modifying schemas:

1. Update this file (SCHEMA.md)
2. Update Rust writer (`inputctl-capture/src/recorder.rs`)
3. Update Python writers (`reflex_train/weak_labels/precompute.py`)
4. Update Python readers (`reflex_train/data/logs.py`, `reflex_train/data/dataset.py`)
5. Update Rust readers (`inputctl-reflex/src/main.rs`)
6. Re-run recording and precompute on test data
7. Verify training and eval still work
