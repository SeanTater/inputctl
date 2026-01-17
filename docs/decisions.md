# Design Decisions

Short notes on architectural choices. Update when you make a change that would
surprise a new contributor.

## 2025-XX-XX: Split training from agent runtime

- `visionctl` remains a Rust-first agent system with minimal Python deps.
- `reflex_train` is a separate Python project for training and weak labeling.
- `reflex_infer` is a Rust ONNX inference stub to validate export + runtime wiring.

## 2025-XX-XX: Offline weak labeling for intent

- Intent labels are generated in a separate preprocessing step and stored as
  `intent.jsonl` alongside recordings.
- Game-specific heuristics (SuperTux sprites, etc.) live under `reflex_train/weak_labels/`.

## 2025-XX-XX: ONNX as the interchange format

- Models export to ONNX with dynamic height/width to simplify Rust inference.
- A JSON manifest accompanies each export to document input/output contract.

## 2025-XX-XX: Dual inference modes in Rust

- `reflex_infer` provides both live (screenshots + real key presses) and eval
  (offline video + logs) inference paths.
- Live mode uses `visionctl` for capture and `inputctl` for action emission.
- Eval mode reads frames from video via ffmpeg and writes predicted actions to JSONL.
