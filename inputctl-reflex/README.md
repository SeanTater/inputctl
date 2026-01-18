# inputctl-reflex

ONNX inference runner for game AI with reflex-trained models.

This crate runs trained models exported from `reflex_train` for real-time game automation.

## Usage

### Live Mode (Real-time Control)

Captures screenshots and executes predicted actions in real-time:

```bash
cargo run -p inputctl-reflex -- \
  --model /path/to/model.onnx \
  live \
  --window "SuperTux" \
  --fps 10 \
  --goal-intent RUN
```

Live mode:
- Captures screenshots via `inputctl-capture`
- Runs inference on each frame
- Sends key events via `inputctl`

### Eval Mode (Offline Evaluation)

Runs inference on recorded data for evaluation:

```bash
cargo run -p inputctl-reflex -- \
  --model /path/to/model.onnx \
  eval \
  --video recording.mp4 \
  --frames frames.jsonl \
  --intents intent.jsonl \
  --out preds.jsonl
```

Eval mode:
- Reads frames from video via ffmpeg
- Runs inference on each frame
- Writes predictions to `preds.jsonl`
- Requires `ffmpeg` on the PATH

## Model Manifest

By default, the runner looks for `/path/to/model_manifest.json` next to the ONNX file. Pass `--manifest` to override.

The manifest contains:
- Input/output tensor names and shapes
- Class labels (key names)
- Training configuration

## ONNX Runtime

Uses the `ort` crate for ONNX inference. If it cannot find or download ONNX Runtime automatically, install it system-wide and set `ORT_LIB_LOCATION`:

```bash
export ORT_LIB_LOCATION=/usr/lib/libonnxruntime.so
```

## Architecture Notes

### Current State

The model predicts which keys to press based on:
- Current screenshot
- Goal intent (e.g., "RUN", "JUMP", "NAVIGATE_TO_MENU")

### Inverse Dynamics (Stub)

The training pipeline has a stub for inverse dynamics prediction - learning actions from (state, next_state) pairs rather than human demonstrations. This is not yet implemented in the inference runner but the training code has placeholder support.

## Dependencies

- `inputctl` - Keyboard input via uinput
- `inputctl-capture` - Screenshot capture
- `ort` - ONNX Runtime bindings
- `image` - Image processing
- `ndarray` - Tensor operations

## License

MIT
