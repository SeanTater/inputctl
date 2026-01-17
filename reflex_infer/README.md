# reflex_infer

Minimal Rust inference stub for ONNX exports of the Reflex Agent.

## Usage

### Live mode (screenshots + real key presses)

```bash
cargo run -p reflex_infer -- \
  --model /path/to/model.onnx \
  live \
  --window "SuperTux" \
  --fps 10 \
  --goal-intent RUN
```

```bash
cargo run -p reflex_infer -- \
  --model /path/to/model.onnx \
  eval \
  --video recording.mp4 \
  --frames frames.jsonl \
  --intents intent.jsonl \
  --out preds.jsonl
```

Live mode captures screenshots via `visionctl` and sends key events via `inputctl`.
Eval mode reads frames from video via ffmpeg and writes `preds.jsonl`.

By default it looks for `/path/to/model_manifest.json` next to the ONNX file.
Pass `--manifest` to override.

Eval mode requires `ffmpeg` on the PATH.

If the `ort` crate cannot find or download ONNX Runtime, install it system-wide
and set `ORT_LIB_LOCATION` to the shared library path.
