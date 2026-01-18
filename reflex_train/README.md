# reflex_train

Training stack for the Reflex Agent. This is intentionally separate from `visionctl`
so the Rust agent package stays lean and the training pipeline can evolve independently.

## What it does

- Builds a vision-to-action model for 2D sidescrollers (SuperTux first)
- Uses weak labeling (offline) to infer high-level intent
- Trains a reflex policy plus an intent predictor

## Layout

- `reflex_train/data/` dataset + logs + key maps
- `reflex_train/models/` model definitions
- `reflex_train/training/` training loop + metrics + config
- `reflex_train/weak_labels/` game-specific weak labelers
- `precompute_intents.py` offline weak labeling step
- `train_reflex.py` training entrypoint

## Install (uv)

```bash
cd reflex_train
uv sync
```

Notes:
- For RTX 50xx (sm_120), this project is configured to use the PyTorch `cu130` wheels (see `pyproject.toml`).
- TorchCodec CUDA decode needs NVIDIA NPP; `uv sync` installs `nvidia-npp` automatically.

## Record data (visionctl)

From the repo root:

```bash
cargo build --release -p visionctl --bin visionctl
./target/release/visionctl record --fps 10 --window "SuperTux" --output dataset/
```

This creates `dataset/run_<timestamp>/` folders with:

- `recording.mp4`
- `frames.jsonl`
- `inputs.jsonl`

## Precompute weak labels

```bash
cd reflex_train
python precompute_intents.py --data_dir ../dataset/ --labeler supertux
```

Outputs `intent.jsonl` next to each recording session.

Key knobs:

- `--intent_horizon 10` (lookahead for intent halo)
- `--sprite_scale 0.5` (speed vs detail)
- `--sprite_threshold 0.85`
- `--sprite_proximity 96`

## Train

Training streams frames sequentially per video (no random seeks) and batches them on the fly.

```bash
cd reflex_train
PYTHONPATH=.. uv run python train_reflex.py --data_dir ../dataset/ --epochs 1 --batch_size 8 --workers 8
```

Notes:
- `--workers` controls the DataLoader worker count (higher can help hide decode time).
- The progress bar shows an estimated total step count based on dataset size.
- Use `--compile-model true` to try `torch.compile`.

## Export ONNX

```bash
cd reflex_train
python export_model.py --checkpoint ../checkpoints/reflex_epoch_0.pth --output reflex.onnx
```

To export a dummy model with random weights (for wiring tests):

```bash
python export_model.py --random --output reflex.onnx
```

Use `--height` and `--width` to match your capture size. These set dummy export sizes; inference can still be dynamic.

## Rust inference stub

There is a minimal Rust runner in `reflex_infer/`:

```bash
cargo run -p reflex_infer -- --model reflex.onnx live --goal-intent RUN
```

This runs live inference (screenshots + key presses).

The exporter also writes `<output>_manifest.json` (input/output contract) next to the ONNX file.

## Rust inference modes

See `reflex_infer/README.md` for live and eval usage.

Important settings (via CLI or env vars, prefixed with `REFLEX_TRAIN_`):

- `data_dir`: dataset root (required)
- `action_horizon`: predict keys this many frames ahead
- `intent_weight`: weight for intent loss
- `checkpoint_dir`: save checkpoints and config snapshot

## Verify dataset (optional)

```bash
cd reflex_train
python -m tests.verify_dataset ../dataset/
```

Generates `debug_stack.png` to visualize the 3-frame stack.
