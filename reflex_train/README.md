# reflex_train

Training stack for the Reflex Agent. This is intentionally separate from `visionctl`
so the Rust agent package stays lean and the training pipeline can evolve independently.

## What it does

- Builds a vision-to-action model for 2D sidescrollers (SuperTux first)
- Uses weak labeling (offline) to generate event/episode/return signals
- Trains a reflex policy with IQL-style weighting and inverse dynamics

## Layout

- `reflex_train/data/` dataset + logs + key maps
- `reflex_train/models/` model definitions
- `reflex_train/training/` training loop + metrics + config
- `reflex_train/weak_labels/` game-specific weak labelers
- `precompute_labels.py` offline weak labeling step
- `train_reflex.py` training entrypoint

## Task Shortcuts (Just)

Install just once:

```bash
cargo install just
```

From the repo root:

```bash
just record
just label
just label-all
just train
just train-awr
just train-no-awr
just train-custom EPOCHS=50 BATCH_SIZE=32 LR=1e-4
just export CKPT=checkpoints/reflex_epoch_0.pth OUT=reflex.onnx
```

Overrides (optional):
- `DATASET_DIR=dataset` (root Justfile)
- `DATASET_DIR=../dataset` (reflex_train Justfile)

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
- `frames.parquet`
- `inputs.parquet`

## Precompute weak labels

```bash
cd reflex_train
python precompute_labels.py --data_dir ../dataset/
```

Outputs per-session label files:
- `events.parquet` - death/attack via template matching; wins via key presses in `inputs.parquet`
- `episodes.parquet` - episode segmentation (spawn to terminal)
- `returns.parquet` - discounted returns with attack bonuses

Key knobs:

- `--sprite_threshold 0.85` (template match confidence)
- `--event_stride 1` (scan every Nth frame for event detection)
- `--attack_reward 0.1` (bonus for killing enemies)
- `--death_reward -1.0` / `--win_reward 1.0`

Uses GPU-accelerated template matching (FFT-based NCC) with torchcodec.
Falls back to CPU PyTorch if CUDA is unavailable.

## Train

Training streams frames sequentially per video (no random seeks) and batches them on the fly.
It uses an IQL-style weighting scheme so higher-return segments of your dataset pull the policy
more strongly.

```bash
cd reflex_train
PYTHONPATH=.. uv run python train_reflex.py --data_dir ../dataset/ --epochs 1 --batch_size 8 --workers 8
```

Notes:
- `--workers` controls the DataLoader worker count (higher can help hide decode time).
- The progress bar shows an estimated total step count based on dataset size.
- Use `--compile-model true` to try `torch.compile`.
- `--inv-dyn-weight` enables the inverse dynamics auxiliary loss.
- `--iql-expectile` and `--iql-adv-temperature` tune the IQL weighting.

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

## Design notes

See `reflex_train/docs/reflex_train_design.md` for a readable overview of the policy/value/IQL setup
and how weak labels feed into returns.

## Rust inference stub

There is a minimal Rust runner in `reflex_infer/`:

```bash
cargo run -p reflex_infer -- --model reflex.onnx live
```

This runs live inference (screenshots + key presses).

The exporter also writes `<output>_manifest.json` (input/output contract) next to the ONNX file.

## Rust inference modes

See `reflex_infer/README.md` for live and eval usage.

Important settings (via CLI or env vars, prefixed with `REFLEX_TRAIN_`):

- `data_dir`: dataset root (required)
- `action_horizon`: predict keys this many frames ahead
- `checkpoint_dir`: save checkpoints and config snapshot

## Verify dataset (optional)

```bash
cd reflex_train
python -m tests.verify_dataset ../dataset/
```

Generates `debug_stack.png` to visualize the 3-frame stack.
