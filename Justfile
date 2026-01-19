DATASET_DIR := "dataset"
CONFIG_FILE := env_var_or_default("CONFIG_FILE", "reflex_train/configs/default.toml")

record DATASET_DIR=DATASET_DIR:
  cargo run --release --bin inputctl-record -- --fps 30 --output {{DATASET_DIR}}

label DATASET_DIR=DATASET_DIR:
  uv run --project reflex_train python reflex_train/precompute_intents.py --data_dir {{DATASET_DIR}} --labeler keys --event_stride 10 --intent_stride 10 --sparkle_threshold 0.9

label-all DATASET_DIR=DATASET_DIR:
  uv run --project reflex_train python reflex_train/precompute_intents.py --data_dir {{DATASET_DIR}} --labeler keys --overwrite true --event_stride 10 --intent_stride 10 --sparkle_threshold 0.9

train CONFIG=CONFIG_FILE:
  uv run --project reflex_train python reflex_train/train_reflex.py {{CONFIG}}

train-mini:
  uv run --project reflex_train python reflex_train/train_reflex.py reflex_train/configs/mini.toml

train-remote:
  ssh sean-gallagher@intuition.local -- uv run --directory /home/sean-gallagher/sandbox/inputctl/ --project reflex_train python reflex_train/train_reflex.py reflex_train/configs/default.toml

train-remote-mini:
  ssh sean-gallagher@intuition.local -- uv run --directory /home/sean-gallagher/sandbox/inputctl/ --project reflex_train python reflex_train/train_reflex.py reflex_train/configs/mini.toml

  
export CKPT OUT="reflex.onnx":
  uv run --project reflex_train python reflex_train/export_model.py --checkpoint {{CKPT}} --output {{OUT}}
