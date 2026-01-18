DATASET_DIR := "dataset"

record DATASET_DIR=DATASET_DIR:
  cargo run --release --bin inputctl-record -- --fps 30 --output {{DATASET_DIR}}

label DATASET_DIR=DATASET_DIR:
  uv run --project reflex_train python reflex_train/precompute_intents.py --data_dir {{DATASET_DIR}} --labeler keys

label-all DATASET_DIR=DATASET_DIR:
  uv run --project reflex_train python reflex_train/precompute_intents.py --data_dir {{DATASET_DIR}} --labeler keys --overwrite true

train DATASET_DIR=DATASET_DIR:
  uv run --project reflex_train python reflex_train/train_reflex.py --data-dir {{DATASET_DIR}} --compile-model true --workers 20

train-awr DATASET_DIR=DATASET_DIR:
  uv run --project reflex_train python reflex_train/train_reflex.py --data-dir {{DATASET_DIR}} --use-awr true --awr-temperature 1.0 --value-weight 0.5

train-no-awr DATASET_DIR=DATASET_DIR:
  uv run --project reflex_train python reflex_train/train_reflex.py --data-dir {{DATASET_DIR}} --use-awr false

train-custom EPOCHS="50" BATCH_SIZE="32" LR="1e-4" DATASET_DIR=DATASET_DIR:
  uv run --project reflex_train python reflex_train/train_reflex.py --data-dir {{DATASET_DIR}} --epochs {{EPOCHS}} --batch-size {{BATCH_SIZE}} --learning-rate {{LR}}

export CKPT OUT="reflex.onnx":
  uv run --project reflex_train python reflex_train/export_model.py --checkpoint {{CKPT}} --output {{OUT}}
