#!/bin/bash

# Sync datasets with ownership rules:
# - Local owns raw capture files (video/frames/inputs)
# - GPU owns generated labels (events/episodes/returns)

LOCAL_DATASET="$HOME/repos/inputctl/dataset/"
REMOTE_DATASET="sean-gallagher@intuition.local:/home/sean-gallagher/sandbox/inputctl/dataset/"

set -euo pipefail

# Push raw captures to GPU (exclude labels to avoid clobbering)
rsync -av --delete \
  --include '*/' \
  --include 'recording.mp4' \
  --include 'frames.parquet' \
  --include 'inputs.parquet' \
  --include 'frames.jsonl' \
  --include 'inputs.jsonl' \
  --include 'mouse.bin' \
  --exclude 'events.parquet' \
  --exclude 'episodes.parquet' \
  --exclude 'returns.parquet' \
  --exclude '*' \
  "$LOCAL_DATASET" "$REMOTE_DATASET"

# Pull labels from GPU (only labeled outputs)
rsync -av --delete \
  --include '*/' \
  --include 'events.parquet' \
  --include 'episodes.parquet' \
  --include 'returns.parquet' \
  --exclude '*' \
  "$REMOTE_DATASET" "$LOCAL_DATASET"
