# AGENTS

This repository already documents most workflows in the READMEs. Start with:

- `README.md` for workspace overview and quick start.
- `reflex_train/README.md` for training/labeling details.

Notes that are not in those READMEs:

- Prefer `just` tasks for record/label/train/export (see `Justfile`), then fall back to raw commands if needed.
- The training pipeline now uses IQL-style weighting plus an inverse dynamics auxiliary loss; design notes are in `reflex_train/docs/reflex_train_design.md`.
- If you modify the model or training losses, update the design doc so future agents have context.
