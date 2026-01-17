# Reflex Agent: Master Implementation Plan

This document contains the complete architectural blueprint and implementation details for the **Reflex Agent** ("The Insect"). It is designed to be a standalone reference for setting up the training pipeline in a new environment.

## 1. Architectural Vision
The **Reflex Agent** is a fast, reactive "spine" for the AI.
-   **Role**: Handles millisecond-level motor control (jump timing, evasion, platforming).
-   **Inputs**:
    -   **Visual**: Standardized 224x224 RGB frames, stacked ($t, t-1, t-2$) to perceive velocity/acceleration.
    -   **Control**: A "Goal Vector" $(dx, dy)$ provided by a higher-level planner (or hardcoded to "Right" for training).
    -   **Context**: 9-channel input tensor (3 frames $\times$ 3 channels).
-   **Outputs**:
    -   **Keyboard**: Logits for keyboard state (Multi-label classification).
    -   **Mouse**: Normalized $(x, y)$ coordinates ($0..1$).

---

## 2. Environment Setup

### System Dependencies
*   **Linux (KDE Plasma 6 recommended for `visionctl` recorder)**
*   **ffmpeg**: Required for video encoding/decoding.
    ```bash
    sudo apt install ffmpeg
    ```

### Python Environment (`uv`)
We use `uv` for fast dependency management.
```bash
# Initialize project
uv init
# Add dependencies
uv add torch torchvision polars opencv-python matplotlib numpy
```

### Rust Recorder (`visionctl`)
The recorder is a custom Rust binary that captures:
1.  **Video**: Raw screen pixels piped to `ffmpeg` (H.264 `yuv420p`).
2.  **Input**: Low-level `evdev` keyboard events.
3.  **Mouse**: Binary stream of cursor positions.

Build the recorder:
```bash
cargo build --release --bin visionctl
```

---

## 3. Data Collection (The Recorder)

We record **cropped** gameplay to focus the agent's attention.

**Command:**
```bash
# Finds window named "SuperTux", crops to it, records at 10fps
./target/release/visionctl record --fps 10 --window "SuperTux" --output dataset/
```

**Data Format per Session:**
-   `recording.mp4`: The visual stream (cropped).
-   `inputs.jsonl`: Discrete keyboard events (`down`, `up`).
-   `frames.jsonl`: Maps video frame index to timestamp.
-   `mouse.bin`: Binary stream `[u64 timestamp][i32 x][i32 y]` (normalized to crop if using crop).

---

## 4. Training Implementation

### A. Data Pipeline (`reflex_train/data/dataset.py`)
Key features: **Lazy Loading** (no RAM explosion), **Frame Stacking** (velocity), **OpenCV Seeking**.

```python
import torch
from torch.utils.data import Dataset
import cv2
import polars as pl
import numpy as np
import os

class MultiStreamDataset(Dataset):
    def __init__(self, run_dirs, transform=None, context_frames=3):
        self.context_frames = context_frames
        self.samples = []
        # ... indexing logic ...
    
    def __getitem__(self, idx):
        # 1. Open Video (Cached per worker)
        # 2. Seek to frame (t - 2)
        # 3. Read 3 frames
        # 4. Stack into (9, H, W) Tensor
        # 5. Return {'pixels': stack, 'label_keys': vec, 'label_mouse': pos}
```
*(See `reflex_train/data/dataset.py` for full implementation)*

### B. Model (`reflex_train/models/reflex_net.py`)
Key features: **ResNet-18 Backbone**, **9-Channel adaptation**, **Multi-Head Output**.

```python
class ReflexNet(nn.Module):
    def __init__(self, context_frames=3, num_keys=128):
        # Modified ResNet-18
        self.backbone = resnet18()
        # Adapt first layer: 3 channels -> 9 channels
        # Weighs averaged to preserve initialization
        
        # Heads
        self.head_keys = Linear(..., num_keys) # BCEWithLogitsLoss
        self.head_mouse = Linear(..., 2)       # MSELoss
```

### C. Training Loop (`reflex_train/train_reflex.py`)
Standard PyTorch loop with **Multi-Objective Loss**.

$$ L_{total} = L_{keys} + L_{mouse} $$

```python
# Keys: Multi-label classification (can press Jump + Right)
criterion_keys = nn.BCEWithLogitsLoss()
# Mouse: Regression
criterion_mouse = nn.MSELoss()
```

---

## 5. Verification Strategy

Before running a 40h training job, verify the pipeline:

1.  **Synthetic Data Check**:
    Run `gen_synthetic_data.py`. This creates a bouncing ball video with predictable key presses.
    
2.  **Visual Stack Inspection**:
    Run `verify_dataset.py`. It outputs `debug_stack.png`.
    *   **Check**: Do you see 3 frames?
    *   **Check**: Is the ball moving? (If frames are identical, stacking is broken).
    *   **Check**: Do keys match the movement?

---

## 6. Future Roadmap: World Models
To improve beyond simple cloning:
1.  **Inverse Dynamics**: Add auxiliary head to predict *Action* given $(Frame_t, Frame_{t+1})$. Forces encoder to learn physics.
2.  **Forward Dynamics**: Predict $Frame_{t+1}$ given $(Frame_t, Action)$. (Dreamer-lite).

**Status**: The code currently contains stubs for `head_inv_dynamics` in `ReflexNet`.
