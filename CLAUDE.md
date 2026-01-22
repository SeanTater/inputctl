# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Workspace Structure

This is a Cargo workspace with five crates:

```
inputctl/           Keyboard/mouse control via uinput
inputctl-capture/   Screen capture via Wayland portal + KWin cursor
inputctl-vision/    LLM agent, tool-calling, web debugger
inputctl-reflex/    ONNX inference for game AI
reflex_train/       Python training pipeline (separate venv)
```

### Dependency Graph

```
inputctl + inputctl-capture (independent primitives)
        ↓
inputctl-vision (LLM + agent, uses both)
        ↓
inputctl-reflex (ONNX inference, uses inputctl + inputctl-capture)
        ↓
reflex_train (Python, produces ONNX models)
```

## Build Commands

```bash
# Build all Rust crates
cargo build --release

# Build specific crate
cargo build --release -p inputctl
cargo build --release -p inputctl-capture
cargo build --release -p inputctl-vision
cargo build --release -p inputctl-reflex

# Run tests
cargo test --workspace
sudo cargo test -p inputctl -- --ignored  # Integration tests need uinput

# Build Python packages
cd inputctl && maturin develop --uv && cd ..
cd inputctl-vision && maturin develop --uv && cd ..

# Run CLIs
inputctl-vision "What's on my screen?"
inputctl-vision agent "Open Firefox"
inputctl-record --output dataset --fps 10
```

## Setup

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get install -y \
  xdg-desktop-portal

# Install maturin
uv tool install maturin

# Build Python packages
uv sync
cd inputctl && maturin develop --uv && cd ..
cd inputctl-vision && maturin develop --uv && cd ..
```

## Crate Details

### inputctl

Low-level keyboard/mouse control via Linux uinput.

**Key files:**
- `src/lib.rs` - Main `InputCtl` struct + pyo3 bindings
- `src/device.rs` - uinput virtual device creation
- `src/keyboard.rs` - ASCII-to-keycode mapping
- `src/interpolation.rs` - Smooth mouse movement with noise

**Design:**
- No daemon needed - holds uinput handle in memory
- ~1 second init delay when `InputCtl::new()` called
- Servo feedback for pointer acceleration compensation

**Requires:** `/dev/uinput` access (root or input group)

### inputctl-capture

Screen capture and cursor tracking primitives.

**Key files:**
- `src/capture/portal.rs` - xdg-desktop-portal ScreenCast
- `src/primitives/cursor.rs` - KWin DBus cursor tracking
- `src/primitives/screenshot.rs` - Screenshot capture + processing
- `src/recorder.rs` - Video + input event recording

**Design:**
- Portal-based capture works on any Wayland compositor
- KWin cursor tracking (KDE Plasma only)
- PipeWire stream processing for video capture

**Binaries:** `inputctl-record` for training data capture

### inputctl-vision

LLM-based GUI automation with autonomous agent.

**Key files:**
- `src/lib.rs` - `VisionCtl` main controller
- `src/agent.rs` - Autonomous agent loop
- `src/llm/` - LLM client with tool-calling
- `src/llm/tools.rs` - Agent tool definitions
- `src/debugger.rs` - Agent state tracking
- `src/server.rs` - Web debugger server
- `src/detection/` - Template matching (rarely used)

**Design:**
- Agent uses normalized coordinates (0-1000)
- Screenshots include cursor marker (red crosshair)
- Web debugger at localhost:3000 with `--debug` flag
- Template matching available but LLM vision preferred

**Environment variables:**
- `VISIONCTL_BACKEND` - ollama, vllm, openai
- `VISIONCTL_URL` - Backend URL
- `VISIONCTL_MODEL` - Model name
- `VISIONCTL_API_KEY` - API key (for OpenAI)

### inputctl-reflex

ONNX inference for game AI.

**Key files:**
- `src/main.rs` - Live and eval mode runners

**Design:**
- Live mode: capture screenshots, run inference, execute keys
- Eval mode: offline evaluation on recorded data
- Uses `ort` crate for ONNX runtime

**Note:** Inverse dynamics is stubbed in training but not inference

### reflex_train

Python training pipeline (separate from Rust workspace).

**Key files:**
- `training/train.py` - Main training script
- `training/data.py` - Dataset loading
- `training/model.py` - Model architecture

**Note:** Template matching in training is for weak labeling (different purpose than detection/ in inputctl-vision)

## Common Pitfalls

**Never use `sudo uv run python`** - creates root-owned venv. Use `sudo .venv/bin/python` instead.

**KDE screenshot permissions** - Run `inputctl-vision install-desktop-file` to grant portal access.

**ONNX Runtime** - If `ort` can't find runtime, set `ORT_LIB_LOCATION=/usr/lib/libonnxruntime.so`

## Python Usage

```python
# Low-level input
from inputctl import InputCtl
ctl = InputCtl()
ctl.type_text("Hello!")
ctl.move_mouse_smooth(100, 50, 1.0, "ease-in-out")
ctl.click("left")

# LLM-based automation
from inputctl_vision import VisionCtl
ctl = VisionCtl(backend="ollama", url="http://localhost:11434", model="llava")
answer = ctl.ask("What's on screen?")
ctl.move_to(500, 300)  # Normalized 0-1000
ctl.click("left")
```

## CLI Examples

```bash
# Ask about screen
inputctl-vision "What windows are open?"

# Run autonomous agent
inputctl-vision agent "Open Firefox and go to example.com"
inputctl-vision agent --debug "Open Settings"  # With web debugger

# Record training data
inputctl-record --output dataset --fps 10 --preset ultrafast

# Run trained model
cargo run -p inputctl-reflex -- --model model.onnx live --window "Game" --fps 10
```
