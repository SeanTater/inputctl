# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace Structure

This is a Cargo workspace with two crates:

- **inputctl** - Linux input automation library (keyboard/mouse control via uinput)
- **visionctl** - Screen capture and LLM vision analysis (Wayland screenshots + local LLM queries)

Each crate has its own Python bindings via PyO3/maturin.

## Build Commands

```bash
# Build all Rust crates in workspace
cargo build --release

# Build specific crate
cargo build --release -p inputctl
cargo build --release -p visionctl

# Run inputctl tests (unit tests only)
cargo test -p inputctl

# Run inputctl integration tests (requires sudo for /dev/uinput)
sudo cargo test -p inputctl -- --ignored

# Build inputctl Python wheel
cd inputctl && maturin build --release

# Build visionctl Python wheel
cd visionctl && maturin build --release

# Install inputctl Python package for development
cd inputctl && maturin develop --uv

# Install visionctl Python package for development
cd visionctl && maturin develop --uv

# Run inputctl Python tests
cd inputctl && uv run pytest python_tests/ -v -m "not integration"

# Run inputctl Python integration tests (requires sudo)
# IMPORTANT: Use .venv/bin/python, NOT "sudo uv run python"
# (sudo uv run creates a root-owned environment)
cd inputctl && sudo ../.venv/bin/python -m pytest python_tests/ -v

# Run visionctl CLI utility
cargo run --release -p visionctl --bin visionctl "What's on my screen?"

# Or with environment variables
VISIONCTL_BACKEND=ollama VISIONCTL_URL=http://localhost:11434 VISIONCTL_MODEL=llava \
  cargo run --release -p visionctl --bin visionctl "What's on my screen?"
```

## Setup

```bash
# Install maturin as a uv tool
uv tool install maturin

# Sync dev dependencies and build both Python packages
uv sync
cd inputctl && maturin develop --uv && cd ..
cd visionctl && maturin develop --uv && cd ..
```

## Architecture

### inputctl

Linux input automation library using uinput, with Python bindings via pyo3.

#### Core Components

- `inputctl/src/lib.rs` - Main `InputCtl` struct and public API, plus pyo3 bindings (behind `python` feature)
- `inputctl/src/device.rs` - Creates the uinput virtual device with keyboard/mouse capabilities
- `inputctl/src/keyboard.rs` - ASCII-to-keycode mapping for US keyboard layout
- `inputctl/src/mouse.rs` - Mouse button enum and key mappings
- `inputctl/src/interpolation.rs` - Smooth mouse movement with low-frequency Perlin-like noise

### Key Design Decisions

- **No daemon needed**: Unlike the C version, this library holds the uinput device handle in memory. The ~1 second initialization delay happens once when `InputCtl::new()` is called.
- **Uses evdev crate**: Wraps Linux uinput via the `evdev` crate
- **Feature-gated Python**: pyo3 bindings only compiled with `--features python`
- **Smooth mouse movement**: Uses low-frequency Perlin-like noise (control points every ~200ms) with cubic interpolation for natural-looking movement paths. Noise amount is configurable (default ±2 pixels).

### Permission Requirements

Requires access to `/dev/uinput`:
- Run as root, OR
- Add user to `input` group with appropriate udev rules

### API Features

**Smooth Mouse Movement** (`move_mouse_smooth`):
- Moves mouse with realistic low-frequency noise variation
- Parameters: `dx`, `dy`, `duration`, `curve` ("linear" or "ease-in-out"), `noise` (default 2.0)
- Noise amount: 0.0 = perfectly smooth, 2.0 = subtle (default), 5.0 = more obvious
- Uses cubic interpolation with control points every ~200ms for natural wavering
- Always hits exact target position despite noise

**Example (Python)**:
```python
from inputctl import InputCtl
ctl = InputCtl()
# Subtle natural variation (default)
ctl.move_mouse_smooth(100, 50, 1.0, "ease-in-out", noise=2.0)
# Perfectly smooth
ctl.move_mouse_smooth(100, 50, 1.0, "linear", noise=0.0)
```

#### Common Pitfalls

⚠️ **NEVER use `sudo uv run python`** - it creates a root-owned venv that causes permission errors. Always use `sudo .venv/bin/python` or `sudo $(which python)` for scripts requiring uinput access.

---

### visionctl

Screen capture and LLM vision analysis library for Wayland, with Python bindings via pyo3.

#### Core Components

- `visionctl/src/lib.rs` - Main `VisionCtl` struct and public API, plus pyo3 bindings (behind `python` feature)
- `visionctl/src/screenshot.rs` - Wrapper for `grim` command to capture Wayland screenshots
- `visionctl/src/llm.rs` - HTTP client for LLM APIs (Ollama, vLLM, OpenAI-compatible)
- `visionctl/src/error.rs` - Error types
- `visionctl/src/bin/visionctl.rs` - CLI utility for querying LLMs about current screen

#### Key Design Decisions

- **Minimal dependencies**: Shells out to `grim` for screenshots instead of complex Wayland protocol handling
- **Blocking HTTP**: Uses reqwest blocking API for simplicity (no async complexity)
- **Multiple LLM backends**: Supports Ollama, vLLM, and OpenAI-compatible APIs
- **Feature-gated Python**: pyo3 bindings only compiled with `--features python`
- **Base64 encoding**: Images sent to LLMs as base64-encoded PNG data

#### System Requirements

- **grim** - Wayland screenshot tool (must be installed)
- **LLM backend** - Ollama, vLLM, or OpenAI-compatible API running locally or remotely

#### API Features

**Two main functions**:
1. `screenshot()` - Capture screenshot and return PNG bytes
2. `ask(question)` - Capture screenshot and query LLM

**Example (Rust)**:
```rust
use visionctl::{VisionCtl, LlmConfig};

let config = LlmConfig::Ollama {
    url: "http://localhost:11434".to_string(),
    model: "llava".to_string(),
};

let ctl = VisionCtl::new(config)?;
let answer = ctl.ask("What's on my screen?")?;
```

**Example (Python)**:
```python
from visionctl import VisionCtl

ctl = VisionCtl(
    backend="ollama",
    url="http://localhost:11434",
    model="llava"
)

answer = ctl.ask("What's on my screen?")
print(answer)
```

**Example (CLI)**:
```bash
# Using environment variables
export VISIONCTL_BACKEND=ollama
export VISIONCTL_URL=http://localhost:11434
export VISIONCTL_MODEL=llava

visionctl "What applications are open?"
# Output: {"question": "What applications are open?", "answer": "..."}
```

#### LLM Backend Configuration

**Ollama** (default):
- `VISIONCTL_BACKEND=ollama`
- `VISIONCTL_URL=http://localhost:11434`
- `VISIONCTL_MODEL=llava` (or other vision model)

**vLLM**:
- `VISIONCTL_BACKEND=vllm`
- `VISIONCTL_URL=http://localhost:8000`
- `VISIONCTL_MODEL=<model-name>`
- `VISIONCTL_API_KEY=<optional-key>`

**OpenAI-compatible**:
- `VISIONCTL_BACKEND=openai`
- `VISIONCTL_URL=<api-endpoint>`
- `VISIONCTL_MODEL=<model-name>`
- `VISIONCTL_API_KEY=<required-key>`
