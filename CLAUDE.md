# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Build Rust library
cargo build --release

# Run Rust tests (unit tests only)
cargo test

# Run Rust integration tests (requires sudo for /dev/uinput)
sudo cargo test -- --ignored

# Build Python wheel
maturin build --release

# Install locally for development (uses uv)
maturin develop --uv

# Run Python tests
uv run pytest python_tests/ -v -m "not integration"

# Run Python integration tests (requires sudo)
# IMPORTANT: Use .venv/bin/python, NOT "sudo uv run python"
# (sudo uv run creates a root-owned environment)
sudo .venv/bin/python -m pytest python_tests/ -v

# Clean rebuild (if venv gets corrupted)
./rebuild_and_demo.sh
```

## Setup

```bash
# Install maturin as a uv tool
uv tool install maturin

# Sync dev dependencies and build
uv sync
maturin develop --uv
```

## Architecture

This is a Rust library for Linux input automation using uinput, with Python bindings via pyo3.

### Core Components

- `src/lib.rs` - Main `InputCtl` struct and public API, plus pyo3 bindings (behind `python` feature)
- `src/device.rs` - Creates the uinput virtual device with keyboard/mouse capabilities
- `src/keyboard.rs` - ASCII-to-keycode mapping for US keyboard layout
- `src/mouse.rs` - Mouse button enum and key mappings
- `src/interpolation.rs` - Smooth mouse movement with low-frequency Perlin-like noise

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

### Common Pitfalls

⚠️ **NEVER use `sudo uv run python`** - it creates a root-owned venv that causes permission errors. Always use `sudo .venv/bin/python` or `sudo $(which python)` for scripts requiring uinput access.
