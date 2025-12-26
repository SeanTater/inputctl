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
sudo $(which python) -m pytest python_tests/ -v
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

### Key Design Decisions

- **No daemon needed**: Unlike the C version, this library holds the uinput device handle in memory. The ~1 second initialization delay happens once when `InputCtl::new()` is called.
- **Uses evdev crate**: Wraps Linux uinput via the `evdev` crate
- **Feature-gated Python**: pyo3 bindings only compiled with `--features python`

### Permission Requirements

Requires access to `/dev/uinput`:
- Run as root, OR
- Add user to `input` group with appropriate udev rules
