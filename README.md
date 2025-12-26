# inputctl

Linux input automation library using uinput, with Python bindings.

Inspired by [ydotool](https://github.com/ReimuNotMoe/ydotool) but designed as a library rather than a CLI tool. Works on X11, Wayland, and headless systems.

## Features

- **No daemon required** - holds the uinput device handle in memory
- **Python bindings** via pyo3/maturin
- **Simple API** - type text, click, move mouse, scroll

## Installation

Requires access to `/dev/uinput` (run as root or configure udev rules).

```bash
# Install build tool
uv tool install maturin

# Clone and build
git clone <repo-url>
cd inputctl
uv sync
maturin develop --uv
```

## Python Usage

```python
import inputctl

# Create device (~1 second initialization)
yd = inputctl.InputCtl()

# Type text
yd.type_text("Hello, World!")

# Mouse operations
yd.move_mouse(100, 50)      # Move relative
yd.click("left")            # Click
yd.scroll(3)                # Scroll up

# Key operations
yd.key_press("enter")
yd.key_down("shift")
yd.key_up("shift")
```

## Rust Usage

```rust
use inputctl::{InputCtl, MouseButton};

let mut yd = InputCtl::new()?;
yd.type_text("Hello!")?;
yd.click(MouseButton::Left)?;
yd.move_mouse(100, 50)?;
```

## Running with sudo

Since `/dev/uinput` typically requires elevated privileges:

```bash
# Run Python example
sudo uv run python examples/mouse_demo.py

# Run tests
sudo cargo test -- --ignored
sudo $(which python) -m pytest python_tests/ -v
```

## Development

```bash
# Rust tests (no sudo needed for unit tests)
cargo test

# Python tests (no sudo needed for import tests)
uv run pytest python_tests/ -v -m "not integration"

# Build release wheel
maturin build --release
```

## License

MIT
