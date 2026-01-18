# inputctl

Linux input automation library using uinput, with Python bindings.

## Repo layout

- `inputctl/`: input automation library (this README)
- `visionctl/`: Rust agent system for screen capture + automation
- `reflex_train/`: training stack for the Reflex Agent (separate Python project)
- `reflex_infer/`: Rust ONNX inference stub for Reflex Agent exports
- `docs/decisions.md`: short architecture notes and rationale

Inspired by [ydotool](https://github.com/ReimuNotMoe/ydotool) but designed as a library rather than a CLI tool. Works on X11, Wayland, and headless systems.

## Features

- **No daemon required** - holds the uinput device handle in memory
- **Python bindings** via pyo3/maturin
- **Simple API** - type text, click, move mouse, scroll
- **Smooth mouse movement** with configurable noise and interpolation
- **Key state tracking** - track held keys and mouse buttons
- **Cursor position tracking** - via KWin DBus interface (KDE Plasma only)
- **Servo control** - automatic compensation for pointer acceleration

## Installation

Requires access to `/dev/uinput` (run as root or configure udev rules).

### System dependencies (visionctl capture)

Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly \
  gstreamer1.0-libav \
  libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev \
  xdg-desktop-portal
```

Arch:
```bash
sudo pacman -S --needed \
  gstreamer \
  gst-plugins-base \
  gst-plugins-good \
  gst-plugins-bad \
  gst-plugins-ugly \
  gst-libav \
  xdg-desktop-portal
```

Fedora:
```bash
sudo dnf install -y \
  gstreamer1 \
  gstreamer1-plugins-base \
  gstreamer1-plugins-good \
  gstreamer1-plugins-bad-free \
  gstreamer1-plugins-ugly \
  gstreamer1-libav \
  gstreamer1-devel \
  gstreamer1-plugins-base-devel \
  xdg-desktop-portal
```

### Python build

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

# Smooth mouse movement with natural variation
yd.move_mouse_smooth(100, 50, 1.0, "ease-in-out", noise=2.0)

# Check held keys
yd.key_down("shift")
print(yd.is_key_held("shift"))  # True

# Release all held keys and buttons
yd.release_all()
```

## Rust Usage

```rust
use inputctl::{InputCtl, MouseButton};

let mut yd = InputCtl::new()?;
yd.type_text("Hello!")?;
yd.click(MouseButton::Left)?;
yd.move_mouse(100, 50)?;

// Smooth movement with servo feedback
yd.move_mouse_smooth(200, 100, 1.5, Curve::EaseInOut, 2.0, 60)?;

// Track held keys
yd.key_down(Key::KEY_A)?;
assert!(yd.is_key_held(Key::KEY_A));

// Release all
yd.release_all()?;
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

## Architecture

### Core Components

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
- **Servo control**: Automatic compensation for pointer acceleration via cursor position feedback

### Smooth Mouse Movement

The `move_mouse_smooth` function provides realistic mouse movement:

- **Parameters**: `dx`, `dy`, `duration`, `curve` ("linear" or "ease-in-out"), `noise` (default 2.0), `fps` (default 60)
- **Noise amount**: 0.0 = perfectly smooth, 2.0 = subtle (default), 5.0 = more obvious
- **Uses cubic interpolation** with control points every ~200ms for natural wavering
- **Always hits exact target** despite noise via servo feedback

**Example (Python)**:
```python
from inputctl import InputCtl
ctl = InputCtl()
# Subtle natural variation (default)
ctl.move_mouse_smooth(100, 50, 1.0, "ease-in-out", noise=2.0)
# Perfectly smooth
ctl.move_mouse_smooth(100, 50, 1.0, "linear", noise=0.0)
```

### Key State Tracking

The library tracks held keys and mouse buttons:

```python
ctl = InputCtl()
ctl.key_down("shift")
ctl.mouse_down("left")

# Check state
print(ctl.is_key_held("shift"))  # True
print(ctl.is_mouse_button_held("left"))  # True

# Get all held items
print(ctl.get_held_keys())
print(ctl.get_held_buttons())

# Release all
ctl.release_all()
```

### Cursor Position Tracking

The library tracks cursor position via KWin DBus interface (KDE Plasma only):

```python
ctl = InputCtl()

# Get current cursor position
x, y = ctl.cursor_pos()

# Check if cursor data is stale
if ctl.cursor_is_stale(1000):  # 1 second timeout
    print("Cursor tracking not available")
```

### Servo Control

The `move_to_position` function uses servo feedback to compensate for pointer acceleration:

```python
ctl = InputCtl()
ctl.move_to_position(100, 100, 2)  # Move to (100, 100) with 2-pixel tolerance
```

## API Reference

### Python API

#### Device Management
- `InputCtl()` - Create new input device
- `cursor_pos()` - Get current cursor position
- `cursor_is_stale(max_age_ms)` - Check if cursor data is stale

#### Text Input
- `type_text(text, key_delay_ms=20, key_hold_ms=20)` - Type text with configurable delays

#### Mouse Operations
- `move_mouse(dx, dy)` - Move mouse by relative amount
- `move_mouse_smooth(dx, dy, duration, curve="linear", noise=2.0, fps=60)` - Smooth movement with interpolation
- `move_to_position(target_x, target_y, tolerance)` - Move to absolute position with servo feedback
- `click(button="left")` - Click mouse button
- `scroll(amount)` - Scroll mouse wheel (positive=up)
- `scroll_horizontal(amount)` - Horizontal scroll

#### Key Operations
- `key_press(key_name)` - Press and release a key
- `key_down(key_name)` - Hold a key down
- `key_up(key_name)` - Release a key
- `is_key_held(key_name)` - Check if key is held
- `get_held_keys()` - Get all held keys

#### Mouse Button Operations
- `mouse_down(button)` - Hold mouse button down
- `mouse_up(button)` - Release mouse button
- `is_mouse_button_held(button)` - Check if button is held
- `get_held_buttons()` - Get all held buttons

#### State Management
- `release_all()` - Release all held keys and buttons
- `get_held_keys()` - Get list of held keys
- `get_held_buttons()` - Get list of held buttons

### Rust API

See `inputctl/src/lib.rs` for full Rust API documentation.

## Examples

### Basic Mouse Movement

```python
import inputctl

ctl = inputctl.InputCtl()
ctl.move_mouse(100, 50)  # Move right 100, down 50
ctl.click("left")
```

### Smooth Movement with Natural Variation

```python
ctl = inputctl.InputCtl()
ctl.move_mouse_smooth(200, 100, 1.5, "ease-in-out", noise=2.0)
```

### Key State Tracking

```python
ctl = inputctl.InputCtl()
ctl.key_down("shift")
ctl.key_down("a")
ctl.key_up("a")
ctl.key_up("shift")
```

### Servo Control Example

```python
ctl = inputctl.InputCtl()
ctl.move_to_position(500, 300, 2)  # Move to (500, 300) with 2-pixel tolerance
```

## Common Pitfalls

⚠️ **NEVER use `sudo uv run python`** - it creates a root-owned venv that causes permission errors. Always use `sudo .venv/bin/python` or `sudo $(which python)` for scripts requiring uinput access.

## Related Projects

- [visionctl](visionctl/README.md) - LLM-based GUI automation toolkit for KDE Plasma
- [ydotool](https://github.com/ReimuNotMoe/ydotool) - Inspiration for this project
- [xdotool](https://github.com/jordansissel/xdotool) - X11 automation tool

## License

MIT

## Support

For issues and questions, please file a GitHub issue or contact the maintainers.

## Changelog

See git log for recent changes:

```bash
git log --oneline -20
```

Recent highlights:
- Added smooth mouse movement with interpolation
- Added key state tracking and hold state management
- Added cursor position tracking via KWin
- Added servo control for pointer acceleration compensation
- Improved Python bindings with better error handling
