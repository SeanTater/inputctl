# inputctl

Linux input automation library using uinput, with Python bindings.

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

### Python

```bash
cd inputctl && maturin develop --uv
```

### Rust

```toml
[dependencies]
inputctl = { path = "../inputctl" }
```

## Usage

### Python

```python
import inputctl

# Create device (~1 second initialization)
ctl = inputctl.InputCtl()

# Type text
ctl.type_text("Hello, World!")

# Mouse operations
ctl.move_mouse(100, 50)      # Move relative
ctl.click("left")            # Click
ctl.scroll(3)                # Scroll up

# Key operations
ctl.key_press("enter")
ctl.key_down("shift")
ctl.key_up("shift")

# Smooth mouse movement with natural variation
ctl.move_mouse_smooth(100, 50, 1.0, "ease-in-out", noise=2.0)

# Key state tracking
ctl.key_down("shift")
print(ctl.is_key_held("shift"))  # True
ctl.release_all()
```

### Rust

```rust
use inputctl::{InputCtl, MouseButton, Curve, Key};

let mut ctl = InputCtl::new()?;
ctl.type_text("Hello!")?;
ctl.click(MouseButton::Left)?;
ctl.move_mouse(100, 50)?;

// Smooth movement with servo feedback
ctl.move_mouse_smooth(200, 100, 1.5, Curve::EaseInOut, 2.0, 60)?;

// Track held keys
ctl.key_down(Key::KEY_A)?;
assert!(ctl.is_key_held(Key::KEY_A));
ctl.release_all()?;
```

## Running with sudo

Since `/dev/uinput` typically requires elevated privileges:

```bash
# Run Python script
sudo .venv/bin/python examples/mouse_demo.py

# Run tests
sudo cargo test -p inputctl -- --ignored
sudo .venv/bin/python -m pytest python_tests/ -v
```

**Warning:** Never use `sudo uv run python` - it creates a root-owned venv that causes permission errors.

## Architecture

### Core Components

- `src/lib.rs` - Main `InputCtl` struct and public API, plus pyo3 bindings
- `src/device.rs` - Creates the uinput virtual device with keyboard/mouse capabilities
- `src/keyboard.rs` - ASCII-to-keycode mapping for US keyboard layout
- `src/mouse.rs` - Mouse button enum and key mappings
- `src/interpolation.rs` - Smooth mouse movement with low-frequency noise

### Key Design Decisions

- **No daemon needed**: Holds the uinput device handle in memory. The ~1 second initialization delay happens once when `InputCtl::new()` is called.
- **Uses evdev crate**: Wraps Linux uinput via the `evdev` crate
- **Feature-gated Python**: pyo3 bindings only compiled with `--features python`
- **Smooth mouse movement**: Uses low-frequency Perlin-like noise with cubic interpolation for natural-looking movement paths

### Smooth Mouse Movement

The `move_mouse_smooth` function provides realistic mouse movement:

- **Parameters**: `dx`, `dy`, `duration`, `curve` ("linear" or "ease-in-out"), `noise` (default 2.0), `fps` (default 60)
- **Noise amount**: 0.0 = perfectly smooth, 2.0 = subtle (default), 5.0 = more obvious
- **Uses cubic interpolation** with control points every ~200ms for natural wavering
- **Always hits exact target** despite noise via servo feedback

## Python API

### Device Management
- `InputCtl()` - Create new input device
- `cursor_pos()` - Get current cursor position (KDE only)
- `cursor_is_stale(max_age_ms)` - Check if cursor data is stale

### Text Input
- `type_text(text, key_delay_ms=20, key_hold_ms=20)` - Type text with configurable delays

### Mouse Operations
- `move_mouse(dx, dy)` - Move mouse by relative amount
- `move_mouse_smooth(dx, dy, duration, curve, noise, fps)` - Smooth movement
- `move_to_position(target_x, target_y, tolerance)` - Move to absolute position with servo feedback
- `click(button)` - Click mouse button
- `scroll(amount)` - Scroll mouse wheel
- `scroll_horizontal(amount)` - Horizontal scroll

### Key Operations
- `key_press(key_name)` - Press and release a key
- `key_down(key_name)` / `key_up(key_name)` - Hold/release key
- `is_key_held(key_name)` - Check if key is held
- `get_held_keys()` - Get all held keys

### Mouse Button Operations
- `mouse_down(button)` / `mouse_up(button)` - Hold/release button
- `is_mouse_button_held(button)` - Check if button is held
- `get_held_buttons()` - Get all held buttons

### State Management
- `release_all()` - Release all held keys and buttons

## License

MIT
