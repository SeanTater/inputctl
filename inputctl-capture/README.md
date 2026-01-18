# inputctl-capture

Screen capture primitives for Linux (Wayland via xdg-desktop-portal).

## Features

- **Portal-based capture** - Works on Wayland using xdg-desktop-portal
- **GStreamer pipeline** - Efficient video processing
- **KWin cursor tracking** - Get cursor position via DBus (KDE Plasma)
- **Recording** - Capture video + input events for training data

## Installation

### System Dependencies

Ubuntu/Debian:
```bash
sudo apt-get install -y \
  gstreamer1.0-tools gstreamer1.0-plugins-{base,good,bad,ugly} \
  gstreamer1.0-libav libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  xdg-desktop-portal
```

Arch:
```bash
sudo pacman -S --needed gstreamer gst-plugins-{base,good,bad,ugly} gst-libav xdg-desktop-portal
```

### Rust

```toml
[dependencies]
inputctl-capture = { path = "../inputctl-capture" }
```

## Usage

### Screenshot

```rust
use inputctl_capture::{capture_screenshot, ScreenshotOptions};

// Simple screenshot
let png_bytes = inputctl_capture::capture_screenshot_simple()?;

// Screenshot with options
let options = ScreenshotOptions {
    mark_cursor: true,
    crop_region: None,
    resize_to_logical: Some((1920, 1080)),
};
let result = capture_screenshot(options)?;
std::fs::write("screenshot.png", result.png_bytes)?;
```

### Cursor Position

```rust
use inputctl_capture::find_cursor;

// Get cursor position via KWin DBus (KDE Plasma only)
let pos = find_cursor()?;
println!("Cursor at ({}, {})", pos.x, pos.y);
```

### Screen Dimensions

```rust
use inputctl_capture::get_screen_dimensions;

let dims = get_screen_dimensions()?;
println!("Screen: {}x{}", dims.width, dims.height);
```

### Window Management

```rust
use inputctl_capture::{find_window, list_windows};

// Find window by title
if let Some(window) = find_window("Firefox")? {
    println!("Firefox at ({}, {})", window.x, window.y);
}

// List all windows
for window in list_windows()? {
    println!("{}: {}x{}", window.caption, window.width, window.height);
}
```

### Streaming Capture

```rust
use inputctl_capture::PortalCapture;

let mut capture = PortalCapture::new()?;

// Get frames in a loop
loop {
    if let Some(frame) = capture.try_get_frame()? {
        // frame.data contains BGRA pixels
        println!("Frame: {}x{}", frame.width, frame.height);
    }
}
```

## Recording CLI

The `inputctl-record` binary captures video and input events for training:

```bash
inputctl-record --output dataset --fps 10 --preset ultrafast

# With options
inputctl-record \
  --output dataset \
  --fps 10 \
  --preset ultrafast \
  --crf 23 \
  --device /dev/input/event5 \
  --region 0,0,1920,1080 \
  --max-seconds 300
```

This creates a dataset directory with:
- `video.mp4` - Screen recording
- `events.jsonl` - Input events (keyboard, mouse)
- `metadata.json` - Recording metadata

## API

### Screenshot Functions
- `capture_screenshot(options)` - Full-featured screenshot
- `capture_screenshot_simple()` - Quick screenshot returning PNG bytes
- `capture_screenshot_image(options)` - Returns DynamicImage

### Cursor Functions
- `find_cursor()` - Get cursor position (KDE Plasma only)
- `CursorPos` - Struct with x, y coordinates

### Screen Functions
- `get_screen_dimensions()` - Get screen size
- `ScreenDimensions` - Struct with width, height

### Window Functions
- `find_window(name)` - Find window by title substring
- `list_windows()` - List all windows
- `Window` - Struct with caption, x, y, width, height

### Streaming
- `PortalCapture` - Streaming frame capture
- `CaptureFrame` - Frame with BGRA data, dimensions, timestamp

### Recording
- `run_recorder(...)` - Start recording session

### Types
- `Region` - x, y, width, height for cropping
- `ScreenshotOptions` - mark_cursor, crop_region, resize_to_logical

## Architecture

### Portal Capture (`capture/portal.rs`)

Uses xdg-desktop-portal's ScreenCast interface:
1. Creates a portal session
2. Selects a screen source
3. Starts a PipeWire stream
4. Uses GStreamer to process frames

### KWin Cursor (`primitives/cursor.rs`)

Gets cursor position via KWin's DBus interface:
- Service: `org.kde.KWin`
- Path: `/org/kde/KWin`
- Interface: `org.kde.KWin.Cursors`

### Recorder (`recorder.rs`)

Combines portal capture with evdev input monitoring:
- Captures frames at target FPS
- Records keyboard/mouse events with timestamps
- Encodes video using x264 via GStreamer

## Platform Requirements

- **Wayland** with xdg-desktop-portal
- **KDE Plasma** for cursor position (optional but recommended)
- **GStreamer** with x264 encoder for recording

## License

MIT
