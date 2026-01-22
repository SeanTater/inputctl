//! inputctl-capture: Screen capture primitives for Linux (Wayland via xdg-desktop-portal)
//!
//! This crate provides low-level screen capture functionality:
//! - Portal-based screenshot capture via PipeWire/GStreamer
//! - KWin cursor position tracking
//! - Screen/window geometry queries
//! - Video recording with input capture for training data

pub mod capture;
pub mod error;
pub mod primitives;
pub mod recorder;

// Re-export common types at crate root
pub use capture::{CaptureFrame, PortalCapture};
pub use error::{Error, Result};
pub use primitives::{
    capture_screenshot, capture_screenshot_image, capture_screenshot_simple, find_cursor,
    find_window, get_screen_dimensions, list_windows, CursorPos, Region, ScreenDimensions,
    ScreenshotOptions, Window,
};
pub use recorder::{run_recorder, RecorderConfig};
