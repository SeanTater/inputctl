//! inputctl-capture: Screen capture primitives for Linux (Wayland via xdg-desktop-portal)
//!
//! This crate provides low-level screen capture functionality:
//! - Portal-based screenshot capture via PipeWire
//! - KWin cursor position tracking
//! - Screen/window geometry queries
//! - Video recording with input capture for training data

pub mod capture;
pub mod error;
pub mod primitives;
pub mod recorder;
pub mod recorder_ops;

// Re-export common types at crate root
pub use capture::{CaptureFrame, FrameSource, PortalCapture};
pub use error::{Error, Result};
pub use primitives::{
    capture_raw_frame_with_source, capture_screenshot, capture_screenshot_image,
    capture_screenshot_image_with_source, capture_screenshot_raw_with_source,
    capture_screenshot_simple, capture_screenshot_with_source,
    capture_screenshot_with_source_default_timeout, find_cursor, find_window,
    get_screen_dimensions, list_windows, CursorPos, Region, ScreenDimensions, ScreenshotData,
    ScreenshotOptions, Window,
};
pub use recorder::{
    run_recorder, run_recorder_with_sources, Encoder, FrameTiming, InputEvent, InputEventSource,
    RecorderConfig, RecorderSummary, VideoWriter,
};
