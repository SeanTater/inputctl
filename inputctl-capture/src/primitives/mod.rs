//! Low-level primitives for screen interaction.
//!
//! This module contains core data structures and functions for screen capture,
//! cursors, regions, and window management on the Linux desktop.

pub mod cursor;
pub mod frame_ops;
pub mod grid;
pub mod screen;
pub mod screenshot;

pub use cursor::find_cursor;
pub use grid::CursorPos;
pub use screen::{
    find_window, get_screen_dimensions, list_windows, Region, ScreenDimensions, Window,
};
pub use screenshot::{
    capture_raw_frame_with_source, capture_screenshot, capture_screenshot_image,
    capture_screenshot_image_with_source, capture_screenshot_raw_with_source,
    capture_screenshot_simple, capture_screenshot_with_source,
    capture_screenshot_with_source_default_timeout, ScreenshotOptions,
};
