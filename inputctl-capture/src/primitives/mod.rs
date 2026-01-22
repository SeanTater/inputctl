//! Low-level primitives for screen interaction.
//!
//! This module contains core data structures and functions for screen capture,
//! cursors, regions, and window management on the Linux desktop.

pub mod cursor;
pub mod grid;
pub mod screen;
pub mod screenshot;

pub use cursor::find_cursor;
pub use grid::CursorPos;
pub use screen::{
    find_window, get_screen_dimensions, list_windows, Region, ScreenDimensions, Window,
};
pub use screenshot::{
    capture_screenshot, capture_screenshot_image, capture_screenshot_simple, ScreenshotOptions,
};
