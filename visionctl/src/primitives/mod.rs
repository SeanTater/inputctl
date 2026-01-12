//! Low-level primitives for screen interaction.
//!
//! This module contains core data structures and functions for screen capture,
//! cursors, regions, and window management on the Linux desktop.

pub mod grid;
pub mod screenshot;
pub mod cursor;
pub mod screen;

pub use grid::CursorPos;
pub use screenshot::{ScreenshotOptions, capture_screenshot, capture_screenshot_simple};
pub use cursor::find_cursor;
pub use screen::{get_screen_dimensions, Region, Window, list_windows, find_window};
