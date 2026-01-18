//! GUI action execution.
//!
//! This module provides the capability to perform physical actions on the
//! desktop, such as clicking, moving the mouse, and typing text, primarily
//! by wrapping the `inputctl` command-line utility.

pub mod input;

pub use input::{
    click, double_click, key_down, key_press, key_up, mouse_down, mouse_up, move_to_pixel, scroll,
    type_text, MouseButton,
};
