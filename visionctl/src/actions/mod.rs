//! GUI action execution.
//!
//! This module provides the capability to perform physical actions on the
//! desktop, such as clicking, moving the mouse, and typing text, primarily
//! by wrapping the `inputctl` command-line utility.

pub mod input;

pub use input::{
    MouseButton,
    click, move_to_pixel,
    type_text, key_press,
};
