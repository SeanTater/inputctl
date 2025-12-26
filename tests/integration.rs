//! Integration tests for inputctl
//!
//! These tests require access to /dev/uinput and are marked #[ignore].
//! Run with: sudo cargo test -- --ignored

use inputctl::{MouseButton, InputCtl};

#[test]
#[ignore = "requires /dev/uinput access (run with sudo)"]
fn create_device() {
    let yd = InputCtl::new();
    assert!(yd.is_ok(), "should create device: {:?}", yd.err());
}

#[test]
#[ignore = "requires /dev/uinput access (run with sudo)"]
fn type_text() {
    let mut yd = InputCtl::new().expect("failed to create device");
    let result = yd.type_text("hello");
    assert!(result.is_ok(), "should type text: {:?}", result.err());
}

#[test]
#[ignore = "requires /dev/uinput access (run with sudo)"]
fn mouse_click() {
    let mut yd = InputCtl::new().expect("failed to create device");
    let result = yd.click(MouseButton::Left);
    assert!(result.is_ok(), "should click: {:?}", result.err());
}

#[test]
#[ignore = "requires /dev/uinput access (run with sudo)"]
fn mouse_move() {
    let mut yd = InputCtl::new().expect("failed to create device");
    let result = yd.move_mouse(10, 20);
    assert!(result.is_ok(), "should move mouse: {:?}", result.err());
}

#[test]
#[ignore = "requires /dev/uinput access (run with sudo)"]
fn scroll() {
    let mut yd = InputCtl::new().expect("failed to create device");
    let result = yd.scroll(3);
    assert!(result.is_ok(), "should scroll: {:?}", result.err());
}
