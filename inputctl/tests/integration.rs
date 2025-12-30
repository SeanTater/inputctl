//! Integration tests for inputctl
//!
//! These tests require access to /dev/uinput and are marked #[ignore].
//! Run with: sudo cargo test -- --ignored

use evdev::Key;
use inputctl::{Curve, MouseButton, InputCtl};

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

#[test]
#[ignore = "requires /dev/uinput access (run with sudo)"]
fn mouse_move_smooth_linear() {
    let mut yd = InputCtl::new().expect("failed to create device");
    let result = yd.move_mouse_smooth(100, 50, 0.5, Curve::Linear, 2.0);
    assert!(result.is_ok(), "should move mouse smoothly: {:?}", result.err());
}

#[test]
#[ignore = "requires /dev/uinput access (run with sudo)"]
fn mouse_move_smooth_ease_in_out() {
    let mut yd = InputCtl::new().expect("failed to create device");
    let result = yd.move_mouse_smooth(100, 50, 1.0, Curve::EaseInOut, 2.0);
    assert!(result.is_ok(), "should move mouse smoothly with ease-in-out: {:?}", result.err());
}

#[test]
#[ignore = "requires /dev/uinput access (run with sudo)"]
fn mouse_move_smooth_negative() {
    let mut yd = InputCtl::new().expect("failed to create device");
    let result = yd.move_mouse_smooth(-50, -30, 0.3, Curve::Linear, 2.0);
    assert!(result.is_ok(), "should move mouse smoothly in negative direction: {:?}", result.err());
}

#[test]
#[ignore = "requires /dev/uinput access (run with sudo)"]
fn mouse_move_smooth_no_noise() {
    let mut yd = InputCtl::new().expect("failed to create device");
    let result = yd.move_mouse_smooth(100, 50, 1.0, Curve::Linear, 0.0);
    assert!(result.is_ok(), "should move mouse smoothly with no noise: {:?}", result.err());
}

#[test]
#[ignore = "requires /dev/uinput access (run with sudo)"]
fn test_hold_state_tracking() {
    let mut ctl = InputCtl::new().expect("failed to create device");
    assert!(!ctl.is_key_held(Key::KEY_A), "key should not be held initially");

    ctl.key_down(Key::KEY_A).expect("should press key down");
    assert!(ctl.is_key_held(Key::KEY_A), "key should be held after key_down");

    ctl.key_up(Key::KEY_A).expect("should release key");
    assert!(!ctl.is_key_held(Key::KEY_A), "key should not be held after key_up");
}

#[test]
#[ignore = "requires /dev/uinput access (run with sudo)"]
fn test_mouse_button_hold_state() {
    let mut ctl = InputCtl::new().expect("failed to create device");
    assert!(!ctl.is_mouse_button_held(MouseButton::Left), "button should not be held initially");

    ctl.mouse_down(MouseButton::Left).expect("should press button down");
    assert!(ctl.is_mouse_button_held(MouseButton::Left), "button should be held after mouse_down");

    ctl.mouse_up(MouseButton::Left).expect("should release button");
    assert!(!ctl.is_mouse_button_held(MouseButton::Left), "button should not be held after mouse_up");
}

#[test]
#[ignore = "requires /dev/uinput access (run with sudo)"]
fn test_release_all() {
    let mut ctl = InputCtl::new().expect("failed to create device");

    // Hold multiple keys and buttons
    ctl.key_down(Key::KEY_LEFTSHIFT).expect("should press shift");
    ctl.key_down(Key::KEY_LEFTCTRL).expect("should press ctrl");
    ctl.mouse_down(MouseButton::Left).expect("should press left button");

    assert_eq!(ctl.get_held_keys().len(), 2, "should have 2 held keys");
    assert_eq!(ctl.get_held_buttons().len(), 1, "should have 1 held button");

    // Release all
    ctl.release_all().expect("should release all");

    assert!(!ctl.is_key_held(Key::KEY_LEFTSHIFT), "shift should not be held");
    assert!(!ctl.is_key_held(Key::KEY_LEFTCTRL), "ctrl should not be held");
    assert!(!ctl.is_mouse_button_held(MouseButton::Left), "left button should not be held");
    assert_eq!(ctl.get_held_keys().len(), 0, "should have no held keys");
    assert_eq!(ctl.get_held_buttons().len(), 0, "should have no held buttons");
}

#[test]
#[ignore = "requires /dev/uinput access (run with sudo)"]
fn test_idempotent_key_down() {
    let mut ctl = InputCtl::new().expect("failed to create device");

    // Press the same key twice
    ctl.key_down(Key::KEY_A).expect("first key_down should succeed");
    ctl.key_down(Key::KEY_A).expect("second key_down should not error");

    assert_eq!(ctl.get_held_keys().len(), 1, "should only track key once");
    assert!(ctl.is_key_held(Key::KEY_A), "key should still be held");
}

#[test]
#[ignore = "requires /dev/uinput access (run with sudo)"]
fn test_idempotent_key_up() {
    let mut ctl = InputCtl::new().expect("failed to create device");

    ctl.key_down(Key::KEY_A).expect("should press key");
    ctl.key_up(Key::KEY_A).expect("first key_up should succeed");
    ctl.key_up(Key::KEY_A).expect("second key_up should not error");

    assert!(!ctl.is_key_held(Key::KEY_A), "key should not be held");
}

#[test]
#[ignore = "requires /dev/uinput access (run with sudo)"]
fn test_get_held_keys() {
    let mut ctl = InputCtl::new().expect("failed to create device");

    assert!(ctl.get_held_keys().is_empty(), "should start with no held keys");

    ctl.key_down(Key::KEY_A).expect("should press A");
    ctl.key_down(Key::KEY_B).expect("should press B");
    ctl.key_down(Key::KEY_LEFTSHIFT).expect("should press shift");

    let held = ctl.get_held_keys();
    assert_eq!(held.len(), 3, "should have 3 held keys");
    assert!(held.contains(&Key::KEY_A), "should contain KEY_A");
    assert!(held.contains(&Key::KEY_B), "should contain KEY_B");
    assert!(held.contains(&Key::KEY_LEFTSHIFT), "should contain KEY_LEFTSHIFT");
}

#[test]
#[ignore = "requires /dev/uinput access (run with sudo)"]
fn test_get_held_buttons() {
    let mut ctl = InputCtl::new().expect("failed to create device");

    assert!(ctl.get_held_buttons().is_empty(), "should start with no held buttons");

    ctl.mouse_down(MouseButton::Left).expect("should press left");
    ctl.mouse_down(MouseButton::Right).expect("should press right");

    let held = ctl.get_held_buttons();
    assert_eq!(held.len(), 2, "should have 2 held buttons");
    assert!(held.contains(&MouseButton::Left), "should contain Left");
    assert!(held.contains(&MouseButton::Right), "should contain Right");
}
