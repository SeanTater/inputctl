use evdev::uinput::{VirtualDevice, VirtualDeviceBuilder};
use evdev::{AttributeSet, Key, RelativeAxisType};
use std::thread;
use std::time::Duration;

use crate::error::Result;

/// Creates and configures a virtual input device
pub fn create_device() -> Result<VirtualDevice> {
    // Register all keys we might need
    let mut keys = AttributeSet::<Key>::new();

    // Letters
    for key in [
        Key::KEY_A,
        Key::KEY_B,
        Key::KEY_C,
        Key::KEY_D,
        Key::KEY_E,
        Key::KEY_F,
        Key::KEY_G,
        Key::KEY_H,
        Key::KEY_I,
        Key::KEY_J,
        Key::KEY_K,
        Key::KEY_L,
        Key::KEY_M,
        Key::KEY_N,
        Key::KEY_O,
        Key::KEY_P,
        Key::KEY_Q,
        Key::KEY_R,
        Key::KEY_S,
        Key::KEY_T,
        Key::KEY_U,
        Key::KEY_V,
        Key::KEY_W,
        Key::KEY_X,
        Key::KEY_Y,
        Key::KEY_Z,
    ] {
        keys.insert(key);
    }

    // Numbers
    for key in [
        Key::KEY_0,
        Key::KEY_1,
        Key::KEY_2,
        Key::KEY_3,
        Key::KEY_4,
        Key::KEY_5,
        Key::KEY_6,
        Key::KEY_7,
        Key::KEY_8,
        Key::KEY_9,
    ] {
        keys.insert(key);
    }

    // Punctuation and symbols
    for key in [
        Key::KEY_SPACE,
        Key::KEY_MINUS,
        Key::KEY_EQUAL,
        Key::KEY_LEFTBRACE,
        Key::KEY_RIGHTBRACE,
        Key::KEY_BACKSLASH,
        Key::KEY_SEMICOLON,
        Key::KEY_APOSTROPHE,
        Key::KEY_GRAVE,
        Key::KEY_COMMA,
        Key::KEY_DOT,
        Key::KEY_SLASH,
        Key::KEY_TAB,
        Key::KEY_ENTER,
        Key::KEY_BACKSPACE,
    ] {
        keys.insert(key);
    }

    // Modifiers
    for key in [
        Key::KEY_LEFTSHIFT,
        Key::KEY_RIGHTSHIFT,
        Key::KEY_LEFTCTRL,
        Key::KEY_RIGHTCTRL,
        Key::KEY_LEFTALT,
        Key::KEY_RIGHTALT,
        Key::KEY_LEFTMETA,
        Key::KEY_RIGHTMETA,
    ] {
        keys.insert(key);
    }

    // Function keys
    for key in [
        Key::KEY_ESC,
        Key::KEY_F1,
        Key::KEY_F2,
        Key::KEY_F3,
        Key::KEY_F4,
        Key::KEY_F5,
        Key::KEY_F6,
        Key::KEY_F7,
        Key::KEY_F8,
        Key::KEY_F9,
        Key::KEY_F10,
        Key::KEY_F11,
        Key::KEY_F12,
    ] {
        keys.insert(key);
    }

    // Navigation
    for key in [
        Key::KEY_UP,
        Key::KEY_DOWN,
        Key::KEY_LEFT,
        Key::KEY_RIGHT,
        Key::KEY_HOME,
        Key::KEY_END,
        Key::KEY_PAGEUP,
        Key::KEY_PAGEDOWN,
        Key::KEY_INSERT,
        Key::KEY_DELETE,
    ] {
        keys.insert(key);
    }

    // Mouse buttons
    for key in [
        Key::BTN_LEFT,
        Key::BTN_RIGHT,
        Key::BTN_MIDDLE,
        Key::BTN_SIDE,
        Key::BTN_EXTRA,
    ] {
        keys.insert(key);
    }

    // Relative axes for mouse movement
    let mut rel_axes = AttributeSet::<RelativeAxisType>::new();
    rel_axes.insert(RelativeAxisType::REL_X);
    rel_axes.insert(RelativeAxisType::REL_Y);
    rel_axes.insert(RelativeAxisType::REL_WHEEL);
    rel_axes.insert(RelativeAxisType::REL_HWHEEL);

    let device = VirtualDeviceBuilder::new()?
        .name("inputctl virtual device")
        .with_keys(&keys)?
        .with_relative_axes(&rel_axes)?
        .build()?;

    // Wait for the kernel to fully recognize the device
    // Without this delay, initial events may be lost
    thread::sleep(Duration::from_secs(1));

    Ok(device)
}
