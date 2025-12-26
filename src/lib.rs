mod device;
pub mod error;
pub mod keyboard;
pub mod mouse;

use evdev::uinput::VirtualDevice;
use evdev::{InputEvent, Key, RelativeAxisType};
use std::thread;
use std::time::Duration;

pub use error::{Error, Result};
pub use mouse::MouseButton;

/// Virtual input device for keyboard and mouse automation
pub struct InputCtl {
    device: VirtualDevice,
}

impl InputCtl {
    /// Create a new virtual input device
    ///
    /// Note: This takes ~1 second as the kernel needs time to recognize the device.
    /// Requires access to /dev/uinput (typically root or input group membership).
    pub fn new() -> Result<Self> {
        let device = device::create_device()?;
        Ok(Self { device })
    }

    /// Type a string of text
    ///
    /// Uses US keyboard layout mapping. Characters that can't be typed are skipped.
    pub fn type_text(&mut self, text: &str) -> Result<()> {
        self.type_text_with_delay(text, 20, 20)
    }

    /// Type a string with custom delays
    ///
    /// - `key_delay_ms`: Delay between each key press
    /// - `key_hold_ms`: How long each key is held down
    pub fn type_text_with_delay(
        &mut self,
        text: &str,
        key_delay_ms: u64,
        key_hold_ms: u64,
    ) -> Result<()> {
        for c in text.chars() {
            if let Some((key, needs_shift)) = keyboard::ascii_to_key(c) {
                if needs_shift {
                    self.key_down(Key::KEY_LEFTSHIFT)?;
                }

                self.key_down(key)?;
                thread::sleep(Duration::from_millis(key_hold_ms));
                self.key_up(key)?;

                if needs_shift {
                    self.key_up(Key::KEY_LEFTSHIFT)?;
                }

                thread::sleep(Duration::from_millis(key_delay_ms));
            }
        }
        Ok(())
    }

    /// Press a key down
    pub fn key_down(&mut self, key: Key) -> Result<()> {
        self.emit_key(key, 1)
    }

    /// Release a key
    pub fn key_up(&mut self, key: Key) -> Result<()> {
        self.emit_key(key, 0)
    }

    /// Press and release a key
    pub fn key_click(&mut self, key: Key) -> Result<()> {
        self.key_down(key)?;
        self.key_up(key)
    }

    /// Click a mouse button (press and release)
    pub fn click(&mut self, button: MouseButton) -> Result<()> {
        self.mouse_down(button)?;
        self.mouse_up(button)
    }

    /// Press a mouse button down
    pub fn mouse_down(&mut self, button: MouseButton) -> Result<()> {
        self.emit_key(button.to_key(), 1)
    }

    /// Release a mouse button
    pub fn mouse_up(&mut self, button: MouseButton) -> Result<()> {
        self.emit_key(button.to_key(), 0)
    }

    /// Move the mouse by relative amount
    pub fn move_mouse(&mut self, dx: i32, dy: i32) -> Result<()> {
        let events = [
            InputEvent::new_now(evdev::EventType::RELATIVE, RelativeAxisType::REL_X.0, dx),
            InputEvent::new_now(evdev::EventType::RELATIVE, RelativeAxisType::REL_Y.0, dy),
        ];
        self.device.emit(&events)?;
        Ok(())
    }

    /// Scroll the mouse wheel
    ///
    /// Positive values scroll up/right, negative scroll down/left
    pub fn scroll(&mut self, amount: i32) -> Result<()> {
        let event = InputEvent::new_now(
            evdev::EventType::RELATIVE,
            RelativeAxisType::REL_WHEEL.0,
            amount,
        );
        self.device.emit(&[event])?;
        Ok(())
    }

    /// Scroll horizontally
    pub fn scroll_horizontal(&mut self, amount: i32) -> Result<()> {
        let event = InputEvent::new_now(
            evdev::EventType::RELATIVE,
            RelativeAxisType::REL_HWHEEL.0,
            amount,
        );
        self.device.emit(&[event])?;
        Ok(())
    }

    /// Emit a raw key event
    fn emit_key(&mut self, key: Key, value: i32) -> Result<()> {
        let event = InputEvent::new_now(evdev::EventType::KEY, key.code(), value);
        self.device.emit(&[event])?;
        Ok(())
    }

    /// Emit a raw input event (for power users)
    ///
    /// This allows sending any type of input event directly.
    pub fn emit_raw(&mut self, event_type: u16, code: u16, value: i32) -> Result<()> {
        let ev_type = evdev::EventType(event_type);
        let event = InputEvent::new_now(ev_type, code, value);
        self.device.emit(&[event])?;
        Ok(())
    }
}

// Python bindings
#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::prelude::*;

    #[pyclass(name = "InputCtl")]
    pub struct PyInputCtl {
        inner: InputCtl,
    }

    #[pymethods]
    impl PyInputCtl {
        #[new]
        fn new() -> PyResult<Self> {
            let inner = InputCtl::new().map_err(|e| {
                pyo3::exceptions::PyOSError::new_err(e.to_string())
            })?;
            Ok(Self { inner })
        }

        /// Type a string of text
        #[pyo3(signature = (text, key_delay_ms=20, key_hold_ms=20))]
        fn type_text(
            &mut self,
            text: &str,
            key_delay_ms: u64,
            key_hold_ms: u64,
        ) -> PyResult<()> {
            self.inner
                .type_text_with_delay(text, key_delay_ms, key_hold_ms)
                .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))
        }

        /// Click a mouse button
        #[pyo3(signature = (button="left"))]
        fn click(&mut self, button: &str) -> PyResult<()> {
            let btn = match button.to_lowercase().as_str() {
                "left" => MouseButton::Left,
                "right" => MouseButton::Right,
                "middle" => MouseButton::Middle,
                _ => return Err(pyo3::exceptions::PyValueError::new_err(
                    "button must be 'left', 'right', or 'middle'"
                )),
            };
            self.inner.click(btn).map_err(|e| {
                pyo3::exceptions::PyOSError::new_err(e.to_string())
            })
        }

        /// Move mouse by relative amount
        fn move_mouse(&mut self, dx: i32, dy: i32) -> PyResult<()> {
            self.inner.move_mouse(dx, dy).map_err(|e| {
                pyo3::exceptions::PyOSError::new_err(e.to_string())
            })
        }

        /// Scroll the mouse wheel (positive=up, negative=down)
        fn scroll(&mut self, amount: i32) -> PyResult<()> {
            self.inner.scroll(amount).map_err(|e| {
                pyo3::exceptions::PyOSError::new_err(e.to_string())
            })
        }

        /// Press a key by name (e.g., "enter", "space", "a")
        fn key_press(&mut self, key_name: &str) -> PyResult<()> {
            let key = parse_key_name(key_name)?;
            self.inner.key_click(key).map_err(|e| {
                pyo3::exceptions::PyOSError::new_err(e.to_string())
            })
        }

        /// Hold a key down
        fn key_down(&mut self, key_name: &str) -> PyResult<()> {
            let key = parse_key_name(key_name)?;
            self.inner.key_down(key).map_err(|e| {
                pyo3::exceptions::PyOSError::new_err(e.to_string())
            })
        }

        /// Release a key
        fn key_up(&mut self, key_name: &str) -> PyResult<()> {
            let key = parse_key_name(key_name)?;
            self.inner.key_up(key).map_err(|e| {
                pyo3::exceptions::PyOSError::new_err(e.to_string())
            })
        }
    }

    fn parse_key_name(name: &str) -> PyResult<Key> {
        let key = match name.to_lowercase().as_str() {
            "enter" | "return" => Key::KEY_ENTER,
            "space" => Key::KEY_SPACE,
            "tab" => Key::KEY_TAB,
            "backspace" => Key::KEY_BACKSPACE,
            "escape" | "esc" => Key::KEY_ESC,
            "shift" | "lshift" => Key::KEY_LEFTSHIFT,
            "rshift" => Key::KEY_RIGHTSHIFT,
            "ctrl" | "control" | "lctrl" => Key::KEY_LEFTCTRL,
            "rctrl" => Key::KEY_RIGHTCTRL,
            "alt" | "lalt" => Key::KEY_LEFTALT,
            "ralt" => Key::KEY_RIGHTALT,
            "super" | "meta" | "win" => Key::KEY_LEFTMETA,
            "up" => Key::KEY_UP,
            "down" => Key::KEY_DOWN,
            "left" => Key::KEY_LEFT,
            "right" => Key::KEY_RIGHT,
            "home" => Key::KEY_HOME,
            "end" => Key::KEY_END,
            "pageup" => Key::KEY_PAGEUP,
            "pagedown" => Key::KEY_PAGEDOWN,
            "insert" => Key::KEY_INSERT,
            "delete" => Key::KEY_DELETE,
            "f1" => Key::KEY_F1,
            "f2" => Key::KEY_F2,
            "f3" => Key::KEY_F3,
            "f4" => Key::KEY_F4,
            "f5" => Key::KEY_F5,
            "f6" => Key::KEY_F6,
            "f7" => Key::KEY_F7,
            "f8" => Key::KEY_F8,
            "f9" => Key::KEY_F9,
            "f10" => Key::KEY_F10,
            "f11" => Key::KEY_F11,
            "f12" => Key::KEY_F12,
            // Single characters
            s if s.len() == 1 => {
                let c = s.chars().next().unwrap();
                match c.to_ascii_lowercase() {
                    'a' => Key::KEY_A, 'b' => Key::KEY_B, 'c' => Key::KEY_C,
                    'd' => Key::KEY_D, 'e' => Key::KEY_E, 'f' => Key::KEY_F,
                    'g' => Key::KEY_G, 'h' => Key::KEY_H, 'i' => Key::KEY_I,
                    'j' => Key::KEY_J, 'k' => Key::KEY_K, 'l' => Key::KEY_L,
                    'm' => Key::KEY_M, 'n' => Key::KEY_N, 'o' => Key::KEY_O,
                    'p' => Key::KEY_P, 'q' => Key::KEY_Q, 'r' => Key::KEY_R,
                    's' => Key::KEY_S, 't' => Key::KEY_T, 'u' => Key::KEY_U,
                    'v' => Key::KEY_V, 'w' => Key::KEY_W, 'x' => Key::KEY_X,
                    'y' => Key::KEY_Y, 'z' => Key::KEY_Z,
                    '0' => Key::KEY_0, '1' => Key::KEY_1, '2' => Key::KEY_2,
                    '3' => Key::KEY_3, '4' => Key::KEY_4, '5' => Key::KEY_5,
                    '6' => Key::KEY_6, '7' => Key::KEY_7, '8' => Key::KEY_8,
                    '9' => Key::KEY_9,
                    _ => return Err(pyo3::exceptions::PyValueError::new_err(
                        format!("Unknown key: {}", name)
                    )),
                }
            }
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unknown key: {}", name)
            )),
        };
        Ok(key)
    }

    #[pymodule]
    fn inputctl(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyInputCtl>()?;
        Ok(())
    }
}
