mod cursor;
mod cursor_daemon;
mod device;
pub mod error;
mod interpolation;
pub mod keyboard;
pub mod mouse;

use cursor::CursorState;
use evdev::uinput::VirtualDevice;
use evdev::{InputEvent, RelativeAxisType};
pub use evdev::Key;
use std::collections::HashSet;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

pub use error::{Error, Result};
pub use interpolation::Curve;
pub use mouse::MouseButton;

/// Parse a key name string into an evdev Key
///
/// Supports:
/// - Special keys: "enter", "space", "tab", "backspace", "escape"
/// - Modifiers: "shift", "ctrl", "alt", "super"
/// - Navigation: "up", "down", "left", "right", "home", "end", "pageup", "pagedown"
/// - Function keys: "f1" through "f12"
/// - Single characters: "a"-"z", "0"-"9"
pub fn parse_key_name(name: &str) -> Result<Key> {
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
                'a' => Key::KEY_A,
                'b' => Key::KEY_B,
                'c' => Key::KEY_C,
                'd' => Key::KEY_D,
                'e' => Key::KEY_E,
                'f' => Key::KEY_F,
                'g' => Key::KEY_G,
                'h' => Key::KEY_H,
                'i' => Key::KEY_I,
                'j' => Key::KEY_J,
                'k' => Key::KEY_K,
                'l' => Key::KEY_L,
                'm' => Key::KEY_M,
                'n' => Key::KEY_N,
                'o' => Key::KEY_O,
                'p' => Key::KEY_P,
                'q' => Key::KEY_Q,
                'r' => Key::KEY_R,
                's' => Key::KEY_S,
                't' => Key::KEY_T,
                'u' => Key::KEY_U,
                'v' => Key::KEY_V,
                'w' => Key::KEY_W,
                'x' => Key::KEY_X,
                'y' => Key::KEY_Y,
                'z' => Key::KEY_Z,
                '0' => Key::KEY_0,
                '1' => Key::KEY_1,
                '2' => Key::KEY_2,
                '3' => Key::KEY_3,
                '4' => Key::KEY_4,
                '5' => Key::KEY_5,
                '6' => Key::KEY_6,
                '7' => Key::KEY_7,
                '8' => Key::KEY_8,
                '9' => Key::KEY_9,
                _ => return Err(Error::UnknownKey(name.to_string())),
            }
        }
        _ => return Err(Error::UnknownKey(name.to_string())),
    };
    Ok(key)
}

/// Virtual input device for keyboard and mouse automation
pub struct InputCtl {
    device: VirtualDevice,
    held_keys: HashSet<Key>,
    held_buttons: HashSet<MouseButton>,
    // Cursor tracking (optional - only on KDE/KWin)
    cursor_state: Arc<CursorState>,
    cursor_thread: Option<JoinHandle<()>>,
}

impl InputCtl {
    /// Create a new virtual input device
    ///
    /// Note: This takes ~1 second as the kernel needs time to recognize the device.
    /// Requires access to /dev/uinput (typically root or input group membership).
    ///
    /// Also starts a background thread for fast cursor position tracking via KWin.
    pub fn new() -> Result<Self> {
        let device = device::create_device()?;

        // Start cursor tracking daemon
        let cursor_state = Arc::new(CursorState::new());
        let state_clone = cursor_state.clone();
        let pid = std::process::id();

        let cursor_thread = thread::spawn(move || {
            if let Err(e) = cursor_daemon::run_daemon(state_clone, pid) {
                // Log error but don't fail - cursor tracking is optional
                eprintln!("[inputctl] Cursor daemon failed (KWin not available?): {}", e);
            }
        });

        let mut ctl = Self {
            device,
            held_keys: HashSet::new(),
            held_buttons: HashSet::new(),
            cursor_state,
            cursor_thread: Some(cursor_thread),
        };

        // Wait briefly for cursor daemon to initialize
        thread::sleep(Duration::from_millis(100));

        // Warm-up movement: The first movement after device creation can be
        // partially dropped by the kernel. Send a small dummy movement to ensure
        // the device is fully initialized and subsequent movements work correctly.
        let _ = ctl.move_mouse(1, 0);
        thread::sleep(Duration::from_millis(50));
        let _ = ctl.move_mouse(-1, 0);
        thread::sleep(Duration::from_millis(50));

        Ok(ctl)
    }

    /// Get current cursor position (instant, <1μs)
    ///
    /// Returns (x, y) screen coordinates. Returns (0, 0) if cursor tracking
    /// is not available (e.g., not running on KDE/KWin).
    pub fn cursor_pos(&self) -> (i32, i32) {
        self.cursor_state.get()
    }

    /// Check if cursor tracking data is stale
    ///
    /// Returns true if no cursor position updates have been received
    /// within the specified time. This can indicate that the KWin
    /// cursor tracking script has stopped working.
    pub fn cursor_is_stale(&self, max_age_ms: u64) -> bool {
        self.cursor_state.is_stale(max_age_ms)
    }

    /// Move mouse to absolute screen position using servo control
    ///
    /// Uses a feedback loop to compensate for pointer acceleration and other
    /// system effects. Queries cursor position after each movement and adjusts
    /// until the target is reached within the specified tolerance.
    ///
    /// # Arguments
    /// * `target_x` - Target X coordinate on screen
    /// * `target_y` - Target Y coordinate on screen
    /// * `tolerance` - Acceptable distance from target (e.g., 2 pixels)
    ///
    /// # Returns
    /// * `Ok(())` if target reached within tolerance
    /// * `Err` if target could not be reached after max attempts
    pub fn move_to_position(&mut self, target_x: i32, target_y: i32, tolerance: i32) -> Result<()> {
        const MAX_ATTEMPTS: usize = 20;
        const SERVO_INTERVAL_MS: u64 = 5;

        for _attempt in 0..MAX_ATTEMPTS {
            let (current_x, current_y) = self.cursor_pos();
            let dx = target_x - current_x;
            let dy = target_y - current_y;

            // Check if we're close enough
            if dx.abs() <= tolerance && dy.abs() <= tolerance {
                return Ok(());
            }

            // Move toward target
            // Use full distance for small corrections, partial for large moves
            let (move_x, move_y) = if dx.abs() > 100 || dy.abs() > 100 {
                // Large distance: move half to avoid overshoot
                (dx / 2, dy / 2)
            } else if dx.abs() > 20 || dy.abs() > 20 {
                // Medium distance: move 75%
                (dx * 3 / 4, dy * 3 / 4)
            } else {
                // Small distance: move full amount
                (dx, dy)
            };

            self.move_mouse(move_x, move_y)?;

            // Wait for cursor to settle and get new position
            thread::sleep(Duration::from_millis(SERVO_INTERVAL_MS));
        }

        Err(Error::DeviceError(format!(
            "Failed to reach target ({}, {}) after {} attempts",
            target_x, target_y, MAX_ATTEMPTS
        )))
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
        self.held_keys.insert(key);
        self.emit_key(key, 1)
    }

    /// Release a key
    pub fn key_up(&mut self, key: Key) -> Result<()> {
        self.held_keys.remove(&key);
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
        self.held_buttons.insert(button);
        self.emit_key(button.to_key(), 1)
    }

    /// Release a mouse button
    pub fn mouse_up(&mut self, button: MouseButton) -> Result<()> {
        self.held_buttons.remove(&button);
        self.emit_key(button.to_key(), 0)
    }

    /// Check if a key is currently held down
    pub fn is_key_held(&self, key: Key) -> bool {
        self.held_keys.contains(&key)
    }

    /// Check if a mouse button is currently held down
    pub fn is_mouse_button_held(&self, button: MouseButton) -> bool {
        self.held_buttons.contains(&button)
    }

    /// Get all currently held keys
    pub fn get_held_keys(&self) -> Vec<Key> {
        self.held_keys.iter().copied().collect()
    }

    /// Get all currently held mouse buttons
    pub fn get_held_buttons(&self) -> Vec<MouseButton> {
        self.held_buttons.iter().copied().collect()
    }

    /// Release all currently held keys and mouse buttons
    ///
    /// This is called automatically when InputCtl is dropped, but can also
    /// be called manually to ensure all inputs are released.
    pub fn release_all(&mut self) -> Result<()> {
        // Clone to avoid borrow issues while iterating and modifying
        let keys: Vec<Key> = self.held_keys.iter().copied().collect();
        let buttons: Vec<MouseButton> = self.held_buttons.iter().copied().collect();

        // Release modifiers first, then mouse buttons
        for key in keys {
            self.key_up(key)?;
        }
        for button in buttons {
            self.mouse_up(button)?;
        }
        Ok(())
    }

    /// Move the mouse by relative amount
    pub fn move_mouse(&mut self, dx: i32, dy: i32) -> Result<()> {
        let events = [
            InputEvent::new_now(evdev::EventType::RELATIVE, RelativeAxisType::REL_X.0, dx),
            InputEvent::new_now(evdev::EventType::RELATIVE, RelativeAxisType::REL_Y.0, dy),
            InputEvent::new_now(evdev::EventType::SYNCHRONIZATION, 0, 0), // SYN_REPORT
        ];
        self.device.emit(&events)?;
        Ok(())
    }

    /// Move mouse smoothly over a duration with interpolation
    ///
    /// Uses servo feedback to compensate for pointer acceleration and guarantee
    /// hitting the target. Movements include low-frequency smooth noise for natural variation.
    ///
    /// # Arguments
    /// * `dx` - Horizontal pixels to move (positive = right)
    /// * `dy` - Vertical pixels to move (positive = down)
    /// * `duration` - Time in seconds (e.g., 1.5)
    /// * `curve` - Interpolation curve type
    /// * `noise` - Maximum deviation in pixels (e.g., 2.0 = ±2 pixels). Use 0.0 for perfectly smooth.
    ///
    /// # Example
    /// ```no_run
    /// use inputctl::{InputCtl, Curve};
    /// let mut ctl = InputCtl::new().unwrap();
    /// // Move with ±2 pixel natural variation
    /// ctl.move_mouse_smooth(100, 50, 1.0, Curve::EaseInOut, 2.0).unwrap();
    /// // Move perfectly smooth with no noise
    /// ctl.move_mouse_smooth(100, 50, 1.0, Curve::Linear, 0.0).unwrap();
    /// ```
    pub fn move_mouse_smooth(
        &mut self,
        dx: i32,
        dy: i32,
        duration: f64,
        curve: Curve,
        noise: f64,
    ) -> Result<()> {
        // Get start position for servo feedback
        let (start_x, start_y) = self.cursor_pos();
        let target_x = start_x + dx;
        let target_y = start_y + dy;

        // Generate waypoints (cumulative offsets from start)
        let waypoints = interpolation::generate_waypoints(dx, dy, duration, curve, noise);
        let delay_ms = ((duration * 1000.0) / waypoints.len() as f64) as u64;

        for (waypoint_x, waypoint_y) in waypoints {
            // Calculate where cursor should be now (absolute position)
            let target_now_x = start_x + waypoint_x;
            let target_now_y = start_y + waypoint_y;

            // Get actual position and calculate servo correction
            let (actual_x, actual_y) = self.cursor_pos();
            let move_x = target_now_x - actual_x;
            let move_y = target_now_y - actual_y;

            self.move_mouse(move_x, move_y)?;
            thread::sleep(Duration::from_millis(delay_ms));
        }

        // Final servo correction to guarantee hitting exact target
        self.move_to_position(target_x, target_y, 2)?;

        Ok(())
    }

    /// Scroll the mouse wheel
    ///
    /// Positive values scroll up/right, negative scroll down/left
    pub fn scroll(&mut self, amount: i32) -> Result<()> {
        let events = [
            InputEvent::new_now(evdev::EventType::RELATIVE, RelativeAxisType::REL_WHEEL.0, amount),
            InputEvent::new_now(evdev::EventType::SYNCHRONIZATION, 0, 0), // SYN_REPORT
        ];
        self.device.emit(&events)?;
        Ok(())
    }

    /// Scroll horizontally
    pub fn scroll_horizontal(&mut self, amount: i32) -> Result<()> {
        let events = [
            InputEvent::new_now(evdev::EventType::RELATIVE, RelativeAxisType::REL_HWHEEL.0, amount),
            InputEvent::new_now(evdev::EventType::SYNCHRONIZATION, 0, 0), // SYN_REPORT
        ];
        self.device.emit(&events)?;
        Ok(())
    }

    /// Emit a raw key event
    fn emit_key(&mut self, key: Key, value: i32) -> Result<()> {
        let events = [
            InputEvent::new_now(evdev::EventType::KEY, key.code(), value),
            InputEvent::new_now(evdev::EventType::SYNCHRONIZATION, 0, 0), // SYN_REPORT
        ];
        self.device.emit(&events)?;
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

impl Drop for InputCtl {
    fn drop(&mut self) {
        // Best-effort cleanup - swallow errors since Drop can't return Result
        let _ = self.release_all();

        // Signal cursor daemon to shut down
        self.cursor_state.signal_shutdown();

        // Wait for cursor thread to finish (with timeout)
        if let Some(thread) = self.cursor_thread.take() {
            // Give it a short time to clean up
            let _ = thread.join();
        }
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

        /// Move mouse smoothly with interpolation
        ///
        /// Movements include low-frequency smooth noise for natural variation.
        ///
        /// # Arguments
        /// * `dx` - Horizontal pixels to move (positive = right)
        /// * `dy` - Vertical pixels to move (positive = down)
        /// * `duration` - Time in seconds (e.g., 1.5)
        /// * `curve` - Interpolation curve: "linear" or "ease-in-out" (default: "linear")
        /// * `noise` - Maximum deviation in pixels (default: 2.0). Use 0.0 for perfectly smooth.
        #[pyo3(signature = (dx, dy, duration, curve="linear", noise=2.0))]
        fn move_mouse_smooth(
            &mut self,
            dx: i32,
            dy: i32,
            duration: f64,
            curve: &str,
            noise: f64,
        ) -> PyResult<()> {
            use super::Curve;
            let curve_enum = match curve.to_lowercase().as_str() {
                "linear" => Curve::Linear,
                "ease-in-out" | "ease_in_out" | "easeinout" => Curve::EaseInOut,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "curve must be 'linear' or 'ease-in-out'",
                    ))
                }
            };

            self.inner
                .move_mouse_smooth(dx, dy, duration, curve_enum, noise)
                .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))
        }

        /// Scroll the mouse wheel (positive=up, negative=down)
        fn scroll(&mut self, amount: i32) -> PyResult<()> {
            self.inner.scroll(amount).map_err(|e| {
                pyo3::exceptions::PyOSError::new_err(e.to_string())
            })
        }

        /// Press a key by name (e.g., "enter", "space", "a")
        fn key_press(&mut self, key_name: &str) -> PyResult<()> {
            let key = py_parse_key_name(key_name)?;
            self.inner.key_click(key).map_err(|e| {
                pyo3::exceptions::PyOSError::new_err(e.to_string())
            })
        }

        /// Hold a key down
        fn key_down(&mut self, key_name: &str) -> PyResult<()> {
            let key = py_parse_key_name(key_name)?;
            self.inner.key_down(key).map_err(|e| {
                pyo3::exceptions::PyOSError::new_err(e.to_string())
            })
        }

        /// Release a key
        fn key_up(&mut self, key_name: &str) -> PyResult<()> {
            let key = py_parse_key_name(key_name)?;
            self.inner.key_up(key).map_err(|e| {
                pyo3::exceptions::PyOSError::new_err(e.to_string())
            })
        }

        /// Press a mouse button down
        fn mouse_down(&mut self, button: &str) -> PyResult<()> {
            let btn = parse_mouse_button(button)?;
            self.inner.mouse_down(btn).map_err(|e| {
                pyo3::exceptions::PyOSError::new_err(e.to_string())
            })
        }

        /// Release a mouse button
        fn mouse_up(&mut self, button: &str) -> PyResult<()> {
            let btn = parse_mouse_button(button)?;
            self.inner.mouse_up(btn).map_err(|e| {
                pyo3::exceptions::PyOSError::new_err(e.to_string())
            })
        }

        /// Check if a key is currently held down
        fn is_key_held(&self, key_name: &str) -> PyResult<bool> {
            let key = py_parse_key_name(key_name)?;
            Ok(self.inner.is_key_held(key))
        }

        /// Check if a mouse button is currently held down
        fn is_mouse_button_held(&self, button: &str) -> PyResult<bool> {
            let btn = parse_mouse_button(button)?;
            Ok(self.inner.is_mouse_button_held(btn))
        }

        /// Get all currently held keys (returns list of evdev key names like "KEY_A")
        fn get_held_keys(&self) -> Vec<String> {
            self.inner
                .get_held_keys()
                .iter()
                .map(|k| format!("{:?}", k))
                .collect()
        }

        /// Get all currently held mouse buttons
        fn get_held_buttons(&self) -> Vec<String> {
            self.inner
                .get_held_buttons()
                .iter()
                .map(|b| match b {
                    MouseButton::Left => "left",
                    MouseButton::Right => "right",
                    MouseButton::Middle => "middle",
                    MouseButton::Side => "side",
                    MouseButton::Extra => "extra",
                })
                .map(String::from)
                .collect()
        }

        /// Release all currently held keys and mouse buttons
        fn release_all(&mut self) -> PyResult<()> {
            self.inner.release_all().map_err(|e| {
                pyo3::exceptions::PyOSError::new_err(e.to_string())
            })
        }
    }

    fn py_parse_key_name(name: &str) -> PyResult<Key> {
        crate::parse_key_name(name).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn parse_mouse_button(name: &str) -> PyResult<MouseButton> {
        let button = match name.to_lowercase().as_str() {
            "left" => MouseButton::Left,
            "right" => MouseButton::Right,
            "middle" => MouseButton::Middle,
            "side" => MouseButton::Side,
            "extra" => MouseButton::Extra,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown mouse button: {}. Must be 'left', 'right', 'middle', 'side', or 'extra'",
                    name
                )))
            }
        };
        Ok(button)
    }

    #[pymodule]
    fn inputctl(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyInputCtl>()?;
        Ok(())
    }
}
