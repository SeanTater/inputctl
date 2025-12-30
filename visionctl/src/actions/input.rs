use crate::error::{Error, Result};
use crate::primitives::{GridConfig, grid_to_pixel, find_cursor};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

// Lazy-initialized global InputCtl instance
static INPUT_CTL: OnceLock<Mutex<inputctl::InputCtl>> = OnceLock::new();

/// Get or initialize the global InputCtl instance
fn get_input_ctl() -> Result<&'static Mutex<inputctl::InputCtl>> {
    // Use get_or_init with panic on error (InputCtl initialization should not fail in normal operation)
    Ok(INPUT_CTL.get_or_init(|| {
        Mutex::new(inputctl::InputCtl::new().expect("Failed to initialize InputCtl - check /dev/uinput permissions"))
    }))
}

/// Mouse button enum (re-exported from inputctl)
pub use inputctl::MouseButton;

/// Easing curve for smooth movement (re-exported from inputctl)
pub use inputctl::Curve;

/// Click at pixel coordinates
///
/// Finds current cursor position, moves to target, and clicks
pub fn click_at_pixel(x: i32, y: i32, button: MouseButton) -> Result<()> {
    let ctl = get_input_ctl()?;
    let mut ctl = ctl.lock().unwrap();

    // Find current cursor position
    let cursor = find_cursor()?;

    // Calculate relative movement
    let dx = x - cursor.x;
    let dy = y - cursor.y;

    // Move smoothly to target
    ctl.move_mouse_smooth(dx, dy, 0.3, Curve::EaseInOut, 2.0)
        .map_err(|e| Error::ScreenshotFailed(format!("Mouse movement failed: {}", e)))?;

    // Small delay before clicking
    std::thread::sleep(Duration::from_millis(50));

    // Click
    ctl.click(button)
        .map_err(|e| Error::ScreenshotFailed(format!("Mouse click failed: {}", e)))?;

    Ok(())
}

/// Move mouse to pixel coordinates
pub fn move_to_pixel(x: i32, y: i32, smooth: bool) -> Result<()> {
    let ctl = get_input_ctl()?;
    let mut ctl = ctl.lock().unwrap();

    // Find current cursor position
    let cursor = find_cursor()?;

    // Calculate relative movement
    let dx = x - cursor.x;
    let dy = y - cursor.y;

    if smooth {
        ctl.move_mouse_smooth(dx, dy, 0.3, Curve::EaseInOut, 2.0)
            .map_err(|e| Error::ScreenshotFailed(format!("Mouse movement failed: {}", e)))?;
    } else {
        ctl.move_mouse(dx, dy)
            .map_err(|e| Error::ScreenshotFailed(format!("Mouse movement failed: {}", e)))?;
    }

    Ok(())
}

/// Click at grid cell
pub fn click_at_grid(cell: &str, config: &GridConfig, button: MouseButton) -> Result<()> {
    let (x, y) = grid_to_pixel(cell, config)?;
    click_at_pixel(x, y, button)
}

/// Move to grid cell
pub fn move_to_grid(cell: &str, config: &GridConfig, smooth: bool) -> Result<()> {
    let (x, y) = grid_to_pixel(cell, config)?;
    move_to_pixel(x, y, smooth)
}

/// Type text using keyboard
pub fn type_text(text: &str) -> Result<()> {
    let ctl = get_input_ctl()?;
    let mut ctl = ctl.lock().unwrap();

    ctl.type_text(text)
        .map_err(|e| Error::ScreenshotFailed(format!("Typing failed: {}", e)))
}

/// Press a key
///
/// For now, this is a simplified implementation that only handles single ASCII characters.
/// Full key support (enter, escape, etc.) will be added later.
pub fn key_press(key: &str) -> Result<()> {
    let ctl = get_input_ctl()?;
    let mut ctl = ctl.lock().unwrap();

    // Simple implementation: if it's a single character, type it
    // TODO: Add support for special keys (enter, escape, tab, etc.)
    if key.len() == 1 {
        ctl.type_text(key)
            .map_err(|e| Error::ScreenshotFailed(format!("Key press failed: {}", e)))
    } else {
        // For now, return error for non-character keys
        Err(Error::ScreenshotFailed(
            format!("Special key '{}' not yet supported - only single characters work for now", key)
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires /dev/uinput access
    fn test_get_input_ctl() {
        let result = get_input_ctl();
        assert!(result.is_ok());
    }
}
