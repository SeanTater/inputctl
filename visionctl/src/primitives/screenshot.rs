use crate::error::{Error, Result};
use crate::primitives::grid::{GridConfig, GridMetadata, CursorPos, draw_grid_overlay};
use std::collections::HashMap;
use std::io::Read;
use nix::unistd::pipe2;
use nix::fcntl::OFlag;
use zbus::blocking::Connection;
use zvariant::{OwnedFd as ZOwnedFd, Value};
use image::{ImageBuffer, RgbaImage, ImageFormat};

/// Options for screenshot capture
#[derive(Clone, Debug, Default)]
pub struct ScreenshotOptions {
    pub grid: Option<GridConfig>,
    pub mark_cursor: bool,
}

/// Screenshot data with metadata
#[derive(Clone, Debug)]
pub struct ScreenshotData {
    pub png_bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub grid: Option<GridMetadata>,
    pub cursor_pos: Option<CursorPos>,
}

/// Capture a screenshot using KWin DBus interface
///
/// Requires KDE Plasma 6.0+ with KWin compositor.
/// Returns PNG bytes without any overlays.
pub fn capture_screenshot_simple() -> Result<Vec<u8>> {
    let data = capture_screenshot(ScreenshotOptions::default())?;
    Ok(data.png_bytes)
}

/// Capture a screenshot with optional grid overlay and cursor marking
pub fn capture_screenshot(options: ScreenshotOptions) -> Result<ScreenshotData> {
    // Create pipe for receiving image data
    let (read_fd, write_fd) = pipe2(OFlag::O_CLOEXEC)
        .map_err(|e| Error::ScreenshotFailed(format!("Failed to create pipe: {}", e)))?;

    // Connect to DBus session bus
    let conn = Connection::session()
        .map_err(|e| Error::ScreenshotFailed(format!("DBus connection failed - is KDE running?: {}", e)))?;

    // Create proxy for KWin ScreenShot2 interface
    let proxy = zbus::blocking::Proxy::new(
        &conn,
        "org.kde.KWin.ScreenShot2",
        "/org/kde/KWin/ScreenShot2",
        "org.kde.KWin.ScreenShot2",
    ).map_err(|e| Error::ScreenshotFailed(format!("KWin DBus interface not found: {}", e)))?;

    // Prepare options (native resolution)
    let mut dbus_options: HashMap<&str, Value> = HashMap::new();
    dbus_options.insert("native-resolution", Value::new(true));

    // Convert FD for DBus
    let z_write_fd = ZOwnedFd::from(write_fd);

    // Call CaptureWorkspace (Plasma 6.0+)
    let response = proxy
        .call_method("CaptureWorkspace", &(dbus_options, z_write_fd))
        .map_err(|e| Error::ScreenshotFailed(format!("Screenshot capture failed: {}", e)))?;

    let body = response.body();
    let reply: HashMap<String, Value> = body
        .deserialize()
        .map_err(|e| Error::ScreenshotFailed(format!("Invalid response format: {}", e)))?;

    // Extract image metadata
    let width = extract_u32(&reply, "width")?;
    let height = extract_u32(&reply, "height")?;
    let stride = extract_u32(&reply, "stride")?;

    // Read raw image data from pipe
    let expected_size = (stride * height) as usize;
    let mut raw_data = vec![0u8; expected_size];
    let mut file = std::fs::File::from(read_fd);
    file.read_exact(&mut raw_data)
        .map_err(|e| Error::ScreenshotFailed(format!("Failed to read image data: {}", e)))?;

    // Convert ARGB32 to RGBA8
    let mut img = argb_to_rgba_image(&raw_data, width, height, stride)?;

    // Get cursor position if needed
    let cursor_pos = if options.mark_cursor || options.grid.is_some() {
        // We'll implement cursor finding in next step
        // For now, return None
        None
    } else {
        None
    };

    // Draw grid overlay if requested
    let grid_metadata = if let Some(ref grid_config) = options.grid {
        draw_grid_overlay(&mut img, grid_config, cursor_pos.as_ref());

        Some(GridMetadata {
            cell_size: grid_config.cell_size,
            cols: (width + grid_config.cell_size - 1) / grid_config.cell_size,
            rows: (height + grid_config.cell_size - 1) / grid_config.cell_size,
            scheme: grid_config.label_scheme.clone(),
        })
    } else {
        None
    };

    // Encode to PNG
    let mut png_bytes = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut png_bytes), ImageFormat::Png)
        .map_err(|e| Error::ScreenshotFailed(format!("PNG encoding failed: {}", e)))?;

    Ok(ScreenshotData {
        png_bytes,
        width,
        height,
        grid: grid_metadata,
        cursor_pos,
    })
}

// Helper: Extract u32 from DBus reply
fn extract_u32(reply: &HashMap<String, Value>, key: &str) -> Result<u32> {
    let value = reply.get(key)
        .ok_or_else(|| Error::ScreenshotFailed(format!("Missing {} in reply", key)))?;

    value.downcast_ref::<u32>()
        .map_err(|_| Error::ScreenshotFailed(format!("Invalid type for {} in reply", key)))
}

// Helper: Convert ARGB32 to RGBA8 image
fn argb_to_rgba_image(argb: &[u8], width: u32, height: u32, stride: u32) -> Result<RgbaImage> {
    // Convert ARGB to RGBA
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        let row_start = (y * stride) as usize;
        for x in 0..width {
            let pixel_start = row_start + (x * 4) as usize;
            // ARGB -> RGBA: [A, R, G, B] -> [R, G, B, A]
            rgba.push(argb[pixel_start + 1]); // R
            rgba.push(argb[pixel_start + 2]); // G
            rgba.push(argb[pixel_start + 3]); // B
            rgba.push(argb[pixel_start + 0]); // A
        }
    }

    // Create image
    ImageBuffer::from_raw(width, height, rgba)
        .ok_or_else(|| Error::ScreenshotFailed("Failed to create image buffer".into()))
}
