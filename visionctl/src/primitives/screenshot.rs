use crate::error::{Error, Result};
use crate::primitives::grid::{CursorPos, draw_cursor_mark};
use crate::primitives::screen::Region;
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
    pub mark_cursor: bool,
    pub crop_region: Option<Region>,
    /// If set, the workspace image will be resized to these dimensions before cropping/drawing.
    /// This allows mapping physical pixels to logical coordinates.
    pub resize_to_logical: Option<(u32, u32)>,
}

/// Screenshot data with metadata
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct ScreenshotData {
    pub png_bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
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

    // Convert BGRA32 to RGBA8
    let mut img = argb_to_rgba_image(&raw_data, width, height, stride)?;

    // Resize to logical dimensions if requested
    // This is the key to fixing Wayland scaling issues.
    if let Some((target_w, target_h)) = options.resize_to_logical {
        if target_w != width || target_h != height {
            img = image::imageops::resize(&img, target_w, target_h, image::imageops::FilterType::Lanczos3);
        }
    }

    // Crop if requested
    if let Some(region) = options.crop_region {
        // Verify crop bounds
        if region.x >= 0 && region.y >= 0 && 
           (region.x + region.width as i32) <= width as i32 && 
           (region.y + region.height as i32) <= height as i32 {
            img = image::imageops::crop(&mut img, region.x as u32, region.y as u32, region.width, region.height).to_image();
        } else {
             return Err(Error::ScreenshotFailed(format!("Crop region {:?} out of bounds for image {}x{}", region, width, height)));
        }
    }

    // Get cursor position and draw marker if requested

    let cursor_pos = if options.mark_cursor {
        let pos = crate::primitives::cursor::find_cursor().ok();
        if let Some(ref p) = pos {
            // If cropping, check if cursor is visible and adjust relative coords for drawing?
            // Actually, we usually draw the mark on the final image.
            // If we cropped, we need to adjust the cursor position relative to the crop to draw it safely.
            // But wait, if we modify `img` above (by cropping), `img` is now the crop.
            // So we need to translate the global cursor `p` to local `p`.
            
            let should_draw = if let Some(region) = options.crop_region {
                // Check if cursor is inside
                region.contains(p.x, p.y)
            } else {
                true
            };

            if should_draw {
                 let draw_x = if let Some(region) = options.crop_region { p.x - region.x } else { p.x };
                 let draw_y = if let Some(region) = options.crop_region { p.y - region.y } else { p.y };
                 
                 // We need a temp local struct to pass to draw_cursor_mark because it probably takes CursorPos
                 // Let's assume draw_cursor_mark handles bounds checking or we should.
                 // For now, let's construct a local pos.
                 let local_pos = CursorPos { x: draw_x, y: draw_y, grid_cell: None };
                 draw_cursor_mark(&mut img, &local_pos);
            }
        }
        pos
    } else {
        None
    };

    // Encode to PNG
    let mut png_bytes = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut png_bytes), ImageFormat::Png)
        .map_err(|e| Error::ScreenshotFailed(format!("PNG encoding failed: {}", e)))?;

    let (final_width, final_height) = if let Some((tw, th)) = options.resize_to_logical {
        (tw, th)
    } else {
        (width, height)
    };

    Ok(ScreenshotData {
        png_bytes,
        width: final_width,
        height: final_height,
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

// Helper: Convert BGRA32 to RGBA8 image
fn argb_to_rgba_image(bgra: &[u8], width: u32, height: u32, stride: u32) -> Result<RgbaImage> {
    // Convert BGRA to RGBA (KWin provides BGRA format on Linux)
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        let row_start = (y * stride) as usize;
        for x in 0..width {
            let pixel_start = row_start + (x * 4) as usize;
            // BGRA -> RGBA: [B, G, R, A] -> [R, G, B, A]
            rgba.push(bgra[pixel_start + 2]); // R
            rgba.push(bgra[pixel_start + 1]); // G
            rgba.push(bgra[pixel_start + 0]); // B
            rgba.push(bgra[pixel_start + 3]); // A
        }
    }

    // Create image
    ImageBuffer::from_raw(width, height, rgba)
        .ok_or_else(|| Error::ScreenshotFailed("Failed to create image buffer".into()))
}
