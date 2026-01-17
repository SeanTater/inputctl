use crate::error::{Error, Result};
use crate::primitives::grid::{draw_cursor_mark, CursorPos};
use crate::primitives::screen::Region;
use image::{ImageBuffer, ImageFormat, RgbaImage};
use nix::fcntl::OFlag;
use nix::unistd::pipe2;
use std::collections::HashMap;
use std::io::Read;
use zbus::blocking::Connection;
use zvariant::{OwnedFd as ZOwnedFd, Value};

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
    let mut img = capture_screenshot_image(options)?;

    // Encode to PNG
    let mut png_bytes = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut png_bytes), ImageFormat::Png)
        .map_err(|e| Error::ScreenshotFailed(format!("PNG encoding failed: {}", e)))?;

    Ok(ScreenshotData {
        png_bytes,
        width: img.width(),
        height: img.height(),
        cursor_pos: None, // We don't return actual cursor pos here anymore in Data, or we need to pass it out from _image function?
                          // Actually, ScreenshotData included cursor_pos for debugging overlays.
                          // Let's make capture_screenshot_image return (RgbaImage, Option<CursorPos>).
    })
}

/// Internal helper: Capture screenshot logic returning RgbaImage
/// Useful for raw access (recorder, etc.)
pub fn capture_screenshot_image(options: ScreenshotOptions) -> Result<RgbaImage> {
    // Create pipe for receiving image data
    let (read_fd, write_fd) = pipe2(OFlag::O_CLOEXEC)
        .map_err(|e| Error::ScreenshotFailed(format!("Failed to create pipe: {}", e)))?;

    // Connect to DBus session bus
    let conn = Connection::session().map_err(|e| {
        Error::ScreenshotFailed(format!("DBus connection failed - is KDE running?: {}", e))
    })?;

    // Create proxy for KWin ScreenShot2 interface
    let proxy = zbus::blocking::Proxy::new(
        &conn,
        "org.kde.KWin.ScreenShot2",
        "/org/kde/KWin/ScreenShot2",
        "org.kde.KWin.ScreenShot2",
    )
    .map_err(|e| Error::ScreenshotFailed(format!("KWin DBus interface not found: {}", e)))?;

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
    if let Some((target_w, target_h)) = options.resize_to_logical {
        if target_w != width || target_h != height {
            img = image::imageops::resize(
                &img,
                target_w,
                target_h,
                image::imageops::FilterType::Lanczos3,
            );
        }
    }

    // Crop if requested
    if let Some(region) = options.crop_region {
        // Verify crop bounds
        if region.x >= 0
            && region.y >= 0
            && (region.x + region.width as i32) <= img.width() as i32
            && (region.y + region.height as i32) <= img.height() as i32
        {
            img = image::imageops::crop(
                &mut img,
                region.x as u32,
                region.y as u32,
                region.width,
                region.height,
            )
            .to_image();
        } else {
            return Err(Error::ScreenshotFailed(format!(
                "Crop region {:?} out of bounds for image {}x{}",
                region,
                img.width(),
                img.height()
            )));
        }
    }

    // Get cursor position and draw marker if requested
    if options.mark_cursor {
        if let Ok(p) = crate::primitives::cursor::find_cursor() {
            let should_draw = if let Some(region) = options.crop_region {
                region.contains(p.x, p.y)
            } else {
                true
            };

            if should_draw {
                let draw_x = if let Some(region) = options.crop_region {
                    p.x - region.x
                } else {
                    p.x
                };
                let draw_y = if let Some(region) = options.crop_region {
                    p.y - region.y
                } else {
                    p.y
                };

                let local_pos = CursorPos {
                    x: draw_x,
                    y: draw_y,
                    grid_cell: None,
                };
                draw_cursor_mark(&mut img, &local_pos);
            }
        }
    }

    Ok(img)
}

/// Capture raw screenshot bytes (RGBA8) without PNG encoding
///
/// Returns raw RGBA8 vector
pub fn capture_screenshot_raw(_width: u32, _height: u32) -> Result<Vec<u8>> {
    // Create pipe for receiving image data
    let (read_fd, write_fd) = pipe2(OFlag::O_CLOEXEC)
        .map_err(|e| Error::ScreenshotFailed(format!("Failed to create pipe: {}", e)))?;

    // Connect to DBus session bus
    let conn = Connection::session().map_err(|e| {
        Error::ScreenshotFailed(format!("DBus connection failed - is KDE running?: {}", e))
    })?;

    // Create proxy for KWin ScreenShot2 interface
    let proxy = zbus::blocking::Proxy::new(
        &conn,
        "org.kde.KWin.ScreenShot2",
        "/org/kde/KWin/ScreenShot2",
        "org.kde.KWin.ScreenShot2",
    )
    .map_err(|e| Error::ScreenshotFailed(format!("KWin DBus interface not found: {}", e)))?;

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
    let width_cap = extract_u32(&reply, "width")?;
    let height_cap = extract_u32(&reply, "height")?;
    let stride = extract_u32(&reply, "stride")?;

    // Read raw image data from pipe
    let expected_size = (stride * height_cap) as usize;
    let mut raw_data = vec![0u8; expected_size];
    let mut file = std::fs::File::from(read_fd);
    file.read_exact(&mut raw_data)
        .map_err(|e| Error::ScreenshotFailed(format!("Failed to read image data: {}", e)))?;

    // Convert BGRA32 -> RGBA8
    let rgba_img = argb_to_rgba_image(&raw_data, width_cap, height_cap, stride)?;

    Ok(rgba_img.into_raw())
}

/// Capture raw screenshot (RGBA8) with cropping
/// Returns raw bytes of the cropped region, plus its width and height
pub fn capture_screenshot_raw_cropped(region: Option<Region>) -> Result<(Vec<u8>, u32, u32)> {
    let img = capture_screenshot_image(ScreenshotOptions {
        mark_cursor: false,
        crop_region: region,
        resize_to_logical: None, // Keep raw pixels for high fidelity ML, unless we specifically want logical?
                                 // ML models usually want consistent resolution.
                                 // If we record raw pixels, we might get different resolutions on different machines.
                                 // But for "Reflex" training on specific machine, raw is fine.
    })?;

    let width = img.width();
    let height = img.height();
    Ok((img.into_raw(), width, height))
}

// Helper: Extract u32 from DBus reply
fn extract_u32(reply: &HashMap<String, Value>, key: &str) -> Result<u32> {
    let value = reply
        .get(key)
        .ok_or_else(|| Error::ScreenshotFailed(format!("Missing {} in reply", key)))?;

    value
        .downcast_ref::<u32>()
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
