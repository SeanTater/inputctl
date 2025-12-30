use crate::error::{Error, Result};
use std::collections::HashMap;
use std::io::Read;
use nix::unistd::pipe2;
use nix::fcntl::OFlag;
use zbus::blocking::Connection;
use zvariant::{OwnedFd as ZOwnedFd, Value};
use image::{ImageBuffer, RgbaImage, ImageFormat};

/// Capture a screenshot using KWin DBus interface
///
/// Requires KDE Plasma 6.0+ with KWin compositor.
/// Returns PNG bytes.
pub fn capture_screenshot() -> Result<Vec<u8>> {
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
    let mut options: HashMap<&str, Value> = HashMap::new();
    options.insert("native-resolution", Value::new(true));

    // Convert FD for DBus
    let z_write_fd = ZOwnedFd::from(write_fd);

    // Call CaptureWorkspace (Plasma 6.0+)
    let response = proxy
        .call_method("CaptureWorkspace", &(options, z_write_fd))
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

    // Convert ARGB32 to RGBA8 and encode to PNG
    argb_to_png(&raw_data, width, height, stride)
}

// Helper: Extract u32 from DBus reply
fn extract_u32(reply: &HashMap<String, Value>, key: &str) -> Result<u32> {
    let value = reply.get(key)
        .ok_or_else(|| Error::ScreenshotFailed(format!("Missing {} in reply", key)))?;

    value.downcast_ref::<u32>()
        .map_err(|_| Error::ScreenshotFailed(format!("Invalid type for {} in reply", key)))
}

// Helper: Convert ARGB32 to RGBA8 and encode as PNG
fn argb_to_png(argb: &[u8], width: u32, height: u32, stride: u32) -> Result<Vec<u8>> {
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

    // Create image and encode to PNG
    let img: RgbaImage = ImageBuffer::from_raw(width, height, rgba)
        .ok_or_else(|| Error::ScreenshotFailed("Failed to create image buffer".into()))?;

    let mut png_bytes = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut png_bytes), ImageFormat::Png)
        .map_err(|e| Error::ScreenshotFailed(format!("PNG encoding failed: {}", e)))?;

    Ok(png_bytes)
}
