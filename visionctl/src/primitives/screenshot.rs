use crate::error::{Error, Result};
use crate::primitives::grid::{draw_cursor_mark, CursorPos};
use crate::primitives::screen::Region;
use image::{ImageBuffer, ImageFormat, RgbaImage};
use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::Duration;

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

/// Capture a screenshot using the portal PipeWire stream
///
/// Requires xdg-desktop-portal with screencast support.
/// Returns PNG bytes without any overlays.
pub fn capture_screenshot_simple() -> Result<Vec<u8>> {
    let data = capture_screenshot(ScreenshotOptions::default())?;
    Ok(data.png_bytes)
}

/// Capture a screenshot with optional grid overlay and cursor marking
pub fn capture_screenshot(options: ScreenshotOptions) -> Result<ScreenshotData> {
    let img = capture_screenshot_image(options)?;

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
    let frame = capture_raw_frame(Duration::from_secs(2))?;
    let mut img = ImageBuffer::from_raw(frame.width, frame.height, frame.rgba)
        .ok_or_else(|| Error::ScreenshotFailed("Failed to create image buffer".into()))?;

    // Resize to logical dimensions if requested
    if let Some((target_w, target_h)) = options.resize_to_logical {
        if target_w != img.width() || target_h != img.height() {
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
    let frame = capture_raw_frame(Duration::from_secs(2))?;
    Ok(frame.rgba)
}

/// Capture raw screenshot (RGBA8) with cropping
/// Returns raw bytes of the cropped region, plus its width and height
pub fn capture_screenshot_raw_cropped(region: Option<Region>) -> Result<(Vec<u8>, u32, u32)> {
    let img = capture_screenshot_image(ScreenshotOptions {
        mark_cursor: false,
        crop_region: region,
        resize_to_logical: None,
    })?;

    let width = img.width();
    let height = img.height();
    Ok((img.into_raw(), width, height))
}

fn capture_raw_frame(timeout: Duration) -> Result<crate::capture::CaptureFrame> {
    let capture = portal_capture()?;
    capture.next_frame(timeout)
}

fn portal_capture() -> Result<std::sync::MutexGuard<'static, crate::capture::PortalCapture>> {
    static CAPTURE: OnceLock<std::result::Result<Mutex<crate::capture::PortalCapture>, Error>> =
        OnceLock::new();
    let capture_result =
        CAPTURE.get_or_init(|| crate::capture::PortalCapture::connect(None).map(Mutex::new));
    let capture = match capture_result {
        Ok(capture) => capture,
        Err(err) => {
            return Err(Error::ScreenshotFailed(format!(
                "Portal capture init failed: {}",
                err
            )))
        }
    };
    let guard = capture
        .lock()
        .map_err(|_| Error::ScreenshotFailed("Capture lock poisoned".into()))?;
    Ok(guard)
}
