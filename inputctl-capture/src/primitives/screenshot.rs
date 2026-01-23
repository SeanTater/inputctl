use crate::capture::FrameSource;
use crate::error::{Error, Result};
use crate::primitives::grid::{draw_cursor_mark, CursorPos};
use crate::primitives::screen::Region;
use fast_image_resize::{
    images::{Image, ImageRef},
    FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer,
};

use image::{ImageBuffer, ImageFormat, RgbaImage};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

static CURSOR_OVERRIDE: Mutex<Option<CursorPos>> = Mutex::new(None);
static RESIZER: OnceLock<Mutex<Resizer>> = OnceLock::new();

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
        cursor_pos: None,
    })
}

pub fn capture_screenshot_with_source_default_timeout(
    source: &mut dyn FrameSource,
    options: ScreenshotOptions,
) -> Result<ScreenshotData> {
    capture_screenshot_with_source(source, options, Duration::from_secs(2))
}

pub(crate) fn resolve_cursor_pos() -> Result<CursorPos> {
    if let Ok(guard) = CURSOR_OVERRIDE.lock() {
        if let Some(pos) = guard.clone() {
            return Ok(pos);
        }
    }

    crate::primitives::cursor::find_cursor()
}

pub fn set_cursor_override(pos: Option<CursorPos>) {
    if let Ok(mut guard) = CURSOR_OVERRIDE.lock() {
        *guard = pos;
    }
}

/// Internal helper: Capture screenshot logic returning RgbaImage
/// Useful for raw access (recorder, etc.)
pub fn capture_screenshot_image(options: ScreenshotOptions) -> Result<RgbaImage> {
    let frame = capture_raw_frame(Duration::from_secs(2))?;
    screenshot_from_frame(&frame, options)
}

pub fn capture_screenshot_image_with_source(
    source: &mut dyn FrameSource,
    options: ScreenshotOptions,
    timeout: Duration,
) -> Result<RgbaImage> {
    let frame = source.next_frame(timeout)?;
    screenshot_from_frame(&frame, options)
}

pub fn capture_screenshot_with_source(
    source: &mut dyn FrameSource,
    options: ScreenshotOptions,
    timeout: Duration,
) -> Result<ScreenshotData> {
    let img = capture_screenshot_image_with_source(source, options, timeout)?;

    let mut png_bytes = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut png_bytes), ImageFormat::Png)
        .map_err(|e| Error::ScreenshotFailed(format!("PNG encoding failed: {}", e)))?;

    Ok(ScreenshotData {
        png_bytes,
        width: img.width(),
        height: img.height(),
        cursor_pos: None,
    })
}

pub fn screenshot_from_frame(
    frame: &crate::capture::CaptureFrame,
    options: ScreenshotOptions,
) -> Result<RgbaImage> {
    let orig_width = frame.width;
    let orig_height = frame.height;
    let logical_full = options
        .resize_to_logical
        .unwrap_or((orig_width, orig_height));

    let mut output_width = orig_width;
    let mut output_height = orig_height;
    let mut crop_box = None;

    if let Some(region) = options.crop_region {
        if region.x < 0
            || region.y < 0
            || region.x as u32 + region.width > logical_full.0
            || region.y as u32 + region.height > logical_full.1
        {
            return Err(Error::ScreenshotFailed(format!(
                "Crop region {:?} out of bounds for image {}x{}",
                region, logical_full.0, logical_full.1
            )));
        }

        let scale_x = orig_width as f64 / logical_full.0 as f64;
        let scale_y = orig_height as f64 / logical_full.1 as f64;
        let left = region.x as f64 * scale_x;
        let top = region.y as f64 * scale_y;
        let width = region.width as f64 * scale_x;
        let height = region.height as f64 * scale_y;
        crop_box = Some((left, top, width, height));
        output_width = region.width;
        output_height = region.height;
    } else if let Some((target_w, target_h)) = options.resize_to_logical {
        output_width = target_w;
        output_height = target_h;
    }

    let rgba = if crop_box.is_some() || output_width != orig_width || output_height != orig_height {
        let src_image = ImageRef::new(orig_width, orig_height, &frame.rgba, PixelType::U8x4)
            .map_err(|e| Error::ScreenshotFailed(format!("Resize source error: {}", e)))?;
        let mut dst_image = Image::new(output_width, output_height, PixelType::U8x4);
        let opts = ResizeOptions::new().resize_alg(ResizeAlg::Interpolation(FilterType::Bilinear));
        let opts = if let Some((left, top, width, height)) = crop_box {
            opts.crop(left, top, width, height)
        } else {
            opts
        };

        let resizer = RESIZER.get_or_init(|| Mutex::new(Resizer::new()));
        let mut resizer = resizer
            .lock()
            .map_err(|_| Error::ScreenshotFailed("Resize lock poisoned".into()))?;
        resizer
            .resize(&src_image, &mut dst_image, Some(&opts))
            .map_err(|e| Error::ScreenshotFailed(format!("Resize failed: {}", e)))?;
        dst_image.into_vec()
    } else {
        frame.rgba.clone()
    };

    let mut img = ImageBuffer::from_raw(output_width, output_height, rgba)
        .ok_or_else(|| Error::ScreenshotFailed("Failed to create image buffer".into()))?;

    if options.mark_cursor {
        if let Ok(p) = resolve_cursor_pos() {
            let mut draw_x = p.x;
            let mut draw_y = p.y;

            if options.resize_to_logical.is_some() {
                let scale_x = logical_full.0 as f32 / orig_width as f32;
                let scale_y = logical_full.1 as f32 / orig_height as f32;
                draw_x = (p.x as f32 * scale_x).round() as i32;
                draw_y = (p.y as f32 * scale_y).round() as i32;
            }

            if let Some(region) = options.crop_region {
                draw_x -= region.x;
                draw_y -= region.y;
            }

            let should_draw = draw_x >= 0
                && draw_y >= 0
                && draw_x < output_width as i32
                && draw_y < output_height as i32;

            if should_draw {
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

pub fn capture_screenshot_raw_with_source(
    source: &mut dyn FrameSource,
    timeout: Duration,
) -> Result<Vec<u8>> {
    let frame = source.next_frame(timeout)?;
    Ok(frame.rgba)
}

pub fn capture_raw_frame_with_source(
    source: &mut dyn FrameSource,
    timeout: Duration,
) -> Result<crate::capture::CaptureFrame> {
    source.next_frame(timeout)
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
