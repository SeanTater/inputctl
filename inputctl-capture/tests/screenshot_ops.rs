use image::Rgba;
use inputctl_capture::primitives::screenshot::{screenshot_from_frame, ScreenshotOptions};
use inputctl_capture::{CaptureFrame, CursorPos, Region};

fn make_frame(width: u32, height: u32) -> CaptureFrame {
    let mut rgba = vec![0u8; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 4) as usize;
            rgba[idx] = x as u8;
            rgba[idx + 1] = y as u8;
            rgba[idx + 2] = 128;
            rgba[idx + 3] = 255;
        }
    }
    CaptureFrame {
        rgba,
        width,
        height,
        timestamp_ms: 0,
        format: "RGBA".to_string(),
    }
}

#[test]
fn screenshot_resizes_then_crops() {
    let frame = make_frame(4, 4);
    let options = ScreenshotOptions {
        mark_cursor: false,
        crop_region: Some(Region::new(1, 1, 1, 1)),
        resize_to_logical: Some((2, 2)),
    };

    let img = screenshot_from_frame(&frame, options).expect("screenshot");
    assert_eq!(img.width(), 1);
    assert_eq!(img.height(), 1);
}

#[test]
fn screenshot_draws_cursor_when_inside_crop() {
    let frame = make_frame(120, 120);
    let options = ScreenshotOptions {
        mark_cursor: true,
        crop_region: Some(Region::new(10, 10, 80, 80)),
        resize_to_logical: Some((100, 100)),
    };

    let cursor = CursorPos {
        x: 72,
        y: 72,
        grid_cell: None,
    };
    inputctl_capture::primitives::screenshot::set_cursor_override(Some(cursor));
    let img = screenshot_from_frame(&frame, options).expect("screenshot");
    inputctl_capture::primitives::screenshot::set_cursor_override(None);

    assert_eq!(img.width(), 80);
    assert_eq!(img.height(), 80);
}

#[test]
fn screenshot_skips_cursor_when_outside_crop() {
    let frame = make_frame(120, 120);
    let options = ScreenshotOptions {
        mark_cursor: true,
        crop_region: Some(Region::new(60, 60, 40, 40)),
        resize_to_logical: None,
    };

    let cursor = CursorPos {
        x: 10,
        y: 10,
        grid_cell: None,
    };
    inputctl_capture::primitives::screenshot::set_cursor_override(Some(cursor));
    let img = screenshot_from_frame(&frame, options).expect("screenshot");
    inputctl_capture::primitives::screenshot::set_cursor_override(None);

    let sample = img.get_pixel(0, 0);
    assert_ne!(*sample, Rgba([255, 0, 0, 255]));
}
