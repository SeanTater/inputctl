use criterion::{black_box, criterion_group, criterion_main, Criterion};
use inputctl_capture::primitives::screenshot::{screenshot_from_frame, ScreenshotOptions};
use inputctl_capture::{CaptureFrame, CursorPos, Region};

fn make_frame(width: u32, height: u32) -> CaptureFrame {
    let mut rgba = vec![0u8; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 4) as usize;
            rgba[idx] = (x % 255) as u8;
            rgba[idx + 1] = (y % 255) as u8;
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

fn bench_crop_resize_cursor(c: &mut Criterion) {
    let frame = make_frame(1920, 1080);
    let cursor = CursorPos {
        x: 960,
        y: 540,
        grid_cell: None,
    };

    let options = ScreenshotOptions {
        mark_cursor: true,
        crop_region: Some(Region::new(0, 0, 1280, 720)),
        resize_to_logical: Some((1920, 1080)),
    };

    inputctl_capture::primitives::screenshot::set_cursor_override(Some(cursor));
    c.bench_function("screenshot_crop_resize_cursor", |b| {
        b.iter(|| {
            let img =
                screenshot_from_frame(black_box(&frame), options.clone()).expect("screenshot");
            black_box(img);
        })
    });
    inputctl_capture::primitives::screenshot::set_cursor_override(None);
}

criterion_group!(benches, bench_crop_resize_cursor);
criterion_main!(benches);
