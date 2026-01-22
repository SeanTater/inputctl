use inputctl_capture::primitives::frame_ops::{copy_frame, non_black_bounds};
use inputctl_capture::Region;

#[test]
fn copy_frame_handles_stride() {
    let mut raw = vec![0u8; 16];
    for (i, v) in raw.iter_mut().enumerate() {
        *v = i as u8;
    }
    let mut warned = false;
    let raw_len = raw.len() as u32;
    let out = copy_frame(&mut raw, 2, 2, 0, 0, raw_len, 4, &mut warned).expect("frame copy");
    assert_eq!(out, raw);
}

#[test]
fn non_black_bounds_detects_pixel() {
    let mut rgba = vec![0u8; 4 * 3 * 4];
    let row_stride = 4 * 4;
    let idx = (2 * row_stride + 1 * 4) as usize;
    rgba[idx] = 255;
    let bounds = non_black_bounds(&rgba, 4, 3).expect("bounds");
    assert_eq!(bounds, Region::new(1, 2, 1, 1));
}
