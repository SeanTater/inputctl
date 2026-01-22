use crate::error::Result;
use crate::primitives::screen::Region;

pub fn copy_frame(
    raw: &mut [u8],
    width: u32,
    height: u32,
    offset: u32,
    stride: i32,
    chunk_size: u32,
    bytes_per_pixel: usize,
    warned_stride: &mut bool,
) -> Option<Vec<u8>> {
    let row_bytes = width as usize * bytes_per_pixel;
    if row_bytes == 0 || height == 0 {
        return None;
    }

    let stride = if stride == 0 {
        row_bytes as i32
    } else {
        stride
    };
    let stride_abs = stride.unsigned_abs() as usize;
    if stride_abs < row_bytes {
        if !*warned_stride {
            eprintln!(
                "PipeWire stride {} smaller than row bytes {}",
                stride_abs, row_bytes
            );
            *warned_stride = true;
        }
        return None;
    }

    let offset = offset as usize;
    if offset >= raw.len() {
        return None;
    }

    let limit = if chunk_size == 0 {
        raw.len()
    } else {
        offset.saturating_add(chunk_size as usize).min(raw.len())
    };

    let mut out = vec![0u8; row_bytes * height as usize];

    if stride == row_bytes as i32 {
        let needed = row_bytes * height as usize;
        let end = offset.saturating_add(needed);
        if end > limit {
            return None;
        }
        out.copy_from_slice(&raw[offset..end]);
        return Some(out);
    }

    for row in 0..height as usize {
        let src_row = if stride > 0 {
            row
        } else {
            height as usize - 1 - row
        };
        let src_start = offset + src_row * stride_abs;
        let src_end = src_start + row_bytes;
        if src_end > limit {
            return None;
        }
        let dst_start = row * row_bytes;
        out[dst_start..dst_start + row_bytes].copy_from_slice(&raw[src_start..src_end]);
    }

    Some(out)
}

pub fn crop_rgba(rgba: &[u8], width: u32, height: u32, region: Option<Region>) -> Result<Vec<u8>> {
    let region = match region {
        Some(region) => region,
        None => return Ok(rgba.to_vec()),
    };

    if region.x < 0
        || region.y < 0
        || region.x as u32 + region.width > width
        || region.y as u32 + region.height > height
    {
        return Ok(rgba.to_vec());
    }

    let crop_x = region.x as u32;
    let crop_y = region.y as u32;
    let crop_w = region.width;
    let crop_h = region.height;

    let mut out = vec![0u8; (crop_w * crop_h * 4) as usize];
    let src_stride = (width * 4) as usize;
    let dst_stride = (crop_w * 4) as usize;

    for row in 0..crop_h {
        let src_start = ((crop_y + row) as usize * src_stride) + (crop_x * 4) as usize;
        let src_end = src_start + dst_stride;
        let dst_start = row as usize * dst_stride;
        out[dst_start..dst_start + dst_stride].copy_from_slice(&rgba[src_start..src_end]);
    }

    Ok(out)
}

pub fn non_black_bounds(rgba: &[u8], width: u32, height: u32) -> Option<Region> {
    if width == 0 || height == 0 {
        return None;
    }

    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut found = false;

    let row_stride = (width * 4) as usize;
    for y in 0..height {
        let row_start = (y as usize) * row_stride;
        for x in 0..width {
            let idx = row_start + (x as usize * 4);
            let r = rgba.get(idx).copied().unwrap_or(0);
            let g = rgba.get(idx + 1).copied().unwrap_or(0);
            let b = rgba.get(idx + 2).copied().unwrap_or(0);
            if r == 0 && g == 0 && b == 0 {
                continue;
            }

            found = true;
            if x < min_x {
                min_x = x;
            }
            if y < min_y {
                min_y = y;
            }
            if x > max_x {
                max_x = x;
            }
            if y > max_y {
                max_y = y;
            }
        }
    }

    if !found {
        return None;
    }

    Some(Region {
        x: min_x as i32,
        y: min_y as i32,
        width: max_x - min_x + 1,
        height: max_y - min_y + 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_frame_with_exact_stride() {
        let mut raw = vec![0u8; 16];
        for (i, v) in raw.iter_mut().enumerate() {
            *v = i as u8;
        }
        let mut warned = false;
        let raw_len = raw.len() as u32;
        let out = copy_frame(&mut raw, 2, 2, 0, 0, raw_len, 4, &mut warned).expect("frame copy");
        assert_eq!(out, raw);
        assert!(!warned);
    }

    #[test]
    fn copy_frame_with_negative_stride() {
        let mut raw = vec![
            1, 2, 3, 4, 5, 6, 7, 8, // row 0
            9, 10, 11, 12, 13, 14, 15, 16, // row 1
        ];
        let mut warned = false;
        let raw_len = raw.len() as u32;
        let out = copy_frame(&mut raw, 2, 2, 0, -8, raw_len, 4, &mut warned).expect("frame copy");
        assert_eq!(
            out,
            vec![9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8]
        );
    }

    #[test]
    fn copy_frame_rejects_too_small_stride() {
        let mut raw = vec![0u8; 16];
        let mut warned = false;
        let raw_len = raw.len() as u32;
        let out = copy_frame(&mut raw, 2, 2, 0, 4, raw_len, 4, &mut warned);
        assert!(out.is_none());
        assert!(warned);
    }

    #[test]
    fn crop_rgba_region() {
        let rgba: Vec<u8> = (0..64).collect();
        let region = Region::new(1, 1, 2, 2);
        let out = crop_rgba(&rgba, 4, 4, Some(region)).expect("crop");
        assert_eq!(out.len(), 16);
        assert_eq!(out[0], rgba[(1 * 4 * 4 + 1 * 4) as usize]);
    }

    #[test]
    fn non_black_bounds_none_when_all_black() {
        let rgba = vec![0u8; 16];
        assert_eq!(non_black_bounds(&rgba, 2, 2), None);
    }

    #[test]
    fn non_black_bounds_single_pixel() {
        let mut rgba = vec![0u8; 4 * 3 * 4];
        let row_stride = 4 * 4;
        let idx = (2 * row_stride + 1 * 4) as usize;
        rgba[idx] = 255;
        let bounds = non_black_bounds(&rgba, 4, 3).expect("bounds");
        assert_eq!(bounds, Region::new(1, 2, 1, 1));
    }
}
