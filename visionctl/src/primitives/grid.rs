use image::{Rgba, RgbaImage};

/// Cursor position with optional grid cell reference
#[derive(Clone, Debug)]
pub struct CursorPos {
    pub x: i32,
    pub y: i32,
    pub grid_cell: Option<String>,
}

/// Draw cursor mark (crosshair + circle) on image
pub fn draw_cursor_mark(image: &mut RgbaImage, pos: &CursorPos) {
    let mark_color = Rgba([255u8, 0u8, 0u8, 255u8]); // Solid red
    let outline_color = Rgba([255u8, 255u8, 255u8, 255u8]); // White outline
    let size = 60u32; // Large crosshair size
    let thickness = 4u32; // Line thickness

    let x = pos.x as u32;
    let y = pos.y as u32;

    // Draw thick crosshair with white outline for visibility
    // Horizontal line
    for dx in 0..size {
        let px = x.saturating_sub(size / 2).saturating_add(dx);
        for t in 0..thickness {
            // White outline (above and below)
            if let Some(pixel) = image
                .get_pixel_mut_checked(px, y.saturating_sub(thickness / 2 + 1).saturating_add(t))
            {
                if t == 0 || t == thickness - 1 {
                    *pixel = outline_color;
                }
            }
            // Red center
            if let Some(pixel) =
                image.get_pixel_mut_checked(px, y.saturating_sub(thickness / 2).saturating_add(t))
            {
                *pixel = mark_color;
            }
        }
    }

    // Vertical line
    for dy in 0..size {
        let py = y.saturating_sub(size / 2).saturating_add(dy);
        for t in 0..thickness {
            // White outline (left and right)
            if let Some(pixel) = image
                .get_pixel_mut_checked(x.saturating_sub(thickness / 2 + 1).saturating_add(t), py)
            {
                if t == 0 || t == thickness - 1 {
                    *pixel = outline_color;
                }
            }
            // Red center
            if let Some(pixel) =
                image.get_pixel_mut_checked(x.saturating_sub(thickness / 2).saturating_add(t), py)
            {
                *pixel = mark_color;
            }
        }
    }

    // Draw a circle around cursor for extra visibility
    let radius = 20u32;
    for angle in 0..360 {
        let rad = (angle as f32) * std::f32::consts::PI / 180.0;
        let cx = x as i32 + (radius as f32 * rad.cos()) as i32;
        let cy = y as i32 + (radius as f32 * rad.sin()) as i32;
        if cx >= 0 && cy >= 0 {
            // White outline
            if let Some(pixel) = image.get_pixel_mut_checked(cx as u32, cy as u32) {
                *pixel = outline_color;
            }
            // Red inner
            let cx2 = x as i32 + ((radius - 1) as f32 * rad.cos()) as i32;
            let cy2 = y as i32 + ((radius - 1) as f32 * rad.sin()) as i32;
            if cx2 >= 0 && cy2 >= 0 {
                if let Some(pixel) = image.get_pixel_mut_checked(cx2 as u32, cy2 as u32) {
                    *pixel = mark_color;
                }
            }
        }
    }
}
