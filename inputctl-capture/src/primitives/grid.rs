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
}
