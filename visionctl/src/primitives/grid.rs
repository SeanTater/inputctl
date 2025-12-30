use crate::error::{Error, Result};
use image::{Rgba, RgbaImage};

/// Grid configuration for overlaying coordinate system on screenshots
#[derive(Clone, Debug)]
pub struct GridConfig {
    pub cell_size: u32,
    pub style: GridStyle,
    pub label_scheme: LabelScheme,
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            cell_size: 100,
            style: GridStyle::Lines,
            label_scheme: LabelScheme::LetterNumber,
        }
    }
}

#[derive(Clone, Debug)]
pub enum GridStyle {
    Lines,
    // Dots,  // Future: grid points instead of lines
}

#[derive(Clone, Debug)]
pub enum LabelScheme {
    LetterNumber,  // A1, B2, C3... (recommended)
    // NumberPair,    // Future: 1-1, 1-2, 2-1...
    // PixelCoords,   // Future: include pixel coords in labels
}

/// Cursor position with optional grid cell reference
#[derive(Clone, Debug)]
pub struct CursorPos {
    pub x: i32,
    pub y: i32,
    pub grid_cell: Option<String>,
}

/// Grid metadata attached to screenshot
#[derive(Clone, Debug)]
pub struct GridMetadata {
    pub cell_size: u32,
    pub cols: u32,
    pub rows: u32,
    pub scheme: LabelScheme,
}

/// Convert pixel coordinates to grid cell reference (e.g., "A1", "B3")
pub fn pixel_to_grid(x: i32, y: i32, config: &GridConfig) -> String {
    match config.label_scheme {
        LabelScheme::LetterNumber => {
            let col = (x as u32 / config.cell_size) as usize;
            let row = (y as u32 / config.cell_size) as usize;

            // Column: A-Z, AA-ZZ, etc.
            let col_label = column_to_letter(col);
            // Row: 1-based indexing
            let row_label = row + 1;

            format!("{}{}", col_label, row_label)
        }
    }
}

/// Convert grid cell reference to pixel coordinates (center of cell)
pub fn grid_to_pixel(cell: &str, config: &GridConfig) -> Result<(i32, i32)> {
    match config.label_scheme {
        LabelScheme::LetterNumber => {
            // Parse letter-number format (e.g., "A1", "B3", "AA10")
            let (col_str, row_str) = split_cell_reference(cell)?;

            let col = letter_to_column(&col_str)?;
            let row = row_str.parse::<usize>()
                .map_err(|_| Error::ScreenshotFailed(format!("Invalid row number: {}", row_str)))?;

            // Convert to 0-based index
            let row = row.saturating_sub(1);

            // Return center of cell
            let x = (col as u32 * config.cell_size + config.cell_size / 2) as i32;
            let y = (row as u32 * config.cell_size + config.cell_size / 2) as i32;

            Ok((x, y))
        }
    }
}

/// Draw grid overlay on image with optional cursor mark
pub fn draw_grid_overlay(
    image: &mut RgbaImage,
    config: &GridConfig,
    cursor: Option<&CursorPos>,
) {
    let (width, height) = image.dimensions();

    match config.style {
        GridStyle::Lines => {
            draw_grid_lines(image, config, width, height);
            draw_grid_labels(image, config, width, height);
        }
    }

    // Optionally mark cursor position
    if let Some(pos) = cursor {
        draw_cursor_mark(image, pos);
    }
}

fn draw_grid_lines(image: &mut RgbaImage, config: &GridConfig, width: u32, height: u32) {
    let line_color = Rgba([200u8, 200u8, 200u8, 80u8]); // Semi-transparent white

    // Draw vertical lines
    let mut x = 0;
    while x < width {
        for y in 0..height {
            if let Some(pixel) = image.get_pixel_mut_checked(x, y) {
                *pixel = blend_pixel(*pixel, line_color);
            }
        }
        x += config.cell_size;
    }

    // Draw horizontal lines
    let mut y = 0;
    while y < height {
        for x in 0..width {
            if let Some(pixel) = image.get_pixel_mut_checked(x, y) {
                *pixel = blend_pixel(*pixel, line_color);
            }
        }
        y += config.cell_size;
    }
}

fn draw_grid_labels(image: &mut RgbaImage, config: &GridConfig, width: u32, height: u32) {
    // Note: Full text rendering requires a font rasterizer (like imageproc + rusttype)
    // For now, we'll skip labels and can add them later if needed with imageproc crate
    // The grid lines alone are sufficient for LLM to understand the coordinate system
    // when combined with the cell references in the tool descriptions

    let _ = (image, config, width, height); // Suppress unused warnings
}

fn draw_cursor_mark(image: &mut RgbaImage, pos: &CursorPos) {
    let mark_color = Rgba([255u8, 0u8, 0u8, 200u8]); // Semi-transparent red
    let size = 10u32; // Crosshair size

    let x = pos.x as u32;
    let y = pos.y as u32;

    // Draw crosshair
    // Horizontal line
    for dx in 0..size {
        let px = x.saturating_sub(size / 2).saturating_add(dx);
        if let Some(pixel) = image.get_pixel_mut_checked(px, y) {
            *pixel = mark_color;
        }
    }

    // Vertical line
    for dy in 0..size {
        let py = y.saturating_sub(size / 2).saturating_add(dy);
        if let Some(pixel) = image.get_pixel_mut_checked(x, py) {
            *pixel = mark_color;
        }
    }
}

/// Alpha blend two pixels
fn blend_pixel(bg: Rgba<u8>, fg: Rgba<u8>) -> Rgba<u8> {
    let alpha = fg.0[3] as f32 / 255.0;
    let inv_alpha = 1.0 - alpha;

    Rgba([
        ((fg.0[0] as f32 * alpha) + (bg.0[0] as f32 * inv_alpha)) as u8,
        ((fg.0[1] as f32 * alpha) + (bg.0[1] as f32 * inv_alpha)) as u8,
        ((fg.0[2] as f32 * alpha) + (bg.0[2] as f32 * inv_alpha)) as u8,
        255, // Full opacity
    ])
}

/// Convert column index to letter(s) (A, B, C, ... Z, AA, AB, ...)
fn column_to_letter(col: usize) -> String {
    let mut result = String::new();
    let mut n = col;

    loop {
        result.insert(0, (b'A' + (n % 26) as u8) as char);
        if n < 26 {
            break;
        }
        n = n / 26 - 1;
    }

    result
}

/// Convert letter(s) to column index (A -> 0, B -> 1, ... Z -> 25, AA -> 26, ...)
fn letter_to_column(s: &str) -> Result<usize> {
    if s.is_empty() {
        return Err(Error::ScreenshotFailed("Empty column label".into()));
    }

    let mut result = 0usize;
    for c in s.chars() {
        if !c.is_ascii_alphabetic() {
            return Err(Error::ScreenshotFailed(format!("Invalid column label: {}", s)));
        }
        result = result * 26 + (c.to_ascii_uppercase() as usize - 'A' as usize + 1);
    }

    Ok(result - 1)
}

/// Split cell reference into column and row parts (e.g., "A1" -> ("A", "1"))
fn split_cell_reference(cell: &str) -> Result<(String, String)> {
    let mut col = String::new();
    let mut row = String::new();
    let mut in_row = false;

    for c in cell.chars() {
        if c.is_ascii_alphabetic() {
            if in_row {
                return Err(Error::ScreenshotFailed(
                    format!("Invalid cell format (letters after numbers): {}", cell)
                ));
            }
            col.push(c);
        } else if c.is_ascii_digit() {
            in_row = true;
            row.push(c);
        } else {
            return Err(Error::ScreenshotFailed(
                format!("Invalid character in cell reference: {}", cell)
            ));
        }
    }

    if col.is_empty() || row.is_empty() {
        return Err(Error::ScreenshotFailed(
            format!("Invalid cell format (missing column or row): {}", cell)
        ));
    }

    Ok((col, row))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_to_letter() {
        assert_eq!(column_to_letter(0), "A");
        assert_eq!(column_to_letter(25), "Z");
        assert_eq!(column_to_letter(26), "AA");
        assert_eq!(column_to_letter(27), "AB");
    }

    #[test]
    fn test_letter_to_column() {
        assert_eq!(letter_to_column("A").unwrap(), 0);
        assert_eq!(letter_to_column("Z").unwrap(), 25);
        assert_eq!(letter_to_column("AA").unwrap(), 26);
        assert_eq!(letter_to_column("AB").unwrap(), 27);
    }

    #[test]
    fn test_pixel_to_grid() {
        let config = GridConfig::default();
        assert_eq!(pixel_to_grid(50, 50, &config), "A1");
        assert_eq!(pixel_to_grid(150, 50, &config), "B1");
        assert_eq!(pixel_to_grid(50, 150, &config), "A2");
        assert_eq!(pixel_to_grid(250, 250, &config), "C3");
    }

    #[test]
    fn test_grid_to_pixel() {
        let config = GridConfig::default();
        assert_eq!(grid_to_pixel("A1", &config).unwrap(), (50, 50));
        assert_eq!(grid_to_pixel("B1", &config).unwrap(), (150, 50));
        assert_eq!(grid_to_pixel("A2", &config).unwrap(), (50, 150));
        assert_eq!(grid_to_pixel("C3", &config).unwrap(), (250, 250));
    }

    #[test]
    fn test_split_cell_reference() {
        assert_eq!(split_cell_reference("A1").unwrap(), ("A".to_string(), "1".to_string()));
        assert_eq!(split_cell_reference("B10").unwrap(), ("B".to_string(), "10".to_string()));
        assert_eq!(split_cell_reference("AA25").unwrap(), ("AA".to_string(), "25".to_string()));

        assert!(split_cell_reference("1A").is_err());
        assert!(split_cell_reference("A").is_err());
        assert!(split_cell_reference("1").is_err());
        assert!(split_cell_reference("").is_err());
    }
}
