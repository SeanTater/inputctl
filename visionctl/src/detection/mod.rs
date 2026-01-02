//! Pure Rust template matching using Normalized Cross-Correlation (NCC).
//!
//! No Python dependencies - fully self-contained.

use crate::{Error, Result};
use image::{GrayImage, imageops};
use serde::{Deserialize, Serialize};

/// Detection result with location and confidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    /// Center X coordinate
    pub x: i32,
    /// Center Y coordinate
    pub y: i32,
    /// Bounding box [x1, y1, x2, y2]
    pub box_coords: [i32; 4],
    /// Detection confidence (0.0 to 1.0)
    pub confidence: f32,
    /// Detection method used
    pub method: String,
    /// Optional label
    #[serde(default)]
    pub label: Option<String>,
}

/// Maximum dimension before downsampling for speed
const MAX_SEARCH_DIM: u32 = 1920;

/// Find a template image in a screenshot using multi-scale NCC.
///
/// Uses a two-pass approach for large images:
/// 1. Coarse search at reduced resolution
/// 2. Refine matches at full resolution
///
/// # Arguments
/// * `screenshot_path` - Path to the screenshot PNG
/// * `template_path` - Path to the template image to find
/// * `threshold` - Confidence threshold (0.0 to 1.0, default 0.8)
///
/// # Returns
/// Vector of detections sorted by confidence (highest first)
pub fn find_template(
    screenshot_path: &str,
    template_path: &str,
    threshold: Option<f32>,
) -> Result<Vec<Detection>> {
    let threshold = threshold.unwrap_or(0.8);

    // Load images
    let screenshot_full = image::open(screenshot_path)
        .map_err(|e| Error::ScreenshotFailed(format!("Failed to load screenshot: {}", e)))?
        .to_luma8();

    let template = image::open(template_path)
        .map_err(|e| Error::ScreenshotFailed(format!("Failed to load template: {}", e)))?
        .to_luma8();

    // Determine if we need to downsample for speed
    let max_dim = screenshot_full.width().max(screenshot_full.height());
    let downsample = if max_dim > MAX_SEARCH_DIM {
        max_dim as f32 / MAX_SEARCH_DIM as f32
    } else {
        1.0
    };

    let (screenshot, scale_factor) = if downsample > 1.0 {
        let new_w = (screenshot_full.width() as f32 / downsample) as u32;
        let new_h = (screenshot_full.height() as f32 / downsample) as u32;
        (
            imageops::resize(&screenshot_full, new_w, new_h, imageops::FilterType::Triangle),
            downsample,
        )
    } else {
        (screenshot_full.clone(), 1.0)
    };

    // Scale template to match downsampled screenshot
    let template_scaled = if downsample > 1.0 {
        let new_w = (template.width() as f32 / downsample).max(1.0) as u32;
        let new_h = (template.height() as f32 / downsample).max(1.0) as u32;
        imageops::resize(&template, new_w, new_h, imageops::FilterType::Triangle)
    } else {
        template.clone()
    };

    // Skip if template is too small after scaling or too large
    if template_scaled.width() < 4 || template_scaled.height() < 4 {
        return Ok(Vec::new());
    }
    if template_scaled.width() >= screenshot.width()
        || template_scaled.height() >= screenshot.height()
    {
        return Ok(Vec::new());
    }

    // Single-scale search (we already handle scale via downsample)
    let matches = ncc_match(&screenshot, &template_scaled, threshold);

    // Convert back to full resolution coordinates
    let mut all_matches: Vec<Detection> = Vec::new();
    for (x, y, conf) in matches {
        // Scale coordinates back to original image size
        let full_x = ((x as f32) * scale_factor) as i32;
        let full_y = ((y as f32) * scale_factor) as i32;
        let tw = template.width() as i32;
        let th = template.height() as i32;

        all_matches.push(Detection {
            x: full_x + tw / 2,
            y: full_y + th / 2,
            box_coords: [full_x, full_y, full_x + tw, full_y + th],
            confidence: conf,
            method: "ncc".to_string(),
            label: None,
        });
    }

    // Sort by confidence (highest first)
    all_matches.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    // Non-maximum suppression
    let mut filtered: Vec<Detection> = Vec::new();
    for m in all_matches {
        let dominated = filtered.iter().any(|existing| {
            let dx = (m.x - existing.x).abs();
            let dy = (m.y - existing.y).abs();
            dx < 30 && dy < 30
        });
        if !dominated {
            filtered.push(m);
        }
    }

    Ok(filtered)
}

/// Perform NCC matching at a single scale
fn ncc_match(img: &GrayImage, tmpl: &GrayImage, threshold: f32) -> Vec<(i32, i32, f32)> {
    let img_w = img.width() as i32;
    let img_h = img.height() as i32;
    let tmpl_w = tmpl.width() as i32;
    let tmpl_h = tmpl.height() as i32;

    // Precompute template statistics
    let tmpl_pixels: Vec<f32> = tmpl.pixels().map(|p| p.0[0] as f32).collect();
    let tmpl_mean: f32 = tmpl_pixels.iter().sum::<f32>() / tmpl_pixels.len() as f32;
    let tmpl_std: f32 = (tmpl_pixels
        .iter()
        .map(|&p| (p - tmpl_mean).powi(2))
        .sum::<f32>()
        / tmpl_pixels.len() as f32)
        .sqrt();

    if tmpl_std < 1e-6 {
        // Template is flat (no variation), can't match reliably
        return Vec::new();
    }

    let mut matches = Vec::new();

    // Stride for faster scanning (balance speed vs precision)
    let stride = 2;

    for y in (0..=(img_h - tmpl_h)).step_by(stride) {
        for x in (0..=(img_w - tmpl_w)).step_by(stride) {
            let ncc = compute_ncc(img, x, y, tmpl, tmpl_w, tmpl_h, tmpl_mean, tmpl_std);
            if ncc >= threshold {
                matches.push((x, y, ncc));
            }
        }
    }

    matches
}

/// Compute NCC score for a single position
fn compute_ncc(
    img: &GrayImage,
    x: i32,
    y: i32,
    tmpl: &GrayImage,
    tmpl_w: i32,
    tmpl_h: i32,
    tmpl_mean: f32,
    tmpl_std: f32,
) -> f32 {
    let mut sum_img = 0.0f32;
    let mut sum_img_sq = 0.0f32;
    let mut sum_cross = 0.0f32;
    let n = (tmpl_w * tmpl_h) as f32;

    for ty in 0..tmpl_h {
        for tx in 0..tmpl_w {
            let img_px = img.get_pixel((x + tx) as u32, (y + ty) as u32).0[0] as f32;
            let tmpl_px = tmpl.get_pixel(tx as u32, ty as u32).0[0] as f32;

            sum_img += img_px;
            sum_img_sq += img_px * img_px;
            sum_cross += img_px * (tmpl_px - tmpl_mean);
        }
    }

    let img_mean = sum_img / n;
    let img_var = sum_img_sq / n - img_mean * img_mean;
    let img_std = img_var.max(0.0).sqrt();

    if img_std < 1e-6 {
        return 0.0;
    }

    // NCC formula
    let ncc = sum_cross / (n * img_std * tmpl_std);
    ncc.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detection_struct() {
        let det = Detection {
            x: 100,
            y: 200,
            box_coords: [50, 150, 150, 250],
            confidence: 0.95,
            method: "ncc".to_string(),
            label: None,
        };
        assert_eq!(det.x, 100);
        assert_eq!(det.confidence, 0.95);
    }
}
