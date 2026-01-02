//! Visual object detection using template matching and ML models.
//!
//! This module provides detection capabilities for finding UI elements in screenshots:
//! - Template matching: Find known icons/elements using a reference image
//! - YOLOE: Find objects by text description (requires `ultralytics` Python package)

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::process::Command;

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
    /// Optional label (for YOLOE text prompts)
    #[serde(default)]
    pub label: Option<String>,
}

/// Find the scripts directory relative to the executable or cwd
fn get_scripts_dir() -> Result<std::path::PathBuf> {
    // Try multiple locations
    let candidates = [
        // Relative to executable
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.join("../scripts"))),
        // Workspace scripts directory
        Some(std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts")),
        // Current directory
        Some(std::path::PathBuf::from("scripts")),
        // Absolute fallback
        Some(std::path::PathBuf::from("/home/sean/sandbox/inputctl/visionctl/scripts")),
    ];

    for candidate in candidates.into_iter().flatten() {
        if candidate.join("template_match.py").exists() {
            return Ok(candidate);
        }
    }

    Err(Error::ScreenshotFailed(
        "Could not find scripts directory with template_match.py".into(),
    ))
}

/// Find a template image in a screenshot using OpenCV template matching.
///
/// This uses SIFT feature matching for scale/rotation invariance,
/// falling back to multi-scale template matching.
///
/// # Arguments
/// * `screenshot` - Path to the screenshot PNG
/// * `template` - Path to the template image to find
/// * `threshold` - Confidence threshold (0.0 to 1.0, default 0.7)
///
/// # Returns
/// Vector of detections sorted by confidence (highest first)
pub fn find_template(screenshot: &str, template: &str, threshold: Option<f32>) -> Result<Vec<Detection>> {
    let scripts_dir = get_scripts_dir()?;
    let python = scripts_dir.join(".venv/bin/python");
    let script = scripts_dir.join("template_match.py");

    if !python.exists() {
        return Err(Error::ScreenshotFailed(format!(
            "Python venv not found at {}. Run setup_detection.sh first.",
            python.display()
        )));
    }

    let threshold_str = threshold.unwrap_or(0.7).to_string();

    let output = Command::new(&python)
        .arg(&script)
        .arg(screenshot)
        .arg(template)
        .arg(&threshold_str)
        .output()
        .map_err(|e| Error::ScreenshotFailed(format!("Failed to run template_match.py: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(Error::ScreenshotFailed(format!(
            "template_match.py failed: {}",
            stderr
        )));
    }

    // Parse JSON output
    let stdout = String::from_utf8_lossy(&output.stdout);

    // The Python script outputs JSON with slightly different field names
    #[derive(Deserialize)]
    struct RawDetection {
        x: i32,
        y: i32,
        #[serde(default)]
        box_coords: Option<[i32; 4]>,
        #[serde(rename = "box", default)]
        box_field: Option<Vec<i32>>,
        confidence: f32,
        method: String,
    }

    let raw: Vec<RawDetection> = serde_json::from_str(&stdout)
        .map_err(|e| Error::ScreenshotFailed(format!("Failed to parse detection results: {}", e)))?;

    Ok(raw
        .into_iter()
        .map(|r| {
            let box_coords = r.box_coords.unwrap_or_else(|| {
                r.box_field
                    .map(|v| [v[0], v[1], v[2], v[3]])
                    .unwrap_or([r.x - 50, r.y - 50, r.x + 50, r.y + 50])
            });
            Detection {
                x: r.x,
                y: r.y,
                box_coords,
                confidence: r.confidence,
                method: r.method,
                label: None,
            }
        })
        .collect())
}

/// Find objects by text description using YOLOE.
///
/// Requires `ultralytics` Python package and downloads model on first use (~800MB).
///
/// # Arguments
/// * `screenshot` - Path to the screenshot PNG
/// * `prompts` - Text descriptions to search for (e.g., ["button", "icon"])
/// * `threshold` - Confidence threshold (0.0 to 1.0, default 0.25)
pub fn find_by_text(screenshot: &str, prompts: &[&str], threshold: Option<f32>) -> Result<Vec<Detection>> {
    let scripts_dir = get_scripts_dir()?;
    let python = scripts_dir.join(".venv/bin/python");
    let script = scripts_dir.join("yoloe_detect.py");

    if !script.exists() {
        return Err(Error::ScreenshotFailed(format!(
            "YOLOE script not found at {}",
            script.display()
        )));
    }

    let threshold_str = threshold.unwrap_or(0.25).to_string();

    let mut cmd = Command::new(&python);
    cmd.arg(&script)
        .arg(screenshot)
        .arg("--threshold")
        .arg(&threshold_str)
        .arg("--text");

    for prompt in prompts {
        cmd.arg(prompt);
    }

    let output = cmd
        .output()
        .map_err(|e| Error::ScreenshotFailed(format!("Failed to run yoloe_detect.py: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(Error::ScreenshotFailed(format!(
            "yoloe_detect.py failed: {}",
            stderr
        )));
    }

    parse_yoloe_output(&output.stdout)
}

/// Find objects similar to a reference image using YOLOE visual prompts.
///
/// # Arguments
/// * `screenshot` - Path to the screenshot PNG
/// * `reference` - Path to the reference image to find
/// * `threshold` - Confidence threshold (0.0 to 1.0, default 0.25)
pub fn find_by_image(screenshot: &str, reference: &str, threshold: Option<f32>) -> Result<Vec<Detection>> {
    let scripts_dir = get_scripts_dir()?;
    let python = scripts_dir.join(".venv/bin/python");
    let script = scripts_dir.join("yoloe_detect.py");

    if !script.exists() {
        return Err(Error::ScreenshotFailed(format!(
            "YOLOE script not found at {}",
            script.display()
        )));
    }

    let threshold_str = threshold.unwrap_or(0.25).to_string();

    let output = Command::new(&python)
        .arg(&script)
        .arg(screenshot)
        .arg("--threshold")
        .arg(&threshold_str)
        .arg("--image")
        .arg(reference)
        .output()
        .map_err(|e| Error::ScreenshotFailed(format!("Failed to run yoloe_detect.py: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(Error::ScreenshotFailed(format!(
            "yoloe_detect.py failed: {}",
            stderr
        )));
    }

    parse_yoloe_output(&output.stdout)
}

fn parse_yoloe_output(stdout: &[u8]) -> Result<Vec<Detection>> {
    let stdout_str = String::from_utf8_lossy(stdout);

    #[derive(Deserialize)]
    struct RawYoloeDetection {
        x: i32,
        y: i32,
        #[serde(rename = "box")]
        box_field: Vec<i32>,
        confidence: f32,
        method: String,
        #[serde(default)]
        label: Option<String>,
    }

    let raw: Vec<RawYoloeDetection> = serde_json::from_str(&stdout_str)
        .map_err(|e| Error::ScreenshotFailed(format!("Failed to parse YOLOE results: {}", e)))?;

    Ok(raw
        .into_iter()
        .map(|r| Detection {
            x: r.x,
            y: r.y,
            box_coords: [r.box_field[0], r.box_field[1], r.box_field[2], r.box_field[3]],
            confidence: r.confidence,
            method: r.method,
            label: r.label,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scripts_dir_exists() {
        let result = get_scripts_dir();
        assert!(result.is_ok(), "Scripts directory should be found");
    }
}
