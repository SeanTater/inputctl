/// Simple Screenshot Test - No LLM Required
///
/// This example captures a screenshot with grid overlay and saves it to a file.
/// Useful for testing that the screenshot functionality works without needing
/// an LLM backend.
///
/// Usage:
///   cargo run --example screenshot_test --release
///
/// Requirements:
///   - KDE Plasma 6.0+ with KWin
///   - Desktop file with screenshot permission (see README)

use visionctl::VisionCtl;
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== VisionCtl Screenshot Test ===\n");

    // Create headless controller (no LLM needed)
    let ctl = VisionCtl::new_headless();

    println!("Step 1: Capturing screenshot with grid overlay...");
    let png_bytes = ctl.screenshot_with_grid()?;
    println!("✓ Screenshot captured: {} bytes\n", png_bytes.len());

    // Save to file
    let output_path = "/tmp/visionctl_test_screenshot.png";
    println!("Step 2: Saving screenshot to {}...", output_path);
    let mut file = File::create(output_path)?;
    file.write_all(&png_bytes)?;
    println!("✓ Screenshot saved\n");

    println!("=== Test Complete ===");
    println!("\nYou can now open the screenshot to verify the grid overlay:");
    println!("  xdg-open {}", output_path);
    println!("\nThe grid should show letter-number coordinates (A1, B2, C3, etc.)");
    println!("with 100px cells by default.\n");

    Ok(())
}
