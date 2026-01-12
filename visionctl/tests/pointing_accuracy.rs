use visionctl::{VisionCtl, parse_coordinates};
use image::{Rgb, RgbImage};
use imageproc::drawing::draw_text_mut;
use ab_glyph::{FontArc, PxScale};
use std::fs;

#[test]
fn test_pointing_accuracy_resolutions() {
    let resolutions = vec![
        (1280, 720),
        (1920, 1080),
        (2560, 1440),
        (3840, 2160),
    ];

    let ctl = VisionCtl::new_from_config().expect("Failed to load config");
    let font_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf";
    let font_data = fs::read(font_path).expect("Failed to read font file");
    let font = FontArc::try_from_vec(font_data).expect("Failed to load font");

    for (w, h) in resolutions {
        for font_size in [24.0, 12.0] {
            println!("Testing resolution {}x{} with font size {}", w, h, font_size);
            let mut img = RgbImage::new(w, h);
            let scale = PxScale::from(font_size);
            
            let tx = (w as f32 * 0.7) as i32;
            let ty = (h as f32 * 0.3) as i32;
            
            // Draw target
            draw_text_mut(&mut img, Rgb([255, 255, 255]), tx, ty, scale, &font, "Target Button");

            // Distractors - add many more distractors to make it harder
            for i in 1..20 {
                let dx = ((i * 137) % w as usize) as i32;
                let dy = ((i * 263) % h as usize) as i32;
                draw_text_mut(&mut img, Rgb([150, 150, 150]), dx, dy, scale, &font, &format!("Button {}", i));
            }
            
            // Look-alikes near the target
            draw_text_mut(&mut img, Rgb([200, 200, 200]), tx - 100, ty + 100, scale, &font, "Target Buton");
            draw_text_mut(&mut img, Rgb([200, 200, 200]), tx + 100, ty - 50, scale, &font, "Target Bottom");
            draw_text_mut(&mut img, Rgb([200, 200, 200]), tx, ty + 50, scale, &font, "Taget Button");
            
            // Distant look-alike
            draw_text_mut(&mut img, Rgb([200, 200, 200]), (w as f32 * 0.2) as i32, (h as f32 * 0.8) as i32, scale, &font, "Targer Button");
            
            // Convert to PNG bytes
            let mut png_bytes = Vec::new();
            let mut cursor = std::io::Cursor::new(&mut png_bytes);
            img.write_to(&mut cursor, image::ImageFormat::Png).expect("Failed to write PNG");

            let prompt = "Find the 'Target Button' and return its coordinates.";
            let response = ctl.query_pointing_image(&png_bytes, prompt);
            
            match response {
                Ok(res) => {
                    if let Some((rx, ry)) = parse_coordinates(&res) {
                        let target_norm_x = tx * 1000 / w as i32;
                        let target_norm_y = ty * 1000 / h as i32;
                        
                        let dx = (rx - target_norm_x).abs();
                        let dy = (ry - target_norm_y).abs();
                        
                        println!("Result: {}x{} @ {}px -> Target: ({}, {}), Recv: ({}, {}), Offset: ({}, {})", w, h, font_size, target_norm_x, target_norm_y, rx, ry, dx, dy);
                        
                        if dx >= 50 || dy >= 50 {
                            println!("FAIL: Low accuracy at {}x{} with {}px font", w, h, font_size);
                        }
                    } else {
                        println!("FAIL: Could not parse response for {}x{} @ {}px: {}", w, h, font_size, res);
                    }
                },
                Err(e) => {
                    println!("ERROR: LLM request failed at {}x{}: {}", w, h, e);
                }
            }
        }
    }
}
