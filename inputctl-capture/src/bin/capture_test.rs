//! Test capture timing to find where lag comes from
//!
//! Run with: cargo run --release -p inputctl-capture --bin capture_test

use inputctl_capture::capture::PortalCapture;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Opening portal capture...\n");

    let capture = PortalCapture::connect(None)?;

    println!("Capture started! Play your game and watch for timing spikes.");
    println!("Press Ctrl+C to stop.\n");

    let running = Arc::new(AtomicBool::new(true));
    let running_ctrlc = running.clone();

    ctrlc::set_handler(move || {
        running_ctrlc.store(false, Ordering::SeqCst);
    })?;

    let mut frame_count = 0u64;
    let mut total_time = Duration::ZERO;
    let mut max_time = Duration::ZERO;
    let mut spikes = 0u64; // frames taking > 40ms (lower threshold)

    let start = Instant::now();

    while running.load(Ordering::SeqCst) {
        let frame_start = Instant::now();

        match capture.next_frame(Duration::from_millis(100)) {
            Ok(_frame) => {
                let elapsed = frame_start.elapsed();
                total_time += elapsed;
                frame_count += 1;

                if elapsed > max_time {
                    max_time = elapsed;
                }

                if elapsed > Duration::from_millis(50) {
                    spikes += 1;
                    println!("SPIKE: frame {} took {:?}", frame_count, elapsed);
                }

                if frame_count % 100 == 0 {
                    let avg = total_time / frame_count as u32;
                    let overall_elapsed = start.elapsed().as_secs_f64();
                    let fps = frame_count as f64 / overall_elapsed;
                    println!(
                        "Frames: {}, FPS: {:.1}, Avg: {:?}, Max: {:?}, Spikes(>50ms): {}",
                        frame_count, fps, avg, max_time, spikes
                    );
                }
            }
            Err(e) => {
                eprintln!("Frame error: {}", e);
            }
        }
    }

    println!("\n--- Final Stats ---");
    println!("Total frames: {}", frame_count);
    println!(
        "Spikes (>50ms): {} ({:.1}%)",
        spikes,
        100.0 * spikes as f64 / frame_count as f64
    );
    println!("Max frame time: {:?}", max_time);
    println!(
        "Avg frame time: {:?}",
        total_time / frame_count.max(1) as u32
    );

    Ok(())
}
