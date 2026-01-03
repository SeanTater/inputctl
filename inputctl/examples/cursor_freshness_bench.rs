//! Measures the actual latency from mouse movement to cursor_pos() update
//!
//! This tests the real feedback loop latency, not just atomic read speed.

use std::time::{Duration, Instant};

fn main() {
    println!("=== Cursor Position Freshness Benchmark ===\n");

    println!("Creating InputCtl (starting cursor daemon)...");
    let mut ctl = inputctl::InputCtl::new().expect("Failed to create InputCtl");

    // Wait for daemon to fully initialize
    std::thread::sleep(Duration::from_millis(500));

    if ctl.cursor_is_stale(1000) {
        println!("ERROR: Cursor tracking not working (no updates received)");
        println!("Make sure you're running on KDE/KWin with a valid DBus session.\n");
        return;
    }

    let (start_x, start_y) = ctl.cursor_pos();
    println!("Initial cursor position: ({}, {})\n", start_x, start_y);

    // Test: move mouse and measure how long until we see the new position
    let iterations = 100;
    let mut latencies = Vec::with_capacity(iterations);

    println!("Measuring update latency ({} iterations)...\n", iterations);

    for i in 0..iterations {
        let (before_x, before_y) = ctl.cursor_pos();

        // Move mouse by small amount
        let dx = if i % 2 == 0 { 10 } else { -10 };
        let start = Instant::now();

        ctl.move_mouse(dx, 0).expect("move_mouse failed");

        // Poll until position changes (with timeout)
        let timeout = Duration::from_millis(100);
        loop {
            let (x, y) = ctl.cursor_pos();
            if x != before_x || y != before_y {
                let elapsed = start.elapsed();
                latencies.push(elapsed.as_micros() as f64);
                break;
            }
            if start.elapsed() > timeout {
                println!("  Iteration {}: TIMEOUT (position didn't change)", i);
                break;
            }
            // Small sleep to avoid burning CPU
            std::thread::sleep(Duration::from_micros(10));
        }
    }

    if latencies.is_empty() {
        println!("ERROR: No successful measurements");
        return;
    }

    // Calculate statistics
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = latencies.first().unwrap();
    let max = latencies.last().unwrap();
    let avg = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let median = latencies[latencies.len() / 2];
    let p95 = latencies[(latencies.len() as f64 * 0.95) as usize];
    let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];

    println!("=== Update Latency (time from move_mouse to cursor_pos change) ===\n");
    println!("Successful: {}/{}", latencies.len(), iterations);
    println!("Min:    {:.1} μs ({:.2} ms)", min, min / 1000.0);
    println!("Max:    {:.1} μs ({:.2} ms)", max, max / 1000.0);
    println!("Avg:    {:.1} μs ({:.2} ms)", avg, avg / 1000.0);
    println!("Median: {:.1} μs ({:.2} ms)", median, median / 1000.0);
    println!("P95:    {:.1} μs ({:.2} ms)", p95, p95 / 1000.0);
    println!("P99:    {:.1} μs ({:.2} ms)", p99, p99 / 1000.0);

    let target_us = 10_000.0; // 10ms target for servo control
    println!(
        "\nFast enough for servo control (<10ms)? {}",
        if p95 < target_us { "YES" } else { "NO" }
    );

    // Return cursor to original position
    let (end_x, _) = ctl.cursor_pos();
    let return_dx = start_x - end_x;
    if return_dx != 0 {
        let _ = ctl.move_mouse(return_dx, 0);
    }
}
