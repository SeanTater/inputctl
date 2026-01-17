use std::time::Instant;

fn main() {
    println!("=== Cursor Position Query Benchmark ===\n");

    // Create InputCtl which starts the cursor daemon
    println!("Creating InputCtl (starting cursor daemon)...");
    let ctl = inputctl::InputCtl::new().expect("Failed to create InputCtl");

    // Wait for daemon to fully initialize
    std::thread::sleep(std::time::Duration::from_millis(500));

    // Check if we're getting updates
    if ctl.cursor_is_stale(1000) {
        println!("WARNING: Cursor tracking appears stale (no updates received)");
        println!("Make sure you're running on KDE/KWin with a valid DBus session.\n");
    } else {
        let (x, y) = ctl.cursor_pos();
        println!("Initial cursor position: ({}, {})\n", x, y);
    }

    let iterations = 10000;
    let mut latencies = Vec::with_capacity(iterations);

    // Warm-up
    for _ in 0..100 {
        let _ = ctl.cursor_pos();
    }

    // Benchmark cursor_pos() - should be instant (atomic read)
    println!("Benchmarking cursor_pos() calls...\n");
    for _ in 0..iterations {
        let start = Instant::now();
        let _ = ctl.cursor_pos();
        let elapsed = start.elapsed();
        latencies.push(elapsed.as_nanos() as f64); // Nanoseconds for high precision
    }

    // Calculate statistics
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = latencies.first().unwrap();
    let max = latencies.last().unwrap();
    let avg = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let median = latencies[latencies.len() / 2];
    let p95 = latencies[(latencies.len() as f64 * 0.95) as usize];
    let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];

    println!("Iterations: {}", iterations);
    println!("Min:    {:.0} ns ({:.3} μs)", min, min / 1000.0);
    println!("Max:    {:.0} ns ({:.3} μs)", max, max / 1000.0);
    println!("Avg:    {:.0} ns ({:.3} μs)", avg, avg / 1000.0);
    println!("Median: {:.0} ns ({:.3} μs)", median, median / 1000.0);
    println!("P95:    {:.0} ns ({:.3} μs)", p95, p95 / 1000.0);
    println!("P99:    {:.0} ns ({:.3} μs)", p99, p99 / 1000.0);

    let target_ns = 10_000.0; // 10μs target
    println!(
        "\nFast enough for servo control (<10μs)? {}",
        if p95 < target_ns { "YES ✓" } else { "NO ✗" }
    );

    // Show distribution
    println!("\n=== Latency Distribution ===");
    let buckets = [
        (0.0, 100.0, "0-100ns"),
        (100.0, 500.0, "100-500ns"),
        (500.0, 1000.0, "500ns-1μs"),
        (1000.0, 5000.0, "1-5μs"),
        (5000.0, 10000.0, "5-10μs"),
        (10000.0, 100000.0, "10-100μs"),
    ];
    for (low, high, label) in buckets {
        let count = latencies.iter().filter(|&&x| x >= low && x < high).count();
        let pct = count as f64 / latencies.len() as f64 * 100.0;
        println!("{:>10}: {:>5} ({:>5.1}%)", label, count, pct);
    }

    // Compare with old visionctl::find_cursor()
    println!("\n=== Comparison with visionctl::find_cursor() ===");
    println!("(This is the OLD slow method - 5 iterations only)");
    let mut old_latencies = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        let _ = visionctl::find_cursor();
        let elapsed = start.elapsed();
        old_latencies.push(elapsed.as_millis() as f64);
    }
    let old_avg = old_latencies.iter().sum::<f64>() / old_latencies.len() as f64;
    println!("Old method avg: {:.1} ms", old_avg);
    println!("New method avg: {:.3} μs", avg / 1000.0);
    println!("Speedup: {:.0}x faster!", old_avg * 1_000_000.0 / avg);
}
