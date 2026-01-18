//! Standalone recorder CLI for capturing supervised gameplay data.

use clap::Parser;
use inputctl_capture::{run_recorder, Region};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "inputctl-record")]
#[command(about = "Record supervised gameplay data (video + input events)")]
#[command(version)]
struct Cli {
    /// Output directory for the dataset
    #[arg(short, long, default_value = "dataset")]
    output: PathBuf,

    /// Target recording FPS
    #[arg(long, default_value_t = 10)]
    fps: u64,

    /// x264 preset (lower CPU = faster presets)
    #[arg(long, default_value = "ultrafast")]
    preset: String,

    /// x264 CRF (higher = smaller, less CPU)
    #[arg(long, default_value_t = 23)]
    crf: u8,

    /// Input device path (optional, will prompt if not provided)
    #[arg(long)]
    device: Option<String>,

    /// Limit recording to a specific region (x,y,w,h)
    #[arg(long, value_parser = parse_region)]
    region: Option<Region>,

    /// Stop recording after this many seconds
    #[arg(long)]
    max_seconds: Option<u64>,

    /// Print performance stats every N seconds
    #[arg(long)]
    stats_interval: Option<u64>,
}

fn parse_region(s: &str) -> Result<Region, String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 4 {
        return Err("Region must be x,y,w,h".to_string());
    }
    let x = parts[0].parse().map_err(|_| "Invalid x".to_string())?;
    let y = parts[1].parse().map_err(|_| "Invalid y".to_string())?;
    let w = parts[2].parse().map_err(|_| "Invalid w".to_string())?;
    let h = parts[3].parse().map_err(|_| "Invalid h".to_string())?;
    Ok(Region {
        x,
        y,
        width: w,
        height: h,
    })
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    run_recorder(
        cli.output,
        cli.fps,
        cli.preset,
        cli.crf,
        cli.device,
        cli.region,
        cli.max_seconds,
        cli.stats_interval,
    )?;

    Ok(())
}
