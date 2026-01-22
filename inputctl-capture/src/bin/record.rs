//! Standalone recorder CLI for capturing supervised gameplay data.

use clap::Parser;
use inputctl_capture::{run_recorder, RecorderConfig};
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
    #[arg(long, default_value = "veryfast")]
    preset: String,

    /// x264 CRF (higher = smaller, less CPU)
    #[arg(long, default_value_t = 23)]
    crf: u8,

    /// Input device path (optional, will prompt if not provided)
    #[arg(long)]
    device: Option<String>,

    /// Stop recording after this many seconds
    #[arg(long)]
    max_seconds: Option<u64>,

    /// Print performance stats every N seconds
    #[arg(long)]
    stats_interval: Option<u64>,

    /// Maximum output resolution (WxH), e.g. 1280x800. Aspect ratio preserved.
    #[arg(long, default_value = "1280x800")]
    max_resolution: String,
}

fn parse_max_resolution(s: &str) -> Option<(u32, u32)> {
    if s.eq_ignore_ascii_case("none") {
        return None;
    }
    let parts: Vec<&str> = s.split('x').collect();
    if parts.len() == 2 {
        if let (Ok(w), Ok(h)) = (parts[0].parse(), parts[1].parse()) {
            return Some((w, h));
        }
    }
    eprintln!("Warning: Invalid max_resolution '{}', using no limit", s);
    None
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let config = RecorderConfig {
        output_dir: cli.output,
        fps: cli.fps,
        preset: cli.preset,
        crf: cli.crf,
        device_path: cli.device,
        max_seconds: cli.max_seconds,
        stats_interval: cli.stats_interval,
        max_resolution: parse_max_resolution(&cli.max_resolution),
    };

    run_recorder(config)?;

    Ok(())
}
