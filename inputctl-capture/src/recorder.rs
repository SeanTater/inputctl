use crate::error::{Error, Result};
use arrow::array::{Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use dialoguer::{theme::ColorfulTheme, Select};
use evdev::{Device, Key};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs::{self, File};
use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Video encoder selection.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum Encoder {
    /// Automatically select best available encoder (VAAPI > x264)
    #[default]
    Auto,
    /// Software x264 encoder
    X264,
    /// Intel/AMD VAAPI hardware encoder
    Vaapi,
}

impl std::str::FromStr for Encoder {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(Encoder::Auto),
            "x264" | "software" | "sw" => Ok(Encoder::X264),
            "vaapi" | "hw" | "hardware" => Ok(Encoder::Vaapi),
            _ => Err(format!("Unknown encoder '{}'. Use: auto, x264, vaapi", s)),
        }
    }
}

/// Configuration for the recorder.
#[derive(Clone)]
pub struct RecorderConfig {
    /// Output directory for the dataset
    pub output_dir: PathBuf,
    /// Target recording FPS
    pub fps: u64,
    /// x264 preset (ultrafast, veryfast, fast, medium, etc.)
    pub preset: String,
    /// x264 CRF quality (0-51, lower = better quality, larger file)
    pub crf: u8,
    /// Input device path (if None, will prompt interactively)
    pub device_path: Option<String>,
    /// Stop recording after this many seconds
    pub max_seconds: Option<u64>,
    /// Print performance stats every N seconds
    pub stats_interval: Option<u64>,
    /// Maximum output resolution (width, height). Aspect ratio preserved.
    pub max_resolution: Option<(u32, u32)>,
    /// Encoder to use (auto, x264, vaapi)
    pub encoder: Encoder,
}

#[derive(Clone)]
struct EventRecord {
    timestamp: u128,
    key_code: u16,
    key_name: String,
    state: &'static str,
}

#[derive(Clone)]
struct FrameTiming {
    frame_idx: u64,
    timestamp: u128,
}

fn write_frames_parquet(
    path: &PathBuf,
    frames: &[FrameTiming],
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let schema = Schema::new(vec![
        Field::new("frame_idx", DataType::Int64, false),
        Field::new("timestamp", DataType::Int64, false),
    ]);

    let frame_indices: Vec<i64> = frames.iter().map(|f| f.frame_idx as i64).collect();
    let timestamps: Vec<i64> = frames.iter().map(|f| f.timestamp as i64).collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from(frame_indices)),
            Arc::new(Int64Array::from(timestamps)),
        ],
    )?;

    let file = File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, Arc::new(schema), Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

/// Check if VAAPI H.264 encoding is available via ffmpeg.
fn detect_vaapi() -> Option<String> {
    // Check if render device exists
    let render_device = "/dev/dri/renderD128";
    if !std::path::Path::new(render_device).exists() {
        return None;
    }

    // Test if ffmpeg can actually use VAAPI for H.264 encoding
    // This runs a quick encode test with null output
    let result = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel", "error",
            "-vaapi_device", render_device,
            "-f", "lavfi",
            "-i", "color=black:s=64x64:d=0.1",
            "-vf", "format=nv12,hwupload",
            "-c:v", "h264_vaapi",
            "-f", "null",
            "-",
        ])
        .output();

    match result {
        Ok(output) if output.status.success() => Some(render_device.to_string()),
        _ => None,
    }
}

/// Resolve encoder choice, falling back if hardware unavailable.
fn resolve_encoder(requested: &Encoder) -> (Encoder, Option<String>) {
    match requested {
        Encoder::Auto => {
            if let Some(device) = detect_vaapi() {
                (Encoder::Vaapi, Some(device))
            } else {
                (Encoder::X264, None)
            }
        }
        Encoder::Vaapi => {
            if let Some(device) = detect_vaapi() {
                (Encoder::Vaapi, Some(device))
            } else {
                eprintln!("Warning: VAAPI not available, falling back to x264");
                (Encoder::X264, None)
            }
        }
        Encoder::X264 => (Encoder::X264, None),
    }
}

fn write_inputs_parquet(
    path: &PathBuf,
    events: &[EventRecord],
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let schema = Schema::new(vec![
        Field::new("timestamp", DataType::Int64, false),
        Field::new("key_code", DataType::Int32, false),
        Field::new("key_name", DataType::Utf8, false),
        Field::new("state", DataType::Utf8, false),
    ]);

    let timestamps: Vec<i64> = events.iter().map(|e| e.timestamp as i64).collect();
    let key_codes: Vec<i32> = events.iter().map(|e| e.key_code as i32).collect();
    let key_names: Vec<&str> = events.iter().map(|e| e.key_name.as_str()).collect();
    let states: Vec<&str> = events.iter().map(|e| e.state).collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from(timestamps)),
            Arc::new(Int32Array::from(key_codes)),
            Arc::new(StringArray::from(key_names)),
            Arc::new(StringArray::from(states)),
        ],
    )?;

    let file = File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, Arc::new(schema), Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

pub fn run_recorder(config: RecorderConfig) -> Result<()> {
    // Check for ffmpeg
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        return Err(Error::ScreenshotFailed(
            "ffmpeg not found in PATH".to_string(),
        ));
    }

    // Select input device
    let device = if let Some(path) = config.device_path {
        Device::open(&path)
            .map_err(|e| Error::ScreenshotFailed(format!("Failed to open device: {}", e)))?
    } else {
        match select_device() {
            Ok(device) => device,
            Err(err) => {
                if std::io::stdout().is_terminal() {
                    return Err(Error::ScreenshotFailed(format!(
                        "Failed to select device: {}",
                        err
                    )));
                }
                return Err(Error::ScreenshotFailed(
                    "Failed to select device: IO error: not a terminal (pass --device)".to_string(),
                ));
            }
        }
    };

    println!("Selected device: {}", device.name().unwrap_or("Unknown"));

    // Prepare output directory
    let timestamp = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S");
    let session_dir = config.output_dir.join(format!("run_{}", timestamp));

    fs::create_dir_all(&session_dir).map_err(|e| {
        Error::ScreenshotFailed(format!("Failed to create output directory: {}", e))
    })?;
    println!("Recording to: {}", session_dir.display());

    // Buffer events in memory for parquet output
    let input_events: Arc<Mutex<Vec<EventRecord>>> = Arc::new(Mutex::new(Vec::new()));
    let frame_timings: Arc<Mutex<Vec<FrameTiming>>> = Arc::new(Mutex::new(Vec::new()));

    let mut stats_file = if config.stats_interval.is_some() {
        Some(
            fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(session_dir.join("stats.jsonl"))
                .map_err(|e| {
                    Error::ScreenshotFailed(format!("Failed to create stats log: {}", e))
                })?,
        )
    } else {
        None
    };

    // Start portal capture and determine dimensions from first frame
    let capture = crate::capture::PortalCapture::connect(None)?;
    let mut pending_frame = Some(
        capture
            .next_frame(Duration::from_millis(2000))
            .map_err(|e| Error::ScreenshotFailed(format!("Capture failed: {}", e)))?,
    );

    let mut width = pending_frame.as_ref().map(|frame| frame.width).unwrap_or(0);
    let mut height = pending_frame
        .as_ref()
        .map(|frame| frame.height)
        .unwrap_or(0);

    // Ensure even dimensions for broader decoder compatibility
    if width % 2 != 0 {
        width -= 1;
    }
    if height % 2 != 0 {
        height -= 1;
    }

    // Trim the first frame to even dimensions if needed.
    if let Some(frame) = pending_frame.take() {
        if frame.width == width && frame.height == height {
            pending_frame = Some(frame);
        } else if frame.width >= width && frame.height >= height {
            let even_len = (width * height * 4) as usize;
            if frame.rgba.len() >= even_len {
                let mut rgba = frame.rgba;
                rgba.truncate(even_len);
                pending_frame = Some(crate::capture::CaptureFrame {
                    rgba,
                    width,
                    height,
                    timestamp_ms: frame.timestamp_ms,
                });
            }
        }
    }

    println!("Recording dimensions: {}x{}", width, height);
    let video_path = session_dir.join("recording.mp4");

    // Resolve encoder choice
    let (encoder, vaapi_device) = resolve_encoder(&config.encoder);
    println!(
        "Encoder: {:?}{}",
        encoder,
        vaapi_device
            .as_ref()
            .map(|d| format!(" ({})", d))
            .unwrap_or_default()
    );

    // Build FFmpeg args based on encoder
    let video_size = format!("{}x{}", width, height);
    let fps_str = config.fps.to_string();
    let qp_str = config.crf.to_string(); // Used as QP for VAAPI, CRF for x264

    // Build filter chain based on encoder and scaling needs
    let filter = match (&encoder, &config.max_resolution) {
        (Encoder::Vaapi, Some((max_w, max_h))) => {
            // VAAPI: convert to nv12, upload to GPU, scale on GPU
            format!(
                "format=nv12,hwupload,scale_vaapi=w='min({},iw)':h='min({},ih)':force_original_aspect_ratio=decrease",
                max_w, max_h
            )
        }
        (Encoder::Vaapi, None) => {
            // VAAPI: just convert and upload
            "format=nv12,hwupload".to_string()
        }
        (_, Some((max_w, max_h))) => {
            // x264 with scaling: CPU scale
            format!(
                "scale='min({},iw)':'min({},ih)':force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2",
                max_w, max_h
            )
        }
        _ => String::new(),
    };

    if !filter.is_empty() {
        println!("Filter: {}", filter);
    }

    let mut args: Vec<String> = Vec::new();

    // VAAPI requires device init before input
    if let Some(ref device) = vaapi_device {
        args.extend([
            "-vaapi_device".to_string(),
            device.clone(),
        ]);
    }

    // Input specification
    args.extend([
        "-f".to_string(), "rawvideo".to_string(),
        "-pixel_format".to_string(), "rgba".to_string(),
        "-video_size".to_string(), video_size,
        "-framerate".to_string(), fps_str,
        "-i".to_string(), "-".to_string(),
    ]);

    // Filter chain
    if !filter.is_empty() {
        args.extend(["-vf".to_string(), filter]);
    }

    // Encoder-specific options
    match encoder {
        Encoder::Vaapi => {
            args.extend([
                "-c:v".to_string(), "h264_vaapi".to_string(),
                "-qp".to_string(), qp_str,
            ]);
        }
        Encoder::X264 | Encoder::Auto => {
            args.extend([
                "-c:v".to_string(), "libx264".to_string(),
                "-preset".to_string(), config.preset.clone(),
                "-crf".to_string(), qp_str,
                "-pix_fmt".to_string(), "yuv444p".to_string(),
            ]);
        }
    }

    // Output
    args.extend([
        "-y".to_string(),
        video_path.to_str().unwrap().to_string(),
    ]);

    let args_ref: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let mut ffmpeg = Command::new("ffmpeg")
        .args(&args_ref)
        .stdin(Stdio::piped())
        .spawn()
        .map_err(|e| Error::ScreenshotFailed(format!("Failed to start ffmpeg: {}", e)))?;

    let mut ffmpeg_stdin = ffmpeg.stdin.take().unwrap();

    let running = Arc::new(AtomicBool::new(true));

    // Start input monitoring thread
    let input_running = running.clone();
    let input_events_clone = input_events.clone();
    thread::spawn(move || {
        monitor_input_stream(device, input_events_clone, input_running);
    });

    // Main capture loop
    println!("Starting recording... Press Ctrl+C to stop.");

    let mut frame_idx = 0;
    let started_at = std::time::Instant::now();
    let mut next_frame_at = started_at;
    let frame_interval = Duration::from_secs_f64(1.0 / config.fps as f64);
    let stats_every = config.stats_interval.map(Duration::from_secs);
    let mut last_stats_at = started_at;
    let mut captured_frames = 0u64;
    let mut dropped_frames = 0u64;
    let mut slow_frames = 0u64;

    // Handle Ctrl+C
    let running_ctrlc = running.clone();
    ctrlc::set_handler(move || {
        println!("\nStopping recording...");
        running_ctrlc.store(false, Ordering::SeqCst);
    })
    .map_err(|e| Error::ScreenshotFailed(format!("Failed to set Ctrl+C handler: {}", e)))?;

    while running.load(Ordering::SeqCst) {
        if let Some(limit) = config.max_seconds {
            if started_at.elapsed().as_secs() >= limit {
                println!("Reached max_seconds={}, stopping...", limit);
                running.store(false, Ordering::SeqCst);
                break;
            }
        }

        let now = std::time::Instant::now();
        if now < next_frame_at {
            thread::sleep(next_frame_at - now);
        }
        if now > next_frame_at + frame_interval {
            dropped_frames += 1;
        }
        next_frame_at += frame_interval;

        let frame = match pending_frame.take() {
            Some(frame) => frame,
            None => match capture.next_frame(Duration::from_millis(2000)) {
                Ok(frame) => frame,
                Err(e) => {
                    eprintln!("Capture failed: {}", e);
                    break;
                }
            },
        };

        let mut rgba = frame.rgba;
        let expected_len = (width * height * 4) as usize;
        if rgba.len() != expected_len {
            if rgba.len() > expected_len {
                rgba.truncate(expected_len);
            } else {
                rgba.resize(expected_len, 0);
            }
        }

        let encode_start = std::time::Instant::now();
        if let Err(e) = ffmpeg_stdin.write_all(&rgba) {
            eprintln!("Failed to write to ffmpeg (pipe closed?): {}", e);
            break;
        }
        if encode_start.elapsed() > frame_interval {
            slow_frames += 1;
        }
        captured_frames += 1;

        let timing = FrameTiming {
            frame_idx,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis(),
        };
        if let Ok(mut timings) = frame_timings.lock() {
            timings.push(timing);
        }

        frame_idx += 1;

        if let Some(interval) = stats_every {
            if last_stats_at.elapsed() >= interval {
                let elapsed = started_at.elapsed().as_secs_f32();
                let fps_actual = if elapsed > 0.0 {
                    captured_frames as f32 / elapsed
                } else {
                    0.0
                };
                if let Some(file) = stats_file.as_mut() {
                    let record = serde_json::json!({
                        "timestamp": SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_millis(),
                        "frames": captured_frames,
                        "fps": fps_actual,
                        "dropped": dropped_frames,
                        "slow_writes": slow_frames,
                    });
                    let _ = writeln!(file, "{}", record);
                }
                last_stats_at = std::time::Instant::now();
            }
        }
    }

    // Cleanup
    drop(ffmpeg_stdin);
    println!("Waiting for encoding to finish...");
    let _ = ffmpeg.wait();

    // Write parquet files
    println!("Writing parquet files...");
    let frames = frame_timings.lock().unwrap();
    if let Err(e) = write_frames_parquet(&session_dir.join("frames.parquet"), &frames) {
        eprintln!("Failed to write frames.parquet: {}", e);
    }
    drop(frames);

    let events = input_events.lock().unwrap();
    if let Err(e) = write_inputs_parquet(&session_dir.join("inputs.parquet"), &events) {
        eprintln!("Failed to write inputs.parquet: {}", e);
    }
    drop(events);

    println!("Recording saved to {}", session_dir.display());
    println!("Total Frames: {}", frame_idx);

    Ok(())
}

fn monitor_input_stream(
    mut device: Device,
    events_buffer: Arc<Mutex<Vec<EventRecord>>>,
    running: Arc<AtomicBool>,
) {
    while running.load(Ordering::Relaxed) {
        match device.fetch_events() {
            Ok(events) => {
                for event in events {
                    if event.event_type() == evdev::EventType::KEY {
                        let val = event.value();
                        let state = match val {
                            0 => "up",
                            1 => "down",
                            2 => "repeat",
                            _ => "unknown",
                        };

                        let key = Key::new(event.code());
                        let record = EventRecord {
                            timestamp: SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap()
                                .as_millis(),
                            key_code: event.code(),
                            key_name: format!("{:?}", key),
                            state,
                        };

                        if let Ok(mut buf) = events_buffer.lock() {
                            buf.push(record);
                        }
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(1));
            }
            Err(_) => {
                break;
            }
        }
    }
}

fn select_device() -> anyhow::Result<Device> {
    let mut devices: Vec<_> = evdev::enumerate().collect();
    devices.retain(|(_, dev)| dev.supported_events().contains(evdev::EventType::KEY));
    if devices.is_empty() {
        return Err(anyhow::anyhow!("No input devices found"));
    }

    let selections: Vec<String> = devices
        .iter()
        .map(|(p, d)| format!("{} ({})", d.name().unwrap_or("?"), p.display()))
        .collect();
    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select Keyboard")
        .items(&selections)
        .default(0)
        .interact()?;

    Ok(Device::open(&devices[selection].0)?)
}
