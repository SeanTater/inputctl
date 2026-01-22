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

    // Build FFmpeg args, optionally adding a scale filter
    let video_size = format!("{}x{}", width, height);
    let fps_str = config.fps.to_string();
    let crf_str = config.crf.to_string();
    let video_path_str = video_path.to_str().unwrap();

    // Scale filter: fit within max resolution, preserve aspect ratio, ensure even dimensions
    let scale_filter = config.max_resolution.map(|(max_w, max_h)| {
        format!(
            "scale='min({},iw)':'min({},ih)':force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2",
            max_w, max_h
        )
    });

    if let Some(ref filter) = scale_filter {
        println!("Output scale filter: {}", filter);
    }

    let mut args = vec![
        "-f", "rawvideo",
        "-pixel_format", "rgba",
        "-video_size", &video_size,
        "-framerate", &fps_str,
        "-i", "-",
    ];

    if let Some(ref filter) = scale_filter {
        args.extend(["-vf", filter.as_str()]);
    }

    args.extend([
        "-c:v", "libx264",
        "-preset", config.preset.as_str(),
        "-crf", &crf_str,
        "-pix_fmt", "yuv444p",
        "-y", video_path_str,
    ]);

    let mut ffmpeg = Command::new("ffmpeg")
        .args(&args)
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
