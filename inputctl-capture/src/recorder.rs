use crate::error::{Error, Result};
use crate::primitives::screen::Region;
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

#[derive(Clone)]
struct EventRecord {
    timestamp: u128,
    event_type: String,
    key_code: Option<u16>,
    key_name: Option<String>,
    state: Option<String>,
    x: Option<i32>,
    y: Option<i32>,
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
        Field::new("event_type", DataType::Utf8, false),
        Field::new("key_code", DataType::Int32, true),
        Field::new("key_name", DataType::Utf8, true),
        Field::new("state", DataType::Utf8, true),
        Field::new("x", DataType::Int32, true),
        Field::new("y", DataType::Int32, true),
    ]);

    let timestamps: Vec<i64> = events.iter().map(|e| e.timestamp as i64).collect();
    let event_types: Vec<&str> = events.iter().map(|e| e.event_type.as_str()).collect();
    let key_codes: Vec<Option<i32>> = events
        .iter()
        .map(|e| e.key_code.map(|k| k as i32))
        .collect();
    let key_names: Vec<Option<&str>> = events.iter().map(|e| e.key_name.as_deref()).collect();
    let states: Vec<Option<&str>> = events.iter().map(|e| e.state.as_deref()).collect();
    let xs: Vec<Option<i32>> = events.iter().map(|e| e.x).collect();
    let ys: Vec<Option<i32>> = events.iter().map(|e| e.y).collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int64Array::from(timestamps)),
            Arc::new(StringArray::from(event_types)),
            Arc::new(Int32Array::from(key_codes)),
            Arc::new(StringArray::from(key_names)),
            Arc::new(StringArray::from(states)),
            Arc::new(Int32Array::from(xs)),
            Arc::new(Int32Array::from(ys)),
        ],
    )?;

    let file = File::create(path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, Arc::new(schema), Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

pub fn run_recorder(
    output_dir: PathBuf,
    fps: u64,
    preset: String,
    crf: u8,
    device_path: Option<String>,
    region_hint: Option<Region>,
    max_seconds: Option<u64>,
    stats_interval: Option<u64>,
) -> Result<()> {
    // Check for ffmpeg
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        return Err(Error::ScreenshotFailed(
            "ffmpeg not found in PATH".to_string(),
        ));
    }

    // 1. Resolve Target Region
    // If window is specified, find it. If region is specified, use it. Else full screen.
    // Note: If both, window takes precedence? Or intersection? Let's say window takes precedence.

    let _target_region = if let Some(r) = region_hint {
        println!("Targeting region: {:?}", r);
        Some(r)
    } else {
        println!("Targeting full screen (portal selection)");
        None
    };

    // 2. Select Input Device
    let device = if let Some(path) = device_path {
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

    // 3. Prepare Output Directory
    let timestamp = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S");
    let session_dir = output_dir.join(format!("run_{}", timestamp));

    fs::create_dir_all(&session_dir).map_err(|e| {
        Error::ScreenshotFailed(format!("Failed to create output directory: {}", e))
    })?;
    println!("Recording to: {}", session_dir.display());

    // Buffer events in memory for parquet output
    let input_events: Arc<Mutex<Vec<EventRecord>>> = Arc::new(Mutex::new(Vec::new()));
    let frame_timings: Arc<Mutex<Vec<FrameTiming>>> = Arc::new(Mutex::new(Vec::new()));

    let mut stats_file = if stats_interval.is_some() {
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

    // 5. Start portal capture and determine dimensions from first frame
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

    // Ensure even dimensions for yuv420p
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

    let mut ffmpeg = Command::new("ffmpeg")
        .args([
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgba", // VisionCtl returns RGBA8
            "-video_size",
            &format!("{}x{}", width, height),
            "-framerate",
            &fps.to_string(),
            "-i",
            "-", // Read from stdin
            "-c:v",
            "libx264",
            "-preset",
            preset.as_str(),
            "-crf",
            &crf.to_string(),
            "-pix_fmt",
            "yuv420p",
            "-y",
            video_path.to_str().unwrap(),
        ])
        .stdin(Stdio::piped())
        .spawn()
        .map_err(|e| Error::ScreenshotFailed(format!("Failed to start ffmpeg: {}", e)))?;

    let mut ffmpeg_stdin = ffmpeg.stdin.take().unwrap();

    // 6. Shared State
    let running = Arc::new(AtomicBool::new(true));

    // 7. Start Threads
    let _running_clone = running.clone();

    let input_running = running.clone();
    let input_events_clone = input_events.clone();
    thread::spawn(move || {
        monitor_input_stream(device, input_events_clone, input_running);
    });

    // 8. Main Rendering Loop (Frame Capture)
    println!("Starting recording... Press Ctrl+C to stop.");

    let mut frame_idx = 0;
    let started_at = std::time::Instant::now();
    let mut next_frame_at = started_at;
    let frame_interval = Duration::from_secs_f64(1.0 / fps as f64);
    let stats_every = stats_interval.map(Duration::from_secs);
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
        if let Some(limit) = max_seconds {
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
                    let _ = writeln!(file, "{}", record.to_string());
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
                            event_type: "key".to_string(),
                            key_code: Some(event.code()),
                            key_name: Some(format!("{:?}", key)),
                            state: Some(state.to_string()),
                            x: None,
                            y: None,
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
    let mut devices = evdev::enumerate()
        .map(|(path, dev)| (path, dev))
        .collect::<Vec<_>>();
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
