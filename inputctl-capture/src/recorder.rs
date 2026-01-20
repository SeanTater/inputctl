use crate::error::{Error, Result};
use crate::primitives::screen::{list_windows, Region};
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

fn write_frames_parquet(path: &PathBuf, frames: &[FrameTiming]) -> std::result::Result<(), Box<dyn std::error::Error>> {
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

fn write_inputs_parquet(path: &PathBuf, events: &[EventRecord]) -> std::result::Result<(), Box<dyn std::error::Error>> {
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
    let key_codes: Vec<Option<i32>> = events.iter().map(|e| e.key_code.map(|k| k as i32)).collect();
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

fn clamp_crop_region(frame_width: u32, frame_height: u32, region: Region) -> Result<Region> {
    if region.x < 0 || region.y < 0 {
        return Err(Error::ScreenshotFailed(format!(
            "Crop origin must be non-negative: {:?}",
            region
        )));
    }

    let max_x = frame_width as i32;
    let max_y = frame_height as i32;
    let end_x = region.x + region.width as i32;
    let end_y = region.y + region.height as i32;

    if end_x > max_x || end_y > max_y {
        return Err(Error::ScreenshotFailed(format!(
            "Crop region {:?} out of bounds for frame {}x{}",
            region, frame_width, frame_height
        )));
    }

    let mut adjusted = region;
    if adjusted.width % 2 != 0 {
        adjusted.width -= 1;
    }
    if adjusted.height % 2 != 0 {
        adjusted.height -= 1;
    }

    if adjusted.width == 0 || adjusted.height == 0 {
        return Err(Error::ScreenshotFailed(format!(
            "Crop region {:?} is too small after even adjustment",
            region
        )));
    }

    Ok(adjusted)
}

fn crop_rgba(frame_width: u32, frame_height: u32, rgba: &[u8], region: Region) -> Result<Vec<u8>> {
    let expected_len = (frame_width * frame_height * 4) as usize;
    if rgba.len() < expected_len {
        return Err(Error::ScreenshotFailed(format!(
            "Frame buffer too small: {} < {}",
            rgba.len(),
            expected_len
        )));
    }

    let bytes_per_pixel = 4usize;
    let src_stride = frame_width as usize * bytes_per_pixel;
    let dst_stride = region.width as usize * bytes_per_pixel;
    let start_x = region.x as usize * bytes_per_pixel;
    let start_y = region.y as usize;

    let mut out = vec![0u8; (region.width * region.height * 4) as usize];
    for row in 0..region.height as usize {
        let src_row = (start_y + row) * src_stride + start_x;
        let dst_row = row * dst_stride;
        let src_slice = &rgba[src_row..src_row + dst_stride];
        let dst_slice = &mut out[dst_row..dst_row + dst_stride];
        dst_slice.copy_from_slice(src_slice);
    }

    Ok(out)
}

fn scale_region_for_frame(
    region: Region,
    frame_width: u32,
    frame_height: u32,
    stream_size: Option<(u32, u32)>,
) -> Region {
    let Some((stream_width, stream_height)) = stream_size else {
        return region;
    };
    if stream_width == 0 || stream_height == 0 {
        return region;
    }

    let scale_x = frame_width as f32 / stream_width as f32;
    let scale_y = frame_height as f32 / stream_height as f32;
    if (scale_x - scale_y).abs() > 0.01 || scale_x <= 0.0 {
        return region;
    }

    let scaled = Region {
        x: (region.x as f32 * scale_x).round() as i32,
        y: (region.y as f32 * scale_y).round() as i32,
        width: (region.width as f32 * scale_x).round() as u32,
        height: (region.height as f32 * scale_y).round() as u32,
    };

    scaled
}

fn parse_region_string(region: &str) -> Result<Region> {
    let parts: Vec<&str> = region.trim().split(',').collect();
    if parts.len() != 4 {
        return Err(Error::ScreenshotFailed(format!(
            "Region must be x,y,w,h; got '{}'",
            region
        )));
    }
    let x = parts[0]
        .parse()
        .map_err(|_| Error::ScreenshotFailed("Invalid x".to_string()))?;
    let y = parts[1]
        .parse()
        .map_err(|_| Error::ScreenshotFailed("Invalid y".to_string()))?;
    let width = parts[2]
        .parse()
        .map_err(|_| Error::ScreenshotFailed("Invalid w".to_string()))?;
    let height = parts[3]
        .parse()
        .map_err(|_| Error::ScreenshotFailed("Invalid h".to_string()))?;
    Ok(Region {
        x,
        y,
        width,
        height,
    })
}

fn slurp_region(stream_size: Option<(u32, u32)>, frame_width: u32, frame_height: u32) -> Result<Option<Region>> {
    let output = match Command::new("slurp")
        .args(["-f", "%x,%y,%w,%h"])
        .output()
    {
        Ok(output) => output,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("slurp not found; install it or pass --region x,y,w,h");
            return Ok(None);
        }
        Err(e) => {
            return Err(Error::ScreenshotFailed(format!(
                "Failed to run slurp: {}",
                e
            )));
        }
    };

    if !output.status.success() {
        eprintln!("slurp exited with {}", output.status);
        return Ok(None);
    }

    let raw = String::from_utf8_lossy(&output.stdout);
    if raw.trim().is_empty() {
        return Ok(None);
    }

    let region = parse_region_string(&raw)?;
    let scaled = scale_region_for_frame(region, frame_width, frame_height, stream_size);
    Ok(Some(scaled))
}

fn pick_window_region(
    stream_size: Option<(u32, u32)>,
    frame_width: u32,
    frame_height: u32,
) -> Result<Option<Region>> {
    let windows = match list_windows() {
        Ok(windows) => windows,
        Err(e) => {
            eprintln!("Failed to list windows: {}", e);
            return slurp_region(stream_size, frame_width, frame_height);
        }
    };

    if windows.is_empty() {
        eprintln!("No windows found for selection.");
        return slurp_region(stream_size, frame_width, frame_height);
    }

    if !std::io::stdout().is_terminal() {
        return Err(Error::ScreenshotFailed(
            "Window selection requires a terminal; pass --region".to_string(),
        ));
    }

    let selections: Vec<String> = windows
        .iter()
        .map(|w| format!("{} ({}x{} @ {},{})", w.title, w.region.width, w.region.height, w.region.x, w.region.y))
        .collect();
    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select Window For Crop")
        .items(&selections)
        .default(0)
        .interact()
        .map_err(|e| Error::ScreenshotFailed(format!("Window selection failed: {}", e)))?;

    let picked = windows
        .get(selection)
        .map(|w| w.region)
        .map(|r| scale_region_for_frame(r, frame_width, frame_height, stream_size))
        .unwrap_or_else(|| windows[0].region);
    Ok(Some(picked))
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

    let target_region = if let Some(r) = region_hint {
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
    println!("{}", capture.stream_debug_line());
    let stream_size = capture.stream_size();
    let first_frame = capture
        .next_frame(Duration::from_millis(2000))
        .map_err(|e| Error::ScreenshotFailed(format!("Capture failed: {}", e)))?;
    println!(
        "Captured source frame: {}x{} ({} bytes)",
        first_frame.width,
        first_frame.height,
        first_frame.rgba.len()
    );

    let base_region = if let Some(region) = target_region {
        region
    } else if let Some(region) = capture.stream_region() {
        println!("Using portal stream region: {:?}", region);
        region
    } else if let Some(region) = pick_window_region(stream_size, first_frame.width, first_frame.height)? {
        println!("Using selected window region: {:?}", region);
        region
    } else {
        println!("Portal did not report a stream region; using full frame.");
        Region {
            x: 0,
            y: 0,
            width: first_frame.width,
            height: first_frame.height,
        }
    };
    let crop_region = clamp_crop_region(first_frame.width, first_frame.height, base_region)?;
    let width = crop_region.width;
    let height = crop_region.height;

    println!(
        "Recording dimensions: {}x{} (crop {},{} @ {}x{})",
        width, height, crop_region.x, crop_region.y, width, height
    );
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

    let mut pending_frame = Some(first_frame);

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
        let expected_len = (frame.width * frame.height * 4) as usize;
        if rgba.len() != expected_len {
            if rgba.len() > expected_len {
                rgba.truncate(expected_len);
            } else {
                rgba.resize(expected_len, 0);
            }
        }

        let rgba = crop_rgba(frame.width, frame.height, &rgba, crop_region)?;

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
