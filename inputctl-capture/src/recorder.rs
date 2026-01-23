use crate::capture::{CaptureFrame, FrameSource, PortalCapture};
use crate::error::{Error, Result};
use crate::recorder_ops::{
    build_ffmpeg_args, normalize_even_dimensions, pipewire_to_ffmpeg_format, run_capture_loop,
    write_summary_outputs,
};
use arrow::array::{Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use dialoguer::{theme::ColorfulTheme, Select};
use evdev::{Device, Key};
use nix::fcntl::{fcntl, FcntlArg, OFlag};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs::{self, File};
use std::io::{BufWriter, IsTerminal, Write};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

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
    /// Maximum output resolution (width, height). Aspect ratio preserved.
    pub max_resolution: Option<(u32, u32)>,
    /// Encoder to use (auto, x264, vaapi)
    pub encoder: Encoder,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InputEvent {
    pub timestamp: u128,
    pub key_code: u16,
    pub key_name: String,
    pub state: &'static str,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FrameTiming {
    pub frame_idx: u64,
    pub timestamp: u128,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RecorderSummary {
    pub frames: Vec<FrameTiming>,
    pub events: Vec<InputEvent>,
    pub captured_frames: u64,
    pub dropped_frames: u64,
    pub slow_frames: u64,
}

pub trait Clock {
    fn now(&self) -> Instant;
    fn now_ms(&self) -> u128;
    fn sleep(&self, duration: Duration);
}

struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }

    fn now_ms(&self) -> u128 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis()
    }

    fn sleep(&self, duration: Duration) {
        thread::sleep(duration);
    }
}

pub trait VideoWriter {
    fn write_frame(&mut self, frame: &[u8]) -> std::io::Result<()>;
    fn flush(&mut self) -> std::io::Result<()>;
    fn finish(&mut self) -> std::io::Result<()>;
}

struct FfmpegWriter {
    process: std::process::Child,
    writer: Option<BufWriter<std::process::ChildStdin>>,
}

impl FfmpegWriter {
    fn new(args: &[String]) -> Result<Self> {
        let args_ref: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
        let mut process = std::process::Command::new("ffmpeg")
            .args(&args_ref)
            .stdin(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| Error::ScreenshotFailed(format!("Failed to start ffmpeg: {}", e)))?;

        let stdin = process
            .stdin
            .take()
            .ok_or_else(|| Error::ScreenshotFailed("Failed to open ffmpeg stdin".to_string()))?;
        let writer = BufWriter::with_capacity(8 * 1024 * 1024, stdin);
        Ok(Self {
            process,
            writer: Some(writer),
        })
    }
}

impl VideoWriter for FfmpegWriter {
    fn write_frame(&mut self, frame: &[u8]) -> std::io::Result<()> {
        if let Some(writer) = self.writer.as_mut() {
            writer.write_all(frame)
        } else {
            Ok(())
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if let Some(writer) = self.writer.as_mut() {
            writer.flush()
        } else {
            Ok(())
        }
    }

    fn finish(&mut self) -> std::io::Result<()> {
        if let Some(mut writer) = self.writer.take() {
            writer.flush()?;
            drop(writer);
        }
        let _ = self.process.wait();
        Ok(())
    }
}

pub trait InputEventSource {
    fn poll_events(&mut self) -> std::io::Result<Vec<InputEvent>>;
}

struct EvdevInputSource {
    device: Device,
}

impl EvdevInputSource {
    fn new(device: Device) -> Self {
        if let Err(e) = set_nonblocking(&device) {
            eprintln!("Failed to set input device non-blocking: {}", e);
        }
        Self { device }
    }
}

impl InputEventSource for EvdevInputSource {
    fn poll_events(&mut self) -> std::io::Result<Vec<InputEvent>> {
        let mut records = Vec::new();
        let events = match self.device.fetch_events() {
            Ok(events) => events,
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                return Ok(records);
            }
            Err(err) => return Err(err),
        };
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
                records.push(InputEvent {
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis(),
                    key_code: event.code(),
                    key_name: format!("{:?}", key),
                    state,
                });
            }
        }
        Ok(records)
    }
}

fn set_nonblocking(device: &Device) -> std::io::Result<()> {
    let flags = OFlag::from_bits_truncate(fcntl(device.as_raw_fd(), FcntlArg::F_GETFL)?);
    let new_flags = flags | OFlag::O_NONBLOCK;
    fcntl(device.as_raw_fd(), FcntlArg::F_SETFL(new_flags))?;
    Ok(())
}

pub(crate) fn write_frames_parquet(
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

pub(crate) fn write_inputs_parquet(
    path: &PathBuf,
    events: &[InputEvent],
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

/// Check if VAAPI H.264 encoding is available via ffmpeg.
fn detect_vaapi() -> Option<String> {
    let render_device = "/dev/dri/renderD128";
    if !std::path::Path::new(render_device).exists() {
        return None;
    }

    let result = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-vaapi_device",
            render_device,
            "-f",
            "lavfi",
            "-i",
            "color=black:s=64x64:d=0.1",
            "-vf",
            "format=nv12,hwupload",
            "-c:v",
            "h264_vaapi",
            "-f",
            "null",
            "-",
        ])
        .output();

    match result {
        Ok(output) if output.status.success() => Some(render_device.to_string()),
        _ => None,
    }
}

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

pub fn run_recorder(config: RecorderConfig) -> Result<()> {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        return Err(Error::ScreenshotFailed(
            "ffmpeg not found in PATH".to_string(),
        ));
    }

    let device = if let Some(path) = config.device_path.clone() {
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

    let timestamp = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S");
    let session_dir = config.output_dir.join(format!("run_{}", timestamp));

    fs::create_dir_all(&session_dir).map_err(|e| {
        Error::ScreenshotFailed(format!("Failed to create output directory: {}", e))
    })?;
    println!("Recording to: {}", session_dir.display());

    let mut capture = PortalCapture::connect(None)?;
    let mut input_source = EvdevInputSource::new(device);
    let mut clock = SystemClock;

    let (mut writer, width, height, pipewire_format, mut pending_frame) =
        prepare_capture(&mut capture, &config, &session_dir)?;
    println!("Recording dimensions: {}x{}", width, height);

    let summary = run_recorder_with_sources(
        &config,
        &mut capture,
        &mut *writer,
        &mut input_source,
        &mut clock,
        width,
        height,
        pipewire_format,
        pending_frame.take(),
    )?;

    write_recorder_outputs(&session_dir, &summary);

    println!("Recording saved to {}", session_dir.display());
    println!("Total Frames: {}", summary.captured_frames);

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn run_recorder_with_sources(
    config: &RecorderConfig,
    frame_source: &mut dyn FrameSource,
    writer: &mut dyn VideoWriter,
    input_source: &mut dyn InputEventSource,
    clock: &mut dyn Clock,
    width: u32,
    height: u32,
    _pipewire_format: String,
    pending_frame: Option<CaptureFrame>,
) -> Result<RecorderSummary> {
    let summary = run_capture_loop(
        config,
        frame_source,
        writer,
        input_source,
        clock,
        width,
        height,
        pending_frame,
    )?;

    if let Err(e) = writer.flush() {
        eprintln!("Warning: failed to flush encoder: {}", e);
    }
    if let Err(e) = writer.finish() {
        eprintln!("Warning: failed to finish encoder: {}", e);
    }

    Ok(summary)
}

fn prepare_capture(
    capture: &mut PortalCapture,
    config: &RecorderConfig,
    session_dir: &Path,
) -> Result<(Box<dyn VideoWriter>, u32, u32, String, Option<CaptureFrame>)> {
    let mut pending_frame = Some(
        capture
            .next_frame(Duration::from_millis(2000))
            .map_err(|e| Error::ScreenshotFailed(format!("Capture failed: {}", e)))?,
    );

    let pipewire_format = pending_frame
        .as_ref()
        .map(|frame| frame.format.clone())
        .unwrap_or_else(|| "BGRx".to_string());

    let (pixel_format, unknown) = pipewire_to_ffmpeg_format(&pipewire_format);
    if let Some(other) = unknown {
        eprintln!("Warning: Unknown format '{}', assuming bgr0", other);
    }
    println!(
        "Pixel format: {} (PipeWire: {})",
        pixel_format, pipewire_format
    );

    let mut width = pending_frame.as_ref().map(|frame| frame.width).unwrap_or(0);
    let mut height = pending_frame
        .as_ref()
        .map(|frame| frame.height)
        .unwrap_or(0);

    if let Some(frame) = pending_frame.take() {
        let (frame, even_width, even_height) = normalize_even_dimensions(frame);
        width = even_width;
        height = even_height;
        pending_frame = Some(frame);
    }

    let video_path = session_dir.join("recording.mp4");
    let (encoder, vaapi_device) = resolve_encoder(&config.encoder);
    println!(
        "Encoder: {:?}{}",
        encoder,
        vaapi_device
            .as_ref()
            .map(|d| format!(" ({})", d))
            .unwrap_or_default()
    );

    let args = build_ffmpeg_args(
        &encoder,
        config.max_resolution,
        width,
        height,
        config.fps,
        config.crf,
        pixel_format,
        &config.preset,
        &video_path,
        vaapi_device.as_deref(),
    );
    let writer = Box::new(FfmpegWriter::new(&args)?);

    Ok((writer, width, height, pipewire_format, pending_frame))
}

fn write_recorder_outputs(session_dir: &Path, summary: &RecorderSummary) {
    write_summary_outputs(session_dir, summary);
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
