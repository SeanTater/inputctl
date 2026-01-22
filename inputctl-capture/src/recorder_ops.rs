use crate::capture::FrameSource;
use crate::error::Result;
use crate::recorder::{
    write_frames_parquet, write_inputs_parquet, Clock, Encoder, FrameTiming, InputEvent,
    InputEventSource, RecorderConfig, RecorderSummary, VideoWriter,
};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Duration;

pub fn pipewire_to_ffmpeg_format(format: &str) -> (&'static str, Option<&str>) {
    match format {
        "BGRx" => ("bgr0", None),
        "BGRA" => ("bgra", None),
        "RGBx" => ("rgb0", None),
        "RGBA" => ("rgba", None),
        "xRGB" => ("0rgb", None),
        "ARGB" => ("argb", None),
        "xBGR" => ("0bgr", None),
        "ABGR" => ("abgr", None),
        other => ("bgr0", Some(other)),
    }
}

pub fn normalize_even_dimensions(
    frame: crate::capture::CaptureFrame,
) -> (crate::capture::CaptureFrame, u32, u32) {
    let mut width = frame.width;
    let mut height = frame.height;

    if width % 2 != 0 {
        width = width.saturating_sub(1);
    }
    if height % 2 != 0 {
        height = height.saturating_sub(1);
    }

    if width == frame.width && height == frame.height {
        return (frame, width, height);
    }

    let even_len = (width * height * 4) as usize;
    let mut rgba = frame.rgba;
    if rgba.len() >= even_len {
        rgba.truncate(even_len);
    }
    (
        crate::capture::CaptureFrame {
            rgba,
            width,
            height,
            timestamp_ms: frame.timestamp_ms,
            format: frame.format.clone(),
        },
        width,
        height,
    )
}

pub(crate) fn maybe_write_stats(
    stats: &mut Option<File>,
    now_ms: u128,
    frames: u64,
    fps: f32,
    dropped: u64,
    slow: u64,
) {
    if let Some(file) = stats.as_mut() {
        let record = serde_json::json!({
            "timestamp": now_ms,
            "frames": frames,
            "fps": fps,
            "dropped": dropped,
            "slow_writes": slow,
        });
        let _ = writeln!(file, "{}", record);
    }
}

pub(crate) fn write_summary_outputs(path: &Path, summary: &RecorderSummary) {
    let frames_path = path.join("frames.parquet");
    if let Err(e) = write_frames_parquet(&frames_path, &summary.frames) {
        eprintln!("Failed to write frames.parquet: {}", e);
    }

    let inputs_path = path.join("inputs.parquet");
    if let Err(e) = write_inputs_parquet(&inputs_path, &summary.events) {
        eprintln!("Failed to write inputs.parquet: {}", e);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run_capture_loop(
    config: &RecorderConfig,
    frame_source: &mut dyn FrameSource,
    writer: &mut dyn VideoWriter,
    input_source: &mut dyn InputEventSource,
    clock: &mut dyn Clock,
    width: u32,
    height: u32,
    pending_frame: Option<crate::capture::CaptureFrame>,
    stats: &mut Option<File>,
) -> Result<RecorderSummary> {
    let frame_interval = Duration::from_secs_f64(1.0 / config.fps as f64);
    let stats_every = config.stats_interval.map(Duration::from_secs);
    let started_at = clock.now();
    let mut last_stats_at = started_at;

    let mut frame_idx = 0;
    let mut next_frame_at = started_at;

    let mut captured_frames = 0u64;
    let mut dropped_frames = 0u64;
    let mut slow_frames = 0u64;
    let mut events: Vec<InputEvent> = Vec::new();
    let mut frames: Vec<FrameTiming> = Vec::new();

    let expected_len = (width * height * 4) as usize;
    let mut frame_buffer = vec![0u8; expected_len];

    let mut pending = pending_frame;

    loop {
        if let Some(limit) = config.max_seconds {
            let elapsed = clock.now().duration_since(started_at);
            if elapsed.as_secs() >= limit {
                break;
            }
        }

        let mut now = clock.now();
        if now < next_frame_at {
            clock.sleep(next_frame_at - now);
            now = clock.now();
        }
        if now > next_frame_at + frame_interval {
            dropped_frames += 1;
        }
        next_frame_at += frame_interval;

        let frame = match pending.take() {
            Some(frame) => frame,
            None => match frame_source.next_frame(Duration::from_millis(2000)) {
                Ok(frame) => frame,
                Err(e) => {
                    eprintln!("Capture failed: {}", e);
                    break;
                }
            },
        };

        let copy_len = frame.rgba.len().min(expected_len);
        frame_buffer[..copy_len].copy_from_slice(&frame.rgba[..copy_len]);
        if copy_len < expected_len {
            frame_buffer[copy_len..].fill(0);
        }

        let encode_start = clock.now();
        if let Err(e) = writer.write_frame(&frame_buffer) {
            eprintln!("Failed to write to encoder: {}", e);
            break;
        }
        if clock.now().duration_since(encode_start) > frame_interval {
            slow_frames += 1;
        }
        captured_frames += 1;

        frames.push(FrameTiming {
            frame_idx,
            timestamp: clock.now_ms(),
        });
        frame_idx += 1;

        match input_source.poll_events() {
            Ok(mut batch) => events.append(&mut batch),
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                clock.sleep(Duration::from_millis(1));
            }
            Err(_) => {}
        }

        if let Some(interval) = stats_every {
            let last_elapsed = clock.now().duration_since(last_stats_at);
            if last_elapsed >= interval {
                let elapsed = clock.now().duration_since(started_at).as_secs_f32();
                let fps_actual = if elapsed > 0.0 {
                    captured_frames as f32 / elapsed
                } else {
                    0.0
                };
                maybe_write_stats(
                    stats,
                    clock.now_ms(),
                    captured_frames,
                    fps_actual,
                    dropped_frames,
                    slow_frames,
                );
                last_stats_at = clock.now();
            }
        }
    }

    Ok(RecorderSummary {
        frames,
        events,
        captured_frames,
        dropped_frames,
        slow_frames,
    })
}

pub fn build_ffmpeg_args(
    encoder: &Encoder,
    max_resolution: Option<(u32, u32)>,
    width: u32,
    height: u32,
    fps: u64,
    crf: u8,
    pixel_format: &str,
    preset: &str,
    output_path: &Path,
    vaapi_device: Option<&str>,
) -> Vec<String> {
    let video_size = format!("{}x{}", width, height);
    let fps_str = fps.to_string();
    let qp_str = crf.to_string();

    let filter = match (encoder, max_resolution) {
        (Encoder::Vaapi, Some((max_w, max_h))) => format!(
            "format=nv12,hwupload,scale_vaapi=w='min({},iw)':h='min({},ih)':force_original_aspect_ratio=decrease",
            max_w, max_h
        ),
        (Encoder::Vaapi, None) => "format=nv12,hwupload".to_string(),
        (_, Some((max_w, max_h))) => format!(
            "scale='min({},iw)':'min({},ih)':force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2",
            max_w, max_h
        ),
        _ => String::new(),
    };

    let mut args: Vec<String> = Vec::new();
    if let Some(device) = vaapi_device {
        args.extend(["-vaapi_device".to_string(), device.to_string()]);
    }

    args.extend([
        "-f".to_string(),
        "rawvideo".to_string(),
        "-pixel_format".to_string(),
        pixel_format.to_string(),
        "-video_size".to_string(),
        video_size,
        "-framerate".to_string(),
        fps_str,
        "-i".to_string(),
        "-".to_string(),
    ]);

    if !filter.is_empty() {
        args.extend(["-vf".to_string(), filter]);
    }

    match encoder {
        Encoder::Vaapi => {
            args.extend([
                "-c:v".to_string(),
                "h264_vaapi".to_string(),
                "-qp".to_string(),
                qp_str,
            ]);
        }
        Encoder::X264 | Encoder::Auto => {
            args.extend([
                "-c:v".to_string(),
                "libx264".to_string(),
                "-preset".to_string(),
                preset.to_string(),
                "-crf".to_string(),
                qp_str,
                "-pix_fmt".to_string(),
                "yuv444p".to_string(),
            ]);
        }
    }

    args.extend(["-y".to_string(), output_path.display().to_string()]);
    args
}
