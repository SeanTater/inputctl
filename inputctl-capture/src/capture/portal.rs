use crate::error::{Error, Result};
use crate::primitives::frame_ops::{copy_frame, copy_frame_region, non_black_bounds_stride};
use crate::primitives::screen::Region;
use ashpd::desktop::screencast::{CursorMode, Screencast, SourceType, Stream};
use ashpd::desktop::PersistMode;
use pipewire as pw;
use pw::spa;
use spa::param::format::{MediaSubtype, MediaType};
use spa::param::format_utils;
use spa::param::video::{VideoFormat, VideoInfoRaw};
use spa::pod::Pod;
use std::os::fd::OwnedFd;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use tokio::runtime;

#[derive(Clone)]
pub struct CaptureFrame {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub timestamp_ms: u128,
    /// Pixel format as PipeWire format string (e.g., "BGRx", "BGRA", "RGBA")
    pub format: String,
}

/// Raw sample from PipeWire before processing
struct RawSample {
    data: Vec<u8>,
    width: u32,
    height: u32,
    format: String,
    timestamp_ms: u128,
}

struct UserData {
    format: VideoInfoRaw,
    format_label: Option<&'static str>,
    bytes_per_pixel: usize,
    warned_unsupported: bool,
    warned_no_data: bool,
    warned_stride: bool,
    auto_crop_region: Option<Region>,
}

struct CaptureState {
    mainloop: pw::main_loop::MainLoop,
    stream: pw::stream::Stream,
    _stream_listener: pw::stream::StreamListener<UserData>,
    _core_listener: pw::core::Listener,
    _portal_stream: Stream,
    running: Arc<AtomicBool>,
    _format_pod: Vec<u8>,
}

pub struct PortalCapture {
    frame_rx: Receiver<RawSample>,
    running: Arc<AtomicBool>,
    thread_handle: Option<thread::JoinHandle<()>>,
}

impl PortalCapture {
    pub fn connect(window_hint: Option<&str>) -> Result<Self> {
        let window_hint = window_hint.map(str::to_string);
        let (frame_tx, frame_rx): (SyncSender<RawSample>, Receiver<RawSample>) =
            mpsc::sync_channel(5);
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let running = Arc::new(AtomicBool::new(true));
        let running_thread = Arc::clone(&running);

        let thread_handle = thread::spawn(move || {
            let init = init_pipewire_capture(window_hint.as_deref(), frame_tx, running_thread);
            match init {
                Ok(state) => {
                    let _ = ready_tx.send(Ok(()));
                    run_pipewire_loop(state);
                }
                Err(err) => {
                    let _ = ready_tx.send(Err(err));
                }
            }
        });

        match ready_rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(err)) => return Err(err),
            Err(_) => {
                return Err(Error::ScreenshotFailed(
                    "PipeWire capture thread exited".to_string(),
                ));
            }
        }

        Ok(Self {
            frame_rx,
            running,
            thread_handle: Some(thread_handle),
        })
    }

    pub fn next_frame(&self, timeout: Duration) -> Result<CaptureFrame> {
        let raw = self
            .frame_rx
            .recv_timeout(timeout)
            .map_err(|_| Error::CaptureTimeout)?;

        let frame_width = raw.width;
        let frame_height = raw.height;
        let format = raw.format;

        let timestamp_ms = raw.timestamp_ms;

        Ok(CaptureFrame {
            rgba: raw.data,
            width: frame_width,
            height: frame_height,
            timestamp_ms,
            format,
        })
    }
}

impl Drop for PortalCapture {
    fn drop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }
}

impl crate::capture::FrameSource for PortalCapture {
    fn next_frame(&mut self, timeout: Duration) -> Result<CaptureFrame> {
        PortalCapture::next_frame(self, timeout)
    }
}

fn init_pipewire_capture(
    window_hint: Option<&str>,
    frame_tx: SyncSender<RawSample>,
    running: Arc<AtomicBool>,
) -> Result<CaptureState> {
    let (portal_stream, fd) = {
        let runtime = runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| Error::ScreenshotFailed(format!("Portal runtime failed: {}", e)))?;
        runtime.block_on(open_portal(window_hint))?
    };
    let pipewire_node_id = portal_stream.pipe_wire_node_id();

    pw::init();

    let mainloop = pw::main_loop::MainLoop::new(None).map_err(map_pipewire_error)?;
    let context = pw::context::Context::new(&mainloop).map_err(map_pipewire_error)?;
    let core = context.connect_fd(fd, None).map_err(map_pipewire_error)?;

    let _core_listener = core
        .add_listener_local()
        .error(|id, seq, res, msg| {
            eprintln!(
                "PipeWire error: id={} seq={} res={} msg={}",
                id, seq, res, msg
            );
        })
        .register();

    let video_stream = pw::stream::Stream::new(
        &core,
        "video-capture",
        pw::properties::properties! {
            *pw::keys::MEDIA_TYPE => "Video",
            *pw::keys::MEDIA_CATEGORY => "Capture",
            *pw::keys::MEDIA_ROLE => "Screen",
            *pw::keys::NODE_ALWAYS_PROCESS => "true",
            *pw::keys::NODE_PAUSE_ON_IDLE => "false",
        },
    )
    .map_err(map_pipewire_error)?;

    let start_time = Instant::now();
    let frame_tx_cb = frame_tx.clone();
    let running_cb = Arc::clone(&running);

    let data = UserData {
        format: VideoInfoRaw::new(),
        format_label: None,
        bytes_per_pixel: 0,
        warned_unsupported: false,
        warned_no_data: false,
        warned_stride: false,
        auto_crop_region: None,
    };

    let _stream_listener = video_stream
        .add_local_listener_with_user_data(data)
        .param_changed(|_stream, user_data, id, param| {
            let Some(param) = param else {
                return;
            };
            if id != spa::param::ParamType::Format.as_raw() {
                return;
            }

            let (media_type, media_subtype) = match format_utils::parse_format(param) {
                Ok(v) => v,
                Err(_) => return,
            };

            if media_type != MediaType::Video || media_subtype != MediaSubtype::Raw {
                return;
            }

            if user_data.format.parse(param).is_err() {
                return;
            }

            let format = user_data.format.format();
            user_data.format_label = format_label(format);
            user_data.bytes_per_pixel = bytes_per_pixel(format).unwrap_or(0);
        })
        .process(move |stream, user_data| {
            if !running_cb.load(Ordering::Relaxed) {
                return;
            }

            let Some(mut buffer) = stream.dequeue_buffer() else {
                return;
            };

            let Some(sample) = build_sample(&mut buffer, user_data, start_time) else {
                return;
            };

            let _ = frame_tx_cb.try_send(sample);
        })
        .register()
        .map_err(map_pipewire_error)?;

    let format_pod = build_format_params()?;
    let params = Pod::from_bytes(&format_pod)
        .ok_or_else(|| Error::ScreenshotFailed("Failed to parse format pod".to_string()))?;
    let mut params_ref = [&*params];

    video_stream
        .connect(
            spa::utils::Direction::Input,
            Some(pipewire_node_id),
            pw::stream::StreamFlags::AUTOCONNECT
                | pw::stream::StreamFlags::MAP_BUFFERS
                | pw::stream::StreamFlags::RT_PROCESS,
            &mut params_ref,
        )
        .map_err(map_pipewire_error)?;

    video_stream.set_active(true).map_err(map_pipewire_error)?;

    Ok(CaptureState {
        mainloop,
        stream: video_stream,
        _stream_listener,
        _core_listener,
        _portal_stream: portal_stream,
        running,
        _format_pod: format_pod,
    })
}

fn run_pipewire_loop(state: CaptureState) {
    while state.running.load(Ordering::SeqCst) {
        state
            .mainloop
            .loop_()
            .iterate(std::time::Duration::from_millis(100));
    }

    drop(state.stream);
}

fn build_sample(
    buffer: &mut pw::buffer::Buffer,
    user_data: &mut UserData,
    start_time: Instant,
) -> Option<RawSample> {
    if user_data.bytes_per_pixel == 0 {
        if !user_data.warned_unsupported {
            eprintln!(
                "Unsupported PipeWire format: {:?}",
                user_data.format.format()
            );
            user_data.warned_unsupported = true;
        }
        return None;
    }

    let size = user_data.format.size();
    let width = size.width;
    let height = size.height;
    if width == 0 || height == 0 {
        return None;
    }

    let format_label = user_data.format_label.unwrap_or("UNKNOWN");

    let datas = buffer.datas_mut();
    if datas.is_empty() {
        return None;
    }

    let data = &mut datas[0];
    let chunk = data.chunk();
    let stride = chunk.stride();
    let offset = chunk.offset();
    let chunk_size = chunk.size();

    let Some(raw) = data.data() else {
        if !user_data.warned_no_data {
            eprintln!("PipeWire buffer not CPU-accessible (dmabuf?)");
            user_data.warned_no_data = true;
        }
        return None;
    };

    if user_data.auto_crop_region.is_none() {
        user_data.auto_crop_region = non_black_bounds_stride(
            raw,
            width,
            height,
            offset,
            stride,
            chunk_size,
            user_data.bytes_per_pixel,
        );
    }

    let region = user_data.auto_crop_region;
    let (frame, out_width, out_height) = if let Some(region) = region {
        let frame = copy_frame_region(
            raw,
            width,
            height,
            offset,
            stride,
            chunk_size,
            user_data.bytes_per_pixel,
            Some(region),
            &mut user_data.warned_stride,
        )?;
        (frame, region.width, region.height)
    } else {
        let frame = copy_frame(
            raw,
            width,
            height,
            offset,
            stride,
            chunk_size,
            user_data.bytes_per_pixel,
            &mut user_data.warned_stride,
        )?;
        (frame, width, height)
    };

    Some(RawSample {
        data: frame,
        width: out_width,
        height: out_height,
        format: format_label.to_string(),
        timestamp_ms: start_time.elapsed().as_millis(),
    })
}

fn build_format_params() -> Result<Vec<u8>> {
    let obj = spa::pod::object!(
        spa::utils::SpaTypes::ObjectParamFormat,
        spa::param::ParamType::EnumFormat,
        spa::pod::property!(
            spa::param::format::FormatProperties::MediaType,
            Id,
            spa::param::format::MediaType::Video
        ),
        spa::pod::property!(
            spa::param::format::FormatProperties::MediaSubtype,
            Id,
            spa::param::format::MediaSubtype::Raw
        ),
        spa::pod::property!(
            spa::param::format::FormatProperties::VideoFormat,
            Choice,
            Enum,
            Id,
            spa::param::video::VideoFormat::RGBx,
            spa::param::video::VideoFormat::BGRx,
            spa::param::video::VideoFormat::RGBA,
            spa::param::video::VideoFormat::BGRA,
            spa::param::video::VideoFormat::xRGB,
            spa::param::video::VideoFormat::xBGR,
            spa::param::video::VideoFormat::ARGB,
            spa::param::video::VideoFormat::ABGR,
        ),
        spa::pod::property!(
            spa::param::format::FormatProperties::VideoSize,
            Choice,
            Range,
            Rectangle,
            spa::utils::Rectangle {
                width: 320,
                height: 240,
            },
            spa::utils::Rectangle {
                width: 1,
                height: 1,
            },
            spa::utils::Rectangle {
                width: 8192,
                height: 8192,
            }
        ),
        spa::pod::property!(
            spa::param::format::FormatProperties::VideoFramerate,
            Choice,
            Range,
            Fraction,
            spa::utils::Fraction { num: 60, denom: 1 },
            spa::utils::Fraction { num: 0, denom: 1 },
            spa::utils::Fraction { num: 240, denom: 1 }
        ),
    );
    let values: Vec<u8> = spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &spa::pod::Value::Object(obj),
    )
    .map_err(|e| Error::ScreenshotFailed(format!("Failed to serialize format pod: {}", e)))?
    .0
    .into_inner();

    Ok(values)
}

fn format_label(format: VideoFormat) -> Option<&'static str> {
    match format {
        VideoFormat::BGRx => Some("BGRx"),
        VideoFormat::BGRA => Some("BGRA"),
        VideoFormat::RGBx => Some("RGBx"),
        VideoFormat::RGBA => Some("RGBA"),
        VideoFormat::xRGB => Some("xRGB"),
        VideoFormat::ARGB => Some("ARGB"),
        VideoFormat::xBGR => Some("xBGR"),
        VideoFormat::ABGR => Some("ABGR"),
        _ => None,
    }
}

fn bytes_per_pixel(format: VideoFormat) -> Option<usize> {
    match format {
        VideoFormat::BGRx
        | VideoFormat::BGRA
        | VideoFormat::RGBx
        | VideoFormat::RGBA
        | VideoFormat::xRGB
        | VideoFormat::ARGB
        | VideoFormat::xBGR
        | VideoFormat::ABGR => Some(4),
        _ => None,
    }
}

async fn open_portal(window_hint: Option<&str>) -> Result<(Stream, OwnedFd)> {
    let proxy = Screencast::new().await.map_err(map_portal_error)?;
    let session = proxy.create_session().await.map_err(map_portal_error)?;
    let source_type = SourceType::Monitor | SourceType::Window;
    let _window_hint = window_hint;

    proxy
        .select_sources(
            &session,
            CursorMode::Embedded,
            source_type,
            true,
            None,
            PersistMode::DoNot,
        )
        .await
        .map_err(map_portal_error)?
        .response()
        .map_err(map_portal_error)?;

    let response = proxy
        .start(&session, None)
        .await
        .map_err(map_portal_error)?
        .response()
        .map_err(map_portal_error)?;
    let stream = response
        .streams()
        .first()
        .ok_or_else(|| Error::ScreenshotFailed("No stream selected".into()))?
        .to_owned();

    let fd = proxy
        .open_pipe_wire_remote(&session)
        .await
        .map_err(map_portal_error)?;

    Ok((stream, fd))
}

fn map_portal_error(err: ashpd::Error) -> Error {
    Error::ScreenshotFailed(format!("Portal error: {}", err))
}

fn map_pipewire_error(err: pw::Error) -> Error {
    Error::ScreenshotFailed(format!("PipeWire error: {}", err))
}
