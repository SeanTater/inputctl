use crate::error::{Error, Result};
use crate::primitives::screen::Region;
use ashpd::desktop::screencast::{CursorMode, Screencast, SourceType, Stream};
use ashpd::desktop::PersistMode;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use std::os::fd::AsRawFd;
use std::time::{Duration, Instant};

pub struct CaptureFrame {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub timestamp_ms: u128,
}

pub struct PortalCapture {
    _pipeline: gst::Pipeline,
    appsink: gst_app::AppSink,
    width: u32,
    height: u32,
    start_time: Instant,
    _stream: Stream,
}

impl PortalCapture {
    pub fn connect(window_hint: Option<&str>) -> Result<Self> {
        gst::init()
            .map_err(|e| Error::ScreenshotFailed(format!("GStreamer init failed: {}", e)))?;

        let window_hint = window_hint.map(str::to_string);
        let (stream, fd) = std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| Error::ScreenshotFailed(format!("Portal runtime failed: {}", e)))?;
            runtime.block_on(open_portal(window_hint.as_deref()))
        })
        .join()
        .map_err(|_| Error::ScreenshotFailed("Portal thread panicked".into()))??;

        let pipewire_node_id = stream.pipe_wire_node_id();
        let stream_fd = fd.as_raw_fd();

        let pipewire_element = gst::ElementFactory::make("pipewiresrc")
            .property("fd", stream_fd)
            .property("path", pipewire_node_id.to_string())
            .property("do-timestamp", false)
            .build()
            .map_err(|_| Error::ScreenshotFailed("Failed to create pipewiresrc".into()))?;

        let convert = gst::ElementFactory::make("videoconvert")
            .build()
            .map_err(|_| Error::ScreenshotFailed("Failed to create videoconvert".into()))?;

        let caps = gst::Caps::builder("video/x-raw")
            .field("format", "RGBA")
            .build();

        let caps_ref = &caps;

        let caps_filter = gst::ElementFactory::make("capsfilter")
            .property("caps", caps_ref)
            .build()
            .map_err(|_| Error::ScreenshotFailed("Failed to create capsfilter".into()))?;

        let appsink = gst::ElementFactory::make("appsink")
            .build()
            .map_err(|_| Error::ScreenshotFailed("Failed to create appsink".into()))?
            .downcast::<gst_app::AppSink>()
            .map_err(|_| Error::ScreenshotFailed("appsink downcast failed".into()))?;

        appsink.set_property("emit-signals", false);
        appsink.set_property("sync", false);
        appsink.set_property("max-buffers", 1u32);
        appsink.set_property("drop", true);
        appsink.set_caps(Some(caps_ref));

        let pipeline = gst::Pipeline::default();
        let appsink_element = appsink.upcast_ref::<gst::Element>();
        pipeline
            .add_many([&pipewire_element, &convert, &caps_filter, appsink_element])
            .map_err(|_| Error::ScreenshotFailed("Failed to build pipeline".into()))?;

        gst::Element::link_many([&pipewire_element, &convert, &caps_filter, appsink_element])
            .map_err(|_| Error::ScreenshotFailed("Failed to link pipeline".into()))?;

        pipeline
            .set_state(gst::State::Playing)
            .map_err(|_| Error::ScreenshotFailed("Failed to start pipeline".into()))?;

        let (width, height) = stream
            .size()
            .map(|s| (s.0 as u32, s.1 as u32))
            .unwrap_or((0, 0));

        Ok(Self {
            _pipeline: pipeline,
            appsink,
            width,
            height,
            start_time: Instant::now(),
            _stream: stream,
        })
    }

    pub fn stream_region(&self) -> Option<Region> {
        let (x, y) = self._stream.position()?;
        let (width, height) = self._stream.size()?;
        if width <= 0 || height <= 0 {
            return None;
        }
        Some(Region {
            x,
            y,
            width: width as u32,
            height: height as u32,
        })
    }

    pub fn stream_size(&self) -> Option<(u32, u32)> {
        let (width, height) = self._stream.size()?;
        if width <= 0 || height <= 0 {
            return None;
        }
        Some((width as u32, height as u32))
    }

    pub fn stream_debug_line(&self) -> String {
        format!(
            "Portal stream: id={:?} source={:?} position={:?} size={:?}",
            self._stream.id(),
            self._stream.source_type(),
            self._stream.position(),
            self._stream.size()
        )
    }

    pub fn next_frame(&self, timeout: Duration) -> Result<CaptureFrame> {
        let timeout_clock = gst::ClockTime::from_nseconds(timeout.as_nanos() as u64);
        let sample = self
            .appsink
            .try_pull_sample(Some(timeout_clock))
            .ok_or_else(|| Error::ScreenshotFailed("Timed out waiting for frame".into()))?;

        let buffer = sample
            .buffer()
            .ok_or_else(|| Error::ScreenshotFailed("Missing buffer".into()))?;

        let map = buffer
            .map_readable()
            .map_err(|_| Error::ScreenshotFailed("Failed to map buffer".into()))?;

        let info = sample
            .caps()
            .and_then(|caps| gst_video_info(caps))
            .unwrap_or((self.width, self.height));

        let timestamp_ms = self.start_time.elapsed().as_millis();

        Ok(CaptureFrame {
            rgba: map.as_slice().to_vec(),
            width: info.0,
            height: info.1,
            timestamp_ms,
        })
    }
}

fn gst_video_info(caps: &gst::CapsRef) -> Option<(u32, u32)> {
    let s = caps.structure(0)?;
    let width = s.get::<i32>("width").ok()? as u32;
    let height = s.get::<i32>("height").ok()? as u32;
    Some((width, height))
}

async fn open_portal(window_hint: Option<&str>) -> Result<(Stream, std::os::fd::OwnedFd)> {
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
