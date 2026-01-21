use crate::error::{Error, Result};
use crate::primitives::screen::Region;
use ashpd::desktop::screencast::{CursorMode, Screencast, SourceType, Stream};
use ashpd::desktop::PersistMode;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use std::os::fd::AsRawFd;
use std::sync::Mutex;
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
    crop_region: Option<Region>,
    alpha_region: Mutex<Option<Region>>,
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

        let crop_region = stream_region(&stream);

        Ok(Self {
            _pipeline: pipeline,
            appsink,
            width,
            height,
            start_time: Instant::now(),
            _stream: stream,
            crop_region,
            alpha_region: Mutex::new(None),
        })
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
        let alpha_region = if self.crop_region.is_some() {
            None
        } else {
            let mut cached = self
                .alpha_region
                .lock()
                .map_err(|_| Error::ScreenshotFailed("Alpha crop lock poisoned".into()))?;
            if cached.is_none() {
                *cached = alpha_bounds(map.as_slice(), info.0, info.1);
            }
            *cached
        };

        let crop_region = self.crop_region.or(alpha_region);
        let rgba = crop_rgba(map.as_slice(), info.0, info.1, crop_region)?;

        let (width, height) = if let Some(region) = crop_region {
            if region.x >= 0
                && region.y >= 0
                && region.x as u32 + region.width <= info.0
                && region.y as u32 + region.height <= info.1
            {
                (region.width, region.height)
            } else {
                (info.0, info.1)
            }
        } else {
            (info.0, info.1)
        };

        Ok(CaptureFrame {
            rgba,
            width,
            height,
            timestamp_ms,
        })
    }
}

fn stream_region(stream: &Stream) -> Option<Region> {
    if stream.source_type() != Some(SourceType::Window) {
        return None;
    }

    let (stream_width, stream_height) = stream.size()?;
    let (x, y) = stream.position().unwrap_or((0, 0));

    if stream_width <= 0 || stream_height <= 0 {
        return None;
    }

    Some(Region {
        x,
        y,
        width: stream_width as u32,
        height: stream_height as u32,
    })
}

fn crop_rgba(rgba: &[u8], width: u32, height: u32, region: Option<Region>) -> Result<Vec<u8>> {
    let region = match region {
        Some(region) => region,
        None => return Ok(rgba.to_vec()),
    };

    if region.x < 0
        || region.y < 0
        || region.x as u32 + region.width > width
        || region.y as u32 + region.height > height
    {
        return Ok(rgba.to_vec());
    }

    let crop_x = region.x as u32;
    let crop_y = region.y as u32;
    let crop_w = region.width;
    let crop_h = region.height;

    let mut out = vec![0u8; (crop_w * crop_h * 4) as usize];
    let src_stride = (width * 4) as usize;
    let dst_stride = (crop_w * 4) as usize;

    for row in 0..crop_h {
        let src_start = ((crop_y + row) as usize * src_stride) + (crop_x * 4) as usize;
        let src_end = src_start + dst_stride;
        let dst_start = (row as usize) * dst_stride;
        out[dst_start..dst_start + dst_stride].copy_from_slice(&rgba[src_start..src_end]);
    }

    Ok(out)
}

fn alpha_bounds(rgba: &[u8], width: u32, height: u32) -> Option<Region> {
    if width == 0 || height == 0 {
        return None;
    }

    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut found = false;

    let row_stride = (width * 4) as usize;
    for y in 0..height {
        let row_start = (y as usize) * row_stride;
        for x in 0..width {
            let idx = row_start + (x as usize * 4) + 3;
            if rgba.get(idx).copied().unwrap_or(0) != 0 {
                found = true;
                if x < min_x {
                    min_x = x;
                }
                if y < min_y {
                    min_y = y;
                }
                if x > max_x {
                    max_x = x;
                }
                if y > max_y {
                    max_y = y;
                }
            }
        }
    }

    if !found {
        return None;
    }

    Some(Region {
        x: min_x as i32,
        y: min_y as i32,
        width: max_x - min_x + 1,
        height: max_y - min_y + 1,
    })
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
