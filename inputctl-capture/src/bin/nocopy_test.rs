//! Test capture with ZERO copying - just count frames in GStreamer callback
//!
//! Run with: cargo run --release -p inputctl-capture --bin nocopy_test

use ashpd::desktop::screencast::{CursorMode, Screencast, SourceType, Stream};
use ashpd::desktop::PersistMode;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

async fn open_portal(
) -> Result<(Stream, std::os::fd::OwnedFd), Box<dyn std::error::Error + Send + Sync>> {
    let proxy = Screencast::new().await?;
    let session = proxy.create_session().await?;

    proxy
        .select_sources(
            &session,
            CursorMode::Embedded,
            SourceType::Monitor | SourceType::Window,
            true,
            None,
            PersistMode::DoNot,
        )
        .await?
        .response()?;

    let response = proxy.start(&session, None).await?.response()?;
    let stream = response.streams().first().ok_or("No stream")?.to_owned();
    let fd = proxy.open_pipe_wire_remote(&session).await?;

    Ok((stream, fd))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    gst::init()?;

    println!("Opening portal...");

    let (stream, fd) = std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        runtime.block_on(open_portal()).expect("portal failed")
    })
    .join()
    .expect("thread panicked");

    let pipewire_node_id = stream.pipe_wire_node_id();
    let stream_fd = fd.as_raw_fd();

    println!(
        "Portal opened! Node: {}, FD: {}",
        pipewire_node_id, stream_fd
    );

    // Build minimal pipeline
    let pipewiresrc = gst::ElementFactory::make("pipewiresrc")
        .property("fd", stream_fd)
        .property("path", pipewire_node_id.to_string())
        .property("do-timestamp", false)
        .build()?;

    let queue = gst::ElementFactory::make("queue")
        .property("max-size-buffers", 10u32)
        .build()?;

    let appsink = gst::ElementFactory::make("appsink")
        .build()?
        .downcast::<gst_app::AppSink>()
        .expect("not appsink");

    appsink.set_property("emit-signals", true);
    appsink.set_property("sync", false);
    appsink.set_property("max-buffers", 1u32);
    appsink.set_property("drop", true);

    // Counter - NO COPY, just count frames
    let frame_count = Arc::new(AtomicU64::new(0));
    let frame_count_cb = frame_count.clone();

    appsink.set_callbacks(
        gst_app::AppSinkCallbacks::builder()
            .new_sample(move |sink| {
                // Pull sample but DON'T copy the data
                let _sample = sink.pull_sample().map_err(|_| gst::FlowError::Error)?;
                // Just count - no map_readable, no to_vec
                frame_count_cb.fetch_add(1, Ordering::Relaxed);
                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );

    let pipeline = gst::Pipeline::default();
    pipeline.add_many([&pipewiresrc, &queue, appsink.upcast_ref()])?;
    gst::Element::link_many([&pipewiresrc, &queue, appsink.upcast_ref()])?;

    // Keep things alive
    let _stream = stream;
    let _fd = fd;

    println!("\nStarting NO-COPY capture test...");
    println!("Play your game and check for lag. Press Ctrl+C to stop.\n");

    pipeline
        .set_state(gst::State::Playing)
        .map_err(|e| format!("Failed: {:?}", e))?;

    let running = Arc::new(AtomicBool::new(true));
    let running_ctrlc = running.clone();
    ctrlc::set_handler(move || {
        running_ctrlc.store(false, Ordering::SeqCst);
    })?;

    let start = std::time::Instant::now();
    while running.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_secs(1));
        let frames = frame_count.load(Ordering::Relaxed);
        let elapsed = start.elapsed().as_secs_f64();
        println!("Frames: {}, FPS: {:.1}", frames, frames as f64 / elapsed);
    }

    pipeline.set_state(gst::State::Null)?;

    let total = frame_count.load(Ordering::Relaxed);
    println!(
        "\nTotal: {} frames in {:.1}s",
        total,
        start.elapsed().as_secs_f64()
    );

    Ok(())
}
