//! Test raw PipeWire capture - bypass GStreamer entirely
//!
//! Run with: cargo run --release -p inputctl-capture --bin pipewire_test

use ashpd::desktop::screencast::{CursorMode, Screencast, SourceType, Stream};
use ashpd::desktop::PersistMode;
use pipewire as pw;
use pw::spa;
use spa::param::format::{MediaSubtype, MediaType};
use spa::param::format_utils;
use spa::param::video::VideoInfoRaw;
use spa::pod::Pod;
use std::cell::Cell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

struct UserData {
    format: VideoInfoRaw,
}

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

    println!("Portal opened! Node: {}", pipewire_node_id);
    println!("\nStarting raw PipeWire capture...");
    println!("Play your game and check for lag. Press Ctrl+C to stop.\n");

    let running = Arc::new(AtomicBool::new(true));
    let running_ctrlc = Arc::clone(&running);

    ctrlc::set_handler(move || {
        running_ctrlc.store(false, Ordering::SeqCst);
    })?;

    // Initialize PipeWire
    pw::init();

    let mainloop = pw::main_loop::MainLoop::new(None)?;
    let context = pw::context::Context::new(&mainloop)?;
    let core = context.connect_fd(fd, None)?;

    let _listener = core
        .add_listener_local()
        .error(|id, seq, res, msg| {
            eprintln!(
                "PipeWire error: id={} seq={} res={} msg={}",
                id, seq, res, msg
            );
        })
        .done(|_id, _seq| {})
        .register();

    // Create stream with target node in properties
    let video_stream = pw::stream::Stream::new(
        &core,
        "video-capture",
        pw::properties::properties! {
            *pw::keys::MEDIA_TYPE => "Video",
            *pw::keys::MEDIA_CATEGORY => "Capture",
            *pw::keys::MEDIA_ROLE => "Screen",
        },
    )?;

    // Frame counter and timing
    let frame_count = Rc::new(Cell::new(0u64));
    let start_time = Rc::new(Cell::new(Instant::now()));
    let last_print = Rc::new(Cell::new(Instant::now()));

    let frame_count_cb = Rc::clone(&frame_count);
    let last_print_cb = Rc::clone(&last_print);
    let frame_count_print = Rc::clone(&frame_count);
    let start_time_print = Rc::clone(&start_time);

    let data = UserData {
        format: VideoInfoRaw::new(),
    };

    let _stream_listener = video_stream
        .add_local_listener_with_user_data(data)
        .state_changed(|_stream, _user_data, _old, new| {
            println!("Stream state: {:?}", new);
        })
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

            user_data
                .format
                .parse(param)
                .expect("Failed to parse video format");

            let size = user_data.format.size();
            println!(
                "Format negotiated: {:?} {}x{}",
                user_data.format.format(),
                size.width,
                size.height
            );
        })
        .process(move |stream, _user_data| {
            frame_count_cb.set(frame_count_cb.get() + 1);

            // Dequeue buffer (minimal work)
            if let Some(_buffer) = stream.dequeue_buffer() {
                // Just drop it
            }

            // Print stats every second
            if last_print_cb.get().elapsed().as_secs() >= 1 {
                let frames = frame_count_print.get();
                let elapsed = start_time_print.get().elapsed().as_secs_f64();
                if elapsed > 0.0 {
                    println!("Frames: {}, FPS: {:.1}", frames, frames as f64 / elapsed);
                }
                last_print_cb.set(Instant::now());
            }
        })
        .register()?;

    // Connect as input to the specific portal node
    // Offer raw video formats to ensure negotiation completes.
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
            spa::param::video::VideoFormat::RGB,
            spa::param::video::VideoFormat::BGR,
            spa::param::video::VideoFormat::I420,
            spa::param::video::VideoFormat::YUY2,
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
    .expect("serialize format pod")
    .0
    .into_inner();
    let mut params = [Pod::from_bytes(&values).expect("format pod")];

    video_stream.connect(
        spa::utils::Direction::Input,
        Some(pipewire_node_id),
        pw::stream::StreamFlags::AUTOCONNECT
            | pw::stream::StreamFlags::MAP_BUFFERS
            | pw::stream::StreamFlags::RT_PROCESS,
        &mut params,
    )?;

    // Some portals start in Paused; explicitly activate after connecting.
    video_stream.set_active(true)?;

    println!("Stream connected, waiting for frames...");

    // Keep stream alive
    let _stream = stream;

    // Run mainloop until Ctrl+C
    start_time.set(Instant::now());

    while running.load(Ordering::SeqCst) {
        mainloop
            .loop_()
            .iterate(std::time::Duration::from_millis(100));
    }

    let total = frame_count.get();
    let elapsed = start_time.get().elapsed().as_secs_f64();
    println!(
        "\nTotal: {} frames in {:.1}s ({:.1} FPS)",
        total,
        elapsed,
        total as f64 / elapsed
    );

    Ok(())
}
