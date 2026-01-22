use inputctl_capture::recorder::Clock;
use inputctl_capture::{
    run_recorder_with_sources, CaptureFrame, Encoder, FrameSource, InputEvent, InputEventSource,
    RecorderConfig, VideoWriter,
};
use std::cell::Cell;
use std::time::{Duration, Instant};

struct TestClock {
    start: Instant,
    now: Cell<Instant>,
}

impl TestClock {
    fn new() -> Self {
        let start = Instant::now();
        Self {
            start,
            now: Cell::new(start),
        }
    }
}

impl Clock for TestClock {
    fn now(&self) -> Instant {
        self.now.get()
    }

    fn now_ms(&self) -> u128 {
        self.now.get().duration_since(self.start).as_millis()
    }

    fn sleep(&self, duration: Duration) {
        let next = self.now.get() + duration;
        self.now.set(next);
    }
}

struct TestFrameSource {
    frames: Vec<CaptureFrame>,
}

impl TestFrameSource {
    fn new(frames: Vec<CaptureFrame>) -> Self {
        Self { frames }
    }
}

impl FrameSource for TestFrameSource {
    fn next_frame(&mut self, _timeout: Duration) -> inputctl_capture::Result<CaptureFrame> {
        if self.frames.is_empty() {
            return Err(inputctl_capture::Error::ScreenshotFailed(
                "no frames".to_string(),
            ));
        }
        Ok(self.frames.remove(0))
    }
}

struct TestWriter {
    writes: usize,
}

impl TestWriter {
    fn new() -> Self {
        Self { writes: 0 }
    }
}

impl VideoWriter for TestWriter {
    fn write_frame(&mut self, _frame: &[u8]) -> std::io::Result<()> {
        self.writes += 1;
        Ok(())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }

    fn finish(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

struct TestInputSource {
    events: Vec<InputEvent>,
}

impl TestInputSource {
    fn new(events: Vec<InputEvent>) -> Self {
        Self { events }
    }
}

impl InputEventSource for TestInputSource {
    fn poll_events(&mut self) -> std::io::Result<Vec<InputEvent>> {
        Ok(self.events.clone())
    }
}

#[test]
fn recorder_pipeline_collects_rows() {
    let config = RecorderConfig {
        output_dir: std::path::PathBuf::from("/tmp"),
        fps: 10,
        preset: "veryfast".to_string(),
        crf: 28,
        device_path: None,
        max_seconds: Some(1),
        stats_interval: None,
        max_resolution: None,
        encoder: Encoder::X264,
    };

    let frame = CaptureFrame {
        rgba: vec![1u8; 16],
        width: 1,
        height: 1,
        timestamp_ms: 0,
        format: "BGRx".to_string(),
    };

    let mut frame_source = TestFrameSource::new(vec![
        CaptureFrame {
            rgba: frame.rgba.clone(),
            width: frame.width,
            height: frame.height,
            timestamp_ms: frame.timestamp_ms,
            format: frame.format.clone(),
        },
        frame,
    ]);
    let mut writer = TestWriter::new();
    let mut input_source = TestInputSource::new(vec![InputEvent {
        timestamp: 123,
        key_code: 30,
        key_name: "KEY_A".to_string(),
        state: "down",
    }]);
    let mut clock = TestClock::new();

    let summary = run_recorder_with_sources(
        &config,
        &mut frame_source,
        &mut writer,
        &mut input_source,
        &mut clock,
        1,
        1,
        "BGRx".to_string(),
        None,
        None,
    )
    .expect("summary");

    assert!(!summary.frames.is_empty());
    assert!(!summary.events.is_empty());
    assert_eq!(summary.frames[0].frame_idx, 0);
    assert_eq!(summary.events[0].key_code, 30);
}
