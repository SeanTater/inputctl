pub mod portal;

use crate::Result;
use std::time::Duration;

pub use portal::{CaptureFrame, PortalCapture};

pub trait FrameSource {
    fn next_frame(&mut self, timeout: Duration) -> Result<CaptureFrame>;
}
