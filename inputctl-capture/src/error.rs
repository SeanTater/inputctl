//! Error types for inputctl-capture.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Screenshot failed: {0}")]
    ScreenshotFailed(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
