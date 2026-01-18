//! Error types and Result alias for inputctl-vision.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Screenshot failed: {0}")]
    ScreenshotFailed(String),

    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("LLM API error: {0}")]
    LlmApiError(String),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Capture error: {0}")]
    Capture(#[from] inputctl_capture::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
