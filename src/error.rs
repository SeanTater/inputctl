use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("failed to create uinput device: {0}")]
    DeviceCreation(#[from] std::io::Error),

    #[error("character '{0}' cannot be typed (no keycode mapping)")]
    UntypableCharacter(char),

    #[error("failed to emit input event")]
    EmitFailed,
}

pub type Result<T> = std::result::Result<T, Error>;
