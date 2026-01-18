//! Persistent configuration management for visionctl.
//!
//! This module handles loading and saving settings from `~/.config/visionctl/config.toml`,
//! including LLM backend details and UI preferences.

use crate::error::Result;
use crate::llm::LlmConfig;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use tracing::warn;

/// Main configuration for visionctl
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    #[serde(default)]
    pub llm: LlmSettings,
    #[serde(default)]
    pub cursor: CursorSettings,
    #[serde(default)]
    pub pointing: Option<LlmSettings>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            llm: LlmSettings::default(),
            cursor: CursorSettings::default(),
            pointing: None,
        }
    }
}

/// LLM backend configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LlmSettings {
    #[serde(default = "default_backend")]
    pub backend: String,
    #[serde(default = "default_base_url")]
    pub base_url: String,
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
}

fn default_temperature() -> f32 {
    0.0 // Deterministic by default for GUI automation
}

fn default_backend() -> String {
    "ollama".to_string()
}

fn default_base_url() -> String {
    "http://localhost:11434".to_string()
}

fn default_model() -> String {
    "qwen3-vl:30b".to_string()
}

impl LlmSettings {
    pub fn to_llm_config(&self) -> Result<LlmConfig> {
        match self.backend.to_lowercase().as_str() {
            "ollama" => Ok(LlmConfig::Ollama {
                url: self.base_url.clone(),
                model: self.model.clone(),
                temperature: self.temperature,
                repetition_penalty: self.repetition_penalty,
            }),
            "vllm" => Ok(LlmConfig::Vllm {
                url: self.base_url.clone(),
                model: self.model.clone(),
                api_key: self.api_key.clone(),
                temperature: self.temperature,
                repetition_penalty: self.repetition_penalty,
            }),
            "openai" => {
                let api_key = self.api_key.clone().ok_or_else(|| {
                    crate::Error::LlmApiError("API key required for OpenAI backend".to_string())
                })?;
                Ok(LlmConfig::OpenAI {
                    url: self.base_url.clone(),
                    model: self.model.clone(),
                    api_key,
                    temperature: self.temperature,
                    repetition_penalty: self.repetition_penalty,
                })
            }
            _ => Err(crate::Error::LlmApiError(format!(
                "Unknown backend: {}. Must be 'ollama', 'vllm', or 'openai'",
                self.backend
            ))),
        }
    }
}

impl Default for LlmSettings {
    fn default() -> Self {
        Self {
            backend: default_backend(),
            base_url: default_base_url(),
            model: default_model(),
            api_key: None,
            temperature: default_temperature(),
            repetition_penalty: None,
        }
    }
}

/// Cursor/mouse movement configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CursorSettings {
    #[serde(default = "default_smooth_fps")]
    pub smooth_fps: u32,
}

fn default_smooth_fps() -> u32 {
    60
}

impl Default for CursorSettings {
    fn default() -> Self {
        Self {
            smooth_fps: default_smooth_fps(),
        }
    }
}

impl Config {
    /// Get the config file path
    pub fn path() -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home).join(".config/visionctl/config.toml")
    }

    /// Load config from file, returning defaults if file doesn't exist
    pub fn load() -> Self {
        let path = Self::path();
        if path.exists() {
            match fs::read_to_string(&path) {
                Ok(contents) => match toml::from_str(&contents) {
                    Ok(config) => return config,
                    Err(e) => warn!(error = %e, "Failed to parse config"),
                },
                Err(e) => warn!(error = %e, "Failed to read config"),
            }
        }
        Self::default()
    }

    /// Save config to file
    pub fn save(&self) -> std::io::Result<()> {
        let path = Self::path();

        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let contents = toml::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        fs::write(path, contents)
    }
}
