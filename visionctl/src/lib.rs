//! `visionctl` is a vision-based GUI automation toolkit for Linux (Wayland).
//!
//! It provides tools for screen capture, LLM-based image analysis, and
//! programmatic control of the mouse and keyboard. It is designed to be used
//! both for simple script-driven automation and for building autonomous agents.
//!
//! # Core Components
//! - [`VisionCtl`]: The main controller for screenshots, coordinate conversion, and LLM interaction.
//! - [`agent`]: An autonomous agent that can execute tools to achieve goals on a desktop.
//! - [`debugger`]: Tools for tracking and visualizing the agent's internal state.
//! - [`server`]: An embedded web server to expose debugging information.
//!
//! # Examples
//! See the `examples/` directory for common usage patterns.

mod actions;
pub mod agent;
pub mod config;
pub mod debugger;
pub mod detection;
mod error;
mod llm;
mod primitives;
pub mod recorder;
pub mod server;

pub use error::{Error, Result};

pub use actions::MouseButton;
pub use agent::{Agent, AgentConfig, AgentResult};
pub use config::{Config, CursorSettings, LlmSettings};
pub use detection::Detection;
pub use llm::{parse_coordinates, LlmConfig};
pub use primitives::{
    capture_screenshot_image, find_cursor, find_window, get_screen_dimensions, list_windows,
    CursorPos, Region, ScreenshotOptions, Window,
};

use crate::debugger::{AgentObserver, NoopObserver};
use llm::LlmClient;
use primitives::{capture_screenshot, capture_screenshot_simple};
use std::sync::Arc;

/// Virtual vision controller for screen capture, LLM queries, and GUI automation
pub struct VisionCtl {
    llm_client: Option<LlmClient>,
    pointing_client: Option<LlmClient>,
    viewport: Option<Region>,
    observer: Arc<dyn AgentObserver>,
}

impl VisionCtl {
    /// Create a new VisionCtl with LLM backend configuration
    pub fn new(config: LlmConfig) -> Result<Self> {
        Ok(Self {
            llm_client: Some(LlmClient::new(config)?),
            pointing_client: None,
            viewport: None,
            observer: Arc::new(NoopObserver),
        })
    }

    /// Create a new VisionCtl with separate models for general reasoning and pointing
    pub fn new_delegated(main_config: LlmConfig, pointing_config: LlmConfig) -> Result<Self> {
        Ok(Self {
            llm_client: Some(LlmClient::new(main_config)?),
            pointing_client: Some(LlmClient::new(pointing_config)?),
            viewport: None,
            observer: Arc::new(NoopObserver),
        })
    }

    /// Create a new VisionCtl based on the system configuration file
    pub fn new_from_config() -> Result<Self> {
        let config = crate::config::Config::load();
        let main_llm = config.llm.to_llm_config()?;
        let pointing_llm = if let Some(p) = config.pointing {
            Some(llm::LlmClient::new(p.to_llm_config()?)?)
        } else {
            None
        };

        Ok(Self {
            llm_client: Some(llm::LlmClient::new(main_llm)?),
            pointing_client: pointing_llm,
            viewport: None,
            observer: Arc::new(NoopObserver),
        })
    }

    /// Create VisionCtl without LLM (for script-driven automation without vision queries)
    pub fn new_headless() -> Self {
        Self {
            llm_client: None,
            pointing_client: None,
            viewport: None,
            observer: Arc::new(NoopObserver),
        }
    }

    /// Set the active viewport region
    pub fn set_viewport(&mut self, region: Option<Region>) {
        self.viewport = region;
        self.observer.on_viewport_change(region);
    }

    /// Get the active viewport region
    pub fn get_viewport(&self) -> Option<Region> {
        self.viewport
    }

    /// Convert normalized coordinates (0-1000) to screen pixels
    /// Respects the active viewport if set
    pub fn to_screen_coords(&self, norm_x: i32, norm_y: i32) -> Result<(i32, i32)> {
        if let Some(region) = self.viewport {
            let px = region.x + (norm_x * region.width as i32 / 1000);
            let py = region.y + (norm_y * region.height as i32 / 1000);
            Ok((px, py))
        } else {
            let dims = primitives::get_screen_dimensions()?;
            let px = norm_x * dims.width as i32 / 1000;
            let py = norm_y * dims.height as i32 / 1000;
            Ok((px, py))
        }
    }

    /// Convert screen pixels to normalized coordinates (0-1000)
    /// Returns None if the point is outside the active viewport
    pub fn to_normalized_coords(&self, screen_x: i32, screen_y: i32) -> Result<Option<(i32, i32)>> {
        if let Some(region) = self.viewport {
            if !region.contains(screen_x, screen_y) {
                return Ok(None);
            }
            let norm_x = (screen_x - region.x) * 1000 / region.width as i32;
            let norm_y = (screen_y - region.y) * 1000 / region.height as i32;
            Ok(Some((norm_x, norm_y)))
        } else {
            let dims = primitives::get_screen_dimensions()?;
            // Clamp to screen bounds for safety, or allow out of bounds?
            // "Normalized" usually implies 0-1000, so we should probably check bounds or clamp.
            // But if it's outside the screen, it's outside the "viewport" of the full screen too.
            if screen_x < 0
                || screen_y < 0
                || screen_x > dims.width as i32
                || screen_y > dims.height as i32
            {
                // Technically outside full screen, but maybe we just clamp?
                // Let's stick to the pattern: if it's wildly out, maybe None?
                // But for full screen, usually we just clamp in move_to.
                // Let's just do the math.
            }
            let norm_x = screen_x * 1000 / dims.width as i32;
            let norm_y = screen_y * 1000 / dims.height as i32;
            Ok(Some((norm_x, norm_y)))
        }
    }

    // === Primitives (both patterns use these) ===

    /// Capture a screenshot and return PNG bytes (static method)
    pub fn screenshot() -> Result<Vec<u8>> {
        capture_screenshot_simple()
    }

    /// Capture raw screenshot (static method)
    pub fn screenshot_raw(width: u32, height: u32) -> Result<Vec<u8>> {
        primitives::screenshot::capture_screenshot_raw(width, height)
    }

    /// Capture raw screenshot with cropping (static method)
    pub fn screenshot_raw_cropped(region: Option<Region>) -> Result<(Vec<u8>, u32, u32)> {
        primitives::screenshot::capture_screenshot_raw_cropped(region)
    }

    pub fn screenshot_with_cursor(&self) -> Result<Vec<u8>> {
        let dims = primitives::get_screen_dimensions()?;
        let options = crate::primitives::ScreenshotOptions {
            mark_cursor: true,
            crop_region: self.viewport,
            resize_to_logical: Some((dims.width, dims.height)),
        };
        let data = capture_screenshot(options)?;
        Ok(data.png_bytes)
    }

    /// Find a window by title (wrapper around primitive)
    pub fn find_window(name: &str) -> Result<Option<Window>> {
        primitives::find_window(name)
    }

    /// Find cursor position
    pub fn find_cursor(&self) -> Result<CursorPos> {
        primitives::find_cursor()
    }

    // === LLM interaction (script-driven pattern) ===

    /// Capture screenshot and query the LLM
    pub fn ask(&self, question: &str) -> Result<String> {
        let client = self.llm_client.as_ref().ok_or_else(|| {
            Error::ScreenshotFailed(
                "No LLM configured - use new() instead of new_headless()".into(),
            )
        })?;
        let image = self.screenshot_with_cursor()?;
        client.query(&image, question)
    }

    /// Query the pointing specialist specifically
    pub fn ask_pointing(&self, question: &str) -> Result<String> {
        let image = self.screenshot_with_cursor()?;
        if let Some(client) = &self.pointing_client {
            client.query(&image, question)
        } else if let Some(client) = &self.llm_client {
            // Fallback to main model if no pointing specialist is configured
            client.query(&image, question)
        } else {
            Err(Error::ScreenshotFailed(
                "No LLM configured for pointing".into(),
            ))
        }
    }
    /// Query the pointing specialist with a provided image
    pub fn query_pointing_image(&self, image: &[u8], question: &str) -> Result<String> {
        if let Some(client) = &self.pointing_client {
            client.query(image, question)
        } else if let Some(client) = &self.llm_client {
            // Fallback to main model if no pointing specialist is configured
            client.query(image, question)
        } else {
            Err(Error::ScreenshotFailed(
                "No LLM configured for pointing".into(),
            ))
        }
    }

    // === Actions (GUI automation) ===

    /// Move cursor to position using 0-1000 normalized coordinates
    pub fn move_to(&self, x: i32, y: i32, smooth: bool) -> Result<()> {
        let (px, py) = self.to_screen_coords(x, y)?;
        actions::move_to_pixel(px, py, smooth)
    }

    /// Click at current cursor position
    pub fn click(&self, button: MouseButton) -> Result<()> {
        actions::click(button)
    }

    /// Type text using keyboard
    pub fn type_text(&self, text: &str) -> Result<()> {
        actions::type_text(text)
    }

    /// Press a key (e.g., "enter", "escape", "ctrl")
    pub fn key_press(&self, key: &str) -> Result<()> {
        actions::key_press(key)
    }

    // === Tool-calling API (LLM-driven pattern) ===

    /// Get tool definitions for LLM tool-calling (Anthropic/OpenAI compatible)
    pub fn get_tool_definitions(&self) -> Vec<llm::ToolDefinition> {
        llm::get_tool_definitions()
    }

    /// Execute a tool by name with parameters (for LLM tool-calling)
    pub fn execute_tool(&self, name: &str, params: serde_json::Value) -> Result<serde_json::Value> {
        llm::execute_tool(self, name, params)
    }

    /// Set an observer for VisionCtl
    pub fn with_observer(mut self, observer: Arc<dyn AgentObserver>) -> Self {
        self.observer = observer;
        self
    }
}

// Python bindings
#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::prelude::*;

    #[pyclass(name = "VisionCtl")]
    pub struct PyVisionCtl {
        inner: VisionCtl,
    }

    #[pymethods]
    impl PyVisionCtl {
        #[new]
        #[pyo3(signature = (backend, url, model, api_key=None, temperature=0.0))]
        fn new(
            backend: &str,
            url: &str,
            model: &str,
            api_key: Option<&str>,
            temperature: f32,
        ) -> PyResult<Self> {
            let settings = LlmSettings {
                backend: backend.to_string(),
                base_url: url.to_string(),
                model: model.to_string(),
                api_key: api_key.map(|s| s.to_string()),
                temperature,
            };

            let config = settings
                .to_llm_config()
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

            let inner = VisionCtl::new(config)
                .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;

            Ok(Self { inner })
        }

        /// Create VisionCtl without LLM (headless mode for script-driven automation)
        #[staticmethod]
        fn new_headless() -> Self {
            Self {
                inner: VisionCtl::new_headless(),
            }
        }

        /// Configure a dedicated pointing model (e.g., Qwen3)
        #[pyo3(signature = (backend, url, model, api_key=None, temperature=0.0))]
        fn set_pointing_model(
            &mut self,
            backend: &str,
            url: &str,
            model: &str,
            api_key: Option<&str>,
            temperature: f32,
        ) -> PyResult<()> {
            let settings = LlmSettings {
                backend: backend.to_string(),
                base_url: url.to_string(),
                model: model.to_string(),
                api_key: api_key.map(|s| s.to_string()),
                temperature,
            };

            let config = settings
                .to_llm_config()
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

            let client = LlmClient::new(config)
                .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;

            self.inner.pointing_client = Some(client);
            Ok(())
        }

        /// Capture a screenshot and return PNG bytes
        #[staticmethod]
        fn screenshot() -> PyResult<Vec<u8>> {
            VisionCtl::screenshot().map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))
        }

        /// Capture screenshot with cursor marker and return PNG bytes
        fn screenshot_with_cursor(&self) -> PyResult<Vec<u8>> {
            self.inner
                .screenshot_with_cursor()
                .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))
        }

        /// Capture screenshot and query LLM
        fn ask(&self, question: &str) -> PyResult<String> {
            self.inner
                .ask(question)
                .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))
        }

        /// Move cursor to position using 0-1000 normalized coordinates
        #[pyo3(signature = (x, y, smooth=true))]
        fn move_to(&self, x: i32, y: i32, smooth: bool) -> PyResult<()> {
            self.inner
                .move_to(x, y, smooth)
                .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))
        }

        /// Click at current cursor position
        #[pyo3(signature = (button="left"))]
        fn click(&self, button: &str) -> PyResult<()> {
            let btn = match button {
                "right" => MouseButton::Right,
                "middle" => MouseButton::Middle,
                _ => MouseButton::Left,
            };
            self.inner
                .click(btn)
                .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))
        }

        /// Type text using keyboard
        fn type_text(&self, text: &str) -> PyResult<()> {
            self.inner
                .type_text(text)
                .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))
        }

        /// Press a key (e.g., "enter", "escape", "ctrl")
        fn key_press(&self, key: &str) -> PyResult<()> {
            self.inner
                .key_press(key)
                .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))
        }

        /// Get tool definitions for LLM tool-calling (returns list of dicts)
        fn get_tool_definitions(&self) -> PyResult<Vec<PyObject>> {
            Python::with_gil(|py| {
                let tools = self.inner.get_tool_definitions();
                tools
                    .iter()
                    .map(|tool| {
                        let dict = pyo3::types::PyDict::new_bound(py);
                        dict.set_item("name", &tool.name)?;
                        dict.set_item("description", &tool.description)?;
                        dict.set_item("input_schema", tool.input_schema.to_string())?;
                        Ok(dict.into())
                    })
                    .collect()
            })
        }

        /// Execute a tool by name with parameters (for LLM tool-calling)
        fn execute_tool(&self, name: &str, params: PyObject) -> PyResult<PyObject> {
            Python::with_gil(|py| {
                // Convert Python dict to JSON Value
                let json_str = params.call_method0(py, "__str__")?;
                let json_str: String = json_str.extract(py)?;
                let params_value: serde_json::Value =
                    serde_json::from_str(&json_str).map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON: {}", e))
                    })?;

                // Execute tool
                let result = self
                    .inner
                    .execute_tool(name, params_value)
                    .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;

                // Convert result back to Python dict
                let result_str = result.to_string();
                let json_module = py.import_bound("json")?;
                let py_result = json_module.call_method1("loads", (result_str,))?;
                Ok(py_result.into())
            })
        }
    }

    #[pymodule]
    fn visionctl(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyVisionCtl>()?;
        Ok(())
    }
}
