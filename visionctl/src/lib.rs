mod error;
mod llm;
mod primitives;
mod actions;
pub mod agent;
pub mod detection;

pub use error::{Error, Result};
pub use llm::LlmConfig;
pub use primitives::{GridConfig, GridStyle, LabelScheme, CursorPos, GridMode, find_cursor};
pub use actions::MouseButton;
pub use agent::{Agent, AgentConfig, AgentResult};
pub use detection::Detection;

use llm::LlmClient;
use primitives::{ScreenshotOptions, capture_screenshot, capture_screenshot_simple};

/// Virtual vision controller for screen capture, LLM queries, and GUI automation
pub struct VisionCtl {
    llm_client: Option<LlmClient>,
    pub(crate) grid_config: GridConfig,
}

impl VisionCtl {
    /// Create a new VisionCtl with LLM backend configuration
    pub fn new(config: LlmConfig) -> Result<Self> {
        Ok(Self {
            llm_client: Some(LlmClient::new(config)?),
            grid_config: GridConfig::default(),
        })
    }

    /// Create VisionCtl without LLM (for script-driven automation without vision queries)
    pub fn new_headless() -> Self {
        Self {
            llm_client: None,
            grid_config: GridConfig::default(),
        }
    }

    /// Set grid configuration for screenshots
    pub fn set_grid_config(&mut self, config: GridConfig) {
        self.grid_config = config;
    }

    // === Primitives (both patterns use these) ===

    /// Capture a screenshot and return PNG bytes (static method)
    pub fn screenshot() -> Result<Vec<u8>> {
        capture_screenshot_simple()
    }

    /// Capture screenshot with grid overlay and cursor marker
    pub fn screenshot_with_grid(&self) -> Result<Vec<u8>> {
        let options = ScreenshotOptions {
            grid: Some(self.grid_config.clone()),
            mark_cursor: true,
        };
        let data = capture_screenshot(options)?;
        Ok(data.png_bytes)
    }

    /// Capture screenshot with cursor marker only (no grid)
    pub fn screenshot_with_cursor(&self) -> Result<Vec<u8>> {
        let options = ScreenshotOptions {
            grid: None,
            mark_cursor: true,
        };
        let data = capture_screenshot(options)?;
        Ok(data.png_bytes)
    }

    /// Find cursor position
    pub fn find_cursor(&self) -> Result<CursorPos> {
        primitives::find_cursor()
    }

    // === LLM interaction (script-driven pattern) ===

    /// Capture screenshot and query the LLM (backward compatible)
    pub fn ask(&self, question: &str) -> Result<String> {
        let client = self.llm_client.as_ref()
            .ok_or_else(|| Error::ScreenshotFailed("No LLM configured - use new() instead of new_headless()".into()))?;
        let image = self.screenshot_with_grid()?;
        client.query(&image, question)
    }

    // === Actions (GUI automation) ===

    /// Click at grid cell (e.g., "B3")
    pub fn click_at_grid(&self, cell: &str) -> Result<()> {
        actions::click_at_grid(cell, &self.grid_config, MouseButton::Left)
    }

    /// Click at grid cell with specific button
    pub fn click_at_grid_with_button(&self, cell: &str, button: MouseButton) -> Result<()> {
        actions::click_at_grid(cell, &self.grid_config, button)
    }

    /// Move mouse to grid cell
    pub fn move_to_grid(&self, cell: &str, smooth: bool) -> Result<()> {
        actions::move_to_grid(cell, &self.grid_config, smooth)
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
        #[pyo3(signature = (backend, url, model, api_key=None))]
        fn new(backend: &str, url: &str, model: &str, api_key: Option<&str>) -> PyResult<Self> {
            let config = match backend.to_lowercase().as_str() {
                "ollama" => LlmConfig::Ollama {
                    url: url.to_string(),
                    model: model.to_string(),
                },
                "vllm" => LlmConfig::Vllm {
                    url: url.to_string(),
                    model: model.to_string(),
                    api_key: api_key.map(|s| s.to_string()),
                },
                "openai" => {
                    let key = api_key.ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err("api_key required for OpenAI backend")
                    })?;
                    LlmConfig::OpenAI {
                        url: url.to_string(),
                        model: model.to_string(),
                        api_key: key.to_string(),
                    }
                }
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        format!("Unknown backend: {}. Must be 'ollama', 'vllm', or 'openai'", backend)
                    ));
                }
            };

            let inner = VisionCtl::new(config).map_err(|e| {
                pyo3::exceptions::PyOSError::new_err(e.to_string())
            })?;

            Ok(Self { inner })
        }

        /// Create VisionCtl without LLM (headless mode for script-driven automation)
        #[staticmethod]
        fn new_headless() -> Self {
            Self {
                inner: VisionCtl::new_headless(),
            }
        }

        /// Capture a screenshot and return PNG bytes
        #[staticmethod]
        fn screenshot() -> PyResult<Vec<u8>> {
            VisionCtl::screenshot()
                .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))
        }

        /// Capture screenshot with grid overlay and return PNG bytes
        #[pyo3(signature = (grid=true))]
        fn screenshot_with_grid(&self, grid: bool) -> PyResult<Vec<u8>> {
            if grid {
                self.inner.screenshot_with_grid()
                    .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))
            } else {
                VisionCtl::screenshot()
                    .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))
            }
        }

        /// Capture screenshot and query LLM
        fn ask(&self, question: &str) -> PyResult<String> {
            self.inner.ask(question)
                .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))
        }

        /// Click at grid cell (e.g., "B3")
        fn click_at_grid(&self, cell: &str) -> PyResult<()> {
            self.inner.click_at_grid(cell)
                .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))
        }

        /// Move mouse to grid cell
        #[pyo3(signature = (cell, smooth=true))]
        fn move_to_grid(&self, cell: &str, smooth: bool) -> PyResult<()> {
            self.inner.move_to_grid(cell, smooth)
                .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))
        }

        /// Type text using keyboard
        fn type_text(&self, text: &str) -> PyResult<()> {
            self.inner.type_text(text)
                .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))
        }

        /// Press a key (currently only single characters supported)
        fn key_press(&self, key: &str) -> PyResult<()> {
            self.inner.key_press(key)
                .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))
        }

        /// Get tool definitions for LLM tool-calling (returns list of dicts)
        fn get_tool_definitions(&self) -> PyResult<Vec<PyObject>> {
            Python::with_gil(|py| {
                let tools = self.inner.get_tool_definitions();
                tools.iter().map(|tool| {
                    let dict = pyo3::types::PyDict::new_bound(py);
                    dict.set_item("name", &tool.name)?;
                    dict.set_item("description", &tool.description)?;
                    dict.set_item("input_schema", tool.input_schema.to_string())?;
                    Ok(dict.into())
                }).collect()
            })
        }

        /// Execute a tool by name with parameters (for LLM tool-calling)
        fn execute_tool(&self, name: &str, params: PyObject) -> PyResult<PyObject> {
            Python::with_gil(|py| {
                // Convert Python dict to JSON Value
                let json_str = params.call_method0(py, "__str__")?;
                let json_str: String = json_str.extract(py)?;
                let params_value: serde_json::Value = serde_json::from_str(&json_str)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON: {}", e)))?;

                // Execute tool
                let result = self.inner.execute_tool(name, params_value)
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
