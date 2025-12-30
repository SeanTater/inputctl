use crate::{Result, VisionCtl};
use serde_json::{json, Value};

/// Tool definition for LLM tool-calling APIs (Anthropic/OpenAI compatible)
#[derive(Clone, Debug)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// Get tool definitions for LLM tool-calling
pub fn get_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "screenshot".to_string(),
            description: "Capture the current screen with optional grid overlay. Returns image data with grid labels (A1, B2, etc.) for spatial reference.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "grid": {
                        "type": "boolean",
                        "description": "Whether to overlay a grid with cell labels (A1, B2, etc.) for spatial reference",
                        "default": true
                    },
                    "mark_cursor": {
                        "type": "boolean",
                        "description": "Whether to mark the cursor position on the screenshot",
                        "default": false
                    }
                }
            })
        },
        ToolDefinition {
            name: "click_at_grid".to_string(),
            description: "Click at a grid cell position (e.g., 'B3'). Use after taking a screenshot with grid overlay to identify the target cell.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "cell": {
                        "type": "string",
                        "description": "Grid cell identifier (e.g., 'A1', 'B3', 'C5')"
                    },
                    "button": {
                        "type": "string",
                        "enum": ["left", "right", "middle"],
                        "description": "Mouse button to click",
                        "default": "left"
                    }
                },
                "required": ["cell"]
            })
        },
        ToolDefinition {
            name: "move_to_grid".to_string(),
            description: "Move mouse to a grid cell without clicking.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "cell": {
                        "type": "string",
                        "description": "Grid cell identifier (e.g., 'A1', 'B3')"
                    },
                    "smooth": {
                        "type": "boolean",
                        "description": "Whether to move smoothly (true) or instantly (false)",
                        "default": true
                    }
                },
                "required": ["cell"]
            })
        },
        ToolDefinition {
            name: "type_text".to_string(),
            description: "Type text using the keyboard at the current cursor/focus position.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to type"
                    }
                },
                "required": ["text"]
            })
        },
        ToolDefinition {
            name: "key_press".to_string(),
            description: "Press a specific key (e.g., 'enter', 'escape', 'tab', 'ctrl', 'alt').".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key to press (e.g., 'enter', 'escape', 'tab', 'ctrl+c')"
                    }
                },
                "required": ["key"]
            })
        },
        ToolDefinition {
            name: "find_cursor".to_string(),
            description: "Get the current cursor position and its grid cell if a grid is configured.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {}
            })
        },
    ]
}

/// Execute a tool by name with parameters
pub fn execute_tool(ctl: &VisionCtl, name: &str, params: Value) -> Result<Value> {
    match name {
        "screenshot" => {
            let _grid = params.get("grid").and_then(|v| v.as_bool()).unwrap_or(true);
            // For now, always use grid since that's what LLMs need for spatial reference
            let png_bytes = ctl.screenshot_with_grid()?;
            Ok(json!({
                "success": true,
                "message": "Screenshot captured with grid overlay",
                "size_bytes": png_bytes.len()
            }))
        }
        "click_at_grid" => {
            let cell = params.get("cell")
                .and_then(|v| v.as_str())
                .ok_or_else(|| crate::Error::ScreenshotFailed("Missing 'cell' parameter".into()))?;

            let button_str = params.get("button")
                .and_then(|v| v.as_str())
                .unwrap_or("left");

            let button = match button_str {
                "right" => crate::MouseButton::Right,
                "middle" => crate::MouseButton::Middle,
                _ => crate::MouseButton::Left,
            };

            ctl.click_at_grid_with_button(cell, button)?;
            Ok(json!({
                "success": true,
                "message": format!("Clicked at grid cell {}", cell)
            }))
        }
        "move_to_grid" => {
            let cell = params.get("cell")
                .and_then(|v| v.as_str())
                .ok_or_else(|| crate::Error::ScreenshotFailed("Missing 'cell' parameter".into()))?;

            let smooth = params.get("smooth")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);

            ctl.move_to_grid(cell, smooth)?;
            Ok(json!({
                "success": true,
                "message": format!("Moved to grid cell {}", cell)
            }))
        }
        "type_text" => {
            let text = params.get("text")
                .and_then(|v| v.as_str())
                .ok_or_else(|| crate::Error::ScreenshotFailed("Missing 'text' parameter".into()))?;

            ctl.type_text(text)?;
            Ok(json!({
                "success": true,
                "message": format!("Typed {} characters", text.len())
            }))
        }
        "key_press" => {
            let key = params.get("key")
                .and_then(|v| v.as_str())
                .ok_or_else(|| crate::Error::ScreenshotFailed("Missing 'key' parameter".into()))?;

            ctl.key_press(key)?;
            Ok(json!({
                "success": true,
                "message": format!("Pressed key '{}'", key)
            }))
        }
        "find_cursor" => {
            let cursor = ctl.find_cursor()?;
            Ok(json!({
                "success": true,
                "x": cursor.x,
                "y": cursor.y,
                "grid_cell": cursor.grid_cell
            }))
        }
        _ => {
            Err(crate::Error::ScreenshotFailed(format!("Unknown tool: {}", name)))
        }
    }
}
