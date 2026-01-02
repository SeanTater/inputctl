use crate::{Result, VisionCtl};
use serde_json::{json, Value};

/// Tool definition for LLM tool-calling APIs (Anthropic/OpenAI compatible)
#[derive(Clone, Debug)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// Get the plan tool (first iteration only)
pub fn get_plan_tool() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "plan".to_string(),
            description: "Create a plan for accomplishing the goal based on what you see on screen.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "observations": {
                        "type": "string",
                        "description": "What you see on screen (windows, buttons, relevant UI elements)"
                    },
                    "target_cell": {
                        "type": "string",
                        "description": "The grid cell containing the target element (e.g., 'R1' for row 1 column R)"
                    },
                    "action": {
                        "type": "string",
                        "description": "What action to take (e.g., 'click the minimize button')"
                    }
                },
                "required": ["observations", "target_cell", "action"]
            })
        },
    ]
}

/// Get action tools (after planning)
pub fn get_action_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "move_direction".to_string(),
            description: "Move the cursor in a direction by a number of pixels. Use this to guide cursor toward a target.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down", "left", "right"],
                        "description": "Direction to move the cursor"
                    },
                    "pixels": {
                        "type": "integer",
                        "description": "Number of pixels to move (typically 50-300)"
                    }
                },
                "required": ["direction", "pixels"]
            })
        },
        ToolDefinition {
            name: "move_to".to_string(),
            description: "Move the cursor to a grid cell. Use this first to position cursor before clicking.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "cell": {
                        "type": "string",
                        "description": "Grid cell to move to (e.g., 'A1', 'R3')"
                    }
                },
                "required": ["cell"]
            })
        },
        ToolDefinition {
            name: "click".to_string(),
            description: "Click at the current cursor position. Only use after cursor is at the target.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "button": {
                        "type": "string",
                        "enum": ["left", "right", "middle"],
                        "description": "Mouse button to click",
                        "default": "left"
                    }
                }
            })
        },
        ToolDefinition {
            name: "type_text".to_string(),
            description: "Type text at the current cursor/focus position.".to_string(),
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
            description: "Press a key (e.g., 'enter', 'escape', 'tab').".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key to press"
                    }
                },
                "required": ["key"]
            })
        },
        ToolDefinition {
            name: "task_complete".to_string(),
            description: "Signal that the task is done (success or failure).".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "success": {
                        "type": "boolean",
                        "description": "Whether the task was completed successfully"
                    },
                    "message": {
                        "type": "string",
                        "description": "Summary of what was done or why it failed"
                    }
                },
                "required": ["success", "message"]
            })
        },
    ]
}

/// Get tool definitions for LLM tool-calling (legacy, returns all tools)
pub fn get_tool_definitions() -> Vec<ToolDefinition> {
    let mut tools = get_plan_tool();
    tools.extend(get_action_tools());
    tools
}

/// Get tools for iterative cursor targeting (simplified set)
pub fn get_targeting_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "move_direction".to_string(),
            description: "Move the cursor toward the target. Call this when cursor is NOT on the target.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down", "left", "right"],
                        "description": "Direction to move the cursor"
                    },
                    "percent": {
                        "type": "integer",
                        "description": "Percentage of screen to move (1-100). E.g., 10 = 10% of screen width/height"
                    }
                },
                "required": ["direction", "percent"]
            })
        },
        ToolDefinition {
            name: "click".to_string(),
            description: "Click at the current cursor position. ONLY call this when the cursor is on the target.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "button": {
                        "type": "string",
                        "enum": ["left", "right", "middle"],
                        "default": "left"
                    }
                }
            })
        },
        ToolDefinition {
            name: "target_reached".to_string(),
            description: "Signal that cursor is on target but no click is needed, or give up.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "success": {
                        "type": "boolean",
                        "description": "true if cursor reached target, false if giving up"
                    },
                    "message": {
                        "type": "string",
                        "description": "Status message"
                    }
                },
                "required": ["success"]
            })
        },
    ]
}

/// Execute a tool by name with parameters
pub fn execute_tool(ctl: &VisionCtl, name: &str, params: Value) -> Result<Value> {
    match name {
        "move_direction" => {
            let direction = params.get("direction")
                .and_then(|v| v.as_str())
                .ok_or_else(|| crate::Error::ScreenshotFailed("Missing 'direction' parameter".into()))?;

            // Support both percent (new) and pixels (legacy)
            let (pixels, is_percent) = if let Some(pct) = params.get("percent").and_then(|v| v.as_i64()) {
                // Convert percentage to pixels (assume ~4K ultrawide: 6912x2160)
                // TODO: Get actual screen dimensions dynamically
                let screen_width = 6912i64;
                let screen_height = 2160i64;
                let px = match direction {
                    "left" | "right" => (screen_width * pct / 100) as i32,
                    "up" | "down" => (screen_height * pct / 100) as i32,
                    _ => 100,
                };
                (px, true)
            } else if let Some(px) = params.get("pixels").and_then(|v| v.as_i64()) {
                (px as i32, false)
            } else {
                return Err(crate::Error::ScreenshotFailed("Missing 'percent' or 'pixels' parameter".into()));
            };

            let (dx, dy) = match direction {
                "up" => (0, -pixels),
                "down" => (0, pixels),
                "left" => (-pixels, 0),
                "right" => (pixels, 0),
                _ => return Err(crate::Error::ScreenshotFailed(format!("Invalid direction: {}", direction))),
            };

            crate::actions::move_relative(dx, dy, true)?;
            let msg = if is_percent {
                format!("Moved cursor {} ({}% = {}px)", direction, params.get("percent").unwrap(), pixels)
            } else {
                format!("Moved cursor {} by {}px", direction, pixels)
            };
            Ok(json!({
                "success": true,
                "message": msg
            }))
        }
        "plan" => {
            // Plan is informational - just echo back the plan
            let observations = params.get("observations").and_then(|v| v.as_str()).unwrap_or("");
            let target_cell = params.get("target_cell").and_then(|v| v.as_str()).unwrap_or("");
            let action = params.get("action").and_then(|v| v.as_str()).unwrap_or("");
            Ok(json!({
                "success": true,
                "observations": observations,
                "target_cell": target_cell,
                "action": action,
                "message": "Plan recorded. Now execute the action."
            }))
        }
        "move_to" => {
            let cell = params.get("cell")
                .and_then(|v| v.as_str())
                .ok_or_else(|| crate::Error::ScreenshotFailed("Missing 'cell' parameter".into()))?;

            ctl.move_to_grid(cell, true)?;
            Ok(json!({
                "success": true,
                "message": format!("Moved cursor to cell {}", cell)
            }))
        }
        "click" => {
            let button_str = params.get("button")
                .and_then(|v| v.as_str())
                .unwrap_or("left");

            let button = match button_str {
                "right" => crate::MouseButton::Right,
                "middle" => crate::MouseButton::Middle,
                _ => crate::MouseButton::Left,
            };

            // Click at current position
            crate::actions::click(button)?;
            Ok(json!({
                "success": true,
                "message": format!("Clicked {} button at current position", button_str)
            }))
        }
        // Legacy tools for backwards compatibility
        "screenshot" => {
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

            ctl.move_to_grid(cell, true)?;
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
        "task_complete" => {
            let success = params.get("success").and_then(|v| v.as_bool()).unwrap_or(false);
            let message = params.get("message").and_then(|v| v.as_str()).unwrap_or("");
            Ok(json!({
                "success": success,
                "message": message,
                "is_task_complete": true
            }))
        }
        "target_reached" => {
            let success = params.get("success").and_then(|v| v.as_bool()).unwrap_or(false);
            let message = params.get("message").and_then(|v| v.as_str()).unwrap_or("");
            Ok(json!({
                "success": success,
                "message": message,
                "is_target_reached": true
            }))
        }
        _ => {
            Err(crate::Error::ScreenshotFailed(format!("Unknown tool: {}", name)))
        }
    }
}
