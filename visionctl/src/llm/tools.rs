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
        ToolDefinition {
            name: "list_templates".to_string(),
            description: "List all available template icons. Use this to discover what icons you can search for with find_template or click_template.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {},
                "required": []
            })
        },
        ToolDefinition {
            name: "find_template".to_string(),
            description: "Find a template icon on screen and report its location. Returns coordinates, confidence, and number of matches. Use when you need to know where an icon is without clicking it.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Template name without .png extension (e.g., 'gwenview'). Use list_templates() to see available names."
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Confidence threshold 0.0-1.0 (default 0.8). Lower = more permissive matching.",
                        "default": 0.8
                    }
                },
                "required": ["name"]
            })
        },
        ToolDefinition {
            name: "click_template".to_string(),
            description: "Find a template icon on screen and click it. Best for clicking small taskbar icons or UI elements that are hard to target with the grid. Combines find + click in one action.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Template name without .png extension (e.g., 'gwenview')"
                    },
                    "button": {
                        "type": "string",
                        "enum": ["left", "right", "middle"],
                        "description": "Mouse button to click",
                        "default": "left"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Confidence threshold 0.0-1.0 (default 0.8)",
                        "default": 0.8
                    }
                },
                "required": ["name"]
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

/// Execute a tool by name with parameters
pub fn execute_tool(ctl: &VisionCtl, name: &str, params: Value) -> Result<Value> {
    match name {
        "move_direction" => {
            let direction = params.get("direction")
                .and_then(|v| v.as_str())
                .ok_or_else(|| crate::Error::ScreenshotFailed("Missing 'direction' parameter".into()))?;

            // Support both percent (new) and pixels (legacy)
            let (pixels, is_percent) = if let Some(pct) = params.get("percent").and_then(|v| v.as_i64()) {
                // Convert percentage to pixels using actual screen dimensions
                let dims = crate::primitives::get_screen_dimensions()?;
                let px = match direction {
                    "left" | "right" => (dims.width as i64 * pct / 100) as i32,
                    "up" | "down" => (dims.height as i64 * pct / 100) as i32,
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
        "list_templates" => {
            use crate::detection;
            let templates = detection::list_available_templates()?;
            Ok(json!({
                "success": true,
                "templates": templates,
                "count": templates.len()
            }))
        }
        "find_template" => {
            use crate::detection;

            let name = params.get("name")
                .and_then(|v| v.as_str())
                .ok_or_else(|| crate::Error::ScreenshotFailed("Missing 'name' parameter".into()))?;

            let threshold = params.get("threshold")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(0.8);

            // Take screenshot
            let screenshot = crate::VisionCtl::screenshot()?;
            let temp_path = "/tmp/visionctl_template_search.png";
            std::fs::write(temp_path, &screenshot)
                .map_err(|e| crate::Error::ScreenshotFailed(format!("Failed to write screenshot: {}", e)))?;

            // Find template file
            let template_path = detection::find_template_file(name)?;

            // Search for template
            let detections = detection::find_template(
                temp_path,
                template_path.to_str().unwrap(),
                Some(threshold)
            )?;

            // Clean up
            let _ = std::fs::remove_file(temp_path);

            if detections.is_empty() {
                Ok(json!({
                    "success": false,
                    "found": false,
                    "message": format!("Template '{}' not found on screen (threshold: {})", name, threshold)
                }))
            } else {
                let best = &detections[0];
                Ok(json!({
                    "success": true,
                    "found": true,
                    "x": best.x,
                    "y": best.y,
                    "confidence": best.confidence,
                    "num_matches": detections.len(),
                    "message": format!("Found {} match(es) for '{}'", detections.len(), name)
                }))
            }
        }
        "click_template" => {
            use crate::detection;
            use crate::actions::{MouseButton, click_at_pixel};

            let name = params.get("name")
                .and_then(|v| v.as_str())
                .ok_or_else(|| crate::Error::ScreenshotFailed("Missing 'name' parameter".into()))?;

            let button_str = params.get("button")
                .and_then(|v| v.as_str())
                .unwrap_or("left");

            let button = match button_str {
                "left" => MouseButton::Left,
                "right" => MouseButton::Right,
                "middle" => MouseButton::Middle,
                _ => MouseButton::Left,
            };

            let threshold = params.get("threshold")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(0.8);

            // Take screenshot
            let screenshot = crate::VisionCtl::screenshot()?;
            let temp_path = "/tmp/visionctl_template_search.png";
            std::fs::write(temp_path, &screenshot)
                .map_err(|e| crate::Error::ScreenshotFailed(format!("Failed to write screenshot: {}", e)))?;

            // Find template file
            let template_path = detection::find_template_file(name)?;

            // Search for template
            let detections = detection::find_template(
                temp_path,
                template_path.to_str().unwrap(),
                Some(threshold)
            )?;

            // Clean up
            let _ = std::fs::remove_file(temp_path);

            if detections.is_empty() {
                Ok(json!({
                    "success": false,
                    "message": format!("Template '{}' not found on screen (threshold: {})", name, threshold)
                }))
            } else {
                let best = &detections[0];

                // Click at template location
                click_at_pixel(best.x, best.y, button)?;

                Ok(json!({
                    "success": true,
                    "x": best.x,
                    "y": best.y,
                    "confidence": best.confidence,
                    "num_matches": detections.len(),
                    "message": format!("Clicked template '{}' at ({}, {}) with {:.2} confidence",
                        name, best.x, best.y, best.confidence)
                }))
            }
        }
        _ => {
            Err(crate::Error::ScreenshotFailed(format!("Unknown tool: {}", name)))
        }
    }
}
