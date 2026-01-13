//! Tool definitions and execution logic for LLM-driven actions.
//!
//! This module defines the tools available to the [`Agent`], such as
//! `type_text`, `move_to`, `click`, and `ask_screen`. It also handles
//! the mapping between LLM function calls and local Rust functions.

use crate::{Result, VisionCtl};
use serde_json::{json, Value};
use serde::{Deserialize, Serialize};

/// Tool definition for LLM tool-calling APIs (Anthropic/OpenAI compatible)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}



/// Get action tools for the agent
pub fn get_action_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "move_to".to_string(),
            description: "Move the cursor to a position using normalized 0-1000 coordinates. (0,0) is top-left, (1000,1000) is bottom-right.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "x": {
                        "type": "integer",
                        "description": "X coordinate (0-1000, where 0=left edge, 1000=right edge)"
                    },
                    "y": {
                        "type": "integer",
                        "description": "Y coordinate (0-1000, where 0=top edge, 1000=bottom edge)"
                    }
                },
                "required": ["x", "y"]
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
            description: "Signal that the FINAl task is done (success or failure).".to_string(),
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
            name: "list_icons".to_string(),
            description: "List all available icon names. Use this to discover what icons you can move to with move_to_icon.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {},
                "required": []
            })
        },
        ToolDefinition {
            name: "move_to_icon".to_string(),
            description: "Find an icon on screen and move the cursor to it. Returns coordinates, confidence, and number of matches.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Icon name without .png extension (e.g., 'gwenview'). Use list_icons() to see available names."
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
            name: "stuck".to_string(),
            description: "Indicate you are stuck and need help.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {},
                "required": []
            })
        },
        ToolDefinition {
            name: "point_at".to_string(),
            description: "Find an object on screen using a secondary vision model (pointing specialist). Moves mouse and returns coordinates. Use for finding specific buttons, icons, or UI elements.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Natural language description of what to find (e.g., 'the close button', 'the search bar')."
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional: Specific pointing model to use (e.g., 'qwen2-vl:2b'). Defaults to the configured LLM."
                    }
                },
                "required": ["description"]
            })
        },
        ToolDefinition {
            name: "ask_screen".to_string(),
            description: "Ask a question about the current screen state. Use this to verify the result of an action or check the state of the UI.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask about the screen content."
                    }
                },
                "required": ["question"]
            })
        },
        ToolDefinition {
            name: "scroll".to_string(),
            description: "Scroll the mouse wheel. Positive dy scrolls up, negative dy scrolls down.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "dx": {
                        "type": "integer",
                        "description": "Horizontal scroll amount (0 for vertical only)"
                    },
                    "dy": {
                        "type": "integer",
                        "description": "Vertical scroll amount (positive=up, negative=down)"
                    }
                },
                "required": ["dx", "dy"]
            })
        },
        ToolDefinition {
            name: "double_click".to_string(),
            description: "Double click at the current cursor position.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "button": {
                        "type": "string",
                        "enum": ["left", "right", "middle"],
                        "description": "Mouse button to double click",
                        "default": "left"
                    }
                }
            })
        },
        ToolDefinition {
            name: "mouse_down".to_string(),
            description: "Press and hold a mouse button at the current position. Use for drag-and-drop.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "button": {
                        "type": "string",
                        "enum": ["left", "right", "middle"],
                        "description": "Mouse button to press down",
                        "default": "left"
                    }
                }
            })
        },
        ToolDefinition {
            name: "mouse_up".to_string(),
            description: "Release a previously pressed mouse button.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "button": {
                        "type": "string",
                        "enum": ["left", "right", "middle"],
                        "description": "Mouse button to release",
                        "default": "left"
                    }
                }
            })
        },
        ToolDefinition {
            name: "key_down".to_string(),
            description: "Press and hold a key (e.g., 'ctrl', 'shift', 'a').".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key to hold down"
                    }
                },
                "required": ["key"]
            })
        },
        ToolDefinition {
            name: "key_up".to_string(),
            description: "Release a previously pressed key.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key to release"
                    }
                },
                "required": ["key"]
            })
        }
    ]
}

/// Get tool definitions for LLM tool-calling (legacy, returns all tools)
pub fn get_tool_definitions() -> Vec<ToolDefinition> {
    get_action_tools()
}

/// Execute a tool by name with parameters
pub fn execute_tool(ctl: &VisionCtl, name: &str, params: Value) -> Result<Value> {
    // Log tool call with arguments
    tracing::debug!(
        tool = %name,
        args = %serde_json::to_string(&params).unwrap_or_else(|_| "serialization failed".to_string()),
        "Executing tool"
    );
    
    let result = match name {
        "move_to" => {
            let x = params.get("x")
                .and_then(|v| v.as_i64())
                .ok_or_else(|| crate::Error::ScreenshotFailed("Missing 'x' parameter".into()))? as i32;
            let y = params.get("y")
                .and_then(|v| v.as_i64())
                .ok_or_else(|| crate::Error::ScreenshotFailed("Missing 'y' parameter".into()))? as i32;

            // Use VisionCtl's coordinate conversion (respects viewport)
            let (px, py) = ctl.to_screen_coords(x, y)?;

            crate::actions::move_to_pixel(px, py, true)?;
            Ok(json!({
                "success": true,
                "message": format!("Moved cursor to ({}, {}) [pixel: ({}, {})]", x, y, px, py)
            }))
        }
        "set_viewport" => {
            let _x = params.get("x").and_then(|v| v.as_i64()).map(|v| v as i32);
            let _y = params.get("y").and_then(|v| v.as_i64()).map(|v| v as i32);
            let _width = params.get("width").and_then(|v| v.as_i64()).map(|v| v as u32);
            let _height = params.get("height").and_then(|v| v.as_i64()).map(|v| v as u32);
            
            // This is a bit tricky because VisionCtl is immutable in execute_tool signature?
            // Ah, execute_tool takes &VisionCtl. We need interior mutability or change signature.
            // But VisionCtl is designed to be shared.
            // Actually, we didn't add interior mutability for viewport in VisionCtl (it's not Mutex/RefCell).
            // That's a blocker. We should have used a RefCell or Mutex for viewport if we want to change it at runtime via tools.
            // Or `execute_tool` should take `&mut VisionCtl`.
            // Let's check `execute_tool` signature. It's `&VisionCtl`.
            // Checking `src/lib.rs`, `VisionCtl` struct fields are just `Option<T>`.
            // We need to change `VisionCtl` to use `Mutex<Option<Region>>` for the viewport if we want to change it.
            // Or assume the agent loop handles viewport changes?
            // The task was "Refactor visionctl to work with ... region", so presumably the tool should be able to set it?
            // Or maybe it's set by the caller?
            // Let's stick to the plan. `set_viewport` is not in the original tool list in `tools.rs`. 
            // The plan didn't explicitly say we'd add a `set_viewport` tool for the LLM. 
            // It said "Update VisionCtl to manage an active viewport".
            // Since we are adding `set_viewport` method to `VisionCtl`, we probably want to expose it?
            // But for now let's just make sure `move_to` works.
            
            // Wait, if I can't set the viewport via tool, how does it get set?
            // Maybe we just expose it for the embedding application?
            // The user request was "refactor visionctl to work with ... a window ...".
            // Usually this means the meaningful "screen" is set programmatically.
            // But for testing we might want to set it.
            // Let's assume for now `move_to` is the priority.
            
            Err(crate::Error::ScreenshotFailed("set_viewport tool not implemented yet".into()))
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

            crate::actions::key_press(key)?;
            Ok(json!({
                "success": true,
                "message": format!("Pressed key '{}'", key)
            }))
        }
        "double_click" => {
            let button_str = params.get("button")
                .and_then(|v| v.as_str())
                .unwrap_or("left");

            let button = match button_str {
                "right" => crate::MouseButton::Right,
                "middle" => crate::MouseButton::Middle,
                _ => crate::MouseButton::Left,
            };

            crate::actions::double_click(button)?;
            Ok(json!({
                "success": true,
                "message": format!("Double clicked {} button at current position", button_str)
            }))
        }
        "mouse_down" => {
            let button_str = params.get("button")
                .and_then(|v| v.as_str())
                .unwrap_or("left");

            let button = match button_str {
                "right" => crate::MouseButton::Right,
                "middle" => crate::MouseButton::Middle,
                _ => crate::MouseButton::Left,
            };

            crate::actions::mouse_down(button)?;
            Ok(json!({
                "success": true,
                "message": format!("Mouse button {} pressed down", button_str)
            }))
        }
        "mouse_up" => {
            let button_str = params.get("button")
                .and_then(|v| v.as_str())
                .unwrap_or("left");

            let button = match button_str {
                "right" => crate::MouseButton::Right,
                "middle" => crate::MouseButton::Middle,
                _ => crate::MouseButton::Left,
            };

            crate::actions::mouse_up(button)?;
            Ok(json!({
                "success": true,
                "message": format!("Mouse button {} released", button_str)
            }))
        }
        "scroll" => {
            let dx = params.get("dx").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
            let dy = params.get("dy").and_then(|v| v.as_i64()).unwrap_or(0) as i32;

            crate::actions::scroll(dx, dy)?;
            Ok(json!({
                "success": true,
                "message": format!("Scrolled dx={}, dy={}", dx, dy)
            }))
        }
        "key_down" => {
            let key = params.get("key")
                .and_then(|v| v.as_str())
                .ok_or_else(|| crate::Error::ScreenshotFailed("Missing 'key' parameter".into()))?;

            crate::actions::key_down(key)?;
            Ok(json!({
                "success": true,
                "message": format!("Key '{}' pressed down", key)
            }))
        }
        "key_up" => {
            let key = params.get("key")
                .and_then(|v| v.as_str())
                .ok_or_else(|| crate::Error::ScreenshotFailed("Missing 'key' parameter".into()))?;

            crate::actions::key_up(key)?;
            Ok(json!({
                "success": true,
                "message": format!("Key '{}' released", key)
            }))
        }
        "find_cursor" => {
            let cursor = ctl.find_cursor()?;
            // Return normalized 0-1000 coordinates relative to viewport
            let norm_res = ctl.to_normalized_coords(cursor.x, cursor.y)?;

            if let Some((norm_x, norm_y)) = norm_res {
                Ok(json!({
                    "success": true,
                    "x": norm_x,
                    "y": norm_y,
                    "pixel_x": cursor.x,
                    "pixel_y": cursor.y,
                    "in_viewport": true
                }))
            } else {
                 Ok(json!({
                    "success": true, 
                    // We don't return x,y if outside viewport
                    "pixel_x": cursor.x,
                    "pixel_y": cursor.y,
                    "in_viewport": false,
                    "message": "Cursor is outside the active viewport"
                }))
            }
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
        "list_icons" | "list_templates" => {
            use crate::detection;
            let templates = detection::list_available_templates()?;
            Ok(json!({
                "success": true,
                "icons": templates,
                "count": templates.len()
            }))
        }
        "move_to_icon" | "find_template" => {
            use crate::detection;
            use crate::actions::move_to_pixel;

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
                    "message": format!("Icon '{}' not found on screen (threshold: {})", name, threshold)
                }))
            } else {
                let best = &detections[0];
                move_to_pixel(best.x, best.y, true)?;
                Ok(json!({
                    "success": true,
                    "found": true,
                    "x": best.x,
                    "y": best.y,
                    "confidence": best.confidence,
                    "num_matches": detections.len(),
                    "message": format!("Moved cursor to icon '{}' ({} match(es))", name, detections.len())
                }))
            }
        }
        "point_at" => {
            let description = params.get("description")
                .and_then(|v| v.as_str())
                .ok_or_else(|| crate::Error::ScreenshotFailed("Missing 'description' parameter".into()))?;
            
            let _model_override = params.get("model").and_then(|v| v.as_str());

            // Check for high resolution which might cause pointing inaccuracies
            let (width, height) = if let Some(region) = ctl.get_viewport() {
                (region.width, region.height)
            } else if let Ok(dims) = crate::primitives::get_screen_dimensions() {
                (dims.width, dims.height)
            } else {
                (0, 0)
            };

            if width > 1920 || height > 1080 {
                tracing::warn!(
                    width = width,
                    height = height,
                    "High resolution detected. Pointing model accuracy may be reduced due to downsampling. Small text may be illegible."
                );
            }

            // 1. Query the pointing model
            let prompt = format!(
                "You are a pointing specialist. Find the following element on screen: {}. \
                Return your response in JSON format: {{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"object_label\"}}. \
                Coordinates should be normalized 0-1000 where 0 is top/left and 1000 is bottom/right. \
                Only return the JSON object, no other text.", 
                description
            );

            let response = ctl.ask_pointing(&prompt)?;
            
            // 3. Parse coordinates (simple greedy [x, y] extraction)
            let coords = parse_coordinates(&response);
            
            match coords {
                Some((x, y)) => {
                    // x, y are normalized within the viewport (0-1000)
                    // VisionCtl::move_to handles translation to pixels using the viewport
                    ctl.move_to(x, y, true)?;
                    Ok(json!({
                        "success": true,
                        "x": x,
                        "y": y,
                        "message": format!("Pointed at '{}' at ({}, {})", description, x, y)
                    }))
                }
                None => {
                    Ok(json!({
                        "success": false,
                        "message": format!("Could not parse coordinates from model response: {}", response)
                    }))
                }
            }
        }
        "ask_screen" => {
            let question = params.get("question")
                .and_then(|v| v.as_str())
                .ok_or_else(|| crate::Error::ScreenshotFailed("Missing 'question' parameter".into()))?;

            let answer = ctl.ask(question)?;
            Ok(json!({
                "success": true,
                "question": question,
                "answer": answer,
                "message": format!("Answer: {}", answer)
            }))
        }
        "stuck" => {
            Ok(json!({
                "success": true,
                "message": "Stuck signal received"
            }))
        }
        _ => {
            Err(crate::Error::ScreenshotFailed(format!("Unknown tool: {}", name)))
        }
    };
    
    // Log the result
    match &result {
        Ok(res) => {
            if let Some(success) = res.get("success").and_then(|v| v.as_bool()) {
                if success {
                    tracing::info!(
                        tool = %name,
                        result = %serde_json::to_string(res).unwrap_or_else(|_| "serialization failed".to_string()),
                        "Tool executed successfully"
                    );
                } else {
                    tracing::warn!(
                        tool = %name,
                        result = %serde_json::to_string(res).unwrap_or_else(|_| "serialization failed".to_string()),
                        "Tool executed but failed"
                    );
                }
            } else {
                tracing::info!(
                    tool = %name,
                    result = %serde_json::to_string(res).unwrap_or_else(|_| "serialization failed".to_string()),
                    "Tool executed (no success field)"
                );
            }
        }
        Err(e) => {
            tracing::error!(
                tool = %name,
                error = %e,
                "Tool execution failed"
            );
        }
    }
    
    result
}

/// Helper to parse normalized coordinates from LLM output.
/// Supports Qwen3 JSON format {"bbox_2d": [x1, y1, x2, y2]} as well as simple [x, y].
pub fn parse_coordinates(s: &str) -> Option<(i32, i32)> {
    // 1. Try to parse as JSON first (Qwen3 style)
    if let Ok(val) = serde_json::from_str::<Value>(s) {
        if let Some(bbox) = val.get("bbox_2d").and_then(|v| v.as_array()) {
            if bbox.len() == 4 {
                let x1 = bbox[0].as_i64()? as i32;
                let y1 = bbox[1].as_i64()? as i32;
                let x2 = bbox[2].as_i64()? as i32;
                let y2 = bbox[3].as_i64()? as i32;
                return Some(((x1 + x2) / 2, (y1 + y2) / 2));
            }
        }
    }

    // 2. Fallback to regex for more robust parsing of messy JSON or plain lists
    let re_bbox = regex::Regex::new(r#""bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]"#).ok()?;
    if let Some(caps) = re_bbox.captures(s) {
        let x1 = caps.get(1)?.as_str().parse::<i32>().ok()?;
        let y1 = caps.get(2)?.as_str().parse::<i32>().ok()?;
        let x2 = caps.get(3)?.as_str().parse::<i32>().ok()?;
        let y2 = caps.get(4)?.as_str().parse::<i32>().ok()?;
        return Some(((x1 + x2) / 2, (y1 + y2) / 2));
    }

    // 3. Simple [x, y] or (x, y) fallback
    let re_simple = regex::Regex::new(r"[\[\(]\s*(\d+)\s*,\s*(\d+)\s*[\]\)]").ok()?;
    if let Some(caps) = re_simple.captures(s) {
        let x = caps.get(1)?.as_str().parse::<i32>().ok()?;
        let y = caps.get(2)?.as_str().parse::<i32>().ok()?;
        return Some((x, y));
    }

    None
}
