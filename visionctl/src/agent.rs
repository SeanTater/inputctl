use crate::error::Result;
use crate::llm::{LlmClient, LlmConfig, Message, get_plan_tool, get_action_tools, execute_tool};
use crate::VisionCtl;
use crate::primitives::{ScreenshotOptions, capture_screenshot, pixel_to_grid, grid_to_pixel};
use serde_json::json;
use std::fs;
use std::time::Duration;
use tracing::{info, debug, warn, error, trace};

const PLAN_PROMPT: &str = r#"You are a GUI automation agent controlling a Linux desktop.

Look at the screenshot and create a plan to accomplish the goal. The screenshot has a grid overlay:
- Column letters (A, B, C...) are shown at top and bottom edges
- Row numbers (1, 2, 3...) are shown at left and right edges
- Each cell is 200x200 pixels, referenced as "A1", "B3", etc.

Analyze the screen and call the plan tool with what you want to do in later turns."#;

const ACTION_PROMPT: &str = r#"You are a GUI automation agent controlling a Linux desktop.

SCREEN INFORMATION:
- Screenshot shows current desktop state
- Red crosshair (+) marks current cursor position
- Grid overlay shows positioning system (if visible)

AVAILABLE ACTIONS:
- plan(text): Create or update a plan for the task. Run this first, and whenever something goes wrong.
- move_direction(direction, pixels): Move cursor in direction (up, down, left, right) by pixel amount
- move_to_cell(cell, offset?): Move cursor to grid cell (with optional pixel offset)
- click(button): Click at current cursor position
- type_text(text): Type text at focus
- key_press(key): Press special keys (enter, escape, tab, etc.)
- move_to_icon(name, threshold?): Find an icon and move the cursor to it. Not all icons are available.
- list_icons(): See available icons
- task_complete(success, message): Signal FINAL task is done (not intermediate)
- stuck(): Indicate you are stuck and need help

PITFALLS:
- Do not repeat the same action multiple times in a row
- Always verify that an action succeeded by checking the screen state
- If something unexpected happens, create a new plan before taking further actions
Use the tools to accomplish the goal step by step."#;

/// Agent configuration
pub struct AgentConfig {
    pub max_iterations: usize,
    pub verbose: bool,
    pub save_screenshots: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            verbose: true,
            save_screenshots: true,
        }
    }
}

/// Result of running the agent
#[derive(Debug)]
pub struct AgentResult {
    pub success: bool,
    pub message: String,
    pub iterations: usize,
    pub actions_taken: Vec<String>,
}

/// Autonomous GUI agent
pub struct Agent {
    ctl: VisionCtl,
    llm_client: LlmClient,
    config: AgentConfig,
}

impl Agent {
    /// Create a new agent with LLM configuration
    pub fn new(llm_config: LlmConfig) -> Result<Self> {
        let ctl = VisionCtl::new(llm_config.clone())?;
        let llm_client = LlmClient::new(llm_config)?;

        Ok(Self {
            ctl,
            llm_client,
            config: AgentConfig::default(),
        })
    }

    /// Set maximum iterations
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.config.max_iterations = max;
        self
    }

    /// Set verbose mode
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
        self
    }

    /// Take screenshot with cursor marked (and grid)
    fn screenshot_with_cursor(&self) -> Result<Vec<u8>> {
        let options = ScreenshotOptions {
            grid: Some(self.ctl.grid_config.clone()),
            mark_cursor: true,
        };
        let data = capture_screenshot(options)?;
        Ok(data.png_bytes)
    }

    /// Run the agent to accomplish a goal
    pub fn run(&self, goal: &str) -> Result<AgentResult> {
        let mut actions_taken = Vec::new();
        let mut plan_target: Option<String> = None;
        let mut scene_notes: Option<String> = None;
        let mut require_replan = false;

        // Phase 1: Planning
        info!("Phase 1: Planning");
        debug!(goal = %goal, "Starting agent");

        let screenshot = self.screenshot_with_cursor()?;
        if self.config.save_screenshots {
            let path = "/tmp/visionctl_agent_plan.png";
            let _ = fs::write(path, &screenshot);
            debug!(path = %path, "Saved planning screenshot");
        }

        let plan_messages = vec![
            Message {
                role: "system".to_string(),
                content: Some(PLAN_PROMPT.to_string()),
                images: None,
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: "user".to_string(),
                content: Some(format!(
                    "Goal: {}\nStatus: {}",
                    goal,
                    self.cursor_status()?
                )),
                images: None,
                tool_calls: None,
                tool_call_id: None,
            },
        ];

        let plan_tools = get_plan_tool();
        debug!("Querying LLM for plan");
        let plan_response = self.llm_client.chat_with_tools(&plan_messages, &plan_tools, Some(&screenshot))?;

        if let Some(tool_calls) = &plan_response.tool_calls {
            if let Some(call) = tool_calls.first() {
                if call.function.name == "plan" {
                    let text = call.function.arguments.get("text")
                        .and_then(|v| v.as_str()).unwrap_or("");

                    info!(text = %text, "Plan created");

                    plan_target = Some(text.to_string());
                    actions_taken.push(format!("plan(text={})", text));
                }
            }
        } else if let Some(content) = &plan_response.content {
            warn!(content = %content, "LLM responded without tool call");
        }

        // Phase 2: Execution
        info!("Phase 2: Execution");
        let mut messages: Vec<Message> = vec![
            Message {
                role: "system".to_string(),
                content: Some(ACTION_PROMPT.to_string()),
                images: None,
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: "user".to_string(),
                content: Some(format!(
                    "Goal: {}\nPlan target: {}\nStatus: {}",
                    goal,
                    plan_target.as_deref().unwrap_or("unknown"),
                    self.cursor_status()?
                )),
                images: None,
                tool_calls: None,
                tool_call_id: None,
            },
        ];

        let action_tools = get_action_tools();

        for iteration in 0..self.config.max_iterations {
            debug!(iteration = iteration + 1, "Starting iteration");

            let screenshot = self.screenshot_with_cursor()?;

            if self.config.save_screenshots {
                let path = format!("/tmp/visionctl_agent_{}.png", iteration);
                let _ = fs::write(&path, &screenshot);
                trace!(path = %path, "Saved iteration screenshot");
            }

            debug!(iteration = iteration + 1, messages = messages.len(), "Querying LLM");

            // Log the last few messages for debugging
            for (i, msg) in messages.iter().rev().take(4).collect::<Vec<_>>().into_iter().rev().enumerate() {
                trace!(
                    msg_idx = messages.len() - 4 + i,
                    role = %msg.role,
                    content = ?msg.content.as_ref().map(|s| if s.len() > 100 { format!("{}...", &s[..100]) } else { s.clone() }),
                    has_tool_calls = msg.tool_calls.is_some(),
                    tool_call_id = ?msg.tool_call_id,
                    "Message in history"
                );
            }

            // Only add a user status message if the last message wasn't a tool response
            // (OpenAI API requires assistant after tool, not user)
            let last_role = messages.last().map(|m| m.role.as_str());
            if last_role != Some("tool") {
                let status_message = if require_replan {
                    format!(
                        "Status: {}\nScene notes: {}\nPrevious tool failed. Replan with plan().",
                        self.cursor_status()?,
                        scene_notes.as_deref().unwrap_or("none")
                    )
                } else {
                    format!(
                        "Status: {}\nScene notes: {}",
                        self.cursor_status()?,
                        scene_notes.as_deref().unwrap_or("none")
                    )
                };
                messages.push(Message {
                    role: "user".to_string(),
                    content: Some(status_message),
                    images: None,
                    tool_calls: None,
                    tool_call_id: None,
                });
            }

            let tools = if require_replan {
                get_plan_tool()
            } else {
                action_tools.clone()
            };

            let response = self.llm_client.chat_with_tools(&messages, &tools, Some(&screenshot))?;

            if let Some(tool_calls) = &response.tool_calls {
                // Process only the FIRST tool call - don't batch execute
                // This ensures we see errors before executing subsequent tools
                if let Some(call) = tool_calls.first() {
                    let tool_name = &call.function.name;
                    let tool_args = &call.function.arguments;

                    info!(tool = %tool_name, args = %tool_args, "Executing tool");

                    if require_replan && tool_name != "plan" {
                        let result = json!({
                            "success": false,
                            "error": "Previous tool failed. Call plan() before any other action."
                        });

                        actions_taken.push(format!("{}({})", tool_name, tool_args));

                        messages.push(Message {
                            role: "assistant".to_string(),
                            content: response.content.clone(),
                            images: None,
                            tool_calls: Some(vec![call.clone()]),
                            tool_call_id: None,
                        });

                        messages.push(Message {
                            role: "tool".to_string(),
                            content: Some(result.to_string()),
                            images: None,
                            tool_calls: None,
                            tool_call_id: call.id.clone(),
                        });

                        std::thread::sleep(Duration::from_millis(250));
                        continue;
                    }

                    // Check for task_complete
                    if tool_name == "task_complete" {
                        let success = tool_args.get("success").and_then(|v| v.as_bool()).unwrap_or(false);
                        let message = tool_args.get("message").and_then(|v| v.as_str()).unwrap_or("").to_string();

                        if success {
                            info!(message = %message, "Task completed successfully");
                        } else {
                            warn!(message = %message, "Task failed");
                        }

                        return Ok(AgentResult {
                            success,
                            message,
                            iterations: iteration + 1,
                            actions_taken,
                        });
                    }

                    if tool_name == "plan" {
                        let observations = tool_args.get("observations")
                            .and_then(|v| v.as_str()).unwrap_or("");
                        let target = tool_args.get("target_cell")
                            .and_then(|v| v.as_str()).unwrap_or("");
                        let action = tool_args.get("action")
                            .and_then(|v| v.as_str()).unwrap_or("");

                        plan_target = Some(target.to_string());
                        actions_taken.push(format!("plan(target={})", target));
                        require_replan = false;

                        let result = json!({
                            "success": true,
                            "observations": observations,
                            "target_cell": target,
                            "action": action,
                            "message": "Plan recorded. Now execute the action."
                        });

                        messages.push(Message {
                            role: "assistant".to_string(),
                            content: response.content.clone(),
                            images: None,
                            tool_calls: Some(vec![call.clone()]),
                            tool_call_id: None,
                        });

                        messages.push(Message {
                            role: "tool".to_string(),
                            content: Some(result.to_string()),
                            images: None,
                            tool_calls: None,
                            tool_call_id: call.id.clone(),
                        });

                        std::thread::sleep(Duration::from_millis(250));
                        continue;
                    }

                    if tool_name == "scene_note" {
                        let text = tool_args.get("text").and_then(|v| v.as_str()).unwrap_or("").trim();
                        if !text.is_empty() {
                            scene_notes = Some(text.to_string());
                        }

                        let result = json!({
                            "success": true,
                            "message": "Scene note recorded",
                            "text": text
                        });

                        actions_taken.push(format!("{}({})", tool_name, tool_args));

                        messages.push(Message {
                            role: "assistant".to_string(),
                            content: response.content.clone(),
                            images: None,
                            tool_calls: Some(vec![call.clone()]),
                            tool_call_id: None,
                        });

                        messages.push(Message {
                            role: "tool".to_string(),
                            content: Some(result.to_string()),
                            images: None,
                            tool_calls: None,
                            tool_call_id: call.id.clone(),
                        });

                        std::thread::sleep(Duration::from_millis(250));
                        continue;
                    }

                    // Execute the tool
                    let (result, tool_failed) = match execute_tool(&self.ctl, tool_name, tool_args.clone()) {
                        Ok(mut r) => {
                            debug!(result = %r, "Tool succeeded");
                            // Include cursor status in result so LLM sees state after action
                            if let Ok(status) = self.cursor_status() {
                                if let Some(obj) = r.as_object_mut() {
                                    obj.insert("cursor_status".to_string(), json!(status));
                                }
                            }
                            (r, false)
                        }
                        Err(e) => {
                            error!(error = %e, tool = %tool_name, "Tool failed");
                            let mut err = json!({"success": false, "error": e.to_string()});
                            if let Ok(status) = self.cursor_status() {
                                if let Some(obj) = err.as_object_mut() {
                                    obj.insert("cursor_status".to_string(), json!(status));
                                }
                            }
                            (err, true)
                        }
                    };

                    actions_taken.push(format!("{}({})", tool_name, tool_args));
                    require_replan = tool_failed;

                    // Add to message history
                    messages.push(Message {
                        role: "assistant".to_string(),
                        content: response.content.clone(),
                        images: None,
                        tool_calls: Some(vec![call.clone()]),
                        tool_call_id: None,
                    });

                    // Include tool_call_id to link response to the original call
                    messages.push(Message {
                        role: "tool".to_string(),
                        content: Some(result.to_string()),
                        images: None,
                        tool_calls: None,
                        tool_call_id: call.id.clone(),
                    });

                    // If there were multiple tool calls but we only executed one,
                    // log that we're skipping the rest
                    if tool_calls.len() > 1 {
                        let skipped: Vec<_> = tool_calls.iter().skip(1).map(|c| &c.function.name).collect();
                        if tool_failed {
                            warn!(skipped = ?skipped, "Skipping remaining tools due to failure");
                        } else {
                            debug!(skipped = ?skipped, "Processing one tool at a time");
                        }
                    }

                    std::thread::sleep(Duration::from_millis(500));
                }
            } else if let Some(content) = &response.content {
                // LLM responded with text instead of a tool call - prompt it to use tools
                debug!(content = %content, "LLM text response (no tool call)");

                messages.push(Message {
                    role: "assistant".to_string(),
                    content: Some(content.clone()),
                    images: None,
                    tool_calls: None,
                    tool_call_id: None,
                });

                messages.push(Message {
                    role: "user".to_string(),
                    content: Some("Use the tools to take an action. What should we do next?".to_string()),
                    images: None,
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
        }

        warn!(max = self.config.max_iterations, "Max iterations reached");
        Ok(AgentResult {
            success: false,
            message: "Max iterations reached without completing the task".to_string(),
            iterations: self.config.max_iterations,
            actions_taken,
        })
    }

    fn cursor_status(&self) -> Result<String> {
        let cursor = self.ctl.find_cursor()?;
        let cell = pixel_to_grid(cursor.x, cursor.y, &self.ctl.grid_config);
        let (cell_x, cell_y) = grid_to_pixel(&cell, &self.ctl.grid_config)?;
        let offset_x = cursor.x - cell_x;
        let offset_y = cursor.y - cell_y;
        Ok(format!(
            "cursor=({}, {}), cell={}, offset=({}, {})",
            cursor.x, cursor.y, cell, offset_x, offset_y
        ))
    }

}
