//! Autonomous agent implementation for GUI automation.
//!
//! This module contains the [`Agent`] which orchestrates the interaction between
//! vision models (via [`VisionCtl`]), and tool execution to accomplish
//! specified goals on a Linux desktop.

use crate::debugger::{AgentObserver, Iteration, NoopObserver};
use crate::error::Result;
use crate::llm::{execute_tool, get_action_tools, LlmClient, LlmConfig, Message};
use inputctl_capture::{capture_screenshot, get_screen_dimensions, ScreenshotOptions};
use crate::VisionCtl;
use serde_json::json;
use std::fs;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, error, info, trace, warn};

const ACTION_PROMPT: &str = r#"You are a GUI automation agent controlling a Linux desktop.

SCREEN INFORMATION:
- Screenshot shows current desktop state (if provided)
- Red crosshair (+) marks current cursor position
- Coordinates use 0-1000 normalized system: (0,0)=top-left, (1000,1000)=bottom-right

Using the provided tools, verify the screen state and perform actions to accomplish the user's goal.

MULTIPLE TOOL CALLS:
- You can and SHOULD call multiple tools in a single turn if they make sense together.
- For example: move_to(x, y) + click(left) is common.
- Or: move_to(x, y) + mouse_down(left) + move_to(x2, y2) + mouse_up(left) for drag-and-drop.
- Tools are executed in the order you provide them.

PITFALLS:
- Do not repeat the same action multiple times in a row
- Always verify that an action succeeded using ask_screen() or by checking the screenshot
- Avoid confirmation bias when verifying (e.g. ask "What is the focused window?" not "Is the window focused?")
Use the tools to accomplish the goal step by step."#;

/// Agent configuration
pub struct AgentConfig {
    pub max_iterations: Option<usize>,
    pub verbose: bool,
    pub save_screenshots: bool,
    pub blind_mode: bool,
    pub no_vision_tools: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: None,
            verbose: true,
            save_screenshots: true,
            blind_mode: false,
            no_vision_tools: false,
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
    observer: Arc<dyn AgentObserver>,
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
            observer: Arc::new(NoopObserver),
        })
    }

    pub fn with_max_iterations(mut self, max: Option<usize>) -> Self {
        self.config.max_iterations = max;
        self
    }

    /// Get mutable access to the underlying VisionCtl
    pub fn ctl_mut(&mut self) -> &mut VisionCtl {
        &mut self.ctl
    }

    /// Set verbose mode
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
        self
    }

    /// Set blind mode (disable screenshots in main loop)
    pub fn with_blind_mode(mut self, blind: bool) -> Self {
        self.config.blind_mode = blind;
        self
    }

    /// Set no vision tools mode (keep screenshots but disable vision tools)
    pub fn with_no_vision_tools(mut self, no_vision: bool) -> Self {
        self.config.no_vision_tools = no_vision;
        self
    }

    /// Set a dedicated pointing model for the agent
    pub fn with_pointing_config(mut self, config: LlmConfig) -> Result<Self> {
        let pointing_client = LlmClient::new(config)?;
        self.ctl.pointing_client = Some(pointing_client);
        Ok(self)
    }

    /// Set an observer for the agent
    pub fn with_observer(mut self, observer: Arc<dyn AgentObserver>) -> Self {
        self.observer = observer;
        self
    }

    /// Take screenshot with cursor marked, respecting viewport
    fn screenshot_with_cursor(&self) -> Result<Vec<u8>> {
        let dims = crate::get_screen_dimensions()?;
        let options = ScreenshotOptions {
            mark_cursor: true,
            crop_region: self.ctl.get_viewport(),
            resize_to_logical: Some((dims.width, dims.height)),
        };
        let data = capture_screenshot(options)?;
        Ok(data.png_bytes)
    }

    /// Take full screen screenshot for context
    fn screenshot_full(&self) -> Result<Vec<u8>> {
        let dims = crate::get_screen_dimensions()?;
        let options = ScreenshotOptions {
            mark_cursor: true,
            crop_region: None,
            resize_to_logical: Some((dims.width, dims.height)),
        };
        let data = capture_screenshot(options)?;
        Ok(data.png_bytes)
    }

    /// Run the agent to accomplish a goal
    pub fn run(&self, goal: &str) -> Result<AgentResult> {
        let mut actions_taken = Vec::new();

        info!("Starting agent");
        debug!(goal = %goal, "Starting agent");
        self.observer.on_run_start(goal);

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
                content: Some(format!("Goal: {}\nStatus: {}", goal, self.cursor_status()?)),
                images: None,
                tool_calls: None,
                tool_call_id: None,
            },
        ];

        let mut action_tools = get_action_tools();
        if self.config.no_vision_tools {
            // Filter out tools that rely on explicit computer vision queries
            // (the model still sees the screenshot but can't use these helpers)
            action_tools.retain(|t| {
                ![
                    "point_at",
                    "move_to_icon",
                    "find_template",
                    "list_icons",
                    "list_templates",
                    "ask_screen",
                ]
                .contains(&t.name.as_str())
            });
        }

        let mut iteration = 0;
        loop {
            if let Some(max) = self.config.max_iterations {
                if iteration >= max {
                    break;
                }
            }
            self.observer.wait_if_paused();

            let injected = self.observer.get_injected_messages();
            for msg in injected {
                messages.push(msg);
            }

            debug!(iteration = iteration + 1, "Starting iteration");

            let model_screenshot = self.screenshot_with_cursor()?;
            let full_screenshot = if self.ctl.get_viewport().is_some() {
                self.screenshot_full().ok()
            } else {
                None
            };

            self.observer.on_iteration_start(&Iteration {
                index: iteration,
                screenshot: full_screenshot,
                screenshot_b64: None,
                model_screenshot: Some(model_screenshot.clone()),
                model_screenshot_b64: None,
                messages: messages.clone(),
                tool_calls: Vec::new(),
                viewport: self.ctl.get_viewport(),
                timestamp: std::time::SystemTime::now(),
            });

            if self.config.save_screenshots {
                let path = format!("/tmp/visionctl_agent_{}.png", iteration);
                let _ = fs::write(&path, &model_screenshot);
                trace!(path = %path, "Saved iteration screenshot");
            }

            debug!(
                iteration = iteration + 1,
                messages = messages.len(),
                "Querying LLM"
            );

            // Log the last few messages for debugging
            for (i, msg) in messages
                .iter()
                .rev()
                .take(4)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .enumerate()
            {
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
                let status_message = format!("Status: {}", self.cursor_status()?,);
                messages.push(Message {
                    role: "user".to_string(),
                    content: Some(status_message),
                    images: None,
                    tool_calls: None,
                    tool_call_id: None,
                });
            }

            let tools = action_tools.clone();

            let iter_screenshot = if self.config.blind_mode {
                None
            } else {
                Some(model_screenshot.as_slice())
            };

            self.observer.on_llm_query(&messages, &tools);
            let response = self.llm_client.chat_with_tools(
                &messages,
                &tools,
                iter_screenshot,
                Some("required".to_string()),
            )?;

            if let Some(tool_calls) = &response.tool_calls {
                // Add the assistant's message with tool calls to history FIRST
                messages.push(Message {
                    role: "assistant".to_string(),
                    content: response.content.clone(),
                    images: None,
                    tool_calls: Some(tool_calls.clone()),
                    tool_call_id: None,
                });

                let mut stop_execution = false;

                for call in tool_calls {
                    if stop_execution {
                        // If a previous tool failed, we must signal that we're skipping this one
                        messages.push(Message {
                            role: "tool".to_string(),
                            content: Some(
                                json!({
                                    "success": false,
                                    "error": "Skipped due to previously failed tool in sequence."
                                })
                                .to_string(),
                            ),
                            images: None,
                            tool_calls: None,
                            tool_call_id: call.id.clone(),
                        });
                        continue;
                    }

                    let tool_name = &call.function.name;
                    let tool_args = &call.function.arguments;

                    info!(tool = %tool_name, args = %tool_args, "Executing tool");
                    self.observer.on_tool_start(tool_name, tool_args);

                    // Check for task_complete
                    if tool_name == "task_complete" {
                        let success = tool_args
                            .get("success")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);
                        let message = tool_args
                            .get("message")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();

                        if success {
                            info!(message = %message, "Task completed successfully");
                        } else {
                            warn!(message = %message, "Task failed");
                        }

                        self.observer.on_task_complete(success, &message);

                        return Ok(AgentResult {
                            success,
                            message,
                            iterations: iteration + 1,
                            actions_taken,
                        });
                    }

                    // Check for stuck - agent is requesting help
                    if tool_name == "stuck" {
                        warn!("Agent signaled stuck");
                        self.observer
                            .on_task_complete(false, "Agent is stuck and needs help");
                        return Ok(AgentResult {
                            success: false,
                            message: "Agent is stuck and needs help".to_string(),
                            iterations: iteration + 1,
                            actions_taken,
                        });
                    }

                    // Execute the tool
                    let (result, tool_failed) =
                        match execute_tool(&self.ctl, tool_name, tool_args.clone()) {
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

                    self.observer.on_tool_end(tool_name, &result);

                    actions_taken.push(format!("{}({})", tool_name, tool_args));

                    if tool_failed {
                        stop_execution = true;
                    }

                    // Include tool_call_id to link response to the original call
                    messages.push(Message {
                        role: "tool".to_string(),
                        content: Some(result.to_string()),
                        images: None,
                        tool_calls: None,
                        tool_call_id: call.id.clone(),
                    });

                    // Small delay between tools in a sequence to allow UI to catch up
                    if !stop_execution {
                        std::thread::sleep(Duration::from_millis(150));
                    }
                }

                self.observer.on_llm_query(&messages, &tools);
                std::thread::sleep(Duration::from_millis(250));
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
                    content: Some(
                        "Use the tools to take an action. What should we do next?".to_string(),
                    ),
                    images: None,
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
            iteration += 1;
        }

        warn!(max = ?self.config.max_iterations, "Iteration limit reached");
        Ok(AgentResult {
            success: false,
            message: format!(
                "Iteration limit {:?} reached without completing the task",
                self.config.max_iterations
            ),
            iterations: iteration,
            actions_taken,
        })
    }

    fn cursor_status(&self) -> Result<String> {
        let cursor = self.ctl.find_cursor()?;
        let dims = get_screen_dimensions()?;
        let norm_x = cursor.x * 1000 / dims.width as i32;
        let norm_y = cursor.y * 1000 / dims.height as i32;
        Ok(format!("cursor=({}, {})", norm_x, norm_y))
    }
}
