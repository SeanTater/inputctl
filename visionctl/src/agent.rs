use crate::error::Result;
use crate::llm::{LlmClient, LlmConfig, Message, get_plan_tool, get_action_tools, execute_tool};
use crate::VisionCtl;
use crate::primitives::{ScreenshotOptions, capture_screenshot};
use serde_json::json;
use std::fs;
use std::time::Duration;

const PLAN_PROMPT: &str = r#"You are a GUI automation agent controlling a Linux desktop.

Look at the screenshot and create a plan to accomplish the goal. The screenshot has a grid overlay:
- Column letters (A, B, C...) are shown at top and bottom edges
- Row numbers (1, 2, 3...) are shown at left and right edges
- Each cell is 50x50 pixels, referenced as "A1", "B3", etc.

Analyze the screen and call the plan tool with:
1. What you see (relevant windows, buttons, UI elements)
2. Where the target element is located (specific grid cell)
3. What action will accomplish the goal"#;

const ACTION_PROMPT: &str = r#"You are a GUI automation agent controlling a Linux desktop.

SCREEN INFORMATION:
- Screenshot shows current desktop state
- Red crosshair (+) marks current cursor position
- Grid overlay shows positioning system (if visible)

POSITIONING METHODS:
1. Grid system (if overlay visible):
   - Column letters: A, B, C, ...
   - Row numbers: 1, 2, 3, ...
   - Cell reference: "A1", "B3", "C5", etc.
   - Each cell is 50x50 pixels
   - Best for: Large UI elements, general positioning

2. Template matching (recommended for icons):
   - Use list_templates() to see available icon templates
   - Use click_template(name) to click precise icons
   - Best for: Small icons, taskbar items, toolbar buttons
   - More accurate than grid for small targets

AVAILABLE ACTIONS:
- move_to(cell): Move cursor to grid cell
- click(button): Click at current cursor position
- type_text(text): Type text at focus
- key_press(key): Press special keys (enter, escape, tab, etc.)
- click_template(name, button?): Find and click an icon template
- find_template(name, threshold?): Locate icon without clicking
- list_templates(): See available icon templates
- task_complete(success, message): Signal task done

STRATEGY:
1. Analyze the screenshot carefully
2. For small icons/buttons: Use click_template() with icon name
3. For large UI areas: Use grid system with move_to() + click()
4. Always verify action success in next screenshot
5. If stuck, try alternative approach or different targeting method

Call task_complete when goal is achieved or if impossible."#;

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

        // Phase 1: Planning
        if self.config.verbose {
            eprintln!("\n[Phase 1] Planning...");
        }

        let screenshot = self.screenshot_with_cursor()?;
        if self.config.save_screenshots {
            let _ = fs::write("/tmp/visionctl_agent_plan.png", &screenshot);
        }

        let plan_messages = vec![
            Message {
                role: "system".to_string(),
                content: Some(PLAN_PROMPT.to_string()),
                images: None,
                tool_calls: None,
            },
            Message {
                role: "user".to_string(),
                content: Some(format!("Goal: {}", goal)),
                images: None,
                tool_calls: None,
            },
        ];

        let plan_tools = get_plan_tool();
        let plan_response = self.llm_client.chat_with_tools(&plan_messages, &plan_tools, Some(&screenshot))?;

        if let Some(tool_calls) = &plan_response.tool_calls {
            if let Some(call) = tool_calls.first() {
                if call.function.name == "plan" {
                    let observations = call.function.arguments.get("observations")
                        .and_then(|v| v.as_str()).unwrap_or("");
                    let target = call.function.arguments.get("target_cell")
                        .and_then(|v| v.as_str()).unwrap_or("");
                    let action = call.function.arguments.get("action")
                        .and_then(|v| v.as_str()).unwrap_or("");

                    if self.config.verbose {
                        eprintln!("[Plan] Observations: {}", observations);
                        eprintln!("[Plan] Target cell: {}", target);
                        eprintln!("[Plan] Action: {}", action);
                    }

                    plan_target = Some(target.to_string());
                    actions_taken.push(format!("plan(target={})", target));
                }
            }
        } else if let Some(content) = &plan_response.content {
            if self.config.verbose {
                eprintln!("[Plan] LLM response (no tool call): {}", content);
            }
        }

        // Phase 2: Execution
        let mut messages: Vec<Message> = vec![
            Message {
                role: "system".to_string(),
                content: Some(ACTION_PROMPT.to_string()),
                images: None,
                tool_calls: None,
            },
            Message {
                role: "user".to_string(),
                content: Some(format!(
                    "Goal: {}\nPlan target: {}",
                    goal,
                    plan_target.as_deref().unwrap_or("unknown")
                )),
                images: None,
                tool_calls: None,
            },
        ];

        let action_tools = get_action_tools();

        for iteration in 0..self.config.max_iterations {
            if self.config.verbose {
                eprintln!("\n[Iteration {}] Taking screenshot...", iteration + 1);
            }

            let screenshot = self.screenshot_with_cursor()?;

            if self.config.save_screenshots {
                let path = format!("/tmp/visionctl_agent_{}.png", iteration);
                let _ = fs::write(&path, &screenshot);
                if self.config.verbose {
                    eprintln!("[Iteration {}] Screenshot saved to {}", iteration + 1, path);
                }
            }

            if self.config.verbose {
                eprintln!("[Iteration {}] Querying LLM...", iteration + 1);
            }

            let response = self.llm_client.chat_with_tools(&messages, &action_tools, Some(&screenshot))?;

            if let Some(tool_calls) = &response.tool_calls {
                for call in tool_calls {
                    let tool_name = &call.function.name;
                    let tool_args = &call.function.arguments;

                    if self.config.verbose {
                        eprintln!("[Iteration {}] Tool call: {}({})", iteration + 1, tool_name, tool_args);
                    }

                    // Check for task_complete
                    if tool_name == "task_complete" {
                        let success = tool_args.get("success").and_then(|v| v.as_bool()).unwrap_or(false);
                        let message = tool_args.get("message").and_then(|v| v.as_str()).unwrap_or("").to_string();

                        return Ok(AgentResult {
                            success,
                            message,
                            iterations: iteration + 1,
                            actions_taken,
                        });
                    }

                    // Execute the tool
                    let result = match execute_tool(&self.ctl, tool_name, tool_args.clone()) {
                        Ok(r) => r,
                        Err(e) => {
                            if self.config.verbose {
                                eprintln!("[Iteration {}] Tool error: {}", iteration + 1, e);
                            }
                            json!({"success": false, "error": e.to_string()})
                        }
                    };

                    actions_taken.push(format!("{}({})", tool_name, tool_args));

                    // Add to message history
                    messages.push(Message {
                        role: "assistant".to_string(),
                        content: response.content.clone(),
                        images: None,
                        tool_calls: Some(vec![call.clone()]),
                    });

                    messages.push(Message {
                        role: "tool".to_string(),
                        content: Some(result.to_string()),
                        images: None,
                        tool_calls: None,
                    });

                    std::thread::sleep(Duration::from_millis(500));
                }
            } else if let Some(content) = &response.content {
                if self.config.verbose {
                    eprintln!("[Iteration {}] LLM says: {}", iteration + 1, content);
                }

                // Try to parse task_complete from text
                if let Some((tool_name, tool_args)) = parse_tool_from_text(content) {
                    if tool_name == "task_complete" {
                        let success = tool_args.get("success").and_then(|v| v.as_bool()).unwrap_or(false);
                        let message = tool_args.get("message").and_then(|v| v.as_str()).unwrap_or("").to_string();

                        return Ok(AgentResult {
                            success,
                            message,
                            iterations: iteration + 1,
                            actions_taken,
                        });
                    }
                }

                messages.push(Message {
                    role: "assistant".to_string(),
                    content: Some(content.clone()),
                    images: None,
                    tool_calls: None,
                });

                messages.push(Message {
                    role: "user".to_string(),
                    content: Some("Use the tools to take an action. What should we do next?".to_string()),
                    images: None,
                    tool_calls: None,
                });
            }
        }

        Ok(AgentResult {
            success: false,
            message: "Max iterations reached without completing the task".to_string(),
            iterations: self.config.max_iterations,
            actions_taken,
        })
    }

}

/// Parse tool calls from text output (fallback for models that output tool syntax as text)
fn parse_tool_from_text(text: &str) -> Option<(String, serde_json::Value)> {
    // Look for patterns like:
    // - task_complete(success=true, message="...")
    // - task_complete({"success": true, "message": "..."})
    // - task_complete(true, "message")

    let text = text.trim();

    // Try to find task_complete specifically
    if let Some(start) = text.find("task_complete(") {
        let after_paren = &text[start + 14..]; // len("task_complete(") = 14

        // Find matching closing paren
        if let Some(end) = after_paren.rfind(')') {
            let args_str = &after_paren[..end].trim();

            // Try to parse as JSON first
            if args_str.starts_with('{') {
                if let Ok(args) = serde_json::from_str(args_str) {
                    return Some(("task_complete".to_string(), args));
                }
            }

            // Try to parse positional args: task_complete(true, "message")
            let parts: Vec<&str> = args_str.splitn(2, ',').collect();
            if parts.len() >= 1 {
                let first = parts[0].trim();
                let success = first == "true" || first == "success=true";
                let message = if parts.len() >= 2 {
                    parts[1].trim().trim_matches('"').trim_matches('\'').to_string()
                } else {
                    String::new()
                };

                // Also check for keyword style in first arg
                if first.contains('=') && !first.starts_with("success") {
                    // Try keyword parsing
                    let mut kw_success = false;
                    let mut kw_message = String::new();

                    for part in args_str.split(',') {
                        let part = part.trim();
                        if let Some(eq_pos) = part.find('=') {
                            let key = part[..eq_pos].trim();
                            let value = part[eq_pos + 1..].trim().trim_matches('"').trim_matches('\'');

                            match key {
                                "success" => kw_success = value == "true",
                                "message" => kw_message = value.to_string(),
                                _ => {}
                            }
                        }
                    }

                    return Some(("task_complete".to_string(), serde_json::json!({
                        "success": kw_success,
                        "message": kw_message
                    })));
                }

                return Some(("task_complete".to_string(), serde_json::json!({
                    "success": success,
                    "message": message
                })));
            }
        }
    }

    None
}
