//! Agent state tracking and debugging infrastructure.
//!
//! This module provides a way to observe the internal state of an [`Agent`] during its execution.
//! It uses an observer pattern (via the [`AgentObserver`] trait) to report events such as
//! starting a new iteration, querying an LLM, or executing a tool.
//!
//! The [`StateStore`] provides a thread-safe implementation of [`AgentObserver`] that
//! keeps the full history of a run in memory and can notify subscribers (like a web server)
//! of changes in real-time.

use serde::{Serialize, Deserialize};
use std::sync::Mutex;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use tokio::sync::broadcast;
use crate::llm::{Message, ToolDefinition};
use crate::primitives::Region;

/// Represents the overall state of an agent session
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentState {
    pub goal: String,
    pub iterations: Vec<Iteration>,
    pub status: String,
    pub screen_width: u32,
    pub screen_height: u32,
}

/// A single step in the agent's execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Iteration {
    /// Zero-based index of the iteration
    pub index: usize,
    /// Raw context screenshot bytes (full screen if possible)
    pub screenshot: Option<Vec<u8>>,
    /// Base64 context screenshot for UI
    pub screenshot_b64: Option<String>,
    /// Raw screenshot sent to the model (might be cropped)
    pub model_screenshot: Option<Vec<u8>>,
    /// Base64 model screenshot for UI
    pub model_screenshot_b64: Option<String>,
    /// Message history at the start of this iteration.
    pub messages: Vec<Message>,
    /// Tools executed during this iteration.
    pub tool_calls: Vec<ToolCall>,
    /// The viewport active during this iteration.
    pub viewport: Option<Region>,
    /// When this iteration started.
    pub timestamp: std::time::SystemTime,
}

/// A tool call and its result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
    pub response: Option<serde_json::Value>,
}

/// Trait for observing agent state changes
pub trait AgentObserver: Send + Sync {
    fn on_run_start(&self, goal: &str);
    fn on_iteration_start(&self, iteration: &Iteration);
    fn on_llm_query(&self, messages: &[Message], tools: &[ToolDefinition]);
    fn on_tool_start(&self, name: &str, args: &serde_json::Value);
    fn on_tool_end(&self, name: &str, result: &serde_json::Value);
    fn on_viewport_change(&self, viewport: Option<Region>);
    fn on_task_complete(&self, success: bool, message: &str);

    /// New: Control methods
    fn wait_if_paused(&self) {}
    fn get_injected_messages(&self) -> Vec<Message> { Vec::new() }
}

/// Default observer that does nothing
pub struct NoopObserver;

impl AgentObserver for NoopObserver {
    fn on_run_start(&self, _goal: &str) {}
    fn on_iteration_start(&self, _iteration: &Iteration) {}
    fn on_llm_query(&self, _messages: &[Message], _tools: &[ToolDefinition]) {}
    fn on_tool_start(&self, _name: &str, _args: &serde_json::Value) {}
    fn on_tool_end(&self, _name: &str, _result: &serde_json::Value) {}
    fn on_viewport_change(&self, _viewport: Option<Region>) {}
    fn on_task_complete(&self, _success: bool, _message: &str) {}
}

/// Thread-safe in-memory store for agent state
pub struct StateStore {
    state: Mutex<AgentState>,
    tx: broadcast::Sender<AgentState>,
    paused: std::sync::atomic::AtomicBool,
    pause_cond: std::sync::Condvar,
    pause_mutex: std::sync::Mutex<()>, // Used by Condvar
    injected_messages: Mutex<Vec<Message>>,
}

impl StateStore {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(100);
        Self {
            state: Mutex::new(AgentState::default()),
            tx,
            paused: std::sync::atomic::AtomicBool::new(false),
            pause_cond: std::sync::Condvar::new(),
            pause_mutex: std::sync::Mutex::new(()),
            injected_messages: Mutex::new(Vec::new()),
        }
    }

    pub fn set_paused(&self, paused: bool) {
        self.paused.store(paused, std::sync::atomic::Ordering::SeqCst);
        if !paused {
            self.pause_cond.notify_all();
        }
        self.notify();
    }

    pub fn inject_message(&self, message: Message) {
        let mut msgs = self.injected_messages.lock().unwrap();
        msgs.push(message);
        self.notify();
    }

    pub fn get_state(&self) -> AgentState {
        self.state.lock().unwrap().clone()
    }

    pub fn subscribe(&self) -> broadcast::Receiver<AgentState> {
        self.tx.subscribe()
    }

    fn notify(&self) {
        let state = self.get_state();
        let _ = self.tx.send(state);
    }
}

impl AgentObserver for StateStore {
    fn on_run_start(&self, goal: &str) {
        let mut state = self.state.lock().unwrap();
        state.goal = goal.to_string();
        state.iterations.clear();
        state.status = "Running".to_string();
        if let Ok(dims) = crate::primitives::get_screen_dimensions() {
            state.screen_width = dims.width;
            state.screen_height = dims.height;
        }
        drop(state);
        self.notify();
    }

    fn on_iteration_start(&self, iteration: &Iteration) {
        let mut state = self.state.lock().unwrap();
        let mut iter = iteration.clone();
        if let Some(screenshot) = &iter.screenshot {
            iter.screenshot_b64 = Some(STANDARD.encode(screenshot));
            iter.screenshot = None;
        }
        if let Some(screenshot) = &iter.model_screenshot {
            iter.model_screenshot_b64 = Some(STANDARD.encode(screenshot));
            iter.model_screenshot = None;
        }
        state.iterations.push(iter);
        drop(state);
        self.notify();
    }

    fn on_llm_query(&self, messages: &[Message], _tools: &[ToolDefinition]) {
        let mut state = self.state.lock().unwrap();
        if let Some(iter) = state.iterations.last_mut() {
            iter.messages = messages.to_vec();
        }
        drop(state);
        self.notify();
    }

    fn on_tool_start(&self, name: &str, args: &serde_json::Value) {
        let mut state = self.state.lock().unwrap();
        if let Some(iter) = state.iterations.last_mut() {
            iter.tool_calls.push(ToolCall {
                name: name.to_string(),
                arguments: args.clone(),
                response: None,
            });
        }
        drop(state);
        self.notify();
    }

    fn on_tool_end(&self, _name: &str, result: &serde_json::Value) {
        let mut state = self.state.lock().unwrap();
        if let Some(iter) = state.iterations.last_mut() {
            if let Some(call) = iter.tool_calls.last_mut() {
                call.response = Some(result.clone());
            }
        }
        drop(state);
        self.notify();
    }

    fn on_viewport_change(&self, viewport: Option<Region>) {
        let mut state = self.state.lock().unwrap();
        if let Some(iter) = state.iterations.last_mut() {
            iter.viewport = viewport;
        }
        drop(state);
        self.notify();
    }

    fn on_task_complete(&self, success: bool, _message: &str) {
        let mut state = self.state.lock().unwrap();
        state.status = if success { "Success".to_string() } else { "Failed".to_string() };
        drop(state);
        self.notify();
    }

    fn wait_if_paused(&self) {
        let mut paused = self.pause_mutex.lock().unwrap();
        while self.paused.load(std::sync::atomic::Ordering::SeqCst) {
            paused = self.pause_cond.wait(paused).unwrap();
        }
    }

    fn get_injected_messages(&self) -> Vec<Message> {
        let mut msgs = self.injected_messages.lock().unwrap();
        std::mem::take(&mut *msgs)
    }
}
