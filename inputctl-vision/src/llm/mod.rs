//! LLM integration for vision-based reasoning and tool calling.
//!
//! This module provides the [`LlmClient`] for interacting with various LLM backends
//! (Ollama, vLLM, OpenAI) and the infrastructure for defining and executing tools
//! that the LLM can call.

pub mod client;
pub mod tools;

pub use client::{LlmClient, LlmConfig, Message};
pub use tools::{
    execute_tool, get_action_tools, get_tool_definitions, parse_coordinates, ToolDefinition,
};
