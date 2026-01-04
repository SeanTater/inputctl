use crate::error::{Error, Result};
use crate::llm::tools::ToolDefinition;
use base64::{engine::general_purpose, Engine as _};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;

#[derive(Debug, Clone)]
pub enum LlmConfig {
    Ollama { url: String, model: String },
    Vllm { url: String, model: String, api_key: Option<String> },
    OpenAI { url: String, model: String, api_key: String },
}

/// Message in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// For tool response messages, links to the original tool call
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Tool call from the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub function: FunctionCall,
}

/// Function call details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: Value,
}

/// Response from chat_with_tools
#[derive(Debug, Clone)]
pub struct ChatResponse {
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

pub struct LlmClient {
    client: Client,
    config: LlmConfig,
}

impl LlmClient {
    pub fn new(config: LlmConfig) -> Result<Self> {
        // Use a very long timeout for slow CPU inference (30 minutes)
        // This is especially important for vision models on CPU
        let client = Client::builder()
            .timeout(Duration::from_secs(1800))
            .build()
            .map_err(|e| Error::LlmApiError(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            client,
            config,
        })
    }

    pub fn query(&self, image_bytes: &[u8], prompt: &str) -> Result<String> {
        let image_base64 = general_purpose::STANDARD.encode(image_bytes);

        match &self.config {
            LlmConfig::Ollama { url, model } => {
                self.query_ollama(url, model, &image_base64, prompt)
            }
            LlmConfig::Vllm { url, model, api_key } => {
                self.query_openai_compatible(url, model, api_key.as_deref(), &image_base64, prompt)
            }
            LlmConfig::OpenAI { url, model, api_key } => {
                self.query_openai_compatible(url, model, Some(api_key.as_str()), &image_base64, prompt)
            }
        }
    }

    /// Chat with tool calling support
    pub fn chat_with_tools(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        image: Option<&[u8]>,
    ) -> Result<ChatResponse> {
        match &self.config {
            LlmConfig::Ollama { url, model } => {
                self.chat_ollama_with_tools(url, model, messages, tools, image)
            }
            LlmConfig::Vllm { url, model, api_key } => {
                self.chat_openai_compatible_with_tools(url, model, api_key.as_deref(), messages, tools, image)
            }
            LlmConfig::OpenAI { url, model, api_key } => {
                self.chat_openai_compatible_with_tools(url, model, Some(api_key.as_str()), messages, tools, image)
            }
        }
    }

    fn chat_ollama_with_tools(
        &self,
        url: &str,
        model: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        image: Option<&[u8]>,
    ) -> Result<ChatResponse> {
        let endpoint = format!("{}/api/chat", url.trim_end_matches('/'));

        // Convert tools to Ollama format
        let ollama_tools: Vec<Value> = tools.iter().map(|t| {
            json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema
                }
            })
        }).collect();

        // Build messages array
        let mut ollama_messages: Vec<Value> = messages.iter().map(|m| {
            let mut msg = json!({ "role": m.role });
            if let Some(content) = &m.content {
                msg["content"] = json!(content);
            }
            if let Some(images) = &m.images {
                msg["images"] = json!(images);
            }
            if let Some(tool_calls) = &m.tool_calls {
                // Include ID in tool calls for Ollama
                let tc_with_ids: Vec<Value> = tool_calls.iter().map(|tc| {
                    let mut obj = json!({
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    });
                    if let Some(id) = &tc.id {
                        obj["id"] = json!(id);
                    }
                    obj
                }).collect();
                msg["tool_calls"] = json!(tc_with_ids);
            }
            // For tool response messages, include the tool_call_id
            if let Some(tool_call_id) = &m.tool_call_id {
                msg["tool_call_id"] = json!(tool_call_id);
            }
            msg
        }).collect();

        // Add image to the last user message if provided
        if let Some(img_bytes) = image {
            let b64 = general_purpose::STANDARD.encode(img_bytes);
            // Find the last user message and add the image
            for msg in ollama_messages.iter_mut().rev() {
                if msg.get("role").and_then(|r| r.as_str()) == Some("user") {
                    msg["images"] = json!([b64]);
                    break;
                }
            }
        }

        let body = json!({
            "model": model,
            "messages": ollama_messages,
            "tools": ollama_tools,
            "stream": false
        });

        let response = self.client
            .post(&endpoint)
            .json(&body)
            .send()?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().unwrap_or_else(|_| "Unknown error".to_string());
            return Err(Error::LlmApiError(format!("Ollama chat API error {}: {}", status, error_text)));
        }

        let json: Value = response.json()?;

        // Parse the response
        let message = json.get("message")
            .ok_or_else(|| Error::LlmApiError("No message in Ollama response".to_string()))?;

        let content = message.get("content")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        // Parse tool calls if present
        let tool_calls = message.get("tool_calls")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter().enumerate().filter_map(|(i, tc)| {
                    let function = tc.get("function")?;
                    let name = function.get("name")?.as_str()?.to_string();
                    // Arguments can be a string or object - Ollama returns it as object
                    let arguments = function.get("arguments").cloned().unwrap_or(json!({}));
                    // Get ID from response or generate one
                    let id = tc.get("id")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                        .or_else(|| Some(format!("call_{}", i)));
                    Some(ToolCall {
                        id,
                        function: FunctionCall { name, arguments }
                    })
                }).collect()
            });

        Ok(ChatResponse { content, tool_calls })
    }

    fn chat_openai_compatible_with_tools(
        &self,
        url: &str,
        model: &str,
        api_key: Option<&str>,
        messages: &[Message],
        tools: &[ToolDefinition],
        image: Option<&[u8]>,
    ) -> Result<ChatResponse> {
        let endpoint = format!("{}/v1/chat/completions", url.trim_end_matches('/'));

        // Convert tools to OpenAI format
        let openai_tools: Vec<Value> = tools.iter().map(|t| {
            json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema
                }
            })
        }).collect();

        // Build messages array - OpenAI format uses content array with text and image_url
        let mut openai_messages: Vec<Value> = Vec::new();

        for (i, m) in messages.iter().enumerate() {
            let is_last_user_msg = m.role == "user" &&
                messages.iter().skip(i + 1).all(|msg| msg.role != "user");

            // If this is the last user message and we have an image, use multimodal content
            if m.role == "user" && is_last_user_msg && image.is_some() {
                let mut content_parts = Vec::new();

                // Add text content
                if let Some(text) = &m.content {
                    content_parts.push(json!({
                        "type": "text",
                        "text": text
                    }));
                }

                // Add image
                if let Some(img_bytes) = image {
                    let b64 = general_purpose::STANDARD.encode(img_bytes);
                    content_parts.push(json!({
                        "type": "image_url",
                        "image_url": {
                            "url": format!("data:image/png;base64,{}", b64)
                        }
                    }));
                }

                openai_messages.push(json!({
                    "role": m.role,
                    "content": content_parts
                }));
            } else {
                // Regular message format
                let mut msg = json!({ "role": m.role });
                if let Some(content) = &m.content {
                    msg["content"] = json!(content);
                }
                if let Some(tool_calls) = &m.tool_calls {
                    // Convert tool calls to OpenAI format with IDs
                    let tc_array: Vec<Value> = tool_calls.iter().enumerate().map(|(idx, tc)| {
                        json!({
                            "id": tc.id.clone().unwrap_or_else(|| format!("call_{}", idx)),
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": serde_json::to_string(&tc.function.arguments).unwrap_or_else(|_| "{}".to_string())
                            }
                        })
                    }).collect();
                    msg["tool_calls"] = json!(tc_array);
                }
                // For tool response messages, include tool_call_id
                if let Some(tool_call_id) = &m.tool_call_id {
                    msg["tool_call_id"] = json!(tool_call_id);
                }
                openai_messages.push(msg);
            }
        }

        let body = json!({
            "model": model,
            "messages": openai_messages,
            "tools": openai_tools,
        });

        let mut request = self.client.post(&endpoint).json(&body);

        if let Some(key) = api_key {
            request = request.header("Authorization", format!("Bearer {}", key));
        }

        let response = request.send()?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().unwrap_or_else(|_| "Unknown error".to_string());
            return Err(Error::LlmApiError(format!("OpenAI chat API error {}: {}", status, error_text)));
        }

        let json: Value = response.json()?;

        // Parse the response - OpenAI format nests message under choices[0]
        let message = json.get("choices")
            .and_then(|v| v.get(0))
            .and_then(|v| v.get("message"))
            .ok_or_else(|| Error::LlmApiError("No message in OpenAI response".to_string()))?;

        let content = message.get("content")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string());

        // Parse tool calls if present
        let tool_calls = message.get("tool_calls")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter().enumerate().filter_map(|(i, tc)| {
                    let function = tc.get("function")?;
                    let name = function.get("name")?.as_str()?.to_string();
                    // OpenAI returns arguments as a JSON string, parse it
                    let arguments = function.get("arguments")
                        .and_then(|v| v.as_str())
                        .and_then(|s| serde_json::from_str(s).ok())
                        .unwrap_or(json!({}));
                    // Get ID from response or generate one
                    let id = tc.get("id")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                        .or_else(|| Some(format!("call_{}", i)));
                    Some(ToolCall {
                        id,
                        function: FunctionCall { name, arguments }
                    })
                }).collect()
            });

        Ok(ChatResponse { content, tool_calls })
    }

    fn query_ollama(&self, url: &str, model: &str, image: &str, prompt: &str) -> Result<String> {
        let endpoint = format!("{}/api/generate", url.trim_end_matches('/'));

        let body = json!({
            "model": model,
            "prompt": prompt,
            "images": [image],
            "stream": false
        });

        let response = self.client
            .post(&endpoint)
            .json(&body)
            .send()?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().unwrap_or_else(|_| "Unknown error".to_string());
            return Err(Error::LlmApiError(format!("Ollama API error {}: {}", status, error_text)));
        }

        let json: Value = response.json()?;

        json.get("response")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| Error::LlmApiError("No response field in Ollama response".to_string()))
    }

    fn query_openai_compatible(
        &self,
        url: &str,
        model: &str,
        api_key: Option<&str>,
        image: &str,
        prompt: &str,
    ) -> Result<String> {
        let endpoint = format!("{}/v1/chat/completions", url.trim_end_matches('/'));

        let body = json!({
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": format!("data:image/png;base64,{}", image)
                            }
                        }
                    ]
                }
            ]
        });

        let mut request = self.client.post(&endpoint).json(&body);

        if let Some(key) = api_key {
            request = request.header("Authorization", format!("Bearer {}", key));
        }

        let response = request.send()?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().unwrap_or_else(|_| "Unknown error".to_string());
            return Err(Error::LlmApiError(format!("OpenAI API error {}: {}", status, error_text)));
        }

        let json: Value = response.json()?;

        json.get("choices")
            .and_then(|v| v.get(0))
            .and_then(|v| v.get("message"))
            .and_then(|v| v.get("content"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| Error::LlmApiError("Invalid OpenAI response format".to_string()))
    }
}
