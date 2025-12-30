use crate::error::{Error, Result};
use base64::{engine::general_purpose, Engine as _};
use reqwest::blocking::Client;
use serde_json::{json, Value};

#[derive(Debug, Clone)]
pub enum LlmConfig {
    Ollama { url: String, model: String },
    Vllm { url: String, model: String, api_key: Option<String> },
    OpenAI { url: String, model: String, api_key: String },
}

pub struct LlmClient {
    client: Client,
    config: LlmConfig,
}

impl LlmClient {
    pub fn new(config: LlmConfig) -> Result<Self> {
        Ok(Self {
            client: Client::new(),
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
