use inputctl_vision::{LlmConfig, VisionCtl};

fn main() -> inputctl_vision::Result<()> {
    // Configure for Ollama (default local setup)
    let config = LlmConfig::Ollama {
        url: "http://localhost:11434".to_string(),
        model: "llava".to_string(),
        temperature: 0.0,
        repetition_penalty: None,
    };

    // Create VisionCtl instance
    let ctl = VisionCtl::new(config)?;

    // Take a screenshot only
    println!("Taking screenshot...");
    let screenshot = VisionCtl::screenshot()?;
    println!("Screenshot captured: {} bytes", screenshot.len());

    // Ask a question about the screen
    println!("\nAsking LLM about the screen...");
    let answer = ctl.ask("What's on my screen?")?;
    println!("Answer: {}", answer);

    Ok(())
}
