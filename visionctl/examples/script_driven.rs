/// Script-Driven Pattern Example
///
/// In this pattern, the script controls the flow and uses the LLM
/// occasionally for perception (understanding what's on screen).
///
/// Usage:
///   cargo run --example script_driven
///
/// Requirements:
///   - KDE Plasma 6.0+ with KWin
///   - Ollama running with llava model
///   - /dev/uinput access

use visionctl::{VisionCtl, LlmConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Script-Driven GUI Automation Example ===\n");

    // Configure LLM backend (Ollama with llava model)
    let config = LlmConfig::Ollama {
        url: "http://localhost:11434".to_string(),
        model: "llava".to_string(),
    };

    let ctl = VisionCtl::new(config)?;

    println!("Step 1: Taking screenshot with grid overlay...");
    let _screenshot = ctl.screenshot_with_grid()?;
    println!("✓ Screenshot captured with grid\n");

    println!("Step 2: Asking LLM to identify something on screen...");
    let answer = ctl.ask("What application windows are visible on the screen? List them briefly.")?;
    println!("LLM Response: {}\n", answer);

    println!("Step 3: Script decides next action based on LLM's response...");
    println!("(In a real scenario, you'd parse the LLM response and take action)\n");

    // Example: Ask LLM where a specific element is
    println!("Step 4: Asking LLM where to click...");
    let location = ctl.ask(
        "Looking at the grid overlay, which grid cell contains a terminal or console window? \
        Just respond with the cell identifier like 'B3' or say 'none' if you don't see one."
    )?;
    println!("LLM says the terminal is at: {}\n", location);

    // Script decides what to do based on LLM's answer
    if !location.to_lowercase().contains("none") {
        println!("Step 5: Script executing action based on LLM guidance...");
        println!("Would click at: {} (skipping actual click for safety)", location);
        // Uncomment to actually click:
        // ctl.click_at_grid(&location)?;
        // println!("✓ Clicked at {}", location);
    } else {
        println!("Step 5: No terminal found, skipping click");
    }

    println!("\n=== Example Complete ===");
    println!("\nKey Points:");
    println!("- Script maintains control flow");
    println!("- LLM used for perception/understanding");
    println!("- Script makes decisions based on LLM responses");
    println!("- Grid overlay helps LLM communicate locations");

    Ok(())
}
