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
        temperature: 0.0,
    };

    let ctl = VisionCtl::new(config)?;

    println!("Step 1: Taking screenshot...");
    let _screenshot = ctl.screenshot_with_cursor()?;
    println!("✓ Screenshot captured\n");

    println!("Step 2: Asking LLM to identify something on screen...");
    let answer = ctl.ask("What application windows are visible on the screen? List them briefly.")?;
    println!("LLM Response: {}\n", answer);

    println!("Step 3: Script decides next action based on LLM's response...");
    println!("(In a real scenario, you'd parse the LLM response and take action)\n");

    // Example: Ask LLM where a specific element is (using 0-1000 coordinates)
    println!("Step 4: Asking LLM where to click...");
    let location = ctl.ask(
        "Looking at the screen, where is a terminal or console window? \
        Respond with normalized coordinates (0-1000 scale where 0,0 is top-left \
        and 1000,1000 is bottom-right). Format: 'x,y' or say 'none' if you don't see one."
    )?;
    println!("LLM says the terminal is at: {}\n", location);

    // Script decides what to do based on LLM's answer
    if !location.to_lowercase().contains("none") {
        println!("Step 5: Script would execute action based on LLM guidance...");
        println!("(In production, you'd parse the coordinates and call ctl.move_to(x, y))");
    } else {
        println!("Step 5: No terminal found, skipping click");
    }

    println!("\n=== Example Complete ===");
    println!("\nKey Points:");
    println!("- Script maintains control flow");
    println!("- LLM used for perception/understanding");
    println!("- Script makes decisions based on LLM responses");
    println!("- Uses 0-1000 normalized coordinates");

    Ok(())
}
