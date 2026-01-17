use serde_json::json;
use visionctl::VisionCtl;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Creating headless controller...");
    let ctl = VisionCtl::new_headless();

    println!("2. Verifying tool definitions...");
    let tools = ctl.get_tool_definitions();
    let has_ask = tools.iter().any(|t| t.name == "ask_screen");
    println!("Has ask_screen tool: {}", has_ask);

    if !has_ask {
        return Err("ask_screen tool definition not found".into());
    }

    println!("3. Verifying execute_tool wiring (expecting error due to headless mode)...");
    let res = ctl.execute_tool("ask_screen", json!({"question": "What do you see?"}));

    match res {
        Ok(val) => {
            println!("Unexpected success: {:?}", val);
        }
        Err(e) => {
            println!("Got expected error: {}", e);
            let msg = e.to_string();
            if !msg.contains("No LLM configured") {
                println!("WARNING: Error message might be wrong, expected 'No LLM configured'");
            }
        }
    }

    println!("Tool verification complete.");
    Ok(())
}
