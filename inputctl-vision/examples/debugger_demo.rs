use inputctl_vision::agent::Agent;
use inputctl_vision::config::Config;
use inputctl_vision::debugger::StateStore;
use inputctl_vision::server::DebugServer;
use std::sync::Arc;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let config = Config::load();
    let llm_config = config.llm.to_llm_config()?;

    // 1. Create the state store
    let state_store = Arc::new(StateStore::new());

    // 2. Start the debugger server in the background
    let server_store = state_store.clone();
    tokio::spawn(async move {
        let server = DebugServer::new(server_store);
        if let Err(e) = server.run(10888).await {
            eprintln!("Debugger server failed: {}", e);
        }
    });

    // 3. Create the agent and attach the state store as an observer
    let agent = Agent::new(llm_config)?.with_observer(state_store);

    println!("Debugger running at http://localhost:10888");
    println!("Starting agent task...");

    // 4. Run the agent
    let goal = "Open the calculator, type 123 + 456, and press enter. Then verify the result.";
    let result = agent.run(goal)?;

    println!(
        "Agent finished: success={}, message={}",
        result.success, result.message
    );

    // Keep server alive for a bit to inspect final state
    println!("Waiting 30s for final inspection...");
    tokio::time::sleep(tokio::time::Duration::from_secs(30)).await;

    Ok(())
}
