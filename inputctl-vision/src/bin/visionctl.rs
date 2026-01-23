use clap::{Parser, Subcommand};
use dialoguer::{Input, Select};
use inputctl_vision::{Agent, Config, Encoder, LlmConfig, RecorderConfig, Region, VisionCtl};
use std::fs;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use tracing::{debug, error, info, warn};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "visionctl")]
#[command(about = "Vision-based GUI automation toolkit")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Increase verbosity (-v info, -vv debug, -vvv trace)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    /// Quiet mode (errors only)
    #[arg(short, long, global = true)]
    quiet: bool,

    /// Question to ask LLM about the screen (default mode)
    #[arg(trailing_var_arg = true)]
    question: Vec<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run autonomous agent to achieve a goal
    /// Run autonomous agent to achieve a goal
    Agent {
        /// Goal for the agent to accomplish
        goal: String,
        /// Limit agent to a specific window (by title keyword)
        #[arg(long)]
        window: Option<String>,
        /// Limit agent to a specific region (x,y,w,h)
        #[arg(long, value_parser = parse_region)]
        region: Option<Region>,
        /// Disable visual input/screenshots (blind mode)
        #[arg(long)]
        blind: bool,
        /// Disable vision tools (keep screenshots but remove point_at, ask_screen, etc.)
        #[arg(long)]
        no_vision_tools: bool,
        /// Enable web debugger
        #[arg(long)]
        debug: bool,
        /// Maximum iterations (0 = unlimited)
        #[arg(long, default_value = "0")]
        max_iterations: usize,
    },
    /// Window management commands
    Window {
        #[command(subcommand)]
        command: WindowCommands,
    },
    /// Capture screenshot to file
    Screenshot {
        /// Output path for screenshot
        #[arg(default_value = "/tmp/visionctl_screenshot.png")]
        output: String,
    },
    /// Find template image in screenshot
    FindTemplate {
        /// Template image to find
        template: String,
        /// Screenshot to search in (auto-captures if not provided)
        #[arg(short, long)]
        screenshot: Option<String>,
        /// Confidence threshold (0.0 to 1.0)
        #[arg(short, long, default_value = "0.8")]
        threshold: f32,
    },
    /// Find template and click on best match
    ClickTemplate {
        /// Template image to find
        template: String,
        /// Confidence threshold (0.0 to 1.0)
        #[arg(short, long, default_value = "0.8")]
        threshold: f32,
    },
    /// Install KDE desktop file for screenshot permissions
    InstallDesktopFile,
    /// Run interactive configuration wizard
    /// Run interactive configuration wizard
    Setup,
    /// Record supervised gameplay data
    Record {
        /// Output directory for the dataset
        #[arg(short, long, default_value = "dataset")]
        output: PathBuf,

        /// Target recording FPS
        #[arg(long, default_value_t = 10)]
        fps: u64,

        /// x264 preset (lower CPU = faster presets)
        #[arg(long, default_value = "ultrafast")]
        preset: String,

        /// Quality (CRF for x264, QP for VAAPI). Higher = smaller files. 28 recommended for VAAPI.
        #[arg(long, default_value_t = 28)]
        crf: u8,

        /// Input device path (optional, will prompt if not provided)
        #[arg(long)]
        device: Option<String>,

        /// Maximum output resolution (WxH, e.g., 1920x1080)
        #[arg(long, value_parser = parse_resolution)]
        max_resolution: Option<(u32, u32)>,

        /// Stop recording after this many seconds
        #[arg(long)]
        max_seconds: Option<u64>,

        /// Video encoder: auto (prefer hw), x264 (software), vaapi (Intel/AMD hw)
        #[arg(long, default_value = "auto")]
        encoder: String,
    },
}

#[derive(Subcommand)]
enum WindowCommands {
    /// List available windows
    List,
}

fn parse_region(s: &str) -> Result<Region, String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 4 {
        return Err("Region must be x,y,w,h".to_string());
    }
    let x = parts[0].parse().map_err(|_| "Invalid x".to_string())?;
    let y = parts[1].parse().map_err(|_| "Invalid y".to_string())?;
    let w = parts[2].parse().map_err(|_| "Invalid w".to_string())?;
    let h = parts[3].parse().map_err(|_| "Invalid h".to_string())?;
    Ok(Region {
        x,
        y,
        width: w,
        height: h,
    })
}

fn parse_resolution(s: &str) -> Result<(u32, u32), String> {
    let parts: Vec<&str> = s.split('x').collect();
    if parts.len() != 2 {
        return Err("Resolution must be WxH (e.g., 1920x1080)".to_string());
    }
    let w = parts[0].parse().map_err(|_| "Invalid width".to_string())?;
    let h = parts[1].parse().map_err(|_| "Invalid height".to_string())?;
    Ok((w, h))
}

fn init_logging(verbose: u8, quiet: bool) {
    let level = if quiet {
        "error"
    } else {
        match verbose {
            0 => "warn",
            1 => "info",
            2 => "debug",
            _ => "trace",
        }
    };

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(format!("visionctl={},inputctl={}", level, level)));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .without_time()
        .init();
}

fn main() -> inputctl_vision::Result<()> {
    let cli = Cli::parse();
    init_logging(cli.verbose, cli.quiet);

    match cli.command {
        Some(Commands::Agent {
            goal,
            window,
            region,
            blind,
            no_vision_tools,
            debug,
            max_iterations,
        }) => run_agent(
            &goal,
            window,
            region,
            blind,
            no_vision_tools,
            debug,
            max_iterations,
        ),
        Some(Commands::Window { command }) => match command {
            WindowCommands::List => run_list_windows(),
        },
        Some(Commands::Screenshot { output }) => run_screenshot(&output),
        Some(Commands::FindTemplate {
            template,
            screenshot,
            threshold,
        }) => run_find_template(&template, screenshot.as_deref(), threshold),
        Some(Commands::ClickTemplate {
            template,
            threshold,
        }) => run_click_template(&template, threshold),
        Some(Commands::InstallDesktopFile) => install_desktop_file(),
        Some(Commands::Setup) => run_setup(),
        Some(Commands::Record {
            output,
            fps,
            preset,
            crf,
            device,
            max_resolution,
            max_seconds,
            encoder,
        }) => {
            let encoder = encoder.parse::<Encoder>().unwrap_or_else(|e| {
                eprintln!("Warning: {}, using auto", e);
                Encoder::Auto
            });
            Ok(inputctl_vision::run_recorder(RecorderConfig {
                output_dir: output,
                fps,
                preset,
                crf,
                device_path: device,
                max_seconds,
                max_resolution,
                encoder,
            })?)
        }
        None => {
            // Default: query LLM with question
            let question = if cli.question.is_empty() {
                // Read from stdin
                let mut buffer = String::new();
                io::stdin().read_to_string(&mut buffer)?;
                buffer.trim().to_string()
            } else {
                cli.question.join(" ")
            };

            if question.is_empty() {
                // Show help if no question provided
                use clap::CommandFactory;
                Cli::command().print_help().ok();
                std::process::exit(1);
            }

            run_query(&question)
        }
    }
}

fn get_llm_config() -> inputctl_vision::Result<LlmConfig> {
    // Load config file (or defaults)
    let file_config = Config::load();

    // Priority: env vars > config file > defaults
    let mut settings = file_config.llm;

    if let Ok(backend) = std::env::var("VISIONCTL_BACKEND") {
        settings.backend = backend;
    }
    if let Ok(url) = std::env::var("VISIONCTL_URL") {
        settings.base_url = url;
    }
    if let Ok(model) = std::env::var("VISIONCTL_MODEL") {
        settings.model = model;
    }
    if let Ok(api_key) = std::env::var("VISIONCTL_API_KEY") {
        settings.api_key = Some(api_key);
    }
    if let Ok(temp_str) = std::env::var("VISIONCTL_TEMPERATURE") {
        if let Ok(temp) = temp_str.parse() {
            settings.temperature = temp;
        }
    }

    settings.to_llm_config()
}

fn run_query(question: &str) -> inputctl_vision::Result<()> {
    let config = get_llm_config()?;
    let ctl = VisionCtl::new(config)?;

    // Capture screenshot with cursor marker and save for debugging
    let screenshot = ctl.screenshot_with_cursor()?;
    let screenshot_path = "/tmp/visionctl_last_screenshot.png";
    fs::write(screenshot_path, &screenshot).map_err(|e| {
        inputctl_vision::Error::ScreenshotFailed(format!("Failed to save screenshot: {}", e))
    })?;
    debug!(path = %screenshot_path, "Screenshot saved");

    let answer = ctl.ask(question)?;
    println!(
        "{}",
        serde_json::json!({
            "question": question,
            "answer": answer
        })
    );

    Ok(())
}

fn run_agent(
    goal: &str,
    window_filter: Option<String>,
    region_filter: Option<Region>,
    blind_mode: bool,
    no_vision_tools: bool,
    debug_mode: bool,
    max_iterations: usize,
) -> inputctl_vision::Result<()> {
    info!(goal = %goal, "Starting agent");

    let main_config = get_llm_config()?;
    let iter_limit = if max_iterations == 0 {
        None
    } else {
        Some(max_iterations)
    };
    let mut agent = Agent::new(main_config)?
        .with_max_iterations(iter_limit)
        .with_verbose(true)
        .with_blind_mode(blind_mode)
        .with_no_vision_tools(no_vision_tools);

    if debug_mode {
        // Start the in-memory state store to track agent events
        let state_store = Arc::new(inputctl_vision::debugger::StateStore::new());
        let server_store = state_store.clone();

        // Spawn the Axum web server in a background task
        tokio::spawn(async move {
            let server = inputctl_vision::server::DebugServer::new(server_store);
            if let Err(e) = server.run(10888).await {
                error!("Debugger server failed: {}", e);
            }
        });

        // Attach the observer to the agent
        agent = agent.with_observer(state_store);
        eprintln!("Debugger running at http://localhost:10888");
    }

    // Apply viewport filters
    if let Some(name) = window_filter {
        info!("Looking for window matching '{}'", name);
        if let Some(win) = inputctl_vision::find_window(&name)? {
            info!(title = %win.title, region = ?win.region, "Found window, restricting viewport");
            agent.ctl_mut().set_viewport(Some(win.region));
        } else {
            error!("Window matching '{}' not found", name);
            return Err(inputctl_vision::Error::ScreenshotFailed(format!(
                "Window '{}' not found",
                name
            )));
        }
    } else if let Some(region) = region_filter {
        info!(region = ?region, "Restricting viewport to region");
        agent.ctl_mut().set_viewport(Some(region));
    }

    // Load config to check for pointing delegation
    let file_config = Config::load();
    if let Some(pointing_settings) = file_config.pointing {
        let pointing_config = pointing_settings.to_llm_config()?;
        agent = agent.with_pointing_config(pointing_config)?;
        info!("Delegated pointing enabled (from config)");
    }

    let agent_result = agent.run(goal);

    if debug_mode {
        if let Err(ref e) = agent_result {
            error!("Agent failed during execution: {}", e);
        }
        eprintln!("\nAgent task finished. Debugger server is still active.");
        eprintln!("Check the final state at: http://localhost:10888");
        eprintln!("Press Enter to stop the debugger and exit...");
        let mut input = String::new();
        let _ = std::io::stdin().read_line(&mut input);
    }

    let result = agent_result?;

    info!(
        success = result.success,
        iterations = result.iterations,
        "Agent complete"
    );
    println!(
        "{}",
        serde_json::json!({
            "success": result.success,
            "message": result.message,
            "iterations": result.iterations,
            "actions": result.actions_taken
        })
    );

    Ok(())
}

fn run_screenshot(output_path: &str) -> inputctl_vision::Result<()> {
    let screenshot = VisionCtl::screenshot()?;
    fs::write(output_path, &screenshot).map_err(|e| {
        inputctl_vision::Error::ScreenshotFailed(format!("Failed to save screenshot: {}", e))
    })?;
    info!(
        path = %output_path,
        bytes = screenshot.len(),
        "Screenshot saved"
    );
    Ok(())
}

fn run_find_template(
    template: &str,
    screenshot: Option<&str>,
    threshold: f32,
) -> inputctl_vision::Result<()> {
    // Take screenshot if not provided
    let screenshot_path = if let Some(path) = screenshot {
        path.to_string()
    } else {
        let screenshot_path = "/tmp/visionctl_detect.png";
        let screenshot_data = VisionCtl::screenshot()?;
        fs::write(screenshot_path, &screenshot_data).map_err(|e| {
            inputctl_vision::Error::ScreenshotFailed(format!("Failed to save screenshot: {}", e))
        })?;
        debug!(path = %screenshot_path, "Screenshot saved");
        screenshot_path.to_string()
    };

    info!(
        template = %template,
        screenshot = %screenshot_path,
        threshold = threshold,
        "Finding template"
    );

    let results =
        inputctl_vision::detection::find_template(&screenshot_path, template, Some(threshold))?;

    // Output JSON
    println!("{}", serde_json::to_string_pretty(&results).unwrap());

    // Summary
    if results.is_empty() {
        warn!("No matches found");
    } else {
        info!(count = results.len(), "Found matches");
        for r in results.iter().take(5) {
            debug!(x = r.x, y = r.y, confidence = r.confidence, "Match");
        }
    }

    Ok(())
}

fn run_click_template(template: &str, threshold: f32) -> inputctl_vision::Result<()> {
    // Take screenshot
    let screenshot_path = "/tmp/visionctl_click_detect.png";
    let screenshot_data = VisionCtl::screenshot()?;
    fs::write(screenshot_path, &screenshot_data).map_err(|e| {
        inputctl_vision::Error::ScreenshotFailed(format!("Failed to save screenshot: {}", e))
    })?;

    info!(template = %template, threshold = threshold, "Finding template to click");

    let results =
        inputctl_vision::detection::find_template(screenshot_path, template, Some(threshold))?;

    if results.is_empty() {
        warn!("No matches found - cannot click");
        return Err(inputctl_vision::Error::ScreenshotFailed(
            "No matches found".to_string(),
        ));
    }

    let best = &results[0];
    info!(
        x = best.x,
        y = best.y,
        confidence = best.confidence,
        "Clicking best match"
    );

    // Click using inputctl
    let status = Command::new("inputctl")
        .args(["click", &best.x.to_string(), &best.y.to_string()])
        .status()
        .map_err(|e| {
            inputctl_vision::Error::ScreenshotFailed(format!("Failed to run inputctl: {}", e))
        })?;

    if !status.success() {
        return Err(inputctl_vision::Error::ScreenshotFailed(
            "inputctl click failed".to_string(),
        ));
    }

    debug!(x = best.x, y = best.y, "Click complete");
    Ok(())
}

fn run_list_windows() -> inputctl_vision::Result<()> {
    let windows = inputctl_vision::list_windows()?;
    if windows.is_empty() {
        println!("No windows found.");
        return Ok(());
    }

    println!(
        "{:<10} {:<10} {:<10} {:<10} {:<10} {}",
        "ID", "X", "Y", "Width", "Height", "Title"
    );
    println!("{}", "-".repeat(80));

    for w in windows {
        println!(
            "{:<10} {:<10} {:<10} {:<10} {:<10} {}",
            w.id.chars().take(8).collect::<String>(), // Truncate ID
            w.region.x,
            w.region.y,
            w.region.width,
            w.region.height,
            w.title
        );
    }
    Ok(())
}

fn install_desktop_file() -> inputctl_vision::Result<()> {
    println!("=== VisionCtl Desktop File Installer ===\n");

    // Get the current executable path
    let exe_path = std::env::current_exe().map_err(|e| {
        inputctl_vision::Error::ScreenshotFailed(format!("Failed to get executable path: {}", e))
    })?;

    let exe_path_str = exe_path.to_str().ok_or_else(|| {
        inputctl_vision::Error::ScreenshotFailed("Invalid executable path".to_string())
    })?;

    println!("Detected executable: {}", exe_path_str);

    // Get desktop file path
    let home = std::env::var("HOME").map_err(|_| {
        inputctl_vision::Error::ScreenshotFailed("HOME environment variable not set".to_string())
    })?;

    let desktop_dir = PathBuf::from(home).join(".local/share/applications");

    // Create directory if it doesn't exist
    fs::create_dir_all(&desktop_dir).map_err(|e| {
        inputctl_vision::Error::ScreenshotFailed(format!(
            "Failed to create desktop directory: {}",
            e
        ))
    })?;

    let desktop_file_path = desktop_dir.join("visionctl.desktop");

    println!(
        "Desktop file will be created at: {}\n",
        desktop_file_path.display()
    );

    // Generate desktop file content
    let desktop_content = format!(
        r#"[Desktop Entry]
Name=VisionCtl
Comment=LLM-based GUI automation toolkit
Exec={exe_path}
Type=Application
Terminal=true
Categories=Development;Utility;
X-KDE-DBUS-Restricted-Interfaces=org.kde.KWin.ScreenShot2
"#,
        exe_path = exe_path_str
    );

    // Check if file already exists
    if desktop_file_path.exists() {
        print!("Desktop file already exists. Overwrite? [y/N]: ");
        io::stdout().flush().unwrap();

        let mut response = String::new();
        io::stdin().read_line(&mut response).unwrap();

        if !response.trim().eq_ignore_ascii_case("y") {
            println!("Aborted.");
            return Ok(());
        }
    }

    // Write desktop file
    fs::write(&desktop_file_path, desktop_content).map_err(|e| {
        inputctl_vision::Error::ScreenshotFailed(format!("Failed to write desktop file: {}", e))
    })?;

    println!("Desktop file created successfully!");

    // Update desktop database
    println!("\nUpdating desktop database...");
    let update_result = Command::new("update-desktop-database")
        .arg(desktop_dir.to_str().unwrap())
        .output();

    match update_result {
        Ok(output) if output.status.success() => {
            println!("Desktop database updated");
        }
        Ok(output) => {
            println!("Desktop database update failed (non-critical):");
            println!("{}", String::from_utf8_lossy(&output.stderr));
        }
        Err(_) => {
            println!("'update-desktop-database' not found (non-critical)");
            println!("You may need to log out and back in for changes to take effect");
        }
    }

    println!("\n=== Installation Complete ===");
    println!("\nThe visionctl application is now authorized to take screenshots on KDE.");
    println!("You can verify this by running:");
    println!("  cargo run --example screenshot_test --release");
    println!("\nNote: If you move the visionctl binary, you'll need to run this command again.");

    Ok(())
}

fn run_setup() -> inputctl_vision::Result<()> {
    println!("=== VisionCtl Configuration Wizard ===\n");

    // Load existing config or defaults
    let mut config = Config::load();

    // LLM Backend selection
    let backends = ["Ollama (Recommended)", "vLLM", "OpenAI"];
    let default_idx = match config.llm.backend.as_str() {
        "vllm" => 1,
        "openai" => 2,
        _ => 0,
    };

    let backend_idx = Select::new()
        .with_prompt("Select LLM backend")
        .items(&backends)
        .default(default_idx)
        .interact()
        .unwrap();

    config.llm.backend = match backend_idx {
        1 => "vllm".to_string(),
        2 => "openai".to_string(),
        _ => "ollama".to_string(),
    };

    // Base URL with backend-specific defaults
    let default_url = match config.llm.backend.as_str() {
        "vllm" => "http://localhost:8000".to_string(),
        "openai" => "https://api.openai.com/v1".to_string(),
        _ => "http://localhost:11434".to_string(),
    };

    let url: String = Input::new()
        .with_prompt("Base URL")
        .default(
            if config.llm.base_url != default_url && !config.llm.base_url.is_empty() {
                config.llm.base_url.clone()
            } else {
                default_url
            },
        )
        .interact_text()
        .unwrap();
    config.llm.base_url = url;

    // Model
    let default_model = match config.llm.backend.as_str() {
        "vllm" => "Qwen/Qwen2.5-VL-7B-Instruct".to_string(),
        "openai" => "gpt-4o".to_string(),
        _ => "qwen3-vl:30b".to_string(),
    };

    let model: String = Input::new()
        .with_prompt("Model name")
        .default(
            if config.llm.model != default_model && !config.llm.model.is_empty() {
                config.llm.model.clone()
            } else {
                default_model
            },
        )
        .interact_text()
        .unwrap();
    config.llm.model = model;

    // API Key (only for vllm/openai)
    if config.llm.backend == "vllm" || config.llm.backend == "openai" {
        let api_key: String = Input::new()
            .with_prompt("API Key (leave empty to skip)")
            .default(config.llm.api_key.clone().unwrap_or_default())
            .allow_empty(true)
            .interact_text()
            .unwrap();
        config.llm.api_key = if api_key.is_empty() {
            None
        } else {
            Some(api_key)
        };
    }

    // Cursor FPS
    let fps: u32 = Input::new()
        .with_prompt("Smooth mouse movement FPS")
        .default(config.cursor.smooth_fps)
        .interact_text()
        .unwrap();
    config.cursor.smooth_fps = fps;

    // Save config
    println!("\n--- Configuration Summary ---");
    println!("Backend:    {}", config.llm.backend);
    println!("Base URL:   {}", config.llm.base_url);
    println!("Model:      {}", config.llm.model);
    if let Some(ref key) = config.llm.api_key {
        println!("API Key:    {}...", &key[..key.len().min(8)]);
    }
    println!("Smooth FPS: {}", config.cursor.smooth_fps);
    println!();

    config.save().map_err(|e| {
        inputctl_vision::Error::ScreenshotFailed(format!("Failed to save config: {}", e))
    })?;

    println!("Configuration saved to: {}", Config::path().display());
    println!("\nYou can edit this file manually or run 'visionctl setup' again.");

    Ok(())
}
