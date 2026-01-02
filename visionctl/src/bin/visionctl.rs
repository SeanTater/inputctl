use clap::{Parser, Subcommand};
use std::fs;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process::Command;
use visionctl::{Agent, LlmConfig, VisionCtl};

#[derive(Parser)]
#[command(name = "visionctl")]
#[command(about = "Vision-based GUI automation toolkit")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Question to ask LLM about the screen (default mode)
    #[arg(trailing_var_arg = true)]
    question: Vec<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run autonomous agent to achieve a goal
    Agent {
        /// Goal for the agent to accomplish
        goal: String,
    },
    /// Guide cursor to target element and click
    Target {
        /// Description of the target element
        description: String,
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
}

fn main() -> visionctl::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Agent { goal }) => run_agent(&goal),
        Some(Commands::Target { description }) => run_target(&description),
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

fn get_llm_config() -> visionctl::Result<LlmConfig> {
    let backend = std::env::var("VISIONCTL_BACKEND").unwrap_or_else(|_| "ollama".to_string());
    let url =
        std::env::var("VISIONCTL_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
    let model = std::env::var("VISIONCTL_MODEL").unwrap_or_else(|_| "llava".to_string());

    let config = match backend.to_lowercase().as_str() {
        "ollama" => LlmConfig::Ollama { url, model },
        "vllm" => LlmConfig::Vllm {
            url,
            model,
            api_key: std::env::var("VISIONCTL_API_KEY").ok(),
        },
        "openai" => LlmConfig::OpenAI {
            url,
            model,
            api_key: std::env::var("VISIONCTL_API_KEY").map_err(|_| {
                visionctl::Error::LlmApiError("VISIONCTL_API_KEY required for OpenAI".to_string())
            })?,
        },
        _ => {
            return Err(visionctl::Error::LlmApiError(format!(
                "Unknown backend: {}",
                backend
            )));
        }
    };

    Ok(config)
}

fn run_query(question: &str) -> visionctl::Result<()> {
    let config = get_llm_config()?;
    let ctl = VisionCtl::new(config)?;

    // Capture screenshot with grid and save for debugging
    let screenshot = ctl.screenshot_with_grid()?;
    let screenshot_path = "/tmp/visionctl_last_screenshot.png";
    fs::write(screenshot_path, &screenshot)
        .map_err(|e| visionctl::Error::ScreenshotFailed(format!("Failed to save screenshot: {}", e)))?;
    eprintln!("Screenshot saved to: {}", screenshot_path);

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

fn run_agent(goal: &str) -> visionctl::Result<()> {
    eprintln!("=== VisionCtl Agent ===");
    eprintln!("Goal: {}\n", goal);

    let config = get_llm_config()?;

    let agent = Agent::new(config)?.with_max_iterations(20).with_verbose(true);

    let result = agent.run(goal)?;

    eprintln!("\n=== Agent Complete ===");
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

fn run_target(target: &str) -> visionctl::Result<()> {
    eprintln!("=== VisionCtl Target Mode ===");
    eprintln!("Target: {}\n", target);

    let config = get_llm_config()?;

    let agent = Agent::new(config)?
        .with_max_iterations(10)
        .with_verbose(true);

    let result = agent.target(target)?;

    eprintln!("\n=== Target Complete ===");
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

fn run_screenshot(output_path: &str) -> visionctl::Result<()> {
    let screenshot = VisionCtl::screenshot()?;
    fs::write(output_path, &screenshot)
        .map_err(|e| visionctl::Error::ScreenshotFailed(format!("Failed to save screenshot: {}", e)))?;
    eprintln!(
        "Screenshot saved to: {} ({} bytes)",
        output_path,
        screenshot.len()
    );
    Ok(())
}

fn run_find_template(
    template: &str,
    screenshot: Option<&str>,
    threshold: f32,
) -> visionctl::Result<()> {
    // Take screenshot if not provided
    let screenshot_path = if let Some(path) = screenshot {
        path.to_string()
    } else {
        let screenshot_path = "/tmp/visionctl_detect.png";
        let screenshot_data = VisionCtl::screenshot()?;
        fs::write(screenshot_path, &screenshot_data)
            .map_err(|e| visionctl::Error::ScreenshotFailed(format!("Failed to save screenshot: {}", e)))?;
        eprintln!("Screenshot saved to: {}", screenshot_path);
        screenshot_path.to_string()
    };

    eprintln!(
        "Finding template '{}' in '{}' (threshold={})...",
        template, screenshot_path, threshold
    );

    let results = visionctl::detection::find_template(&screenshot_path, template, Some(threshold))?;

    // Output JSON
    println!("{}", serde_json::to_string_pretty(&results).unwrap());

    // Summary
    if results.is_empty() {
        eprintln!("\nNo matches found");
    } else {
        eprintln!("\nFound {} match(es):", results.len());
        for r in results.iter().take(5) {
            eprintln!("  ({}, {}) conf={:.3}", r.x, r.y, r.confidence);
        }
        if results.len() > 5 {
            eprintln!("  ... and {} more", results.len() - 5);
        }
    }

    Ok(())
}

fn run_click_template(template: &str, threshold: f32) -> visionctl::Result<()> {
    // Take screenshot
    let screenshot_path = "/tmp/visionctl_click_detect.png";
    let screenshot_data = VisionCtl::screenshot()?;
    fs::write(screenshot_path, &screenshot_data)
        .map_err(|e| visionctl::Error::ScreenshotFailed(format!("Failed to save screenshot: {}", e)))?;

    eprintln!("Finding template '{}' (threshold={})...", template, threshold);

    let results = visionctl::detection::find_template(screenshot_path, template, Some(threshold))?;

    if results.is_empty() {
        eprintln!("No matches found - cannot click");
        return Err(visionctl::Error::ScreenshotFailed(
            "No matches found".to_string(),
        ));
    }

    let best = &results[0];
    eprintln!(
        "Best match at ({}, {}) conf={:.3}",
        best.x, best.y, best.confidence
    );
    eprintln!("Clicking...");

    // Click using inputctl
    let status = Command::new("inputctl")
        .args(["click", &best.x.to_string(), &best.y.to_string()])
        .status()
        .map_err(|e| visionctl::Error::ScreenshotFailed(format!("Failed to run inputctl: {}", e)))?;

    if !status.success() {
        return Err(visionctl::Error::ScreenshotFailed(
            "inputctl click failed".to_string(),
        ));
    }

    eprintln!("Clicked at ({}, {})", best.x, best.y);
    Ok(())
}

fn install_desktop_file() -> visionctl::Result<()> {
    println!("=== VisionCtl Desktop File Installer ===\n");

    // Get the current executable path
    let exe_path = std::env::current_exe()
        .map_err(|e| visionctl::Error::ScreenshotFailed(format!("Failed to get executable path: {}", e)))?;

    let exe_path_str = exe_path
        .to_str()
        .ok_or_else(|| visionctl::Error::ScreenshotFailed("Invalid executable path".to_string()))?;

    println!("Detected executable: {}", exe_path_str);

    // Get desktop file path
    let home = std::env::var("HOME")
        .map_err(|_| visionctl::Error::ScreenshotFailed("HOME environment variable not set".to_string()))?;

    let desktop_dir = PathBuf::from(home).join(".local/share/applications");

    // Create directory if it doesn't exist
    fs::create_dir_all(&desktop_dir)
        .map_err(|e| visionctl::Error::ScreenshotFailed(format!("Failed to create desktop directory: {}", e)))?;

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
    fs::write(&desktop_file_path, desktop_content)
        .map_err(|e| visionctl::Error::ScreenshotFailed(format!("Failed to write desktop file: {}", e)))?;

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
