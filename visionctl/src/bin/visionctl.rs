use visionctl::{VisionCtl, LlmConfig, Agent, detection};
use std::io::{self, Read, Write};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() -> visionctl::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // Check for subcommands
    if args.len() > 1 {
        match args[1].as_str() {
            "agent" => {
                let goal = args.get(2).map(|s| s.as_str()).unwrap_or("");
                if goal.is_empty() {
                    eprintln!("Usage: visionctl agent <goal>");
                    eprintln!("Example: visionctl agent \"switch to the haruna window\"");
                    std::process::exit(1);
                }
                return run_agent(goal);
            }
            "target" => {
                let target = args.get(2).map(|s| s.as_str()).unwrap_or("");
                if target.is_empty() {
                    eprintln!("Usage: visionctl target <description>");
                    eprintln!("Example: visionctl target \"the minimize button of the Haruna window\"");
                    std::process::exit(1);
                }
                return run_target(target);
            }
            "install-desktop-file" => {
                return install_desktop_file();
            }
            "screenshot" => {
                let output_path = args.get(2).map(|s| s.as_str()).unwrap_or("/tmp/visionctl_screenshot.png");
                return run_screenshot(output_path);
            }
            "find-template" => {
                let template = args.get(2).map(|s| s.as_str()).unwrap_or("");
                if template.is_empty() {
                    eprintln!("Usage: visionctl find-template <template.png> [screenshot.png] [threshold]");
                    eprintln!("Example: visionctl find-template refs/orb.png /tmp/screen.png 0.7");
                    eprintln!("         visionctl find-template refs/orb.png 0.7  # auto-screenshot");
                    std::process::exit(1);
                }
                // Smart arg parsing: if arg3 looks like a number, it's threshold; else screenshot
                let (screenshot, threshold) = match args.get(3).map(|s| s.as_str()) {
                    Some(arg) if arg.parse::<f32>().is_ok() => {
                        // arg3 is a number -> threshold, auto-screenshot
                        (None, arg.parse().ok())
                    }
                    Some(screenshot_path) => {
                        // arg3 is a path -> screenshot, arg4 is threshold
                        (Some(screenshot_path), args.get(4).and_then(|s| s.parse().ok()))
                    }
                    None => (None, None),
                };
                return run_find_template(template, screenshot, threshold);
            }
            "click-template" => {
                let template = args.get(2).map(|s| s.as_str()).unwrap_or("");
                if template.is_empty() {
                    eprintln!("Usage: visionctl click-template <template.png> [threshold]");
                    eprintln!("Example: visionctl click-template refs/orb.png 0.7");
                    std::process::exit(1);
                }
                let threshold = args.get(3).and_then(|s| s.parse().ok());
                return run_click_template(template, threshold);
            }
            "find-text" => {
                if args.len() < 3 {
                    eprintln!("Usage: visionctl find-text <prompt> [prompt2] ... [--screenshot <path>] [--threshold <0-1>]");
                    eprintln!("Example: visionctl find-text \"button\" \"icon\"");
                    std::process::exit(1);
                }
                return run_find_text(&args[2..]);
            }
            "click-text" => {
                let prompt = args.get(2).map(|s| s.as_str()).unwrap_or("");
                if prompt.is_empty() {
                    eprintln!("Usage: visionctl click-text <prompt> [threshold]");
                    eprintln!("Example: visionctl click-text \"minimize button\" 0.3");
                    std::process::exit(1);
                }
                let threshold = args.get(3).and_then(|s| s.parse().ok());
                return run_click_text(prompt, threshold);
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            _ => {
                // Fall through to original behavior (question query)
            }
        }
    }

    // Original behavior: query LLM with question
    run_query()
}

fn install_desktop_file() -> visionctl::Result<()> {
    println!("=== VisionCtl Desktop File Installer ===\n");

    // Get the current executable path
    let exe_path = std::env::current_exe()
        .map_err(|e| visionctl::Error::ScreenshotFailed(format!("Failed to get executable path: {}", e)))?;

    let exe_path_str = exe_path.to_str()
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

    println!("Desktop file will be created at: {}\n", desktop_file_path.display());

    // Generate desktop file content
    let desktop_content = format!(r#"[Desktop Entry]
Name=VisionCtl
Comment=LLM-based GUI automation toolkit
Exec={exe_path}
Type=Application
Terminal=true
Categories=Development;Utility;
X-KDE-DBUS-Restricted-Interfaces=org.kde.KWin.ScreenShot2
"#, exe_path = exe_path_str);

    // Check if file already exists
    if desktop_file_path.exists() {
        println!("⚠️  Desktop file already exists. Overwrite? [y/N]: ");
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

    println!("✓ Desktop file created successfully!");

    // Update desktop database
    println!("\nUpdating desktop database...");
    let update_result = Command::new("update-desktop-database")
        .arg(desktop_dir.to_str().unwrap())
        .output();

    match update_result {
        Ok(output) if output.status.success() => {
            println!("✓ Desktop database updated");
        }
        Ok(output) => {
            println!("⚠️  Desktop database update failed (non-critical):");
            println!("{}", String::from_utf8_lossy(&output.stderr));
        }
        Err(_) => {
            println!("⚠️  'update-desktop-database' not found (non-critical)");
            println!("   You may need to log out and back in for changes to take effect");
        }
    }

    println!("\n=== Installation Complete ===");
    println!("\nThe visionctl application is now authorized to take screenshots on KDE.");
    println!("You can verify this by running:");
    println!("  cargo run --example screenshot_test --release");
    println!("\nNote: If you move the visionctl binary, you'll need to run this command again.");

    Ok(())
}

fn run_query() -> visionctl::Result<()> {
    // Read config from environment variables
    let backend = std::env::var("VISIONCTL_BACKEND")
        .unwrap_or_else(|_| "ollama".to_string());
    let url = std::env::var("VISIONCTL_URL")
        .unwrap_or_else(|_| "http://localhost:11434".to_string());
    let model = std::env::var("VISIONCTL_MODEL")
        .unwrap_or_else(|_| "llava".to_string());

    let config = match backend.to_lowercase().as_str() {
        "ollama" => LlmConfig::Ollama { url, model },
        "vllm" => LlmConfig::Vllm {
            url,
            model,
            api_key: std::env::var("VISIONCTL_API_KEY").ok()
        },
        "openai" => LlmConfig::OpenAI {
            url,
            model,
            api_key: std::env::var("VISIONCTL_API_KEY")
                .expect("VISIONCTL_API_KEY environment variable required for OpenAI backend"),
        },
        _ => {
            eprintln!("Unknown backend: {}", backend);
            eprintln!("Valid backends: ollama, vllm, openai");
            std::process::exit(1);
        }
    };

    let ctl = VisionCtl::new(config)?;

    // Read question from CLI args or stdin
    let question = if let Some(q) = std::env::args().nth(1) {
        q
    } else {
        let mut buffer = String::new();
        io::stdin().read_to_string(&mut buffer)?;
        buffer.trim().to_string()
    };

    if question.is_empty() {
        print_help();
        std::process::exit(1);
    }

    // Capture screenshot with grid and save it for debugging
    let screenshot = ctl.screenshot_with_grid()?;
    let screenshot_path = "/tmp/visionctl_last_screenshot.png";
    fs::write(screenshot_path, &screenshot)
        .map_err(|e| visionctl::Error::ScreenshotFailed(format!("Failed to save screenshot: {}", e)))?;
    eprintln!("Screenshot saved to: {}", screenshot_path);

    // Query LLM (which will capture screenshot again, but that's okay)
    let answer = ctl.ask(&question)?;
    println!("{}", serde_json::json!({
        "question": question,
        "answer": answer
    }));

    Ok(())
}

fn run_agent(goal: &str) -> visionctl::Result<()> {
    eprintln!("=== VisionCtl Agent ===");
    eprintln!("Goal: {}\n", goal);

    let config = get_llm_config()?;

    let agent = Agent::new(config)?
        .with_max_iterations(20)
        .with_verbose(true);

    let result = agent.run(goal)?;

    eprintln!("\n=== Agent Complete ===");
    println!("{}", serde_json::json!({
        "success": result.success,
        "message": result.message,
        "iterations": result.iterations,
        "actions": result.actions_taken
    }));

    Ok(())
}

fn run_screenshot(output_path: &str) -> visionctl::Result<()> {
    let screenshot = VisionCtl::screenshot()?;
    fs::write(output_path, &screenshot)
        .map_err(|e| visionctl::Error::ScreenshotFailed(format!("Failed to save screenshot: {}", e)))?;
    eprintln!("Screenshot saved to: {} ({} bytes)", output_path, screenshot.len());
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
    println!("{}", serde_json::json!({
        "success": result.success,
        "message": result.message,
        "iterations": result.iterations,
        "actions": result.actions_taken
    }));

    Ok(())
}

fn get_llm_config() -> visionctl::Result<LlmConfig> {
    let backend = std::env::var("VISIONCTL_BACKEND")
        .unwrap_or_else(|_| "ollama".to_string());
    let url = std::env::var("VISIONCTL_URL")
        .unwrap_or_else(|_| "http://localhost:11434".to_string());
    let model = std::env::var("VISIONCTL_MODEL")
        .unwrap_or_else(|_| "llava".to_string());

    let config = match backend.to_lowercase().as_str() {
        "ollama" => LlmConfig::Ollama { url, model },
        "vllm" => LlmConfig::Vllm {
            url,
            model,
            api_key: std::env::var("VISIONCTL_API_KEY").ok()
        },
        "openai" => LlmConfig::OpenAI {
            url,
            model,
            api_key: std::env::var("VISIONCTL_API_KEY")
                .map_err(|_| visionctl::Error::LlmApiError("VISIONCTL_API_KEY required for OpenAI".to_string()))?,
        },
        _ => {
            return Err(visionctl::Error::LlmApiError(format!("Unknown backend: {}", backend)));
        }
    };

    Ok(config)
}

fn run_find_template(template: &str, screenshot: Option<&str>, threshold: Option<f32>) -> visionctl::Result<()> {
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

    eprintln!("Finding template '{}' in '{}'...", template, screenshot_path);

    let results = detection::find_template(&screenshot_path, template, threshold)?;

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

fn run_click_template(template: &str, threshold: Option<f32>) -> visionctl::Result<()> {
    // Take screenshot
    let screenshot_path = "/tmp/visionctl_click_detect.png";
    let screenshot_data = VisionCtl::screenshot()?;
    fs::write(screenshot_path, &screenshot_data)
        .map_err(|e| visionctl::Error::ScreenshotFailed(format!("Failed to save screenshot: {}", e)))?;

    eprintln!("Finding template '{}'...", template);

    let results = detection::find_template(screenshot_path, template, threshold)?;

    if results.is_empty() {
        eprintln!("No matches found - cannot click");
        return Err(visionctl::Error::ScreenshotFailed("No matches found".to_string()));
    }

    let best = &results[0];
    eprintln!("Best match at ({}, {}) conf={:.3}", best.x, best.y, best.confidence);
    eprintln!("Clicking...");

    // Click using inputctl
    let status = Command::new("inputctl")
        .args(["click", &best.x.to_string(), &best.y.to_string()])
        .status()
        .map_err(|e| visionctl::Error::ScreenshotFailed(format!("Failed to run inputctl: {}", e)))?;

    if !status.success() {
        return Err(visionctl::Error::ScreenshotFailed("inputctl click failed".to_string()));
    }

    eprintln!("Clicked at ({}, {})", best.x, best.y);
    Ok(())
}

fn run_find_text(args: &[String]) -> visionctl::Result<()> {
    // Parse arguments: prompts [--screenshot path] [--threshold val]
    let mut prompts: Vec<&str> = Vec::new();
    let mut screenshot: Option<&str> = None;
    let mut threshold: Option<f32> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--screenshot" | "-s" => {
                if i + 1 < args.len() {
                    screenshot = Some(&args[i + 1]);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--threshold" | "-t" => {
                if i + 1 < args.len() {
                    threshold = args[i + 1].parse().ok();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            _ => {
                prompts.push(&args[i]);
                i += 1;
            }
        }
    }

    if prompts.is_empty() {
        return Err(visionctl::Error::ScreenshotFailed("No prompts provided".to_string()));
    }

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

    eprintln!("Finding objects matching: {:?}", prompts);

    let results = detection::find_by_text(&screenshot_path, &prompts, threshold)?;

    // Output JSON
    println!("{}", serde_json::to_string_pretty(&results).unwrap());

    // Summary
    if results.is_empty() {
        eprintln!("\nNo detections found");
    } else {
        eprintln!("\nFound {} detection(s):", results.len());
        for r in results.iter().take(5) {
            let label = r.label.as_deref().unwrap_or("unknown");
            eprintln!("  {}: ({}, {}) conf={:.3}", label, r.x, r.y, r.confidence);
        }
        if results.len() > 5 {
            eprintln!("  ... and {} more", results.len() - 5);
        }
    }

    Ok(())
}

fn run_click_text(prompt: &str, threshold: Option<f32>) -> visionctl::Result<()> {
    // Take screenshot
    let screenshot_path = "/tmp/visionctl_click_detect.png";
    let screenshot_data = VisionCtl::screenshot()?;
    fs::write(screenshot_path, &screenshot_data)
        .map_err(|e| visionctl::Error::ScreenshotFailed(format!("Failed to save screenshot: {}", e)))?;

    eprintln!("Finding objects matching '{}'...", prompt);

    let results = detection::find_by_text(screenshot_path, &[prompt], threshold)?;

    if results.is_empty() {
        eprintln!("No detections found - cannot click");
        return Err(visionctl::Error::ScreenshotFailed("No detections found".to_string()));
    }

    let best = &results[0];
    let label = best.label.as_deref().unwrap_or("object");
    eprintln!("Best match '{}' at ({}, {}) conf={:.3}", label, best.x, best.y, best.confidence);
    eprintln!("Clicking...");

    // Click using inputctl
    let status = Command::new("inputctl")
        .args(["click", &best.x.to_string(), &best.y.to_string()])
        .status()
        .map_err(|e| visionctl::Error::ScreenshotFailed(format!("Failed to run inputctl: {}", e)))?;

    if !status.success() {
        return Err(visionctl::Error::ScreenshotFailed("inputctl click failed".to_string()));
    }

    eprintln!("Clicked at ({}, {})", best.x, best.y);
    Ok(())
}

fn print_help() {
    println!("VisionCtl - LLM-based GUI automation toolkit\n");
    println!("USAGE:");
    println!("    visionctl <question>                    Query LLM about current screen");
    println!("    visionctl agent <goal>                  Run autonomous agent to accomplish goal");
    println!("    visionctl target <description>          Guide cursor to target and click it");
    println!("    visionctl screenshot [path]             Capture screenshot");
    println!("    visionctl find-template <img> [screen]  Find template image in screenshot");
    println!("    visionctl click-template <img>          Find template and click it");
    println!("    visionctl find-text <prompts...>        Find objects by text (YOLOE)");
    println!("    visionctl click-text <prompt>           Find by text and click (YOLOE)");
    println!("    visionctl install-desktop-file          Install KDE screenshot permission");
    println!("    visionctl --help                        Show this help\n");
    println!("EXAMPLES:");
    println!("    visionctl \"What's on my screen?\"");
    println!("    visionctl agent \"switch to the haruna window\"");
    println!("    visionctl screenshot /tmp/test.png");
    println!("    visionctl find-template refs/orb.png /tmp/screen.png 0.7");
    println!("    visionctl click-template refs/minimize_button.png");
    println!("    visionctl find-text \"button\" \"icon\"");
    println!("    visionctl click-text \"minimize button\"\n");
    println!("ENVIRONMENT:");
    println!("    VISIONCTL_BACKEND    LLM backend (ollama, vllm, openai) [default: ollama]");
    println!("    VISIONCTL_URL        Backend URL [default: http://localhost:11434]");
    println!("    VISIONCTL_MODEL      Model name [default: llava]");
    println!("    VISIONCTL_API_KEY    API key (required for openai backend)\n");
    println!("SETUP:");
    println!("    Template matching: cd visionctl/scripts && ./setup_detection.sh");
    println!("    YOLOE (optional):  cd visionctl/scripts && uv sync --extra ml\n");
}
