use visionctl::{VisionCtl, LlmConfig};
use std::io::{self, Read, Write};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() -> visionctl::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // Check for subcommands
    if args.len() > 1 {
        match args[1].as_str() {
            "install-desktop-file" => {
                return install_desktop_file();
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

    // Query and output JSON
    let answer = ctl.ask(&question)?;
    println!("{}", serde_json::json!({
        "question": question,
        "answer": answer
    }));

    Ok(())
}

fn print_help() {
    println!("VisionCtl - LLM-based GUI automation toolkit\n");
    println!("USAGE:");
    println!("    visionctl <question>           Query LLM about current screen");
    println!("    visionctl install-desktop-file Install KDE screenshot permission");
    println!("    visionctl --help               Show this help\n");
    println!("EXAMPLES:");
    println!("    visionctl \"What's on my screen?\"");
    println!("    visionctl install-desktop-file\n");
    println!("ENVIRONMENT:");
    println!("    VISIONCTL_BACKEND    LLM backend (ollama, vllm, openai) [default: ollama]");
    println!("    VISIONCTL_URL        Backend URL [default: http://localhost:11434]");
    println!("    VISIONCTL_MODEL      Model name [default: llava]");
    println!("    VISIONCTL_API_KEY    API key (required for openai backend)\n");
}
