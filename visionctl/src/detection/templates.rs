use std::path::PathBuf;
use crate::Result;

/// Get user's template directory (~/.config/visionctl/templates/)
pub fn get_template_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".config/visionctl/templates")
}

/// Create template directory if it doesn't exist
pub fn ensure_template_dir() -> Result<()> {
    let dir = get_template_dir();
    if !dir.exists() {
        std::fs::create_dir_all(&dir)
            .map_err(|e| crate::Error::ScreenshotFailed(format!("Failed to create template directory: {}", e)))?;
    }
    Ok(())
}

/// List all available template names (without .png extension)
pub fn list_available_templates() -> Result<Vec<String>> {
    ensure_template_dir()?;
    let dir = get_template_dir();

    let mut templates = Vec::new();
    for entry in std::fs::read_dir(dir)
        .map_err(|e| crate::Error::ScreenshotFailed(format!("Failed to read template directory: {}", e)))?
    {
        let entry = entry
            .map_err(|e| crate::Error::ScreenshotFailed(format!("Failed to read directory entry: {}", e)))?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("png") {
            if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                templates.push(name.to_string());
            }
        }
    }

    templates.sort();
    Ok(templates)
}

/// Find template file path by name
pub fn find_template_file(name: &str) -> Result<PathBuf> {
    ensure_template_dir()?;
    let path = get_template_dir().join(format!("{}.png", name));

    if !path.exists() {
        let available = list_available_templates()?;
        let available_str = if available.is_empty() {
            "none".to_string()
        } else {
            available.join(", ")
        };
        return Err(crate::Error::ScreenshotFailed(format!(
            "Template '{}' not found. Available: {}. Add templates to ~/.config/visionctl/templates/",
            name, available_str
        )));
    }

    Ok(path)
}
