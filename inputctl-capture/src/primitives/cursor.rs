use crate::error::{Error, Result};
use crate::primitives::grid::CursorPos;
use std::fs;
use std::process::Command;
use std::time::Duration;
use tempfile::NamedTempFile;
use zbus::blocking::Connection;
use zvariant::ObjectPath;

/// Find cursor position using KWin
///
/// Returns cursor coordinates
pub fn find_cursor() -> Result<CursorPos> {
    // Generate a unique marker for this query
    let marker = format!("VISIONCTL_CURSOR_{}", std::process::id());

    // Create temp script that prints cursor position with marker
    let script_file = NamedTempFile::with_suffix(".js")
        .map_err(|e| Error::ScreenshotFailed(format!("Failed to create script file: {}", e)))?;

    let script_content = format!(
        r#"
        var pos = workspace.cursorPos;
        print("{}_" + pos.x + "_" + pos.y);
    "#,
        marker
    );

    fs::write(script_file.path(), &script_content)
        .map_err(|e| Error::ScreenshotFailed(format!("Failed to write script: {}", e)))?;

    // Connect to DBus and load the script
    let conn = Connection::session()
        .map_err(|e| Error::ScreenshotFailed(format!("DBus connection failed: {}", e)))?;

    let proxy = zbus::blocking::Proxy::new(
        &conn,
        "org.kde.KWin",
        "/Scripting",
        "org.kde.kwin.Scripting",
    )
    .map_err(|e| Error::ScreenshotFailed(format!("KWin Scripting interface not found: {}", e)))?;

    // Load the script
    let script_path = script_file.path().to_string_lossy().to_string();
    let script_id: i32 = proxy
        .call_method("loadScript", &(&script_path,))
        .map_err(|e| Error::ScreenshotFailed(format!("Failed to load script: {}", e)))?
        .body()
        .deserialize()
        .map_err(|e| Error::ScreenshotFailed(format!("Invalid script ID: {}", e)))?;

    // Run the script
    let script_path_str = format!("/Scripting/Script{}", script_id);
    let script_obj_path = ObjectPath::try_from(script_path_str.as_str())
        .map_err(|e| Error::ScreenshotFailed(format!("Invalid script path: {}", e)))?;

    let script_proxy = zbus::blocking::Proxy::new(
        &conn,
        "org.kde.KWin",
        script_obj_path,
        "org.kde.kwin.Script",
    )
    .map_err(|e| Error::ScreenshotFailed(format!("Script proxy failed: {}", e)))?;

    script_proxy
        .call_method("run", &())
        .map_err(|e| Error::ScreenshotFailed(format!("Failed to run script: {}", e)))?;

    // Give script time to execute
    std::thread::sleep(Duration::from_millis(100));

    // Read from journalctl
    let output = Command::new("journalctl")
        .args([
            "--user",
            "-u",
            "plasma-kwin_wayland",
            "-n",
            "50",
            "--no-pager",
            "-o",
            "cat",
        ])
        .output()
        .map_err(|e| Error::ScreenshotFailed(format!("Failed to run journalctl: {}", e)))?;

    let journal_output = String::from_utf8_lossy(&output.stdout);

    // Unload the script
    let _ = proxy.call_method("unloadScript", &(&script_path,));

    // Find our marker in the output
    for line in journal_output.lines().rev() {
        if line.contains(&marker) {
            // Parse "MARKER_x_y" format
            let parts: Vec<&str> = line.split('_').collect();
            if parts.len() >= 3 {
                let x: i32 = parts[parts.len() - 2]
                    .parse()
                    .map_err(|_| Error::ScreenshotFailed(format!("Invalid x in: {}", line)))?;
                let y: i32 = parts[parts.len() - 1]
                    .parse()
                    .map_err(|_| Error::ScreenshotFailed(format!("Invalid y in: {}", line)))?;

                return Ok(CursorPos {
                    x,
                    y,
                    grid_cell: None,
                });
            }
        }
    }

    Err(Error::ScreenshotFailed(format!(
        "Cursor position not found in journal (marker: {})",
        marker
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires KDE Plasma running
    fn test_find_cursor() {
        let pos = find_cursor();
        if pos.is_ok() {
            let cursor = pos.unwrap();
            println!("Cursor at: ({}, {})", cursor.x, cursor.y);
            assert!(cursor.x >= 0);
            assert!(cursor.y >= 0);
        }
    }
}
