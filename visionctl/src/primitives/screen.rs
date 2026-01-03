use crate::error::{Error, Result};
use std::fs;
use std::process::Command;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tempfile::NamedTempFile;
use zbus::blocking::Connection;
use zvariant::ObjectPath;

/// Screen dimensions
#[derive(Clone, Copy, Debug)]
pub struct ScreenDimensions {
    pub width: u32,
    pub height: u32,
}

/// Cached screen dimensions with timestamp
struct CachedDimensions {
    dims: ScreenDimensions,
    fetched_at: Instant,
}

/// Global cache for screen dimensions (30 second TTL)
static SCREEN_CACHE: Mutex<Option<CachedDimensions>> = Mutex::new(None);
const CACHE_TTL: Duration = Duration::from_secs(30);

/// Query screen dimensions via KWin (cached for 30 seconds)
///
/// Returns the workspace dimensions (total screen area across all monitors).
/// Results are cached for 30 seconds to avoid repeated DBus queries while
/// still reacting to display configuration changes.
pub fn get_screen_dimensions() -> Result<ScreenDimensions> {
    // Check cache first
    {
        let cache = SCREEN_CACHE.lock().unwrap();
        if let Some(ref cached) = *cache {
            if cached.fetched_at.elapsed() < CACHE_TTL {
                return Ok(cached.dims);
            }
        }
    }

    // Cache miss or stale - query fresh
    let dims = query_screen_dimensions_uncached()?;

    // Update cache
    {
        let mut cache = SCREEN_CACHE.lock().unwrap();
        *cache = Some(CachedDimensions {
            dims,
            fetched_at: Instant::now(),
        });
    }

    Ok(dims)
}

/// Query screen dimensions from KWin (uncached)
fn query_screen_dimensions_uncached() -> Result<ScreenDimensions> {
    // Generate a unique marker for this query
    let marker = format!("VISIONCTL_SCREEN_{}", std::process::id());

    // Create temp script that prints screen dimensions with marker
    let script_file = NamedTempFile::with_suffix(".js")
        .map_err(|e| Error::ScreenshotFailed(format!("Failed to create script file: {}", e)))?;

    let script_content = format!(
        r#"
        print("{}_" + workspace.workspaceWidth + "_" + workspace.workspaceHeight);
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
            // Parse "MARKER_width_height" format
            let parts: Vec<&str> = line.split('_').collect();
            if parts.len() >= 3 {
                let width: u32 = parts[parts.len() - 2]
                    .parse()
                    .map_err(|_| Error::ScreenshotFailed(format!("Invalid width in: {}", line)))?;
                let height: u32 = parts[parts.len() - 1]
                    .parse()
                    .map_err(|_| Error::ScreenshotFailed(format!("Invalid height in: {}", line)))?;

                return Ok(ScreenDimensions { width, height });
            }
        }
    }

    Err(Error::ScreenshotFailed(format!(
        "Screen dimensions not found in journal (marker: {})",
        marker
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires KDE Plasma running
    fn test_get_screen_dimensions() {
        let dims = get_screen_dimensions();
        if dims.is_ok() {
            let d = dims.unwrap();
            println!("Screen dimensions: {}x{}", d.width, d.height);
            assert!(d.width > 0);
            assert!(d.height > 0);
        }
    }
}
