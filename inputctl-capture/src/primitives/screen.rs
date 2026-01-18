//! Screen and window management primitives.
//!
//! This module provides the [`Region`] and [`Window`] types, as well as functions
//! to list windows and get screen dimensions, primarily targeting Wayland desktops
//! via D-Bus interfaces.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
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

/// A rectangular region on the screen
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Region {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl Region {
    pub fn new(x: i32, y: i32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Check if a point is within the region
    pub fn contains(&self, x: i32, y: i32) -> bool {
        x >= self.x
            && x < self.x + self.width as i32
            && y >= self.y
            && y < self.y + self.height as i32
    }
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

/// A window on the screen
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Window {
    pub id: String,
    pub title: String,
    pub region: Region,
}

/// List all available windows (clients) from KWin
pub fn list_windows() -> Result<Vec<Window>> {
    // Generate a unique marker for this query
    let marker = format!("VISIONCTL_WINDOWS_{}", std::process::id());

    // Create temp script that prints window info
    let script_file = NamedTempFile::with_suffix(".js")
        .map_err(|e| Error::ScreenshotFailed(format!("Failed to create script file: {}", e)))?;

    // We use workspace.windowList() (Plasma 6) or workspace.clientList() (Plasma 5/Early 6)
    // trying windowList() first as it's cleaner for modern KWin
    let script_content = format!(
        r#"
        const clients = workspace.windowList();
        for (let i = 0; i < clients.length; i++) {{
            const c = clients[i];
            // Format: MARKER_internalId_x_y_width_height_caption
            print("{}_" + c.internalId + "_" + c.x + "_" + c.y + "_" + c.width + "_" + c.height + "_" + c.caption);
        }}
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
    std::thread::sleep(Duration::from_millis(200));

    // Read from journalctl
    let output = Command::new("journalctl")
        .args([
            "--user",
            "-u",
            "plasma-kwin_wayland",
            "-n",
            "200", // Need more lines for windows
            "--no-pager",
            "-o",
            "cat",
        ])
        .output()
        .map_err(|e| Error::ScreenshotFailed(format!("Failed to run journalctl: {}", e)))?;

    let journal_output = String::from_utf8_lossy(&output.stdout);

    // Unload the script
    let _ = proxy.call_method("unloadScript", &(&script_path,));

    let mut windows = Vec::new();
    let marker_pat = format!("{}_", marker);

    for line in journal_output.lines().rev() {
        if let Some(data) = line.split(&marker_pat).nth(1) {
            // Parse "internalId_x_y_width_height_caption"
            // Note: caption might contain underscores, so LIMIT split
            let parts: Vec<&str> = data.splitn(6, '_').collect();
            if parts.len() == 6 {
                let id = parts[0].to_string();
                let x = parts[1].parse::<f64>().unwrap_or(0.0) as i32;
                let y = parts[2].parse::<f64>().unwrap_or(0.0) as i32;
                let width = parts[3].parse::<f64>().unwrap_or(0.0) as u32;
                let height = parts[4].parse::<f64>().unwrap_or(0.0) as u32;
                let title = parts[5].to_string();

                windows.push(Window {
                    id,
                    title,
                    region: Region {
                        x,
                        y,
                        width,
                        height,
                    },
                });
            }
        }
    }

    // Reverse to match original order (we iterated rev() to find recent first)
    // But since unrelated lines are skipped, order might be mixed if we read too far back.
    // Ideally we filter by unique ID and take the most recent occurrence?
    // Yes, KWin prints once. `journalctl -n 200` might contain stale runs if we run fast.
    // The marker includes process ID, so that helps uniqueness per run.
    windows.reverse();
    Ok(windows)
}

/// Find a window by title substring
pub fn find_window(name: &str) -> Result<Option<Window>> {
    let windows = list_windows()?;
    let name_lower = name.to_lowercase();

    // Find best match (exact match > substring match)
    // Or just first substring match
    for win in windows {
        if win.title.to_lowercase().contains(&name_lower) {
            return Ok(Some(win));
        }
    }

    Ok(None)
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
