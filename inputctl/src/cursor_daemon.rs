//! Background daemon for fast cursor position tracking
//!
//! Registers a DBus service that receives cursor position updates from a
//! KWin script, enabling sub-millisecond cursor position queries.

use crate::cursor::CursorState;
use std::sync::Arc;
use std::time::Duration;
use zbus::blocking::connection::Builder;
use zbus::interface;

/// DBus interface for receiving cursor updates
struct CursorService {
    state: Arc<CursorState>,
}

#[interface(name = "org.inputctl.Cursor")]
impl CursorService {
    /// Called by KWin script when cursor position changes
    fn update(&self, x: i32, y: i32) {
        self.state.update(x, y);
    }
}

/// Error type for daemon operations
#[derive(Debug)]
pub enum DaemonError {
    Dbus(String),
    Script(String),
    Io(std::io::Error),
}

impl std::fmt::Display for DaemonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DaemonError::Dbus(s) => write!(f, "DBus error: {}", s),
            DaemonError::Script(s) => write!(f, "Script error: {}", s),
            DaemonError::Io(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for DaemonError {}

impl From<std::io::Error> for DaemonError {
    fn from(e: std::io::Error) -> Self {
        DaemonError::Io(e)
    }
}

/// Run the cursor daemon
///
/// This blocks until shutdown is signaled via the CursorState.
/// Should be called from a dedicated thread.
pub fn run_daemon(state: Arc<CursorState>, pid: u32) -> Result<(), DaemonError> {
    use zbus::names::WellKnownName;

    // DBus names can't have elements starting with digits, so use "p" prefix
    let service_name = format!("org.inputctl.Cursor.p{}", pid);
    let well_known_name: WellKnownName = service_name
        .as_str()
        .try_into()
        .map_err(|e| DaemonError::Dbus(format!("Invalid service name: {}", e)))?;

    // Build connection with our service
    let conn = Builder::session()
        .map_err(|e| DaemonError::Dbus(format!("Failed to create session builder: {}", e)))?
        .name(well_known_name)
        .map_err(|e| DaemonError::Dbus(format!("Failed to register name {}: {}", service_name, e)))?
        .serve_at(
            "/Cursor",
            CursorService {
                state: state.clone(),
            },
        )
        .map_err(|e| DaemonError::Dbus(format!("Failed to serve interface: {}", e)))?
        .build()
        .map_err(|e| DaemonError::Dbus(format!("Failed to build connection: {}", e)))?;

    // Load and run KWin script
    let script_id = load_kwin_script(&conn, pid)?;

    // Keep running until shutdown is signaled
    // The connection handles incoming DBus calls automatically
    while !state.should_shutdown() {
        std::thread::sleep(Duration::from_millis(10));
    }

    // Cleanup
    let _ = unload_kwin_script(&conn, script_id);

    Ok(())
}

/// Load the KWin cursor tracking script
fn load_kwin_script(conn: &zbus::blocking::Connection, pid: u32) -> Result<i32, DaemonError> {
    // Create script content that calls our DBus service on cursor changes
    let script = format!(
        r#"
var service = "org.inputctl.Cursor.p{}";
workspace.cursorPosChanged.connect(function() {{
    var pos = workspace.cursorPos;
    callDBus(service, "/Cursor", "org.inputctl.Cursor", "Update", pos.x, pos.y);
}});
// Send initial position
var pos = workspace.cursorPos;
callDBus(service, "/Cursor", "org.inputctl.Cursor", "Update", pos.x, pos.y);
"#,
        pid
    );

    // Write to temp file
    let script_path = format!("/tmp/inputctl_cursor_{}.js", pid);
    std::fs::write(&script_path, &script)?;

    // Load via KWin DBus
    let proxy =
        zbus::blocking::Proxy::new(conn, "org.kde.KWin", "/Scripting", "org.kde.kwin.Scripting")
            .map_err(|e| DaemonError::Dbus(format!("KWin Scripting interface not found: {}", e)))?;

    let script_id: i32 = proxy
        .call_method("loadScript", &(&script_path,))
        .map_err(|e| DaemonError::Script(format!("Failed to load script: {}", e)))?
        .body()
        .deserialize()
        .map_err(|e| DaemonError::Script(format!("Invalid script ID response: {}", e)))?;

    // Run the script
    let script_obj_path = format!("/Scripting/Script{}", script_id);
    let script_proxy = zbus::blocking::Proxy::new(
        conn,
        "org.kde.KWin",
        &*script_obj_path,
        "org.kde.kwin.Script",
    )
    .map_err(|e| DaemonError::Script(format!("Script proxy failed: {}", e)))?;

    script_proxy
        .call_method("run", &())
        .map_err(|e| DaemonError::Script(format!("Failed to run script: {}", e)))?;

    Ok(script_id)
}

/// Unload the KWin script and cleanup
fn unload_kwin_script(
    conn: &zbus::blocking::Connection,
    script_id: i32,
) -> Result<(), DaemonError> {
    // Try to stop the script
    let script_obj_path = format!("/Scripting/Script{}", script_id);
    if let Ok(proxy) = zbus::blocking::Proxy::new(
        conn,
        "org.kde.KWin",
        &*script_obj_path,
        "org.kde.kwin.Script",
    ) {
        let _ = proxy.call_method("stop", &());
    }

    // Clean up temp file
    let script_path = format!("/tmp/inputctl_cursor_{}.js", std::process::id());
    let _ = std::fs::remove_file(&script_path);

    Ok(())
}
