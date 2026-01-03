//! Shared cursor state for fast position tracking
//!
//! Uses atomics for lock-free access from main thread while
//! background thread receives updates via DBus.

use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Shared cursor state between main thread and daemon thread
pub struct CursorState {
    x: AtomicI32,
    y: AtomicI32,
    last_update_ms: AtomicU64,
    shutdown: AtomicBool,
}

impl CursorState {
    pub fn new() -> Self {
        Self {
            x: AtomicI32::new(0),
            y: AtomicI32::new(0),
            last_update_ms: AtomicU64::new(0),
            shutdown: AtomicBool::new(false),
        }
    }

    /// Update cursor position (called by daemon thread)
    pub fn update(&self, x: i32, y: i32) {
        self.x.store(x, Ordering::Relaxed);
        self.y.store(y, Ordering::Relaxed);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        self.last_update_ms.store(now, Ordering::Relaxed);
    }

    /// Get current cursor position (instant, <1μs)
    pub fn get(&self) -> (i32, i32) {
        (
            self.x.load(Ordering::Relaxed),
            self.y.load(Ordering::Relaxed),
        )
    }

    /// Check if cursor tracking is stale (no updates in given time)
    pub fn is_stale(&self, max_age_ms: u64) -> bool {
        let last = self.last_update_ms.load(Ordering::Relaxed);
        if last == 0 {
            // Never updated yet
            return true;
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        now.saturating_sub(last) > max_age_ms
    }

    /// Signal the daemon thread to shut down
    pub fn signal_shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }

    /// Check if shutdown was signaled
    pub fn should_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Relaxed)
    }
}

impl Default for CursorState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_and_get() {
        let state = CursorState::new();
        assert_eq!(state.get(), (0, 0));

        state.update(100, 200);
        assert_eq!(state.get(), (100, 200));

        state.update(-50, 300);
        assert_eq!(state.get(), (-50, 300));
    }

    #[test]
    fn test_staleness() {
        let state = CursorState::new();

        // Never updated = stale
        assert!(state.is_stale(1000));

        // Just updated = not stale
        state.update(0, 0);
        assert!(!state.is_stale(1000));

        // Still fresh with short max age
        assert!(!state.is_stale(100));
    }

    #[test]
    fn test_shutdown() {
        let state = CursorState::new();
        assert!(!state.should_shutdown());

        state.signal_shutdown();
        assert!(state.should_shutdown());
    }
}
