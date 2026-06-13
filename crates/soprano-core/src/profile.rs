//! Env-gated timing instrumentation (`SOPRANO_PROFILE=1`).
//!
//! Emits `[profile]` lines on stderr at pipeline boundaries so on-device
//! runs can attribute latency to a specific stage. Zero overhead beyond a
//! cached bool check when disabled.

use std::sync::OnceLock;

pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("SOPRANO_PROFILE").is_some())
}

macro_rules! profile_log {
    ($($arg:tt)*) => {
        if crate::profile::enabled() {
            eprintln!("[profile] {}", format!($($arg)*));
        }
    };
}

pub(crate) use profile_log;
