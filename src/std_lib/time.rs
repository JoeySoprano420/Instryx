//! Time and duration utilities

pub use std::time::{Duration, Instant, SystemTime};

pub fn now() -> Instant {
    Instant::now()
}

pub fn sleep(millis: u64) -> Duration {
    Duration::from_millis(millis)
}

pub fn timestamp() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}