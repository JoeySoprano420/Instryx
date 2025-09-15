//! File system operations

pub use std::fs::{File, OpenOptions};
pub use std::io::{Read, Write, Result};
pub use std::path::{Path, PathBuf};

pub fn read_to_string<P: AsRef<Path>>(path: P) -> Result<String> {
    std::fs::read_to_string(path)
}

pub fn write<P: AsRef<Path>>(path: P, contents: &str) -> Result<()> {
    std::fs::write(path, contents)
}

pub fn exists<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().exists()
}