//! Standard library for the Instryx language
//! 
//! Provides essential functionality including I/O, collections, 
//! string manipulation, and system interfaces.

pub mod io;
pub mod collections;
pub mod string;
pub mod math;
pub mod time;
pub mod async_;
pub mod fs;
pub mod net;

/// Re-export commonly used items
pub use io::{println, print, input};
pub use collections::{Vec, HashMap, HashSet};
pub use string::String;
pub use math::*;

/// Standard library initialization
pub fn init() {
    println!("Instryx Standard Library v0.1.0");
}