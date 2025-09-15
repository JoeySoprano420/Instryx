//! Input/Output operations

/// Print a line to stdout
pub fn println(message: &str) {
    println!("{}", message);
}

/// Print to stdout without newline
pub fn print(message: &str) {
    print!("{}", message);
}

/// Read input from stdin
pub fn input() -> String {
    use std::io::{self, Write};
    
    let mut buffer = String::new();
    io::stdout().flush().unwrap();
    io::stdin().read_line(&mut buffer).unwrap();
    buffer.trim().to_string()
}