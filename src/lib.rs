//! Instryx Programming Language
//! 
//! A modern systems programming language designed for safety, performance,
//! and expressiveness while maintaining backwards compatibility.

pub mod lexer;
pub mod ast;
pub mod parser;
pub mod semantic;
pub mod codegen;
pub mod runtime;
pub mod std_lib;

use std::fs;
use std::path::Path;

/// Result type for Instryx operations
pub type Result<T> = std::result::Result<T, Error>;

/// Error types for the Instryx compiler
#[derive(Debug, Clone)]
pub enum Error {
    /// I/O errors
    Io(String),
    
    /// Lexical errors
    Lexer(String),
    
    /// Parse errors
    Parser(String),
    
    /// Semantic analysis errors
    Semantic(String),
    
    /// Code generation errors
    Codegen(String),
    
    /// Runtime errors
    Runtime(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(msg) => write!(f, "I/O Error: {}", msg),
            Error::Lexer(msg) => write!(f, "Lexer Error: {}", msg),
            Error::Parser(msg) => write!(f, "Parser Error: {}", msg),
            Error::Semantic(msg) => write!(f, "Semantic Error: {}", msg),
            Error::Codegen(msg) => write!(f, "Codegen Error: {}", msg),
            Error::Runtime(msg) => write!(f, "Runtime Error: {}", msg),
        }
    }
}

impl std::error::Error for Error {}

/// Compiler configuration
#[derive(Debug, Clone)]
pub struct Config {
    pub target: Target,
    pub optimization_level: OptimizationLevel,
    pub debug_info: bool,
    pub warnings_as_errors: bool,
    pub output_file: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            target: Target::Native,
            optimization_level: OptimizationLevel::Debug,
            debug_info: true,
            warnings_as_errors: false,
            output_file: None,
        }
    }
}

/// Compilation targets
#[derive(Debug, Clone)]
pub enum Target {
    Native,
    Wasm,
    JavaScript,
    // More targets to be added
}

/// Optimization levels
#[derive(Debug, Clone)]
pub enum OptimizationLevel {
    Debug,    // No optimizations, full debug info
    Release,  // Full optimizations, minimal debug info
    Size,     // Optimize for size
    Speed,    // Optimize for speed
}

/// Main compiler interface
pub struct Compiler {
    config: Config,
}

impl Compiler {
    pub fn new(config: Config) -> Self {
        Compiler { config }
    }
    
    /// Compile a source file
    pub fn compile_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let source = fs::read_to_string(path.as_ref())
            .map_err(|e| Error::Io(format!("Failed to read file: {}", e)))?;
            
        self.compile_source(&source)
    }
    
    /// Compile source code from a string
    pub fn compile_source(&self, source: &str) -> Result<()> {
        // Step 1: Lexical analysis
        let mut lexer = lexer::Lexer::new(source);
        let tokens = lexer.tokenize();
        
        // Check for lexer errors
        for token in &tokens {
            if let lexer::TokenType::Invalid(msg) = &token.token_type {
                return Err(Error::Lexer(format!("Invalid token at line {}, column {}: {}", 
                    token.position.line, token.position.column, msg)));
            }
        }
        
        // Step 2: Parsing
        let mut parser = parser::Parser::new(tokens);
        let _ast = parser.parse().map_err(|e| Error::Parser(e))?;
        
        // Step 3: Semantic analysis (placeholder)
        // let mut analyzer = semantic::Analyzer::new();
        // analyzer.analyze(&ast)?;
        
        // Step 4: Code generation (placeholder)
        // let mut codegen = codegen::Generator::new(&self.config);
        // codegen.generate(&ast)?;
        
        println!("Compilation successful!");
        Ok(())
    }
    
    /// Run source code directly (interpreter mode)
    pub fn run_source(&self, source: &str) -> Result<()> {
        // For now, just compile
        self.compile_source(source)
    }
}

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const VERSION_MAJOR: u32 = 0;
pub const VERSION_MINOR: u32 = 1;
pub const VERSION_PATCH: u32 = 0;

/// Language edition for backwards compatibility
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edition {
    Edition2024,  // Initial edition
    // Future editions for backwards compatibility
}

impl Default for Edition {
    fn default() -> Self {
        Edition::Edition2024
    }
}

/// Feature flags for experimental features
#[derive(Debug, Clone)]
pub struct FeatureFlags {
    pub async_await: bool,
    pub pattern_matching: bool,
    pub type_inference: bool,
    pub memory_safety: bool,
    pub generic_functions: bool,
    pub modules: bool,
}

impl Default for FeatureFlags {
    fn default() -> Self {
        FeatureFlags {
            async_await: true,
            pattern_matching: true,
            type_inference: true,
            memory_safety: true,
            generic_functions: true,
            modules: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_compilation() {
        let compiler = Compiler::new(Config::default());
        let source = r#"
            fn main() -> () {
                let x = 42;
                println("Hello, Instryx!");
            }
        "#;
        
        // Should not panic and should succeed
        assert!(compiler.compile_source(source).is_ok());
    }
    
    #[test]
    fn test_lexer_integration() {
        let compiler = Compiler::new(Config::default());
        let source = "let x: Int = 42 + 10;";
        
        assert!(compiler.compile_source(source).is_ok());
    }
}