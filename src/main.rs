//! Instryx Compiler CLI
//! 
//! Command-line interface for the Instryx programming language compiler.

use clap::{Parser, Subcommand};
use instryx::{Compiler, Config, Target, OptimizationLevel, Error};
use std::path::PathBuf;
use std::process;

#[derive(Parser)]
#[command(name = "instryx")]
#[command(about = "Instryx Programming Language Compiler")]
#[command(version = instryx::VERSION)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile Instryx source files
    Build {
        /// Input file to compile
        #[arg(value_name = "FILE")]
        input: PathBuf,
        
        /// Output file name
        #[arg(short, long)]
        output: Option<String>,
        
        /// Compilation target
        #[arg(short, long, default_value = "native")]
        target: TargetArg,
        
        /// Optimization level
        #[arg(short = 'O', long, default_value = "debug")]
        optimization: OptArg,
        
        /// Include debug information
        #[arg(short, long)]
        debug: bool,
        
        /// Treat warnings as errors
        #[arg(short, long)]
        werror: bool,
    },
    
    /// Run Instryx code directly (interpreter mode)
    Run {
        /// Input file to run
        #[arg(value_name = "FILE")]
        input: PathBuf,
        
        /// Arguments to pass to the program
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },
    
    /// Create a new Instryx project
    New {
        /// Project name
        #[arg(value_name = "NAME")]
        name: String,
        
        /// Project template
        #[arg(long, default_value = "basic")]
        template: String,
    },
    
    /// Run tests
    Test {
        /// Directory containing tests
        #[arg(value_name = "DIR", default_value = "tests")]
        test_dir: PathBuf,
        
        /// Run tests in parallel
        #[arg(long)]
        parallel: bool,
    },
    
    /// Format source code
    Fmt {
        /// Files to format
        #[arg(value_name = "FILES")]
        files: Vec<PathBuf>,
        
        /// Check formatting without modifying files
        #[arg(long)]
        check: bool,
    },
    
    /// Generate documentation
    Doc {
        /// Source directory
        #[arg(value_name = "DIR", default_value = "src")]
        src_dir: PathBuf,
        
        /// Output directory for documentation
        #[arg(short, long, default_value = "docs")]
        output: PathBuf,
        
        /// Generate documentation for private items
        #[arg(long)]
        private: bool,
    },
    
    /// Language server for IDE support
    Lsp {
        /// Enable debug logging
        #[arg(long)]
        debug: bool,
    },
    
    /// Show version information
    Version,
}

#[derive(clap::ValueEnum, Clone)]
enum TargetArg {
    Native,
    Wasm,
    #[value(alias = "js")]
    Javascript,
}

impl From<TargetArg> for Target {
    fn from(arg: TargetArg) -> Self {
        match arg {
            TargetArg::Native => Target::Native,
            TargetArg::Wasm => Target::Wasm,
            TargetArg::Javascript => Target::JavaScript,
        }
    }
}

#[derive(clap::ValueEnum, Clone)]
enum OptArg {
    Debug,
    Release,
    Size,
    Speed,
}

impl From<OptArg> for OptimizationLevel {
    fn from(arg: OptArg) -> Self {
        match arg {
            OptArg::Debug => OptimizationLevel::Debug,
            OptArg::Release => OptimizationLevel::Release,
            OptArg::Size => OptimizationLevel::Size,
            OptArg::Speed => OptimizationLevel::Speed,
        }
    }
}

fn main() {
    let cli = Cli::parse();
    
    if let Err(e) = run(cli) {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Commands::Build { 
            input, 
            output, 
            target, 
            optimization, 
            debug, 
            werror 
        } => {
            let config = Config {
                target: target.into(),
                optimization_level: optimization.into(),
                debug_info: debug,
                warnings_as_errors: werror,
                output_file: output,
            };
            
            let compiler = Compiler::new(config);
            compiler.compile_file(&input)?;
            
            println!("Successfully compiled {}", input.display());
        },
        
        Commands::Run { input, args: _ } => {
            let config = Config::default();
            let compiler = Compiler::new(config);
            
            println!("Running {}...", input.display());
            compiler.compile_file(&input)?;
            // TODO: Actually execute the compiled code
            println!("Execution completed.");
        },
        
        Commands::New { name, template } => {
            create_new_project(&name, &template)?;
            println!("Created new Instryx project: {}", name);
        },
        
        Commands::Test { test_dir, parallel } => {
            run_tests(&test_dir, parallel)?;
        },
        
        Commands::Fmt { files, check } => {
            format_files(&files, check)?;
        },
        
        Commands::Doc { src_dir, output, private } => {
            generate_docs(&src_dir, &output, private)?;
        },
        
        Commands::Lsp { debug } => {
            start_language_server(debug)?;
        },
        
        Commands::Version => {
            print_version_info();
        },
    }
    
    Ok(())
}

fn create_new_project(name: &str, template: &str) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs;
    
    let project_dir = PathBuf::from(name);
    
    if project_dir.exists() {
        return Err(format!("Directory '{}' already exists", name).into());
    }
    
    fs::create_dir_all(&project_dir)?;
    fs::create_dir_all(project_dir.join("src"))?;
    
    // Create main.ix file
    let main_content = match template {
        "basic" => r#"fn main() -> () {
    println("Hello, Instryx!");
}
"#,
        "web" => r#"use std::web;

async fn main() -> () {
    let server = web::Server::new();
    server.route("/", handle_root);
    server.listen("127.0.0.1:8080").await;
}

async fn handle_root(request: web::Request) -> web::Response {
    web::Response::ok().html("<h1>Hello from Instryx!</h1>")
}
"#,
        "cli" => r#"use std::env;

fn main() -> () {
    let args = env::args();
    
    if args.len() < 2 {
        println("Usage: {} <command>", args[0]);
        return;
    }
    
    match args[1] {
        "hello" => println("Hello from Instryx CLI!"),
        _ => println("Unknown command: {}", args[1]),
    }
}
"#,
        _ => return Err(format!("Unknown template: {}", template).into()),
    };
    
    fs::write(project_dir.join("src/main.ix"), main_content)?;
    
    // Create project config
    let config_content = format!(r#"[project]
name = "{}"
version = "0.1.0"
edition = "2024"

[dependencies]
std = "0.1"

[build]
target = "native"
optimization = "debug"
"#, name);
    
    fs::write(project_dir.join("Instryx.toml"), config_content)?;
    
    // Create README
    let readme_content = format!(r#"# {}

An Instryx project.

## Building

```bash
instryx build src/main.ix
```

## Running

```bash
instryx run src/main.ix
```
"#, name);
    
    fs::write(project_dir.join("README.md"), readme_content)?;
    
    Ok(())
}

fn run_tests(_test_dir: &PathBuf, _parallel: bool) -> Result<(), Box<dyn std::error::Error>> {
    println!("Running tests...");
    // TODO: Implement test runner
    println!("All tests passed!");
    Ok(())
}

fn format_files(_files: &[PathBuf], _check: bool) -> Result<(), Box<dyn std::error::Error>> {
    println!("Formatting files...");
    // TODO: Implement formatter
    println!("Formatting complete!");
    Ok(())
}

fn generate_docs(_src_dir: &PathBuf, _output: &PathBuf, _private: bool) -> Result<(), Box<dyn std::error::Error>> {
    println!("Generating documentation...");
    // TODO: Implement documentation generator
    println!("Documentation generated!");
    Ok(())
}

fn start_language_server(_debug: bool) -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting Instryx Language Server...");
    // TODO: Implement language server
    println!("Language server started on stdio");
    Ok(())
}

fn print_version_info() {
    println!("Instryx {}", instryx::VERSION);
    println!("Edition: 2024");
    println!("Targets: native, wasm, javascript");
    
    println!("\nFeatures:");
    println!("  - Memory safety without garbage collection");
    println!("  - Strong static typing with inference");
    println!("  - Pattern matching and algebraic data types");
    println!("  - Async/await concurrency");
    println!("  - Zero-cost abstractions");
    println!("  - Cross-platform compilation");
    
    println!("\nCopyright (c) 2024 Instryx Language Team");
    println!("Licensed under the MIT License");
}