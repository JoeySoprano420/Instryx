# Instryx Programming Language

A modern, high-performance programming language designed for the future of software development.

## Features

- **Memory Safety**: Built-in memory safety without garbage collection overhead
- **Strong Type System**: Static typing with intelligent type inference
- **High Performance**: Compiled to native code with zero-cost abstractions
- **Concurrency**: First-class async/await and concurrent programming support
- **Pattern Matching**: Powerful pattern matching and algebraic data types
- **Cross-Platform**: Compile to multiple targets including native, WASM, and JVM
- **Modern Syntax**: Clean, expressive syntax inspired by the best modern languages
- **Backwards Compatible**: Designed for long-term stability and backwards compatibility

## Quick Start

```bash
# Install Instryx (coming soon)
curl -sSf https://instryx.dev/install.sh | sh

# Create a new project
instryx new hello-world
cd hello-world

# Run your first program
instryx run
```

## Example Code

```instryx
// Hello World
fn main() -> () {
    println("Hello, World!");
}

// Type-safe variables with inference
let message: String = "Instryx is awesome!";
let count = 42; // inferred as Int

// Pattern matching
match count {
    0 => println("Zero"),
    1..10 => println("Small number"),
    _ => println("Large number"),
}

// Async functions
async fn fetch_data(url: String) -> Result<Data, Error> {
    let response = await http::get(url)?;
    Ok(response.json()?)
}
```

## Project Status

🚧 **Under Active Development** 🚧

This project is in active development. The language specification and implementation are being continuously expanded and improved.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.