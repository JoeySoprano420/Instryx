# Instryx Language Specification

Version: 0.1.0 (Draft)

## Overview

Instryx is a modern systems programming language designed for safety, performance, and expressiveness. It combines the best features of contemporary languages while maintaining backwards compatibility and extensibility for future evolution.

## Design Principles

1. **Memory Safety**: Prevent common memory-related errors without runtime overhead
2. **Zero-Cost Abstractions**: High-level features should not impact performance
3. **Expressiveness**: Clean, readable syntax that scales from simple scripts to large systems
4. **Interoperability**: Easy integration with existing codebases and languages
5. **Backwards Compatibility**: Long-term stability and upgrade paths
6. **Extensibility**: Built-in mechanisms for language evolution

## Syntax Overview

### Variables and Types

```instryx
// Immutable by default
let name = "John";          // String (inferred)
let age: Int = 30;          // Explicit type annotation
let mut counter = 0;        // Mutable variable

// Type inference
let numbers = [1, 2, 3, 4, 5];  // Array<Int>
let point = (x: 10, y: 20);     // Tuple(Int, Int)
```

### Functions

```instryx
// Function definition
fn add(a: Int, b: Int) -> Int {
    a + b
}

// Generic function
fn identity<T>(value: T) -> T {
    value
}

// Async function
async fn fetch_data(url: String) -> Result<Data, Error> {
    // Implementation
}
```

### Control Flow

```instryx
// If expressions
let result = if condition {
    "true branch"
} else {
    "false branch"
};

// Pattern matching
match value {
    0 => "zero",
    1..10 => "small",
    n if n > 100 => "large",
    _ => "other",
}

// Loops
for item in collection {
    println(item);
}

while condition {
    // Loop body
}
```

### Data Types

```instryx
// Algebraic data types
enum Option<T> {
    Some(T),
    None,
}

// Structs
struct Person {
    name: String,
    age: Int,
    email: String,
}

// Traits (interfaces)
trait Display {
    fn display(self) -> String;
}
```

### Error Handling

```instryx
// Result types for error handling
fn divide(a: Int, b: Int) -> Result<Int, String> {
    if b == 0 {
        Err("Division by zero")
    } else {
        Ok(a / b)
    }
}

// Error propagation with ?
fn calculate() -> Result<Int, String> {
    let a = parse_int("10")?;
    let b = parse_int("5")?;
    divide(a, b)
}
```

### Modules and Imports

```instryx
// Module definition
mod math {
    pub fn add(a: Int, b: Int) -> Int {
        a + b
    }
    
    fn private_helper() -> Int {
        42
    }
}

// Imports
use std::collections::HashMap;
use math::{add, subtract};
```

### Memory Management

```instryx
// Ownership and borrowing (like Rust but simplified)
fn process_data(data: Vec<Int>) -> Vec<Int> {
    data.map(|x| x * 2)  // Takes ownership
}

fn read_data(data: &Vec<Int>) -> Int {
    data.len()  // Borrows immutably
}

fn modify_data(data: &mut Vec<Int>) {
    data.push(42);  // Borrows mutably
}
```

### Concurrency

```instryx
// Async/await
async fn main() {
    let future1 = fetch_url("https://api1.com");
    let future2 = fetch_url("https://api2.com");
    
    let (result1, result2) = await (future1, future2);
}

// Channels for communication
let (sender, receiver) = channel<String>();

spawn async {
    sender.send("Hello from thread").await;
};

let message = receiver.recv().await;
```

## Type System

### Primitive Types

- `Int`: Platform-sized integer
- `Int8`, `Int16`, `Int32`, `Int64`: Sized integers
- `UInt`, `UInt8`, `UInt16`, `UInt32`, `UInt64`: Unsigned integers
- `Float32`, `Float64`: Floating-point numbers
- `Bool`: Boolean values
- `Char`: Unicode scalar values
- `String`: UTF-8 strings
- `()`: Unit type

### Composite Types

- `Array<T>`: Fixed-size arrays
- `Vec<T>`: Dynamic arrays
- `HashMap<K, V>`: Hash maps
- `Option<T>`: Optional values
- `Result<T, E>`: Error handling
- `(T, U, ...)`: Tuples

### Advanced Types

- Generic types with constraints
- Higher-kinded types
- Dependent types (future)
- Linear types for resource management

## Standard Library

The standard library provides essential functionality:

- Collections: `Vec`, `HashMap`, `Set`, etc.
- I/O: File operations, networking, streams
- Async: Futures, async I/O, channels
- String manipulation and Unicode support
- Mathematical functions
- Date and time handling
- JSON and serialization
- Testing framework
- Benchmarking tools

## Tooling

- `instryx`: Main compiler and interpreter
- `instryx run`: Execute scripts directly
- `instryx build`: Compile projects
- `instryx test`: Run tests
- `instryx doc`: Generate documentation
- `instryx fmt`: Code formatting
- `instryx lint`: Static analysis
- Package manager integration

## Backwards Compatibility

Instryx maintains backwards compatibility through:

- Semantic versioning
- Deprecation warnings before removal
- Migration tools for major version upgrades
- Stable language edition system
- Conservative evolution approach

## Future Expansion

The language is designed to evolve:

- Plugin system for language extensions
- Macro system for code generation
- Foreign function interface (FFI)
- Custom operators and syntax
- Domain-specific language (DSL) support
- Compile-time computation expansion

This specification is a living document that will evolve with the language.