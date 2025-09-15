//! Integration tests for the Instryx compiler

use instryx::{Compiler, Config};

#[test]
fn test_hello_world_compilation() {
    let compiler = Compiler::new(Config::default());
    let source = r#"
        fn main() -> () {
            println("Hello, World!");
        }
    "#;
    
    assert!(compiler.compile_source(source).is_ok());
}

#[test]
fn test_variable_declaration() {
    let compiler = Compiler::new(Config::default());
    let source = r#"
        fn main() -> () {
            let x = 42;
            let mut y = "hello";
            let z: Bool = true;
        }
    "#;
    
    assert!(compiler.compile_source(source).is_ok());
}

#[test]
fn test_function_definition() {
    let compiler = Compiler::new(Config::default());
    let source = r#"
        fn add(a: Int, b: Int) -> Int {
            a + b
        }
        
        fn main() -> () {
            let result = add(5, 10);
            println("Result: {}", result);
        }
    "#;
    
    assert!(compiler.compile_source(source).is_ok());
}

#[test]
fn test_struct_definition() {
    let compiler = Compiler::new(Config::default());
    let source = r#"
        struct Point {
            pub x: Int,
            pub y: Int,
        }
        
        fn main() -> () {
            let p = Point { x: 10, y: 20 };
            println("Point: ({}, {})", p.x, p.y);
        }
    "#;
    
    assert!(compiler.compile_source(source).is_ok());
}

#[test]
fn test_enum_definition() {
    let compiler = Compiler::new(Config::default());
    let source = r#"
        enum Color {
            Red,
            Green,
            Blue,
        }
        
        fn main() -> () {
            let color = Color::Red;
            match color {
                Color::Red => println("Red color"),
                Color::Green => println("Green color"),
                Color::Blue => println("Blue color"),
            }
        }
    "#;
    
    assert!(compiler.compile_source(source).is_ok());
}

#[test]
fn test_lexer_errors() {
    let compiler = Compiler::new(Config::default());
    let source = "let x = @#$%^&*;"; // Invalid characters
    
    let result = compiler.compile_source(source);
    assert!(result.is_err());
    
    if let Err(error) = result {
        assert!(error.to_string().contains("Lexer Error"));
    }
}

#[test]
fn test_complex_program() {
    let compiler = Compiler::new(Config::default());
    let source = r#"
        struct User {
            pub name: String,
            pub age: Int,
            email: String,
        }
        
        enum Status {
            Active,
            Inactive,
            Pending(String),
        }
        
        fn create_user(name: String, age: Int, email: String) -> User {
            User { name, age, email }
        }
        
        fn main() -> () {
            let user = create_user("Alice", 30, "alice@example.com");
            let status = Status::Active;
            
            println("User: {} ({})", user.name, user.age);
            
            match status {
                Status::Active => println("User is active"),
                Status::Inactive => println("User is inactive"),
                Status::Pending(reason) => println("User pending: {}", reason),
            }
        }
    "#;
    
    assert!(compiler.compile_source(source).is_ok());
}