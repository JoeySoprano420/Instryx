//! Abstract Syntax Tree (AST) for the Instryx language
//!
//! This module defines the data structures that represent the parsed syntax
//! of Instryx source code.

use crate::lexer::Position;
use std::collections::HashMap;
use std::fmt;

/// AST node with position information for error reporting
#[derive(Debug, Clone, PartialEq)]
pub struct AstNode<T> {
    pub node: T,
    pub position: Position,
}

impl<T> AstNode<T> {
    pub fn new(node: T, position: Position) -> Self {
        AstNode { node, position }
    }
}

/// Type annotations
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    /// Primitive types
    Int,
    Float,
    Bool,
    Char,
    String,
    Unit,  // ()
    
    /// Generic type parameter
    TypeVar(String),
    
    /// Function type: (param_types) -> return_type
    Function {
        params: Vec<Type>,
        return_type: Box<Type>,
    },
    
    /// Array type: [T; size] or [T] for dynamic
    Array {
        element_type: Box<Type>,
        size: Option<usize>,
    },
    
    /// Tuple type: (T, U, V, ...)
    Tuple(Vec<Type>),
    
    /// Optional type: Option<T>
    Option(Box<Type>),
    
    /// Result type: Result<T, E>
    Result {
        ok_type: Box<Type>,
        err_type: Box<Type>,
    },
    
    /// User-defined type
    UserDefined {
        name: String,
        type_args: Vec<Type>,
    },
    
    /// Reference type: &T or &mut T
    Reference {
        mutable: bool,
        inner: Box<Type>,
    },
}

/// Expressions in the language
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Literals
    IntLiteral(i64),
    FloatLiteral(f64),
    StringLiteral(String),
    CharLiteral(char),
    BoolLiteral(bool),
    UnitLiteral,
    
    /// Variable reference
    Identifier(String),
    
    /// Binary operations
    BinaryOp {
        left: Box<AstNode<Expr>>,
        op: BinaryOperator,
        right: Box<AstNode<Expr>>,
    },
    
    /// Unary operations
    UnaryOp {
        op: UnaryOperator,
        operand: Box<AstNode<Expr>>,
    },
    
    /// Function call
    Call {
        function: Box<AstNode<Expr>>,
        args: Vec<AstNode<Expr>>,
    },
    
    /// Method call
    MethodCall {
        receiver: Box<AstNode<Expr>>,
        method: String,
        args: Vec<AstNode<Expr>>,
    },
    
    /// Field access
    FieldAccess {
        object: Box<AstNode<Expr>>,
        field: String,
    },
    
    /// Index access
    Index {
        object: Box<AstNode<Expr>>,
        index: Box<AstNode<Expr>>,
    },
    
    /// Array literal
    Array(Vec<AstNode<Expr>>),
    
    /// Tuple literal
    Tuple(Vec<AstNode<Expr>>),
    
    /// Struct literal
    StructLiteral {
        name: String,
        fields: Vec<(String, AstNode<Expr>)>,
    },
    
    /// If expression
    If {
        condition: Box<AstNode<Expr>>,
        then_branch: Box<AstNode<Expr>>,
        else_branch: Option<Box<AstNode<Expr>>>,
    },
    
    /// Match expression
    Match {
        expression: Box<AstNode<Expr>>,
        arms: Vec<MatchArm>,
    },
    
    /// Block expression
    Block(Vec<AstNode<Stmt>>),
    
    /// Lambda/closure
    Lambda {
        params: Vec<Parameter>,
        body: Box<AstNode<Expr>>,
    },
    
    /// Async expression
    Async(Box<AstNode<Expr>>),
    
    /// Await expression
    Await(Box<AstNode<Expr>>),
    
    /// Return expression
    Return(Option<Box<AstNode<Expr>>>),
    
    /// Break expression
    Break(Option<Box<AstNode<Expr>>>),
    
    /// Continue expression
    Continue,
}

/// Binary operators
#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOperator {
    // Arithmetic
    Add, Subtract, Multiply, Divide, Modulo,
    
    // Comparison
    Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual,
    
    // Logical
    And, Or,
    
    // Bitwise
    BitAnd, BitOr, BitXor, LeftShift, RightShift,
    
    // Assignment
    Assign, AddAssign, SubtractAssign, MultiplyAssign, DivideAssign,
}

/// Unary operators
#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOperator {
    Plus, Minus, Not, BitNot, Deref, AddressOf, MutableRef,
}

/// Match arm in a match expression
#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub pattern: AstNode<Pattern>,
    pub guard: Option<AstNode<Expr>>,
    pub body: AstNode<Expr>,
}

/// Patterns for destructuring and matching
#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    /// Wildcard pattern _
    Wildcard,
    
    /// Variable binding
    Identifier(String),
    
    /// Literal patterns
    IntLiteral(i64),
    FloatLiteral(f64),
    StringLiteral(String),
    CharLiteral(char),
    BoolLiteral(bool),
    
    /// Tuple pattern
    Tuple(Vec<AstNode<Pattern>>),
    
    /// Array pattern
    Array(Vec<AstNode<Pattern>>),
    
    /// Struct pattern
    Struct {
        name: String,
        fields: Vec<(String, AstNode<Pattern>)>,
    },
    
    /// Enum variant pattern
    Enum {
        name: String,
        variant: String,
        fields: Vec<AstNode<Pattern>>,
    },
    
    /// Range pattern
    Range {
        start: Box<AstNode<Pattern>>,
        end: Box<AstNode<Pattern>>,
        inclusive: bool,
    },
    
    /// Or pattern (pattern1 | pattern2)
    Or(Vec<AstNode<Pattern>>),
    
    /// Reference pattern
    Ref {
        mutable: bool,
        pattern: Box<AstNode<Pattern>>,
    },
}

/// Statements
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    /// Expression statement
    Expression(AstNode<Expr>),
    
    /// Variable declaration
    Let {
        mutable: bool,
        name: String,
        type_annotation: Option<Type>,
        value: Option<AstNode<Expr>>,
    },
    
    /// Assignment statement
    Assignment {
        target: AstNode<Expr>,
        value: AstNode<Expr>,
    },
    
    /// While loop
    While {
        condition: AstNode<Expr>,
        body: Box<AstNode<Stmt>>,
    },
    
    /// For loop
    For {
        variable: String,
        iterable: AstNode<Expr>,
        body: Box<AstNode<Stmt>>,
    },
    
    /// Loop
    Loop {
        body: Box<AstNode<Stmt>>,
    },
    
    /// Block statement
    Block(Vec<AstNode<Stmt>>),
}

/// Function parameter
#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    pub name: String,
    pub param_type: Type,
    pub default_value: Option<AstNode<Expr>>,
}

/// Generic type parameter
#[derive(Debug, Clone, PartialEq)]
pub struct GenericParam {
    pub name: String,
    pub bounds: Vec<String>, // Trait bounds
}

/// Function declaration
#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: String,
    pub generic_params: Vec<GenericParam>,
    pub params: Vec<Parameter>,
    pub return_type: Option<Type>,
    pub body: Box<AstNode<Stmt>>,
    pub is_async: bool,
    pub is_public: bool,
}

/// Struct declaration
#[derive(Debug, Clone, PartialEq)]
pub struct StructDecl {
    pub name: String,
    pub generic_params: Vec<GenericParam>,
    pub fields: Vec<StructField>,
    pub is_public: bool,
}

/// Struct field
#[derive(Debug, Clone, PartialEq)]
pub struct StructField {
    pub name: String,
    pub field_type: Type,
    pub is_public: bool,
}

/// Enum declaration
#[derive(Debug, Clone, PartialEq)]
pub struct EnumDecl {
    pub name: String,
    pub generic_params: Vec<GenericParam>,
    pub variants: Vec<EnumVariant>,
    pub is_public: bool,
}

/// Enum variant
#[derive(Debug, Clone, PartialEq)]
pub struct EnumVariant {
    pub name: String,
    pub fields: Vec<Type>, // Tuple variants or unit variants
}

/// Trait declaration
#[derive(Debug, Clone, PartialEq)]
pub struct TraitDecl {
    pub name: String,
    pub generic_params: Vec<GenericParam>,
    pub methods: Vec<TraitMethod>,
    pub is_public: bool,
}

/// Trait method signature
#[derive(Debug, Clone, PartialEq)]
pub struct TraitMethod {
    pub name: String,
    pub params: Vec<Parameter>,
    pub return_type: Option<Type>,
    pub default_impl: Option<Box<AstNode<Stmt>>>,
}

/// Implementation block
#[derive(Debug, Clone, PartialEq)]
pub struct ImplBlock {
    pub trait_name: Option<String>, // None for inherent impl
    pub type_name: String,
    pub generic_params: Vec<GenericParam>,
    pub methods: Vec<Function>,
}

/// Use/import declaration
#[derive(Debug, Clone, PartialEq)]
pub struct UseDecl {
    pub path: Vec<String>,
    pub alias: Option<String>,
    pub items: Option<Vec<String>>, // For use module::{item1, item2}
}

/// Module declaration
#[derive(Debug, Clone, PartialEq)]
pub struct ModuleDecl {
    pub name: String,
    pub items: Vec<AstNode<Item>>,
    pub is_public: bool,
}

/// Top-level items
#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    Function(Function),
    Struct(StructDecl),
    Enum(EnumDecl),
    Trait(TraitDecl),
    Impl(ImplBlock),
    Use(UseDecl),
    Module(ModuleDecl),
}

/// Complete AST for a file/module
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub items: Vec<AstNode<Item>>,
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Int => write!(f, "Int"),
            Type::Float => write!(f, "Float"),
            Type::Bool => write!(f, "Bool"),
            Type::Char => write!(f, "Char"),
            Type::String => write!(f, "String"),
            Type::Unit => write!(f, "()"),
            Type::TypeVar(name) => write!(f, "{}", name),
            Type::Function { params, return_type } => {
                write!(f, "(")?;
                for (i, param) in params.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", param)?;
                }
                write!(f, ") -> {}", return_type)
            },
            Type::Array { element_type, size } => {
                if let Some(s) = size {
                    write!(f, "[{}; {}]", element_type, s)
                } else {
                    write!(f, "[{}]", element_type)
                }
            },
            Type::Tuple(types) => {
                write!(f, "(")?;
                for (i, t) in types.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", t)?;
                }
                write!(f, ")")
            },
            Type::Option(inner) => write!(f, "Option<{}>", inner),
            Type::Result { ok_type, err_type } => write!(f, "Result<{}, {}>", ok_type, err_type),
            Type::UserDefined { name, type_args } => {
                write!(f, "{}", name)?;
                if !type_args.is_empty() {
                    write!(f, "<")?;
                    for (i, arg) in type_args.iter().enumerate() {
                        if i > 0 { write!(f, ", ")?; }
                        write!(f, "{}", arg)?;
                    }
                    write!(f, ">")?;
                }
                Ok(())
            },
            Type::Reference { mutable, inner } => {
                if *mutable {
                    write!(f, "&mut {}", inner)
                } else {
                    write!(f, "&{}", inner)
                }
            },
        }
    }
}

/// Visitor trait for AST traversal
pub trait Visitor<T> {
    fn visit_program(&mut self, program: &Program) -> T;
    fn visit_item(&mut self, item: &AstNode<Item>) -> T;
    fn visit_function(&mut self, function: &Function) -> T;
    fn visit_stmt(&mut self, stmt: &AstNode<Stmt>) -> T;
    fn visit_expr(&mut self, expr: &AstNode<Expr>) -> T;
    fn visit_pattern(&mut self, pattern: &AstNode<Pattern>) -> T;
    fn visit_type(&mut self, type_: &Type) -> T;
}