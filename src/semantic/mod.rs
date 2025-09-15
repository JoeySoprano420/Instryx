//! Semantic analysis module
//! 
//! Performs type checking, name resolution, and other semantic validation.

use crate::ast::*;
use std::collections::HashMap;

/// Semantic analyzer for the Instryx language
pub struct Analyzer {
    symbol_table: SymbolTable,
    type_checker: TypeChecker,
    errors: Vec<String>,
}

impl Analyzer {
    pub fn new() -> Self {
        Analyzer {
            symbol_table: SymbolTable::new(),
            type_checker: TypeChecker::new(),
            errors: Vec::new(),
        }
    }
    
    /// Analyze a program for semantic correctness
    pub fn analyze(&mut self, program: &Program) -> Result<(), Vec<String>> {
        // Phase 1: Build symbol table
        self.build_symbol_table(program);
        
        // Phase 2: Resolve names and check types
        self.check_types(program);
        
        // Phase 3: Check for other semantic errors
        self.check_semantics(program);
        
        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(self.errors.clone())
        }
    }
    
    fn build_symbol_table(&mut self, program: &Program) {
        for item in &program.items {
            match &item.node {
                Item::Function(func) => {
                    self.symbol_table.add_function(func.name.clone(), func);
                }
                Item::Struct(struct_decl) => {
                    self.symbol_table.add_struct(struct_decl.name.clone(), struct_decl);
                }
                Item::Enum(enum_decl) => {
                    self.symbol_table.add_enum(enum_decl.name.clone(), enum_decl);
                }
                _ => {} // Handle other items
            }
        }
    }
    
    fn check_types(&mut self, _program: &Program) {
        // TODO: Implement type checking
        // This would traverse the AST and verify that all expressions have valid types
    }
    
    fn check_semantics(&mut self, _program: &Program) {
        // TODO: Implement semantic checks
        // - Unreachable code detection
        // - Unused variable warnings
        // - Borrow checker (if implementing ownership)
        // - Lifetime analysis
    }
    
    fn add_error(&mut self, message: String) {
        self.errors.push(message);
    }
}

/// Symbol table for name resolution
#[derive(Debug)]
pub struct SymbolTable {
    scopes: Vec<Scope>,
    functions: HashMap<String, FunctionInfo>,
    types: HashMap<String, TypeInfo>,
}

impl SymbolTable {
    pub fn new() -> Self {
        let mut table = SymbolTable {
            scopes: vec![Scope::new()], // Global scope
            functions: HashMap::new(),
            types: HashMap::new(),
        };
        
        // Add built-in types and functions
        table.add_builtins();
        table
    }
    
    fn add_builtins(&mut self) {
        // Add built-in types
        self.types.insert("Int".to_string(), TypeInfo {
            name: "Int".to_string(),
            kind: TypeKind::Primitive,
        });
        self.types.insert("Float".to_string(), TypeInfo {
            name: "Float".to_string(),
            kind: TypeKind::Primitive,
        });
        self.types.insert("Bool".to_string(), TypeInfo {
            name: "Bool".to_string(),
            kind: TypeKind::Primitive,
        });
        self.types.insert("String".to_string(), TypeInfo {
            name: "String".to_string(),
            kind: TypeKind::Primitive,
        });
        
        // Add built-in functions
        self.functions.insert("println".to_string(), FunctionInfo {
            name: "println".to_string(),
            param_types: vec![Type::String],
            return_type: Type::Unit,
            is_builtin: true,
        });
    }
    
    pub fn add_function(&mut self, name: String, func: &Function) {
        let param_types = func.params.iter()
            .map(|p| p.param_type.clone())
            .collect();
            
        let return_type = func.return_type.clone()
            .unwrap_or(Type::Unit);
            
        self.functions.insert(name.clone(), FunctionInfo {
            name,
            param_types,
            return_type,
            is_builtin: false,
        });
    }
    
    pub fn add_struct(&mut self, name: String, _struct_decl: &StructDecl) {
        self.types.insert(name.clone(), TypeInfo {
            name,
            kind: TypeKind::Struct,
        });
    }
    
    pub fn add_enum(&mut self, name: String, _enum_decl: &EnumDecl) {
        self.types.insert(name.clone(), TypeInfo {
            name,
            kind: TypeKind::Enum,
        });
    }
    
    pub fn push_scope(&mut self) {
        self.scopes.push(Scope::new());
    }
    
    pub fn pop_scope(&mut self) {
        self.scopes.pop();
    }
    
    pub fn lookup_function(&self, name: &str) -> Option<&FunctionInfo> {
        self.functions.get(name)
    }
    
    pub fn lookup_type(&self, name: &str) -> Option<&TypeInfo> {
        self.types.get(name)
    }
}

#[derive(Debug)]
struct Scope {
    variables: HashMap<String, VariableInfo>,
}

impl Scope {
    fn new() -> Self {
        Scope {
            variables: HashMap::new(),
        }
    }
}

#[derive(Debug)]
struct VariableInfo {
    name: String,
    var_type: Type,
    mutable: bool,
}

#[derive(Debug)]
struct FunctionInfo {
    name: String,
    param_types: Vec<Type>,
    return_type: Type,
    is_builtin: bool,
}

#[derive(Debug)]
struct TypeInfo {
    name: String,
    kind: TypeKind,
}

#[derive(Debug)]
enum TypeKind {
    Primitive,
    Struct,
    Enum,
    Function,
}

/// Type checker for expressions and statements
pub struct TypeChecker {
    // Type inference engine
    // Constraint solver
    // Generic type instantiation
}

impl TypeChecker {
    pub fn new() -> Self {
        TypeChecker {}
    }
    
    /// Infer the type of an expression
    pub fn infer_type(&mut self, _expr: &AstNode<Expr>) -> Result<Type, String> {
        // TODO: Implement type inference
        // This would use algorithms like Hindley-Milner for type inference
        Ok(Type::Unit)
    }
    
    /// Check if two types are compatible
    pub fn types_compatible(&self, _type1: &Type, _type2: &Type) -> bool {
        // TODO: Implement type compatibility checking
        true
    }
    
    /// Unify two types (for generics)
    pub fn unify(&mut self, _type1: &Type, _type2: &Type) -> Result<Type, String> {
        // TODO: Implement type unification for generic type inference
        Ok(Type::Unit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_symbol_table_creation() {
        let table = SymbolTable::new();
        
        // Should have built-in types
        assert!(table.lookup_type("Int").is_some());
        assert!(table.lookup_type("String").is_some());
        
        // Should have built-in functions
        assert!(table.lookup_function("println").is_some());
    }
    
    #[test]
    fn test_analyzer_creation() {
        let analyzer = Analyzer::new();
        assert!(analyzer.errors.is_empty());
    }
}