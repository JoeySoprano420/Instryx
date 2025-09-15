//! Parser module for the Instryx language
//! 
//! Converts tokens from the lexer into an Abstract Syntax Tree (AST).

use crate::lexer::{Token, TokenType};
use crate::ast::*;

/// Parser for Instryx source code
pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, current: 0 }
    }
    
    /// Parse tokens into a Program AST
    pub fn parse(&mut self) -> Result<Program, String> {
        let mut items = Vec::new();
        
        while !self.is_at_end() {
            if let Some(item) = self.parse_item()? {
                items.push(item);
            }
        }
        
        Ok(Program { items })
    }
    
    /// Parse a top-level item
    fn parse_item(&mut self) -> Result<Option<AstNode<Item>>, String> {
        // Skip newlines
        while self.match_token(&TokenType::Newline) {
            // Continue
        }
        
        if self.is_at_end() {
            return Ok(None);
        }
        
        let position = self.peek().position.clone();
        
        // Check for visibility modifier
        let is_public = self.match_token(&TokenType::Pub);
        
        match &self.peek().token_type {
            TokenType::Fn => {
                let func = self.parse_function(is_public)?;
                Ok(Some(AstNode::new(Item::Function(func), position)))
            }
            TokenType::Struct => {
                let struct_decl = self.parse_struct(is_public)?;
                Ok(Some(AstNode::new(Item::Struct(struct_decl), position)))
            }
            TokenType::Enum => {
                let enum_decl = self.parse_enum(is_public)?;
                Ok(Some(AstNode::new(Item::Enum(enum_decl), position)))
            }
            TokenType::Use => {
                let use_decl = self.parse_use()?;
                Ok(Some(AstNode::new(Item::Use(use_decl), position)))
            }
            _ => Err(format!("Unexpected token: {:?}", self.peek().token_type))
        }
    }
    
    /// Parse a function declaration
    fn parse_function(&mut self, is_public: bool) -> Result<Function, String> {
        self.consume(&TokenType::Fn, "Expected 'fn'")?;
        
        let name = match &self.advance().token_type {
            TokenType::Identifier(name) => name.clone(),
            _ => return Err("Expected function name".to_string()),
        };
        
        // TODO: Implement full function parsing
        // For now, create a minimal function structure
        Ok(Function {
            name,
            generic_params: Vec::new(),
            params: Vec::new(),
            return_type: None,
            body: Box::new(AstNode::new(
                Stmt::Block(Vec::new()),
                self.peek().position.clone()
            )),
            is_async: false,
            is_public,
        })
    }
    
    /// Parse a struct declaration  
    fn parse_struct(&mut self, is_public: bool) -> Result<StructDecl, String> {
        self.consume(&TokenType::Struct, "Expected 'struct'")?;
        
        let name = match &self.advance().token_type {
            TokenType::Identifier(name) => name.clone(),
            _ => return Err("Expected struct name".to_string()),
        };
        
        // TODO: Implement full struct parsing
        Ok(StructDecl {
            name,
            generic_params: Vec::new(),
            fields: Vec::new(),
            is_public,
        })
    }
    
    /// Parse an enum declaration
    fn parse_enum(&mut self, is_public: bool) -> Result<EnumDecl, String> {
        self.consume(&TokenType::Enum, "Expected 'enum'")?;
        
        let name = match &self.advance().token_type {
            TokenType::Identifier(name) => name.clone(),
            _ => return Err("Expected enum name".to_string()),
        };
        
        // TODO: Implement full enum parsing
        Ok(EnumDecl {
            name,
            generic_params: Vec::new(),
            variants: Vec::new(),
            is_public,
        })
    }
    
    /// Parse a use declaration
    fn parse_use(&mut self) -> Result<UseDecl, String> {
        self.consume(&TokenType::Use, "Expected 'use'")?;
        
        // TODO: Implement full use parsing
        Ok(UseDecl {
            path: vec!["placeholder".to_string()],
            alias: None,
            items: None,
        })
    }
    
    /// Helper methods for token manipulation
    fn peek(&self) -> &Token {
        &self.tokens[self.current]
    }
    
    fn advance(&mut self) -> &Token {
        if !self.is_at_end() {
            self.current += 1;
        }
        &self.tokens[self.current - 1]
    }
    
    fn is_at_end(&self) -> bool {
        matches!(self.peek().token_type, TokenType::Eof)
    }
    
    fn match_token(&mut self, token_type: &TokenType) -> bool {
        if std::mem::discriminant(&self.peek().token_type) == std::mem::discriminant(token_type) {
            self.advance();
            true
        } else {
            false
        }
    }
    
    fn consume(&mut self, token_type: &TokenType, message: &str) -> Result<&Token, String> {
        if std::mem::discriminant(&self.peek().token_type) == std::mem::discriminant(token_type) {
            Ok(self.advance())
        } else {
            Err(format!("{}: got {:?}", message, self.peek().token_type))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    
    #[test]
    fn test_basic_parsing() {
        let source = "fn main() {}";
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        
        let mut parser = Parser::new(tokens);
        let result = parser.parse();
        
        assert!(result.is_ok());
        let program = result.unwrap();
        assert_eq!(program.items.len(), 1);
    }
}