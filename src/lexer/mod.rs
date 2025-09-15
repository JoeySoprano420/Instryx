//! Instryx Language Lexer
//! 
//! The lexer is responsible for breaking down source code into tokens
//! that can be processed by the parser.

use std::collections::HashMap;
use std::fmt;

/// Position information for error reporting
#[derive(Debug, Clone, PartialEq)]
pub struct Position {
    pub line: usize,
    pub column: usize,
    pub offset: usize,
}

impl Position {
    pub fn new() -> Self {
        Position { line: 1, column: 1, offset: 0 }
    }
    
    pub fn advance(&mut self, ch: char) {
        if ch == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        self.offset += ch.len_utf8();
    }
}

/// Token types in the Instryx language
#[derive(Debug, Clone, PartialEq)]
pub enum TokenType {
    // Literals
    IntLiteral(i64),
    FloatLiteral(f64),
    StringLiteral(String),
    CharLiteral(char),
    BoolLiteral(bool),
    
    // Identifiers
    Identifier(String),
    
    // Keywords
    Fn,
    Let,
    Mut,
    If,
    Else,
    Match,
    For,
    While,
    Loop,
    Break,
    Continue,
    Return,
    True,
    False,
    Async,
    Await,
    Pub,
    Use,
    Mod,
    Struct,
    Enum,
    Trait,
    Impl,
    Self_,
    Super,
    
    // Operators
    Plus,
    Minus,
    Multiply,
    Divide,
    Modulo,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    And,
    Or,
    Not,
    BitAnd,
    BitOr,
    BitXor,
    BitNot,
    LeftShift,
    RightShift,
    
    // Assignment
    Assign,
    PlusAssign,
    MinusAssign,
    MultiplyAssign,
    DivideAssign,
    
    // Punctuation
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    LeftBracket,
    RightBracket,
    Comma,
    Semicolon,
    Colon,
    DoubleColon,
    Dot,
    Arrow,
    FatArrow,
    Question,
    At,
    Hash,
    Dollar,
    
    // Special
    Newline,
    Eof,
    Invalid(String),
}

/// A token with position information
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub token_type: TokenType,
    pub position: Position,
    pub lexeme: String,
}

impl Token {
    pub fn new(token_type: TokenType, position: Position, lexeme: String) -> Self {
        Token { token_type, position, lexeme }
    }
}

/// The main lexer struct
pub struct Lexer {
    input: Vec<char>,
    current: usize,
    position: Position,
    keywords: HashMap<String, TokenType>,
}

impl Lexer {
    pub fn new(input: &str) -> Self {
        let mut keywords = HashMap::new();
        
        // Populate keywords
        keywords.insert("fn".to_string(), TokenType::Fn);
        keywords.insert("let".to_string(), TokenType::Let);
        keywords.insert("mut".to_string(), TokenType::Mut);
        keywords.insert("if".to_string(), TokenType::If);
        keywords.insert("else".to_string(), TokenType::Else);
        keywords.insert("match".to_string(), TokenType::Match);
        keywords.insert("for".to_string(), TokenType::For);
        keywords.insert("while".to_string(), TokenType::While);
        keywords.insert("loop".to_string(), TokenType::Loop);
        keywords.insert("break".to_string(), TokenType::Break);
        keywords.insert("continue".to_string(), TokenType::Continue);
        keywords.insert("return".to_string(), TokenType::Return);
        keywords.insert("true".to_string(), TokenType::True);
        keywords.insert("false".to_string(), TokenType::False);
        keywords.insert("async".to_string(), TokenType::Async);
        keywords.insert("await".to_string(), TokenType::Await);
        keywords.insert("pub".to_string(), TokenType::Pub);
        keywords.insert("use".to_string(), TokenType::Use);
        keywords.insert("mod".to_string(), TokenType::Mod);
        keywords.insert("struct".to_string(), TokenType::Struct);
        keywords.insert("enum".to_string(), TokenType::Enum);
        keywords.insert("trait".to_string(), TokenType::Trait);
        keywords.insert("impl".to_string(), TokenType::Impl);
        keywords.insert("self".to_string(), TokenType::Self_);
        keywords.insert("super".to_string(), TokenType::Super);
        
        Lexer {
            input: input.chars().collect(),
            current: 0,
            position: Position::new(),
            keywords,
        }
    }
    
    /// Get the current character
    fn current_char(&self) -> Option<char> {
        self.input.get(self.current).copied()
    }
    
    /// Advance to the next character
    fn advance(&mut self) -> Option<char> {
        if let Some(ch) = self.current_char() {
            self.position.advance(ch);
            self.current += 1;
            Some(ch)
        } else {
            None
        }
    }
    
    /// Peek at the next character without advancing
    fn peek(&self) -> Option<char> {
        self.input.get(self.current + 1).copied()
    }
    
    /// Skip whitespace characters
    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.current_char() {
            if ch.is_whitespace() && ch != '\n' {
                self.advance();
            } else {
                break;
            }
        }
    }
    
    /// Read a number literal
    fn read_number(&mut self) -> Token {
        let start_pos = self.position.clone();
        let mut lexeme = String::new();
        let mut is_float = false;
        
        while let Some(ch) = self.current_char() {
            if ch.is_ascii_digit() {
                lexeme.push(ch);
                self.advance();
            } else if ch == '.' && !is_float && self.peek().map_or(false, |c| c.is_ascii_digit()) {
                is_float = true;
                lexeme.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        
        let token_type = if is_float {
            match lexeme.parse::<f64>() {
                Ok(value) => TokenType::FloatLiteral(value),
                Err(_) => TokenType::Invalid(format!("Invalid float literal: {}", lexeme)),
            }
        } else {
            match lexeme.parse::<i64>() {
                Ok(value) => TokenType::IntLiteral(value),
                Err(_) => TokenType::Invalid(format!("Invalid integer literal: {}", lexeme)),
            }
        };
        
        Token::new(token_type, start_pos, lexeme)
    }
    
    /// Read a string literal
    fn read_string(&mut self) -> Token {
        let start_pos = self.position.clone();
        let quote_char = self.advance().unwrap(); // Skip opening quote
        let mut value = String::new();
        let mut lexeme = String::from(quote_char);
        
        while let Some(ch) = self.current_char() {
            lexeme.push(ch);
            
            if ch == quote_char {
                self.advance(); // Skip closing quote
                return Token::new(TokenType::StringLiteral(value), start_pos, lexeme);
            } else if ch == '\\' {
                self.advance(); // Skip backslash
                if let Some(escaped) = self.current_char() {
                    lexeme.push(escaped);
                    let escaped_char = match escaped {
                        'n' => '\n',
                        't' => '\t',
                        'r' => '\r',
                        '\\' => '\\',
                        '\'' => '\'',
                        '"' => '"',
                        _ => escaped,
                    };
                    value.push(escaped_char);
                    self.advance();
                }
            } else {
                value.push(ch);
                self.advance();
            }
        }
        
        Token::new(TokenType::Invalid("Unterminated string literal".to_string()), start_pos, lexeme)
    }
    
    /// Read an identifier or keyword
    fn read_identifier(&mut self) -> Token {
        let start_pos = self.position.clone();
        let mut lexeme = String::new();
        
        while let Some(ch) = self.current_char() {
            if ch.is_alphanumeric() || ch == '_' {
                lexeme.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        
        let token_type = self.keywords.get(&lexeme)
            .cloned()
            .unwrap_or_else(|| TokenType::Identifier(lexeme.clone()));
        
        Token::new(token_type, start_pos, lexeme)
    }
    
    /// Get the next token
    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace();
        
        let start_pos = self.position.clone();
        
        match self.current_char() {
            None => Token::new(TokenType::Eof, start_pos, "".to_string()),
            Some('\n') => {
                self.advance();
                Token::new(TokenType::Newline, start_pos, "\n".to_string())
            }
            Some(ch) if ch.is_ascii_digit() => self.read_number(),
            Some(ch) if ch.is_alphabetic() || ch == '_' => self.read_identifier(),
            Some('"') | Some('\'') => self.read_string(),
            Some(ch) => {
                let mut lexeme = String::from(ch);
                self.advance();
                
                let token_type = match ch {
                    '+' => match self.current_char() {
                        Some('=') => { self.advance(); lexeme.push('='); TokenType::PlusAssign }
                        _ => TokenType::Plus,
                    },
                    '-' => match self.current_char() {
                        Some('=') => { self.advance(); lexeme.push('='); TokenType::MinusAssign }
                        Some('>') => { self.advance(); lexeme.push('>'); TokenType::Arrow }
                        _ => TokenType::Minus,
                    },
                    '*' => match self.current_char() {
                        Some('=') => { self.advance(); lexeme.push('='); TokenType::MultiplyAssign }
                        _ => TokenType::Multiply,
                    },
                    '/' => match self.current_char() {
                        Some('=') => { self.advance(); lexeme.push('='); TokenType::DivideAssign }
                        Some('/') => {
                            // Line comment - skip to end of line
                            while let Some(c) = self.current_char() {
                                if c == '\n' { break; }
                                self.advance();
                            }
                            return self.next_token();
                        },
                        _ => TokenType::Divide,
                    },
                    '%' => TokenType::Modulo,
                    '=' => match self.current_char() {
                        Some('=') => { self.advance(); lexeme.push('='); TokenType::Equal }
                        Some('>') => { self.advance(); lexeme.push('>'); TokenType::FatArrow }
                        _ => TokenType::Assign,
                    },
                    '!' => match self.current_char() {
                        Some('=') => { self.advance(); lexeme.push('='); TokenType::NotEqual }
                        _ => TokenType::Not,
                    },
                    '<' => match self.current_char() {
                        Some('=') => { self.advance(); lexeme.push('='); TokenType::LessEqual }
                        Some('<') => { self.advance(); lexeme.push('<'); TokenType::LeftShift }
                        _ => TokenType::Less,
                    },
                    '>' => match self.current_char() {
                        Some('=') => { self.advance(); lexeme.push('='); TokenType::GreaterEqual }
                        Some('>') => { self.advance(); lexeme.push('>'); TokenType::RightShift }
                        _ => TokenType::Greater,
                    },
                    '&' => match self.current_char() {
                        Some('&') => { self.advance(); lexeme.push('&'); TokenType::And }
                        _ => TokenType::BitAnd,
                    },
                    '|' => match self.current_char() {
                        Some('|') => { self.advance(); lexeme.push('|'); TokenType::Or }
                        _ => TokenType::BitOr,
                    },
                    '^' => TokenType::BitXor,
                    '~' => TokenType::BitNot,
                    '(' => TokenType::LeftParen,
                    ')' => TokenType::RightParen,
                    '{' => TokenType::LeftBrace,
                    '}' => TokenType::RightBrace,
                    '[' => TokenType::LeftBracket,
                    ']' => TokenType::RightBracket,
                    ',' => TokenType::Comma,
                    ';' => TokenType::Semicolon,
                    ':' => match self.current_char() {
                        Some(':') => { self.advance(); lexeme.push(':'); TokenType::DoubleColon }
                        _ => TokenType::Colon,
                    },
                    '.' => TokenType::Dot,
                    '?' => TokenType::Question,
                    '@' => TokenType::At,
                    '#' => TokenType::Hash,
                    '$' => TokenType::Dollar,
                    _ => TokenType::Invalid(format!("Unexpected character: {}", ch)),
                };
                
                Token::new(token_type, start_pos, lexeme)
            }
        }
    }
    
    /// Tokenize the entire input
    pub fn tokenize(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        
        loop {
            let token = self.next_token();
            let is_eof = matches!(token.token_type, TokenType::Eof);
            tokens.push(token);
            
            if is_eof {
                break;
            }
        }
        
        tokens
    }
}

impl fmt::Display for TokenType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenType::IntLiteral(n) => write!(f, "IntLiteral({})", n),
            TokenType::FloatLiteral(n) => write!(f, "FloatLiteral({})", n),
            TokenType::StringLiteral(s) => write!(f, "StringLiteral(\"{}\")", s),
            TokenType::CharLiteral(c) => write!(f, "CharLiteral('{}')", c),
            TokenType::BoolLiteral(b) => write!(f, "BoolLiteral({})", b),
            TokenType::Identifier(s) => write!(f, "Identifier({})", s),
            _ => write!(f, "{:?}", self),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_tokens() {
        let mut lexer = Lexer::new("let x = 42;");
        let tokens = lexer.tokenize();
        
        assert_eq!(tokens.len(), 6); // let, x, =, 42, ;, EOF
        assert!(matches!(tokens[0].token_type, TokenType::Let));
        assert!(matches!(tokens[1].token_type, TokenType::Identifier(_)));
        assert!(matches!(tokens[2].token_type, TokenType::Assign));
        assert!(matches!(tokens[3].token_type, TokenType::IntLiteral(42)));
        assert!(matches!(tokens[4].token_type, TokenType::Semicolon));
        assert!(matches!(tokens[5].token_type, TokenType::Eof));
    }
    
    #[test]
    fn test_string_literals() {
        let mut lexer = Lexer::new("\"hello world\"");
        let tokens = lexer.tokenize();
        
        assert_eq!(tokens.len(), 2); // string, EOF
        match &tokens[0].token_type {
            TokenType::StringLiteral(s) => assert_eq!(s, "hello world"),
            _ => panic!("Expected string literal"),
        }
    }
    
    #[test]
    fn test_operators() {
        let mut lexer = Lexer::new("+ - * / == != <= >= && ||");
        let tokens = lexer.tokenize();
        
        let expected = vec![
            TokenType::Plus, TokenType::Minus, TokenType::Multiply, TokenType::Divide,
            TokenType::Equal, TokenType::NotEqual, TokenType::LessEqual, TokenType::GreaterEqual,
            TokenType::And, TokenType::Or, TokenType::Eof
        ];
        
        for (i, expected_type) in expected.iter().enumerate() {
            assert_eq!(std::mem::discriminant(&tokens[i].token_type), 
                      std::mem::discriminant(expected_type));
        }
    }
}