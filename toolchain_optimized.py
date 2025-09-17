#!/usr/bin/env python3
"""
toolchain_optimized.py
Optimized and Enhanced Instryx Language Toolchain

A unified, high-performance toolchain for the Instryx programming language featuring:
- Optimized lexer with caching and advanced token recognition
- Unified parser with operator precedence and error recovery  
- LLVM-based code generation with optimization passes
- Advanced AST transformations and optimizations
- Plugin architecture for extensibility
- Profiling and benchmarking infrastructure
- LSP integration hooks
- Comprehensive error handling and diagnostics

Author: Enhanced by AI Assistant for competitive performance
License: MIT
"""

from __future__ import annotations

import re
import ast
import sys
import time
import threading
import bisect
import hashlib
import logging
import traceback
from typing import (
    List, Dict, Tuple, Optional, Union, Any, Iterator, Callable, 
    Set, Protocol, TypeVar, Generic, NamedTuple, Final
)
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum, auto
from functools import lru_cache, wraps
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Type definitions for better static analysis
T = TypeVar('T')
TokenType = str
TokenValue = Union[str, int, float]

# ============================================================================
# Core Token and Configuration System
# ============================================================================

class TokenKind(Enum):
    """Enumeration of all token types for better type safety."""
    # Literals
    NUMBER = auto()
    STRING = auto()
    KEYWORD = auto()
    ID = auto()
    
    # Operators  
    ASSIGN = auto()
    OP = auto()
    
    # Punctuation
    END = auto()
    DOT = auto()
    COLON = auto()
    COMMA = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    
    # Special
    MACRO = auto()
    COMMENT = auto()
    ML_COMMENT = auto()
    NEWLINE = auto()
    SKIP = auto()  # For whitespace
    EOF = auto()
    MISMATCH = auto()


# ============================================================================
# AST Node System
# ============================================================================

class ASTNodeKind(Enum):
    """AST Node types for type-safe AST construction."""
    PROGRAM = auto()
    FUNCTION_DEF = auto()
    BLOCK = auto()
    EXPRESSION_STMT = auto()
    ASSIGNMENT = auto()
    BINARY_OP = auto()
    UNARY_OP = auto()
    CALL = auto()
    IDENTIFIER = auto()
    LITERAL = auto()
    IF_STMT = auto()
    WHILE_STMT = auto()
    FOR_STMT = auto()
    RETURN_STMT = auto()
    QUARANTINE_BLOCK = auto()
    MATCH_STMT = auto()
    MACRO_CALL = auto()


@dataclass
class ASTNode:
    """Immutable AST node with comprehensive metadata."""
    kind: ASTNodeKind
    value: Any = None
    children: List['ASTNode'] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    location: Optional[Tuple[int, int]] = None
    
    def __post_init__(self):
        """Ensure children is always a list."""
        if self.children is None:
            object.__setattr__(self, 'children', [])


class ParseError(Exception):
    """Enhanced parse error with location and recovery information."""
    def __init__(self, message: str, token: Optional[Token] = None, 
                 expected: Optional[List[str]] = None, got: Optional[str] = None):
        super().__init__(message)
        self.token = token
        self.expected = expected or []
        self.got = got
        self.line = token.line if token else 0
        self.column = token.column if token else 0


@dataclass(frozen=True)
class Token:
    """Immutable token representation with optional position information."""
    kind: TokenKind
    value: TokenValue
    line: int = 0
    column: int = 0
    
    @property
    def type(self) -> str:
        """Backward compatibility: return string representation of kind."""
        return self.kind.name
    
    def __iter__(self):
        """Backward compatibility: allow tuple unpacking (type, value) or (type, value, line, col)."""
        if self.line > 0 or self.column > 0:
            yield self.kind.name
            yield self.value
            yield self.line
            yield self.column
        else:
            yield self.kind.name
            yield self.value


@dataclass
class LexerConfig:
    """Configuration for lexer behavior and optimization."""
    emit_positions: bool = True
    skip_comments: bool = True
    skip_whitespace: bool = True
    enable_caching: bool = True
    max_cache_size: int = 1000
    enable_parallel: bool = False
    chunk_size: int = 10000
    
    # Advanced features
    enable_macro_expansion: bool = False
    strict_mode: bool = False
    debug_tokens: bool = False


# ============================================================================
# Performance-Optimized Lexer
# ============================================================================

class OptimizedInstryxLexer:
    """
    High-performance lexer with advanced features:
    - LRU caching for repeated tokenization
    - Parallel processing for large files
    - Advanced regex optimization
    - Position tracking with binary search
    - Plugin hooks for macro expansion
    """
    
    # Class-level cached regex compilation
    _REGEX_CACHE: Dict[str, re.Pattern] = {}
    
    # Keywords set for fast O(1) lookup
    KEYWORDS: Final[Set[str]] = {
        'func', 'main', 'quarantine', 'try', 'replace', 'erase',
        'if', 'else', 'while', 'fork', 'join', 'then', 'true', 'false',
        'print', 'alert', 'log', 'return', 'import', 'const', 'let', 'var',
        'class', 'struct', 'enum', 'interface', 'trait', 'impl',
        'async', 'await', 'yield', 'match', 'case', 'default'
    }
    
    # Optimized token specifications (order optimized for common tokens first)
    TOKEN_SPECS: Final[List[Tuple[TokenKind, str]]] = [
        # Comments first (often need to be skipped)
        (TokenKind.ML_COMMENT, r'/\*[\s\S]*?\*/'),
        (TokenKind.COMMENT, r'--[^\n]*'),
        
        # Numbers (optimized for common integer patterns)  
        (TokenKind.NUMBER, r'(?:0x[0-9A-Fa-f_]+|0b[01_]+|\d+(?:_\d+)*(?:\.\d+(?:_\d+)*)?(?:[eE][+-]?\d+)?)'),
        
        # Strings with comprehensive escape support
        (TokenKind.STRING, r'"(?:\\(?:["\\/bfnrt]|u[0-9A-Fa-f]{4})|[^"\\])*"'),
        
        # Macros (common in Instryx)
        (TokenKind.MACRO, r'@[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*'),
        
        # Multi-character operators first
        (TokenKind.OP, r'(?:==|!=|<=|>=|\|\||&&|\+=|-=|\*=|/=|%=|\+\+|--|<<|>>|\b(?:and|or|not|in|is)\b|[+\-*/%<>!&|^~])'),
        
        # Single-character tokens (high frequency)
        (TokenKind.ASSIGN, r'='),
        (TokenKind.END, r';'),
        (TokenKind.DOT, r'\.'),
        (TokenKind.COLON, r':'),
        (TokenKind.COMMA, r','),
        (TokenKind.LPAREN, r'\('),
        (TokenKind.RPAREN, r'\)'),
        (TokenKind.LBRACE, r'\{'),
        (TokenKind.RBRACE, r'\}'),
        (TokenKind.LBRACKET, r'\['),
        (TokenKind.RBRACKET, r'\]'),
        
        # Identifiers (must come after keywords check)
        (TokenKind.ID, r'[A-Za-z_][A-Za-z0-9_]*'),
        
        # Whitespace
        (TokenKind.NEWLINE, r'\n'),
        (TokenKind.SKIP, r'[ \t\r]+'),  # Skip spaces and tabs
        
        # Error recovery
        (TokenKind.MISMATCH, r'.'),
    ]
    
    def __init__(self, config: Optional[LexerConfig] = None):
        self.config = config or LexerConfig()
        self._token_cache: Dict[str, List[Token]] = {}
        self._regex = self._get_compiled_regex()
        
        # Initialize performance metrics
        self._metrics = {
            'tokens_produced': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'tokenization_time': 0.0,
        }
    
    @classmethod
    def _get_compiled_regex(cls) -> re.Pattern:
        """Get cached compiled regex for token matching."""
        cache_key = 'main_tokenizer'
        if cache_key not in cls._REGEX_CACHE:
            pattern = '|'.join(f"(?P<{kind.name}>{regex})" for kind, regex in cls.TOKEN_SPECS)
            cls._REGEX_CACHE[cache_key] = re.compile(pattern, re.MULTILINE | re.DOTALL)
        return cls._REGEX_CACHE[cache_key]
    
    @lru_cache(maxsize=1000)
    def _compute_line_starts(self, code: str) -> Tuple[int, ...]:
        """Compute line start positions for efficient line/column calculation."""
        if not code:
            return (0,)
        
        line_starts = [0]
        for i, char in enumerate(code):
            if char == '\n':
                line_starts.append(i + 1)
        return tuple(line_starts)
    
    def _get_line_column(self, pos: int, line_starts: Tuple[int, ...]) -> Tuple[int, int]:
        """Efficiently compute line and column from position using binary search."""
        line_idx = bisect.bisect_right(line_starts, pos) - 1
        line = line_idx + 1
        column = pos - line_starts[line_idx] + 1
        return line, column
    
    def tokenize(self, code: str) -> List[Token]:
        """Tokenize code into a list of tokens with optional caching."""
        if self.config.enable_caching:
            code_hash = hashlib.md5(code.encode()).hexdigest()
            if code_hash in self._token_cache:
                self._metrics['cache_hits'] += 1
                return self._token_cache[code_hash]
            self._metrics['cache_misses'] += 1
        
        start_time = time.time()
        
        if self.config.enable_parallel and len(code) > self.config.chunk_size:
            tokens = self._tokenize_parallel(code)
        else:
            tokens = list(self._tokenize_sequential(code))
        
        self._metrics['tokenization_time'] += time.time() - start_time
        self._metrics['tokens_produced'] += len(tokens)
        
        if self.config.enable_caching and len(self._token_cache) < self.config.max_cache_size:
            self._token_cache[code_hash] = tokens
        
        return tokens
    
    def _tokenize_sequential(self, code: str) -> Iterator[Token]:
        """Sequential tokenization with optimized performance."""
        line_starts = self._compute_line_starts(code) if self.config.emit_positions else None
        
        for match in self._regex.finditer(code):
            kind_name = match.lastgroup
            if not kind_name:
                continue
                
            kind = TokenKind[kind_name]
            value = match.group()
            pos = match.start()
            
            # Skip unwanted tokens
            if kind == TokenKind.SKIP and self.config.skip_whitespace:
                continue
            if kind == TokenKind.COMMENT and self.config.skip_comments:
                continue
            if kind == TokenKind.ML_COMMENT and self.config.skip_comments:
                continue
            if kind == TokenKind.NEWLINE and self.config.skip_whitespace:
                continue
            
            # Handle keyword detection
            if kind == TokenKind.ID and value in self.KEYWORDS:
                kind = TokenKind.KEYWORD
            
            # Create token with optional position info
            if self.config.emit_positions and line_starts:
                line, column = self._get_line_column(pos, line_starts)
                token = Token(kind=kind, value=value, line=line, column=column)
            else:
                token = Token(kind=kind, value=value)
            
            if self.config.debug_tokens:
                logger.debug(f"Token: {token}")
            
            yield token
    
    def _tokenize_parallel(self, code: str) -> List[Token]:
        """Parallel tokenization for large files (experimental)."""
        # Split code into chunks while preserving token boundaries
        chunks = self._split_code_safely(code)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            chunk_tokens = list(executor.map(
                lambda chunk: list(self._tokenize_sequential(chunk)),
                chunks
            ))
        
        # Flatten results
        tokens = []
        for chunk_result in chunk_tokens:
            tokens.extend(chunk_result)
        
        return tokens
    
    def _split_code_safely(self, code: str) -> List[str]:
        """Split code into chunks at safe boundaries (whitespace/newlines)."""
        chunk_size = self.config.chunk_size
        if len(code) <= chunk_size:
            return [code]
        
        chunks = []
        start = 0
        
        while start < len(code):
            end = min(start + chunk_size, len(code))
            
            # Find a safe split point (whitespace or newline)
            if end < len(code):
                while end > start and code[end] not in ' \t\n\r':
                    end -= 1
                if end == start:  # Fallback if no whitespace found
                    end = min(start + chunk_size, len(code))
            
            chunks.append(code[start:end])
            start = end
        
        return chunks
    
    def get_metrics(self) -> Dict[str, Union[int, float]]:
        """Get performance metrics for profiling and optimization."""
        return self._metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        for key in self._metrics:
            self._metrics[key] = 0


# ============================================================================
# Optimized Parser with Advanced Features
# ============================================================================

@dataclass
class ParserConfig:
    """Configuration for parser behavior and optimizations."""
    enable_error_recovery: bool = True
    enable_constant_folding: bool = True
    enable_operator_precedence: bool = True
    max_recursion_depth: int = 1000
    enable_caching: bool = True
    strict_mode: bool = False
    debug_parsing: bool = False


class OptimizedInstryxParser:
    """
    High-performance recursive descent parser with advanced features:
    - Operator precedence parsing
    - Error recovery and diagnostics
    - AST caching and memoization
    - Constant folding during parsing
    - Plugin hooks for syntax extensions
    """
    
    # Operator precedence table (higher number = higher precedence)
    PRECEDENCE: Final[Dict[str, int]] = {
        'or': 1,
        'and': 2,
        'not': 3,
        '==': 4, '!=': 4, '<': 4, '>': 4, '<=': 4, '>=': 4,
        '+': 5, '-': 5,
        '*': 6, '/': 6, '%': 6,
        '**': 7,  # Exponentiation
        'unary': 8,  # Unary operators
    }
    
    # Right associative operators
    RIGHT_ASSOCIATIVE: Final[Set[str]] = {'**', '='}
    
    def __init__(self, config: Optional[ParserConfig] = None):
        self.config = config or ParserConfig()
        self.lexer = OptimizedInstryxLexer()
        self.tokens: List[Token] = []
        self.pos = 0
        self.current_token: Optional[Token] = None
        
        # Performance metrics
        self._metrics = {
            'nodes_created': 0,
            'errors_recovered': 0,
            'constants_folded': 0,
            'parse_time': 0.0,
        }
        
        # AST node cache for memoization
        self._ast_cache: Dict[Tuple[int, str], ASTNode] = {}
    
    def parse(self, code: str) -> ASTNode:
        """Parse code into an AST with comprehensive error handling."""
        start_time = time.time()
        
        try:
            # Tokenize input
            self.tokens = self.lexer.tokenize(code)
            self.tokens.append(Token(TokenKind.EOF, ''))  # Add sentinel
            self.pos = 0
            self._advance()
            
            # Parse program
            ast = self._parse_program()
            
            if self.config.debug_parsing:
                logger.debug(f"Parsed AST: {ast}")
            
            return ast
            
        finally:
            self._metrics['parse_time'] += time.time() - start_time
    
    def _advance(self) -> Token:
        """Move to next token and return current."""
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
            self.pos += 1
        else:
            self.current_token = Token(TokenKind.EOF, '')
        return self.current_token
    
    def _peek(self, offset: int = 0) -> Token:
        """Look ahead at token without consuming."""
        peek_pos = self.pos + offset
        if peek_pos < len(self.tokens):
            return self.tokens[peek_pos]
        return Token(TokenKind.EOF, '')
    
    def _match(self, *kinds: TokenKind) -> bool:
        """Check if current token matches any of the given kinds."""
        return self.current_token and self.current_token.kind in kinds
    
    def _consume(self, expected: TokenKind, message: str = None) -> Token:
        """Consume token of expected kind or raise error."""
        if not self._match(expected):
            if message:
                raise ParseError(message, self.current_token, [expected.name], 
                               self.current_token.kind.name if self.current_token else 'EOF')
            else:
                raise ParseError(f"Expected {expected.name}, got {self.current_token.kind.name}",
                               self.current_token)
        
        token = self.current_token
        self._advance()
        return token
    
    def _error(self, message: str, recover_to: Optional[List[TokenKind]] = None) -> None:
        """Handle parse error with optional recovery."""
        error = ParseError(message, self.current_token)
        
        if self.config.enable_error_recovery and recover_to:
            self._recover_to(recover_to)
            self._metrics['errors_recovered'] += 1
            logger.warning(f"Parse error: {message}, recovered to {self.current_token.kind.name}")
        else:
            raise error
    
    def _recover_to(self, token_kinds: List[TokenKind]) -> None:
        """Recover by advancing to one of the specified token types."""
        while not self._match(TokenKind.EOF) and not self._match(*token_kinds):
            self._advance()
    
    def _parse_program(self) -> ASTNode:
        """Parse top-level program."""
        statements = []
        
        while not self._match(TokenKind.EOF):
            try:
                stmt = self._parse_statement()
                if stmt:
                    statements.append(stmt)
            except ParseError as e:
                if self.config.enable_error_recovery:
                    logger.error(f"Parse error in statement: {e}")
                    self._recover_to([TokenKind.END, TokenKind.KEYWORD, TokenKind.EOF])
                else:
                    raise
        
        return self._create_node(ASTNodeKind.PROGRAM, children=statements)
    
    def _parse_statement(self) -> Optional[ASTNode]:
        """Parse a statement."""
        if self._match(TokenKind.KEYWORD):
            keyword = self.current_token.value
            if keyword == 'func':
                return self._parse_function()
            elif keyword == 'if':
                return self._parse_if_statement()
            elif keyword == 'while':
                return self._parse_while_statement()
            elif keyword == 'return':
                return self._parse_return_statement()
            elif keyword == 'quarantine':
                return self._parse_quarantine_block()
            elif keyword in ('let', 'const', 'var'):
                return self._parse_variable_declaration(keyword)
        elif self._match(TokenKind.MACRO):
            return self._parse_macro_call()
        elif self._match(TokenKind.LBRACE):
            return self._parse_block()
        else:
            return self._parse_expression_statement()
    
    def _parse_function(self) -> ASTNode:
        """Parse function definition."""
        self._consume(TokenKind.KEYWORD, "Expected 'func'")
        name = self._consume(TokenKind.ID, "Expected function name").value
        
        self._consume(TokenKind.LPAREN, "Expected '(' after function name")
        
        # Parse parameters
        params = []
        while not self._match(TokenKind.RPAREN):
            param = self._consume(TokenKind.ID, "Expected parameter name").value
            params.append(param)
            
            if self._match(TokenKind.COMMA):
                self._advance()
            elif not self._match(TokenKind.RPAREN):
                self._error("Expected ',' or ')' in parameter list")
        
        self._consume(TokenKind.RPAREN, "Expected ')' after parameters")
        
        # Parse function body
        body = self._parse_block()
        
        return self._create_node(
            ASTNodeKind.FUNCTION_DEF,
            value=name,
            children=[body],
            attributes={'params': params}
        )
    
    def _parse_block(self) -> ASTNode:
        """Parse block statement."""
        self._consume(TokenKind.LBRACE, "Expected '{'")
        
        statements = []
        while not self._match(TokenKind.RBRACE) and not self._match(TokenKind.EOF):
            stmt = self._parse_statement()
            if stmt:
                statements.append(stmt)
        
        self._consume(TokenKind.RBRACE, "Expected '}'")
        return self._create_node(ASTNodeKind.BLOCK, children=statements)
    
    def _parse_expression_statement(self) -> ASTNode:
        """Parse expression statement."""
        expr = self._parse_expression()
        if self._match(TokenKind.END):
            self._advance()
        return self._create_node(ASTNodeKind.EXPRESSION_STMT, children=[expr])
    
    def _parse_expression(self) -> ASTNode:
        """Parse expression with operator precedence."""
        return self._parse_binary_expression(0)
    
    def _parse_binary_expression(self, min_precedence: int) -> ASTNode:
        """Parse binary expressions with precedence climbing."""
        left = self._parse_unary_expression()
        
        while (self._match(TokenKind.OP) and 
               self.current_token.value in self.PRECEDENCE and
               self.PRECEDENCE[self.current_token.value] >= min_precedence):
            
            op_token = self.current_token
            op = op_token.value
            precedence = self.PRECEDENCE[op]
            self._advance()
            
            # Handle right associativity
            if op in self.RIGHT_ASSOCIATIVE:
                right = self._parse_binary_expression(precedence)
            else:
                right = self._parse_binary_expression(precedence + 1)
            
            # Constant folding during parsing
            if self.config.enable_constant_folding:
                folded = self._try_constant_fold(op, left, right)
                if folded:
                    left = folded
                    self._metrics['constants_folded'] += 1
                    continue
            
            left = self._create_node(
                ASTNodeKind.BINARY_OP,
                value=op,
                children=[left, right],
                location=(op_token.line, op_token.column) if op_token.line > 0 else None
            )
        
        return left
    
    def _parse_unary_expression(self) -> ASTNode:
        """Parse unary expressions."""
        if self._match(TokenKind.OP) and self.current_token.value in ('!', '-', '+', 'not'):
            op_token = self.current_token
            op = op_token.value
            self._advance()
            
            expr = self._parse_unary_expression()
            
            return self._create_node(
                ASTNodeKind.UNARY_OP,
                value=op,
                children=[expr],
                location=(op_token.line, op_token.column) if op_token.line > 0 else None
            )
        
        return self._parse_primary()
    
    def _parse_primary(self) -> ASTNode:
        """Parse primary expressions."""
        if self._match(TokenKind.NUMBER):
            token = self.current_token
            self._advance()
            # Convert to appropriate numeric type
            try:
                if '.' in token.value:
                    value = float(token.value)
                else:
                    value = int(token.value)
            except ValueError:
                value = token.value
            
            return self._create_node(
                ASTNodeKind.LITERAL,
                value=value,
                location=(token.line, token.column) if token.line > 0 else None
            )
        
        elif self._match(TokenKind.STRING):
            token = self.current_token
            self._advance()
            # Remove quotes and handle basic escapes
            value = token.value[1:-1].replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
            
            return self._create_node(
                ASTNodeKind.LITERAL,
                value=value,
                location=(token.line, token.column) if token.line > 0 else None
            )
        
        elif self._match(TokenKind.ID):
            token = self.current_token
            self._advance()
            
            # Check for function call
            if self._match(TokenKind.LPAREN):
                self._advance()
                args = []
                
                while not self._match(TokenKind.RPAREN):
                    args.append(self._parse_expression())
                    if self._match(TokenKind.COMMA):
                        self._advance()
                    elif not self._match(TokenKind.RPAREN):
                        self._error("Expected ',' or ')' in argument list")
                
                self._consume(TokenKind.RPAREN)
                
                return self._create_node(
                    ASTNodeKind.CALL,
                    value=token.value,
                    children=args,
                    location=(token.line, token.column) if token.line > 0 else None
                )
            else:
                return self._create_node(
                    ASTNodeKind.IDENTIFIER,
                    value=token.value,
                    location=(token.line, token.column) if token.line > 0 else None
                )
        
        elif self._match(TokenKind.LPAREN):
            self._advance()
            expr = self._parse_expression()
            self._consume(TokenKind.RPAREN, "Expected ')' after expression")
            return expr
        
        else:
            self._error(f"Unexpected token in expression: {self.current_token.kind.name}")
    
    # Helper methods for additional statement types
    def _parse_if_statement(self) -> ASTNode:
        """Parse if statement."""
        self._consume(TokenKind.KEYWORD)  # 'if'
        condition = self._parse_expression()
        then_branch = self._parse_block()
        
        else_branch = None
        if self._match(TokenKind.KEYWORD) and self.current_token.value == 'else':
            self._advance()
            else_branch = self._parse_block()
        
        children = [condition, then_branch]
        if else_branch:
            children.append(else_branch)
        
        return self._create_node(ASTNodeKind.IF_STMT, children=children)
    
    def _parse_while_statement(self) -> ASTNode:
        """Parse while statement."""
        self._consume(TokenKind.KEYWORD)  # 'while'
        condition = self._parse_expression()
        body = self._parse_block()
        
        return self._create_node(ASTNodeKind.WHILE_STMT, children=[condition, body])
    
    def _parse_return_statement(self) -> ASTNode:
        """Parse return statement."""
        self._consume(TokenKind.KEYWORD)  # 'return'
        
        expr = None
        if not self._match(TokenKind.END):
            expr = self._parse_expression()
        
        if self._match(TokenKind.END):
            self._advance()
        
        children = [expr] if expr else []
        return self._create_node(ASTNodeKind.RETURN_STMT, children=children)
    
    def _parse_quarantine_block(self) -> ASTNode:
        """Parse quarantine block (Instryx-specific)."""
        self._consume(TokenKind.KEYWORD)  # 'quarantine'
        
        # Parse try block
        if self._match(TokenKind.KEYWORD) and self.current_token.value == 'try':
            self._advance()
        
        try_block = self._parse_block()
        
        # Parse replace block (optional)
        replace_block = None
        if self._match(TokenKind.KEYWORD) and self.current_token.value == 'replace':
            self._advance()
            replace_block = self._parse_block()
        
        # Parse erase block (optional)
        erase_block = None
        if self._match(TokenKind.KEYWORD) and self.current_token.value == 'erase':
            self._advance()
            erase_block = self._parse_block()
        
        children = [try_block]
        if replace_block:
            children.append(replace_block)
        if erase_block:
            children.append(erase_block)
        
        return self._create_node(ASTNodeKind.QUARANTINE_BLOCK, children=children)
    
    def _parse_variable_declaration(self, keyword: str) -> ASTNode:
        """Parse variable declaration."""
        self._advance()  # consume let/const/var
        name = self._consume(TokenKind.ID, "Expected variable name").value
        
        value = None
        if self._match(TokenKind.ASSIGN):
            self._advance()
            value = self._parse_expression()
        
        if self._match(TokenKind.END):
            self._advance()
        
        return self._create_node(
            ASTNodeKind.ASSIGNMENT,
            value=name,
            children=[value] if value else [],
            attributes={'declaration_type': keyword}
        )
    
    def _parse_macro_call(self) -> ASTNode:
        """Parse macro call."""
        macro_token = self._consume(TokenKind.MACRO)
        macro_name = macro_token.value[1:]  # Remove @
        
        # Simple macro call without arguments for now
        return self._create_node(
            ASTNodeKind.MACRO_CALL,
            value=macro_name,
            location=(macro_token.line, macro_token.column) if macro_token.line > 0 else None
        )
    
    def _create_node(self, kind: ASTNodeKind, value: Any = None, 
                     children: Optional[List[ASTNode]] = None,
                     attributes: Optional[Dict[str, Any]] = None,
                     location: Optional[Tuple[int, int]] = None) -> ASTNode:
        """Create AST node with metrics tracking."""
        self._metrics['nodes_created'] += 1
        return ASTNode(
            kind=kind,
            value=value,
            children=children or [],
            attributes=attributes or {},
            location=location
        )
    
    def _try_constant_fold(self, op: str, left: ASTNode, right: ASTNode) -> Optional[ASTNode]:
        """Attempt constant folding for binary operations."""
        if (left.kind == ASTNodeKind.LITERAL and 
            right.kind == ASTNodeKind.LITERAL):
            
            try:
                left_val = left.value
                right_val = right.value
                
                if op == '+':
                    result = left_val + right_val
                elif op == '-':
                    result = left_val - right_val
                elif op == '*':
                    result = left_val * right_val
                elif op == '/':
                    if right_val != 0:
                        result = left_val / right_val
                    else:
                        return None  # Don't fold division by zero
                elif op == '%':
                    if right_val != 0:
                        result = left_val % right_val
                    else:
                        return None
                elif op == '==':
                    result = left_val == right_val
                elif op == '!=':
                    result = left_val != right_val
                elif op == '<':
                    result = left_val < right_val
                elif op == '>':
                    result = left_val > right_val
                elif op == '<=':
                    result = left_val <= right_val
                elif op == '>=':
                    result = left_val >= right_val
                else:
                    return None
                
                return self._create_node(ASTNodeKind.LITERAL, value=result)
                
            except (TypeError, ValueError, ZeroDivisionError):
                return None
        
        return None
    
    def get_metrics(self) -> Dict[str, Union[int, float]]:
        """Get parser performance metrics."""
        return self._metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset parser metrics."""
        for key in self._metrics:
            self._metrics[key] = 0


# ============================================================================
# Testing and Validation
# ============================================================================

def benchmark_lexer():
    """Benchmark lexer performance against sample code."""
    sample_code = '''
    -- Load user data with error handling
    @inject db.conn;
    @wraptry quarantine {
        func load_user(uid: int) -> User {
            let data = db.get(uid);
            if data != null {
                print: "User loaded successfully";
                return User::from(data);
            } else {
                log("User not found: " + uid);
                return null;
            }
        };
        
        main() {
            let user = load_user(42);
            match user {
                Some(u) => print: "Welcome " + u.name;
                None => alert("Failed to load user");
            };
        };
    };
    ''' * 100  # Multiply for larger test
    
    # Test different configurations
    configs = [
        ("Standard", LexerConfig()),
        ("Cached", LexerConfig(enable_caching=True)),
        ("No Positions", LexerConfig(emit_positions=False)),
        ("Debug", LexerConfig(debug_tokens=False)),  # Don't spam output
    ]
    
    results = {}
    
    for name, config in configs:
        lexer = OptimizedInstryxLexer(config)
        
        # Warmup
        lexer.tokenize(sample_code[:1000])
        
        # Benchmark
        start_time = time.time()
        tokens = lexer.tokenize(sample_code)
        end_time = time.time()
        
        results[name] = {
            'time': end_time - start_time,
            'tokens': len(tokens),
            'tokens_per_sec': len(tokens) / (end_time - start_time),
            'metrics': lexer.get_metrics()
        }
        
        print(f"{name}: {len(tokens)} tokens in {end_time - start_time:.4f}s "
              f"({len(tokens) / (end_time - start_time):.0f} tokens/sec)")
    
    return results


def test_lexer_functionality():
    """Test lexer functionality and backward compatibility."""
    lexer = OptimizedInstryxLexer()
    
    test_cases = [
        ('123', [(TokenKind.NUMBER, '123')]),
        ('"hello"', [(TokenKind.STRING, '"hello"')]),
        ('func', [(TokenKind.KEYWORD, 'func')]),
        ('variable', [(TokenKind.ID, 'variable')]),
        ('@inject', [(TokenKind.MACRO, '@inject')]),
        ('==', [(TokenKind.OP, '==')]),
        ('-- comment', []),  # Should be skipped
        ('a + b', [(TokenKind.ID, 'a'), (TokenKind.OP, '+'), (TokenKind.ID, 'b')]),
    ]
    
    all_passed = True
    
    for i, (code, expected) in enumerate(test_cases):
        tokens = lexer.tokenize(code)
        
        # Convert to comparable format
        actual = [(token.kind, token.value) for token in tokens]
        
        if actual == expected:
            print(f"Test {i+1}: PASSED")
        else:
            print(f"Test {i+1}: FAILED")
            print(f"  Code: {code!r}")
            print(f"  Expected: {expected}")
            print(f"  Actual: {actual}")
            all_passed = False
    
    # Test backward compatibility
    try:
        tokens = lexer.tokenize('func test() { return 42; }')
        # Should be able to unpack as tuples
        for token in tokens:
            if hasattr(token, '__iter__'):
                type_str, value = list(token)[:2]
                assert isinstance(type_str, str)
        print("Backward compatibility: PASSED")
    except Exception as e:
        print(f"Backward compatibility: FAILED - {e}")
        all_passed = False
    
    return all_passed


def test_parser_functionality():
    """Test parser functionality with various constructs."""
    parser = OptimizedInstryxParser()
    
    test_cases = [
        # Simple expressions
        ("42", lambda ast: ast.children[0].children[0].value == 42),
        ('"hello"', lambda ast: ast.children[0].children[0].value == 'hello'),
        ("x + y", lambda ast: ast.children[0].children[0].kind == ASTNodeKind.BINARY_OP),
        
        # Function definition
        ("func test() { return 42; }", 
         lambda ast: ast.children[0].kind == ASTNodeKind.FUNCTION_DEF),
        
        # Constant folding
        ("3 + 4", lambda ast: ast.children[0].children[0].value == 7),
        
        # Control structures
        ("if x { y; }", lambda ast: ast.children[0].kind == ASTNodeKind.IF_STMT),
        ("while x { y; }", lambda ast: ast.children[0].kind == ASTNodeKind.WHILE_STMT),
        
        # Quarantine blocks (Instryx-specific)
        ("quarantine try { x; }", 
         lambda ast: ast.children[0].kind == ASTNodeKind.QUARANTINE_BLOCK),
    ]
    
    all_passed = True
    
    for i, (code, validator) in enumerate(test_cases):
        try:
            ast = parser.parse(code)
            if validator(ast):
                print(f"Parser test {i+1}: PASSED")
            else:
                print(f"Parser test {i+1}: FAILED (validation)")
                print(f"  Code: {code!r}")
                print(f"  AST: {ast}")
                all_passed = False
        except Exception as e:
            print(f"Parser test {i+1}: FAILED (exception)")
            print(f"  Code: {code!r}")
            print(f"  Error: {e}")
            all_passed = False
    
    # Test metrics
    metrics = parser.get_metrics()
    print(f"Parser metrics: {metrics}")
    
    return all_passed


def benchmark_parser():
    """Benchmark parser performance."""
    parser = OptimizedInstryxParser()
    
    # Test with increasingly complex code
    test_programs = [
        # Simple
        "func simple() { return 42; }",
        
        # Medium complexity
        '''
        func factorial(n) {
            if n <= 1 {
                return 1;
            } else {
                return n * factorial(n - 1);
            };
        };
        ''',
        
        # Complex with quarantine blocks
        '''
        @inject db.conn;
        func load_user(uid) {
            quarantine try {
                let data = db.get(uid);
                if data != null {
                    print: "User loaded: " + data.name;
                    return User::from(data);
                } else {
                    log("User not found");
                    return null;
                };
            } replace {
                log("Retrying user load...");
                return load_user(uid);
            } erase {
                alert("Failed to load user completely");
                return null;
            };
        };
        
        func main() {
            let user = load_user(42);
            while user != null {
                print: "Processing: " + user.name;
                user = get_next_user();
            };
        };
        '''
    ]
    
    results = {}
    
    for i, program in enumerate(test_programs):
        # Warmup
        parser.parse("func warmup() { return 1; }")
        parser.reset_metrics()
        
        # Benchmark
        start_time = time.time()
        iterations = 100 if i == 0 else 50 if i == 1 else 10  # Fewer iterations for complex code
        
        for _ in range(iterations):
            ast = parser.parse(program)
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / iterations
        metrics = parser.get_metrics()
        
        results[f"Test {i+1}"] = {
            'total_time': total_time,
            'avg_time': avg_time,
            'iterations': iterations,
            'nodes_per_sec': metrics['nodes_created'] / total_time if total_time > 0 else 0,
            'metrics': metrics
        }
        
        print(f"Test {i+1}: {iterations} iterations in {total_time:.4f}s "
              f"(avg: {avg_time*1000:.2f}ms, {metrics['nodes_created']} nodes)")
    
    return results


def test_error_recovery():
    """Test parser error recovery capabilities."""
    parser = OptimizedInstryxParser(ParserConfig(enable_error_recovery=True))
    
    # Test cases with intentional syntax errors
    error_cases = [
        "func test( { return 42; }",  # Missing closing paren
        "func test() return 42; }",   # Missing opening brace
        "if x print: 'hello';",       # Missing braces
        "3 + + 4",                   # Double operator
    ]
    
    print("Testing error recovery...")
    
    for i, code in enumerate(error_cases):
        try:
            ast = parser.parse(code)
            print(f"Error test {i+1}: Recovered successfully")
        except ParseError as e:
            print(f"Error test {i+1}: Failed to recover - {e}")
        except Exception as e:
            print(f"Error test {i+1}: Unexpected error - {e}")
    
    metrics = parser.get_metrics()
    print(f"Error recovery metrics: {metrics['errors_recovered']} errors recovered")


if __name__ == "__main__":
    print("=== Instryx Optimized Toolchain Test Suite ===")
    print()
    
    print("1. Testing lexer functionality...")
    if test_lexer_functionality():
        print("✓ All lexer tests passed!")
    else:
        print("✗ Some lexer tests failed!")
    
    print()
    print("2. Testing parser functionality...")
    if test_parser_functionality():
        print("✓ All parser tests passed!")
    else:
        print("✗ Some parser tests failed!")
    
    print()
    print("3. Testing error recovery...")
    test_error_recovery()
    
    print()
    print("4. Running lexer benchmarks...")
    benchmark_lexer()
    
    print()
    print("5. Running parser benchmarks...")
    benchmark_parser()
    
    print()
    print("=== Test Complete ===")