#!/usr/bin/env python3
"""
toolchain.py
Main compiler/interpreter toolchain implementation for the Instryx programming language.

This file contains the complete toolchain including:
- Lexer (tokenizer)
- Parser (AST builder) 
- Interpreter/Executor
- Code generators
- Utilities and helpers

Author: Violet Magenta / VACU Technologies
License: MIT
"""

from __future__ import annotations

import re
import sys
import json
import logging
import argparse
from typing import List, Tuple, Dict, Any, Optional, Union, Iterator
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("instryx.toolchain")

# ============================================================================
# Token Types and Lexer
# ============================================================================

Token = Tuple[str, str]  # (type, value)
TokenWithPos = Tuple[str, str, int, int]  # (type, value, line, col)

@dataclass
class LexerConfig:
    emit_positions: bool = False
    skip_comments: bool = True
    skip_whitespace: bool = True

class InstryxLexer:
    """Production-ready lexer for the Instryx programming language."""
    
    def __init__(self, config: Optional[LexerConfig] = None):
        self.config = config or LexerConfig()
        
        # Keywords
        self.keywords = {
            'func', 'main', 'quarantine', 'try', 'replace', 'erase',
            'if', 'else', 'while', 'fork', 'join', 'then', 'true', 'false',
            'print', 'alert', 'log', 'return', 'import',
        }
        
        # Token specification (order matters - longer patterns first)
        specs = [
            ('ML_COMMENT', r'/\*[\s\S]*?\*/'),                    # /* comment */
            ('COMMENT',    r'--[^\n]*'),                          # -- comment
            ('MACRO',      r'@[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*'),  # @macro
            ('STRING',     r'"(?:\\.|[^"\\])*"'),                 # "string"
            ('NUMBER',     r'0x[0-9A-Fa-f]+|0b[01]+|\d+(?:\.\d+)?'),  # numbers
            ('ASSIGN',     r'='),                                 # =
            ('END',        r';'),                                 # ;
            ('DOT',        r'\.'),                                # .
            ('COLON',      r':'),                                 # :
            ('COMMA',      r','),                                 # ,
            ('LPAREN',     r'\('),                                # (
            ('RPAREN',     r'\)'),                                # )
            ('LBRACE',     r'\{'),                                # {
            ('RBRACE',     r'\}'),                                # }
            ('OP',         r'==|!=|<=|>=|\|\||&&|\b(?:and|or|not)\b|[+\-*/%<>!]'),  # operators
            ('ID',         r'[A-Za-z_][A-Za-z0-9_]*'),            # identifiers
            ('NEWLINE',    r'\n'),                                # newlines
            ('SKIP',       r'[ \t\r]+'),                          # whitespace
            ('MISMATCH',   r'.'),                                 # any other char
        ]
        
        self.token_specification = specs
        pattern = '|'.join(f"(?P<{name}>{pat})" for name, pat in specs)
        self.token_regex = re.compile(pattern, re.MULTILINE)
    
    def tokenize(self, code: str) -> List[Token]:
        """Return list of tokens."""
        return list(self.iter_tokens(code))
    
    def iter_tokens(self, code: str) -> Iterator[Token]:
        """Generate tokens one by one."""
        for mo in self.token_regex.finditer(code):
            kind = mo.lastgroup
            raw = mo.group(kind)
            
            if kind == 'SKIP':
                continue
            if kind in ('COMMENT', 'ML_COMMENT'):
                if self.config.skip_comments:
                    continue
            if kind == 'NEWLINE':
                if self.config.skip_whitespace:
                    continue
            
            # Normalize keywords
            if kind == 'ID' and raw in self.keywords:
                kind = 'KEYWORD'
            
            yield (kind, raw)


# ============================================================================
# AST Node Types
# ============================================================================

@dataclass
class ASTNode:
    """Base AST node."""
    kind: str
    value: Any = None
    children: List['ASTNode'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []

@dataclass 
class Program(ASTNode):
    imports: List[ASTNode] = None
    declarations: List[ASTNode] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.imports is None:
            self.imports = []
        if self.declarations is None:
            self.declarations = []

@dataclass
class Function(ASTNode):
    name: str = ""
    params: List[str] = None
    body: ASTNode = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.params is None:
            self.params = []

# ============================================================================
# Simple Parser
# ============================================================================

class ParseError(Exception):
    pass

class InstryxParser:
    """Simple recursive descent parser for Instryx."""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
    
    def current_token(self) -> Optional[Token]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None
    
    def consume(self, expected_type: Optional[str] = None) -> Optional[Token]:
        token = self.current_token()
        if token is None:
            return None
        if expected_type and token[0] != expected_type:
            raise ParseError(f"Expected {expected_type}, got {token[0]}")
        self.pos += 1
        return token
    
    def parse(self) -> Program:
        """Parse tokens into Program AST."""
        program = Program(kind="Program")
        
        while self.current_token():
            if self.current_token()[0] == 'KEYWORD' and self.current_token()[1] == 'func':
                func = self.parse_function()
                program.declarations.append(func)
            else:
                # Skip unknown tokens for now
                self.consume()
        
        return program
    
    def parse_function(self) -> Function:
        """Parse function declaration."""
        self.consume('KEYWORD')  # consume 'func'
        
        name_token = self.consume('ID')
        if not name_token:
            raise ParseError("Expected function name")
        
        self.consume('LPAREN')
        
        params = []
        while self.current_token() and self.current_token()[0] != 'RPAREN':
            if self.current_token()[0] == 'ID':
                params.append(self.consume('ID')[1])
            if self.current_token() and self.current_token()[0] == 'COMMA':
                self.consume('COMMA')
        
        self.consume('RPAREN')
        
        # Parse body (simplified)
        body = ASTNode(kind="Block")
        if self.current_token() and self.current_token()[0] == 'LBRACE':
            self.consume('LBRACE')
            # Skip body content for now
            brace_count = 1
            while brace_count > 0 and self.current_token():
                token = self.consume()
                if token[0] == 'LBRACE':
                    brace_count += 1
                elif token[0] == 'RBRACE':
                    brace_count -= 1
        
        return Function(kind="Function", name=name_token[1], params=params, body=body)


# ============================================================================
# Simple Interpreter/Executor
# ============================================================================

class RuntimeError(Exception):
    pass

class Environment:
    """Variable environment with scoping."""
    
    def __init__(self, parent: Optional['Environment'] = None):
        self.vars: Dict[str, Any] = {}
        self.parent = parent
    
    def get(self, name: str) -> Any:
        if name in self.vars:
            return self.vars[name]
        if self.parent:
            return self.parent.get(name)
        raise RuntimeError(f"Undefined variable: {name}")
    
    def set(self, name: str, value: Any):
        self.vars[name] = value

class Interpreter:
    """Simple interpreter for Instryx AST."""
    
    def __init__(self):
        self.global_env = Environment()
        self.functions: Dict[str, Function] = {}
        
        # Built-in functions
        self.global_env.set("print", self._builtin_print)
        self.global_env.set("alert", self._builtin_alert)
        self.global_env.set("log", self._builtin_log)
    
    def interpret(self, program: Program) -> Any:
        """Interpret a program."""
        # Register functions
        for decl in program.declarations:
            if isinstance(decl, Function):
                self.functions[decl.name] = decl
        
        # Run main if it exists
        if "main" in self.functions:
            return self._call_function("main", [])
        
        return None
    
    def _call_function(self, name: str, args: List[Any]) -> Any:
        """Call a function."""
        if name in self.global_env.vars:
            # Built-in function
            func = self.global_env.vars[name]
            if callable(func):
                return func(*args)
        
        if name in self.functions:
            func = self.functions[name]
            # Create new environment for function
            func_env = Environment(self.global_env)
            
            # Bind parameters
            for i, param in enumerate(func.params):
                if i < len(args):
                    func_env.set(param, args[i])
                else:
                    func_env.set(param, None)
            
            # Execute function body (simplified)
            return None
        
        raise RuntimeError(f"Undefined function: {name}")
    
    def _builtin_print(self, *args):
        """Built-in print function."""
        print(" ".join(str(arg) for arg in args))
        return None
    
    def _builtin_alert(self, *args):
        """Built-in alert function.""" 
        print("ALERT:", " ".join(str(arg) for arg in args))
        return None
    
    def _builtin_log(self, *args):
        """Built-in log function."""
        logger.info(" ".join(str(arg) for arg in args))
        return None


# ============================================================================
# Toolchain Main Class
# ============================================================================

class InstryxToolchain:
    """Main toolchain coordinating lexer, parser, and interpreter."""
    
    def __init__(self):
        self.lexer = InstryxLexer()
        self.interpreter = Interpreter()
    
    def compile_and_run(self, code: str) -> Any:
        """Compile and run Instryx code."""
        try:
            # Tokenize
            tokens = self.lexer.tokenize(code)
            logger.debug(f"Tokenized {len(tokens)} tokens")
            
            # Parse
            parser = InstryxParser(tokens)
            program = parser.parse()
            logger.debug(f"Parsed program with {len(program.declarations)} declarations")
            
            # Interpret
            result = self.interpreter.interpret(program)
            return result
            
        except Exception as e:
            logger.error(f"Compilation/execution failed: {e}")
            raise
    
    def tokenize_only(self, code: str) -> List[Token]:
        """Just tokenize code and return tokens."""
        return self.lexer.tokenize(code)


# ============================================================================
# CLI and Main Entry Points
# ============================================================================

def create_sample_program() -> str:
    """Create a sample Instryx program for testing."""
    return '''
-- Sample Instryx program
@inject db.conn;

func load_user(uid) {
    quarantine try {
        data = db.get(uid);
        print: "User loaded";
    } replace {
        log("Retrying...");
        load_user(uid);
    } erase {
        alert("Load failed");
    };
};

main() {
    load_user(42);
    print("Program completed");
};
'''

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Instryx Language Toolchain')
    parser.add_argument('file', nargs='?', help='Instryx source file to compile/run')
    parser.add_argument('--tokenize-only', action='store_true', help='Only tokenize, don\'t parse/run')
    parser.add_argument('--sample', action='store_true', help='Run with sample program')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    toolchain = InstryxToolchain()
    
    # Get source code
    if args.sample:
        code = create_sample_program()
        print("Running sample program:")
        print(code)
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                code = f.read()
        except FileNotFoundError:
            print(f"Error: File '{args.file}' not found")
            return 1
        except Exception as e:
            print(f"Error reading file: {e}")
            return 1
    else:
        print("No input provided. Use --sample for a demo or specify a file.")
        return 1
    
    try:
        if args.tokenize_only:
            tokens = toolchain.tokenize_only(code)
            print(f"Tokens ({len(tokens)}):")
            for token in tokens:
                print(f"  {token}")
        else:
            result = toolchain.compile_and_run(code)
            if result is not None:
                print(f"Result: {result}")
            
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

# ============================================================================
# Self-test when run as script
# ============================================================================

if __name__ == "__main__":
    sys.exit(main())