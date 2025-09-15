# instryx_lexer.py
# Production-ready Lexer for the Instryx Programming Language
# Author: Violet Magenta / VACU Technologies
# License: MIT

import re
from typing import List, Tuple

Token = Tuple[str, str]

class InstryxLexer:
    def __init__(self):
        self.token_specification = [
            ('MACRO',      r'@\w+'),                   # CIAMS macros (e.g., @inject, @ffi)
            ('COMMENT',    r'--[^\n]*'),               # Single-line comment
            ('STRING',     r'"[^"\n]*"'),              # Double-quoted string
            ('NUMBER',     r'\d+(\.\d+)?'),           # Integer or decimal number
            ('ASSIGN',     r'='),                      # Assignment operator
            ('END',        r';'),                      # Statement terminator
            ('ID',         r'[A-Za-z_][A-Za-z0-9_]*'), # Identifiers
            ('NEWLINE',    r'\n'),                     # Line endings
            ('SKIP',       r'[ \t]+'),                 # Skip over spaces and tabs
            ('LPAREN',     r'\('),                     # Open parenthesis
            ('RPAREN',     r'\)'),                     # Close parenthesis
            ('LBRACE',     r'\{'),                     # Open brace
            ('RBRACE',     r'\}'),                     # Close brace
            ('OP',         r'[\+\-\*/<>!]=?|==|!=|>=|<=|and|or|not'),  # Operators
            ('COLON',      r':'),                      # Colon for directives
        ]
        self.token_regex = re.compile('|'.join(f'(?P<{name}>{pattern})' for name, pattern in self.token_specification))
        self.keywords = {
            'func', 'main', 'quarantine', 'try', 'replace', 'erase',
            'if', 'else', 'while', 'fork', 'join', 'then', 'true', 'false',
            'print', 'alert', 'log', 'return',
        }

    def tokenize(self, code: str) -> List[Token]:
        tokens = []
        for mo in self.token_regex.finditer(code):
            kind = mo.lastgroup
            value = mo.group()
            if kind == 'NEWLINE' or kind == 'SKIP':
                continue
            if kind == 'ID' and value in self.keywords:
                kind = 'KEYWORD'
            tokens.append((kind, value))
        return tokens


# Test block (can be removed in production)
if __name__ == "__main__":
    lexer = InstryxLexer()
    code = """
    -- Load user data
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
    };
    """
    for token in lexer.tokenize(code):
        print(token)
