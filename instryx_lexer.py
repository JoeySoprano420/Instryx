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

# instryx_lexer.py
# Production-ready Lexer for the Instryx Programming Language — supreme boosters
# Author: Violet Magenta / VACU Technologies (modified)
# License: MIT
"""
Enhanced, fully-implemented lexer for Instryx.

Features:
 - Robust, high-performance regex tokenization with named groups
 - Support for: macros (@...), identifiers, keywords, numbers (int/float/hex/bin), strings with escapes
 - Single-line (--) and multi-line (/* ... */) comments
 - Operators (symbols and word-ops: and/or/not), punctuation (., :, ;, parentheses, braces, dot)
 - Emits tokens as tuples: (type, value, lineno, col) (backwards-compatible with simple (type,value))
 - Tokenization options and small fallback behavior for isolated tests
 - Lightweight public API: tokenize(code) -> List[Token] (or generator via iter_tokens)
"""

from __future__ import annotations
import re
from typing import List, Tuple, Iterator, Optional, Iterable, Union, Dict
from dataclasses import dataclass

# Canonical token tuple: (type, value, lineno, col)
Token = Tuple[str, str, int, int]


@dataclass
class LexerConfig:
    emit_positions: bool = True   # include lineno/col in tokens
    skip_comments: bool = True    # drop comments
    skip_whitespace: bool = True  # skip spaces/tabs/newlines (NEWLINE may still be emitted if desired)


class InstryxLexer:
    def __init__(self, config: Optional[LexerConfig] = None):
        self.config = config or LexerConfig()

        # Keyword set (kept small and overridable)
        self.keywords = {
            'func', 'main', 'quarantine', 'try', 'replace', 'erase',
            'if', 'else', 'while', 'fork', 'join', 'then', 'true', 'false',
            'print', 'alert', 'log', 'return', 'import',
        }

        # Token specification: order matters (longer patterns / multi-char first)
        specs = [
            ('ML_COMMENT', r'/\*[\s\S]*?\*/'),                   # Multi-line comment
            ('COMMENT',    r'--[^\n]*'),                         # Single-line comment
            ('MACRO',      r'@[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*'),  # @macro or @macro.mod
            ('STRING',     r'"(?:\\.|[^"\\])*"'),                # Double-quoted string with escapes
            ('NUMBER',     r'0x[0-9A-Fa-f_]+|0b[01_]+|\d+(?:_\d+)*(?:\.\d+(?:_\d+)*)?'),  # int/float/hex/bin with underscores
            ('ASSIGN',     r'='),                                # Assignment
            ('END',        r';'),                                # Statement terminator
            ('DOT',        r'\.'),                               # Dot
            ('COLON',      r':'),                                # Colon for directives
            ('COMMA',      r','),                                # Comma
            ('LPAREN',     r'\('),                               # (
            ('RPAREN',     r'\)'),                               # )
            ('LBRACE',     r'\{'),                               # {
            ('RBRACE',     r'\}'),                               # }
            # Operators: words and symbols; word-ops bounded
            ('OP',         r'==|!=|<=|>=|\|\||&&|\b(?:and|or|not)\b|[+\-*/%<>!]'),
            ('MACSYM',     r'@'),                                # stray @
            ('ID',         r'[A-Za-z_][A-Za-z0-9_]*'),           # Identifiers
            ('NEWLINE',    r'\n'),                               # Newlines
            ('SKIP',       r'[ \t\r]+'),                         # Spaces, tabs, CR (skipped)
            ('MISMATCH',   r'.'),                                # Any other single char
        ]

        # Compile combined regex
        self.token_specification = specs
        regex_parts = (f"(?P<{name}>{pattern})" for name, pattern in self.token_specification)
        self.token_regex = re.compile('|'.join(regex_parts), re.MULTILINE)

        # Precompute keyword lookup for quick membership check
        self._keywords = set(self.keywords)

    # Public API: returns a list of tokens (type, value) or (type, value, lineno, col)
    def tokenize(self, code: str) -> List[Union[Tuple[str, str], Token]]:
        return list(self.iter_tokens(code))

    # Iterator API: yields tokens progressively (better for large inputs)
    def iter_tokens(self, code: str) -> Iterator[Union[Tuple[str, str], Token]]:
        for tok in self._iter_tokens_internal(code):
            yield tok

    def _iter_tokens_internal(self, code: str) -> Iterator[Union[Tuple[str, str], Token]]:
        get_pos = self._compute_position
        for mo in self.token_regex.finditer(code):
            kind = mo.lastgroup
            raw = mo.group(kind)
            if kind == 'SKIP':
                continue
            if kind in ('COMMENT', 'ML_COMMENT'):
                if self.config.skip_comments:
                    continue
                else:
                    lineno, col = get_pos(code, mo.start())
                    yield self._mk_token('COMMENT', raw, lineno, col)
                    continue
            if kind == 'NEWLINE':
                # by default skip newlines (parser doesn't need them)
                if self.config.skip_whitespace:
                    continue
                else:
                    lineno, col = get_pos(code, mo.start())
                    yield self._mk_token('NEWLINE', raw, lineno, col)
                    continue

            # Normalize strings: keep escapes intact but store raw including quotes
            if kind == 'ID' and raw in self._keywords:
                kind = 'KEYWORD'

            # Normalize numbers: strip underscores for numeric parsing but keep original value as string/int/float
            if kind == 'NUMBER':
                value = raw
            else:
                value = raw

            lineno, col = get_pos(code, mo.start())
            yield self._mk_token(kind, value, lineno, col)

    def _mk_token(self, kind: str, value: str, lineno: int, col: int) -> Union[Tuple[str, str], Token]:
        if self.config.emit_positions:
            return (kind, value, lineno, col)
        else:
            return (kind, value)

    @staticmethod
    def _compute_position(code: str, index: int) -> Tuple[int, int]:
        # Compute line number and column for index (1-based line, 0-based column)
        # This is O(1) amortized if code is not extremely large; for simplicity we compute via rfind
        # Find last newline before index
        last_nl = code.rfind('\n', 0, index)
        if last_nl == -1:
            lineno = 1
            col = index
        else:
            lineno = code.count('\n', 0, index) + 1
            col = index - last_nl - 1
        return lineno, col


# -------------------------
# Simple CLI test (executable)
# -------------------------
if __name__ == "__main__":
    sample = """
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

    lexer = InstryxLexer()
    for tok in lexer.iter_tokens(sample):
        print(tok)

# instryx_lexer.py
# Production-ready Lexer for the Instryx Programming Language — supreme boosters (final)
# Author: Violet Magenta / VACU Technologies (modified)
# License: MIT
"""
High-performance, robust lexer for Instryx with superior boosters:
 - fast regex-based tokenization with named groups
 - supports macros, keywords, identifiers, numbers (int/float/hex/bin), strings with escapes
 - single-line (--) and multi-line (/* ... */) comments
 - operators (symbol + word ops), punctuation, dot accessor
 - emits tokens as (type, value) or (type, value, lineno, col) depending on config
 - fast position computation using precomputed line starts + bisect
 - iter_tokens generator + tokenize convenience
 - backward compatible with older (type, value) output
 - self-test with asserts in __main__
"""

from __future__ import annotations
import re
from typing import List, Tuple, Iterator, Optional, Union
from dataclasses import dataclass, field
import bisect

# Canonical token tuple (type, value, lineno, col) when positions enabled
TokenWithPos = Tuple[str, str, int, int]
TokenSimple = Tuple[str, str]
Token = Union[TokenSimple, TokenWithPos]


@dataclass
class LexerConfig:
    emit_positions: bool = True     # include lineno/col in tokens
    skip_comments: bool = True      # drop comments
    skip_whitespace: bool = True    # drop whitespace/newlines


class InstryxLexer:
    def __init__(self, config: Optional[LexerConfig] = None):
        self.config = config or LexerConfig()

        # Keywords (extendable)
        self.keywords = {
            'func', 'main', 'quarantine', 'try', 'replace', 'erase',
            'if', 'else', 'while', 'fork', 'join', 'then', 'true', 'false',
            'print', 'alert', 'log', 'return', 'import',
        }

        # Token specification (order matters)
        specs = [
            ('ML_COMMENT', r'/\*[\s\S]*?\*/'),                                   # /* ... */
            ('COMMENT',    r'--[^\n]*'),                                         # -- ...
            ('MACRO',      r'@[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*'),  # @name or @name.mod
            ('STRING',     r'"(?:\\.|[^"\\])*"'),                                # "..." with escapes
            ('NUMBER',     r'0x[0-9A-Fa-f_]+|0b[01_]+|\d+(?:_\d+)*(?:\.\d+(?:_\d+)*)?'),  # ints/floats/bin/hex with underscores
            ('ASSIGN',     r'='),                                                 # =
            ('END',        r';'),                                                 # ;
            ('DOT',        r'\.'),                                                # .
            ('COLON',      r':'),                                                 # :
            ('COMMA',      r','),                                                 # ,
            ('LPAREN',     r'\('),                                                # (
            ('RPAREN',     r'\)'),                                                # )
            ('LBRACE',     r'\{'),                                                # {
            ('RBRACE',     r'\}'),                                                # }
            ('OP',         r'==|!=|<=|>=|\|\||&&|\b(?:and|or|not)\b|[+\-*/%<>!]'), # operators
            ('ID',         r'[A-Za-z_][A-Za-z0-9_]*'),                            # identifiers
            ('NEWLINE',    r'\n'),                                                # newline
            ('SKIP',       r'[ \t\r]+'),                                          # whitespace
            ('MISMATCH',   r'.'),                                                 # any other single char
        ]

        self.token_specification = specs
        regex_parts = (f"(?P<{name}>{pattern})" for name, pattern in self.token_specification)
        self.token_regex = re.compile('|'.join(regex_parts), re.MULTILINE)

    # Convenience: returns list of tokens
    def tokenize(self, code: str) -> List[Token]:
        return list(self.iter_tokens(code))

    # Generator: yields tokens one by one
    def iter_tokens(self, code: str) -> Iterator[Token]:
        # Precompute line start indices for fast (lineno, col) queries
        line_starts = self._compute_line_starts(code)
        for mo in self.token_regex.finditer(code):
            kind = mo.lastgroup
            raw = mo.group(kind)
            if kind == 'SKIP':
                continue
            if kind in ('COMMENT', 'ML_COMMENT'):
                if self.config.skip_comments:
                    continue
                # else fallthrough to emit comment token
            if kind == 'NEWLINE':
                if self.config.skip_whitespace:
                    continue
                # else emit NEWLINE token

            # Keyword normalization
            if kind == 'ID' and raw in self.keywords:
                kind = 'KEYWORD'

            # Numbers and strings preserved as raw text; parser will handle conversion
            value = raw

            if self.config.emit_positions:
                lineno, col = self._pos_from_index(line_starts, mo.start())
                yield (kind, value, lineno, col)
            else:
                yield (kind, value)

    # Internal: compute line start indices (0-based index of each line's start)
    @staticmethod
    def _compute_line_starts(code: str) -> List[int]:
        # first line starts at 0
        starts = [0]
        # find newline positions
        for m in re.finditer(r'\n', code):
            starts.append(m.end())  # start of next line
        return starts

    # Internal: map byte index to (lineno, col) using bisect on line_starts
    @staticmethod
    def _pos_from_index(line_starts: List[int], index: int) -> Tuple[int, int]:
        # lineno is 1-based, col is 0-based
        # bisect_right returns first start > index, so subtract 1
        line_no = bisect.bisect_right(line_starts, index) - 1
        lineno = line_no + 1
        col = index - line_starts[line_no]
        return lineno, col

    # Backwards-compatible simple tokenize API (no positions)
    def tokenize_simple(self, code: str) -> List[Tuple[str, str]]:
        cfg_saved = self.config.emit_positions
        try:
            self.config.emit_positions = False
            return [t for t in self.iter_tokens(code)]
        finally:
            self.config.emit_positions = cfg_saved


# -------------------------
# CLI self-test (executable)
# -------------------------
if __name__ == "__main__":
    SAMPLE = """
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

    # default config: emit positions, skip comments/whitespace
    lexer = InstryxLexer()
    tokens = lexer.tokenize(SAMPLE)
    # show a few tokens
    for t in tokens[:20]:
        print(t)

    # simple tokens (backwards-compatible)
    simple = lexer.tokenize_simple(SAMPLE)
    assert all(len(tok) == 2 for tok in simple), "simple tokens must be (type,value)"

    # ensure keywords normalized
    assert any(tok[0] == 'KEYWORD' and tok[1] == 'func' for tok in simple), "keyword detection failed"

    print("Lexer self-test passed.")

# instryx_lexer.py
# Production-ready Lexer for the Instryx Programming Language — supreme boosters
# Author: Violet Magenta / VACU Technologies (modified)
# License: MIT
"""
High-performance, robust lexer for Instryx with superior boosters:
 - fast regex-based tokenization with named groups
 - supports macros, keywords, identifiers, numbers (int/float/hex/bin), strings with escapes
 - single-line (--) and multi-line (/* ... */) comments
 - operators (symbol + word ops), punctuation, dot accessor
 - emits tokens as (type, value) or (type, value, lineno, col) depending on config
 - fast position computation using precomputed line starts + bisect
 - iter_tokens generator + tokenize convenience
 - backward compatible with older (type, value) output
"""

from __future__ import annotations
import re
import bisect
from typing import List, Tuple, Iterator, Optional, Union
from dataclasses import dataclass

# Token shapes:
TokenWithPos = Tuple[str, str, int, int]
TokenSimple = Tuple[str, str]
Token = Union[TokenSimple, TokenWithPos]


@dataclass
class LexerConfig:
    emit_positions: bool = True     # include lineno/col in tokens
    skip_comments: bool = True      # drop comments
    skip_whitespace: bool = True    # drop whitespace/newlines


class InstryxLexer:
    def __init__(self, config: Optional[LexerConfig] = None):
        self.config = config or LexerConfig()

        # Keyword set (overrideable)
        self.keywords = {
            'func', 'main', 'quarantine', 'try', 'replace', 'erase',
            'if', 'else', 'while', 'fork', 'join', 'then', 'true', 'false',
            'print', 'alert', 'log', 'return', 'import',
        }

        # Token specification - order matters
        specs = [
            ('ML_COMMENT', r'/\*[\s\S]*?\*/'),
            ('COMMENT',    r'--[^\n]*'),
            ('MACRO',      r'@[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*'),
            ('STRING',     r'"(?:\\.|[^"\\])*"'),
            ('NUMBER',     r'0x[0-9A-Fa-f_]+|0b[01_]+|\d+(?:_\d+)*(?:\.\d+(?:_\d+)*)?'),
            ('ASSIGN',     r'='),
            ('END',        r';'),
            ('DOT',        r'\.'),
            ('COLON',      r':'),
            ('COMMA',      r','),
            ('LPAREN',     r'\('),
            ('RPAREN',     r'\)'),
            ('LBRACE',     r'\{'),
            ('RBRACE',     r'\}'),
            ('OP',         r'==|!=|<=|>=|\|\||&&|\b(?:and|or|not)\b|[+\-*/%<>!]'),
            ('ID',         r'[A-Za-z_][A-Za-z0-9_]*'),
            ('NEWLINE',    r'\n'),
            ('SKIP',       r'[ \t\r]+'),
            ('MISMATCH',   r'.'),
        ]

        self.token_specification = specs
        pattern = '|'.join(f"(?P<{name}>{pat})" for name, pat in self.token_specification)
        self.token_regex = re.compile(pattern, re.MULTILINE)

    def tokenize(self, code: str) -> List[Token]:
        """Convenience: return list of tokens."""
        return list(self.iter_tokens(code))

    def iter_tokens(self, code: str) -> Iterator[Token]:
        """Generator yielding tokens (type,value[,lineno,col])."""
        line_starts = self._compute_line_starts(code)
        for mo in self.token_regex.finditer(code):
            kind = mo.lastgroup
            raw = mo.group(kind)

            if kind == 'SKIP':
                continue
            if kind in ('COMMENT', 'ML_COMMENT'):
                if self.config.skip_comments:
                    continue
                # else emit comment token
            if kind == 'NEWLINE':
                if self.config.skip_whitespace:
                    continue
                # else emit newline token

            # Keyword normalization
            if kind == 'ID' and raw in self.keywords:
                kind = 'KEYWORD'

            value = raw

            if self.config.emit_positions:
                lineno, col = self._pos_from_index(line_starts, mo.start())
                yield (kind, value, lineno, col)
            else:
                yield (kind, value)

    # Backwards-compatible simple tokenize (no positions)
    def tokenize_simple(self, code: str) -> List[TokenSimple]:
        saved = self.config.emit_positions
        try:
            self.config.emit_positions = False
            return [t for t in self.iter_tokens(code)]
        finally:
            self.config.emit_positions = saved

    @staticmethod
    def _compute_line_starts(code: str) -> List[int]:
        starts = [0]
        for m in re.finditer(r'\n', code):
            starts.append(m.end())
        return starts

    @staticmethod
    def _pos_from_index(line_starts: List[int], index: int) -> Tuple[int, int]:
        # lineno 1-based, col 0-based
        line_no = bisect.bisect_right(line_starts, index) - 1
        lineno = line_no + 1
        col = index - line_starts[line_no]
        return lineno, col


# -------------------------
# CLI self-test (executable)
# -------------------------
if __name__ == "__main__":
    SAMPLE = """
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

    lexer = InstryxLexer()
    tokens = lexer.tokenize(SAMPLE)
    for t in tokens[:40]:
        print(t)

    simple = lexer.tokenize_simple(SAMPLE)
    assert all(len(tok) == 2 for tok in simple), "simple tokens must be (type,value)"
    assert any(tok[0] == 'KEYWORD' and tok[1] == 'func' for tok in simple), "keyword detection failed"
    print("Lexer self-test passed.")

# instryx_lexer.py
# Production-ready Lexer for the Instryx Programming Language — supreme boosters
# Author: Violet Magenta / VACU Technologies (modified)
# License: MIT
"""
Enhanced, fully-implemented lexer for Instryx.

Key features:
 - Fast regex tokenization with named groups
 - Supports macros (@...), identifiers, keywords, numbers (int/float/hex/bin with underscores), escaped strings
 - Single-line (--) and multi-line (/* ... */) comments
 - Symbol and word operators (and/or/not), punctuation, dot accessor
 - Streaming API (`iter_tokens`) and convenience `tokenize`
 - Backwards-compatible simple tokens via `tokenize_simple`
 - Optional position emission (lineno, col) via LexerConfig (default: False for compatibility)
"""

from __future__ import annotations
import re
import bisect
from typing import List, Tuple, Iterator, Optional, Union
from dataclasses import dataclass

# Token shapes:
TokenWithPos = Tuple[str, str, int, int]
TokenSimple = Tuple[str, str]
Token = Union[TokenSimple, TokenWithPos]


@dataclass
class LexerConfig:
    emit_positions: bool = False    # default False for backward compatibility
    skip_comments: bool = True      # drop comments by default
    skip_whitespace: bool = True    # drop whitespace/newlines by default


class InstryxLexer:
    """
    Usage:
      lexer = InstryxLexer()                      # defaults (no positions)
      tokens = lexer.tokenize(code)               # list of (type, value)
      lexer_pos = InstryxLexer(LexerConfig(emit_positions=True))
      tokens_pos = lexer_pos.tokenize(code)       # list of (type, value, lineno, col)
      for tok in lexer.iter_tokens(code): ...    # generator
    """

    def __init__(self, config: Optional[LexerConfig] = None):
        self.config = config or LexerConfig()

        # Keyword set (extendable)
        self.keywords = {
            'func', 'main', 'quarantine', 'try', 'replace', 'erase',
            'if', 'else', 'while', 'fork', 'join', 'then', 'true', 'false',
            'print', 'alert', 'log', 'return', 'import',
        }

        # Token specification (order matters: longer patterns first)
        specs = [
            ('ML_COMMENT', r'/\*[\s\S]*?\*/'),                                   # /* ... */
            ('COMMENT',    r'--[^\n]*'),                                         # -- comment
            ('MACRO',      r'@[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*'),  # @name or @name.mod
            ('STRING',     r'"(?:\\.|[^"\\])*"'),                                # "..." with escapes
            ('NUMBER',     r'0x[0-9A-Fa-f_]+|0b[01_]+|\d+(?:_\d+)*(?:\.\d+(?:_\d+)*)?'),  # ints/floats/bin/hex underscores
            ('ASSIGN',     r'='),                                                 # =
            ('END',        r';'),                                                 # ;
            ('DOT',        r'\.'),                                                # .
            ('COLON',      r':'),                                                 # :
            ('COMMA',      r','),                                                 # ,
            ('LPAREN',     r'\('),                                                # (
            ('RPAREN',     r'\)'),                                                # )
            ('LBRACE',     r'\{'),                                                # {
            ('RBRACE',     r'\}'),                                                # }
            ('OP',         r'==|!=|<=|>=|\|\||&&|\b(?:and|or|not)\b|[+\-*/%<>!]'), # operators
            ('ID',         r'[A-Za-z_][A-Za-z0-9_]*'),                            # identifiers
            ('NEWLINE',    r'\n'),                                                # newline
            ('SKIP',       r'[ \t\r]+'),                                          # whitespace (skip)
            ('MISMATCH',   r'.'),                                                 # any other single char
        ]
        self.token_specification = specs
        pattern = '|'.join(f"(?P<{name}>{pat})" for name, pat in self.token_specification)
        self.token_regex = re.compile(pattern, re.MULTILINE)

    # Convenience: return full list of tokens
    def tokenize(self, code: str) -> List[Token]:
        return list(self.iter_tokens(code))

    # Generator: iterate tokens (efficient for large inputs)
    def iter_tokens(self, code: str) -> Iterator[Token]:
        line_starts = self._compute_line_starts(code) if self.config.emit_positions else ()
        for mo in self.token_regex.finditer(code):
            kind = mo.lastgroup
            raw = mo.group(kind)

            if kind == 'SKIP':
                continue
            if kind in ('COMMENT', 'ML_COMMENT'):
                if self.config.skip_comments:
                    continue
                # else fallthrough and emit comment token
            if kind == 'NEWLINE':
                if self.config.skip_whitespace:
                    continue
                # else fallthrough and emit NEWLINE token

            # Normalize identifiers that are keywords
            if kind == 'ID' and raw in self.keywords:
                kind = 'KEYWORD'

            value = raw

            if self.config.emit_positions:
                lineno, col = self._pos_from_index(line_starts, mo.start())
                yield (kind, value, lineno, col)
            else:
                yield (kind, value)

    # Backwards-compatible simple tokenize (no positions)
    def tokenize_simple(self, code: str) -> List[TokenSimple]:
        saved = self.config.emit_positions
        try:
            self.config.emit_positions = False
            return [t for t in self.iter_tokens(code)]
        finally:
            self.config.emit_positions = saved

    # Compute line start offsets for fast lookups
    @staticmethod
    def _compute_line_starts(code: str) -> List[int]:
        starts = [0]
        for m in re.finditer(r'\n', code):
            starts.append(m.end())
        return starts

    # Map index -> (lineno, col) using bisect on precomputed starts
    @staticmethod
    def _pos_from_index(line_starts: List[int], index: int) -> Tuple[int, int]:
        line_no = bisect.bisect_right(line_starts, index) - 1
        lineno = line_no + 1
        col = index - line_starts[line_no]
        return lineno, col


# -------------------------
# CLI self-test (executable)
# -------------------------
if __name__ == "__main__":
    SAMPLE = """
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

    lexer = InstryxLexer()  # default: emit_positions=False for compatibility
    tokens = lexer.tokenize(SAMPLE)
    for t in tokens[:40]:
        print(t)

    # quick assertions for basic self-test
    simple = lexer.tokenize_simple(SAMPLE)
    assert all(len(tok) == 2 for tok in simple), "simple tokens must be (type,value)"
    assert any(tok[0] == 'KEYWORD' and tok[1] == 'func' for tok in simple), "keyword detection failed"
    print("Lexer self-test passed.")

# instryx_lexer.py
# Production-ready Lexer for the Instryx Programming Language — supreme boosters
# Author: Violet Magenta / VACU Technologies (modified)
# License: MIT
"""
High-performance, robust lexer for Instryx with superior boosters:
 - fast regex-based tokenization with named groups
 - supports macros (@...), keywords, identifiers, numbers (int/float/hex/bin with underscores), escaped strings
 - single-line (--) and multi-line (/* ... */) comments
 - symbol and word operators (and/or/not), punctuation, dot accessor
 - streaming API (iter_tokens) and convenience tokenize/tokenize_simple
 - optional position emission (lineno, col) via LexerConfig
 - backwards-compatible outputs
"""

from __future__ import annotations
import re
import bisect
from typing import List, Tuple, Iterator, Optional, Union
from dataclasses import dataclass

# Token shapes:
TokenWithPos = Tuple[str, str, int, int]
TokenSimple = Tuple[str, str]
Token = Union[TokenSimple, TokenWithPos]


@dataclass
class LexerConfig:
    emit_positions: bool = True     # include lineno/col in tokens (set False for legacy (type,value))
    skip_comments: bool = True      # drop comments
    skip_whitespace: bool = True    # drop whitespace/newlines


class InstryxLexer:
    """
    Usage:
      lexer = InstryxLexer()                               # defaults: positions True
      tokens = lexer.tokenize(code)                        # list of tokens (type,value,lineno,col)
      simple = lexer.tokenize_simple(code)                 # list of (type,value)
      for tok in lexer.iter_tokens(code): ...              # generator
    """

    def __init__(self, config: Optional[LexerConfig] = None):
        self.config = config or LexerConfig()

        # Keyword set (extendable by consumer)
        self.keywords = {
            'func', 'main', 'quarantine', 'try', 'replace', 'erase',
            'if', 'else', 'while', 'fork', 'join', 'then', 'true', 'false',
            'print', 'alert', 'log', 'return', 'import',
        }

        # Token specification (order matters: longer/multi-char first)
        specs = [
            ('ML_COMMENT', r'/\*[\s\S]*?\*/'),                                   # /* ... */
            ('COMMENT',    r'--[^\n]*'),                                         # -- comment
            ('MACRO',      r'@[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*'),  # @name or @name.mod
            ('STRING',     r'"(?:\\.|[^"\\])*"'),                                # "..." with escapes
            ('NUMBER',     r'0x[0-9A-Fa-f_]+|0b[01_]+|\d+(?:_\d+)*(?:\.\d+(?:_\d+)*)?'), # ints/floats/bin/hex underscores
            ('ASSIGN',     r'='),                                                 # =
            ('END',        r';'),                                                 # ;
            ('DOT',        r'\.'),                                                # .
            ('COLON',      r':'),                                                 # :
            ('COMMA',      r','),                                                 # ,
            ('LPAREN',     r'\('),                                                # (
            ('RPAREN',     r'\)'),                                                # )
            ('LBRACE',     r'\{'),                                                # {
            ('RBRACE',     r'\}'),                                                # }
            ('OP',         r'==|!=|<=|>=|\|\||&&|\b(?:and|or|not)\b|[+\-*/%<>!]'), # operators (symbol + word ops)
            ('ID',         r'[A-Za-z_][A-Za-z0-9_]*'),                            # identifiers
            ('NEWLINE',    r'\n'),                                                # newline
            ('SKIP',       r'[ \t\r]+'),                                          # whitespace (skip)
            ('MISMATCH',   r'.'),                                                 # any other single char
        ]

        self.token_specification = specs
        pattern = '|'.join(f"(?P<{name}>{pat})" for name, pat in self.token_specification)
        self.token_regex = re.compile(pattern, re.MULTILINE)

    # Convenience: return full list of tokens
    def tokenize(self, code: str) -> List[Token]:
        return list(self.iter_tokens(code))

    # Generator: yield tokens one by one (efficient streaming)
    def iter_tokens(self, code: str) -> Iterator[Token]:
        line_starts = self._compute_line_starts(code) if self.config.emit_positions else ()
        for mo in self.token_regex.finditer(code):
            kind = mo.lastgroup
            raw = mo.group(kind)

            if kind == 'SKIP':
                continue
            if kind in ('COMMENT', 'ML_COMMENT'):
                if self.config.skip_comments:
                    continue
                # else emit comment token
            if kind == 'NEWLINE':
                if self.config.skip_whitespace:
                    continue
                # else emit NEWLINE token

            # Normalize IDs that are keywords
            if kind == 'ID' and raw in self.keywords:
                kind = 'KEYWORD'

            value = raw

            if self.config.emit_positions:
                lineno, col = self._pos_from_index(line_starts, mo.start())
                yield (kind, value, lineno, col)
            else:
                yield (kind, value)

    # Backwards-compatible simple tokenize (no positions)
    def tokenize_simple(self, code: str) -> List[TokenSimple]:
        saved = self.config.emit_positions
        try:
            self.config.emit_positions = False
            return [t for t in self.iter_tokens(code)]
        finally:
            self.config.emit_positions = saved

    # Precompute line start offsets for position lookup (fast)
    @staticmethod
    def _compute_line_starts(code: str) -> List[int]:
        starts = [0]
        for m in re.finditer(r'\n', code):
            starts.append(m.end())
        return starts

    # Map index -> (lineno, col) using bisect on precomputed line_starts
    @staticmethod
    def _pos_from_index(line_starts: List[int], index: int) -> Tuple[int, int]:
        line_no = bisect.bisect_right(line_starts, index) - 1
        lineno = line_no + 1
        col = index - line_starts[line_no]
        return lineno, col


# -------------------------
# CLI self-test (executable)
# -------------------------
if __name__ == "__main__":
    SAMPLE = """
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

    lexer = InstryxLexer()  # default: positions enabled (rich tokens)
    tokens = lexer.tokenize(SAMPLE)
    for t in tokens[:40]:
        print(t)

    # legacy simple tokens
    simple = lexer.tokenize_simple(SAMPLE)
    assert all(len(tok) == 2 for tok in simple), "simple tokens must be (type,value)"
    assert any(tok[0] == 'KEYWORD' and tok[1] == 'func' for tok in simple), "keyword detection failed"
    print("Lexer self-test passed.")

# instryx_lexer.py
# Enhanced, production-ready lexer for the Instryx Programming Language
# Author: Violet Magenta / VACU Technologies (modified)
# License: MIT
"""
InstryxLexer — robust, fast, fully-implemented and executable.

Features:
 - Regex-based tokenization with named groups
 - Macros (@...), identifiers, keywords, numbers (int/float/hex/bin with underscores), escaped strings
 - Single-line (--) and multi-line (/* ... */) comments
 - Symbol and word operators (and/or/not), punctuation, dot accessor
 - Streaming API (`iter_tokens`) and convenience `tokenize`/`tokenize_simple`
 - Optional position emission (lineno, col) via LexerConfig (default: False for backward compatibility)
 - Backward-compatible output shape
"""

from __future__ import annotations
import re
import bisect
from typing import List, Tuple, Iterator, Optional, Union
from dataclasses import dataclass

# Token shapes
TokenWithPos = Tuple[str, str, int, int]
TokenSimple = Tuple[str, str]
Token = Union[TokenSimple, TokenWithPos]


@dataclass
class LexerConfig:
    emit_positions: bool = False   # default False -> (type, value) tokens (legacy)
    skip_comments: bool = True     # drop comments by default
    skip_whitespace: bool = True   # drop whitespace/newlines by default


class InstryxLexer:
    """
    Usage:
      lexer = InstryxLexer()  # legacy-style tokens (type, value)
      tokens = lexer.tokenize(code)
      lexer_pos = InstryxLexer(LexerConfig(emit_positions=True))
      tokens_pos = lexer_pos.tokenize(code)  # (type, value, lineno, col)
      for tok in lexer.iter_tokens(code): ...
    """

    def __init__(self, config: Optional[LexerConfig] = None):
        self.config = config or LexerConfig()

        # Keywords extensible by consumer
        self.keywords = {
            'func', 'main', 'quarantine', 'try', 'replace', 'erase',
            'if', 'else', 'while', 'fork', 'join', 'then', 'true', 'false',
            'print', 'alert', 'log', 'return', 'import',
        }

        # Token specification (order matters — longer / multi-char first)
        specs = [
            ('ML_COMMENT', r'/\*[\s\S]*?\*/'),
            ('COMMENT',    r'--[^\n]*'),
            ('MACRO',      r'@[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*'),
            ('STRING',     r'"(?:\\.|[^"\\])*"'),
            ('NUMBER',     r'0x[0-9A-Fa-f_]+|0b[01_]+|\d+(?:_\d+)*(?:\.\d+(?:_\d+)*)?'),
            ('ASSIGN',     r'='),
            ('END',        r';'),
            ('DOT',        r'\.'),
            ('COLON',      r':'),
            ('COMMA',      r','),
            ('LPAREN',     r'\('),
            ('RPAREN',     r'\)'),
            ('LBRACE',     r'\{'),
            ('RBRACE',     r'\}'),
            ('OP',         r'==|!=|<=|>=|\|\||&&|\b(?:and|or|not)\b|[+\-*/%<>!]'),
            ('ID',         r'[A-Za-z_][A-Za-z0-9_]*'),
            ('NEWLINE',    r'\n'),
            ('SKIP',       r'[ \t\r]+'),
            ('MISMATCH',   r'.'),
        ]
        self.token_specification = specs
        pattern = '|'.join(f"(?P<{name}>{pat})" for name, pat in specs)
        self.token_regex = re.compile(pattern, re.MULTILINE)

    def tokenize(self, code: str) -> List[Token]:
        """Return a list of tokens. Shape depends on LexerConfig.emit_positions."""
        return list(self.iter_tokens(code))

    def iter_tokens(self, code: str) -> Iterator[Token]:
        """Generator yielding tokens. Efficient for large inputs."""
        line_starts = self._compute_line_starts(code) if self.config.emit_positions else ()
        for mo in self.token_regex.finditer(code):
            kind = mo.lastgroup
            raw = mo.group(kind)

            # Skip trivial whitespace
            if kind == 'SKIP':
                continue

            # Comments
            if kind in ('COMMENT', 'ML_COMMENT'):
                if self.config.skip_comments:
                    continue
                # else emit comment token as-is

            # Newlines
            if kind == 'NEWLINE':
                if self.config.skip_whitespace:
                    continue
                # else emit NEWLINE token

            # Normalize identifiers that are keywords
            if kind == 'ID' and raw in self.keywords:
                kind = 'KEYWORD'

            value = raw

            if self.config.emit_positions:
                lineno, col = self._pos_from_index(line_starts, mo.start())
                yield (kind, value, lineno, col)
            else:
                yield (kind, value)

    def tokenize_simple(self, code: str) -> List[TokenSimple]:
        """Backward-compatible list of simple tokens (type, value)."""
        saved = self.config.emit_positions
        try:
            self.config.emit_positions = False
            return [t for t in self.iter_tokens(code)]  # type: ignore[list-item]
        finally:
            self.config.emit_positions = saved

    # -- Utilities ---------------------------------------------------------

    @staticmethod
    def _compute_line_starts(code: str) -> List[int]:
        """Return list of start indices for each line (0-based)."""
        starts = [0]
        for m in re.finditer(r'\n', code):
            starts.append(m.end())
        return starts

    @staticmethod
    def _pos_from_index(line_starts: List[int], index: int) -> Tuple[int, int]:
        """Map character index to (lineno, col). lineno is 1-based; col is 0-based."""
        # bisect to find line number
        i = bisect.bisect_right(line_starts, index) - 1
        lineno = i + 1
        col = index - line_starts[i]
        return lineno, col


# -------------------------
# CLI self-test (executable)
# -------------------------
if __name__ == "__main__":
    SAMPLE = r'''
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
'''

    # Default lexer (legacy-style simple tokens)
    lexer = InstryxLexer()
    simple = lexer.tokenize_simple(SAMPLE)
    for t in simple:
        print(t)

    # With positions (for diagnostics / LSP)
    lexer_pos = InstryxLexer(LexerConfig(emit_positions=True))
    tokens_pos = lexer_pos.tokenize(SAMPLE)
    # print a subset
    for t in tokens_pos[:24]:
        print(t)

    # Basic self-checks
    assert any(tok[0] == 'KEYWORD' and tok[1] == 'func' for tok in simple), "keyword detection failed"
    print("instryx_lexer self-test passed.")

# instryx_lexer.py
# Instryx Lexer — superior boosters, enhancers, tooling, optimizations
# Author: Violet Magenta / VACU Technologies (modified)
# License: MIT
"""
High-performance, fully-implemented lexer for the Instryx language.

Extra boosters & optimizations:
 - Robust token model (Token dataclass) with optional positional info and span
 - Fast precompiled combined regex with ordered, named patterns
 - String unescaping (via ast.literal_eval), numeric normalization (int/float, hex/bin, underscores)
 - Multi-line and single-line comment support; optional comment/whitespace skipping
 - Streaming generator API + convenience list API + simple backwards-compatible API
 - Optional per-instance LRU caching for repeated inputs
 - TokenStream helper with peek/next for parser convenience
 - Executable self-test in __main__
"""

from __future__ import annotations
import re
import bisect
import ast
from dataclasses import dataclass
from typing import List, Tuple, Iterator, Optional, Union, Iterable, Dict
from functools import lru_cache

# Public token shapes
TokenSimple = Tuple[str, str]
TokenWithPos = Tuple[str, Union[str, int, float], int, int]
Token = Union[TokenSimple, TokenWithPos]


@dataclass(frozen=True)
class TokenObj:
    type: str
    value: Union[str, int, float]
    lineno: Optional[int] = None
    col: Optional[int] = None
    start: Optional[int] = None
    end: Optional[int] = None

    def as_tuple(self, emit_pos: bool) -> Token:
        if emit_pos and self.lineno is not None and self.col is not None:
            return (self.type, self.value, self.lineno, self.col)
        return (self.type, self.value)


@dataclass
class LexerConfig:
    emit_positions: bool = False        # default False for backward compatibility
    skip_comments: bool = True
    skip_whitespace: bool = True
    enable_cache: bool = False
    cache_size: int = 128


class InstryxLexer:
    def __init__(self, config: Optional[LexerConfig] = None):
        self.config = config or LexerConfig()

        # Keyword set (overrideable)
        self.keywords = {
            'func', 'main', 'quarantine', 'try', 'replace', 'erase',
            'if', 'else', 'while', 'fork', 'join', 'then', 'true', 'false',
            'print', 'alert', 'log', 'return', 'import',
        }

        # Token specification (order matters: longest/multi-char first)
        specs = [
            ('ML_COMMENT', r'/\*[\s\S]*?\*/'),
            ('COMMENT',    r'--[^\n]*'),
            ('MACRO',      r'@[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*'),
            ('STRING',     r'"(?:\\.|[^"\\])*"'),
            ('NUMBER',     r'0x[0-9A-Fa-f_]+|0b[01_]+|\d+(?:_\d+)*(?:\.\d+(?:_\d+)*)?'),
            ('ASSIGN',     r'='),
            ('END',        r';'),
            ('DOT',        r'\.'),
            ('COLON',      r':'),
            ('COMMA',      r','),
            ('LPAREN',     r'\('),
            ('RPAREN',     r'\)'),
            ('LBRACE',     r'\{'),
            ('RBRACE',     r'\}'),
            ('OP',         r'==|!=|<=|>=|\|\||&&|\b(?:and|or|not)\b|[+\-*/%<>!]'),
            ('ID',         r'[A-Za-z_][A-Za-z0-9_]*'),
            ('NEWLINE',    r'\n'),
            ('SKIP',       r'[ \t\r]+'),
            ('MISMATCH',   r'.'),
        ]
        self.token_specification = specs
        pattern = '|'.join(f"(?P<{name}>{pat})" for name, pat in specs)
        # compiled once per instance
        self._token_regex = re.compile(pattern, re.MULTILINE)

        # internal caches
        self._keyword_set = set(self.keywords)
        if self.config.enable_cache:
            # per-instance LRU cache for tokenize_simple
            self._tokenize_simple_cached = lru_cache(maxsize=self.config.cache_size)(self._tokenize_simple_uncached)
        else:
            self._tokenize_simple_cached = None

    # Public convenience API
    def tokenize(self, code: str) -> List[Token]:
        """Return list of tokens using configured emission (with positions if enabled)."""
        return list(self.iter_tokens(code))

    def iter_tokens(self, code: str) -> Iterator[Token]:
        """Streaming token generator (yields tuples or tuples-with-pos)."""
        line_starts = self._compute_line_starts(code) if self.config.emit_positions else ()
        for mo in self._token_regex.finditer(code):
            kind = mo.lastgroup
            raw = mo.group(kind)
            start_idx = mo.start()
            end_idx = mo.end()

            if kind == 'SKIP':
                continue

            if kind in ('COMMENT', 'ML_COMMENT'):
                if self.config.skip_comments:
                    continue
                # else emit comment token

            if kind == 'NEWLINE':
                if self.config.skip_whitespace:
                    continue
                # else emit newline token

            # Normalize IDs that are keywords
            if kind == 'ID' and raw in self._keyword_set:
                kind = 'KEYWORD'

            # Normalize and convert values
            value: Union[str, int, float] = raw
            if kind == 'STRING':
                # safely unescape using ast.literal_eval
                try:
                    value = ast.literal_eval(raw)
                except Exception:
                    # fallback: strip quotes
                    value = raw[1:-1]
            elif kind == 'NUMBER':
                value = self._normalize_number(raw)
            # MACRO, ID, OP, etc. kept as raw strings

            lineno: Optional[int] = None
            col: Optional[int] = None
            if self.config.emit_positions:
                lineno, col = self._pos_from_index(line_starts, start_idx)

            token = TokenObj(kind, value, lineno, col, start_idx, end_idx)
            yield token.as_tuple(self.config.emit_positions)

    # Backwards-compatible simple tokenize (type,value), with optional caching
    def tokenize_simple(self, code: str) -> List[TokenSimple]:
        if self.config.enable_cache and self._tokenize_simple_cached is not None:
            return list(self._tokenize_simple_cached(code))
        return list(self._tokenize_simple_uncached(code))

    def _tokenize_simple_uncached(self, code: str) -> Iterator[TokenSimple]:
        # produce (type, value) tuples regardless of emit_positions
        for t in self.iter_tokens_simple(code):
            yield t

    def iter_tokens_simple(self, code: str) -> Iterator[TokenSimple]:
        """Generator yielding (type, value) tuples (fast path)"""
        for mo in self._token_regex.finditer(code):
            kind = mo.lastgroup
            raw = mo.group(kind)
            if kind in ('SKIP',):
                continue
            if kind in ('COMMENT', 'ML_COMMENT') and self.config.skip_comments:
                continue
            if kind == 'NEWLINE' and self.config.skip_whitespace:
                continue
            if kind == 'ID' and raw in self._keyword_set:
                kind = 'KEYWORD'
            # For numbers and strings, return normalized values (string/unescaped/number) to aid older consumers
            if kind == 'STRING':
                try:
                    value = ast.literal_eval(raw)
                except Exception:
                    value = raw[1:-1]
            elif kind == 'NUMBER':
                value = self._normalize_number(raw)
            else:
                value = raw
            yield (kind, value)  # type: ignore

    # Utility: return a TokenStream wrapper for parser convenience
    def stream(self, code: str) -> "TokenStream":
        return TokenStream(self.iter_tokens(code), emit_pos=self.config.emit_positions)

    # Helpers ----------------------------------------------------------------
    @staticmethod
    def _normalize_number(raw: str) -> Union[int, float, str]:
        r = raw.replace('_', '')
        try:
            if r.startswith(('0x', '0X')):
                return int(r, 16)
            if r.startswith(('0b', '0B')):
                return int(r, 2)
            if '.' in r:
                return float(r)
            return int(r)
        except Exception:
            return raw  # return raw if parsing fails

    @staticmethod
    def _compute_line_starts(code: str) -> List[int]:
        starts: List[int] = [0]
        for m in re.finditer(r'\n', code):
            starts.append(m.end())
        return starts

    @staticmethod
    def _pos_from_index(line_starts: List[int], index: int) -> Tuple[int, int]:
        # lineno 1-based, col 0-based
        i = bisect.bisect_right(line_starts, index) - 1
        lineno = i + 1
        col = index - line_starts[i]
        return lineno, col


class TokenStream:
    """Simple wrapper around token iterable to provide peek/next and convenience for parsers."""

    def __init__(self, token_iter: Iterable[Token], emit_pos: bool = False):
        self._it = iter(token_iter)
        self._buffer: List[Token] = []
        self.emit_pos = emit_pos

    def _fill(self, n: int = 1) -> None:
        try:
            while len(self._buffer) < n:
                self._buffer.append(next(self._it))
        except StopIteration:
            return

    def peek(self, n: int = 0) -> Optional[Token]:
        self._fill(n + 1)
        return self._buffer[n] if n < len(self._buffer) else None

    def next(self) -> Optional[Token]:
        self._fill(1)
        return self._buffer.pop(0) if self._buffer else None

    def __iter__(self):
        while True:
            t = self.next()
            if t is None:
                break
            yield t


# -------------------------
# CLI self-test (executable)
# -------------------------
if __name__ == "__main__":
    SAMPLE = r'''
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
'''

    # Legacy simple usage
    lexer = InstryxLexer(LexerConfig(emit_positions=False))
    simple = lexer.tokenize_simple(SAMPLE)
    print("Simple tokens (first 20):")
    for t in simple[:20]:
        print(t)

    # Rich tokens with positions
    lexer_pos = InstryxLexer(LexerConfig(emit_positions=True))
    tokens_pos = lexer_pos.tokenize(SAMPLE)
    print("\nTokens with positions (first 24):")
    for t in tokens_pos[:24]:
        print(t)

    # TokenStream usage (parser-friendly)
    stream = lexer_pos.stream(SAMPLE)
    print("\nStream peek/next demo:")
    print("peek0:", stream.peek(0))
    print("next:", stream.next())
    print("peek0:", stream.peek(0))

    # Basic self-checks
    assert any(tok[0] == 'KEYWORD' and tok[1] == 'func' for tok in simple), "keyword detection failed"
    print("\ninstryx_lexer self-test passed.")
