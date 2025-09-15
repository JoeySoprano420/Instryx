# instryx_parser.py
# Production-ready Recursive Descent Parser for the Instryx Language
# Author: Violet Magenta / VACU Technologies
# License: MIT

from instryx_lexer import InstryxLexer, Token

class ASTNode:
    def __init__(self, node_type: str, value=None, children=None):
        self.node_type = node_type
        self.value = value
        self.children = children if children else []

    def __repr__(self):
        return f"ASTNode({self.node_type!r}, {self.value!r}, {self.children!r})"


class InstryxParser:
    def __init__(self):
        self.lexer = InstryxLexer()
        self.tokens = []
        self.pos = 0

    def parse(self, code: str) -> ASTNode:
        self.tokens = self.lexer.tokenize(code)
        self.pos = 0
        return self.program()

    def current(self) -> Token:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else ('EOF', '')

    def consume(self, expected_type=None) -> Token:
        token = self.current()
        if expected_type and token[0] != expected_type:
            raise SyntaxError(f"Expected {expected_type}, got {token[0]} ({token[1]})")
        self.pos += 1
        return token

    def match(self, *types) -> bool:
        return self.current()[0] in types

    def program(self) -> ASTNode:
        statements = []
        while not self.match('EOF'):
            stmt = self.statement()
            statements.append(stmt)
        return ASTNode('Program', children=statements)

    def statement(self) -> ASTNode:
        if self.match('KEYWORD') and self.current()[1] == 'func':
            return self.function_definition()
        elif self.match('MACRO'):
            macro = self.consume('MACRO')
            id1 = self.consume('ID')
            if self.match('DOT'):
                self.consume('DOT')
                id2 = self.consume('ID')
                self.consume('END')
                return ASTNode('Macro', macro[1], [ASTNode('ID', f"{id1[1]}.{id2[1]}")])
            self.consume('END')
            return ASTNode('Macro', macro[1], [ASTNode('ID', id1[1])])
        elif self.match('KEYWORD') and self.current()[1] == 'quarantine':
            return self.quarantine_block()
        elif self.match('KEYWORD') and self.current()[1] == 'main':
            return self.main_block()
        else:
            return self.expression_statement()

    def main_block(self) -> ASTNode:
        self.consume('KEYWORD')  # main
        self.consume('LPAREN')
        self.consume('RPAREN')
        body = self.block()
        return ASTNode('Main', children=[body])

    def function_definition(self) -> ASTNode:
        self.consume('KEYWORD')  # func
        name = self.consume('ID')
        self.consume('LPAREN')
        params = []
        if not self.match('RPAREN'):
            params.append(self.consume('ID')[1])
            while self.match('COMMA'):
                self.consume('COMMA')
                params.append(self.consume('ID')[1])
        self.consume('RPAREN')
        body = self.block()
        return ASTNode('Function', name[1], [ASTNode('Params', children=[ASTNode('ID', p) for p in params]), body])

    def quarantine_block(self) -> ASTNode:
        self.consume('KEYWORD')  # quarantine
        try_block = None
        replace_block = None
        erase_block = None
        if self.match('KEYWORD') and self.current()[1] == 'try':
            self.consume('KEYWORD')
            try_block = self.block()
        if self.match('KEYWORD') and self.current()[1] == 'replace':
            self.consume('KEYWORD')
            replace_block = self.block()
        if self.match('KEYWORD') and self.current()[1] == 'erase':
            self.consume('KEYWORD')
            erase_block = self.block()
        return ASTNode('Quarantine', children=[try_block, replace_block, erase_block])

    def block(self) -> ASTNode:
        self.consume('LBRACE')
        stmts = []
        while not self.match('RBRACE'):
            stmts.append(self.statement())
        self.consume('RBRACE')
        return ASTNode('Block', children=stmts)

    def expression_statement(self) -> ASTNode:
        expr = self.expression()
        self.consume('END')
        return ASTNode('ExprStmt', children=[expr])

    def expression(self) -> ASTNode:
        if self.match('ID'):
            id_token = self.consume('ID')
            if self.match('ASSIGN'):
                self.consume('ASSIGN')
                value = self.expression()
                return ASTNode('Assign', id_token[1], [value])
            elif self.match('LPAREN'):
                self.consume('LPAREN')
                args = []
                if not self.match('RPAREN'):
                    args.append(self.expression())
                    while self.match('COMMA'):
                        self.consume('COMMA')
                        args.append(self.expression())
                self.consume('RPAREN')
                return ASTNode('Call', id_token[1], args)
            else:
                return ASTNode('ID', id_token[1])
        elif self.match('STRING'):
            return ASTNode('String', self.consume('STRING')[1])
        elif self.match('NUMBER'):
            return ASTNode('Number', self.consume('NUMBER')[1])
        else:
            raise SyntaxError(f"Unexpected token: {self.current()}")


# Test block (can be removed in production)
if __name__ == "__main__":
    parser = InstryxParser()
    sample_code = """
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
    ast = parser.parse(sample_code)
    print(ast)
