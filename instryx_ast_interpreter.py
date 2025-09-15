# instryx_ast_interpreter.py
# Production-ready AST Interpreter for the Instryx Language
# Author: Violet Magenta / VACU Technologies
# License: MIT

from instryx_parser import InstryxParser, ASTNode

class RuntimeContext:
    def __init__(self):
        self.variables = {}
        self.functions = {}
        self.output = []

    def log(self, message):
        self.output.append(message)
        print(message)


class InstryxInterpreter:
    def __init__(self):
        self.parser = InstryxParser()
        self.ctx = RuntimeContext()

    def interpret(self, code: str):
        ast = self.parser.parse(code)
        self.eval_node(ast)

    def eval_node(self, node: ASTNode):
        method_name = f'eval_{node.node_type}'
        method = getattr(self, method_name, self.eval_unknown)
        return method(node)

    def eval_unknown(self, node: ASTNode):
        raise Exception(f"Unknown AST node type: {node.node_type}")

    def eval_Program(self, node: ASTNode):
        for child in node.children:
            self.eval_node(child)

    def eval_Block(self, node: ASTNode):
        for stmt in node.children:
            self.eval_node(stmt)

    def eval_Main(self, node: ASTNode):
        self.ctx.log("▶ Executing main()")
        self.eval_node(node.children[0])

    def eval_Function(self, node: ASTNode):
        name = node.value
        self.ctx.functions[name] = node
        self.ctx.log(f"🛠 Registered function: {name}")

    def eval_ExprStmt(self, node: ASTNode):
        return self.eval_node(node.children[0])

    def eval_Assign(self, node: ASTNode):
        var_name = node.value
        value = self.eval_node(node.children[0])
        self.ctx.variables[var_name] = value
        self.ctx.log(f"🔧 Assigned {var_name} = {value}")

    def eval_ID(self, node: ASTNode):
        return self.ctx.variables.get(node.value, None)

    def eval_Number(self, node: ASTNode):
        return float(node.value) if '.' in node.value else int(node.value)

    def eval_String(self, node: ASTNode):
        return node.value.strip('"')

    def eval_Call(self, node: ASTNode):
        func_name = node.value
        args = [self.eval_node(arg) for arg in node.children]
        if func_name == "print":
            self.ctx.log(f"🖨️  {' '.join(map(str, args))}")
        elif func_name == "log":
            self.ctx.log(f"📝 {args[0]}")
        elif func_name == "alert":
            self.ctx.log(f"⚠️ ALERT: {args[0]}")
        elif func_name in self.ctx.functions:
            func_def = self.ctx.functions[func_name]
            param_nodes = func_def.children[0].children
            local_vars_backup = self.ctx.variables.copy()
            for i, param in enumerate(param_nodes):
                param_name = param.value
                self.ctx.variables[param_name] = args[i] if i < len(args) else None
            self.eval_node(func_def.children[1])
            self.ctx.variables = local_vars_backup
        else:
            raise Exception(f"Undefined function: {func_name}")

    def eval_Macro(self, node: ASTNode):
        self.ctx.log(f"⚙️ Macro: {node.value} on {node.children[0].value}")

    def eval_Quarantine(self, node: ASTNode):
        self.ctx.log("🛡️ Entering quarantine block...")
        try:
            if node.children[0]:
                self.eval_node(node.children[0])  # try
        except Exception as e:
            self.ctx.log(f"⚠️ Exception: {str(e)}")
            if node.children[1]:
                self.eval_node(node.children[1])  # replace
            elif node.children[2]:
                self.eval_node(node.children[2])  # erase
            else:
                raise

# Test block (can be removed in production)
if __name__ == "__main__":
    interpreter = InstryxInterpreter()
    sample_code = """
    -- Load user data
    @inject db.conn;

    func load_user(uid) {
        quarantine try {
            data = "User42";
            print: "Loaded", data;
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
    interpreter.interpret(sample_code)
