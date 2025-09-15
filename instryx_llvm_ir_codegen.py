# instryx_llvm_ir_codegen.py
# Production-ready LLVM IR Code Generator for Instryx
# Author: Violet Magenta / VACU Technologies
# License: MIT

from instryx_parser import InstryxParser, ASTNode
from llvmlite import ir, binding

class InstryxLLVMCodegen:
    def __init__(self):
        self.parser = InstryxParser()
        self.module = ir.Module(name="instryx")
        self.builder = None
        self.funcs = {}
        self.printf = None

    def generate(self, code: str) -> str:
        ast = self.parser.parse(code)
        self._declare_builtins()
        self._eval_node(ast)
        return str(self.module)

    def _declare_builtins(self):
        voidptr_ty = ir.IntType(8).as_pointer()
        printf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
        self.printf = ir.Function(self.module, printf_ty, name="printf")

    def _eval_node(self, node: ASTNode):
        method = getattr(self, f'_eval_{node.node_type}', self._eval_unknown)
        return method(node)

    def _eval_unknown(self, node: ASTNode):
        raise NotImplementedError(f"Unknown AST node type: {node.node_type}")

    def _eval_Program(self, node: ASTNode):
        for child in node.children:
            self._eval_node(child)

    def _eval_Function(self, node: ASTNode):
        name = node.value
        params_node, body_node = node.children
        param_names = [p.value for p in params_node.children]
        func_ty = ir.FunctionType(ir.VoidType(), [ir.IntType(32)] * len(param_names))
        func = ir.Function(self.module, func_ty, name=name)
        self.funcs[name] = func

        block = func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)

        for i, arg in enumerate(func.args):
            arg.name = param_names[i]

        self._eval_node(body_node)
        self.builder.ret_void()

    def _eval_Main(self, node: ASTNode):
        func_ty = ir.FunctionType(ir.VoidType(), [])
        func = ir.Function(self.module, func_ty, name="main")
        block = func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)
        self._eval_node(node.children[0])
        self.builder.ret_void()

    def _eval_Block(self, node: ASTNode):
        for stmt in node.children:
            self._eval_node(stmt)

    def _eval_ExprStmt(self, node: ASTNode):
        return self._eval_node(node.children[0])

    def _eval_Call(self, node: ASTNode):
        func_name = node.value
        args = [self._eval_node(arg) for arg in node.children]

        if func_name == "print":
            fmt = "%s\n\0"
            c_fmt = ir.Constant(ir.ArrayType(ir.IntType(8), len(fmt)),
                                bytearray(fmt.encode("utf8")))
            global_fmt = ir.GlobalVariable(self.module, c_fmt.type, name="fstr")
            global_fmt.linkage = 'internal'
            global_fmt.global_constant = True
            global_fmt.initializer = c_fmt
            fmt_ptr = self.builder.bitcast(global_fmt, ir.IntType(8).as_pointer())
            self.builder.call(self.printf, [fmt_ptr] + args)
        elif func_name in self.funcs:
            return self.builder.call(self.funcs[func_name], args)

    def _eval_String(self, node: ASTNode):
        string_val = node.value.strip('"') + "\0"
        const_str = ir.Constant(ir.ArrayType(ir.IntType(8), len(string_val)),
                                bytearray(string_val.encode("utf8")))
        global_str = ir.GlobalVariable(self.module, const_str.type, name="str")
        global_str.linkage = 'internal'
        global_str.global_constant = True
        global_str.initializer = const_str
        return self.builder.bitcast(global_str, ir.IntType(8).as_pointer())


# Test block (can be removed in production)
if __name__ == "__main__":
    generator = InstryxLLVMCodegen()
    sample_code = """
    func greet(uid) {
        print: "Hello from Instryx";
    };

    main() {
        greet(0);
    };
    """
    llvm_ir = generator.generate(sample_code)
    print(llvm_ir)
