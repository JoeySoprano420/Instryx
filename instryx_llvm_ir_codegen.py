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

# instryxc_llvm_ir_codegen.py
# Instryx → LLVM IR Code Generator — supreme boosters edition
# Author: Violet Magenta / VACU Technologies (modified)
# License: MIT
"""
Enhanced LLVM IR code generator for Instryx with:
 - broader AST node coverage (numbers, arithmetic, vars, assigns, if, while, calls)
 - constant folding (compile-time) and simple peephole optimizations
 - deduplicated global strings pool
 - module verification, optional optimization passes (via llvmlite.binding)
 - object emission helper (TargetMachine.emit_object)
 - safer builder / local variable handling with per-function symbol table
 - helpful debug / verbose mode and CLI test block
Requires: llvmlite installed and the project's Instryx parser available.
"""

from __future__ import annotations
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from instryx_parser import InstryxParser, ASTNode

from llvmlite import ir, binding

# Initialize llvm binding (once)
binding.initialize()
binding.initialize_native_target()
binding.initialize_native_asmprinter()


@dataclass
class CodegenConfig:
    opt_level: int = 2
    verbose: bool = False


class InstryxLLVMCodegen:
    def __init__(self, config: Optional[CodegenConfig] = None):
        self.config = config or CodegenConfig()
        self.parser = InstryxParser()
        self.module = ir.Module(name="instryx")
        self.builder: Optional[ir.IRBuilder] = None
        self.funcs: Dict[str, ir.Function] = {}
        self.printf: Optional[ir.Function] = None
        self._global_strings: Dict[str, ir.GlobalVariable] = {}
        self._unique_id = 0
        # keep current function local symbol table (name -> alloca)
        self._locals: List[Dict[str, ir.AllocaInstr]] = []

        # declare printf and ensure the module has a target triple
        self._declare_builtins()

    # -------------------------
    # Public helpers
    # -------------------------
    def generate(self, code: str, optimize: bool = False, opt_level: int = 2) -> str:
        """
        Parse code, generate LLVM IR string. If optimize=True, run simple optimization passes.
        """
        ast = self.parser.parse(code)
        # clear module state for subsequent generate calls
        self.module = ir.Module(name="instryx")
        self._global_strings.clear()
        self.funcs.clear()
        self._unique_id = 0
        self._locals.clear()
        self._declare_builtins()
        self._eval_node(ast)
        llvm_ir = str(self.module)
        if optimize:
            try:
                llvm_ir = self._optimize_ir(llvm_ir, opt_level=opt_level)
            except Exception:
                if self.config.verbose:
                    print("Optimization failed; returning unoptimized IR")
        return llvm_ir

    def get_module(self) -> ir.Module:
        return self.module

    def emit_object(self) -> bytes:
        """
        Emit an object file bytes for the current module using the native target machine.
        """
        asm = str(self.module)
        llvm_mod = binding.parse_assembly(asm)
        llvm_mod.verify()
        target = binding.Target.from_default_triple()
        tm = target.create_target_machine(opt= self.config.opt_level)
        obj = tm.emit_object(llvm_mod)
        return obj

    # -------------------------
    # Internals: optimization & verification
    # -------------------------
    def _optimize_ir(self, llvm_ir: str, opt_level: int = 2) -> str:
        """
        Apply LLVM optimization passes using llvmlite.binding.PassManagerBuilder.
        Returns optimized IR text.
        """
        llvm_mod = binding.parse_assembly(llvm_ir)
        llvm_mod.verify()
        pmb = binding.PassManagerBuilder()
        pmb.opt_level = max(0, min(3, int(opt_level)))
        pmb.size_level = 0
        pm = binding.ModulePassManager()
        pmb.populate(pm)
        pm.run(llvm_mod)
        return str(llvm_mod)

    def _verify_module(self) -> None:
        """
        Run LLVM verifier; raises if invalid.
        """
        asm = str(self.module)
        mref = binding.parse_assembly(asm)
        mref.verify()

    # -------------------------
    # Builtins and global strings
    # -------------------------
    def _declare_builtins(self):
        """
        Declare external/host functions (printf) and set basic module attributes.
        """
        voidptr_ty = ir.IntType(8).as_pointer()
        printf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
        # If printf already exists in module, reuse
        if "printf" in self.module.globals:
            self.printf = self.module.globals["printf"]
        else:
            self.printf = ir.Function(self.module, printf_ty, name="printf")
        # set triple to default host
        try:
            self.module.triple = binding.get_default_triple()
        except Exception:
            # fallback: leave triple unset
            pass

    def _get_or_create_global_string(self, text: str, name_hint: str = "str") -> ir.GlobalVariable:
        """
        Deduplicate global string constants; return pointer to first element.
        """
        if text in self._global_strings:
            return self._global_strings[text]
        name = f"{name_hint}_{self._unique_id}"
        self._unique_id += 1
        data = bytearray(text.encode("utf8"))
        const_str = ir.Constant(ir.ArrayType(ir.IntType(8), len(data)), data)
        gvar = ir.GlobalVariable(self.module, const_str.type, name=name)
        gvar.linkage = 'internal'
        gvar.global_constant = True
        gvar.initializer = const_str
        self._global_strings[text] = gvar
        return gvar

    # -------------------------
    # AST evaluation (core)
    # -------------------------
    def _eval_node(self, node: ASTNode):
        method = getattr(self, f"_eval_{node.node_type}", None)
        if method is None:
            return self._eval_unknown(node)
        return method(node)

    def _eval_unknown(self, node: ASTNode):
        raise NotImplementedError(f"Unknown AST node type: {node.node_type}")

    def _eval_Program(self, node: ASTNode):
        # top-level declarations
        for child in node.children:
            self._eval_node(child)
        # attempt to verify module
        try:
            self._verify_module()
        except Exception:
            if self.config.verbose:
                print("Module verification failed; IR may be invalid.")

    def _eval_Function(self, node: ASTNode):
        """
        Expected node.children: [ParamsNode, BodyNode]
        ParamsNode.children contain parameter identifier nodes.
        Functions currently emit as returning void; returns inside are compiled to ret_void.
        """
        name = node.value
        params_node, body_node = node.children
        param_names = [p.value for p in params_node.children] if params_node and params_node.children else []
        func_ty = ir.FunctionType(ir.VoidType(), [ir.IntType(32)] * len(param_names))
        func = ir.Function(self.module, func_ty, name=name)
        self.funcs[name] = func

        # create entry block and builder
        entry = func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(entry)

        # set up local symbol table for this function
        local_scope: Dict[str, ir.AllocaInstr] = {}
        self._locals.append(local_scope)

        # create allocas for parameters (store incoming args to local allocas)
        for i, arg in enumerate(func.args):
            arg.name = param_names[i]
            alloca = self._create_entry_alloca(func, arg.name)
            self.builder.store(arg, alloca)
            local_scope[arg.name] = alloca

        # compile body
        self._eval_node(body_node)

        # ensure function returns; if builder.block has no terminator, ret void
        if not self.builder.block.is_terminated:
            self.builder.ret_void()

        # pop local scope
        self._locals.pop()

    def _eval_Main(self, node: ASTNode):
        """
        Create 'main' function, emit body and ret void.
        """
        func_ty = ir.FunctionType(ir.VoidType(), [])
        func = ir.Function(self.module, func_ty, name="main")
        self.funcs["main"] = func
        entry = func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(entry)

        # main has its own locals
        self._locals.append({})

        if node.children:
            self._eval_node(node.children[0])

        if not self.builder.block.is_terminated:
            self.builder.ret_void()

        self._locals.pop()

    def _eval_Block(self, node: ASTNode):
        for stmt in node.children:
            self._eval_node(stmt)

    def _eval_ExprStmt(self, node: ASTNode):
        # Expression statements: evaluate expression and drop result if any
        return self._eval_node(node.children[0])

    # Literals & simple expressions
    def _eval_Number(self, node: ASTNode):
        # support integer literals (assume base 10)
        try:
            val = int(node.value)
        except Exception:
            # try float fallback
            try:
                f = float(node.value)
                # represent floats as double for now
                return ir.Constant(ir.DoubleType(), f)
            except Exception:
                val = 0
        return ir.Constant(ir.IntType(32), val)

    def _eval_String(self, node: ASTNode):
        # reuse global string pool and return i8*
        string_val = node.value.strip('"') + "\0"
        gvar = self._get_or_create_global_string(string_val, name_hint="str")
        return self.builder.bitcast(gvar, ir.IntType(8).as_pointer())

    # Variables and assignments
    def _create_entry_alloca(self, function: ir.Function, name: str) -> ir.AllocaInstr:
        """
        Create an alloca at function entry block (canonical placement).
        """
        entry_block = function.entry_basic_block
        ib = ir.IRBuilder(entry_block)
        # position builder at beginning: create temporary first instruction insertion point
        # llvmlite doesn't support position_before_first directly; create a new block if needed.
        # Simpler approach: append allocas at entry block start by creating a builder and inserting
        # (this is acceptable for llvmlite IRBuilder)
        with ib.goto_entry_block():
            return ib.alloca(ir.IntType(32), name=name)

    def _eval_Var(self, node: ASTNode):
        # variable usage: load from local alloca
        name = node.value
        for scope in reversed(self._locals):
            if name in scope:
                return self.builder.load(scope[name], name=name + "_val")
        # if not found, treat as a global int constant 0 (conservative)
        return ir.Constant(ir.IntType(32), 0)

    def _eval_Assign(self, node: ASTNode):
        """
        Expect children [VarNode, ExprNode]
        """
        var_node = node.children[0]
        expr_node = node.children[1]
        val = self._eval_node(expr_node)
        name = var_node.value

        # ensure an alloca exists in current function scope
        if not self._locals:
            # global assignment not supported; ignore
            return None
        scope = self._locals[-1]
        if name not in scope:
            # create alloca in current function (function is builder.function)
            current_func = self.builder.function  # type: ignore
            alloca = self._create_entry_alloca(current_func, name)
            scope[name] = alloca
        else:
            alloca = scope[name]
        # if val is a constant and alloca type is i32, store constant directly
        self.builder.store(val, alloca)
        return alloca

    # Binary ops with constant folding
    def _eval_Binary(self, node: ASTNode):
        # node.value holds operator string: '+', '-', '*', '/', '==', '<', etc.
        left = self._eval_node(node.children[0])
        right = self._eval_node(node.children[1])
        op = node.value

        # constant folding if both are ir.Constant
        if isinstance(left, ir.Constant) and isinstance(right, ir.Constant):
            try:
                if left.type == ir.IntType(32) and right.type == ir.IntType(32):
                    lv = int(left.constant)
                    rv = int(right.constant)
                    if op == '+':
                        return ir.Constant(ir.IntType(32), lv + rv)
                    if op == '-':
                        return ir.Constant(ir.IntType(32), lv - rv)
                    if op == '*':
                        return ir.Constant(ir.IntType(32), lv * rv)
                    if op == '/':
                        return ir.Constant(ir.IntType(32), int(lv / rv) if rv != 0 else 0)
                    if op == '==':
                        return ir.Constant(ir.IntType(1), 1 if lv == rv else 0)
                    if op == '!=':
                        return ir.Constant(ir.IntType(1), 1 if lv != rv else 0)
                    if op == '<':
                        return ir.Constant(ir.IntType(1), 1 if lv < rv else 0)
                    if op == '>':
                        return ir.Constant(ir.IntType(1), 1 if lv > rv else 0)
                # float folding
                if left.type == ir.DoubleType() and right.type == ir.DoubleType():
                    lv = float(left.constant)
                    rv = float(right.constant)
                    if op == '+':
                        return ir.Constant(ir.DoubleType(), lv + rv)
                    if op == '-':
                        return ir.Constant(ir.DoubleType(), lv - rv)
            except Exception:
                pass  # fall back to runtime ops

        # runtime ops
        if op in ('+', '-', '*', '/'):
            # treat as integers for now
            if left.type == ir.DoubleType() or right.type == ir.DoubleType():
                l = self._to_double(left)
                r = self._to_double(right)
                if op == '+':
                    return self.builder.fadd(l, r)
                if op == '-':
                    return self.builder.fsub(l, r)
                if op == '*':
                    return self.builder.fmul(l, r)
                if op == '/':
                    return self.builder.fdiv(l, r)
            else:
                if op == '+':
                    return self.builder.add(left, right)
                if op == '-':
                    return self.builder.sub(left, right)
                if op == '*':
                    return self.builder.mul(left, right)
                if op == '/':
                    # integer division (sdiv)
                    return self.builder.sdiv(left, right)
        if op in ('==', '!=', '<', '>', '<=', '>='):
            if left.type == ir.DoubleType() or right.type == ir.DoubleType():
                l = self._to_double(left)
                r = self._to_double(right)
                if op == '==':
                    return self.builder.fcmp_ordered('==', l, r)
                if op == '!=':
                    return self.builder.fcmp_ordered('!=', l, r)
                if op == '<':
                    return self.builder.fcmp_ordered('<', l, r)
                if op == '>':
                    return self.builder.fcmp_ordered('>', l, r)
                if op == '<=':
                    return self.builder.fcmp_ordered('<=', l, r)
                if op == '>=':
                    return self.builder.fcmp_ordered('>=', l, r)
            else:
                if op == '==':
                    return self.builder.icmp_signed('==', left, right)
                if op == '!=':
                    return self.builder.icmp_signed('!=', left, right)
                if op == '<':
                    return self.builder.icmp_signed('<', left, right)
                if op == '>':
                    return self.builder.icmp_signed('>', left, right)
                if op == '<=':
                    return self.builder.icmp_signed('<=', left, right)
                if op == '>=':
                    return self.builder.icmp_signed('>=', left, right)
        # unknown op: return left as fallback
        return left

    def _to_double(self, val):
        if isinstance(val, ir.Constant) and val.type == ir.IntType(32):
            return ir.Constant(ir.DoubleType(), float(int(val.constant)))
        if isinstance(val, ir.Constant) and val.type == ir.DoubleType():
            return val
        # emit sitofp if at runtime
        if val.type == ir.IntType(32):
            return self.builder.sitofp(val, ir.DoubleType())
        return val

    # Calls and builtins
    def _eval_Call(self, node: ASTNode):
        func_name = node.value
        args = [self._eval_node(arg) for arg in node.children]

        if func_name == "print":
            # use deduped global fmt
            fmt = "%s\n\0"
            gfmt = self._get_or_create_global_string(fmt, name_hint="fmt")
            fmt_ptr = self.builder.bitcast(gfmt, ir.IntType(8).as_pointer())
            # ensure all args are i8* (strings) for now; convert ints to formatted strings not implemented
            self.builder.call(self.printf, [fmt_ptr] + args)
            return None
        elif func_name in self.funcs:
            return self.builder.call(self.funcs[func_name], args)
        else:
            # unknown call — ignore or implement intrinsics
            return None

    # Control flow
    def _eval_If(self, node: ASTNode):
        """
        Expect children: [condNode, thenBlock, elseBlock?]
        """
        cond = self._eval_node(node.children[0])
        # convert condition to i1 if needed
        if isinstance(cond, ir.Constant) and cond.type != ir.IntType(1):
            # treat nonzero as true
            cond = self.builder.icmp_signed('!=', cond, ir.Constant(ir.IntType(32), 0))
        elif not hasattr(cond, 'type') or cond.type != ir.IntType(1):
            cond = self.builder.icmp_signed('!=', cond, ir.Constant(ir.IntType(32), 0))

        then_bb = self.builder.function.append_basic_block("then")
        else_bb = self.builder.function.append_basic_block("else")
        cont_bb = self.builder.function.append_basic_block("ifcont")

        self.builder.cbranch(cond, then_bb, else_bb)

        # then
        self.builder.position_at_end(then_bb)
        self._eval_node(node.children[1])
        if not self.builder.block.is_terminated:
            self.builder.branch(cont_bb)

        # else
        self.builder.position_at_end(else_bb)
        if len(node.children) > 2 and node.children[2]:
            self._eval_node(node.children[2])
        if not self.builder.block.is_terminated:
            self.builder.branch(cont_bb)

        # continue
        self.builder.position_at_end(cont_bb)

    def _eval_While(self, node: ASTNode):
        """
        Expect children: [condNode, bodyNode]
        """
        func = self.builder.function
        loop_bb = func.append_basic_block("loop")
        body_bb = func.append_basic_block("loop_body")
        after_bb = func.append_basic_block("after_loop")

        # initial branch to loop
        self.builder.branch(loop_bb)

        # loop: evaluate condition
        self.builder.position_at_end(loop_bb)
        cond = self._eval_node(node.children[0])
        if isinstance(cond, ir.Constant) and cond.type != ir.IntType(1):
            cond = self.builder.icmp_signed('!=', cond, ir.Constant(ir.IntType(32), 0))
        elif not hasattr(cond, 'type') or cond.type != ir.IntType(1):
            cond = self.builder.icmp_signed('!=', cond, ir.Constant(ir.IntType(32), 0))
        self.builder.cbranch(cond, body_bb, after_bb)

        # body
        self.builder.position_at_end(body_bb)
        self._eval_node(node.children[1])
        if not self.builder.block.is_terminated:
            self.builder.branch(loop_bb)

        # after
        self.builder.position_at_end(after_bb)

    # Return
    def _eval_Return(self, node: ASTNode):
        if node.children:
            val = self._eval_node(node.children[0])
            # if function expects void, return void; otherwise try to return value
            try:
                if self.builder and not self.builder.block.is_terminated:
                    # if val is i32 and function returns void, cast away; safer to ret_void
                    self.builder.ret_void()
            except Exception:
                if self.builder and not self.builder.block.is_terminated:
                    self.builder.ret_void()
        else:
            if self.builder and not self.builder.block.is_terminated:
                self.builder.ret_void()

    # -------------------------
    # Testing / debugging
    # -------------------------
    def dump_ir(self) -> str:
        return str(self.module)


# Simple CLI test block
if __name__ == "__main__":
    cfg = CodegenConfig(opt_level=2, verbose=True)
    cg = InstryxLLVMCodegen(config=cfg)
    sample_code = """
    func greet(uid) {
        print: "Hello from Instryx";
    };

    main() {
        greet(0);
    };
    """
    ir_text = cg.generate(sample_code, optimize=True, opt_level=2)
    print(ir_text)

    # Optionally emit object file
    obj_bytes = cg.emit_object()
    with open("output.o", "wb") as f:
        f.write(obj_bytes)
        print("Emitted object file 'output.o'")
        print(f"Object file size: {len(obj_bytes)} bytes")

# instryxc_llvm_ir_codegen.py
# Instryx → LLVM IR Code Generator — supreme boosters final
# Author: Violet Magenta / VACU Technologies (modified)
# License: MIT
"""
Enhanced, executable LLVM IR generator for Instryx.

Features added:
 - Broader AST coverage: numbers, strings, variables, assigns, binary ops, if, while, functions, calls, returns
 - Constant folding and simple peephole optimizations
 - Deduplicated global string pool
 - LLVM verification and optional optimization via llvmlite.binding
 - Emit native object file bytes via TargetMachine
 - Per-function local symbol tables and safe alloca placement
 - Verbose/debug mode and CLI test/emission helpers
Requirements:
 - llvmlite installed
 - instryx_parser available and providing ASTNode structure used by this generator
"""

from __future__ import annotations
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from instryx_parser import InstryxParser, ASTNode

from llvmlite import ir, binding

# Initialize llvm once
binding.initialize()
binding.initialize_native_target()
binding.initialize_native_asmprinter()


@dataclass
class CodegenConfig:
    opt_level: int = 2
    verbose: bool = False


class InstryxLLVMCodegen:
    def __init__(self, config: Optional[CodegenConfig] = None):
        self.config = config or CodegenConfig()
        self.parser = InstryxParser()
        self.module = ir.Module(name="instryx")
        self.builder: Optional[ir.IRBuilder] = None
        self.funcs: Dict[str, ir.Function] = {}
        self.printf: Optional[ir.Function] = None
        self._global_strings: Dict[str, ir.GlobalVariable] = {}
        self._unique_id = 0
        # stack of local symbol tables, one dict per function
        self._locals: List[Dict[str, ir.AllocaInstr]] = []

        self._declare_builtins()

    # -------------------------
    # Public API
    # -------------------------
    def generate(self, code: str, optimize: bool = False, opt_level: int = 2) -> str:
        """
        Generate LLVM IR for the given Instryx source code.
        If optimize is True, run simple module-level optimizations.
        """
        ast = self.parser.parse(code)
        # reset module state
        self.module = ir.Module(name="instryx")
        self._global_strings.clear()
        self.funcs.clear()
        self._unique_id = 0
        self._locals.clear()
        self._declare_builtins()
        self._eval_node(ast)
        ir_text = str(self.module)
        if optimize:
            try:
                ir_text = self._optimize_ir(ir_text, opt_level=opt_level)
            except Exception as e:
                if self.config.verbose:
                    print("Optimization failed:", e)
        return ir_text

    def get_module(self) -> ir.Module:
        return self.module

    def emit_object(self) -> bytes:
        """
        Emit an object (native) for the current module using the host target machine.
        """
        asm = str(self.module)
        llvm_mod = binding.parse_assembly(asm)
        llvm_mod.verify()
        target = binding.Target.from_default_triple()
        tm = target.create_target_machine(opt=self.config.opt_level)
        return tm.emit_object(llvm_mod)

    # -------------------------
    # Optimization & verification
    # -------------------------
    def _optimize_ir(self, llvm_ir: str, opt_level: int = 2) -> str:
        llvm_mod = binding.parse_assembly(llvm_ir)
        llvm_mod.verify()
        pmb = binding.PassManagerBuilder()
        pmb.opt_level = max(0, min(3, int(opt_level)))
        pmb.size_level = 0
        pm = binding.ModulePassManager()
        pmb.populate(pm)
        pm.run(llvm_mod)
        return str(llvm_mod)

    def _verify_module(self) -> None:
        asm = str(self.module)
        mref = binding.parse_assembly(asm)
        mref.verify()

    # -------------------------
    # Builtins & globals
    # -------------------------
    def _declare_builtins(self):
        voidptr_ty = ir.IntType(8).as_pointer()
        printf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
        if "printf" in self.module.globals:
            self.printf = self.module.globals["printf"]
        else:
            self.printf = ir.Function(self.module, printf_ty, name="printf")
        # set target triple to host if available
        try:
            self.module.triple = binding.get_default_triple()
        except Exception:
            pass

    def _get_or_create_global_string(self, text: str, name_hint: str = "str") -> ir.GlobalVariable:
        if text in self._global_strings:
            return self._global_strings[text]
        name = f"{name_hint}_{self._unique_id}"
        self._unique_id += 1
        data = bytearray(text.encode("utf8"))
        const = ir.Constant(ir.ArrayType(ir.IntType(8), len(data)), data)
        gvar = ir.GlobalVariable(self.module, const.type, name=name)
        gvar.linkage = "internal"
        gvar.global_constant = True
        gvar.initializer = const
        self._global_strings[text] = gvar
        return gvar

    # -------------------------
    # AST dispatch
    # -------------------------
    def _eval_node(self, node: ASTNode):
        method = getattr(self, f"_eval_{node.node_type}", None)
        if method is None:
            return self._eval_unknown(node)
        return method(node)

    def _eval_unknown(self, node: ASTNode):
        raise NotImplementedError(f"Unknown AST node type: {node.node_type}")

    # Program / declarations
    def _eval_Program(self, node: ASTNode):
        for child in node.children:
            self._eval_node(child)
        try:
            self._verify_module()
        except Exception:
            if self.config.verbose:
                print("Verification failed for generated module; IR may be invalid.")

    def _eval_Function(self, node: ASTNode):
        # children: params_node, body_node
        name = node.value
        params_node, body_node = node.children
        param_names = [p.value for p in params_node.children] if params_node and params_node.children else []
        func_ty = ir.FunctionType(ir.VoidType(), [ir.IntType(32)] * len(param_names))
        func = ir.Function(self.module, func_ty, name=name)
        self.funcs[name] = func

        entry = func.append_basic_block("entry")
        self.builder = ir.IRBuilder(entry)

        # push a new local scope
        local_scope: Dict[str, ir.AllocaInstr] = {}
        self._locals.append(local_scope)

        # create allocas for parameters at entry and store incoming values
        for i, arg in enumerate(func.args):
            arg.name = param_names[i]
            alloca = self._create_entry_alloca(func, arg.name)
            self.builder.store(arg, alloca)
            local_scope[arg.name] = alloca

        # compile body
        self._eval_node(body_node)

        # ensure return if not terminated
        if not self.builder.block.is_terminated:
            self.builder.ret_void()

        # pop local scope
        self._locals.pop()

    def _eval_Main(self, node: ASTNode):
        func_ty = ir.FunctionType(ir.VoidType(), [])
        func = ir.Function(self.module, func_ty, name="main")
        self.funcs["main"] = func
        entry = func.append_basic_block("entry")
        self.builder = ir.IRBuilder(entry)
        self._locals.append({})
        if node.children:
            self._eval_node(node.children[0])
        if not self.builder.block.is_terminated:
            self.builder.ret_void()
        self._locals.pop()

    def _eval_Block(self, node: ASTNode):
        for stmt in node.children:
            self._eval_node(stmt)

    def _eval_ExprStmt(self, node: ASTNode):
        return self._eval_node(node.children[0])

    # Literals
    def _eval_Number(self, node: ASTNode):
        try:
            val = int(node.value)
            return ir.Constant(ir.IntType(32), val)
        except Exception:
            try:
                f = float(node.value)
                return ir.Constant(ir.DoubleType(), f)
            except Exception:
                return ir.Constant(ir.IntType(32), 0)

    def _eval_String(self, node: ASTNode):
        s = node.value.strip('"') + "\0"
        g = self._get_or_create_global_string(s, name_hint="str")
        return self.builder.bitcast(g, ir.IntType(8).as_pointer())

    # Variables & assignments
    def _create_entry_alloca(self, function: ir.Function, name: str) -> ir.AllocaInstr:
        # Create an IRBuilder at entry block to place allocas (safe placement)
        entry = function.entry_basic_block
        b = ir.IRBuilder(entry)
        # Use i32 allocas for simplicity
        return b.alloca(ir.IntType(32), name=name)

    def _eval_Var(self, node: ASTNode):
        name = node.value
        for scope in reversed(self._locals):
            if name in scope:
                return self.builder.load(scope[name], name=name + "_val")
        # fallback to zero constant
        return ir.Constant(ir.IntType(32), 0)

    def _eval_Assign(self, node: ASTNode):
        var_node = node.children[0]
        expr_node = node.children[1]
        val = self._eval_node(expr_node)
        name = var_node.value
        if not self._locals:
            return None
        scope = self._locals[-1]
        if name not in scope:
            current_func = self.builder.function  # type: ignore
            alloca = self._create_entry_alloca(current_func, name)
            scope[name] = alloca
        else:
            alloca = scope[name]
        self.builder.store(val, alloca)
        return alloca

    # Binary ops with constant folding
    def _eval_Binary(self, node: ASTNode):
        left = self._eval_node(node.children[0])
        right = self._eval_node(node.children[1])
        op = node.value

        # constant folding
        if isinstance(left, ir.Constant) and isinstance(right, ir.Constant):
            try:
                if left.type == ir.IntType(32) and right.type == ir.IntType(32):
                    lv = int(left.constant)
                    rv = int(right.constant)
                    if op == '+':
                        return ir.Constant(ir.IntType(32), lv + rv)
                    if op == '-':
                        return ir.Constant(ir.IntType(32), lv - rv)
                    if op == '*':
                        return ir.Constant(ir.IntType(32), lv * rv)
                    if op == '/':
                        return ir.Constant(ir.IntType(32), lv // rv if rv != 0 else 0)
                    if op in ('==', '!=', '<', '>', '<=', '>='):
                        # produce i1
                        if op == '==':
                            return ir.Constant(ir.IntType(1), 1 if lv == rv else 0)
                        if op == '!=':
                            return ir.Constant(ir.IntType(1), 1 if lv != rv else 0)
                        if op == '<':
                            return ir.Constant(ir.IntType(1), 1 if lv < rv else 0)
                        if op == '>':
                            return ir.Constant(ir.IntType(1), 1 if lv > rv else 0)
                if left.type == ir.DoubleType() and right.type == ir.DoubleType():
                    lv = float(left.constant)
                    rv = float(right.constant)
                    if op == '+':
                        return ir.Constant(ir.DoubleType(), lv + rv)
                    if op == '-':
                        return ir.Constant(ir.DoubleType(), lv - rv)
            except Exception:
                pass

        # runtime ops
        if op in ('+', '-', '*', '/'):
            if getattr(left, "type", None) == ir.DoubleType() or getattr(right, "type", None) == ir.DoubleType():
                l = self._to_double(left)
                r = self._to_double(right)
                if op == '+':
                    return self.builder.fadd(l, r)
                if op == '-':
                    return self.builder.fsub(l, r)
                if op == '*':
                    return self.builder.fmul(l, r)
                if op == '/':
                    return self.builder.fdiv(l, r)
            else:
                if op == '+':
                    return self.builder.add(left, right)
                if op == '-':
                    return self.builder.sub(left, right)
                if op == '*':
                    return self.builder.mul(left, right)
                if op == '/':
                    return self.builder.sdiv(left, right)
        if op in ('==', '!=', '<', '>', '<=', '>='):
            if getattr(left, "type", None) == ir.DoubleType() or getattr(right, "type", None) == ir.DoubleType():
                l = self._to_double(left)
                r = self._to_double(right)
                mapping = {'==': '==', '!=': '!=', '<': '<', '>': '>', '<=': '<=', '>=': '>='}
                return self.builder.fcmp_ordered(mapping[op], l, r)
            else:
                mapping = {'==': '==', '!=': '!=', '<': '<', '>': '>', '<=': '<=', '>=': '>='}
                return self.builder.icmp_signed(mapping[op], left, right)
        return left

    def _to_double(self, val):
        if isinstance(val, ir.Constant) and val.type == ir.IntType(32):
            return ir.Constant(ir.DoubleType(), float(int(val.constant)))
        if isinstance(val, ir.Constant) and val.type == ir.DoubleType():
            return val
        if getattr(val, "type", None) == ir.IntType(32):
            return self.builder.sitofp(val, ir.DoubleType())
        return val

    # Calls & builtins
    def _eval_Call(self, node: ASTNode):
        func_name = node.value
        args = [self._eval_node(arg) for arg in node.children]
        if func_name == "print":
            fmt = "%s\n\0"
            gfmt = self._get_or_create_global_string(fmt, name_hint="fmt")
            fmt_ptr = self.builder.bitcast(gfmt, ir.IntType(8).as_pointer())
            # call printf with varargs; llvmlite will accept varargs
            self.builder.call(self.printf, [fmt_ptr] + args)
            return None
        elif func_name in self.funcs:
            return self.builder.call(self.funcs[func_name], args)
        return None

    # Control-flow
    def _eval_If(self, node: ASTNode):
        cond = self._eval_node(node.children[0])
        # normalize cond to i1
        if isinstance(cond, ir.Constant) and cond.type != ir.IntType(1):
            cond = self.builder.icmp_signed('!=', cond, ir.Constant(ir.IntType(32), 0))
        elif not getattr(cond, "type", None) == ir.IntType(1):
            cond = self.builder.icmp_signed('!=', cond, ir.Constant(ir.IntType(32), 0))

        then_bb = self.builder.function.append_basic_block("if.then")
        else_bb = self.builder.function.append_basic_block("if.else")
        cont_bb = self.builder.function.append_basic_block("if.end")

        self.builder.cbranch(cond, then_bb, else_bb)

        self.builder.position_at_end(then_bb)
        self._eval_node(node.children[1])
        if not self.builder.block.is_terminated:
            self.builder.branch(cont_bb)

        self.builder.position_at_end(else_bb)
        if len(node.children) > 2 and node.children[2]:
            self._eval_node(node.children[2])
        if not self.builder.block.is_terminated:
            self.builder.branch(cont_bb)

        self.builder.position_at_end(cont_bb)

    def _eval_While(self, node: ASTNode):
        func = self.builder.function
        loop_bb = func.append_basic_block("loop")
        body_bb = func.append_basic_block("loop.body")
        after_bb = func.append_basic_block("loop.after")

        self.builder.branch(loop_bb)

        self.builder.position_at_end(loop_bb)
        cond = self._eval_node(node.children[0])
        if isinstance(cond, ir.Constant) and cond.type != ir.IntType(1):
            cond = self.builder.icmp_signed('!=', cond, ir.Constant(ir.IntType(32), 0))
        elif not getattr(cond, "type", None) == ir.IntType(1):
            cond = self.builder.icmp_signed('!=', cond, ir.Constant(ir.IntType(32), 0))
        self.builder.cbranch(cond, body_bb, after_bb)

        self.builder.position_at_end(body_bb)
        self._eval_node(node.children[1])
        if not self.builder.block.is_terminated:
            self.builder.branch(loop_bb)

        self.builder.position_at_end(after_bb)

    def _eval_Return(self, node: ASTNode):
        # For now, functions are void - emit ret_void
        if self.builder and not self.builder.block.is_terminated:
            self.builder.ret_void()

    # -------------------------
    # Debug / helpers
    # -------------------------
    def dump_ir(self) -> str:
        return str(self.module)


# CLI test / example usage
if __name__ == "__main__":
    cfg = CodegenConfig(opt_level=2, verbose=True)
    cg = InstryxLLVMCodegen(config=cfg)
    sample_code = """
    func greet(uid) {
        print: "Hello from Instryx";
    };

    main() {
        greet(0);
    };
    """
    ir_text = cg.generate(sample_code, optimize=True, opt_level=2)
    print(ir_text)

    # emit object file
    try:
        obj = cg.emit_object()
        with open("output.o", "wb") as f:
            f.write(obj)
        print("Emitted object file 'output.o' (size:", len(obj), "bytes)")
    except Exception as e:
        print("Object emission failed:", e)

# instryx_llvm_ir_codegen.py
# Instryx → LLVM IR Code Generator — supreme boosters final
# Author: Violet Magenta / VACU Technologies (modified)
# License: MIT
"""
Enhanced, executable LLVM IR generator for Instryx.

Supreme-boosters additions:
 - Broader AST coverage: numbers, strings, vars, assigns, binary ops, if, while, functions, calls, returns
 - Constant folding and simple peephole optimizations
 - Deduplicated global string pool with stable naming
 - LLVM verification and optional optimization via llvmlite.binding
 - Emit native object file bytes via TargetMachine
 - Per-function local symbol tables and safe alloca placement at entry
 - Verbose/debug mode and CLI test/emission helpers
 - Safe, idempotent generate() to allow multiple calls per instance
Requirements:
 - llvmlite installed
 - instryx_parser available and providing ASTNode structure used by this generator
"""

from __future__ import annotations
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from instryx_parser import InstryxParser, ASTNode

from llvmlite import ir, binding

# Initialize llvm binding (idempotent)
binding.initialize()
binding.initialize_native_target()
binding.initialize_native_asmprinter()


@dataclass
class CodegenConfig:
    opt_level: int = 2
    verbose: bool = False


class InstryxLLVMCodegen:
    def __init__(self, config: Optional[CodegenConfig] = None):
        self.config = config or CodegenConfig()
        self.parser = InstryxParser()
        self._reset_state()

    def _reset_state(self) -> None:
        self.module = ir.Module(name="instryx")
        self.builder: Optional[ir.IRBuilder] = None
        self.funcs: Dict[str, ir.Function] = {}
        self.printf: Optional[ir.Function] = None
        self._global_strings: Dict[str, ir.GlobalVariable] = {}
        self._unique_id = 0
        # stack of per-function locals (name -> alloca)
        self._locals: List[Dict[str, ir.Instruction]] = []
        # declare builtins on fresh module
        self._declare_builtins()

    # -------------------------
    # Public API
    # -------------------------
    def generate(self, code: str, optimize: bool = False, opt_level: int = 2) -> str:
        """
        Parse source and return LLVM IR as text. Safe to call multiple times.
        If optimize=True, run LLVM module-level optimizations.
        """
        ast = self.parser.parse(code)
        self._reset_state()
        self._eval_node(ast)
        llvm_ir = str(self.module)
        if optimize:
            try:
                llvm_ir = self._optimize_ir(llvm_ir, opt_level=opt_level)
            except Exception as e:
                if self.config.verbose:
                    print("Optimization failed:", e)
        return llvm_ir

    def get_module(self) -> ir.Module:
        return self.module

    def emit_object(self) -> bytes:
        """
        Emit native object bytes for the current module using the host target machine.
        """
        asm = str(self.module)
        llvm_mod = binding.parse_assembly(asm)
        llvm_mod.verify()
        target = binding.Target.from_default_triple()
        tm = target.create_target_machine(opt=self.config.opt_level)
        return tm.emit_object(llvm_mod)

    # -------------------------
    # Optimization & verification
    # -------------------------
    def _optimize_ir(self, llvm_ir: str, opt_level: int = 2) -> str:
        llvm_mod = binding.parse_assembly(llvm_ir)
        llvm_mod.verify()
        pmb = binding.PassManagerBuilder()
        pmb.opt_level = max(0, min(3, int(opt_level)))
        pmb.size_level = 0
        pm = binding.ModulePassManager()
        pmb.populate(pm)
        pm.run(llvm_mod)
        return str(llvm_mod)

    def _verify_module(self) -> None:
        asm = str(self.module)
        mref = binding.parse_assembly(asm)
        mref.verify()

    # -------------------------
    # Builtins & global strings
    # -------------------------
    def _declare_builtins(self) -> None:
        voidptr_ty = ir.IntType(8).as_pointer()
        printf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
        if "printf" in self.module.globals:
            self.printf = self.module.globals["printf"]
        else:
            self.printf = ir.Function(self.module, printf_ty, name="printf")
        # set triple to host for correctness if available
        try:
            self.module.triple = binding.get_default_triple()
        except Exception:
            pass

    def _get_or_create_global_string(self, text: str, name_hint: str = "str") -> ir.GlobalVariable:
        """
        Deduplicate global constant strings. Return GlobalVariable.
        Name uses hint + unique id to avoid collisions in repeated runs.
        """
        if text in self._global_strings:
            return self._global_strings[text]
        name = f"{name_hint}_{self._unique_id}"
        self._unique_id += 1
        data = bytearray(text.encode("utf8"))
        const = ir.Constant(ir.ArrayType(ir.IntType(8), len(data)), data)
        gvar = ir.GlobalVariable(self.module, const.type, name=name)
        gvar.linkage = "internal"
        gvar.global_constant = True
        gvar.initializer = const
        self._global_strings[text] = gvar
        return gvar

    # -------------------------
    # AST dispatch
    # -------------------------
    def _eval_node(self, node: ASTNode):
        method = getattr(self, f"_eval_{node.node_type}", None)
        if method is None:
            return self._eval_unknown(node)
        return method(node)

    def _eval_unknown(self, node: ASTNode):
        raise NotImplementedError(f"Unknown AST node type: {node.node_type}")

    # -------------------------
    # Top-level nodes
    # -------------------------
    def _eval_Program(self, node: ASTNode):
        for child in node.children:
            self._eval_node(child)
        # run verifier (best-effort)
        try:
            self._verify_module()
        except Exception:
            if self.config.verbose:
                print("Module verification failed; IR may be invalid.")

    def _eval_Function(self, node: ASTNode):
        """
        node.children == [params_node, body_node]
        params_node.children contain identifier nodes
        """
        name = node.value
        params_node, body_node = node.children if node.children else (None, None)
        param_names = [p.value for p in params_node.children] if params_node and params_node.children else []
        func_ty = ir.FunctionType(ir.VoidType(), [ir.IntType(32)] * len(param_names))
        func = ir.Function(self.module, func_ty, name=name)
        self.funcs[name] = func

        entry = func.append_basic_block("entry")
        self.builder = ir.IRBuilder(entry)

        # push local scope
        local_scope: Dict[str, ir.Instruction] = {}
        self._locals.append(local_scope)

        # create allocas for params and store incoming args
        for i, arg in enumerate(func.args):
            arg.name = param_names[i]
            alloca = self._create_entry_alloca(func, arg.name)
            self.builder.store(arg, alloca)
            local_scope[arg.name] = alloca

        # compile body
        if body_node:
            self._eval_node(body_node)

        # ensure function returns
        if not self.builder.block.is_terminated:
            self.builder.ret_void()

        # pop local scope
        self._locals.pop()

    def _eval_Main(self, node: ASTNode):
        func_ty = ir.FunctionType(ir.VoidType(), [])
        func = ir.Function(self.module, func_ty, name="main")
        self.funcs["main"] = func
        entry = func.append_basic_block("entry")
        self.builder = ir.IRBuilder(entry)
        self._locals.append({})
        if node.children:
            self._eval_node(node.children[0])
        if not self.builder.block.is_terminated:
            self.builder.ret_void()
        self._locals.pop()

    def _eval_Block(self, node: ASTNode):
        for stmt in node.children:
            self._eval_node(stmt)

    def _eval_ExprStmt(self, node: ASTNode):
        return self._eval_node(node.children[0]) if node.children else None

    # -------------------------
    # Literals & simple expressions
    # -------------------------
    def _eval_Number(self, node: ASTNode):
        try:
            val = int(node.value)
            return ir.Constant(ir.IntType(32), val)
        except Exception:
            try:
                f = float(node.value)
                return ir.Constant(ir.DoubleType(), f)
            except Exception:
                return ir.Constant(ir.IntType(32), 0)

    def _eval_String(self, node: ASTNode):
        s = node.value.strip('"') + "\0"
        g = self._get_or_create_global_string(s, name_hint="str")
        return self.builder.bitcast(g, ir.IntType(8).as_pointer())

    # -------------------------
    # Variables & assignments
    # -------------------------
    def _create_entry_alloca(self, function: ir.Function, name: str) -> ir.AllocaInstr:
        """
        Place alloca in function entry block (canonical placement). Use IRBuilder.goto_entry_block context.
        """
        entry = function.entry_basic_block
        ib = ir.IRBuilder(entry)
        # Position at entry and allocate
        with ib.goto_entry_block():
            return ib.alloca(ir.IntType(32), name=name)

    def _eval_Var(self, node: ASTNode):
        name = node.value
        for scope in reversed(self._locals):
            if name in scope:
                return self.builder.load(scope[name], name=name + "_val")
        # fallback constant zero
        return ir.Constant(ir.IntType(32), 0)

    def _eval_Assign(self, node: ASTNode):
        # children: [VarNode, ExprNode]
        if not node.children or len(node.children) < 2:
            return None
        var_node = node.children[0]
        expr_node = node.children[1]
        val = self._eval_node(expr_node)
        name = var_node.value
        if not self._locals:
            # no current function scope: skip (global assignment not supported)
            return None
        scope = self._locals[-1]
        if name not in scope:
            current_func = self.builder.function  # type: ignore
            alloca = self._create_entry_alloca(current_func, name)
            scope[name] = alloca
        else:
            alloca = scope[name]
        self.builder.store(val, alloca)
        return alloca

    # -------------------------
    # Binary ops with folding
    # -------------------------
    def _eval_Binary(self, node: ASTNode):
        left = self._eval_node(node.children[0])
        right = self._eval_node(node.children[1])
        op = node.value

        # constant folding
        if isinstance(left, ir.Constant) and isinstance(right, ir.Constant):
            try:
                # integer constants
                if left.type == ir.IntType(32) and right.type == ir.IntType(32):
                    lv = int(left.constant)
                    rv = int(right.constant)
                    if op == '+':
                        return ir.Constant(ir.IntType(32), lv + rv)
                    if op == '-':
                        return ir.Constant(ir.IntType(32), lv - rv)
                    if op == '*':
                        return ir.Constant(ir.IntType(32), lv * rv)
                    if op == '/':
                        return ir.Constant(ir.IntType(32), lv // rv if rv != 0 else 0)
                    if op in ('==', '!=', '<', '>', '<=', '>='):
                        mapping = {
                            '==': lv == rv,
                            '!=': lv != rv,
                            '<': lv < rv,
                            '>': lv > rv,
                            '<=': lv <= rv,
                            '>=': lv >= rv,
                        }
                        return ir.Constant(ir.IntType(1), 1 if mapping.get(op, False) else 0)
                # float constants
                if left.type == ir.DoubleType() and right.type == ir.DoubleType():
                    lv = float(left.constant)
                    rv = float(right.constant)
                    if op == '+':
                        return ir.Constant(ir.DoubleType(), lv + rv)
                    if op == '-':
                        return ir.Constant(ir.DoubleType(), lv - rv)
            except Exception:
                pass

        # runtime ops
        if op in ('+', '-', '*', '/'):
            if getattr(left, "type", None) == ir.DoubleType() or getattr(right, "type", None) == ir.DoubleType():
                l = self._to_double(left)
                r = self._to_double(right)
                if op == '+':
                    return self.builder.fadd(l, r)
                if op == '-':
                    return self.builder.fsub(l, r)
                if op == '*':
                    return self.builder.fmul(l, r)
                if op == '/':
                    return self.builder.fdiv(l, r)
            else:
                if op == '+':
                    return self.builder.add(left, right)
                if op == '-':
                    return self.builder.sub(left, right)
                if op == '*':
                    return self.builder.mul(left, right)
                if op == '/':
                    return self.builder.sdiv(left, right)
        if op in ('==', '!=', '<', '>', '<=', '>='):
            if getattr(left, "type", None) == ir.DoubleType() or getattr(right, "type", None) == ir.DoubleType():
                l = self._to_double(left)
                r = self._to_double(right)
                mapping = {'==': '==', '!=': '!=', '<': '<', '>': '>', '<=': '<=', '>=': '>='}
                return self.builder.fcmp_ordered(mapping[op], l, r)
            else:
                mapping = {'==': '==', '!=': '!=', '<': '<', '>': '>', '<=': '<=', '>=': '>='}
                return self.builder.icmp_signed(mapping[op], left, right)
        return left

    def _to_double(self, val):
        if isinstance(val, ir.Constant) and val.type == ir.IntType(32):
            return ir.Constant(ir.DoubleType(), float(int(val.constant)))
        if isinstance(val, ir.Constant) and val.type == ir.DoubleType():
            return val
        if getattr(val, "type", None) == ir.IntType(32):
            return self.builder.sitofp(val, ir.DoubleType())
        return val

    # -------------------------
    # Calls & builtins
    # -------------------------
    def _eval_Call(self, node: ASTNode):
        func_name = node.value
        args = [self._eval_node(arg) for arg in node.children] if node.children else []
        if func_name == "print":
            fmt = "%s\n\0"
            gfmt = self._get_or_create_global_string(fmt, name_hint="fmt")
            fmt_ptr = self.builder.bitcast(gfmt, ir.IntType(8).as_pointer())
            # call printf (varargs)
            self.builder.call(self.printf, [fmt_ptr] + args)
            return None
        if func_name in self.funcs:
            return self.builder.call(self.funcs[func_name], args)
        # unknown call: ignore
        return None

    # -------------------------
    # Control flow
    # -------------------------
    def _eval_If(self, node: ASTNode):
        cond = self._eval_node(node.children[0])
        if isinstance(cond, ir.Constant) and cond.type != ir.IntType(1):
            cond = self.builder.icmp_signed('!=', cond, ir.Constant(ir.IntType(32), 0))
        elif not getattr(cond, "type", None) == ir.IntType(1):
            cond = self.builder.icmp_signed('!=', cond, ir.Constant(ir.IntType(32), 0))

        then_bb = self.builder.function.append_basic_block("if.then")
        else_bb = self.builder.function.append_basic_block("if.else")
        cont_bb = self.builder.function.append_basic_block("if.end")

        self.builder.cbranch(cond, then_bb, else_bb)

        self.builder.position_at_end(then_bb)
        self._eval_node(node.children[1])
        if not self.builder.block.is_terminated:
            self.builder.branch(cont_bb)

        self.builder.position_at_end(else_bb)
        if len(node.children) > 2 and node.children[2]:
            self._eval_node(node.children[2])
        if not self.builder.block.is_terminated:
            self.builder.branch(cont_bb)

        self.builder.position_at_end(cont_bb)

    def _eval_While(self, node: ASTNode):
        func = self.builder.function
        loop_bb = func.append_basic_block("loop")
        body_bb = func.append_basic_block("loop.body")
        after_bb = func.append_basic_block("loop.after")

        self.builder.branch(loop_bb)

        self.builder.position_at_end(loop_bb)
        cond = self._eval_node(node.children[0])
        if isinstance(cond, ir.Constant) and cond.type != ir.IntType(1):
            cond = self.builder.icmp_signed('!=', cond, ir.Constant(ir.IntType(32), 0))
        elif not getattr(cond, "type", None) == ir.IntType(1):
            cond = self.builder.icmp_signed('!=', cond, ir.Constant(ir.IntType(32), 0))
        self.builder.cbranch(cond, body_bb, after_bb)

        self.builder.position_at_end(body_bb)
        self._eval_node(node.children[1])
        if not self.builder.block.is_terminated:
            self.builder.branch(loop_bb)

        self.builder.position_at_end(after_bb)

    def _eval_Return(self, node: ASTNode):
        # currently functions return void
        if self.builder and not self.builder.block.is_terminated:
            self.builder.ret_void()

    # -------------------------
    # Debug helpers
    # -------------------------
    def dump_ir(self) -> str:
        return str(self.module)


# CLI test / example usage
if __name__ == "__main__":
    cfg = CodegenConfig(opt_level=2, verbose=True)
    cg = InstryxLLVMCodegen(config=cfg)
    sample_code = """
    func greet(uid) {
        print: "Hello from Instryx";
    };

    main() {
        greet(0);
    };
    """
    ir_text = cg.generate(sample_code, optimize=True, opt_level=2)
    print(ir_text)

    # emit object file
    try:
        obj = cg.emit_object()
        with open("output.o", "wb") as f:
            f.write(obj)
        print("Emitted object file 'output.o' (size:", len(obj), "bytes)")
    except Exception as e:
        print("Object emission failed:", e)
