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

    print("\nFinal Variables State:", interpreter.ctx.variables)
    # instryx_ast_interpreter.py
    print("Execution Log:", interpreter.ctx.output)
    # instryx_ast_interpreter.py
    print("Functions Defined:", list(interpreter.ctx.functions.keys()))
    # instryx_ast_interpreter.py
    print("Output Log:", interpreter.ctx.output)
    print("Execution Log:", interpreter.ctx.output)
    print("Functions Defined:", list(interpreter.ctx.functions.keys()))
    print("Final Variables State:", interpreter.ctx.variables)
    print("Execution Log:", interpreter.ctx.output)
    print("Functions Defined:", list(interpreter.ctx.functions.keys()))
    print("Final Variables State:", interpreter.ctx.variables)
    print("Execution Log:", interpreter.ctx.output)
    print("Functions Defined:", list(interpreter.ctx.functions.keys()))
    print("Final Variables State:", interpreter.ctx.variables)
        
# instryx_ast_interpreter.py
# Production-ready AST Interpreter for the Instryx Language
# Author: Violet Magenta / VACU Technologies (extended)
# License: MIT

from __future__ import annotations
import copy
import time
import logging
from typing import Any, Dict, List, Optional

from instryx_parser import InstryxParser, ASTNode

logger = logging.getLogger("instryx.ast_interpreter")
logger.addHandler(logging.NullHandler())


# --- internal control signals ---
class _ReturnSignal(Exception):
    def __init__(self, value: Any):
        self.value = value


class _BreakSignal(Exception):
    pass


class _ContinueSignal(Exception):
    pass


# --- runtime frame (lexical scope) ---
class Frame:
    def __init__(self, parent: Optional["Frame"] = None):
        self.vars: Dict[str, Any] = {}
        self.parent = parent

    def get(self, name: str) -> Any:
        if name in self.vars:
            return self.vars[name]
        if self.parent:
            return self.parent.get(name)
        raise NameError(f"Undefined identifier '{name}'")

    def set(self, name: str, value: Any) -> None:
        self.vars[name] = value

    def snapshot(self):
        """Return deep snapshot of frame chain (shallow per-frame copy)."""
        chain = []
        f = self
        while f:
            chain.append(copy.deepcopy(f.vars))
            f = f.parent
        return chain

    def restore(self, snapshot):
        """Restore frame chain from snapshot list (closest-first)."""
        f = self
        i = 0
        for snap in snapshot:
            if not f:
                break
            f.vars = copy.deepcopy(snap)
            f = f.parent
            i += 1


# --- runtime context ---
class RuntimeContext:
    def __init__(self):
        self.global_frame = Frame(parent=None)
        self.current_frame = self.global_frame
        self.functions: Dict[str, ASTNode] = {}
        self.output: List[str] = []
        self.builtins: Dict[str, Any] = {}
        self.counters: Dict[str, int] = {}  # simple profiler (node counts)

    def push_frame(self, frame: Frame):
        frame.parent = self.current_frame
        self.current_frame = frame

    def pop_frame(self):
        if self.current_frame.parent:
            self.current_frame = self.current_frame.parent
        else:
            self.current_frame = self.global_frame

    def log(self, message: str):
        self.output.append(message)
        print(message)

    def inc(self, key: str):
        self.counters[key] = self.counters.get(key, 0) + 1


# --- Interpreter ---
class InstryxInterpreter:
    def __init__(self):
        self.parser = InstryxParser()
        self.ctx = RuntimeContext()
        # dispatch cache for faster method lookup
        self._dispatch_cache: Dict[str, Any] = {}
        # register default builtins
        self.register_builtin("print", self._builtin_print)
        self.register_builtin("sleep_ms", self._builtin_sleep_ms)
        self.register_builtin("fail", self._builtin_fail)
        # try to auto-wire allocator / async runtime (best-effort)
        self._auto_wire_subsystems()

    # --- plumbing ---
    def register_builtin(self, name: str, fn):
        if name in self.ctx.builtins:
            logger.debug("Overwriting builtin %s", name)
        self.ctx.builtins[name] = fn

    def register_extern(self, name: str, fn):
        # alias to same registry for now
        self.register_builtin(name, fn)

    def interpret(self, code: str):
        ast = self.parser.parse(code)
        return self.eval_node(ast)

    def eval_node(self, node: ASTNode):
        if node is None:
            return None
        self.ctx.inc(node.node_type)
        # dispatch cache lookup
        method = self._dispatch_cache.get(node.node_type)
        if method is None:
            method = getattr(self, f"eval_{node.node_type}", None)
            if method is None:
                method = self.eval_unknown
            self._dispatch_cache[node.node_type] = method
        return method(node)

    def eval_unknown(self, node: ASTNode):
        raise Exception(f"Unknown AST node type: {node.node_type}")

    # --- program / blocks ---
    def eval_Program(self, node: ASTNode):
        result = None
        for child in node.children:
            result = self.eval_node(child)
        return result

    def eval_Block(self, node: ASTNode):
        result = None
        for stmt in node.children:
            try:
                result = self.eval_node(stmt)
            except _ReturnSignal as r:
                # bubble return up the call chain
                raise r
            except _BreakSignal:
                raise
            except _ContinueSignal:
                raise
        return result

    def eval_Main(self, node: ASTNode):
        self.ctx.log("▶ Executing main()")
        # main body may be a block or function call node
        return self.eval_node(node.children[0])

    # --- functions ---
    def eval_Function(self, node: ASTNode):
        name = node.value
        self.ctx.functions[name] = node
        self.ctx.log(f"🛠 Registered function: {name}")
        return None

    def _call_user_function(self, func_node: ASTNode, args: List[Any]):
        # param nodes commonly stored as first child; body as second child
        params_node = func_node.children[0] if func_node.children else None
        body_node = func_node.children[1] if len(func_node.children) > 1 else None
        param_nodes = params_node.children if params_node else []
        # push new frame
        new_frame = Frame(parent=self.ctx.current_frame)
        # bind params
        for i, param in enumerate(param_nodes):
            pname = param.value
            new_frame.set(pname, args[i] if i < len(args) else None)
        # push and execute
        self.ctx.push_frame(new_frame)
        try:
            try:
                self.eval_node(body_node)
            except _ReturnSignal as r:
                return r.value
        finally:
            self.ctx.pop_frame()
        return None

    # --- statements / expressions ---
    def eval_ExprStmt(self, node: ASTNode):
        return self.eval_node(node.children[0]) if node.children else None

    def eval_Assign(self, node: ASTNode):
        var_name = node.value
        value = self.eval_node(node.children[0]) if node.children else None
        self.ctx.current_frame.set(var_name, value)
        self.ctx.log(f"🔧 Assigned {var_name} = {value}")
        return value

    def eval_ID(self, node: ASTNode):
        return self.ctx.current_frame.get(node.value)

    def eval_Number(self, node: ASTNode):
        try:
            return float(node.value) if "." in node.value else int(node.value)
        except Exception:
            return int(node.value)

    def eval_String(self, node: ASTNode):
        return node.value.strip('"').strip("'")

    # binary ops
    def eval_BinaryOp(self, node: ASTNode):
        # assumed structure: value holds operator, children are left/right
        op = node.value
        left = self.eval_node(node.children[0])
        # short-circuit for and/or
        if op == "and":
            if not left:
                return left
            right = self.eval_node(node.children[1])
            return right
        if op == "or":
            if left:
                return left
            right = self.eval_node(node.children[1])
            return right
        right = self.eval_node(node.children[1])
        try:
            if op == "+":
                return left + right
            if op == "-":
                return left - right
            if op == "*":
                return left * right
            if op == "/":
                return left / right
            if op == "==":
                return left == right
            if op == "!=":
                return left != right
            if op == ">":
                return left > right
            if op == "<":
                return left < right
            if op == ">=":
                return left >= right
            if op == "<=":
                return left <= right
        except Exception as e:
            raise RuntimeError(f"Binary operation {op} failed: {e}")
        raise RuntimeError(f"Unsupported operator: {op}")

    # control flow
    def eval_If(self, node: ASTNode):
        test = self.eval_node(node.children[0])
        if test:
            return self.eval_node(node.children[1])
        elif len(node.children) > 2 and node.children[2]:
            return self.eval_node(node.children[2])
        return None

    def eval_While(self, node: ASTNode):
        test_node = node.children[0]
        body_node = node.children[1] if len(node.children) > 1 else None
        while self.eval_node(test_node):
            try:
                self.eval_node(body_node)
            except _ContinueSignal:
                continue
            except _BreakSignal:
                break
        return None

    def eval_For(self, node: ASTNode):
        # simplistic for loop: children [init, test, update, body]
        init = node.children[0] if len(node.children) > 0 else None
        test = node.children[1] if len(node.children) > 1 else None
        update = node.children[2] if len(node.children) > 2 else None
        body = node.children[3] if len(node.children) > 3 else None
        if init:
            self.eval_node(init)
        while (self.eval_node(test) if test else True):
            try:
                self.eval_node(body)
            except _ContinueSignal:
                pass
            except _BreakSignal:
                break
            if update:
                self.eval_node(update)
        return None

    def eval_Return(self, node: ASTNode):
        val = self.eval_node(node.children[0]) if node.children else None
        raise _ReturnSignal(val)

    def eval_Break(self, node: ASTNode):
        raise _BreakSignal()

    def eval_Continue(self, node: ASTNode):
        raise _ContinueSignal()

    # call dispatch with builtins and user functions, with TCO attempt
    def eval_Call(self, node: ASTNode):
        func_name = node.value
        args = [self.eval_node(arg) for arg in node.children]
        # builtins first
        if func_name in self.ctx.builtins:
            try:
                return self.ctx.builtins[func_name](*args)
            except Exception as e:
                raise RuntimeError(f"Builtin {func_name} raised: {e}")
        # user function
        if func_name in self.ctx.functions:
            func_def = self.ctx.functions[func_name]
            # Simple tail-call optimization: if the last statement in func body is a Call to another function,
            # we could implement trampolining. Here we do a single direct call.
            return self._call_user_function(func_def, args)
        # runtime helpers
        if func_name == "print":
            return self._builtin_print(*args)
        raise NameError(f"Undefined function: {func_name}")

    # macros (lightweight)
    def eval_Macro(self, node: ASTNode):
        self.ctx.log(f"⚙️ Macro: {node.value} on {node.children[0].value if node.children else '<unknown>'}")

    # quarantine semantics with frame + global snapshot and optional allocator integration
    def eval_Quarantine(self, node: ASTNode):
        # expected children: try, replace, erase (each may be Block)
        try_block = node.children[0] if len(node.children) > 0 else None
        replace_block = node.children[1] if len(node.children) > 1 else None
        erase_block = node.children[2] if len(node.children) > 2 else None

        # snapshot frame chain and counters
        frame_snap = self.ctx.current_frame.snapshot()
        counters_snap = copy.deepcopy(self.ctx.counters)
        try:
            return self.eval_node(try_block)
        except Exception as e_try:
            self.ctx.log(f"⚠️ Quarantine caught: {e_try}")
            # restore state
            self.ctx.current_frame.restore(frame_snap)
            self.ctx.counters = counters_snap
            # attempt replace
            try:
                if replace_block:
                    return self.eval_node(replace_block)
            except Exception as e_rep:
                self.ctx.log(f"⚠️ Quarantine replace failed: {e_rep}")
                self.ctx.current_frame.restore(frame_snap)
                self.ctx.counters = counters_snap
                if erase_block:
                    try:
                        return self.eval_node(erase_block)
                    except Exception as e_erase:
                        self.ctx.log(f"⚠️ Quarantine erase failed: {e_erase}")
                        return None
                return None

    # --- builtins ---
    def _builtin_print(self, *args):
        msg = " ".join(map(str, args))
        self.ctx.log(f"🖨️  {msg}")
        return None

    def _builtin_sleep_ms(self, ms: float):
        try:
            time.sleep(float(ms) / 1000.0)
        except Exception:
            pass
        return None

    def _builtin_fail(self, message: str = ""):
        raise RuntimeError(f"fail(): {message}")

    # --- auto-wire subsystems (best-effort) ---
    def _auto_wire_subsystems(self):
        try:
            import instryx_heap_gc_allocator as allocator  # type: ignore
            alloc = allocator.HeapGCAllocator()
            # expose a few methods
            self.register_extern("alloc_object", alloc.alloc_object)
            self.register_extern("alloc_array", alloc.alloc_array)
            self.register_extern("collect_heap", alloc.collect)
            logger.debug("Wired HeapGCAllocator to interpreter")
        except Exception:
            logger.debug("HeapGCAllocator not available")

        try:
            import instryx_async_threading_runtime as art  # type: ignore
            runtime = art.get_runtime()
            self.register_extern("spawn", runtime.spawn)
            self.register_extern("run_sync", runtime.run_sync)
            logger.debug("Wired AsyncThreadingRuntime to interpreter")
        except Exception:
            logger.debug("Async runtime not available")

    # --- utilities / debugging ---
    def dump_state(self):
        state = {
            "current_vars": self.ctx.current_frame.vars,
            "functions": list(self.ctx.functions.keys()),
            "counters": dict(self.ctx.counters)
        }
        return state

    def profile_report(self):
        return dict(self.ctx.counters)


# --- quick self-test (kept concise) ---
if __name__ == "__main__":
    interpreter = InstryxInterpreter()
    sample_code = """
    -- Load user data
    @inject db.conn;

    func load_user(uid) {
        quarantine try {
            data = "User42";
            print: "Loaded", data;
            return data;
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
    print("STATE:", interpreter.dump_state())

    print("PROFILE:", interpreter.profile_report())
    print("OUTPUT LOG:", interpreter.ctx.output)
    print("STATE:", interpreter.dump_state())
    print("PROFILE:", interpreter.profile_report())
 
# instryx_ast_interpreter.py
# Production-ready AST Interpreter for the Instryx Language
# Author: Violet Magenta / VACU Technologies (extended)
# License: MIT

from __future__ import annotations
import copy
import time
import os
import logging
from typing import Any, Dict, List, Optional, Callable, Tuple

from instryx_parser import InstryxParser, ASTNode

logger = logging.getLogger("instryx.ast_interpreter")
logger.addHandler(logging.NullHandler())


# --- internal control signals / trampoline ---
class _ReturnSignal(Exception):
    def __init__(self, value: Any):
        self.value = value


class _BreakSignal(Exception):
    pass


class _ContinueSignal(Exception):
    pass


class _TailCall(Exception):
    """Signal used internally to implement tail-call trampolining."""
    def __init__(self, callee: str, args: List[Any]):
        self.callee = callee
        self.args = args


# --- runtime frame (lexical scope) ---
class Frame:
    def __init__(self, parent: Optional["Frame"] = None):
        self.vars: Dict[str, Any] = {}
        self.parent = parent

    def get(self, name: str) -> Any:
        if name in self.vars:
            return self.vars[name]
        if self.parent:
            return self.parent.get(name)
        raise NameError(f"Undefined identifier '{name}'")

    def set(self, name: str, value: Any) -> None:
        self.vars[name] = value

    def snapshot(self) -> List[Dict[str, Any]]:
        """Return deep snapshot of frame chain (closest-first)."""
        chain = []
        f = self
        while f:
            chain.append(copy.deepcopy(f.vars))
            f = f.parent
        return chain

    def restore(self, snapshot: List[Dict[str, Any]]) -> None:
        """Restore frame chain from snapshot list (closest-first)."""
        f = self
        i = 0
        for snap in snapshot:
            if not f:
                break
            f.vars = copy.deepcopy(snap)
            f = f.parent
            i += 1


# --- runtime context ---
class RuntimeContext:
    def __init__(self):
        self.global_frame = Frame(parent=None)
        self.current_frame = self.global_frame
        self.functions: Dict[str, ASTNode] = {}
        self.output: List[str] = []
        self.builtins: Dict[str, Callable[..., Any]] = {}
        self.counters: Dict[str, int] = {}  # simple profiler (node counts)
        self.node_time: Dict[str, float] = {}  # cumulative time per node type

    def push_frame(self, frame: Frame):
        frame.parent = self.current_frame
        self.current_frame = frame

    def pop_frame(self):
        if self.current_frame.parent:
            self.current_frame = self.current_frame.parent
        else:
            self.current_frame = self.global_frame

    def log(self, message: str):
        self.output.append(message)
        print(message)

    def inc(self, key: str):
        self.counters[key] = self.counters.get(key, 0) + 1

    def add_time(self, key: str, dt: float):
        self.node_time[key] = self.node_time.get(key, 0.0) + dt


# --- Interpreter ---
class InstryxInterpreter:
    def __init__(self):
        self.parser = InstryxParser()
        self.ctx = RuntimeContext()
        # dispatch cache for faster method lookup
        self._dispatch_cache: Dict[str, Callable[[ASTNode], Any]] = {}
        # hooks for instrumentation
        self.before_eval_hooks: List[Callable[[ASTNode], None]] = []
        self.after_eval_hooks: List[Callable[[ASTNode, Any], None]] = []
        # runtime guards & limits
        self._instr_count = 0
        self._max_instructions = int(os.environ.get("INSTRYX_MAX_INSTRUCTIONS", "1000000"))
        self._max_recursion_depth = int(os.environ.get("INSTRYX_MAX_RECURSION", "1000"))
        # tail-call candidate node (used to implement trampolining)
        self._tail_call_candidate_node: Optional[ASTNode] = None
        # register default builtins
        self.register_builtin("print", self._builtin_print)
        self.register_builtin("sleep_ms", self._builtin_sleep_ms)
        self.register_builtin("fail", self._builtin_fail)
        # try to auto-wire allocator / async runtime (best-effort)
        self._auto_wire_subsystems()

    # --- plumbing & instrumentation ---
    def register_builtin(self, name: str, fn: Callable[..., Any]):
        if name in self.ctx.builtins:
            logger.debug("Overwriting builtin %s", name)
        self.ctx.builtins[name] = fn

    def register_extern(self, name: str, fn: Callable[..., Any]):
        # alias to same registry for now
        self.register_builtin(name, fn)

    def register_before_hook(self, fn: Callable[[ASTNode], None]) -> None:
        self.before_eval_hooks.append(fn)

    def register_after_hook(self, fn: Callable[[ASTNode, Any], None]) -> None:
        self.after_eval_hooks.append(fn)

    def _run_before_hooks(self, node: ASTNode) -> None:
        for h in self.before_eval_hooks:
            try:
                h(node)
            except Exception:
                logger.exception("before hook failed")

    def _run_after_hooks(self, node: ASTNode, result: Any) -> None:
        for h in self.after_eval_hooks:
            try:
                h(node, result)
            except Exception:
                logger.exception("after hook failed")

    def interpret(self, code: str):
        ast = self.parser.parse(code)
        return self.eval_node(ast)

    def eval_node(self, node: Optional[ASTNode]):
        if node is None:
            return None
        # instruction counting guard
        self._instr_count += 1
        if self._instr_count > self._max_instructions:
            raise RuntimeError("Instruction budget exceeded")
        self.ctx.inc(node.node_type)
        # dispatch cache lookup
        method = self._dispatch_cache.get(node.node_type)
        if method is None:
            method = getattr(self, f"eval_{node.node_type}", None)
            if method is None:
                method = self.eval_unknown
            self._dispatch_cache[node.node_type] = method
        # timing
        t0 = time.perf_counter()
        self._run_before_hooks(node)
        try:
            result = method(node)
        finally:
            dt = time.perf_counter() - t0
            self.ctx.add_time(node.node_type, dt)
            self._run_after_hooks(node, locals().get("result", None))
        return result

    def eval_unknown(self, node: ASTNode):
        raise Exception(f"Unknown AST node type: {node.node_type}")

    # --- program / blocks ---
    def eval_Program(self, node: ASTNode):
        result = None
        for child in node.children:
            result = self.eval_node(child)
        return result

    def eval_Block(self, node: ASTNode):
        result = None
        for stmt in node.children:
            try:
                result = self.eval_node(stmt)
            except _ReturnSignal as r:
                # bubble return up the call chain
                raise r
            except _BreakSignal:
                raise
            except _ContinueSignal:
                raise
        return result

    def eval_Main(self, node: ASTNode):
        self.ctx.log("▶ Executing main()")
        return self.eval_node(node.children[0]) if node.children else None

    # --- functions & tail-call trampolining ---
    def eval_Function(self, node: ASTNode):
        name = node.value
        self.ctx.functions[name] = node
        self.ctx.log(f"🛠 Registered function: {name}")
        return None

    def _call_user_function(self, func_node: ASTNode, args: List[Any]):
        """
        Call user function with a simple tail-call trampolining optimization.
        If the function body ends with `Return(Call(...))` we catch that pattern
        and avoid growing the Python stack by looping.
        """
        current_func = func_node
        current_args = args

        # limit recursion depth by loop count and maintain a safety counter
        call_depth = 0
        while True:
            call_depth += 1
            if call_depth > self._max_recursion_depth:
                raise RuntimeError("Max recursion depth exceeded (trampoline)")

            params_node = current_func.children[0] if current_func.children else None
            body_node = current_func.children[1] if len(current_func.children) > 1 else None
            param_nodes = params_node.children if params_node else []

            # detect tail-call candidate: body last statement is Return(Call(...))
            tail_candidate = None
            if body_node and getattr(body_node, "children", None):
                last_stmt = body_node.children[-1]
                if last_stmt.node_type == "Return" and last_stmt.children:
                    ret_child = last_stmt.children[0]
                    if ret_child.node_type == "Call":
                        tail_candidate = ret_child

            # set tail candidate context so eval_Return can raise _TailCall
            self._tail_call_candidate_node = tail_candidate

            # create frame and bind params
            new_frame = Frame(parent=self.ctx.current_frame)
            for i, param in enumerate(param_nodes):
                pname = param.value
                new_frame.set(pname, current_args[i] if i < len(current_args) else None)

            self.ctx.push_frame(new_frame)
            try:
                try:
                    # execute body; a _ReturnSignal will carry a normal return value
                    self.eval_node(body_node)
                    # no return encountered
                    return None
                except _TailCall as tc:
                    # tail-call request: set up next func & args and iterate
                    next_callee = tc.callee
                    next_args = tc.args
                    # find target function node
                    if next_callee in self.ctx.functions:
                        next_func_node = self.ctx.functions[next_callee]
                        current_func = next_func_node
                        current_args = next_args
                        # restore frame and loop to trampolined call
                        continue
                    else:
                        # unresolved: perform normal call resolution (may be extern)
                        # unwind and call normally
                        self.ctx.pop_frame()
                        return self.eval_Call(ASTNode(node_type="Call", value=next_callee, children=[]))  # type: ignore
                except _ReturnSignal as r:
                    return r.value
            finally:
                # always ensure frame popped before next iteration or return
                if self.ctx.current_frame is new_frame:
                    self.ctx.pop_frame()
                # clear tail candidate in-case of nested calls
                self._tail_call_candidate_node = None

    # --- statements / expressions ---
    def eval_ExprStmt(self, node: ASTNode):
        return self.eval_node(node.children[0]) if node.children else None

    def eval_Assign(self, node: ASTNode):
        var_name = node.value
        value = self.eval_node(node.children[0]) if node.children else None
        self.ctx.current_frame.set(var_name, value)
        self.ctx.log(f"🔧 Assigned {var_name} = {value}")
        return value

    def eval_ID(self, node: ASTNode):
        return self.ctx.current_frame.get(node.value)

    def eval_Number(self, node: ASTNode):
        try:
            return float(node.value) if "." in node.value else int(node.value)
        except Exception:
            return int(node.value)

    def eval_String(self, node: ASTNode):
        return node.value.strip('"').strip("'")

    # binary ops
    def eval_BinaryOp(self, node: ASTNode):
        op = node.value
        left = self.eval_node(node.children[0])
        if op == "and":
            if not left:
                return left
            right = self.eval_node(node.children[1])
            return right
        if op == "or":
            if left:
                return left
            right = self.eval_node(node.children[1])
            return right
        right = self.eval_node(node.children[1])
        try:
            if op == "+":
                return left + right
            if op == "-":
                return left - right
            if op == "*":
                return left * right
            if op == "/":
                return left / right
            if op == "==":
                return left == right
            if op == "!=":
                return left != right
            if op == ">":
                return left > right
            if op == "<":
                return left < right
            if op == ">=":
                return left >= right
            if op == "<=":
                return left <= right
        except Exception as e:
            raise RuntimeError(f"Binary operation {op} failed: {e}")
        raise RuntimeError(f"Unsupported operator: {op}")

    # control flow
    def eval_If(self, node: ASTNode):
        test = self.eval_node(node.children[0])
        if test:
            return self.eval_node(node.children[1])
        elif len(node.children) > 2 and node.children[2]:
            return self.eval_node(node.children[2])
        return None

    def eval_While(self, node: ASTNode):
        test_node = node.children[0]
        body_node = node.children[1] if len(node.children) > 1 else None
        while self.eval_node(test_node):
            try:
                self.eval_node(body_node)
            except _ContinueSignal:
                continue
            except _BreakSignal:
                break
        return None

    def eval_For(self, node: ASTNode):
        init = node.children[0] if len(node.children) > 0 else None
        test = node.children[1] if len(node.children) > 1 else None
        update = node.children[2] if len(node.children) > 2 else None
        body = node.children[3] if len(node.children) > 3 else None
        if init:
            self.eval_node(init)
        while (self.eval_node(test) if test else True):
            try:
                self.eval_node(body)
            except _ContinueSignal:
                pass
            except _BreakSignal:
                break
            if update:
                self.eval_node(update)
        return None

    def eval_Return(self, node: ASTNode):
        # Tail-call detection: if return child is the previously marked tail candidate Call node,
        # evaluate its callee + args and raise _TailCall to be handled by caller trampoline.
        if node.children:
            child = node.children[0]
            if child.node_type == "Call" and child is self._tail_call_candidate_node:
                # evaluate call arguments in current environment and emit _TailCall
                callee = child.value
                args = [self.eval_node(a) for a in child.children]
                raise _TailCall(callee, args)
            val = self.eval_node(child)
            raise _ReturnSignal(val)
        else:
            raise _ReturnSignal(None)

    def eval_Break(self, node: ASTNode):
        raise _BreakSignal()

    def eval_Continue(self, node: ASTNode):
        raise _ContinueSignal()

    # call dispatch with builtins and user functions
    def eval_Call(self, node: ASTNode):
        func_name = node.value
        args = [self.eval_node(arg) for arg in node.children]
        # builtins first
        if func_name in self.ctx.builtins:
            try:
                return self.ctx.builtins[func_name](*args)
            except Exception as e:
                raise RuntimeError(f"Builtin {func_name} raised: {e}")
        # user function
        if func_name in self.ctx.functions:
            func_def = self.ctx.functions[func_name]
            return self._call_user_function(func_def, args)
        # runtime helpers (backwards compatibility)
        if func_name == "print":
            return self._builtin_print(*args)
        raise NameError(f"Undefined function: {func_name}")

    # macros (lightweight)
    def eval_Macro(self, node: ASTNode):
        self.ctx.log(f"⚙️ Macro: {node.value} on {node.children[0].value if node.children else '<unknown>'}")

    # quarantine semantics with frame + snapshot and optional allocator integration
    def eval_Quarantine(self, node: ASTNode):
        try_block = node.children[0] if len(node.children) > 0 else None
        replace_block = node.children[1] if len(node.children) > 1 else None
        erase_block = node.children[2] if len(node.children) > 2 else None

        frame_snap = self.ctx.current_frame.snapshot()
        counters_snap = copy.deepcopy(self.ctx.counters)
        try:
            return self.eval_node(try_block)
        except Exception as e_try:
            self.ctx.log(f"⚠️ Quarantine caught: {e_try}")
            # restore state
            self.ctx.current_frame.restore(frame_snap)
            self.ctx.counters = counters_snap
            try:
                if replace_block:
                    return self.eval_node(replace_block)
            except Exception as e_rep:
                self.ctx.log(f"⚠️ Quarantine replace failed: {e_rep}")
                self.ctx.current_frame.restore(frame_snap)
                self.ctx.counters = counters_snap
                if erase_block:
                    try:
                        return self.eval_node(erase_block)
                    except Exception as e_erase:
                        self.ctx.log(f"⚠️ Quarantine erase failed: {e_erase}")
                        return None
                return None

    # --- builtins ---
    def _builtin_print(self, *args):
        msg = " ".join(map(str, args))
        self.ctx.log(f"🖨️  {msg}")
        return None

    def _builtin_sleep_ms(self, ms: float):
        try:
            time.sleep(float(ms) / 1000.0)
        except Exception:
            pass
        return None

    def _builtin_fail(self, message: str = ""):
        raise RuntimeError(f"fail(): {message}")

    # --- auto-wire subsystems (best-effort) ---
    def _auto_wire_subsystems(self):
        try:
            import instryx_heap_gc_allocator as allocator  # type: ignore
            alloc = allocator.HeapGCAllocator()
            self.register_extern("alloc_object", alloc.alloc_object)
            self.register_extern("alloc_array", alloc.alloc_array)
            self.register_extern("collect_heap", alloc.collect)
            logger.debug("Wired HeapGCAllocator to interpreter")
        except Exception:
            logger.debug("HeapGCAllocator not available")

        try:
            import instryx_async_threading_runtime as art  # type: ignore
            runtime = art.get_runtime()
            self.register_extern("spawn", runtime.spawn)
            self.register_extern("run_sync", runtime.run_sync)
            logger.debug("Wired AsyncThreadingRuntime to interpreter")
        except Exception:
            logger.debug("Async runtime not available")

    # --- utilities / debugging / profiling ---
    def dump_state(self) -> Dict[str, Any]:
        state = {
            "current_vars": self.ctx.current_frame.vars,
            "functions": list(self.ctx.functions.keys()),
            "counters": dict(self.ctx.counters),
            "node_time": dict(self.ctx.node_time),
            "instr_count": self._instr_count
        }
        return state

    def profile_report(self) -> Dict[str, Any]:
        return dict(self.ctx.counters)

    def reset_counters(self):
        self.ctx.counters.clear()
        self.ctx.node_time.clear()
        self._instr_count = 0


# --- quick self-test (concise) ---
if __name__ == "__main__":
    interpreter = InstryxInterpreter()
    sample_code = """
    -- Load user data
    @inject db.conn;

    func load_user(uid) {
        quarantine try {
            data = "User42";
            print: "Loaded", data;
            return data;
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
    print("STATE:", interpreter.dump_state())
    print("PROFILE:", interpreter.profile_report())
    print("OUTPUT LOG:", interpreter.ctx.output)
