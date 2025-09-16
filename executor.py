"""
Instryx executor - hardwarebridge.executor

A resilient, minimal executor for the Instryx AST. Designed to:
 - Execute a high-level Instryx AST (dict-based node representation).
 - Provide quarantine semantics (try → replace → erase) with shadowed state.
 - Host builtin functions and allow external/FFI function registration.
 - Keep no external dependencies so it can be used during toolchain stages.

Expectations about AST:
 - Node is a dict with a "type" key (e.g. "Program", "FunctionDef", "Call",
   "Assign", "If", "While", "Quarantine", "Literal", "Identifier", "Block", "Return")
 - Function bodies are lists of statements.
 - This executor is intentionally conservative and safe: runtime errors inside
   quarantine blocks are caught and handled per quarantine semantics, and never
   re-raise to the process.
"""

from __future__ import annotations
import copy
import logging
import threading
import asyncio
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("instryx.executor")
logger.addHandler(logging.NullHandler())


class RuntimeErrorInInstryx(Exception):
    """Wrapped runtime error to differentiate from host errors."""


class Frame:
    def __init__(self, parent: Optional["Frame"] = None):
        self.vars: Dict[str, Any] = {}
        self.parent = parent

    def get(self, name: str) -> Any:
        if name in self.vars:
            return self.vars[name]
        if self.parent:
            return self.parent.get(name)
        raise RuntimeErrorInInstryx(f"Undefined identifier '{name}'")

    def set(self, name: str, value: Any) -> None:
        self.vars[name] = value

    def snapshot(self) -> Dict[str, Any]:
        # Shallow copy is intentional for primitive safety; deep copy when needed.
        return copy.deepcopy(self.vars)

    def restore(self, snap: Dict[str, Any]) -> None:
        self.vars = copy.deepcopy(snap)


class Function:
    def __init__(self, name: str, params: List[str], body: List[Dict]):
        self.name = name
        self.params = params
        self.body = body


class Executor:
    def __init__(self):
        self.global_frame = Frame()
        self.functions: Dict[str, Function] = {}
        self.externs: Dict[str, Callable[..., Any]] = {}
        # register safe builtins
        self.register_builtin("print", self._builtin_print)
        self.register_builtin("fail", self._builtin_fail)
        self.register_builtin("sleep_ms", self._builtin_sleep_ms)
        # allows external runtime to plug into executor (WASM host, FFI, etc.)
        self._stop_requested = False

    # --- registration / hosting API ---
    def register_function(self, name: str, params: List[str], body: List[Dict]) -> None:
        self.functions[name] = Function(name, params, body)

    def register_extern(self, name: str, fn: Callable[..., Any]) -> None:
        self.externs[name] = fn

    def register_builtin(self, name: str, fn: Callable[..., Any]) -> None:
        self.register_extern(name, fn)

    # --- execution entrypoints ---
    def load_ast(self, ast: Dict) -> None:
        """
        Load top-level AST. This will register functions found at root.
        Accepts a Program node or a dict with function blocks.
        """
        node_type = ast.get("type") if isinstance(ast, dict) else None
        if node_type == "Program":
            for stmt in ast.get("body", []):
                if stmt.get("type") == "FunctionDef":
                    self._register_function_node(stmt)
                else:
                    # top-level statement executed at load time
                    self.execute(stmt, self.global_frame)
        else:
            # tolerant: try to scan for function defs at top-level
            for k, v in (ast.items() if isinstance(ast, dict) else []):
                if isinstance(v, dict) and v.get("type") == "FunctionDef":
                    self._register_function_node(v)

    def run_main(self, args: Optional[List[Any]] = None) -> Any:
        """
        Run the `main` function if present. Returns its returned value.
        """
        if "main" not in self.functions:
            raise RuntimeErrorInInstryx("No entrypoint 'main' found.")
        return self._call_function("main", args or [], self.global_frame)

    # --- core execution ---
    def execute(self, node: Any, frame: Frame) -> Any:
        """
        Execute a statement node.
        """
        if node is None:
            return None
        if isinstance(node, list):
            result = None
            for stmt in node:
                result = self.execute(stmt, frame)
                if isinstance(result, _ReturnSignal):
                    return result
            return result

        t = node.get("type")
        if t == "Block":
            return self.execute(node.get("body", []), Frame(parent=frame))
        if t == "Assign":
            value = self.eval_expr(node.get("value"), frame)
            target = node.get("target")
            if not isinstance(target, str):
                raise RuntimeErrorInInstryx("Unsupported assignment target")
            frame.set(target, value)
            return value
        if t == "ExpressionStatement":
            return self.eval_expr(node.get("expression"), frame)
        if t == "If":
            cond = self.eval_expr(node.get("test"), frame)
            if cond:
                return self.execute(node.get("consequent", {}), frame)
            else:
                alt = node.get("alternate")
                if alt:
                    return self.execute(alt, frame)
            return None
        if t == "While":
            res = None
            while self.eval_expr(node.get("test"), frame):
                res = self.execute(node.get("body"), frame)
                if isinstance(res, _ReturnSignal):
                    return res
            return res
        if t == "Return":
            val = self.eval_expr(node.get("value"), frame)
            return _ReturnSignal(val)
        if t == "FunctionDef":
            self._register_function_node(node)
            return None
        if t == "Quarantine":
            return self._execute_quarantine(node, frame)
        # fallback: attempt expression eval for unknown statements
        return self.eval_expr(node, frame)

    def eval_expr(self, node: Any, frame: Frame) -> Any:
        """
        Evaluate expression nodes and return a value.
        """
        if node is None:
            return None
        if isinstance(node, (int, float, bool, str)):
            return node
        if isinstance(node, list):
            return [self.eval_expr(n, frame) for n in node]

        t = node.get("type")
        if t == "Literal":
            return node.get("value")
        if t == "Identifier":
            return frame.get(node.get("name"))
        if t == "BinaryOp":
            left = self.eval_expr(node.get("left"), frame)
            right = self.eval_expr(node.get("right"), frame)
            op = node.get("op")
            return _binary_op(op, left, right)
        if t == "Call":
            callee = node.get("callee")
            # callee could be a name or expression
            if isinstance(callee, dict) and callee.get("type") == "Identifier":
                name = callee.get("name")
            elif isinstance(callee, str):
                name = callee
            else:
                name = callee.get("name") if isinstance(callee, dict) else None

            args = [self.eval_expr(a, frame) for a in node.get("arguments", [])]
            return self._call_function(name, args, frame)
        if t == "Array":
            return [self.eval_expr(x, frame) for x in node.get("elements", [])]
        if t == "Object":
            return {k: self.eval_expr(v, frame) for k, v in node.get("properties", {}).items()}

        # Unknown node types: try direct mapping (for flexible ASTs)
        if "value" in node:
            return node["value"]

        raise RuntimeErrorInInstryx(f"Unknown expression node type: {t}")

    # --- internal helpers ---
    def _register_function_node(self, node: Dict) -> None:
        name = node.get("name")
        params = node.get("params", [])
        body = node.get("body", [])
        self.functions[name] = Function(name, params, body)
        logger.debug("Registered function %s(%s)", name, params)

    def _call_function(self, name: str, args: List[Any], caller_frame: Frame) -> Any:
        """
        Call either a registered Instryx function or an extern/builtin.
        """
        # extern/builtin override
        if name in self.externs:
            try:
                return self.externs[name](*args)
            except Exception as ex:
                # externs are host-level, convert to wrapped runtime error
                logger.exception("Extern function '%s' raised", name)
                raise RuntimeErrorInInstryx(f"Extern '{name}' error: {ex}") from ex

        if name not in self.functions:
            raise RuntimeErrorInInstryx(f"Undefined function '{name}'")

        fn = self.functions[name]
        if len(args) != len(fn.params):
            raise RuntimeErrorInInstryx(f"Function '{name}' expected {len(fn.params)} args, got {len(args)}")
        call_frame = Frame(parent=self.global_frame)
        for p, v in zip(fn.params, args):
            call_frame.set(p, v)

        # Execute function body
        res = self.execute(fn.body, call_frame)
        if isinstance(res, _ReturnSignal):
            return res.value
        return None

    def _execute_quarantine(self, node: Dict, frame: Frame) -> Any:
        """
        Quarantine semantics:
         - Execute the 'try' block in a shadowed environment snapshot.
         - On exception: restore snapshot, execute 'replace'.
         - If 'replace' fails: restore snapshot again and execute 'erase'.
         - No exceptions escape; all handled and logged.
        """
        try_block = node.get("try")
        replace_block = node.get("replace")
        erase_block = node.get("erase")

        # snapshot both local frame and global frame to emulate shadow-heap rollback
        frame_snap = frame.snapshot()
        global_snap = self.global_frame.snapshot()

        try:
            logger.debug("Quarantine: entering try block")
            result = self.execute(try_block, frame)
            if isinstance(result, _ReturnSignal):
                return result
            return result
        except Exception as e_try:
            logger.exception("Quarantine try block failed: %s", e_try)
            # restore state before attempting replace
            frame.restore(frame_snap)
            self.global_frame.restore(global_snap)
            try:
                logger.debug("Quarantine: entering replace block")
                res_replace = self.execute(replace_block, frame)
                if isinstance(res_replace, _ReturnSignal):
                    return res_replace
                return res_replace
            except Exception as e_replace:
                logger.exception("Quarantine replace block failed: %s", e_replace)
                # final attempt: restore and run erase (best-effort)
                frame.restore(frame_snap)
                self.global_frame.restore(global_snap)
                try:
                    logger.debug("Quarantine: entering erase block")
                    res_erase = self.execute(erase_block, frame)
                    if isinstance(res_erase, _ReturnSignal):
                        return res_erase
                    return res_erase
                except Exception as e_erase:
                    # Nothing else to do — log and swallow to guarantee no crash.
                    logger.exception("Quarantine erase block failed: %s", e_erase)
                    return None

    # --- builtins ---
    def _builtin_print(self, *args) -> None:
        # safe print: convert to str and write to stdout
        try:
            out = " ".join(str(a) for a in args)
            print(out)
        except Exception:
            # ensure no exception escapes
            logger.exception("Builtin print failed")

    def _builtin_fail(self, message: str = "") -> None:
        raise RuntimeErrorInInstryx(f"fail(): {message}")

    def _builtin_sleep_ms(self, ms: float) -> None:
        try:
            import time
            time.sleep(ms / 1000.0)
        except Exception:
            logger.exception("sleep_ms failed")

    # --- utility ---
    def request_stop(self) -> None:
        self._stop_requested = True


# --- small helpers / signals ---
class _ReturnSignal:
    def __init__(self, value: Any):
        self.value = value


def _binary_op(op: str, left: Any, right: Any) -> Any:
    try:
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return left / right
        if op == "and":
            return left and right
        if op == "or":
            return left or right
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
        logger.exception("Binary op failed: %s %s %s", left, op, right)
        raise RuntimeErrorInInstryx(f"Binary operation error: {e}") from e
    raise RuntimeErrorInInstryx(f"Unsupported binary operator '{op}'")

"""
Instryx executor — node readers adapted to the Instryx.SyntaxTree shapes.

This executor is tolerant: it supports ASTs shaped like the sample
Instryx structs (Expr/Stmt/FuncDecl/Program) while also accepting
older/alternate shapes (keys like "type", "kind" or "node").

Key mappings implemented:
 - Program -> {'imports': [...], 'declarations': [...]}
 - Function  -> FuncDecl style: {'kind': 'Function', 'name', 'params', 'body', 'returns'}
 - Stmt      -> {'kind': <NodeKind>, 'exprs': [...], 'stmts': [...]}
 - Expr      -> {'kind': <NodeKind>, 'value': ...}
 - BinaryOp  -> value may be (op, left, right) or {'op','left','right'}
 - Identifier-> value is the identifier name
 - Literal   -> value is the literal
 - Call      -> value may be {'callee': Expr-or-name, 'args': [Expr]} or (callee, [args])
 - Assign/Declare -> Stmt.exprs = [leftExpr, rightExpr]

Integration hooks:
 - Attempts to register externs from instryx_heap_gc_allocator and
   instryx_async_threading_runtime if they are importable.
"""

from __future__ import annotations
import copy
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("instryx.executor")
logger.addHandler(logging.NullHandler())


class RuntimeErrorInInstryx(Exception):
    pass


class _ReturnSignal:
    def __init__(self, value: Any):
        self.value = value


class Frame:
    def __init__(self, parent: Optional["Frame"] = None):
        self.vars: Dict[str, Any] = {}
        self.parent = parent

    def get(self, name: str) -> Any:
        if name in self.vars:
            return self.vars[name]
        if self.parent:
            return self.parent.get(name)
        raise RuntimeErrorInInstryx(f"Undefined identifier '{name}'")

    def set(self, name: str, value: Any) -> None:
        self.vars[name] = value

    def snapshot(self) -> Dict[str, Any]:
        return copy.deepcopy(self.vars)

    def restore(self, snap: Dict[str, Any]) -> None:
        self.vars = copy.deepcopy(snap)


class Function:
    def __init__(self, name: str, params: List[str], body: Dict):
        self.name = name
        self.params = params
        self.body = body


class Executor:
    def __init__(self):
        self.global_frame = Frame()
        self.functions: Dict[str, Function] = {}
        self.externs: Dict[str, Callable[..., Any]] = {}
        # register minimal builtins
        self.register_extern("print", self._builtin_print)
        self.register_extern("fail", self._builtin_fail)
        # try to auto-wire memory/async subsystems if available
        self._auto_wire_subsystems()

    # -------------------------
    # registration / hosting API
    # -------------------------
    def register_extern(self, name: str, fn: Callable[..., Any]) -> None:
        self.externs[name] = fn

    def register_function(self, name: str, params: List[str], body: Dict) -> None:
        self.functions[name] = Function(name, params, body)

    # -------------------------
    # AST ingestion
    # -------------------------
    def load_ast(self, ast: Dict) -> None:
        """
        Accepts Program-shaped root or tolerant dict.
        Program: {'imports': [...], 'declarations': [...]}
        """
        if not isinstance(ast, dict):
            raise RuntimeErrorInInstryx("AST root must be a dict")

        # prefer explicit Program shape
        if "imports" in ast or "declarations" in ast:
            for decl in ast.get("declarations", []):
                self._maybe_register_top_level(decl)
        else:
            # tolerant scan: register any top-level function declarations found
            for v in ast.values():
                if isinstance(v, dict) and self._node_kind(v) in ("Function", "FuncDecl", "FunctionDecl"):
                    self._maybe_register_top_level(v)

    def run_main(self, args: Optional[List[Any]] = None) -> Any:
        if "main" not in self.functions:
            raise RuntimeErrorInInstryx("No entrypoint 'main' found")
        return self._call_function("main", args or [])

    # -------------------------
    # execution
    # -------------------------
    def execute_stmt(self, stmt: Dict, frame: Frame) -> Any:
        kind = self._node_kind(stmt)
        # Block: contains stmts
        if kind in ("Block",):
            body = stmt.get("stmts", []) or stmt.get("body", [])
            return self._execute_block(body, Frame(parent=frame))
        if kind in ("If",):
            cond_expr = self._first_expr(stmt)
            cond = self.eval_expr(cond_expr, frame)
            if cond:
                return self.execute_stmt(self._first_stmt(stmt), frame)
            else:
                # else may be second stmt
                alt = (stmt.get("stmts") or [None])[1] if len(stmt.get("stmts", [])) > 1 else None
                if alt:
                    return self.execute_stmt(alt, frame)
                return None
        if kind in ("While",):
            cond_expr = self._first_expr(stmt)
            body_stmt = self._first_stmt(stmt)
            result = None
            while self.eval_expr(cond_expr, frame):
                result = self.execute_stmt(body_stmt, frame)
                if isinstance(result, _ReturnSignal):
                    return result
            return result
        if kind in ("Return",):
            val_expr = self._first_expr(stmt)
            return _ReturnSignal(self.eval_expr(val_expr, frame))
        if kind in ("Break", "Continue"):
            # simple interpreter: raise to caller (not implemented here)
            raise RuntimeErrorInInstryx(f"{kind} not implemented in this executor")
        if kind in ("Assign", "Declare"):
            exprs = stmt.get("exprs", [])
            if len(exprs) < 2:
                raise RuntimeErrorInInstryx("Assign requires left and right expression")
            left = exprs[0]
            right = exprs[1]
            # left expected to be Identifier expr or name
            name = self._identifier_name(left)
            value = self.eval_expr(right, frame)
            frame.set(name, value)
            return value
        if kind in ("ExpressionStatement",):
            ex = self._first_expr(stmt)
            return self.eval_expr(ex, frame)
        if kind in ("Function", "FuncDecl", "FunctionDecl"):
            self._maybe_register_top_level(stmt)
            return None
        # fallback: if stmt contains exprs execute sequentially
        for expr in stmt.get("exprs", []) if isinstance(stmt.get("exprs", []), list) else []:
            res = self.eval_expr(expr, frame)
            if isinstance(res, _ReturnSignal):
                return res
        # finally execute nested stmts
        for s in stmt.get("stmts", []) if isinstance(stmt.get("stmts", []), list) else []:
            res = self.execute_stmt(s, frame)
            if isinstance(res, _ReturnSignal):
                return res
        return None

    def eval_expr(self, expr: Any, frame: Frame) -> Any:
        if expr is None:
            return None
        # literal python primitives
        if isinstance(expr, (int, float, bool, str)):
            return expr
        if isinstance(expr, list):
            return [self.eval_expr(e, frame) for e in expr]
        if not isinstance(expr, dict):
            # unexpected node, return as-is
            return expr

        kind = self._node_kind(expr)
        if kind in ("Literal",):
            return expr.get("value")
        if kind in ("Identifier",):
            name = expr.get("value") if "value" in expr else expr.get("name") or expr.get("identifier")
            if not isinstance(name, str):
                raise RuntimeErrorInInstryx("Identifier value malformed")
            return frame.get(name)
        if kind in ("BinaryOp",):
            val = expr.get("value")
            # support tuple or dict shapes
            if isinstance(val, (list, tuple)) and len(val) == 3:
                op, left_e, right_e = val
            elif isinstance(val, dict):
                op = val.get("op")
                left_e = val.get("left")
                right_e = val.get("right")
            else:
                raise RuntimeErrorInInstryx("BinaryOp value malformed")
            left = self.eval_expr(left_e, frame)
            right = self.eval_expr(right_e, frame)
            return _binary_op(op, left, right)
        if kind in ("UnaryOp",):
            val = expr.get("value")
            op = val.get("op") if isinstance(val, dict) else (val[0] if isinstance(val, (list, tuple)) else None)
            operand = val.get("operand") if isinstance(val, dict) else (val[1] if isinstance(val, (list, tuple)) else None)
            v = self.eval_expr(operand, frame)
            if op == "-":
                return -v
            if op == "not":
                return not v
            raise RuntimeErrorInInstryx(f"Unsupported unary op {op}")
        if kind in ("Call",):
            val = expr.get("value")
            # multiple shapes: dict {'callee', 'args'} or tuple (callee, args)
            if isinstance(val, dict):
                callee = val.get("callee")
                args = val.get("args", []) or val.get("arguments", [])
            elif isinstance(val, (list, tuple)) and len(val) == 2:
                callee, args = val
            else:
                # fallback: expression may embed callee directly in 'callee' key
                callee = expr.get("callee") or expr.get("value")
                args = expr.get("arguments", []) or expr.get("args", [])
            # callee can be identifier expr or raw name
            if isinstance(callee, dict) and self._node_kind(callee) == "Identifier":
                name = callee.get("value")
            elif isinstance(callee, str):
                name = callee
            else:
                # evaluate callee expression (e.g., higher-order)
                evaluated = self.eval_expr(callee, frame)
                if callable(evaluated):
                    evaluated_args = [self.eval_expr(a, frame) for a in args]
                    return evaluated(*evaluated_args)
                raise RuntimeErrorInInstryx("Unsupported callee shape")
            evaluated_args = [self.eval_expr(a, frame) for a in (args or [])]
            return self._call_function(name, evaluated_args, frame)
        if kind in ("Array",):
            elems = expr.get("value") or expr.get("elements") or []
            return [self.eval_expr(e, frame) for e in elems]
        if kind in ("Object",):
            props = expr.get("value") or expr.get("properties") or {}
            return {k: self.eval_expr(v, frame) for k, v in props.items()}
        # fallback: some ASTs store literal in 'value'
        if "value" in expr and not isinstance(expr["value"], (dict, list)):
            return expr["value"]

        raise RuntimeErrorInInstryx(f"Unknown expression kind: {kind}")

    # -------------------------
    # internal helpers
    # -------------------------
    def _maybe_register_top_level(self, node: Dict) -> None:
        kind = self._node_kind(node)
        if kind in ("Function", "FuncDecl", "FunctionDecl"):
            name = node.get("name")
            params = []
            for p in node.get("params", []) or node.get("parameters", []):
                if isinstance(p, dict):
                    params.append(p.get("name"))
                else:
                    params.append(str(p))
            body = node.get("body") or node.get("body_stmt") or {}
            self.register_function(name, params, body)
            logger.debug("Registered function %s(%s)", name, params)

    def _call_function(self, name: str, args: List[Any], caller_frame: Optional[Frame] = None) -> Any:
        # externs/builtins first
        if name in self.externs:
            try:
                return self.externs[name](*args)
            except Exception as e:
                logger.exception("Extern '%s' raised", name)
                raise RuntimeErrorInInstryx(f"Extern '{name}' error: {e}") from e

        if name not in self.functions:
            raise RuntimeErrorInInstryx(f"Undefined function '{name}'")
        fn = self.functions[name]
        if len(args) != len(fn.params):
            # allow missing args as None for leniency
            if len(args) < len(fn.params):
                args = args + [None] * (len(fn.params) - len(args))
            else:
                raise RuntimeErrorInInstryx(f"Function '{name}' expected {len(fn.params)} args, got {len(args)}")
        call_frame = Frame(parent=self.global_frame)
        for p, v in zip(fn.params, args):
            call_frame.set(p, v)
        res = self.execute_stmt(fn.body, call_frame)
        if isinstance(res, _ReturnSignal):
            return res.value
        return None

    def _first_expr(self, stmt: Dict) -> Optional[Dict]:
        exprs = stmt.get("exprs", [])
        return exprs[0] if exprs else None

    def _first_stmt(self, stmt: Dict) -> Optional[Dict]:
        stmts = stmt.get("stmts", [])
        return stmts[0] if stmts else None

    def _identifier_name(self, expr: Any) -> str:
        if isinstance(expr, dict) and self._node_kind(expr) == "Identifier":
            name = expr.get("value")
            if not isinstance(name, str):
                raise RuntimeErrorInInstryx("Identifier malformed")
            return name
        if isinstance(expr, str):
            return expr
        raise RuntimeErrorInInstryx("Unsupported assignment target")

    def _node_kind(self, node: Dict) -> str:
        # tolerant kind detection
        if not isinstance(node, dict):
            return ""
        return node.get("kind") or node.get("type") or node.get("node") or ""

    # -------------------------
    # builtins & integrations
    # -------------------------
    def _builtin_print(self, *args) -> None:
        try:
            print(" ".join(str(a) for a in args))
        except Exception:
            logger.exception("print builtin failed")

    def _builtin_fail(self, msg: str = "") -> None:
        raise RuntimeErrorInInstryx(f"fail(): {msg}")

    def _auto_wire_subsystems(self) -> None:
        """
        Try importing allocator and async runtime and register helpful externs.
        This keeps the executor useful out-of-the-box and allows production integration
        by replacing externs with real modules.
        """
        try:
            import instryx_heap_gc_allocator as allocator  # type: ignore
            # if allocator exposes 'alloc'/'free' functions, register them
            if hasattr(allocator, "alloc"):
                self.register_extern("alloc", getattr(allocator, "alloc"))
            if hasattr(allocator, "free"):
                self.register_extern("free", getattr(allocator, "free"))
            logger.debug("Wired instryx_heap_gc_allocator externs")
        except Exception:
            logger.debug("No instryx_heap_gc_allocator available (skipping)")

        try:
            import instryx_async_threading_runtime as aruntime  # type: ignore
            if hasattr(aruntime, "spawn"):
                self.register_extern("spawn", getattr(aruntime, "spawn"))
            if hasattr(aruntime, "sleep_ms"):
                self.register_extern("sleep_ms", getattr(aruntime, "sleep_ms"))
            logger.debug("Wired instryx_async_threading_runtime externs")
        except Exception:
            logger.debug("No instryx_async_threading_runtime available (skipping)")
# -------------------------
# small helpers
# -------------------------
def _binary_op(op: str, left: Any, right: Any) -> Any:
    try:
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return left / right
        if op == "and":
            return left and right
        if op == "or":
            return left or right
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
            return left < right
    except Exception as e:
        logger.exception("Binary op failed: %s %s %s", left, op, right)
        raise RuntimeErrorInInstryx(f"Binary operation error: {e}") from e
    raise RuntimeErrorInInstryx(f"Unsupported binary operator '{op}'")

"""
Instryx executor — hardwarebridge.executor

Updates:
 - Expanded Dodecagram translator: nested functions, params, many node kinds.
 - Match now supports pattern-binding (capture) and simple object-field patterns.
 - Break / Continue implemented via signals handled by loops.
 - Wires to project's allocator and async runtime with adapter wrappers when available.
 - Tolerant to both legacy ("type") AST shapes and Instryx.SyntaxTree ("kind"/"exprs"/"stmts").

Keep AST shapes conservative; translator produces Instryx.SyntaxTree Program shape:
 - Program: {"imports": [], "declarations": [ ... ]}
 - Function: {"kind":"Function", "name":..., "params":[...], "body": {"kind":"Block","stmts":[...]}}
 - Stmt: {"kind": NodeKind, "exprs": [...], "stmts": [...], ...}
 - Expr: {"kind":"Literal"|"Identifier"|"Call"|"BinaryOp"|..., "value": ...}

Dodecagram translator expectations (representative example included in docstring).
"""

from __future__ import annotations
import copy
import logging
import threading
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("instryx.executor")
logger.addHandler(logging.NullHandler())


class RuntimeErrorInInstryx(Exception):
    pass


class _ReturnSignal(Exception):
    def __init__(self, value: Any):
        super().__init__("return")
        self.value = value


class _BreakSignal(Exception):
    pass


class _ContinueSignal(Exception):
    pass


class Frame:
    def __init__(self, parent: Optional["Frame"] = None):
        self.vars: Dict[str, Any] = {}
        self.parent = parent

    def get(self, name: str) -> Any:
        if name in self.vars:
            return self.vars[name]
        if self.parent:
            return self.parent.get(name)
        raise RuntimeErrorInInstryx(f"Undefined identifier '{name}'")

    def set(self, name: str, value: Any) -> None:
        self.vars[name] = value

    def snapshot(self) -> Dict[str, Any]:
        return copy.deepcopy(self.vars)

    def restore(self, snap: Dict[str, Any]) -> None:
        self.vars = copy.deepcopy(snap)


class Function:
    def __init__(self, name: str, params: List[str], body: Dict):
        self.name = name
        self.params = params
        self.body = body


class Executor:
    def __init__(self):
        self.global_frame = Frame()
        self.functions: Dict[str, Function] = {}
        self.externs: Dict[str, Callable[..., Any]] = {}
        # builtin shims
        self.register_extern("print", self._builtin_print)
        self.register_extern("fail", self._builtin_fail)
        # integration adapters (wired when modules present)
        self._wire_lock = threading.Lock()
        self._auto_wire_subsystems()

    # registration / hosting API
    def register_extern(self, name: str, fn: Callable[..., Any]) -> None:
        self.externs[name] = fn

    def register_function(self, name: str, params: List[str], body: Dict) -> None:
        self.functions[name] = Function(name, params, body)

    # AST ingestion
    def load_ast(self, ast: Dict) -> None:
        if not isinstance(ast, dict):
            raise RuntimeErrorInInstryx("AST root must be a dict")

        # detect dodecagram root (contains '𝕘')
        if any("𝕘" in str(k) for k in ast.keys()):
            ast = self.translate_dodecagram(ast)

        # program-shaped AST
        if "declarations" in ast or "imports" in ast:
            for decl in ast.get("declarations", []):
                # register functions
                if self._node_kind(decl) in ("Function", "FuncDecl", "FunctionDecl"):
                    self._maybe_register_top_level(decl)
            # execute any top-level non-function statements
            for decl in ast.get("declarations", []):
                if self._node_kind(decl) not in ("Function", "FuncDecl", "FunctionDecl"):
                    self.execute_stmt(decl, self.global_frame)
            return

        # legacy: scan for FunctionDef nodes
        for v in ast.values():
            if isinstance(v, dict) and self._node_kind(v) in ("Function", "FunctionDef"):
                self._maybe_register_top_level(v)

    def run_main(self, args: Optional[List[Any]] = None) -> Any:
        if "main" not in self.functions:
            raise RuntimeErrorInInstryx("No entrypoint 'main' found")
        return self._call_function("main", args or [])

    # execution entrypoints
    def execute(self, node: Any, frame: Frame) -> Any:
        if node is None:
            return None
        # legacy minimal dispatch: if 'type' exists and is legacy, map to legacy path
        if isinstance(node, dict) and "type" in node:
            return self._legacy_execute(node, frame)
        return self.execute_stmt(node, frame)

    def execute_stmt(self, stmt: Any, frame: Frame) -> Any:
        """
        Execute Instryx.SyntaxTree shaped Stmt nodes.
        """
        if stmt is None:
            return None
        if isinstance(stmt, list):
            result = None
            for s in stmt:
                result = self.execute_stmt(s, frame)
            return result
        if not isinstance(stmt, dict):
            # treat as expression statement
            return self.eval_expr(stmt, frame)

        kind = self._node_kind(stmt)

        # Block
        if kind == "Block":
            body = stmt.get("stmts", []) or stmt.get("body", [])
            return self._execute_block(body, Frame(parent=frame))

        # If
        if kind == "If":
            cond = self.eval_expr(self._first_expr(stmt), frame)
            if cond:
                return self.execute_stmt(self._first_stmt(stmt), frame)
            else:
                # else branch may be second stmts entry or 'alternate'
                alt = stmt.get("alternate") or ((stmt.get("stmts") or [None])[1] if len(stmt.get("stmts", [])) > 1 else None)
                if alt:
                    return self.execute_stmt(alt, frame)
                return None

        # While
        if kind == "While":
            cond_expr = stmt.get("test") or self._first_expr(stmt)
            body = stmt.get("body") or self._first_stmt(stmt)
            try:
                while self.eval_expr(cond_expr, frame):
                    try:
                        res = self.execute_stmt(body, frame)
                    except _ContinueSignal:
                        continue
                    if isinstance(res, _ReturnSignal):
                        return res
            except _BreakSignal:
                return None
            return None

        # For (C-style)
        if kind == "For":
            init = stmt.get("init")
            test = stmt.get("test")
            update = stmt.get("update")
            body = stmt.get("body") or stmt.get("stmts", [])
            if init:
                self.execute_stmt(init, frame)
            try:
                while True:
                    if test and not self.eval_expr(test, frame):
                        break
                    try:
                        res = self.execute_stmt(body, frame)
                        if isinstance(res, _ReturnSignal):
                            return res
                    except _ContinueSignal:
                        pass
                    if update:
                        self.execute_stmt(update, frame)
            except _BreakSignal:
                return None
            return None

        # ForEach
        if kind in ("ForEach", "Foreach"):
            iter_var = stmt.get("iterVar") or stmt.get("var")
            iterable = stmt.get("iterable") or (self._first_expr(stmt))
            body = stmt.get("body") or stmt.get("stmts", [])
            items = self.eval_expr(iterable, frame)
            if items is None:
                return None
            try:
                for it in items:
                    frame.set(iter_var, it)
                    try:
                        res = self.execute_stmt(body, frame)
                        if isinstance(res, _ReturnSignal):
                            return res
                    except _ContinueSignal:
                        continue
            except _BreakSignal:
                return None
            return None

        # Match / pattern-binding
        if kind == "Match":
            subject = self.eval_expr(stmt.get("expr") or self._first_expr(stmt), frame)
            cases = stmt.get("cases", [])
            default = stmt.get("default")
            for c in cases:
                pat = c.get("pattern")
                body = c.get("body")
                matched, bind_map = self._match_pattern(pat, subject, frame)
                if matched:
                    # execute body in child frame with binds
                    child = Frame(parent=frame)
                    for k, v in bind_map.items():
                        child.set(k, v)
                    return self.execute_stmt(body, child)
            if default:
                return self.execute_stmt(default, frame)
            return None

        # Return
        if kind == "Return":
            val = self.eval_expr(self._first_expr(stmt), frame)
            raise _ReturnSignal(val)

        # Break / Continue
        if kind == "Break":
            raise _BreakSignal()
        if kind == "Continue":
            raise _ContinueSignal()

        # Assign / Declare
        if kind in ("Assign", "Declare"):
            exprs = stmt.get("exprs", [])
            if len(exprs) < 2:
                raise RuntimeErrorInInstryx("Assign requires left and right expression")
            left = exprs[0]
            right = exprs[1]
            name = self._identifier_name(left)
            value = self.eval_expr(right, frame)
            frame.set(name, value)
            return value

        # Quarantine
        if kind == "Quarantine":
            return self._execute_quarantine(stmt, frame)

        # Function declaration
        if kind in ("Function", "FuncDecl", "FunctionDecl"):
            self._maybe_register_top_level(stmt)
            return None

        # Class / Struct / Enum registration
        if kind in ("Class", "Struct"):
            tname = stmt.get("name")
            fields = stmt.get("fields") or {f.get("name"): f.get("default") for f in stmt.get("fields", [])}
            types = self.global_frame.vars.setdefault("_types", {})
            types[tname] = {"kind": kind, "fields": fields}
            return None

        if kind == "Enum":
            tname = stmt.get("name")
            variants = stmt.get("variants") or {v.get("name"): v.get("value") for v in stmt.get("variants", [])}
            types = self.global_frame.vars.setdefault("_types", {})
            types[tname] = {"kind": "Enum", "variants": variants}
            return None

        # ExpressionStatement
        if kind in ("ExpressionStatement", ""):
            ex = self._first_expr(stmt)
            return self.eval_expr(ex, frame)

        # fallbacks: eval exprs then stmts
        for expr in stmt.get("exprs", []) if isinstance(stmt.get("exprs", []), list) else []:
            res = self.eval_expr(expr, frame)
            if isinstance(res, _ReturnSignal):
                return res
        for s in stmt.get("stmts", []) if isinstance(stmt.get("stmts", []), list) else []:
            res = self.execute_stmt(s, frame)
            if isinstance(res, _ReturnSignal):
                return res
        return None

    def _execute_block(self, body: List[Dict], block_frame: Frame) -> Any:
        for s in body:
            res = self.execute_stmt(s, block_frame)
            if isinstance(res, _ReturnSignal):
                return res
        return None

    def eval_expr(self, expr: Any, frame: Frame) -> Any:
        if expr is None:
            return None
        if isinstance(expr, (int, float, bool, str)):
            return expr
        if isinstance(expr, list):
            return [self.eval_expr(e, frame) for e in expr]
        if not isinstance(expr, dict):
            return expr

        kind = self._node_kind(expr)

        if kind == "Literal":
            return expr.get("value")
        if kind == "Identifier":
            name = expr.get("value") or expr.get("name") or expr.get("identifier")
            if not isinstance(name, str):
                raise RuntimeErrorInInstryx("Identifier malformed")
            return frame.get(name)
        if kind == "BinaryOp":
            val = expr.get("value")
            if isinstance(val, (list, tuple)) and len(val) == 3:
                op, l, r = val
            elif isinstance(val, dict):
                op = val.get("op"); l = val.get("left"); r = val.get("right")
            else:
                raise RuntimeErrorInInstryx("BinaryOp malformed")
            left = self.eval_expr(l, frame)
            right = self.eval_expr(r, frame)
            return _binary_op(op, left, right)
        if kind == "UnaryOp":
            val = expr.get("value")
            op = val.get("op") if isinstance(val, dict) else (val[0] if isinstance(val, (list, tuple)) else None)
            operand = val.get("operand") if isinstance(val, dict) else (val[1] if isinstance(val, (list, tuple)) else None)
            v = self.eval_expr(operand, frame)
            if op == "-": return -v
            if op in ("not", "!"): return not v
            raise RuntimeErrorInInstryx(f"Unsupported unary op {op}")
        if kind == "Call":
            val = expr.get("value")
            if isinstance(val, dict):
                callee = val.get("callee"); args = val.get("args", []) or val.get("arguments", [])
            elif isinstance(val, (list, tuple)) and len(val) == 2:
                callee, args = val
            else:
                callee = expr.get("callee") or expr.get("value"); args = expr.get("arguments", []) or expr.get("args", [])
            # resolve callee
            if isinstance(callee, dict) and self._node_kind(callee) == "Identifier":
                name = callee.get("value")
            elif isinstance(callee, str):
                name = callee
            else:
                evaluated = self.eval_expr(callee, frame)
                if callable(evaluated):
                    evaluated_args = [self.eval_expr(a, frame) for a in args]
                    return evaluated(*evaluated_args)
                raise RuntimeErrorInInstryx("Unsupported callee shape")
            evaluated_args = [self.eval_expr(a, frame) for a in (args or [])]
            return self._call_function(name, evaluated_args, frame)
        if kind == "Array":
            elems = expr.get("value") or expr.get("elements") or []
            return [self.eval_expr(e, frame) for e in elems]
        if kind == "Object":
            props = expr.get("value") or expr.get("properties") or {}
            return {k: self.eval_expr(v, frame) for k, v in props.items()}

        # Type instantiation by name (Class/Struct)
        if kind in ("TypeRef",) or (kind == "Identifier" and expr.get("value") in self.global_frame.vars.get("_types", {})):
            tname = expr.get("value")
            types = self.global_frame.vars.get("_types", {})
            t = types.get(tname)
            if t and t["kind"] in ("Class", "Struct"):
                inst = {"__type__": tname}
                for fname, fdef in t["fields"].items():
                    inst[fname] = fdef if not isinstance(fdef, dict) else fdef.get("default")
                return inst
            if t and t["kind"] == "Enum":
                return t["variants"]

        if "value" in expr and not isinstance(expr["value"], (dict, list)):
            return expr["value"]

        raise RuntimeErrorInInstryx(f"Unknown expression kind: {kind}")

    # legacy compatibility
    def _legacy_execute(self, node: Dict, frame: Frame) -> Any:
        t = node.get("type")
        if t == "Block":
            return self._execute_block(node.get("body", []), Frame(parent=frame))
        if t == "Assign":
            value = self.eval_expr(node.get("value"), frame)
            target = node.get("target")
            if not isinstance(target, str):
                raise RuntimeErrorInInstryx("Unsupported assignment target")
            frame.set(target, value)
            return value
        if t == "If":
            cond = self.eval_expr(node.get("test"), frame)
            if cond:
                return self._legacy_execute(node.get("consequent", {}), frame)
            else:
                alt = node.get("alternate")
                if alt:
                    return self._legacy_execute(alt, frame)
            return None
        if t == "While":
            res = None
            while self.eval_expr(node.get("test"), frame):
                res = self._legacy_execute(node.get("body"), frame)
            return res
        if t == "Return":
            val = self.eval_expr(node.get("value"), frame)
            raise _ReturnSignal(val)
        if t == "Quarantine":
            return self._execute_quarantine(node, frame)
        # fallback
        return self.eval_expr(node, frame)

    # quarantine semantics
    def _execute_quarantine(self, node: Dict, frame: Frame) -> Any:
        try_block = node.get("try") or node.get("try_block")
        replace_block = node.get("replace")
        erase_block = node.get("erase")

        frame_snap = frame.snapshot()
        global_snap = self.global_frame.snapshot()

        try:
            return self.execute_stmt(try_block, frame)
        except Exception as e_try:
            logger.exception("Quarantine try failed: %s", e_try)
            frame.restore(frame_snap); self.global_frame.restore(global_snap)
            try:
                return self.execute_stmt(replace_block, frame)
            except Exception as e_rep:
                logger.exception("Quarantine replace failed: %s", e_rep)
                frame.restore(frame_snap); self.global_frame.restore(global_snap)
                try:
                    return self.execute_stmt(erase_block, frame)
                except Exception as e_erase:
                    logger.exception("Quarantine erase failed: %s", e_erase)
                    return None

    # internal helpers
    def _maybe_register_top_level(self, node: Dict) -> None:
        kind = self._node_kind(node)
        if kind in ("Function", "FuncDecl", "FunctionDecl", "FunctionDef"):
            name = node.get("name")
            params = []
            for p in node.get("params", []) or node.get("parameters", []):
                if isinstance(p, dict):
                    params.append(p.get("name"))
                else:
                    params.append(str(p))
            body = node.get("body") or node.get("body_stmt") or node.get("stmts") or []
            self.register_function(name, params, body)
            logger.debug("Registered function %s(%s)", name, params)

    def _call_function(self, name: str, args: List[Any], caller_frame: Optional[Frame] = None) -> Any:
        if name in self.externs:
            try:
                return self.externs[name](*args)
            except Exception as e:
                logger.exception("Extern '%s' raised", name)
                raise RuntimeErrorInInstryx(f"Extern '{name}' error: {e}") from e

        if name not in self.functions:
            # try constructor from types registry
            types = self.global_frame.vars.get("_types", {})
            if name in types:
                t = types[name]
                if t["kind"] in ("Class", "Struct"):
                    inst = {"__type__": name}
                    for fname, fdef in t["fields"].items():
                        inst[fname] = fdef if not isinstance(fdef, dict) else fdef.get("default")
                    return inst
                if t["kind"] == "Enum":
                    return t["variants"]
            raise RuntimeErrorInInstryx(f"Undefined function '{name}'")
        fn = self.functions[name]
        if len(args) != len(fn.params):
            if len(args) < len(fn.params):
                args = args + [None] * (len(fn.params) - len(args))
            else:
                raise RuntimeErrorInInstryx(f"Function '{name}' expected {len(fn.params)} args, got {len(args)}")
        call_frame = Frame(parent=self.global_frame)
        for p, v in zip(fn.params, args):
            call_frame.set(p, v)
        try:
            return self.execute_stmt(fn.body, call_frame)
        except _ReturnSignal as r:
            return r.value

    def _first_expr(self, stmt: Dict) -> Optional[Dict]:
        exprs = stmt.get("exprs", [])
        return exprs[0] if exprs else stmt.get("expr")

    def _first_stmt(self, stmt: Dict) -> Optional[Dict]:
        stmts = stmt.get("stmts", [])
        return stmts[0] if stmts else stmt.get("body")

    def _identifier_name(self, expr: Any) -> str:
        if isinstance(expr, dict) and self._node_kind(expr) == "Identifier":
            name = expr.get("value") or expr.get("name")
            if not isinstance(name, str):
                raise RuntimeErrorInInstryx("Identifier malformed")
            return name
        if isinstance(expr, str):
            return expr
        raise RuntimeErrorInInstryx("Unsupported assignment target")

    def _node_kind(self, node: Any) -> str:
        if not isinstance(node, dict):
            return ""
        return node.get("kind") or node.get("type") or node.get("node") or ""

    # pattern matching helper
    def _match_pattern(self, pat: Any, subject: Any, frame: Frame) -> (bool, Dict[str, Any]):
        """
        Returns (matched:bool, binds:dict).
        Supported patterns:
         - "_" wildcard (string or Literal with value "_")
         - {'bind': 'name'} binds subject to name
         - literal values compare equality
         - Identifier pattern binds if pattern.kind == 'Identifier' and has 'bind' flag (heuristic)
         - object pattern: {'fields': {'f1': pattern, 'f2': pattern, ...}}
        """
        binds: Dict[str, Any] = {}
        # wildcard
        if pat == "_" or (isinstance(pat, dict) and pat.get("value") == "_"):
            return True, {}
        # bind shorthand: {'bind':'name'}
        if isinstance(pat, dict) and "bind" in pat:
            binds[pat["bind"]] = subject
            return True, binds
        # Identifier pattern that is string name preceded by '$' or marked as bind
        if isinstance(pat, dict) and self._node_kind(pat) == "Identifier":
            pname = pat.get("value") or pat.get("name")
            # if identifier value begins with '$' treat as bind
            if isinstance(pname, str) and pname.startswith("$"):
                binds[pname.lstrip("$")] = subject
                return True, binds
            # otherwise compare identifier literal to subject (rare)
            return (pname == subject), {}
        # object pattern
        if isinstance(pat, dict) and "fields" in pat and isinstance(subject, dict):
            for k, subpat in pat["fields"].items():
                if k not in subject:
                    return False, {}
                ok, b = self._match_pattern(subpat, subject[k], frame)
                if not ok:
                    return False, {}
                binds.update(b)
            return True, binds
        # list/array pattern
        if isinstance(pat, list) and isinstance(subject, (list, tuple)):
            if len(pat) != len(subject):
                return False, {}
            for psub, ssub in zip(pat, subject):
                ok, b = self._match_pattern(psub, ssub, frame)
                if not ok:
                    return False, {}
                binds.update(b)
            return True, binds
        # literal compare
        if isinstance(pat, dict) and pat.get("kind") == "Literal":
            return (pat.get("value") == subject), {}
        if not isinstance(pat, dict):
            return (pat == subject), {}
        # fallback fail
        return False, {}

    # builtins
    def _builtin_print(self, *args) -> None:
        try:
            print(" ".join(str(a) for a in args))
        except Exception:
            logger.exception("print builtin failed")

    def _builtin_fail(self, msg: str = "") -> None:
        raise RuntimeErrorInInstryx(f"fail(): {msg}")

    # auto-wire adapters for allocator and async runtime
    def _auto_wire_subsystems(self) -> None:
        with self._wire_lock:
            # allocator adapter
            try:
                import instryx_heap_gc_allocator as allocator  # type: ignore
                alloc_instance = None
                # prefer existing singleton or create one
                if hasattr(allocator, "HeapGCAllocator"):
                    try:
                        alloc_instance = allocator.HeapGCAllocator()
                    except Exception:
                        alloc_instance = None
                # if module provides convenience names, try to use them
                if alloc_instance:
                    self.register_extern("alloc_object", alloc_instance.alloc_object)
                    self.register_extern("alloc_array", alloc_instance.alloc_array)
                    self.register_extern("alloc_bytes", alloc_instance.alloc_bytes)
                    self.register_extern("alloc_large", alloc_instance.alloc_large_object)
                    self.register_extern("box_value", alloc_instance.box_value)
                    self.register_extern("get_field", alloc_instance.get_field)
                    self.register_extern("set_field", alloc_instance.set_field)
                    self.register_extern("get_index", alloc_instance.get_index)
                    self.register_extern("set_index", alloc_instance.set_index)
                    self.register_extern("collect_heap", alloc_instance.collect)
                    self.register_extern("heap_snapshot", alloc_instance.heap_snapshot)
                    self.register_extern("compact_heap", alloc_instance.compact_heap)
                    self.register_extern("register_root", alloc_instance.register_root)
                    self.register_extern("unregister_root", alloc_instance.unregister_root)
                    logger.debug("Wired HeapGCAllocator adapters")
            except Exception:
                logger.debug("HeapGCAllocator not available (skipping)")

            # async runtime adapter
            try:
                import instryx_async_threading_runtime as aruntime  # type: ignore
                # prefer get_runtime function if present
                runtime = None
                if hasattr(aruntime, "get_runtime"):
                    try:
                        runtime = aruntime.get_runtime()
                    except Exception:
                        runtime = None
                elif hasattr(aruntime, "AsyncThreadingRuntime"):
                    try:
                        runtime = aruntime.AsyncThreadingRuntime()
                    except Exception:
                        runtime = None
                if runtime:
                    self.register_extern("spawn", runtime.spawn)
                    self.register_extern("submit", runtime.submit)
                    self.register_extern("schedule_later", runtime.schedule_later)
                    self.register_extern("schedule_repeating", runtime.schedule_repeating)
                    self.register_extern("run_sync", runtime.run_sync)
                    self.register_extern("runtime_metrics", runtime.metrics)
                    logger.debug("Wired AsyncThreadingRuntime adapters")
            except Exception:
                logger.debug("Async runtime not available (skipping)")

    # dodecagram translator (expanded)
    def translate_dodecagram(self, dodec: Dict) -> Dict:
        """
        Representative Dodecagram JSON -> Instryx.SyntaxTree Program translator.

        Representative sample dodecagram (for the translator to cover):
        {
          "𝕘12": {
            "node": "main",
            "branch_𝕓0": { "call": "init", "args": [] },
            "branch_𝕓1": {
               "node": "fetchData",
               "params": ["url"],
               "body": {
                   "branch_𝕓0": { "call": "net.request", "args": ["url"] },
                   "branch_𝕓1": { "call": "process", "args": [{"call":"json.parse","args":["#0"]}] }
               }
            },
            "branch_𝕓2": { "assign": ["x", 42] },
            "branch_𝕓3": { "quarantine": {
                "try": { "call": "risky" },
                "replace": { "call": "retry" },
                "erase": { "call": "abort" }
            } }
          }
        }

        The translator aims to:
         - Convert dodecagram "node" entries into Function declarations.
         - Map 'call', 'func', 'assign', 'literal', 'params', 'body', 'quarantine',
           'if', 'while', 'for', 'match', 'class', 'struct', 'enum' keys to Instryx nodes.
         - Recurse into branch_𝕓* children (and plain numeric keys).
        """
        def conv_node(n) -> Any:
            # primitive
            if not isinstance(n, dict):
                return self._lit_or_ident(n)
            # call
            if "call" in n:
                callee = n["call"]
                args = [conv_node(a) for a in n.get("args", [])]
                return {"kind": "Call", "value": {"callee": callee if isinstance(callee, str) else conv_node(callee), "args": args}}
            # func / function declaration (nested)
            if "func" in n or "function" in n:
                fname = n.get("func") or n.get("function")
                params = n.get("params") or n.get("parameters") or []
                body_raw = n.get("body") or n.get("blocks") or n.get("branches") or {}
                body_stmts = []
                # if body is dict of branches
                if isinstance(body_raw, dict):
                    for bk, bv in body_raw.items():
                        body_stmts.append({"kind": "ExpressionStatement", "exprs": [conv_node(bv)]})
                elif isinstance(body_raw, list):
                    for item in body_raw:
                        body_stmts.append({"kind": "ExpressionStatement", "exprs": [conv_node(item)]})
                return {"kind": "Function", "name": fname, "params": params, "body": {"kind": "Block", "stmts": body_stmts}}
            # assign
            if "assign" in n:
                left, right = n["assign"][0], n["assign"][1]
                return {"kind": "Assign", "exprs": [self._lit_or_ident(left), conv_node(right)]}
            # quarantine
            if "quarantine" in n:
                q = n["quarantine"]
                return {"kind": "Quarantine", "try": conv_node(q.get("try")), "replace": conv_node(q.get("replace")), "erase": conv_node(q.get("erase"))}
            # if/while/for
            if "if" in n:
                cond = conv_node(n["if"].get("cond") or n["if"].get("test"))
                then_b = conv_node(n["if"].get("then") or n["if"].get("body"))
                alt_b = conv_node(n["if"].get("else")) if n["if"].get("else") else None
                return {"kind": "If", "exprs": [cond], "stmts": [then_b, alt_b] if alt_b else [then_b]}
            if "while" in n:
                return {"kind": "While", "test": conv_node(n["while"].get("test") or n["while"].get("cond")), "body": conv_node(n["while"].get("body"))}
            if "for" in n:
                f = n["for"]
                return {"kind": "For", "init": conv_node(f.get("init")) if f.get("init") else None,
                        "test": conv_node(f.get("test")) if f.get("test") else None,
                        "update": conv_node(f.get("update")) if f.get("update") else None,
                        "body": conv_node(f.get("body"))}
            # match
            if "match" in n:
                subject = conv_node(n["match"].get("expr"))
                cases = []
                for c in n["match"].get("cases", []):
                    cases.append({"pattern": c.get("pattern"), "body": conv_node(c.get("body"))})
                return {"kind": "Match", "expr": subject, "cases": cases, "default": conv_node(n["match"].get("default"))}
            # class / struct / enum
            if "class" in n or "struct" in n:
                name = n.get("class") or n.get("struct")
                fields = n.get("fields", [])
                return {"kind": "Class" if "class" in n else "Struct", "name": name, "fields": fields}
            if "enum" in n:
                name = n.get("enum")
                variants = n.get("variants", [])
                return {"kind": "Enum", "name": name, "variants": variants}
            # nested node: treat as expression statement or block
            # if node contains 'node' and branches, convert to function
            if "node" in n:
                fname = n["node"]
                params = n.get("params", [])
                # convert branches to statements
                stmts = []
                for k, v in n.items():
                    if k == "node":
                        continue
                    if isinstance(v, dict):
                        stmts.append({"kind": "ExpressionStatement", "exprs": [conv_node(v)]})
                    else:
                        stmts.append({"kind": "ExpressionStatement", "exprs": [self._lit_or_ident(v)]})
                return {"kind": "Function", "name": fname, "params": params, "body": {"kind": "Block", "stmts": stmts}}
            # unknown dict -> try to convert fields to object literal
            obj = {}
            for k, v in n.items():
                obj[k] = conv_node(v) if isinstance(v, (dict, list)) else v
            return {"kind": "Object", "value": obj}

        # top-level translator: build program with possible multiple functions
        root_key = next((k for k in dodec.keys() if "𝕘" in str(k)), None)
        root = dodec.get(root_key) if root_key else dodec
        prog: Dict = {"imports": [], "declarations": []}
        # if root is a single function-like node
        if isinstance(root, dict) and root.get("node"):
            func_node = conv_node(root)
            if func_node.get("kind") == "Function":
                prog["declarations"].append(func_node)
            else:
                # wrap into main
                prog["declarations"].append({"kind": "Function", "name": "main", "params": [], "body": {"kind": "Block", "stmts": [{"kind": "ExpressionStatement", "exprs": [func_node]}]}})
            return prog

        # otherwise convert each child
        for k, v in dodec.items():
            if isinstance(v, dict):
                conv = conv_node(v)
                if isinstance(conv, dict) and conv.get("kind") == "Function":
                    prog["declarations"].append(conv)
                else:
                    prog["declarations"].append({"kind": "ExpressionStatement", "exprs": [conv]})
        return prog

    def _lit_or_ident(self, v: Any) -> Dict:
        if isinstance(v, str):
            if v.startswith("http://") or v.startswith("https://") or " " in v or v.startswith('"') or v.startswith("'"):
                return {"kind": "Literal", "value": v}
            return {"kind": "Identifier", "value": v}
        if isinstance(v, (int, float, bool)):
            return {"kind": "Literal", "value": v}
        if isinstance(v, dict):
            # try to interpret as dodec branch
            return self.translate_dodecagram({ "tmp": v })["declarations"][0] if v.get("node") else {"kind": "Object", "value": v}
        return {"kind": "Literal", "value": v}


# small helpers
def _binary_op(op: str, left: Any, right: Any) -> Any:
    try:
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return left / right
        if op == "and":
            return left and right
        if op == "or":
            return left or right
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
            return left < right
    except Exception as e:
        logger.exception("Binary op failed: %s %s %s", left, op, right)
        raise RuntimeErrorInInstryx(f"Binary operation error: {e}") from e
    raise RuntimeErrorInInstryx(f"Unsupported binary operator '{op}'")

class _ReturnSignal(Exception):
    def __init__(self, value: Any):
        self.value = value
        super().__init__()
# --- small helpers / signals ---
class _ReturnSignal(Exception):
    """Signal used to return a value from a function body / block."""
    def __init__(self, value: Any):
        super().__init__("return")
        self.value = value


class _BreakSignal(Exception):
    """Internal signal to break out of the nearest loop.

    This is raised by `Break` statements and caught by loop handlers
    (While/For/ForEach) to perform a controlled loop exit without
    propagating a catchable exception up to the host process.
    """
    pass


class _ContinueSignal(Exception):
    """Internal signal to skip to the next loop iteration.

    This is raised by `Continue` statements and caught by loop handlers
    to continue execution at the top of the loop.
    """
    pass


def _binary_op(op: str, left: Any, right: Any) -> Any:
    try:
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return left / right
        if op == "and":
            return left and right
        if op == "or":
            return left or right
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
        logger.exception("Binary op failed: %s %s %s", left, op, right)
        raise RuntimeErrorInInstryx(f"Binary operation error: {e}") from e
    raise RuntimeErrorInInstryx(f"Unsupported binary operator '{op}'")


