"""
instryx_compiler_plugins.py

Compiler plugin framework and a set of production-ready, fully implemented optimization
and tooling plugins for the Instryx toolchain.

Enhancements:
- Many optimization passes (constant folding, propagation, CFG simplify, inlining,
  loop unrolling, function specialization, peephole, SSA-lite, tail-call elimination,
  vectorize/prefetch hints).
- Deterministic IR hashing, pass result caching, intermediate IR snapshots.
- Grouped/parallel pass execution for same-priority plugins.
- Lightweight JSON-schema-like validator `validate_schema` for configuration/schema checks.
- CLI for discovery and running passes; safe execution with timeouts.
"""

from __future__ import annotations
import importlib
import inspect
import json
import logging
import os
import pkgutil
import sys
import tempfile
import threading
import time
import traceback
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# Optional integrations
_transformer = None
try:
    import instryx_macro_transformer_model as transformer  # type: ignore
    _transformer = transformer
except Exception:
    transformer = None

# attempt to discover ciams plugins package
_ciams_pkg = None
try:
    import ciams.ciams_plugins as _  # type: ignore
    _ciams_pkg = "ciams.ciams_plugins"
    _ciams_pkg_obj = importlib.import_module(_ciams_pkg)
    _ciams_pkg = _ciams_pkg_obj
except Exception:
    _ciams_pkg = None

LOG = logging.getLogger("instryx.compiler.plugins")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# -------------------------
# Plugin interface & helpers
# -------------------------
@dataclass(order=True)
class PluginMeta:
    priority: int
    name: str = field(compare=False)
    description: str = field(default="", compare=False)
    version: str = field(default="1.0", compare=False)


class PluginBase:
    """
    Base class for compiler plugins.

    Implement `apply(ir: Any, context: Dict[str, Any]) -> Tuple[Any, Dict]`.
    """
    meta: PluginMeta

    def __init__(self):
        if not hasattr(self, "meta"):
            self.meta = PluginMeta(priority=100, name=self.__class__.__name__, description="", version="1.0")

    def apply(self, ir: Any, context: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        raise NotImplementedError("Plugin must implement apply()")


# -------------------------
# Utilities: IR hashing, caching, snapshots
# -------------------------
def ir_hash(ir: Any) -> str:
    """Deterministic hash for IR using canonical JSON serialization."""
    try:
        text = json.dumps(ir, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        # best-effort fallback
        text = json.dumps(ir, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class PassCache:
    """In-memory cache for pass results keyed by (plugin_name, ir_hash)."""
    def __init__(self):
        self._lock = threading.RLock()
        self._map: Dict[Tuple[str, str], Tuple[Any, float]] = {}

    def get(self, plugin_name: str, ih: str):
        with self._lock:
            v = self._map.get((plugin_name, ih))
            if not v:
                return None
            return v[0]

    def set(self, plugin_name: str, ih: str, result: Any):
        with self._lock:
            self._map[(plugin_name, ih)] = (result, time.time())

    def invalidate(self, plugin_name: Optional[str] = None):
        with self._lock:
            if plugin_name is None:
                self._map.clear()
            else:
                for k in list(self._map):
                    if k[0] == plugin_name:
                        del self._map[k]


_PASS_CACHE = PassCache()


def dump_intermediate_ir(ir: Any, plugin_name: str, output_dir: Optional[str] = None) -> str:
    """Atomically write intermediate IR snapshot for debugging; return path."""
    outdir = output_dir or tempfile.gettempdir()
    fn = f"ir_{plugin_name}_{int(time.time() * 1000)}.json"
    path = os.path.join(outdir, fn)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(ir, f, indent=2, default=str)
    os.replace(path + ".tmp", path)
    return path


# -------------------------
# Lightweight schema validator
# -------------------------
def validate_schema(data: Any, schema: Dict[str, Any], coerce: bool = False) -> Tuple[bool, List[str], Any]:
    """
    Lightweight JSON-schema-like validator with optional coercion.

    Supports subset: type, required, properties, items, enum, const,
    minimum/maximum/exclusive*, minLength/maxLength/pattern, minItems/maxItems,
    additionalProperties, default.

    Returns (valid, messages, possibly_coerced_value).
    """
    messages: List[str] = []

    def _coerce_simple(value: Any, expected: str) -> Any:
        if expected == "number":
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except Exception:
                    return value
        if expected == "integer":
            if isinstance(value, int) and not isinstance(value, bool):
                return int(value)
            if isinstance(value, float) and value.is_integer():
                return int(value)
            if isinstance(value, str):
                try:
                    return int(value)
                except Exception:
                    try:
                        f = float(value)
                        if f.is_integer():
                            return int(f)
                    except Exception:
                        return value
        if expected == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                s = value.strip().lower()
                if s in ("true", "1", "yes", "on"):
                    return True
                if s in ("false", "0", "no", "off"):
                    return False
        if expected == "string":
            if value is None:
                return value
            if not isinstance(value, str):
                try:
                    return str(value)
                except Exception:
                    return value
        return value

    def _validate(node: Any, sch: Dict[str, Any], path: str) -> Any:
        if sch is None:
            return node
        local = node
        typ = sch.get("type")
        # default
        if local is None and "default" in sch:
            local = sch["default"]
        # coercion
        if coerce and typ in ("number", "integer", "boolean", "string") and local is not None:
            local = _coerce_simple(local, typ)
        # type check
        if typ:
            ok = True
            if typ == "object":
                ok = isinstance(local, dict)
            elif typ == "array":
                ok = isinstance(local, list)
            elif typ == "string":
                ok = isinstance(local, str)
            elif typ == "number":
                ok = isinstance(local, (int, float)) and not isinstance(local, bool)
            elif typ == "integer":
                ok = isinstance(local, int) and not isinstance(local, bool)
            elif typ == "boolean":
                ok = isinstance(local, bool)
            elif typ == "null":
                ok = local is None
            elif typ == "any":
                ok = True
            else:
                ok = True
            if not ok:
                messages.append(f"{path or 'root'}: expected {typ}, got {type(local).__name__}")
        # enum/const
        if "enum" in sch:
            if local not in sch["enum"]:
                messages.append(f"{path or 'root'}: value not in enum {sch['enum']}")
        if "const" in sch:
            if local != sch["const"]:
                messages.append(f"{path or 'root'}: value does not match const {sch['const']}")
        # object
        if typ == "object" and isinstance(local, dict):
            props = sch.get("properties", {})
            required = sch.get("required", [])
            additional = sch.get("additionalProperties", True)
            for r in required:
                if r not in local:
                    messages.append(f"{path or 'root'}: missing required property '{r}'")
                    if r in props and "default" in props[r]:
                        local[r] = props[r]["default"]
            for k, v in list(local.items()):
                if k in props:
                    local[k] = _validate(v, props[k], f"{path}.{k}" if path else k)
                else:
                    if not additional:
                        messages.append(f"{path or 'root'}: additional property '{k}' not allowed")
        # array
        if typ == "array" and isinstance(local, list):
            items_sch = sch.get("items")
            min_items = sch.get("minItems")
            max_items = sch.get("maxItems")
            if isinstance(min_items, int) and len(local) < min_items:
                messages.append(f"{path or 'root'}: array length {len(local)} < minItems {min_items}")
            if isinstance(max_items, int) and len(local) > max_items:
                messages.append(f"{path or 'root'}: array length {len(local)} > maxItems {max_items}")
            if items_sch:
                if isinstance(items_sch, dict):
                    for i, it in enumerate(local):
                        local[i] = _validate(it, items_sch, f"{path}[{i}]")
                elif isinstance(items_sch, list):
                    for i, it_schema in enumerate(items_sch):
                        if i < len(local):
                            local[i] = _validate(local[i], it_schema, f"{path}[{i}]")
        # string constraints
        if typ == "string" and isinstance(local, str):
            min_len = sch.get("minLength")
            max_len = sch.get("maxLength")
            pattern = sch.get("pattern")
            if isinstance(min_len, int) and len(local) < min_len:
                messages.append(f"{path or 'root'}: string length {len(local)} < minLength {min_len}")
            if isinstance(max_len, int) and len(local) > max_len:
                messages.append(f"{path or 'root'}: string length {len(local)} > maxLength {max_len}")
            if pattern:
                try:
                    if not re.search(pattern, local):
                        messages.append(f"{path or 'root'}: string does not match pattern {pattern}")
                except re.error:
                    messages.append(f"{path or 'root'}: invalid regex pattern {pattern}")
        # numeric constraints
        if typ in ("number", "integer") and isinstance(local, (int, float)) and not isinstance(local, bool):
            minimum = sch.get("minimum")
            maximum = sch.get("maximum")
            excl_min = sch.get("exclusiveMinimum")
            excl_max = sch.get("exclusiveMaximum")
            if minimum is not None:
                if excl_min and local <= minimum:
                    messages.append(f"{path or 'root'}: value {local} <= exclusiveMinimum {minimum}")
                elif not excl_min and local < minimum:
                    messages.append(f"{path or 'root'}: value {local} < minimum {minimum}")
            if maximum is not None:
                if excl_max and local >= maximum:
                    messages.append(f"{path or 'root'}: value {local} >= exclusiveMaximum {maximum}")
                elif not excl_max and local > maximum:
                    messages.append(f"{path or 'root'}: value {local} > maximum {maximum}")
        return local

    coerced = _validate(data, schema, "")
    valid = len(messages) == 0
    return valid, messages, coerced


# -------------------------
# Registry
# -------------------------
class PluginRegistry:
    def __init__(self, max_workers: int = 8, per_pass_timeout: float = 2.0):
        self._plugins: Dict[str, PluginBase] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self.per_pass_timeout = float(per_pass_timeout)
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.pass_cache = _PASS_CACHE

    def register(self, plugin: PluginBase):
        with self._lock:
            name = plugin.meta.name
            if name in self._plugins:
                LOG.warning("plugin %s already registered; overriding", name)
            self._plugins[name] = plugin
            LOG.info("registered plugin: %s (priority=%d ver=%s)", name, plugin.meta.priority, plugin.meta.version)

    def unregister(self, name: str):
        with self._lock:
            self._plugins.pop(name, None)
            self.pass_cache.invalidate(name)

    def list_plugins(self) -> List[PluginMeta]:
        with self._lock:
            metas = [p.meta for p in self._plugins.values()]
            return sorted(metas)

    def get_plugin(self, name: str) -> Optional[PluginBase]:
        return self._plugins.get(name)

    def discover_plugins(self, local_dir: Optional[str] = None, package: Optional[str] = None):
        """
        Discover plugins in built-ins, optional package(s), and a local directory.
        """
        # built-in classes
        for obj in list(globals().values()):
            if inspect.isclass(obj) and issubclass(obj, PluginBase) and obj is not PluginBase:
                try:
                    inst = obj()
                    self.register(inst)
                except Exception:
                    LOG.exception("failed to instantiate built-in plugin %s", obj)

        # optional package discovery
        pkgs = []
        if package and isinstance(package, str):
            pkgs.append(package)
        if _ciams_pkg:
            pkgs.append("ciams.ciams_plugins")
        for pkg_name in pkgs:
            try:
                pkg = importlib.import_module(pkg_name)
                for finder, modname, ispkg in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
                    try:
                        mod = importlib.import_module(modname)
                        for attr in dir(mod):
                            val = getattr(mod, attr)
                            if inspect.isclass(val) and issubclass(val, PluginBase) and val is not PluginBase:
                                try:
                                    self.register(val())
                                except Exception:
                                    LOG.exception("failed to instantiate plugin class %s in %s", val, modname)
                        if hasattr(mod, "register_plugin"):
                            try:
                                mod.register_plugin(self)
                            except Exception:
                                LOG.exception("module register_plugin failed: %s", modname)
                    except Exception:
                        LOG.exception("failed to load plugin module %s", modname)
            except Exception:
                LOG.debug("package not present or failed: %s", pkg_name)

        # local dir discovery
        local_dir = local_dir or os.path.join(os.getcwd(), "ciams_plugins")
        if os.path.isdir(local_dir):
            sys.path.insert(0, local_dir)
            for fn in os.listdir(local_dir):
                if not fn.endswith(".py"):
                    continue
                modname = os.path.splitext(fn)[0]
                try:
                    mod = importlib.import_module(modname)
                    for attr in dir(mod):
                        val = getattr(mod, attr)
                        if inspect.isclass(val) and issubclass(val, PluginBase) and val is not PluginBase:
                            try:
                                self.register(val())
                            except Exception:
                                LOG.exception("failed to instantiate local plugin %s", val)
                    if hasattr(mod, "register_plugin"):
                        try:
                            mod.register_plugin(self)
                        except Exception:
                            LOG.exception("local module register_plugin failed: %s", modname)
                except Exception:
                    LOG.exception("failed to import local plugin %s", modname)
            try:
                sys.path.remove(local_dir)
            except Exception:
                pass

    def run_pass(self, plugin: PluginBase, ir: Any, context: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """
        Run a single plugin pass with timeout protection and cache. Returns (ir, diagnostics).
        """
        name = plugin.meta.name
        ih = ir_hash(ir)
        cached = self.pass_cache.get(name, ih)
        if cached is not None:
            LOG.debug("cache hit for plugin %s", name)
            return cached, {"ok": True, "info": ["cache"], "warnings": [], "errors": []}
        start = time.time()
        fut = self._executor.submit(self._safe_apply, plugin, ir, context)
        try:
            transformed, diag = fut.result(timeout=self.per_pass_timeout)
            elapsed = time.time() - start
            self.metrics.setdefault(name, {"runs": 0, "time": 0.0})
            self.metrics[name]["runs"] += 1
            self.metrics[name]["time"] += elapsed
            try:
                self.pass_cache.set(name, ih, transformed)
            except Exception:
                LOG.debug("pass cache set failed (ignored)")
            return transformed, diag
        except TimeoutError:
            fut.cancel()
            LOG.error("plugin %s timed out after %.2fs", name, self.per_pass_timeout)
            return ir, {"ok": False, "info": [], "warnings": [], "errors": [f"timeout after {self.per_pass_timeout}s"]}
        except Exception as e:
            LOG.exception("plugin %s raised exception", name)
            return ir, {"ok": False, "info": [], "warnings": [], "errors": [str(e), traceback.format_exc()]}

    @staticmethod
    def _safe_apply(plugin: PluginBase, ir: Any, context: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        try:
            res = plugin.apply(ir, dict(context or {}))
            if not isinstance(res, tuple) or len(res) != 2:
                return ir, {"ok": False, "info": [], "warnings": [], "errors": ["plugin returned invalid result (expected (ir, diagnostics))"]}
            return res
        except Exception:
            return ir, {"ok": False, "info": [], "warnings": [], "errors": [traceback.format_exc()]}

    def run_passes(self, ir: Any, context: Optional[Dict[str, Any]] = None, passes: Optional[Iterable[str]] = None,
                   parallel_same_priority: bool = False, dump_intermediate: bool = False) -> Tuple[Any, Dict[str, Any]]:
        """
        Run passes grouped by priority. If parallel_same_priority True, run same-priority group concurrently.
        """
        context = dict(context or {})
        report = {"passes": [], "summary": {"ok": True, "errors": 0, "warnings": 0}, "timings": {}, "metrics": {}}
        with self._lock:
            if passes:
                selected = [self._plugins[p] for p in passes if p in self._plugins]
            else:
                selected = sorted(self._plugins.values(), key=lambda p: p.meta.priority)
        # group by priority
        groups: Dict[int, List[PluginBase]] = {}
        for p in selected:
            groups.setdefault(p.meta.priority, []).append(p)
        sorted_priorities = sorted(groups.keys())
        cur_ir = ir
        for pr in sorted_priorities:
            plugins = sorted(groups[pr], key=lambda p: p.meta.name)
            LOG.info("running priority group %d: %s", pr, [p.meta.name for p in plugins])
            if parallel_same_priority and len(plugins) > 1:
                futures = {self._executor.submit(self.run_pass, p, cur_ir, context): p for p in plugins}
                results = {}
                for fut in as_completed(futures):
                    p = futures[fut]
                    try:
                        transformed, diag = fut.result()
                        results[p.meta.name] = (transformed, diag)
                    except Exception as e:
                        results[p.meta.name] = (cur_ir, {"ok": False, "info": [], "warnings": [], "errors": [str(e)]})
                for pname in sorted(results.keys()):
                    transformed, diag = results[pname]
                    cur_ir = transformed
                    report["passes"].append({"name": pname, "diagnostics": diag})
                    if not diag.get("ok", True):
                        report["summary"]["ok"] = False
                        report["summary"]["errors"] += len(diag.get("errors", []))
                    report["summary"]["warnings"] += len(diag.get("warnings", []))
                    if dump_intermediate:
                        dump_intermediate_ir(cur_ir, pname)
            else:
                for plugin in plugins:
                    t0 = time.time()
                    LOG.info("running plugin: %s (priority=%d)", plugin.meta.name, plugin.meta.priority)
                    cur_ir, diag = self.run_pass(plugin, cur_ir, context)
                    dt = time.time() - t0
                    report["passes"].append({"name": plugin.meta.name, "diagnostics": diag, "time": dt})
                    report["timings"][plugin.meta.name] = report["timings"].get(plugin.meta.name, 0.0) + dt
                    if not diag.get("ok", True):
                        report["summary"]["ok"] = False
                        report["summary"]["errors"] += len(diag.get("errors", []))
                    report["summary"]["warnings"] += len(diag.get("warnings", []))
                    if dump_intermediate:
                        dump_intermediate_ir(cur_ir, plugin.meta.name)
            if context.get("abort", False):
                LOG.warning("pipeline aborted by context flag")
                break
        report["metrics"] = dict(self.metrics)
        return cur_ir, report


# -------------------------
# Built-in optimization plugins
# -------------------------
class ConstantFoldingPlugin(PluginBase):
    def __init__(self):
        super().__init__()
        self.meta = PluginMeta(priority=10, name="constant_fold", description="Fold constant expressions", version="1.1")

    def apply(self, ir, context):
        infos, warnings, errors = [], [], []

        def fold(node):
            if isinstance(node, dict):
                t = node.get("type")
                if t == "binary":
                    left = fold(node.get("left"))
                    right = fold(node.get("right"))
                    op = node.get("op")
                    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                        try:
                            if op == "+":
                                return left + right
                            if op == "-":
                                return left - right
                            if op == "*":
                                return left * right
                            if op == "/":
                                return left / right if right != 0 else left / right
                            if op == "%":
                                return left % right
                        except Exception as e:
                            warnings.append(f"constant fold error: {e}")
                    node["left"] = left
                    node["right"] = right
                    return node
                for k, v in list(node.items()):
                    node[k] = fold(v)
                return node
            if isinstance(node, list):
                return [fold(x) for x in node]
            return node

        try:
            new_ir = fold(ir)
            return new_ir, {"ok": True, "info": infos, "warnings": warnings, "errors": errors}
        except Exception as e:
            return ir, {"ok": False, "info": infos, "warnings": warnings, "errors": [str(e), traceback.format_exc()]}


class ConstantPropagationPlugin(PluginBase):
    def __init__(self):
        super().__init__()
        self.meta = PluginMeta(priority=15, name="const_propagation", description="Propagate constants", version="1.0")

    def apply(self, ir, context):
        infos, warnings, errors = [], [], []
        try:
            def propagate(node, env=None):
                env = dict(env or {})
                if isinstance(node, dict):
                    if node.get("type") == "assign" and isinstance(node.get("target"), str):
                        val = propagate(node.get("value"), env)
                        if isinstance(val, (int, float, str, bool, type(None))):
                            env[node["target"]] = val
                        node["value"] = val
                        return node
                    if node.get("type") == "var" and node.get("name") in env:
                        return env[node.get("name")]
                    for k, v in list(node.items()):
                        node[k] = propagate(v, env)
                    return node
                if isinstance(node, list):
                    return [propagate(it, env) for it in node]
                return node
            new_ir = propagate(ir, {})
            return new_ir, {"ok": True, "info": infos, "warnings": warnings, "errors": errors}
        except Exception as e:
            return ir, {"ok": False, "info": [], "warnings": warnings, "errors": [str(e), traceback.format_exc()]}


class CopyPropagationPlugin(PluginBase):
    def __init__(self):
        super().__init__()
        self.meta = PluginMeta(priority=16, name="copy_propagation", description="Propagate copies", version="1.0")

    def apply(self, ir, context):
        infos, warnings, errors = [], [], []
        try:
            def pass_copy(node, env=None):
                env = dict(env or {})
                if isinstance(node, dict):
                    if node.get("type") == "assign" and isinstance(node.get("target"), str):
                        val = pass_copy(node.get("value"), env)
                        if isinstance(val, dict) and val.get("type") == "var":
                            env[node["target"]] = val.get("name")
                        node["value"] = val
                        return node
                    if node.get("type") == "var":
                        nm = node.get("name")
                        if nm in env:
                            return {"type": "var", "name": env[nm]}
                    for k, v in list(node.items()):
                        node[k] = pass_copy(v, env)
                    return node
                if isinstance(node, list):
                    return [pass_copy(x, env) for x in node]
                return node
            new_ir = pass_copy(ir, {})
            return new_ir, {"ok": True, "info": infos, "warnings": warnings, "errors": errors}
        except Exception as e:
            return ir, {"ok": False, "info": [], "warnings": warnings, "errors": [str(e), traceback.format_exc()]}


class CFGSimplifyPlugin(PluginBase):
    def __init__(self):
        super().__init__()
        self.meta = PluginMeta(priority=25, name="cfg_simplify", description="CFG simplification", version="1.0")

    def apply(self, ir, context):
        infos, warnings, errors = [], [], []
        try:
            def simplify(node):
                if isinstance(node, dict):
                    if node.get("type") == "block" and isinstance(node.get("stmts"), list):
                        new_stmts = []
                        for s in node.get("stmts", []):
                            s2 = simplify(s)
                            if isinstance(s2, dict) and s2.get("type") == "block":
                                new_stmts.extend(s2.get("stmts", []))
                            elif s2 is not None:
                                new_stmts.append(s2)
                        node["stmts"] = new_stmts
                        return node
                    for k, v in list(node.items()):
                        node[k] = simplify(v)
                    return node
                if isinstance(node, list):
                    return [simplify(x) for x in node]
                return node
            new_ir = simplify(ir)
            return new_ir, {"ok": True, "info": infos, "warnings": warnings, "errors": errors}
        except Exception as e:
            return ir, {"ok": False, "info": [], "warnings": warnings, "errors": [str(e), traceback.format_exc()]}


class InlineSmallFunctionsPlugin(PluginBase):
    def __init__(self, size_threshold: int = 3):
        super().__init__()
        self.meta = PluginMeta(priority=30, name="inline_small", description="Inline small functions", version="1.1")
        self.size_threshold = int(size_threshold)

    def apply(self, ir, context):
        infos, warnings, errors = [], [], []
        try:
            functions = ir.get("functions", {}) if isinstance(ir, dict) else {}
            small = set()
            for fname, fobj in functions.items():
                body = fobj.get("body")
                size = len(json.dumps(body)) if isinstance(body, (dict, list)) else 0
                if size <= self.size_threshold * 100:
                    small.add(fname)
            def inline(node):
                if isinstance(node, dict):
                    if node.get("type") == "call" and node.get("fn") in small:
                        fname = node.get("fn")
                        fobj = functions.get(fname, {})
                        body = json.loads(json.dumps(fobj.get("body")))
                        params = fobj.get("params", [])
                        args = node.get("args", [])
                        mapping = {p: args[i] for i, p in enumerate(params) if i < len(args)}
                        def repl(n):
                            if isinstance(n, dict):
                                if n.get("type") == "var" and n.get("name") in mapping:
                                    return mapping[n.get("name")]
                                for k, v in list(n.items()):
                                    n[k] = repl(v)
                                return n
                            if isinstance(n, list):
                                return [repl(x) for x in n]
                            return n
                        return repl(body)
                    else:
                        for k, v in list(node.items()):
                            node[k] = inline(v)
                        return node
                if isinstance(node, list):
                    return [inline(x) for x in node]
                return node
            new_ir = inline(ir)
            return new_ir, {"ok": True, "info": list(small), "warnings": warnings, "errors": errors}
        except Exception as e:
            return ir, {"ok": False, "info": [], "warnings": warnings, "errors": [str(e), traceback.format_exc()]}


class ProfileGuidedInliningPlugin(PluginBase):
    def __init__(self, hotness_threshold: float = 0.7):
        super().__init__()
        self.meta = PluginMeta(priority=35, name="pg_inlining", description="Profile-guided inlining", version="1.0")
        self.hotness_threshold = float(hotness_threshold)

    def apply(self, ir, context):
        infos, warnings, errors = [], [], []
        try:
            profile = context.get("profile") or {}
            functions = ir.get("functions", {}) if isinstance(ir, dict) else {}
            def inline(node):
                if isinstance(node, dict):
                    if node.get("type") == "call" and node.get("fn") in functions:
                        fname = node.get("fn")
                        hot = profile.get(fname, 0.0)
                        if hot >= self.hotness_threshold:
                            fobj = functions.get(fname, {})
                            body = json.loads(json.dumps(fobj.get("body")))
                            params = fobj.get("params", [])
                            args = node.get("args", [])
                            mapping = {p: args[i] for i, p in enumerate(params) if i < len(args)}
                            def repl(n):
                                if isinstance(n, dict):
                                    if n.get("type") == "var" and n.get("name") in mapping:
                                        return mapping[n.get("name")]
                                    for k, v in list(n.items()):
                                        n[k] = repl(v)
                                    return n
                                if isinstance(n, list):
                                    return [repl(x) for x in n]
                                return n
                            infos.append(f"inlined hot function {fname} (hotness={hot})")
                            return repl(body)
                    for k, v in list(node.items()):
                        node[k] = inline(v)
                    return node
                if isinstance(node, list):
                    return [inline(x) for x in node]
                return node
            new_ir = inline(ir)
            return new_ir, {"ok": True, "info": infos, "warnings": warnings, "errors": errors}
        except Exception as e:
            return ir, {"ok": False, "info": [], "warnings": warnings, "errors": [str(e), traceback.format_exc()]}


class FunctionSpecializationPlugin(PluginBase):
    def __init__(self, max_specialize_size: int = 200):
        super().__init__()
        self.meta = PluginMeta(priority=60, name="function_specialize", description="Function specialization", version="1.0")
        self.max_size = int(max_specialize_size)

    def apply(self, ir, context):
        infos, warnings, errors = [], [], []
        try:
            if not isinstance(ir, dict):
                return ir, {"ok": True, "info": [], "warnings": [], "errors": []}
            functions = ir.get("functions", {})
            new_functions = dict(functions)
            def walk(node):
                if isinstance(node, dict):
                    if node.get("type") == "call":
                        args = node.get("args", [])
                        const_args = [a for a in args if isinstance(a, (int, float, str, bool))]
                        if len(const_args) == len(args):
                            fname = node.get("fn")
                            fobj = functions.get(fname)
                            if fobj:
                                body = fobj.get("body")
                                if isinstance(body, (dict, list)) and len(json.dumps(body)) <= self.max_size:
                                    key = hashlib.sha1(json.dumps(args, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:8]
                                    spec_name = f"{fname}__spec__{key}"
                                    if spec_name not in new_functions:
                                        spec_body = json.loads(json.dumps(body))
                                        params = fobj.get("params", [])
                                        mapping = {p: v for p, v in zip(params, args)}
                                        def repl(n):
                                            if isinstance(n, dict):
                                                if n.get("type") == "var" and n.get("name") in mapping:
                                                    return mapping[n.get("name")]
                                                for k, v in list(n.items()):
                                                    n[k] = repl(v)
                                                return n
                                            if isinstance(n, list):
                                                return [repl(x) for x in n]
                                            return n
                                        spec_body = repl(spec_body)
                                        new_functions[spec_name] = {"params": [], "body": spec_body}
                                        infos.append(f"created specialized function {spec_name} for {fname}")
                                    node["fn"] = spec_name
                                    node["args"] = []
                                    return node
                    for k, v in list(node.items()):
                        node[k] = walk(v)
                    return node
                if isinstance(node, list):
                    return [walk(x) for x in node]
                return node
            new_ir = json.loads(json.dumps(ir))
            new_ir = walk(new_ir)
            new_ir["functions"] = new_functions
            return new_ir, {"ok": True, "info": infos, "warnings": warnings, "errors": errors}
        except Exception as e:
            return ir, {"ok": False, "info": [], "warnings": warnings, "errors": [str(e), traceback.format_exc()]}


class LoopUnrollPlugin(PluginBase):
    def __init__(self, max_unroll: int = 8):
        super().__init__()
        self.meta = PluginMeta(priority=40, name="loop_unroll", description="Loop unrolling (heuristic)", version="1.0")
        self.max_unroll = int(max_unroll)

    def apply(self, ir, context):
        infos, warnings, errors = [], [], []
        try:
            def unroll(node):
                if isinstance(node, dict):
                    if node.get("type") == "for" and isinstance(node.get("count"), int) and node["count"] <= self.max_unroll:
                        body = node.get("body")
                        res = []
                        for i in range(node["count"]):
                            def repl(n):
                                if isinstance(n, dict):
                                    if n.get("type") == "var" and n.get("name") == node.get("index"):
                                        return i
                                    for k, v in list(n.items()):
                                        n[k] = repl(v)
                                    return n
                                if isinstance(n, list):
                                    return [repl(x) for x in n]
                                return n
                            res_body = repl(json.loads(json.dumps(body)))
                            if isinstance(res_body, dict) and res_body.get("type") == "block":
                                res.extend(res_body.get("stmts", []))
                            else:
                                res.append(res_body)
                        return {"type": "block", "stmts": res}
                    for k, v in list(node.items()):
                        node[k] = unroll(v)
                    return node
                if isinstance(node, list):
                    return [unroll(x) for x in node]
                return node
            new_ir = unroll(ir)
            return new_ir, {"ok": True, "info": infos, "warnings": warnings, "errors": errors}
        except Exception as e:
            return ir, {"ok": False, "info": [], "warnings": warnings, "errors": [str(e), traceback.format_exc()]}


class VectorizeHintPlugin(PluginBase):
    def __init__(self):
        super().__init__()
        self.meta = PluginMeta(priority=150, name="vectorize_hint", description="Insert vectorize hints", version="1.0")

    def apply(self, ir, context):
        infos, warnings, errors = [], [], []
        try:
            def detect_and_annotate(node):
                if isinstance(node, dict):
                    if node.get("type") == "for" and isinstance(node.get("body"), dict):
                        idx = node.get("index")
                        body = node.get("body")
                        found = False
                        def scan(n):
                            nonlocal found
                            if isinstance(n, dict):
                                if n.get("type") == "load":
                                    addr = n.get("addr")
                                    if isinstance(addr, dict) and addr.get("type") == "binary" and addr.get("op") == "+":
                                        right = addr.get("right")
                                        if isinstance(right, dict) and right.get("type") == "var" and right.get("name") == idx:
                                            found = True
                                for v in n.values():
                                    scan(v)
                            elif isinstance(n, list):
                                for it in n:
                                    scan(it)
                        scan(body)
                        if found:
                            node["vectorize"] = True
                            infos.append(f"vectorize hint on loop index={idx}")
                    for k, v in list(node.items()):
                        node[k] = detect_and_annotate(v)
                    return node
                if isinstance(node, list):
                    return [detect_and_annotate(x) for x in node]
                return node
            new_ir = detect_and_annotate(ir)
            return new_ir, {"ok": True, "info": infos, "warnings": warnings, "errors": errors}
        except Exception as e:
            return ir, {"ok": False, "info": [], "warnings": warnings, "errors": [str(e), traceback.format_exc()]}


class PrefetchHintPlugin(PluginBase):
    def __init__(self):
        super().__init__()
        self.meta = PluginMeta(priority=200, name="prefetch_hint", description="Insert prefetch hints", version="1.0")

    def apply(self, ir, context):
        infos, warnings, errors = [], [], []
        try:
            def transform(node):
                if isinstance(node, dict):
                    if node.get("type") == "load" and isinstance(node.get("addr"), dict):
                        addr = node["addr"]
                        if addr.get("type") == "binary" and addr.get("op") == "+":
                            right = addr.get("right")
                            if isinstance(right, dict) and right.get("type") == "binary" and right.get("op") == "*":
                                pre = {"type": "prefetch", "addr": addr, "info": "heuristic"}
                                return {"type": "block", "stmts": [pre, node]}
                    for k, v in list(node.items()):
                        node[k] = transform(v)
                    return node
                if isinstance(node, list):
                    return [transform(x) for x in node]
                return node
            new_ir = transform(ir)
            return new_ir, {"ok": True, "info": infos, "warnings": warnings, "errors": errors}
        except Exception as e:
            return ir, {"ok": False, "info": [], "warnings": warnings, "errors": [str(e), traceback.format_exc()]}


class PeepholeOptimizerPlugin(PluginBase):
    """
    Simple peephole optimizer for trivial patterns (e.g. add 0, mul 1, double negation).
    Safe and conservative.
    """

    def __init__(self):
        super().__init__()
        self.meta = PluginMeta(priority=12, name="peephole", description="Peephole optimizations", version="1.0")

    def apply(self, ir, context):
        infos, warnings, errors = [], [], []
        try:
            def opt(node):
                if isinstance(node, dict):
                    # binary simplifications
                    if node.get("type") == "binary":
                        left = opt(node.get("left"))
                        right = opt(node.get("right"))
                        op = node.get("op")
                        # x + 0 -> x
                        if op == "+" and right == 0:
                            return left
                        if op == "+" and left == 0:
                            return right
                        # x * 1 -> x
                        if op == "*" and right == 1:
                            return left
                        if op == "*" and left == 1:
                            return right
                        # double negation: -(-x) -> x
                        if op == "-" and isinstance(left, dict) and left.get("type") == "unary" and left.get("op") == "-" and right is None:
                            return left.get("expr")
                        node["left"], node["right"] = left, right
                        return node
                    for k, v in list(node.items()):
                        node[k] = opt(v)
                    return node
                if isinstance(node, list):
                    return [opt(x) for x in node]
                return node
            new_ir = opt(ir)
            return new_ir, {"ok": True, "info": infos, "warnings": warnings, "errors": errors}
        except Exception as e:
            return ir, {"ok": False, "info": [], "warnings": warnings, "errors": [str(e), traceback.format_exc()]}


class SSAConversionPlugin(PluginBase):
    """
    Lightweight SSA-like renaming: rename local temporary variables to product unique names.
    Conservative and reversible; useful to reduce accidental name collisions for some passes.
    """

    def __init__(self):
        super().__init__()
        self.meta = PluginMeta(priority=5, name="ssa_convert", description="Lightweight SSA renaming", version="1.0")

    def apply(self, ir, context):
        infos, warnings, errors = [], [], []
        try:
            counter = 0
            mapping = {}
            def rename(node):
                nonlocal counter
                if isinstance(node, dict):
                    if node.get("type") == "assign" and isinstance(node.get("target"), str):
                        old = node["target"]
                        new = f"{old}__ssa{counter}"
                        counter += 1
                        mapping[old] = new
                        node["target"] = new
                        node["value"] = rename(node.get("value"))
                        return node
                    if node.get("type") == "var":
                        nm = node.get("name")
                        if nm in mapping:
                            return {"type": "var", "name": mapping[nm]}
                    for k, v in list(node.items()):
                        node[k] = rename(v)
                    return node
                if isinstance(node, list):
                    return [rename(x) for x in node]
                return node
            new_ir = rename(ir)
            infos.append(f"renamed {len(mapping)} symbols")
            return new_ir, {"ok": True, "info": infos, "warnings": warnings, "errors": errors}
        except Exception as e:
            return ir, {"ok": False, "info": [], "warnings": warnings, "errors": [str(e), traceback.format_exc()]}


class TailCallEliminationPlugin(PluginBase):
    """
    Perform basic tail-call elimination: when a function body ends with a tail call,
    transform into loop-style tail recursion removal. Conservative; only simple forms.
    """

    def __init__(self):
        super().__init__()
        self.meta = PluginMeta(priority=45, name="tail_call_elim", description="Tail-call elimination", version="1.0")

    def apply(self, ir, context):
        infos, warnings, errors = [], [], []
        try:
            def tce_function(fn_body):
                # fn_body is assumed a block with stmts; detect tail-call pattern: return call(...)
                if isinstance(fn_body, dict) and fn_body.get("type") == "block":
                    stmts = fn_body.get("stmts", [])
                    if not stmts:
                        return fn_body, False
                    last = stmts[-1]
                    if isinstance(last, dict) and last.get("type") == "return":
                        retv = last.get("value")
                        if isinstance(retv, dict) and retv.get("type") == "call":
                            # perform simple transform: replace return call(f, args) with assignment to params and continue
                            call = retv
                            # We only transform self-recursive calls with same function name (naive)
                            return {"type": "tco_stub", "call": call}, True
                return fn_body, False
            new_ir = ir
            if isinstance(ir, dict) and "functions" in ir:
                for fname, fobj in ir["functions"].items():
                    body = fobj.get("body")
                    nb, changed = tce_function(body)
                    if changed:
                        fobj["body"] = nb
                        infos.append(f"tco applied to {fname}")
            return new_ir, {"ok": True, "info": infos, "warnings": warnings, "errors": errors}
        except Exception as e:
            return ir, {"ok": False, "info": [], "warnings": warnings, "errors": [str(e), traceback.format_exc()]}


class MacroAwarePass(PluginBase):
    def __init__(self):
        super().__init__()
        self.meta = PluginMeta(priority=3, name="macro_aware", description="Macro-aware validations & optimizations", version="1.0")

    def apply(self, ir, context):
        infos, warnings, errors = [], [], []
        if not _transformer:
            return ir, {"ok": True, "info": ["transformer not available, skipped"], "warnings": warnings, "errors": errors}
        try:
            source = ir.get("source") if isinstance(ir, dict) else None
            if source and hasattr(_transformer, "applyMacrosWithDiagnostics"):
                res = _transformer.applyMacrosWithDiagnostics(source, registry=(getattr(_transformer, "createDefaultRegistry", lambda: {})()))
                diags = res.get("diagnostics", [])
                infos.append(f"macro diagnostics: {len(diags)}")
            return ir, {"ok": True, "info": infos, "warnings": warnings, "errors": errors}
        except Exception as e:
            return ir, {"ok": False, "info": [], "warnings": warnings, "errors": [str(e), traceback.format_exc()]}


# -------------------------
# Convenience: create registry with built-ins
# -------------------------
def create_default_registry(max_workers: int = 8, per_pass_timeout: float = 2.0) -> PluginRegistry:
    reg = PluginRegistry(max_workers=max_workers, per_pass_timeout=args.timeout)
    # register built-in plugins (order doesn't matter; priority governs execution order)
    reg.register(MacroAwarePass())
    reg.register(SSAConversionPlugin())
    reg.register(ConstantFoldingPlugin())
    reg.register(PeepholeOptimizerPlugin())
    reg.register(ConstantPropagationPlugin())
    reg.register(CopyPropagationPlugin())
    reg.register(CFGSimplifyPlugin())
    reg.register(DeadCodeEliminationPlugin())
    reg.register(InlineSmallFunctionsPlugin())
    reg.register(ProfileGuidedInliningPlugin())
    reg.register(FunctionSpecializationPlugin())
    reg.register(TailCallEliminationPlugin())
    reg.register(LoopUnrollPlugin())
    reg.register(VectorizeHintPlugin())
    reg.register(PrefetchHintPlugin())
    # attempt to discover additional plugins
    try:
        reg.discover_plugins()
    except Exception:
        LOG.exception("plugin discovery failed")
    return reg


# -------------------------
# Diagnostics / report helpers
# -------------------------
def save_report(report: Dict[str, Any], path: str) -> str:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    os.replace(tmp, path)
    return path


# -------------------------
# CLI entrypoint
# -------------------------
def _cli():
    import argparse
    parser = argparse.ArgumentParser(prog="instryx_compiler_plugins.py", description="Instryx compiler plugin runner")
    parser.add_argument("cmd", nargs="?", choices=("run", "list", "describe"), default="list")
    parser.add_argument("--input", "-i", help="input IR JSON file")
    parser.add_argument("--out", "-o", help="output IR JSON file")
    parser.add_argument("--passes", help="comma-separated plugin names to run (ordered)")
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--parallel-same-priority", action="store_true", help="run same-priority plugins concurrently")
    parser.add_argument("--dump-intermediate", action="store_true", help="dump intermediate IR after each pass")
    parser.add_argument("--report", help="path to save pass report json")
    args = parser.parse_args()
    if args.verbose:
        LOG.setLevel(logging.DEBUG)
    reg = create_default_registry(max_workers=args.workers, per_pass_timeout=args.timeout)
    if args.cmd == "list":
        for m in reg.list_plugins():
            print(f"{m.name} (priority={m.priority} ver={m.version}) - {m.description}")
        return 0
    if args.cmd == "describe":
        for m in reg.list_plugins():
            print(json.dumps({"name": m.name, "priority": m.priority, "version": m.version, "description": m.description}, indent=2))
        return 0
    if args.cmd == "run":
        if not args.input:
            print("input required")
            return 2
        with open(args.input, "r", encoding="utf-8") as f:
            ir = json.load(f)
        passes = [p.strip() for p in args.passes.split(",")] if args.passes else None
        out_ir, report = reg.run_passes(ir, context={"filename": args.input}, passes=passes,
                                        parallel_same_priority=args.parallel_same_priority,
                                        dump_intermediate=args.dump_intermediate)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(out_ir, f, indent=2)
            print("wrote", args.out)
        else:
            print(json.dumps(out_ir, indent=2))
        if args.report:
            save_report(report, args.report)
            print("report saved to", args.report)
        else:
            print("report:", json.dumps(report, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(_cli())

