"""
instryx_macro_transformer_model.py

Production-ready macro transformer for Instryx textual macros.

Features
- Macro registry with runtime register/unregister
- Safe scanner for '@macro ...;' invocations (skips strings/comments)
- applyMacrosWithDiagnostics(...) returns transformed text + structured diagnostics
- applyMacros(...) convenience wrapper
- File preview/apply with transactional backup and rollback
- Plugin discovery + automatic loading of Python plugins (ciams_plugins)
- Built-in macros:
  - match_pattern(EnumName, varName) -> uses instryx_match_enum_struct if available
  - emit_helper(name, ...) -> uses instryx_memory_math_loops_codegen if available
  - vectorize_hint(loopHeader) -> emits a vectorize hint
  - tile_loop(N,tileSize) -> emits tiled-loop hint
- Optional AST-based lowering via instryx_syntax_morph when available
- CLI: list, preview, apply, serve (HTTP preview) and export-registry
"""

from __future__ import annotations
import argparse
import importlib
import inspect
import io
import json
import logging
import os
import re
import shutil
import socket
import sys
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple

# Optional integrations (best-effort)
_try_match_tool = None
try:
    from instryx_match_enum_struct import DMatchTool  # type: ignore
    _try_match_tool = DMatchTool()
except Exception:
    _try_match_tool = None

_try_codegen = None
try:
    import instryx_memory_math_loops_codegen as codegen  # type: ignore
    _try_codegen = codegen
except Exception:
    _try_codegen = None

_try_syntax_morph = None
try:
    import instryx_syntax_morph as syntax_morph  # type: ignore
    _try_syntax_morph = syntax_morph
except Exception:
    _try_syntax_morph = None

# Logging
LOG = logging.getLogger("instryx.macro.transformer.model")
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOG.addHandler(h)

# Types
MacroFn = Callable[[List[str], Dict[str, Any]], str]  # returns replacement text
Registry = Dict[str, MacroFn]

# Scanner regex for macro name (after '@')
_RE_MACRO_NAME = re.compile(r"@([A-Za-z_][\w]*)", flags=re.ASCII)

# Default registry export path (optional)
DEFAULT_REGISTRY_EXPORT = "instryx_macro_registry.json"

_LOCK = threading.RLock()


# -------------------------
# Low-level scanner
# -------------------------
def _scan_macros(source: str) -> List[Tuple[int, int, str, str]]:
    """
    Scan source and return list of (start_idx, end_idx, macro_name, raw_args_text)
    `end_idx` is exclusive index (position after semicolon).
    Skips string literals and single-line comments to reduce false positives.
    """
    res: List[Tuple[int, int, str, str]] = []
    i = 0
    L = len(source)
    in_string = None
    while i < L:
        ch = source[i]
        # skip string contents
        if in_string:
            if ch == in_string and source[i - 1] != "\\":
                in_string = None
            i += 1
            continue
        if source.startswith("//", i):
            nl = source.find("\n", i)
            i = nl + 1 if nl != -1 else L
            continue
        if ch in ('"', "'"):
            in_string = ch
            i += 1
            continue
        if ch == "@":
            m = _RE_MACRO_NAME.match(source, i)
            if not m:
                i += 1
                continue
            name = m.group(1)
            payload_start = m.end()
            # find semicolon that is top-level (not inside parentheses/braces/strings)
            j = payload_start
            depth_paren = depth_brace = depth_brack = 0
            in_s = None
            found = False
            while j < L:
                c = source[j]
                if in_s:
                    if c == in_s and source[j - 1] != "\\":
                        in_s = None
                    j += 1
                    continue
                if c in ('"', "'"):
                    in_s = c
                    j += 1
                    continue
                if c == "(":
                    depth_paren += 1
                elif c == ")":
                    depth_paren = max(0, depth_paren - 1)
                elif c == "{":
                    depth_brace += 1
                elif c == "}":
                    depth_brace = max(0, depth_brace - 1)
                elif c == "[":
                    depth_brack += 1
                elif c == "]":
                    depth_brack = max(0, depth_brack - 1)
                elif c == ";" and depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
                    raw_args = source[payload_start:j].strip()
                    res.append((i, j + 1, name, raw_args))
                    found = True
                    j += 1
                    break
                j += 1
            if not found:
                # ignore unterminated macro invocation
                i = payload_start
                continue
            i = j
            continue
        i += 1
    return res


# -------------------------
# Arg parsing (safe-ish)
# -------------------------
def _parse_macro_args(raw: str) -> List[str]:
    """
    Parse macro raw args into top-level comma separated strings.
    Preserves string quoting inside args.
    """
    s = raw.strip()
    if not s:
        return []
    # remove wrapping parentheses if any
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    in_s = None
    for ch in s:
        if in_s:
            buf.append(ch)
            if ch == in_s and (len(buf) < 2 or buf[-2] != "\\"):
                in_s = None
            continue
        if ch in ('"', "'"):
            buf.append(ch)
            in_s = ch
            continue
        if ch == "(":
            depth += 1
            buf.append(ch)
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
            continue
        if ch == "," and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
            continue
        buf.append(ch)
    if buf:
        p = "".join(buf).strip()
        if p:
            parts.append(p)
    return parts


# -------------------------
# Registry + macros
# -------------------------
class MacroRegistry:
    def __init__(self):
        self._macros: Registry = {}
        self._lock = threading.RLock()

    def register(self, name: str, fn: MacroFn) -> None:
        with self._lock:
            self._macros[name] = fn
            LOG.info("macro registered: %s", name)

    def unregister(self, name: str) -> None:
        with self._lock:
            if name in self._macros:
                del self._macros[name]
                LOG.info("macro unregistered: %s", name)

    def get(self, name: str) -> Optional[MacroFn]:
        with self._lock:
            return self._macros.get(name)

    def list(self) -> List[str]:
        with self._lock:
            return sorted(list(self._macros.keys()))

    def as_dict(self) -> Dict[str, str]:
        with self._lock:
            return {k: (getattr(v, "__name__", "<callable>")) for k, v in self._macros.items()}


_REGISTRY = MacroRegistry()


# -------------------------
# Built-in macro implementations
# -------------------------
def _macro_match_pattern(args: List[str], ctx: Dict[str, Any]) -> str:
    """
    @match_pattern EnumName, varName;
    Expands to a match skeleton using instryx_match_enum_struct.DMatchTool if available.
    If the enum is not found in src, emits a helpful placeholder comment.
    """
    enum_name = args[0] if len(args) >= 1 else ""
    var_name = args[1] if len(args) >= 2 else "v"
    src = ctx.get("source", "") or ""
    if not _try_match_tool:
        return f"/* match_pattern: DMatchTool not available (requested {enum_name},{var_name}) */"
    # Attempt to find the enum in provided source
    try:
        enums = _try_match_tool.find_enums(src)
        ed = next((e for e in enums if e.name == enum_name), None)
        if ed:
            return _try_match_tool.generate_match_stub(ed, var_name=var_name)
        # fallback: try to search globally (not implemented) -> return placeholder
        return f"/* match_pattern: enum {enum_name} not found in source */\n// match {var_name} {{ /* add arms */ }}"
    except Exception as e:
        LOG.exception("match_pattern failed")
        return f"/* match_pattern error: {e} */"


def _macro_emit_helper(args: List[str], ctx: Dict[str, Any]) -> str:
    """
    @emit_helper name, arg1, arg2;
    Calls into instryx_memory_math_loops_codegen when available to emit helper text.
    """
    if not _try_codegen:
        return f"/* emit_helper: codegen module not available for args={args} */"
    if not args:
        return "/* emit_helper: missing helper name */"
    name = args[0]
    helper_args = []
    if len(args) > 1:
        helper_args = args[1:]
    try:
        if hasattr(_try_codegen, "emit_helper"):
            return _try_codegen.emit_helper(name, *helper_args)
        fn = getattr(_try_codegen, f"generate_{name}", None)
        if callable(fn):
            return fn(*helper_args)
        return f"/* emit_helper: helper {name} not found */"
    except Exception as e:
        LOG.exception("emit_helper error")
        return f"/* emit_helper error: {e} */"


def _macro_vectorize_hint(args: List[str], ctx: Dict[str, Any]) -> str:
    """
    @vectorize_hint loopHeader;
    Emits a textual vectorize hint wrapper around a loop header snippet.
    Example usage: @vectorize_hint for (i=0; i<len(arr); i+=1);
    """
    if not args:
        return "/* vectorize_hint: missing loop header */"
    header = args[0]
    # naive wrap: user provided header should be the loop header (without body)
    return f"/* vectorize hint */\n{header} {{ /* { 'vectorized body' } */ }}\n"


def _macro_tile_loop(args: List[str], ctx: Dict[str, Any]) -> str:
    """
    @tile_loop forHeader, tileSize;
    Emits a simple loop tiling scaffold.
    """
    if not args:
        return "/* tile_loop: missing args */"
    header = args[0]
    tile = args[1] if len(args) > 1 else "64"
    return f"/* tiled loop (tile={tile}) */\n{header} {{ /* inner tiled body placeholder */ }}\n"


# -------------------------
# Registry initialization
# -------------------------
def createDefaultRegistry() -> Registry:
    reg = {}
    reg["match_pattern"] = _macro_match_pattern
    reg["emit_helper"] = _macro_emit_helper
    reg["vectorize_hint"] = _macro_vectorize_hint
    reg["tile_loop"] = _macro_tile_loop
    # register into global MacroRegistry for convenience
    for k, v in reg.items():
        _REGISTRY.register(k, v)
    return reg


def createFullRegistry() -> Registry:
    # For now same as default; external callers can modify result
    return createDefaultRegistry()


# -------------------------
# Core transformer API
# -------------------------
def applyMacrosWithDiagnostics(source: str, registry: Optional[Registry] = None, opts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Apply macros in `source` using `registry` (or default).
    Returns dict: {"result": {"transformed": str}, "diagnostics": [ {level,msg,macro,range,...} ] }
    """
    if registry is None:
        registry = createDefaultRegistry()
    diagnostics: List[Dict[str, Any]] = []
    matches = _scan_macros(source)
    if not matches:
        return {"result": {"transformed": source}, "diagnostics": diagnostics}

    out_parts: List[str] = []
    last = 0
    for start_idx, end_idx, name, raw_args in matches:
        # copy preceding text
        out_parts.append(source[last:start_idx])
        args = _parse_macro_args(raw_args)
        macro_fn = registry.get(name) or _REGISTRY.get(name)
        if macro_fn is None:
            diagnostics.append({"level": "warning", "message": f"macro '{name}' not found", "macro": name, "range": [start_idx, end_idx]})
            out_parts.append(source[start_idx:end_idx])  # keep as-is
            last = end_idx
            continue
        try:
            # macro may accept context
            ctx = {"source": source, "opts": opts or {}, "registry": registry}
            repl = macro_fn(args, ctx)
            if repl is None:
                repl = ""
            if not isinstance(repl, str):
                repl = str(repl)
            out_parts.append(repl)
            diagnostics.append({"level": "info", "message": f"macro '{name}' expanded", "macro": name, "range": [start_idx, end_idx]})
        except Exception as e:
            LOG.exception("macro expansion failed: %s", name)
            diagnostics.append({"level": "error", "message": f"macro '{name}' error: {e}", "macro": name, "range": [start_idx, end_idx], "trace": traceback.format_exc()})
            out_parts.append(source[start_idx:end_idx])  # preserve original
        last = end_idx
    out_parts.append(source[last:])
    transformed = "".join(out_parts)

    # Optionally run AST-level lowerings if enabled in opts and available
    try:
        if opts and opts.get("ast_lowering") and _try_syntax_morph:
            try:
                transformed = _try_syntax_morph.apply_lowerings(transformed)
                diagnostics.append({"level": "info", "message": "AST lowerings applied", "macro": "ast_lowering"})
            except Exception:
                LOG.exception("AST lowering failed")
                diagnostics.append({"level": "warning", "message": "AST lowering failed", "macro": "ast_lowering", "trace": traceback.format_exc()})
    except Exception:
        # non-fatal
        pass

    return {"result": {"transformed": transformed}, "diagnostics": diagnostics}


def applyMacros(source: str, registry: Optional[Registry] = None, opts: Optional[Dict[str, Any]] = None) -> str:
    return applyMacrosWithDiagnostics(source, registry=registry, opts=opts)["result"]["transformed"]


# -------------------------
# File preview / apply helpers
# -------------------------
def preview_apply_file(path: str, registry: Optional[Registry] = None, opts: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """
    Read file, apply macros in preview mode (do not write). Returns (ok, transformed_text, diagnostics)
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
    except Exception as e:
        return False, f"read failed: {e}", []
    res = applyMacrosWithDiagnostics(src, registry=registry, opts=opts)
    transformed = res.get("result", {}).get("transformed", src)
    diagnostics = res.get("diagnostics", [])
    return True, transformed, diagnostics


def apply_to_file_atomic(path: str, registry: Optional[Registry] = None, opts: Optional[Dict[str, Any]] = None, backup: bool = True) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """
    Apply macros and write transformed content to path (atomic via temp -> replace).
    Creates a .bak backup if backup True.
    Returns (ok, out_path, diagnostics)
    """
    ok, transformed_or_msg, diagnostics = preview_apply_file(path, registry=registry, opts=opts)
    if not ok:
        return False, transformed_or_msg, diagnostics or []
    transformed = transformed_or_msg
    # safety check: don't allow huge expansions by default
    orig_size = os.path.getsize(path) if os.path.exists(path) else len(transformed)
    new_size = len(transformed.encode("utf-8"))
    max_growth = opts.get("max_growth_bytes", 200_000) if opts else 200_000
    if new_size - orig_size > max_growth and (not (opts or {}).get("force", False)):
        return False, "expansion too large; aborting (use force=true in opts to override)", diagnostics
    # write to temp then replace
    dirp = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".__instryx_macro_tmp_", dir=dirp, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(transformed)
        if backup and os.path.exists(path):
            bak = path + ".bak"
            shutil.copy2(path, bak)
        os.replace(tmp, path)
        return True, path, diagnostics
    except Exception as e:
        LOG.exception("apply_to_file_atomic failed")
        try:
            os.unlink(tmp)
        except Exception:
            pass
        return False, f"write failed: {e}", diagnostics


# -------------------------
# Plugin discovery (Python modules under ciams_plugins)
# -------------------------
def load_plugins_from_dir(plugins_dir: Optional[str] = None, pass_registry: Optional[MacroRegistry] = None) -> List[str]:
    """
    Load plugins: Python files in plugins_dir exporting `register(registry_or_module)` function.
    Returns list of loaded module names.
    """
    loaded = []
    plugins_dir = plugins_dir or os.path.join(os.path.dirname(__file__), "ciams_plugins")
    if not os.path.isdir(plugins_dir):
        return loaded
    sys.path.insert(0, plugins_dir)
    for fn in os.listdir(plugins_dir):
        if not fn.endswith(".py"):
            continue
        modname = fn[:-3]
        try:
            mod = importlib.import_module(modname)
            if hasattr(mod, "register") and callable(mod.register):
                try:
                    # pass either MacroRegistry or module-level registry dict
                    regobj = pass_registry or _REGISTRY
                    # If register signature accepts two params, also pass createDefaultRegistry
                    sig = inspect.signature(mod.register)
                    if len(sig.parameters) == 2:
                        mod.register(regobj, createDefaultRegistry)
                    else:
                        mod.register(regobj)
                except Exception:
                    LOG.exception("plugin %s register failed", modname)
                loaded.append(modname)
        except Exception:
            LOG.exception("loading plugin %s failed", modname)
    try:
        sys.path.remove(plugins_dir)
    except Exception:
        pass
    return loaded


# -------------------------
# CLI & HTTP mini-server
# -------------------------
class _HTTPHandler(BaseHTTPRequestHandler):
    registry: Optional[Registry] = None

    def _send_json(self, obj: Any, status: int = 200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/list"):
            names = list((_REGISTRY.list()))
            self._send_json({"macros": names})
            return
        if self.path.startswith("/preview"):
            # expect query param file=...
            qs = {}
            if "?" in self.path:
                _, q = self.path.split("?", 1)
                for pair in q.split("&"):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        qs[k] = v
            file = qs.get("file")
            if not file or not os.path.exists(file):
                self._send_json({"error": "file missing"}, 400)
                return
            ok, transformed, diagnostics = preview_apply_file(file, registry=self.registry)
            if not ok:
                self._send_json({"error": transformed}, 500)
                return
            self._send_json({"transformed": transformed, "diagnostics": diagnostics})
            return
        self._send_json({"error": "unknown endpoint"}, 404)

    def do_POST(self):
        if self.path != "/apply":
            self._send_json({"error": "unknown endpoint"}, 404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        try:
            data = json.loads(body or "{}")
            file = data.get("file")
            if not file or not os.path.exists(file):
                self._send_json({"error": "file missing"}, 400)
                return
            opts = data.get("opts", {})
            ok, out, diagnostics = apply_to_file_atomic(file, registry=self.registry, opts=opts)
            self._send_json({"ok": ok, "out": out, "diagnostics": diagnostics})
        except Exception as e:
            LOG.exception("HTTP apply failed")
            self._send_json({"error": str(e)}, 500)


def serve_http(port: int = 8787, host: str = "127.0.0.1", registry: Optional[Registry] = None):
    handler = _HTTPHandler
    handler.registry = registry or createDefaultRegistry()
    server = HTTPServer((host, port), handler)
    LOG.info("serving macro transformer API on %s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("server stopped")
    finally:
        server.server_close()


# -------------------------
# Export / import registry helpers
# -------------------------
def export_registry(path: str = DEFAULT_REGISTRY_EXPORT) -> str:
    d = _REGISTRY.as_dict()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
        return path
    except Exception as e:
        LOG.exception("export_registry failed")
        raise


# -------------------------
# CLI entrypoint
# -------------------------
def _cli():
    p = argparse.ArgumentParser(prog="instryx_macro_transformer_model.py")
    p.add_argument("cmd", nargs="?", help="command (list, preview, apply, serve, export-registry, load-plugins)")
    p.add_argument("target", nargs="?", help="file target or plugin dir")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.cmd in (None, "help"):
        p.print_help()
        return 0

    if args.cmd == "list":
        print("registered macros:", _REGISTRY.list())
        return 0

    if args.cmd == "preview":
        if not args.target:
            print("file required")
            return 2
        ok, transformed, diagnostics = preview_apply_file(args.target, registry=createDefaultRegistry(), opts={"force": args.force})
        if not ok:
            print("preview failed:", transformed)
            return 2
        print(transformed)
        if diagnostics:
            print("\nDiagnostics:")
            print(json.dumps(diagnostics, indent=2))
        return 0

    if args.cmd == "apply":
        if not args.target:
            print("file required")
            return 2
        ok, out, diagnostics = apply_to_file_atomic(args.target, registry=createDefaultRegistry(), opts={"force": args.force})
        if not ok:
            print("apply failed:", out)
            return 2
        print("wrote", out)
        if diagnostics:
            print("diagnostics:", json.dumps(diagnostics, indent=2))
        return 0

    if args.cmd == "serve":
        serve_http(port=args.port, host=args.host, registry=createDefaultRegistry())
        return 0

    if args.cmd == "export-registry":
        path = args.target or DEFAULT_REGISTRY_EXPORT
        export_registry(path)
        print("exported registry ->", path)
        return 0

    if args.cmd == "load-plugins":
        dirp = args.target or os.path.join(os.path.dirname(__file__), "ciams_plugins")
        loaded = load_plugins_from_dir(dirp)
        print("loaded plugins:", loaded)
        return 0

    print("unknown command", args.cmd)
    p.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(_cli())

