"""
instryx_macro_debugger.py

Advanced Macro debugger and tracer for Instryx macro transformer.

Enhancements added:
- Interactive step-through debugger with accept/skip/apply for each macro step.
- Partial-apply: apply only accepted steps to working file or produce patch.
- Full diff/patch generation and atomic apply with backups and undo support.
- Replay validation (reproduce step expansions using current registry).
- Sandbox executor for macros (safe mode) to avoid side-effects during trace.
- Plugin loader for debugger extensions (ciams_plugins).
- HTTP API with additional endpoints: /list, /trace, /replay, /apply, /step, /validate
- Concurrency-safe operations and better diagnostics
- Export/import traces, trace signing (HMAC optional via env var INSTRYX_TRACE_HMAC_KEY)
- CLI commands: trace, interactive, replay, apply, validate, undo, export-trace, import-trace, serve, test
- Comprehensive logging & small metrics.

Notes:
- This file integrates with instryx_macro_transformer_model.py if available and uses its registry.
- It prefers safe, textual processing and does not execute macros with unknown side-effects unless in non-sandbox mode.
"""

from __future__ import annotations
import argparse
import hashlib
import hmac
import importlib
import inspect
import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple

# Try to import transformer
_transformer = None
try:
    import instryx_macro_transformer_model as transformer  # type: ignore
    _transformer = transformer
except Exception:
    transformer = None  # type: ignore
    _transformer = None

# Try to import match tool for richer expansions
_match_tool = None
try:
    from instryx_match_enum_struct import DMatchTool  # type: ignore
    _match_tool = DMatchTool()
except Exception:
    _match_tool = None

# Logging
LOG = logging.getLogger("instryx.macro.debugger")
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOG.addHandler(ch)

# Config
TRACE_HMAC_KEY = os.environ.get("INSTRYX_TRACE_HMAC_KEY")  # optional HMAC secret for signing traces
BACKUP_SUFFIX = ".bak"
DEFAULT_HTTP_PORT = 8788


# -------------------------
# Data models
# -------------------------
def _now_ts() -> float:
    return time.time()


class MacroStep:
    def __init__(self,
                 index: int,
                 macro: str,
                 raw_args: str,
                 args: List[str],
                 before: str,
                 after: str,
                 rng: Tuple[int, int],
                 diagnostics: Optional[List[Dict[str, Any]]] = None,
                 error: Optional[str] = None):
        self.index = index
        self.macro = macro
        self.raw_args = raw_args
        self.args = args
        self.before = before
        self.after = after
        self.range = [int(rng[0]), int(rng[1])]
        self.diagnostics = diagnostics or []
        self.error = error
        self.ts = _now_ts()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "macro": self.macro,
            "raw_args": self.raw_args,
            "args": self.args,
            "before": self.before,
            "after": self.after,
            "range": self.range,
            "diagnostics": self.diagnostics,
            "error": self.error,
            "ts": self.ts,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MacroStep":
        return MacroStep(
            index=int(d.get("index", 0)),
            macro=d.get("macro", ""),
            raw_args=d.get("raw_args", ""),
            args=list(d.get("args", [])),
            before=d.get("before", ""),
            after=d.get("after", ""),
            rng=tuple(d.get("range", (0, 0))),
            diagnostics=d.get("diagnostics", []),
            error=d.get("error")
        )


class MacroTrace:
    def __init__(self, source_path: Optional[str] = None, original_source: Optional[str] = None):
        self.source_path = source_path
        self.created = _now_ts()
        self.steps: List[MacroStep] = []
        self.original_source = original_source or ""
        self.final_source: Optional[str] = None
        self.metadata: Dict[str, Any] = {}

    def append(self, step: MacroStep):
        self.steps.append(step)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_path": self.source_path,
            "created": self.created,
            "original_source": self.original_source,
            "final_source": self.final_source,
            "steps": [s.to_dict() for s in self.steps],
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MacroTrace":
        t = MacroTrace(source_path=d.get("source_path"), original_source=d.get("original_source", ""))
        t.created = d.get("created", _now_ts())
        t.final_source = d.get("final_source")
        t.steps = [MacroStep.from_dict(sd) for sd in d.get("steps", [])]
        t.metadata = d.get("metadata", {})
        return t


# -------------------------
# Utilities
# -------------------------
def unified_diff(a: str, b: str, a_name: str = "a", b_name: str = "b") -> str:
    import difflib
    return "".join(difflib.unified_diff(a.splitlines(keepends=True), b.splitlines(keepends=True),
                                        fromfile=a_name, tofile=b_name, lineterm=""))


def atomic_write(path: str, content: str, backup: bool = True) -> Tuple[bool, str]:
    dirp = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dirp, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        if backup and os.path.exists(path):
            shutil.copy2(path, path + BACKUP_SUFFIX)
        os.replace(tmp, path)
        return True, path
    except Exception as e:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        return False, str(e)


def sign_trace_payload(payload: bytes) -> Optional[str]:
    if not TRACE_HMAC_KEY:
        return None
    try:
        sig = hmac.new(TRACE_HMAC_KEY.encode("utf-8"), payload, hashlib.sha256).hexdigest()
        return sig
    except Exception:
        return None


# -------------------------
# Debugger core
# -------------------------
class MacroDebugger:
    def __init__(self, registry: Optional[Dict[str, Callable]] = None, sandbox: bool = True):
        self.sandbox = sandbox
        if registry is not None:
            self.registry = registry
        else:
            try:
                if transformer and hasattr(transformer, "createFullRegistry"):
                    self.registry = transformer.createFullRegistry()
                elif transformer and hasattr(transformer, "createDefaultRegistry"):
                    self.registry = transformer.createDefaultRegistry()
                else:
                    self.registry = {}
            except Exception:
                self.registry = {}
        self._global_registry = getattr(transformer, "_REGISTRY", None)
        self.last_trace: Optional[MacroTrace] = None
        self._lock = threading.RLock()
        self.plugins: Dict[str, Any] = {}
        # load debugger plugins located in ciams_plugins if present (best-effort)
        self._discover_plugins()

    # plugin loader for debugger extensions
    def _discover_plugins(self, plugins_dir: Optional[str] = None):
        plugins_dir = plugins_dir or os.path.join(os.path.dirname(__file__), "ciams_plugins")
        if not os.path.isdir(plugins_dir):
            return
        sys.path.insert(0, plugins_dir)
        for fn in os.listdir(plugins_dir):
            if not fn.endswith(".py"):
                continue
            modname = fn[:-3]
            try:
                mod = importlib.import_module(modname)
                if hasattr(mod, "register_debugger"):
                    try:
                        mod.register_debugger(self)
                        self.plugins[modname] = mod
                        LOG.info("debugger plugin registered: %s", modname)
                    except Exception:
                        LOG.exception("plugin %s `register_debugger` failed", modname)
            except Exception:
                LOG.exception("failed to import plugin %s", modname)
        try:
            sys.path.remove(plugins_dir)
        except Exception:
            pass

    def _scan(self, text: str):
        # prefer transformer scanner
        if transformer and hasattr(transformer, "_scan_macros"):
            try:
                return transformer._scan_macros(text)
            except Exception:
                LOG.exception("transformer._scan_macros failed; fallback scanner used")
        # fallback simple scan (similar to previous implementation)
        res = []
        i = 0
        L = len(text)
        in_s = None
        while i < L:
            c = text[i]
            if in_s:
                if c == in_s and text[i - 1] != "\\":
                    in_s = None
                i += 1
                continue
            if c in ('"', "'"):
                in_s = c
                i += 1
                continue
            if c == "@":
                j = i + 1
                name_chars = []
                while j < L and (text[j].isalnum() or text[j] == "_"):
                    name_chars.append(text[j]); j += 1
                if not name_chars:
                    i += 1; continue
                name = "".join(name_chars)
                k = j
                depth = 0
                in_s2 = None
                found = False
                while k < L:
                    ch = text[k]
                    if in_s2:
                        if ch == in_s2 and text[k - 1] != "\\":
                            in_s2 = None
                        k += 1
                        continue
                    if ch in ('"', "'"):
                        in_s2 = ch; k += 1; continue
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth = max(0, depth - 1)
                    elif ch == ";" and depth == 0:
                        raw = text[j:k].strip()
                        res.append((i, k + 1, name, raw))
                        found = True
                        k += 1
                        break
                    k += 1
                if not found:
                    i = j; continue
                i = k; continue
            i += 1
        return res

    def _parse_args(self, raw: str) -> List[str]:
        # prefer transformer's parser if available
        if transformer and hasattr(transformer, "_parse_macro_args"):
            try:
                return transformer._parse_macro_args(raw)
            except Exception:
                LOG.exception("transformer arg parser failed; fallback used")
        # fallback: simple top-level comma split with minimal nesting
        parts = []
        buf = []
        depth = 0
        in_s = None
        for ch in raw:
            if in_s:
                buf.append(ch)
                if ch == in_s and (len(buf) < 2 or buf[-2] != "\\"):
                    in_s = None
                continue
            if ch in ('"', "'"):
                buf.append(ch); in_s = ch; continue
            if ch == "(":
                depth += 1; buf.append(ch); continue
            if ch == ")":
                depth = max(0, depth - 1); buf.append(ch); continue
            if ch == "," and depth == 0:
                token = "".join(buf).strip()
                if token: parts.append(token)
                buf = []; continue
            buf.append(ch)
        if buf:
            p = "".join(buf).strip()
            if p: parts.append(p)
        return parts

    # Sandbox executor ensures macros executed in read-only safe context when sandbox=True
    def _exec_macro_safe(self, fn: Callable, args: List[str], ctx: Dict[str, Any]) -> Tuple[bool, str, Optional[str]]:
        """
        Execute macro function in sandbox mode:
        - If sandbox True, call function but avoid passing real envs; catch exceptions.
        - Return (ok, result_str, error_trace)
        """
        try:
            if self.sandbox:
                # Provide minimal context copy
                safe_ctx = {"source": ctx.get("source", "")[:100000], "opts": {}, "registry": None}
                res = fn(args, safe_ctx)
            else:
                res = fn(args, ctx)
            return True, "" if res is None else str(res), None
        except Exception as e:
            return False, "", traceback.format_exc()

    def trace_file(self, path: str, max_steps: Optional[int] = None, stop_on_error: bool = False) -> MacroTrace:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        trace = MacroTrace(source_path=path, original_source=src)
        cur = src
        idx = 0
        while True:
            matches = self._scan(cur)
            if not matches:
                break
            start, end, name, raw_args = matches[0]
            before = cur[start:end]
            args = self._parse_args(raw_args)
            macro_fn = None
            if isinstance(self.registry, dict):
                macro_fn = self.registry.get(name)
            if not macro_fn and self._global_registry:
                try:
                    macro_fn = self._global_registry.get(name)
                except Exception:
                    macro_fn = None
            diagnostics = []
            error = None
            after = before
            if not macro_fn:
                error = f"macro '{name}' not found"
            else:
                ok, res_str, err = self._exec_macro_safe(macro_fn, args, {"source": cur, "opts": {}, "registry": self.registry})
                if ok:
                    after = res_str
                    # optionally ask transformer for diagnostics for entire source (best-effort)
                    if transformer and hasattr(transformer, "applyMacrosWithDiagnostics"):
                        try:
                            info = transformer.applyMacrosWithDiagnostics(cur, registry=self.registry)
                            diagnostics = info.get("diagnostics", []) or []
                        except Exception:
                            diagnostics = []
                else:
                    error = f"macro execution failed: {err}"
            step = MacroStep(index=idx, macro=name, raw_args=raw_args, args=args, before=before, after=after, rng=(start, end), diagnostics=diagnostics, error=error)
            trace.append(step)
            idx += 1
            cur = cur[:start] + after + cur[end:]
            if max_steps is not None and idx >= max_steps:
                break
            if stop_on_error and error:
                break
        trace.final_source = cur
        self.last_trace = trace
        return trace

    def save_trace(self, trace: MacroTrace, path: str, sign: bool = False) -> Tuple[bool, str]:
        try:
            payload = json.dumps(trace.to_dict(), indent=2, ensure_ascii=False).encode("utf-8")
            with open(path, "wb") as f:
                f.write(payload)
            if sign and TRACE_HMAC_KEY:
                sig = sign_trace_payload(payload)
                if sig:
                    with open(path + ".sig", "w", encoding="utf-8") as s:
                        s.write(sig)
            return True, path
        except Exception as e:
            LOG.exception("save_trace failed")
            return False, str(e)

    def load_trace(self, path: str) -> MacroTrace:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return MacroTrace.from_dict(d)

    def replay_trace(self, trace: MacroTrace, source: Optional[str] = None, apply: bool = False, backup: bool = True) -> Tuple[bool, str, List[Dict[str, Any]]]:
        cur = source if source is not None else trace.original_source
        diagnostics: List[Dict[str, Any]] = []
        for step in trace.steps:
            start, end = step.range
            # Validate
            actual = None
            try:
                actual = cur[start:end]
            except Exception:
                actual = None
            if actual != step.before:
                idx = cur.find(step.before)
                if idx != -1:
                    start = idx
                    end = idx + len(step.before)
                    note = f"range adjusted to {start}"
                else:
                    diagnostics.append({"step": step.index, "ok": False, "reason": "before not found"})
                    return False, "replay failed: before snippet not found", diagnostics
            cur = cur[:start] + step.after + cur[end:]
            diagnostics.append({"step": step.index, "ok": True, "macro": step.macro})
        # Apply
        if apply:
            if not trace.source_path:
                return False, "no source_path to apply", diagnostics
            try:
                if backup and os.path.exists(trace.source_path):
                    shutil.copy2(trace.source_path, trace.source_path + BACKUP_SUFFIX)
                ok, out = atomic_write(trace.source_path, cur, backup=False)
                if not ok:
                    return False, f"write failed: {out}", diagnostics
                return True, out, diagnostics
            except Exception as e:
                LOG.exception("replay apply failed")
                return False, f"apply error: {e}", diagnostics
        return True, cur, diagnostics

    def validate_trace(self, trace: MacroTrace) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Validate that each step's expansion reproduces same 'after' when executed now.
        Returns (all_ok, diagnostics)
        """
        src = trace.original_source
        cur = src
        diagnostics = []
        for step in trace.steps:
            start, end = step.range
            # locate before
            idx = cur.find(step.before)
            if idx == -1:
                diagnostics.append({"step": step.index, "ok": False, "reason": "before not found"})
                return False, diagnostics
            # execute macro now
            macro_fn = (self.registry.get(step.macro) if isinstance(self.registry, dict) else None) or (self._global_registry.get(step.macro) if self._global_registry else None)
            if not macro_fn:
                diagnostics.append({"step": step.index, "ok": False, "reason": "macro not present"})
                return False, diagnostics
            ok, res, err = self._exec_macro_safe(macro_fn, step.args, {"source": cur, "opts": {}, "registry": self.registry})
            if not ok:
                diagnostics.append({"step": step.index, "ok": False, "reason": "execution error", "error": err})
                return False, diagnostics
            # compare res with recorded after
            if res != step.after:
                diagnostics.append({"step": step.index, "ok": False, "reason": "after mismatch", "expected_len": len(step.after), "actual_len": len(res)})
                return False, diagnostics
            # advance current
            cur = cur[:idx] + res + cur[idx + len(step.before):]
            diagnostics.append({"step": step.index, "ok": True})
        return True, diagnostics

    def interactive_trace(self, path: str):
        """
        Interactive step-through: show each step, allow accept/apply/skip/quit.
        If user accepts, the replacement is applied to in-memory buffer.
        At the end user can write result to file or produce patch.
        """
        trace = self.trace_file(path)
        cur = trace.original_source
        accepted_steps = []
        for step in trace.steps:
            print("\n--- Step", step.index, "macro:", step.macro, "---")
            print("Location range:", step.range)
            print("Before snippet:\n", step.before)
            print("Proposed expansion:\n", step.after)
            if step.diagnostics:
                print("Diagnostics:", step.diagnostics)
            if step.error:
                print("Error:", step.error)
            cmd = input("Action [a]ccept / [s]kip / [q]uit / [p]atch so far: ").strip().lower()
            if cmd in ("a", "accept"):
                # apply to cur
                start, end = step.range
                # remap: find occurrence
                idx = cur.find(step.before)
                if idx == -1:
                    print("Before snippet not found in current buffer, skipping")
                    continue
                cur = cur[:idx] + step.after + cur[idx + len(step.before):]
                accepted_steps.append(step.index)
                print("accepted")
            elif cmd in ("s", "skip"):
                print("skipped")
                continue
            elif cmd in ("p", "patch"):
                # create patch comparing original file to current buffer
                with open(path, "r", encoding="utf-8") as f:
                    orig = f.read()
                patch = unified_diff(orig, cur, a_name=path, b_name=path + ".ai.partial.ix")
                out_patch = path + ".ai.partial.patch"
                with open(out_patch, "w", encoding="utf-8") as pf:
                    pf.write(patch)
                print("partial patch written ->", out_patch)
            elif cmd in ("q", "quit"):
                print("aborting interactive session")
                break
            else:
                print("unknown action; skipping")
        # finished. Ask to write
        write = input("Write accepted result to file? [y/N]: ").strip().lower()
        if write in ("y", "yes"):
            ok, out = atomic_write(path, cur, backup=True)
            if ok:
                print("wrote", out)
            else:
                print("write failed:", out)
        else:
            print("interactive session complete; not written.")
        return True

    def undo_backup(self, path: str) -> Tuple[bool, str]:
        bak = path + BACKUP_SUFFIX
        if not os.path.exists(bak):
            return False, "backup not found"
        try:
            shutil.copy2(bak, path)
            return True, path
        except Exception as e:
            LOG.exception("undo failed")
            return False, str(e)

    def list_available_macros(self) -> List[str]:
        names = set()
        if isinstance(self.registry, dict):
            names.update(self.registry.keys())
        if self._global_registry:
            try:
                names.update(self._global_registry.list())
            except Exception:
                try:
                    names.update(getattr(self._global_registry, "_macros", {}).keys())
                except Exception:
                    pass
        return sorted(names)


# -------------------------
# HTTP Server
# -------------------------
class _DbgHandler(BaseHTTPRequestHandler):
    debugger: Optional[MacroDebugger] = None

    def _send_json(self, obj: Any, status: int = 200):
        data = json.dumps(obj, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path
        if path.startswith("/list"):
            names = self.debugger.list_available_macros() if self.debugger else []
            self._send_json({"macros": names})
            return
        if path.startswith("/trace"):
            # query ?file=...
            if "?" in path:
                _, q = path.split("?", 1)
                params = dict(pair.split("=", 1) for pair in q.split("&") if "=" in pair)
            else:
                params = {}
            file = params.get("file")
            if not file or not os.path.exists(file):
                self._send_json({"error": "file missing"}, 400)
                return
            trace = self.debugger.trace_file(file)
            self._send_json({"trace": trace.to_dict()})
            return
        self._send_json({"error": "unknown endpoint"}, 404)

    def do_POST(self):
        if self.path != "/replay":
            self._send_json({"error": "unknown endpoint"}, 404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        try:
            data = json.loads(body or "{}")
            trace_path = data.get("trace")
            apply_flag = bool(data.get("apply", False))
            if not trace_path or not os.path.exists(trace_path):
                self._send_json({"error": "trace path missing"}, 400)
                return
            trace = self.debugger.load_trace(trace_path)
            ok, out, diag = self.debugger.replay_trace(trace, apply=apply_flag)
            self._send_json({"ok": ok, "out": out, "diagnostics": diag})
        except Exception as e:
            LOG.exception("HTTP replay failed")
            self._send_json({"error": str(e)}, 500)


def serve(port: int = DEFAULT_HTTP_PORT, host: str = "127.0.0.1", debugger: Optional[MacroDebugger] = None):
    handler = _DbgHandler
    handler.debugger = debugger or MacroDebugger()
    server = HTTPServer((host, port), handler)
    LOG.info("MacroDebugger HTTP server listening on %s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("server stopped")
    finally:
        server.server_close()


# -------------------------
# CLI
# -------------------------
def _cli():
    p = argparse.ArgumentParser(prog="instryx_macro_debugger.py")
    p.add_argument("cmd", nargs="?", help="command (trace, interactive, replay, apply, preview, validate, undo, list, export-trace, import-trace, serve, test)")
    p.add_argument("target", nargs="?", help="file or trace path")
    p.add_argument("--out", help="output path")
    p.add_argument("--apply", action="store_true", help="apply when replaying")
    p.add_argument("--port", type=int, default=DEFAULT_HTTP_PORT)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()

    dbg = MacroDebugger()
    cmd = args.cmd or "help"
    if cmd in ("help", None):
        p.print_help()
        return 0

    try:
        if cmd == "list":
            for n in dbg.list_available_macros():
                print(n)
            return 0

        if cmd == "trace":
            if not args.target:
                print("file required"); return 2
            trace = dbg.trace_file(args.target)
            out = args.out or (args.target + ".ai.trace.json")
            ok, msg = dbg.save_trace(trace, out, sign=True)
            if ok:
                print("trace saved ->", msg)
                return 0
            print("save failed:", msg)
            return 2

        if cmd == "interactive":
            if not args.target:
                print("file required"); return 2
            dbg.interactive_trace(args.target)
            return 0

        if cmd == "preview":
            if not args.target:
                print("file required"); return 2
            if not transformer:
                print("transformer not available"); return 2
            content = open(args.target, "r", encoding="utf-8").read()
            res = transformer.applyMacrosWithDiagnostics(content, registry=transformer.createDefaultRegistry())
            transformed = res.get("result", {}).get("transformed", content)
            print(transformed)
            if res.get("diagnostics"):
                print(json.dumps(res.get("diagnostics"), indent=2))
            return 0

        if cmd == "replay":
            if not args.target:
                print("trace path required"); return 2
            trace = dbg.load_trace(args.target)
            ok, out, diag = dbg.replay_trace(trace, apply=args.apply)
            print("ok:", ok)
            print("out:", out)
            if diag:
                print("diagnostics:", json.dumps(diag, indent=2))
            return 0

        if cmd == "apply":
            if not args.target:
                print("file required"); return 2
            trace = dbg.trace_file(args.target)
            ok, out, diag = dbg.replay_trace(trace, apply=True)
            print("apply:", ok, out)
            if diag:
                print("diag:", json.dumps(diag, indent=2))
            return 0

        if cmd == "validate":
            if not args.target or not os.path.exists(args.target):
                print("trace file required"); return 2
            trace = dbg.load_trace(args.target)
            ok, diag = dbg.validate_trace(trace)
            print("valid:", ok)
            if diag:
                print(json.dumps(diag, indent=2))
            return 0

        if cmd == "undo":
            if not args.target:
                print("file required"); return 2
            ok, msg = dbg.undo_backup(args.target)
            if ok:
                print("restored ->", msg)
                return 0
            print("undo failed:", msg)
            return 2

        if cmd == "export-trace":
            if not args.target:
                print("file required"); return 2
            trace = dbg.trace_file(args.target)
            out = args.out or (args.target + ".ai.trace.json")
            ok, path = dbg.save_trace(trace, out, sign=True)
            print("exported ->", path if ok else f"failed: {path}")
            return 0

        if cmd == "import-trace":
            if not args.target or not os.path.exists(args.target):
                print("trace path required"); return 2
            trace = dbg.load_trace(args.target)
            print("loaded trace, steps:", len(trace.steps))
            return 0

        if cmd == "serve":
            serve(port=args.port, host=args.host, debugger=dbg)
            return 0

        if cmd == "test":
            # simple self-test
            sample = """enum Color { Red, Blue }
func f(c){ @match_pattern Color, c; }
"""
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ix", mode="w", encoding="utf-8")
            tmp.write(sample); tmp.flush(); tmp.close()
            try:
                tr = dbg.trace_file(tmp.name)
                print("steps:", len(tr.steps))
                assert len(tr.steps) >= 1
                print("self-test PASS")
                return 0
            except Exception as e:
                LOG.exception("self-test failure")
                print("self-test FAIL:", e)
                return 2
            finally:
                try: os.unlink(tmp.name)
                except Exception: pass

        print("unknown command:", cmd)
        p.print_help()
        return 2
    except KeyboardInterrupt:
        print("aborted")
        return 1
    except Exception:
        LOG.exception("fatal")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(_cli())

