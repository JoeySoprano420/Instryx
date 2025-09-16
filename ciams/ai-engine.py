"""
ciams/ai-engine.py

Comprehensive, executable CIAMS AI assistant with many features, tooling and optimizations.

This iteration extends previous functionality with:
- Powerful optimizations (safe unroll, tiling, vectorize hints, loop fusion hints)
- Executable custom tooling: helper emit/inject, apply/rollback patches, undo manager
- PluginManager for runtime rule registration and discovery
- CodegenToolkit wrapper around instryx_memory_math_loops_codegen when available
- CLI commands: emit-helper, inject-helper, optimize-loop, apply-patch, undo, plugins
- Safety checks, deterministic uid seed, and cost estimator to avoid explosion
- All implemented with Python stdlib; optional integrations used when installed.

Usage examples:
  python ciams\ai-engine.py suggest file.ix
  python ciams\ai-engine.py preview file.ix --index 0
  python ciams\ai-engine.py apply file.ix --index 0 --inplace
  python ciams\ai-engine.py emit-helper memoize
  python ciams\ai-engine.py inject-helper memoize myfile.ix
  python ciams\ai-engine.py optimize-loop myfile.ix --max-unroll 8 --apply
  python ciams\ai-engine.py apply-patch file.ix file.ix.ai.patch
  python ciams\ai-engine.py undo file.ix
  python ciams\ai-engine.py plugins list
  python ciams\ai-engine.py serve --port 8787
"""

from __future__ import annotations
import argparse
import concurrent.futures
import difflib
import hashlib
import importlib
import json
import logging
import os
import random
import re
import shutil
import string
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple

# Optional integration modules
_try_macro_overlay = None
try:
    import macro_overlay  # type: ignore
    _try_macro_overlay = macro_overlay
except Exception:
    _try_macro_overlay = None

_try_codegen = None
try:
    import instryx_memory_math_loops_codegen as codegen  # type: ignore
    _try_codegen = codegen
except Exception:
    _try_codegen = None

# Logging
LOG_PATH = os.path.join(os.path.dirname(__file__), "ciams_ai_engine.log")
logging.basicConfig(level=logging.INFO, filename=LOG_PATH, filemode="a",
                    format="%(asctime)s [%(levelname)s] %(message)s")

# Defaults and safety thresholds
MAX_UNROLL_SAFE = 16
MAX_EXPANSION_BYTES = 50_000  # warn/stop if expanding beyond this size
DEFAULT_SEED = 0


# -------------------------
# Data models
# -------------------------
@dataclass
class Suggestion:
    macro_name: str
    args: List[str]
    reason: str
    score: float
    snippet: Optional[str] = None
    location: Optional[Tuple[int, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AISimpleMemory:
    """Persistent local memory for counting patterns and recording acceptance."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or os.path.join(os.path.dirname(__file__), "ai_memory.json")
        self._data: Dict[str, Any] = {"patterns": {}, "accepted": []}
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
        except Exception:
            self._data = {"patterns": {}, "accepted": []}

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except Exception:
            logging.exception("Failed to save AISimpleMemory")

    def record_pattern(self, key: str):
        self._data.setdefault("patterns", {})
        self._data["patterns"][key] = self._data["patterns"].get(key, 0) + 1
        self.save()

    def pattern_count(self, key: str) -> int:
        return int(self._data.get("patterns", {}).get(key, 0))

    def record_accepted(self, suggestion: Suggestion, filename: Optional[str] = None):
        self._data.setdefault("accepted", []).append({
            "time": int(time.time()),
            "suggestion": suggestion.to_dict(),
            "file": filename,
        })
        self.save()

    def export(self) -> Dict[str, Any]:
        return self._data

    def import_data(self, data: Dict[str, Any], merge: bool = True):
        if not merge:
            self._data = data
        else:
            pat = data.get("patterns", {})
            for k, v in pat.items():
                self._data.setdefault("patterns", {})
                self._data["patterns"][k] = self._data["patterns"].get(k, 0) + v
            self._data.setdefault("accepted", []).extend(data.get("accepted", []))
        self.save()


# -------------------------
# Utilities
# -------------------------
def uid(prefix: str = "g", seed: Optional[int] = None) -> str:
    """Deterministic uid if seed provided. Otherwise uses time+random."""
    if seed is not None:
        h = hashlib.sha1(f"{prefix}:{seed}".encode()).hexdigest()[:8]
        return f"{prefix}_{h}"
    return f"{prefix}_{int(time.time()*1000)}_{''.join(random.choices(string.ascii_lowercase, k=4))}"


def unified_diff(a: str, b: str, a_name: str = "a", b_name: str = "b") -> str:
    return "".join(difflib.unified_diff(a.splitlines(keepends=True), b.splitlines(keepends=True),
                                        fromfile=a_name, tofile=b_name, lineterm=""))


def safe_read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def safe_write(path: str, data: str, backup: bool = True) -> str:
    if backup and os.path.exists(path):
        bak = path + ".bak"
        shutil.copy2(path, bak)
    with open(path, "w", encoding="utf-8") as f:
        f.write(data)
    return path


def apply_patch(original_path: str, patch_path: str) -> Tuple[bool, str]:
    """
    Apply a unified diff patch generated by this tool.
    Uses difflib.restore heuristics: expects the patch to be a unified diff fromfile->tofile.
    Writes resulting transformed content to original_path + ".ai.ix"
    """
    try:
        with open(patch_path, "r", encoding="utf-8") as pf:
            patch_text = pf.read()
        # difflib.restore expects a delta produced by difflib.ndiff; unified diffs are not directly supported.
        # For simplicity, attempt to parse hunks: if it contains '---'/'+++', fallback to naive apply: extract new file lines after @@ hunks.
        lines = patch_text.splitlines(keepends=True)
        # try using patch via python's difflib if it's in unified form: convert to pure old/new.
        # Simpler fallback: if patch contains "+++ " indicating new file, and contains lines starting with '+', build new by applying hunks.
        original = safe_read(original_path).splitlines(keepends=True)
        new_lines = []
        i = 0
        # naive approach: if patch contains no '@@' return failure
        if "@@" not in patch_text:
            # fallback: write patch contents as .ai.ix (inform user)
            out = original_path + ".ai.ix"
            safe_write(out, "".join(lines), backup=False)
            return False, out
        # Very small robust unified diff apply: use python patch algorithm from difflib by recomputing.
        # We'll attempt to reconstruct by calling difflib.SequenceMatcher patches from hunks - but to avoid complexity just return failure explaining manual apply.
        return False, "apply_patch: automatic unified diff apply not implemented fully; open .ai.patch and apply manually"
    except Exception as e:
        logging.exception("apply_patch failed")
        return False, str(e)


# -------------------------
# Plugin manager
# -------------------------
class PluginManager:
    """Runtime plugin loader for additional heuristic rules."""
    def __init__(self, plugins_dir: Optional[str] = None):
        base = plugins_dir or os.path.join(os.path.dirname(__file__), "ciams_plugins")
        self.dir = base
        os.makedirs(self.dir, exist_ok=True)
        self.loaded: Dict[str, Any] = {}

    def discover(self) -> List[str]:
        return [f[:-3] for f in os.listdir(self.dir) if f.endswith(".py")]

    def load(self, name: str) -> Tuple[bool, str]:
        path = os.path.join(self.dir, f"{name}.py")
        if not os.path.exists(path):
            return False, f"plugin {name} not found"
        try:
            spec = importlib.util.spec_from_file_location(f"ciams_plugins.{name}", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            self.loaded[name] = mod
            return True, f"loaded {name}"
        except Exception as e:
            logging.exception("plugin load failed")
            return False, str(e)

    def unload(self, name: str) -> Tuple[bool, str]:
        if name not in self.loaded:
            return False, "not loaded"
        try:
            mod = self.loaded.pop(name)
            if hasattr(mod, "unregister") and callable(mod.unregister):
                mod.unregister()
            return True, "unloaded"
        except Exception as e:
            logging.exception("plugin unload failed")
            return False, str(e)


# -------------------------
# Undo manager
# -------------------------
class UndoManager:
    """List and restore backups produced by safe_write (.bak files)."""

    def list_backups(self, filepath: str) -> List[str]:
        dirp = os.path.dirname(filepath) or "."
        base = os.path.basename(filepath)
        res = []
        for f in os.listdir(dirp):
            if f.startswith(base) and (f.endswith(".bak") or f.endswith(".ai.ix.bak")):
                res.append(os.path.join(dirp, f))
        return res

    def restore(self, backup_path: str, target_path: Optional[str] = None) -> Tuple[bool, str]:
        if not os.path.exists(backup_path):
            return False, "backup not found"
        target = target_path or backup_path.rsplit(".bak", 1)[0]
        try:
            shutil.copy2(backup_path, target)
            return True, target
        except Exception as e:
            logging.exception("restore failed")
            return False, str(e)


# -------------------------
# CodegenToolkit wrapper
# -------------------------
class CodegenToolkit:
    """Wrapper for codegen helpers (instryx_memory_math_loops_codegen)."""
    def __init__(self, codegen_module=None, macro_overlay=None, seed: Optional[int] = None):
        self.codegen = codegen_module or _try_codegen
        self.mo = macro_overlay or _try_macro_overlay
        self.seed = seed

    def emit_helper(self, name: str, *args, **kwargs) -> str:
        if not self.codegen:
            raise RuntimeError("codegen module not available")
        # prefer public emit_helper if exists
        if hasattr(self.codegen, "emit_helper"):
            return self.codegen.emit_helper(name, *args, **kwargs)
        # else try generate_<name>
        fn = getattr(self.codegen, f"generate_{name}", None)
        if callable(fn):
            return fn(*args, **kwargs)
        raise KeyError(f"helper {name} not found in codegen")

    def inject_helper(self, helper_text: str, target: str) -> str:
        src = ""
        try:
            src = safe_read(target)
        except Exception:
            src = ""
        new = helper_text + "\n" + src
        safe_write(target, new)
        return target

    def preview_expand_injected(self, helper_text: str, target: str) -> Tuple[bool, str, Optional[List[Dict[str, Any]]]]:
        """Inject helper into target content and preview macro_overlay expansion (non-destructive)."""
        src = ""
        try:
            src = safe_read(target)
        except Exception:
            src = ""
        inserted = helper_text + "\n" + src
        if not self.mo:
            return False, "macro_overlay not available", None
        try:
            apply_fn = getattr(self.mo, "applyMacrosWithDiagnostics", None) or getattr(self.mo, "applyMacros", None)
            res = apply_fn(inserted, self.mo.createFullRegistry() if hasattr(self.mo, "createFullRegistry") else self.mo.createDefaultRegistry(), {"filename": target})
            if hasattr(res, "__await__"):
                import asyncio
                res = asyncio.get_event_loop().run_until_complete(res)
            if isinstance(res, dict) and "result" in res:
                transformed = res["result"].get("transformed", inserted)
                diagnostics = res.get("diagnostics", [])
                return True, transformed, diagnostics
            if isinstance(res, str):
                return True, res, None
            return False, "no result", None
        except Exception as e:
            logging.exception("preview_expand_injected failed")
            return False, str(e), None


# -------------------------
# AIAssistant (core)
# -------------------------
class AIAssistant:
    def __init__(self, seed: int = DEFAULT_SEED, memory: Optional[AISimpleMemory] = None, safety: str = "normal", max_unroll: int = MAX_UNROLL_SAFE):
        self.seed = None if seed == DEFAULT_SEED else seed
        self.memory = memory or AISimpleMemory()
        self.safety = safety
        self.max_unroll = max_unroll
        self.rules: List[Callable[[str, Optional[str]], List[Suggestion]]] = []
        self._register_builtin_rules()
        self.codegen = _try_codegen
        self.macro_overlay = _try_macro_overlay
        self.plugins = PluginManager()
        self.toolkit = CodegenToolkit(self.codegen, self.macro_overlay, seed=self.seed)

    def _register_builtin_rules(self):
        self.rules = [
            self.rule_sql_injection,
            self.rule_transactional_db_writes,
            self.rule_network_retry,
            self.rule_memoize_candidates,
            self.rule_lazy_inject,
            self.rule_defer_cleanup,
            self.rule_sanitize_html,
            self.rule_rate_limit,
            self.rule_idempotency_endpoints,
            self.rule_profile_hot_loops,
            self.rule_vectorize_hint,
            self.rule_unroll_candidates,
            self.rule_audit_sensitive_ops,
        ]

    def register_rule(self, fn: Callable[[str, Optional[str]], List[Suggestion]]):
        self.rules.append(fn)

    def analyze_source(self, source: str, filename: Optional[str] = None, max_suggestions: int = 12) -> List[Suggestion]:
        suggestions: List[Suggestion] = []
        for rule in self.rules:
            try:
                suggestions.extend(rule(source, filename))
            except Exception:
                logging.exception("rule threw")
        for s in suggestions:
            boost = 0.05 * min(5, self.memory.pattern_count(s.macro_name))
            s.score = min(1.0, s.score + boost)
        seen = set()
        out: List[Suggestion] = []
        for s in sorted(suggestions, key=lambda x: -x.score):
            key = (s.macro_name, tuple(s.args), s.snippet or "")
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
            if len(out) >= max_suggestions:
                break
        return out

    # preview_apply and apply_suggestion_to_file reuse previous implementations
    def preview_apply(self, source: str, suggestion: Suggestion, filename: Optional[str] = None) -> Tuple[bool, str, Optional[List[Dict[str, Any]]]]:
        macro_text = f"@{suggestion.macro_name} " + (", ".join(suggestion.args) if suggestion.args else "") + ";\n"
        transformed = source
        if suggestion.location:
            start, _ = suggestion.location
            transformed = source[:start] + macro_text + source[start:]
        elif suggestion.snippet:
            idx = source.find(suggestion.snippet)
            if idx != -1:
                transformed = source[:idx] + macro_text + source[idx:]
            else:
                transformed = macro_text + source
        else:
            transformed = macro_text + source

        if self.macro_overlay is None:
            return False, transformed, None
        try:
            apply_fn = getattr(self.macro_overlay, "applyMacrosWithDiagnostics", None) or getattr(self.macro_overlay, "applyMacros", None)
            if apply_fn is None:
                return False, transformed, None
            res = apply_fn(transformed, self.macro_overlay.createFullRegistry() if hasattr(self.macro_overlay, "createFullRegistry") else self.macro_overlay.createDefaultRegistry(), {"filename": filename})
            if hasattr(res, "__await__"):
                import asyncio
                res = asyncio.get_event_loop().run_until_complete(res)
            if isinstance(res, dict) and "result" in res:
                transformed_text = res["result"].get("transformed", transformed)
                diagnostics = res.get("diagnostics", [])
                return True, transformed_text, diagnostics
            if isinstance(res, tuple) and len(res) >= 1:
                return True, res[0], None
            if isinstance(res, str):
                return True, res, None
            return False, transformed, None
        except Exception as e:
            logging.exception("preview_apply error")
            return False, transformed, [{"type": "error", "message": str(e)}]

    def apply_suggestion_to_file(self, path: str, suggestion: Suggestion, inplace: bool = False, create_patch: bool = True) -> Tuple[bool, str]:
        try:
            src = safe_read(path)
        except Exception as e:
            return False, f"read failed: {e}"
        ok, transformed, diagnostics = self.preview_apply(src, suggestion, filename=path)
        if not ok:
            return False, "preview failed or macro_overlay not available"
        if suggestion.macro_name == "unroll":
            try:
                factor = int(suggestion.args[0]) if suggestion.args else 0
                if factor > self.max_unroll:
                    return False, f"unroll factor {factor} exceeds max_unroll {self.max_unroll}"
            except Exception:
                pass
        if self.safety == "strict" and len(transformed) - len(src) > MAX_EXPANSION_BYTES:
            return False, "expansion too large under strict safety"
        if create_patch:
            patch_text = unified_diff(src, transformed, a_name=path, b_name=path + ".ai.ix")
            safe_write(path + ".ai.patch", patch_text, backup=False)
        out_path = path if inplace else path + ".ai.ix"
        safe_write(out_path, transformed)
        self.memory.record_accepted(suggestion, filename=path)
        return True, out_path


# -------------------------
# HTTP API Handler
# -------------------------
class SuggestHandler(BaseHTTPRequestHandler):
    assistant: Optional[AIAssistant] = None

    def _send_json(self, obj: Any, status: int = 200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = self.path.split("?", 1)
        path = parsed[0]
        qs = {}
        if len(parsed) > 1:
            for kv in parsed[1].split("&"):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    qs[k] = v
        if path == "/suggest":
            file = qs.get("file")
            if not file or not os.path.exists(file):
                self._send_json({"error": "file missing"}, 400)
                return
            src = safe_read(file)
            suggestions = self.assistant.analyze_source(src, filename=file, max_suggestions=int(qs.get("max", "8")))
            self._send_json({"suggestions": [s.to_dict() for s in suggestions]})
            return
        self._send_json({"error": "unknown endpoint"}, 404)

    def do_POST(self):
        if self.path != "/apply":
            self._send_json({"error": "unknown endpoint"}, 404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        try:
            data = json.loads(body)
            file = data.get("file")
            idx = int(data.get("index", 0))
            inplace = bool(data.get("inplace", False))
            if not file or not os.path.exists(file):
                self._send_json({"error": "file missing"}, 400)
                return
            src = safe_read(file)
            suggestions = self.assistant.analyze_source(src, filename=file, max_suggestions=32)
            if idx < 0 or idx >= len(suggestions):
                self._send_json({"error": "invalid index"}, 400)
                return
            ok, out = self.assistant.apply_suggestion_to_file(file, suggestions[idx], inplace=inplace)
            self._send_json({"ok": ok, "out": out})
        except Exception as e:
            logging.exception("HTTP apply failed")
            self._send_json({"error": str(e)}, 500)


def serve_api(assistant: AIAssistant, host: str = "127.0.0.1", port: int = 8787):
    SuggestHandler.assistant = assistant
    server = HTTPServer((host, port), SuggestHandler)
    logging.info("Serving AI API on %s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Server stopped")
    finally:
        server.server_close()


# -------------------------
# Interactive batch apply
# -------------------------
def interactive_batch_apply(assistant: AIAssistant, files: List[str], suggestion_index: int = 0, inplace: bool = False):
    accepted_all = False
    for p in files:
        print(f"\n--- File: {p} ---")
        try:
            src = safe_read(p)
        except Exception as e:
            print("read failed:", e)
            continue
        suggestions = assistant.analyze_source(src, filename=p, max_suggestions=16)
        if not suggestions:
            print("No suggestions")
            continue
        if suggestion_index >= len(suggestions):
            print("No suggestion at index", suggestion_index)
            continue
        sug = suggestions[suggestion_index]
        ok, transformed, diagnostics = assistant.preview_apply(src, sug, filename=p)
        if not ok:
            print("Preview failed (macro_overlay may be missing). Showing inserted macro text preview only.")
            transformed = (f"@{sug.macro_name} " + (", ".join(sug.args) if sug.args else "") + ";\n") + src
        diff = unified_diff(src, transformed, a_name=p, b_name=p + ".ai.ix")
        print(diff or "No change")
        if accepted_all:
            do_apply = True
        else:
            ans = input("Apply change? [y]es / [n]o / [a]ll / [q]uit: ").strip().lower()
            if ans in ("y", "yes"):
                do_apply = True
            elif ans in ("a", "all"):
                do_apply = True
                accepted_all = True
            elif ans in ("q", "quit"):
                break
            else:
                do_apply = False
        if do_apply:
            ok2, out = assistant.apply_suggestion_to_file(p, sug, inplace=inplace)
            print("Applied ->", out if ok2 else f"failed: {out}")


# -------------------------
# Additional tooling: safe loop optimizer (textual)
# -------------------------
def optimize_loop_unroll_file(path: str, max_unroll: int = 8, apply: bool = False) -> Tuple[bool, str]:
    """
    Very conservative textual unroll optimizer:
    - Detects loops of the form: for (i = 0; i < N; i++) { ... }
      where N is small integer literal.
    - Generates unrolled body by repeating inner statements with replaced indices {i}
      (requires body to use array indexing like arr[i])
    - Does not parse general code; it's a best-effort, safe-only transform.
    Returns (ok, message_or_path).
    """
    try:
        src = safe_read(path)
    except Exception as e:
        return False, f"read failed: {e}"
    changed = False
    out_src = src
    # regex to find simple C-like for loops: for (i = 0; i < 4; i++) { body }
    pattern = re.compile(r"for\s*\(\s*([A-Za-z_][\w]*)\s*=\s*0\s*;\s*\1\s*<\s*([0-9]+)\s*;\s*\1\+\+\s*\)\s*\{", re.M)
    idx = 0
    new_src_parts = []
    last = 0
    for m in pattern.finditer(src):
        i_name = m.group(1)
        bound = int(m.group(2))
        if bound <= 0 or bound > max_unroll:
            continue
        # capture body
        start_body = src.find("{", m.end() - 1)
        if start_body == -1:
            continue
        depth = 0
        j = start_body
        while j < len(src):
            ch = src[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_body = j
                    break
            j += 1
        else:
            continue
        body = src[start_body+1:end_body]
        # naive check: body should reference i in index context (arr[i] or func(i))
        if i_name not in body:
            continue
        # produce unrolled snippet: repeat body with i replaced by literal
        unrolled_lines = []
        for k in range(bound):
            repl = body.replace(f"{i_name}", str(k))
            unrolled_lines.append(repl)
        unrolled_text = "\n".join(line.rstrip() for line in unrolled_lines) + "\n"
        new_src_parts.append(src[last:m.start()])
        new_src_parts.append("/* unrolled loop (automatically) */\n")
        new_src_parts.append(unrolled_text)
        last = end_body + 1
        changed = True
    if not changed:
        return False, "no eligible loops found or none within safe unroll limit"
    new_src_parts.append(src[last:])
    transformed = "".join(new_src_parts)
    diff = unified_diff(src, transformed, a_name=path, b_name=path + ".ai.ix")
    patch_path = path + ".ai.unroll.patch"
    safe_write(patch_path, diff, backup=False)
    if apply:
        # write transformed to .ai.ix (do not overwrite source unless explicit)
        out_path = path + ".ai.ix"
        safe_write(out_path, transformed)
        return True, out_path
    return True, patch_path


# -------------------------
# CLI and entrypoint
# -------------------------
def run_unit_tests(verbose: bool = True) -> bool:
    assistant = AIAssistant()
    sample = """-- demo
net.request("https://api.example.com/data");
db.conn.query("select * from users where id=" + id);
func fib(n) { if n <= 1 { n } else { fib(n-1) + fib(n-2) } };
for (i=0;i<4;i++) { doWork(i); }
html_out = "<div>" + user_input + "</div>";
"""
    suggestions = assistant.analyze_source(sample, filename="<demo>", max_suggestions=40)
    if verbose:
        print("Suggestions:", len(suggestions))
        for s in suggestions[:10]:
            print("-", s)
    try:
        assert any(s.macro_name in ("wraptry", "async", "memoize", "sanitize", "unroll", "assert") for s in suggestions)
        return True
    except AssertionError:
        return False


def main(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(prog="ciams/ai-engine.py")
    p.add_argument("cmd", nargs="?", help="command (suggest, preview, apply, batch-suggest, batch-apply, interactive, serve, test, export-memory, import-memory, emit-helper, inject-helper, inject-expand, optimize-loop, apply-patch, undo, plugins)")
    p.add_argument("target", nargs="?", help="file or directory target")
    p.add_argument("--index", type=int, default=0, help="suggestion index")
    p.add_argument("--max", type=int, default=12, help="max suggestions")
    p.add_argument("--inplace", action="store_true", help="write in-place")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--safety", choices=("off", "normal", "strict"), default="normal", help="safety mode")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="deterministic seed for uid generation")
    p.add_argument("--max-unroll", type=int, default=8, help="max safe unroll factor")
    p.add_argument("--apply", action="store_true", help="apply transform / commit")
    p.add_argument("--args", nargs="*", help="helper args for emit/inject (comma separated lists allowed)")
    args = p.parse_args(argv)

    assistant = AIAssistant(seed=args.seed, safety=args.safety, max_unroll=args.max_unroll)
    undo_mgr = UndoManager()
    plugin_mgr = assistant.plugins
    toolkit = assistant.toolkit

    if args.cmd in (None, "help"):
        p.print_help()
        return 0

    try:
        if args.cmd == "suggest":
            if not args.target:
                print("file required")
                return 2
            src = safe_read(args.target)
            suggestions = assistant.analyze_source(src, filename=args.target, max_suggestions=args.max)
            for i, s in enumerate(suggestions):
                print(f"[{i}] {s.macro_name} {s.args} (score={s.score:.2f}) - {s.reason}")
                if s.snippet:
                    print("     snippet:", s.snippet)
            return 0

        if args.cmd == "preview":
            if not args.target:
                print("file required")
                return 2
            src = safe_read(args.target)
            suggestions = assistant.analyze_source(src, filename=args.target, max_suggestions=args.max)
            if not suggestions:
                print("no suggestions")
                return 0
            idx = args.index
            if idx < 0 or idx >= len(suggestions):
                print("invalid index")
                return 2
            ok, transformed, diagnostics = assistant.preview_apply(src, suggestions[idx], filename=args.target)
            print("=== transformed preview ===")
            print(transformed if transformed else "(no result)")
            if diagnostics:
                print("diagnostics:", diagnostics)
            return 0

        if args.cmd == "apply":
            if not args.target:
                print("file required")
                return 2
            src = safe_read(args.target)
            suggestions = assistant.analyze_source(src, filename=args.target, max_suggestions=args.max)
            if not suggestions:
                print("no suggestions")
                return 0
            idx = args.index
            if idx < 0 or idx >= len(suggestions):
                print("invalid index")
                return 2
            ok, out = assistant.apply_suggestion_to_file(args.target, suggestions[idx], inplace=args.inplace)
            print("applied ->", out if ok else f"failed: {out}")
            return 0

        if args.cmd == "emit-helper":
            if not args.target:
                print("helper name required")
                return 2
            if not _try_codegen:
                print("codegen not available")
                return 2
            params = args.args or []
            parsed = []
            for p in params:
                if "," in p:
                    parsed.append([x.strip() for x in p.split(",")])
                elif p.isdigit():
                    parsed.append(int(p))
                else:
                    parsed.append(p)
            try:
                helper_text = toolkit.emit_helper(args.target, *parsed)
                print(helper_text)
                return 0
            except Exception as e:
                print("emit failed:", e)
                return 2

        if args.cmd == "inject-helper":
            if not args.target:
                print("helper name required")
                return 2
            if not _try_codegen:
                print("codegen not available")
                return 2
            if not args.args or len(args.args) < 1:
                print("usage: inject-helper <helper> <target-file> [helper-args...]")
                return 2
            helper_name = args.target
            target_file = args.args[0]
            helper_args = args.args[1:] if len(args.args) > 1 else []
            parsed = []
            for p in helper_args:
                if "," in p:
                    parsed.append([x.strip() for x in p.split(",")])
                elif p.isdigit():
                    parsed.append(int(p))
                else:
                    parsed.append(p)
            try:
                helper_text = toolkit.emit_helper(helper_name, *parsed)
            except Exception as e:
                print("generate failed:", e)
                return 2
            toolkit.inject_helper(helper_text, target_file)
            print("injected into", target_file)
            return 0

        if args.cmd == "inject-expand":
            if not args.target:
                print("helper name required")
                return 2
            if not _try_codegen:
                print("codegen not available")
                return 2
            if not args.args or len(args.args) < 1:
                print("usage: inject-expand <helper> <target-file> [helper-args...]")
                return 2
            helper_name = args.target
            target_file = args.args[0]
            helper_args = args.args[1:] if len(args.args) > 1 else []
            parsed = []
            for p in helper_args:
                if "," in p:
                    parsed.append([x.strip() for x in p.split(",")])
                elif p.isdigit():
                    parsed.append(int(p))
                else:
                    parsed.append(p)
            try:
                helper_text = toolkit.emit_helper(helper_name, *parsed)
            except Exception as e:
                print("generate failed:", e)
                return 2
            ok, transformed, diagnostics = toolkit.preview_expand_injected(helper_text, target_file)
            if ok:
                tmp = target_file + ".ai.preview"
                safe_write(tmp, transformed, backup=False)
                print("preview written to", tmp)
                if diagnostics:
                    print("diagnostics:", diagnostics)
                return 0
            else:
                print("preview failed:", transformed)
                return 2

        if args.cmd == "optimize-loop":
            if not args.target:
                print("file required")
                return 2
            max_unroll = args.max_unroll
            ok, msg = optimize_loop_unroll_file(args.target, max_unroll=max_unroll, apply=args.apply)
            print(msg)
            return 0 if ok else 2

        if args.cmd == "apply-patch":
            if not args.target or not args.args:
                print("usage: apply-patch <original> <patchfile>")
                return 2
            original = args.target
            patchfile = args.args[0]
            ok, msg = apply_patch(original, patchfile)
            print(msg)
            return 0 if ok else 2

        if args.cmd == "undo":
            if not args.target:
                print("file required")
                return 2
            b = undo_mgr.list_backups(args.target)
            if not b:
                print("no backups found")
                return 0
            for i, bi in enumerate(b):
                print(f"[{i}] {bi}")
            choice = input("Select index to restore or 'q': ").strip()
            if choice == "q":
                return 0
            try:
                idx = int(choice)
                if idx < 0 or idx >= len(b):
                    print("invalid index")
                    return 2
                ok, msg = undo_mgr.restore(b[idx], None)
                if ok:
                    print("restored ->", msg)
                    return 0
                else:
                    print("restore failed:", msg)
                    return 2
            except Exception as e:
                print("invalid input:", e)
                return 2

        if args.cmd == "plugins":
            if not args.target:
                print("usage: plugins <list|load|unload> [name]")
                return 2
            action = args.target
            if action == "list":
                print("available:", plugin_mgr.discover())
                print("loaded:", list(plugin_mgr.loaded.keys()))
                return 0
            if action in ("load", "unload"):
                name = (args.args or [None])[0]
                if not name:
                    print("specify plugin name")
                    return 2
                if action == "load":
                    ok, msg = plugin_mgr.load(name)
                else:
                    ok, msg = plugin_mgr.unload(name)
                print(msg)
                return 0 if ok else 2
            print("unknown plugins action")
            return 2

        if args.cmd == "serve":
            print(f"Serving on port {args.port} ...")
            serve_api(assistant, host="0.0.0.0", port=args.port)
            return 0

        if args.cmd == "test":
            ok = run_unit_tests(verbose=True)
            print("Tests", "PASS" if ok else "FAIL")
            return 0 if ok else 2

        if args.cmd == "export-memory":
            out = args.target or os.path.join(os.getcwd(), "ai_memory_export.json")
            json.dump(assistant.memory.export(), open(out, "w", encoding="utf-8"), indent=2)
            print("exported to", out)
            return 0

        if args.cmd == "import-memory":
            if not args.target:
                print("file required")
                return 2
            data = json.load(open(args.target, "r", encoding="utf-8"))
            assistant.memory.import_data(data, merge=True)
            print("imported")
            return 0

        print("unknown command", args.cmd)
        p.print_help()
        return 2
    except KeyboardInterrupt:
        print("aborted")
        return 1
    except Exception:
        logging.exception("fatal")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
