"""
instryx_shell_embedded.py

Extended Instryx Embedded Shell
- Adds many developer-facing features, tools and utilities while remaining fully executable
  using only Python stdlib plus optional repo modules.
- New features:
  - readline-based tab completion and command history
  - plugin system (load modules from ./plugins or installed modules)
  - persistent config (~/.instryx_shell.json)
  - :lint, :format, :test, :build, :pack, :serve, :generate, :open, :search, :plugins commands
  - background task manager for long-running compile/run jobs
  - logging to file + verbose mode
  - simple static HTTP server for serving build artifacts
  - project scaffolding / sample generation
  - integration hooks for instryx_syntax_morph, macro_overlay, instryx_wasm_host_runtime, emitter modules if available
  - graceful fallback to instryxc CLI when emitter absent
- Designed for local development in VS / terminal. No external deps required.
"""

from __future__ import annotations
import shutil
import subprocess
import tempfile
import sys
import os
import readline
import textwrap
import difflib
import importlib
import asyncio
import json
import http.server
import socketserver
import threading
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Callable

# Optional integrations (lazy imported)
_syntax_morph_mod = None
_macro_overlay_mod = None
_emitter_mod = None
_wasm_host_mod = None

# Shell config path
CONFIG_PATH = Path.home() / ".instryx_shell.json"
LOG_PATH = Path.cwd() / "instryx_shell.log"

# Default commands for completion
BASE_COMMANDS = [
    ":help", ":load", ":show", ":morph", ":expand", ":diff", ":compile", ":run", ":save", ":edit", ":clear",
    ":lint", ":format", ":test", ":build", ":pack", ":serve", ":generate", ":open", ":search", ":plugins",
    ":plugins.load", ":plugins.list", ":plugins.unload", ":history", ":quit", ":exit"
]

# Task manager
_background_tasks: Dict[str, asyncio.Task] = {}

# -------------------------
# Utilities and optional module loading
# -------------------------
def _try_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None

def load_optional_modules():
    global _syntax_morph_mod, _macro_overlay_mod, _emitter_mod, _wasm_host_mod
    _syntax_morph_mod = _try_import("instryx_syntax_morph")
    _macro_overlay_mod = _try_import("macro_overlay")
    _emitter_mod = _try_import("instryx_wasm_and_exe_backend_emitter")
    _wasm_host_mod = _try_import("instryx_wasm_host_runtime")

def _log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass

def read_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def write_config(cfg: Dict[str, Any]):
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception as e:
        _log(f"failed to write config: {e}")

# -------------------------
# Existing helpers (unchanged behavior)
# -------------------------
def unified_diff(a: str, b: str, a_name: str = "original", b_name: str = "transformed") -> str:
    a_lines = a.splitlines(keepends=True)
    b_lines = b.splitlines(keepends=True)
    return "".join(difflib.unified_diff(a_lines, b_lines, fromfile=a_name, tofile=b_name, lineterm=""))


async def apply_macro_overlay_async(source: str, filename: Optional[str] = None) -> Tuple[str, list]:
    """
    Attempt to run macro overlay expansion using macro_overlay.applyMacrosWithDiagnostics.
    Supports both synchronous and asynchronous implementations in the module.
    Returns (transformed_source, diagnostics_list)
    """
    global _macro_overlay_mod
    if _macro_overlay_mod is None:
        _macro_overlay_mod = _try_import("macro_overlay")
    if _macro_overlay_mod is None:
        return source, []

    # create registry if available
    registry = None
    if hasattr(_macro_overlay_mod, "createFullRegistry"):
        registry = getattr(_macro_overlay_mod, "createFullRegistry")()
    elif hasattr(_macro_overlay_mod, "createDefaultRegistry"):
        registry = getattr(_macro_overlay_mod, "createDefaultRegistry")()

    apply_fn = getattr(_macro_overlay_mod, "applyMacrosWithDiagnostics", None) or getattr(_macro_overlay_mod, "applyMacros", None)
    if apply_fn is None:
        return source, []

    # applyMacrosWithDiagnostics signature in this repo: (source, registry, ctx)
    try:
        maybe = apply_fn(source, registry, {"filename": filename})
        if asyncio.iscoroutine(maybe):
            res = await maybe
        else:
            # might be sync
            res = maybe
        # result expected shape: { result: { ok, transformed, ...}, diagnostics: [...] } in our code
        if isinstance(res, dict):
            result = res.get("result")
            diagnostics = res.get("diagnostics", [])
            if isinstance(result, dict) and "transformed" in result:
                return result["transformed"], diagnostics
        # fallback: if apply_fn returned ExpansionResult directly
        if isinstance(res, tuple) and len(res) >= 1:
            # not expected, fallback
            return res[0], []
    except Exception as e:
        return source, [{"type": "error", "message": f"macro overlay failed: {e}"}]

    return source, []


def apply_macro_overlay(source: str, filename: Optional[str] = None) -> Tuple[str, list]:
    """
    Synchronous wrapper around async apply_macro_overlay_async.
    """
    try:
        return asyncio.get_event_loop().run_until_complete(apply_macro_overlay_async(source, filename))
    except RuntimeError:
        # no running loop
        return asyncio.new_event_loop().run_until_complete(apply_macro_overlay_async(source, filename))


def try_compile_with_emitter(source: str, out_wasm: str) -> Tuple[bool, str]:
    """
    Try to compile using instryx_wasm_and_exe_backend_emitter or similar module.
    This function probes for common function names; returns (success, message or path).
    """
    global _emitter_mod
    if _emitter_mod is None:
        _emitter_mod = _try_import("instryx_wasm_and_exe_backend_emitter")
    if _emitter_mod is None:
        return False, "emitter module not available"

    # Common function name variants to try
    fn_names = [
        "compile_source_to_wasm",
        "compile_to_wasm",
        "emit_wasm",
        "compile_wasm",
        "compile_ix_to_wasm",
        "compile_to_exe",
    ]
    for name in fn_names:
        fn = getattr(_emitter_mod, name, None)
        if callable(fn):
            try:
                # many likely expect (source, out_path) or (file, out_path)
                ret = fn(source, out_wasm)
                # If it returns bool or path, accept
                if isinstance(ret, bool):
                    return ret, out_wasm if ret else "emitter reported failure"
                if isinstance(ret, str):
                    return True, ret
                return True, out_wasm
            except Exception as e:
                return False, f"emitter.{name} failed: {e}"
    return False, "no compatible compile function found in emitter module"


def try_compile_with_cli(source: str, out_wasm: str, target: str = "wasm") -> Tuple[bool, str]:
    """
    Fallback: write source to temp .ix and call `instryxc` to produce wasm.
    Returns (success, message_or_outpath)
    """
    instryxc = shutil.which("instryxc")
    if instryxc is None:
        return False, "instryxc not found in PATH"
    with tempfile.NamedTemporaryFile("w", suffix=".ix", delete=False, encoding="utf-8") as tf:
        tf.write(source)
        src_path = tf.name
    try:
        if target == "wasm":
            cmd = [instryxc, src_path, "--target", "wasm", "-o", out_wasm]
        else:
            cmd = [instryxc, src_path, "-o", out_wasm]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            return False, f"instryxc failed: {proc.stderr.strip()}"
        return True, out_wasm
    except Exception as e:
        return False, f"instryxc invocation failed: {e}"
    finally:
        try:
            os.unlink(src_path)
        except Exception:
            pass


def compile_to_wasm(source: str, out_wasm: Optional[str] = None) -> Tuple[bool, str]:
    """
    Try multiple strategies to compile to wasm.
    Returns (success, out_path_or_message)
    """
    if out_wasm is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wasm", delete=False)
        out_wasm = tmp.name
        tmp.close()

    # Try emitter module first
    ok, msg = try_compile_with_emitter(source, out_wasm)
    if ok:
        return True, msg

    # Fallback to CLI
    ok, msg = try_compile_with_cli(source, out_wasm, target="wasm")
    if ok:
        return True, msg

    return False, f"compile failed: {msg}"


def compile_to_native(source: str, out_exe: Optional[str] = None) -> Tuple[bool, str]:
    """
    Try to compile to a native executable.
    Returns (success, out_path_or_message)
    """
    if out_exe is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".exe" if os.name == "nt" else ".out", delete=False)
        out_exe = tmp.name
        tmp.close()

    # Try emitter module first
    ok, msg = try_compile_with_emitter(source, out_exe)
    if ok:
        return True, msg

    # Fallback to CLI
    ok, msg = try_compile_with_cli(source, out_exe, target="native")
    if ok:
        return True, msg

    return False, f"compile failed: {msg}"


def run_wasm_module(wasm_path: str, func: str = "main", args: Optional[Tuple[str, ...]] = None) -> Tuple[bool, Any]:
    """
    Instantiate and run the wasm module using instryx_wasm_host_runtime.WasmHostRuntime
    Returns (success, result_or_error)
    """
    global _wasm_host_mod
    if _wasm_host_mod is None:
        _wasm_host_mod = _try_import("instryx_wasm_host_runtime")
    if _wasm_host_mod is None:
        return False, "instryx_wasm_host_runtime module not available"

    try:
        Runtime = getattr(_wasm_host_mod, "WasmHostRuntime")
        rt = Runtime(enable_wasi=True)
        rt.instantiate(wasm_path)
        if args:
            # try calling with string args
            res = rt.call_with_strings(func, args)
            return True, res
        else:
            res = rt.call(func)
            return True, res
    except Exception as e:
        return False, f"runtime error: {e}"


# -------------------------
# REPL / Shell
# -------------------------
REPL_BANNER = """Instryx Embedded Shell (extended)
Type :help for commands.
"""

class PluginManager:
    """
    Simple plugin loader; plugins are Python modules that export `register(shell)` function.
    Plugins can live in ./plugins directory or be installed packages.
    """
    def __init__(self, shell):
        self.shell = shell
        self.loaded: Dict[str, Any] = {}
        self.plugins_dir = Path.cwd() / "plugins"
        self.plugins_dir.mkdir(exist_ok=True)

    def discover(self) -> List[str]:
        names = []
        for p in self.plugins_dir.glob("*.py"):
            names.append(p.stem)
        return names

    def load(self, name: str) -> Tuple[bool, str]:
        # try local plugins first
        try:
            full = f"plugins.{name}"
            if (self.plugins_dir / f"{name}.py").exists():
                spec = importlib.util.spec_from_file_location(full, str(self.plugins_dir / f"{name}.py"))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
            else:
                mod = importlib.import_module(name)
            if hasattr(mod, "register") and callable(mod.register):
                mod.register(self.shell)
            self.loaded[name] = mod
            return True, f"plugin {name} loaded"
        except Exception as e:
            return False, f"failed to load plugin {name}: {e}"

    def unload(self, name: str) -> Tuple[bool, str]:
        if name not in self.loaded:
            return False, "not loaded"
        try:
            mod = self.loaded.pop(name)
            if hasattr(mod, "unregister") and callable(mod.unregister):
                mod.unregister(self.shell)
            return True, f"plugin {name} unloaded"
        except Exception as e:
            return False, f"unload failed: {e}"

    def list_loaded(self) -> List[str]:
        return list(self.loaded.keys())

class InstryxShell:
    def __init__(self):
        load_optional_modules()
        self.source_original: Optional[str] = None
        self.source_morphed: Optional[str] = None
        self.source_expanded: Optional[str] = None
        self.last_wasm_path: Optional[str] = None
        self.last_native_path: Optional[str] = None
        self.current_file: Optional[str] = None
        self.history_file = Path.home() / ".instryx_shell_history"
        self.cfg = read_config()
        self.plugin_mgr = PluginManager(self)
        self._setup_readline()
        self._load_history()

    # ----- readline / completion / history -----
    def _setup_readline(self):
        try:
            readline.parse_and_bind("tab: complete")
            readline.set_completer(self._completer)
        except Exception:
            pass

    def _completer(self, text, state):
        options = [c for c in BASE_COMMANDS + list(self.plugin_mgr.discover()) if c.startswith(text)]
        try:
            return options[state]
        except Exception:
            return None

    def _load_history(self):
        try:
            if self.history_file.exists():
                readline.read_history_file(str(self.history_file))
        except Exception:
            pass

    def _save_history(self):
        try:
            readline.write_history_file(str(self.history_file))
        except Exception:
            pass

    # ----- REPL -----
    def repl(self):
        print(REPL_BANNER)
        try:
            while True:
                try:
                    line = input("instryx> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not line:
                    continue
                if line.startswith(":"):
                    parts = line.split()
                    cmd = parts[0][1:]
                    args = parts[1:]
                    method = getattr(self, f"cmd_{cmd.replace('.', '_')}", None)
                    if method:
                        try:
                            method(args)
                        except Exception as e:
                            _log(f"command {cmd} failed: {e}")
                            print(f"[error] command {cmd} failed: {e}")
                    else:
                        # plugin dispatch
                        if cmd.startswith("plugins."):
                            sub = cmd.split(".", 1)[1]
                            handler = getattr(self, f"plugins_{sub}", None)
                            if handler:
                                handler(args)
                                continue
                        print(f"Unknown command: {cmd}. Type :help")
                else:
                    if self.source_original is None:
                        self.source_original = line + "\n"
                    else:
                        self.source_original += line + "\n"
                    self.source_morphed = None
                    self.source_expanded = None
                    print("[ok] appended line to source buffer")
        finally:
            self._save_history()

    # ----- core commands -----
    def cmd_help(self, _args):
        print(REPL_BANNER)
        print("Commands:")
        print("  :load <file>           load an Instryx (.ix) file")
        print("  :show                  show current buffer (expanded > morphed > original)")
        print("  :morph                 apply morphology/formatting")
        print("  :format                alias for :morph (writes back if --inplace configured)")
        print("  :expand                run macro overlay expansion (if available)")
        print("  :diff                  show unified diff (original -> morphed -> expanded)")
        print("  :compile [out.wasm]    compile to wasm")
        print("  :build [out.exe]       compile to native executable")
        print("  :run [wasm] [func]     run wasm (uses host runtime if available)")
        print("  :lint                  run basic lint checks")
        print("  :test                  run unit tests (syntax morph + macro overlay tests)")
        print("  :pack <out.zip>        package build artifacts into zip")
        print("  :serve [dir] [port]    serve directory over HTTP (static)")
        print("  :generate <type> <name> generate sample project (app/library) in ./<name>")
        print("  :open <file>           open file with system-default editor")
        print("  :search <pattern>      search files for pattern (simple grep)")
        print("  :plugins.list          list available plugin files")
        print("  :plugins.load <name>   load plugin by name")
        print("  :plugins.unload <name> unload plugin")
        print("  :history               show readline history path")
        print("  :quit / :exit          exit")
        print("Use :help <command> for more details (not implemented per-command).")

    def cmd_load(self, args):
        if not args:
            print("Usage: :load <file>")
            return
        path = args[0]
        try:
            txt = Path(path).read_text(encoding="utf-8")
            self.source_original = txt
            self.source_morphed = None
            self.source_expanded = None
            self.current_file = path
            print(f"Loaded {path} ({len(txt)} bytes)")
        except Exception as e:
            print(f"Failed to load {path}: {e}")

    def cmd_show(self, _args):
        src = self.source_expanded or self.source_morphed or self.source_original
        if src is None:
            print("[no source loaded]")
            return
        print("----- CURRENT SOURCE -----")
        print(src)

    def cmd_morph(self, _args):
        if self.source_original is None:
            print("No source loaded")
            return
        if _syntax_morph_mod is None:
            print("instryx_syntax_morph module not available")
            return
        try:
            SyntaxMorph = getattr(_syntax_morph_mod, "SyntaxMorph")
            sm = SyntaxMorph()
            res = sm.morph(self.source_original)
            self.source_morphed = res.transformed
            self.source_expanded = None
            print(f"[morph] applied {len(res.edits)} edits")
            _log(f"morph applied {len(res.edits)} edits")
        except Exception as e:
            print(f"[morph] error: {e}")
            _log(f"morph error: {e}")

    # :format alias
    def cmd_format(self, args):
        self.cmd_morph(args)

    def cmd_expand(self, _args):
        if self.source_morphed is None and self.source_original is None:
            print("No source to expand")
            return
        src = self.source_morphed or self.source_original
        transformed, diagnostics = apply_macro_overlay(src, self.current_file)
        self.source_expanded = transformed
        print(f"[expand] applied (diagnostics: {len(diagnostics)})")
        for d in diagnostics[:10]:
            print(f" - {d.get('type','info')}: {d.get('message')}")
        if len(diagnostics) > 10:
            print(f" - ... {len(diagnostics)-10} more")
        _log(f"expand diagnostics: {len(diagnostics)}")

    def cmd_diff(self, _args):
        if self.source_original is None:
            print("No source loaded")
            return
        morphed = self.source_morphed or self.source_original
        expanded = self.source_expanded or morphed
        print("---- morph diff ----")
        print(unified_diff(self.source_original, morphed, "original", "morphed"))
        if expanded != morphed:
            print("---- expand diff ----")
            print(unified_diff(morphed, expanded, "morphed", "expanded"))

    def cmd_compile(self, args):
        if self.source_expanded is None and self.source_morphed is None and self.source_original is None:
            print("No source to compile")
            return
        src = self.source_expanded or self.source_morphed or self.source_original
        out = args[0] if args else None
        _log("compile requested")
        # run in background to avoid blocking REPL
        async def _job():
            ok, msg = compile_to_wasm(src, out)
            if ok:
                self.last_wasm_path = msg
                print(f"[compile] success -> {msg}")
                _log(f"compile success -> {msg}")
            else:
                print(f"[compile] failed: {msg}")
                _log(f"compile failed: {msg}")
        asyncio.create_task(_job())

    def cmd_build(self, args):
        if self.source_expanded is None and self.source_morphed is None and self.source_original is None:
            print("No source to build")
            return
        src = self.source_expanded or self.source_morphed or self.source_original
        out = args[0] if args else None
        _log("build requested")
        async def _job():
            ok, msg = compile_to_native(src, out)
            if ok:
                self.last_native_path = msg
                print(f"[build] success -> {msg}")
                _log(f"build success -> {msg}")
            else:
                print(f"[build] failed: {msg}")
                _log(f"build failed: {msg}")
        asyncio.create_task(_job())

    def cmd_run(self, args):
        wasm = None
        func = "main"
        run_args = None
        if args:
            wasm = args[0]
            if len(args) > 1:
                func = args[1]
        if wasm is None:
            wasm = self.last_wasm_path
        if wasm is None:
            print("No wasm module available. Use :compile or specify path.")
            return
        _log(f"run requested {wasm} {func}")
        async def _job():
            ok, res = run_wasm_module(wasm, func, run_args)
            if ok:
                print(f"[run] result: {res}")
                _log(f"run result: {res}")
            else:
                print(f"[run] failed: {res}")
                _log(f"run failed: {res}")
        asyncio.create_task(_job())

    def cmd_save(self, args):
        if not args:
            print("Usage: :save <file>")
            return
        path = args[0]
        src = self.source_expanded or self.source_morphed or self.source_original
        if src is None:
            print("No source to save")
            return
        Path(path).write_text(src, encoding="utf-8")
        print(f"Wrote {path}")

    def cmd_edit(self, _args):
        print("Enter multi-line source. End with a single '.' on its own line.")
        lines = []
        try:
            while True:
                ln = input()
                if ln.strip() == ".":
                    break
                lines.append(ln)
        except (EOFError, KeyboardInterrupt):
            print()
        text = "\n".join(lines) + "\n"
        self.source_original = (self.source_original or "") + text
        self.source_morphed = None
        self.source_expanded = None
        print(f"[edit] appended {len(lines)} lines")

    def cmd_clear(self, _args):
        self.source_original = None
        self.source_morphed = None
        self.source_expanded = None
        self.last_wasm_path = None
        self.last_native_path = None
        self.current_file = None
        print("[cleared buffers]")

    def cmd_lint(self, _args):
        """
        Lightweight lint checks:
        - unmatched quotes
        - unbalanced parentheses/braces
        - trailing whitespace
        - missing semicolons heuristics
        """
        src = self.source_expanded or self.source_morphed or self.source_original
        if src is None:
            print("No source to lint")
            return
        issues = []
        # unmatched quotes
        for q in ['"', "'"]:
            if src.count(q) % 2 != 0:
                issues.append(f"unmatched quote: {q}")
        # braces
        if src.count("{") != src.count("}"):
            issues.append("unbalanced braces")
        if src.count("(") != src.count(")"):
            issues.append("unbalanced parentheses")
        # trailing whitespace
        for i, line in enumerate(src.splitlines(), 1):
            if line.rstrip() != line:
                issues.append(f"trailing whitespace on line {i}")
        # missing semicolon heuristic
        lines = [l.rstrip() for l in src.splitlines()]
        for i, l in enumerate(lines, 1):
            if l and not l.endswith((';', '{', '}', ':')) and not l.strip().startswith('--') and len(l) < 200:
                # simple heuristic - skip lines that look like keywords
                if not any(l.strip().startswith(k) for k in ('func', 'if', 'while', 'quarantine', 'else', 'return')):
                    issues.append(f"maybe missing semicolon at line {i}: {l.strip()[:80]}")
        if not issues:
            print("[lint] no issues found")
        else:
            print("[lint] issues:")
            for it in issues:
                print(" -", it)

    def cmd_test(self, _args):
        passed = True
        # run syntax morph unit tests if available
        if _syntax_morph_mod:
            try:
                fn = getattr(_syntax_morph_mod, "run_unit_tests", None)
                if callable(fn):
                    ok = fn(verbose=False)
                    print(f"syntax morph tests: {'PASS' if ok else 'FAIL'}")
                    passed = passed and ok
            except Exception as e:
                print("syntax morph tests failed:", e)
                passed = False
        else:
            print("syntax morph module unavailable; skipping")
        # attempt macro overlay demo/test if available
        if _macro_overlay_mod is None:
            _macro_overlay_mod = _try_import("macro_overlay")
        if _macro_overlay_mod:
            try:
                demo = getattr(_macro_overlay_mod, "demoExpand", None)
                if callable(demo):
                    res = demo()
                    # demoExpand might be async or sync
                    if asyncio.iscoroutine(res):
                        res = asyncio.get_event_loop().run_until_complete(res)
                    print("macro overlay demo executed")
                else:
                    print("no demoExpand in macro_overlay")
            except Exception as e:
                print("macro overlay demo failed:", e)
                passed = False
        else:
            print("macro_overlay module unavailable; skipping")
        print("tests complete")
        return passed

    def cmd_pack(self, args):
        if not args:
            print("Usage: :pack <out.zip>")
            return
        out = args[0]
        files = []
        if self.last_wasm_path and Path(self.last_wasm_path).exists():
            files.append(self.last_wasm_path)
        if self.last_native_path and Path(self.last_native_path).exists():
            files.append(self.last_native_path)
        if self.current_file:
            files.append(self.current_file)
        if not files:
            print("No artifacts to pack")
            return
        try:
            import zipfile
            with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for f in files:
                    z.write(f, arcname=Path(f).name)
            print(f"Packed {len(files)} files into {out}")
        except Exception as e:
            print("pack failed:", e)

    def cmd_serve(self, args):
        dir_to_serve = args[0] if args else "."
        port = int(args[1]) if len(args) > 1 else 8000
        handler = http.server.SimpleHTTPRequestHandler
        prev_cwd = os.getcwd()
        os.chdir(dir_to_serve)
        httpd = socketserver.TCPServer(("", port), handler)
        print(f"Serving {dir_to_serve} at http://localhost:{port} (Ctrl-C to stop)")
        _log(f"serve started on {port} for {dir_to_serve}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("Server stopped")
        finally:
            httpd.server_close()
            os.chdir(prev_cwd)

    def cmd_generate(self, args):
        """ generate sample project: :generate app myapp """
        if len(args) < 2:
            print("Usage: :generate <app|lib> <name>")
            return
        typ, name = args[0], args[1]
        target = Path(name)
        if target.exists():
            print(f"{name} already exists")
            return
        target.mkdir(parents=True)
        # basic files
        main = target / "main.ix"
        readme = target / "README.md"
        build_sh = target / "build.sh"
        if typ == "app":
            main.write_text(textwrap.dedent("""\
                -- Generated Instryx app
                func greet(name) {
                    print: "Hello, " + name + "!";
                };

                main() {
                    greet("Instryx");
                };
                """), encoding="utf-8")
        else:
            main.write_text(textwrap.dedent("""\
                -- Generated library
                func lib_fn() {
                    print: "library function";
                };
                """), encoding="utf-8")
        readme.write_text(f"# {name}\n\nGenerated by Instryx shell.", encoding="utf-8")
        build_sh.write_text("#!/bin/sh\ninstryxc main.ix -o main.wasm\n", encoding="utf-8")
        try:
            build_sh.chmod(0o755)
        except Exception:
            pass
        print(f"Generated {typ} {name}")

    def cmd_open(self, args):
        if not args:
            print("Usage: :open <file>")
            return
        path = args[0]
        try:
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.run(["open", path])
            else:
                subprocess.run(["xdg-open", path])
            print(f"Opened {path}")
        except Exception as e:
            print("open failed:", e)

    def cmd_search(self, args):
        if not args:
            print("Usage: :search <pattern>")
            return
        pat = args[0]
        root = Path(".")
        matches = []
        for f in root.rglob("*.ix"):
            try:
                text = f.read_text(encoding="utf-8")
                if pat in text:
                    matches.append(str(f))
            except Exception:
                continue
        if not matches:
            print("No matches")
        else:
            for m in matches:
                print(m)

    def cmd_plugins_list(self, _args):
        available = self.plugin_mgr.discover()
        loaded = self.plugin_mgr.list_loaded()
        print("Available plugins:", ", ".join(available) if available else "(none)")
        print("Loaded plugins:", ", ".join(loaded) if loaded else "(none)")

    def cmd_plugins_load(self, args):
        if not args:
            print("Usage: :plugins.load <name>")
            return
        name = args[0]
        ok, msg = self.plugin_mgr.load(name)
        print(msg)

    def cmd_plugins_unload(self, args):
        if not args:
            print("Usage: :plugins.unload <name>")
            return
        name = args[0]
        ok, msg = self.plugin_mgr.unload(name)
        print(msg)

    def cmd_history(self, _args):
        print("History file:", str(self.history_file))

    def cmd_quit(self, _args):
        print("Bye.")
        raise SystemExit(0)

    def cmd_exit(self, _args):
        self.cmd_quit(_args)

# -------------------------
# Entry point
# -------------------------
def main():
    shell = InstryxShell()
    try:
        shell.repl()
    except SystemExit:
        pass
    except Exception as e:
        _log(f"fatal: {e}")
        print(f"[fatal] {e}", file=sys.stderr)

if __name__ == "__main__":
    main()

    import os
    os._exit(0)  # ensure all threads exit
    import sys
    sys.exit(0)
    """
    """
    # instryx_shell_embedded.py
    import os
    os._exit(0)  # ensure all threads exit
    import sys
    sys.exit(0)
    
# instryx_shell_enhancements.py
# Enhancements for instryx_shell_embedded.py:
# - TaskManager (async background tasks with persistence)
# - FileWatcher (watchdog if available, fallback to polling)
# - MetricsServer (simple /metrics HTTP endpoint for Prometheus)
# - Optional ncurses UI overlay for task monitoring
#
# Designed to be imported and used by instryx_shell_embedded.py

from __future__ import annotations
import asyncio
import gzip
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# optional imports
try:
    from watchdog.observers import Observer  # type: ignore
    from watchdog.events import FileSystemEventHandler  # type: ignore
    _HAS_WATCHDOG = True
except Exception:
    _HAS_WATCHDOG = False

try:
    import curses  # type: ignore
    _HAS_CURSES = True
except Exception:
    _HAS_CURSES = False

# persistent cache dir for shell
CACHE_DIR = Path.home() / ".instryx_shell_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_TASKS_PERSIST = CACHE_DIR / "tasks.json"
_TASKS_LOCK = threading.RLock()

# simple global metrics store (prometheus-style counters)
_metrics_lock = threading.RLock()
_metrics: Dict[str, int] = {
    "instryx_compile_requests_total": 0,
    "instryx_compile_success_total": 0,
    "instryx_compile_failure_total": 0,
    "instryx_run_requests_total": 0,
    "instryx_run_success_total": 0,
    "instryx_run_failure_total": 0,
}

# threadpool to run blocking work from shell
_BLOCKING_POOL = ThreadPoolExecutor(max_workers=4)


class TaskManager:
    """
    Lightweight TaskManager.
    create(label, coro_func) -> tid
    list() -> dict of metadata
    cancel(tid) -> bool
    Persist metadata to disk to remain inspectable across shell restarts.
    """

    def __init__(self):
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._load()

    def _load(self):
        try:
            if _TASKS_PERSIST.exists():
                with _TASKS_LOCK:
                    self._tasks = json.loads(_TASKS_PERSIST.read_text(encoding="utf-8"))
        except Exception:
            self._tasks = {}

    def _save(self):
        try:
            with _TASKS_LOCK:
                _TASKS_PERSIST.write_text(json.dumps(self._tasks, default=str, indent=2), encoding="utf-8")
        except Exception:
            pass

    def create(self, label: str, coro_fn: Callable[[], Any]) -> str:
        tid = hashlib.sha1(f"{label}:{time.time()}".encode()).hexdigest()[:12]
        meta = {
            "id": tid,
            "label": label,
            "status": "queued",
            "start_ts": None,
            "finish_ts": None,
            "result": None,
        }
        with self._lock:
            self._tasks[tid] = meta
            self._save()

        async def _runner():
            meta["status"] = "running"
            meta["start_ts"] = time.time()
            self._save()
            try:
                res = await coro_fn()
                meta["status"] = "done"
                meta["result"] = {"ok": True, "value": res}
            except asyncio.CancelledError:
                meta["status"] = "cancelled"
            except Exception as e:
                meta["status"] = "error"
                meta["result"] = {"ok": False, "error": str(e)}
            meta["finish_ts"] = time.time()
            self._save()

        task = asyncio.create_task(_runner())
        with self._lock:
            self._tasks[tid]["task_ref"] = task
            self._save()
        return tid

    def list(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            out = {}
            for k, v in self._tasks.items():
                copyv = dict(v)
                copyv.pop("task_ref", None)
                out[k] = copyv
            return out

    def cancel(self, tid: str) -> bool:
        with self._lock:
            entry = self._tasks.get(tid)
            if not entry:
                return False
            tref = entry.get("task_ref")
            if tref and not tref.done():
                tref.cancel()
                entry["status"] = "cancelling"
                self._save()
                return True
        return False


class _PollingWatcher(threading.Thread):
    """Lightweight fallback watcher (polling)."""

    def __init__(self, callback: Callable[[str], None], poll_interval: float = 0.5):
        super().__init__(daemon=True)
        self._watched: Dict[str, float] = {}
        self._cb = callback
        self._interval = poll_interval
        self._stop = threading.Event()

    def watch(self, path: str):
        p = Path(path)
        self._watched[path] = p.stat().st_mtime if p.exists() else 0.0

    def unwatch(self, path: str):
        self._watched.pop(path, None)
        if not self._watched:
            self._stop.set()

    def run(self):
        while not self._stop.is_set() and self._watched:
            for p, last in list(self._watched.items()):
                try:
                    if Path(p).exists():
                        m = Path(p).stat().st_mtime
                        if m != last:
                            self._watched[p] = m
                            try:
                                self._cb(p)
                            except Exception:
                                pass
                    else:
                        self._watched.pop(p, None)
                except Exception:
                    continue
            time.sleep(self._interval)

    def stop(self):
        self._stop.set()


class FileWatcher:
    """
    FileWatcher that uses watchdog when available; falls back to polling otherwise.

    API:
      fw = FileWatcher()
      fw.watch(path, callback)
      fw.unwatch(path)
    """

    def __init__(self):
        self._poller: Optional[_PollingWatcher] = None
        self._observer = None
        self._handlers: Dict[str, Callable[[str], None]] = {}
        if _HAS_WATCHDOG:
            class _Handler(FileSystemEventHandler):
                def __init__(self, outer):
                    super().__init__()
                    self._outer = outer

                def on_modified(self, event):
                    if not event.is_directory:
                        cb = self._outer._handlers.get(event.src_path)
                        if cb:
                            try:
                                cb(event.src_path)
                            except Exception:
                                pass

                def on_created(self, event):
                    if not event.is_directory:
                        cb = self._outer._handlers.get(event.src_path)
                        if cb:
                            try:
                                cb(event.src_path)
                            except Exception:
                                pass
            self._wd_handler_cls = _Handler
        else:
            self._wd_handler_cls = None

    def watch(self, path: str, callback: Callable[[str], None]):
        path = str(Path(path).resolve())
        self._handlers[path] = callback
        if _HAS_WATCHDOG:
            if self._observer is None:
                self._observer = Observer()
                self._observer.start()
            handler = self._wd_handler_cls(self)
            self._observer.schedule(handler, os.path.dirname(path) or ".", recursive=False)
        else:
            if self._poller is None or not self._poller.is_alive():
                self._poller = _PollingWatcher(callback)
                self._poller.start()
            self._poller.watch(path)

    def unwatch(self, path: str):
        path = str(Path(path).resolve())
        self._handlers.pop(path, None)
        if _HAS_WATCHDOG and self._observer:
            # watchdog doesn't provide per-path unschedule easily here; best-effort leave running
            pass
        else:
            if self._poller:
                self._poller.unwatch(path)

    def stop(self):
        if _HAS_WATCHDOG and self._observer:
            try:
                self._observer.stop()
                self._observer.join(timeout=1.0)
            except Exception:
                pass
        if self._poller:
            self._poller.stop()


# Simple HTTP metrics endpoint for Prometheus
@dataclass
class MetricsServer:
    host: str = "127.0.0.1"
    port: int = 8001
    _server: Optional[threading.Thread] = None
    _httpd: Any = None

    def _make_handler(self):
        metrics_ref = _metrics
        metrics_lock = _metrics_lock

        class _Handler(http.server.BaseHTTPRequestHandler):  # type: ignore
            def do_GET(self):
                if self.path != "/metrics":
                    self.send_response(404)
                    self.end_headers()
                    return
                with metrics_lock:
                    payload = "\n".join(f"{k} {v}" for k, v in metrics_ref.items()) + "\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload.encode("utf-8"))

            def log_message(self, format, *args):
                return  # silence

        return _Handler

    def start(self):
        if self._server and self._server.is_alive():
            return
        handler = self._make_handler()
        import socketserver
        self._httpd = socketserver.TCPServer((self.host, self.port), handler)
        def _serve():
            try:
                self._httpd.serve_forever()
            except Exception:
                pass
        self._server = threading.Thread(target=_serve, daemon=True)
        self._server.start()

    def stop(self):
        if self._httpd:
            try:
                self._httpd.shutdown()
            except Exception:
                pass
        if self._server:
            self._server.join(timeout=1.0)


# Simple ncurses UI for viewing tasks (optional)
def start_tasks_ui(task_mgr: TaskManager):
    if not _HAS_CURSES:
        raise RuntimeError("curses not available on this platform")
    import curses

    def _ui(stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        while True:
            stdscr.erase()
            tasks = task_mgr.list()
            stdscr.addstr(0, 0, "Instryx Tasks (q to quit)".ljust(80), curses.A_REVERSE)
            row = 1
            for tid, meta in tasks.items():
                label = meta.get("label", "")[:40]
                status = meta.get("status", "")
                start_ts = meta.get("start_ts") or ""
                finish_ts = meta.get("finish_ts") or ""
                line = f"{tid} {label:40} {status:10} start={start_ts} finish={finish_ts}"
                stdscr.addstr(row, 0, line[:curses.COLS-1])
                row += 1
                if row >= curses.LINES - 1:
                    break
            stdscr.refresh()
            try:
                ch = stdscr.getch()
                if ch in (ord("q"), ord("Q")):
                    break
            except Exception:
                pass
            time.sleep(0.2)

    curses.wrapper(_ui)

# tests/test_shell_enhancements.py
# Unit tests for TaskManager and FileWatcher (fallback behavior)
import asyncio
import os
import tempfile
import time
import unittest
from pathlib import Path

from instryx_shell_enhancements import TaskManager, FileWatcher, MetricsServer, _metrics, _metrics_lock

class TestTaskManager(unittest.IsolatedAsyncioTestCase):
    async def test_create_and_complete_task(self):
        tm = TaskManager()
        async def coro():
            await asyncio.sleep(0.01)
            return "ok"
        tid = tm.create("quick", coro)
        # wait until task finishes
        for _ in range(100):
            lst = tm.list()
            if lst[tid]["status"] in ("done", "error"):
                break
            await asyncio.sleep(0.01)
        lst = tm.list()
        self.assertIn(tid, lst)
        self.assertIn(lst[tid]["status"], ("done", "error"))

    async def test_cancel(self):
        tm = TaskManager()
        async def long_coro():
            await asyncio.sleep(2)
            return "done"
        tid = tm.create("long", long_coro)
        # cancel immediately
        ok = tm.cancel(tid)
        self.assertTrue(ok)
        # wait a bit
        await asyncio.sleep(0.05)
        lst = tm.list()
        self.assertIn(tid, lst)
        self.assertIn(lst[tid]["status"], ("cancelling", "cancelled", "done", "error"))

class TestFileWatcher(unittest.TestCase):
    def test_polling_watch(self):
        fw = FileWatcher()
        tf = tempfile.NamedTemporaryFile(delete=False)
        path = tf.name
        tf.write(b"hello")
        tf.flush()
        tf.close()
        seen = []
        def cb(p):
            seen.append(p)
        try:
            fw.watch(path, cb)
            # modify file
            time.sleep(0.1)
            with open(path, "w") as f:
                f.write("changed")
            # wait for callback
            for _ in range(50):
                if seen:
                    break
                time.sleep(0.05)
            self.assertTrue(len(seen) >= 1)
        finally:
            try:
                fw.unwatch(path)
            except Exception:
                pass
            try:
                os.unlink(path)
            except Exception:
                pass

class TestMetricsServer(unittest.TestCase):
    def test_metrics_endpoint(self):
        ms = MetricsServer(host="127.0.0.1", port=18081)
        ms.start()
        try:
            # increment a metric
            with _metrics_lock:
                _metrics["instryx_compile_requests_total"] += 1
            import urllib.request
            res = urllib.request.urlopen("http://127.0.0.1:18081/metrics")
            body = res.read().decode()
            self.assertIn("instryx_compile_requests_total", body)
        finally:
            ms.stop()

if __name__ == "__main__":
    unittest.main()

# (append) Integration glue for the new enhancements module
# This block can be appended to the end of instryx_shell_embedded.py to wire in the new features.
try:
    from instryx_shell_enhancements import TaskManager, FileWatcher, MetricsServer, start_tasks_ui, _metrics, _metrics_lock
except Exception:
    TaskManager = None
    FileWatcher = None
    MetricsServer = None
    start_tasks_ui = None

# Attach optional instances and commands to InstryxShell by monkey-patching if available.
if TaskManager is not None and FileWatcher is not None:
    def _attach_enhancements(shell_cls):
        # provide lazy-initialized components on shell instances
        def _ensure_components(self):
            if not hasattr(self, "task_mgr"):
                self.task_mgr = TaskManager()
            if not hasattr(self, "file_watcher"):
                self.file_watcher = FileWatcher()
            if not hasattr(self, "metrics_server"):
                self.metrics_server = None
        shell_cls._ensure_components = _ensure_components

        async def _wrap_coro(fn):
            # helper to adapt blocking functions to coroutine for TaskManager
            return await fn()

        def cmd_tasks(self, _args):
            self._ensure_components()
            tasks = self.task_mgr.list()
            if not tasks:
                print("No background tasks")
                return
            for tid, meta in tasks.items():
                print(f"{tid} {meta.get('label')} status={meta.get('status')} start={meta.get('start_ts')} finish={meta.get('finish_ts')}")
        shell_cls.cmd_tasks = cmd_tasks

        def cmd_task_cancel(self, args):
            if not args:
                print("Usage: :task.cancel <id>")
                return
            self._ensure_components()
            ok = self.task_mgr.cancel(args[0])
            print("cancelled" if ok else "not found or already done")
        shell_cls.cmd_task_cancel = cmd_task_cancel

        def cmd_watch(self, args):
            if not args:
                print("Usage: :watch <file>")
                return
            self._ensure_components()
            path = args[0]
            def cb(p):
                print(f"[watch] change detected: {p}")
                try:
                    txt = Path(p).read_text(encoding="utf-8")
                    self.source_original = txt
                    self.source_morphed = None
                    self.source_expanded = None
                    print("[watch] reloaded source buffer")
                except Exception as e:
                    print("[watch] reload failed:", e)
            self.file_watcher.watch(path, cb)
            print(f"Watching {path} for changes")
        shell_cls.cmd_watch = cmd_watch

        def cmd_metrics_start(self, args):
            host = args[0] if args else "127.0.0.1"
            port = int(args[1]) if len(args) > 1 else 8001
            self._ensure_components()
            if getattr(self, "metrics_server", None) is None:
                self.metrics_server = MetricsServer(host=host, port=port)
                self.metrics_server.start()
                print(f"metrics server started at http://{host}:{port}/metrics")
            else:
                print("metrics server already running")
        shell_cls.cmd_metrics_start = cmd_metrics_start

        def cmd_metrics_stop(self, _args):
            if getattr(self, "metrics_server", None):
                try:
                    self.metrics_server.stop()
                except Exception:
                    pass
                self.metrics_server = None
                print("metrics server stopped")
            else:
                print("metrics server not running")
        shell_cls.cmd_metrics_stop = cmd_metrics_stop

        def cmd_tasks_ui(self, _args):
            self._ensure_components()
            if start_tasks_ui is None:
                print("ncurses UI not available on this platform")
                return
            try:
                start_tasks_ui(self.task_mgr)
            except Exception as e:
                print("tasks UI failed:", e)
        shell_cls.cmd_tasks_ui = cmd_tasks_ui

    try:
        _attach_enhancements(InstryxShell)
    except Exception:
        pass

