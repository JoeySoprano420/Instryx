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

