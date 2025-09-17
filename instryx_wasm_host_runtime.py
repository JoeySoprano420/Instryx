# instryxc.py
# Final CLI Compiler Wrapper for the Instryx Language
# Author: Violet Magenta / VACU Technologies
# License: MIT

import argparse
import sys
import os
from instryx_parser import InstryxParser
from instryx_ast_interpreter import InstryxInterpreter
from instryx_llvm_ir_codegen import InstryxLLVMCodegen
from instryx_jit_aot_runner import InstryxRunner
from instryx_wasm_and_exe_backend_emitter import InstryxEmitter
from instryx_dodecagram_ast_visualizer import DodecagramExporter

def compile_and_run(args):
    if args.run:
        runner = InstryxRunner()
        runner.run(args.file.read())
    elif args.emit == "llvm":
        codegen = InstryxLLVMCodegen()
        print(codegen.generate(args.file.read()))
    elif args.emit in ["exe", "wasm"]:
        emitter = InstryxEmitter()
        emitter.emit(args.file.read(), target=args.emit, output_name=args.output)
    elif args.emit == "ast":
        parser = InstryxParser()
        ast = parser.parse(args.file.read())
        print(ast)
    elif args.emit == "visual":
        viz = DodecagramExporter()
        viz.parse_code(args.file.read())
        viz.export_to_graphviz(f"{args.output}_ast")
    elif args.emit == "json":
        viz = DodecagramExporter()
        viz.parse_code(args.file.read())
        viz.export_to_json(f"{args.output}_ast.json")
    elif args.emit == "interpret":
        interpreter = InstryxInterpreter()
        interpreter.interpret(args.file.read())
    else:
        print("Unknown emit mode:", args.emit)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="🧠 Instryx CLI Compiler")
    parser.add_argument("file", type=argparse.FileType("r"), help="Instryx source file (.ix)")
    parser.add_argument("-o", "--output", type=str, default="program", help="Output file name prefix")
    parser.add_argument("--emit", type=str, choices=["llvm", "exe", "wasm", "ast", "visual", "json", "interpret"],
                        help="What to emit: LLVM IR, binary, AST, visual, etc.")
    parser.add_argument("--run", action="store_true", help="JIT compile and run the code")

    args = parser.parse_args()
    compile_and_run(args)

if __name__ == "__main__":
    main()

# instryxc.py
# Final CLI Compiler Wrapper for the Instryx Language — boosted edition
# Author: Violet Magenta / VACU Technologies (modified)
# License: MIT
"""
Enhancements:
 - watch mode (polling fallback) to recompile/re-run on source changes
 - metrics HTTP endpoint (/metrics) for lightweight Prometheus scraping
 - batch mode: compile/emit many files in parallel
 - file-based IR/artifact cache to avoid redundant work
 - improved CLI, logging, verbosity and error handling
 - fully implemented, self-contained (pure stdlib)
"""

from __future__ import annotations
import argparse
import sys
import os
import time
import hashlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence, Dict, Any

from instryx_parser import InstryxParser
from instryx_ast_interpreter import InstryxInterpreter
from instryx_llvm_ir_codegen import InstryxLLVMCodegen
from instryx_jit_aot_runner import InstryxRunner
from instryx_wasm_and_exe_backend_emitter import InstryxEmitter
from instryx_dodecagram_ast_visualizer import DodecagramExporter
import http.server

LOG = logging.getLogger("instryxc")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


CACHE_DIR = Path.home() / ".instryx_cli_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _file_hash(contents: str) -> str:
    return hashlib.sha256(contents.encode("utf-8")).hexdigest()


def _cache_write(name: str, contents: str) -> Path:
    p = CACHE_DIR / name
    p.write_text(contents, encoding="utf-8")
    return p


def _metrics_http_server_thread(host: str, port: int, metrics_supplier: callable) -> threading.Thread:
    class Handler(http.server.BaseHTTPRequestHandler):  # type: ignore
        def do_GET(self):
            if self.path != "/metrics":
                self.send_response(404)
                self.end_headers()
                return
            lines = []
            try:
                metrics = metrics_supplier()
                for k, v in metrics.items():
                    lines.append(f"{k} {v}")
                payload = "\n".join(lines) + "\n"
            except Exception as e:
                payload = f"# metrics error: {e}\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload.encode("utf-8"))

        def log_message(self, fmt, *args):  # silence
            return

    server = http.server.ThreadingHTTPServer((host, port), Handler)
    th = threading.Thread(target=server.serve_forever, daemon=True, name="instryx-metrics")
    th.start()
    LOG.info("Metrics HTTP server started at http://%s:%d/metrics", host, port)
    return th


class _PollingWatcher:
    """Simple polling-based file watcher (cross-platform)."""
    def __init__(self, poll_interval: float = 0.5):
        self._watched: Dict[Path, float] = {}
        self._poll_interval = poll_interval
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._cb = None
        self._thread: Optional[threading.Thread] = None

    def watch(self, path: Path, callback):
        path = path.resolve()
        with self._lock:
            self._watched[path] = path.stat().st_mtime if path.exists() else 0.0
            self._cb = callback
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, daemon=True, name="instryx-watcher")
                self._stop.clear()
                self._thread.start()

    def unwatch(self, path: Path):
        with self._lock:
            self._watched.pop(path.resolve(), None)
            if not self._watched:
                self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            with self._lock:
                for p, last in list(self._watched.items()):
                    try:
                        if p.exists():
                            m = p.stat().st_mtime
                            if m != last:
                                self._watched[p] = m
                                try:
                                    if self._cb:
                                        self._cb(p)
                                except Exception:
                                    LOG.exception("watch callback error")
                        else:
                            # removed
                            self._watched.pop(p, None)
                    except Exception:
                        continue
            time.sleep(self._poll_interval)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.2)


# Simple in-memory metrics for this CLI tool
_CLI_METRICS = {
    "instryxc_compiles_total": 0,
    "instryxc_runs_total": 0,
    "instryxc_emits_total": 0,
    "instryxc_errors_total": 0,
}


def _inc(metric: str, n: int = 1):
    _CLI_METRICS[metric] = _CLI_METRICS.get(metric, 0) + n


def compile_and_run(args: argparse.Namespace) -> int:
    try:
        code = args.file.read()
        if args.emit == "llvm":
            codegen = InstryxLLVMCodegen()
            llvm_ir = codegen.generate(code)
            print(llvm_ir)
            _inc("instryxc_emits_total")
            # cache IR
            _cache_write(f"{_file_hash(code)}.ll", llvm_ir)
            return 0
        if args.emit in ("exe", "wasm"):
            emitter = InstryxEmitter()
            emitter.emit(code, target=args.emit, output_name=args.output)
            _inc("instryxc_emits_total")
            return 0
        if args.emit == "ast":
            parser = InstryxParser()
            ast = parser.parse(code)
            print(ast)
            _inc("instryxc_emits_total")
            return 0
        if args.emit == "visual":
            viz = DodecagramExporter()
            viz.parse_code(code)
            viz.export_to_graphviz(f"{args.output}_ast")
            _inc("instryxc_emits_total")
            return 0
        if args.emit == "json":
            viz = DodecagramExporter()
            viz.parse_code(code)
            viz.export_to_json(f"{args.output}_ast.json")
            _inc("instryxc_emits_total")
            return 0
        if args.emit == "interpret":
            interpreter = InstryxInterpreter()
            interpreter.interpret(code)
            _inc("instryxc_emits_total")
            return 0
        if args.run:
            runner = InstryxRunner(verbose=args.verbose)
            _inc("instryxc_runs_total")
            runner.run(code, invoke_main=True, timeout=args.timeout)
            return 0
        print("Unknown emit mode:", args.emit)
        return 2
    except Exception as e:
        LOG.exception("compile_and_run failed")
        _inc("instryxc_errors_total", 1)
        print("Error:", e, file=sys.stderr)
        return 3


def _batch_process_directory(dirpath: Path, emit_mode: str, parallel: int = 4, output_prefix: Optional[str] = None) -> int:
    """
    Walk directory, find .ix files, and emit according to emit_mode (llvm/exe/wasm/ast/json/interpret/run).
    Runs in parallel threads.
    """
    dirpath = dirpath.resolve()
    if not dirpath.is_dir():
        LOG.error("Batch target is not a directory: %s", dirpath)
        return 2
    files = sorted(dirpath.rglob("*.ix"))
    if not files:
        LOG.info("No .ix files found under %s", dirpath)
        return 0
    LOG.info("Batch processing %d files with %d workers (emit=%s)", len(files), parallel, emit_mode)
    results = {}
    def _process(p: Path):
        with p.open("r", encoding="utf-8") as fh:
            code = fh.read()
        # small wrapper to avoid duplicating logic; reuse compile_and_run by creating a fake args namespace
        args = argparse.Namespace(file=type("F", (), {"read": lambda self=code: code})(), output=(output_prefix or p.stem), emit=emit_mode, run=(emit_mode=="run"), verbose=False, timeout=5)
        return p, compile_and_run(args)

    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = {ex.submit(_process, f): f for f in files}
        ok = 0
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                filep, rc = fut.result()
                results[str(filep)] = rc
                if rc == 0:
                    ok += 1
            except Exception as e:
                LOG.exception("batch processing %s failed", p)
                results[str(p)] = 3
    LOG.info("Batch finished: %d/%d successful", ok, len(files))
    return 0 if ok == len(files) else 1


def main():
    parser = argparse.ArgumentParser(description="🧠 Instryx CLI Compiler — supreme boosters edition")
    parser.add_argument("file", nargs="?", type=argparse.FileType("r"), help="Instryx source file (.ix). If omitted use --batch")
    parser.add_argument("-o", "--output", type=str, default="program", help="Output file name prefix")
    parser.add_argument("--emit", type=str, choices=["llvm", "exe", "wasm", "ast", "visual", "json", "interpret"], default="llvm",
                        help="What to emit: LLVM IR, binary, AST, visual, etc.")
    parser.add_argument("--run", action="store_true", help="JIT compile and run the code")
    parser.add_argument("--watch", "-w", action="store_true", help="Watch file for changes and re-run/re-emit on change")
    parser.add_argument("--metrics-port", type=int, default=0, help="Start local /metrics endpoint (port 0 disabled)")
    parser.add_argument("--batch", type=str, default=None, help="Batch process a directory of .ix files")
    parser.add_argument("--parallel", type=int, default=4, help="Parallel workers for batch mode")
    parser.add_argument("--timeout", type=int, default=10, help="Timeout seconds for running code")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        LOG.setLevel(logging.DEBUG)
        LOG.debug("Verbose mode enabled")

    # start metrics server if requested
    metrics_thread = None
    if args.metrics_port and args.metrics_port > 0:
        def _metrics_supplier():
            return _CLI_METRICS
        metrics_thread = _metrics_http_server_thread("127.0.0.1", args.metrics_port, _metrics_supplier)

    # batch mode
    if args.batch:
        rc = _batch_process_directory(Path(args.batch), emit_mode=args.emit, parallel=args.parallel, output_prefix=args.output)
        sys.exit(rc)

    if not args.file:
        LOG.error("No input file provided and no --batch specified")
        parser.print_help()
        sys.exit(2)

    # Normal compile/run path
    rc = compile_and_run(args)
    if args.watch:
        # start polling watcher and re-run on file change
        watcher = _PollingWatcher()
        src_path = Path(args.file.name).resolve()
        LOG.info("Watching %s for changes...", src_path)
        def _on_change(p: Path):
            LOG.info("Change detected in %s — re-running", p)
            # reopen file and run
            try:
                with p.open("r", encoding="utf-8") as fh:
                    code = fh.read()
                fake_args = argparse.Namespace(file=type("F", (), {"read": lambda self=code: code})(), output=args.output, emit=args.emit, run=args.run, verbose=args.verbose, timeout=args.timeout)
                compile_and_run(fake_args)
            except Exception:
                LOG.exception("Error during watched re-run")
        watcher.watch(src_path, _on_change)
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            LOG.info("Stopping watcher")
            watcher.stop()
    sys.exit(rc)


if __name__ == "__main__":
    main()

    # instryxc.py
    # Final CLI Compiler Wrapper for the Instryx Language — boosted edition

# instryxc.py
# Final CLI Compiler Wrapper for the Instryx Language — supreme boosters (final)
# Author: Violet Magenta / VACU Technologies (modified)
# License: MIT
"""
Supreme-boosters final edition — executable, self-contained CLI with enhanced tooling.

Features:
 - emit/run/interpret/ast/visual/json modes
 - watch mode (polling watcher) to recompile/re-run on change
 - batch mode for parallel processing of many .ix files
 - lightweight /metrics HTTP endpoint (Prometheus text format)
 - file-based IR cache under ~/.instryx_cli_cache
 - simple in-memory CLI metrics and optional metrics HTTP server
 - safe runner instantiation with graceful fallback (compatible with multiple InstryxRunner versions)
 - robust error handling and logging
"""

from __future__ import annotations
import argparse
import sys
import os
import time
import hashlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence, Dict, Any, Callable

# Import the components provided by the Instryx toolchain in the workspace.
from instryx_parser import InstryxParser
from instryx_ast_interpreter import InstryxInterpreter
from instryx_llvm_ir_codegen import InstryxLLVMCodegen
from instryx_jit_aot_runner import InstryxRunner
from instryx_wasm_and_exe_backend_emitter import InstryxEmitter
from instryx_dodecagram_ast_visualizer import DodecagramExporter
import http.server

LOG = logging.getLogger("instryxc")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Cache directory for emitted IR/artifacts
CACHE_DIR = Path.home() / ".instryx_cli_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Simple CLI metrics
_CLI_METRICS: Dict[str, int] = {
    "instryxc_compiles_total": 0,
    "instryxc_runs_total": 0,
    "instryxc_emits_total": 0,
    "instryxc_errors_total": 0,
}


def _inc(metric: str, n: int = 1) -> None:
    _CLI_METRICS[metric] = _CLI_METRICS.get(metric, 0) + n


def _file_hash(contents: str) -> str:
    return hashlib.sha256(contents.encode("utf-8")).hexdigest()


def _cache_write(name: str, contents: str) -> Path:
    p = CACHE_DIR / name
    p.write_text(contents, encoding="utf-8")
    return p


def _metrics_http_server_thread(host: str, port: int, metrics_supplier: Callable[[], Dict[str, int]]) -> threading.Thread:
    class Handler(http.server.BaseHTTPRequestHandler):  # type: ignore
        def do_GET(self):
            if self.path != "/metrics":
                self.send_response(404)
                self.end_headers()
                return
            try:
                metrics = metrics_supplier()
                payload = "\n".join(f"{k} {v}" for k, v in metrics.items()) + "\n"
            except Exception as e:
                payload = f"# metrics error: {e}\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload.encode("utf-8"))

        def log_message(self, format, *args):  # silence default logging
            return

    server = http.server.ThreadingHTTPServer((host, port), Handler)
    th = threading.Thread(target=server.serve_forever, daemon=True, name="instryx-metrics")
    th.start()
    LOG.info("Metrics HTTP server started at http://%s:%d/metrics", host, port)
    return th


class _PollingWatcher:
    """Cross-platform polling file watcher (small, lightweight)."""
    def __init__(self, poll_interval: float = 0.5):
        self._watched: Dict[Path, float] = {}
        self._poll_interval = poll_interval
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._cb = None
        self._thread: Optional[threading.Thread] = None

    def watch(self, path: Path, callback: Callable[[Path], None]) -> None:
        path = path.resolve()
        with self._lock:
            self._watched[path] = path.stat().st_mtime if path.exists() else 0.0
            self._cb = callback
            if self._thread is None or not self._thread.is_alive():
                self._stop.clear()
                self._thread = threading.Thread(target=self._run, daemon=True, name="instryx-watcher")
                self._thread.start()

    def unwatch(self, path: Path) -> None:
        with self._lock:
            self._watched.pop(path.resolve(), None)
            if not self._watched:
                self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                for p, last in list(self._watched.items()):
                    try:
                        if p.exists():
                            m = p.stat().st_mtime
                            if m != last:
                                self._watched[p] = m
                                try:
                                    if self._cb:
                                        self._cb(p)
                                except Exception:
                                    LOG.exception("watch callback error")
                        else:
                            self._watched.pop(p, None)
                    except Exception:
                        continue
            time.sleep(self._poll_interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.2)


def compile_and_run(args: argparse.Namespace) -> int:
    """
    Central compile/run function. Reads args.file (file-like with .read()).
    Returns process-like exit code (0 success, >0 error).
    """
    try:
        code = args.file.read()
        if args.emit == "llvm":
            codegen = InstryxLLVMCodegen()
            llvm_ir = codegen.generate(code)
            print(llvm_ir)
            _inc("instryxc_emits_total")
            _cache_write(f"{_file_hash(code)}.ll", llvm_ir)
            return 0

        if args.emit in ("exe", "wasm"):
            emitter = InstryxEmitter()
            emitter.emit(code, target=args.emit, output_name=args.output)
            _inc("instryxc_emits_total")
            return 0

        if args.emit == "ast":
            parser = InstryxParser()
            ast = parser.parse(code)
            print(ast)
            _inc("instryxc_emits_total")
            return 0

        if args.emit == "visual":
            viz = DodecagramExporter()
            viz.parse_code(code)
            viz.export_to_graphviz(f"{args.output}_ast")
            _inc("instryxc_emits_total")
            return 0

        if args.emit == "json":
            viz = DodecagramExporter()
            viz.parse_code(code)
            viz.export_to_json(f"{args.output}_ast.json")
            _inc("instryxc_emits_total")
            return 0

        if args.emit == "interpret":
            interpreter = InstryxInterpreter()
            interpreter.interpret(code)
            _inc("instryxc_emits_total")
            return 0

        if args.run:
            # instantiate runner with verbose if supported
            try:
                runner = InstryxRunner(verbose=args.verbose)  # newer runners accept verbose
            except TypeError:
                runner = InstryxRunner()  # fallback to basic runner
            _inc("instryxc_runs_total")
            runner.run(code, invoke_main=True, timeout=args.timeout)
            return 0

        print("Unknown emit mode:", args.emit)
        return 2

    except Exception as e:
        LOG.exception("compile_and_run failed")
        _inc("instryxc_errors_total", 1)
        print("Error:", e, file=sys.stderr)
        return 3


def _process_file_path(p: Path, emit_mode: str, output_prefix: Optional[str]) -> Tuple[str, int]:
    """
    Helper used by batch mode: returns (path_str, return_code)
    """
    try:
        with p.open("r", encoding="utf-8") as fh:
            code = fh.read()
        fake_file = type("F", (), {"read": lambda self=code: code})()
        args = argparse.Namespace(file=fake_file, output=(output_prefix or p.stem), emit=emit_mode, run=(emit_mode == "run"), verbose=False, timeout=5)
        rc = compile_and_run(args)
        return str(p), rc
    except Exception:
        LOG.exception("Batch processing file failed: %s", p)
        return str(p), 3


def _batch_process_directory(dirpath: Path, emit_mode: str, parallel: int = 4, output_prefix: Optional[str] = None) -> int:
    dirpath = dirpath.resolve()
    if not dirpath.is_dir():
        LOG.error("Batch target is not a directory: %s", dirpath)
        return 2
    files = sorted(dirpath.rglob("*.ix"))
    if not files:
        LOG.info("No .ix files found under %s", dirpath)
        return 0
    LOG.info("Batch processing %d files with %d workers (emit=%s)", len(files), parallel, emit_mode)
    ok = 0
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = {ex.submit(_process_file_path, f, emit_mode, output_prefix): f for f in files}
        results = {}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                path_str, rc = fut.result()
            except Exception:
                LOG.exception("Error processing %s", p)
                path_str, rc = str(p), 3
            results[path_str] = rc
            if rc == 0:
                ok += 1
    LOG.info("Batch finished: %d/%d successful", ok, len(files))
    return 0 if ok == len(files) else 1


def main():
    parser = argparse.ArgumentParser(description="🧠 Instryx CLI Compiler — supreme boosters")
    parser.add_argument("file", nargs="?", type=argparse.FileType("r"), help="Instryx source file (.ix). If omitted use --batch")
    parser.add_argument("-o", "--output", type=str, default="program", help="Output file name prefix")
    parser.add_argument("--emit", type=str, choices=["llvm", "exe", "wasm", "ast", "visual", "json", "interpret"], default="llvm",
                        help="What to emit: LLVM IR, binary, AST, visual, etc.")
    parser.add_argument("--run", action="store_true", help="JIT compile and run the code")
    parser.add_argument("--watch", "-w", action="store_true", help="Watch file for changes and re-run/re-emit on change")
    parser.add_argument("--metrics-port", type=int, default=0, help="Start local /metrics endpoint (port 0 disabled)")
    parser.add_argument("--batch", type=str, default=None, help="Batch process a directory of .ix files")
    parser.add_argument("--parallel", type=int, default=4, help="Parallel workers for batch mode")
    parser.add_argument("--timeout", type=int, default=10, help="Timeout seconds for running code")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        LOG.setLevel(logging.DEBUG)
        LOG.debug("Verbose mode enabled")

    metrics_thread = None
    if args.metrics_port and args.metrics_port > 0:
        metrics_thread = _metrics_http_server_thread("127.0.0.1", args.metrics_port, lambda: _CLI_METRICS)

    if args.batch:
        rc = _batch_process_directory(Path(args.batch), emit_mode=args.emit, parallel=args.parallel, output_prefix=args.output)
        sys.exit(rc)

    if not args.file:
        LOG.error("No input file provided and no --batch specified")
        parser.print_help()
        sys.exit(2)

    rc = compile_and_run(args)

    if args.watch:
        watcher = _PollingWatcher()
        src_path = Path(args.file.name).resolve()
        LOG.info("Watching %s for changes...", src_path)

        def _on_change(p: Path) -> None:
            LOG.info("Change detected in %s — re-running", p)
            try:
                with p.open("r", encoding="utf-8") as fh:
                    code = fh.read()
                fake_file = type("F", (), {"read": lambda self=code: code})()
                fake_args = argparse.Namespace(file=fake_file, output=args.output, emit=args.emit, run=args.run, verbose=args.verbose, timeout=args.timeout)
                compile_and_run(fake_args)
            except Exception:
                LOG.exception("Error during watched re-run")

        watcher.watch(src_path, _on_change)
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            LOG.info("Stopping watcher")
            watcher.stop()

    sys.exit(rc)


if __name__ == "__main__":
    main()

    # instryxc.py
    # Final CLI Compiler Wrapper for the Instryx Language — supreme boosters (final)

# instryxc.py
# Final CLI Compiler Wrapper for the Instryx Language — supreme boosters (enhanced)
# Author: Violet Magenta / VACU Technologies (modified)
# License: MIT
"""
Supreme boosters — additional tooling, caching, sandboxed runs, opt-levels, bench, config, and cleanup.
Fully implemented, pure-stdlib, executable.
"""

from __future__ import annotations
import argparse
import sys
import os
import time
import json
import hashlib
import logging
import threading
import tempfile
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence, Dict, Any, Callable, Tuple

from instryx_parser import InstryxParser
from instryx_ast_interpreter import InstryxInterpreter
from instryx_llvm_ir_codegen import InstryxLLVMCodegen
from instryx_jit_aot_runner import InstryxRunner
from instryx_wasm_and_exe_backend_emitter import InstryxEmitter
from instryx_dodecagram_ast_visualizer import DodecagramExporter
import http.server

LOG = logging.getLogger("instryxc")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Cache directory for emitted IR/artifacts
CACHE_DIR = Path.home() / ".instryx_cli_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Simple CLI metrics
_CLI_METRICS: Dict[str, int] = {
    "instryxc_compiles_total": 0,
    "instryxc_runs_total": 0,
    "instryxc_emits_total": 0,
    "instryxc_errors_total": 0,
    "instryxc_bench_runs": 0,
}


def _inc(metric: str, n: int = 1) -> None:
    _CLI_METRICS[metric] = _CLI_METRICS.get(metric, 0) + n


def _file_hash(contents: str, extra: Optional[Dict[str, Any]] = None) -> str:
    h = hashlib.sha256(contents.encode("utf-8"))
    if extra:
        h.update(json.dumps(extra, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def _cache_write(name: str, contents: str) -> Path:
    p = CACHE_DIR / name
    p.write_text(contents, encoding="utf-8")
    return p


def _cache_read(name: str) -> Optional[str]:
    p = CACHE_DIR / name
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return None


def _metrics_http_server_thread(host: str, port: int, metrics_supplier: Callable[[], Dict[str, int]]) -> threading.Thread:
    class Handler(http.server.BaseHTTPRequestHandler):  # type: ignore
        def do_GET(self):
            if self.path != "/metrics":
                self.send_response(404)
                self.end_headers()
                return
            try:
                metrics = metrics_supplier()
                payload = "\n".join(f"{k} {v}" for k, v in metrics.items()) + "\n"
            except Exception as e:
                payload = f"# metrics error: {e}\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload.encode("utf-8"))

        def log_message(self, format, *args):  # silence default logging
            return

    server = http.server.ThreadingHTTPServer((host, port), Handler)
    th = threading.Thread(target=server.serve_forever, daemon=True, name="instryx-metrics")
    th.start()
    LOG.info("Metrics HTTP server started at http://%s:%d/metrics", host, port)
    return th


class _PollingWatcher:
    """Cross-platform polling file watcher (small, lightweight)."""
    def __init__(self, poll_interval: float = 0.5):
        self._watched: Dict[Path, float] = {}
        self._poll_interval = poll_interval
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._cb = None
        self._thread: Optional[threading.Thread] = None

    def watch(self, path: Path, callback: Callable[[Path], None]) -> None:
        path = path.resolve()
        with self._lock:
            self._watched[path] = path.stat().st_mtime if path.exists() else 0.0
            self._cb = callback
            if self._thread is None or not self._thread.is_alive():
                self._stop.clear()
                self._thread = threading.Thread(target=self._run, daemon=True, name="instryx-watcher")
                self._thread.start()

    def unwatch(self, path: Path) -> None:
        with self._lock:
            self._watched.pop(path.resolve(), None)
            if not self._watched:
                self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                for p, last in list(self._watched.items()):
                    try:
                        if p.exists():
                            m = p.stat().st_mtime
                            if m != last:
                                self._watched[p] = m
                                try:
                                    if self._cb:
                                        self._cb(p)
                                except Exception:
                                    LOG.exception("watch callback error")
                        else:
                            self._watched.pop(p, None)
                    except Exception:
                        continue
            time.sleep(self._poll_interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.2)


# -------------------------------------------------------------------------
# Core operations
# -------------------------------------------------------------------------
def emit_llvm(code: str, opt_level: int = 0, force: bool = False) -> Tuple[int, Optional[str]]:
    meta = {"opt_level": int(opt_level)}
    key = _file_hash(code, extra=meta)
    cache_name = f"{key}.ll"
    if not force:
        cached = _cache_read(cache_name)
        if cached is not None:
            LOG.debug("Using cached IR for key=%s", key)
            return 0, cached
    codegen = InstryxLLVMCodegen()
    try:
        llvm_ir = codegen.generate(code, opt_level=opt_level) if hasattr(codegen, "generate") else codegen.generate(code)
    except TypeError:
        # fallback if generate doesn't accept opt_level
        llvm_ir = codegen.generate(code)
    _cache_write(cache_name, llvm_ir)
    _inc("instryxc_compiles_total", 1)
    return 0, llvm_ir


def run_in_sandbox(code: str, timeout: int = 10) -> int:
    """
    Run code in a separate Python process to isolate the host.
    Uses InstryxRunner in the subprocess if available.
    """
    stub = f"""
import sys
from instryx_jit_aot_runner import InstryxRunner
runner = InstryxRunner()
code = r'''{code.replace("'''", "\\'\\'\\'")}'''
try:
    runner.run(code, invoke_main=True, timeout={timeout})
    sys.exit(0)
except Exception as e:
    print("sandbox error:", e, file=sys.stderr)
    sys.exit(2)
"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(stub)
        path = tf.name
    try:
        proc = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout + 5)
        if proc.returncode != 0:
            LOG.error("Sandbox run failed: %s", proc.stderr.strip())
        else:
            LOG.debug("Sandbox stdout: %s", proc.stdout.strip())
        return proc.returncode
    except subprocess.TimeoutExpired:
        LOG.error("Sandbox timed out")
        return 124
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


def clean_cache() -> int:
    try:
        if CACHE_DIR.exists():
            for f in CACHE_DIR.iterdir():
                try:
                    if f.is_file():
                        f.unlink()
                    elif f.is_dir():
                        shutil.rmtree(f)
                except Exception:
                    LOG.debug("Failed to remove cache item %s", f)
        LOG.info("Cache cleared at %s", CACHE_DIR)
        return 0
    except Exception:
        LOG.exception("clean_cache failed")
        return 2


def bench_compile_and_run(code: str, iterations: int = 5, warmup: int = 1, opt_level: int = 0, isolate: bool = False) -> Dict[str, Any]:
    """Benchmark compile + run latency (simple)"""
    times = []
    for i in range(iterations):
        if i < warmup:
            # warmup run (not measured)
            if isolate:
                run_in_sandbox(code, timeout=5)
            else:
                try:
                    runner = InstryxRunner()
                    runner.run(code, invoke_main=True, timeout=5)
                except Exception:
                    pass
            continue
        t0 = time.time()
        if isolate:
            run_in_sandbox(code, timeout=5)
        else:
            try:
                runner = InstryxRunner()
                runner.run(code, invoke_main=True, timeout=5)
            except Exception:
                pass
        times.append(time.time() - t0)
        _inc("instryxc_bench_runs", 1)
    stats = {"runs": len(times), "avg_s": (sum(times) / len(times)) if times else None, "min_s": min(times) if times else None, "max_s": max(times) if times else None}
    return stats


# -------------------------------------------------------------------------
# CLI glue (compile_and_run upgraded)
# -------------------------------------------------------------------------
def compile_and_run(args: argparse.Namespace) -> int:
    try:
        code = args.file.read()
        # support opt-level and force flags if present
        opt_level = getattr(args, "opt_level", 0)
        force = getattr(args, "force", False)
        isolate = getattr(args, "isolate", False)
        # emit LLVM (with caching)
        if args.emit == "llvm":
            rc, llvm_ir = emit_llvm(code, opt_level=opt_level, force=force)
            if rc == 0 and llvm_ir:
                print(llvm_ir)
            _inc("instryxc_emits_total")
            return 0 if rc == 0 else rc

        if args.emit in ("exe", "wasm"):
            emitter = InstryxEmitter()
            emitter.emit(code, target=args.emit, output_name=args.output, opt_level=opt_level) if hasattr(emitter, "emit") else emitter.emit(code, target=args.emit, output_name=args.output)
            _inc("instryxc_emits_total")
            return 0

        if args.emit == "ast":
            parser = InstryxParser()
            ast = parser.parse(code)
            print(ast)
            _inc("instryxc_emits_total")
            return 0

        if args.emit == "visual":
            viz = DodecagramExporter()
            viz.parse_code(code)
            viz.export_to_graphviz(f"{args.output}_ast")
            _inc("instryxc_emits_total")
            return 0

        if args.emit == "json":
            viz = DodecagramExporter()
            viz.parse_code(code)
            viz.export_to_json(f"{args.output}_ast.json")
            _inc("instryxc_emits_total")
            return 0

        if args.emit == "interpret":
            interpreter = InstryxInterpreter()
            interpreter.interpret(code)
            _inc("instryxc_emits_total")
            return 0

        if args.run:
            _inc("instryxc_runs_total")
            if isolate:
                return run_in_sandbox(code, timeout=args.timeout)
            # prefer runner with verbose support, fallback otherwise
            try:
                runner = InstryxRunner(verbose=args.verbose)
            except TypeError:
                runner = InstryxRunner()
            runner.run(code, invoke_main=True, timeout=args.timeout, use_subprocess=False)
            return 0

        print("Unknown emit mode:", args.emit)
        return 2

    except Exception as e:
        LOG.exception("compile_and_run failed")
        _inc("instryxc_errors_total", 1)
        print("Error:", e, file=sys.stderr)
        return 3


# -------------------------------------------------------------------------
# Batch helpers
# -------------------------------------------------------------------------
def _process_file_path(p: Path, emit_mode: str, output_prefix: Optional[str], force: bool, opt_level: int, isolate: bool) -> Tuple[str, int]:
  
        with p.open("r", encoding="utf-8") as fh:
            code = fh.read()
        fake_file = type("F", (), {"read": lambda self=code: code})()
instryxc.py
# instryxc.py
# Final CLI Compiler Wrapper for the Instryx Language — clean, enhanced, executable
# Author: Violet Magenta / VACU Technologies (modified)
# License: MIT

"""
Supreme-boosters — implemented safely and cleanly.

Features:
 - emit/run/interpret/ast/visual/json modes
 - watch mode (polling watcher) to recompile/re-run on change
 - batch mode for parallel processing of many .ix files
 - lightweight /metrics HTTP endpoint (Prometheus text format)
 - file-based IR cache under ~/.instryx_cli_cache
 - opt-level, force, isolate, bench, clean-cache CLI flags
 - sandboxed subprocess runs for isolation
 - pure-stdlib, defensive, fully executable
"""

from __future__ import annotations
import argparse
import sys
import os
import time
import json
import hashlib
import logging
import threading
import tempfile
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence, Dict, Any, Callable, Tuple

# Project imports (assume available in workspace)
from instryx_parser import InstryxParser
from instryx_ast_interpreter import InstryxInterpreter
from instryx_llvm_ir_codegen import InstryxLLVMCodegen
from instryx_jit_aot_runner import InstryxRunner
from instryx_wasm_and_exe_backend_emitter import InstryxEmitter
from instryx_dodecagram_ast_visualizer import DodecagramExporter
import http.server

LOG = logging.getLogger("instryxc")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Cache directory for emitted IR/artifacts
CACHE_DIR = Path.home() / ".instryx_cli_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Simple CLI metrics
_CLI_METRICS: Dict[str, int] = {
    "instryxc_compiles_total": 0,
    "instryxc_runs_total": 0,
    "instryxc_emits_total": 0,
    "instryxc_errors_total": 0,
    "instryxc_bench_runs": 0,
}


def _inc(metric: str, n: int = 1) -> None:
    _CLI_METRICS[metric] = _CLI_METRICS.get(metric, 0) + n


def _file_hash(contents: str, extra: Optional[Dict[str, Any]] = None) -> str:
    h = hashlib.sha256(contents.encode("utf-8"))
    if extra:
        h.update(json.dumps(extra, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def _cache_write(name: str, contents: str) -> Path:
    p = CACHE_DIR / name
    p.write_text(contents, encoding="utf-8")
    return p


def _cache_read(name: str) -> Optional[str]:
    p = CACHE_DIR / name
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return None


def _metrics_http_server_thread(host: str, port: int, metrics_supplier: Callable[[], Dict[str, int]]) -> threading.Thread:
    class Handler(http.server.BaseHTTPRequestHandler):  # type: ignore
        def do_GET(self):
            if self.path != "/metrics":
                self.send_response(404)
                self.end_headers()
                return
            try:
                metrics = metrics_supplier()
                payload = "\n".join(f"{k} {v}" for k, v in metrics.items()) + "\n"
            except Exception as e:
                payload = f"# metrics error: {e}\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload.encode("utf-8"))

        def log_message(self, format, *args):  # silence default logging
            return

    server = http.server.ThreadingHTTPServer((host, port), Handler)
    th = threading.Thread(target=server.serve_forever, daemon=True, name="instryx-metrics")
    th.start()
    LOG.info("Metrics HTTP server started at http://%s:%d/metrics", host, port)
    return th


class _PollingWatcher:
    """Cross-platform polling file watcher (small, lightweight)."""
    def __init__(self, poll_interval: float = 0.5):
        self._watched: Dict[Path, float] = {}
        self._poll_interval = poll_interval
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._cb: Optional[Callable[[Path], None]] = None
        self._thread: Optional[threading.Thread] = None

    def watch(self, path: Path, callback: Callable[[Path], None]) -> None:
        path = path.resolve()
        with self._lock:
            self._watched[path] = path.stat().st_mtime if path.exists() else 0.0
            self._cb = callback
            if self._thread is None or not self._thread.is_alive():
                self._stop.clear()
                self._thread = threading.Thread(target=self._run, daemon=True, name="instryx-watcher")
                self._thread.start()

    def unwatch(self, path: Path) -> None:
        with self._lock:
            self._watched.pop(path.resolve(), None)
            if not self._watched:
                self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                for p, last in list(self._watched.items()):
                    try:
                        if p.exists():
                            m = p.stat().st_mtime
                            if m != last:
                                self._watched[p] = m
                                try:
                                    if self._cb:
                                        self._cb(p)
                                except Exception:
                                    LOG.exception("watch callback error")
                        else:
                            self._watched.pop(p, None)
                    except Exception:
                        continue
            time.sleep(self._poll_interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.2)


# -------------------------------------------------------------------------
# Core operations
# -------------------------------------------------------------------------
def emit_llvm(code: str, opt_level: int = 0, force: bool = False) -> Tuple[int, Optional[str]]:
    meta = {"opt_level": int(opt_level)}
    key = _file_hash(code, extra=meta)
    cache_name = f"{key}.ll"
    if not force:
        cached = _cache_read(cache_name)
        if cached is not None:
            LOG.debug("Using cached IR for key=%s", key)
            return 0, cached
    codegen = InstryxLLVMCodegen()
    try:
        # prefer generate(code, opt_level) if available
        if hasattr(codegen.generate, "__call__"):
            try:
                llvm_ir = codegen.generate(code, opt_level=opt_level)  # type: ignore[arg-type]
            except TypeError:
                llvm_ir = codegen.generate(code)  # type: ignore[call-arg]
        else:
            llvm_ir = codegen.generate(code)
    except Exception:
        LOG.exception("emit_llvm failed")
        return 2, None
    _cache_write(cache_name, llvm_ir)
    _inc("instryxc_compiles_total", 1)
    return 0, llvm_ir


def run_in_sandbox(code: str, timeout: int = 10) -> int:
    """
    Run code in a separate Python process to isolate the host.
    Implementation writes the code to a temp file and runs a small runner stub that reads it.
    """
    with tempfile.TemporaryDirectory() as td:
        code_path = Path(td) / "program.ix"
        stub_path = Path(td) / "runner_stub.py"
        code_path.write_text(code, encoding="utf-8")
        stub = (
            "import sys\n"
            "from instryx_jit_aot_runner import InstryxRunner\n"
            "p = sys.argv[1]\n"
            "with open(p, 'r', encoding='utf-8') as fh:\n"
            "    code = fh.read()\n"
            "runner = InstryxRunner()\n"
            "try:\n"
            "    runner.run(code, invoke_main=True, timeout=%d)\n"
            "    sys.exit(0)\n"
            "except Exception as e:\n"
            "    print('sandbox error:', e, file=sys.stderr)\n"
            "    sys.exit(2)\n"
        ) % (timeout,)
        stub_path.write_text(stub, encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, str(stub_path), str(code_path)], capture_output=True, text=True, timeout=timeout + 5)
            if proc.returncode != 0:
                LOG.error("Sandbox run failed: %s", proc.stderr.strip())
            else:
                LOG.debug("Sandbox stdout: %s", proc.stdout.strip())
            return proc.returncode
        except subprocess.TimeoutExpired:
            LOG.error("Sandbox timed out")
            return 124


def clean_cache() -> int:
    try:
        if CACHE_DIR.exists():
            for f in CACHE_DIR.iterdir():
                try:
                    if f.is_file():
                        f.unlink()
                    elif f.is_dir():
                        shutil.rmtree(f)
                except Exception:
                    LOG.debug("Failed to remove cache item %s", f)
        LOG.info("Cache cleared at %s", CACHE_DIR)
        return 0
    except Exception:
        LOG.exception("clean_cache failed")
        return 2


def bench_compile_and_run(code: str, iterations: int = 5, warmup: int = 1, opt_level: int = 0, isolate: bool = False) -> Dict[str, Any]:
    """Benchmark compile + run latency (simple)"""
    times = []
    for i in range(iterations):
        if i < warmup:
            # warmup run (not measured)
            if isolate:
                run_in_sandbox(code, timeout=5)
            else:
                try:
                    runner = InstryxRunner()
                    runner.run(code, invoke_main=True, timeout=5)
                except Exception:
                    pass
            continue
        t0 = time.time()
        if isolate:
            run_in_sandbox(code, timeout=5)
        else:
            try:
                runner = InstryxRunner()
                runner.run(code, invoke_main=True, timeout=5)
            except Exception:
                pass
        times.append(time.time() - t0)
        _inc("instryxc_bench_runs", 1)
    stats = {"runs": len(times), "avg_s": (sum(times) / len(times)) if times else None, "min_s": min(times) if times else None, "max_s": max(times) if times else None}
    return stats


# -------------------------------------------------------------------------
# CLI glue
# -------------------------------------------------------------------------
def compile_and_run(args: argparse.Namespace) -> int:
    try:
        code = args.file.read()
        opt_level = getattr(args, "opt_level", 0)
        force = getattr(args, "force", False)
        isolate = getattr(args, "isolate", False)

        if args.emit == "llvm":
            rc, llvm_ir = emit_llvm(code, opt_level=opt_level, force=force)
            if rc == 0 and llvm_ir:
                print(llvm_ir)
            _inc("instryxc_emits_total")
            return 0 if rc == 0 else rc

        if args.emit in ("exe", "wasm"):
            emitter = InstryxEmitter()
            # prefer opt-level if supported
            try:
                emitter.emit(code, target=args.emit, output_name=args.output, opt_level=opt_level)  # type: ignore[arg-type]
            except TypeError:
                emitter.emit(code, target=args.emit, output_name=args.output)
            _inc("instryxc_emits_total")
            return 0

        if args.emit == "ast":
            parser = InstryxParser()
            ast = parser.parse(code)
            print(ast)
            _inc("instryxc_emits_total")
            return 0

        if args.emit == "visual":
            viz = DodecagramExporter()
            viz.parse_code(code)
            viz.export_to_graphviz(f"{args.output}_ast")
            _inc("instryxc_emits_total")
            return 0

        if args.emit == "json":
            viz = DodecagramExporter()
            viz.parse_code(code)
            viz.export_to_json(f"{args.output}_ast.json")
            _inc("instryxc_emits_total")
            return 0

        if args.emit == "interpret":
            interpreter = InstryxInterpreter()
            interpreter.interpret(code)
            _inc("instryxc_emits_total")
            return 0

        if args.run:
            _inc("instryxc_runs_total")
            if isolate:
                return run_in_sandbox(code, timeout=args.timeout)
            try:
                runner = InstryxRunner(verbose=args.verbose)
            except TypeError:
                runner = InstryxRunner()
            runner.run(code, invoke_main=True, timeout=args.timeout, use_subprocess=False)
            return 0

        print("Unknown emit mode:", args.emit)
        return 2

    except Exception:
        LOG.exception("compile_and_run failed")
        _inc("instryxc_errors_total", 1)
        print("Error during compile_and_run", file=sys.stderr)
        return 3


# -------------------------------------------------------------------------
# Batch helpers
# -------------------------------------------------------------------------
def _process_file_path(p: Path, emit_mode: str, output_prefix: Optional[str], force: bool, opt_level: int, isolate: bool) -> Tuple[str, int]:
    try:
        with p.open("r", encoding="utf-8") as fh:
            code = fh.read()
        fake_file = type("F", (), {"read": lambda self=code: code})()
        args = argparse.Namespace(file=fake_file, output=(output_prefix or p.stem), emit=emit_mode, run=(emit_mode == "run"),
                                  verbose=False, timeout=5, force=force, opt_level=opt_level, isolate=isolate)
        rc = compile_and_run(args)
        return str(p), rc
    except Exception:
        LOG.exception("Batch processing file failed: %s", p)
        return str(p), 3


def _batch_process_directory(dirpath: Path, emit_mode: str, parallel: int = 4, output_prefix: Optional[str] = None, force: bool = False, opt_level: int = 0, isolate: bool = False) -> int:
    dirpath = dirpath.resolve()
    if not dirpath.is_dir():
        LOG.error("Batch target is not a directory: %s", dirpath)
        return 2
    files = sorted(dirpath.rglob("*.ix"))
    if not files:
        LOG.info("No .ix files found under %s", dirpath)
        return 0
    LOG.info("Batch processing %d files with %d workers (emit=%s)", len(files), parallel, emit_mode)
    ok = 0
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = {ex.submit(_process_file_path, f, emit_mode, output_prefix, force, opt_level, isolate): f for f in files}
        results: Dict[str, int] = {}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                path_str, rc = fut.result()
            except Exception:
                LOG.exception("Error processing %s", p)
                path_str, rc = str(p), 3
            results[path_str] = rc
            if rc == 0:
                ok += 1
    LOG.info("Batch finished: %d/%d successful", ok, len(files))
    return 0 if ok == len(files) else 1


# -------------------------------------------------------------------------
# CLI entrypoint
# -------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="🧠 Instryx CLI Compiler — supreme boosters")
    parser.add_argument("file", nargs="?", type=argparse.FileType("r"), help="Instryx source file (.ix). If omitted use --batch")
    parser.add_argument("-o", "--output", type=str, default="program", help="Output file name prefix")
    parser.add_argument("--emit", type=str, choices=["llvm", "exe", "wasm", "ast", "visual", "json", "interpret"], default="llvm",
                        help="What to emit: LLVM IR, binary, AST, visual, etc.")
    parser.add_argument("--run", action="store_true", help="JIT compile and run the code")
    parser.add_argument("--watch", "-w", action="store_true", help="Watch file for changes and re-run/re-emit on change")
    parser.add_argument("--metrics-port", type=int, default=0, help="Start local /metrics endpoint (port 0 disabled)")
    parser.add_argument("--batch", type=str, default=None, help="Batch process a directory of .ix files")
    parser.add_argument("--parallel", type=int, default=4, help="Parallel workers for batch mode")
    parser.add_argument("--timeout", type=int, default=10, help="Timeout seconds for running code")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--force", action="store_true", help="Bypass cache and force recompilation")
    parser.add_argument("--opt-level", type=int, default=0, help="Optimization level for codegen/emitter (0-3)")
    parser.add_argument("--isolate", action="store_true", help="Run program in isolated subprocess sandbox")
    parser.add_argument("--clean-cache", action="store_true", help="Remove CLI cache and exit")
    parser.add_argument("--bench", action="store_true", help="Run quick compile+run benchmark on the file")
    args = parser.parse_args(argv)

    if args.verbose:
        LOG.setLevel(logging.DEBUG)
        LOG.debug("Verbose mode enabled")

    if args.clean_cache:
        return clean_cache()

    metrics_thread = None
    if args.metrics_port and args.metrics_port > 0:
        metrics_thread = _metrics_http_server_thread("127.0.0.1", args.metrics_port, lambda: _CLI_METRICS)

    if args.batch:
        rc = _batch_process_directory(Path(args.batch), emit_mode=args.emit, parallel=args.parallel, output_prefix=args.output, force=args.force, opt_level=args.opt_level, isolate=args.isolate)
        return rc

    if not args.file:
        LOG.error("No input file provided and no --batch specified")
        parser.print_help()
        return 2

    rc = compile_and_run(args)

    if args.bench:
        with args.file as fh:
            code = fh.read()
        stats = bench_compile_and_run(code, iterations=5, warmup=1, opt_level=args.opt_level, isolate=args.isolate)
        print("Bench stats:", json.dumps(stats, indent=2))

    if args.watch:
        watcher = _PollingWatcher()
        src_path = Path(args.file.name).resolve()
        LOG.info("Watching %s for changes...", src_path)

        def _on_change(p: Path) -> None:
            LOG.info("Change detected in %s — re-running", p)
            try:
                with p.open("r", encoding="utf-8") as fh:
                    code = fh.read()
                fake_file = type("F", (), {"read": lambda self=code: code})()
                fake_args = argparse.Namespace(file=fake_file, output=args.output, emit=args.emit, run=args.run, verbose=args.verbose, timeout=args.timeout, force=args.force, opt_level=args.opt_level, isolate=args.isolate)
                compile_and_run(fake_args)
            except Exception:
                LOG.exception("Error during watched re-run")

        watcher.watch(src_path, _on_change)
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            LOG.info("Stopping watcher")
            watcher.stop()

    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        LOG.exception("Fatal error in instryxc")
        sys.exit(3)

        def _process_file_path(p: Path, emit_mode: str, output_prefix: Optional[str], force: bool, opt_level: int, isolate: bool) -> Tuple[str, int]:
            try:
                with p.open("r", encoding="utf-8") as fh:
                    code = fh.read()
                fake_file = type("F", (), {"read": lambda self=code: code})()
                args = argparse.Namespace(file=fake_file, output=(output_prefix or p.stem), emit=emit_mode, run=(emit_mode == "run"),
                                          verbose=False, timeout=5, force=force, opt_level=opt_level, isolate=isolate)
                rc = compile_and_run(args)
                return str(p), rc
            except Exception:
                LOG.exception("Batch processing file failed: %s", p)
                return str(p), 3
            except Exception:
                LOG.exception("Batch processing file failed: %s", p)
                return str(p), 3

"""
instryx_wasm_host_runtime.py

Lightweight Wasm host runtime tailored for Instryx-produced WebAssembly modules.
Features:
- Uses wasmtime (pip install wasmtime) to load/instantiate WASM modules with WASI.
- Provides common host functions expected by Instryx-compiled WASM:
  - host.log(ptr, len)
  - host.time_now() -> i64 milliseconds
  - host.fail(ptr, len) -> traps the guest (raises HostTrap)
  - host.system_get(ptr, len) -> i32 (pointer to allocated guest string)
- Helpers for string <-> guest memory marshalling using guest `alloc` export.
- Simple CLI and demo usage.

Notes:
- This is a pragmatic host shim. For full reliability integrate with the Instryx
  ABI expectations (allocator name, memory layout, calling convention).
- If `wasmtime` is not installed the module will raise an informative ImportError.
"""

from typing import Optional, Callable, Tuple
import time
import sys
import json

try:
    from wasmtime import (
        Store,
        Module,
        Linker,
        Instance,
        Func,
        FuncType,
        ValType,
        Memory,
        Caller,
        WasiConfig,
        Trap,
    )
except Exception as e:
    raise ImportError(
        "wasmtime Python bindings are required. Install with: pip install wasmtime\n"
        f"Underlying import error: {e}"
    )


class WasmHostRuntime:
    """
    Minimal Wasm host runtime for Instryx modules.
    Usage:
      runtime = WasmHostRuntime()
      runtime.instantiate("module.wasm")
      runtime.call("main")
    """

    def __init__(self, enable_wasi: bool = True):
        self.store = Store()
        self.linker = Linker(self.store.engine)
        self.module: Optional[Module] = None
        self.instance: Optional[Instance] = None
        self.memory: Optional[Memory] = None
        self.enable_wasi = enable_wasi

        if enable_wasi:
            wasi_cfg = WasiConfig()
            wasi_cfg.inherit_stdout()
            wasi_cfg.inherit_stderr()
            # keep environment minimal; user can extend
            self.store.set_wasi(wasi_cfg)

        # register host functions under module name "host"
        self._register_host_functions()

    # -----------------------
    # Host functions
    # -----------------------
    def _register_host_functions(self):
        # log(ptr: i32, len: i32)
        def _log(caller: Caller, ptr: int, length: int):
            try:
                mem = self._get_memory_from_caller(caller)
                data = mem.read(self.store, ptr, length)
                print(data.decode("utf-8", errors="replace"))
            except Exception as e:
                print(f"[host.log] error reading memory: {e}", file=sys.stderr)

        # time_now() -> i64 milliseconds
        def _time_now() -> int:
            return int(time.time() * 1000)

        # fail(ptr: i32, len: i32) -> trap
        def _fail(caller: Caller, ptr: int, length: int):
            try:
                mem = self._get_memory_from_caller(caller)
                data = mem.read(self.store, ptr, length)
                msg = data.decode("utf-8", errors="replace")
            except Exception:
                msg = "<failed to read message>"
            raise Trap(f"host.fail called: {msg}")

        # system_get(ptr: i32, len: i32) -> i32 (pointer in guest memory)
        # Host will allocate memory in guest via exported `alloc` and write the string there.
        def _system_get(caller: Caller, ptr: int, length: int) -> int:
            try:
                mem = self._get_memory_from_caller(caller)
                key = mem.read(self.store, ptr, length).decode("utf-8")
                # Resolve some simple system values
                value = self._resolve_system_key(key)
                # Serialize as JSON string if complex
                if not isinstance(value, (str, bytes)):
                    value = json.dumps(value)
                if isinstance(value, str):
                    value_bytes = value.encode("utf-8")
                else:
                    value_bytes = value
                # allocate in guest
                alloc = self._get_exported_alloc()
                if alloc is None:
                    raise RuntimeError("Guest module does not expose an 'alloc' export required for system_get")
                guest_ptr = alloc(len(value_bytes))
                # write into guest memory
                mem.write(self.store, guest_ptr, value_bytes)
                return guest_ptr
            except Exception as e:
                raise Trap(f"host.system_get error: {e}")

        # Register functions on linker under "host"
        self.linker.define("host", "log", Func(self.store, FuncType([ValType.i32(), ValType.i32()], []), _log))
        self.linker.define("host", "time_now", Func(self.store, FuncType([], [ValType.i64()]), _time_now))
        self.linker.define("host", "fail", Func(self.store, FuncType([ValType.i32(), ValType.i32()], []), _fail))
        self.linker.define("host", "system_get", Func(self.store, FuncType([ValType.i32(), ValType.i32()], [ValType.i32()]), _system_get))

    def _get_memory_from_caller(self, caller: Caller) -> Memory:
        # try to find memory exported by the instance or the caller
        mem = None
        try:
            mem = caller.get_export("memory")
        except Exception:
            mem = None
        if mem is None:
            # fallback to previously cached memory
            if self.memory is None:
                raise RuntimeError("Wasm memory not available")
            return self.memory
        return mem

    # -----------------------
    # Module lifecycle
    # -----------------------
    def instantiate(self, wasm_path: str):
        """
        Load and instantiate a wasm module from file.
        After instantiation exported memory and functions will be available on the runtime.
        """
        self.module = Module(self.store.engine, open(wasm_path, "rb").read())
        self.instance = self.linker.instantiate(self.store, self.module)
        # attempt to locate linear memory
        try:
            mem = self.instance.get_export(self.store, "memory")
            if isinstance(mem, Memory):
                self.memory = mem
        except Exception:
            self.memory = None

    # -----------------------
    # Helpers: allocator/exports
    # -----------------------
    def _get_exported_alloc(self) -> Optional[Callable[[int], int]]:
        """
        Returns a callable alloc(size)->ptr if the guest exports one named 'alloc' or '_alloc'.
        The returned function will be invoked with (size) and return an integer pointer.
        """
        if self.instance is None:
            return None
        alloc_export = None
        for name in ("alloc", "_alloc", "ix_alloc"):
            try:
                fn = self.instance.get_export(self.store, name)
                if fn is not None:
                    alloc_export = fn
                    break
            except Exception:
                continue
        if alloc_export is None:
            return None

        def alloc_fn(size: int) -> int:
            res = alloc_export(self.store, size)
            # Some wasm allocs return i32 or i64; ensure int
            return int(res) if res is not None else 0

        return alloc_fn

    # -----------------------
    # High-level call helpers
    # -----------------------
    def call(self, func_name: str, *args):
        """
        Call an exported function with numeric args. Returns raw result(s).
        """
        if not self.instance:
            raise RuntimeError("Module not instantiated")
        fn = self.instance.get_export(self.store, func_name)
        if fn is None:
            raise RuntimeError(f"Export {func_name} not found")
        return fn(self.store, *args)

    def call_with_strings(self, func_name: str, str_args: Tuple[str, ...]) -> Optional[str]:
        """
        Call exported function that accepts string pointers (ptr,len) pairs for each argument.
        The function is expected to return a pointer to a result string allocated via guest `alloc`.
        This helper:
          - finds guest alloc
          - writes strings into guest memory
          - calls function passing (ptr, len) pairs
          - reads returned pointer as zero-terminated/length-unknown string by reading until next null
            or (if guest also returns length via convention) this helper can be extended.
        """
        alloc = self._get_exported_alloc()
        if alloc is None:
            raise RuntimeError("Guest module does not export an 'alloc' function required for passing strings")

        if not self.instance or self.memory is None:
            raise RuntimeError("Module not instantiated or memory unavailable")

        mem = self.memory
        ptrs_and_lens = []
        for s in str_args:
            b = s.encode("utf-8")
            ptr = alloc(len(b))
            mem.write(self.store, ptr, b)
            ptrs_and_lens.extend([ptr, len(b)])

        # Call
        res = self.call(func_name, *ptrs_and_lens)

        # If result is integer pointer, attempt to read a length-prefixed string or null-terminated.
        try:
            res_ptr = int(res)
        except Exception:
            return None

        # Attempt to read a length-prefixed value by convention: many ABI's return (ptr,len) but here we only have ptr.
        # Strategy: try to read until a NUL (0) up to a reasonable limit
        max_read = 4096
        collected = bytearray()
        for i in range(max_read):
            try:
                b = mem.read(self.store, res_ptr + i, 1)
            except Exception:
                break
            if not b:
                break
            if b[0] == 0:
                break
            collected.append(b[0])
        return collected.decode("utf-8", errors="replace")

    # -----------------------
    # Utilities & resolution
    # -----------------------
    def _resolve_system_key(self, key: str):
        """
        Provide simple system-level keys. Extend to integrate with host services.
        Examples supported:
          - "net.api" -> placeholder info
          - "time" -> current time value
        """
        if key == "time.now":
            return int(time.time() * 1000)
        if key == "host.env":
            return dict()  # empty env placeholder
        if key == "net.api":
            return {"base_url": "https://example.com/api"}
        # default: return an informative string
        return f"<system:{key}"

    # -----------------------
    # CLI / Demo
    # -----------------------
def _demo_main():
    import argparse
    p = argparse.ArgumentParser(description="Instryx Wasm host runtime demo")
    p.add_argument("wasm", help="WASM module file path")
    p.add_argument("--call", help="Exported function to call (default 'main')", default="main")
    p.add_argument("--args", help="Comma-separated string args", default="")
    args = p.parse_args()

    rt = WasmHostRuntime(enable_wasi=True)
    print(f"Loading {args.wasm} ...")
    rt.instantiate(args.wasm)
    func = args.call
    if args.args:
        sargs = tuple(a for a in args.args.split(","))
        out = rt.call_with_strings(func, sargs)
        print("Call result (string):", out)
    else:
        try:
            res = rt.call(func)
            print("Call result:", res)
        except Trap as t:
            print("Guest trapped:", t)

if __name__ == "__main__":
    _demo_main()
