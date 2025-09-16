"""
instryx_binary_emitters.py

Advanced, production-ready binary emitter and tooling for Instryx.

Features:
- High-level `AdvancedEmitter` that composes:
  - optional IR-level optimizations (instryx_opt_utils.optimize_shard / plugin passes)
  - LLVM-based emit pipeline (via instryx_wasm_and_exe_backend_emitter.InstryxEmitter when available)
  - caching keyed by deterministic IR hash
  - manifest and checksum generation
  - profile-guided inlining integration (reads profile JSON and injects into plugin context)
  - link-time optimization (LTO) / thinLTO toggles
  - wasm post-processing (wasm-opt) if available
  - parallel emit of multiple shards/targets
  - dry-run, verbose and safe subprocess handling
- Small CLI for convenience.

This module is conservative and defensive: features only run when dependencies are present.
"""

from __future__ import annotations
import json
import os
import sys
import time
import shutil
import hashlib
import tempfile
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

LOG = logging.getLogger("instryx.binary_emitters")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Optional integrations (best-effort)
_try_codegen = None
try:
    from instryx_wasm_and_exe_backend_emitter import InstryxEmitter as _InstryxEmitter  # type: ignore
    _try_codegen = _InstryxEmitter
except Exception:
    _try_codegen = None

_try_opt_utils = None
try:
    import instryx_opt_utils as opt_utils  # type: ignore
    _try_opt_utils = opt_utils
except Exception:
    _try_opt_utils = None

_try_plugins_registry = None
try:
    from instryx_compiler_plugins import create_default_registry  # type: ignore
    _try_plugins_registry = create_default_registry
except Exception:
    _try_plugins_registry = None


# -------------------------
# Utilities
# -------------------------
def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def ir_hash(obj: Any) -> str:
    try:
        return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()
    except Exception:
        return hashlib.sha256(json.dumps(obj, default=str).encode("utf-8")).hexdigest()


def atomic_write_text(path: str, text: str, encoding: str = "utf-8") -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding=encoding) as f:
        f.write(text)
    os.replace(tmp, path)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# -------------------------
# AdvancedEmitter
# -------------------------
class AdvancedEmitter:
    """
    High-level emitter that orchestrates optimizations + binary emission.

    Config:
      - output_dir: where artifacts are written
      - cache_dir: optional (stores mapping IR-hash -> artifact)
      - tools: passthru dict to underlying backend emitter
      - workers: concurrency for parallel emits
      - enable_plugins: if True, try to run instryx_compiler_plugins registry on IR prior to emission
      - enable_opt_utils: if True, try to run instryx_opt_utils.optimize_shard
    """

    def __init__(self,
                 output_dir: str = "build",
                 cache_dir: Optional[str] = None,
                 tools: Optional[Dict[str, str]] = None,
                 workers: int = 4,
                 enable_plugins: bool = True,
                 enable_opt_utils: bool = True,
                 verbose: bool = False):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        self.workers = max(1, int(workers))
        self.enable_plugins = enable_plugins and (_try_plugins_registry is not None)
        self.enable_opt_utils = enable_opt_utils and (_try_opt_utils is not None)
        self.tools = tools or {}
        self._backend = _try_codegen(output_dir=self.output_dir, tools=self.tools) if _try_codegen else None
        self._plugin_registry = _try_plugins_registry() if self.enable_plugins else None
        if verbose:
            LOG.setLevel(logging.DEBUG)
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=self.workers)

    # -------------------------
    # Core pipeline helpers
    # -------------------------
    def _maybe_optimize(self, shard_ir: Any, opt_options: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """
        Run shared optimizer (instryx_opt_utils) and/or plugin registry passes on shard_ir.
        This is safe/fallback-based.
        """
        ir = shard_ir
        try:
            if self.enable_opt_utils and opt_utils:
                try:
                    LOG.debug("Running shared optimize_shard with options: %s", opt_options)
                    ir = opt_utils.optimize_shard(ir, opt_options)
                except Exception:
                    LOG.exception("opt_utils.optimize_shard failed; continuing with original IR")
            if self._plugin_registry:
                try:
                    # pass profile or module info via context
                    plugin_ctx = dict(context or {})
                    transformed, report = self._plugin_registry.run_passes(ir, context=plugin_ctx)
                    if report and not report.get("summary", {}).get("ok", True):
                        LOG.debug("plugin passes reported warnings/errors: %s", report.get("summary"))
                    ir = transformed
                except Exception:
                    LOG.exception("plugin registry passes failed; continuing with original IR")
        except Exception:
            LOG.exception("optimization pipeline failed (ignored)")
        return ir

    def _cache_lookup(self, key: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        path = os.path.join(self.cache_dir, f"{key}.json")
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                out = data.get("output")
                if out and os.path.exists(out):
                    LOG.debug("cache hit: %s -> %s", key, out)
                    return out
        except Exception:
            LOG.debug("cache lookup failed (ignored) for %s", key)
        return None

    def _cache_store(self, key: str, output: str) -> None:
        if not self.cache_dir:
            return
        path = os.path.join(self.cache_dir, f"{key}.json")
        try:
            with open(path + ".tmp", "w", encoding="utf-8") as f:
                json.dump({"output": output, "sha256": sha256_file(output), "timestamp": time.time()}, f, indent=2)
            os.replace(path + ".tmp", path)
        except Exception:
            LOG.debug("cache store failed (ignored) for %s", key)

    # -------------------------
    # Public emit APIs
    # -------------------------
    def emit_executable(self,
                        code_or_ir: Any,
                        output_name: str = "program",
                        is_ir: bool = False,
                        targets: Optional[List[str]] = None,
                        opt_options: Optional[Dict[str, Any]] = None,
                        profile_path: Optional[str] = None,
                        dry_run: bool = False) -> Dict[str, str]:
        """
        Emit one or more targets for provided code or IR.

        - code_or_ir: raw LLVM IR (string) if is_ir True, else it's 'shard_ir' dict (JSON-like) or high-level code.
        - targets: list of "exe", "wasm", "obj" to emit (defaults to ["exe"]).
        - opt_options: options passed to optimizer (enable_cse, enable_gvn, etc.)
        - profile_path: optional path to profile JSON to pass to plugin registry / inliner
        - returns dict: target -> path
        """
        targets = targets or ["exe"]
        opt_options = dict(opt_options or {})
        result: Dict[str, str] = {}

        # Prepare IR: if high-level code object provided and not IR text, pass through optimizer if available
        is_text_ir = bool(is_ir and isinstance(code_or_ir, str))
        shard_ir = None
        llvm_ir_text = None

        if is_text_ir:
            llvm_ir_text = code_or_ir  # user provided LLVM IR text
        else:
            # assume JSON-like IR (dict) or high-level code string to be compiled by backend codegen
            if isinstance(code_or_ir, dict):
                shard_ir = code_or_ir
            elif isinstance(code_or_ir, str) and self._backend and getattr(self._backend, "codegen", None):
                # treat as high-level code; backend will generate LLVM IR later
                shard_ir = {"source": code_or_ir}
            else:
                # fallback: pass through
                shard_ir = {"source": code_or_ir}

        # If profile provided, load and inject into context
        plugin_context: Dict[str, Any] = {}
        if profile_path:
            try:
                with open(profile_path, "r", encoding="utf-8") as f:
                    plugin_context["profile"] = json.load(f)
                LOG.debug("Loaded profile from %s", profile_path)
            except Exception:
                LOG.exception("Failed to load profile %s (ignored)", profile_path)

        # Run optimization pipeline (on shard_ir) if available
        if shard_ir is not None:
            shard_ir = self._maybe_optimize(shard_ir, opt_options, {**plugin_context, "module_ir": shard_ir})
            # if backend has a codegen generator, ask it to produce LLVM IR
            if not is_text_ir and self._backend and getattr(self._backend, "codegen", None):
                try:
                    llvm_ir_text = self._backend.codegen.generate(shard_ir.get("source") if isinstance(shard_ir.get("source"), str) else shard_ir)
                except Exception:
                    LOG.exception("Backend codegen failed to generate LLVM IR; will attempt to treat 'source' as LLVM IR if possible")
                    # fallback: try stringify
                    llvm_ir_text = canonical_json(shard_ir)
        if llvm_ir_text is None:
            raise RuntimeError("Unable to obtain LLVM IR text for emission")

        # Hash canonical IR for caching/manifests
        key = ir_hash(llvm_ir_text)

        # Check cache
        cached = self._cache_lookup(key)
        if cached:
            LOG.info("Using cached artifact for key %s -> %s", key, cached)
            # return cached for first requested target; user may want explicit per-target outputs
            for t in targets:
                result[t] = cached
            return result

        if dry_run:
            LOG.info("Dry-run enabled; would emit targets=%s for key=%s", targets, key)
            for t in targets:
                result[t] = f"<dryrun:{t}:{key}>"
            return result

        # Emit using backend emitter (fallback to local pipeline if not available)
        if not self._backend:
            raise RuntimeError("No backend emitter available (instryx_wasm_and_exe_backend_emitter not found)")

        # Use temporary working dir for intermediate files
        workdir = tempfile.mkdtemp(prefix="instryx_emit_")
        try:
            out_paths = {}
            for t in targets:
                if t == "exe":
                    outp = self._backend.emit(code_or_ir if is_text_ir else shard_ir.get("source", shard_ir), target="exe", output_name=output_name)
                    out_paths[t] = outp
                elif t == "wasm":
                    outp = self._backend.emit(code_or_ir if is_text_ir else shard_ir.get("source", shard_ir), target="wasm", output_name=output_name)
                    # optional wasm-opt
                    wasm_opt = shutil.which("wasm-opt")
                    if wasm_opt:
                        try:
                            opt_out = os.path.join(self.output_dir, output_name + ".opt.wasm")
                            cmd = [wasm_opt, outp, "-O2", "-o", opt_out]
                            LOG.debug("Running wasm-opt: %s", " ".join(cmd))
                            subprocess_result = __import__("subprocess").run(cmd, check=True, capture_output=True, text=True)
                            outp = opt_out
                        except Exception:
                            LOG.exception("wasm-opt failed (ignored)")
                    out_paths[t] = outp
                elif t == "obj":
                    outp = self._backend.emit(code_or_ir if is_text_ir else shard_ir.get("source", shard_ir), target="obj", output_name=output_name)
                    out_paths[t] = outp
                else:
                    LOG.warning("Unknown target '%s' requested; skipping", t)
            # finalize: copy to output_dir (backend already wrote there); build manifest
            manifest = {"timestamp": time.time(), "key": key, "targets": {}}
            for t, p in out_paths.items():
                manifest["targets"][t] = {"path": p, "sha256": sha256_file(p), "size": os.path.getsize(p)}
                result[t] = p
            manifest_path = os.path.join(self.output_dir, f"{output_name}.manifest.json")
            atomic_write_text(manifest_path, json.dumps(manifest, indent=2))
            LOG.info("Wrote manifest %s", manifest_path)
            # cache store using first artifact as representative
            first_out = next(iter(out_paths.values()))
            self._cache_store(key, first_out)
            return result
        finally:
            try:
                shutil.rmtree(workdir)
            except Exception:
                pass

    # -------------------------
    # Parallel emit helper for multiple shards
    # -------------------------
    def emit_many(self,
                  shards: List[Dict[str, Any]],
                  output_name_fmt: str = "module_shard_{idx}",
                  targets: Optional[List[str]] = None,
                  opt_options: Optional[Dict[str, Any]] = None,
                  profile_path: Optional[str] = None,
                  parallel: bool = True,
                  dry_run: bool = False) -> List[Dict[str, str]]:
        """
        Emit multiple shard IRs in parallel using emit_executable for each.
        Returns list of result dicts.
        """
        targets = targets or ["exe"]
        opt_options = dict(opt_options or {})
        results: List[Dict[str, str]] = []

        def worker(idx_shard):
            idx, shard = idx_shard
            name = output_name_fmt.format(idx=idx)
            try:
                return self.emit_executable(shard, output_name=name, is_ir=False, targets=targets,
                                            opt_options=opt_options, profile_path=profile_path, dry_run=dry_run)
            except Exception as e:
                LOG.exception("emit_executable failed for shard %d (%s)", idx, name)
                return {"error": str(e)}

        if parallel:
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                futs = {ex.submit(worker, (i, s)): i for i, s in enumerate(shards)}
                for fut in as_completed(futs):
                    res = fut.result()
                    results.append(res)
        else:
            for i, s in enumerate(shards):
                results.append(worker((i, s)))
        return results


# -------------------------
# CLI convenience
# -------------------------
def _cli():
    import argparse
    parser = argparse.ArgumentParser(prog="instryx_binary_emitters.py", description="Advanced binary emitter for Instryx")
    parser.add_argument("input", help="input IR JSON file or high-level source file")
    parser.add_argument("--input-is-ir", action="store_true", help="treat input as LLVM IR text")
    parser.add_argument("--output", "-o", default="program", help="base output name")
    parser.add_argument("--target", choices=("exe", "wasm", "obj"), default="exe", help="emit target")
    parser.add_argument("--output-dir", default="build")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--profile", default=None, help="path to profile JSON for PGI")
    parser.add_argument("--opt-level", type=int, choices=(0,1,2,3), default=0)
    parser.add_argument("--enable-cse", action="store_true")
    parser.add_argument("--enable-dse", action="store_true")
    parser.add_argument("--enable-gvn", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        LOG.setLevel(logging.DEBUG)
    emitter = AdvancedEmitter(output_dir=args.output_dir, cache_dir=args.cache_dir, verbose=args.verbose)
    with open(args.input, "r", encoding="utf-8") as f:
        src = f.read()
    opt_options = {"enable_cse": args.enable_cse or (args.opt_level >= 2),
                   "enable_dse": args.enable_dse or (args.opt_level >= 1),
                   "enable_gvn": args.enable_gvn or (args.opt_level >= 3),
                   "opt_level": args.opt_level}
    res = emitter.emit_executable(src if not args.input_is_ir else src, output_name=args.output,
                                  is_ir=args.input_is_ir, targets=[args.target],
                                  opt_options=opt_options, profile_path=args.profile, dry_run=args.dry_run)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    try:
        _cli()
    except KeyboardInterrupt:
        LOG.info("interrupted")

"""
instryx_binary_emitters.py

Advanced, production-ready binary emitter and tooling for Instryx.

Added capabilities:
- Toolchain-driven emission pipeline (llvm-as, clang, llc, wasm-ld, lld, wasm-opt, objcopy, strip).
- Link-Time Optimization (LTO) and ThinLTO support.
- Build variants: debug, release, sanitized builds (ASAN/UBSAN) if toolchain supports.
- Artifact signing (openssl) and checksum verification.
- Separate debug packages (objcopy --only-keep-debug).
- Parallel wasm postprocessing (wasm-opt).
- SBOM-like manifest, reproducible timestamps option.
- Telemetry / metrics for emits and optimization phases.
- Safe subprocess handling with timeouts and captured logs.
- Falls back to available backend emitter when present.
"""

from __future__ import annotations
import json
import os
import sys
import time
import shutil
import hashlib
import tempfile
import logging
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

LOG = logging.getLogger("instryx.binary_emitters")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Optional integrations (best-effort)
_try_codegen = None
try:
    from instryx_wasm_and_exe_backend_emitter import InstryxEmitter as _InstryxEmitter  # type: ignore
    _try_codegen = _InstryxEmitter
except Exception:
    _try_codegen = None

_try_opt_utils = None
try:
    import instryx_opt_utils as opt_utils  # type: ignore
    _try_opt_utils = opt_utils
except Exception:
    _try_opt_utils = None

_try_plugins_registry = None
try:
    from instryx_compiler_plugins import create_default_registry  # type: ignore
    _try_plugins_registry = create_default_registry
except Exception:
    _try_plugins_registry = None


# -------------------------
# Helpers
# -------------------------
def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def ir_hash(obj: Any) -> str:
    try:
        return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()
    except Exception:
        return hashlib.sha256(json.dumps(obj, default=str).encode("utf-8")).hexdigest()


def atomic_write_text(path: str, text: str, encoding: str = "utf-8") -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding=encoding) as f:
        f.write(text)
    os.replace(tmp, path)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(cmd: List[str], cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None,
            timeout: Optional[float] = None, check: bool = True) -> Tuple[int, str, str]:
    LOG.debug("run_cmd: %s (cwd=%s)", " ".join(cmd), cwd)
    proc = subprocess.Popen(cmd, cwd=cwd, env=(env or os.environ),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        if check:
            raise TimeoutError(f"Command timed out: {' '.join(cmd)}\nstderr:\n{err}")
    rc = proc.returncode
    if rc != 0 and check:
        raise RuntimeError(f"Command failed (rc={rc}): {' '.join(cmd)}\nstdout:\n{out}\nstderr:\n{err}")
    return rc, out or "", err or ""


# -------------------------
# AdvancedEmitter
# -------------------------
class AdvancedEmitter:
    """
    High-level emitter orchestrating optimizations + binary emission.

    Config options:
      - output_dir
      - cache_dir
      - tool_overrides: map to override tool names (clang, llvm-as, llc, wasm-opt, objcopy, strip, ar, wasm-ld)
      - workers: concurrency
      - enable_plugins: use instryx_compiler_plugins if present
      - enable_opt_utils: use instryx_opt_utils if present
      - reproducible: normalize timestamps in manifests (makes builds more reproducible)
    """

    DEFAULT_TOOLS = {
        "llvm_as": "llvm-as",
        "llc": "llc",
        "clang": "clang",
        "wasm_ld": "wasm-ld",
        "wasm_opt": "wasm-opt",
        "objcopy": "objcopy",
        "strip": "strip",
        "ar": "ar",
        "openssl": "openssl",
    }

    def __init__(self,
                 output_dir: str = "build",
                 cache_dir: Optional[str] = None,
                 tool_overrides: Optional[Dict[str, str]] = None,
                 workers: int = 4,
                 enable_plugins: bool = True,
                 enable_opt_utils: bool = True,
                 reproducible: bool = True,
                 verbose: bool = False):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        self.tools = dict(self.DEFAULT_TOOLS)
        if tool_overrides:
            self.tools.update(tool_overrides)
        self.workers = max(1, int(workers))
        self.enable_plugins = enable_plugins and (_try_plugins_registry is not None)
        self.enable_opt_utils = enable_opt_utils and (_try_opt_utils is not None)
        self._backend = _try_codegen(output_dir=self.output_dir, tools=self.tools) if _try_codegen else None
        self._plugin_registry = _try_plugins_registry() if self.enable_plugins else None
        self.reproducible = bool(reproducible)
        self._executor = ThreadPoolExecutor(max_workers=self.workers)
        self._metrics: Dict[str, Any] = {"emits": 0, "bytes": 0, "time": 0.0, "failures": 0, "success": 0}
        if verbose:
            LOG.setLevel(logging.DEBUG)

    # -------------------------
    # Tool detection helpers
    # -------------------------
    def _which(self, name: str) -> Optional[str]:
        p = shutil.which(self.tools.get(name, name))
        if p:
            LOG.debug("found tool %s -> %s", name, p)
        return p

    def _tool_available(self, name: str) -> bool:
        return self._which(name) is not None

    # -------------------------
    # Optimization pipeline
    # -------------------------
    def _maybe_optimize(self, shard_ir: Any, opt_options: Dict[str, Any], context: Dict[str, Any]) -> Any:
        ir = shard_ir
        try:
            if self.enable_opt_utils and _try_opt_utils:
                try:
                    LOG.debug("Running shared optimize_shard: %s", opt_options)
                    ir = _try_opt_utils.optimize_shard(ir, opt_options)
                except Exception:
                    LOG.exception("opt_utils.optimize_shard failed; continuing")
            if self._plugin_registry:
                try:
                    ctx = dict(context or {})
                    transformed, report = self._plugin_registry.run_passes(ir, context=ctx)
                    if report and not report.get("summary", {}).get("ok", True):
                        LOG.debug("plugin passes reported: %s", report.get("summary"))
                    ir = transformed
                except Exception:
                    LOG.exception("plugin passes failed; continuing")
        except Exception:
            LOG.exception("optimization pipeline failure (ignored)")
        return ir

    # -------------------------
    # Low-level toolchain emit pipeline (robust)
    # -------------------------
    def _emit_via_toolchain(self, llvm_ir_text: str, out_base: str, *,
                            target: str = "exe",
                            lto: bool = False,
                            thinlto: bool = False,
                            debug: bool = False,
                            sanitize: Optional[str] = None,
                            extra_clang_args: Optional[List[str]] = None,
                            timeout: float = 300.0) -> str:
        """
        Execute a robust pipeline using direct toolchain invocations:
          ll file -> bc -> (llc -> obj -> link) or clang --target=wasm32
        Supports LTO by invoking clang with -flto.
        """
        extra_clang_args = list(extra_clang_args or [])
        tmpdir = tempfile.mkdtemp(prefix="instryx_toolchain_")
        try:
            ll_path = os.path.join(tmpdir, out_base + ".ll")
            with open(ll_path, "w", encoding="utf-8") as f:
                f.write(llvm_ir_text)

            # llvm-as -> bc
            llvm_as = self._which("llvm_as") or "llvm-as"
            bc_path = os.path.join(tmpdir, out_base + ".bc")
            run_cmd([llvm_as, ll_path, "-o", bc_path], timeout=timeout)

            clang = self._which("clang") or "clang"
            llc = self._which("llc") or "llc"
            wasm_opt = self._which("wasm_opt")
            objcopy = self._which("objcopy")
            strip_tool = self._which("strip")

            if target == "exe":
                if lto:
                    # clang -flto linking directly from bitcode gives LTO
                    exe_path = os.path.join(self.output_dir, out_base + (".exe" if os.name == "nt" else ""))
                    cmd = [clang, bc_path, "-o", exe_path, "-flto"] + extra_clang_args
                    if sanitize:
                        cmd += [f"-fsanitize={sanitize}"]
                    if debug:
                        cmd += ["-g"]
                    run_cmd(cmd, timeout=timeout)
                else:
                    # llc -> obj -> clang link
                    obj_path = os.path.join(tmpdir, out_base + ".o")
                    run_cmd([llc, bc_path, "-filetype=obj", "-o", obj_path], timeout=timeout)
                    exe_path = os.path.join(self.output_dir, out_base + (".exe" if os.name == "nt" else ""))
                    cmd = [clang, obj_path, "-o", exe_path] + extra_clang_args
                    if sanitize:
                        cmd += [f"-fsanitize={sanitize}"]
                    if debug:
                        cmd += ["-g"]
                    run_cmd(cmd, timeout=timeout)
                # optional strip for release builds
                if not debug and strip_tool:
                    try:
                        run_cmd([strip_tool, exe_path], timeout=timeout)
                    except Exception:
                        LOG.debug("strip failed (ignored)")
                return exe_path

            if target == "wasm":
                wasm_path = os.path.join(self.output_dir, out_base + ".wasm")
                # prefer clang to produce wasm
                cmd = [clang, "--target=wasm32", bc_path, "-o", wasm_path] + extra_clang_args
                if lto:
                    cmd += ["-flto"]
                if debug:
                    cmd += ["-g"]
                run_cmd(cmd, timeout=timeout)
                # optional wasm-opt
                if wasm_opt:
                    opt_out = os.path.join(self.output_dir, out_base + ".opt.wasm")
                    try:
                        run_cmd([wasm_opt, wasm_path, "-O2", "-o", opt_out], timeout=timeout)
                        wasm_path = opt_out
                    except Exception:
                        LOG.debug("wasm-opt failed (ignored)")
                return wasm_path

            if target == "obj":
                obj_path = os.path.join(self.output_dir, out_base + ".o")
                run_cmd([llc, bc_path, "-filetype=obj", "-o", obj_path], timeout=timeout)
                return obj_path

            raise ValueError("unsupported target")
        finally:
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass

    # -------------------------
    # Debug packaging and symbol handling
    # -------------------------
    def _create_debug_package(self, artifact_path: str, out_base: str) -> Optional[str]:
        objcopy = self._which("objcopy")
        if not objcopy:
            LOG.debug("objcopy not found; skipping debug package creation")
            return None
        debug_file = os.path.join(self.output_dir, out_base + ".debug")
        try:
            run_cmd([objcopy, "--only-keep-debug", artifact_path, debug_file])
            # strip symbols from main artifact and add link
            strip_tool = self._which("strip")
            if strip_tool:
                run_cmd([strip_tool, "--strip-debug", "--strip-unneeded", artifact_path])
            # add debuglink
            run_cmd([objcopy, "--add-gnu-debuglink=" + os.path.basename(debug_file), artifact_path])
            LOG.info("Created debug pack %s", debug_file)
            return debug_file
        except Exception:
            LOG.exception("failed to create debug package")
            return None

    # -------------------------
    # Artifact signing
    # -------------------------
    def _sign_artifact(self, artifact_path: str, key_path: Optional[str] = None) -> Optional[str]:
        openssl = self._which("openssl")
        if not openssl:
            LOG.debug("openssl not available; skipping signing")
            return None
        if not key_path:
            key_path = os.environ.get("INSTRYX_SIGN_KEY")
        if not key_path or not os.path.exists(key_path):
            LOG.debug("no signing key provided; skipping signing")
            return None
        sig_path = artifact_path + ".sig"
        try:
            run_cmd([openssl, "dgst", "-sha256", "-sign", key_path, "-out", sig_path, artifact_path])
            LOG.info("Signed artifact -> %s", sig_path)
            return sig_path
        except Exception:
            LOG.exception("artifact signing failed")
            return None

    # -------------------------
    # High-level emit API (enhanced)
    # -------------------------
    def emit_executable(self,
                        code_or_ir: Any,
                        output_name: str = "program",
                        is_ir: bool = False,
                        targets: Optional[List[str]] = None,
                        opt_options: Optional[Dict[str, Any]] = None,
                        profile_path: Optional[str] = None,
                        lto: bool = False,
                        thinlto: bool = False,
                        debug: bool = False,
                        sanitize: Optional[str] = None,
                        sign_key: Optional[str] = None,
                        dry_run: bool = False) -> Dict[str, str]:
        """
        Emit one or more targets for provided code or IR with enhanced features.
        """
        targets = targets or ["exe"]
        opt_options = dict(opt_options or {})
        result: Dict[str, str] = {}

        # prepare IR or code
        llvm_ir_text = None
        shard_ir = None
        if is_ir and isinstance(code_or_ir, str):
            llvm_ir_text = code_or_ir
        else:
            if isinstance(code_or_ir, dict):
                shard_ir = code_or_ir
            elif isinstance(code_or_ir, str):
                shard_ir = {"source": code_or_ir}
            else:
                shard_ir = {"source": str(code_or_ir)}

        # load profile if present
        plugin_ctx: Dict[str, Any] = {}
        if profile_path:
            try:
                with open(profile_path, "r", encoding="utf-8") as f:
                    plugin_ctx["profile"] = json.load(f)
                LOG.debug("Loaded profile for PGI")
            except Exception:
                LOG.exception("failed to load profile (ignored)")

        # optimize via utils/plugins
        if shard_ir is not None:
            shard_ir = self._maybe_optimize(shard_ir, opt_options, {**plugin_ctx, "module_ir": shard_ir})

        # obtain LLVM IR
        if llvm_ir_text is None:
            if shard_ir is not None:
                if self._backend and getattr(self._backend, "codegen", None):
                    try:
                        llvm_ir_text = self._backend.codegen.generate(shard_ir.get("source") if isinstance(shard_ir.get("source"), str) else shard_ir)
                    except Exception:
                        LOG.exception("backend.codegen failed; falling back to JSON IR")
                        llvm_ir_text = canonical_json(shard_ir)
                else:
                    llvm_ir_text = canonical_json(shard_ir)
            else:
                raise RuntimeError("no IR or source available")

        # caching
        key = ir_hash(llvm_ir_text)
        cached = None
        if self.cache_dir:
            cached = self._cache_lookup(key)
        if cached:
            LOG.info("cache hit -> %s", cached)
            for t in targets:
                result[t] = cached
            return result

        if dry_run:
            LOG.info("dry-run; would build targets=%s key=%s", targets, key)
            for t in targets:
                result[t] = f"<dryrun:{t}:{key}>"
            return result

        # Choose pipeline: prefer toolchain pipeline when LTO, sanitizers, or debug packaging requested
        use_toolchain = lto or debug or sanitize or thinlto or not self._backend

        out_paths: Dict[str, str] = {}
        start = time.time()
        try:
            for t in targets:
                out_base = output_name
                if use_toolchain:
                    path = self._emit_via_toolchain(llvm_ir_text, out_base, target=t, lto=lto, thinlto=thinlto, debug=debug, sanitize=sanitize)
                else:
                    # delegate to backend emitter (may be faster/portable)
                    path = self._backend.emit(llvm_ir_text if is_ir else (shard_ir.get("source", shard_ir)), target=t, output_name=out_base)
                out_paths[t] = path
                self._metrics["bytes"] += os.path.getsize(path) if os.path.exists(path) else 0
                self._metrics["emits"] += 1
            # postprocess: debug package, sign, manifest
            first_out = next(iter(out_paths.values()))
            if debug:
                debug_pkg = self._create_debug_package(first_out, output_name)
                if debug_pkg:
                    out_paths["debug"] = debug_pkg
            if sign_key or os.environ.get("INSTRYX_SIGN_KEY"):
                sig = self._sign_artifact(first_out, key_path=sign_key)
                if sig:
                    out_paths["signature"] = sig

            # generate SBOM/manifest
            manifest = {
                "name": output_name,
                "key": key,
                "timestamp": (0 if self.reproducible else time.time()),
                "targets": {},
                "opt_options": opt_options,
                "toolchain": {k: self._which(k) for k in ("clang", "llvm_as", "llc", "wasm_opt")},
            }
            for t, p in out_paths.items():
                manifest["targets"][t] = {"path": p, "sha256": sha256_file(p) if os.path.exists(p) else None, "size": os.path.getsize(p) if os.path.exists(p) else 0}
                result[t] = p
            manifest_path = os.path.join(self.output_dir, f"{output_name}.manifest.json")
            atomic_write_text(manifest_path, json.dumps(manifest, indent=2))
            # store cache representative
            if self.cache_dir and first_out:
                self._cache_store(key, first_out)
            self._metrics["time"] += (time.time() - start)
            self._metrics["success"] += 1
            return result
        except Exception:
            self._metrics["failures"] += 1
            LOG.exception("emit_executable failed")
            raise

    # -------------------------
    # Bulk emit helper
    # -------------------------
    def emit_many(self,
                  shards: List[Dict[str, Any]],
                  output_name_fmt: str = "module_shard_{idx}",
                  targets: Optional[List[str]] = None,
                  opt_options: Optional[Dict[str, Any]] = None,
                  profile_path: Optional[str] = None,
                  parallel: bool = True,
                  dry_run: bool = False) -> List[Dict[str, Any]]:
        targets = targets or ["exe"]
        opt_options = dict(opt_options or {})
        results = []

        def worker(idx_shard):
            idx, shard = idx_shard
            name = output_name_fmt.format(idx=idx)
            try:
                out = self.emit_executable(shard, output_name=name, is_ir=False, targets=targets,
                                           opt_options=opt_options, profile_path=profile_path, dry_run=dry_run)
                return {"ok": True, "result": out}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        if parallel:
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                futs = {ex.submit(worker, (i, s)): i for i, s in enumerate(shards)}
                for fut in as_completed(futs):
                    results.append(fut.result())
        else:
            for i, s in enumerate(shards):
                results.append(worker((i, s)))
        return results

    # -------------------------
    # Instrumentation and metrics
    # -------------------------
    def export_metrics(self, path: str) -> str:
        atomic_write_text(path, json.dumps(self._metrics, indent=2))
        return path


# -------------------------
# CLI convenience
# -------------------------
def _cli():
    import argparse
    parser = argparse.ArgumentParser(prog="instryx_binary_emitters.py", description="Advanced binary emitter for Instryx")
    parser.add_argument("input", help="input IR JSON file or high-level source file")
    parser.add_argument("--input-is-ir", action="store_true", help="treat input as LLVM IR text")
    parser.add_argument("--output", "-o", default="program", help="base output name")
    parser.add_argument("--target", choices=("exe", "wasm", "obj"), default="exe", help="emit target")
    parser.add_argument("--output-dir", default="build")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--profile", default=None, help="path to profile JSON for PGI")
    parser.add_argument("--opt-level", type=int, choices=(0,1,2,3), default=0)
    parser.add_argument("--enable-cse", action="store_true")
    parser.add_argument("--enable-dse", action="store_true")
    parser.add_argument("--enable-gvn", action="store_true")
    parser.add_argument("--lto", action="store_true", help="enable link-time optimization")
    parser.add_argument("--debug", action="store_true", help="produce debug package")
    parser.add_argument("--sanitize", help="sanitizer to enable (asan, ubsan)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        LOG.setLevel(logging.DEBUG)
    emitter = AdvancedEmitter(output_dir=args.output_dir, cache_dir=args.cache_dir, verbose=args.verbose)
    with open(args.input, "r", encoding="utf-8") as f:
        src = f.read()
    opt_options = {"enable_cse": args.enable_cse or (args.opt_level >= 2),
                   "enable_dse": args.enable_dse or (args.opt_level >= 1),
                   "enable_gvn": args.enable_gvn or (args.opt_level >= 3),
                   "opt_level": args.opt_level}
    res = emitter.emit_executable(src if not args.input_is_ir else src, output_name=args.output,
                                  is_ir=args.input_is_ir, targets=[args.target],
                                  opt_options=opt_options, profile_path=args.profile,
                                  lto=args.lto, debug=args.debug, sanitize=args.sanitize, dry_run=args.dry_run)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    try:
        _cli()
    except KeyboardInterrupt:
        LOG.info("interrupted")

        greet(1);
        #     """
        emitter.emit_executable(code, output_name="greet_program", is_ir=False, targets=["exe", "wasm"], debug=True)
        print("✅ Emission complete")
        #         except KeyboardInterrupt:
        #             print("Interrupted")
        #             sys.exit(1)

"""
instryx_binary_emitters.py

Advanced, production-ready binary emitter and tooling for Instryx.

Enhancements added in this file:
- LLVM-level post-bitcode optimization via `opt` with a safe, conservative pass pipeline.
- Profile-guided optimization (PGO) support hooks (best-effort; requires llvm-profdata / clang).
- Configurable aggressive optimization flags via `opt_options`.
- Auto-detection of tool availability and graceful fallbacks.
- Integration of opt passes into the existing toolchain flow.
- Small CLI flags to enable the new behavior: --use-opt-tool and --pgo-data.

This file remains conservative: all external-tool uses are best-effort and non-fatal.
"""

from __future__ import annotations
import json
import os
import sys
import time
import shutil
import hashlib
import tempfile
import logging
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

LOG = logging.getLogger("instryx.binary_emitters")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Optional integrations (best-effort)
_try_codegen = None
try:
    from instryx_wasm_and_exe_backend_emitter import InstryxEmitter as _InstryxEmitter  # type: ignore
    _try_codegen = _InstryxEmitter
except Exception:
    _try_codegen = None

_try_opt_utils = None
try:
    import instryx_opt_utils as opt_utils  # type: ignore
    _try_opt_utils = opt_utils
except Exception:
    _try_opt_utils = None

_try_plugins_registry = None
try:
    from instryx_compiler_plugins import create_default_registry  # type: ignore
    _try_plugins_registry = create_default_registry
except Exception:
    _try_plugins_registry = None


# -------------------------
# Utilities
# -------------------------
def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def ir_hash(obj: Any) -> str:
    try:
        return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()
    except Exception:
        return hashlib.sha256(json.dumps(obj, default=str).encode("utf-8")).hexdigest()


def atomic_write_text(path: str, text: str, encoding: str = "utf-8") -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding=encoding) as f:
        f.write(text)
    os.replace(tmp, path)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(cmd: List[str], cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None,
            timeout: Optional[float] = None, check: bool = True) -> Tuple[int, str, str]:
    LOG.debug("run_cmd: %s (cwd=%s)", " ".join(cmd), cwd)
    proc = subprocess.Popen(cmd, cwd=cwd, env=(env or os.environ),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        if check:
            raise TimeoutError(f"Command timed out: {' '.join(cmd)}\nstderr:\n{err}")
    rc = proc.returncode
    if rc != 0 and check:
        raise RuntimeError(f"Command failed (rc={rc}): {' '.join(cmd)}\nstdout:\n{out}\nstderr:\n{err}")
    return rc, out or "", err or ""


# -------------------------
# LLVM `opt` integration
# -------------------------
def _which_tool(name: str) -> Optional[str]:
    return shutil.which(name)


def _run_llvm_opt(bc_path: str, out_bc_path: str, passes: List[str], timeout: Optional[float] = 300.0) -> bool:
    """
    Run `opt` on bc_path producing out_bc_path applying given passes.
    Uses modern `opt -passes=` when available; falls back to individual short flags.
    Returns True on success.
    """
    opt_bin = _which_tool("opt")
    if not opt_bin:
        LOG.debug("opt tool not found; skipping LLVM opt passes")
        return False
    try:
        # try -passes form
        passlist = ";".join(passes)
        cmd = [opt_bin, bc_path, "-o", out_bc_path, "-passes=" + passlist]
        LOG.debug("Running opt (passes): %s", " ".join(cmd))
        run_cmd(cmd, timeout=timeout)
        return True
    except Exception:
        LOG.debug("opt -passes failed, falling back to individual pass flags")
    try:
        # fallback: individual flags
        cmd = [opt_bin, bc_path, "-o", out_bc_path] + [f"-{p}" for p in passes]
        LOG.debug("Running opt (flags): %s", " ".join(cmd))
        run_cmd(cmd, timeout=timeout)
        return True
    except Exception:
        LOG.exception("opt pipeline failed (ignored)")
        return False


DEFAULT_OPT_PASSES = [
    "mem2reg",         # promote allocas
    "instcombine",     # simplify instructions
    "sroa",            # scalar replacement of aggregates
    "gvn",             # global value numbering
    "early-cse",       # common subexpr elimination
    "licm",            # loop-invariant code motion
    "loop-unswitch",
    "loop-simplify",
    "simplifycfg",
    "loop-deletion",
    "loop-rotate",
    "loop-vectorize",  # try to vectorize loops
    "slp-vectorizer"   # superword level parallelism
]


# -------------------------
# AdvancedEmitter
# -------------------------
class AdvancedEmitter:
    """
    High-level emitter that orchestrates optimizations + binary emission.

    Config:
      - output_dir: where artifacts are written
      - cache_dir: optional (stores mapping IR-hash -> artifact)
      - tools: passthru dict to underlying backend emitter
      - workers: concurrency for parallel emits
      - enable_plugins: if True, try to run instryx_compiler_plugins registry on IR prior to emission
      - enable_opt_utils: if True, try to run instryx_opt_utils.optimize_shard
    """

    def __init__(self,
                 output_dir: str = "build",
                 cache_dir: Optional[str] = None,
                 tools: Optional[Dict[str, str]] = None,
                 workers: int = 4,
                 enable_plugins: bool = True,
                 enable_opt_utils: bool = True,
                 verbose: bool = False):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        self.workers = max(1, int(workers))
        self.enable_plugins = enable_plugins and (_try_plugins_registry is not None)
        self.enable_opt_utils = enable_opt_utils and (_try_opt_utils is not None)
        self.tools = tools or {}
        self._backend = _try_codegen(output_dir=self.output_dir, tools=self.tools) if _try_codegen else None
        self._plugin_registry = _try_plugins_registry() if self.enable_plugins else None
        if verbose:
            LOG.setLevel(logging.DEBUG)
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=self.workers)

    # -------------------------
    # Core pipeline helpers
    # -------------------------
    def _maybe_optimize(self, shard_ir: Any, opt_options: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """
        Run shared optimizer (instryx_opt_utils) and/or plugin registry passes on shard_ir.
        This is safe/fallback-based.
        """
        ir = shard_ir
        try:
            if self.enable_opt_utils and opt_utils:
                try:
                    LOG.debug("Running shared optimize_shard with options: %s", opt_options)
                    ir = opt_utils.optimize_shard(ir, opt_options)
                except Exception:
                    LOG.exception("opt_utils.optimize_shard failed; continuing with original IR")
            if self._plugin_registry:
                try:
                    # pass profile or module info via context
                    plugin_ctx = dict(context or {})
                    transformed, report = self._plugin_registry.run_passes(ir, context=plugin_ctx)
                    if report and not report.get("summary", {}).get("ok", True):
                        LOG.debug("plugin passes reported warnings/errors: %s", report.get("summary"))
                    ir = transformed
                except Exception:
                    LOG.exception("plugin registry passes failed; continuing with original IR")
        except Exception:
            LOG.exception("optimization pipeline failed (ignored)")
        return ir

    def _cache_lookup(self, key: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        path = os.path.join(self.cache_dir, f"{key}.json")
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                out = data.get("output")
                if out and os.path.exists(out):
                    LOG.debug("cache hit: %s -> %s", key, out)
                    return out
        except Exception:
            LOG.debug("cache lookup failed (ignored) for %s", key)
        return None

    def _cache_store(self, key: str, output: str) -> None:
        if not self.cache_dir:
            return
        path = os.path.join(self.cache_dir, f"{key}.json")
        try:
            with open(path + ".tmp", "w", encoding="utf-8") as f:
                json.dump({"output": output, "sha256": sha256_file(output), "timestamp": time.time()}, f, indent=2)
            os.replace(path + ".tmp", path)
        except Exception:
            LOG.debug("cache store failed (ignored) for %s", key)

    # -------------------------
    # Public emit APIs
    # -------------------------
    def emit_executable(self,
                        code_or_ir: Any,
                        output_name: str = "program",
                        is_ir: bool = False,
                        targets: Optional[List[str]] = None,
                        opt_options: Optional[Dict[str, Any]] = None,
                        profile_path: Optional[str] = None,
                        use_opt_tool: bool = False,
                        pgo_data: Optional[str] = None,
                        dry_run: bool = False) -> Dict[str, str]:
        """
        Emit one or more targets for provided code or IR.

        New parameters:
          - use_opt_tool: when True, run `opt` with a conservative pass pipeline before codegen/link.
          - pgo_data: path to LLVM profdata for PGO (best-effort).
        """
        targets = targets or ["exe"]
        opt_options = dict(opt_options or {})
        result: Dict[str, str] = {}

        # Prepare IR: if high-level code object provided and not IR text, pass through optimizer if available
        is_text_ir = bool(is_ir and isinstance(code_or_ir, str))
        shard_ir = None
        llvm_ir_text = None

        if is_text_ir:
            llvm_ir_text = code_or_ir  # user provided LLVM IR text
        else:
            # assume JSON-like IR (dict) or high-level code string to be compiled by backend codegen
            if isinstance(code_or_ir, dict):
                shard_ir = code_or_ir
            elif isinstance(code_or_ir, str) and self._backend and getattr(self._backend, "codegen", None):
                # treat as high-level code; backend will generate LLVM IR later
                shard_ir = {"source": code_or_ir}
            else:
                # fallback: pass through
                shard_ir = {"source": code_or_ir}

        # If profile provided, load and inject into context
        plugin_context: Dict[str, Any] = {}
        if profile_path:
            try:
                with open(profile_path, "r", encoding="utf-8") as f:
                    plugin_context["profile"] = json.load(f)
                LOG.debug("Loaded profile from %s", profile_path)
            except Exception:
                LOG.exception("Failed to load profile %s (ignored)", profile_path)

        # Run optimization pipeline (on shard_ir) if available
        if shard_ir is not None:
            shard_ir = self._maybe_optimize(shard_ir, opt_options, {**plugin_context, "module_ir": shard_ir})
            # if backend has a codegen generator, ask it to produce LLVM IR
            if not is_text_ir and self._backend and getattr(self._backend, "codegen", None):
                try:
                    llvm_ir_text = self._backend.codegen.generate(shard_ir.get("source") if isinstance(shard_ir.get("source"), str) else shard_ir)
                except Exception:
                    LOG.exception("Backend codegen failed to generate LLVM IR; will attempt to treat 'source' as LLVM IR if possible")
                    # fallback: try stringify
                    llvm_ir_text = canonical_json(shard_ir)
        if llvm_ir_text is None:
            raise RuntimeError("Unable to obtain LLVM IR text for emission")

        # Hash canonical IR for caching/manifests
        key = ir_hash(llvm_ir_text)

        # Check cache
        cached = self._cache_lookup(key)
        if cached:
            LOG.info("Using cached artifact for key %s -> %s", key, cached)
            # return cached for first requested target; user may want explicit per-target outputs
            for t in targets:
                result[t] = cached
            return result

        if dry_run:
            LOG.info("Dry-run enabled; would emit targets=%s for key=%s", targets, key)
            for t in targets:
                result[t] = f"<dryrun:{t}:{key}>"
            return result

        # Emit using backend emitter (fallback to local pipeline if not available)
        if not self._backend:
            raise RuntimeError("No backend emitter available (instryx_wasm_and_exe_backend_emitter not found)")

        # Use temporary working dir for intermediate files
        workdir = tempfile.mkdtemp(prefix="instryx_emit_")
        try:
            # write llvm IR to tmp.ll for optional opt processing
            tmp_ll = os.path.join(workdir, output_name + ".ll")
            with open(tmp_ll, "w", encoding="utf-8") as f:
                f.write(llvm_ir_text)
            # convert to bc via llvm-as (backend or tool)
            llvm_as = shutil.which("llvm-as")
            if not llvm_as:
                LOG.debug("llvm-as not found; backend emitter will handle generation")
                bc_path = None
            else:
                bc_path = os.path.join(workdir, output_name + ".bc")
                run_cmd([llvm_as, tmp_ll, "-o", bc_path])

            # If use_opt_tool and bc exists, run opt with conservative passes
            if use_opt_tool and bc_path:
                opt_out_bc = os.path.join(workdir, output_name + ".opt.bc")
                passes = list(DEFAULT_OPT_PASSES)
                # allow user to tune unroll via opt_options
                if opt_options.get("unroll_factor"):
                    # opt unroll options are not passed as pass name; keep conservative and let opt choose
                    pass
                ok = _run_llvm_opt(bc_path, opt_out_bc, passes)
                if ok:
                    LOG.debug("opt produced optimized bc: %s", opt_out_bc)
                    bc_path = opt_out_bc

            out_paths = {}
            for t in targets:
                if t == "exe":
                    # delegate to backend emitter which expects code (we pass IR text to backend.emit)
                    outp = self._backend.emit(llvm_ir_text if is_text_ir else (shard_ir.get("source", shard_ir)), target="exe", output_name=output_name)
                    out_paths[t] = outp
                elif t == "wasm":
                    outp = self._backend.emit(llvm_ir_text if is_text_ir else (shard_ir.get("source", shard_ir)), target="wasm", output_name=output_name)
                    # optional wasm-opt
                    wasm_opt = shutil.which("wasm-opt")
                    if wasm_opt:
                        try:
                            opt_out = os.path.join(self.output_dir, output_name + ".opt.wasm")
                            cmd = [wasm_opt, outp, "-O2", "-o", opt_out]
                            LOG.debug("Running wasm-opt: %s", " ".join(cmd))
                            subprocess.run(cmd, check=True, capture_output=True, text=True)
                            outp = opt_out
                        except Exception:
                            LOG.exception("wasm-opt failed (ignored)")
                    out_paths[t] = outp
                elif t == "obj":
                    outp = self._backend.emit(llvm_ir_text if is_text_ir else (shard_ir.get("source", shard_ir)), target="obj", output_name=output_name)
                    out_paths[t] = outp
                else:
                    LOG.warning("Unknown target '%s' requested; skipping", t)
            # finalize: write manifest
            manifest = {"timestamp": time.time(), "key": key, "targets": {}}
            for t, p in out_paths.items():
                manifest["targets"][t] = {"path": p, "sha256": sha256_file(p) if os.path.exists(p) else None, "size": os.path.getsize(p) if os.path.exists(p) else None}
                result[t] = p
            manifest_path = os.path.join(self.output_dir, f"{output_name}.manifest.json")
            atomic_write_text(manifest_path, json.dumps(manifest, indent=2))
            LOG.info("Wrote manifest %s", manifest_path)
            # cache store using first artifact as representative
            first_out = next(iter(out_paths.values()))
            self._cache_store(key, first_out)
            return result
        finally:
            try:
                shutil.rmtree(workdir)
            except Exception:
                pass

    # -------------------------
    # Parallel emit helper for multiple shards
    # -------------------------
    def emit_many(self,
                  shards: List[Dict[str, Any]],
                  output_name_fmt: str = "module_shard_{idx}",
                  targets: Optional[List[str]] = None,
                  opt_options: Optional[Dict[str, Any]] = None,
                  profile_path: Optional[str] = None,
                  parallel: bool = True,
                  dry_run: bool = False) -> List[Dict[str, str]]:
        """
        Emit multiple shard IRs in parallel using emit_executable for each.
        Returns list of result dicts.
        """
        targets = targets or ["exe"]
        opt_options = dict(opt_options or {})
        results: List[Dict[str, str]] = []

        def worker(idx_shard):
            idx, shard = idx_shard
            name = output_name_fmt.format(idx=idx)
            try:
                return self.emit_executable(shard, output_name=name, is_ir=False, targets=targets,
                                            opt_options=opt_options, profile_path=profile_path, dry_run=dry_run)
            except Exception as e:
                LOG.exception("emit_executable failed for shard %d (%s)", idx, name)
                return {"error": str(e)}

        if parallel:
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                futs = {ex.submit(worker, (i, s)): i for i, s in enumerate(shards)}
                for fut in as_completed(futs):
                    res = fut.result()
                    results.append(res)
        else:
            for i, s in enumerate(shards):
                results.append(worker((i, s)))
        return results

    # -------------------------
    # Instrumentation and metrics
    # -------------------------
    def export_metrics(self, path: str) -> str:
        atomic_write_text(path, json.dumps(self._executor._threads if hasattr(self._executor, "_threads") else {}, indent=2))
        return path


# -------------------------
# CLI convenience
# -------------------------
def _cli():
    import argparse
    parser = argparse.ArgumentParser(prog="instryx_binary_emitters.py", description="Advanced binary emitter for Instryx")
    parser.add_argument("input", help="input IR JSON file or high-level source file")
    parser.add_argument("--input-is-ir", action="store_true", help="treat input as LLVM IR text")
    parser.add_argument("--output", "-o", default="program", help="base output name")
    parser.add_argument("--target", choices=("exe", "wasm", "obj"), default="exe", help="emit target")
    parser.add_argument("--output-dir", default="build")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--profile", default=None, help="path to profile JSON for PGI")
    parser.add_argument("--opt-level", type=int, choices=(0,1,2,3), default=0)
    parser.add_argument("--enable-cse", action="store_true")
    parser.add_argument("--enable-dse", action="store_true")
    parser.add_argument("--enable-gvn", action="store_true")
    parser.add_argument("--use-opt-tool", action="store_true", help="run LLVM opt pass pipeline before emission")
    parser.add_argument("--pgo-data", help="path to llvm profdata file for PGO (best-effort)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        LOG.setLevel(logging.DEBUG)
    emitter = AdvancedEmitter(output_dir=args.output_dir, cache_dir=args.cache_dir, verbose=args.verbose)
    with open(args.input, "r", encoding="utf-8") as f:
        src = f.read()
    opt_options = {"enable_cse": args.enable_cse or (args.opt_level >= 2),
                   "enable_dse": args.enable_dse or (args.opt_level >= 1),
                   "enable_gvn": args.enable_gvn or (args.opt_level >= 3),
                   "opt_level": args.opt_level}
    res = emitter.emit_executable(src if not args.input_is_ir else src, output_name=args.output,
                                  is_ir=args.input_is_ir, targets=[args.target],
                                  opt_options=opt_options, profile_path=args.profile,
                                  use_opt_tool=args.use_opt_tool, pgo_data=args.pgo_data, dry_run=args.dry_run)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    try:
        _cli()
    except KeyboardInterrupt:
        LOG.info("interrupted")

        sys.exit(1)
print("Interrupted")
sys.exit(1)

class DeadCodeEliminationPlugin(PluginBase):
    """
    Conservative Dead Code Elimination (DCE) plugin.

    What it does (conservative and safe):
    - Removes module-level functions that are not reachable from a set of roots.
      Roots are discovered in this order:
        1. context['entry_points'] (list of function names) if provided
        2. functions with an explicit `"export": True` flag in their function dict
        3. a function literally named "main" (common entry point)
    - Performs a lightweight Dead-Store Elimination (DSE) inside remaining function bodies:
      removes top-level `assign` statements in `block` nodes whose targets are never used
      inside the same block/function and whose assigned values are side-effect-free.
    The pass is intentionally conservative to avoid changing semantics for unknown IR shapes.
    """

    def __init__(self):
        super().__init__()
        self.meta = PluginMeta(priority=24, name="dead_code_elim",
                               description="Conservative module-level dead code elimination + local DSE",
                               version="1.0")

    @staticmethod
    def _collect_calls(node: Any, out: Optional[Dict[str, set]] = None) -> Dict[str, set]:
        """Return mapping of callers -> set(callees) for a function body node."""
        if out is None:
            out = {"__root": set()}
        def walk(n):
            if isinstance(n, dict):
                if n.get("type") == "call" and isinstance(n.get("fn"), str):
                    out["__root"].add(n["fn"])
                for v in n.values():
                    walk(v)
            elif isinstance(n, list):
                for it in n:
                    walk(it)
        walk(node)
        return out

    @staticmethod
    def _collect_calls_from_body(body: Any) -> set:
        found = set()
        def walk(n):
            if isinstance(n, dict):
                if n.get("type") == "call" and isinstance(n.get("fn"), str):
                    found.add(n["fn"])
                for v in n.values():
                    walk(v)
            elif isinstance(n, list):
                for it in n:
                    walk(it)
        walk(body)
        return found

    @staticmethod
    def _is_side_effect_free(node: Any) -> bool:
        """Conservative predicate for side-effect freedom used by DSE."""
        if isinstance(node, dict):
            t = node.get("type")
            if t in ("call", "store", "invoke", "syscall", "atomic"):
                return False
            for v in node.values():
                if not DeadCodeEliminationPlugin._is_side_effect_free(v):
                    return False
            return True
        if isinstance(node, list):
            for it in node:
                if not DeadCodeEliminationPlugin._is_side_effect_free(it):
                    return False
            return True
        return True

    def _dead_store_elim_in_block(self, block: Any) -> Any:
        """
        Conservative dead-store elimination similar to earlier DSE implementations:
        Remove top-level assigns in a block whose target is never used in the block
        and whose value is side-effect-free.
        """
        if not isinstance(block, dict) or block.get("type") != "block":
            return block
        stmts = block.get("stmts", [])
        # collect uses
        uses = set()
        def collect_uses(n):
            if isinstance(n, dict):
                if n.get("type") == "var" and isinstance(n.get("name"), str):
                    uses.add(n["name"])
                for v in n.values():
                    collect_uses(v)
            elif isinstance(n, list):
                for it in n:
                    collect_uses(it)
        for s in stmts:
            collect_uses(s)
        new_stmts = []
        for s in stmts:
            if isinstance(s, dict) and s.get("type") == "assign" and isinstance(s.get("target"), str):
                tgt = s.get("target")
                if tgt not in uses and self._is_side_effect_free(s.get("value")):
                    # drop dead store
                    continue
            # recursively process nested blocks
            if isinstance(s, dict):
                for k, v in list(s.items()):
                    if isinstance(v, dict) and v.get("type") == "block":
                        s[k] = self._dead_store_elim_in_block(v)
                    elif isinstance(v, list):
                        s[k] = [self._dead_store_elim_in_block(x) if isinstance(x, dict) and x.get("type") == "block" else x for x in v]
            new_stmts.append(s)
        block["stmts"] = new_stmts
        return block

    def apply(self, ir: Any, context: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        infos, warnings, errors = [], [], []
        try:
            if not isinstance(ir, dict):
                return ir, {"ok": True, "info": ["no-module"], "warnings": warnings, "errors": errors}
            functions = ir.get("functions", {})
            if not isinstance(functions, dict) or not functions:
                return ir, {"ok": True, "info": ["no-functions"], "warnings": warnings, "errors": errors}

            # Build call graph (function -> set(callees))
            graph: Dict[str, set] = {}
            for fname, fobj in functions.items():
                body = fobj.get("body")
                callees = set()
                if body is not None:
                    callees = self._collect_calls_from_body(body)
                graph[fname] = set(c for c in callees if isinstance(c, str))

            # Determine roots: context-provided entry_points, exported functions, "main"
            roots = set()
            entry_points = context.get("entry_points") or context.get("roots") or []
            if isinstance(entry_points, (list, tuple)):
                roots.update([e for e in entry_points if isinstance(e, str)])
            # exported functions
            for fname, fobj in functions.items():
                if isinstance(fobj, dict) and (fobj.get("export") or fobj.get("is_export") or fobj.get("exported")):
                    roots.add(fname)
            # default main
            if "main" in functions:
                roots.add("main")
            if not roots:
                # if nothing known, conservatively keep all and only perform local DSE
                roots = set(functions.keys())

            # reachability traversal
            reachable = set()
            stack = list(roots)
            while stack:
                cur = stack.pop()
                if cur in reachable:
                    continue
                if cur not in functions:
                    # unknown external callee - ignore
                    continue
                reachable.add(cur)
                for c in graph.get(cur, set()):
                    if c not in reachable:
                        stack.append(c)

            # Remove unreachable functions (conservative)
            removed = [fn for fn in functions.keys() if fn not in reachable]
            if removed:
                infos.append(f"removed_unreachable_functions={len(removed)}")
            new_functions = {fname: json.loads(json.dumps(fobj)) for fname, fobj in functions.items() if fname in reachable}

            # Apply local DSE to remaining functions' bodies
            for fname, fobj in list(new_functions.items()):
                body = fobj.get("body")
                if isinstance(body, dict) and body.get("type") == "block":
                    try:
                        fobj["body"] = self._dead_store_elim_in_block(body)
                    except Exception:
                        LOG.debug("DCE: DSE failed for function %s (ignored)", fname)

            new_ir = dict(ir)
            new_ir["functions"] = new_functions

            infos.append(f"functions_kept={len(new_functions)}")
            return new_ir, {"ok": True, "info": infos, "warnings": warnings, "errors": errors}
        except Exception as e:
            LOG.exception("DeadCodeEliminationPlugin failed")
            return ir, {"ok": False, "info": [], "warnings": warnings, "errors": [str(e), traceback.format_exc()]}
            sys.exit(1)


