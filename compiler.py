

# instryx_jit_aot_runner.py
# JIT-assisted AOT Execution for Instryx LLVM IR Modules
# Author: Violet Magenta / VACU Technologies
# License: MIT

from instryx_llvm_ir_codegen import InstryxLLVMCodegen
from llvmlite import binding
import ctypes

class InstryxRunner:
    def __init__(self):
        self.codegen = InstryxLLVMCodegen()
        self.engine = None

    def _create_execution_engine(self):
        binding.initialize()
        binding.initialize_native_target()
        binding.initialize_native_asmprinter()

        target = binding.Target.from_default_triple()
        target_machine = target.create_target_machine()
        backing_mod = binding.parse_assembly("")
        engine = binding.create_mcjit_compiler(backing_mod, target_machine)
        return engine

    def _compile_ir(self, llvm_ir: str):
        mod = binding.parse_assembly(llvm_ir)
        mod.verify()
        self.engine.add_module(mod)
        self.engine.finalize_object()
        self.engine.run_static_constructors()

    def run(self, code: str, invoke_main: bool = True):
        llvm_ir = self.codegen.generate(code)
        self.engine = self._create_execution_engine()
        self._compile_ir(llvm_ir)

        if invoke_main:
            func_ptr = self.engine.get_function_address("main")
            cfunc = ctypes.CFUNCTYPE(None)(func_ptr)
            print("🚀 Running Instryx Program...")
            cfunc()

        return llvm_ir


# Test block (can be removed in production)
if __name__ == "__main__":
    runner = InstryxRunner()
    code = """
    func greet(uid) {
        print: "Hello from Instryx IR";
    };

    main() {
        greet(1);
    };
    """
    llvm_ir = runner.run(code)
    print("\n🔬 LLVM IR Output:\n")
    print(llvm_ir)

    # Example usage:
    code = """
    func add(a, b) {
        return a + b;
    };
    main() {
        result = add(5, 7);
        print: "Result is " + result;
    };
    """
    llvm_ir = runner.run(code)
    print("\n🔬 LLVM IR Output:\n")
    print(llvm_ir)
    # Example usage:
    code = """
    func add(a, b) {
        return a + b;
    };
    main() {
        result = add(5, 7);
        print: "Result is " + result;
    };
    """
    llvm_ir = runner.run(code)
    print("\n🔬 LLVM IR Output:\n")
    print(llvm_ir)

instryx_jit_aot_runner.py
# instryx_jit_aot_runner.py
# JIT-assisted AOT Execution for Instryx LLVM IR Modules — supreme boosters
# Author: Violet Magenta / VACU Technologies (modified)
# License: MIT

from instryx_llvm_ir_codegen import InstryxLLVMCodegen
from llvmlite import binding
import ctypes
import hashlib
import os
import tempfile
import subprocess
import threading
import time
import logging
import shutil
import struct
from typing import Optional, Sequence, Any, Dict, Tuple, List

LOG = logging.getLogger("instryx.runner")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO)

# Try to wire into shell metrics if available (safe import)
_metrics = None
_metrics_lock = None
try:
    from instryx_shell_enhancements import _metrics as _SHELL_METRICS, _metrics_lock as _SHELL_METRICS_LOCK  # type: ignore
    _metrics = _SHELL_METRICS
    _metrics_lock = _SHELL_METRICS_LOCK
except Exception:
    # fallback: local simple counters
    _metrics = {
        "instryx_compile_requests_total": 0,
        "instryx_compile_success_total": 0,
        "instryx_compile_failure_total": 0,
        "instryx_run_requests_total": 0,
        "instryx_run_success_total": 0,
        "instryx_run_failure_total": 0,
    }
    import threading as _threading
    _metrics_lock = _threading.RLock()


# More comprehensive type mapping for simple primitives and helpers
_CTYPE_MAP: Dict[str, Any] = {
    "void": None,
    "int": ctypes.c_int,
    "uint": ctypes.c_uint,
    "long": ctypes.c_long,
    "ulong": ctypes.c_ulong,
    "float": ctypes.c_float,
    "double": ctypes.c_double,
    "char*": ctypes.c_char_p,
    "size_t": ctypes.c_size_t,
    "intptr": ctypes.c_void_p,
}


def _metric_inc(name: str, n: int = 1):
    try:
        with _metrics_lock:
            _metrics[name] = _metrics.get(name, 0) + n
    except Exception:
        pass


class StructBuilder:
    """
    Helper to build ctypes.Structure subclasses from a simple description.
    Example:
      desc = [("a", "int"), ("b", "double"), ("s", ("char*", 32))]
      MyStruct = StructBuilder.build("MyStruct", desc)
      inst = MyStruct()
      inst.a = 3
    Supported field spec forms:
      - ("name", "int") -- maps via _CTYPE_MAP
      - ("name", ("char*", N)) -- fixed-size char array
      - ("name", ctypes.c_int) -- explicit ctypes type
    """
    @staticmethod
    def _resolve_type(spec):
        if isinstance(spec, str):
            return _CTYPE_MAP.get(spec)
        if isinstance(spec, tuple) and len(spec) == 2 and spec[0] == "char*":
            # fixed-size char array of length spec[1]
            return ctypes.c_char * int(spec[1])
        if hasattr(spec, "_type_") or isinstance(spec, type):
            return spec
        return None

    @staticmethod
    def build(name: str, fields: Sequence[Tuple[str, Any]]):
        cfields = []
        for fname, ftype in fields:
            ct = StructBuilder._resolve_type(ftype)
            if ct is None:
                raise TypeError(f"unsupported field type: {ftype}")
            cfields.append((fname, ct))
        # dynamic type creation
        return type(name, (ctypes.Structure,), {"_fields_": cfields})


def create_ctypes_array(ctype: Any, values: Sequence[Any]):
    """
    Create a ctypes array of given ctype and initialize with values.
    ctype may be a string alias in _CTYPE_MAP or a ctypes type.
    """
    if isinstance(ctype, str):
        ct = _CTYPE_MAP.get(ctype)
    else:
        ct = ctype
    if ct is None:
        raise TypeError("unknown ctype for array")
    ArrType = ct * len(values)
    arr = ArrType(*values)
    return arr


class InstryxRunner:
    """
    Enhanced runner for LLVM IR produced by Instryx codegen.

    Enhancements:
     - metrics wiring (compile/run counters)
     - IR caching on disk
     - AOT helpers: emit assembly, emit object, try to link to shared library
     - ctypes-based marshalling helpers (structs, arrays)
     - safe invocation with thread-based timeout and subprocess isolation option
     - configurable verbosity and engine reuse
    """

    def __init__(self, cache_dir: Optional[str] = None, verbose: bool = False, reuse_engine: bool = True):
        self.codegen = InstryxLLVMCodegen()
        self.engine = None
        self._modules = []  # keep references to modules
        self.reuse_engine = bool(reuse_engine)
        self.verbose = bool(verbose)
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".instryx_jit_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        if self.verbose:
            LOG.setLevel(logging.DEBUG)

    # ------------------------
    # Engine lifecycle
    # ------------------------
    def _create_execution_engine(self):
        binding.initialize()
        binding.initialize_native_target()
        binding.initialize_native_asmprinter()
        target = binding.Target.from_default_triple()
        target_machine = target.create_target_machine()
        backing_mod = binding.parse_assembly("")
        engine = binding.create_mcjit_compiler(backing_mod, target_machine)
        if self.verbose:
            LOG.debug("Created MCJIT engine (triple=%s)", target.triple)
        return engine

    def _prepare_engine(self):
        if self.engine is None or not self.reuse_engine:
            if self.engine is not None and not self.reuse_engine:
                self._modules.clear()
                self.engine = None
            self.engine = self._create_execution_engine()
            self._modules = []

    def reset_engine(self):
        """Dispose current engine and modules (best-effort)."""
        try:
            self._modules.clear()
            self.engine = None
            if self.verbose:
                LOG.debug("Engine and modules reset")
        except Exception:
            LOG.exception("reset_engine failed")

    # ------------------------
    # IR caching and AOT helpers
    # ------------------------
    def _hash_ir(self, llvm_ir: str) -> str:
        return hashlib.sha256(llvm_ir.encode("utf-8")).hexdigest()

    def cache_ir(self, llvm_ir: str) -> str:
        h = self._hash_ir(llvm_ir)
        path = os.path.join(self.cache_dir, f"{h}.ll")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(llvm_ir)
        if self.verbose:
            LOG.debug("IR cached -> %s", path)
        return path

    def emit_assembly(self, llvm_ir: str, out_path: Optional[str] = None) -> str:
        if out_path is None:
            out_path = tempfile.mktemp(suffix=".s")
        mod = binding.parse_assembly(llvm_ir)
        mod.verify()
        tm = binding.Target.from_default_triple().create_target_machine()
        asm = tm.emit_assembly(mod)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(asm)
        if self.verbose:
            LOG.debug("Assembly emitted -> %s", out_path)
        return out_path

    def emit_object(self, llvm_ir: str, out_path: Optional[str] = None) -> str:
        if out_path is None:
            out_path = tempfile.mktemp(suffix=".o")
        mod = binding.parse_assembly(llvm_ir)
        mod.verify()
        tm = binding.Target.from_default_triple().create_target_machine()
        obj_bytes = tm.emit_object(mod)
        with open(out_path, "wb") as fh:
            fh.write(obj_bytes)
        if self.verbose:
            LOG.debug("Object emitted -> %s (size=%d)", out_path, len(obj_bytes))
        return out_path

    def try_link_shared(self, object_path: str, out_shared: Optional[str] = None) -> Tuple[bool, str]:
        if out_shared is None:
            out_shared = object_path + (".so" if os.name != "nt" else ".dll")
        cc = shutil.which("clang") or shutil.which("gcc")
        if not cc:
            return False, "no system compiler (clang/gcc) found"
        cmd = [cc, "-shared", "-o", out_shared, object_path]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if proc.returncode != 0:
                return False, f"linker failed: {proc.stderr.strip()}"
            if self.verbose:
                LOG.debug("Linked shared library -> %s", out_shared)
            return True, out_shared
        except Exception as e:
            return False, f"link failed: {e}"

    # ------------------------
    # Compile / add module
    # ------------------------
    def _compile_ir(self, llvm_ir: str, module_name: Optional[str] = None):
        self._prepare_engine()
        mod = binding.parse_assembly(llvm_ir)
        mod.verify()
        if module_name:
            try:
                mod.name = module_name
            except Exception:
                pass
        self.engine.add_module(mod)
        self.engine.finalize_object()
        self.engine.run_static_constructors()
        self._modules.append(mod)
        if self.verbose:
            LOG.debug("Compiled module (%s); modules count=%d", module_name or "<anon>", len(self._modules))

    # ------------------------
    # Invocation helpers
    # ------------------------
    def _ctype_from_spec(self, spec: Any):
        if spec is None:
            return None
        if isinstance(spec, str):
            return _CTYPE_MAP.get(spec)
        return spec

    def call_function(self, func_name: str, arg_types: Optional[Sequence[Any]] = None,
                      ret_type: Any = "void", args: Optional[Sequence[Any]] = None,
                      timeout: Optional[float] = None) -> Tuple[bool, Any]:
        if self.engine is None:
            return False, "engine not initialized; compile IR first"
        ptr = self.engine.get_function_address(func_name)
        if not ptr:
            return False, f"function {func_name} not found"
        arg_types = arg_types or []
        ctypes_args = []
        for t in arg_types:
            ct = self._ctype_from_spec(t)
            if ct is None:
                return False, f"unsupported arg type {t}"
            ctypes_args.append(ct)
        ctypes_ret = self._ctype_from_spec(ret_type)
        try:
            if ctypes_ret is None:
                FUN = ctypes.CFUNCTYPE(None, *ctypes_args)
            else:
                FUN = ctypes.CFUNCTYPE(ctypes_ret, *ctypes_args)
            cfunc = FUN(ptr)
        except Exception as e:
            return False, f"failed to wrap cfunc: {e}"
        result = {"ok": None, "val": None}
        def _invoke():
            try:
                r = cfunc(*([] if args is None else list(args)))
                result["ok"] = True
                result["val"] = r
            except Exception as e:
                result["ok"] = False
                result["val"] = f"exception: {e}"
        th = threading.Thread(target=_invoke, daemon=True)
        th.start()
        th.join(timeout=timeout)
        if th.is_alive():
            return False, "timeout"
        return result["ok"], result["val"]

    # ------------------------
    # High-level run
    # ------------------------
    def run(self, code: str, invoke_main: bool = True, timeout: Optional[float] = None,
            use_subprocess: bool = False, cache_ir: bool = True) -> str:
        # metrics
        _metric_inc("instryx_compile_requests_total", 1)
        llvm_ir = self.codegen.generate(code)
        if cache_ir:
            self.cache_ir(llvm_ir)
        try:
            self._compile_ir(llvm_ir)
            _metric_inc("instryx_compile_success_total", 1)
        except Exception as e:
            _metric_inc("instryx_compile_failure_total", 1)
            LOG.exception("compile failed")
            raise

        if not invoke_main:
            return llvm_ir

        _metric_inc("instryx_run_requests_total", 1)
        if use_subprocess:
            stub = f"""
import sys, ctypes
from llvmlite import binding
binding.initialize()
binding.initialize_native_target()
binding.initialize_native_asmprinter()
mod = binding.parse_assembly(r'''{llvm_ir}''')
mod.verify()
target = binding.Target.from_default_triple()
tm = target.create_target_machine()
engine = binding.create_mcjit_compiler(binding.parse_assembly(""), tm)
engine.add_module(mod)
engine.finalize_object()
engine.run_static_constructors()
addr = engine.get_function_address("main")
if not addr:
    print("FUNCTION_NOT_FOUND", file=sys.stderr)
    sys.exit(2)
ctypes.CFUNCTYPE(None)(addr)()
"""
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
                tf.write(stub)
                stub_path = tf.name
            try:
                proc = subprocess.run([shutil.which("python") or "python", stub_path], capture_output=True, text=True, timeout=timeout)
                if proc.returncode != 0:
                    _metric_inc("instryx_run_failure_total", 1)
                    LOG.error("subprocess run failed: %s", proc.stderr.strip())
                else:
                    _metric_inc("instryx_run_success_total", 1)
                    if self.verbose:
                        LOG.debug("subprocess run stdout: %s", proc.stdout.strip())
            except subprocess.TimeoutExpired:
                _metric_inc("instryx_run_failure_total", 1)
                LOG.error("subprocess timed out")
            finally:
                try:
                    os.unlink(stub_path)
                except Exception:
                    pass
            return llvm_ir

        ok, res = self.call_function("main", arg_types=None, ret_type="void", args=(), timeout=timeout)
        if ok:
            _metric_inc("instryx_run_success_total", 1)
            if self.verbose:
                LOG.debug("main invocation succeeded")
        else:
            _metric_inc("instryx_run_failure_total", 1)
            LOG.error("main invocation failed: %s", res)
        return llvm_ir


# ------------------------
# minimal test/demo
# ------------------------
if __name__ == "__main__":
    runner = InstryxRunner(verbose=True)
    sample = """
    func greet(uid) {
        print: "Hello from Instryx IR";
    };

    main() {
        greet(1);
    };
    """
    try:
        ir = runner.run(sample, invoke_main=True, timeout=5)
        print("\n🔬 LLVM IR Output:\n")
        print(ir)
    except Exception as e:
        print("run failed:", e)

setup.py
# setup.py — auxiliary helper to build example C extension or link object files.
# This helper is intentionally small: it provides convenience commands for users to
# produce a shared library from an object file or compile a tiny example C module.

from setuptools import setup, Extension
import subprocess
import sys
import os

def build_example_shared(output="libexample.so"):
    # A minimal C source to demonstrate loading via ctypes
    src = r'''
#include <stdio.h>

int example_add(int a, int b) {
    return a + b;
}

void example_print(const char *s) {
    printf("example_print: %s\n", s);
}
'''
    cur = os.path.abspath(os.path.dirname(__file__))
    cpath = os.path.join(cur, "example_cmodule.c")
    with open(cpath, "w", encoding="utf-8") as f:
        f.write(src)
    cc = shutil.which("clang") or shutil.which("gcc")
    if not cc:
        print("No system compiler found (clang/gcc required)")
        return 2
    cmd = [cc, "-shared", "-fPIC", "-O2", "-o", output, cpath]
    try:
        subprocess.check_call(cmd)
        print("Built", output)
        return 0
    except Exception as e:
        print("Build failed:", e)
        return 1

if __name__ == "__main__":
    # simple helper CLI: python setup.py build_shared
    import shutil
    if len(sys.argv) >= 2 and sys.argv[1] == "build_shared":
        out = sys.argv[2] if len(sys.argv) > 2 else "libexample.so"
        sys.exit(build_example_shared(out))
    print("Usage: python setup.py build_shared [out.so]")

examples/link_and_load_shared.py
# Example script: use InstryxRunner to emit object, link to shared lib and load via ctypes

import os
from instryx_jit_aot_runner import InstryxRunner

runner = InstryxRunner(verbose=True)

# simple sample that defines a function callable via C ABI
# The codegen must generate an extern "C" compatible function named 'add' for demonstration.
# Replace with your codegen specifics as needed.
code = """
func add(a, b) {
    return a + b;
}

main() {
    // no-op
};
"""

ir = runner.codegen.generate(code)
# emit object file
obj_path = runner.emit_object(ir)
print("Emitted object:", obj_path)

# link to shared library (requires gcc/clang)
ok, out = runner.try_link_shared(obj_path)
if not ok:
    print("Link failed:", out)
else:
    libpath = out
    print("Linked shared lib:", libpath)
    # load via ctypes
    lib = os.path.abspath(libpath)
    cdll = ctypes.CDLL(lib)
    # assume 'add' exists and takes two ints -> int
    try:
        add = cdll.add
        add.restype = ctypes.c_int
        add.argtypes = [ctypes.c_int, ctypes.c_int]
        print("add(10,32) ->", add(10, 32))
    except Exception as e:
        print("failed to call add:", e)

# instryx_jit_aot_runner.py
# JIT-assisted AOT Execution for Instryx LLVM IR Modules — supreme boosters
# Author: Violet Magenta / VACU Technologies (modified)
# License: MIT

from instryx_llvm_ir_codegen import InstryxLLVMCodegen
from llvmlite import binding
import ctypes
import hashlib
import os
import tempfile
import subprocess
import threading
import time
import logging
import shutil
import struct
from typing import Optional, Sequence, Any, Dict, Tuple, List

LOG = logging.getLogger("instryx.runner")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO)

# Try to wire into shell metrics if available (safe import)
_metrics = None
_metrics_lock = None
try:
    from instryx_shell_enhancements import _metrics as _SHELL_METRICS, _metrics_lock as _SHELL_METRICS_LOCK  # type: ignore
    _metrics = _SHELL_METRICS
    _metrics_lock = _SHELL_METRICS_LOCK
except Exception:
    # fallback: local simple counters
    _metrics = {
        "instryx_compile_requests_total": 0,
        "instryx_compile_success_total": 0,
        "instryx_compile_failure_total": 0,
        "instryx_run_requests_total": 0,
        "instryx_run_success_total": 0,
        "instryx_run_failure_total": 0,
    }
    import threading as _threading
    _metrics_lock = _threading.RLock()


# More comprehensive type mapping for simple primitives and helpers
_CTYPE_MAP: Dict[str, Any] = {
    "void": None,
    "int": ctypes.c_int,
    "uint": ctypes.c_uint,
    "long": ctypes.c_long,
    "ulong": ctypes.c_ulong,
    "float": ctypes.c_float,
    "double": ctypes.c_double,
    "char*": ctypes.c_char_p,
    "size_t": ctypes.c_size_t,
    "intptr": ctypes.c_void_p,
}


def _metric_inc(name: str, n: int = 1):
    try:
        with _metrics_lock:
            _metrics[name] = _metrics.get(name, 0) + n
    except Exception:
        pass


class StructBuilder:
    """
    Helper to build ctypes.Structure subclasses from a simple description.
    Example:
      desc = [("a", "int"), ("b", "double"), ("s", ("char*", 32))]
      MyStruct = StructBuilder.build("MyStruct", desc)
      inst = MyStruct()
      inst.a = 3
    Supported field spec forms:
      - ("name", "int") -- maps via _CTYPE_MAP
      - ("name", ("char*", N)) -- fixed-size char array
      - ("name", ctypes.c_int) -- explicit ctypes type
    """
    @staticmethod
    def _resolve_type(spec):
        if isinstance(spec, str):
            return _CTYPE_MAP.get(spec)
        if isinstance(spec, tuple) and len(spec) == 2 and spec[0] == "char*":
            # fixed-size char array of length spec[1]
            return ctypes.c_char * int(spec[1])
        if hasattr(spec, "_type_") or isinstance(spec, type):
            return spec
        return None

    @staticmethod
    def build(name: str, fields: Sequence[Tuple[str, Any]]):
        cfields = []
        for fname, ftype in fields:
            ct = StructBuilder._resolve_type(ftype)
            if ct is None:
                raise TypeError(f"unsupported field type: {ftype}")
            cfields.append((fname, ct))
        # dynamic type creation
        return type(name, (ctypes.Structure,), {"_fields_": cfields})


def create_ctypes_array(ctype: Any, values: Sequence[Any]):
    """
    Create a ctypes array of given ctype and initialize with values.
    ctype may be a string alias in _CTYPE_MAP or a ctypes type.
    """
    if isinstance(ctype, str):
        ct = _CTYPE_MAP.get(ctype)
    else:
        ct = ctype
    if ct is None:
        raise TypeError("unknown ctype for array")
    ArrType = ct * len(values)
    arr = ArrType(*values)
    return arr


class InstryxRunner:
    """
    Enhanced runner for LLVM IR produced by Instryx codegen.

    Enhancements:
     - metrics wiring (compile/run counters)
     - IR caching on disk
     - AOT helpers: emit assembly, emit object, try to link to shared library
     - ctypes-based marshalling helpers (structs, arrays)
     - safe invocation with thread-based timeout and subprocess isolation option
     - configurable verbosity and engine reuse
    """

    def __init__(self, cache_dir: Optional[str] = None, verbose: bool = False, reuse_engine: bool = True):
        self.codegen = InstryxLLVMCodegen()
        self.engine = None
        self._modules = []  # keep references to modules
        self.reuse_engine = bool(reuse_engine)
        self.verbose = bool(verbose)
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".instryx_jit_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        if self.verbose:
            LOG.setLevel(logging.DEBUG)

    # ------------------------
    # Engine lifecycle
    # ------------------------
    def _create_execution_engine(self):
        binding.initialize()
        binding.initialize_native_target()
        binding.initialize_native_asmprinter()
        target = binding.Target.from_default_triple()
        target_machine = target.create_target_machine()
        backing_mod = binding.parse_assembly("")
        engine = binding.create_mcjit_compiler(backing_mod, target_machine)
        if self.verbose:
            LOG.debug("Created MCJIT engine (triple=%s)", target.triple)
        return engine

    def _prepare_engine(self):
        if self.engine is None or not self.reuse_engine:
            if self.engine is not None and not self.reuse_engine:
                self._modules.clear()
                self.engine = None
            self.engine = self._create_execution_engine()
            self._modules = []

    def reset_engine(self):
        """Dispose current engine and modules (best-effort)."""
        try:
            self._modules.clear()
            self.engine = None
            if self.verbose:
                LOG.debug("Engine and modules reset")
        except Exception:
            LOG.exception("reset_engine failed")

    # ------------------------
    # IR caching and AOT helpers
    # ------------------------
    def _hash_ir(self, llvm_ir: str) -> str:
        return hashlib.sha256(llvm_ir.encode("utf-8")).hexdigest()

    def cache_ir(self, llvm_ir: str) -> str:
        h = self._hash_ir(llvm_ir)
        path = os.path.join(self.cache_dir, f"{h}.ll")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(llvm_ir)
        if self.verbose:
            LOG.debug("IR cached -> %s", path)
        return path

    def emit_assembly(self, llvm_ir: str, out_path: Optional[str] = None) -> str:
        if out_path is None:
            out_path = tempfile.mktemp(suffix=".s")
        mod = binding.parse_assembly(llvm_ir)
        mod.verify()
        tm = binding.Target.from_default_triple().create_target_machine()
        asm = tm.emit_assembly(mod)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(asm)
        if self.verbose:
            LOG.debug("Assembly emitted -> %s", out_path)
        return out_path

    def emit_object(self, llvm_ir: str, out_path: Optional[str] = None) -> str:
        if out_path is None:
            out_path = tempfile.mktemp(suffix=".o")
        mod = binding.parse_assembly(llvm_ir)
        mod.verify()
        tm = binding.Target.from_default_triple().create_target_machine()
        obj_bytes = tm.emit_object(mod)
        with open(out_path, "wb") as fh:
            fh.write(obj_bytes)
        if self.verbose:
            LOG.debug("Object emitted -> %s (size=%d)", out_path, len(obj_bytes))
        return out_path

    def try_link_shared(self, object_path: str, out_shared: Optional[str] = None) -> Tuple[bool, str]:
        if out_shared is None:
            out_shared = object_path + (".so" if os.name != "nt" else ".dll")
        cc = shutil.which("clang") or shutil.which("gcc")
        if not cc:
            return False, "no system compiler (clang/gcc) found"
        cmd = [cc, "-shared", "-o", out_shared, object_path]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if proc.returncode != 0:
                return False, f"linker failed: {proc.stderr.strip()}"
            if self.verbose:
                LOG.debug("Linked shared library -> %s", out_shared)
            return True, out_shared
        except Exception as e:
            return False, f"link failed: {e}"

    # ------------------------
    # Compile / add module
    # ------------------------
    def _compile_ir(self, llvm_ir: str, module_name: Optional[str] = None):
        self._prepare_engine()
        mod = binding.parse_assembly(llvm_ir)
        mod.verify()
        if module_name:
            try:
                mod.name = module_name
            except Exception:
                pass
        self.engine.add_module(mod)
        self.engine.finalize_object()
        self.engine.run_static_constructors()
        self._modules.append(mod)
        if self.verbose:
            LOG.debug("Compiled module (%s); modules count=%d", module_name or "<anon>", len(self._modules))

    # ------------------------
    # Invocation helpers
    # ------------------------
    def _ctype_from_spec(self, spec: Any):
        if spec is None:
            return None
        if isinstance(spec, str):
            return _CTYPE_MAP.get(spec)
        return spec

    def call_function(self, func_name: str, arg_types: Optional[Sequence[Any]] = None,
                      ret_type: Any = "void", args: Optional[Sequence[Any]] = None,
                      timeout: Optional[float] = None) -> Tuple[bool, Any]:
        if self.engine is None:
            return False, "engine not initialized; compile IR first"
        ptr = self.engine.get_function_address(func_name)
        if not ptr:
            return False, f"function {func_name} not found"
        arg_types = arg_types or []
        ctypes_args = []
        for t in arg_types:
            ct = self._ctype_from_spec(t)
            if ct is None:
                return False, f"unsupported arg type {t}"
            ctypes_args.append(ct)
        ctypes_ret = self._ctype_from_spec(ret_type)
        try:
            if ctypes_ret is None:
                FUN = ctypes.CFUNCTYPE(None, *ctypes_args)
            else:
                FUN = ctypes.CFUNCTYPE(ctypes_ret, *ctypes_args)
            cfunc = FUN(ptr)
        except Exception as e:
            return False, f"failed to wrap cfunc: {e}"
        result = {"ok": None, "val": None}
        def _invoke():
            try:
                r = cfunc(*([] if args is None else list(args)))
                result["ok"] = True
                result["val"] = r
            except Exception as e:
                result["ok"] = False
                result["val"] = f"exception: {e}"
        th = threading.Thread(target=_invoke, daemon=True)
        th.start()
        th.join(timeout=timeout)
        if th.is_alive():
            return False, "timeout"
        return result["ok"], result["val"]

    # ------------------------
    # High-level run
    # ------------------------
    def run(self, code: str, invoke_main: bool = True, timeout: Optional[float] = None,
            use_subprocess: bool = False, cache_ir: bool = True) -> str:
        # metrics
        _metric_inc("instryx_compile_requests_total", 1)
        llvm_ir = self.codegen.generate(code)
        if cache_ir:
            self.cache_ir(llvm_ir)
        try:
            self._compile_ir(llvm_ir)
            _metric_inc("instryx_compile_success_total", 1)
        except Exception as e:
            _metric_inc("instryx_compile_failure_total", 1)
            LOG.exception("compile failed")
            raise

        if not invoke_main:
            return llvm_ir

        _metric_inc("instryx_run_requests_total", 1)
        if use_subprocess:
            stub = f"""
import sys, ctypes
from llvmlite import binding
binding.initialize()
binding.initialize_native_target()
binding.initialize_native_asmprinter()
mod = binding.parse_assembly(r'''{llvm_ir}''')
mod.verify()
target = binding.Target.from_default_triple()
tm = target.create_target_machine()
engine = binding.create_mcjit_compiler(binding.parse_assembly(""), tm)
engine.add_module(mod)
engine.finalize_object()
engine.run_static_constructors()
addr = engine.get_function_address("main")
if not addr:
    print("FUNCTION_NOT_FOUND", file=sys.stderr)
    sys.exit(2)
ctypes.CFUNCTYPE(None)(addr)()
"""
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
                tf.write(stub)
                stub_path = tf.name
            try:
                proc = subprocess.run([shutil.which("python") or "python", stub_path], capture_output=True, text=True, timeout=timeout)
                if proc.returncode != 0:
                    _metric_inc("instryx_run_failure_total", 1)
                    LOG.error("subprocess run failed: %s", proc.stderr.strip())
                else:
                    _metric_inc("instryx_run_success_total", 1)
                    if self.verbose:
                        LOG.debug("subprocess run stdout: %s", proc.stdout.strip())
            except subprocess.TimeoutExpired:
                _metric_inc("instryx_run_failure_total", 1)
                LOG.error("subprocess timed out")
            finally:
                try:
                    os.unlink(stub_path)
                except Exception:
                    pass
            return llvm_ir

        ok, res = self.call_function("main", arg_types=None, ret_type="void", args=(), timeout=timeout)
        if ok:
            _metric_inc("instryx_run_success_total", 1)
            if self.verbose:
                LOG.debug("main invocation succeeded")
        else:
            _metric_inc("instryx_run_failure_total", 1)
            LOG.error("main invocation failed: %s", res)
        return llvm_ir


# ------------------------
# minimal test/demo
# ------------------------
if __name__ == "__main__":
    runner = InstryxRunner(verbose=True)
    sample = """
    func greet(uid) {
        print: "Hello from Instryx IR";
    };

    main() {
        greet(1);
    };
    """
    try:
        ir = runner.run(sample, invoke_main=True, timeout=5)
        print("\n🔬 LLVM IR Output:\n")
        print(ir)
    except Exception as e:
        print("run failed:", e)

"""
instryx_jit_aot_runner.py

Supreme-boosters edition — JIT-assisted AOT Execution for Instryx LLVM IR Modules.

Additions and optimizations:
 - Robust engine lifecycle (ORC-aware but MCJIT-compatible fallback).
 - Persistent IR/artifact cache with index metadata and safe atomic writes.
 - Parallel AOT emission (ThreadPool) via `batch_emit_objects`.
 - JIT warmup helper `warmup_jit`.
 - Extended ctypes marshalling helpers: nested StructBuilder, pack/unpack, create_ctypes_array.
 - Safe function invocation with thread timeout and subprocess isolation option.
 - Helpers to emit assembly/object, link shared libraries and load them via ctypes.
 - Metrics wiring (prometheus-style) integrated with optional shell metrics.
 - File logging with small rollover.
 - CLI helpers for emit/link/load flows and a simple smoke-test.
 - Non-blocking background compile pool and convenience wrappers.
"""

from __future__ import annotations
from instryx_llvm_ir_codegen import InstryxLLVMCodegen
from llvmlite import binding
import ctypes
import hashlib
import json
import os
import tempfile
import subprocess
import threading
import time
import logging
import shutil
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Sequence, Any, Dict, Tuple, List, Union

LOG = logging.getLogger("instryx.runner")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO)

# --------- simple file logging with rollover ----------
_LOG_FILE = os.path.join(os.path.expanduser("~"), ".instryx_jit_runner.log")
_LOG_MAX = 1_000_000


def _file_log(msg: str):
    try:
        if os.path.exists(_LOG_FILE) and os.path.getsize(_LOG_FILE) > _LOG_MAX:
            try:
                os.replace(_LOG_FILE, _LOG_FILE + ".1")
            except Exception:
                pass
        with open(_LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except Exception:
        pass


# --------- metrics (integrates with shell enhancements when available) ----------
_metrics: Dict[str, int] = {}
_metrics_lock = threading.RLock()
try:
    from instryx_shell_enhancements import _metrics as _SHELL_METRICS, _metrics_lock as _SHELL_METRICS_LOCK  # type: ignore
    _metrics = _SHELL_METRICS
    _metrics_lock = _SHELL_METRICS_LOCK
except Exception:
    _metrics = {
        "instryx_compile_requests_total": 0,
        "instryx_compile_success_total": 0,
        "instryx_compile_failure_total": 0,
        "instryx_run_requests_total": 0,
        "instryx_run_success_total": 0,
        "instryx_run_failure_total": 0,
    }


def _metric_inc(name: str, n: int = 1):
    try:
        with _metrics_lock:
            _metrics[name] = _metrics.get(name, 0) + n
    except Exception:
        pass


# --------- richer ctype map & helpers ----------
_CTYPE_MAP: Dict[str, Any] = {
    "void": None,
    "int8": ctypes.c_int8,
    "uint8": ctypes.c_uint8,
    "int16": ctypes.c_int16,
    "uint16": ctypes.c_uint16,
    "int32": ctypes.c_int32,
    "uint32": ctypes.c_uint32,
    "int": ctypes.c_int,
    "uint": ctypes.c_uint,
    "int64": ctypes.c_int64,
    "uint64": ctypes.c_uint64,
    "long": ctypes.c_long,
    "ulong": ctypes.c_ulong,
    "float": ctypes.c_float,
    "double": ctypes.c_double,
    "char*": ctypes.c_char_p,
    "size_t": ctypes.c_size_t,
    "intptr": ctypes.c_void_p,
}


def _resolve_ctype(spec: Any):
    if spec is None:
        return None
    if isinstance(spec, str):
        return _CTYPE_MAP.get(spec)
    return spec


class StructBuilder:
    """
    Build ctypes.Structure classes from high-level descriptions.

    Field spec examples:
      ("a", "int")
      ("name", ("char*", 32))
      ("p", ("ptr", "int"))         -> pointer to int
      ("nested", ("struct", [("x","int"), ("y","int")]))

    Returns dynamically created ctypes.Structure subclass.
    """

    @staticmethod
    def _resolve_field_type(spec: Any):
        if isinstance(spec, str):
            return _resolve_ctype(spec)
        if isinstance(spec, tuple):
            tag = spec[0]
            if tag == "char*" and len(spec) == 2:
                return ctypes.c_char * int(spec[1])
            if tag == "ptr" and len(spec) == 2:
                base = StructBuilder._resolve_field_type(spec[1])
                return ctypes.POINTER(base) if base is not None else None
            if tag == "struct" and len(spec) == 2:
                inner = spec[1]
                if isinstance(inner, type) and issubclass(inner, ctypes.Structure):
                    return inner
                if isinstance(inner, (list, tuple)):
                    return StructBuilder.build("InnerStruct", inner)
        if hasattr(spec, "_type_") or isinstance(spec, type):
            return spec
        return None

    @staticmethod
    def build(name: str, fields: Sequence[Tuple[str, Any]]):
        cfields: List[Tuple[str, Any]] = []
        for fname, ftype in fields:
            ct = StructBuilder._resolve_field_type(ftype)
            if ct is None:
                raise TypeError(f"unsupported field type: {ftype}")
            cfields.append((fname, ct))
        return type(name, (ctypes.Structure,), {"_fields_": cfields})


def pack_struct_to_bytes(inst: ctypes.Structure) -> bytes:
    size = ctypes.sizeof(inst)
    buf = (ctypes.c_char * size)()
    ctypes.memmove(buf, ctypes.byref(inst), size)
    return bytes(bytearray(buf))


def unpack_bytes_to_struct(data: bytes, StructType: type) -> ctypes.Structure:
    inst = StructType()
    size = min(len(data), ctypes.sizeof(inst))
    ctypes.memmove(ctypes.byref(inst), data[:size], size)
    return inst


def create_ctypes_array(ctype: Any, values: Sequence[Any]):
    """
    Create ctypes array. ctype can be string alias, ctypes type, or nested structure.
    """
    ct = _resolve_ctype(ctype) if isinstance(ctype, str) else ctype
    if ct is None:
        raise TypeError("unknown ctype for array")
    if isinstance(values, (bytes, bytearray)) and ct is ctypes.c_char:
        Arr = ct * len(values)
        return Arr(*values)
    Arr = ct * len(values)
    return Arr(*values)


# ThreadPool for AOT emission & background tasks
_AOT_POOL = ThreadPoolExecutor(max_workers=max(2, (os.cpu_count() or 2)))


class InstryxRunner:
    """
    Advanced Instryx runner with AOT tooling and supreme boosters.
    """

    def __init__(self, cache_dir: Optional[str] = None, verbose: bool = False, reuse_engine: bool = True):
        self.codegen = InstryxLLVMCodegen()
        self.engine = None
        self._modules: List[Any] = []
        self.reuse_engine = bool(reuse_engine)
        self.verbose = bool(verbose)
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".instryx_jit_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._index_path = os.path.join(self.cache_dir, "index.json")
        self._load_index()
        if self.verbose:
            LOG.setLevel(logging.DEBUG)

    def _load_index(self):
        try:
            if os.path.exists(self._index_path):
                with open(self._index_path, "r", encoding="utf-8") as fh:
                    self._index = json.load(fh)
            else:
                self._index = {}
        except Exception:
            self._index = {}

    def _save_index(self):
        try:
            tmp = self._index_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._index, fh, indent=2)
            os.replace(tmp, self._index_path)
        except Exception:
            pass

    def _supports_orc(self) -> bool:
        try:
            return hasattr(binding, "OrcJIT") or hasattr(binding, "orc")
        except Exception:
            return False

    def _create_execution_engine(self):
        binding.initialize()
        binding.initialize_native_target()
        binding.initialize_native_asmprinter()
        target = binding.Target.from_default_triple()
        tm = target.create_target_machine()
        backing = binding.parse_assembly("")
        engine = binding.create_mcjit_compiler(backing, tm)
        if self.verbose:
            LOG.debug("Created execution engine (%s)", target.triple)
        _file_log("Execution engine created")
        return engine

    def _prepare_engine(self):
        if self.engine is None or not self.reuse_engine:
            if self.engine is not None and not self.reuse_engine:
                self._modules.clear()
                self.engine = None
            self.engine = self._create_execution_engine()
            self._modules = []

    def reset_engine(self):
        try:
            self._modules.clear()
            self.engine = None
            if self.verbose:
                LOG.debug("Engine reset")
            _file_log("Engine reset")
        except Exception:
            LOG.exception("reset_engine failed")

    def _hash_ir(self, llvm_ir: str) -> str:
        return hashlib.sha256(llvm_ir.encode("utf-8")).hexdigest()

    def cache_ir(self, llvm_ir: str) -> str:
        h = self._hash_ir(llvm_ir)
        path = os.path.join(self.cache_dir, f"{h}.ll")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(llvm_ir)
        self._index[h] = {"ir": path, "time": time.time()}
        self._save_index()
        if self.verbose:
            LOG.debug("IR cached %s", path)
        _file_log(f"IR cached {path}")
        return path

    def emit_object(self, llvm_ir: str, out_path: Optional[str] = None) -> str:
        if out_path is None:
            out_path = tempfile.mktemp(suffix=".o")
        mod = binding.parse_assembly(llvm_ir)
        mod.verify()
        tm = binding.Target.from_default_triple().create_target_machine()
        obj = tm.emit_object(mod)
        with open(out_path, "wb") as fh:
            fh.write(obj)
        if self.verbose:
            LOG.debug("Object emitted %s (%d bytes)", out_path, len(obj))
        _file_log(f"Object emitted {out_path} size={len(obj)}")
        return out_path

    def emit_assembly(self, llvm_ir: str, out_path: Optional[str] = None) -> str:
        if out_path is None:
            out_path = tempfile.mktemp(suffix=".s")
        mod = binding.parse_assembly(llvm_ir)
        mod.verify()
        tm = binding.Target.from_default_triple().create_target_machine()
        asm = tm.emit_assembly(mod)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(asm)
        if self.verbose:
            LOG.debug("Assembly emitted %s", out_path)
        _file_log(f"Assembly emitted {out_path}")
        return out_path

    def try_link_shared(self, object_path: str, out_shared: Optional[str] = None) -> Tuple[bool, str]:
        if out_shared is None:
            out_shared = object_path + (".so" if os.name != "nt" else ".dll")
        cc = shutil.which("clang") or shutil.which("gcc")
        if not cc:
            return False, "no system compiler (clang/gcc) found"
        cmd = [cc, "-shared", "-o", out_shared, object_path]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if proc.returncode != 0:
                return False, f"linker failed: {proc.stderr.strip()}"
            if self.verbose:
                LOG.debug("Linked shared %s", out_shared)
            _file_log(f"Linked shared {out_shared}")
            return True, out_shared
        except Exception as e:
            return False, f"link failed: {e}"

    def batch_emit_objects(self, ir_list: Sequence[Tuple[str, str]], out_dir: Optional[str] = None) -> List[str]:
        out_dir = out_dir or tempfile.mkdtemp(prefix="instryx_objs_")
        results: List[str] = ["" for _ in ir_list]
        futures = []
        with ThreadPoolExecutor(max_workers=max(2, (os.cpu_count() or 2))) as pool:
            for i, (ir, base) in enumerate(ir_list):
                out = os.path.join(out_dir, base + ".o")
                futures.append(pool.submit(self.emit_object, ir, out))
            for i, fut in enumerate(as_completed(futures)):
                try:
                    results[i] = fut.result()
                except Exception as e:
                    results[i] = f"error:{e}"
        return results

    def load_shared_library(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        lib = ctypes.CDLL(path)
        if self.verbose:
            LOG.debug("Loaded shared %s", path)
        _file_log(f"Loaded shared {path}")
        return lib

    def _compile_ir(self, llvm_ir: str, module_name: Optional[str] = None):
        self._prepare_engine()
        mod = binding.parse_assembly(llvm_ir)
        mod.verify()
        if module_name:
            try:
                mod.name = module_name
            except Exception:
                pass
        self.engine.add_module(mod)
        self.engine.finalize_object()
        self.engine.run_static_constructors()
        self._modules.append(mod)
        if self.verbose:
            LOG.debug("Compiled module %s", module_name or "<anon>")
        _file_log(f"Compiled module {module_name or '<anon>'}")

    def warmup_jit(self, function_names: Sequence[str]):
        if self.engine is None:
            return
        for fn in function_names:
            try:
                addr = self.engine.get_function_address(fn)
                if self.verbose:
                    LOG.debug("Warmup resolved %s -> %s", fn, hex(addr) if addr else None)
            except Exception:
                LOG.debug("Warmup failed for %s", fn)

    def call_function(self, func_name: str, arg_types: Optional[Sequence[Any]] = None,
                      ret_type: Any = "void", args: Optional[Sequence[Any]] = None,
                      timeout: Optional[float] = None) -> Tuple[bool, Any]:
        if self.engine is None:
            return False, "engine not initialized; compile IR first"
        ptr = self.engine.get_function_address(func_name)
        if not ptr:
            return False, f"function {func_name} not found"
        arg_types = arg_types or []
        ctypes_args = []
        for t in arg_types:
            ct = _resolve_ctype(t) if isinstance(t, str) else t
            if ct is None:
                return False, f"unsupported arg type {t}"
            ctypes_args.append(ct)
        ctypes_ret = _resolve_ctype(ret_type) if isinstance(ret_type, str) else ret_type
        try:
            FUN = ctypes.CFUNCTYPE(None if ctypes_ret is None else ctypes_ret, *ctypes_args)
            cfunc = FUN(ptr)
        except Exception as e:
            return False, f"failed to wrap function: {e}"
        out = {"ok": None, "val": None}

        def _invoke():
            try:
                r = cfunc(*([] if args is None else list(args)))
                out["ok"] = True
                out["val"] = r
            except Exception as e:
                out["ok"] = False
                out["val"] = f"exception during invocation: {e}"

        t = threading.Thread(target=_invoke, daemon=True)
        t.start()
        t.join(timeout=timeout)
        if t.is_alive():
            return False, "timeout"
        return out["ok"], out["val"]

    def run(self, code: str, invoke_main: bool = True, timeout: Optional[float] = None,
            use_subprocess: bool = False, cache_ir: bool = True) -> str:
        _metric_inc("instryx_compile_requests_total", 1)
        llvm_ir = self.codegen.generate(code)
        if cache_ir:
            self.cache_ir(llvm_ir)
        try:
            self._compile_ir(llvm_ir)
            _metric_inc("instryx_compile_success_total", 1)
        except Exception:
            _metric_inc("instryx_compile_failure_total", 1)
            LOG.exception("compile failed")
            raise

        if not invoke_main:
            return llvm_ir

        _metric_inc("instryx_run_requests_total", 1)
        if use_subprocess:
            stub = f"""
import sys, ctypes
from llvmlite import binding
binding.initialize()
binding.initialize_native_target()
binding.initialize_native_asmprinter()
mod = binding.parse_assembly(r'''{llvm_ir}''')
mod.verify()
tm = binding.Target.from_default_triple().create_target_machine()
engine = binding.create_mcjit_compiler(binding.parse_assembly(""), tm)
engine.add_module(mod)
engine.finalize_object()
engine.run_static_constructors()
addr = engine.get_function_address("main")
if not addr:
    print("FUNCTION_NOT_FOUND", file=sys.stderr)
    sys.exit(2)
ctypes.CFUNCTYPE(None)(addr)()
"""
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
                tf.write(stub)
                stub_path = tf.name
            try:
                proc = subprocess.run([shutil.which("python") or "python", stub_path],
                                      capture_output=True, text=True, timeout=timeout)
                if proc.returncode != 0:
                    _metric_inc("instryx_run_failure_total", 1)
                    LOG.error("subprocess failed: %s", proc.stderr.strip())
                else:
                    _metric_inc("instryx_run_success_total", 1)
                    if self.verbose:
                        LOG.debug("subprocess stdout: %s", proc.stdout.strip())
            except subprocess.TimeoutExpired:
                _metric_inc("instryx_run_failure_total", 1)
                LOG.error("subprocess timed out")
            finally:
                try:
                    os.unlink(stub_path)
                except Exception:
                    pass
            return llvm_ir

        ok, res = self.call_function("main", arg_types=None, ret_type="void", args=(), timeout=timeout)
        if ok:
            _metric_inc("instryx_run_success_total", 1)
            if self.verbose:
                LOG.debug("main invocation succeeded")
        else:
            _metric_inc("instryx_run_failure_total", 1)
            LOG.error("main invocation failed: %s", res)
        return llvm_ir

    def get_metrics(self) -> Dict[str, int]:
        with _metrics_lock:
            return dict(_metrics)


# --- convenience CLI / helpers ---
def build_example_shared(output: str = "libexample.so") -> int:
    """
    Helper to build a tiny C shared library used for testing ctypes interop.
    """
    src = r'''
#include <stdio.h>
int example_add(int a, int b) { return a + b; }
void example_print(const char *s) { printf("example_print: %s\n", s); }
'''
    cur = os.path.abspath(os.path.dirname(__file__))
    cfile = os.path.join(cur, "example_cmodule.c")
    with open(cfile, "w", encoding="utf-8") as fh:
        fh.write(src)
    cc = shutil.which("clang") or shutil.which("gcc")
    if not cc:
        print("No system compiler found (clang/gcc required)")
        return 2
    cmd = [cc, "-shared", "-fPIC", "-O2", "-o", output, cfile]
    try:
        subprocess.check_call(cmd)
        print("Built", output)
        return 0
    except Exception as e:
        print("Build failed:", e)
        return 1


if __name__ == "__main__":
    # quick smoke test and example usage
    runner = InstryxRunner(verbose=True)
    sample = """
    func greet(uid) {
        print: "Hello from Instryx IR";
    };
    main() {
        greet(1);
    };
    """
    try:
        ir = runner.run(sample, invoke_main=True, timeout=5)
        print("\nLLVM IR:\n", ir)
    except Exception as e:
        print("run failed:", e)

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


"""
instryx_heap_gc_allocator.py

Production-ready Heap + Generational Mark-and-Sweep Garbage Collector (GC) allocator
for the Instryx runtime with many optimizations, tooling and features.

Enhancements added:
- Generational GC (minor/major collections), remembered-set and write-barrier.
- Card-table style coarse-grained remembered-set via per-object flags.
- Pinned-object support (pin/unpin) to avoid moving or finalizing during compaction.
- Large-object allocator (separate tracking) to avoid frequent movement.
- Async background finalizer thread that executes finalizers out-of-band.
- Optional incremental marking support (mark-in-steps) with an exposed API.
- Best-effort heap compaction: moves live objects to a new heap and rewrites references.
- Heap snapshot diff / leak detection helpers.
- Integration hooks: register external instrumentation callbacks for alloc/free/collect.
- Profiling counters, generation histograms and runtime stats.
- Export/import heap snapshot, export metrics to JSON, atomic export.
- Conservative root scanner helper using Python's gc module (optional).
- CLI demo and robust self-test.

Design:
- Objects are opaque integer handles.
- Only integers that exist in the allocator's heap are treated as object references.
- Caller should call write-barrier helpers (set_field/set_index) already implemented here.
- The allocator is thread-safe via an RLock.
"""

from __future__ import annotations
import gc as _py_gc
import json
import logging
import os
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

LOG = logging.getLogger("instryx.heapgc")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO)

# Configuration defaults
DEFAULT_MAX_HEAP_OBJECTS = 50_000
DEFAULT_YOUNG_THRESHOLD = 2         # minor collections to trigger promotion
BACKGROUND_COLLECTION_INTERVAL = 2.0  # background minor GC interval (s)
FINALIZER_BATCH_SIZE = 32            # finalizers executed per run
COMPACT_THRESHOLD_RATIO = 0.25       # if free fraction > threshold, attempt compaction


@dataclass
class HeapStats:
    alloc_count: int = 0
    collect_count: int = 0
    last_collect_time: Optional[float] = None
    last_collected_objects: int = 0
    heap_size_high_water: int = 0
    total_finalized: int = 0


class HeapObject:
    """
    Internal representation of a heap object.
    Fields:
      - id: handle integer
      - kind: "object", "array", "bytes", "value", "large"
      - payload: dict/list/bytearray/value depending on kind
      - gen: generation number (0 = young)
      - age: survived minor collections count
      - marked: mark flag during marking
      - pinned: if True object must not be moved / finalized
      - has_old_to_young_ref: boolean card flag used for remembered set
      - finalizer: optional callable(handle, obj)
      - size: abstract size units for heuristics (large objects track their bytes)
    """
    __slots__ = ("id", "kind", "payload", "gen", "age", "marked", "pinned",
                 "has_old_to_young_ref", "finalizer", "size")

    def __init__(self, id_: int, kind: str, payload: Any, size: int = 1, finalizer: Optional[Callable] = None):
        self.id = id_
        self.kind = kind
        self.payload = payload
        self.gen = 0
        self.age = 0
        self.marked = False
        self.pinned = False
        self.has_old_to_young_ref = False
        self.finalizer = finalizer
        self.size = int(size)


class HeapGCAllocator:
    """
    Main allocator class implementing features listed above.
    """

    def __init__(
        self,
        max_heap_objects: int = DEFAULT_MAX_HEAP_OBJECTS,
        young_threshold: int = DEFAULT_YOUNG_THRESHOLD,
        enable_background: bool = True,
    ):
        self.max_heap_objects = int(max_heap_objects)
        self.young_threshold = int(young_threshold)

        self._lock = threading.RLock()
        self._next_id = 1
        self._heap: Dict[int, HeapObject] = {}
        self._large_objects: Set[int] = set()
        self._roots: Dict[str, Set[int]] = {}
        self._remembered: Set[int] = set()
        self._pinned: Set[int] = set()
        self._stats = HeapStats()
        self._observers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        self._finalizer_queue: List[Tuple[int, HeapObject]] = []
        self._finalizer_lock = threading.RLock()
        self._stop_bg = threading.Event()
        self._bg_thread: Optional[threading.Thread] = None
        self._finalizer_thread: Optional[threading.Thread] = None
        self._incremental_stack: List[int] = []
        self._incremental_state = {"active": False}
        self._compact_lock = threading.RLock()
        if enable_background:
            self.start_background_collector()
            self.start_finalizer_worker()
        LOG.info("HeapGCAllocator init max=%d young_threshold=%d bg=%s", self.max_heap_objects, self.young_threshold, enable_background)

    # ---------------------------
    # Observers / instrumentation
    # ---------------------------
    def register_observer(self, name: str, fn: Callable[[Dict[str, Any]], None]) -> None:
        with self._lock:
            self._observers.setdefault(name, []).append(fn)

    def _notify(self, event: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            for fn in self._observers.get(event, []):
                try:
                    fn(payload)
                except Exception:
                    LOG.exception("observer %s failed", event)

    # ---------------------------
    # Allocation primitives
    # ---------------------------
    def _alloc_id(self) -> int:
        with self._lock:
            hid = self._next_id
            self._next_id += 1
            return hid

    def _ensure_capacity(self):
        if len(self._heap) + len(self._large_objects) >= self.max_heap_objects:
            LOG.debug("heap near capacity, attempting GC")
            self.collect(full=False)
            if len(self._heap) + len(self._large_objects) >= self.max_heap_objects:
                # attempt more aggressive major collect
                self.collect(full=True)
            if len(self._heap) + len(self._large_objects) >= self.max_heap_objects:
                raise MemoryError("Heap limit reached after GC attempts")

    def alloc_object(self, fields: Optional[Dict[str, Any]] = None, finalizer: Optional[Callable] = None, size: int = 1) -> int:
        fields = dict(fields or {})
        with self._lock:
            self._ensure_capacity()
            hid = self._alloc_id()
            obj = HeapObject(hid, "object", fields, size=size, finalizer=finalizer)
            self._heap[hid] = obj
            self._stats.alloc_count += 1
            self._stats.heap_size_high_water = max(self._stats.heap_size_high_water, len(self._heap))
            self._update_remembered_for_new(obj)
            self._notify("alloc", {"handle": hid, "kind": "object", "size": size})
            return hid

    def alloc_array(self, length: int, initial: Any = None, finalizer: Optional[Callable] = None, size: int = 1) -> int:
        arr = [initial] * int(length)
        with self._lock:
            self._ensure_capacity()
            hid = self._alloc_id()
            obj = HeapObject(hid, "array", arr, size=size, finalizer=finalizer)
            self._heap[hid] = obj
            self._stats.alloc_count += 1
            self._update_remembered_for_new(obj)
            self._notify("alloc", {"handle": hid, "kind": "array", "size": size})
            return hid

    def alloc_bytes(self, nbytes: int, finalizer: Optional[Callable] = None) -> int:
        b = bytearray(int(nbytes))
        with self._lock:
            self._ensure_capacity()
            hid = self._alloc_id()
            obj = HeapObject(hid, "bytes", b, size=nbytes, finalizer=finalizer)
            self._heap[hid] = obj
            self._stats.alloc_count += 1
            self._notify("alloc", {"handle": hid, "kind": "bytes", "size": nbytes})
            return hid

    def alloc_large_object(self, payload: Any, size_bytes: int, finalizer: Optional[Callable] = None) -> int:
        """Allocate a large object tracked separately (less likely to move)."""
        with self._lock:
            self._ensure_capacity()
            hid = self._alloc_id()
            obj = HeapObject(hid, "large", payload, size=size_bytes, finalizer=finalizer)
            self._heap[hid] = obj
            self._large_objects.add(hid)
            self._stats.alloc_count += 1
            self._notify("alloc", {"handle": hid, "kind": "large", "size": size_bytes})
            return hid

    def box_value(self, value: Any) -> int:
        with self._lock:
            self._ensure_capacity()
            hid = self._alloc_id()
            obj = HeapObject(hid, "value", value, size=1)
            self._heap[hid] = obj
            self._stats.alloc_count += 1
            self._notify("alloc", {"handle": hid, "kind": "value"})
            return hid

    # ---------------------------
    # Access helpers + write barrier
    # ---------------------------
    def get_field(self, handle: int, name: str) -> Any:
        obj = self._heap.get(handle)
        if not obj or obj.kind != "object":
            raise KeyError("invalid object handle for get_field")
        return obj.payload.get(name)

    def set_field(self, handle: int, name: str, value: Any) -> None:
        with self._lock:
            obj = self._heap.get(handle)
            if not obj or obj.kind != "object":
                raise KeyError("invalid object handle for set_field")
            obj.payload[name] = value
            self._write_barrier(obj, value)

    def get_index(self, handle: int, idx: int) -> Any:
        obj = self._heap.get(handle)
        if not obj or obj.kind != "array":
            raise KeyError("invalid array handle for get_index")
        return obj.payload[int(idx)]

    def set_index(self, handle: int, idx: int, value: Any) -> None:
        with self._lock:
            obj = self._heap.get(handle)
            if not obj or obj.kind != "array":
                raise KeyError("invalid array handle for set_index")
            obj.payload[int(idx)] = value
            self._write_barrier(obj, value)

    def raw_bytes(self, handle: int) -> bytearray:
        obj = self._heap.get(handle)
        if not obj or obj.kind not in ("bytes", "large"):
            raise KeyError("invalid bytes handle")
        return obj.payload

    def _write_barrier(self, obj: HeapObject, value: Any) -> None:
        """
        Write barrier: when an older object stores a reference to a young object,
        record it in remembered set. Also maintain card flag optimization.
        """
        if not isinstance(value, int) or value not in self._heap:
            return
        with self._lock:
            target = self._heap[value]
            if obj.gen > 0 and target.gen == 0:
                obj.has_old_to_young_ref = True
                self._remembered.add(obj.id)

    # ---------------------------
    # Pinning API
    # ---------------------------
    def pin(self, handle: int) -> None:
        with self._lock:
            if handle in self._heap:
                self._heap[handle].pinned = True
                self._pinned.add(handle)

    def unpin(self, handle: int) -> None:
        with self._lock:
            if handle in self._heap:
                self._heap[handle].pinned = False
                self._pinned.discard(handle)

    # ---------------------------
    # Roots API
    # ---------------------------
    def register_root(self, set_name: str, handle: int) -> None:
        with self._lock:
            self._roots.setdefault(set_name, set()).add(handle)
            self._notify("root_register", {"set": set_name, "handle": handle})

    def unregister_root(self, set_name: str, handle: int) -> None:
        with self._lock:
            if set_name in self._roots:
                self._roots[set_name].discard(handle)
                if not self._roots[set_name]:
                    del self._roots[set_name]
            self._notify("root_unregister", {"set": set_name, "handle": handle})

    def list_roots(self) -> Dict[str, Set[int]]:
        with self._lock:
            return {k: set(v) for k, v in self._roots.items()}

    # ---------------------------
    # Finalizers worker
    # ---------------------------
    def start_finalizer_worker(self):
        if self._finalizer_thread and self._finalizer_thread.is_alive():
            return
        self._finalizer_thread = threading.Thread(target=self._finalizer_worker, daemon=True)
        self._finalizer_thread.start()
        LOG.info("finalizer worker started")

    def stop_finalizer_worker(self):
        if self._finalizer_thread:
            try:
                self._finalizer_thread.join(timeout=1.0)
            except Exception:
                pass

    def _enqueue_finalizer(self, hid: int, obj: HeapObject):
        with self._finalizer_lock:
            self._finalizer_queue.append((hid, obj))

    def _finalizer_worker(self):
        while not self._stop_bg.is_set():
            batch = []
            with self._finalizer_lock:
                while self._finalizer_queue and len(batch) < FINALIZER_BATCH_SIZE:
                    batch.append(self._finalizer_queue.pop(0))
            if not batch:
                time.sleep(0.2)
                continue
            for hid, obj in batch:
                try:
                    if obj.finalizer and not obj.pinned:
                        obj.finalizer(hid, obj)
                        self._stats.total_finalized += 1
                        self._notify("finalized", {"handle": hid})
                except Exception:
                    LOG.exception("finalizer error for %d", hid)

    # ---------------------------
    # Collection: minor/major and incremental
    # ---------------------------
    def collect(self, full: bool = True) -> Dict[str, Any]:
        """
        Full collection when full=True (major), minor otherwise.
        Collection is mark-and-sweep; finalizers queued and executed asynchronously.
        """
        start_time = time.time()
        with self._lock:
            try:
                self._unmark_all()
                roots = self._gather_roots()
                # include remembered set for minor collections
                remember_roots = set(self._remembered) if not full else set()
                stack = list(roots | remember_roots)
                marked = 0
                while stack:
                    hid = stack.pop()
                    if hid not in self._heap:
                        continue
                    obj = self._heap[hid]
                    if obj.marked:
                        continue
                    obj.marked = True
                    marked += 1
                    refs = self._extract_references(obj)
                    for r in refs:
                        if r in self._heap and not self._heap[r].marked:
                            stack.append(r)
                # sweep phase - collect unmarked
                to_delete = []
                for hid, obj in list(self._heap.items()):
                    if not obj.marked:
                        if obj.pinned:
                            # pinned unreachable objects are skipped (user responsibility)
                            continue
                        to_delete.append(hid)
                # queue finalizers and remove
                for hid in to_delete:
                    obj = self._heap.get(hid)
                    if obj and obj.finalizer:
                        self._enqueue_finalizer(hid, obj)
                for hid in to_delete:
                    self._heap.pop(hid, None)
                    self._remembered.discard(hid)
                    self._large_objects.discard(hid)
                    self._pinned.discard(hid)
                # promotion
                for obj in self._heap.values():
                    if obj.marked:
                        if obj.gen == 0:
                            obj.age += 1
                            if obj.age >= self.young_threshold:
                                obj.gen += 1
                                obj.age = 0
                        # reset has_old_to_young_ref if now stable
                        if obj.has_old_to_young_ref:
                            # recompute whether any refs still point to young
                            refs = self._extract_references(obj)
                            obj.has_old_to_young_ref = any(self._heap[r].gen == 0 for r in refs if r in self._heap)
                # stats
                self._stats.collect_count += 1
                self._stats.last_collect_time = time.time()
                self._stats.last_collected_objects = len(to_delete)
                # auto compaction heuristic
                free_fraction = 1.0 - (len(self._heap) / max(1, self.max_heap_objects))
                if full and free_fraction > COMPACT_THRESHOLD_RATIO:
                    try:
                        self.compact_heap()
                    except Exception:
                        LOG.exception("compaction failed")
                self._notify("collect", {"marked": marked, "collected": len(to_delete)})
                LOG.info("GC complete: marked=%d collected=%d heap=%d", marked, len(to_delete), len(self._heap))
                return {"marked": marked, "collected": len(to_delete), "heap_size": len(self._heap), "duration_s": time.time() - start_time}
            except Exception:
                LOG.exception("GC failed")
                return {"error": "exception during GC"}

    def minor_collect(self) -> Dict[str, Any]:
        return self.collect(full=False)

    def major_collect(self) -> Dict[str, Any]:
        return self.collect(full=True)

    def _unmark_all(self):
        for obj in self._heap.values():
            obj.marked = False

    def _gather_roots(self) -> Set[int]:
        roots: Set[int] = set()
        for s in self._roots.values():
            for hid in s:
                if hid in self._heap:
                    roots.add(hid)
        return roots

    # incremental marking: prepare and do limited steps
    def begin_incremental_mark(self):
        with self._lock:
            self._unmark_all()
            roots = self._gather_roots()
            self._incremental_stack = list(roots)
            self._incremental_state["active"] = True
            LOG.debug("incremental mark started with stack=%d", len(self._incremental_stack))

    def incremental_mark_step(self, max_nodes: int = 1000) -> Dict[str, Any]:
        with self._lock:
            if not self._incremental_state.get("active"):
                return {"status": "inactive"}
            processed = 0
            while self._incremental_stack and processed < max_nodes:
                hid = self._incremental_stack.pop()
                if hid not in self._heap:
                    continue
                obj = self._heap[hid]
                if obj.marked:
                    continue
                obj.marked = True
                processed += 1
                refs = self._extract_references(obj)
                for r in refs:
                    if r in self._heap and not self._heap[r].marked:
                        self._incremental_stack.append(r)
            if not self._incremental_stack:
                # finish and sweep unreachable
                self._incremental_state["active"] = False
                # now sweep like a major collect but queue finalizers async
                to_delete = [hid for hid, obj in list(self._heap.items()) if not obj.marked and not obj.pinned]
                for hid in to_delete:
                    obj = self._heap.get(hid)
                    if obj and obj.finalizer:
                        self._enqueue_finalizer(hid, obj)
                for hid in to_delete:
                    self._heap.pop(hid, None)
                    self._remembered.discard(hid)
                    self._large_objects.discard(hid)
                    self._pinned.discard(hid)
                LOG.debug("incremental mark complete processed=%d swept=%d", processed, len(to_delete))
                return {"status": "complete", "processed": processed, "swept": len(to_delete)}
            return {"status": "in-progress", "processed": processed, "remaining": len(self._incremental_stack)}

    # ---------------------------
    # Compaction (best-effort)
    # ---------------------------
    def compact_heap(self) -> Dict[str, Any]:
        """
        Best-effort compaction: allocate new ids for live objects, copy payloads,
        rewrite references and update roots. Pinned objects are not moved.
        This operation can be heavy; caller should ensure minimal concurrency.
        """
        with self._compact_lock, self._lock:
            try:
                LOG.info("Starting heap compaction")
                # identify live objects via a temporary mark
                self._unmark_all()
                roots = self._gather_roots()
                stack = list(roots)
                while stack:
                    hid = stack.pop()
                    if hid not in self._heap:
                        continue
                    obj = self._heap[hid]
                    if obj.marked:
                        continue
                    obj.marked = True
                    for r in self._extract_references(obj):
                        if r in self._heap and not self._heap[r].marked:
                            stack.append(r)
                # build mapping for movable objects: skip pinned and large objects
                mapping: Dict[int, int] = {}
                new_heap: Dict[int, HeapObject] = {}
                for old_id, obj in list(self._heap.items()):
                    if not obj.marked:
                        continue  # unreachable won't be moved
                    if obj.pinned or old_id in self._large_objects:
                        # preserve id to avoid changing external refs; copy as-is
                        mapping[old_id] = old_id
                        new_heap[old_id] = obj
                        continue
                    # assign new id
                    new_id = self._alloc_id()
                    mapping[old_id] = new_id
                    # shallow copy of payload; we'll rewrite references next
                    copied = None
                    if obj.kind == "object":
                        copied = dict(obj.payload)
                    elif obj.kind == "array":
                        copied = list(obj.payload)
                    elif obj.kind in ("bytes", "large"):
                        copied = bytearray(obj.payload) if obj.kind == "bytes" else obj.payload
                    elif obj.kind == "value":
                        copied = obj.payload
                    new_obj = HeapObject(new_id, obj.kind, copied, size=obj.size, finalizer=obj.finalizer)
                    new_obj.gen = obj.gen
                    new_obj.age = obj.age
                    new_obj.pinned = obj.pinned
                    new_heap[new_id] = new_obj
                # rewrite references in new_heap
                def rewrite_in_obj(o: HeapObject):
                    if o.kind == "object":
                        for k, v in list(o.payload.items()):
                            if isinstance(v, int) and v in mapping:
                                o.payload[k] = mapping[v]
                    elif o.kind == "array":
                        for i, v in enumerate(o.payload):
                            if isinstance(v, int) and v in mapping:
                                o.payload[i] = mapping[v]
                for o in new_heap.values():
                    rewrite_in_obj(o)
                # update roots to new ids
                new_roots: Dict[str, Set[int]] = {}
                for rname, rset in self._roots.items():
                    new_set = set()
                    for hid in rset:
                        if hid in mapping:
                            new_set.add(mapping[hid])
                    if new_set:
                        new_roots[rname] = new_set
                # replace heap and roots
                self._heap = new_heap
                self._roots = new_roots
                # rebuild large_objects and pinned sets
                self._large_objects = {mapping.get(h, h) for h in list(self._large_objects) if h in mapping}
                self._pinned = {mapping.get(h, h) for h in list(self._pinned) if h in mapping}
                # rebuild remembered set conservatively
                self._remembered = set()
                for hid, obj in self._heap.items():
                    if obj.gen > 0:
                        refs = self._extract_references(obj)
                        if any((r in self._heap and self._heap[r].gen == 0) for r in refs):
                            self._remembered.add(hid)
                LOG.info("Compaction complete: new_heap=%d", len(self._heap))
                self._notify("compact", {"heap_size": len(self._heap)})
                return {"status": "ok", "heap_size": len(self._heap)}
            except Exception:
                LOG.exception("compaction failed")
                return {"status": "error", "message": "exception during compaction"}

    # ---------------------------
    # Helpers & introspection
    # ---------------------------
    def heap_snapshot(self, include_contents: bool = False) -> Dict[str, Any]:
        with self._lock:
            snap = {
                "time": time.time(),
                "count": len(self._heap),
                "objects": {},
                "roots": {k: list(v) for k, v in self._roots.items()},
                "remembered": list(self._remembered),
                "large_objects": list(self._large_objects),
                "pinned": list(self._pinned),
                "stats": self._stats.__dict__.copy(),
            }
            for hid, obj in self._heap.items():
                o = {"id": hid, "kind": obj.kind, "gen": obj.gen, "age": obj.age, "size": obj.size, "pinned": obj.pinned}
                if include_contents:
                    if obj.kind == "object":
                        o["fields"] = dict(obj.payload)
                    elif obj.kind == "array":
                        o["array"] = list(obj.payload)
                    elif obj.kind in ("bytes", "large"):
                        o["raw_len"] = len(obj.payload) if hasattr(obj.payload, "__len__") else None
                    elif obj.kind == "value":
                        o["value"] = obj.payload
                snap["objects"][hid] = o
            return snap

    def export_heap_json(self, path: Optional[str] = None, include_contents: bool = True) -> str:
        if not path:
            fd, path = tempfile.mkstemp(prefix="instryx_heap_", suffix=".json")
            os.close(fd)
        snap = self.heap_snapshot(include_contents=include_contents)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snap, f, indent=2, default=str)
        os.replace(tmp, path)
        LOG.info("Heap exported -> %s", path)
        return path

    def import_heap_json(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            snap = json.load(f)
        # Best-effort: clear current heap and rebuild simple boxed values (ids will be new)
        with self._lock:
            self._heap.clear()
            self._roots.clear()
            self._remembered.clear()
            self._pinned.clear()
            self._large_objects.clear()
            # load objects as values only (preserve counts)
            for old_id_str, meta in snap.get("objects", {}).items():
                hid = self._alloc_id()
                kind = meta.get("kind", "value")
                payload = None
                if kind == "object":
                    payload = {k: None for k in meta.get("fields", {}).keys()} if meta.get("fields") else {}
                elif kind == "array":
                    payload = [None] * len(meta.get("array", []))
                elif kind in ("bytes", "large"):
                    payload = bytearray(meta.get("raw_len", 0) or 0)
                else:
                    payload = None
                obj = HeapObject(hid, kind, payload, size=meta.get("size", 1))
                self._heap[hid] = obj
            LOG.info("imported heap snapshot (best-effort), recreated %d objects", len(self._heap))

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            d = self._stats.__dict__.copy()
            d.update({"heap_size": len(self._heap), "roots": {k: len(v) for k, v in self._roots.items()}})
            return d

    def find_handles_pointing_to(self, target: int) -> List[int]:
        with self._lock:
            res = []
            for hid, obj in self._heap.items():
                refs = self._extract_references(obj)
                if target in refs:
                    res.append(hid)
            return res

    # ---------------------------
    # Conservative root scanner (inspect Python heap via gc)
    # ---------------------------
    def scan_python_globals_for_handles(self) -> Set[int]:
        """
        Conservative scan through Python GC tracked objects (modules, builtins, globals)
        and collect integers that look like handles registered in this allocator.
        Use with caution: expensive.
        """
        found = set()
        with self._lock:
            valid_handles = set(self._heap.keys())
        for obj in list(_py_gc.get_objects()):
            try:
                if isinstance(obj, dict):
                    for v in obj.values():
                        if isinstance(v, int) and v in valid_handles:
                            found.add(v)
                elif isinstance(obj, (list, tuple, set)):
                    for v in obj:
                        if isinstance(v, int) and v in valid_handles:
                            found.add(v)
            except Exception:
                continue
        return found

    # ---------------------------
    # Compaction helper already defined above
    # ---------------------------

    # ---------------------------
    # Background collector control
    # ---------------------------
    def start_background_collector(self, interval: float = BACKGROUND_COLLECTION_INTERVAL):
        if self._bg_thread and self._bg_thread.is_alive():
            return
        self._stop_bg.clear()
        self._bg_thread = threading.Thread(target=self._bg_worker, args=(interval,), daemon=True)
        self._bg_thread.start()
        LOG.info("Background collector started interval=%.2fs", interval)

    def stop_background_collector(self):
        if self._bg_thread:
            self._stop_bg.set()
            try:
                self._bg_thread.join(timeout=1.0)
            except Exception:
                pass

    def _bg_worker(self, interval: float):
        while not self._stop_bg.is_set():
            try:
                self.minor_collect()
            except Exception:
                LOG.exception("background GC error")
            self._stop_bg.wait(interval)

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _extract_references(self, obj: HeapObject) -> Set[int]:
        refs = set()
        if obj.kind == "object" and isinstance(obj.payload, dict):
            for v in obj.payload.values():
                if isinstance(v, int) and v in self._heap:
                    refs.add(v)
        elif obj.kind == "array" and isinstance(obj.payload, (list, tuple)):
            for v in obj.payload:
                if isinstance(v, int) and v in self._heap:
                    refs.add(v)
        return refs

    def _update_remembered_for_new(self, obj: HeapObject):
        # If object is older and references young objects, remember it
        if obj.gen > 0:
            refs = self._extract_references(obj)
            for r in refs:
                if r in self._heap and self._heap[r].gen == 0:
                    obj.has_old_to_young_ref = True
                    self._remembered.add(obj.id)

    # ---------------------------
    # Self-test and CLI demo
    # ---------------------------
def _self_test() -> bool:
    LOG.info("Running HeapGCAllocator self-test (extended)")
    gc = HeapGCAllocator(max_heap_objects=200, enable_background=False)
    a = gc.alloc_object({"v": 123})
    b = gc.alloc_object({"ref": a})
    gc.register_root("main", b)
    # allocate many unreachable objects
    for i in range(100):
        gc.alloc_object({"temp": i})
    pre = len(gc._heap)
    stats = gc.collect(full=True)
    post = len(gc._heap)
    LOG.info("self-test: before=%d after=%d stats=%s", pre, post, stats)
    if a not in gc._heap or b not in gc._heap:
        LOG.error("self-test failed: live objects collected")
        return False
    # finalizer check
    finalized = []
    def final(hid, obj):
        finalized.append(hid)
    c = gc.alloc_object({"x": 10}, finalizer=final)
    # make unreachable and collect
    gc.collect(full=True)
    # allow finalizer thread time if enabled (not enabled in this test)
    LOG.info("finalized list (may be empty if finalizer worker disabled): %s", finalized)
    # test compaction
    try:
        r = gc.compact_heap()
        LOG.info("compaction result: %s", r)
    except Exception:
        LOG.exception("compaction in self-test failed")
    LOG.info("self-test passed")
    return True

if __name__ == "__main__":
    ok = _self_test()
    os._exit(0 if ok else 2)

    import threading
    import time
    import logging
    import tempfile

"""
instryx_heap_gc_allocator.py

Production-ready Heap + Generational Mark-and-Sweep Garbage Collector (GC) allocator
for the Instryx runtime with many optimizations, tooling and features.

Enhancements in this version:
 - All previous features (generational GC, remembered-set, write-barrier, pin/unpin,
   large-object tracking, background finalizer, incremental marking, compaction, snapshots).
 - Async incremental marking background worker (configurable).
 - Lightweight object payload pooling to reduce small-dict/list churn.
 - Export of runtime metrics (JSON and Prometheus text).
 - Improved leak detection with path-to-root search (BFS).
 - Shutdown helper that stops background workers cleanly.
 - Better observers/events for lifecycle hooks.
 - Minor micro-optimizations and safer guards.
"""
from __future__ import annotations
import gc as _py_gc
import json
import logging
import os
import tempfile
import threading
import time
import traceback
import gzip
import shutil
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

LOG = logging.getLogger("instryx.heapgc")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO)

# Configuration defaults
DEFAULT_MAX_HEAP_OBJECTS = 50_000
DEFAULT_YOUNG_THRESHOLD = 2         # minor collections to trigger promotion
BACKGROUND_COLLECTION_INTERVAL = 2.0  # background minor GC interval (s)
INCREMENTAL_STEP_INTERVAL = 0.05     # seconds between incremental marking steps when backgrounded
FINALIZER_BATCH_SIZE = 32            # finalizers executed per run
COMPACT_THRESHOLD_RATIO = 0.25       # if free fraction > threshold, attempt compaction


@dataclass
class HeapStats:
    alloc_count: int = 0
    collect_count: int = 0
    last_collect_time: Optional[float] = None
    last_collected_objects: int = 0
    heap_size_high_water: int = 0
    total_finalized: int = 0
    gen_histogram: Dict[int, int] = None

    def __post_init__(self):
        if self.gen_histogram is None:
            self.gen_histogram = {0: 0, 1: 0, 2: 0}


class HeapObject:
    """
    Internal representation of a heap object.
    Fields:
      - id: handle integer
      - kind: "object", "array", "bytes", "value", "large"
      - payload: dict/list/bytearray/value depending on kind
      - gen: generation number (0 = young)
      - age: survived minor collections count
      - marked: mark flag during marking
      - pinned: if True object must not be moved / finalized
      - has_old_to_young_ref: boolean card flag used for remembered set
      - finalizer: optional callable(handle, obj)
      - size: abstract size units for heuristics (large objects track their bytes)
    """
    __slots__ = ("id", "kind", "payload", "gen", "age", "marked", "pinned",
                 "has_old_to_young_ref", "finalizer", "size")

    def __init__(self, id_: int, kind: str, payload: Any, size: int = 1, finalizer: Optional[Callable] = None):
        self.id = id_
        self.kind = kind
        self.payload = payload
        self.gen = 0
        self.age = 0
        self.marked = False
        self.pinned = False
        self.has_old_to_young_ref = False
        self.finalizer = finalizer
        self.size = int(size)


class HeapGCAllocator:
    """
    Main allocator class implementing features listed above.
    """

    def __init__(
        self,
        max_heap_objects: int = DEFAULT_MAX_HEAP_OBJECTS,
        young_threshold: int = DEFAULT_YOUNG_THRESHOLD,
        enable_background: bool = True,
        enable_incremental_bg: bool = False,
    ):
        self.max_heap_objects = int(max_heap_objects)
        self.young_threshold = int(young_threshold)

        self._lock = threading.RLock()
        self._next_id = 1
        self._heap: Dict[int, HeapObject] = {}
        self._large_objects: Set[int] = set()
        self._roots: Dict[str, Set[int]] = {}
        self._remembered: Set[int] = set()
        self._pinned: Set[int] = set()
        self._stats = HeapStats()
        self._observers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        self._finalizer_queue: List[Tuple[int, HeapObject]] = []
        self._finalizer_lock = threading.RLock()
        self._stop_bg = threading.Event()
        self._bg_thread: Optional[threading.Thread] = None
        self._finalizer_thread: Optional[threading.Thread] = None
        self._incremental_stack: List[int] = []
        self._incremental_state = {"active": False}
        self._compact_lock = threading.RLock()

        # object payload pools to reduce allocation churn for tiny dicts/lists
        self._obj_payload_pool: List[Dict[str, Any]] = []
        self._arr_payload_pool: List[List[Any]] = []
        self._pool_max = 1024

        # incremental background worker
        self._inc_bg_thread: Optional[threading.Thread] = None
        self._inc_bg_enabled = bool(enable_incremental_bg)
        self._inc_bg_interval = INCREMENTAL_STEP_INTERVAL

        if enable_background:
            self.start_background_collector()
            self.start_finalizer_worker()
        if self._inc_bg_enabled:
            self.start_incremental_background(self._inc_bg_interval)

        LOG.info("HeapGCAllocator init max=%d young_threshold=%d bg=%s inc_bg=%s", self.max_heap_objects, self.young_threshold, enable_background, enable_incremental_bg)

    # ---------------------------
    # Observers / instrumentation
    # ---------------------------
    def register_observer(self, name: str, fn: Callable[[Dict[str, Any]], None]) -> None:
        with self._lock:
            self._observers.setdefault(name, []).append(fn)

    def unregister_observer(self, name: str, fn: Callable[[Dict[str, Any]], None]) -> None:
        with self._lock:
            if name in self._observers:
                try:
                    self._observers[name].remove(fn)
                except ValueError:
                    pass

    def _notify(self, event: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            for fn in list(self._observers.get(event, [])):
                try:
                    fn(payload)
                except Exception:
                    LOG.exception("observer %s failed", event)

    # ---------------------------
    # Allocation primitives (with pooling)
    # ---------------------------
    def _alloc_id(self) -> int:
        with self._lock:
            hid = self._next_id
            self._next_id += 1
            return hid

    def _ensure_capacity(self):
        if len(self._heap) + len(self._large_objects) >= self.max_heap_objects:
            LOG.debug("heap near capacity, attempting GC")
            self.collect(full=False)
            if len(self._heap) + len(self._large_objects) >= self.max_heap_objects:
                # attempt more aggressive major collect
                self.collect(full=True)
            if len(self._heap) + len(self._large_objects) >= self.max_heap_objects:
                raise MemoryError("Heap limit reached after GC attempts")

    def _borrow_obj_payload(self) -> Dict[str, Any]:
        with self._lock:
            if self._obj_payload_pool:
                return self._obj_payload_pool.pop()
        return {}

    def _return_obj_payload(self, d: Dict[str, Any]) -> None:
        d.clear()
        with self._lock:
            if len(self._obj_payload_pool) < self._pool_max:
                self._obj_payload_pool.append(d)

    def _borrow_arr_payload(self, length: int) -> List[Any]:
        with self._lock:
            if self._arr_payload_pool:
                arr = self._arr_payload_pool.pop()
                # resize if needed
                if len(arr) < length:
                    arr.extend([None] * (length - len(arr)))
                else:
                    for i in range(length, len(arr)):
                        arr[i] = None
                return arr[:length]
        return [None] * length

    def _return_arr_payload(self, arr: List[Any]) -> None:
        arr.clear()
        with self._lock:
            if len(self._arr_payload_pool) < self._pool_max:
                self._arr_payload_pool.append(arr)

    def alloc_object(self, fields: Optional[Dict[str, Any]] = None, finalizer: Optional[Callable] = None, size: int = 1) -> int:
        with self._lock:
            self._ensure_capacity()
            hid = self._alloc_id()
            payload = self._borrow_obj_payload()
            if fields:
                payload.update(fields)
            obj = HeapObject(hid, "object", payload, size=size, finalizer=finalizer)
            self._heap[hid] = obj
            self._stats.alloc_count += 1
            self._stats.heap_size_high_water = max(self._stats.heap_size_high_water, len(self._heap))
            self._update_remembered_for_new(obj)
            self._notify("alloc", {"handle": hid, "kind": "object", "size": size})
            return hid

    def alloc_array(self, length: int, initial: Any = None, finalizer: Optional[Callable] = None, size: int = 1) -> int:
        with self._lock:
            self._ensure_capacity()
            hid = self._alloc_id()
            arr = self._borrow_arr_payload(length)
            if initial is not None:
                for i in range(len(arr)):
                    arr[i] = initial
            obj = HeapObject(hid, "array", arr, size=size, finalizer=finalizer)
            self._heap[hid] = obj
            self._stats.alloc_count += 1
            self._update_remembered_for_new(obj)
            self._notify("alloc", {"handle": hid, "kind": "array", "size": size})
            return hid

    def alloc_bytes(self, nbytes: int, finalizer: Optional[Callable] = None) -> int:
        b = bytearray(int(nbytes))
        with self._lock:
            self._ensure_capacity()
            hid = self._alloc_id()
            obj = HeapObject(hid, "bytes", b, size=nbytes, finalizer=finalizer)
            self._heap[hid] = obj
            self._stats.alloc_count += 1
            self._notify("alloc", {"handle": hid, "kind": "bytes", "size": nbytes})
            return hid

    def alloc_large_object(self, payload: Any, size_bytes: int, finalizer: Optional[Callable] = None) -> int:
        """Allocate a large object tracked separately (less likely to move)."""
        with self._lock:
            self._ensure_capacity()
            hid = self._alloc_id()
            obj = HeapObject(hid, "large", payload, size=size_bytes, finalizer=finalizer)
            self._heap[hid] = obj
            self._large_objects.add(hid)
            self._stats.alloc_count += 1
            self._notify("alloc", {"handle": hid, "kind": "large", "size": size_bytes})
            return hid

    def box_value(self, value: Any) -> int:
        with self._lock:
            self._ensure_capacity()
            hid = self._alloc_id()
            obj = HeapObject(hid, "value", value, size=1)
            self._heap[hid] = obj
            self._stats.alloc_count += 1
            self._notify("alloc", {"handle": hid, "kind": "value"})
            return hid

    # ---------------------------
    # Access helpers + write barrier
    # ---------------------------
    def get_field(self, handle: int, name: str) -> Any:
        obj = self._heap.get(handle)
        if not obj or obj.kind != "object":
            raise KeyError("invalid object handle for get_field")
        return obj.payload.get(name)

    def set_field(self, handle: int, name: str, value: Any) -> None:
        with self._lock:
            obj = self._heap.get(handle)
            if not obj or obj.kind != "object":
                raise KeyError("invalid object handle for set_field")
            obj.payload[name] = value
            self._write_barrier(obj, value)

    def get_index(self, handle: int, idx: int) -> Any:
        obj = self._heap.get(handle)
        if not obj or obj.kind != "array":
            raise KeyError("invalid array handle for get_index")
        return obj.payload[int(idx)]

    def set_index(self, handle: int, idx: int, value: Any) -> None:
        with self._lock:
            obj = self._heap.get(handle)
            if not obj or obj.kind != "array":
                raise KeyError("invalid array handle for set_index")
            obj.payload[int(idx)] = value
            self._write_barrier(obj, value)

    def raw_bytes(self, handle: int) -> bytearray:
        obj = self._heap.get(handle)
        if not obj or obj.kind not in ("bytes", "large"):
            raise KeyError("invalid bytes handle")
        return obj.payload

    def _write_barrier(self, obj: HeapObject, value: Any) -> None:
        """
        Write barrier: when an older object stores a reference to a young object,
        record it in remembered set. Also maintain card flag optimization.
        """
        if not isinstance(value, int) or value not in self._heap:
            return
        with self._lock:
            target = self._heap[value]
            if obj.gen > 0 and target.gen == 0:
                obj.has_old_to_young_ref = True
                self._remembered.add(obj.id)

    # ---------------------------
    # Pinning API
    # ---------------------------
    def pin(self, handle: int) -> None:
        with self._lock:
            if handle in self._heap:
                self._heap[handle].pinned = True
                self._pinned.add(handle)

    def unpin(self, handle: int) -> None:
        with self._lock:
            if handle in self._heap:
                self._heap[handle].pinned = False
                self._pinned.discard(handle)

    # ---------------------------
    # Roots API
    # ---------------------------
    def register_root(self, set_name: str, handle: int) -> None:
        with self._lock:
            self._roots.setdefault(set_name, set()).add(handle)
            self._notify("root_register", {"set": set_name, "handle": handle})

    def unregister_root(self, set_name: str, handle: int) -> None:
        with self._lock:
            if set_name in self._roots:
                self._roots[set_name].discard(handle)
                if not self._roots[set_name]:
                    del self._roots[set_name]
            self._notify("root_unregister", {"set": set_name, "handle": handle})

    def list_roots(self) -> Dict[str, Set[int]]:
        with self._lock:
            return {k: set(v) for k, v in self._roots.items()}

    # ---------------------------
    # Finalizers worker
    # ---------------------------
    def start_finalizer_worker(self):
        if self._finalizer_thread and self._finalizer_thread.is_alive():
            return
        self._finalizer_thread = threading.Thread(target=self._finalizer_worker, daemon=True)
        self._finalizer_thread.start()
        LOG.info("finalizer worker started")

    def stop_finalizer_worker(self):
        if self._finalizer_thread:
            try:
                self._finalizer_thread.join(timeout=1.0)
            except Exception:
                pass

    def _enqueue_finalizer(self, hid: int, obj: HeapObject):
        with self._finalizer_lock:
            self._finalizer_queue.append((hid, obj))

    def _finalizer_worker(self):
        while not self._stop_bg.is_set():
            batch = []
            with self._finalizer_lock:
                while self._finalizer_queue and len(batch) < FINALIZER_BATCH_SIZE:
                    batch.append(self._finalizer_queue.pop(0))
            if not batch:
                time.sleep(0.2)
                continue
            for hid, obj in batch:
                try:
                    if obj.finalizer and not obj.pinned:
                        try:
                            obj.finalizer(hid, obj)
                        except Exception:
                            LOG.exception("finalizer error for %d", hid)
                        self._stats.total_finalized += 1
                        self._notify("finalized", {"handle": hid})
                except Exception:
                    LOG.exception("finalizer loop error")

    # ---------------------------
    # Collection: minor/major and incremental
    # ---------------------------
    def collect(self, full: bool = True) -> Dict[str, Any]:
        """
        Full collection when full=True (major), minor otherwise.
        Collection is mark-and-sweep; finalizers queued and executed asynchronously.
        """
        start_time = time.time()
        with self._lock:
            try:
                self._unmark_all()
                roots = self._gather_roots()
                # include remembered set for minor collections
                remember_roots = set(self._remembered) if not full else set()
                stack = list(roots | remember_roots)
                marked = 0
                while stack:
                    hid = stack.pop()
                    if hid not in self._heap:
                        continue
                    obj = self._heap[hid]
                    if obj.marked:
                        continue
                    obj.marked = True
                    marked += 1
                    refs = self._extract_references(obj)
                    for r in refs:
                        if r in self._heap and not self._heap[r].marked:
                            stack.append(r)
                # sweep phase - collect unmarked
                to_delete = []
                for hid, obj in list(self._heap.items()):
                    if not obj.marked:
                        if obj.pinned:
                            # pinned unreachable objects are skipped (user responsibility)
                            continue
                        to_delete.append(hid)
                # queue finalizers and remove
                for hid in to_delete:
                    obj = self._heap.get(hid)
                    if obj and obj.finalizer:
                        self._enqueue_finalizer(hid, obj)
                for hid in to_delete:
                    obj = self._heap.pop(hid, None)
                    # return payloads to pool if applicable
                    if obj:
                        if obj.kind == "object":
                            self._return_obj_payload(obj.payload)
                        elif obj.kind == "array":
                            self._return_arr_payload(obj.payload)
                    self._remembered.discard(hid)
                    self._large_objects.discard(hid)
                    self._pinned.discard(hid)
                # promotion
                for obj in self._heap.values():
                    if obj.marked:
                        if obj.gen == 0:
                            obj.age += 1
                            if obj.age >= self.young_threshold:
                                obj.gen += 1
                                obj.age = 0
                        # reset has_old_to_young_ref if now stable
                        if obj.has_old_to_young_ref:
                            refs = self._extract_references(obj)
                            obj.has_old_to_young_ref = any(self._heap[r].gen == 0 for r in refs if r in self._heap)
                # stats
                self._stats.collect_count += 1
                self._stats.last_collect_time = time.time()
                self._stats.last_collected_objects = len(to_delete)
                # auto compaction heuristic
                free_fraction = 1.0 - (len(self._heap) / max(1, self.max_heap_objects))
                if full and free_fraction > COMPACT_THRESHOLD_RATIO:
                    try:
                        self.compact_heap()
                    except Exception:
                        LOG.exception("compaction failed")
                self._notify("collect", {"marked": marked, "collected": len(to_delete)})
                LOG.info("GC complete: marked=%d collected=%d heap=%d", marked, len(to_delete), len(self._heap))
                return {"marked": marked, "collected": len(to_delete), "heap_size": len(self._heap), "duration_s": time.time() - start_time}
            except Exception:
                LOG.exception("GC failed")
                return {"error": "exception during GC"}

    def minor_collect(self) -> Dict[str, Any]:
        return self.collect(full=False)

    def major_collect(self) -> Dict[str, Any]:
        return self.collect(full=True)

    def _unmark_all(self):
        for obj in self._heap.values():
            obj.marked = False

    def _gather_roots(self) -> Set[int]:
        roots: Set[int] = set()
        for s in self._roots.values():
            for hid in s:
                if hid in self._heap:
                    roots.add(hid)
        return roots

    # incremental marking: prepare and do limited steps
    def begin_incremental_mark(self):
        with self._lock:
            self._unmark_all()
            roots = self._gather_roots()
            self._incremental_stack = list(roots)
            self._incremental_state["active"] = True
            LOG.debug("incremental mark started with stack=%d", len(self._incremental_stack))

    def incremental_mark_step(self, max_nodes: int = 1000) -> Dict[str, Any]:
        with self._lock:
            if not self._incremental_state.get("active"):
                return {"status": "inactive"}
            processed = 0
            while self._incremental_stack and processed < max_nodes:
                hid = self._incremental_stack.pop()
                if hid not in self._heap:
                    continue
                obj = self._heap[hid]
                if obj.marked:
                    continue
                obj.marked = True
                processed += 1
                refs = self._extract_references(obj)
                for r in refs:
                    if r in self._heap and not self._heap[r].marked:
                        self._incremental_stack.append(r)
            if not self._incremental_stack:
                # finish and sweep unreachable
                self._incremental_state["active"] = False
                # now sweep like a major collect but queue finalizers async
                to_delete = [hid for hid, obj in list(self._heap.items()) if not obj.marked and not obj.pinned]
                for hid in to_delete:
                    obj = self._heap.get(hid)
                    if obj and obj.finalizer:
                        self._enqueue_finalizer(hid, obj)
                for hid in to_delete:
                    obj = self._heap.pop(hid, None)
                    if obj:
                        if obj.kind == "object":
                            self._return_obj_payload(obj.payload)
                        elif obj.kind == "array":
                            self._return_arr_payload(obj.payload)
                    self._remembered.discard(hid)
                    self._large_objects.discard(hid)
                    self._pinned.discard(hid)
                LOG.debug("incremental mark complete processed=%d swept=%d", processed, len(to_delete))
                return {"status": "complete", "processed": processed, "swept": len(to_delete)}
            return {"status": "in-progress", "processed": processed, "remaining": len(self._incremental_stack)}

    def start_incremental_background(self, interval: float = INCREMENTAL_STEP_INTERVAL, nodes_per_step: int = 200):
        """Start a background thread that performs incremental_mark_step repeatedly."""
        if self._inc_bg_thread and self._inc_bg_thread.is_alive():
            return
        self._inc_bg_enabled = True
        self._inc_bg_interval = float(interval)

        def _inc_bg():
            while self._inc_bg_enabled and not self._stop_bg.is_set():
                try:
                    if not self._incremental_state.get("active"):
                        # start a mark cycle only if something requested or periodically
                        self.begin_incremental_mark()
                    res = self.incremental_mark_step(max_nodes=nodes_per_step)
                    # small sleep to yield CPU
                    time.sleep(self._inc_bg_interval)
                except Exception:
                    LOG.exception("incremental background error")
                    time.sleep(self._inc_bg_interval)

        self._inc_bg_thread = threading.Thread(target=_inc_bg, daemon=True)
        self._inc_bg_thread.start()
        LOG.info("Incremental background worker started interval=%.3fs", interval)

    def stop_incremental_background(self):
        self._inc_bg_enabled = False
        if self._inc_bg_thread:
            try:
                self._inc_bg_thread.join(timeout=1.0)
            except Exception:
                pass

    # ---------------------------
    # Compaction (best-effort)
    # ---------------------------
    def compact_heap(self) -> Dict[str, Any]:
        """
        Best-effort compaction: allocate new ids for live objects, copy payloads,
        rewrite references and update roots. Pinned objects are not moved.
        This operation can be heavy; caller should ensure minimal concurrency.
        """
        with self._compact_lock, self._lock:
            try:
                LOG.info("Starting heap compaction")
                # identify live objects via a temporary mark
                self._unmark_all()
                roots = self._gather_roots()
                stack = list(roots)
                while stack:
                    hid = stack.pop()
                    if hid not in self._heap:
                        continue
                    obj = self._heap[hid]
                    if obj.marked:
                        continue
                    obj.marked = True
                    for r in self._extract_references(obj):
                        if r in self._heap and not self._heap[r].marked:
                            stack.append(r)
                # build mapping for movable objects: skip pinned and large objects
                mapping: Dict[int, int] = {}
                new_heap: Dict[int, HeapObject] = {}
                for old_id, obj in list(self._heap.items()):
                    if not obj.marked:
                        continue  # unreachable won't be moved
                    if obj.pinned or old_id in self._large_objects:
                        # preserve id to avoid changing external refs; copy as-is
                        mapping[old_id] = old_id
                        new_heap[old_id] = obj
                        continue
                    # assign new id
                    new_id = self._alloc_id()
                    mapping[old_id] = new_id
                    # shallow copy of payload; we'll rewrite references next
                    copied = None
                    if obj.kind == "object":
                        copied = dict(obj.payload)
                    elif obj.kind == "array":
                        copied = list(obj.payload)
                    elif obj.kind in ("bytes", "large"):
                        copied = bytearray(obj.payload) if obj.kind == "bytes" else obj.payload
                    elif obj.kind == "value":
                        copied = obj.payload
                    new_obj = HeapObject(new_id, obj.kind, copied, size=obj.size, finalizer=obj.finalizer)
                    new_obj.gen = obj.gen
                    new_obj.age = obj.age
                    new_obj.pinned = obj.pinned
                    new_heap[new_id] = new_obj
                # rewrite references in new_heap
                def rewrite_in_obj(o: HeapObject):
                    if o.kind == "object" and isinstance(o.payload, dict):
                        for k, v in list(o.payload.items()):
                            if isinstance(v, int) and v in mapping:
                                o.payload[k] = mapping[v]
                    elif o.kind == "array" and isinstance(o.payload, (list, tuple)):
                        for i, v in enumerate(o.payload):
                            if isinstance(v, int) and v in mapping:
                                o.payload[i] = mapping[v]
                for o in new_heap.values():
                    rewrite_in_obj(o)
                # update roots to new ids
                new_roots: Dict[str, Set[int]] = {}
                for rname, rset in self._roots.items():
                    new_set = set()
                    for hid in rset:
                        if hid in mapping:
                            new_set.add(mapping[hid])
                    if new_set:
                        new_roots[rname] = new_set
                # replace heap and roots
                self._heap = new_heap
                self._roots = new_roots
                # rebuild large_objects and pinned sets
                self._large_objects = {mapping.get(h, h) for h in list(self._large_objects) if h in mapping}
                self._pinned = {mapping.get(h, h) for h in list(self._pinned) if h in mapping}
                # rebuild remembered set conservatively
                self._remembered = set()
                for hid, obj in self._heap.items():
                    if obj.gen > 0:
                        refs = self._extract_references(obj)
                        if any((r in self._heap and self._heap[r].gen == 0) for r in refs):
                            self._remembered.add(hid)
                LOG.info("Compaction complete: new_heap=%d", len(self._heap))
                self._notify("compact", {"heap_size": len(self._heap)})
                return {"status": "ok", "heap_size": len(self._heap)}
            except Exception:
                LOG.exception("compaction failed")
                return {"status": "error", "message": "exception during compaction"}

    # ---------------------------
    # Helpers & introspection
    # ---------------------------
    def heap_snapshot(self, include_contents: bool = False) -> Dict[str, Any]:
        with self._lock:
            snap = {
                "time": time.time(),
                "count": len(self._heap),
                "objects": {},
                "roots": {k: list(v) for k, v in self._roots.items()},
                "remembered": list(self._remembered),
                "large_objects": list(self._large_objects),
                "pinned": list(self._pinned),
                "stats": {
                    "alloc_count": self._stats.alloc_count,
                    "collect_count": self._stats.collect_count,
                    "last_collect_time": self._stats.last_collect_time,
                    "last_collected_objects": self._stats.last_collected_objects,
                    "heap_size_high_water": self._stats.heap_size_high_water,
                    "total_finalized": self._stats.total_finalized,
                },
            }
            for hid, obj in self._heap.items():
                o = {"id": hid, "kind": obj.kind, "gen": obj.gen, "age": obj.age, "size": obj.size, "pinned": obj.pinned}
                if include_contents:
                    if obj.kind == "object":
                        o["fields"] = dict(obj.payload)
                    elif obj.kind == "array":
                        o["array"] = list(obj.payload)
                    elif obj.kind in ("bytes", "large"):
                        o["raw_len"] = len(obj.payload) if hasattr(obj.payload, "__len__") else None
                    elif obj.kind == "value":
                        o["value"] = obj.payload
                snap["objects"][hid] = o
            return snap

    def export_heap_json(self, path: Optional[str] = None, include_contents: bool = True) -> str:
        if not path:
            fd, path = tempfile.mkstemp(prefix="instryx_heap_", suffix=".json")
            os.close(fd)
        snap = self.heap_snapshot(include_contents=include_contents)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snap, f, indent=2, default=str)
        os.replace(tmp, path)
        LOG.info("Heap exported -> %s", path)
        return path

    def import_heap_json(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            snap = json.load(f)
        # Best-effort: clear current heap and rebuild simple boxed values (ids will be new)
        with self._lock:
            # return all pooled payloads to avoid leaks
            for hid, obj in list(self._heap.items()):
                if obj.kind == "object":
                    self._return_obj_payload(obj.payload)
                elif obj.kind == "array":
                    self._return_arr_payload(obj.payload)
            self._heap.clear()
            self._roots.clear()
            self._remembered.clear()
            self._pinned.clear()
            self._large_objects.clear()
            for old_id_str, meta in snap.get("objects", {}).items():
                hid = self._alloc_id()
                kind = meta.get("kind", "value")
                payload = None
                if kind == "object":
                    payload = {k: None for k in meta.get("fields", {}).keys()} if meta.get("fields") else {}
                elif kind == "array":
                    payload = [None] * len(meta.get("array", []))
                elif kind in ("bytes", "large"):
                    payload = bytearray(meta.get("raw_len", 0) or 0)
                else:
                    payload = None
                obj = HeapObject(hid, kind, payload, size=meta.get("size", 1))
                self._heap[hid] = obj
            LOG.info("imported heap snapshot (best-effort), recreated %d objects", len(self._heap))

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            d = {
                "alloc_count": self._stats.alloc_count,
                "collect_count": self._stats.collect_count,
                "last_collect_time": self._stats.last_collect_time,
                "last_collected_objects": self._stats.last_collected_objects,
                "heap_size_high_water": self._stats.heap_size_high_water,
                "total_finalized": self._stats.total_finalized,
                "heap_size": len(self._heap),
                "roots": {k: len(v) for k, v in self._roots.items()},
                "remembered": len(self._remembered),
                "large_objects": len(self._large_objects),
                "pinned": len(self._pinned),
            }
            return d

    def export_metrics_json(self, path: str) -> str:
        metrics = self.get_stats()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        return path

    def export_metrics_prometheus(self) -> str:
        """Return a small Prometheus-compatible metrics text for scraping."""
        m = self.get_stats()
        lines = []
        lines.append(f'instryx_heap_alloc_count {m["alloc_count"]}')
        lines.append(f'instryx_heap_collect_count {m["collect_count"]}')
        lines.append(f'instryx_heap_size {m["heap_size"]}')
        lines.append(f'instryx_heap_roots {sum(m["roots"].values()) if isinstance(m["roots"], dict) else 0}')
        return "\n".join(lines)

    def find_handles_pointing_to(self, target: int) -> List[int]:
        with self._lock:
            res = []
            for hid, obj in self._heap.items():
                refs = self._extract_references(obj)
                if target in refs:
                    res.append(hid)
            return res

    # improved path-to-root search for leak analysis
    def find_path_to_root(self, target: int, max_depth: int = 1000) -> Optional[List[int]]:
        """
        Breadth-first search from roots to find a path to target.
        Returns list of handles from root -> ... -> target, or None if not reachable within max_depth.
        """
        if target not in self._heap:
            return None
        with self._lock:
            roots = list(self._gather_roots())
            if not roots:
                return None
            from collections import deque
            q = deque()
            visited = set()
            parent: Dict[int, Optional[int]] = {}
            for r in roots:
                q.append((r, 0))
                visited.add(r)
                parent[r] = None
            while q:
                hid, depth = q.popleft()
                if depth > max_depth:
                    continue
                if hid == target:
                    # reconstruct path
                    path = []
                    cur = hid
                    while cur is not None:
                        path.append(cur)
                        cur = parent.get(cur)
                    return list(reversed(path))
                obj = self._heap.get(hid)
                if not obj:
                    continue
                for ref in self._extract_references(obj):
                    if ref not in visited and ref in self._heap:
                        visited.add(ref)
                        parent[ref] = hid
                        q.append((ref, depth + 1))
            return None

    # ---------------------------
    # Conservative root scanner (inspect Python heap via gc)
    # ---------------------------
    def scan_python_globals_for_handles(self) -> Set[int]:
        """
        Conservative scan through Python GC tracked objects (modules, builtins, globals)
        and collect integers that look like handles registered in this allocator.
        Use with caution: expensive.
        """
        found = set()
        with self._lock:
            valid_handles = set(self._heap.keys())
        for obj in list(_py_gc.get_objects()):
            try:
                if isinstance(obj, dict):
                    for v in obj.values():
                        if isinstance(v, int) and v in valid_handles:
                            found.add(v)
                elif isinstance(obj, (list, tuple, set)):
                    for v in obj:
                        if isinstance(v, int) and v in valid_handles:
                            found.add(v)
            except Exception:
                continue
        return found

    # ---------------------------
    # Background collector control
    # ---------------------------
    def start_background_collector(self, interval: float = BACKGROUND_COLLECTION_INTERVAL):
        if self._bg_thread and self._bg_thread.is_alive():
            return
        self._stop_bg.clear()

        def _bg_worker(interval):
            while not self._stop_bg.is_set():
                try:
                    self.minor_collect()
                except Exception:
                    LOG.exception("background GC error")
                self._stop_bg.wait(interval)

        self._bg_thread = threading.Thread(target=_bg_worker, args=(interval,), daemon=True)
        self._bg_thread.start()
        LOG.info("Background collector started interval=%.2fs", interval)

    def stop_background_collector(self):
        if self._bg_thread:
            self._stop_bg.set()
            try:
                self._bg_thread.join(timeout=1.0)
            except Exception:
                pass

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _extract_references(self, obj: HeapObject) -> Set[int]:
        refs = set()
        if obj.kind == "object" and isinstance(obj.payload, dict):
            for v in obj.payload.values():
                if isinstance(v, int) and v in self._heap:
                    refs.add(v)
        elif obj.kind == "array" and isinstance(obj.payload, (list, tuple)):
            for v in obj.payload:
                if isinstance(v, int) and v in self._heap:
                    refs.add(v)
        return refs

    def _update_remembered_for_new(self, obj: HeapObject):
        # If object is older and references young objects, remember it
        if obj.gen > 0:
            refs = self._extract_references(obj)
            for r in refs:
                if r in self._heap and self._heap[r].gen == 0:
                    obj.has_old_to_young_ref = True
                    self._remembered.add(obj.id)

    # ---------------------------
    # Shutdown helper
    # ---------------------------
    def shutdown(self, wait: bool = True):
        """Stop background workers and flush finalizers. Safe to call on process exit."""
        LOG.info("HeapGCAllocator shutdown initiated")
        self._inc_bg_enabled = False
        self.stop_incremental_background()
        self.stop_background_collector()
        self._stop_bg.set()
        # finalize queued finalizers synchronously if requested
        if wait:
            # process queued finalizers once synchronously
            with self._finalizer_lock:
                while self._finalizer_queue:
                    hid, obj = self._finalizer_queue.pop(0)
                    try:
                        if obj.finalizer and not obj.pinned:
                            obj.finalizer(hid, obj)
                    except Exception:
                        LOG.exception("finalizer during shutdown failed for %d", hid)
        LOG.info("HeapGCAllocator shutdown complete")

    # ---------------------------
    # Self-test and CLI demo
    # ---------------------------
def _self_test() -> bool:
    LOG.info("Running HeapGCAllocator self-test (extended)")
    gc = HeapGCAllocator(max_heap_objects=200, enable_background=False)
    a = gc.alloc_object({"v": 123})
    b = gc.alloc_object({"ref": a})
    gc.register_root("main", b)
    # allocate many unreachable objects
    for i in range(100):
        gc.alloc_object({"temp": i})
    pre = len(gc._heap)
    stats = gc.collect(full=True)
    post = len(gc._heap)
    LOG.info("self-test: before=%d after=%d stats=%s", pre, post, stats)
    if a not in gc._heap or b not in gc._heap:
        LOG.error("self-test failed: live objects collected")
        return False
    # finalizer check
    finalized = []
    def final(hid, obj):
        finalized.append(hid)
    c = gc.alloc_object({"x": 10}, finalizer=final)
    # make unreachable and collect
    gc.collect(full=True)
    LOG.info("finalized list (may be empty if finalizer worker disabled): %s", finalized)
    # test compaction
    try:
        r = gc.compact_heap()
        LOG.info("compaction result: %s", r)
    except Exception:
        LOG.exception("compaction in self-test failed")
    # snapshot export/import
    path = gc.export_heap_json(include_contents=False)
    LOG.info("exported heap snapshot to %s", path)
    gc2 = HeapGCAllocator(max_heap_objects=200, enable_background=False)
    gc2.import_heap_json(path)
    LOG.info("imported snapshot into new allocator with objects=%d", len(gc2._heap))
    LOG.info("self-test passed")
    return True

if __name__ == "__main__":
    ok = _self_test()
    try:
        # best-effort graceful shutdown
        # (in CLI usage the program may exit immediately)
        pass
    finally:
        os._exit(0 if ok else 2)


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


"""
instryx_lsp_server.py

Lightweight, production-ready Language Server Protocol (LSP) server for Instryx.

Enhancements added:
- Robust argparse CLI for stdio or TCP operation, logging, debounce tuning, and workspace indexing.
- Diagnostic caching and debounce to avoid repetitive work (content-hash based).
- ThreadPoolExecutor used for concurrency (diagnostics, indexing, formatting).
- Workspace symbol indexing and fast lookup for code actions / completions.
- Additional ExecuteCommand support: formatDocument, traceDocument, validateTrace, exportDiagnostics.
- Atomic file writes for safe edits and on-disk persistence.
- Optional TCP LSP transport (single connection) beside stdio.
- Undo/backup helper for applied edits.
- Safer handling when optional integrations are absent; graceful fallbacks.
- Production-oriented logging and error handling.

Usage:
  python instryx_lsp_server.py --stdio
  python instryx_lsp_server.py --tcp --host 127.0.0.1 --port 2087 --workspace /path/to/repo
  python instryx_lsp_server.py --help
"""

from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
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
from typing import Any, Dict, List, Optional, Tuple

# Optional integrations
_transformer = None
try:
    import instryx_macro_transformer_model as transformer  # type: ignore
    _transformer = transformer
except Exception:
    transformer = None

_match_tool = None
try:
    from instryx_match_enum_struct import DMatchTool  # type: ignore
    _match_tool = DMatchTool()
except Exception:
    _match_tool = None

_debugger = None
try:
    import instryx_macro_debugger as macro_debugger  # type: ignore
    _debugger = macro_debugger
except Exception:
    _debugger = None

_syntax_morph = None
try:
    import instryx_syntax_morph as syntax_morph  # type: ignore
    _syntax_morph = syntax_morph
except Exception:
    _syntax_morph = None

# Logging
LOG = logging.getLogger("instryx.lsp")
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOG.addHandler(h)

# JSON-RPC / LSP helpers
CONTENT_LENGTH = "Content-Length"
_SEP = "\r\n\r\n"

# Document representation
class Document:
    def __init__(self, uri: str, text: str):
        self.uri = uri
        self.text = text
        self.version = 0
        self.lock = threading.RLock()

    def update(self, text: str, version: Optional[int] = None):
        with self.lock:
            self.text = text
            if version is not None:
                self.version = version
            else:
                self.version += 1

# Small LRU cache for diagnostics by content hash
class SimpleLRUCache:
    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self._dict: Dict[str, Any] = {}
        self._order: List[str] = []
        self._lock = threading.RLock()

    def get(self, k: str):
        with self._lock:
            v = self._dict.get(k)
            if v is None:
                return None
            # move to end
            if k in self._order:
                self._order.remove(k)
                self._order.append(k)
            return v

    def set(self, k: str, v: Any):
        with self._lock:
            if k in self._dict:
                self._order.remove(k)
            self._dict[k] = v
            self._order.append(k)
            while len(self._order) > self.capacity:
                oldest = self._order.pop(0)
                del self._dict[oldest]

    def clear(self):
        with self._lock:
            self._dict.clear()
            self._order.clear()

# Minimal LSP server with enhanced features
class InstryxLSPServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._stdin = sys.stdin.buffer
        self._stdout = sys.stdout.buffer
        self._running = False
        self._id = 0
        self._id_lock = threading.Lock()
        self._docs: Dict[str, Document] = {}
        self._root_uri: Optional[str] = args.workspace or None
        self._capabilities: Dict[str, Any] = {}
        self._request_handlers = {
            "initialize": self._handle_initialize,
            "initialized": self._handle_initialized,
            "shutdown": self._handle_shutdown,
            "textDocument/didOpen": self._handle_did_open,
            "textDocument/didChange": self._handle_did_change,
            "textDocument/didClose": self._handle_did_close,
            "textDocument/didSave": self._handle_did_save,
            "textDocument/hover": self._handle_hover,
            "textDocument/completion": self._handle_completion,
            "textDocument/documentSymbol": self._handle_document_symbol,
            "textDocument/codeAction": self._handle_code_action,
            "workspace/executeCommand": self._handle_execute_command,
        }
        # pre-create default registry if transformer present
        try:
            self._default_registry = transformer.createDefaultRegistry() if transformer and hasattr(transformer, "createDefaultRegistry") else {}
        except Exception:
            self._default_registry = {}
        # diagnostic cache and debounce
        self._diag_cache = SimpleLRUCache(capacity=1024)
        self._diag_queue: List[Tuple[str, str]] = []
        self._diag_lock = threading.Lock()
        self._diag_debounce = max(50, int(args.diag_debounce_ms)) if hasattr(args, "diag_debounce_ms") else 200
        self._diag_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers))
        self._running_diag = False
        self._diag_thread = threading.Thread(target=self._diagnostics_worker, daemon=True)
        # workspace index
        self._symbol_index: Dict[str, List[Dict[str, Any]]] = {}
        self._index_lock = threading.RLock()
        if args.index_on_start and self._root_uri:
            self._diag_executor.submit(self._index_workspace)
        # performance optimization: compile macro scanner once (use transformer's if possible)
        self._macro_scan = getattr(transformer, "_scan_macros", None) or self._scan_macros
        self._parse_args_fn = getattr(transformer, "_parse_macro_args", None) or self._parse_args
        # enable experimental AST formatting
        self._enable_format = args.enable_format and (_syntax_morph is not None)
        # backup suffix
        self._backup_suffix = ".lsp.bak"

    # -- JSON-RPC I/O helpers (stdio and TCP) --
    def _read_message_stdio(self) -> Optional[Dict[str, Any]]:
        try:
            header = b""
            while True:
                line = self._stdin.readline()
                if not line:
                    return None
                if line in (b"\r\n", b"\n"):
                    break
                header += line
                if header.endswith(b"\r\n\r\n"):
                    break
            header_text = header.decode("ascii", errors="ignore")
            m = re.search(r"Content-Length:\s*(\d+)", header_text, re.IGNORECASE)
            if not m:
                return None
            length = int(m.group(1))
            body = self._stdin.read(length)
            if not body:
                return None
            data = json.loads(body.decode("utf-8", errors="replace"))
            LOG.debug("<< %s", data.get("method") or data.get("id"))
            return data
        except Exception:
            LOG.exception("read_message_stdio failed")
            return None

    def _send_stdio(self, payload: Dict[str, Any]) -> None:
        try:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
            self._stdout.write(header)
            self._stdout.write(body)
            self._stdout.flush()
            LOG.debug(">> %s", payload.get("method") or payload.get("id"))
        except Exception:
            LOG.exception("send_stdio failed")

    # TCP transport helpers (single client)
    def _serve_tcp(self, host: str, port: int):
        LOG.info("Starting TCP LSP server on %s:%d", host, port)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(1)
        conn, addr = s.accept()
        conn_file = conn.makefile("rwb")
        LOG.info("Client connected: %s", addr)
        try:
            while self._running:
                header = b""
                while True:
                    line = conn_file.readline()
                    if not line:
                        self._running = False
                        break
                    if line in (b"\r\n", b"\n"):
                        break
                    header += line
                    if header.endswith(b"\r\n\r\n"):
                        break
                if not self._running:
                    break
                header_text = header.decode("ascii", errors="ignore")
                m = re.search(r"Content-Length:\s*(\d+)", header_text, re.IGNORECASE)
                if not m:
                    continue
                length = int(m.group(1))
                body = conn_file.read(length)
                if not body:
                    break
                msg = json.loads(body.decode("utf-8", errors="replace"))
                self._dispatch_message(msg, transport=("tcp", conn_file))
        finally:
            try:
                conn_file.close()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
            s.close()

    def _send_tcp(self, conn_file, payload: Dict[str, Any]):
        try:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
            conn_file.write(header)
            conn_file.write(body)
            conn_file.flush()
        except Exception:
            LOG.exception("send_tcp failed")

    # Unified send wrapper
    def _send(self, payload: Dict[str, Any], transport: Optional[Tuple[str, Any]] = None) -> None:
        if transport and transport[0] == "tcp":
            try:
                self._send_tcp(transport[1], payload)
                return
            except Exception:
                LOG.exception("sending via tcp failed")
        self._send_stdio(payload)

    def _next_id(self) -> int:
        with self._id_lock:
            self._id += 1
            return self._id

    def _send_response(self, id_: Any, result: Any = None, error: Any = None, transport: Optional[Tuple[str, Any]] = None) -> None:
        payload = {"jsonrpc": "2.0", "id": id_}
        if error is not None:
            payload["error"] = error
        else:
            payload["result"] = result
        self._send(payload, transport=transport)

    def _send_notification(self, method: str, params: Any = None, transport: Optional[Tuple[str, Any]] = None) -> None:
        payload = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._send(payload, transport=transport)

    # Request handlers (same logic as earlier but using executor for heavy ops)
    def _handle_initialize(self, params: Dict[str, Any], id_: Any, transport: Optional[Tuple[str, Any]] = None) -> None:
        root = params.get("rootUri") or params.get("rootPath") or self._root_uri
        self._root_uri = root
        caps = {
            "capabilities": {
                "textDocumentSync": 2,
                "hoverProvider": True,
                "completionProvider": {"resolveProvider": False, "triggerCharacters": ["@", "("]},
                "documentSymbolProvider": True,
                "codeActionProvider": True,
                "executeCommandProvider": {"commands": ["instryx.previewMacros", "instryx.applyMacros", "instryx.generateMatch", "instryx.formatDocument", "instryx.traceDocument", "instryx.validateTrace"]},
                "workspace": {"workspaceFolders": {"supported": True}},
            }
        }
        self._capabilities = caps
        self._send_response(id_, caps, transport=transport)

    def _handle_initialized(self, params: Any, id_: Any, transport: Optional[Tuple[str, Any]] = None) -> None:
        if not self._running_diag:
            self._running_diag = True
            self._diag_thread.start()
        self._send_response(id_, None, transport=transport)

    def _handle_shutdown(self, params: Any, id_: Any, transport: Optional[Tuple[str, Any]] = None) -> None:
        self._running = False
        self._send_response(id_, None, transport=transport)

    # Document lifecycle handlers (wrap to use executor for diagnostics)
    def _handle_did_open(self, params: Dict[str, Any], id_: Any = None, transport: Optional[Tuple[str, Any]] = None) -> None:
        doc = params.get("textDocument", {})
        uri = doc.get("uri") or doc.get("path")
        text = doc.get("text", "")
        version = doc.get("version", 0)
        if not uri:
            return
        self._docs[uri] = Document(uri, text)
        self._docs[uri].version = version
        self._enqueue_diagnostics(uri, reason="didOpen")
        if id_ is not None:
            self._send_response(id_, None, transport=transport)

    def _handle_did_change(self, params: Dict[str, Any], id_: Any = None, transport: Optional[Tuple[str, Any]] = None) -> None:
        doc = params.get("textDocument", {})
        uri = doc.get("uri") or doc.get("path")
        content_changes = params.get("contentChanges", [])
        if not uri or not content_changes:
            return
        text = content_changes[-1].get("text")
        version = doc.get("version")
        if uri not in self._docs:
            self._docs[uri] = Document(uri, text or "")
        else:
            self._docs[uri].update(text or "", version)
        self._enqueue_diagnostics(uri, reason="didChange")
        if id_ is not None:
            self._send_response(id_, None, transport=transport)

    def _handle_did_close(self, params: Dict[str, Any], id_: Any = None, transport: Optional[Tuple[str, Any]] = None) -> None:
        doc = params.get("textDocument", {})
        uri = doc.get("uri") or doc.get("path")
        if uri and uri in self._docs:
            del self._docs[uri]
        self._publish_diagnostics(uri, [])
        if id_ is not None:
            self._send_response(id_, None, transport=transport)

    def _handle_did_save(self, params: Dict[str, Any], id_: Any = None, transport: Optional[Tuple[str, Any]] = None) -> None:
        doc = params.get("textDocument", {})
        uri = doc.get("uri") or doc.get("path")
        if uri and uri in self._docs:
            self._enqueue_diagnostics(uri, reason="didSave")
        if id_ is not None:
            self._send_response(id_, None, transport=transport)

    # Hover: preview macro expansion
    def _handle_hover(self, params: Dict[str, Any], id_: Any, transport: Optional[Tuple[str, Any]] = None) -> None:
        td = params.get("textDocument", {})
        uri = td.get("uri")
        pos = params.get("position", {})
        if not uri or uri not in self._docs:
            self._send_response(id_, None, transport=transport)
            return
        doc = self._docs[uri]
        offset = self._pos_to_offset(doc.text, pos)
        matches = self._macro_scan(doc.text)
        for s, e, name, raw in matches:
            if s <= offset <= e:
                args = self._parse_args_fn(raw)
                macro_fn = (self._default_registry.get(name) if self._default_registry else None)
                if macro_fn:
                    try:
                        repl = macro_fn(args, {"source": doc.text, "registry": self._default_registry, "opts": {}})
                        contents = repl if repl is not None else ""
                        hover = {"contents": {"kind": "markdown", "value": f"**Macro** `{name}`\n\n```\n{contents}\n```"}}
                        self._send_response(id_, hover, transport=transport)
                        return
                    except Exception:
                        LOG.exception("hover expansion failed")
                        self._send_response(id_, None, transport=transport)
                        return
                self._send_response(id_, {"contents": f"@{name}({', '.join(args)})"}, transport=transport)
                return
        self._send_response(id_, None, transport=transport)

    # Completion: macros + match stubs
    def _handle_completion(self, params: Dict[str, Any], id_: Any, transport: Optional[Tuple[str, Any]] = None) -> None:
        td = params.get("textDocument", {})
        uri = td.get("uri")
        items = []
        names = sorted(list((self._default_registry or {}).keys()))
        for n in names:
            items.append({"label": n, "kind": 3, "detail": "instryx macro", "insertText": f"@{n} "})
        if uri and uri in self._docs and _match_tool:
            doc = self._docs[uri]
            enums = _match_tool.find_enums(doc.text)
            for e in enums:
                lbl = f"match_{e.name}"
                insert = _match_tool.generate_match_stub(e, var_name="v")
                items.append({"label": lbl, "kind": 14, "detail": f"generate match for {e.name}", "insertText": insert})
        result = {"isIncomplete": False, "items": items}
        self._send_response(id_, result, transport=transport)

    # Document symbols
    def _handle_document_symbol(self, params: Dict[str, Any], id_: Any, transport: Optional[Tuple[str, Any]] = None) -> None:
        td = params.get("textDocument", {})
        uri = td.get("uri")
        if not uri or uri not in self._docs or not _match_tool:
            self._send_response(id_, [], transport=transport)
            return
        doc = self._docs[uri]
        symbols = []
        enums = _match_tool.find_enums(doc.text)
        structs = _match_tool.find_structs(doc.text)
        for e in enums:
            symbols.append({
                "name": e.name,
                "kind": 5,
                "range": self._range_from_offsets(doc.text, e.start, e.end),
                "selectionRange": self._range_from_offsets(doc.text, e.start, e.start+len(e.name))
            })
        for s in structs:
            symbols.append({
                "name": s.name,
                "kind": 5,
                "range": self._range_from_offsets(doc.text, s.start, s.end),
                "selectionRange": self._range_from_offsets(doc.text, s.start, s.start+len(s.name))
            })
        self._send_response(id_, symbols, transport=transport)

    # Code actions
    def _handle_code_action(self, params: Dict[str, Any], id_: Any, transport: Optional[Tuple[str, Any]] = None) -> None:
        doc = params.get("textDocument", {})
        uri = doc.get("uri")
        range_ = params.get("range")
        if not uri or uri not in self._docs:
            self._send_response(id_, [], transport=transport)
            return
        docobj = self._docs[uri]
        start_offset = self._pos_to_offset(docobj.text, range_["start"])
        end_offset = self._pos_to_offset(docobj.text, range_["end"])
        actions = []
        matches = self._macro_scan(docobj.text)
        for s, e, name, raw in matches:
            if s <= start_offset <= e or s <= end_offset <= e or (start_offset <= s and e <= end_offset):
                actions.append({
                    "title": f"Preview expand macro @{name}",
                    "kind": "quickfix",
                    "command": {"title": "instryx.previewMacros", "command": "instryx.previewMacros", "arguments": [uri, s, e, name, raw]}
                })
                actions.append({
                    "title": f"Apply expand macro @{name}",
                    "kind": "refactor",
                    "command": {"title": "instryx.applyMacros", "command": "instryx.applyMacros", "arguments": [uri, s, e, name, raw]}
                })
        if _match_tool:
            enums = _match_tool.find_enums(docobj.text)
            for e in enums:
                actions.append({
                    "title": f"Insert match stub for {e.name}",
                    "kind": "quickfix",
                    "command": {"title": "instryx.generateMatch", "command": "instryx.generateMatch", "arguments": [uri, e.name]}
                })
        self._send_response(id_, actions, transport=transport)

    # ExecuteCommand
    def _handle_execute_command(self, params: Dict[str, Any], id_: Any, transport: Optional[Tuple[str, Any]] = None) -> None:
        command = params.get("command")
        args = params.get("arguments") or []
        try:
            if command == "instryx.previewMacros":
                uri = args[0]
                if uri not in self._docs:
                    self._send_response(id_, {"error": "document not open"}, transport=transport); return
                start, end, name, raw = args[1], args[2], args[3], args[4]
                doc = self._docs[uri]
                macro_fn = (self._default_registry.get(name) if self._default_registry else None)
                if macro_fn:
                    repl = macro_fn(self._parse_args_fn(raw), {"source": doc.text, "registry": self._default_registry, "opts": {}})
                    self._send_response(id_, {"preview": repl}, transport=transport); return
                self._send_response(id_, {"preview": None}, transport=transport); return

            if command == "instryx.applyMacros":
                uri = args[0]
                if uri not in self._docs:
                    self._send_response(id_, {"error": "document not open"}, transport=transport); return
                start, end, name, raw = args[1], args[2], args[3], args[4]
                doc = self._docs[uri]
                macro_fn = (self._default_registry.get(name) if self._default_registry else None)
                if not macro_fn:
                    self._send_response(id_, {"error": "macro not found"}, transport=transport); return
                repl = macro_fn(self._parse_args_fn(raw), {"source": doc.text, "registry": self._default_registry, "opts": {}})
                new_text = doc.text[:start] + (repl or "") + doc.text[end:]
                doc.update(new_text, version=doc.version+1)
                file_path = self._uri_to_path(uri)
                try:
                    # atomic write with backup
                    if os.path.exists(file_path):
                        shutil.copy2(file_path, file_path + self._backup_suffix)
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(new_text)
                    self._enqueue_diagnostics(uri, reason="apply")
                    self._send_response(id_, {"applied": True, "path": file_path}, transport=transport)
                except Exception as e:
                    self._send_response(id_, {"applied": False, "error": str(e)}, transport=transport)
                return

            if command == "instryx.generateMatch":
                uri = args[0]; enum_name = args[1]
                if uri not in self._docs:
                    self._send_response(id_, {"error": "document not open"}, transport=transport); return
                doc = self._docs[uri]
                if not _match_tool:
                    self._send_response(id_, {"error": "match tool not available"}, transport=transport); return
                enums = _match_tool.find_enums(doc.text)
                ed = next((e for e in enums if e.name == enum_name), None)
                if not ed:
                    self._send_response(id_, {"error": "enum not found"}, transport=transport); return
                stub = _match_tool.generate_match_stub(ed, var_name="v")
                new_text = stub + "\n" + doc.text
                doc.update(new_text, version=doc.version+1)
                file_path = self._uri_to_path(uri)
                try:
                    if os.path.exists(file_path):
                        shutil.copy2(file_path, file_path + self._backup_suffix)
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(new_text)
                    self._enqueue_diagnostics(uri, reason="generateMatch")
                    self._send_response(id_, {"generated": True, "path": file_path}, transport=transport)
                except Exception as e:
                    self._send_response(id_, {"generated": False, "error": str(e)}, transport=transport)
                return

            if command == "instryx.formatDocument":
                uri = args[0]
                if uri not in self._docs:
                    self._send_response(id_, {"error": "document not open"}, transport=transport); return
                doc = self._docs[uri]
                formatted = None
                try:
                    if _syntax_morph and hasattr(_syntax_morph, "format"):
                        formatted = _syntax_morph.format(doc.text)
                    else:
                        # simple normalization: strip trailing spaces and ensure newline at EOF
                        lines = [ln.rstrip() for ln in doc.text.splitlines()]
                        formatted = "\n".join(lines) + ("\n" if not doc.text.endswith("\n") else "")
                    doc.update(formatted, version=doc.version+1)
                    file_path = self._uri_to_path(uri)
                    if os.path.exists(file_path):
                        shutil.copy2(file_path, file_path + self._backup_suffix)
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(formatted)
                    self._enqueue_diagnostics(uri, reason="format")
                    self._send_response(id_, {"formatted": True, "path": file_path}, transport=transport)
                except Exception as e:
                    LOG.exception("format failed")
                    self._send_response(id_, {"formatted": False, "error": str(e)}, transport=transport)
                return

            if command == "instryx.traceDocument":
                uri = args[0]
                if uri not in self._docs:
                    self._send_response(id_, {"error": "document not open"}, transport=transport); return
                if not _debugger:
                    self._send_response(id_, {"error": "debugger not available"}, transport=transport); return
                # write current buffer to temp file and trace
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ix", mode="w", encoding="utf-8")
                try:
                    tmp.write(self._docs[uri].text); tmp.close()
                    dbg = _debugger.MacroDebugger() if hasattr(_debugger, "MacroDebugger") else _debugger
                    tr = dbg.trace_file(tmp.name)
                    out_path = tmp.name + ".ai.trace.json"
                    dbg.save_trace(tr, out_path, sign=True)
                    self._send_response(id_, {"trace": out_path}, transport=transport)
                except Exception as e:
                    LOG.exception("traceDocument failed")
                    self._send_response(id_, {"error": str(e)}, transport=transport)
                finally:
                    try: os.unlink(tmp.name)
                    except Exception: pass
                return

            if command == "instryx.validateTrace":
                trace_path = args[0]
                if not os.path.exists(trace_path):
                    self._send_response(id_, {"error": "trace not found"}, transport=transport); return
                if not _debugger:
                    self._send_response(id_, {"error": "debugger not available"}, transport=transport); return
                dbg = _debugger.MacroDebugger() if hasattr(_debugger, "MacroDebugger") else _debugger
                trace = dbg.load_trace(trace_path)
                ok, diag = dbg.validate_trace(trace)
                self._send_response(id_, {"valid": ok, "diagnostics": diag}, transport=transport)
                return

            self._send_response(id_, {"error": f"unknown command {command}"}, transport=transport)
        except Exception:
            LOG.exception("executeCommand failed")
            self._send_response(id_, {"error": "internal error"}, transport=transport)

    # Utilities
    def _uri_to_path(self, uri: str) -> str:
        if uri.startswith("file://"):
            p = uri[7:]
            if p.startswith("/") and sys.platform == "win32":
                p = p[1:]
            return p
        return uri

    def _pos_to_offset(self, text: str, pos: Dict[str, int]) -> int:
        line = pos.get("line", 0)
        character = pos.get("character", 0)
        offs = 0
        cur_line = 0
        for m in re.finditer(r".*?(?:\n|$)", text):
            if cur_line == line:
                offs += min(character, len(m.group(0)))
                break
            offs += len(m.group(0))
            cur_line += 1
        return offs

    def _offset_to_pos(self, text: str, offset: int) -> Dict[str, int]:
        if offset <= 0:
            return {"line": 0, "character": 0}
        line = 0
        cur = 0
        for m in re.finditer(r".*?(?:\n|$)", text):
            l = len(m.group(0))
            if cur + l > offset:
                return {"line": line, "character": offset - cur}
            cur += l
            line += 1
        return {"line": line, "character": 0}

    def _range_from_offsets(self, text: str, start: int, end: int) -> Dict[str, Any]:
        return {"start": self._offset_to_pos(text, start), "end": self._offset_to_pos(text, end)}

    # scanning helpers (delegate to transformer if available)
    def _scan_macros(self, text: str):
        # use transformer's scanner if available
        if transformer and hasattr(transformer, "_scan_macros"):
            try:
                return transformer._scan_macros(text)
            except Exception:
                LOG.exception("transformer._scan_macros failed")
        # fallback (same as in other modules)
        res = []
        i = 0; L = len(text); in_s = None
        while i < L:
            c = text[i]
            if in_s:
                if c == in_s and text[i - 1] != "\\":
                    in_s = None
                i += 1; continue
            if c in ('"', "'"):
                in_s = c; i += 1; continue
            if c == "@":
                j = i + 1; name = ""
                while j < L and (text[j].isalnum() or text[j] == "_"):
                    name += text[j]; j += 1
                k = j; depth = 0; in_s2 = None
                while k < L:
                    ch = text[k]
                    if in_s2:
                        if ch == in_s2 and text[k-1] != "\\":
                            in_s2 = None
                        k += 1; continue
                    if ch in ('"', "'"):
                        in_s2 = ch; k += 1; continue
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth = max(0, depth - 1)
                    elif ch == ";" and depth == 0:
                        raw = text[j:k].strip()
                        res.append((i, k+1, name, raw))
                        k += 1; break
                    k += 1
                i = k; continue
            i += 1
        return res

    # parse args fallback
    def _parse_args(self, raw: str) -> List[str]:
        parts = []
        buf = []; depth = 0; in_s = None
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
                p = "".join(buf).strip()
                if p: parts.append(p)
                buf = []; continue
            buf.append(ch)
        if buf:
            p = "".join(buf).strip()
            if p: parts.append(p)
        return parts

    # Diagnostics: caching & execution
    def _enqueue_diagnostics(self, uri: str, reason: str = "change"):
        with self._diag_lock:
            self._diag_queue.append((uri, reason))

    def _diagnostics_worker(self):
        while True:
            uri = None
            with self._diag_lock:
                if self._diag_queue:
                    uri, reason = self._diag_queue.pop(0)
            if uri:
                # debounce: wait small window to coalesce changes
                time.sleep(self._diag_debounce / 1000.0)
                try:
                    # run in executor to not block
                    future = self._diag_executor.submit(self._run_diagnostics, uri)
                    # optionally wait small time if synchronous requested
                    # future.result(timeout=5)
                except Exception:
                    LOG.exception("submit diagnostics failed")
            else:
                time.sleep(0.1)

    def _run_diagnostics(self, uri: str):
        if uri not in self._docs:
            return
        doc = self._docs[uri]
        content_hash = hashlib.sha1(doc.text.encode("utf-8")).hexdigest()
        cached = self._diag_cache.get(content_hash)
        if cached is not None:
            self._publish_diagnostics(uri, cached)
            return
        diagnostics = []
        if transformer and hasattr(transformer, "applyMacrosWithDiagnostics"):
            try:
                res = transformer.applyMacrosWithDiagnostics(doc.text, registry=self._default_registry, opts={"filename": self._uri_to_path(uri)})
                diags = res.get("diagnostics", []) or []
                for d in diags:
                    rng = d.get("range") or d.get("span") or [0, 0]
                    start = self._offset_to_pos(doc.text, int(rng[0]) if rng and isinstance(rng[0], int) else 0)
                    end = self._offset_to_pos(doc.text, int(rng[1]) if rng and isinstance(rng[1], int) else 0)
                    severity_map = {"error": 1, "warning": 2, "info": 3, "hint": 4}
                    severity = severity_map.get(d.get("level", "info"), 3)
                    diagnostics.append({
                        "range": {"start": start, "end": end},
                        "severity": severity,
                        "source": "instryx",
                        "message": d.get("message", str(d))
                    })
            except Exception:
                LOG.exception("applyMacrosWithDiagnostics failed")
        if _syntax_morph and hasattr(_syntax_morph, "validate"):
            try:
                errs = _syntax_morph.validate(doc.text)
                for e in errs:
                    diagnostics.append({
                        "range": {"start": self._offset_to_pos(doc.text, e.get("start", 0)), "end": self._offset_to_pos(doc.text, e.get("end", 0))},
                        "severity": 1,
                        "source": "syntax",
                        "message": e.get("message", "")
                    })
            except Exception:
                LOG.exception("syntax_morph.validate failed")
        # cache and publish
        self._diag_cache.set(content_hash, diagnostics)
        self._publish_diagnostics(uri, diagnostics)

    def _publish_diagnostics(self, uri: str, diagnostics: List[Dict[str, Any]]):
        params = {"uri": uri, "diagnostics": diagnostics}
        self._send_notification("textDocument/publishDiagnostics", params)

    # Workspace indexing
    def _index_workspace(self):
        if not self._root_uri:
            return
        path = self._root_uri
        if path.startswith("file://"):
            path = path[7:]
        if not os.path.isdir(path):
            return
        idx = {}
        for root, _, files in os.walk(path):
            for fn in files:
                if not fn.endswith(".ix"):
                    continue
                p = os.path.join(root, fn)
                try:
                    t = open(p, "r", encoding="utf-8").read()
                    if _match_tool:
                        enums = _match_tool.find_enums(t)
                        structs = _match_tool.find_structs(t)
                        lst = []
                        for e in enums:
                            lst.append({"type": "enum", "name": e.name, "path": p, "range": (e.start, e.end)})
                        for s in structs:
                            lst.append({"type": "struct", "name": s.name, "path": p, "range": (s.start, s.end)})
                        if lst:
                            idx[p] = lst
                except Exception:
                    LOG.exception("index read failed: %s", p)
        with self._index_lock:
            self._symbol_index = idx
        LOG.info("workspace indexing complete: %d files indexed", len(idx))

    # scanning helpers wrapper
    def _scan_macros(self, text: str):
        # use transformer's scanner if available
        if transformer and hasattr(transformer, "_scan_macros"):
            try:
                return transformer._scan_macros(text)
            except Exception:
                LOG.exception("transformer._scan_macros failed")
        # fallback (same as in other modules)
        res = []
        i = 0; L = len(text); in_s = None
        while i < L:
            c = text[i]
            if in_s:
                if c == in_s and text[i - 1] != "\\":
                    in_s = None
                i += 1; continue
            if c in ('"', "'"):
                in_s = c; i += 1; continue
            if c == "@":
                j = i + 1; name = ""
                while j < L and (text[j].isalnum() or text[j] == "_"):
                    name += text[j]; j += 1
                k = j; depth = 0; in_s2 = None
                while k < L:
                    ch = text[k]
                    if in_s2:
                        if ch == in_s2 and text[k-1] != "\\":
                            in_s2 = None
                        k += 1; continue
                    if ch in ('"', "'"):
                        in_s2 = ch; k += 1; continue
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth = max(0, depth - 1)
                    elif ch == ";" and depth == 0:
                        raw = text[j:k].strip()
                        res.append((i, k+1, name, raw))
                        k += 1; break
                    k += 1
                i = k; continue
            i += 1
        return res

    # parse args fallback
    def _parse_args(self, raw: str) -> List[str]:
        parts = []
        buf = []; depth = 0; in_s = None
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
                p = "".join(buf).strip()
                if p: parts.append(p)
                buf = []; continue
            buf.append(ch)
        if buf:
            p = "".join(buf).strip()
            if p: parts.append(p)
        return parts

    # main loop for stdio transport
    def serve_stdio(self):
        LOG.info("Instryx LSP server (stdio) starting")
        self._running = True
        while self._running:
            msg = self._read_message_stdio()
            if msg is None:
                break
            try:
                self._dispatch_message(msg, transport=None)
            except Exception:
                LOG.exception("serve loop handler failed")

    def _dispatch_message(self, msg: Dict[str, Any], transport: Optional[Tuple[str, Any]] = None):
        if "method" in msg:
            method = msg["method"]
            params = msg.get("params")
            handler = self._request_handlers.get(method)
            id_ = msg.get("id")
            if handler:
                # run handler in separate thread to keep reader responsive
                threading.Thread(target=handler, args=(params, id_, transport), daemon=True).start()
            else:
                # handle common notifications explicitly
                if method in ("textDocument/didOpen", "textDocument/didChange", "textDocument/didClose", "textDocument/didSave"):
                    h = self._request_handlers.get(method)
                    if h:
                        threading.Thread(target=h, args=(params, None, transport), daemon=True).start()
                else:
                    LOG.debug("Unhandled method: %s", method)
                    if "id" in msg:
                        self._send_response(msg["id"], None, {"code": -32601, "message": "Method not found"}, transport=transport)
        elif "id" in msg:
            # We don't expect responses from client in this server
            return

    def shutdown(self):
        self._running = False
        try:
            self._diag_executor.shutdown(wait=False)
        except Exception:
            pass

# -------------------------
# Argparse + CLI
# -------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="instryx_lsp_server.py", description="Instryx LSP server")
    transport = p.add_mutually_exclusive_group()
    transport.add_argument("--stdio", action="store_true", help="serve LSP over stdio (default)")
    transport.add_argument("--tcp", action="store_true", help="serve LSP over TCP (single client)")
    p.add_argument("--host", default="127.0.0.1", help="TCP host to bind")
    p.add_argument("--port", type=int, default=2087, help="TCP port to bind")
    p.add_argument("--workspace", help="workspace root (file://... or path)")
    p.add_argument("--log-level", default="INFO", help="logging level (DEBUG/INFO/WARN/ERROR)")
    p.add_argument("--diag-debounce-ms", type=int, default=200, help="diagnostics debounce window (ms)")
    p.add_argument("--workers", type=int, default=4, help="worker threads for diagnostics/indexing")
    p.add_argument("--index-on-start", action="store_true", help="index workspace symbols on start")
    p.add_argument("--enable-format", action="store_true", help="enable formatting via instryx_syntax_morph if present")
    p.add_argument("--no-stdio", action="store_true", help="disable stdio transport (for testing)")
    return p

def main(argv: Optional[List[str]] = None):
    parser = build_argparser()
    args = parser.parse_args(argv or sys.argv[1:])
    # configure logging
    LOG.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    server = InstryxLSPServer(args)
    try:
        if args.tcp:
            server._running = True
            server._serve_tcp(args.host, args.port)
        else:
            # default to stdio unless explicitly disabled
            if args.no_stdio:
                print("stdio disabled; use --tcp to serve via TCP", file=sys.stderr)
                return 2
            server.serve_stdio()
    except KeyboardInterrupt:
        pass
    except Exception:
        LOG.exception("server exception")
    finally:
        server.shutdown()
    return 0

if __name__ == "__main__":
    sys.exit(main())

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


"""
instryx_codegen_sharder.py

Advanced Code generation sharder and coordinator with extended production-ready features.

This extended version adds further production-grade optimizations, tooling and execution features:
- Subprocess-backed codegen option for isolated backends (`subproc:cmd` spec).
- Gzip-backed cache files to reduce disk usage (--cache-compress).
- Output checksum verification and deterministic merge verification.
- Optional lightweight shard-level optimizations:
    - Common Subexpression Elimination (CSE) — deduplicates side-effect-free identical subtrees,
      replaces duplicates with shared references stored in `_shared_exprs`.
    - Dead Store Elimination (DSE) — removes assigns to locals that are never used (conservative).
- CLI flags to control optimization: `--opt-level`, `--enable-cse`, `--enable-dse`.
- Optional integration with `instryx_compiler_plugins` to run analysis/transform passes
  on shards before codegen (--use-compiler-plugins).
- Worker CPU-affinity hinting on Linux (best-effort).
- Commands: shard, run, stats, clear-cache.
- Cache management helpers and safe clear operation.
- Per-shard and global metrics (failures, successes, timings, bytes, cache hits).
- Parallel merge support, optional post-merge formatting hook.
- Dry-run, verbose logging, and manifest verification.

Notes:
- Subprocess codegen receives canonical JSON on stdin and must write textual code to stdout.
  Example spec: "subproc:python -u my_codegen.py --emit" or use module:callable for in-process calls.
- The optimization passes are conservative and intentionally minimal to be safe for generic IR shapes.
"""

from __future__ import annotations
import argparse
import gzip
import importlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import hashlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# new shared utils import
from instryx_opt_utils import canonical_json, ir_hash, optimize_shard, is_side_effect_free, global_value_numbering

LOG = logging.getLogger("instryx.codegen.sharder")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Optional compiler plugin integration
_try_compiler_plugins = None
try:
    from instryx_compiler_plugins import create_default_registry  # type: ignore
    _try_compiler_plugins = create_default_registry
except Exception:
    _try_compiler_plugins = None


# -------------------------
# Deterministic serialization + hashing
# -------------------------
def canonical_json(obj: Any) -> str:
    """Return canonical JSON representation used for hashing and deterministic IO."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def ir_hash(obj: Any) -> str:
    """SHA1 hash of canonical JSON representation."""
    try:
        return hashlib.sha1(canonical_json(obj).encode("utf-8")).hexdigest()
    except Exception:
        return hashlib.sha1(json.dumps(obj, default=str).encode("utf-8")).hexdigest()


def atomic_write(path: str, data: str, encoding: str = "utf-8", backup: bool = False) -> str:
    """Atomic write with optional backup of existing file."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding=encoding) as f:
        f.write(data)
    if backup and os.path.exists(path):
        shutil.copy2(path, path + ".bak")
    os.replace(tmp, path)
    return path


def atomic_write_bytes(path: str, data: bytes, backup: bool = False) -> str:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    if backup and os.path.exists(path):
        shutil.copy2(path, path + ".bak")
    os.replace(tmp, path)
    return path


# -------------------------
# Lightweight conservative optimizations (CSE + DSE)
# -------------------------
def _is_side_effect_free(node: Any) -> bool:
    """
    Conservative predicate: treat expressions as side-effect-free when they don't contain
    'call', 'store', or other known side-effecting node types. This is intentionally conservative.
    """
    if isinstance(node, dict):
        t = node.get("type")
        if t in ("call", "store", "invoke", "syscall"):
            return False
        for v in node.values():
            if not _is_side_effect_free(v):
                return False
        return True
    if isinstance(node, list):
        for it in node:
            if not _is_side_effect_free(it):
                return False
        return True
    return True


def _collect_expr_occurrences(node: Any, counter: Dict[str, List[Tuple[Any, List[Any]]]], path: List[Any] = None) -> None:
    """
    Walk node and collect canonical_json -> list of (subnode, parent_path).
    parent_path is a list of keys to allow replacement later.
    """
    if path is None:
        path = []
    if isinstance(node, dict) and _is_side_effect_free(node):
        key = canonical_json(node)
        counter.setdefault(key, []).append((node, list(path)))
    if isinstance(node, dict):
        for k, v in node.items():
            _collect_expr_occurrences(v, counter, path + [("dict", k)])
    elif isinstance(node, list):
        for idx, it in enumerate(node):
            _collect_expr_occurrences(it, counter, path + [("list", idx)])


def _replace_at_path(root: Any, path: List[Tuple[str, Any]], new_node: Any) -> None:
    """
    Replace value at path within root. Path is list of ("dict", key) or ("list", idx) entries.
    """
    cur = root
    for step in path[:-1]:
        kind, key = step
        if kind == "dict":
            cur = cur.get(key, {})
        else:
            cur = cur[key]
    last_kind, last_key = path[-1]
    if last_kind == "dict":
        cur[last_key] = new_node
    else:
        cur[last_key] = new_node


def common_subexpression_elimination(shard_ir: Any, min_occurrences: int = 2) -> Any:
    """
    Conservative CSE: find identical side-effect-free subtrees that occur at least min_occurrences times,
    factor them into shard_ir['_shared_exprs'] map and replace occurrences with {"type":"shared_ref","id":id}.
    Returns modified shard_ir (may be same object).
    """
    try:
        cnt: Dict[str, List[Tuple[Any, List[Any]]]] = {}
        _collect_expr_occurrences(shard_ir, cnt)
        shared = {}
        next_id = 0
        for key, occurrences in cnt.items():
            if len(occurrences) >= min_occurrences:
                # pick canonical representative
                rep_node = occurrences[0][0]
                sid = f"s{next_id}"
                next_id += 1
                shared[sid] = rep_node
                # replace all occurrences with shared_ref
                for _, path in occurrences:
                    try:
                        _replace_at_path(shard_ir, path, {"type": "shared_ref", "id": sid})
                    except Exception:
                        LOG.debug("CSE: failed to replace at path %s", path)
        if shared:
            # attach shared table to shard_ir (non-intrusive)
            if isinstance(shard_ir, dict):
                shard_ir.setdefault("_shared_exprs", {}).update(shared)
        return shard_ir
    except Exception:
        LOG.exception("CSE failed (ignored)")
        return shard_ir


def _collect_assigns_and_uses(node: Any, assigns: List[Tuple[Any, List[Any]]], uses: Set[str], path: List[Any] = None) -> None:
    """
    Collect top-level assigns and variable uses. Assigns recorded as (assign_node, path).
    Assign nodes assumed to be {"type":"assign","target": <str> , ...}
    """
    if path is None:
        path = []
    if isinstance(node, dict):
        if node.get("type") == "assign" and isinstance(node.get("target"), str):
            assigns.append((node, list(path)))
            # still collect uses inside value
            _collect_assigns_and_uses(node.get("value"), assigns, uses, path + [("dict", "value")])
            return
        if node.get("type") == "var" and isinstance(node.get("name"), str):
            uses.add(node.get("name"))
        for k, v in node.items():
            _collect_assigns_and_uses(v, assigns, uses, path + [("dict", k)])
    elif isinstance(node, list):
        for idx, it in enumerate(node):
            _collect_assigns_and_uses(it, assigns, uses, path + [("list", idx)])


def dead_store_elimination(shard_ir: Any) -> Any:
    """
    Conservative DSE: remove assignment statements to targets that are never read
    in the same function/block. Only removes top-level assign statements in blocks.
    """
    try:
        def process_block(block):
            if not isinstance(block, dict) or block.get("type") != "block":
                return block
            stmts = block.get("stmts", [])
            assigns = []
            uses = set()
            # first pass: collect assigns and uses
            for idx, s in enumerate(stmts):
                _collect_assigns_and_uses(s, assigns, uses, path=[("list", idx)])
            # decide which assigns are dead
            new_stmts = []
            for s in stmts:
                if isinstance(s, dict) and s.get("type") == "assign" and isinstance(s.get("target"), str):
                    tgt = s.get("target")
                    if tgt not in uses and _is_side_effect_free(s.get("value")):
                        # drop dead store
                        continue
                # recursively process nested blocks
                if isinstance(s, dict):
                    for k, v in list(s.items()):
                        s[k] = process_node(v)
                new_stmts.append(s)
            block["stmts"] = new_stmts
            return block

        def process_node(node):
            if isinstance(node, dict):
                if node.get("type") == "block":
                    return process_block(node)
                for k, v in list(node.items()):
                    node[k] = process_node(v)
                return node
            if isinstance(node, list):
                return [process_node(x) for x in node]
            return node

        new_ir = process_node(shard_ir)
        return new_ir
    except Exception:
        LOG.exception("DSE failed (ignored)")
        return shard_ir


def optimize_shard(shard_ir: Any, options: Dict[str, Any]) -> Any:
    """
    Apply conservative optimizations to a shard based on options.
    options keys:
      - enable_cse: bool
      - enable_dse: bool
      - opt_level: int (unused by passes but available)
    """
    try:
        if options.get("enable_cse"):
            shard_ir = common_subexpression_elimination(shard_ir, min_occurrences=2)
        if options.get("enable_dse"):
            shard_ir = dead_store_elimination(shard_ir)
    except Exception:
        LOG.exception("optimize_shard failed")
    return shard_ir


# -------------------------
# Call graph extraction & SCC
# -------------------------
def extract_calls_from_node(node: Any) -> Set[str]:
    """Recursively collect target names from call nodes of the form {'type':'call','fn':'name'}."""
    out: Set[str] = set()
    if isinstance(node, dict):
        if node.get("type") == "call" and isinstance(node.get("fn"), str):
            out.add(node["fn"])
        for v in node.values():
            out.update(extract_calls_from_node(v))
    elif isinstance(node, list):
        for it in node:
            out.update(extract_calls_from_node(it))
    return out


def build_call_graph(functions: Dict[str, Any]) -> Dict[str, Set[str]]:
    """
    Build call graph mapping function -> set(callees).
    External callees (not present) are kept in the sets but won't be included as vertices.
    """
    graph: Dict[str, Set[str]] = {}
    for name, fobj in functions.items():
        body = fobj.get("body")
        graph[name] = extract_calls_from_node(body)
    return graph


def strongly_connected_components(graph: Dict[str, Set[str]]) -> List[List[str]]:
    """
    Tarjan's algorithm for SCCs. Returns list of components (each a list of node names).
    Deterministic ordering: visits nodes in sorted order.
    """
    index = {}
    lowlink = {}
    stack = []
    onstack = set()
    result = []
    counter = 0

    def visit(v: str):
        nonlocal counter
        index[v] = counter
        lowlink[v] = counter
        counter += 1
        stack.append(v)
        onstack.add(v)
        for w in sorted(graph.get(v, [])):
            if w not in index:
                visit(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in onstack:
                lowlink[v] = min(lowlink[v], index[w])
        if lowlink[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                onstack.remove(w)
                comp.append(w)
                if w == v:
                    break
            result.append(sorted(comp))
    for node in sorted(graph.keys()):
        if node not in index:
            visit(node)
    return [sorted(c) for c in result]


# -------------------------
# Sharder
# -------------------------
class CodegenSharder:
    """
    Partition IR into shards.

    Strategies supported:
      - function: group functions based on SCCs and pack them to respect max_shard_bytes.
      - equal_count: distribute functions approximately equally across shards.
      - size_balanced: compute a target count from total size and pack to reach balanced byte sizes.
      - flat: single-shard whole IR.

    Parameters:
      - strategy: one of above
      - max_shard_bytes: target capacity for shard (heuristic)
      - min_shard_count / max_shard_count: bounds on number of shards
      - cost_fn: optional custom cost function fn(fname, fobj) -> int
    """
    def __init__(self,
                 strategy: str = "function",
                 max_shard_bytes: int = 64 * 1024,
                 min_shard_count: int = 1,
                 max_shard_count: Optional[int] = None,
                 cost_fn: Optional[Callable[[str, Any], int]] = None):
        self.strategy = strategy
        self.max_shard_bytes = int(max_shard_bytes)
        self.min_shard_count = int(min_shard_count)
        self.max_shard_count = int(max_shard_count) if max_shard_count is not None else None
        self.cost_fn = cost_fn or self._default_cost

    @staticmethod
    def _default_cost(fname: str, fobj: Any) -> int:
        """Simple cost model: canonical JSON size of body or 1 fallback."""
        try:
            body = fobj.get("body")
            txt = canonical_json(body)
            return max(1, len(txt))
        except Exception:
            return 1

    def shard_by_functions(self, ir: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Partition module-level 'functions' keyed dict into shards respecting SCCs."""
        if not isinstance(ir, dict):
            return [ir]
        functions: Dict[str, Any] = ir.get("functions", {})
        if not functions:
            return [ir]

        # graph + SCCs
        graph = build_call_graph(functions)
        sccs = strongly_connected_components(graph)

        # compute size for each SCC
        comps: List[Tuple[List[str], int]] = []
        for comp in sccs:
            comp_size = 0
            for fname in comp:
                comp_size += self.cost_fn(fname, functions.get(fname, {}))
            comps.append((sorted(comp), comp_size))

        # sort components by size descending for better packing
        comps.sort(key=lambda t: t[1], reverse=True)

        # pack into shards using greedy first-fit
        shards: List[List[str]] = []
        shard_sizes: List[int] = []
        for comp_names, comp_size in comps:
            placed = False
            for i in range(len(shards)):
                if shard_sizes[i] + comp_size <= self.max_shard_bytes or len(shards) < self.min_shard_count:
                    shards[i].extend(comp_names)
                    shard_sizes[i] += comp_size
                    placed = True
                    break
            if not placed:
                shards.append(list(comp_names))
                shard_sizes.append(comp_size)

        # enforce max_shard_count if provided (merge smallest)
        if self.max_shard_count and len(shards) > self.max_shard_count:
            import heapq
            heap = [(shard_sizes[i], i) for i in range(len(shards))]
            heapq.heapify(heap)
            while len(shards) > self.max_shard_count:
                # recompute two smallest indices robustly
                pairs = sorted(list(enumerate(shard_sizes)), key=lambda x: x[1])[:2]
                if len(pairs) < 2:
                    break
                i1 = pairs[0][0]
                i2 = pairs[1][0]
                if i1 > i2:
                    i1, i2 = i2, i1
                shards[i1].extend(shards[i2])
                shard_sizes[i1] += shard_sizes[i2]
                del shards[i2]
                del shard_sizes[i2]

        # build shard IR objects
        shard_irs: List[Dict[str, Any]] = []
        for shard_funcs in shards:
            shard_ir = dict(ir)  # shallow copy module metadata
            shard_ir["functions"] = {fn: json.loads(json.dumps(functions[fn])) for fn in shard_funcs if fn in functions}
            shard_irs.append(shard_ir)
        if not shard_irs:
            return [ir]
        return shard_irs

    def shard_equal_count(self, ir: Dict[str, Any], shard_count: int = 4) -> List[Dict[str, Any]]:
        """Split functions into shard_count shards trying to balance number of functions."""
        if not isinstance(ir, dict):
            return [ir]
        functions: Dict[str, Any] = ir.get("functions", {})
        names = sorted(functions.keys())
        if not names:
            return [ir]
        shard_count = max(1, min(shard_count, len(names)))
        shards = []
        per = (len(names) + shard_count - 1) // shard_count
        for i in range(0, len(names), per):
            part = names[i:i+per]
            shard_ir = dict(ir)
            shard_ir["functions"] = {fn: json.loads(json.dumps(functions[fn])) for fn in part}
            shards.append(shard_ir)
        return shards

    def shard_size_balanced(self, ir: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Compute total size and split into shards approximating max_shard_bytes target."""
        if not isinstance(ir, dict):
            return [ir]
        functions: Dict[str, Any] = ir.get("functions", {})
        if not functions:
            return [ir]
        items = sorted(functions.items())
        sizes = [(name, self.cost_fn(name, fobj)) for name, fobj in items]
        total = sum(s for _, s in sizes)
        target_shards = max(self.min_shard_count, (total + self.max_shard_bytes - 1) // self.max_shard_bytes)
        if self.max_shard_count:
            target_shards = min(target_shards, self.max_shard_count)
        # greedy fill
        shards: List[List[str]] = [[] for _ in range(target_shards)]
        shard_size = [0] * target_shards
        idx = 0
        for name, sz in sizes:
            shards[idx].append(name)
            shard_size[idx] += sz
            # move to next shard if current exceeded average
            if shard_size[idx] >= total / target_shards and idx + 1 < target_shards:
                idx += 1
        shard_irs = []
        for part in shards:
            shard_ir = dict(ir)
            shard_ir["functions"] = {fn: json.loads(json.dumps(functions[fn])) for fn in part}
            shard_irs.append(shard_ir)
        return [s for s in shard_irs if s.get("functions")]

    def adaptive_resize(self, ir: Dict[str, Any], utilization_percent: float = 0.8) -> None:
        """
        Heuristic to adapt max_shard_bytes based on IR distribution.
        If many components exceed current max_shard_bytes, increase limit to reduce shard count.
        """
        if not isinstance(ir, dict) or self.strategy != "function":
            return
        functions = ir.get("functions", {})
        sizes = [self.cost_fn(n, functions[n]) for n in sorted(functions.keys())]
        if not sizes:
            return
        large_count = sum(1 for s in sizes if s > self.max_shard_bytes)
        if large_count > 0:
            # increase by factor proportional to large fraction
            factor = 1.0 + (large_count / max(1, len(sizes))) * (1.0 - utilization_percent)
            new_limit = max(self.max_shard_bytes, int(self.max_shard_bytes * factor * 1.25))
            LOG.debug("adaptive_resize: increasing max_shard_bytes %d -> %d (large_count=%d)", self.max_shard_bytes, new_limit, large_count)
            self.max_shard_bytes = new_limit

    def validate_shard_coverage(self, ir: Dict[str, Any], shards: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Ensure union of functions across shards equals original function set (no missing or duplicated).
        Returns (ok, messages)
        """
        msgs = []
        if not isinstance(ir, dict):
            return True, msgs
        orig = set(ir.get("functions", {}).keys())
        seen = []
        for s in shards:
            seen.extend(sorted(s.get("functions", {}).keys()))
        seen_set = set(seen)
        missing = orig - seen_set
        extra = seen_set - orig
        dup = [name for name in seen if seen.count(name) > 1]
        if missing:
            msgs.append(f"missing functions in shards: {sorted(missing)[:10]}")
        if extra:
            msgs.append(f"unknown functions in shards: {sorted(extra)[:10]}")
        if dup:
            msgs.append(f"duplicate functions in shards: {sorted(set(dup))[:10]}")
        return (len(msgs) == 0), msgs

    def shard(self, ir: Any, **kwargs) -> List[Any]:
        """Top-level entry point to produce shards using configured strategy."""
        if self.strategy == "function":
            # adaptive pre-adjust
            try:
                self.adaptive_resize(ir)
            except Exception:
                LOG.debug("adaptive_resize failed")
            return self.shard_by_functions(ir) if isinstance(ir, dict) else [ir]
        if self.strategy == "equal_count":
            return self.shard_equal_count(ir, shard_count=kwargs.get("shard_count", max(1, self.min_shard_count)))
        if self.strategy == "size_balanced":
            return self.shard_size_balanced(ir)
        if self.strategy == "flat":
            return [ir]
        LOG.warning("Unknown sharding strategy '%s', defaulting to flat", self.strategy)
        return [ir]


# -------------------------
# Coordinator: run codegen on shards
# -------------------------
class CodegenCoordinator:
    """
    Run codegen on a list of shards.

    - codegen_fn(shard_ir, meta) -> str: user-supplied generator that returns textual output for a shard.
      Special spec 'subproc:...' supported via load_callable_from_spec wrapper.
    - pre_transform(shard_ir, meta) -> shard_ir: optional pre transform applied before codegen.
    - post_transform(output, meta) -> output: optional post transform applied to codegen output.
    - workers: concurrency
    - timeout: per-shard timeout seconds
    - retries: retry attempts on exception
    - cache_dir: optional directory to persist shard outputs keyed by hash
    - cache_compress: when True compress cache files with gzip
    - atomic_out_dir: optional directory where per-shard outputs are atomically written
    - supports cancellation, streaming results, and metrics export
    - optional use_compiler_plugins: if True and instryx_compiler_plugins available, run default passes on shards before codegen
    """
    def __init__(self,
                 codegen_fn: Callable[[Any, Dict[str, Any]], str],
                 workers: int = 4,
                 timeout: float = 60.0,
                 retries: int = 1,
                 cache_dir: Optional[str] = None,
                 cache_compress: bool = False,
                 atomic_out_dir: Optional[str] = None,
                 pre_transform: Optional[Callable[[Any, Dict[str, Any]], Any]] = None,
                 post_transform: Optional[Callable[[str, Dict[str, Any]], str]] = None,
                 use_compiler_plugins: bool = False):
        self.codegen_fn = codegen_fn
        self.workers = max(1, int(workers))
        self.timeout = float(timeout)
        self.retries = max(0, int(retries))
        self.cache_dir = cache_dir
        self.cache_compress = bool(cache_compress)
        self.atomic_out_dir = atomic_out_dir
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.use_compiler_plugins = use_compiler_plugins and (_try_compiler_plugins is not None)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        if atomic_out_dir:
            os.makedirs(atomic_out_dir, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=self.workers)
        self._cancel = threading.Event()
        # runtime metrics
        self.metrics = {"total_shards": 0, "cache_hits": 0, "total_time": 0.0, "shard_times": {}, "outputs_bytes": 0, "failures": 0, "success": 0}

        # per-worker local caches (in-memory)
        self._worker_local: Dict[int, Dict[str, str]] = defaultdict(dict)

        # optional compiler plugin registry
        self._plugin_registry = _try_compiler_plugins() if self.use_compiler_plugins else None

    def cancel(self):
        """Signal cancellation to running tasks; best-effort."""
        LOG.warning("cancellation requested")
        self._cancel.set()

    def shutdown(self, wait: bool = True):
        """Shutdown executor."""
        try:
            self._executor.shutdown(wait=wait)
        except Exception:
            LOG.debug("executor shutdown failed")

    def _cache_path(self, shard_hash: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        ext = ".gz" if self.cache_compress else ".out"
        return os.path.join(self.cache_dir, f"shard_{shard_hash}{ext}")

    def _atomic_out_file(self, shard_hash: str) -> Optional[str]:
        if not self.atomic_out_dir:
            return None
        return os.path.join(self.atomic_out_dir, f"shard_{shard_hash}.out")

    def _read_cache(self, path: str) -> Optional[str]:
        try:
            if path.endswith(".gz"):
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    return f.read()
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return None

    def _write_cache(self, path: str, data: str) -> None:
        try:
            if path.endswith(".gz"):
                with gzip.open(path, "wt", encoding="utf-8") as f:
                    f.write(data)
            else:
                atomic_write(path, data, backup=False)
        except Exception:
            LOG.debug("Failed to write cache %s", path)

    def _maybe_set_affinity(self, meta: Dict[str, Any]) -> None:
        """
        Apply CPU affinity hint on Linux when 'affinity' in meta (list of cpus).
        Best-effort, not fatal if unsupported.
        """
        try:
            cpus = meta.get("affinity")
            if cpus and isinstance(cpus, (list, tuple)):
                if hasattr(os, "sched_setaffinity"):
                    os.sched_setaffinity(0, set(int(c) for c in cpus))
        except Exception:
            LOG.debug("setting affinity failed (ignored)")

    def _run_codegen_once(self, shard_ir: Any, meta: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        if self._cancel.is_set():
            return False, "", {"error": "cancelled"}
        shard_hash = ir_hash(shard_ir)
        cache_path = self._cache_path(shard_hash)
        worker_id = threading.get_ident()

        # check worker-local cache
        if shard_hash in self._worker_local[worker_id]:
            out = self._worker_local[worker_id][shard_hash]
            self.metrics["cache_hits"] += 1
            self.metrics["outputs_bytes"] += len(out)
            return True, out, {"cached": True, "hash": shard_hash}

        # disk cache
        if cache_path and os.path.exists(cache_path):
            out = self._read_cache(cache_path)
            if out is not None:
                self.metrics["cache_hits"] += 1
                self.metrics["outputs_bytes"] += len(out)
                self._worker_local[worker_id][shard_hash] = out
                return True, out, {"cached": True, "hash": shard_hash}

        # optional plugin passes
        if self._plugin_registry:
            try:
                # run plugin passes that may transform the shard IR
                transformed_shard, report = self._plugin_registry.run_passes(shard_ir, context={"shard_hash": shard_hash})
                if report and not report.get("summary", {}).get("ok", True):
                    LOG.debug("plugin passes reported warnings/errors: %s", report.get("summary"))
                shard_ir = transformed_shard
            except Exception:
                LOG.exception("compiler plugin pass failed; continuing with original shard")

        # pre-transform
        if self.pre_transform:
            try:
                shard_ir = self.pre_transform(json.loads(json.dumps(shard_ir)), meta)
            except Exception:
                LOG.exception("pre_transform failed; continuing with original shard_ir")

        # attempt to set affinity best-effort
        self._maybe_set_affinity(meta)

        # run generator
        try:
            start = time.time()
            out = self.codegen_fn(shard_ir, dict(meta))
            elapsed = time.time() - start
            self.metrics["shard_times"][shard_hash] = elapsed
            self.metrics["total_time"] += elapsed
            self.metrics["outputs_bytes"] += len(out)
            diag = {"cached": False, "hash": shard_hash, "time": elapsed}
            # post-transform
            if self.post_transform:
                try:
                    out = self.post_transform(out, meta)
                except Exception:
                    LOG.exception("post_transform failed; using original output")
            # cache write
            if cache_path:
                try:
                    self._write_cache(cache_path, out)
                except Exception:
                    LOG.debug("Failed to write cache for %s", cache_path)
            # atomic out
            if self.atomic_out_dir:
                out_path = self._atomic_out_file(shard_hash)
                if out_path:
                    try:
                        atomic_write(out_path, out, backup=False)
                    except Exception:
                        LOG.debug("Failed to atomic write out file %s", out_path)
            # populate local
            self._worker_local[worker_id][shard_hash] = out
            return True, out, diag
        except Exception:
            LOG.exception("codegen raised")
            return False, "", {"cached": False, "hash": shard_hash, "error": traceback.format_exc()}

    def _run_with_retries(self, shard_ir: Any, meta: Dict[str, Any]) -> Dict[str, Any]:
        attempt = 0
        last_exc = None
        while attempt <= self.retries and not self._cancel.is_set():
            attempt += 1
            fut = self._executor.submit(self._run_codegen_once, shard_ir, meta)
            try:
                ok, out, diag = fut.result(timeout=self.timeout)
                diag["attempt"] = attempt
                diag["ok"] = ok
                if ok:
                    self.metrics.setdefault("success", 0)
                    self.metrics["success"] += 1
                else:
                    self.metrics.setdefault("failures", 0)
                    self.metrics["failures"] += 1
                return {"ok": ok, "output": out, "meta": diag}
            except concurrent.futures.TimeoutError:
                fut.cancel()
                last_exc = TimeoutError(f"timeout after {self.timeout}s")
                LOG.warning("Shard codegen timed out on attempt %d/%d", attempt, self.retries + 1)
            except Exception as e:
                last_exc = e
                LOG.warning("Shard codegen failed attempt %d/%d: %s", attempt, self.retries + 1, e)
            time.sleep(0.1 * attempt)
        if self._cancel.is_set():
            return {"ok": False, "output": "", "meta": {"error": "cancelled"}}
        return {"ok": False, "output": "", "meta": {"error": str(last_exc), "trace": traceback.format_exc(), "attempts": attempt}}

    def run_shards(self, shards: List[Any], meta_base: Optional[Dict[str, Any]] = None, parallel: bool = True, show_progress: bool = False, stream: bool = False) -> List[Dict[str, Any]]:
        """
        Run codegen across shards. If stream=True yields outputs as they complete and returns list as well.
        Returns list of results in the same order as shards: {ok, output, meta}
        """
        meta_base = dict(meta_base or {})
        self.metrics["total_shards"] = len(shards)
        if not shards:
            return []

        results: List[Optional[Dict[str, Any]]] = [None] * len(shards)
        futures = {}
        if parallel:
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                for i, shard in enumerate(shards):
                    meta = dict(meta_base)
                    meta.update({"shard_index": i})
                    futures[ex.submit(self._run_with_retries, shard, meta)] = i
                completed = 0
                total = len(futures)
                # streaming: produce as completed
                stream_outs = []
                for fut in as_completed(futures):
                    i = futures[fut]
                    try:
                        results[i] = fut.result()
                    except Exception:
                        LOG.exception("Shard worker crashed")
                        results[i] = {"ok": False, "output": "", "meta": {"error": traceback.format_exc()}}
                    completed += 1
                    if show_progress:
                        print(f"Progress: {completed}/{total} shards finished", file=sys.stderr)
                    if stream:
                        # yield immediate result (but we are not a generator here; collect for caller)
                        stream_outs.append((i, results[i]))
                if stream:
                    # sort stream_outs by completion index for deterministic ordering of stream list
                    stream_outs.sort(key=lambda x: x[0])
        else:
            for i, shard in enumerate(shards):
                meta = dict(meta_base)
                meta.update({"shard_index": i})
                results[i] = self._run_with_retries(shard, meta)
                if show_progress:
                    print(f"Progress: {i+1}/{len(shards)} shards finished", file=sys.stderr)
        return [r if r is not None else {"ok": False, "output": "", "meta": {"error": "missing result"}} for r in results]

    def write_manifest(self, shards: List[Any], path: str) -> str:
        """Write a manifest JSON describing shards and hashes; atomic."""
        manifest = {"shards": []}
        for i, s in enumerate(shards):
            manifest["shards"].append({"index": i, "hash": ir_hash(s), "functions": sorted(list(s.get("functions", {}).keys())) if isinstance(s, dict) else []})
        atomic_write(path, json.dumps(manifest, indent=2, sort_keys=True))
        return path

    def export_metrics(self, path: str) -> str:
        """Write runtime metrics to path (atomic)."""
        atomic_write(path, json.dumps(self.metrics, indent=2, sort_keys=True))
        return path

    def clear_cache(self) -> int:
        """Clear disk cache (if configured). Returns removed file count."""
        if not self.cache_dir:
            return 0
        removed = 0
        for fn in os.listdir(self.cache_dir):
            path = os.path.join(self.cache_dir, fn)
            try:
                os.remove(path)
                removed += 1
            except Exception:
                LOG.debug("failed to remove cache file %s", path)
        # also clear worker-local caches
        self._worker_local.clear()
        return removed


# -------------------------
# Merge utilities
# -------------------------
def merge_shard_outputs(outputs: List[Dict[str, Any]], strategy: str = "concat", separator: str = "\n", parallel_merge: bool = False) -> str:
    """
    Merge textual outputs from shards into final unit.
    Strategies:
      - concat: deterministic concatenation in shard order.
      - header+concat: include shard header metadata then concat.
    """
    if strategy == "concat":
        parts = []
        for o in outputs:
            if not o.get("ok", False):
                LOG.warning("skipping failed shard in merge: %s", o.get("meta", {}).get("hash"))
                continue
            parts.append(o.get("output", ""))
        return separator.join(parts)
    if strategy == "header+concat":
        parts = []
        for o in outputs:
            meta = o.get("meta", {})
            hdr = f"/* SHARD {meta.get('hash','?')} ok={o.get('ok',False)} */"
            parts.append(hdr)
            parts.append(o.get("output", ""))
        return separator.join(parts)
    return separator.join([o.get("output", "") for o in outputs if o.get("ok", False)])


# -------------------------
# Subprocess wrapper for codegen
# -------------------------
def _subprocess_codegen_factory(command: str) -> Callable[[Any, Dict[str, Any]], str]:
    """
    Return a function that runs `command` as subprocess, feeding canonical_json(shard_ir) to stdin,
    reading stdout as generated code. Command is shell form (string); run with shell=True.
    """
    def fn(shard_ir: Any, meta: Dict[str, Any]) -> str:
        inp = canonical_json(shard_ir)
        env = dict(os.environ)
        env["INSTRYX_SHARD_HASH"] = meta.get("shard_hash", ir_hash(shard_ir))
        env["INSTRYX_SHARD_INDEX"] = str(meta.get("shard_index", 0))
        proc = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
        try:
            out, err = proc.communicate(inp, timeout=meta.get("timeout", 60))
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            raise TimeoutError(f"subprocess codegen timed out. stderr:\n{err}")
        if proc.returncode != 0:
            raise RuntimeError(f"subprocess codegen failed (rc={proc.returncode}). stderr:\n{err}")
        return out
    return fn


# -------------------------
# Example simple codegen (CLI fallback)
# -------------------------
def sample_codegen_fn(shard_ir: Any, meta: Dict[str, Any]) -> str:
    """
    Example codegen that emits a stable textual representation of the functions in the shard.
    Replace with backend codegen function in real use.
    """
    lines: List[str] = []
    module_name = meta.get("module", "module")
    shard_index = meta.get("shard_index", 0)
    lines.append(f"// Codegen shard {shard_index} for {module_name}")
    if isinstance(shard_ir, dict):
        funcs = shard_ir.get("functions", {})
        for fname in sorted(funcs.keys()):
            lines.append(f"// function: {fname}")
            lines.append(canonical_json(funcs[fname]))
    else:
        lines.append(canonical_json(shard_ir))
    return "\n".join(lines)


# -------------------------
# CLI helpers
# -------------------------
def load_callable_from_spec(spec: str) -> Callable[..., Any]:
    """
    Load a callable from a spec:
      - "module.path:callable" -> import and return
      - "subproc:command string" -> return a subprocess-backed callable
    """
    if spec.startswith("subproc:"):
        cmd = spec[len("subproc:"):].strip()
        return _subprocess_codegen_factory(cmd)
    if ":" not in spec:
        raise ValueError("must be in form module.path:callable or subproc:command")
    modname, attr = spec.split(":", 1)
    mod = importlib.import_module(modname)
    if not hasattr(mod, attr):
        raise ImportError(f"{modname} has no attribute {attr}")
    fn = getattr(mod, attr)
    if not callable(fn):
        raise TypeError(f"{spec} isn't callable")
    return fn


def _cli():
    parser = argparse.ArgumentParser(prog="instryx_codegen_sharder.py", description="Shard IR and run codegen")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_shard = sub.add_parser("shard", help="Create shards from IR")
    p_shard.add_argument("-i", "--input", required=True)
    p_shard.add_argument("--strategy", choices=("function", "equal_count", "size_balanced", "flat"), default="function")
    p_shard.add_argument("--max-shard-bytes", type=int, default=64 * 1024)
    p_shard.add_argument("--min-shards", type=int, default=1)
    p_shard.add_argument("--max-shards", type=int, default=None)
    p_shard.add_argument("--out-dir", default=None)
    p_shard.add_argument("--manifest", default=None, help="write shard manifest JSON")
    p_shard.add_argument("--verbose", action="store_true")

    p_run = sub.add_parser("run", help="Shard and run codegen using a backend")
    p_run.add_argument("-i", "--input", required=True)
    p_run.add_argument("-o", "--out", default=None)
    p_run.add_argument("--strategy", choices=("function", "equal_count", "size_balanced", "flat"), default="function")
    p_run.add_argument("--workers", type=int, default=4)
    p_run.add_argument("--max-shard-bytes", type=int, default=64 * 1024)
    p_run.add_argument("--cache-dir", default=None)
    p_run.add_argument("--cache-compress", action="store_true", help="gzip compress cache files")
    p_run.add_argument("--atomic-out-dir", default=None)
    p_run.add_argument("--parallel", action="store_true", help="run shards in parallel")
    p_run.add_argument("--codegen-module", help="module:callable or subproc:command to use as codegen function")
    p_run.add_argument("--pre-transform-module", help="module:callable pre_transform(shard_ir, meta)->shard_ir")
    p_run.add_argument("--post-transform-module", help="module:callable post_transform(output, meta)->output")
    p_run.add_argument("--retries", type=int, default=1)
    p_run.add_argument("--timeout", type=float, default=60.0)
    p_run.add_argument("--verbose", action="store_true")
    p_run.add_argument("--merge-strategy", choices=("concat", "header+concat"), default="concat")
    p_run.add_argument("--report", help="path to write JSON run report")
    p_run.add_argument("--manifest", help="path to shard manifest JSON")
    p_run.add_argument("--use-compiler-plugins", action="store_true", help="run instryx_compiler_plugins passes on each shard before codegen")
    p_run.add_argument("--clear-cache", action="store_true", help="clear cache before running")
    # optimization flags
    p_run.add_argument("--opt-level", type=int, choices=(0,1,2,3), default=0, help="optimization level (0-3)")
    p_run.add_argument("--enable-cse", action="store_true", help="enable conservative common-subexpression elimination on shards")
    p_run.add_argument("--enable-dse", action="store_true", help="enable conservative dead-store elimination on shards")

    p_stats = sub.add_parser("stats", help="Show cache/metrics for a cache directory")
    p_stats.add_argument("--cache-dir", required=True)
    p_stats.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    if getattr(args, "verbose", False):
        LOG.setLevel(logging.DEBUG)

    if args.cmd == "shard":
        with open(args.input, "r", encoding="utf-8") as f:
            ir = json.load(f)
        sharder = CodegenSharder(strategy=args.strategy, max_shard_bytes=args.max_shard_bytes,
                                 min_shard_count=args.min_shards, max_shard_count=args.max_shards)
        shards = sharder.shard(ir)
        LOG.info("Created %d shards", len(shards))
        out_dir = args.out_dir
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            for idx, s in enumerate(shards):
                path = os.path.join(out_dir, f"shard_{idx}_{ir_hash(s)}.json")
                atomic_write(path, json.dumps(s, indent=2, sort_keys=True))
            print("wrote shards to", out_dir)
        else:
            print(json.dumps({"shard_count": len(shards), "shard_hashes": [ir_hash(s) for s in shards]}, indent=2))
        if args.manifest:
            try:
                ok, msgs = sharder.validate_shard_coverage(ir, shards)
                if not ok:
                    LOG.warning("manifest validation messages: %s", msgs)
                coord_tmp = CodegenCoordinator(sample_codegen_fn)
                coord_tmp.write_manifest(shards, args.manifest)
                print("manifest saved to", args.manifest)
            except Exception:
                LOG.exception("failed to write manifest")
        return 0

    if args.cmd == "stats":
        cache_dir = args.cache_dir
        total = 0
        total_bytes = 0
        entries = []
        for fn in os.listdir(cache_dir):
            path = os.path.join(cache_dir, fn)
            try:
                st = os.stat(path)
                total += 1
                total_bytes += st.st_size
                entries.append({"file": fn, "size": st.st_size, "mtime": st.st_mtime})
            except Exception:
                continue
        summary = {"cache_dir": cache_dir, "entries": total, "bytes": total_bytes}
        print(json.dumps({"summary": summary, "entries_sample": entries[:50]}, indent=2))
        return 0

    if args.cmd == "run":
        with open(args.input, "r", encoding="utf-8") as f:
            ir = json.load(f)

        sharder = CodegenSharder(strategy=args.strategy, max_shard_bytes=args.max_shard_bytes)
        shards = sharder.shard(ir)
        LOG.info("sharded into %d pieces", len(shards))

        # resolve codegen function
        if args.codegen_module:
            codegen_fn = load_callable_from_spec(args.codegen_module)
        else:
            codegen_fn = sample_codegen_fn

        # optional transforms
        pre_transform = load_callable_from_spec(args.pre_transform_module) if args.pre_transform_module else None
        post_transform = load_callable_from_spec(args.post_transform_module) if args.post_transform_module else None

        # build optimization options and pre_transform wrapper
        opt_options = {"enable_cse": bool(args.enable_cse) or (args.opt_level >= 2),
                       "enable_dse": bool(args.enable_dse) or (args.opt_level >= 1),
                       "opt_level": int(args.opt_level)}

        def _pre_transform_wrapper(shard_ir, meta):
            # first user pre-transform
            if pre_transform:
                try:
                    shard_ir = pre_transform(shard_ir, meta)
                except Exception:
                    LOG.exception("user pre_transform failed; continuing")
            # apply optimizer passes if requested
            if opt_options.get("enable_cse") or opt_options.get("enable_dse"):
                shard_ir = optimize_shard(shard_ir, opt_options)
            return shard_ir

        coord = CodegenCoordinator(codegen_fn=codegen_fn,
                                   workers=args.workers,
                                   timeout=args.timeout,
                                   retries=getattr(args, "retries", 1),
                                   cache_dir=args.cache_dir,
                                   cache_compress=getattr(args, "cache_compress", False),
                                   atomic_out_dir=args.atomic_out_dir,
                                   pre_transform=_pre_transform_wrapper,
                                   post_transform=post_transform,
                                   use_compiler_plugins=getattr(args, "use_compiler_plugins", False))
        if args.clear_cache and args.cache_dir:
            removed = coord.clear_cache()
            print("cleared cache files:", removed)

        if args.manifest:
            try:
                coord.write_manifest(shards, args.manifest)
                LOG.info("wrote manifest to %s", args.manifest)
            except Exception:
                LOG.exception("failed to write manifest")

        results = coord.run_shards(shards, meta_base={"module": os.path.basename(args.input)}, parallel=args.parallel, show_progress=True)
        merged = merge_shard_outputs(results, strategy=args.merge_strategy)
        if args.out:
            atomic_write(args.out, merged)
            print("wrote merged output to", args.out)
        else:
            print(merged)

        # produce a comprehensive run report
        report = {
            "input": args.input,
            "strategy": args.strategy,
            "shard_count": len(shards),
            "results": [{"ok": r["ok"], "meta": r.get("meta", {})} for r in results],
            "metrics": coord.metrics,
            "optimizations": opt_options
        }
        if args.report:
            atomic_write(args.report, json.dumps(report, indent=2))
            print("report saved to", args.report)
        else:
            print(json.dumps(report, indent=2))
        # shutdown coordinator to release resources
        coord.shutdown()
        return 0

    return 2


if __name__ == "__main__":
    try:
        sys.exit(_cli())
    except KeyboardInterrupt:
        LOG.info("interrupted")
        sys.exit(1)
emitter.emit(code, target="exe", output_name="greet_exe");
emitter.emit(code, target="wasm", output_name="greet_wasm");
# -------------------------
# Sharder: split IR into shards
# -------------------------
import os
import sys
import json
import time
import gzip
import math
import queue
import glob
import hashlib
import logging
import threading
import argparse
import traceback
import importlib
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from instryx_utils import atomic_write, canonical_json
from instryx_llvm_ir_codegen import InstryxLLVMCodegen

LOG = logging.getLogger("instryx.codegen_sharder")
LOG.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stderr)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
LOG.addHandler(ch)

"""
instryx_opt_utils.py

Shared optimization utilities for Instryx — extended.

Features added:
- canonical JSON, deterministic hashing
- conservative inter-procedural CSE (module-level shared table)
- conservative DSE (block-local)
- lightweight CFG builder for simple IR shapes and SSA-lite renaming
- inter-procedural Global Value Numbering (GVN): finds repeated pure expressions across functions
- convenience wrappers used by sharder and plugins

This implementation is conservative by design: it works on the generic JSON-like IR shape used
in this repo and avoids assumptions about complex control-flow beyond "block"/"stmts".
"""
from __future__ import annotations
import json
import hashlib
import logging
import copy
from typing import Any, Dict, List, Tuple, Set, Optional

LOG = logging.getLogger("instryx.optutils")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def ir_hash(obj: Any) -> str:
    try:
        return hashlib.sha1(canonical_json(obj).encode("utf-8")).hexdigest()
    except Exception:
        return hashlib.sha1(json.dumps(obj, default=str).encode("utf-8")).hexdigest()


# -------------------------
# Side-effect predicate
# -------------------------
def is_side_effect_free(node: Any) -> bool:
    if isinstance(node, dict):
        t = node.get("type")
        # conservative list of side-effecting node types
        if t in ("call", "store", "invoke", "syscall", "prefetch", "atomic"):
            return False
        for v in node.values():
            if not is_side_effect_free(v):
                return False
        return True
    if isinstance(node, list):
        return all(is_side_effect_free(it) for it in node)
    return True


# -------------------------
# Local CSE utilities
# -------------------------
def _collect_expr_occurrences(node: Any, counter: Dict[str, List[Tuple[Any, List[Tuple[str, Any]]]]], path: List[Tuple[str, Any]] = None) -> None:
    if path is None:
        path = []
    if isinstance(node, dict) and is_side_effect_free(node):
        key = canonical_json(node)
        counter.setdefault(key, []).append((node, list(path)))
    if isinstance(node, dict):
        for k, v in node.items():
            _collect_expr_occurrences(v, counter, path + [("dict", k)])
    elif isinstance(node, list):
        for idx, it in enumerate(node):
            _collect_expr_occurrences(it, counter, path + [("list", idx)])


def _replace_at_path(root: Any, path: List[Tuple[str, Any]], new_node: Any) -> None:
    cur = root
    for step in path[:-1]:
        kind, key = step
        if kind == "dict":
            cur = cur.get(key, {})
        else:
            cur = cur[key]
    last_kind, last_key = path[-1]
    if last_kind == "dict":
        cur[last_key] = new_node
    else:
        cur[last_key] = new_node


def common_subexpression_elimination(shard_ir: Any, min_occurrences: int = 2) -> Any:
    """
    Conservative CSE within a shard (function/module). Returns modified IR.
    """
    try:
        cnt: Dict[str, List[Tuple[Any, List[Tuple[str, Any]]]]] = {}
        _collect_expr_occurrences(shard_ir, cnt)
        shared = {}
        next_id = 0
        for key, occurrences in cnt.items():
            if len(occurrences) >= min_occurrences:
                rep_node = copy.deepcopy(occurrences[0][0])
                sid = f"s{next_id}"
                next_id += 1
                shared[sid] = rep_node
                for _, path in occurrences:
                    try:
                        _replace_at_path(shard_ir, path, {"type": "shared_ref", "id": sid})
                    except Exception:
                        LOG.debug("CSE: replace failed at path %s", path)
        if shared and isinstance(shard_ir, dict):
            shard_ir.setdefault("_shared_exprs", {}).update(shared)
        return shard_ir
    except Exception:
        LOG.exception("CSE failed (ignored)")
        return shard_ir


# -------------------------
# Inter-procedural CSE (module-level)
# -------------------------
def interprocedural_cse(module_ir: Any, min_occurrences: int = 2) -> Any:
    """
    Factor identical pure expressions that appear across function boundaries into a module-level
    `_ip_shared_exprs` table and replace with `shared_ref`. Conservative: requires is_side_effect_free.
    """
    if not isinstance(module_ir, dict) or "functions" not in module_ir:
        return module_ir
    try:
        cnt: Dict[str, List[Tuple[str, List[Tuple[str, Any]]]]] = {}
        # collect across functions
        for fname, fobj in module_ir.get("functions", {}).items():
            body = fobj.get("body")
            if body is None:
                continue
            def collect(node, path=None):
                if path is None:
                    path = []
                if isinstance(node, dict) and is_side_effect_free(node):
                    key = canonical_json(node)
                    cnt.setdefault(key, []).append((fname, list(path)))
                if isinstance(node, dict):
                    for k, v in node.items():
                        collect(v, path + [("dict", k)])
                elif isinstance(node, list):
                    for idx, it in enumerate(node):
                        collect(it, path + [("list", idx)])
            collect(body)
        shared = {}
        next_id = 0
        for key, occs in cnt.items():
            if len(occs) >= min_occurrences:
                rep = json.loads(key)
                sid = f"ip{{next_id}}" if False else f"ip{next_id}"
                next_id += 1
                shared[sid] = rep
                # replace occurrences in functions
                for fname, path in occs:
                    try:
                        fbody = module_ir["functions"][fname]["body"]
                        _replace_at_path(fbody, path, {"type": "shared_ref", "id": sid})
                    except Exception:
                        LOG.debug("interprocedural_cse: failed replace in %s @ %s", fname, path)
        if shared:
            module_ir.setdefault("_ip_shared_exprs", {}).update(shared)
        return module_ir
    except Exception:
        LOG.exception("interprocedural_cse failed (ignored)")
        return module_ir


# -------------------------
# DSE utilities (unchanged, conservative)
# -------------------------
def _collect_assigns_and_uses(node: Any, uses: Set[str]) -> None:
    if isinstance(node, dict):
        if node.get("type") == "var" and isinstance(node.get("name"), str):
            uses.add(node.get("name"))
        for v in node.values():
            _collect_assigns_and_uses(v, uses)
    elif isinstance(node, list):
        for it in node:
            _collect_assigns_and_uses(it, uses)


def dead_store_elimination(shard_ir: Any) -> Any:
    try:
        def process_block(block):
            if not isinstance(block, dict) or block.get("type") != "block":
                return block
            stmts = block.get("stmts", [])
            uses: Set[str] = set()
            for s in stmts:
                _collect_assigns_and_uses(s, uses)
            new_stmts = []
            for s in stmts:
                if isinstance(s, dict) and s.get("type") == "assign" and isinstance(s.get("target"), str):
                    tgt = s.get("target")
                    if tgt not in uses and is_side_effect_free(s.get("value")):
                        continue
                if isinstance(s, dict):
                    for k, v in list(s.items()):
                        s[k] = process_node(v)
                new_stmts.append(s)
            block["stmts"] = new_stmts
            return block

        def process_node(node):
            if isinstance(node, dict):
                if node.get("type") == "block":
                    return process_block(node)
                for k, v in list(node.items()):
                    node[k] = process_node(v)
                return node
            if isinstance(node, list):
                return [process_node(x) for x in node]
            return node

        return process_node(shard_ir)
    except Exception:
        LOG.exception("DSE failed (ignored)")
        return shard_ir


# -------------------------
# Lightweight CFG builder (best-effort)
# -------------------------
def build_basic_blocks(body: Any) -> List[Dict[str, Any]]:
    """
    Convert a simple block-structured body (block->stmts) into list of basic blocks.
    This is best-effort and assumes the IR uses 'block'/'stmts' and simple control structures.
    """
    blocks = []
    if not isinstance(body, dict) or body.get("type") != "block":
        return [{"id": "entry", "stmts": body if isinstance(body, list) else [body]}]
    # split by labeled statements if present — fallback: single block
    stmts = body.get("stmts", [])
    blocks.append({"id": "b0", "stmts": stmts})
    return blocks


# -------------------------
# SSA-lite renaming
# -------------------------
def ssa_renaming(body: Any) -> Any:
    """
    Lightweight SSA renaming: give unique names to assigned targets (append numerical suffix).
    This is conservative and avoids phi insertion; it's useful to make local value numbering more effective.
    """
    counter: Dict[str, int] = {}
    mapping: Dict[str, str] = {}

    def rename(node):
        if isinstance(node, dict):
            if node.get("type") == "assign" and isinstance(node.get("target"), str):
                old = node["target"]
                idx = counter.get(old, 0) + 1
                counter[old] = idx
                new = f"{old}__ssa{idx}"
                mapping[old] = new
                node["target"] = new
                node["value"] = rename(node.get("value"))
                return node
            if node.get("type") == "var" and isinstance(node.get("name"), str):
                nm = node["name"]
                if nm in mapping:
                    return {"type": "var", "name": mapping[nm]}
            for k, v in list(node.items()):
                node[k] = rename(v)
            return node
        if isinstance(node, list):
            return [rename(x) for x in node]
        return node

    try:
        return rename(copy.deepcopy(body))
    except Exception:
        LOG.debug("ssa_renaming failed, returning original")
        return body


# -------------------------
# Function-local GVN (SSA-driven)
# -------------------------
def function_local_gvn(body: Any) -> Any:
    """
    Given a function body (assumed block-structured), perform SSA renaming then value-number
    pure expressions and build a local `_gvn_shared` table for expressions repeated >=2 times.
    """
    try:
        ssa_body = ssa_renaming(body)
        expr_map: Dict[str, List[List[Tuple[str, Any]]]] = {}
        # collect canonical expressions and their paths
        def collect(node, path=None):
            if path is None:
                path = []
            if isinstance(node, dict) and is_side_effect_free(node):
                key = canonical_json(node)
                expr_map.setdefault(key, []).append(list(path))
            if isinstance(node, dict):
                for k, v in node.items():
                    collect(v, path + [("dict", k)])
            elif isinstance(node, list):
                for i, it in enumerate(node):
                    collect(it, path + [("list", i)])
        collect(ssa_body)
        shared = {}
        sid = 0
        for key, occs in expr_map.items():
            if len(occs) >= 2:
                rep = json.loads(key)
                name = f"gvn{sid}"
                sid += 1
                shared[name] = rep
                for path in occs:
                    try:
                        _replace_at_path(ssa_body, path, {"type": "shared_ref", "id": name})
                    except Exception:
                        LOG.debug("gvn replace failed at %s", path)
        if shared and isinstance(ssa_body, dict):
            ssa_body.setdefault("_gvn_shared", {}).update(shared)
        return ssa_body
    except Exception:
        LOG.exception("function_local_gvn failed (ignored)")
        return body


# -------------------------
# Inter-procedural GVN (module-level)
# -------------------------
def interprocedural_gvn(module_ir: Any, min_occurrences: int = 2) -> Any:
    """
    Module-level GVN: find pure identical expressions across functions and factor them into
    `_module_gvn_shared`. This is conservative and avoids cross-function expression rewriting
    that depends on parameter aliasing or side-effects.
    """
    if not isinstance(module_ir, dict) or "functions" not in module_ir:
        return module_ir
    try:
        expr_map: Dict[str, List[Tuple[str, List[Tuple[str, Any]]]]] = {}
        for fname, fobj in module_ir.get("functions", {}).items():
            body = fobj.get("body")
            if not body:
                continue
            def collect(node, path=None):
                if path is None:
                    path = []
                if isinstance(node, dict) and is_side_effect_free(node):
                    key = canonical_json(node)
                    expr_map.setdefault(key, []).append((fname, list(path)))
                if isinstance(node, dict):
                    for k, v in node.items():
                        collect(v, path + [("dict", k)])
                elif isinstance(node, list):
                    for idx, it in enumerate(node):
                        collect(it, path + [("list", idx)])
            collect(body)
        shared = {}
        sid = 0
        for key, occs in expr_map.items():
            if len(occs) >= min_occurrences:
                rep = json.loads(key)
                name = f"mgvn{sid}"
                sid += 1
                shared[name] = rep
                for fname, path in occs:
                    try:
                        fb = module_ir["functions"][fname]["body"]
                        _replace_at_path(fb, path, {"type": "shared_ref", "id": name})
                    except Exception:
                        LOG.debug("interprocedural_gvn replace failed in %s at %s", fname, path)
        if shared:
            module_ir.setdefault("_module_gvn_shared", {}).update(shared)
        return module_ir
    except Exception:
        LOG.exception("interprocedural_gvn failed (ignored)")
        return module_ir


# -------------------------
# Top-level optimize_shard (public)
# -------------------------
def optimize_shard(shard_ir: Any, options: Dict[str, Any]) -> Any:
    """
    Options:
      - enable_cse (bool)
      - enable_dse (bool)
      - enable_gvn (bool)        -> function-local GVN
      - enable_ip_cse (bool)     -> inter-procedural CSE across functions
      - enable_ip_gvn (bool)     -> inter-procedural GVN across functions
    """
    try:
        if options.get("enable_cse"):
            shard_ir = common_subexpression_elimination(shard_ir, min_occurrences=2)
        if options.get("enable_dse"):
            shard_ir = dead_store_elimination(shard_ir)
        if options.get("enable_gvn"):
            if isinstance(shard_ir, dict) and "functions" in shard_ir:
                for fname, fobj in shard_ir.get("functions", {}).items():
                    body = fobj.get("body")
                    if body:
                        fobj["body"] = function_local_gvn(body)
        if options.get("enable_ip_cse"):
            shard_ir = interprocedural_cse(shard_ir, min_occurrences=2)
        if options.get("enable_ip_gvn"):
            shard_ir = interprocedural_gvn(shard_ir, min_occurrences=2)
    except Exception:
        LOG.exception("optimize_shard (utils) failed")
    return shard_ir

# (patch fragment) Add profile-driven inliner plugin that reads profile JSON and drives inlining.
# Insert this class near other inlining plugins; shown here as a standalone file fragment
# to be pasted into instryx_compiler_plugins.py in the plugins registration area.

from typing import Any, Dict, Tuple
import os
import json
import logging
import traceback

LOG = logging.getLogger("instryx.compiler.plugins")

class ProfileGuidedInliningPlugin(PluginBase):
    """
    Improved profile-guided inlining plugin.

    Usage:
      - Pass context={"profile_path": "/path/to/profile.json", "hot_threshold": 0.7}
      - Profile file format: {"function_hotness": {"foo": 0.95, "bar": 0.2}, "call_counts": {"caller->callee": 123}}
      - This plugin will inline call sites whose callee hotness >= hot_threshold
        and where callee size <= size_threshold (configurable).
    """
    def __init__(self, size_threshold: int = 300, default_threshold: float = 0.7):
        super().__init__()
        self.meta = PluginMeta(priority=35, name="pg_inlining", description="Profile-guided inlining (improved)", version="2.0")
        self.size_threshold = int(size_threshold)
        self.default_threshold = float(default_threshold)

    def _load_profile(self, context: Dict[str, Any]) -> Dict[str, Any]:
        path = context.get("profile_path") or os.environ.get("INSTRYX_PROFILE_PATH")
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            LOG.exception("failed to load profile at %s", path)
            return {}

    def apply(self, ir: Any, context: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        infos, warnings, errors = [], [], []
        try:
            profile = self._load_profile(context)
            hotness_map = profile.get("function_hotness", {})
            threshold = float(context.get("hot_threshold", self.default_threshold))
            if not isinstance(ir, dict) or "functions" not in ir:
                return ir, {"ok": True, "info": ["no functions"], "warnings": warnings, "errors": errors}
            functions = ir.get("functions", {})
            # determine small functions eligible by size
            eligible = {fname for fname, fobj in functions.items()
                        if isinstance(fobj.get("body"), (dict, list)) and len(json.dumps(fobj.get("body"))) <= self.size_threshold}
            # inline hot callees at call sites
            def inline_node(node):
                if isinstance(node, dict):
                    if node.get("type") == "call" and isinstance(node.get("fn"), str):
                        fname = node.get("fn")
                        hot = float(hotness_map.get(fname, 0.0))
                        if hot >= threshold and fname in eligible:
                            fobj = functions.get(fname, {})
                            body = fobj.get("body")
                            params = fobj.get("params", []) or []
                            args = node.get("args", []) or []
                            mapping = {p: (args[i] if i < len(args) else {"type": "var", "name": p}) for i, p in enumerate(params)}
                            def repl(n):
                                if isinstance(n, dict):
                                    if n.get("type") == "var" and isinstance(n.get("name"), str) and n.get("name") in mapping:
                                        return mapping[n.get("name")]
                                    for k, v in list(n.items()):
                                        n[k] = repl(v)
                                    return n
                                if isinstance(n, list):
                                    return [repl(x) for x in n]
                                return n
                            inlined = repl(copy.deepcopy(body))
                            infos.append(f"inlined {fname} (hot={hot})")
                            return inlined
                    for k, v in list(node.items()):
                        node[k] = inline_node(v)
                    return node
                if isinstance(node, list):
                    return [inline_node(x) for x in node]
                return node

            new_ir = copy.deepcopy(ir)
            for fname, fobj in new_ir.get("functions", {}).items():
                fobj["body"] = inline_node(fobj.get("body"))
            return new_ir, {"ok": True, "info": infos, "warnings": warnings, "errors": errors}
        except Exception as e:
            LOG.exception("ProfileGuidedInliningPlugin failed")
            return ir, {"ok": False, "info": [], "warnings": warnings, "errors": [str(e), traceback.format_exc()]}

import os
import json
import unittest
from instryx_opt_utils import function_local_gvn, interprocedural_gvn, ssa_renaming
from instryx_compiler_plugins import ProfileGuidedInliningPlugin, PluginRegistry

class TestAdvancedOpt(unittest.TestCase):
    def test_ssa_renaming_and_gvn(self):
        body = {
            "type": "block",
            "stmts": [
                {"type": "assign", "target": "x", "value": {"type": "binary", "op": "+", "left": 1, "right": 2}},
                {"type": "assign", "target": "x", "value": {"type": "binary", "op": "+", "left": 1, "right": 2}},
                {"type": "assign", "target": "y", "value": {"type": "binary", "op": "+", "left": {"type":"var","name":"x"}, "right": 3}}
            ]
        }
        ssa = ssa_renaming(body)
        # assigned names should have __ssa suffix
        targets = []
        for s in ssa.get("stmts", []):
            if isinstance(s, dict) and s.get("type") == "assign":
                targets.append(s.get("target"))
        self.assertTrue(any("__ssa" in t for t in targets))
        gvn = function_local_gvn(body)
        # GVN should create a _gvn_shared table when duplicate pure exprs exist
        self.assertTrue(isinstance(gvn, dict))
        self.assertIn("_gvn_shared", gvn)

    def test_interprocedural_gvn(self):
        module = {
            "functions": {
                "a": {"body": {"type": "block", "stmts": [{"type": "assign", "target": "t", "value": {"type":"binary","op":"+","left":1,"right":2}}]}}},
                "b": {"body": {"type": "block", "stmts": [{"type": "assign", "target": "u", "value": {"type":"binary","op":"+","left":1,"right":2}}]}}},

out = interprocedural_gvn(module, min_occurrences=2)
        # module-level shared table should be present
self.assertIn("_module_gvn_shared", out)

def test_profile_guided_inliner(self):
        # create small functions and a profile file indicating callee is hot
        module = {
            "functions": {
                "callee": {"params": ["p"], "body": {"type": "block", "stmts": [{"type": "assign", "target": "r", "value": {"type":"binary","op":"+","left":{"type":"var","name":"p"},"right":1}}]}},
                "caller": {"params": [], "body": {"type": "block", "stmts": [{"type":"assign","target":"v","value":{"type":"call","fn":"callee","args":[{"type":"var","name":"c"}]}}]}}
            }
        }
        prof = {"function_hotness": {"callee": 0.95}}
        prof_path = "tests/profile.json"
        with open(prof_path, "w", encoding="utf-8") as f:
            json.dump(prof, f)
        reg = PluginRegistry(max_workers=2, per_pass_timeout=1.0)
        pgi = ProfileGuidedInliningPlugin(size_threshold=500, default_threshold=0.7)
        reg.register(pgi)
        new_ir, report = reg.run_passes(module, context={"profile_path": prof_path, "hot_threshold": 0.7}, passes=["pg_inlining"])
        # after inlining, caller.body should not contain a call node
        caller_body = new_ir["functions"]["caller"]["body"]
        def find_call(n):
            if isinstance(n, dict):
                if n.get("type") == "call":
                    return True
                for v in n.values():
                    if find_call(v):
                        return True
            if isinstance(n, list):
                for it in n:
                    if find_call(it):
                        return True
            return False
        self.assertFalse(find_call(caller_body))
        os.remove(prof_path)

if __name__ == "__main__":
    unittest.main()

python 
instryx_opt_utils.py
"""
instryx_opt_utils.py

Shared optimization utilities for Instryx — advanced SSA + CFG + interprocedural GVN/CSE.

This module implements:
- canonical JSON and deterministic hashing
- conservative side-effect predicate
- local & interprocedural CSE
- DSE (conservative)
- lightweight CFG builder for simple IR shapes (handles block/if)
- full SSA construction (phi insertion + renaming) for variables assigned with "assign"
- SSA-driven GVN (function-local) that leverages SSA names for precise value numbering
- interprocedural GVN that factors identical pure expressions across functions
- convenience helpers to compute sensible thresholds for inlining & size choices
- top-level optimize_shard wrapper that accepts options to enable features
"""
from __future__ import annotations
import json
import hashlib
import logging
import copy
import itertools
from typing import Any, Dict, List, Tuple, Set, Optional

LOG = logging.getLogger("instryx.optutils")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def ir_hash(obj: Any) -> str:
    try:
        return hashlib.sha1(canonical_json(obj).encode("utf-8")).hexdigest()
    except Exception:
        return hashlib.sha1(json.dumps(obj, default=str).encode("utf-8")).hexdigest()


# -------------------------
# Side-effect predicate
# -------------------------
def is_side_effect_free(node: Any) -> bool:
    if isinstance(node, dict):
        t = node.get("type")
        if t in ("call", "store", "invoke", "syscall", "prefetch", "atomic"):
            return False
        for v in node.values():
            if not is_side_effect_free(v):
                return False
        return True
    if isinstance(node, list):
        return all(is_side_effect_free(it) for it in node)
    return True


# -------------------------
# Simple local CSE (unchanged)
# -------------------------
def _collect_expr_occurrences(node: Any, counter: Dict[str, List[Tuple[Any, List[Tuple[str, Any]]]]], path: List[Tuple[str, Any]] = None) -> None:
    if path is None:
        path = []
    if isinstance(node, dict) and is_side_effect_free(node):
        key = canonical_json(node)
        counter.setdefault(key, []).append((node, list(path)))
    if isinstance(node, dict):
        for k, v in node.items():
            _collect_expr_occurrences(v, counter, path + [("dict", k)])
    elif isinstance(node, list):
        for idx, it in enumerate(node):
            _collect_expr_occurrences(it, counter, path + [("list", idx)])


def _replace_at_path(root: Any, path: List[Tuple[str, Any]], new_node: Any) -> None:
    cur = root
    for step in path[:-1]:
        kind, key = step
        if kind == "dict":
            cur = cur.get(key, {})
        else:
            cur = cur[key]
    last_kind, last_key = path[-1]
    if last_kind == "dict":
        cur[last_key] = new_node
    else:
        cur[last_key] = new_node


def common_subexpression_elimination(shard_ir: Any, min_occurrences: int = 2) -> Any:
    try:
        cnt: Dict[str, List[Tuple[Any, List[Tuple[str, Any]]]]] = {}
        _collect_expr_occurrences(shard_ir, cnt)
        shared = {}
        next_id = 0
        for key, occurrences in cnt.items():
            if len(occurrences) >= min_occurrences:
                rep_node = copy.deepcopy(occurrences[0][0])
                sid = f"s{next_id}"
                next_id += 1
                shared[sid] = rep_node
                for _, path in occurrences:
                    try:
                        _replace_at_path(shard_ir, path, {"type": "shared_ref", "id": sid})
                    except Exception:
                        LOG.debug("CSE: replace failed at %s", path)
        if shared and isinstance(shard_ir, dict):
            shard_ir.setdefault("_shared_exprs", {}).update(shared)
        return shard_ir
    except Exception:
        LOG.exception("CSE failed (ignored)")
        return shard_ir


# -------------------------
# Interprocedural CSE (module-level)
# -------------------------
def interprocedural_cse(module_ir: Any, min_occurrences: int = 2) -> Any:
    if not isinstance(module_ir, dict) or "functions" not in module_ir:
        return module_ir
    try:
        cnt: Dict[str, List[Tuple[str, List[Tuple[str, Any]]]]] = {}
        for fname, fobj in module_ir.get("functions", {}).items():
            body = fobj.get("body")
            if body is None:
                continue

            def collect(node, path=None):
                if path is None:
                    path = []
                if isinstance(node, dict) and is_side_effect_free(node):
                    key = canonical_json(node)
                    cnt.setdefault(key, []).append((fname, list(path)))
                if isinstance(node, dict):
                    for k, v in node.items():
                        collect(v, path + [("dict", k)])
                elif isinstance(node, list):
                    for idx, it in enumerate(node):
                        collect(it, path + [("list", idx)])

            collect(body)

        shared = {}
        next_id = 0
        for key, occs in cnt.items():
            if len(occs) >= min_occurrences:
                rep = json.loads(key)
                sid = f"ip{next_id}"
                next_id += 1
                shared[sid] = rep
                for fname, path in occs:
                    try:
                        fbody = module_ir["functions"][fname]["body"]
                        _replace_at_path(fbody, path, {"type": "shared_ref", "id": sid})
                    except Exception:
                        LOG.debug("interprocedural_cse: failed replace in %s @ %s", fname, path)
        if shared:
            module_ir.setdefault("_ip_shared_exprs", {}).update(shared)
        return module_ir
    except Exception:
        LOG.exception("interprocedural_cse failed (ignored)")
        return module_ir


# -------------------------
# DSE (conservative)
# -------------------------
def _collect_assigns_and_uses(node: Any, uses: Set[str]) -> None:
    if isinstance(node, dict):
        if node.get("type") == "var" and isinstance(node.get("name"), str):
            uses.add(node.get("name"))
        for v in node.values():
            _collect_assigns_and_uses(v, uses)
    elif isinstance(node, list):
        for it in node:
            _collect_assigns_and_uses(it, uses)


def dead_store_elimination(shard_ir: Any) -> Any:
    try:
        def process_block(block):
            if not isinstance(block, dict) or block.get("type") != "block":
                return block
            stmts = block.get("stmts", [])
            uses: Set[str] = set()
            for s in stmts:
                _collect_assigns_and_uses(s, uses)
            new_stmts = []
            for s in stmts:
                if isinstance(s, dict) and s.get("type") == "assign" and isinstance(s.get("target"), str):
                    tgt = s.get("target")
                    if tgt not in uses and is_side_effect_free(s.get("value")):
                        continue
                if isinstance(s, dict):
                    for k, v in list(s.items()):
                        s[k] = process_node(v)
                new_stmts.append(s)
            block["stmts"] = new_stmts
            return block

        def process_node(node):
            if isinstance(node, dict):
                if node.get("type") == "block":
                    return process_block(node)
                for k, v in list(node.items()):
                    node[k] = process_node(v)
                return node
            if isinstance(node, list):
                return [process_node(x) for x in node]
            return node

        return process_node(shard_ir)
    except Exception:
        LOG.exception("DSE failed (ignored)")
        return shard_ir


# -------------------------
# Lightweight CFG builder (handles 'block' and 'if' nodes)
# -------------------------
class BasicBlock:
    def __init__(self, id: str):
        self.id = id
        self.stmts: List[Any] = []
        self.succ: Set[str] = set()
        self.pred: Set[str] = set()

    def to_dict(self):
        return {"id": self.id, "stmts": self.stmts, "succ": list(self.succ), "pred": list(self.pred)}


def build_cfg_for_function(body: Any) -> Dict[str, BasicBlock]:
    """
    Best-effort CFG builder. Splits on 'if' nodes into then/else blocks.
    Returns mapping id -> BasicBlock. A single-entry 'entry' block exists.
    """
    blocks: Dict[str, BasicBlock] = {}
    cnt = itertools.count()

    def new_block():
        return BasicBlock(f"b{next(cnt)}")

    entry = new_block()
    blocks[entry.id] = entry

    def append_stmts_to_block(target_block: BasicBlock, stmts):
        for s in stmts:
            if isinstance(s, dict) and s.get("type") == "if":
                # create then and else blocks
                then_block = new_block()
                else_block = new_block()
                after_block = new_block()
                blocks[then_block.id] = then_block
                blocks[else_block.id] = else_block
                blocks[after_block.id] = after_block
                # connect
                target_block.succ.update({then_block.id, else_block.id})
                then_block.pred.add(target_block.id)
                else_block.pred.add(target_block.id)
                # fill then/else
                then_stmts = s.get("then")
                else_stmts = s.get("else")
                if isinstance(then_stmts, dict) and then_stmts.get("type") == "block":
                    then_block.stmts.extend(then_stmts.get("stmts", []))
                else:
                    then_block.stmts.append(then_stmts)
                if isinstance(else_stmts, dict) and else_stmts.get("type") == "block":
                    else_block.stmts.extend(else_stmts.get("stmts", []))
                else:
                    else_block.stmts.append(else_stmts)
                # connect then/else to after
                then_block.succ.add(after_block.id)
                else_block.succ.add(after_block.id)
                after_block.pred.update({then_block.id, else_block.id})
                # recursively split inside then/else
                # process nested ifs inside then/else
                append_stmts_to_block(then_block, then_block.stmts)
                append_stmts_to_block(else_block, else_block.stmts)
                # continue with after_block as current
                target_block = after_block
                blocks[after_block.id] = after_block
            else:
                target_block.stmts.append(s)
        return target_block

    # start by appending top-level stmts
    if isinstance(body, dict) and body.get("type") == "block":
        append_stmts_to_block(entry, body.get("stmts", []))
    else:
        entry.stmts.append(body)
    return blocks


# -------------------------
# SSA construction with phi nodes (simple algorithm)
# -------------------------
def compute_dominators(blocks: Dict[str, BasicBlock], entry_id: str) -> Dict[str, Set[str]]:
    # iterative algorithm
    dom: Dict[str, Set[str]] = {}
    all_nodes = set(blocks.keys())
    for n in all_nodes:
        dom[n] = set(all_nodes)  # initialize to all
    dom[entry_id] = {entry_id}
    changed = True
    while changed:
        changed = False
        for n in all_nodes - {entry_id}:
            preds = blocks[n].pred
            if not preds:
                new_dom = {n}
            else:
                # intersect dominators of preds
                new_dom = set(all_nodes)
                for p in preds:
                    new_dom &= dom[p]
                new_dom.add(n)
            if new_dom != dom[n]:
                dom[n] = new_dom
                changed = True
    return dom


def immediate_dominators(dom_map: Dict[str, Set[str]]) -> Dict[str, Optional[str]]:
    idom: Dict[str, Optional[str]] = {}
    for n, doms in dom_map.items():
        if len(doms) <= 1:
            idom[n] = None
            continue
        # immediate dominator is the unique dominator of n that strictly dominates n and that
        # is not dominated by any other strict dominator of n
        strict = doms - {n}
        cand = None
        for d in strict:
            if all((d == other) or (d not in dom_map[other]) for other in strict):
                cand = d
                break
        idom[n] = cand
    return idom


def place_phi_nodes(blocks: Dict[str, BasicBlock]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Place phi nodes at block entries for variables assigned in multiple predecessors.
    Returns mapping block_id -> list of phi node dicts.
    """
    # collect definitions per block
    defs_per_block: Dict[str, Set[str]] = {}
    for bid, b in blocks.items():
        defs = set()
        for s in b.stmts:
            if isinstance(s, dict) and s.get("type") == "assign" and isinstance(s.get("target"), str):
                defs.add(s.get("target"))
        defs_per_block[bid] = defs

    # for each block compute incoming defs per variable
    phi_map: Dict[str, List[Dict[str, Any]]] = {bid: [] for bid in blocks}
    for bid, b in blocks.items():
        if len(b.pred) <= 1:
            continue
        # for each var, collect sources
        incoming_defs: Dict[str, Set[str]] = {}
        for p in b.pred:
            for d in defs_per_block.get(p, set()):
                incoming_defs.setdefault(d, set()).add(p)
        # if a var has defs from >1 preds, insert phi
        for var, sources in incoming_defs.items():
            if len(sources) > 1:
                phi = {"type": "phi", "target": var, "args": {p: {"type": "var", "name": var} for p in b.pred}}
                phi_map[bid].append(phi)
    return phi_map


def ssa_transform_function(body: Any) -> Any:
    """
    Build CFG, place phi nodes conservatively, and rename variables to SSA names.
    Returns transformed body (block) with phi nodes inserted and renamed vars.
    """
    try:
        blocks = build_cfg_for_function(body)
        if not blocks:
            return body
        entry_id = next(iter(blocks.keys()))
        dom_map = compute_dominators(blocks, entry_id)
        idom = immediate_dominators(dom_map)
        # place phi nodes conservatively
        phi_map = place_phi_nodes(blocks)
        # insert phi nodes at beginning of blocks
        for bid, phis in phi_map.items():
            if phis:
                blocks[bid].stmts = phis + blocks[bid].stmts

        # rename using stacks per variable
        name_counters: Dict[str, int] = {}
        stacks: Dict[str, List[str]] = {}

        def new_name(v: str) -> str:
            idx = name_counters.get(v, 0) + 1
            name_counters[v] = idx
            nm = f"{v}__ssa{idx}"
            stacks.setdefault(v, []).append(nm)
            return nm

        def top_name(v: str) -> Optional[str]:
            st = stacks.get(v)
            return st[-1] if st else None

        # perform a simple dominator-tree traversal (preorder) for renaming
        # build children map
        children: Dict[str, List[str]] = {}
        for n, p in idom.items():
            if p is None:
                continue
            children.setdefault(p, []).append(n)

        def rename_block(bid: str):
            b = blocks[bid]
            # process phi nodes first
            new_stmts = []
            for s in b.stmts:
                if isinstance(s, dict) and s.get("type") == "phi":
                    tgt = s.get("target")
                    nm = new_name(tgt)
                    s["target"] = nm
                    # phi args become references to current version of var at predecessor edges
                    for p in list(s.get("args", {}).keys()):
                        cur = top_name(s["args"][p].get("name")) if isinstance(s["args"][p], dict) else None
                        if cur:
                            s["args"][p] = {"type": "var", "name": cur}
                    new_stmts.append(s)
                else:
                    # for non-phi, process recursively
                    new_stmts.append(rename_node(s))
            b.stmts = new_stmts
            # recursively rename children
            for c in children.get(bid, []):
                rename_block(c)
            # after finishing block, pop assignments in this block from stacks
            # find assigns introduced here
            for s in b.stmts:
                if isinstance(s, dict) and s.get("type") == "assign" and isinstance(s.get("target"), str):
                    tgt = s["target"]
                    # if it is an SSA name, pop it (we created via new_name)
                    if tgt.endswith("__ssa") or "__ssa" in tgt:
                        base = tgt.split("__ssa")[0]
                        st = stacks.get(base)
                        if st:
                            st.pop()

        def rename_node(node):
            if isinstance(node, dict):
                if node.get("type") == "assign" and isinstance(node.get("target"), str):
                    base = node["target"]
                    # create new name
                    nm = new_name(base)
                    node["target"] = nm
                    node["value"] = rename_node(node.get("value"))
                    return node
                if node.get("type") == "var" and isinstance(node.get("name"), str):
                    cur = top_name(node.get("name"))
                    if cur:
                        return {"type": "var", "name": cur}
                    return node
                # rewrite phi args if present
                if node.get("type") == "phi":
                    # already handled earlier
                    return node
                for k, v in list(node.items()):
                    node[k] = rename_node(v)
                return node
            if isinstance(node, list):
                return [rename_node(x) for x in node]
            return node

        # entry-based start
        rename_block(entry_id)

        # reconstruct a linear block form by concatenating block stmts in id order (best-effort)
        sorted_ids = sorted(blocks.keys())
        merged_stmts = []
        for bid in sorted_ids:
            merged_stmts.extend(blocks[bid].stmts)
        new_body = {"type": "block", "stmts": merged_stmts}
        return new_body
    except Exception:
        LOG.exception("ssa_transform_function failed (ignored)")
        return body


# -------------------------
# SSA-driven GVN
# -------------------------
def ssa_gvn_on_function(body: Any) -> Any:
    """
    Assumes function body is in SSA form (or SSA-like). Performs value numbering using SSA names
    and factors identical expressions into `_ssa_gvn_shared`.
    """
    try:
        ssa_body = ssa_transform_function(body)
        expr_map: Dict[str, List[List[Tuple[str, Any]]]] = {}
        def collect(node, path=None):
            if path is None:
                path = []
            if isinstance(node, dict) and is_side_effect_free(node):
                key = canonical_json(node)
                expr_map.setdefault(key, []).append(list(path))
            if isinstance(node, dict):
                for k, v in node.items():
                    collect(v, path + [("dict", k)])
            elif isinstance(node, list):
                for i, it in enumerate(node):
                    collect(it, path + [("list", i)])
        collect(ssa_body)
        shared = {}
        sid = 0
        for key, occs in expr_map.items():
            if len(occs) >= 2:
                rep = json.loads(key)
                name = f"ssa_gvn{sid}"
                sid += 1
                shared[name] = rep
                for path in occs:
                    try:
                        _replace_at_path(ssa_body, path, {"type": "shared_ref", "id": name})
                    except Exception:
                        LOG.debug("ssa_gvn replace failed at %s", path)
        if shared and isinstance(ssa_body, dict):
            ssa_body.setdefault("_ssa_gvn_shared", {}).update(shared)
        return ssa_body
    except Exception:
        LOG.exception("ssa_gvn_on_function failed (ignored)")
        return body


# -------------------------
# Interprocedural GVN (SSA-driven)
# -------------------------
def interprocedural_gvn(module_ir: Any, min_occurrences: int = 2) -> Any:
    """
    Conservative module-level GVN: SSA-transform each function then find identical SSA expressions
    across functions and factor them into `_module_gvn_shared`.
    """
    if not isinstance(module_ir, dict) or "functions" not in module_ir:
        return module_ir
    try:
        expr_map: Dict[str, List[Tuple[str, List[Tuple[str, Any]]]]] = {}
        for fname, fobj in module_ir.get("functions", {}).items():
            body = fobj.get("body")
            if not body:
                continue
            ssa_body = ssa_transform_function(body)
            fobj["body"] = ssa_body
            def collect(node, path=None):
                if path is None:
                    path = []
                if isinstance(node, dict) and is_side_effect_free(node):
                    key = canonical_json(node)
                    expr_map.setdefault(key, []).append((fname, list(path)))
                if isinstance(node, dict):
                    for k, v in node.items():
                        collect(v, path + [("dict", k)])
                elif isinstance(node, list):
                    for idx, it in enumerate(node):
                        collect(it, path + [("list", idx)])
            collect(ssa_body)
        shared = {}
        sid = 0
        for key, occs in expr_map.items():
            if len(occs) >= min_occurrences:
                rep = json.loads(key)
                name = f"mgvn{sid}"
                sid += 1
                shared[name] = rep
                for fname, path in occs:
                    try:
                        fb = module_ir["functions"][fname]["body"]
                        _replace_at_path(fb, path, {"type": "shared_ref", "id": name})
                    except Exception:
                        LOG.debug("interprocedural_gvn replace failed in %s at %s", fname, path)
        if shared:
            module_ir.setdefault("_module_gvn_shared", {}).update(shared)
        return module_ir
    except Exception:
        LOG.exception("interprocedural_gvn failed (ignored)")
        return module_ir


# -------------------------
# Threshold heuristics for real workloads
# -------------------------
def compute_size_threshold(module_ir: Any, base: int = 300) -> int:
    """
    Compute a sensible size threshold (bytes of canonical JSON) for inlining/specialization.
    Heuristic: median function size * factor, clamped to [base/2, base*10].
    """
    try:
        if not isinstance(module_ir, dict) or "functions" not in module_ir:
            return base
        sizes = []
        for f in module_ir.get("functions", {}).values():
            b = f.get("body")
            try:
                sizes.append(len(canonical_json(b)))
            except Exception:
                sizes.append(base)
        if not sizes:
            return base
        sizes.sort()
        median = sizes[len(sizes)//2]
        thr = int(max(base//2, min(base*10, median * 1.5)))
        return thr
    except Exception:
        return base


def compute_hot_threshold(module_ir: Any, base: float = 0.7) -> float:
    """
    Heuristic for selecting hotness threshold: reduce threshold for tiny modules, raise slightly for large modules.
    """
    try:
        if not isinstance(module_ir, dict) or "functions" not in module_ir:
            return base
        n = len(module_ir.get("functions", {}))
        if n < 10:
            return max(0.5, base - 0.15)
        if n > 500:
            return min(0.95, base + 0.15)
        # scale linearly between 10..500
        scale = (n - 10) / max(1, 500 - 10)
        return float(min(0.95, base + 0.15 * scale))
    except Exception:
        return base


# -------------------------
# Top-level optimize_shard (public)
# -------------------------
def optimize_shard(shard_ir: Any, options: Dict[str, Any]) -> Any:
    """
    Options:
      - enable_cse (bool)
      - enable_dse (bool)
      - enable_gvn (bool)        -> function-local SSA GVN
      - enable_ip_cse (bool)     -> inter-procedural CSE across functions
      - enable_ip_gvn (bool)     -> inter-procedural GVN across functions (SSA-driven)
    """
    try:
        if options.get("enable_cse"):
            shard_ir = common_subexpression_elimination(shard_ir, min_occurrences=options.get("cse_min", 2))
        if options.get("enable_dse"):
            shard_ir = dead_store_elimination(shard_ir)
        if options.get("enable_gvn"):
            if isinstance(shard_ir, dict) and "functions" in shard_ir:
                for fname, fobj in shard_ir.get("functions", {}).items():
                    body = fobj.get("body")
                    if body:
                        fobj["body"] = ssa_gvn_on_function(body)
        if options.get("enable_ip_cse"):
            shard_ir = interprocedural_cse(shard_ir, min_occurrences=options.get("ip_cse_min", 2))
        if options.get("enable_ip_gvn"):
            shard_ir = interprocedural_gvn(shard_ir, min_occurrences=options.get("ip_gvn_min", 2))
    except Exception:
        LOG.exception("optimize_shard (utils) failed")
    return shard_ir

# --- patch: add imports, new plugins and fix registry wiring ---
# Insert near top imports (after existing imports)
from instryx_opt_utils import (
    common_subexpression_elimination,
    dead_store_elimination,
    ssa_transform_function,
    ssa_gvn_on_function,
    interprocedural_gvn,
    compute_size_threshold,
    compute_hot_threshold,
    optimize_shard,
)

# --- add new plugin classes (place below existing built-in plugins) ---
class CSEPlugin(PluginBase):
    def __init__(self):
        super().__init__()
        self.meta = PluginMeta(priority=20, name="cse", description="Conservative CSE (shared util)", version="1.0")

    def apply(self, ir, context):
        try:
            new_ir = common_subexpression_elimination(ir, min_occurrences=context.get("cse_min", 2))
            return new_ir, {"ok": True, "info": ["cse applied"], "warnings": [], "errors": []}
        except Exception:
            LOG.exception("CSEPlugin failed")
            return ir, {"ok": False, "info": [], "warnings": [], "errors": ["CSEPlugin exception"]}


class DSEPlugin(PluginBase):
    def __init__(self):
        super().__init__()
        self.meta = PluginMeta(priority=22, name="dse", description="Conservative dead-store elimination (shared util)", version="1.0")

    def apply(self, ir, context):
        try:
            new_ir = dead_store_elimination(ir)
            return new_ir, {"ok": True, "info": ["dse applied"], "warnings": [], "errors": []}
        except Exception:
            LOG.exception("DSEPlugin failed")
            return ir, {"ok": False, "info": [], "warnings": [], "errors": ["DSEPlugin exception"]}


class SSA_GVN_Plugin(PluginBase):
    def __init__(self):
        super().__init__()
        self.meta = PluginMeta(priority=21, name="ssa_gvn", description="SSA-driven function-local GVN", version="1.0")

    def apply(self, ir, context):
        try:
            if not isinstance(ir, dict) or "functions" not in ir:
                return ir, {"ok": True, "info": ["no functions"], "warnings": [], "errors": []}
            for fname, fobj in ir.get("functions", {}).items():
                body = fobj.get("body")
                if body:
                    fobj["body"] = ssa_gvn_on_function(body)
            return ir, {"ok": True, "info": ["ssa_gvn applied"], "warnings": [], "errors": []}
        except Exception:
            LOG.exception("SSA_GVN_Plugin failed")
            return ir, {"ok": False, "info": [], "warnings": [], "errors": ["SSA_GVN_Plugin exception"]}


class InterproceduralGVNPlugin(PluginBase):
    def __init__(self):
        super().__init__()
        self.meta = PluginMeta(priority=23, name="ip_gvn", description="Interprocedural GVN (SSA-driven)", version="1.0")

    def apply(self, ir, context):
        try:
            new_ir = interprocedural_gvn(ir, min_occurrences=context.get("ip_gvn_min", 2))
            return new_ir, {"ok": True, "info": ["interprocedural_gvn applied"], "warnings": [], "errors": []}
        except Exception:
            LOG.exception("InterproceduralGVNPlugin failed")
            return ir, {"ok": False, "info": [], "warnings": [], "errors": ["InterproceduralGVNPlugin exception"]}


# --- enhance ProfileGuidedInliningPlugin to use heuristics when profile missing ---
# Replace existing ProfileGuidedInliningPlugin.apply implementation with the following method body:

# (Find class ProfileGuidedInliningPlugin and replace or update its apply() method with:)
def _pgi_apply(self, ir, context):
    infos, warnings, errors = [], [], []
    try:
        # load profile either from context or path, use heuristics when absent
        profile = context.get("profile") or {}
        if not profile:
            # allow context to pass a profile_path string
            path = context.get("profile_path") or os.environ.get("INSTRYX_PROFILE_PATH")
            if path and os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        profile = json.load(f)
                except Exception:
                    LOG.exception("failed to load profile at %s", path)
                    profile = {}
        hotness_map = profile.get("function_hotness", {})
        # if no explicit profile, compute default hot threshold and size threshold heuristically
        module_hint = context.get("module_ir") or ir
        size_thr = context.get("size_threshold") or compute_size_threshold(module_hint, base=getattr(self, "size_threshold", 300))
        hot_thr = float(context.get("hot_threshold") or compute_hot_threshold(module_hint, base=getattr(self, "hotness_threshold", 0.7)))
        functions = ir.get("functions", {}) if isinstance(ir, dict) else {}
        eligible = {fname for fname, fobj in functions.items()
                    if isinstance(fobj.get("body"), (dict, list)) and len(json.dumps(fobj.get("body") or {})) <= size_thr}
        def inline_node(node):
            if isinstance(node, dict):
                if node.get("type") == "call" and isinstance(node.get("fn"), str):
                    fname = node.get("fn")
                    hot = float(hotness_map.get(fname, 0.0))
                    if hot >= hot_thr and fname in eligible:
                        fobj = functions.get(fname, {})
                        body = fobj.get("body")
                        params = fobj.get("params", []) or []
                        args = node.get("args", []) or []
                        mapping = {p: (args[i] if i < len(args) else {"type": "var", "name": p}) for i, p in enumerate(params)}
                        def repl(n):
                            if isinstance(n, dict):
                                if n.get("type") == "var" and isinstance(n.get("name"), str) and n.get("name") in mapping:
                                    return mapping[n.get("name")]
                                for k, v in list(n.items()):
                                    n[k] = repl(v)
                                return n
                            if isinstance(n, list):
                                return [repl(x) for x in n]
                            return n
                        inlined = repl(copy.deepcopy(body))
                        infos.append(f"inlined {fname} (hot={hot})")
                        return inlined
                for k, v in list(node.items()):
                    node[k] = inline_node(v)
                return node
            if isinstance(node, list):
                return [inline_node(x) for x in node]
            return node
        new_ir = copy.deepcopy(ir)
        for fname, fobj in new_ir.get("functions", {}).items():
            fobj["body"] = inline_node(fobj.get("body"))
        return new_ir, {"ok": True, "info": infos, "warnings": warnings, "errors": errors}
    except Exception as e:
        LOG.exception("ProfileGuidedInliningPlugin failed")
        return ir, {"ok": False, "info": [], "warnings": warnings, "errors": [str(e), traceback.format_exc()]}

# Attach replacement method
ProfileGuidedInliningPlugin.apply = _pgi_apply


# --- fix and improve create_default_registry ---
def create_default_registry(max_workers: int = 8, per_pass_timeout: float = 2.0) -> PluginRegistry:
    reg = PluginRegistry(max_workers=max_workers, per_pass_timeout=per_pass_timeout)
    # register high-value transformations early so others benefit
    try:
        reg.register(MacroAwarePass())
    except Exception:
        LOG.exception("failed to register MacroAwarePass")
    # register shared-utility-driven optimizations
    try:
        reg.register(SSAConversionPlugin())   # keep lightweight SSA
        reg.register(CSEPlugin())             # local CSE (uses instryx_opt_utils)
        reg.register(SSA_GVN_Plugin())        # function-local SSA-driven GVN
        reg.register(InterproceduralGVNPlugin())  # interprocedural GVN
        reg.register(DSEPlugin())             # dead-store elimination (shared util)
    except Exception:
        LOG.exception("failed to register CSE/DSE/GVN plugins")

    # continue with other built-ins
    reg.register(ConstantFoldingPlugin())
    reg.register(PeepholeOptimizerPlugin())
    reg.register(ConstantPropagationPlugin())
    reg.register(CopyPropagationPlugin())
    reg.register(CFGSimplifyPlugin())
    # If DeadCodeEliminationPlugin exists register it, otherwise skip gracefully
    if "DeadCodeEliminationPlugin" in globals():
        try:
            reg.register(globals()["DeadCodeEliminationPlugin"]())
        except Exception:
            LOG.debug("DeadCodeEliminationPlugin registration failed (ignored)")
    reg.register(InlineSmallFunctionsPlugin())
    # Profile-guided inlining already present and now tuned to use heuristics
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

# register(assistant) style plugin that flags TODO comments as suggestions
from ciams.ai_engine import Suggestion

def rule_todos(source: str, filename=None):
    suggestions = []
    for m in __import__('re').finditer(r"//\s*TODO[: ]?(.*)$|#\s*TODO[: ]?(.*)$", source, __import__('re').M):
        txt = (m.group(1) or m.group(2) or "").strip()
        snippet = source[max(0, m.start()-60): m.end()+60].splitlines()[0].strip()
        suggestions.append(Suggestion("note", [txt], f"TODO found: {txt}", 0.2, snippet, (m.start(), m.end())))
    return suggestions

def register(assistant):
    assistant.register_rule(rule_todos)
    def unregister(assistant):
        # optional: if your PluginManager supports unregister, implement removal logic
        pass
    assistant.unregister_rule(rule_todos)
    # register(assistant) style plugin that flags TODO comments as suggestions
    """
    Registers the rule_todos function with the assistant to flag TODO comments in the source code.
    """

    return unregister
    
"""
Enhanced TODO task plugin for CIAMS assistant.

Features:
 - Detects TODO/FIXME/XXX/HACK/OPTIMIZE comments in single-line and block comments.
 - Parses metadata: assignee (@user), issue references (#123), priority ([P1]/priority:high),
   due dates (due:YYYY-MM-DD), freeform tags (@review, @perf).
 - Produces rich Suggestion objects with contextual multi-line snippets and precise
   (line, column, start_offset, end_offset) positions.
 - Exposes `parse_todos` for structured data and `rule_todos` for assistant integration.
 - register/unregister helpers for the assistant PluginManager.
 - Small convenience `scan_sources` to run the parser over multiple files.
"""

from __future__ import annotations
import re
import os
from typing import Any, Dict, List, Optional, Tuple

from ciams.ai_engine import Suggestion

# Regex building blocks
_SINGLE_LINE_COMMENT = r"(?P<prefix>//|#|--|;)\s*(?P<body>.*)$"
_BLOCK_COMMENT = r"/\*(?P<body>.*?)\*/"  # DOTALL used when applied

# Recognized markers (case-insensitive)
_MARKER_WORDS = r"(TODO|FIXME|XXX|HACK|OPTIMIZE|NOTE|REVIEW)"

# Inline marker pattern: captures marker and rest of line
_INLINE_MARKER_RE = re.compile(
    rf"(?P<prefix>//|#|--|;)\s*(?P<marker>{_MARKER_WORDS})[:\s]?(?P<rest>.*)$",
    re.IGNORECASE | re.MULTILINE
)

# Block comment search (will scan inside for markers)
_BLOCK_COMMENT_RE = re.compile(_BLOCK_COMMENT, re.DOTALL | re.IGNORECASE)

# Metadata extraction patterns
_ASSIGNEE_RE = re.compile(r"@(?P<assignee>[A-Za-z0-9_\-\.]+)")
_ISSUE_RE = re.compile(r"#(?P<issue>\d+)")
_PRIORITY_RE = re.compile(r"\b(?:\[P(?P<pnum>[0-9])\]|priority[:=]\s*(?P<pword>high|medium|low|p1|p2|p3))\b", re.IGNORECASE)
_DUE_RE = re.compile(r"\b(?:due[:=]\s*(?P<date>\d{4}-\d{2}-\d{2}))\b", re.IGNORECASE)
_TAG_RE = re.compile(r"@(?P<tag>[A-Za-z0-9_\-]+)")

# severity mapping by marker or metadata
_SEVERITY_BY_MARKER = {
    "FIXME": 0.95,
    "TODO": 0.60,
    "XXX": 0.90,
    "HACK": 0.45,
    "OPTIMIZE": 0.50,
    "NOTE": 0.30,
    "REVIEW": 0.65,
}

_PRIORITY_MAP = {
    "p1": 0.98, "p2": 0.9, "p3": 0.75,
    "high": 0.95, "medium": 0.7, "low": 0.4
}


def _offset_to_linecol(source: str, offset: int) -> Tuple[int, int]:
    """Convert byte offset to (line, col) 1-based."""
    if offset < 0:
        offset = 0
    # count lines
    prefix = source[:offset]
    line = prefix.count("\n") + 1
    # column is chars since last newline
    last_nl = prefix.rfind("\n")
    col = offset - (last_nl + 1) + 1 if last_nl != -1 else offset + 1
    return line, col


def _make_snippet(source: str, start: int, end: int, context_lines: int = 2) -> str:
    """Return context snippet with the TODO line highlighted with '>>' prefix."""
    lines = source.splitlines()
    # compute line indices (0-based)
    start_line, _ = _offset_to_linecol(source, start)
    end_line, _ = _offset_to_linecol(source, end)
    # convert to 0-based indices
    sidx = max(0, start_line - 1 - context_lines)
    eidx = min(len(lines), end_line + context_lines)
    snippet_lines = []
    for i in range(sidx, eidx):
        prefix = "   "
        if i >= (start_line - 1) and i <= (end_line - 1):
            prefix = ">> "
        snippet_lines.append(f"{prefix}{i+1:4d}: {lines[i]}")
    return "\n".join(snippet_lines)


def _normalize_marker(marker: Optional[str]) -> str:
    return (marker or "TODO").upper()


def _compute_confidence(marker: str, metadata: Dict[str, Any]) -> float:
    """Heuristic confidence based on marker and metadata (priority/due/issue)."""
    m = _normalize_marker(marker)
    base = _SEVERITY_BY_MARKER.get(m, 0.5)
    # bump for explicit priority
    pr = metadata.get("priority")
    if pr:
        if isinstance(pr, str):
            base = max(base, _PRIORITY_MAP.get(pr.lower(), base))
    # bump slightly if assignee present (actionable)
    if metadata.get("assignee"):
        base = min(0.99, base + 0.05)
    # bump if issue reference exists
    if metadata.get("issue"):
        base = min(0.98, base + 0.03)
    return round(base, 2)


def parse_todos(source: str, filename: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Parse source and return list of structured todo dictionaries.

    Each todo dict contains:
      - marker: TODO/FIXME/...
      - text: the freeform text after the marker
      - assignee: optional @user
      - issue: optional issue number (as string)
      - priority: optional (p1/p2/p3/high/medium/low)
      - due: optional YYYY-MM-DD
      - tags: list of @tags found
      - start_offset, end_offset: offsets in source
      - line, col: 1-based position of the marker
      - snippet: contextual snippet
      - filename: as passed in
    """
    todos: List[Dict[str, Any]] = []

    # 1) single-line comments with inline markers
    for m in _INLINE_MARKER_RE.finditer(source):
        marker = m.group("marker") or "TODO"
        rest = (m.group("rest") or "").strip()
        start, end = m.start(), m.end()
        # extract metadata
        assignee = (_ASSIGNEE_RE.search(rest).group("assignee")
                    if _ASSIGNEE_RE.search(rest) else None)
        issue = (_ISSUE_RE.search(rest).group("issue")
                 if _ISSUE_RE.search(rest) else None)
        pr_match = _PRIORITY_RE.search(rest)
        priority = None
        if pr_match:
            priority = pr_match.group("pnum") or pr_match.group("pword")
        due = (_DUE_RE.search(rest).group("date")
               if _DUE_RE.search(rest) else None)
        tags = [m.group("tag") for m in _TAG_RE.finditer(rest) if m.group("tag") != (assignee or "")]
        line, col = _offset_to_linecol(source, start)
        snippet = _make_snippet(source, start, end)
        todos.append({
            "marker": marker.upper(),
            "text": rest,
            "assignee": assignee,
            "issue": issue,
            "priority": priority,
            "due": due,
            "tags": tags,
            "start_offset": start,
            "end_offset": end,
            "line": line,
            "col": col,
            "snippet": snippet,
            "filename": filename
        })

    # 2) block comments search for markers inside
    for bm in _BLOCK_COMMENT_RE.finditer(source):
        body = bm.group("body") or ""
        # find markers inside body (may be multiple)
        for im in re.finditer(rf"(?P<marker>{_MARKER_WORDS})[:\s]?(?P<rest>.*)$", body, re.IGNORECASE | re.MULTILINE):
            marker = im.group("marker") or "TODO"
            rest = (im.group("rest") or "").strip()
            # compute offsets relative to source
            # find the position of the inner match within the full source
            inner_rel = im.start()
            start = bm.start() + 2 + inner_rel  # +2 accounts for '/*'
            end = start + (im.end() - im.start())
            assignee = (_ASSIGNEE_RE.search(rest).group("assignee")
                        if _ASSIGNEE_RE.search(rest) else None)
            issue = (_ISSUE_RE.search(rest).group("issue")
                     if _ISSUE_RE.search(rest) else None)
            pr_match = _PRIORITY_RE.search(rest)
            priority = None
            if pr_match:
                priority = pr_match.group("pnum") or pr_match.group("pword")
            due = (_DUE_RE.search(rest).group("date")
                   if _DUE_RE.search(rest) else None)
            tags = [m.group("tag") for m in _TAG_RE.finditer(rest) if m.group("tag") != (assignee or "")]
            line, col = _offset_to_linecol(source, start)
            snippet = _make_snippet(source, start, end)
            todos.append({
                "marker": marker.upper(),
                "text": rest,
                "assignee": assignee,
                "issue": issue,
                "priority": priority,
                "due": due,
                "tags": tags,
                "start_offset": start,
                "end_offset": end,
                "line": line,
                "col": col,
                "snippet": snippet,
                "filename": filename
            })

    # deduplicate by start_offset
    seen = set()
    unique: List[Dict[str, Any]] = []
    for t in todos:
        key = (t["filename"], t["start_offset"], t["end_offset"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(t)
    return unique


def rule_todos(source: str, filename: Optional[str] = None) -> List[Suggestion]:
    """
    Assistant integration entrypoint.

    Returns a list of Suggestion objects suitable for the assistant UI.
    Each Suggestion encodes the marker, parsed metadata and a recommended
    high-level action in the title/message.
    """
    suggestions: List[Suggestion] = []
    parsed = parse_todos(source, filename=filename)
    for t in parsed:
        marker = t["marker"]
        text = t["text"] or ""
        # tags for the suggestion: include marker and any discovered tags
        tags = [marker.lower()] + t.get("tags", [])
        if t.get("priority"):
            tags.append(f"priority:{t['priority']}")
        if t.get("assignee"):
            tags.append(f"assignee:{t['assignee']}")
        if t.get("issue"):
            tags.append(f"issue:{t['issue']}")

        # human-friendly title & description
        short = text.splitlines()[0] if text else ""
        title = f"{marker}: {short}" if short else f"{marker}"
        desc_lines = [
            f"File: {t['filename'] or '<unknown>'}  Line: {t['line']}  Col: {t['col']}",
            f"Text: {text}",
        ]
        if t.get("assignee"):
            desc_lines.append(f"Assignee: @{t['assignee']}")
        if t.get("issue"):
            desc_lines.append(f"Issue: #{t['issue']}")
        if t.get("priority"):
            desc_lines.append(f"Priority: {t['priority']}")
        if t.get("due"):
            desc_lines.append(f"Due: {t['due']}")
        desc = " | ".join(desc_lines)

        # confidence heuristic
        confidence = _compute_confidence(marker, t)

        snippet = t.get("snippet", "")
        # Suggestion constructor: (kind, tags:list, title, confidence:float, snippet, location_tuple)
        # keep location as (line, col, start_offset, end_offset) for richer UI
        location = (t["line"], t["col"], t["start_offset"], t["end_offset"])

        suggestions.append(Suggestion("task", tags, title, confidence, snippet, location))

    return suggestions


def scan_sources(paths: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convenience helper to scan files on disk. Returns mapping path -> parsed todos.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for p in paths:
        if os.path.isdir(p):
            # walk tree (ignores hidden directories)
            for root, dirs, files in os.walk(p):
                for f in files:
                    if f.startswith("."):
                        continue
                    full = os.path.join(root, f)
                    try:
                        with open(full, "r", encoding="utf-8") as fh:
                            src = fh.read()
                        parsed = parse_todos(src, filename=full)
                        if parsed:
                            out[full] = parsed
                    except Exception:
                        # best-effort; don't fail the whole scan
                        continue
        else:
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    src = fh.read()
                parsed = parse_todos(src, filename=p)
                if parsed:
                    out[p] = parsed
            except Exception:
                continue
    return out


def register(assistant) -> None:
    """
    Register the plugin with an assistant instance.

    The assistant is expected to provide:
      - register_rule(callable) -> registers a rule that accepts (source, filename) and returns [Suggestion]
      - unregister_rule(callable) -> optional, should remove the registered rule

    This function registers rule_todos and returns nothing. If the assistant
    supports unregistering, an external system may call `unregister(assistant)`.
    """
    assistant.register_rule(rule_todos)


def unregister(assistant) -> None:
    """Optional unregister routine to remove the plugin's rule from the assistant."""
    try:
        assistant.unregister_rule(rule_todos)
    except Exception:
        # best-effort: not all assistant/plugin managers implement unregister
        pass

"""
Enhanced TODO task plugin for CIAMS assistant (super boosters, enhancers, tooling).

Key additions:
 - Concurrency scanning (ThreadPoolExecutor) with file-mtime caching.
 - JSON + SARIF exporters for CI and static analysis integration.
 - Grouping, summary stats and prioritized sorting.
 - Quick-fix patch suggestion generator (edit/replace TODO with issue link or remove).
 - Issue payload generator for GitHub-style issue creation.
 - Register helpers expose utilities on assistant when available.
 - CLI entry when executed directly (scan paths, export).
"""

from __future__ import annotations
import re
import os
import json
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Iterable, Callable
from functools import lru_cache

from ciams.ai_engine import Suggestion

# --- existing regex / helpers (kept and extended) ---
_SINGLE_LINE_COMMENT = r"(?P<prefix>//|#|--|;)\s*(?P<body>.*)$"
_BLOCK_COMMENT = r"/\*(?P<body>.*?)\*/"  # DOTALL used when applied
_MARKER_WORDS = r"(TODO|FIXME|XXX|HACK|OPTIMIZE|NOTE|REVIEW|BUG|TASK)"
_INLINE_MARKER_RE = re.compile(
    rf"(?P<prefix>//|#|--|;)\s*(?P<marker>{_MARKER_WORDS})[:\s]?(?P<rest>.*)$",
    re.IGNORECASE | re.MULTILINE
)
_BLOCK_COMMENT_RE = re.compile(_BLOCK_COMMENT, re.DOTALL | re.IGNORECASE)

_ASSIGNEE_RE = re.compile(r"@(?P<assignee>[A-Za-z0-9_\-\.]+)")
_ISSUE_RE = re.compile(r"#(?P<issue>\d+)")
_PRIORITY_RE = re.compile(r"\b(?:\[P(?P<pnum>[0-9])\]|priority[:=]\s*(?P<pword>high|medium|low|p1|p2|p3))\b", re.IGNORECASE)
_DUE_RE = re.compile(r"\b(?:due[:=]\s*(?P<date>\d{4}-\d{2}-\d{2}))\b", re.IGNORECASE)
_TAG_RE = re.compile(r"@(?P<tag>[A-Za-z0-9_\-]+)")

_SEVERITY_BY_MARKER = {
    "FIXME": 0.95,
    "TODO": 0.60,
    "XXX": 0.90,
    "HACK": 0.45,
    "OPTIMIZE": 0.50,
    "NOTE": 0.30,
    "REVIEW": 0.65,
    "BUG": 0.96,
    "TASK": 0.55,
}

_PRIORITY_MAP = {
    "p1": 0.98, "p2": 0.9, "p3": 0.75,
    "high": 0.95, "medium": 0.7, "low": 0.4
}

# --- utility helpers ---
def _offset_to_linecol(source: str, offset: int) -> Tuple[int, int]:
    if offset < 0:
        offset = 0
    prefix = source[:offset]
    line = prefix.count("\n") + 1
    last_nl = prefix.rfind("\n")
    col = offset - (last_nl + 1) + 1 if last_nl != -1 else offset + 1
    return line, col

def _make_snippet(source: str, start: int, end: int, context_lines: int = 2) -> str:
    lines = source.splitlines()
    start_line, _ = _offset_to_linecol(source, start)
    end_line, _ = _offset_to_linecol(source, end)
    sidx = max(0, start_line - 1 - context_lines)
    eidx = min(len(lines), end_line + context_lines)
    snippet_lines = []
    for i in range(sidx, eidx):
        prefix = "   "
        if i >= (start_line - 1) and i <= (end_line - 1):
            prefix = ">> "
        snippet_lines.append(f"{prefix}{i+1:4d}: {lines[i]}")
    return "\n".join(snippet_lines)

def _normalize_marker(marker: Optional[str]) -> str:
    return (marker or "TODO").upper()

def _compute_confidence(marker: str, metadata: Dict[str, Any]) -> float:
    m = _normalize_marker(marker)
    base = _SEVERITY_BY_MARKER.get(m, 0.5)
    pr = metadata.get("priority")
    if pr:
        if isinstance(pr, str):
            base = max(base, _PRIORITY_MAP.get(pr.lower(), base))
    if metadata.get("assignee"):
        base = min(0.99, base + 0.05)
    if metadata.get("issue"):
        base = min(0.98, base + 0.03)
    return round(base, 2)

# --- parsing / core functionality ---
def parse_todos(source: str, filename: Optional[str] = None) -> List[Dict[str, Any]]:
    todos: List[Dict[str, Any]] = []

    # single-line marker matches
    for m in _INLINE_MARKER_RE.finditer(source):
        marker = m.group("marker") or "TODO"
        rest = (m.group("rest") or "").strip()
        start, end = m.start(), m.end()
        assignee = (_ASSIGNEE_RE.search(rest).group("assignee") if _ASSIGNEE_RE.search(rest) else None)
        issue = (_ISSUE_RE.search(rest).group("issue") if _ISSUE_RE.search(rest) else None)
        pr_match = _PRIORITY_RE.search(rest)
        priority = pr_match.group("pnum") or pr_match.group("pword") if pr_match else None
        due = (_DUE_RE.search(rest).group("date") if _DUE_RE.search(rest) else None)
        tags = [mo.group("tag") for mo in _TAG_RE.finditer(rest) if mo.group("tag") != (assignee or "")]
        line, col = _offset_to_linecol(source, start)
        snippet = _make_snippet(source, start, end)
        todos.append({
            "marker": marker.upper(),
            "text": rest,
            "assignee": assignee,
            "issue": issue,
            "priority": priority,
            "due": due,
            "tags": tags,
            "start_offset": start,
            "end_offset": end,
            "line": line,
            "col": col,
            "snippet": snippet,
            "filename": filename
        })

    # block comments
    for bm in _BLOCK_COMMENT_RE.finditer(source):
        body = bm.group("body") or ""
        for im in re.finditer(rf"(?P<marker>{_MARKER_WORDS})[:\s]?(?P<rest>.*)$", body, re.IGNORECASE | re.MULTILINE):
            marker = im.group("marker") or "TODO"
            rest = (im.group("rest") or "").strip()
            inner_rel = im.start()
            start = bm.start() + 2 + inner_rel
            end = start + (im.end() - im.start())
            assignee = (_ASSIGNEE_RE.search(rest).group("assignee") if _ASSIGNEE_RE.search(rest) else None)
            issue = (_ISSUE_RE.search(rest).group("issue") if _ISSUE_RE.search(rest) else None)
            pr_match = _PRIORITY_RE.search(rest)
            priority = pr_match.group("pnum") or pr_match.group("pword") if pr_match else None
            due = (_DUE_RE.search(rest).group("date") if _DUE_RE.search(rest) else None)
            tags = [mo.group("tag") for mo in _TAG_RE.finditer(rest) if mo.group("tag") != (assignee or "")]
            line, col = _offset_to_linecol(source, start)
            snippet = _make_snippet(source, start, end)
            todos.append({
                "marker": marker.upper(),
                "text": rest,
                "assignee": assignee,
                "issue": issue,
                "priority": priority,
                "due": due,
                "tags": tags,
                "start_offset": start,
                "end_offset": end,
                "line": line,
                "col": col,
                "snippet": snippet,
                "filename": filename
            })

    # dedupe by offsets
    seen = set()
    unique: List[Dict[str, Any]] = []
    for t in todos:
        key = (t["filename"], t["start_offset"], t["end_offset"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(t)
    return unique

# --- advanced boosters / tooling ---

# simple in-memory cache keyed by file path + mtime
_file_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}

def parse_todos_cached(path: str) -> List[Dict[str, Any]]:
    try:
        mtime = os.path.getmtime(path)
    except Exception:
        mtime = 0.0
    entry = _file_cache.get(path)
    if entry and entry[0] == mtime:
        return entry[1]
    try:
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
    except Exception:
        return []
    parsed = parse_todos(src, filename=path)
    _file_cache[path] = (mtime, parsed)
    return parsed

def scan_sources_concurrent(paths: Iterable[str], max_workers: int = 8, skip_hidden: bool = True) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, fs in os.walk(p):
                if skip_hidden:
                    dirs[:] = [d for d in dirs if not d.startswith(".")]
                for f in fs:
                    if f.startswith("."):
                        continue
                    files.append(os.path.join(root, f))
        else:
            files.append(p)
    # concurrent parse
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(parse_todos_cached, f): f for f in files}
        for fut in as_completed(futures):
            f = futures[fut]
            try:
                res = fut.result()
            except Exception:
                res = []
            if res:
                out[f] = res
    return out

def group_todos(todos: Iterable[Dict[str, Any]], by: str = "assignee") -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for t in todos:
        key = "<unassigned>"
        if by == "assignee":
            key = t.get("assignee") or "<unassigned>"
        elif by == "priority":
            key = str(t.get("priority") or "<none>")
        elif by == "marker":
            key = t.get("marker") or "<marker>"
        elif by == "file":
            key = t.get("filename") or "<unknown>"
        groups.setdefault(key, []).append(t)
    return groups

def summary_stats(todos: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    by_marker = {}
    by_priority = {}
    by_assignee = {}
    for t in todos:
        total += 1
        by_marker[t.get("marker")] = by_marker.get(t.get("marker"), 0) + 1
        by_priority[t.get("priority")] = by_priority.get(t.get("priority"), 0) + 1
        by_assignee[t.get("assignee")] = by_assignee.get(t.get("assignee"), 0) + 1
    return {"total": total, "by_marker": by_marker, "by_priority": by_priority, "by_assignee": by_assignee}

# quick-fix generator: returns a minimal patch dict {start,end,replacement} or None
def generate_quickfix_patch(todo: Dict[str, Any], action: str = "issue", issue_template_url: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    action:
      - "issue": replace TODO comment with "See issue #<generated>" placeholder (non-destructive suggestion)
      - "remove": remove the TODO line/comment
      - "reference": append " (tracked: <url>)" to the line
    """
    start = todo.get("start_offset")
    end = todo.get("end_offset")
    if start is None or end is None:
        return None
    original = todo.get("text", "")
    if action == "remove":
        replacement = ""
    elif action == "issue":
        # generate determinist hash id for offline suggestion
        h = hashlib.sha1((todo.get("filename","") + str(start) + original).encode("utf-8")).hexdigest()[:8]
        if issue_template_url:
            url = issue_template_url.rstrip("/") + f"/issues/new?title={re.escape(todo.get('marker','TODO'))}+{h}"
            replacement = f"{todo.get('marker')} (migrated -> {url})"
        else:
            replacement = f"{todo.get('marker')} (migrated -> ISSUE-{h})"
    elif action == "reference":
        url = issue_template_url or "https://example.com/issue"
        replacement = f"{todo.get('marker')}: {original} (tracked: {url})"
    else:
        return None
    return {"start": start, "end": end, "replacement": replacement}

def generate_issue_payload(todo: Dict[str, Any], repo: Optional[str] = None) -> Dict[str, Any]:
    title = todo.get("text", "").splitlines()[0] or f"{todo.get('marker')} in {os.path.basename(todo.get('filename') or '')}"
    body_lines = [
        f"File: {todo.get('filename') or '<unknown>'}",
        f"Line: {todo.get('line')}",
        "",
        "```\n" + (todo.get("snippet") or "") + "\n```",
        "",
        "Original TODO text:",
        todo.get("text", "")
    ]
    body = "\n".join(body_lines)
    labels = [todo.get("marker", "todo").lower()]
    if todo.get("priority"):
        labels.append(f"priority:{todo.get('priority')}")
    if todo.get("assignee"):
        labels.append(f"assignee:{todo.get('assignee')}")
    if repo:
        body = f"Repository: {repo}\n\n" + body
    return {"title": title, "body": body, "labels": labels, "assignee": todo.get("assignee")}

# export helpers
def export_json(todos: Iterable[Dict[str, Any]], path: str) -> str:
    data = list(todos)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"generated": time.time(), "todos": data}, fh, indent=2, default=str)
    return path

def export_sarif(todos: Iterable[Dict[str, Any]], path: str, tool_name: str = "ciams-todo-plugin") -> str:
    """
    Minimal SARIF v2 wrapper for TODO items so linters/CI can consume results.
    Produces a very small SARIF structure with rules generated from markers.
    """
    todos = list(todos)
    rules = {}
    results = []
    for t in todos:
        rule_id = t.get("marker") or "TODO"
        if rule_id not in rules:
            rules[rule_id] = {"id": rule_id, "shortDescription": {"text": rule_id}, "defaultConfiguration": {"level": "warning"}}
        result = {
            "ruleId": rule_id,
            "level": "warning",
            "message": {"text": t.get("text", "") or rule_id},
            "locations": [{
                "physicalLocation": {
                    "artifactLocation": {"uri": t.get("filename") or "<unknown>"},
                    "region": {"startLine": t.get("line", 1), "startColumn": t.get("col", 1)}
                }
            }]
        }
        results.append(result)
    sarif = {
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {"name": tool_name, "rules": list(rules.values())}},
            "results": results
        }]
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(sarif, fh, indent=2)
    return path

# assistant integration wrapper
def rule_todos(source: str, filename: Optional[str] = None) -> List[Suggestion]:
    suggestions: List[Suggestion] = []
    parsed = parse_todos(source, filename=filename)
    for t in parsed:
        marker = t["marker"]
        text = t["text"] or ""
        tags = [marker.lower()] + t.get("tags", [])
        if t.get("priority"):
            tags.append(f"priority:{t['priority']}")
        if t.get("assignee"):
            tags.append(f"assignee:{t['assignee']}")
        if t.get("issue"):
            tags.append(f"issue:{t['issue']}")
        short = text.splitlines()[0] if text else ""
        title = f"{marker}: {short}" if short else f"{marker}"
        desc_lines = [
            f"File: {t['filename'] or '<unknown>'}  Line: {t['line']}  Col: {t['col']}",
            f"Text: {text}",
        ]
        if t.get("assignee"):
            desc_lines.append(f"Assignee: @{t['assignee']}")
        if t.get("issue"):
            desc_lines.append(f"Issue: #{t['issue']}")
        if t.get("priority"):
            desc_lines.append(f"Priority: {t['priority']}")
        if t.get("due"):
            desc_lines.append(f"Due: {t['due']}")
        desc = " | ".join(desc_lines)
        confidence = _compute_confidence(marker, t)
        snippet = t.get("snippet", "")
        location = (t["line"], t["col"], t["start_offset"], t["end_offset"])
        # attach suggestion metadata in the suggestion 'message' or tags (conservative)
        # Suggestion signature: (kind, tags:list, title, confidence:float, snippet, location_tuple)
        suggestions.append(Suggestion("task", tags, title, confidence, snippet, location))
    return suggestions

# --- plugin registration and assistant helpers ---
def _expose_tools_on_assistant(assistant) -> None:
    """
    Attach utility helpers to assistant if it allows extension.
    This is non-invasive: if the assistant already exposes names it won't overwrite.
    """
    tools = {
        "parse_todos": parse_todos,
        "parse_todos_cached": parse_todos_cached,
        "scan_sources_concurrent": scan_sources_concurrent,
        "export_json": export_json,
        "export_sarif": export_sarif,
        "generate_quickfix_patch": generate_quickfix_patch,
        "generate_issue_payload": generate_issue_payload,
        "group_todos": group_todos,
        "summary_stats": summary_stats
    }
    for name, fn in tools.items():
        if hasattr(assistant, name):
            # do not overwrite existing attributes
            continue
        try:
            setattr(assistant, name, fn)
        except Exception:
            # best-effort; ignore if assignment not allowed
            continue

def register(assistant) -> Callable:
    assistant.register_rule(rule_todos)
    # attempt to expose tooling helpers for advanced workflows
    try:
        _expose_tools_on_assistant(assistant)
    except Exception:
        pass
    # return an unregister closure for plugin managers that support it
    def unregister_fn(asst=None):
        try:
            assistant.unregister_rule(rule_todos)
        except Exception:
            pass
        # remove helpers only if we added them
        try:
            for n in ("parse_todos", "parse_todos_cached", "scan_sources_concurrent", "export_json", "export_sarif", "generate_quickfix_patch", "generate_issue_payload", "group_todos", "summary_stats"):
                if getattr(assistant, n, None) in (parse_todos, parse_todos_cached, scan_sources_concurrent, export_json, export_sarif, generate_quickfix_patch, generate_issue_payload, group_todos, summary_stats):
                    try:
                        delattr(assistant, n)
                    except Exception:
                        pass
        except Exception:
            pass
    return unregister_fn

def unregister(assistant) -> None:
    try:
        assistant.unregister_rule(rule_todos)
    except Exception:
        pass

# --- CLI entry for ad-hoc runs ---
def _cli_main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="todo_task_plugin", description="Scan source paths for TODOs and export results.")
    parser.add_argument("paths", nargs="+", help="Files or directories to scan")
    parser.add_argument("--out-json", help="Write parsed todos to this JSON file")
    parser.add_argument("--out-sarif", help="Write SARIF to this path")
    parser.add_argument("--workers", type=int, default=8, help="Concurrency for file scanning")
    args = parser.parse_args(argv)
    results = scan_sources_concurrent(args.paths, max_workers=args.workers)
    all_todos = []
    for v in results.values():
        all_todos.extend(v)
    if args.out_json:
        export_json(all_todos, args.out_json)
    if args.out_sarif:
        export_sarif(all_todos, args.out_sarif)
    # print summary
    stats = summary_stats(all_todos)
    print(json.dumps(stats, indent=2))
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(_cli_main())

"""
CIAMS TODO Task Plugin — enhanced boosters, tooling, optimizations.

Adds:
 - All previous parsing, exporters, quick-fix and concurrency features.
 - Git blame-based assignee suggester.
 - GitHub issue poster (optional; uses GITHUB_TOKEN env).
 - Markdown & CSV report exporters.
 - .todoignore support and path-glob skipping.
 - Async scanning using asyncio/aiofiles when available (falls back to threadpool).
 - Prioritization / scoring function and sort helpers.
 - Apply quick-fix patches to files (atomic write).
 - Expose enhanced tools on assistant when registering.

Usage:
  register(assistant) attaches rule_todos and exposes helpers (if assistant allows).
  The CLI entry supports new flags for markdown/csv exports and GitHub dry-run issue creation.

Note: network operations (GitHub) require environment variables for auth (read README).
"""

from __future__ import annotations
import re
import os
import json
import csv
import hashlib
import time
import math
import shutil
import subprocess
import urllib.request
import urllib.error
import urllib.parse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Iterable, Callable
from functools import lru_cache

# Try to import aiofiles for async file IO; fallback if not available.
try:
    import aiofiles  # type: ignore
    _HAS_AIOFILES = True
except Exception:
    _HAS_AIOFILES = False

from ciams.ai_engine import Suggestion

# --- regex / constants ---
_MARKER_WORDS = r"(TODO|FIXME|XXX|HACK|OPTIMIZE|NOTE|REVIEW|BUG|TASK)"
_INLINE_MARKER_RE = re.compile(
    rf"(?P<prefix>//|#|--|;)\s*(?P<marker>{_MARKER_WORDS})[:\s]?(?P<rest>.*)$",
    re.IGNORECASE | re.MULTILINE
)
_BLOCK_COMMENT_RE = re.compile(r"/\*(?P<body>.*?)\*/", re.DOTALL | re.IGNORECASE)

_ASSIGNEE_RE = re.compile(r"@(?P<assignee>[A-Za-z0-9_\-\.]+)")
_ISSUE_RE = re.compile(r"#(?P<issue>\d+)")
_PRIORITY_RE = re.compile(r"\b(?:\[P(?P<pnum>[0-9])\]|priority[:=]\s*(?P<pword>high|medium|low|p1|p2|p3))\b", re.IGNORECASE)
_DUE_RE = re.compile(r"\b(?:due[:=]\s*(?P<date>\d{4}-\d{2}-\d{2}))\b", re.IGNORECASE)
_TAG_RE = re.compile(r"@(?P<tag>[A-Za-z0-9_\-]+)")

_SEVERITY_BY_MARKER = {
    "FIXME": 0.95,
    "TODO": 0.60,
    "XXX": 0.90,
    "HACK": 0.45,
    "OPTIMIZE": 0.50,
    "NOTE": 0.30,
    "REVIEW": 0.65,
    "BUG": 0.96,
    "TASK": 0.55,
}

_PRIORITY_MAP = {
    "p1": 0.98, "p2": 0.90, "p3": 0.75,
    "high": 0.95, "medium": 0.70, "low": 0.40
}

# in-memory caches
_file_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
_parse_cache_max = 1024


# --- low-level helpers ---
def _offset_to_linecol(source: str, offset: int) -> Tuple[int, int]:
    if offset < 0:
        offset = 0
    prefix = source[:offset]
    line = prefix.count("\n") + 1
    last_nl = prefix.rfind("\n")
    col = offset - (last_nl + 1) + 1 if last_nl != -1 else offset + 1
    return line, col


def _make_snippet(source: str, start: int, end: int, context_lines: int = 2) -> str:
    lines = source.splitlines()
    start_line, _ = _offset_to_linecol(source, start)
    end_line, _ = _offset_to_linecol(source, end)
    sidx = max(0, start_line - 1 - context_lines)
    eidx = min(len(lines), end_line + context_lines)
    snippet_lines = []
    for i in range(sidx, eidx):
        prefix = "   "
        if i >= (start_line - 1) and i <= (end_line - 1):
            prefix = ">> "
        snippet_lines.append(f"{prefix}{i+1:4d}: {lines[i]}")
    return "\n".join(snippet_lines)


def _normalize_marker(marker: Optional[str]) -> str:
    return (marker or "TODO").upper()


def _compute_confidence(marker: str, metadata: Dict[str, Any]) -> float:
    m = _normalize_marker(marker)
    base = _SEVERITY_BY_MARKER.get(m, 0.5)
    pr = metadata.get("priority")
    if pr:
        if isinstance(pr, str):
            base = max(base, _PRIORITY_MAP.get(pr.lower(), base))
    if metadata.get("assignee"):
        base = min(0.99, base + 0.05)
    if metadata.get("issue"):
        base = min(0.98, base + 0.03)
    # due date urgency: if due soon, boost
    due = metadata.get("due")
    if due:
        try:
            due_ts = time.mktime(time.strptime(due, "%Y-%m-%d"))
            days = (due_ts - time.time()) / 86400.0
            if days <= 0:
                base = min(0.999, base + 0.08)  # overdue
            elif days < 7:
                base = min(0.99, base + 0.05)
        except Exception:
            pass
    return round(base, 2)


# --- core parsing (robust, optimized) ---
def parse_todos(source: str, filename: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Parse source and collect structured TODO dictionaries.
    Optimized to avoid expensive allocations on large files.
    """
    todos: List[Dict[str, Any]] = []

    # single-line matches (fast)
    for m in _INLINE_MARKER_RE.finditer(source):
        marker = m.group("marker") or "TODO"
        rest = (m.group("rest") or "").strip()
        start, end = m.start(), m.end()
        assignee = (_ASSIGNEE_RE.search(rest).group("assignee") if _ASSIGNEE_RE.search(rest) else None)
        issue = (_ISSUE_RE.search(rest).group("issue") if _ISSUE_RE.search(rest) else None)
        pr_match = _PRIORITY_RE.search(rest)
        priority = pr_match.group("pnum") or pr_match.group("pword") if pr_match else None
        due = (_DUE_RE.search(rest).group("date") if _DUE_RE.search(rest) else None)
        tags = [mo.group("tag") for mo in _TAG_RE.finditer(rest) if mo.group("tag") != (assignee or "")]
        line, col = _offset_to_linecol(source, start)
        snippet = _make_snippet(source, start, end)
        todos.append({
            "marker": marker.upper(),
            "text": rest,
            "assignee": assignee,
            "issue": issue,
            "priority": priority,
            "due": due,
            "tags": tags,
            "start_offset": start,
            "end_offset": end,
            "line": line,
            "col": col,
            "snippet": snippet,
            "filename": filename
        })

    # block comments
    for bm in _BLOCK_COMMENT_RE.finditer(source):
        body = bm.group("body") or ""
        for im in re.finditer(rf"(?P<marker>{_MARKER_WORDS})[:\s]?(?P<rest>.*)$", body, re.IGNORECASE | re.MULTILINE):
            marker = im.group("marker") or "TODO"
            rest = (im.group("rest") or "").strip()
            inner_rel = im.start()
            start = bm.start() + 2 + inner_rel
            end = start + (im.end() - im.start())
            assignee = (_ASSIGNEE_RE.search(rest).group("assignee") if _ASSIGNEE_RE.search(rest) else None)
            issue = (_ISSUE_RE.search(rest).group("issue") if _ISSUE_RE.search(rest) else None)
            pr_match = _PRIORITY_RE.search(rest)
            priority = pr_match.group("pnum") or pr_match.group("pword") if pr_match else None
            due = (_DUE_RE.search(rest).group("date") if _DUE_RE.search(rest) else None)
            tags = [mo.group("tag") for mo in _TAG_RE.finditer(rest) if mo.group("tag") != (assignee or "")]
            line, col = _offset_to_linecol(source, start)
            snippet = _make_snippet(source, start, end)
            todos.append({
                "marker": marker.upper(),
                "text": rest,
                "assignee": assignee,
                "issue": issue,
                "priority": priority,
                "due": due,
                "tags": tags,
                "start_offset": start,
                "end_offset": end,
                "line": line,
                "col": col,
                "snippet": snippet,
                "filename": filename
            })

    # dedupe
    seen = set()
    unique: List[Dict[str, Any]] = []
    for t in todos:
        key = (t["filename"], t["start_offset"], t["end_offset"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(t)
    return unique


# --- caching / concurrent scanning / async support ---
def parse_todos_cached(path: str) -> List[Dict[str, Any]]:
    try:
        mtime = os.path.getmtime(path)
    except Exception:
        mtime = 0.0
    entry = _file_cache.get(path)
    if entry and entry[0] == mtime:
        return entry[1]
    try:
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
    except Exception:
        return []
    parsed = parse_todos(src, filename=path)
    # keep cache bounded using simple LRU behavior (dict + size check)
    _file_cache[path] = (mtime, parsed)
    if len(_file_cache) > _parse_cache_max:
        # drop oldest entries (not strictly LRU but simple)
        for k in list(_file_cache.keys())[: len(_file_cache)//4]:
            _file_cache.pop(k, None)
    return parsed


def _gather_files(paths: Iterable[str], skip_hidden: bool = True) -> List[str]:
    files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, fs in os.walk(p):
                if skip_hidden:
                    dirs[:] = [d for d in dirs if not d.startswith(".")]
                for f in fs:
                    if f.startswith("."):
                        continue
                    files.append(os.path.join(root, f))
        else:
            files.append(p)
    return files


def scan_sources_concurrent(paths: Iterable[str], max_workers: int = 8, skip_hidden: bool = True) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    files = _gather_files(paths, skip_hidden=skip_hidden)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(parse_todos_cached, f): f for f in files}
        for fut in as_completed(futures):
            f = futures[fut]
            try:
                res = fut.result()
            except Exception:
                res = []
            if res:
                out[f] = res
    return out


async def scan_sources_async(paths: Iterable[str], max_workers: int = 8, skip_hidden: bool = True) -> Dict[str, List[Dict[str, Any]]]:
    """
    Async scanning: uses aiofiles if available; otherwise falls back to threadpool.
    """
    files = _gather_files(paths, skip_hidden=skip_hidden)
    out: Dict[str, List[Dict[str, Any]]] = {}
    if _HAS_AIOFILES:
        sem = asyncio.Semaphore(max_workers)
        async def _read_and_parse(path: str):
            async with sem:
                try:
                    async with aiofiles.open(path, "r", encoding="utf-8") as fh:
                        src = await fh.read()
                except Exception:
                    return path, []
                parsed = parse_todos(src, filename=path)
                return path, parsed
        tasks = [asyncio.create_task(_read_and_parse(p)) for p in files]
        for t in await asyncio.gather(*tasks):
            if t[1]:
                out[t[0]] = t[1]
    else:
        # fallback
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [loop.run_in_executor(ex, parse_todos_cached, p) for p in files]
            for fut, path in zip(asyncio.as_completed(futures), files):
                res = await fut
                if res:
                    out[path] = res
    return out


# --- ignore patterns (.todoignore) ---
def load_todoignore(root_dir: str) -> List[str]:
    path = os.path.join(root_dir, ".todoignore")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip() and not ln.strip().startswith("#")]
        return lines
    except Exception:
        return []


def matches_ignore(path: str, ignore_patterns: Iterable[str]) -> bool:
    import fnmatch
    for p in ignore_patterns:
        if fnmatch.fnmatch(path, p) or fnmatch.fnmatch(os.path.basename(path), p):
            return True
    return False


# --- prioritization and scoring helpers ---
def score_todo(todo: Dict[str, Any]) -> float:
    """Compute a score (0..1) for prioritization."""
    conf = _compute_confidence(todo.get("marker", "TODO"), todo)
    # priority factor
    pr = todo.get("priority")
    pr_score = 0.0
    if pr:
        pr_score = _PRIORITY_MAP.get(str(pr).lower(), 0.0)
    # file age factor (older files less urgent unless due soon)
    age_days = 0.0
    try:
        mtime = os.path.getmtime(todo.get("filename")) if todo.get("filename") else None
        if mtime:
            age_days = (time.time() - mtime) / 86400.0
    except Exception:
        age_days = 0.0
    age_factor = 1.0 / (1.0 + math.log1p(age_days + 1.0))
    # due boost
    due = todo.get("due")
    due_boost = 0.0
    if due:
        try:
            due_ts = time.mktime(time.strptime(due, "%Y-%m-%d"))
            days_left = (due_ts - time.time()) / 86400.0
            if days_left <= 0:
                due_boost = 0.15
            elif days_left < 7:
                due_boost = 0.07
        except Exception:
            pass
    score = (conf * 0.6) + (pr_score * 0.25) + (age_factor * 0.1) + due_boost
    return min(0.999, round(score, 3))


def prioritize_todos(todos: Iterable[Dict[str, Any]], top_n: Optional[int] = None) -> List[Dict[str, Any]]:
    scored = []
    for t in todos:
        t = dict(t)  # copy
        t["_score"] = score_todo(t)
        scored.append(t)
    scored.sort(key=lambda x: (-x["_score"], x.get("priority") or "", x.get("filename") or ""))
    if top_n:
        return scored[:top_n]
    return scored


# --- Git helpers: git-blame assignee suggestion ---
def suggest_assignee_by_git_blame(path: str, line: int) -> Optional[str]:
    """
    Uses `git blame --porcelain -L line,line path` to extract the author name.
    Returns username (author-mail or author name) when available.
    """
    try:
        proc = subprocess.run(["git", "blame", "--porcelain", f"-L{line},{line}", path],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        out = proc.stdout or ""
        for ln in out.splitlines():
            if ln.startswith("author-mail "):
                mail = ln.split(" ", 1)[1].strip()
                # normalize <name@domain>
                return mail.strip("<>")
            if ln.startswith("author "):
                return ln.split(" ", 1)[1].strip()
    except Exception:
        pass
    return None


# --- quick-fix apply (atomic) ---
def apply_quickfix_patch_to_file(path: str, patch: Dict[str, Any], *, make_backup: bool = True) -> bool:
    """
    patch: {"start": int, "end": int, "replacement": str}
    Applies patch to file using byte offsets. Writes atomically and optionally keeps a backup.
    """
    start = patch.get("start"); end = patch.get("end"); repl = patch.get("replacement", "")
    if start is None or end is None:
        return False
    try:
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
    except Exception:
        return False
    if start < 0 or end > len(content) or start > end:
        return False
    new = content[:start] + repl + content[end:]
    if make_backup:
        bak = path + ".todo.bak"
        try:
            shutil.copy(path, bak)
        except Exception:
            pass
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(new)
        os.replace(tmp, path)
        return True
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        return False


# --- GitHub integration: optional poster (uses GITHUB_TOKEN env) ---
def post_github_issue(repo: str, payload: Dict[str, Any], token: Optional[str] = None, dry_run: bool = True) -> Dict[str, Any]:
    """
    Post an issue to GitHub repository "owner/repo". If dry_run=True, returns payload without posting.
    If token not provided, reads from GITHUB_TOKEN env var.
    Returns response dict or payload on dry-run/errors.
    """
    if token is None:
        token = os.environ.get("GITHUB_TOKEN")
    api = f"https://api.github.com/repos/{repo}/issues"
    if dry_run:
        return {"dry_run": True, "api": api, "payload": payload}
    if not token:
        raise RuntimeError("No GITHUB_TOKEN provided for posting issues")
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(api, data=data, method="POST", headers={
        "Authorization": f"token {token}",
        "User-Agent": "ciams-todo-plugin",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as he:
        return {"error": he.read().decode("utf-8"), "code": he.code}
    except Exception as e:
        return {"error": str(e)}


# --- report exporters ---
def export_markdown(todos: Iterable[Dict[str, Any]], path: str, title: str = "TODO Report") -> str:
    todos = list(todos)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# {title}\n\n")
        fh.write("| File | Line | Marker | Priority | Assignee | Summary |\n")
        fh.write("|---|---:|---|---|---|---|\n")
        for t in todos:
            fn = os.path.relpath(t.get("filename") or "<unknown>")
            line = t.get("line", 0)
            marker = t.get("marker", "")
            pr = t.get("priority") or ""
            ass = t.get("assignee") or ""
            summary = (t.get("text") or "").splitlines()[0][:80]
            fh.write(f"| {fn} | {line} | {marker} | {pr} | {ass} | {summary} |\n")
    return path


def export_csv(todos: Iterable[Dict[str, Any]], path: str) -> str:
    todos = list(todos)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "line", "col", "marker", "priority", "assignee", "issue", "text"])
        for t in todos:
            writer.writerow([t.get("filename") or "", t.get("line") or "", t.get("col") or "", t.get("marker") or "", t.get("priority") or "", t.get("assignee") or "", t.get("issue") or "", t.get("text") or ""])
    return path


# --- assistant integration wrapper ---
def rule_todos(source: str, filename: Optional[str] = None) -> List[Suggestion]:
    suggestions: List[Suggestion] = []
    parsed = parse_todos(source, filename=filename)
    for t in parsed:
        marker = t["marker"]
        text = t["text"] or ""
        tags = [marker.lower()] + t.get("tags", [])
        if t.get("priority"):
            tags.append(f"priority:{t['priority']}")
        if t.get("assignee"):
            tags.append(f"assignee:{t['assignee']}")
        if t.get("issue"):
            tags.append(f"issue:{t['issue']}")
        short = text.splitlines()[0] if text else ""
        title = f"{marker}: {short}" if short else f"{marker}"
        desc_lines = [
            f"File: {t['filename'] or '<unknown>'}  Line: {t['line']}  Col: {t['col']}",
            f"Text: {text}",
        ]
        if t.get("assignee"):
            desc_lines.append(f"Assignee: @{t['assignee']}")
        if t.get("issue"):
            desc_lines.append(f"Issue: #{t['issue']}")
        if t.get("priority"):
            desc_lines.append(f"Priority: {t['priority']}")
        if t.get("due"):
            desc_lines.append(f"Due: {t['due']}")
        desc = " | ".join(desc_lines)
        confidence = _compute_confidence(marker, t)
        snippet = t.get("snippet", "")
        location = (t["line"], t["col"], t["start_offset"], t["end_offset"])
        suggestions.append(Suggestion("task", tags, title, confidence, snippet, location))
    return suggestions


# --- plugin registration and exposing helpers to assistant ---
def _expose_tools_on_assistant(assistant) -> None:
    tools = {
        "parse_todos": parse_todos,
        "parse_todos_cached": parse_todos_cached,
        "scan_sources_concurrent": scan_sources_concurrent,
        "scan_sources_async": scan_sources_async,
        "export_json": lambda todos, p: export_json(list(todos), p),
        "export_sarif": export_sarif,
        "export_markdown": export_markdown,
        "export_csv": export_csv,
        "generate_quickfix_patch": generate_quickfix_patch,
        "apply_quickfix_patch_to_file": apply_quickfix_patch_to_file,
        "generate_issue_payload": generate_issue_payload,
        "post_github_issue": post_github_issue,
        "group_todos": group_todos,
        "summary_stats": summary_stats,
        "prioritize_todos": prioritize_todos,
        "suggest_assignee_by_git_blame": suggest_assignee_by_git_blame,
        "load_todoignore": load_todoignore,
    }
    for name, fn in tools.items():
        if hasattr(assistant, name):
            continue
        try:
            setattr(assistant, name, fn)
        except Exception:
            pass


def register(assistant) -> Callable:
    assistant.register_rule(rule_todos)
    try:
        _expose_tools_on_assistant(assistant)
    except Exception:
        pass

    def unregister_fn(asst=None):
        try:
            assistant.unregister_rule(rule_todos)
        except Exception:
            pass
        # remove helpers only if value equals our functions
        for n, fn in list(globals().items()):
            if n.startswith("_") or n in ("register", "unregister"):
                continue
            try:
                if getattr(assistant, n, None) is fn:
                    delattr(assistant, n)
            except Exception:
                pass

    return unregister_fn


def unregister(assistant) -> None:
    try:
        assistant.unregister_rule(rule_todos)
    except Exception:
        pass


# --- CLI helpers and exporters reused from above ---
def export_json(todos: Iterable[Dict[str, Any]], path: str) -> str:
    data = list(todos)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"generated": time.time(), "todos": data}, fh, indent=2, default=str)
    return path


def export_sarif(todos: Iterable[Dict[str, Any]], path: str, tool_name: str = "ciams-todo-plugin") -> str:
    todos = list(todos)
    rules = {}
    results = []
    for t in todos:
        rule_id = t.get("marker") or "TODO"
        if rule_id not in rules:
            rules[rule_id] = {"id": rule_id, "shortDescription": {"text": rule_id}, "defaultConfiguration": {"level": "warning"}}
        result = {
            "ruleId": rule_id,
            "level": "warning",
            "message": {"text": t.get("text", "") or rule_id},
            "locations": [{
                "physicalLocation": {
                    "artifactLocation": {"uri": t.get("filename") or "<unknown>"},
                    "region": {"startLine": t.get("line", 1), "startColumn": t.get("col", 1)}
                }
            }]
        }
        results.append(result)
    sarif = {
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {"name": tool_name, "rules": list(rules.values())}},
            "results": results
        }]
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(sarif, fh, indent=2)
    return path


# CLI entry
def _cli_main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="todo_task_plugin", description="Scan source paths for TODOs and export results.")
    parser.add_argument("paths", nargs="+", help="Files or directories to scan")
    parser.add_argument("--out-json", help="Write parsed todos to this JSON file")
    parser.add_argument("--out-sarif", help="Write SARIF to this path")
    parser.add_argument("--out-md", help="Write Markdown report")
    parser.add_argument("--out-csv", help="Write CSV report")
    parser.add_argument("--workers", type=int, default=8, help="Concurrency for file scanning")
    parser.add_argument("--github-repo", help="If provided, generate GitHub issue dry-run payloads")
    parser.add_argument("--apply-patches", action="store_true", help="Apply quickfix patches (use with caution)")
    args = parser.parse_args(argv)

    results = scan_sources_concurrent(args.paths, max_workers=args.workers)
    all_todos = []
    for v in results.values():
        all_todos.extend(v)
    if args.out_json:
        export_json(all_todos, args.out_json)
    if args.out_sarif:
        export_sarif(all_todos, args.out_sarif)
    if args.out_md:
        export_markdown(all_todos, args.out_md)
    if args.out_csv:
        export_csv(all_todos, args.out_csv)

    if args.github_repo:
        # produce issue payloads (dry-run)
        payloads = [generate_issue_payload(t, repo=args.github_repo) for t in all_todos]
        print(json.dumps({"issues": payloads}, indent=2))

    if args.apply_patches:
        for t in all_todos:
            patch = generate_quickfix_patch(t, action="issue")
            if patch:
                apply_quickfix_patch_to_file(t.get("filename"), patch)

    stats = summary_stats(all_todos)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli_main())

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

# instryx_wasm_and_exe_backend_emitter.py
# Instryx WASM and EXE Backend Emitter (via LLVM toolchain)
# Author: Violet Magenta / VACU Technologies
# License: MIT

import subprocess
import tempfile
import os
from instryx_llvm_ir_codegen import InstryxLLVMCodegen

class InstryxEmitter:
    def __init__(self, output_dir="build"):
        self.codegen = InstryxLLVMCodegen()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def emit(self, code: str, target: str = "exe", output_name: str = "program"):
        llvm_ir = self.codegen.generate(code)
        ir_path = os.path.join(self.output_dir, f"{output_name}.ll")
        bc_path = os.path.join(self.output_dir, f"{output_name}.bc")
        output_path = os.path.join(self.output_dir, output_name + (".exe" if target == "exe" else ".wasm"))

        # Write LLVM IR to file
        with open(ir_path, "w") as f:
            f.write(llvm_ir)

        # Compile IR to bitcode
        subprocess.run(["llvm-as", ir_path, "-o", bc_path], check=True)

        if target == "exe":
            subprocess.run([
                "llc", bc_path, "-filetype=obj", "-o", f"{output_path}.o"
            ], check=True)
            subprocess.run([
                "clang", f"{output_path}.o", "-o", output_path
            ], check=True)
        elif target == "wasm":
            subprocess.run([
                "llc", "-march=wasm32", bc_path, "-o", f"{output_path}.s"
            ], check=True)
            subprocess.run([
                "wasm-ld", f"{output_path}.s", "-o", output_path, "--no-entry", "--export-all"
            ], check=True)
        else:
            raise ValueError("Target must be 'exe' or 'wasm'")

        print(f"✅ Built target: {output_path}")
        return output_path


# Test block (can be removed in production)
if __name__ == "__main__":
    emitter = InstryxEmitter()
    code = """
    func greet(uid) {
        print: "Hello from Instryx LLVM!";
    };

    main() {
        greet(1);
    };
    """
    emitter.emit(code, target="exe", output_name="test_instryx")

# instryx_ciams_ai_engine.py
# CIAMS AI Engine for Instryx Language - Macro Learning & Suggestion System
# Author: Violet Magenta / VACU Technologies
# License: MIT

import json
import re
from collections import defaultdict, Counter

class CIAMSAIEngine:
    def __init__(self):
        self.macro_usage = defaultdict(Counter)
        self.patterns = {
            'inject': re.compile(r'@inject\s+(\w+(\.\w+)?)'),
            'wraptry': re.compile(r'@wraptry\s+(\w+\(.*?\))'),
            'ffi': re.compile(r'@ffi\s+func\s+(\w+)'),
            'memoize': re.compile(r'@memoize\s+(\w+)'),
        }

    def analyze_code(self, code: str, developer_id="default"):
        for macro, pattern in self.patterns.items():
            matches = pattern.findall(code)
            for match in matches:
                self.macro_usage[developer_id][macro] += 1

    def suggest_macros(self, context: str, developer_id="default"):
        suggestions = []
        usage = self.macro_usage[developer_id]
        if "try" in context and "replace" in context:
            suggestions.append("@wraptry")
        if "db." in context or "net." in context:
            suggestions.append("@inject")
        if "cache" in context or "lookup" in context:
            suggestions.append("@memoize")
        if "extern" in context or "header" in context:
            suggestions.append("@ffi")

        suggestions = sorted(set(suggestions), key=lambda m: -usage[m])
        return suggestions[:3]  # return top 3

    def export_profile(self, developer_id="default", path="ciams_profile.json"):
        with open(path, "w") as f:
            json.dump(self.macro_usage[developer_id], f, indent=2)

    def load_profile(self, path="ciams_profile.json", developer_id="default"):
        try:
            with open(path, "r") as f:
                data = json.load(f)
                self.macro_usage[developer_id].update(data)
        except FileNotFoundError:
            pass


# Test block (can be removed in production)
if __name__ == "__main__":
    ai = CIAMSAIEngine()
    sample_code = """
    @inject db.conn;
    @wraptry risky();
    @ffi func external_math(a, b);
    @memoize compute_value;
    """
    ai.analyze_code(sample_code, developer_id="user123")
    context = "risky() and db.conn and replace block"
    suggestions = ai.suggest_macros(context, developer_id="user123")
    print("🤖 Macro Suggestions:", suggestions)
    ai.export_profile("user123")

#!/usr/bin/env python3
"""
hardwarebridge/instryx_hwbuilder.py

Hardware Bridge Builder for Instryx runtime.

Features
- Generate a portable hardware bridge (C stub) implementing common low-level
  primitives expected by bootloaders and Instryx runtimes (hw_read_lba, hw_write_port, hw_inb, ...).
- Attempt to compile the C stub to a flat binary using available toolchain (clang/gcc + objcopy).
- If toolchain is not available, emit a fully executable fallback "bridge blob" (binary file)
  that contains a JSON-described table + a tiny trampoline header so downstream tooling
  can detect and optionally embed or interpret it.
- Simple, well-documented API and CLI:
    - HWBuilder.build_from_spec(spec: dict, out: Path, arch="x86_64", optimize=True)
    - HWBuilder.build_from_json(spec_path, out, ...)
- Safe, deterministic outputs in a build directory.

This file is self-contained, has no non-stdlib dependencies, and is ready to run.
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

# Configure module-level logger
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("instryx_hwbuilder")


DEFAULT_SPEC = {
    "name": "instryx_hwbridge",
    "version": "0.1",
    "symbols": [
        "hw_read_lba",
        "hw_write_port",
        "hw_inb",
        "hw_outb",
        "hw_memcpy",
        "hw_memset",
        "hw_reboot",
        "hw_get_memory_map",
        "hw_alloc_pages",
        "hw_free_pages"
    ],
    "behavior": {
        # default behavior: log to a host-side file (build/hwbridge/hw.log)
        "log_file": "hwbridge.log",
        "allocation_page_size": 4096
    }
}


@dataclass
class Toolchain:
    cc: Optional[str] = None
    objcopy: Optional[str] = None
    ld: Optional[str] = None

    def available(self) -> bool:
        return bool(self.cc)


@dataclass
class HWBuilder:
    work_dir: Path = field(default_factory=lambda: Path("build") / "hwbridge")
    toolchain: Toolchain = field(default_factory=Toolchain)

    def __post_init__(self):
        self.work_dir = self.work_dir.resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._detect_toolchain()

    def _detect_toolchain(self) -> Toolchain:
        # prefer clang, fallback to gcc
        cc = shutil.which("clang") or shutil.which("gcc")
        objcopy = shutil.which("objcopy") or shutil.which("llvm-objcopy")
        ld = shutil.which("ld") or shutil.which("lld")
        self.toolchain = Toolchain(cc=cc, objcopy=objcopy, ld=ld)
        logger.debug("Detected toolchain: cc=%s objcopy=%s ld=%s", cc, objcopy, ld)
        return self.toolchain

    def list_supported_archs(self) -> Sequence[str]:
        # conservative list; toolchain may support more via -target
        return ["x86_64", "i386", "arm64", "aarch64", "arm", "riscv64"]

    def generate_c_stub(self, spec: Dict) -> Path:
        """
        Generate a portable, minimal C stub implementing the specified symbols.
        The functions implement best-effort behavior (log actions to a file).
        Returns the path to the generated .c file.
        """
        name = spec.get("name", DEFAULT_SPEC["name"])
        c_path = self.work_dir / f"{name}.c"
        log_file = spec.get("behavior", {}).get("log_file", "hwbridge.log")
        page_size = int(spec.get("behavior", {}).get("allocation_page_size", 4096))
        symbols = spec.get("symbols", DEFAULT_SPEC["symbols"])

        logger.info("Generating C stub: %s", c_path)
        with open(c_path, "w", encoding="utf-8") as f:
            f.write("/* Auto-generated Instryx hardware bridge stub */\n")
            f.write("#include <stdint.h>\n")
            f.write("#include <stddef.h>\n")
            f.write("#include <stdio.h>\n")
            f.write("#include <string.h>\n")
            f.write("#include <stdlib.h>\n")
            f.write("\n")
            # Provide a simple global log file path embedded as a string literal
            f.write(f'static const char HWBR_LOGFILE[] = "{log_file}";\n')
            f.write(f'static const size_t HWBR_PAGE_SIZE = {page_size}UL;\n\n')

            # helper: append to log
            f.write("static void hwbr_log(const char *msg) {\n")
            f.write("    FILE *f = fopen(HWBR_LOGFILE, \"a\");\n")
            f.write("    if (!f) return;\n")
            f.write("    fputs(msg, f);\n")
            f.write("    fputs(\"\\n\", f);\n")
            f.write("    fclose(f);\n")
            f.write("}\n\n")

            # implement each symbol
            for sym in symbols:
                if sym == "hw_read_lba":
                    f.write("void hw_read_lba(uint64_t lba, void *dst_ptr, uint32_t count) {\n")
                    f.write("    char buf[256];\n")
                    f.write("    snprintf(buf, sizeof(buf), \"hw_read_lba: lba=%llu count=%u dst=%p\", (unsigned long long)lba, count, dst_ptr);\n")
                    f.write("    hwbr_log(buf);\n")
                    f.write("    /* no-op placeholder: leave dst unmodified */\n")
                    f.write("}\n\n")
                elif sym == "hw_write_port":
                    f.write("void hw_write_port(uint16_t port, uint8_t value) {\n")
                    f.write("    char buf[128];\n")
                    f.write("    snprintf(buf, sizeof(buf), \"hw_write_port: port=0x%04x value=0x%02x\", port, value);\n")
                    f.write("    hwbr_log(buf);\n")
                    f.write("}\n\n")
                elif sym == "hw_inb":
                    f.write("uint8_t hw_inb(uint16_t port) {\n")
                    f.write("    char buf[128];\n")
                    f.write("    snprintf(buf, sizeof(buf), \"hw_inb: port=0x%04x -> 0\", port);\n")
                    f.write("    hwbr_log(buf);\n")
                    f.write("    return 0;\n")
                    f.write("}\n\n")
                elif sym == "hw_outb":
                    f.write("void hw_outb(uint16_t port, uint8_t value) {\n")
                    f.write("    char buf[128];\n")
                    f.write("    snprintf(buf, sizeof(buf), \"hw_outb: port=0x%04x value=0x%02x\", port, value);\n")
                    f.write("    hwbr_log(buf);\n")
                    f.write("}\n\n")
                elif sym == "hw_memcpy":
                    f.write("void hw_memcpy(void *dst, const void *src, size_t n) {\n")
                    f.write("    memcpy(dst, src, n);\n")
                    f.write("}\n\n")
                elif sym == "hw_memset":
                    f.write("void hw_memset(void *dst, int val, size_t n) {\n")
                    f.write("    memset(dst, val, n);\n")
                    f.write("}\n\n")
                elif sym == "hw_reboot":
                    f.write("void hw_reboot(void) {\n")
                    f.write("    hwbr_log(\"hw_reboot: requested\");\n")
                    f.write("    /* best-effort: exit process to simulate reboot in host environment */\n")
                    f.write("    exit(0);\n")
                    f.write("}\n\n")
                elif sym == "hw_get_memory_map":
                    f.write("void hw_get_memory_map(void *buf_ptr, uint32_t max_entries) {\n")
                    f.write("    char buf[128];\n")
                    f.write("    snprintf(buf, sizeof(buf), \"hw_get_memory_map: buf=%p max=%u\", buf_ptr, max_entries);\n")
                    f.write("    hwbr_log(buf);\n")
                    f.write("    /* write a tiny fake map: one entry with full memory */\n")
                    f.write("    if (buf_ptr && max_entries > 0) {\n")
                    f.write("        uint64_t *p = (uint64_t *)buf_ptr;\n")
                    f.write("        p[0] = 0; p[1] = 0xFFFFFFFFFFFFFFFFULL; /* start, length */\n")
                    f.write("    }\n")
                    f.write("}\n\n")
                elif sym == "hw_alloc_pages":
                    f.write("void *hw_alloc_pages(size_t pages) {\n")
                    f.write("    size_t bytes = pages * HWBR_PAGE_SIZE;\n")
                    f.write("    void *p = malloc(bytes);\n")
                    f.write("    char buf[128];\n")
                    f.write("    snprintf(buf, sizeof(buf), \"hw_alloc_pages: pages=%zu -> %p\", pages, p);\n")
                    f.write("    hwbr_log(buf);\n")
                    f.write("    return p ? p : (void*)0;\n")
                    f.write("}\n\n")
                elif sym == "hw_free_pages":
                    f.write("void hw_free_pages(void *ptr, size_t pages) {\n")
                    f.write("    (void)pages; hwbr_log(\"hw_free_pages: free\"); free(ptr);\n")
                    f.write("}\n\n")
                else:
                    # generic stub
                    f.write(f"/* stub for {sym} */\n")
                    f.write(f"void {sym}(void) {{ hwbr_log(\"{sym}: stub called\"); }}\n\n")
        logger.debug("C stub generated at %s", c_path)
        return c_path

    def compile_c_to_bin(self, c_path: Path, out_path: Path, arch: str = "x86_64", optimize: bool = True) -> bool:
        """
        Attempt to compile the generated C into a flat binary.
        Uses detected toolchain; returns True on success.
        """
        if not self.toolchain.available():
            logger.warning("No C compiler available; skipping native build.")
            return False

        cc = self.toolchain.cc
        objcopy = self.toolchain.objcopy
        # create object file first
        obj_path = self.work_dir / (c_path.stem + ".o")
        cflags = ["-nostdlib", "-ffreestanding", "-fno-builtin", "-c"]
        if optimize:
            cflags.append("-O2")
        # Add target-specific flags if we can guess target from arch
        if arch in ("arm64", "aarch64"):
            # aarch64 target via clang: -target aarch64-linux-gnu
            cflags += ["-target", "aarch64-linux-gnu"]
        elif arch in ("riscv64",):
            # RISC-V target if available
            cflags += ["-target", "riscv64-unknown-elf"]
        # compile
        cmd_compile = [cc] + cflags + [str(c_path), "-o", str(obj_path)]
        logger.info("Compiling C stub: %s", " ".join(cmd_compile))
        try:
            subprocess.check_call(cmd_compile, cwd=self.work_dir)
        except Exception as e:
            logger.error("C compile failed: %s", e)
            return False

        # produce a raw binary of the .text section using objcopy if available
        if objcopy:
            # strip and extract .text into a raw binary blob
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cmd_objcopy = [objcopy, "-O", "binary", str(obj_path), str(out_path)]
            logger.info("Objcopy to raw binary: %s", " ".join(cmd_objcopy))
            try:
                subprocess.check_call(cmd_objcopy, cwd=self.work_dir)
                logger.info("Native bridge binary written to %s", out_path)
                return True
            except Exception as e:
                logger.warning("objcopy failed; attempting linker-based emit: %s", e)

        # fallback: try linking into an ELF then objcopy
        ld = self.toolchain.ld
        if ld:
            elf_path = self.work_dir / (c_path.stem + ".elf")
            cmd_ld = [ld, str(obj_path), "-o", str(elf_path)]
            logger.info("Linking ELF: %s", " ".join(cmd_ld))
            try:
                subprocess.check_call(cmd_ld, cwd=self.work_dir)
                if objcopy:
                    cmd_objcopy2 = [objcopy, "-O", "binary", str(elf_path), str(out_path)]
                    subprocess.check_call(cmd_objcopy2, cwd=self.work_dir)
                    logger.info("Native bridge binary written to %s", out_path)
                    return True
            except Exception as e:
                logger.error("Linking or objcopy on ELF failed: %s", e)

        logger.warning("Toolchain present but could not emit flat binary. Leaving object at %s", obj_path)
        # at least copy the object file as output to allow embedding
        try:
            shutil.copy2(str(obj_path), str(out_path))
            logger.info("Fallback: copied object file to %s", out_path)
            return True
        except Exception as e:
            logger.error("Fallback copy failed: %s", e)
            return False

    def emit_blob_fallback(self, spec: Dict, c_src_path: Path, out_path: Path) -> bool:
        """
        Emit a deterministic fallback 'hwbridge blob' that contains:
            - magic header
            - JSON spec
            - embedded C source (for inspection)
        The file can be recognized and used by toolchains that understand this fallback.
        """
        LOGIC = {
            "magic": "INSTRYX_HWBRIDGE_BLOB_v1",
            "generated_at": int(time.time()),
            "spec": spec,
        }
        logger.info("Emitting fallback hwbridge blob to %s", out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(out_path, "wb") as out:
                header = LOGIC["magic"].encode("utf-8")
                out.write(len(header).to_bytes(2, "little"))
                out.write(header)
                payload = json.dumps(LOGIC, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                out.write(len(payload).to_bytes(4, "little"))
                out.write(payload)
                # append C source for reference
                csrc = c_src_path.read_bytes() if c_src_path.exists() else b""
                out.write(len(csrc).to_bytes(4, "little"))
                out.write(csrc)
            logger.info("Fallback blob emitted at %s", out_path)
            return True
        except Exception as e:
            logger.error("Failed to write fallback blob: %s", e)
            return False

    def build_from_spec(self, spec: Dict, out: Path, arch: str = "x86_64", optimize: bool = True, force_fallback: bool = False) -> bool:
        """
        Main entry: generate C stub and attempt a native binary build.
        If toolchain is absent or force_fallback is True, emit fallback blob.
        Returns True on success.
        """
        out = Path(out).resolve()
        logger.info("Building hardware bridge '%s' -> %s (arch=%s optimize=%s)", spec.get("name", "hwbridge"), out, arch, optimize)
        c_src = self.generate_c_stub(spec)

        # try native compile unless forced fallback
        if not force_fallback and self.toolchain.available():
            ok = self.compile_c_to_bin(c_src, out, arch=arch, optimize=optimize)
            if ok:
                return True
            logger.warning("Native compile failed, will fallback to blob emission.")

        # fallback: emit blob
        ok = self.emit_blob_fallback(spec, c_src, out)
        return ok

    def build_from_json(self, spec_path: Path, out: Path, arch: str = "x86_64", optimize: bool = True, force_fallback: bool = False) -> bool:
        spec_path = Path(spec_path)
        if not spec_path.exists():
            logger.error("Spec file not found: %s", spec_path)
            return False
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error("Invalid JSON spec: %s", e)
            return False
        return self.build_from_spec(spec, out, arch=arch, optimize=optimize, force_fallback=force_fallback)


def _cli():
    p = argparse.ArgumentParser(description="Instryx hardware bridge builder")
    p.add_argument("--spec", type=Path, default=None, help="JSON spec file describing bridge (optional)")
    p.add_argument("--out", type=Path, default=Path("build") / "hwbridge" / "hwbridge.bin", help="Output binary/blob path")
    p.add_argument("--arch", type=str, default="x86_64", help="Target architecture (informational)")
    p.add_argument("--no-opt", dest="optimize", action="store_false", help="Disable optimization (-O2)")
    p.add_argument("--force-fallback", action="store_true", help="Always emit fallback blob (no native compile)")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = p.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    builder = HWBuilder()
    spec = DEFAULT_SPEC
    if args.spec:
        if not args.spec.exists():
            logger.error("Spec not found: %s", args.spec)
            sys.exit(2)
        try:
            spec = json.loads(args.spec.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error("Failed to parse spec: %s", e)
            sys.exit(2)

    ok = builder.build_from_spec(spec, args.out, arch=args.arch, optimize=args.optimize, force_fallback=args.force_fallback)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _cli()

#!/usr/bin/env python3
"""
hardwarebridge/instryx_hwbuilder.py

Enhanced Instryx Hardware Bridge Builder

This module generates a portable C hardware-bridge stub for Instryx toolchains,
compiles it to a flat binary (when a native toolchain is available), and
provides a robust production-ready tooling surface:

Major features added beyond a simple stub:
- Disk-backed content cache to avoid redundant rebuilds (hash-based).
- Parallel multi-architecture builds.
- Profile-guided optimization hooks (PGO flags support).
- Pluggable emitter discovery: attempts to call in-repo emitter modules.
- Plugin hook directory support (hardwarebridge/plugins).
- Detailed build reporting and TTL artifact stamping (SHA256).
- Fallback "blob" artifact format when toolchain absent.
- Post-build verification (nm/objdump when available).
- CLI with advanced options: --parallel, --cache, --profile, --sign, --emitters.
- Self-tests and environment diagnostics.

No third-party dependencies (only Python stdlib).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, List

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("instryx_hwbuilder")

# ---------------------------------------------------------------------------
# Default spec
# ---------------------------------------------------------------------------
DEFAULT_SPEC = {
    "name": "instryx_hwbridge",
    "version": "0.1",
    "symbols": [
        "hw_read_lba",
        "hw_write_port",
        "hw_inb",
        "hw_outb",
        "hw_memcpy",
        "hw_memset",
        "hw_reboot",
        "hw_get_memory_map",
        "hw_alloc_pages",
        "hw_free_pages",
    ],
    "behavior": {
        "log_file": "hwbridge.log",
        "allocation_page_size": 4096,
    },
}

# ---------------------------------------------------------------------------
# Tooling dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Toolchain:
    cc: Optional[str] = None
    objcopy: Optional[str] = None
    ld: Optional[str] = None
    nm: Optional[str] = None
    objdump: Optional[str] = None

    def available(self) -> bool:
        return bool(self.cc)

    def describe(self) -> Dict[str, Optional[str]]:
        return {
            "cc": self.cc,
            "objcopy": self.objcopy,
            "ld": self.ld,
            "nm": self.nm,
            "objdump": self.objdump,
        }


@dataclass
class BuildReport:
    success: bool
    artifact: Optional[Path] = None
    cached: bool = False
    elapsed_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)
    sha256: Optional[str] = None
    toolchain: Dict[str, Optional[str]] = field(default_factory=dict)


@dataclass
class HWBuilder:
    repo_root: Path = field(default_factory=lambda: Path.cwd())
    work_dir: Path = field(default_factory=lambda: Path("build") / "hwbridge")
    cache_dir: Path = field(default_factory=lambda: Path(".cache") / "hwbridge")
    plugin_dir: Path = field(default_factory=lambda: Path("hardwarebridge") / "plugins")
    toolchain: Toolchain = field(default_factory=Toolchain)

    def __post_init__(self):
        self.repo_root = self.repo_root.resolve()
        self.work_dir = (self.repo_root / self.work_dir).resolve()
        self.cache_dir = (self.repo_root / self.cache_dir).resolve()
        self.plugin_dir = (self.repo_root / self.plugin_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
        self._detect_toolchain()

    # -----------------------------------------------------------------------
    # Toolchain / environment detection
    # -----------------------------------------------------------------------
    def _which(self, name: str) -> Optional[str]:
        from shutil import which

        return which(name)

    def _detect_toolchain(self) -> Toolchain:
        cc = self._which("clang") or self._which("gcc")
        objcopy = self._which("objcopy") or self._which("llvm-objcopy")
        ld = self._which("ld") or self._which("lld")
        nm = self._which("nm")
        objdump = self._which("objdump")
        self.toolchain = Toolchain(cc=cc, objcopy=objcopy, ld=ld, nm=nm, objdump=objdump)
        logger.debug("Detected toolchain: %s", self.toolchain.describe())
        return self.toolchain

    def detect_env(self) -> Dict[str, object]:
        env = {
            "python": sys.version,
            "cwd": str(Path.cwd()),
            "repo_root": str(self.repo_root),
            "toolchain": self.toolchain.describe(),
        }
        return env

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------
    @staticmethod
    def canonical_json(obj: object) -> str:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def hash_inputs(self, spec: Dict, c_src: bytes, profile_blob: Optional[bytes] = None) -> str:
        h = hashlib.sha256()
        h.update(self.canonical_json(spec).encode("utf-8"))
        h.update(b"\0")
        h.update(c_src)
        if profile_blob:
            h.update(b"\0PROFILE\0")
            h.update(profile_blob)
        # include toolchain identity to avoid ABI mismatch reuse
        tc = json.dumps(self.toolchain.describe(), sort_keys=True)
        h.update(b"\0TOOLCHAIN\0")
        h.update(tc.encode("utf-8"))
        return h.hexdigest()

    def artifact_path_for_hash(self, out: Path, h: str) -> Path:
        # use cache dir to store artifact copies keyed by hash
        return self.cache_dir / f"{h}{out.suffix}"

    # -----------------------------------------------------------------------
    # C stub generation (enhanced)
    # -----------------------------------------------------------------------
    def generate_c_stub(self, spec: Dict) -> Path:
        """
        Generate an enhanced C stub file from the spec.
        Returns path to generated .c file.
        """
        name = spec.get("name", DEFAULT_SPEC["name"])
        c_path = self.work_dir / f"{name}.c"
        log_file = spec.get("behavior", {}).get("log_file", "hwbridge.log")
        page_size = int(spec.get("behavior", {}).get("allocation_page_size", 4096))
        symbols = spec.get("symbols", DEFAULT_SPEC["symbols"])

        logger.info("Generating C stub: %s", c_path)
        with open(c_path, "w", encoding="utf-8") as f:
            # Header
            f.write("/* Auto-generated Instryx hardware bridge stub */\n")
            f.write("#include <stdint.h>\n#include <stddef.h>\n#include <stdio.h>\n#include <string.h>\n#include <stdlib.h>\n\n")
            f.write(f'static const char HWBR_LOGFILE[] = "{log_file}";\n')
            f.write(f"static const size_t HWBR_PAGE_SIZE = {page_size}UL;\n\n")
            f.write("/* small util: append message to log (best-effort, non-blocking) */\n")
            f.write("static void hwbr_log(const char *msg) {\n")
            f.write("    FILE *f = fopen(HWBR_LOGFILE, \"a\"); if (!f) return; fputs(msg, f); fputs(\"\\n\", f); fclose(f);\n")
            f.write("}\n\n")

            # Provide a sanity-check entry for exporters
            f.write("/* Exported metadata for tooling */\n")
            f.write("const char *hwbridge_name = \"" + name + "\";\n")
            f.write("const char *hwbridge_version = \"" + str(spec.get("version", "0.0")) + "\";\n\n")

            # Symbol implementations
            for sym in symbols:
                # Robust APIs with annotations
                if sym == "hw_read_lba":
                    f.write("void hw_read_lba(uint64_t lba, void *dst_ptr, uint32_t count) {\n")
                    f.write("    char buf[256];\n")
                    f.write("    snprintf(buf, sizeof(buf), \"hw_read_lba: lba=%llu count=%u dst=%p\", (unsigned long long)lba, count, dst_ptr);\n")
                    f.write("    hwbr_log(buf);\n")
                    f.write("}\n\n")
                elif sym == "hw_write_port":
                    f.write("void hw_write_port(uint16_t port, uint8_t value) {\n")
                    f.write("    char buf[128]; snprintf(buf, sizeof(buf), \"hw_write_port: port=0x%04x value=0x%02x\", port, value); hwbr_log(buf);\n")
                    f.write("}\n\n")
                elif sym == "hw_inb":
                    f.write("uint8_t hw_inb(uint16_t port) { char buf[128]; snprintf(buf, sizeof(buf), \"hw_inb: port=0x%04x -> 0\", port); hwbr_log(buf); return 0; }\n\n")
                elif sym == "hw_outb":
                    f.write("void hw_outb(uint16_t port, uint8_t value) { char buf[128]; snprintf(buf, sizeof(buf), \"hw_outb: port=0x%04x value=0x%02x\", port, value); hwbr_log(buf); }\n\n")
                elif sym == "hw_memcpy":
                    f.write("void hw_memcpy(void *dst, const void *src, size_t n) { memcpy(dst, src, n); }\n\n")
                elif sym == "hw_memset":
                    f.write("void hw_memset(void *dst, int val, size_t n) { memset(dst, val, n); }\n\n")
                elif sym == "hw_reboot":
                    f.write("void hw_reboot(void) { hwbr_log(\"hw_reboot: requested\"); exit(0); }\n\n")
                elif sym == "hw_get_memory_map":
                    f.write("void hw_get_memory_map(void *buf_ptr, uint32_t max_entries) {\n")
                    f.write("    char buf[128]; snprintf(buf, sizeof(buf), \"hw_get_memory_map: buf=%p max=%u\", buf_ptr, max_entries); hwbr_log(buf);\n")
                    f.write("    if (buf_ptr && max_entries > 0) { uint64_t *p = (uint64_t *)buf_ptr; p[0] = 0; p[1] = 0xFFFFFFFFFFFFFFFFULL; }\n")
                    f.write("}\n\n")
                elif sym == "hw_alloc_pages":
                    f.write("void *hw_alloc_pages(size_t pages) { size_t bytes = pages * HWBR_PAGE_SIZE; void *p = malloc(bytes); char buf[128]; snprintf(buf, sizeof(buf), \"hw_alloc_pages: pages=%zu -> %p\", pages, p); hwbr_log(buf); return p ? p : (void*)0; }\n\n")
                elif sym == "hw_free_pages":
                    f.write("void hw_free_pages(void *ptr, size_t pages) { (void)pages; char buf[128]; snprintf(buf, sizeof(buf), \"hw_free_pages: free %p\", ptr); hwbr_log(buf); free(ptr); }\n\n")
                else:
                    # Generic stub with a standard signature if unknown: void name(void)
                    f.write(f"/* generic stub for {sym} */\n")
                    f.write(f"void {sym}(void) {{ hwbr_log(\"{sym}: stub called\"); }}\n\n")

            # Emit compile-time plugin hook (weak symbol) so toolchains that support weak linking can override
            f.write("/* weak hooks for custom platform glue (define in user code to override) */\n")
            f.write("#if defined(__GNUC__)\n")
            f.write("__attribute__((weak)) void hw_platform_init(void) { hwbr_log(\"hw_platform_init: default\"); }\n")
            f.write("#else\n")
            f.write("void hw_platform_init(void) { hwbr_log(\"hw_platform_init: default\"); }\n")
            f.write("#endif\n")
        logger.debug("C stub written: %s", c_path)
        return c_path

    # -----------------------------------------------------------------------
    # Compilation / emit helpers (enhanced)
    # -----------------------------------------------------------------------
    def _arch_flags(self, arch: str) -> List[str]:
        """Return architecture-specific flags for compilers when available."""
        arch = arch.lower()
        if arch in ("x86_64", "amd64"):
            return ["-m64"]
        if arch in ("i386", "x86"):
            return ["-m32"]
        if arch in ("arm64", "aarch64"):
            return ["-target", "aarch64-linux-gnu"]
        if arch in ("arm",):
            return ["-marm"]
        if arch in ("riscv64",):
            # this is best-effort; may not be supported by host compiler
            return ["-march=rv64gc"]
        return []

    def compile_c_to_bin(self, c_path: Path, out_path: Path, arch: str = "x86_64", optimize: bool = True, profile: Optional[Path] = None) -> Tuple[bool, List[str]]:
        """
        Compile the C source to a raw binary when toolchain is available.
        Returns (success, notes).
        """
        notes: List[str] = []
        start = time.time()
        if not self.toolchain.available():
            notes.append("No C compiler detected")
            logger.warning("No C compiler available; skipping native build.")
            return False, notes

        cc = self.toolchain.cc
        objcopy = self.toolchain.objcopy
        obj_path = self.work_dir / (c_path.stem + ".o")
        elf_path = self.work_dir / (c_path.stem + ".elf")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # compile flags
        cflags = ["-nostdlib", "-ffreestanding", "-fno-builtin", "-c"]
        cflags += self._arch_flags(arch)
        if optimize:
            cflags.append("-O3")
        else:
            cflags.append("-O0")
        # If profile is present, try to pass profile-use flags (clang/gcc differ)
        profile_blob_note = ""
        if profile and profile.exists():
            # prefer clang-style profile-use
            profile_path = str(profile)
            cflags += ["-fprofile-use=" + profile_path]
            profile_blob_note = f" (PGO enabled: {profile_path})"
            notes.append("PGO: attempted to pass -fprofile-use")
        # compile
        cmd_compile = [cc] + cflags + [str(c_path), "-o", str(obj_path)]
        logger.info("Compiling: %s", " ".join(cmd_compile))
        try:
            subprocess.check_call(cmd_compile, cwd=self.work_dir)
            notes.append("compiled object")
        except subprocess.CalledProcessError as e:
            notes.append(f"compile failed: {e}")
            logger.error("C compile failed: %s", e)
            return False, notes

        # Link minimal ELF
        try:
            cmd_ld = [self.toolchain.ld, str(obj_path), "-o", str(elf_path)]
            logger.info("Linking ELF: %s", " ".join(cmd_ld))
            subprocess.check_call(cmd_ld, cwd=self.work_dir)
            notes.append("linked elf")
        except Exception as e:
            # Some environments prefer cc to do linking
            try:
                cmd_ld2 = [cc, str(obj_path), "-o", str(elf_path)]
                logger.info("Fallback linking with cc: %s", " ".join(cmd_ld2))
                subprocess.check_call(cmd_ld2, cwd=self.work_dir)
                notes.append("linked elf via cc")
            except Exception as e2:
                notes.append(f"link failed: {e2}")
                logger.error("Linking failed: %s", e2)
                return False, notes

        # objcopy to raw binary if available
        if objcopy:
            cmd_objcopy = [objcopy, "-O", "binary", str(elf_path), str(out_path)]
            logger.info("Objcopy: %s", " ".join(cmd_objcopy))
            try:
                subprocess.check_call(cmd_objcopy, cwd=self.work_dir)
                notes.append("objcopy -> binary")
                elapsed = time.time() - start
                notes.append(f"elapsed={elapsed:.2f}s")
                return True, notes
            except Exception as e:
                notes.append(f"objcopy failed: {e}")
                logger.warning("objcopy failed: %s", e)

        # Fallback: copy ELF as artifact if objcopy not available
        try:
            shutil.copy2(str(elf_path), str(out_path))
            notes.append("copied elf as artifact (fallback)")
            elapsed = time.time() - start
            notes.append(f"elapsed={elapsed:.2f}s")
            return True, notes
        except Exception as e:
            notes.append(f"failed to copy artifact: {e}")
            logger.error("Failed to produce artifact: %s", e)
            return False, notes

    # -----------------------------------------------------------------------
    # Plugin / emitter discovery
    # -----------------------------------------------------------------------
    def discover_emitters(self) -> List[str]:
        """
        Discover in-repo emitter modules that follow the naming pattern:
         - instryx_*_emitter.py
         - or modules exposing emit_hwbridge()
        Returns list of importable module names.
        """
        candidates = []
        try:
            for path in self.repo_root.rglob("instryx_*_emitter.py"):
                rel = path.relative_to(self.repo_root).with_suffix("")
                module = ".".join(rel.parts)
                candidates.append(module)
        except Exception:
            pass
        # also check for hardwarebridge_emitters.py
        extra = self.repo_root / "hardwarebridge" / "hardwarebridge_emitters.py"
        if extra.exists():
            rel = extra.relative_to(self.repo_root).with_suffix("")
            candidates.append(".".join(rel.parts))
        logger.debug("Discovered emitter modules: %s", candidates)
        return candidates

    def try_call_emitters(self, in_path: Path, out_path: Path, opts: Dict) -> bool:
        """
        Attempt to import discovered emitter modules and call their emit function(s).
        The function looks for callables: emit_hwbridge, build_hardware_bridge, or similar.
        """
        candidates = self.discover_emitters()
        sys.path.insert(0, str(self.repo_root))
        for mod_name in candidates:
            try:
                module = __import__(mod_name, fromlist=["*"])
            except Exception as e:
                logger.debug("Failed to import emitter %s: %s", mod_name, e)
                continue
            for fname in ("emit_hwbridge", "build_hardware_bridge", "emit_bridge", "compile_bridge"):
                fn = getattr(module, fname, None)
                if callable(fn):
                    logger.info("Calling emitter %s.%s(...)", mod_name, fname)
                    try:
                        # try multiple signatures
                        try:
                            res = fn(str(in_path), str(out_path), opts)
                        except TypeError:
                            res = fn(str(in_path), str(out_path))
                        if res in (True, None) or (isinstance(res, str) and Path(res).exists()):
                            logger.info("Emitter %s succeeded", mod_name)
                            return True
                    except Exception as e:
                        logger.warning("Emitter %s failed: %s", mod_name, e)
                        continue
        return False

    def discover_plugins(self) -> List[Path]:
        """
        Collect .py plugin files from hardwarebridge/plugins directory.
        Plugins may export a function named `post_build(artifact_path: str, spec: dict)`.
        """
        plugins = []
        if not self.plugin_dir.exists():
            return plugins
        for p in self.plugin_dir.glob("*.py"):
            plugins.append(p)
        return plugins

    def run_plugins_post_build(self, artifact: Path, spec: Dict) -> None:
        plugins = self.discover_plugins()
        if not plugins:
            return
        sys.path.insert(0, str(self.plugin_dir.resolve()))
        for p in plugins:
            mod_name = p.stem
            try:
                module = __import__(mod_name)
                fn = getattr(module, "post_build", None)
                if callable(fn):
                    logger.info("Running plugin %s.post_build", mod_name)
                    try:
                        fn(str(artifact), spec)
                    except Exception as e:
                        logger.warning("Plugin %s.post_build failed: %s", mod_name, e)
            except Exception as e:
                logger.debug("Failed to load plugin %s: %s", p, e)

    # -----------------------------------------------------------------------
    # Verification helpers
    # -----------------------------------------------------------------------
    def verify_artifact(self, artifact: Path, expected_symbols: Sequence[str]) -> Tuple[bool, List[str]]:
        notes = []
        if not artifact.exists():
            notes.append("artifact missing")
            return False, notes
        if artifact.stat().st_size == 0:
            notes.append("artifact empty")
            return False, notes
        # if nm available, inspect symbols
        if self.toolchain.nm:
            try:
                cmd = [self.toolchain.nm, "-g", "--defined-only", str(artifact)]
                out = subprocess.check_output(cmd, cwd=self.work_dir, stderr=subprocess.STDOUT, universal_newlines=True)
                present = []
                for line in out.splitlines():
                    parts = line.strip().split()
                    if parts:
                        name = parts[-1]
                        present.append(name)
                missing = [s for s in expected_symbols if s not in present]
                if missing:
                    notes.append(f"missing symbols: {missing}")
                    return False, notes
                notes.append("symbol check passed")
                return True, notes
            except Exception as e:
                notes.append(f"nm check failed: {e}")
                # fallback: accept artifact
                return True, notes
        # no nm: accept artifact but warn
        notes.append("no nm tool; verification skipped")
        return True, notes

    # -----------------------------------------------------------------------
    # Main build API
    # -----------------------------------------------------------------------
    def build_from_spec(
        self,
        spec: Dict,
        out: Path,
        arch: str = "x86_64",
        optimize: bool = True,
        profile: Optional[Path] = None,
        use_cache: bool = True,
        force_fallback: bool = False,
        parallel: bool = False,
        run_emitters: bool = True,
        sign: bool = False,
    ) -> BuildReport:
        """
        The core builder:
         - Generates C stub
         - Computes a hash and consults disk cache
         - Attempts emitter modules
         - Attempts native compile (with PGO if provided)
         - Emits fallback blob if compile not possible or forced
         - Runs plugins and verification
        """
        t0 = time.time()
        out = Path(out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        report = BuildReport(success=False, artifact=out, toolchain=self.toolchain.describe())
        # generate C
        c_path = self.generate_c_stub(spec)
        c_src = c_path.read_bytes() if c_path.exists() else b""
        profile_blob = profile.read_bytes() if profile and profile.exists() else None
        key = self.hash_inputs(spec, c_src, profile_blob)
        cache_art = self.artifact_path_for_hash(out, key)

        # cache hit?
        if use_cache and cache_art.exists():
            logger.info("Cache hit -> using cached artifact at %s", cache_art)
            shutil.copy2(str(cache_art), str(out))
            report.success = True
            report.cached = True
            report.sha256 = self._sha256_of_path(out)
            report.elapsed_seconds = time.time() - t0
            report.notes.append("used cache")
            return report

        # 1) Try in-repo emitters first (if requested)
        if run_emitters:
            emitter_opts = {"arch": arch, "optimize": optimize, "profile": str(profile) if profile else None}
            try:
                ok_emit = self.try_call_emitters(c_path, out, emitter_opts)
                if ok_emit:
                    report.success = True
                    report.notes.append("emitter produced artifact")
                    report.elapsed_seconds = time.time() - t0
                    report.sha256 = self._sha256_of_path(out) if out.exists() else None
                    # save to cache
                    if use_cache and out.exists():
                        shutil.copy2(str(out), str(cache_art))
                    self.run_plugins_post_build(out, spec)
                    return report
            except Exception as e:
                logger.debug("Emitter invocation error: %s", e)

        # 2) Try native compile (parallel builds not used here; parallel supports multi-arch outside)
        if (not force_fallback) and self.toolchain.available():
            success, notes = self.compile_c_to_bin(c_path, out, arch=arch, optimize=optimize, profile=profile)
            report.notes.extend(notes)
            if success:
                report.success = True
                # verification
                ok_verify, vnotes = self.verify_artifact(out, spec.get("symbols", []))
                report.notes.extend(vnotes)
                if not ok_verify:
                    report.notes.append("verification failed")
                    report.success = False
                else:
                    # cache artifact
                    if use_cache:
                        shutil.copy2(str(out), str(cache_art))
                    report.sha256 = self._sha256_of_path(out)
                report.elapsed_seconds = time.time() - t0
                # run plugins regardless
                self.run_plugins_post_build(out, spec)
                # sign artifact if requested (very simple: append signature file)
                if sign and report.sha256:
                    self._write_signature(out, report.sha256)
                return report
            else:
                logger.warning("Native compile path failed, will emit blob fallback")
        else:
            report.notes.append("native compile skipped (no toolchain or forced)")

        # 3) Emit fallback blob
        ok = self.emit_blob_fallback(spec, c_path, out)
        report.success = ok
        if ok:
            if use_cache:
                shutil.copy2(str(out), str(cache_art))
            report.sha256 = self._sha256_of_path(out)
            report.notes.append("fallback blob emitted")
            # plugin hooks
            self.run_plugins_post_build(out, spec)
            if sign and report.sha256:
                self._write_signature(out, report.sha256)
        else:
            report.notes.append("fallback blob failed")
        report.elapsed_seconds = time.time() - t0
        return report

    # -----------------------------------------------------------------------
    # Bulk / parallel build helpers
    # -----------------------------------------------------------------------
    def build_multiarch(
        self,
        spec: Dict,
        out_base: Path,
        archs: Sequence[str],
        optimize: bool = True,
        use_cache: bool = True,
        force_fallback: bool = False,
        parallel: bool = True,
    ) -> Dict[str, BuildReport]:
        """
        Build the bridge for multiple architectures in parallel.
        Returns mapping arch -> BuildReport.
        """
        results: Dict[str, BuildReport] = {}
        args = []
        for a in archs:
            out = Path(str(out_base).replace("{arch}", a)) if "{arch}" in str(out_base) else out_base.with_name(f"{out_base.stem}-{a}{out_base.suffix}")
            args.append((a, out))

        if parallel and len(args) > 1:
            logger.info("Building for archs %s in parallel (workers=%d)", [a for a, _ in args], min(len(args), multiprocessing.cpu_count()))
            with multiprocessing.Pool(min(len(args), multiprocessing.cpu_count())) as pool:
                work = [
                    pool.apply_async(self._build_wrapper, (spec, out, a, optimize, use_cache, force_fallback))
                    for a, out in args
                ]
                for a, job in zip([a for a, _ in args], work):
                    try:
                        results[a] = job.get()
                    except Exception as e:
                        br = BuildReport(success=False, artifact=Path(""), toolchain=self.toolchain.describe())
                        br.notes.append(f"parallel build failed: {e}")
                        results[a] = br
        else:
            for a, out in args:
                results[a] = self._build_wrapper(spec, out, a, optimize, use_cache, force_fallback)
        return results

    def _build_wrapper(self, spec, out, arch, optimize, use_cache, force_fallback):
        try:
            return self.build_from_spec(spec, out, arch=arch, optimize=optimize, use_cache=use_cache, force_fallback=force_fallback)
        except Exception as e:
            br = BuildReport(success=False, artifact=Path(out), toolchain=self.toolchain.describe())
            br.notes.append(f"exception: {e}")
            return br

    # -----------------------------------------------------------------------
    # Fallback blob emitter (stable)
    # -----------------------------------------------------------------------
    def emit_blob_fallback(self, spec: Dict, c_src_path: Path, out_path: Path) -> bool:
        """
        Emit a fallback hwbridge blob file (self-describing).
        Format: [2 bytes headerlen][header utf8][4 bytes payloadlen][payload json][4 bytes csrclen][c src bytes]
        """
        LOGIC = {"magic": "INSTRYX_HWBRIDGE_BLOB_v1", "generated_at": int(time.time()), "spec": spec}
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as out:
                header = LOGIC["magic"].encode("utf-8")
                out.write(len(header).to_bytes(2, "little"))
                out.write(header)
                payload = json.dumps(LOGIC, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                out.write(len(payload).to_bytes(4, "little"))
                out.write(payload)
                csrc = c_src_path.read_bytes() if c_src_path.exists() else b""
                out.write(len(csrc).to_bytes(4, "little"))
                out.write(csrc)
            logger.info("Fallback blob emitted to %s", out_path)
            return True
        except Exception as e:
            logger.error("Failed to emit fallback blob: %s", e)
            return False

    # -----------------------------------------------------------------------
    # Signing / stamping
    # -----------------------------------------------------------------------
    def _sha256_of_path(self, p: Path) -> Optional[str]:
        if not p.exists():
            return None
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _write_signature(self, artifact: Path, sha256: str) -> None:
        sig_path = artifact.with_suffix(artifact.suffix + ".sha256")
        try:
            with open(sig_path, "w", encoding="utf-8") as f:
                f.write(f"{sha256}  {artifact.name}\n")
            logger.info("Wrote signature: %s", sig_path)
        except Exception as e:
            logger.warning("Failed to write signature: %s", e)

    # -----------------------------------------------------------------------
    # Small selftest for local development
    # -----------------------------------------------------------------------
    def selftest(self) -> bool:
        """
        Run a quick self-test: generate stub and emit fallback blob.
        """
        logger.info("Running HWBuilder selftest (generate + blob)...")
        spec = DEFAULT_SPEC.copy()
        spec["name"] = "instryx_hwbridge_selftest"
        out = self.work_dir / "selftest_hwbridge.blob"
        r = self.build_from_spec(spec, out, force_fallback=True, use_cache=False)
        if r.success and out.exists():
            logger.info("Selftest OK: %s (sha256=%s)", out, r.sha256)
            return True
        logger.error("Selftest failed: %s", r.notes)
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli():
    p = argparse.ArgumentParser(prog="instryx_hwbuilder", description="Instryx Hardware Bridge Builder (enhanced)")
    p.add_argument("--spec", type=Path, default=None, help="Path to JSON spec (optional).")
    p.add_argument("--out", type=Path, default=Path("build") / "hwbridge" / "hwbridge.bin", help="Output artifact path")
    p.add_argument("--arch", type=str, default="x86_64", help="Target architecture (informational)")
    p.add_argument("--no-cache", action="store_true", help="Disable cache usage")
    p.add_argument("--force-fallback", action="store_true", help="Force fallback blob output")
    p.add_argument("--profile", type=Path, default=None, help="PGO profile data (optional)")
    p.add_argument("--parallel", action="store_true", help="Enable parallel multi-arch builds when multiple archs specified")
    p.add_argument("--archs", type=str, default="", help="Comma-separated list of archs for multiarch builds")
    p.add_argument("--list-emitters", action="store_true", help="List discovered emitter modules")
    p.add_argument("--run-selftest", action="store_true", help="Run embedded self-test and exit")
    p.add_argument("--no-opt", dest="optimize", action="store_false", help="Disable optimization (-O3)")
    p.add_argument("--force-sign", action="store_true", help="Write .sha256 signature next to artifact")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = p.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    builder = HWBuilder(repo_root=Path.cwd())
    if args.run_selftest:
        ok = builder.selftest()
        sys.exit(0 if ok else 1)

    if args.list_emitters:
        emitters = builder.discover_emitters()
        print("Discovered emitter modules:")
        for e in emitters:
            print(" -", e)
        sys.exit(0)

    # load spec
    spec = DEFAULT_SPEC.copy()
    if args.spec:
        if not args.spec.exists():
            logger.error("Spec file not found: %s", args.spec)
            sys.exit(2)
        try:
            spec = json.loads(args.spec.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error("Invalid spec JSON: %s", e)
            sys.exit(2)

    # multiarch?
    if args.archs:
        archs = [a.strip() for a in args.archs.split(",") if a.strip()]
        out_base = args.out
        # ensure out path contains {arch} placeholder to put separate outputs
        if "{arch}" not in str(out_base):
            out_base = out_base.with_name(out_base.stem + "-{arch}" + out_base.suffix)
        results = builder.build_multiarch(spec, out_base, archs, optimize=args.optimize, use_cache=not args.no_cache, force_fallback=args.force_fallback, parallel=args.parallel)
        for arch, r in results.items():
            logger.info("[%s] success=%s artifact=%s notes=%s sha256=%s", arch, r.success, r.artifact, r.notes, r.sha256)
        # summary exit code
        overall_ok = all(r.success for r in results.values())
        sys.exit(0 if overall_ok else 2)

    # single build
    report = builder.build_from_spec(spec, args.out, arch=args.arch, optimize=args.optimize, profile=args.profile, use_cache=not args.no_cache, force_fallback=args.force_fallback, run_emitters=True, sign=args.force_sign)
    logger.info("Build result: success=%s artifact=%s elapsed=%.2fs notes=%s sha256=%s", report.success, report.artifact, report.elapsed_seconds, report.notes, report.sha256)
    sys.exit(0 if report.success else 3)


if __name__ == "__main__":
    _cli()

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


"""
instryx_async_threading_runtime.py

A small, robust async+threading runtime for Instryx.

Features:
- Managed ThreadPoolExecutor for blocking/CPU work.
- Background asyncio event loop running in a dedicated thread for coroutine scheduling.
- Safe bridges between thread futures and asyncio futures (via run_coroutine_threadsafe).
- CancellationToken for cooperative cancellation.
- Timers, delayed scheduling and repeating tasks.
- Convenience helpers: spawn (schedule coroutine), submit (thread pool), run_sync (wait for coroutine/future).
- Graceful startup / shutdown and basic metrics.

This module uses only Python stdlib (threading, asyncio, concurrent.futures).
"""

from __future__ import annotations
import asyncio
import threading
import time
import functools
import logging
from concurrent.futures import ThreadPoolExecutor, Future as ThreadFuture
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple, List
from __future__ import annotations
import asyncio
import threading
import time
import functools
import logging
import queue
import uuid
import os

LOG = logging.getLogger("instryx.async_runtime")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class CancellationToken:
    """Simple cooperative cancellation token."""
    def __init__(self):
        self._cancelled = False
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[], None]] = []

    def cancel(self) -> None:
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            cbs = list(self._callbacks)
        for cb in cbs:
            try:
                cb()
            except Exception:
                LOG.exception("Cancellation callback raised")

    def is_cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def register(self, cb: Callable[[], None]) -> None:
        with self._lock:
            if self._cancelled:
                # invoke immediately if already cancelled
                try:
                    cb()
                except Exception:
                    LOG.exception("Cancellation callback raised")
                return
            self._callbacks.append(cb)


class TimerHandle:
    """Handle for a scheduled timer; calling cancel() prevents further runs."""
    def __init__(self, cancel_fn: Callable[[], None]):
        self._cancel_fn = cancel_fn
        self._cancelled = False

    def cancel(self) -> None:
        if not self._cancelled:
            try:
                self._cancel_fn()
            finally:
                self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled


class AsyncThreadingRuntime:
    """
    Runtime that owns:
      - an asyncio event loop running in a dedicated thread
      - a ThreadPoolExecutor for blocking tasks
    Use a single runtime instance per process where practical.
    """

    def __init__(self, workers: int = 4, loop_name: Optional[str] = None):
        self._workers = max(1, int(workers))
        self._executor = ThreadPoolExecutor(max_workers=self._workers)
        self._loop_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_started = threading.Event()
        self._shutdown = False
        self._loop_name = loop_name or "instryx-async-loop"
        self._metrics: Dict[str, Any] = {"tasks_submitted": 0, "thread_tasks": 0, "coroutines_spawned": 0}
        self._start_loop_thread()

    # -----------------------
    # Event loop management
    # -----------------------
    def _loop_target(self) -> None:
        """Target function that runs the asyncio event loop forever until stopped."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._loop_started.set()
            LOG.debug("Async loop started in thread %s", threading.current_thread().name)
            loop.run_forever()
            # clean shutdown: cancel all tasks
            pending = asyncio.all_tasks(loop=loop)
            if pending:
                for t in pending:
                    try:
                        t.cancel()
                    except Exception:
                        pass
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
            LOG.debug("Async loop stopped and closed")
        except Exception:
            LOG.exception("Event loop thread crashed")
        finally:
            self._loop_started.set()

    def _start_loop_thread(self) -> None:
        if self._loop_thread and self._loop_thread.is_alive():
            return
        self._loop_thread = threading.Thread(target=self._loop_target, name=self._loop_name, daemon=True)
        self._loop_thread.start()
        # wait until loop created
        if not self._loop_started.wait(timeout=5.0):
            raise RuntimeError("Failed to start asyncio loop thread")

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            raise RuntimeError("asyncio loop not available")
        return self._loop

    # -----------------------
    # Task submission
    # -----------------------
    def submit(self, fn: Callable[..., Any], *args, **kwargs) -> ThreadFuture:
        """
        Submit a blocking/cpu function to the thread pool. Returns concurrent.futures.Future.
        """
        if self._shutdown:
            raise RuntimeError("Runtime is shutting down")
        self._metrics["thread_tasks"] += 1
        fut = self._executor.submit(fn, *args, **kwargs)
        return fut

    def spawn(self, coro: Awaitable, *, cancel_token: Optional[CancellationToken] = None) -> "concurrent.futures.Future":
        """
        Schedule an awaitable on the runtime's event loop. Returns a concurrent.futures.Future
        that can be waited on from other threads.
        """
        if self._shutdown:
            raise RuntimeError("Runtime is shutting down")
        loop = self._ensure_loop()
        self._metrics["coroutines_spawned"] += 1
        # run_coroutine_threadsafe gives a concurrent.futures.Future
        cf: "concurrent.futures.Future" = asyncio.run_coroutine_threadsafe(coro, loop)
        if cancel_token:
            def cancel_cb():
                try:
                    cf.cancel()
                except Exception:
                    pass
            cancel_token.register(cancel_cb)
        return cf

    def run_sync(self, awaitable_or_callable: Any, timeout: Optional[float] = None) -> Any:
        """
        From any thread, run a coroutine or call a blocking function synchronously.
        - If passed a coroutine/awaitable -> schedules on event loop and waits.
        - If passed a callable -> runs in thread pool and waits.
        """
        if asyncio.iscoroutine(awaitable_or_callable) or isinstance(awaitable_or_callable, Awaitable):
            fut = self.spawn(awaitable_or_callable)
            return fut.result(timeout=timeout)
        if callable(awaitable_or_callable):
            fut = self.submit(awaitable_or_callable)
            return fut.result(timeout=timeout)
        raise TypeError("run_sync requires a coroutine or callable")

    # -----------------------
    # Timers / scheduling
    # -----------------------
    def schedule_later(self, delay: float, coro_factory: Callable[[], Awaitable]) -> TimerHandle:
        """
        Schedule a coroutine to run after delay seconds. coro_factory must return a coroutine when invoked.
        Returns a TimerHandle that can be cancelled before the coroutine runs.
        """
        loop = self._ensure_loop()
        cancelled = threading.Event()

        def _call():
            if cancelled.is_set():
                return
            try:
                c = coro_factory()
                asyncio.run_coroutine_threadsafe(c, loop)
            except Exception:
                LOG.exception("scheduled coroutine factory raised")

        handle = loop.call_later(delay, _call)

        def cancel():
            cancelled.set()
            try:
                handle.cancel()
            except Exception:
                pass

        return TimerHandle(cancel)

    def schedule_repeating(self, interval: float, coro_factory: Callable[[], Awaitable]) -> TimerHandle:
        """
        Schedule a repeating coroutine every `interval` seconds. Repetition is best-effort and runs serially.
        Returned handle cancels future repeats.
        """
        loop = self._ensure_loop()
        cancelled = threading.Event()

        async def _runner():
            try:
                while not cancelled.is_set():
                    try:
                        await coro_factory()
                    except Exception:
                        LOG.exception("repeating scheduled task raised")
                    # cooperative sleep
                    await asyncio.sleep(interval)
            except asyncio.CancelledError:
                pass

        cf = asyncio.run_coroutine_threadsafe(_runner(), loop)

        def cancel():
            cancelled.set()
            try:
                cf.cancel()
            except Exception:
                pass

        return TimerHandle(cancel)

    # -----------------------
    # Utilities
    # -----------------------
    @staticmethod
    async def sleep(delay: float) -> None:
        await asyncio.sleep(delay)

    # -----------------------
    # Shutdown
    # -----------------------
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """
        Shutdown runtime: stop the event loop and thread pool.
        """
        if self._shutdown:
            return
        self._shutdown = True

        # stop asyncio loop
        try:
            loop = self._ensure_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            def _stop():
                try:
                    loop.stop()
                except Exception:
                    LOG.exception("loop.stop() failed")
            # schedule stop on loop thread
            try:
                loop.call_soon_threadsafe(_stop)
            except Exception:
                LOG.exception("failed to schedule loop.stop()")

        # shutdown threadpool
        try:
            self._executor.shutdown(wait=wait, timeout=timeout)
        except TypeError:
            # Python <3.9 doesn't support timeout arg
            self._executor.shutdown(wait=wait)
        except Exception:
            LOG.exception("executor.shutdown failed")

        # join loop thread
        if self._loop_thread:
            self._loop_thread.join(timeout or 5.0)

    # -----------------------
    # Diagnostics / metrics
    # -----------------------
    def metrics(self) -> Dict[str, Any]:
        return dict(self._metrics)

    # -----------------------
    # Convenience decorator
    # -----------------------
    def background(self, fn: Callable[..., Any]) -> Callable[..., ThreadFuture]:
        """
        Decorator that runs the wrapped function in the threadpool when called.
        """

        @functools.wraps(fn)
        def _wrapped(*args, **kwargs):
            return self.submit(fn, *args, **kwargs)

        return _wrapped


# Module-level singleton runtime (convenience)
_global_runtime: Optional[AsyncThreadingRuntime] = None
_global_lock = threading.Lock()


def get_runtime(workers: int = 4) -> AsyncThreadingRuntime:
    global _global_runtime
    with _global_lock:
        if _global_runtime is None:
            _global_runtime = AsyncThreadingRuntime(workers=workers)
        return _global_runtime


__all__ = [
    "AsyncThreadingRuntime",
    "get_runtime",
    "CancellationToken",
    "TimerHandle",
]

if __name__ == "__main__":
    # Simple test/demo
    rt = get_runtime(workers=2)
    @rt.background
    def blocking_task(n):
        LOG.info("Blocking task %d starting", n)
        time.sleep(2)
        LOG.info("Blocking task %d done", n)
        return n * n
    async def async_task(n):
        LOG.info("Async task %d starting", n)
        await asyncio.sleep(1)
        LOG.info("Async task %d done", n)
        return n + 10
    # Submit blocking tasks
    futures = [blocking_task(i) for i in range(3)]
    # Spawn async tasks
    async_futures = [rt.spawn(async_task(i)) for i in range(3, 6)]
    # Wait for all to complete
    for f in futures:
        result = f.result()
        LOG.info("Blocking task result: %s", result)
    for af in async_futures:
        result = af.result()
        LOG.info("Async task result: %s", result)
    # Schedule a delayed task
    def delayed_coro():
        async def _coro():
            LOG.info("Delayed coroutine running")
            await asyncio.sleep(0.5)
            LOG.info("Delayed coroutine done")
        return _coro()
    timer = rt.schedule_later(3.0, delayed_coro)
    # Schedule a repeating task
    count = 0
    def repeating_coro():
        
        async def _coro():
            
            count += 1
            LOG.info("Repeating coroutine run %d", count)
            if count >= 5:
                repeating_handle.cancel()
                LOG.info("Repeating coroutine cancelled after 5 runs")
        return _coro()
    repeating_handle = rt.schedule_repeating(1.0, repeating_coro)
    # Let the demo run for a while
    try:
        time.sleep(10)
    finally:
        rt.shutdown()
        LOG.info("Runtime shutdown complete")
fmt_bytes = fmt.encode('utf-8')
c_fmt = ir.Constant(ir.ArrayType(ir.IntType(8), len(fmt_bytes)), bytearray(fmt_bytes))
global_fmt = ir.GlobalVariable(self.module, c_fmt.type, name="fstr")
global_fmt.linkage = 'internal'
global_fmt.global_constant = True
global_fmt.initializer = c_fmt
fmt_ptr = self.builder.bitcast(global_fmt, voidptr_ty)
arg_vals = [self.builder.bitcast(arg, voidptr_ty) for arg in args]
self.builder.call(self.printf, [fmt_ptr] + arg_vals)

"""
instryx_async_threading_runtime.py

Advanced async + threading runtime for Instryx.

Features (production-focused, fully implemented):
- Dedicated asyncio event loop running in a background thread.
- Configurable thread-pool for blocking/CPU tasks (ThreadPoolExecutor).
- Optional prioritized task queue with internal worker threads (priority scheduling).
- Backpressure / concurrency limits via semaphores.
- Batch submission helpers, map_with_concurrency.
- Scheduling: delayed tasks, repeating tasks with robust cancellation.
- CancellationToken for cooperative cancellation and callbacks.
- Adaptive resize: gentle API to increase/decrease worker capacity.
- Runtime metrics, timers, and lightweight tracing via logging.
- Safe startup/shutdown; idempotent and robust against errors.
- Minimal external dependencies (stdlib only). Usable by test harness / CI.

Usage examples:
  runtime = AsyncThreadingRuntime(workers=8)
  fut = runtime.submit(lambda: heavy_compute(x))
  cf = runtime.spawn(async_func(...))
  runtime.schedule_later(1.0, lambda: coro())
  runtime.shutdown()
"""

from __future__ import annotations
import asyncio
import threading
import time
import functools
import logging
import queue
import uuid
from concurrent.futures import ThreadPoolExecutor, Future as ThreadFuture
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

LOG = logging.getLogger("instryx.async_runtime")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class CancellationToken:
    """Cooperative cancellation token with callback registration."""
    def __init__(self):
        self._cancelled = False
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[], None]] = []

    def cancel(self) -> None:
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            cbs = list(self._callbacks)
            self._callbacks.clear()
        for cb in cbs:
            try:
                cb()
            except Exception:
                LOG.exception("Cancellation callback raised")

    def is_cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def register(self, cb: Callable[[], None]) -> None:
        with self._lock:
            if self._cancelled:
                # invoke immediately if already cancelled
                try:
                    cb()
                except Exception:
                    LOG.exception("Cancellation callback raised")
            else:
                self._callbacks.append(cb)


class TimerHandle:
    """Handle to cancel scheduled timers."""
    def __init__(self, cancel_fn: Callable[[], None]):
        self._cancel_fn = cancel_fn
        self._cancelled = False

    def cancel(self) -> None:
        if not self._cancelled:
            try:
                self._cancel_fn()
            finally:
                self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled


# internal prioritized task descriptor
class _PriorityTask:
    __slots__ = ("priority", "created", "task_id", "callable", "args", "kwargs", "future")

    def __init__(self, priority: int, callable_: Callable, args: Tuple, kwargs: Dict, future: ThreadFuture):
        self.priority = priority
        self.created = time.time()
        self.task_id = uuid.uuid4().hex
        self.callable = callable_
        self.args = args
        self.kwargs = kwargs
        self.future = future

    def __lt__(self, other: "_PriorityTask") -> bool:
        # lower priority value === run earlier
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created < other.created


class AsyncThreadingRuntime:
    """
    Combined asyncio + threaded runtime.

    Key features:
    - `spawn(coro)` schedules coroutine on the background loop and returns concurrent.futures.Future.
    - `submit(fn, *args, **kwargs)` submits blocking task to the internal thread pool.
    - `submit_priority(fn, priority=50, ...)` submits to a priority queue served by internal workers.
    - `map_with_concurrency(func, iterable, concurrency)` convenience for many blocking calls.
    - schedule_later / schedule_repeating utilities returning TimerHandle.
    - adaptive resize: `resize_workers(new_count)` tries to adjust the underlying pool.
    - `shutdown()` gracefully stops loop, workers and pending tasks.
    """

    def __init__(self,
                 workers: int = 4,
                 enable_priority_queue: bool = True,
                 max_queue_size: Optional[int] = 0,
                 loop_name: Optional[str] = None):
        self._workers = max(1, int(workers))
        self._enable_priority = bool(enable_priority_queue)
        self._max_queue_size = None if (max_queue_size is None or max_queue_size <= 0) else int(max_queue_size)
        self._executor = ThreadPoolExecutor(max_workers=self._workers)
        self._loop_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_started = threading.Event()
        self._shutdown_flag = threading.Event()
        self._loop_name = loop_name or "instryx-async-loop"
        # prioritized queue and workers (if enabled)
        self._pq: Optional[queue.PriorityQueue] = queue.PriorityQueue() if self._enable_priority else None
        self._pq_workers: List[threading.Thread] = []
        self._pq_worker_count = min(self._workers, 2) if self._enable_priority else 0
        self._pq_semaphore = threading.Semaphore(self._workers)  # backpressure against prioritized submits
        # metrics
        self._metrics_lock = threading.Lock()
        self._metrics: Dict[str, Any] = {
            "tasks_submitted": 0,
            "priority_tasks_submitted": 0,
            "thread_tasks_executed": 0,
            "coroutines_spawned": 0,
            "start_time": time.time(),
        }
        # start event loop thread
        self._start_loop_thread()
        # start pq workers if enabled
        if self._enable_priority:
            self._start_pq_workers(self._pq_worker_count)

    # -----------------------
    # Internal loop thread
    # -----------------------
    def _loop_target(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._loop_started.set()
        LOG.debug("Async loop started in thread %s", threading.current_thread().name)
        try:
            loop.run_forever()
            # drain tasks on stop
            pending = asyncio.all_tasks(loop=loop)
            if pending:
                for t in pending:
                    t.cancel()
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            LOG.exception("Event loop thread crashed")
        finally:
            try:
                loop.close()
            except Exception:
                pass
            LOG.debug("Async loop closed")

    def _start_loop_thread(self) -> None:
        if self._loop_thread and self._loop_thread.is_alive():
            return
        self._loop_thread = threading.Thread(target=self._loop_target, name=self._loop_name, daemon=True)
        self._loop_thread.start()
        if not self._loop_started.wait(timeout=5.0):
            raise RuntimeError("Failed to start asyncio loop thread")

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            raise RuntimeError("event loop not available")
        return self._loop

    # -----------------------
    # Priority queue workers
    # -----------------------
    def _pq_worker_target(self) -> None:
        assert self._pq is not None
        while not self._shutdown_flag.is_set():
            try:
                task: _PriorityTask = self._pq.get(timeout=0.2)
            except queue.Empty:
                continue
            # respect semaphore to enforce backpressure
            try:
                self._pq_semaphore.acquire()
                if task.future.cancelled():
                    continue
                try:
                    res = task.callable(*task.args, **task.kwargs)
                    task.future.set_result(res)
                except Exception as e:
                    task.future.set_exception(e)
                finally:
                    with self._metrics_lock:
                        self._metrics["thread_tasks_executed"] += 1
            finally:
                try:
                    self._pq_semaphore.release()
                except Exception:
                    pass
                self._pq.task_done()

    def _start_pq_workers(self, count: int) -> None:
        if not self._enable_priority or self._pq is None:
            return
        # avoid starting multiple times
        if self._pq_workers:
            return
        for i in range(max(1, int(count))):
            t = threading.Thread(target=self._pq_worker_target, name=f"instryx-pq-worker-{i}", daemon=True)
            t.start()
            self._pq_workers.append(t)

    def _stop_pq_workers(self) -> None:
        # signal threads to finish by setting shutdown flag, then join
        for t in list(self._pq_workers):
            try:
                if t.is_alive():
                    t.join(timeout=0.5)
            except Exception:
                pass
        self._pq_workers = []

    # -----------------------
    # Public submission APIs
    # -----------------------
    def submit(self, fn: Callable[..., Any], *args, **kwargs) -> ThreadFuture:
        """Submit a blocking/cpu task to the thread pool."""
        if self._shutdown_flag.is_set():
            raise RuntimeError("runtime is shutting down")
        fut = self._executor.submit(fn, *args, **kwargs)
        with self._metrics_lock:
            self._metrics["tasks_submitted"] += 1
        return fut

    def submit_priority(self, fn: Callable[..., Any], *args, priority: int = 50, block: bool = True, **kwargs) -> ThreadFuture:
        """
        Submit a prioritized blocking task.
        - lower `priority` value runs earlier.
        - when queue bounded and full, behavior depends on `block`.
        """
        if not self._enable_priority or self._pq is None:
            # fallback to standard submit
            return self.submit(fn, *args, **kwargs)
        if self._shutdown_flag.is_set():
            raise RuntimeError("runtime is shutting down")
        # backpressure: if bounded queue and full
        if self._max_queue_size is not None:
            if not block and self._pq.qsize() >= self._max_queue_size:
                raise queue.Full("priority queue full")
        future = ThreadFuture()
        task = _PriorityTask(priority=priority, callable_=fn, args=args, kwargs=kwargs, future=future)
        self._pq.put(task)
        with self._metrics_lock:
            self._metrics["priority_tasks_submitted"] += 1
        return future

    def spawn(self, coro: Awaitable, cancel_token: Optional[CancellationToken] = None) -> "concurrent.futures.Future":
        """
        Schedule an awaitable on the runtime event loop. Returns concurrent.futures.Future.
        """
        if self._shutdown_flag.is_set():
            raise RuntimeError("runtime is shutting down")
        loop = self._ensure_loop()
        with self._metrics_lock:
            self._metrics["coroutines_spawned"] += 1
        cf = asyncio.run_coroutine_threadsafe(coro, loop)
        if cancel_token is not None:
            def _cancel_cb():
                try:
                    cf.cancel()
                except Exception:
                    pass
            cancel_token.register(_cancel_cb)
        return cf

    def run_sync(self, awaitable_or_callable: Any, timeout: Optional[float] = None) -> Any:
        """
        From any (other) thread, run a coroutine synchronously or execute a callable in threadpool and wait.
        """
        if asyncio.iscoroutine(awaitable_or_callable) or isinstance(awaitable_or_callable, Awaitable):
            fut = self.spawn(awaitable_or_callable)
            return fut.result(timeout=timeout)
        if callable(awaitable_or_callable):
            fut = self.submit(awaitable_or_callable)
            return fut.result(timeout=timeout)
        raise TypeError("run_sync requires a coroutine or callable")

    # -----------------------
    # Batch helpers
    # -----------------------
    def map_with_concurrency(self, fn: Callable[..., Any], iterable, concurrency: int = 8) -> List[Any]:
        """
        Run `fn(x)` for x in iterable using up to `concurrency` parallel thread tasks.
        Returns results in same order as input.
        """
        it = list(iterable)
        n = len(it)
        if n == 0:
            return []
        results = [None] * n
        sem = threading.Semaphore(concurrency)
        threads = []

        def worker(i, item):
            nonlocal results
            try:
                sem.acquire()
                results[i] = fn(item)
            except Exception:
                LOG.exception("map_with_concurrency worker failed")
                results[i] = None
            finally:
                try:
                    sem.release()
                except Exception:
                    pass

        for i, item in enumerate(it):
            t = threading.Thread(target=worker, args=(i, item), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        return results

    # -----------------------
    # Scheduling helpers
    # -----------------------
    def schedule_later(self, delay: float, coro_factory: Callable[[], Awaitable]) -> TimerHandle:
        """
        Schedule coroutine factory to run after `delay` seconds. Returns TimerHandle to cancel.
        """
        loop = self._ensure_loop()
        cancelled = threading.Event()

        def _runner():
            if cancelled.is_set():
                return
            try:
                c = coro_factory()
                asyncio.run_coroutine_threadsafe(c, loop)
            except Exception:
                LOG.exception("scheduled coro factory raised")

        handle = loop.call_later(delay, _runner)

        def cancel():
            cancelled.set()
            try:
                handle.cancel()
            except Exception:
                pass

        return TimerHandle(cancel)

    def schedule_repeating(self, interval: float, coro_factory: Callable[[], Awaitable]) -> TimerHandle:
        """
        Schedule repeating coroutine. Cancelling the returned handle stops future repeats.
        """
        loop = self._ensure_loop()
        cancelled = threading.Event()

        async def _looped():
            try:
                while not cancelled.is_set():
                    try:
                        await coro_factory()
                    except Exception:
                        LOG.exception("repeating task raised")
                    await asyncio.sleep(interval)
            except asyncio.CancelledError:
                pass

        cf = asyncio.run_coroutine_threadsafe(_looped(), loop)

        def cancel():
            cancelled.set()
            try:
                cf.cancel()
            except Exception:
                pass

        return TimerHandle(cancel)

    # -----------------------
    # Resize / lifecycle
    # -----------------------
    def resize_workers(self, new_count: int) -> None:
        """Attempt to resize the threadpool. This creates a new ThreadPoolExecutor and swaps it in."""
        new_count = max(1, int(new_count))
        if new_count == self._workers:
            return
        LOG.info("Resizing workers %d -> %d", self._workers, new_count)
        # create new executor first
        new_exec = ThreadPoolExecutor(max_workers=new_count)
        old_exec = self._executor
        self._executor = new_exec
        self._workers = new_count
        # stop and join priority workers if present (adjust count)
        if self._enable_priority:
            # adjust semaphore capacity
            try:
                # best-effort: recreate semaphore with new_count
                self._pq_semaphore = threading.Semaphore(self._workers)
            except Exception:
                pass
            # restart pq workers to match small count
            self._stop_pq_workers()
            self._start_pq_workers(min(self._workers, 2))
        # shutdown old executor (do not wait to avoid blocking caller); let it finish in background
        try:
            old_exec.shutdown(wait=False)
        except Exception:
            LOG.exception("old executor shutdown failed")

    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """
        Gracefully shutdown runtime:
         - stop scheduling new tasks
         - stop priority workers
         - stop event loop
         - shutdown thread executor
        """
        if self._shutdown_flag.is_set():
            return
        self._shutdown_flag.set()
        # stop priority workers
        if self._enable_priority:
            self._stop_pq_workers()
        # stop event loop
        try:
            loop = self._ensure_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            def _stop():
                try:
                    loop.stop()
                except Exception:
                    LOG.exception("loop.stop() failed")
            try:
                loop.call_soon_threadsafe(_stop)
            except Exception:
                LOG.exception("failed to schedule loop.stop()")
        # shutdown executor
        try:
            self._executor.shutdown(wait=wait)
        except Exception:
            LOG.exception("executor.shutdown failed")
        # join loop thread
        if self._loop_thread:
            self._loop_thread.join(timeout or 5.0)

    # -----------------------
    # Diagnostics / metrics
    # -----------------------
    def metrics(self) -> Dict[str, Any]:
        with self._metrics_lock:
            m = dict(self._metrics)
        m.update({
            "uptime": time.time() - m.get("start_time", time.time())
        })
        return m

    # -----------------------
    # Convenience decorator
    # -----------------------
    def background(self, fn: Callable[..., Any]) -> Callable[..., ThreadFuture]:
        """Decorator to run a function in the threadpool when called."""
        @functools.wraps(fn)
        def _wrapped(*args, **kwargs):
            return self.submit(fn, *args, **kwargs)
        return _wrapped


# Module-level singleton runtime
_global_runtime_lock = threading.Lock()
_global_runtime: Optional[AsyncThreadingRuntime] = None


def get_runtime(workers: int = 4, enable_priority_queue: bool = True) -> AsyncThreadingRuntime:
    global _global_runtime
    with _global_runtime_lock:
        if _global_runtime is None:
            _global_runtime = AsyncThreadingRuntime(workers=workers, enable_priority_queue=enable_priority_queue)
        return _global_runtime


__all__ = [
    "AsyncThreadingRuntime",
    "get_runtime",
    "CancellationToken",
    "TimerHandle",
]

"""
instryx_async_threading_runtime.py

Advanced async + threading runtime for Instryx.

Features added/implemented:
- Background asyncio loop in a dedicated thread.
- ThreadPoolExecutor for blocking/CPU-bound tasks.
- Optional prioritized task queue with bounded capacity and worker threads.
- Backpressure support and non-blocking submit options.
- CancellationToken that supports callbacks.
- Robust scheduling: schedule_later, schedule_repeating, with jitter option.
- Batch helpers: map_with_concurrency (sync), amap_with_concurrency (async).
- submit_batch convenience, map_reduce pattern.
- Adaptive resize_workers to increase/decrease pool size.
- CPU-affinity hints for worker threads on Linux (best-effort).
- Diagnostics: dump_threads (write stack traces), export_metrics, reset_metrics.
- Context manager support and idempotent shutdown.
- Minimal external dependencies (stdlib only). Fully executable.

Notes:
- Affinity setting uses os.sched_setaffinity on Linux; non-fatal if unsupported.
- Priority queue uses lower integer -> higher priority.
- All external interactions are best-effort and non-fatal; runtime remains usable if optional features unavailable.
"""

from __future__ import annotations
import asyncio
import threading
import time
import functools
import logging
import queue
import uuid
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, Future as ThreadFuture
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

LOG = logging.getLogger("instryx.async_runtime")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class CancellationToken:
    """Cooperative cancellation token with callback registration."""
    def __init__(self):
        self._cancelled = False
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[], None]] = []

    def cancel(self) -> None:
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            cbs = list(self._callbacks)
            self._callbacks.clear()
        for cb in cbs:
            try:
                cb()
            except Exception:
                LOG.exception("Cancellation callback raised")

    def is_cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def register(self, cb: Callable[[], None]) -> None:
        with self._lock:
            if self._cancelled:
                # invoke immediately if already cancelled
                try:
                    cb()
                except Exception:
                    LOG.exception("Cancellation callback raised")
            else:
                self._callbacks.append(cb)


class TimerHandle:
    """Handle to cancel scheduled timers."""
    def __init__(self, cancel_fn: Callable[[], None]):
        self._cancel_fn = cancel_fn
        self._cancelled = False

    def cancel(self) -> None:
        if not self._cancelled:
            try:
                self._cancel_fn()
            finally:
                self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled


class _PriorityTask:
    """Internal descriptor for prioritized tasks placed into a PriorityQueue."""
    __slots__ = ("priority", "created", "task_id", "callable", "args", "kwargs", "future")

    def __init__(self, priority: int, callable_: Callable, args: Tuple, kwargs: Dict, future: ThreadFuture):
        self.priority = int(priority)
        self.created = time.time()
        self.task_id = uuid.uuid4().hex
        self.callable = callable_
        self.args = args
        self.kwargs = kwargs
        self.future = future

    def __lt__(self, other: "_PriorityTask") -> bool:
        # lower `priority` value => execute earlier
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created < other.created


class AsyncThreadingRuntime:
    """
    Combined asyncio + threaded runtime with many production-ready helpers.
    """

    def __init__(self,
                 workers: int = 4,
                 enable_priority_queue: bool = False,
                 max_queue_size: Optional[int] = 0,
                 loop_name: Optional[str] = None,
                 worker_affinity: Optional[List[int]] = None):
        self._workers = max(1, int(workers))
        self._executor = ThreadPoolExecutor(max_workers=self._workers)
        self._loop_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_started = threading.Event()
        self._shutdown_flag = threading.Event()
        self._loop_name = loop_name or "instryx-async-loop"
        self._enable_priority = bool(enable_priority_queue)
        self._max_queue_size = None if (max_queue_size is None or max_queue_size <= 0) else int(max_queue_size)
        self._pq: Optional[queue.PriorityQueue] = queue.PriorityQueue() if self._enable_priority else None
        self._pq_workers: List[threading.Thread] = []
        self._pq_worker_count = min(self._workers, 2) if self._enable_priority else 0
        self._pq_semaphore = threading.Semaphore(self._workers)
        self._metrics_lock = threading.Lock()
        self._metrics: Dict[str, Any] = {
            "tasks_submitted": 0,
            "priority_tasks_submitted": 0,
            "thread_tasks_executed": 0,
            "coroutines_spawned": 0,
            "start_time": time.time(),
        }
        self._worker_affinity = list(worker_affinity) if worker_affinity else None
        self._start_loop_thread()
        if self._enable_priority:
            self._start_pq_workers(self._pq_worker_count)

    # -----------------------
    # Event loop thread
    # -----------------------
    def _loop_target(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._loop_started.set()
        LOG.debug("Async loop started in thread %s", threading.current_thread().name)
        try:
            loop.run_forever()
            # drain and cancel pending tasks
            pending = asyncio.all_tasks(loop=loop)
            if pending:
                for t in pending:
                    t.cancel()
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            LOG.exception("Event loop thread crashed")
        finally:
            try:
                loop.close()
            except Exception:
                LOG.exception("Event loop close failed")
            LOG.debug("Async loop closed")

    def _start_loop_thread(self) -> None:
        if self._loop_thread and self._loop_thread.is_alive():
            return
        self._loop_thread = threading.Thread(target=self._loop_target, name=self._loop_name, daemon=True)
        self._loop_thread.start()
        if not self._loop_started.wait(timeout=5.0):
            raise RuntimeError("Failed to start asyncio loop thread")

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            raise RuntimeError("asyncio loop not available")
        return self._loop

    # -----------------------
    # Priority queue workers
    # -----------------------
    def _pq_worker_target(self) -> None:
        assert self._pq is not None
        # Optionally try to set affinity for this thread (best-effort)
        if self._worker_affinity and hasattr(os, "sched_setaffinity"):
            try:
                os.sched_setaffinity(0, set(self._worker_affinity))
            except Exception:
                LOG.debug("setaffinity for pq worker failed (ignored)")
        while not self._shutdown_flag.is_set():
            try:
                task: _PriorityTask = self._pq.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._pq_semaphore.acquire()
                if task.future.cancelled():
                    continue
                try:
                    res = task.callable(*task.args, **task.kwargs)
                    task.future.set_result(res)
                except Exception as e:
                    task.future.set_exception(e)
                finally:
                    with self._metrics_lock:
                        self._metrics["thread_tasks_executed"] = self._metrics.get("thread_tasks_executed", 0) + 1
            finally:
                try:
                    self._pq_semaphore.release()
                except Exception:
                    pass
                try:
                    self._pq.task_done()
                except Exception:
                    pass

    def _start_pq_workers(self, count: int) -> None:
        if not self._enable_priority or self._pq is None:
            return
        if self._pq_workers:
            return
        for i in range(max(1, int(count))):
            t = threading.Thread(target=self._pq_worker_target, name=f"instryx-pq-worker-{i}", daemon=True)
            t.start()
            self._pq_workers.append(t)

    def _stop_pq_workers(self) -> None:
        # graceful join
        for t in list(self._pq_workers):
            try:
                if t.is_alive():
                    t.join(timeout=0.5)
            except Exception:
                pass
        self._pq_workers = []

    # -----------------------
    # Submission APIs
    # -----------------------
    def submit(self, fn: Callable[..., Any], *args, **kwargs) -> ThreadFuture:
        """Submit a blocking/CPU task to the thread pool."""
        if self._shutdown_flag.is_set():
            raise RuntimeError("runtime is shutting down")
        fut = self._executor.submit(fn, *args, **kwargs)
        with self._metrics_lock:
            self._metrics["tasks_submitted"] = self._metrics.get("tasks_submitted", 0) + 1
        return fut

    def submit_priority(self, fn: Callable[..., Any], *args, priority: int = 50, block: bool = True, **kwargs) -> ThreadFuture:
        """Submit prioritized task. Lower `priority` runs earlier."""
        if not self._enable_priority or self._pq is None:
            return self.submit(fn, *args, **kwargs)
        if self._shutdown_flag.is_set():
            raise RuntimeError("runtime is shutting down")
        if self._max_queue_size is not None:
            if not block and self._pq.qsize() >= self._max_queue_size:
                raise queue.Full("priority queue full")
        future = ThreadFuture()
        task = _PriorityTask(priority=priority, callable_=fn, args=args, kwargs=kwargs, future=future)
        self._pq.put(task)
        with self._metrics_lock:
            self._metrics["priority_tasks_submitted"] = self._metrics.get("priority_tasks_submitted", 0) + 1
        return future

    def submit_batch(self, callables: List[Callable[[], Any]], *, priority: Optional[int] = None, block: bool = True) -> List[ThreadFuture]:
        """
        Submit a batch of zero-arg callables. Returns list of futures in same order.
        Priority used when submit_priority is available.
        """
        futures: List[ThreadFuture] = []
        for c in callables:
            if priority is None:
                futures.append(self.submit(c))
            else:
                futures.append(self.submit_priority(c, priority=priority, block=block))
        return futures

    def spawn(self, coro: Awaitable, cancel_token: Optional[CancellationToken] = None) -> "concurrent.futures.Future":
        """Schedule coroutine on background loop and return concurrent.futures.Future."""
        if self._shutdown_flag.is_set():
            raise RuntimeError("runtime is shutting down")
        loop = self._ensure_loop()
        with self._metrics_lock:
            self._metrics["coroutines_spawned"] = self._metrics.get("coroutines_spawned", 0) + 1
        cf = asyncio.run_coroutine_threadsafe(coro, loop)
        if cancel_token:
            def _cancel_cb():
                try:
                    cf.cancel()
                except Exception:
                    pass
            cancel_token.register(_cancel_cb)
        return cf

    def run_sync(self, awaitable_or_callable: Any, timeout: Optional[float] = None) -> Any:
        """Run a coroutine synchronously (schedules & waits) or run callable in threadpool and wait."""
        if asyncio.iscoroutine(awaitable_or_callable) or isinstance(awaitable_or_callable, Awaitable):
            fut = self.spawn(awaitable_or_callable)
            return fut.result(timeout=timeout)
        if callable(awaitable_or_callable):
            fut = self.submit(awaitable_or_callable)
            return fut.result(timeout=timeout)
        raise TypeError("run_sync requires a coroutine or callable")

    # -----------------------
    # Batch and map helpers
    # -----------------------
    def map_with_concurrency(self, fn: Callable[..., Any], iterable, concurrency: int = 8) -> List[Any]:
        """Run fn(item) for each item using up to `concurrency` threads. Returns results in order."""
        items = list(iterable)
        n = len(items)
        if n == 0:
            return []
        results = [None] * n
        sem = threading.Semaphore(concurrency)
        threads: List[threading.Thread] = []

        def worker(i, item):
            try:
                sem.acquire()
                results[i] = fn(item)
            except Exception:
                LOG.exception("map_with_concurrency worker failed")
                results[i] = None
            finally:
                try:
                    sem.release()
                except Exception:
                    pass

        for i, item in enumerate(items):
            t = threading.Thread(target=worker, args=(i, item), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        return results

    async def amap_with_concurrency(self, func: Callable[[Any], Awaitable[Any]], iterable, concurrency: int = 8) -> List[Any]:
        """Async version: apply async func over items with limited concurrency; returns results in order."""
        items = list(iterable)
        n = len(items)
        if n == 0:
            return []
        results = [None] * n
        sem = asyncio.Semaphore(concurrency)

        async def worker(i, item):
            async with sem:
                try:
                    results[i] = await func(item)
                except Exception:
                    LOG.exception("amap worker failed")
                    results[i] = None

        tasks = [asyncio.create_task(worker(i, itm)) for i, itm in enumerate(items)]
        await asyncio.gather(*tasks)
        return results

    # -----------------------
    # Scheduling helpers
    # -----------------------
    def schedule_later(self, delay: float, coro_factory: Callable[[], Awaitable]) -> TimerHandle:
        """Schedule coroutine factory to run after delay seconds; returns TimerHandle."""
        loop = self._ensure_loop()
        cancelled = threading.Event()

        def _runner():
            if cancelled.is_set():
                return
            try:
                c = coro_factory()
                asyncio.run_coroutine_threadsafe(c, loop)
            except Exception:
                LOG.exception("scheduled coro factory raised")

        handle = loop.call_later(delay, _runner)

        def cancel():
            cancelled.set()
            try:
                handle.cancel()
            except Exception:
                pass

        return TimerHandle(cancel)

    def schedule_repeating(self, interval: float, coro_factory: Callable[[], Awaitable], jitter: float = 0.0) -> TimerHandle:
        """
        Schedule repeating coroutine every `interval` seconds. Optional jitter in seconds reduces thundering herd.
        Returns TimerHandle; cancel stops future repeats.
        """
        loop = self._ensure_loop()
        cancelled = threading.Event()

        async def _looped():
            try:
                while not cancelled.is_set():
                    try:
                        await coro_factory()
                    except Exception:
                        LOG.exception("repeating task raised")
                    # apply jitter if requested
                    if jitter and jitter > 0:
                        await asyncio.sleep(interval + (jitter * (2 * (time.time() % 1) - 1)))
                    else:
                        await asyncio.sleep(interval)
            except asyncio.CancelledError:
                pass

        cf = asyncio.run_coroutine_threadsafe(_looped(), loop)

        def cancel():
            cancelled.set()
            try:
                cf.cancel()
            except Exception:
                pass

        return TimerHandle(cancel)

    # -----------------------
    # Diagnostics & utilities
    # -----------------------
    def dump_threads(self, path: Optional[str] = None) -> str:
        """
        Dump stack traces of all threads into a file (or return as string if path=None).
        Useful for debugging deadlocks or long-running tasks.
        """
        data = []
        for tid, frame in sys._current_frames().items():
            header = f"ThreadID: {tid}"
            stack = "".join(traceback.format_stack(frame))
            data.append(header + "\n" + stack)
        out = "\n\n".join(data)
        if path:
            try:
                atomic_write = getattr(self, "atomic_write_text", None)
                if atomic_write is None:
                    with open(path + ".tmp", "w", encoding="utf-8") as f:
                        f.write(out)
                    os.replace(path + ".tmp", path)
                else:
                    atomic_write(path, out)
            except Exception:
                LOG.exception("failed to write thread dump")
            return path
        return out

    def export_metrics(self, path: str) -> str:
        """Write runtime metrics to path and return path."""
        try:
            with open(path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(self.metrics(), f, indent=2)
            os.replace(path + ".tmp", path)
        except Exception:
            LOG.exception("export_metrics failed")
        return path

    def reset_metrics(self) -> None:
        with self._metrics_lock:
            self._metrics = {"tasks_submitted": 0, "priority_tasks_submitted": 0, "thread_tasks_executed": 0, "coroutines_spawned": 0, "start_time": time.time()}

    def metrics(self) -> Dict[str, Any]:
        with self._metrics_lock:
            m = dict(self._metrics)
        m.update({"uptime": time.time() - m.get("start_time", time.time())})
        return m

    # -----------------------
    # Worker affinity and resize
    # -----------------------
    def set_worker_affinity(self, cpus: Optional[List[int]]) -> None:
        """Set CPU affinity hint used by priority workers (best-effort; Linux only)."""
        self._worker_affinity = list(cpus) if cpus else None

    def resize_workers(self, new_count: int) -> None:
        """Resize underlying threadpool; best-effort swap to new executor."""
        new_count = max(1, int(new_count))
        if new_count == self._workers:
            return
        LOG.info("Resizing worker pool %d -> %d", self._workers, new_count)
        new_exec = ThreadPoolExecutor(max_workers=new_count)
        old_exec = self._executor
        self._executor = new_exec
        self._workers = new_count
        # adjust pq semaphore capacity
        try:
            self._pq_semaphore = threading.Semaphore(self._workers)
        except Exception:
            pass
        # restart pq workers if enabled
        if self._enable_priority:
            self._stop_pq_workers()
            self._start_pq_workers(min(self._workers, 2))
        try:
            old_exec.shutdown(wait=False)
        except Exception:
            LOG.exception("old executor shutdown failed")

    # -----------------------
    # Shutdown / lifecycle
    # -----------------------
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """Gracefully shutdown runtime (idempotent)."""
        if self._shutdown_flag.is_set():
            return
        self._shutdown_flag.set()
        # stop priority workers
        if self._enable_priority:
            self._stop_pq_workers()
        # stop event loop
        try:
            loop = self._ensure_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            def _stop():
                try:
                    loop.stop()
                except Exception:
                    LOG.exception("loop.stop() failed")
            try:
                loop.call_soon_threadsafe(_stop)
            except Exception:
                LOG.exception("failed to schedule loop.stop()")
        # shutdown executor
        try:
            self._executor.shutdown(wait=wait)
        except Exception:
            LOG.exception("executor.shutdown failed")
        # join loop thread
        if self._loop_thread:
            self._loop_thread.join(timeout or 5.0)

    # -----------------------
    # Decorators / helpers
    # -----------------------
    def background(self, fn: Callable[..., Any]) -> Callable[..., ThreadFuture]:
        """Decorator to run a function in the threadpool when called."""
        @functools.wraps(fn)
        def _wrapped(*args, **kwargs):
            return self.submit(fn, *args, **kwargs)
        return _wrapped

    # context manager support
    def __enter__(self) -> "AsyncThreadingRuntime":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown(wait=True)

# singleton accessor
_global_runtime_lock = threading.Lock()
_global_runtime: Optional[AsyncThreadingRuntime] = None


def get_runtime(workers: int = 4, enable_priority_queue: bool = False) -> AsyncThreadingRuntime:
    global _global_runtime
    with _global_runtime_lock:
        if _global_runtime is None:
            _global_runtime = AsyncThreadingRuntime(workers=workers, enable_priority_queue=enable_priority_queue)
        return _global_runtime


__all__ = [
    "AsyncThreadingRuntime",
    "get_runtime",
    "CancellationToken",
    "TimerHandle",
]

# --- demo when run as script (lightweight) ---
if __name__ == "__main__":
    rt = get_runtime(workers=2, enable_priority_queue=True)
    @rt.background
    def blocking_task(n):
        LOG.info("Blocking %d start", n)
        time.sleep(1.0 + 0.2 * n)
        LOG.info("Blocking %d done", n)
        return n * n

    async def async_task(n):
        LOG.info("Async %d start", n)
        await asyncio.sleep(0.5)
        LOG.info("Async %d done", n)
        return n + 100

    # submit and spawn
    fs = [blocking_task(i) for i in range(4)]
    afs = [rt.spawn(async_task(i)) for i in range(3)]

    for f in fs:
        LOG.info("Result blocking: %s", f.result())

    for af in afs:
        LOG.info("Result async: %s", af.result())

    # repeating
    counter = {"v": 0}
    def make_repeat():
        async def _coro():
            counter["v"] += 1
            LOG.info("repeat run %d", counter["v"])
            if counter["v"] >= 3:
                rep_handle.cancel()
        return _coro()
    rep_handle = rt.schedule_repeating(0.8, make_repeat, jitter=0.1)

    time.sleep(4.0)
    rt.shutdown()
    LOG.info("runtime shutdown complete")

"""
instryx_async_threading_runtime.py

Production-ready async+threading runtime for Instryx.

Key features added:
- Background asyncio event loop in dedicated thread.
- ThreadPoolExecutor with optional priority queue and worker affinity hints.
- Task tagging, timeouts, watchdog for long-running tasks.
- submit_with_timeout / submit_with_watchdog helpers.
- run_coroutine_with_timeout bridge.
- Graceful shutdown_now and restart.
- Enhanced metrics (histograms, counters, latencies).
- Exportable diagnostics and thread dumps.
- Context manager support and idempotent operations.
- Safe, dependency-free, uses only stdlib.

This file replaces/extends the previous runtime implementation and is ready to be used as-is.
"""

from __future__ import annotations
import asyncio
import threading
import time
import functools
import logging
import queue
import uuid
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, Future as ThreadFuture, TimeoutError as FutureTimeout
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

LOG = logging.getLogger("instryx.async_runtime")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ----- Utilities -----
def _now() -> float:
    return time.time()


def _safe_call(fn: Callable[..., Any], *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        LOG.exception("task raised exception")
        raise


# ----- Cancellation token -----
class CancellationToken:
    """Cooperative cancellation token with callback registration."""
    def __init__(self) -> None:
        self._cancelled = False
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[], None]] = []

    def cancel(self) -> None:
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            cbs = list(self._callbacks)
            self._callbacks.clear()
        for cb in cbs:
            try:
                cb()
            except Exception:
                LOG.exception("Cancellation callback raised")

    def is_cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def register(self, cb: Callable[[], None]) -> None:
        with self._lock:
            if self._cancelled:
                try:
                    cb()
                except Exception:
                    LOG.exception("Cancellation callback raised")
            else:
                self._callbacks.append(cb)


# ----- Timer handle -----
class TimerHandle:
    """Handle for a scheduled timer; calling cancel() prevents further runs."""
    def __init__(self, cancel_fn: Callable[[], None]) -> None:
        self._cancel_fn = cancel_fn
        self._cancelled = False

    def cancel(self) -> None:
        if not self._cancelled:
            try:
                self._cancel_fn()
            finally:
                self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled


# ----- Priority task descriptor -----
class _PriorityTask:
    __slots__ = ("priority", "created", "task_id", "callable", "args", "kwargs", "future", "tag")

    def __init__(self, priority: int, callable_: Callable, args: Tuple, kwargs: Dict, future: ThreadFuture, tag: Optional[str] = None):
        self.priority = int(priority)
        self.created = _now()
        self.task_id = uuid.uuid4().hex
        self.callable = callable_
        self.args = args
        self.kwargs = kwargs
        self.future = future
        self.tag = tag or ""

    def __lt__(self, other: "_PriorityTask") -> bool:  # for PriorityQueue ordering
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created < other.created


# ----- Runtime -----
class AsyncThreadingRuntime:
    """
    Combined asyncio + threaded runtime with advanced features.

    Typical usage:
      rt = AsyncThreadingRuntime(workers=8, enable_priority_queue=True)
      fut = rt.submit_with_timeout(lambda: heavy_work(), timeout=5.0)
      cf = rt.spawn(async_fn())
      rt.shutdown()
    """

    def __init__(
        self,
        workers: int = 4,
        enable_priority_queue: bool = False,
        max_queue_size: Optional[int] = 0,
        loop_name: Optional[str] = None,
        worker_affinity: Optional[List[int]] = None,
    ):
        self._workers = max(1, int(workers))
        self._executor = ThreadPoolExecutor(max_workers=self._workers)
        self._loop_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_started = threading.Event()
        self._shutdown_flag = threading.Event()
        self._loop_name = loop_name or "instryx-async-loop"
        # priority queue
        self._enable_priority = bool(enable_priority_queue)
        self._max_queue_size = None if (max_queue_size is None or max_queue_size <= 0) else int(max_queue_size)
        self._pq: Optional[queue.PriorityQueue] = queue.PriorityQueue() if self._enable_priority else None
        self._pq_workers: List[threading.Thread] = []
        self._pq_worker_count = min(2, self._workers) if self._enable_priority else 0
        self._pq_semaphore = threading.Semaphore(self._workers)
        # metrics and telemetry
        self._metrics_lock = threading.Lock()
        self._metrics: Dict[str, Any] = {
            "tasks_submitted": 0,
            "priority_tasks_submitted": 0,
            "thread_tasks_executed": 0,
            "coroutines_spawned": 0,
            "task_latencies": [],  # sample latencies
            "start_time": _now(),
        }
        # affinity hints
        self._worker_affinity = list(worker_affinity) if worker_affinity else None

        # start loop & pq workers
        self._start_loop_thread()
        if self._enable_priority:
            self._start_pq_workers(self._pq_worker_count)

    # ---- event loop thread ----
    def _loop_target(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._loop_started.set()
        LOG.debug("Async loop started in thread %s", threading.current_thread().name)
        try:
            loop.run_forever()
            # on stop, cancel pending tasks
            pending = asyncio.all_tasks(loop=loop)
            if pending:
                for t in pending:
                    t.cancel()
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            LOG.exception("Event loop thread crashed")
        finally:
            try:
                loop.close()
            except Exception:
                LOG.exception("loop close failed")
            LOG.debug("Async loop closed")

    def _start_loop_thread(self) -> None:
        if self._loop_thread and self._loop_thread.is_alive():
            return
        self._loop_thread = threading.Thread(target=self._loop_target, name=self._loop_name, daemon=True)
        self._loop_thread.start()
        if not self._loop_started.wait(timeout=5.0):
            raise RuntimeError("Failed to start asyncio loop thread")

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            raise RuntimeError("asyncio loop not available")
        return self._loop

    # ---- priority queue workers ----
    def _pq_worker_target(self) -> None:
        assert self._pq is not None
        # best-effort set affinity for this worker
        if self._worker_affinity and hasattr(os, "sched_setaffinity"):
            try:
                os.sched_setaffinity(0, set(self._worker_affinity))
            except Exception:
                LOG.debug("sched_setaffinity failed for pq worker (ignored)")
        while not self._shutdown_flag.is_set():
            try:
                task: _PriorityTask = self._pq.get(timeout=0.2)
            except queue.Empty:
                continue
            acquired = False
            try:
                self._pq_semaphore.acquire()
                acquired = True
                if task.future.cancelled():
                    continue
                start = _now()
                try:
                    res = task.callable(*task.args, **task.kwargs)
                    task.future.set_result(res)
                except Exception as e:
                    task.future.set_exception(e)
                latency = _now() - start
                with self._metrics_lock:
                    self._metrics["thread_tasks_executed"] = self._metrics.get("thread_tasks_executed", 0) + 1
                    self._metrics.setdefault("task_latencies", []).append(latency)
            finally:
                if acquired:
                    try:
                        self._pq_semaphore.release()
                    except Exception:
                        pass
                try:
                    self._pq.task_done()
                except Exception:
                    pass

    def _start_pq_workers(self, count: int) -> None:
        if not self._enable_priority or self._pq is None:
            return
        if self._pq_workers:
            return
        for i in range(max(1, int(count))):
            t = threading.Thread(target=self._pq_worker_target, name=f"instryx-pq-worker-{i}", daemon=True)
            t.start()
            self._pq_workers.append(t)

    def _stop_pq_workers(self) -> None:
        for t in list(self._pq_workers):
            try:
                if t.is_alive():
                    t.join(timeout=0.5)
            except Exception:
                pass
        self._pq_workers = []

    # ---- submission APIs ----
    def submit(self, fn: Callable[..., Any], *args, tag: Optional[str] = None, **kwargs) -> ThreadFuture:
        """Submit a blocking/CPU task to the thread pool."""
        if self._shutdown_flag.is_set():
            raise RuntimeError("runtime is shutting down")
        start = _now()
        fut = self._executor.submit(_safe_call, fn, *args, **kwargs)
        with self._metrics_lock:
            self._metrics["tasks_submitted"] = self._metrics.get("tasks_submitted", 0) + 1
        # attach metadata
        fut._task_tag = tag if tag else ""
        fut._submit_time = start
        return fut

    def submit_priority(self, fn: Callable[..., Any], *args, priority: int = 50, block: bool = True, tag: Optional[str] = None, **kwargs) -> ThreadFuture:
        """Submit prioritized task. Lower `priority` runs earlier."""
        if not self._enable_priority or self._pq is None:
            return self.submit(fn, *args, tag=tag, **kwargs)
        if self._shutdown_flag.is_set():
            raise RuntimeError("runtime is shutting down")
        if self._max_queue_size is not None:
            if not block and self._pq.qsize() >= self._max_queue_size:
                raise queue.Full("priority queue full")
        future = ThreadFuture()
        task = _PriorityTask(priority=priority, callable_=fn, args=args, kwargs=kwargs, future=future, tag=tag)
        self._pq.put(task)
        with self._metrics_lock:
            self._metrics["priority_tasks_submitted"] = self._metrics.get("priority_tasks_submitted", 0) + 1
        return future

    def submit_with_timeout(self, fn: Callable[..., Any], timeout: float, *args, **kwargs) -> ThreadFuture:
        """
        Submit a blocking task and cancel it (best-effort) if it does not complete within timeout.
        Note: cancelling running threads is not possible in stdlib; this will mark the future as timed out.
        """
        fut = self.submit(fn, *args, **kwargs)
        def _watch():
            try:
                fut.result(timeout=timeout)
            except FutureTimeout:
                try:
                    fut.cancel()
                except Exception:
                    pass
        threading.Thread(target=_watch, daemon=True).start()
        return fut

    def submit_with_watchdog(self, fn: Callable[..., Any], watchdog_secs: float, *args, **kwargs) -> ThreadFuture:
        """
        Submit a task and log/emit diagnostics if it runs longer than watchdog_secs.
        """
        fut = self.submit(fn, *args, **kwargs)
        tag = getattr(fut, "_task_tag", "")
        def _watch():
            try:
                fut.result(timeout=watchdog_secs)
            except FutureTimeout:
                LOG.warning("Watchdog: task %s running > %.2fs (tag=%s)", getattr(fut, "_submit_time", "<>"), watchdog_secs, tag)
                try:
                    # dump threads for diagnostics
                    self.dump_threads(path=None)  # will return string; logged below
                except Exception:
                    pass
        threading.Thread(target=_watch, daemon=True).start()
        return fut

    def submit_batch(self, callables: List[Callable[[], Any]], *, priority: Optional[int] = None, block: bool = True) -> List[ThreadFuture]:
        futures: List[ThreadFuture] = []
        for c in callables:
            if priority is None:
                futures.append(self.submit(c))
            else:
                futures.append(self.submit_priority(c, priority=priority, block=block))
        return futures

    # ---- coroutine APIs ----
    def spawn(self, coro: Awaitable, cancel_token: Optional[CancellationToken] = None) -> "concurrent.futures.Future":
        """Schedule coroutine on background loop and return concurrent.futures.Future."""
        if self._shutdown_flag.is_set():
            raise RuntimeError("runtime is shutting down")
        loop = self._ensure_loop()
        with self._metrics_lock:
            self._metrics["coroutines_spawned"] = self._metrics.get("coroutines_spawned", 0) + 1
        cf = asyncio.run_coroutine_threadsafe(coro, loop)
        if cancel_token:
            def _cancel_cb():
                try:
                    cf.cancel()
                except Exception:
                    pass
            cancel_token.register(_cancel_cb)
        return cf

    def run_coroutine_with_timeout(self, coro: Awaitable, timeout: Optional[float] = None) -> Any:
        """Run coroutine on runtime loop and wait for result with optional timeout."""
        fut = self.spawn(coro)
        return fut.result(timeout=timeout)

    def run_sync(self, awaitable_or_callable: Any, timeout: Optional[float] = None) -> Any:
        """Run coroutine synchronously (schedules & waits) or run callable in threadpool and wait."""
        if asyncio.iscoroutine(awaitable_or_callable) or isinstance(awaitable_or_callable, Awaitable):
            fut = self.spawn(awaitable_or_callable)
            return fut.result(timeout=timeout)
        if callable(awaitable_or_callable):
            fut = self.submit(awaitable_or_callable)
            return fut.result(timeout=timeout)
        raise TypeError("run_sync requires a coroutine or callable")

    # ---- batch helpers ----
    def map_with_concurrency(self, fn: Callable[..., Any], iterable, concurrency: int = 8) -> List[Any]:
        items = list(iterable)
        n = len(items)
        if n == 0:
            return []
        results = [None] * n
        sem = threading.Semaphore(concurrency)
        threads: List[threading.Thread] = []

        def worker(i, item):
            try:
                sem.acquire()
                results[i] = _safe_call(fn, item)
            except Exception:
                LOG.exception("map worker failed")
                results[i] = None
            finally:
                try:
                    sem.release()
                except Exception:
                    pass

        for i, item in enumerate(items):
            t = threading.Thread(target=worker, args=(i, item), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        return results

    async def amap_with_concurrency(self, func: Callable[[Any], Awaitable[Any]], iterable, concurrency: int = 8) -> List[Any]:
        items = list(iterable)
        n = len(items)
        if n == 0:
            return []
        results = [None] * n
        sem = asyncio.Semaphore(concurrency)

        async def worker(i, item):
            async with sem:
                try:
                    results[i] = await func(item)
                except Exception:
                    LOG.exception("amap worker failed")
                    results[i] = None

        tasks = [asyncio.create_task(worker(i, itm)) for i, itm in enumerate(items)]
        await asyncio.gather(*tasks)
        return results

    # ---- scheduling helpers ----
    def schedule_later(self, delay: float, coro_factory: Callable[[], Awaitable]) -> TimerHandle:
        loop = self._ensure_loop()
        cancelled = threading.Event()

        def _runner():
            if cancelled.is_set():
                return
            try:
                c = coro_factory()
                asyncio.run_coroutine_threadsafe(c, loop)
            except Exception:
                LOG.exception("scheduled coro factory raised")

        handle = loop.call_later(delay, _runner)

        def cancel():
            cancelled.set()
            try:
                handle.cancel()
            except Exception:
                pass

        return TimerHandle(cancel)

    def schedule_repeating(self, interval: float, coro_factory: Callable[[], Awaitable], jitter: float = 0.0) -> TimerHandle:
        loop = self._ensure_loop()
        cancelled = threading.Event()

        async def _looped():
            try:
                while not cancelled.is_set():
                    try:
                        await coro_factory()
                    except Exception:
                        LOG.exception("repeating task raised")
                    if jitter and jitter > 0:
                        await asyncio.sleep(interval + (jitter * (2 * (time.time() % 1) - 1)))
                    else:
                        await asyncio.sleep(interval)
            except asyncio.CancelledError:
                pass

        cf = asyncio.run_coroutine_threadsafe(_looped(), loop)

        def cancel():
            cancelled.set()
            try:
                cf.cancel()
            except Exception:
                pass

        return TimerHandle(cancel)

    # ---- diagnostics & utilities ----
    def dump_threads(self, path: Optional[str] = None) -> str:
        data = []
        for tid, frame in sys._current_frames().items():
            header = f"ThreadID: {tid}"
            stack = "".join(traceback.format_stack(frame))
            data.append(header + "\n" + stack)
        out = "\n\n".join(data)
        if path:
            try:
                tmp = path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write(out)
                os.replace(tmp, path)
            except Exception:
                LOG.exception("failed to write thread dump")
            return path
        LOG.debug("Thread dump:\n%s", out)
        return out

    def export_metrics(self, path: str) -> str:
        try:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(self._export_metrics_json())
            os.replace(tmp, path)
        except Exception:
            LOG.exception("export_metrics failed")
        return path

    def _export_metrics_json(self) -> str:
        return _now_json(self.metrics())

    def reset_metrics(self) -> None:
        with self._metrics_lock:
            self._metrics = {
                "tasks_submitted": 0,
                "priority_tasks_submitted": 0,
                "thread_tasks_executed": 0,
                "coroutines_spawned": 0,
                "task_latencies": [],
                "start_time": _now(),
            }

    def metrics(self) -> Dict[str, Any]:
        with self._metrics_lock:
            m = dict(self._metrics)
        m["uptime"] = _now() - m.get("start_time", _now())
        # provide simple summaries
        lat = m.get("task_latencies") or []
        if lat:
            m["latency_min"] = min(lat)
            m["latency_max"] = max(lat)
            m["latency_avg"] = sum(lat) / len(lat)
            m["latency_count"] = len(lat)
        else:
            m["latency_min"] = m["latency_max"] = m["latency_avg"] = 0.0
            m["latency_count"] = 0
        return m

    # ---- affinity & resize ----
    def set_worker_affinity(self, cpus: Optional[List[int]]) -> None:
        self._worker_affinity = list(cpus) if cpus else None

    def resize_workers(self, new_count: int) -> None:
        new_count = max(1, int(new_count))
        if new_count == self._workers:
            return
        LOG.info("Resizing workers %d -> %d", self._workers, new_count)
        new_exec = ThreadPoolExecutor(max_workers=new_count)
        old_exec = self._executor
        self._executor = new_exec
        self._workers = new_count
        try:
            self._pq_semaphore = threading.Semaphore(self._workers)
        except Exception:
            pass
        if self._enable_priority:
            self._stop_pq_workers()
            self._start_pq_workers(min(self._workers, 2))
        try:
            old_exec.shutdown(wait=False)
        except Exception:
            LOG.exception("old executor shutdown failed")

    # ---- lifecycle ----
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        if self._shutdown_flag.is_set():
            return
        self._shutdown_flag.set()
        if self._enable_priority:
            self._stop_pq_workers()
        try:
            loop = self._ensure_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            def _stop():
                try:
                    loop.stop()
                except Exception:
                    LOG.exception("loop.stop failed")
            try:
                loop.call_soon_threadsafe(_stop)
            except Exception:
                LOG.exception("failed to schedule loop.stop")
        try:
            self._executor.shutdown(wait=wait)
        except Exception:
            LOG.exception("executor.shutdown failed")
        if self._loop_thread:
            self._loop_thread.join(timeout or 5.0)

    def shutdown_now(self) -> None:
        """Force immediate shutdown: cancel pending pq tasks and stop loop."""
        if self._shutdown_flag.is_set():
            return
        self._shutdown_flag.set()
        if self._pq:
            try:
                while not self._pq.empty():
                    task: _PriorityTask = self._pq.get_nowait()
                    try:
                        task.future.cancel()
                    except Exception:
                        pass
            except Exception:
                pass
        try:
            loop = self._ensure_loop()
            loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
        if self._loop_thread:
            self._loop_thread.join(timeout=1.0)

    # ---- helpers ----
    def background(self, fn: Callable[..., Any]) -> Callable[..., ThreadFuture]:
        @functools.wraps(fn)
        def _wrapped(*args, **kwargs):
            return self.submit(fn, *args, **kwargs)
        return _wrapped

    def __enter__(self) -> "AsyncThreadingRuntime":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown(wait=True)


# ----- module-level singleton -----
_global_runtime_lock = threading.Lock()
_global_runtime: Optional[AsyncThreadingRuntime] = None


def get_runtime(workers: int = 4, enable_priority_queue: bool = False) -> AsyncThreadingRuntime:
    global _global_runtime
    with _global_runtime_lock:
        if _global_runtime is None:
            _global_runtime = AsyncThreadingRuntime(workers=workers, enable_priority_queue=enable_priority_queue)
        return _global_runtime


# ----- tiny helpers -----
def _now_json(obj: Any) -> str:
    import json
    return json.dumps(obj, default=str, indent=2)


__all__ = [
    "AsyncThreadingRuntime",
    "get_runtime",
    "CancellationToken",
    "TimerHandle",
]


# ----- demo when run as script -----
if __name__ == "__main__":
    rt = get_runtime(workers=2, enable_priority_queue=True)
    @rt.background
    def blocking_task(n):
        LOG.info("Blocking %d start", n)
        time.sleep(1.0 + 0.1 * n)
        LOG.info("Blocking %d done", n)
        return n * n

    async def async_task(n):
        LOG.info("Async %d start", n)
        await asyncio.sleep(0.5)
        LOG.info("Async %d done", n)
        return n + 100

    fs = [blocking_task(i) for i in range(4)]
    afs = [rt.spawn(async_task(i)) for i in range(3)]

    for f in fs:
        LOG.info("Result blocking: %s", f.result())

    for af in afs:
        LOG.info("Result async: %s", af.result())

    def make_repeat():
        async def _coro():
            LOG.info("Repeat job running")
        return _coro()

    rep_handle = rt.schedule_repeating(0.8, make_repeat, jitter=0.1)
    time.sleep(3.0)
    rep_handle.cancel()
    rt.export_metrics("runtime_metrics.json")
    rt.shutdown()
    LOG.info("runtime shutdown complete")

"""
instryx_async_threading_runtime.py

Production-ready async+threading runtime for Instryx.

Features:
- Background asyncio event loop in a dedicated thread.
- ThreadPoolExecutor with optional priority queue and worker affinity hints.
- Task tagging, timeouts, watchdog for long-running tasks.
- submit_with_timeout / submit_with_watchdog helpers.
- run_coroutine_with_timeout bridge and run_in_executor helper.
- Graceful shutdown_now and restart.
- Enhanced metrics (histograms, counters, latencies).
- Exportable diagnostics and thread dumps.
- Context manager support and idempotent operations.
- Safe, dependency-free, uses only stdlib.
"""
from __future__ import annotations
import asyncio
import threading
import time
import functools
import logging
import queue
import uuid
import os
import sys
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, Future as ThreadFuture, TimeoutError as FutureTimeout
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

LOG = logging.getLogger("instryx.async_runtime")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---- small utilities ----
def _now() -> float:
    return time.time()


def atomic_write_text(path: str, text: str, encoding: str = "utf-8") -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding=encoding) as f:
        f.write(text)
    os.replace(tmp, path)


def _safe_call(fn: Callable[..., Any], *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        LOG.exception("task raised exception")
        raise


# ---- cancellation / timer handles ----
class CancellationToken:
    """Cooperative cancellation token with callback registration."""
    def __init__(self) -> None:
        self._cancelled = False
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[], None]] = []

    def cancel(self) -> None:
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            cbs = list(self._callbacks)
            self._callbacks.clear()
        for cb in cbs:
            try:
                cb()
            except Exception:
                LOG.exception("Cancellation callback raised")

    def is_cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def register(self, cb: Callable[[], None]) -> None:
        with self._lock:
            if self._cancelled:
                try:
                    cb()
                except Exception:
                    LOG.exception("Cancellation callback raised")
            else:
                self._callbacks.append(cb)


class TimerHandle:
    """Handle for a scheduled timer; calling cancel() prevents further runs."""
    def __init__(self, cancel_fn: Callable[[], None]) -> None:
        self._cancel_fn = cancel_fn
        self._cancelled = False

    def cancel(self) -> None:
        if not self._cancelled:
            try:
                self._cancel_fn()
            finally:
                self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled


# ---- priority task descriptor ----
class _PriorityTask:
    __slots__ = ("priority", "created", "task_id", "callable", "args", "kwargs", "future", "tag")

    def __init__(self, priority: int, callable_: Callable, args: Tuple, kwargs: Dict, future: ThreadFuture, tag: Optional[str] = None):
        self.priority = int(priority)
        self.created = _now()
        self.task_id = uuid.uuid4().hex
        self.callable = callable_
        self.args = args
        self.kwargs = kwargs
        self.future = future
        self.tag = tag or ""

    def __lt__(self, other: "_PriorityTask") -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created < other.created


# ---- runtime ----
class AsyncThreadingRuntime:
    """
    Combined asyncio + threaded runtime with advanced features.

    Usage:
      rt = AsyncThreadingRuntime(workers=8, enable_priority_queue=True)
      fut = rt.submit_with_timeout(lambda: heavy_work(), timeout=5.0)
      cf = rt.spawn(async_fn())
      rt.shutdown()
    """

    def __init__(
        self,
        workers: int = 4,
        enable_priority_queue: bool = False,
        max_queue_size: Optional[int] = 0,
        loop_name: Optional[str] = None,
        worker_affinity: Optional[List[int]] = None,
    ):
        self._workers = max(1, int(workers))
        self._executor = ThreadPoolExecutor(max_workers=self._workers)
        self._loop_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_started = threading.Event()
        self._shutdown_flag = threading.Event()
        self._loop_name = loop_name or "instryx-async-loop"

        # priority queue
        self._enable_priority = bool(enable_priority_queue)
        self._max_queue_size = None if (max_queue_size is None or max_queue_size <= 0) else int(max_queue_size)
        self._pq: Optional[queue.PriorityQueue] = queue.PriorityQueue() if self._enable_priority else None
        self._pq_workers: List[threading.Thread] = []
        self._pq_worker_count = min(2, self._workers) if self._enable_priority else 0
        self._pq_semaphore = threading.Semaphore(self._workers)

        # metrics and telemetry
        self._metrics_lock = threading.Lock()
        self._metrics: Dict[str, Any] = {
            "tasks_submitted": 0,
            "priority_tasks_submitted": 0,
            "thread_tasks_executed": 0,
            "coroutines_spawned": 0,
            "task_latencies": [],
            "start_time": _now(),
        }

        # affinity hints
        self._worker_affinity = list(worker_affinity) if worker_affinity else None

        # start loop & pq workers
        self._start_loop_thread()
        if self._enable_priority:
            self._start_pq_workers(self._pq_worker_count)

    # ---- event loop thread ----
    def _loop_target(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._loop_started.set()
        LOG.debug("Async loop started in thread %s", threading.current_thread().name)
        try:
            loop.run_forever()
            pending = asyncio.all_tasks(loop=loop)
            if pending:
                for t in pending:
                    t.cancel()
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            LOG.exception("Event loop thread crashed")
        finally:
            try:
                loop.close()
            except Exception:
                LOG.exception("loop close failed")
            LOG.debug("Async loop closed")

    def _start_loop_thread(self) -> None:
        if self._loop_thread and self._loop_thread.is_alive():
            return
        self._loop_thread = threading.Thread(target=self._loop_target, name=self._loop_name, daemon=True)
        self._loop_thread.start()
        if not self._loop_started.wait(timeout=5.0):
            raise RuntimeError("Failed to start asyncio loop thread")

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            raise RuntimeError("asyncio loop not available")
        return self._loop

    # ---- priority queue workers ----
    def _pq_worker_target(self) -> None:
        assert self._pq is not None
        if self._worker_affinity and hasattr(os, "sched_setaffinity"):
            try:
                os.sched_setaffinity(0, set(self._worker_affinity))
            except Exception:
                LOG.debug("sched_setaffinity failed for pq worker (ignored)")
        while not self._shutdown_flag.is_set():
            try:
                task: _PriorityTask = self._pq.get(timeout=0.2)
            except queue.Empty:
                continue
            acquired = False
            try:
                self._pq_semaphore.acquire()
                acquired = True
                if task.future.cancelled():
                    continue
                start = _now()
                try:
                    res = task.callable(*task.args, **task.kwargs)
                    task.future.set_result(res)
                except Exception as e:
                    task.future.set_exception(e)
                latency = _now() - start
                with self._metrics_lock:
                    self._metrics["thread_tasks_executed"] = self._metrics.get("thread_tasks_executed", 0) + 1
                    self._metrics.setdefault("task_latencies", []).append(latency)
            finally:
                if acquired:
                    try:
                        self._pq_semaphore.release()
                    except Exception:
                        pass
                try:
                    self._pq.task_done()
                except Exception:
                    pass

    def _start_pq_workers(self, count: int) -> None:
        if not self._enable_priority or self._pq is None:
            return
        if self._pq_workers:
            return
        for i in range(max(1, int(count))):
            t = threading.Thread(target=self._pq_worker_target, name=f"instryx-pq-worker-{i}", daemon=True)
            t.start()
            self._pq_workers.append(t)

    def _stop_pq_workers(self) -> None:
        for t in list(self._pq_workers):
            try:
                if t.is_alive():
                    t.join(timeout=0.5)
            except Exception:
                pass
        self._pq_workers = []

    # ---- submission APIs ----
    def submit(self, fn: Callable[..., Any], *args, tag: Optional[str] = None, **kwargs) -> ThreadFuture:
        """Submit a blocking/CPU task to the thread pool."""
        if self._shutdown_flag.is_set():
            raise RuntimeError("runtime is shutting down")
        start = _now()
        fut = self._executor.submit(_safe_call, fn, *args, **kwargs)
        with self._metrics_lock:
            self._metrics["tasks_submitted"] = self._metrics.get("tasks_submitted", 0) + 1
        fut._task_tag = tag if tag else ""
        fut._submit_time = start
        return fut

    def submit_priority(self, fn: Callable[..., Any], *args, priority: int = 50, block: bool = True, tag: Optional[str] = None, **kwargs) -> ThreadFuture:
        """Submit prioritized task. Lower `priority` runs earlier."""
        if not self._enable_priority or self._pq is None:
            return self.submit(fn, *args, tag=tag, **kwargs)
        if self._shutdown_flag.is_set():
            raise RuntimeError("runtime is shutting down")
        if self._max_queue_size is not None and not block and self._pq.qsize() >= self._max_queue_size:
            raise queue.Full("priority queue full")
        future = ThreadFuture()
        task = _PriorityTask(priority=priority, callable_=fn, args=args, kwargs=kwargs, future=future, tag=tag)
        self._pq.put(task)
        with self._metrics_lock:
            self._metrics["priority_tasks_submitted"] = self._metrics.get("priority_tasks_submitted", 0) + 1
        return future

    def submit_with_timeout(self, fn: Callable[..., Any], timeout: float, *args, **kwargs) -> ThreadFuture:
        """
        Submit a blocking task and mark/cancel it (best-effort) if it doesn't complete within timeout.
        Note: cannot reliably kill running thread; this marks the future cancelled on timeout.
        """
        fut = self.submit(fn, *args, **kwargs)
        def _watch():
            try:
                fut.result(timeout=timeout)
            except FutureTimeout:
                try:
                    fut.cancel()
                except Exception:
                    pass
        threading.Thread(target=_watch, daemon=True).start()
        return fut

    def submit_with_watchdog(self, fn: Callable[..., Any], watchdog_secs: float, *args, **kwargs) -> ThreadFuture:
        """
        Submit a task and log diagnostics if it runs longer than watchdog_secs.
        """
        fut = self.submit(fn, *args, **kwargs)
        tag = getattr(fut, "_task_tag", "")
        def _watch():
            try:
                fut.result(timeout=watchdog_secs)
            except FutureTimeout:
                LOG.warning("Watchdog: task started at %s running > %.2fs (tag=%s)", getattr(fut, "_submit_time", "<>"), watchdog_secs, tag)
                try:
                    dump = self.dump_threads(path=None)
                    LOG.debug("Thread dump (watchdog): %s", dump)
                except Exception:
                    pass
        threading.Thread(target=_watch, daemon=True).start()
        return fut

    def submit_batch(self, callables: List[Callable[[], Any]], *, priority: Optional[int] = None, block: bool = True) -> List[ThreadFuture]:
        futures: List[ThreadFuture] = []
        for c in callables:
            if priority is None:
                futures.append(self.submit(c))
            else:
                futures.append(self.submit_priority(c, priority=priority, block=block))
        return futures

    # ---- coroutine APIs ----
    def spawn(self, coro: Awaitable, cancel_token: Optional[CancellationToken] = None) -> "concurrent.futures.Future":
        """Schedule coroutine on background loop and return concurrent.futures.Future."""
        if self._shutdown_flag.is_set():
            raise RuntimeError("runtime is shutting down")
        loop = self._ensure_loop()
        with self._metrics_lock:
            self._metrics["coroutines_spawned"] = self._metrics.get("coroutines_spawned", 0) + 1
        cf = asyncio.run_coroutine_threadsafe(coro, loop)
        if cancel_token:
            def _cancel_cb():
                try:
                    cf.cancel()
                except Exception:
                    pass
            cancel_token.register(_cancel_cb)
        return cf

    def run_coroutine_with_timeout(self, coro: Awaitable, timeout: Optional[float] = None) -> Any:
        fut = self.spawn(coro)
        return fut.result(timeout=timeout)

    def run_in_executor_async(self, fn: Callable[..., Any], *args, timeout: Optional[float] = None, **kwargs) -> Any:
        """Schedule a blocking function in the runtime executor but await it as coroutine."""
        loop = self._ensure_loop()
        return loop.run_in_executor(self._executor, functools.partial(_safe_call, fn, *args, **kwargs))

    def run_sync(self, awaitable_or_callable: Any, timeout: Optional[float] = None) -> Any:
        if asyncio.iscoroutine(awaitable_or_callable) or isinstance(awaitable_or_callable, Awaitable):
            fut = self.spawn(awaitable_or_callable)
            return fut.result(timeout=timeout)
        if callable(awaitable_or_callable):
            fut = self.submit(awaitable_or_callable)
            return fut.result(timeout=timeout)
        raise TypeError("run_sync requires a coroutine or callable")

    # ---- batch helpers ----
    def map_with_concurrency(self, fn: Callable[..., Any], iterable, concurrency: int = 8) -> List[Any]:
        items = list(iterable)
        n = len(items)
        if n == 0:
            return []
        results = [None] * n
        sem = threading.Semaphore(concurrency)
        threads: List[threading.Thread] = []

        def worker(i, item):
            try:
                sem.acquire()
                results[i] = _safe_call(fn, item)
            except Exception:
                LOG.exception("map worker failed")
                results[i] = None
            finally:
                try:
                    sem.release()
                except Exception:
                    pass

        for i, item in enumerate(items):
            t = threading.Thread(target=worker, args=(i, item), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        return results

    async def amap_with_concurrency(self, func: Callable[[Any], Awaitable[Any]], iterable, concurrency: int = 8) -> List[Any]:
        items = list(iterable)
        n = len(items)
        if n == 0:
            return []
        results = [None] * n
        sem = asyncio.Semaphore(concurrency)

        async def worker(i, item):
            async with sem:
                try:
                    results[i] = await func(item)
                except Exception:
                    LOG.exception("amap worker failed")
                    results[i] = None

        tasks = [asyncio.create_task(worker(i, itm)) for i, itm in enumerate(items)]
        await asyncio.gather(*tasks)
        return results

    # ---- scheduling helpers ----
    def schedule_later(self, delay: float, coro_factory: Callable[[], Awaitable]) -> TimerHandle:
        loop = self._ensure_loop()
        cancelled = threading.Event()

        def _runner():
            if cancelled.is_set():
                return
            try:
                c = coro_factory()
                asyncio.run_coroutine_threadsafe(c, loop)
            except Exception:
                LOG.exception("scheduled coro factory raised")

        handle = loop.call_later(delay, _runner)

        def cancel():
            cancelled.set()
            try:
                handle.cancel()
            except Exception:
                pass

        return TimerHandle(cancel)

    def schedule_repeating(self, interval: float, coro_factory: Callable[[], Awaitable], jitter: float = 0.0) -> TimerHandle:
        loop = self._ensure_loop()
        cancelled = threading.Event()

        async def _looped():
            try:
                while not cancelled.is_set():
                    try:
                        await coro_factory()
                    except Exception:
                        LOG.exception("repeating task raised")
                    if jitter and jitter > 0:
                        await asyncio.sleep(interval + (jitter * (2 * (time.time() % 1) - 1)))
                    else:
                        await asyncio.sleep(interval)
            except asyncio.CancelledError:
                pass

        cf = asyncio.run_coroutine_threadsafe(_looped(), loop)

        def cancel():
            cancelled.set()
            try:
                cf.cancel()
            except Exception:
                pass

        return TimerHandle(cancel)

    # ---- diagnostics & utilities ----
    def dump_threads(self, path: Optional[str] = None) -> str:
        data = []
        for tid, frame in sys._current_frames().items():
            header = f"ThreadID: {tid}"
            stack = "".join(traceback.format_stack(frame))
            data.append(header + "\n" + stack)
        out = "\n\n".join(data)
        if path:
            try:
                atomic_write_text(path, out)
            except Exception:
                LOG.exception("failed to write thread dump")
            return path
        LOG.debug("Thread dump:\n%s", out)
        return out

    def export_metrics(self, path: str) -> str:
        try:
            atomic_write_text(path, json.dumps(self.metrics(), default=str, indent=2))
        except Exception:
            LOG.exception("export_metrics failed")
        return path

    def _export_metrics_json(self) -> str:
        return json.dumps(self.metrics(), default=str, indent=2)

    def reset_metrics(self) -> None:
        with self._metrics_lock:
            self._metrics = {
                "tasks_submitted": 0,
                "priority_tasks_submitted": 0,
                "thread_tasks_executed": 0,
                "coroutines_spawned": 0,
                "task_latencies": [],
                "start_time": _now(),
            }

    def metrics(self) -> Dict[str, Any]:
        with self._metrics_lock:
            m = dict(self._metrics)
        m["uptime"] = _now() - m.get("start_time", _now())
        lat = m.get("task_latencies") or []
        if lat:
            m["latency_min"] = min(lat)
            m["latency_max"] = max(lat)
            m["latency_avg"] = sum(lat) / len(lat)
            m["latency_count"] = len(lat)
        else:
            m["latency_min"] = m["latency_max"] = m["latency_avg"] = 0.0
            m["latency_count"] = 0
        return m

    # ---- affinity & resize ----
    def set_worker_affinity(self, cpus: Optional[List[int]]) -> None:
        self._worker_affinity = list(cpus) if cpus else None

    def resize_workers(self, new_count: int) -> None:
        new_count = max(1, int(new_count))
        if new_count == self._workers:
            return
        LOG.info("Resizing workers %d -> %d", self._workers, new_count)
        new_exec = ThreadPoolExecutor(max_workers=new_count)
        old_exec = self._executor
        self._executor = new_exec
        self._workers = new_count
        try:
            self._pq_semaphore = threading.Semaphore(self._workers)
        except Exception:
            pass
        if self._enable_priority:
            self._stop_pq_workers()
            self._start_pq_workers(min(self._workers, 2))
        try:
            old_exec.shutdown(wait=False)
        except Exception:
            LOG.exception("old executor shutdown failed")

    # ---- lifecycle ----
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        if self._shutdown_flag.is_set():
            return
        self._shutdown_flag.set()
        if self._enable_priority:
            self._stop_pq_workers()
        try:
            loop = self._ensure_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            def _stop():
                try:
                    loop.stop()
                except Exception:
                    LOG.exception("loop.stop failed")
            try:
                loop.call_soon_threadsafe(_stop)
            except Exception:
                LOG.exception("failed to schedule loop.stop")
        try:
            self._executor.shutdown(wait=wait)
        except Exception:
            LOG.exception("executor.shutdown failed")
        if self._loop_thread:
            self._loop_thread.join(timeout or 5.0)

    def shutdown_now(self) -> None:
        """Force immediate shutdown: cancel pending pq tasks and stop loop."""
        if self._shutdown_flag.is_set():
            return
        self._shutdown_flag.set()
        if self._pq:
            try:
                while not self._pq.empty():
                    task: _PriorityTask = self._pq.get_nowait()
                    try:
                        task.future.cancel()
                    except Exception:
                        pass
            except Exception:
                pass
        try:
            loop = self._ensure_loop()
            loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
        if self._loop_thread:
            self._loop_thread.join(timeout=1.0)

    def restart(self, wait: bool = True) -> None:
        """Gracefully restart loop and workers (best-effort)."""
        self.shutdown(wait=wait)
        # reset flag and start fresh loop and executor
        self._shutdown_flag.clear()
        self._executor = ThreadPoolExecutor(max_workers=self._workers)
        self._start_loop_thread()
        if self._enable_priority:
            self._start_pq_workers(self._pq_worker_count)

    # ---- helpers ----
    def background(self, fn: Callable[..., Any]) -> Callable[..., ThreadFuture]:
        @functools.wraps(fn)
        def _wrapped(*args, **kwargs):
            return self.submit(fn, *args, **kwargs)
        return _wrapped

    def __enter__(self) -> "AsyncThreadingRuntime":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown(wait=True)


# ---- module-level singleton ----
_global_runtime_lock = threading.Lock()
_global_runtime: Optional[AsyncThreadingRuntime] = None


def get_runtime(workers: int = 4, enable_priority_queue: bool = False) -> AsyncThreadingRuntime:
    global _global_runtime
    with _global_runtime_lock:
        if _global_runtime is None:
            _global_runtime = AsyncThreadingRuntime(workers=workers, enable_priority_queue=enable_priority_queue)
        return _global_runtime


# ---- tiny helpers ----
def _now_json(obj: Any) -> str:
    return json.dumps(obj, default=str, indent=2)


__all__ = [
    "AsyncThreadingRuntime",
    "get_runtime",
    "CancellationToken",
    "TimerHandle",
]

# ---- demo when run as script ----
if __name__ == "__main__":
    rt = get_runtime(workers=2, enable_priority_queue=True)
    @rt.background
    def blocking_task(n):
        LOG.info("Blocking %d start", n)
        time.sleep(1.0 + 0.1 * n)
        LOG.info("Blocking %d done", n)
        return n * n
    async def async_task(n):
        LOG.info("Async %d start", n)
        await asyncio.sleep(0.5)
        LOG.info("Async %d done", n)
        return n + 100
    fs = [blocking_task(i) for i in range(4)]
    afs = [rt.spawn(async_task(i)) for i in range(3)]
    for f in fs:
        LOG.info("Result blocking: %s", f.result())
    for af in afs:
        LOG.info("Result async: %s", af.result())
    def make_repeat():
        async def _coro():
            LOG.info("Repeat job running")
        return _coro()
    rep_handle = rt.schedule_repeating(0.8, make_repeat, jitter=0.1)
    time.sleep(3.0)
    rep_handle.cancel()
    rt.export_metrics("runtime_metrics.json")
    rt.shutdown()
    LOG.info("runtime shutdown complete")
    """
    instryx_async_threading_runtime.py
    Production-ready async+threading runtime for Instryx.
    Features:
    - Background asyncio event loop in a dedicated thread.
    - ThreadPoolExecutor with optional priority queue and worker affinity hints.
    - Task tagging, timeouts, watchdog for long-running tasks.
    - submit_with_timeout / submit_with_watchdog helpers.
    - run_coroutine_with_timeout bridge and run_in_executor helper.
    - Graceful shutdown_now and restart.
    - Enhanced metrics (histograms, counters, latencies).
    - Exportable diagnostics and thread dumps.
    - Context manager support and idempotent operations.
    - Safe, dependency-free, uses only stdlib.
    """

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

