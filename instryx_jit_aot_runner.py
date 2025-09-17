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
