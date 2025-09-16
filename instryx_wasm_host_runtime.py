
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
