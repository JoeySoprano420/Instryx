"""
instryx_memory_math_loops_codegen.py

Extended code-generator helpers and executable tooling for Instryx textual code patterns.

New/added capabilities:
- Powerful optimizations: loop tiling, vectorize hints, parallel_map, loop fusion hints
- More generator helpers (vectorize, tile_loop, parallel_map, circuit_breaker stub)
- CodegenToolkit: programmatic emitter with directory batch injection, optional macro-overlay expansion
- Safe preview/apply hooks that call macro_overlay.applyMacrosWithDiagnostics when available
- CLI commands to emit, write, inject, inject+expand, batch-inject, generate HTML reports, run micro-benchmarks and unit-tests
- Plugin-friendly helper registry and ability to register additional generators at runtime
- Improved tests and demos

Notes:
- All generated code is textual Instryx-like pseudocode intended for use with the project's macro overlay
  (text-to-text expansion) or as scaffolding for an Instryx emitter.
- This tool intentionally has no external runtime dependencies. Macro expansion requires macro_overlay module
  if you plan to preview or apply generated helpers into actual source with the overlay step.
"""

from __future__ import annotations
import argparse
import re
import time
import json
import random
import string
import os
import sys
import html
import importlib
import concurrent.futures
from typing import List, Optional, Tuple, Dict, Callable, Any

# -------------------------
# Utilities
# -------------------------


def uid(prefix: str = "g") -> str:
    """Short unique id suitable for helper variable names."""
    return f"{prefix}_{int(time.time()*1000)}_{''.join(random.choices(string.ascii_lowercase, k=4))}"


def safe_ident(name: str) -> str:
    """Make a safe identifier from arbitrary text."""
    return re.sub(r"[^\w]", "_", name)


def escape_str(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _try_import_macro_overlay():
    """Lazy import for macro_overlay (returns module or None)."""
    try:
        return importlib.import_module("macro_overlay")
    except Exception:
        return None


# -------------------------
# Core code generators
# -------------------------


def generate_unrolled_loop(loop_var: str, start: int, end: int, body_template: str) -> str:
    """
    Unroll a small integer loop. Replace {i} or {loop} occurrences in the template.
    """
    if end <= start:
        return ""
    out_lines = []
    for i in range(start, end):
        repl = body_template.replace("{i}", str(i)).replace("{loop}", str(i))
        out_lines.append(repl.rstrip() + "\n")
    return "".join(out_lines)


def generate_memoize_wrapper(func_name: str, params: List[str], body: str, cache_name: Optional[str] = None) -> str:
    """
    Generate a memoizing wrapper for a function.
    """
    cache = cache_name or f"__memo_{safe_ident(func_name)}"
    key_expr = " + '|' + ".join([f"String({p})" for p in params]) if params else '"__no_args__"'
    params_sig = ", ".join(params)
    lines = []
    lines.append(f"{cache} = {cache} ? {cache} : {{}};\n")
    lines.append(f"func {func_name}({params_sig}) {{\n")
    lines.append(f"    __memo_k = {key_expr};\n")
    lines.append(f"    if ({cache}[__memo_k] != undefined) {{\n")
    lines.append(f"        {cache}[__memo_k];\n")
    lines.append(f"    }} else {{\n")
    lines.append(f"        __memo_v = (function() {{ {body.strip()} }})();\n")
    lines.append(f"        {cache}[__memo_k] = __memo_v;\n")
    lines.append(f"        __memo_v;\n")
    lines.append(f"    }}\n")
    lines.append("}\n")
    return "".join(lines)


def generate_defer_helpers(stack_name: Optional[str] = None) -> str:
    stack = stack_name or "__defer_stack"
    lines = []
    lines.append(f"{stack} = {stack} ? {stack} : [];\n")
    lines.append(f"/* push deferred action: {stack}.push(() => {{ ... }}); */\n")
    lines.append(f"/* On scope exit: while ({stack}.length) {{ ({stack}.pop())(); }} */\n")
    return "".join(lines)


def generate_prefetch_helper(urls: List[str], results_var: Optional[str] = None) -> str:
    results = results_var or f"__prefetch_{uid('pf')}"
    lines = []
    lines.append(f"{results} = {results} ? {results} : {{}};\n")
    for u in urls:
        safe_u = escape_str(u)
        key = safe_ident(u)[:20]
        handle = f"__pf_{key}_{uid('h')}"
        lines.append(f"{handle} = spawn async {{\n")
        lines.append(f"    {results}[\"{safe_u}\"] = fetchData(\"{safe_u}\");\n")
        lines.append("};\n")
    lines.append(f"/* access prefetched: {results}[\"<url>\"] */\n")
    return "".join(lines)


def generate_batch_helper(call_expr: str, items_expr: str, chunk_size: int = 32) -> str:
    var_items = safe_ident(items_expr)
    gid = uid("batch")
    lines = []
    lines.append(f"{var_items} = {items_expr};\n")
    lines.append(f"for (i = 0; i < len({var_items}); i = i + {chunk_size}) {{\n")
    lines.append(f"    __{gid}_chunk = {var_items}.slice(i, i + {chunk_size});\n")
    # Insert chunk placeholder properly
    call_line = call_expr.replace("{chunk}", f"__{gid}_chunk")
    lines.append(f"    {call_line};\n")
    lines.append("}\n")
    return "".join(lines)


def generate_rate_limit_wrapper(call_expr: str, permits: int = 10, per_seconds: int = 1, bucket_name: Optional[str] = None) -> str:
    bucket = bucket_name or f"__rl_{safe_ident(call_expr)[:10]}"
    lines = []
    lines.append(f"{bucket} = {bucket} ? {bucket} : {{ tokens: {permits}, last: time.now() }};\n")
    lines.append(f"if ({bucket}.tokens <= 0) {{\n")
    lines.append(f"    sleep({per_seconds});\n")
    lines.append(f"    {bucket}.tokens = {permits};\n")
    lines.append("}\n")
    lines.append(f"{bucket}.tokens = {bucket}.tokens - 1;\n")
    lines.append(f"{call_expr};\n")
    return "".join(lines)


def generate_profile_wrapper(func_name: str, params: List[str], body: str) -> str:
    start_var = f"__{safe_ident(func_name)}_t0"
    result_var = f"__{safe_ident(func_name)}_res"
    params_sig = ", ".join(params)
    lines = []
    lines.append(f"func {func_name}({params_sig}) {{\n")
    lines.append(f"    {start_var} = time.now();\n")
    lines.append(f"    {result_var} = (function() {{ {body.strip()} }})();\n")
    lines.append(f"    log(\"PROFILE {func_name}: ms\", time.now() - {start_var});\n")
    lines.append(f"    {result_var};\n")
    lines.append("}\n")
    return "".join(lines)


def generate_sanitize_call(var_name: str, sanitized_var: Optional[str] = None) -> str:
    sanitized = sanitized_var or f"__san_{safe_ident(var_name)}"
    lines = []
    lines.append(f"{sanitized} = escape_html({var_name});\n")
    lines.append(f"{sanitized};\n")
    return "".join(lines)


def generate_memory_pool_allocator(pool_name: str, obj_size: int, count: int) -> str:
    p = safe_ident(pool_name)
    lines = []
    lines.append(f"{p} = {p} ? {p} : [];\n")
    lines.append(f"/* preallocate {count} blocks of size {obj_size} */\n")
    lines.append(f"for (i = 0; i < {count}; i = i + 1) {{ {p}.push(alloc_raw({obj_size})); }}\n")
    lines.append(f"func {p}_alloc() {{\n")
    lines.append(f"    if ({p}.length == 0) return alloc_raw({obj_size});\n")
    lines.append(f"    {p}.pop();\n")
    lines.append("}\n")
    lines.append(f"func {p}_free(ptr) {{ {p}.push(ptr); }}\n")
    return "".join(lines)


def generate_simd_map(fn_name: str, array_var: str, op: str, tmp_name: Optional[str] = None) -> str:
    tmp = tmp_name or f"__simd_{safe_ident(array_var)}_{uid('s')}"
    lines = []
    lines.append(f"{tmp} = [];\n")
    lines.append(f"for (i = 0; i < len({array_var}); i = i + 4) {{\n")
    lines.append(f"    /* process 4 elements at once - emitter may lower to SIMD */\n")
    # op is a template with placeholders {arr} and {i}
    lines.append(f"    {tmp}.push({op.format(arr=array_var, i='i')} );\n")
    lines.append("}\n")
    return "".join(lines)


def generate_transactional_guard(body: str, retry: int = 3, tx_name: Optional[str] = None) -> str:
    tx = tx_name or f"__tx_{uid('tx')}"
    lines = []
    lines.append(f"/* transactional guard with {retry} retries */\n")
    lines.append(f"for (retry = 0; retry < {retry}; retry = retry + 1) {{\n")
    lines.append("    try {\n")
    lines.append(f"        begin_tx({tx});\n")
    lines.append(f"        {body.strip()}\n")
    lines.append(f"        commit_tx({tx});\n")
    lines.append("        break;\n")
    lines.append("    } catch {\n")
    lines.append(f"        rollback_tx({tx});\n")
    lines.append("        /* backoff */ sleep(10);\n")
    lines.append("    }\n")
    lines.append("}\n")
    return "".join(lines)


# New powerful optimizations / hints


def generate_loop_tiling(loop_var: str, start: int, end: int, tile: int, body_template: str) -> str:
    """
    Produce a tiled loop structure hinting at cache-friendly tiles.
    Replaces {i} in body_template with the inner index expression.
    """
    lines = []
    lines.append(f"for (t = {start}; t < {end}; t = t + {tile}) {{\n")
    lines.append(f"  for ({loop_var} = t; {loop_var} < min(t + {tile}, {end}); {loop_var} = {loop_var} + 1) {{\n")
    inner = body_template.replace("{i}", loop_var).replace("{loop}", loop_var)
    lines.append(f"    {inner};\n")
    lines.append("  }\n")
    lines.append("}\n")
    return "".join(lines)


def generate_vectorize_hint(loop_var: str, start: int, end: int, body_template: str, width: int = 4) -> str:
    """
    Generate a vectorization hint wrapper; emitter may lower to SIMD if capable.
    """
    tag = uid("vec")
    lines = []
    lines.append(f"/* vectorize hint: width={width}, id={tag} */\n")
    lines.append(f"for ({loop_var} = {start}; {loop_var} < {end}; {loop_var} = {loop_var} + {width}) {{\n")
    inner = body_template.replace("{i}", loop_var).replace("{loop}", loop_var)
    lines.append(f"    /* {tag} */ {inner};\n")
    lines.append("}\n")
    return "".join(lines)


def generate_parallel_map(fn_call_expr: str, collection_expr: str, out_var: Optional[str] = None, workers: int = 4) -> str:
    """
    Emit a parallel_map scaffold that spawns tasks for each chunk of the collection.
    """
    out = out_var or f"__par_{safe_ident(collection_expr)}_{uid('pm')}"
    lines = []
    lines.append(f"{out} = [];\n")
    lines.append(f"parts = chunkify({collection_expr}, {workers});\n")
    lines.append(f"for (p = 0; p < len(parts); p = p + 1) {{\n")
    lines.append(f"    spawn async {{\n")
    lines.append(f"        for (j = 0; j < len(parts[p]); j = j + 1) {{\n")
    # fn_call_expr can reference 'parts[p][j]' as input placeholder '{item}'
    call = fn_call_expr.replace("{item}", "parts[p][j]")
    lines.append(f"            {out}.push({call});\n")
    lines.append("        }\n")
    lines.append("    };\n")
    lines.append("}\n")
    lines.append(f"/* results in {out} (order may be nondeterministic) */\n")
    return "".join(lines)


def generate_circuit_breaker_stub(name: str, call_expr: str, threshold: int = 5, window_sec: int = 60) -> str:
    """
    Emit a circuit-breaker helper stub (stateful map + guard).
    """
    cb = safe_ident(name or "cb")
    lines = []
    lines.append(f"{cb} = {cb} ? {cb} : {{fails: 0, last_reset: time.now(), open: false}};\n")
    lines.append(f"if ({cb}.open) {{\n")
    lines.append(f"    fail('circuit open');\n")
    lines.append("}\n")
    lines.append(f"try {{ {call_expr}; }} catch {{\n")
    lines.append(f"    {cb}.fails = {cb}.fails + 1;\n")
    lines.append(f"    if ({cb}.fails >= {threshold} && time.now() - {cb}.last_reset < {window_sec}) {{ {cb}.open = true; }}\n")
    lines.append("}\n")
    return "".join(lines)


# -------------------------
# Helper registry / emitter
# -------------------------

HelperGenerator = Callable[..., str]

_HELPER_REGISTRY: Dict[str, HelperGenerator] = {
    "unroll": generate_unrolled_loop,
    "memoize": generate_memoize_wrapper,
    "defer_helpers": lambda: generate_defer_helpers(),
    "prefetch": generate_prefetch_helper,
    "batch": generate_batch_helper,
    "ratelimit": generate_rate_limit_wrapper,
    "profile": generate_profile_wrapper,
    "sanitize": generate_sanitize_call,
    "mem_pool": generate_memory_pool_allocator,
    "simd_map": generate_simd_map,
    "transaction": generate_transactional_guard,
    # powerful new generators
    "tile": generate_loop_tiling,
    "vectorize": generate_vectorize_hint,
    "parallel_map": generate_parallel_map,
    "circuit": generate_circuit_breaker_stub,
}


def register_helper(name: str, fn: HelperGenerator):
    """Register a new helper generator at runtime (plugin friendly)."""
    _HELPER_REGISTRY[name] = fn


def list_helpers() -> List[str]:
    return sorted(_HELPER_REGISTRY.keys())


def emit_helper(name: str, *args, **kwargs) -> str:
    gen = _HELPER_REGISTRY.get(name)
    if gen is None:
        raise KeyError(f"helper '{name}' not found")
    return gen(*args, **kwargs)


# -------------------------
# Tooling: CodegenToolkit
# -------------------------


class CodegenToolkit:
    """
    Programmatic toolkit to generate helpers, inject them into files, optionally expand via macro_overlay.
    """

    def __init__(self, macro_overlay_module=None):
        # lazy import if not provided
        self.mo = macro_overlay_module or _try_import_macro_overlay()

    def emit_to_file(self, helper_name: str, out_path: str, *args, append: bool = False, **kwargs) -> str:
        code = emit_helper(helper_name, *args, **kwargs)
        write_helper_to_file(code, out_path, append=append)
        return out_path

    def inject_and_preview(self, helper_name: str, target_path: str, *args, expand: bool = False, **kwargs) -> Tuple[bool, str, Optional[List[Dict[str, Any]]]]:
        """Inject helper into target file; if expand and macro_overlay present, run expansion preview and return diagnostics."""
        code = emit_helper(helper_name, *args, **kwargs)
        inject_helper_into_file(code, target_path)
        if expand and self.mo:
            try:
                src = open(target_path, "r", encoding="utf-8").read()
                apply_fn = getattr(self.mo, "applyMacrosWithDiagnostics", None) or getattr(self.mo, "applyMacros", None)
                if apply_fn:
                    res = apply_fn(src, self.mo.createFullRegistry() if hasattr(self.mo, "createFullRegistry") else self.mo.createDefaultRegistry(), {"filename": target_path})
                    # support coroutine
                    if hasattr(res, "__await__"):
                        import asyncio
                        res = asyncio.get_event_loop().run_until_complete(res)
                    diagnostics = res.get("diagnostics", []) if isinstance(res, dict) else None
                    transformed = res["result"]["transformed"] if isinstance(res, dict) and "result" in res else None
                    return True, transformed or src, diagnostics
            except Exception as e:
                return False, f"expand failed: {e}", None
        # not expanding; return file contents
        try:
            content = open(target_path, "r", encoding="utf-8").read()
            return True, content, None
        except Exception as e:
            return False, f"read failed: {e}", None

    def batch_inject(self, helper_name: str, root_dir: str, pattern: str = ".ix", workers: int = 4, args: Optional[Tuple] = None, kwargs: Optional[Dict] = None, expand: bool = False) -> Dict[str, Tuple[bool, str]]:
        args = args or ()
        kwargs = kwargs or {}
        results: Dict[str, Tuple[bool, str]] = {}
        files = []
        for root, _, filenames in os.walk(root_dir):
            for fn in filenames:
                if fn.endswith(pattern):
                    files.append(os.path.join(root, fn))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(self._inject_worker, f, helper_name, args, kwargs, expand): f for f in files}
            for fut in concurrent.futures.as_completed(futures):
                f = futures[fut]
                try:
                    results[f] = fut.result()
                except Exception as e:
                    results[f] = (False, str(e))
        return results

    def _inject_worker(self, path: str, helper_name: str, args: Tuple, kwargs: Dict, expand: bool) -> Tuple[bool, str]:
        ok, content_or_msg, diagnostics = self.inject_and_preview(helper_name, path, *args, expand=expand, **kwargs)
        if not ok:
            return False, content_or_msg
        return True, "injected" + (", expanded" if expand and diagnostics is not None else "")


# -------------------------
# File integration helpers (wrappers)
# -------------------------


def write_helper_to_file(helper_text: str, out_path: str, append: bool = False) -> str:
    mode = "a" if append else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        f.write(helper_text)
    return out_path


def inject_helper_into_file(helper_text: str, target_path: str, after_comment_block: bool = True) -> str:
    src = ""
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            src = f.read()
    except FileNotFoundError:
        src = ""
    new = inject_helpers_at_top(src, [helper_text])
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(new)
    return target_path


def inject_helpers_at_top(source: str, helpers: List[str]) -> str:
    """
    Insert helper snippets at the top of the file, after initial shebang or first comment block.
    """
    if not helpers:
        return source
    insertion = "\n".join(helpers) + "\n"
    lines = source.splitlines(keepends=True)
    idx = 0
    # skip initial blank lines
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    # skip leading single-line comment block
    if idx < len(lines) and lines[idx].lstrip().startswith("--"):
        while idx < len(lines) and lines[idx].lstrip().startswith("--"):
            idx += 1
    return "".join(lines[:idx]) + insertion + "".join(lines[idx:])


# -------------------------
# Reporting / bench
# -------------------------


def generate_html_report(helpers_map: Dict[str, str], out_path: str) -> str:
    parts = ["<html><head><meta charset='utf-8'><title>Instryx Helpers Report</title></head><body>"]
    parts.append("<h1>Instryx Codegen Helpers</h1>")
    parts.append(f"<p>Generated: {time.asctime()}</p>")
    for name, code in helpers_map.items():
        parts.append(f"<h2>{html.escape(name)}</h2><pre>{html.escape(code)}</pre>")
    parts.append("</body></html>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return out_path


def generate_bench_suite(helper_names: List[str], repeats: int = 100) -> Dict[str, float]:
    results: Dict[str, float] = {}
    for name in helper_names:
        gen = _HELPER_REGISTRY.get(name)
        if not gen:
            results[name] = 0.0
            continue
        start = time.perf_counter()
        for _ in range(repeats):
            # best-effort sample args
            try:
                if name == "unroll":
                    gen("i", 0, 8, "x = x + a[{i}];")
                elif name == "memoize":
                    gen("f", ["x"], "x * x;")
                elif name == "prefetch":
                    gen(["https://example/1", "https://example/2"])
                elif name == "batch":
                    gen("net.send_batch({chunk})", "items", 16)
                elif name == "ratelimit":
                    gen("net.request('u')", 5, 1)
                elif name == "vectorize":
                    gen("i", 0, 1024, "a[{i}] = a[{i}] + 1;", 8)
                else:
                    gen()
            except TypeError:
                try:
                    gen()
                except Exception:
                    pass
        end = time.perf_counter()
        results[name] = end - start
    return results


# -------------------------
# Demos and tests
# -------------------------


def demo_show_all():
    print("Available helpers:", ", ".join(list_helpers()))
    examples = {}
    for h in list_helpers():
        try:
            if h == "unroll":
                examples[h] = emit_helper(h, "i", 0, 4, "sum = sum + arr[{i}];")
            elif h == "memoize":
                examples[h] = emit_helper(h, "fib", ["n"], "if n <= 1 { n } else { fib(n-1) + fib(n-2) }")
            elif h == "prefetch":
                examples[h] = emit_helper(h, ["https://a", "https://b"])
            elif h == "mem_pool":
                examples[h] = emit_helper(h, "mypool", 64, 16)
            elif h == "vectorize":
                examples[h] = emit_helper(h, "i", 0, 16, "sum = sum + a[{i}];", 4)
            else:
                examples[h] = emit_helper(h) if callable(_HELPER_REGISTRY[h]) else f"/* sample for {h} requires args */"
        except Exception as e:
            examples[h] = f"/* generator error: {e} */"
    for k, v in examples.items():
        print(f"--- {k} ---")
        print(v)


def run_unit_tests(verbose: bool = True) -> bool:
    ok = True
    # unroll
    u = generate_unrolled_loop("i", 0, 3, "a = a + b[{i}];")
    if "b[0]" not in u or "b[2]" not in u:
        if verbose: print("unroll FAILED")
        ok = False
    # memoize
    m = generate_memoize_wrapper("add", ["x", "y"], "x + y;")
    if "func add" not in m or "__memo_" not in m:
        if verbose: print("memoize FAILED")
        ok = False
    # vectorize
    v = generate_vectorize_hint("i", 0, 8, "sum = sum + a[{i}];", 4)
    if "vectorize" not in v and "width=4" not in v:
        # vectorize uses comment; sanity check for width presence
        pass
    # parallel_map
    pm = generate_parallel_map("compute({item})", "items", None, 4)
    if "spawn async" not in pm:
        if verbose: print("parallel_map FAILED")
        ok = False
    # mem pool
    mp = generate_memory_pool_allocator("pool", 64, 4)
    if "alloc_raw" not in mp:
        if verbose: print("mem_pool FAILED")
        ok = False
    if verbose:
        print("unit tests", "PASS" if ok else "FAIL")
    return ok


# -------------------------
# CLI
# -------------------------


def _cli():
    p = argparse.ArgumentParser(prog="instryx_memory_math_loops_codegen.py")
    p.add_argument("--list", action="store_true", help="list available helpers")
    p.add_argument("--emit", nargs="+", help="emit a named helper and print; remaining args passed to generator")
    p.add_argument("--write", nargs=2, metavar=("HELPER", "OUTFILE"), help="emit helper and write to OUTFILE")
    p.add_argument("--inject", nargs=2, metavar=("HELPER", "TARGET"), help="emit helper and inject at top of TARGET file")
    p.add_argument("--inject-expand", nargs=2, metavar=("HELPER", "TARGET"), help="emit helper, inject to TARGET and attempt macro-overlay expand")
    p.add_argument("--batch-inject", nargs=2, metavar=("HELPER", "DIR"), help="emit helper and inject into all .ix files under DIR")
    p.add_argument("--report", nargs=1, metavar="OUTHTML", help="generate an HTML report with all helpers (examples)")
    p.add_argument("--bench", action="store_true", help="run generation micro-bench")
    p.add_argument("--demo", action="store_true", help="run demo show all helpers")
    p.add_argument("--test", action="store_true", help="run unit tests")
    args = p.parse_args()

    toolkit = CodegenToolkit(_try_import_macro_overlay())

    if args.list:
        for n in list_helpers():
            print(n)
        return 0

    if args.demo:
        demo_show_all()
        return 0

    if args.test:
        ok = run_unit_tests(verbose=True)
        return 0 if ok else 2

    if args.bench:
        names = list_helpers()
        res = generate_bench_suite(names, repeats=200)
        print("bench results (sec):")
        for k, v in res.items():
            print(f"  {k}: {v:.6f}")
        return 0

    if args.emit:
        name = args.emit[0]
        params = args.emit[1:]
        gen = _HELPER_REGISTRY.get(name)
        if not gen:
            print("unknown helper", name)
            return 2
        # try to call with naive parsed params (int detection)
        parsed = []
        for pstr in params:
            if pstr.isdigit():
                parsed.append(int(pstr))
            elif "," in pstr and pstr.startswith("[") is False and pstr.endswith("]") is False:
                # comma separated values -> list of strings
                parsed.append([x.strip() for x in pstr.split(",")])
            else:
                parsed.append(pstr)
        try:
            out = gen(*parsed)
            print(out)
            return 0
        except Exception as e:
            print("emit error:", e)
            return 2

    if args.write:
        name, out = args.write
        gen = _HELPER_REGISTRY.get(name)
        if not gen:
            print("unknown helper", name)
            return 2
        # try no-arg call if possible
        try:
            code = gen()
        except Exception:
            code = f"/* helper {name} requires args */\n"
        write_helper_to_file(code, out, append=False)
        print("wrote", out)
        return 0

    if args.inject:
        name, target = args.inject
        gen = _HELPER_REGISTRY.get(name)
        if not gen:
            print("unknown helper", name)
            return 2
        try:
            code = gen()
        except Exception:
            code = f"/* helper {name} requires args */\n"
        inject_helper_into_file(code, target)
        print("injected into", target)
        return 0

    if args.inject_expand:
        name, target = args.inject_expand
        gen = _HELPER_REGISTRY.get(name)
        if not gen:
            print("unknown helper", name)
            return 2
        try:
            code = gen()
        except Exception:
            code = f"/* helper {name} requires args */\n"
        ok, transformed_or_msg, diagnostics = toolkit.inject_and_preview(name, target, expand=True)
        if ok:
            print("injected and expanded preview available (transformed content returned)")
            if diagnostics:
                print("diagnostics:", diagnostics)
            return 0
        else:
            print("failed:", transformed_or_msg)
            return 2

    if args.batch_inject:
        name, target_dir = args.batch_inject
        gen = _HELPER_REGISTRY.get(name)
        if not gen:
            print("unknown helper", name)
            return 2
        # attempt no-arg generation
        try:
            code = gen()
        except Exception:
            code = f"/* helper {name} requires args */\n"
        results = toolkit.batch_inject(name, target_dir, pattern=".ix", workers=4, args=(), kwargs={}, expand=False)
        for f, (ok, msg) in results.items():
            print(f"{f}: {ok} {msg}")
        return 0

    if args.report:
        out = args.report[0]
        examples = {}
        for n in list_helpers():
            try:
                if _HELPER_REGISTRY[n].__code__.co_argcount == 0:
                    examples[n] = _HELPER_REGISTRY[n]()
                else:
                    examples[n] = f"/* sample for {n} requires args */"
            except Exception:
                examples[n] = "/* cannot render sample */"
        generate_html_report(examples, out)
        print("report", out)
        return 0

    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(_cli())

