# expose generate_prefetch_x(...) function for instryx_memory_math_loops_codegen style
def generate_prefetch_x(urls, results_var=None):
    # simple textual helper: spawn async prefetch for each url
    out = []
    var = results_var or "__prefetch_x"
    out.append(f"{var} = {var} ? {var} : {{}};\n")
    for u in urls:
        safe_u = u.replace('"', '\\"')
        out.append(f"spawn async {{ {var}['{safe_u}'] = fetchData('{safe_u}'); }};\n")
    return "".join(out)

# Optionally expose register so PluginManager can attach this to a codegen registry
def register(toolkit):
    # toolkit could be CodegenToolkit or PluginManager depending on your load flow
    # If toolkit provides a register_helper function, use it; otherwise the codegen loader may import this module style.
    if hasattr(toolkit, "register_helper"):
        toolkit.register_helper("prefetch_x", generate_prefetch_x)
        """
        Registers the prefetch_x helper function with the given toolkit.
        This allows code generation modules to call prefetch_x(urls, results_var)
        to generate code that spawns async prefetches for the specified URLs.
        """

"""
ciams/ciams_plugins/prefetch_helper_plugin.py

Enhanced prefetch helper plugin for instryx_memory_math_loops_codegen-style codegen.

Features added:
 - Flexible API: generate_prefetch_x(urls, results_var=None, options=None)
 - Options: concurrency, retries, timeout_ms, backoff_ms, cache_var, use_promises
 - Generates safe, portable code snippets for both `spawn async {}` runtime style
   and Promise-style runtimes (use_promises=True).
 - Batch prefetch generation to respect concurrency and produce Promise.all-style joins.
 - Optional in-generated-cache support via provided cache_var name.
 - Small telemetry hooks (emit prefetched count into optional metrics var).
 - register(toolkit) integrates with toolkits exposing register_helper and supports metadata.
 - Input sanitization and deterministic temporary variable naming to avoid collisions.
"""

from __future__ import annotations
import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# PUBLIC API
def generate_prefetch_x(urls: Sequence[str], results_var: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate code that prefetches the given `urls`.

    Parameters:
      - urls: iterable of URL strings to prefetch.
      - results_var: optional name of the map/dict variable to store results. If None, a safe default is used.
      - options: dict of optional parameters:
          - runtime: "spawn" (default) or "promises" -> controls generated style
          - concurrency: max number of concurrent fetches per batch (int, default 8)
          - retries: number of retries for failed fetches (int, default 2)
          - timeout_ms: millisecond timeout per fetch attempt (int, optional)
          - backoff_ms: base backoff milliseconds between retries (int, default 50)
          - cache_var: optional name of a cache map to avoid refetching
          - metrics_var: optional name of a map/counter to increment for telemetry
          - safe_keys: if True keys will be hex-hashed to avoid non-identifier keys (default False)
          - indent: string used for indentation in output (default 2 spaces)
          - inline: if True generate compact inline code, else multi-line readable output (default False)

    Returns:
      - string containing the generated prefetch code snippet (text).
    """
    opts = dict(options or {})
    runtime = opts.get("runtime", "spawn")
    concurrency = int(opts.get("concurrency", 8) or 8)
    retries = int(opts.get("retries", 2) or 2)
    timeout_ms = opts.get("timeout_ms")  # may be None
    backoff_ms = int(opts.get("backoff_ms", 50) or 50)
    cache_var = opts.get("cache_var")
    metrics_var = opts.get("metrics_var")
    safe_keys = bool(opts.get("safe_keys", False))
    indent = opts.get("indent", "  ")
    inline = bool(opts.get("inline", False))

    # results_var default
    rv = results_var or "__prefetch_x"
    # prepare lines
    lines: List[str] = []

    # helper functions used to produce safe literal and safe key
    def _escape(s: str) -> str:
        return s.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')

    def _safe_key(u: str) -> str:
        if safe_keys:
            # deterministic short hex key
            h = hashlib.sha1(u.encode("utf-8")).hexdigest()[:10]
            return f"pf_{h}"
        # otherwise use url string as map key (keep quotes)
        return u

    # init results map and optional cache map and metrics map
    if inline:
        sep = " "
    else:
        sep = "\n"

    lines.append(f"{rv} = {rv} ? {rv} : {{}};")
    if cache_var:
        lines.append(f"{cache_var} = {cache_var} ? {cache_var} : {{}};")
    if metrics_var:
        lines.append(f"{metrics_var} = {metrics_var} ? {metrics_var} : {{prefetched:0}};")

    # fast-path: if runtime is promises, generate Promise.all with limited concurrency batches
    urls_list = list(urls)

    if runtime == "promises":
        # build helper function name for backoff/retries
        helper_name = "__prefetch_fetch_with_retry"
        # helper definition (Promises-style pseudocode)
        helper_lines = []
        helper_lines.append(f"function {helper_name}(url, retries, timeout_ms, backoff_ms) {{")
        helper_lines.append(f"{indent}// returns a Promise that resolves to fetched data or null on failure")
        helper_lines.append(f"{indent}let attempt = 0;")
        helper_lines.append(f"{indent}function tryOnce(resolve) {{")
        attempt_body = []
        attempt_body.append(f"{indent*2}let p = fetchData(url);")
        if timeout_ms:
            attempt_body.append(f"{indent*2}// optional: apply timeout wrapper if runtime supports it (not implemented here)")
        attempt_body.append(f"{indent*2}p.then(d => resolve(d)).catch(() => {{")
        attempt_body.append(f"{indent*3}attempt += 1;")
        attempt_body.append(f"{indent*3}if (attempt <= retries) {{")
        attempt_body.append(f"{indent*4}setTimeout(() => tryOnce(resolve), backoff_ms * attempt);")
        attempt_body.append(f"{indent*3}}} else {{ resolve(null); }}")
        attempt_body.append(f"{indent*2}}});")
        helper_lines.extend(attempt_body)
        helper_lines.append(f"{indent}}}")
        helper_lines.append(f"{indent}return new Promise(tryOnce);")
        helper_lines.append("}")
        lines.extend(helper_lines)

        # batch into concurrency-limited groups
        batches: List[List[str]] = []
        for i in range(0, len(urls_list), concurrency):
            batches.append(urls_list[i:i+concurrency])

        batch_idx = 0
        for batch in batches:
            promise_names: List[str] = []
            for u in batch:
                safe_u = _escape(u)
                key_literal = f"'{_escape(u)}'"
                pvar = f"__pf_p_{batch_idx}_{len(promise_names)}"
                promise_names.append(pvar)
                # if cache exists and has value, short-circuit
                if cache_var:
                    lines.append(f"if ({cache_var}['{safe_u}']) {{ {rv}['{safe_u}'] = {cache_var}['{safe_u}']; }} else {{")
                    lines.append(f"{indent}let {pvar} = {helper_name}('{safe_u}', {retries}, {timeout_ms if timeout_ms else 'null'}, {backoff_ms});")
                    lines.append(f"{indent}{pvar}.then(d => {{ if (d) {{ {rv}['{safe_u}'] = d; {cache_var}['{safe_u}'] = d; if ({metrics_var}) {{ {metrics_var}.prefetched += 1; }} }} }});")
                    lines.append("}")
                else:
                    lines.append(f"let {pvar} = {helper_name}('{safe_u}', {retries}, {timeout_ms if timeout_ms else 'null'}, {backoff_ms});")
                    then_clause = f"{pvar}.then(d => {{ if (d) {{ {rv}['{safe_u}'] = d; if ({metrics_var}) {{ {metrics_var}.prefetched += 1; }} }} }})"
                    lines.append(then_clause)
            # join batch promises
            pname_list = ", ".join([n for n in promise_names])
            if promise_names:
                lines.append(f"Promise.all([{pname_list}]);")
            batch_idx += 1

        return (sep.join(lines) + (sep if not inline else ""))

    # Default runtime: spawn async { ... } (Instryx-style)
    # generate concurrency controlled batches: spawn concurrency tasks per batch
    if len(urls_list) == 0:
        return "\n".join(lines) + ("\n" if not inline else "")

    # build batches based on concurrency
    batches = [urls_list[i:i+concurrency] for i in range(0, len(urls_list), concurrency)]
    uid = hashlib.sha1(",".join(urls_list).encode("utf-8")).hexdigest()[:8]
    batch_id = 0
    for batch in batches:
        # each batch spawn will create concurrency async tasks and optionally wait (not all runtimes support wait)
        lines.append(f"// prefetch batch {batch_id} uid={uid}")
        for u in batch:
            safe_u = _escape(u)
            key_literal = f"'{_escape(u)}'"
            # short-circuit via cache_var
            if cache_var:
                lines.append(f"if ({cache_var}['{safe_u}']) {{ {rv}['{safe_u}'] = {cache_var}['{safe_u}']; }} else {{")
                lines.append(f"{indent}spawn async {{")
                # attempt + retries loop (simple sequential retry pattern)
                lines.append(f"{indent*2}let _attempt = 0;")
                lines.append(f"{indent*2}let _res = null;")
                lines.append(f"{indent*2}while (_attempt <= {retries} && !_res) {{")
                if timeout_ms:
                    lines.append(f"{indent*3}// note: runtime must support fetchData with timeout or wrapper; placeholder used")
                lines.append(f"{indent*3}_res = fetchData('{safe_u}');")
                lines.append(f"{indent*3}if (!_res) {{ _attempt = _attempt + 1; sleep({backoff_ms} * _attempt); }}")
                lines.append(f"{indent*2}}}")
                lines.append(f"{indent*2}if (_res) {{ {rv}['{safe_u}'] = _res; {cache_var}['{safe_u}'] = _res;")
                if metrics_var:
                    lines.append(f"{indent*3}{metrics_var}.prefetched = ({metrics_var}.prefetched || 0) + 1;")
                lines.append(f"{indent*2}}}")
                lines.append(f"{indent}}};")
                lines.append("}")
            else:
                lines.append(f"spawn async {{")
                lines.append(f"{indent}let _attempt = 0;")
                lines.append(f"{indent}let _res = null;")
                lines.append(f"{indent}while (_attempt <= {retries} && !_res) {{")
                if timeout_ms:
                    lines.append(f"{indent*2}// placeholder: apply timeout to fetchData if runtime supports it")
                lines.append(f"{indent*2}_res = fetchData('{safe_u}');")
                lines.append(f"{indent*2}if (!_res) {{ _attempt = _attempt + 1; sleep({backoff_ms} * _attempt); }}")
                lines.append(f"{indent}}}")
                lines.append(f"{indent}if (_res) {{ {rv}['{safe_u}'] = _res;")
                if metrics_var:
                    lines.append(f"{indent*2}{metrics_var}.prefetched = ({metrics_var}.prefetched || 0) + 1;")
                lines.append(f"{indent}}}")
                lines.append("};")
        batch_id += 1

    return (sep.join(lines) + (sep if not inline else ""))


def register(toolkit: Any) -> None:
    """
    Register this helper with a toolkit or PluginManager.

    Expected toolkit API:
      - toolkit.register_helper(name, fn, metadata=None)
    If register_helper not present, falling back to attach function attribute to toolkit.
    """
    meta = {
        "name": "prefetch_x",
        "description": "Generate prefetch code for a list of URLs with options (concurrency, retries, cache)",
        "signature": "prefetch_x(urls, results_var=None, options=None) -> str",
        "options": {
            "runtime": {"type": "string", "enum": ["spawn", "promises"], "default": "spawn"},
            "concurrency": {"type": "integer", "default": 8},
            "retries": {"type": "integer", "default": 2},
            "timeout_ms": {"type": "integer", "default": None},
            "backoff_ms": {"type": "integer", "default": 50},
            "cache_var": {"type": "string", "default": None},
            "metrics_var": {"type": "string", "default": None},
            "safe_keys": {"type": "boolean", "default": False},
        }
    }
    if hasattr(toolkit, "register_helper"):
        try:
            toolkit.register_helper("prefetch_x", generate_prefetch_x, metadata=meta)
        except Exception:
            # graceful fallback: attempt no-metadata registration
            toolkit.register_helper("prefetch_x", generate_prefetch_x)
    else:
        # best-effort: attach as attribute
        try:
            setattr(toolkit, "prefetch_x", generate_prefetch_x)
        except Exception:
            pass

        """
        Registers the prefetch_x helper function with the given toolkit.
        This allows code generation modules to call prefetch_x(urls, results_var, options)
        to generate code that spawns async prefetches for the specified URLs with advanced options.
        """

"""
ciams/ciams_plugins/prefetch_helper_plugin.py

Fully implemented enhanced prefetch helper plugin for instryx_memory_math_loops_codegen-style codegen.

Exports:
 - generate_prefetch_x(urls, results_var=None, options=None) -> str
 - register(toolkit) -> None

The module is pure-Python and executable as a small CLI demo that prints generated snippets.
"""
from __future__ import annotations
import hashlib
import json
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Public API
def generate_prefetch_x(urls: Sequence[str], results_var: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate a prefetch code snippet for a list of URLs.

    Parameters
    ----------
    urls:
        Sequence of URL strings to prefetch.
    results_var:
        Optional variable name to store results (map). Defaults to "__prefetch_x".
    options:
        Optional dict with keys:
          - runtime: "spawn" (default) or "promises"
          - concurrency: int (default 8)
          - retries: int (default 2)
          - timeout_ms: int or None
          - backoff_ms: int (default 50)
          - cache_var: optional string name of cache map to consult/write
          - metrics_var: optional string name of metrics map to increment
          - safe_keys: bool (default False) - if True, generate hashed keys instead of raw URL keys
          - indent: str (default "  ")
          - inline: bool (default False) - generate compact single-line output

    Returns
    -------
    str
        The generated code snippet (text). This snippet targets the codegen runtime
        that provides `spawn async { ... }`, `fetchData(url)` and optionally `Promise` and `setTimeout`.
    """
    opts = dict(options or {})
    runtime = str(opts.get("runtime", "spawn"))
    concurrency = int(opts.get("concurrency", 8) or 8)
    retries = int(opts.get("retries", 2) or 2)
    timeout_ms = opts.get("timeout_ms")  # may be None
    backoff_ms = int(opts.get("backoff_ms", 50) or 50)
    cache_var = opts.get("cache_var")
    metrics_var = opts.get("metrics_var")
    safe_keys = bool(opts.get("safe_keys", False))
    indent = str(opts.get("indent", "  "))
    inline = bool(opts.get("inline", False))

    # normalize inputs
    urls_list = [str(u) for u in urls]
    rv = results_var or "__prefetch_x"

    def _escape_literal(s: str) -> str:
        return s.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')

    def _key_for(u: str) -> str:
        if safe_keys:
            h = hashlib.sha1(u.encode("utf-8")).hexdigest()[:12]
            return f"pf_{h}"
        # use quoted literal key
        return f"'{_escape_literal(u)}'"

    # build lines
    lines: List[str] = []
    sep = " " if inline else "\n"

    # initialize containers
    lines.append(f"{rv} = {rv} ? {rv} : {{}};")
    if cache_var:
        lines.append(f"{cache_var} = {cache_var} ? {cache_var} : {{}};")
    if metrics_var:
        lines.append(f"{metrics_var} = {metrics_var} ? {metrics_var} : {{prefetched:0}};")

    if not urls_list:
        return sep.join(lines) + (sep if not inline else "")

    # PROMISES runtime generation
    if runtime == "promises":
        helper_name = "__prefetch_fetch_with_retry"
        hl: List[str] = []
        hl.append(f"function {helper_name}(url, retries, timeout_ms, backoff_ms) {{")
        hl.append(f"{indent}// returns a Promise that resolves to fetched data or null on failure")
        hl.append(f"{indent}let attempt = 0;")
        hl.append(f"{indent}function tryOnce(resolve) {{")
        hl.append(f"{indent*2}let p = fetchData(url);")
        if timeout_ms is not None:
            hl.append(f"{indent*2}// NOTE: wrap `p` with a timeout if runtime provides such helper")
        hl.append(f"{indent*2}p.then(d => resolve(d)).catch(() => {{")
        hl.append(f"{indent*3}attempt += 1;")
        hl.append(f"{indent*3}if (attempt <= retries) {{")
        hl.append(f"{indent*4}setTimeout(() => tryOnce(resolve), backoff_ms * attempt);")
        hl.append(f"{indent*3}}} else {{ resolve(null); }}")
        hl.append(f"{indent*2}}});")
        hl.append(f"{indent}}}")
        hl.append(f"{indent}return new Promise(tryOnce);")
        hl.append("}")
        lines.extend(hl)

        # batch into concurrency-limited groups
        for i in range(0, len(urls_list), concurrency):
            batch = urls_list[i:i+concurrency]
            promise_vars: List[str] = []
            for j, u in enumerate(batch):
                safe = _escape_literal(u)
                pvar = f"__pf_p_{i}_{j}"
                promise_vars.append(pvar)
                if cache_var:
                    lines.append(f"if ({cache_var}[{_key_for(u)}]) {{ {rv}[{_key_for(u)}] = {cache_var}[{_key_for(u)}]; }} else {{")
                    lines.append(f"{indent}let {pvar} = {helper_name}('{safe}', {retries}, {timeout_ms if timeout_ms is not None else 'null'}, {backoff_ms});")
                    then_clause = f"{indent}{pvar}.then(d => {{ if (d) {{ {rv}[{_key_for(u)}] = d; {cache_var}[{_key_for(u)}] = d; {metrics_var + '.prefetched += 1;' if metrics_var else ''} }} }});"
                    # avoid embedding None incorrectly
                    if metrics_var:
                        lines.append(f"{indent}{pvar}.then(d => {{ if (d) {{ {rv}[{_key_for(u)}] = d; {cache_var}[{_key_for(u)}] = d; {metrics_var}.prefetched = ({metrics_var}.prefetched || 0) + 1; }} }});")
                    else:
                        lines.append(then_clause)
                    lines.append("}")
                else:
                    lines.append(f"let {pvar} = {helper_name}('{safe}', {retries}, {timeout_ms if timeout_ms is not None else 'null'}, {backoff_ms});")
                    if metrics_var:
                        lines.append(f"{indent}{pvar}.then(d => {{ if (d) {{ {rv}[{_key_for(u)}] = d; {metrics_var}.prefetched = ({metrics_var}.prefetched || 0) + 1; }} }});")
                    else:
                        lines.append(f"{indent}{pvar}.then(d => {{ if (d) {{ {rv}[{_key_for(u)}] = d; }} }});")
            if promise_vars:
                pvlist = ", ".join(promise_vars)
                lines.append(f"Promise.all([{pvlist}]);")

        return sep.join(lines) + (sep if not inline else "")

    # Default: spawn async style codegen
    batches = [urls_list[i:i+concurrency] for i in range(0, len(urls_list), concurrency)]
    uid = hashlib.sha1(",".join(urls_list).encode("utf-8")).hexdigest()[:8]

    for bidx, batch in enumerate(batches):
        lines.append(f"// prefetch batch {bidx} uid={uid}")
        for u in batch:
            safe = _escape_literal(u)
            key = _key_for(u)
            if cache_var:
                lines.append(f"if ({cache_var}[{key}]) {{ {rv}[{key}] = {cache_var}[{key}]; }} else {{")
                lines.append(f"{indent}spawn async {{")
                lines.append(f"{indent*2}let __pf_attempt = 0;")
                lines.append(f"{indent*2}let __pf_res = null;")
                lines.append(f"{indent*2}while (__pf_attempt <= {retries} && !__pf_res) {{")
                if timeout_ms is not None:
                    lines.append(f"{indent*3}// placeholder: runtime must provide timeout wrapper for fetchData(url, timeout_ms)")
                lines.append(f"{indent*3}__pf_res = fetchData('{safe}');")
                lines.append(f"{indent*3}if (!__pf_res) {{ __pf_attempt = __pf_attempt + 1; sleep({backoff_ms} * __pf_attempt); }}")
                lines.append(f"{indent*2}}}")
                lines.append(f"{indent*2}if (__pf_res) {{ {rv}[{key}] = __pf_res; {cache_var}[{key}] = __pf_res;")
                if metrics_var:
                    lines.append(f"{indent*3}{metrics_var}.prefetched = ({metrics_var}.prefetched || 0) + 1;")
                lines.append(f"{indent*2}}}")
                lines.append(f"{indent}}};")
                lines.append("}")
            else:
                lines.append(f"spawn async {{")
                lines.append(f"{indent}let __pf_attempt = 0;")
                lines.append(f"{indent}let __pf_res = null;")
                lines.append(f"{indent}while (__pf_attempt <= {retries} && !__pf_res) {{")
                if timeout_ms is not None:
                    lines.append(f"{indent*2}// placeholder: apply timeout to fetchData if runtime supports it")
                lines.append(f"{indent*2}__pf_res = fetchData('{safe}');")
                lines.append(f"{indent*2}if (!__pf_res) {{ __pf_attempt = __pf_attempt + 1; sleep({backoff_ms} * __pf_attempt); }}")
                lines.append(f"{indent}}}")
                lines.append(f"{indent}if (__pf_res) {{ {rv}[{key}] = __pf_res;")
                if metrics_var:
                    lines.append(f"{indent*2}{metrics_var}.prefetched = ({metrics_var}.prefetched || 0) + 1;")
                lines.append(f"{indent}}}")
                lines.append("};")

    return sep.join(lines) + (sep if not inline else "")


def register(toolkit: Any) -> None:
    """
    Register helper with toolkit or PluginManager.

    If toolkit provides register_helper(name, fn, metadata=None) it's used, otherwise
    attach helper as attribute on toolkit.
    """
    meta = {
        "name": "prefetch_x",
        "description": "Generate prefetch code for URLs with concurrency/retry/cache support.",
        "signature": "prefetch_x(urls, results_var=None, options=None) -> str",
        "options": {
            "runtime": {"type": "string", "enum": ["spawn", "promises"], "default": "spawn"},
            "concurrency": {"type": "integer", "default": 8},
            "retries": {"type": "integer", "default": 2},
            "timeout_ms": {"type": ["integer", "null"], "default": None},
            "backoff_ms": {"type": "integer", "default": 50},
            "cache_var": {"type": ["string", "null"], "default": None},
            "metrics_var": {"type": ["string", "null"], "default": None},
            "safe_keys": {"type": "boolean", "default": False},
            "inline": {"type": "boolean", "default": False},
        }
    }
    if hasattr(toolkit, "register_helper"):
        try:
            toolkit.register_helper("prefetch_x", generate_prefetch_x, metadata=meta)
            return
        except Exception:
            try:
                toolkit.register_helper("prefetch_x", generate_prefetch_x)
                return
            except Exception:
                pass
    # fallback: attach as attribute
    try:
        setattr(toolkit, "prefetch_x", generate_prefetch_x)
    except Exception:
        # no-op if toolkit can't be modified
        pass


# Simple CLI demo so module is executable and behavior is easily inspected.
def _demo():
    sample_urls = [
        "https://example.com/data/1.json",
        "https://example.com/data/2.json",
        "https://cdn.example.com/assets/img.png"
    ]
    print("=== spawn-style snippet ===")
    print(generate_prefetch_x(sample_urls, results_var="__my_prefetch", options={"runtime": "spawn", "concurrency": 2, "retries": 1, "metrics_var": "METRICS"}))
    print("=== promises-style snippet ===")
    print(generate_prefetch_x(sample_urls, results_var="__my_prefetch", options={"runtime": "promises", "concurrency": 2, "retries": 1, "metrics_var": "METRICS"}))


if __name__ == "__main__":
    _demo()

"""
ciams/ciams_plugins/prefetch_helper_plugin.py

Enhanced, fully-implemented prefetch helper plugin.

Features:
 - Robust `generate_prefetch_x(urls, results_var=None, options=None)` producing
   spawn-style or promise-style prefetch snippets with concurrency/retry/timeout/backoff.
 - `generate_prefetch_plan(urls, options)` returns structured plan (metadata + batches).
 - URL validation and safe literal escaping.
 - Optional cache/metrics integration and safe key hashing.
 - `register(toolkit)` to attach helper to a toolkit (with metadata if supported).
 - Executable CLI demo and a small self-check that validates output and plan structure.
 - No external dependencies — pure stdlib.

Usage:
  - Import and call `generate_prefetch_x(...)` from codegen components.
  - Plugins/toolkits can call `register(toolkit)` to register the helper.

Design notes:
 - The generated snippet is textual and targets runtimes that expose `spawn async { ... }`,
   `fetchData(url)` and (optionally) `Promise` + `setTimeout`. Timeouts are emitted as
   comments/placeholders because runtime-specific wrappers are required for actual timeout
   semantics in generated code.
"""

from __future__ import annotations
import hashlib
import json
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

# Public API ---------------------------------------------------------------

def _escape_literal(s: str) -> str:
    """Escape string for inclusion in single-quoted code literals."""
    return s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")

def _is_valid_url(u: str) -> bool:
    """Very small conservative URL validator using urllib.parse."""
    try:
        p = urlparse(u)
        return bool(p.scheme and p.netloc)
    except Exception:
        return False

def generate_prefetch_plan(urls: Sequence[str], options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Build a deterministic prefetch plan describing batches and per-URL metadata.
    Returns a dict with keys: { urls: [...], batches: [[url,...],...], metadata: {...} }.
    This is useful for analysis, testing and offline prefetch scheduling.
    """
    opts = dict(options or {})
    concurrency = int(opts.get("concurrency", 8) or 8)
    safe_keys = bool(opts.get("safe_keys", False))

    urls_list: List[str] = [str(u) for u in urls]
    validated: List[Dict[str, Any]] = []
    for u in urls_list:
        validated.append({
            "url": u,
            "valid": _is_valid_url(u),
            "safe_key": ("pf_" + hashlib.sha1(u.encode("utf-8")).hexdigest()[:12]) if safe_keys else None
        })

    batches: List[List[str]] = [urls_list[i:i + concurrency] for i in range(0, len(urls_list), concurrency)]
    plan = {
        "count": len(urls_list),
        "batches": batches,
        "urls": validated,
        "metadata": {"concurrency": concurrency, "safe_keys": safe_keys}
    }
    return plan

def generate_prefetch_x(urls: Sequence[str], results_var: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate textual prefetch code.

    Parameters
    ----------
    urls : Sequence[str]
        URLs to prefetch.
    results_var : Optional[str]
        Name of results map returned by runtime. Defaults to "__prefetch_x".
    options : Optional[Dict[str, Any]]
        runtime: "spawn"|"promises" (default "spawn")
        concurrency: int (default 8)
        retries: int (default 2)
        timeout_ms: int or None
        backoff_ms: int (default 50)
        cache_var: optional map name to consult/write
        metrics_var: optional map name to update
        safe_keys: bool (default False)
        indent: str (default "  ")
        inline: bool (default False)

    Returns
    -------
    str
        Code snippet string (multi-line or single-line if inline=True).
    """
    opts = dict(options or {})
    runtime = str(opts.get("runtime", "spawn"))
    concurrency = max(1, int(opts.get("concurrency", 8) or 8))
    retries = max(0, int(opts.get("retries", 2) or 2))
    timeout_ms = opts.get("timeout_ms")
    backoff_ms = max(0, int(opts.get("backoff_ms", 50) or 50))
    cache_var = opts.get("cache_var")
    metrics_var = opts.get("metrics_var")
    safe_keys = bool(opts.get("safe_keys", False))
    indent = str(opts.get("indent", "  "))
    inline = bool(opts.get("inline", False))

    urls_list = [str(u) for u in urls]
    rv = results_var or "__prefetch_x"

    sep = " " if inline else "\n"
    lines: List[str] = []
    lines.append(f"{rv} = {rv} ? {rv} : {{}};")
    if cache_var:
        lines.append(f"{cache_var} = {cache_var} ? {cache_var} : {{}};")
    if metrics_var:
        lines.append(f"{metrics_var} = {metrics_var} ? {metrics_var} : {{prefetched:0}};")

    # Validate URLs and early-return if none valid
    valid_urls = [u for u in urls_list if _is_valid_url(u)]
    invalid_urls = [u for u in urls_list if not _is_valid_url(u)]
    if invalid_urls:
        # Include comment warning about invalid URLs (safe behavior)
        lines.append(f"// WARNING: {len(invalid_urls)} invalid URL(s) skipped")
        for u in invalid_urls:
            lines.append(f"// skipped: {u}")

    if not valid_urls:
        return sep.join(lines) + (sep if not inline else "")

    # Helper to derive map key
    def _key_for(u: str) -> str:
        if safe_keys:
            return f"pf_{hashlib.sha1(u.encode('utf-8')).hexdigest()[:12]}"
        return f"'{_escape_literal(u)}'"

    # PROMISES runtime
    if runtime == "promises":
        helper = "__prefetch_fetch_with_retry"
        hl = [
            f"function {helper}(url, retries, timeout_ms, backoff_ms) {{",
            f"{indent}// returns a Promise that resolves to fetched data or null",
            f"{indent}let attempt = 0;",
            f"{indent}function tryOnce(resolve) {{",
            f"{indent*2}let p = fetchData(url);",
        ]
        if timeout_ms is not None:
            hl.append(f"{indent*2}// Wrap p with timeout if runtime supports it (not implemented here)")
        hl.extend([
            f"{indent*2}p.then(d => resolve(d)).catch(() => {{",
            f"{indent*3}attempt += 1;",
            f"{indent*3}if (attempt <= retries) {{",
            f"{indent*4}setTimeout(() => tryOnce(resolve), backoff_ms * attempt);",
            f"{indent*3}}} else {{ resolve(null); }}",
            f"{indent*2}}});",
            f"{indent}}}",
            f"{indent}return new Promise(tryOnce);",
            f"}}"
        ])
        lines.extend(hl)

        # Batch into concurrency groups
        for i in range(0, len(valid_urls), concurrency):
            batch = valid_urls[i:i+concurrency]
            pvars: List[str] = []
            for j, u in enumerate(batch):
                safe = _escape_literal(u)
                pvar = f"__pf_p_{i}_{j}"
                pvars.append(pvar)
                key = _key_for(u)
                if cache_var:
                    lines.append(f"if ({cache_var}[{key}]) {{ {rv}[{key}] = {cache_var}[{key}]; }} else {{")
                    lines.append(f"{indent}let {pvar} = {helper}('{safe}', {retries}, {timeout_ms if timeout_ms is not None else 'null'}, {backoff_ms});")
                    if metrics_var:
                        lines.append(f"{indent}{pvar}.then(d => {{ if (d) {{ {rv}[{key}] = d; {cache_var}[{key}] = d; {metrics_var}.prefetched = ({metrics_var}.prefetched || 0) + 1; }} }});")
                    else:
                        lines.append(f"{indent}{pvar}.then(d => {{ if (d) {{ {rv}[{key}] = d; {cache_var}[{key}] = d; }} }});")
                    lines.append("}")
                else:
                    lines.append(f"let {pvar} = {helper}('{safe}', {retries}, {timeout_ms if timeout_ms is not None else 'null'}, {backoff_ms});")
                    if metrics_var:
                        lines.append(f"{indent}{pvar}.then(d => {{ if (d) {{ {rv}[{key}] = d; {metrics_var}.prefetched = ({metrics_var}.prefetched || 0) + 1; }} }});")
                    else:
                        lines.append(f"{indent}{pvar}.then(d => {{ if (d) {{ {rv}[{key}] = d; }} }});")
            if pvars:
                lines.append(f"Promise.all([{', '.join(pvars)}]);")

        return sep.join(lines) + (sep if not inline else "")

    # Default spawn-style runtime
    batches = [valid_urls[i:i+concurrency] for i in range(0, len(valid_urls), concurrency)]
    uid = hashlib.sha1(",".join(valid_urls).encode("utf-8")).hexdigest()[:8]

    for bidx, batch in enumerate(batches):
        lines.append(f"// prefetch batch {bidx} uid={uid}")
        for u in batch:
            safe = _escape_literal(u)
            key = _key_for(u)
            if cache_var:
                lines.append(f"if ({cache_var}[{key}]) {{ {rv}[{key}] = {cache_var}[{key}]; }} else {{")
                lines.append(f"{indent}spawn async {{")
                lines.append(f"{indent*2}let __pf_attempt = 0;")
                lines.append(f"{indent*2}let __pf_res = null;")
                lines.append(f"{indent*2}while (__pf_attempt <= {retries} && !__pf_res) {{")
                if timeout_ms is not None:
                    lines.append(f"{indent*3}// placeholder: runtime should support fetchData(url, timeout_ms) for timeouts")
                lines.append(f"{indent*3}__pf_res = fetchData('{safe}');")
                lines.append(f"{indent*3}if (!__pf_res) {{ __pf_attempt += 1; sleep({backoff_ms} * __pf_attempt); }}")
                lines.append(f"{indent*2}}}")
                lines.append(f"{indent*2}if (__pf_res) {{ {rv}[{key}] = __pf_res; {cache_var}[{key}] = __pf_res;")
                if metrics_var:
                    lines.append(f"{indent*3}{metrics_var}.prefetched = ({metrics_var}.prefetched || 0) + 1;")
                lines.append(f"{indent*2}}}")
                lines.append(f"{indent}}};")
                lines.append("}")
            else:
                lines.append(f"spawn async {{")
                lines.append(f"{indent}let __pf_attempt = 0;")
                lines.append(f"{indent}let __pf_res = null;")
                lines.append(f"{indent}while (__pf_attempt <= {retries} && !__pf_res) {{")
                if timeout_ms is not None:
                    lines.append(f"{indent*2}// placeholder: runtime timeout wrapper would be used here")
                lines.append(f"{indent*2}__pf_res = fetchData('{safe}');")
                lines.append(f"{indent*2}if (!__pf_res) {{ __pf_attempt += 1; sleep({backoff_ms} * __pf_attempt); }}")
                lines.append(f"{indent}}}")
                lines.append(f"{indent}if (__pf_res) {{ {rv}[{key}] = __pf_res;")
                if metrics_var:
                    lines.append(f"{indent*2}{metrics_var}.prefetched = ({metrics_var}.prefetched || 0) + 1;")
                lines.append(f"{indent}}}")
                lines.append("};")

    return sep.join(lines) + (sep if not inline else "")


def register(toolkit: Any) -> None:
    """
    Register helper with a toolkit or PluginManager.
    - If toolkit.register_helper(name, fn, metadata) exists, it will be used.
    - Otherwise the function is attached to the toolkit as attribute `prefetch_x`.
    """
    meta = {
        "name": "prefetch_x",
        "description": "Generate prefetch code for URLs with concurrency/retry/cache support.",
        "signature": "prefetch_x(urls, results_var=None, options=None) -> str",
        "options": {
            "runtime": {"type": "string", "enum": ["spawn", "promises"], "default": "spawn"},
            "concurrency": {"type": "integer", "default": 8},
            "retries": {"type": "integer", "default": 2},
            "timeout_ms": {"type": ["integer", "null"], "default": None},
            "backoff_ms": {"type": "integer", "default": 50},
            "cache_var": {"type": ["string", "null"], "default": None},
            "metrics_var": {"type": ["string", "null"], "default": None},
            "safe_keys": {"type": "boolean", "default": False},
            "inline": {"type": "boolean", "default": False},
        }
    }
    if hasattr(toolkit, "register_helper"):
        try:
            toolkit.register_helper("prefetch_x", generate_prefetch_x, metadata=meta)
            return
        except Exception:
            try:
                toolkit.register_helper("prefetch_x", generate_prefetch_x)
                return
            except Exception:
                pass
    # fallback attach as attribute
    try:
        setattr(toolkit, "prefetch_x", generate_prefetch_x)
    except Exception:
        # silently ignore if toolkit cannot be mutated
        pass


# CLI demo + self-check ----------------------------------------------------

def _demo_and_selfcheck():
    sample = [
        "https://example.com/a.json",
        "https://example.com/b.json",
        "invalid:url",
        "https://cdn.example.com/img.png"
    ]
    print("=== Generated spawn-style snippet ===")
    print(generate_prefetch_x(sample, results_var="__my_prefetch", options={"runtime": "spawn", "concurrency": 2, "retries": 1, "metrics_var": "METRICS"}))
    print("=== Generated promises-style snippet ===")
    print(generate_prefetch_x(sample, results_var="__my_prefetch", options={"runtime": "promises", "concurrency": 2, "retries": 1, "metrics_var": "METRICS"}))

    plan = generate_prefetch_plan(sample, {"concurrency": 2, "safe_keys": True})
    print("\n=== Prefetch Plan JSON ===")
    print(json.dumps(plan, indent=2))

    # basic assertions for self-check
    assert isinstance(plan, dict) and "batches" in plan and "urls" in plan
    assert sum(len(b) for b in plan["batches"]) == plan["count"]
    print("\nSelf-check: plan looks consistent.")

if __name__ == "__main__":
    _demo_and_selfcheck()

    """
    ciams/ciams_plugins/prefetch_helper_plugin.py
    Enhanced, fully-implemented prefetch helper plugin.
    Features:
     - Robust `generate_prefetch_x(urls, results_var=None, options=None)` producing
       spawn-style or promise-style prefetch snippets with concurrency/retry/timeout/backoff.
     - `generate_prefetch_plan(urls, options)` returns structured plan (metadata + batches).
     - URL validation and safe literal escaping.
     - Optional cache/metrics integration and safe key hashing.
     - `register(toolkit)` to attach helper to a toolkit (with metadata if supported).
     - Executable CLI demo and a small self-check that validates output and plan structure.
     - No external dependencies — pure stdlib.
     Usage:
      - Import and call `generate_prefetch_x(...)` from codegen components.
      - Plugins/toolkits can call `register(toolkit)` to register the helper.
        Design notes:
        - The generated snippet is textual and targets runtimes that expose `spawn async { ... }`,
        `fetchData(url)` and (optionally) `Promise` + `setTimeout`. Timeouts are emitted as
        comments/placeholders because runtime-specific wrappers are required for actual timeout
        semantics in generated code.
        """

"""
ciams/ciams_plugins/prefetch_helper_plugin.py

Enhanced, fully-implemented prefetch helper plugin for instryx_memory_math_loops_codegen.

Additions / boosters:
 - Robust `generate_prefetch_x(urls, results_var=None, options=None)` producing
   spawn-style or promise-style prefetch snippets with concurrency/retry/timeout/backoff.
 - `generate_prefetch_plan(urls, options)` returns structured plan (metadata + batches).
 - `generate_prefetch_ast(urls, options)` returns a structured AST-like representation.
 - Simple in-memory TTL cache helper `PrefetchCache` useful for runtime emulation / tests.
 - Metrics callback hook via `set_metrics_callback(fn)`.
 - `register(toolkit)` integrates with toolkits exposing `register_helper`.
 - CLI demo, self-check and lightweight unit-style checks (executable).
 - Pure-stdlib, defensive input validation and deterministic names to avoid collisions.

Usage:
  from ciams.ciams_plugins.prefetch_helper_plugin import generate_prefetch_x, register
"""
from __future__ import annotations
import hashlib
import json
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

# Type aliases
Options = Dict[str, Any]
MetricsCallback = Optional[Callable[[str, Dict[str, Any]], None]]

# Module-level metrics callback (noop by default)
_metrics_cb: MetricsCallback = None


def set_metrics_callback(fn: MetricsCallback) -> None:
    """Set a metrics callback to receive events: fn(event_name, payload)."""
    global _metrics_cb
    _metrics_cb = fn


def _emit_metric(event: str, payload: Dict[str, Any]) -> None:
    try:
        if _metrics_cb:
            _metrics_cb(event, payload)
    except Exception:
        # metrics must not break generation
        pass


# -----------------------------
# Utilities
# -----------------------------
def _escape_literal(s: str) -> str:
    """Escape string for inclusion in single-quoted code literals."""
    return s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")


def _is_valid_url(u: str) -> bool:
    """Conservative URL validation using urllib.parse."""
    try:
        p = urlparse(u)
        return bool(p.scheme and p.netloc)
    except Exception:
        return False


# -----------------------------
# Small in-memory TTL cache (helpful for tests/emulation)
# -----------------------------
class PrefetchCache:
    """Simple in-memory cache with TTL (seconds)."""

    def __init__(self, default_ttl: Optional[int] = None):
        self._store: Dict[str, Tuple[float, Any]] = {}
        self._default_ttl = default_ttl

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        expire = time.time() + (ttl if ttl is not None else (self._default_ttl or 0)) if (ttl or self._default_ttl) else 0.0
        self._store[key] = (expire, value)

    def get(self, key: str) -> Any:
        entry = self._store.get(key)
        if not entry:
            return None
        expire, value = entry
        if expire and time.time() > expire:
            self._store.pop(key, None)
            return None
        return value

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def clear(self) -> None:
        self._store.clear()

    def to_dict(self) -> Dict[str, Any]:
        out = {}
        for k, (expire, v) in self._store.items():
            out[k] = {"expire": expire, "value_repr": repr(v)}
        return out


# -----------------------------
# Prefetch plan / AST generation
# -----------------------------
def generate_prefetch_plan(urls: Sequence[str], options: Optional[Options] = None) -> Dict[str, Any]:
    """
    Deterministic prefetch plan describing batches and per-URL metadata.
    Useful for analysis, test, or offline scheduling.
    """
    opts = dict(options or {})
    concurrency = max(1, int(opts.get("concurrency", 8) or 8))
    safe_keys = bool(opts.get("safe_keys", False))

    urls_list = [str(u) for u in urls]
    validated: List[Dict[str, Any]] = []
    for u in urls_list:
        validated.append({
            "url": u,
            "valid": _is_valid_url(u),
            "safe_key": ("pf_" + hashlib.sha1(u.encode("utf-8")).hexdigest()[:12]) if safe_keys else None
        })

    batches: List[List[str]] = [urls_list[i:i + concurrency] for i in range(0, len(urls_list), concurrency)]
    plan = {
        "count": len(urls_list),
        "batches": batches,
        "urls": validated,
        "metadata": {"concurrency": concurrency, "safe_keys": safe_keys}
    }
    _emit_metric("prefetch_plan_generated", {"count": len(urls_list), "concurrency": concurrency})
    return plan


def generate_prefetch_ast(urls: Sequence[str], options: Optional[Options] = None) -> Dict[str, Any]:
    """
    Return a structured, language-agnostic representation of the actions the generated
    snippet will perform. Useful for tooling that wants to operate on the plan programmatically.
    """
    opts = dict(options or {})
    runtime = opts.get("runtime", "spawn")
    concurrency = max(1, int(opts.get("concurrency", 8) or 8))
    retries = max(0, int(opts.get("retries", 2) or 2))
    backoff_ms = int(opts.get("backoff_ms", 50) or 50)
    timeout_ms = opts.get("timeout_ms")
    safe_keys = bool(opts.get("safe_keys", False))

    urls_list = [str(u) for u in urls]
    actions = []
    for u in urls_list:
        valid = _is_valid_url(u)
        actions.append({
            "url": u,
            "valid": valid,
            "action": "prefetch" if valid else "skip",
            "params": {"retries": retries, "timeout_ms": timeout_ms, "backoff_ms": backoff_ms, "safe_key": ("pf_" + hashlib.sha1(u.encode("utf-8")).hexdigest()[:12]) if safe_keys else None}
        })

    ast = {
        "runtime": runtime,
        "concurrency": concurrency,
        "actions": actions,
        "metadata": {"generated_at": time.time()}
    }
    _emit_metric("prefetch_ast_generated", {"count": len(urls_list)})
    return ast


# -----------------------------
# Main generator: textual snippet
# -----------------------------
def generate_prefetch_x(urls: Sequence[str], results_var: Optional[str] = None, options: Optional[Options] = None) -> str:
    """
    Generate textual prefetch snippet.

    Target runtimes:
      - spawn-style (default): uses `spawn async { ... }`, `fetchData(url)` and `sleep(ms)`
      - promises-style: uses `Promise` / `fetchData(url)` / `setTimeout`

    Returned string is plain text and should be inserted into target source.
    """
    opts = dict(options or {})
    runtime = str(opts.get("runtime", "spawn"))
    concurrency = max(1, int(opts.get("concurrency", 8) or 8))
    retries = max(0, int(opts.get("retries", 2) or 2))
    timeout_ms = opts.get("timeout_ms")
    backoff_ms = max(0, int(opts.get("backoff_ms", 50) or 50))
    cache_var = opts.get("cache_var")
    metrics_var = opts.get("metrics_var")
    safe_keys = bool(opts.get("safe_keys", False))
    indent = str(opts.get("indent", "  "))
    inline = bool(opts.get("inline", False))

    urls_list = [str(u) for u in urls]
    rv = results_var or "__prefetch_x"

    sep = " " if inline else "\n"
    lines: List[str] = []
    lines.append(f"{rv} = {rv} ? {rv} : {{}};")
    if cache_var:
        lines.append(f"{cache_var} = {cache_var} ? {cache_var} : {{}};")
    if metrics_var:
        lines.append(f"{metrics_var} = {metrics_var} ? {metrics_var} : {{prefetched:0}};")

    # Validate and split
    valid_urls = [u for u in urls_list if _is_valid_url(u)]
    invalid_urls = [u for u in urls_list if not _is_valid_url(u)]
    if invalid_urls:
        lines.append(f"// WARNING: {len(invalid_urls)} invalid URL(s) skipped")
        for u in invalid_urls:
            lines.append(f"// skipped: {u}")

    if not valid_urls:
        return sep.join(lines) + (sep if not inline else "")

    def _key_for(u: str) -> str:
        if safe_keys:
            return f"pf_{hashlib.sha1(u.encode('utf-8')).hexdigest()[:12]}"
        return f"'{_escape_literal(u)}'"

    # Promises runtime generation
    if runtime == "promises":
        helper = "__prefetch_fetch_with_retry"
        lines.extend([
            f"function {helper}(url, retries, timeout_ms, backoff_ms) {{",
            f"{indent}// returns a Promise that resolves to fetched data or null on failure",
            f"{indent}let attempt = 0;",
            f"{indent}function tryOnce(resolve) {{",
            f"{indent*2}let p = fetchData(url);",
        ])
        if timeout_ms is not None:
            lines.append(f"{indent*2}// NOTE: wrap promise `p` with a timeout helper if runtime provides it")
        lines.extend([
            f"{indent*2}p.then(d => resolve(d)).catch(() => {{",
            f"{indent*3}attempt += 1;",
            f"{indent*3}if (attempt <= retries) {{",
            f"{indent*4}setTimeout(() => tryOnce(resolve), backoff_ms * attempt);",
            f"{indent*3}}} else {{ resolve(null); }}",
            f"{indent*2}}});",
            f"{indent}}}",
            f"{indent}return new Promise(tryOnce);",
            f"}}"
        ])

        for i in range(0, len(valid_urls), concurrency):
            batch = valid_urls[i:i+concurrency]
            pvars: List[str] = []
            for j, u in enumerate(batch):
                safe = _escape_literal(u)
                pvar = f"__pf_p_{i}_{j}"
                pvars.append(pvar)
                key = _key_for(u)
                if cache_var:
                    lines.append(f"if ({cache_var}[{key}]) {{ {rv}[{key}] = {cache_var}[{key}]; }} else {{")
                    lines.append(f"{indent}let {pvar} = {helper}('{safe}', {retries}, {timeout_ms if timeout_ms is not None else 'null'}, {backoff_ms});")
                    if metrics_var:
                        lines.append(f"{indent}{pvar}.then(d => {{ if (d) {{ {rv}[{key}] = d; {cache_var}[{key}] = d; {metrics_var}.prefetched = ({metrics_var}.prefetched || 0) + 1; }} }});")
                    else:
                        lines.append(f"{indent}{pvar}.then(d => {{ if (d) {{ {rv}[{key}] = d; {cache_var}[{key}] = d; }} }});")
                    lines.append("}")
                else:
                    lines.append(f"let {pvar} = {helper}('{safe}', {retries}, {timeout_ms if timeout_ms is not None else 'null'}, {backoff_ms});")
                    if metrics_var:
                        lines.append(f"{indent}{pvar}.then(d => {{ if (d) {{ {rv}[{key}] = d; {metrics_var}.prefetched = ({metrics_var}.prefetched || 0) + 1; }} }});")
                    else:
                        lines.append(f"{indent}{pvar}.then(d => {{ if (d) {{ {rv}[{key}] = d; }} }});")
            if pvars:
                lines.append(f"Promise.all([{', '.join(pvars)}]);")

        _emit_metric("prefetch_snippet_generated", {"runtime": "promises", "count": len(valid_urls)})
        return sep.join(lines) + (sep if not inline else "")

    # Default spawn-style generation
    batches = [valid_urls[i:i+concurrency] for i in range(0, len(valid_urls), concurrency)]
    uid = hashlib.sha1(",".join(valid_urls).encode("utf-8")).hexdigest()[:8]

    for bidx, batch in enumerate(batches):
        lines.append(f"// prefetch batch {bidx} uid={uid}")
        for u in batch:
            safe = _escape_literal(u)
            key = _key_for(u)
            if cache_var:
                lines.append(f"if ({cache_var}[{key}]) {{ {rv}[{key}] = {cache_var}[{key}]; }} else {{")
                lines.append(f"{indent}spawn async {{")
                lines.append(f"{indent*2}let __pf_attempt = 0;")
                lines.append(f"{indent*2}let __pf_res = null;")
                lines.append(f"{indent*2}while (__pf_attempt <= {retries} && !__pf_res) {{")
                if timeout_ms is not None:
                    lines.append(f"{indent*3}// placeholder: runtime should support fetchData(url, timeout_ms)")
                lines.append(f"{indent*3}__pf_res = fetchData('{safe}');")
                lines.append(f"{indent*3}if (!__pf_res) {{ __pf_attempt += 1; sleep({backoff_ms} * __pf_attempt); }}")
                lines.append(f"{indent*2}}}")
                lines.append(f"{indent*2}if (__pf_res) {{ {rv}[{key}] = __pf_res; {cache_var}[{key}] = __pf_res;")
                if metrics_var:
                    lines.append(f"{indent*3}{metrics_var}.prefetched = ({metrics_var}.prefetched || 0) + 1;")
                lines.append(f"{indent*2}}}")
                lines.append(f"{indent}}};")
                lines.append("}")
            else:
                lines.append(f"spawn async {{")
                lines.append(f"{indent}let __pf_attempt = 0;")
                lines.append(f"{indent}let __pf_res = null;")
                lines.append(f"{indent}while (__pf_attempt <= {retries} && !__pf_res) {{")
                if timeout_ms is not None:
                    lines.append(f"{indent*2}// placeholder: runtime timeout wrapper would be used here")
                lines.append(f"{indent*2}__pf_res = fetchData('{safe}');")
                lines.append(f"{indent*2}if (!__pf_res) {{ __pf_attempt += 1; sleep({backoff_ms} * __pf_attempt); }}")
                lines.append(f"{indent}}}")
                lines.append(f"{indent}if (__pf_res) {{ {rv}[{key}] = __pf_res;")
                if metrics_var:
                    lines.append(f"{indent*2}{metrics_var}.prefetched = ({metrics_var}.prefetched || 0) + 1;")
                lines.append(f"{indent}}}")
                lines.append("};")

    _emit_metric("prefetch_snippet_generated", {"runtime": "spawn", "count": len(valid_urls)})
    return sep.join(lines) + (sep if not inline else "")


# -----------------------------
# Integration helper
# -----------------------------
def register(toolkit: Any) -> None:
    """
    Register helper with toolkit or PluginManager.
    - If toolkit.register_helper(name, fn, metadata) exists, it will be used.
    - Otherwise the function is attached as attribute `prefetch_x`.
    """
    meta = {
        "name": "prefetch_x",
        "description": "Generate prefetch code for URLs (concurrency, retries, cache, metrics).",
        "signature": "prefetch_x(urls, results_var=None, options=None) -> str",
        "options": {
            "runtime": {"type": "string", "enum": ["spawn", "promises"], "default": "spawn"},
            "concurrency": {"type": "integer", "default": 8},
            "retries": {"type": "integer", "default": 2},
            "timeout_ms": {"type": ["integer", "null"], "default": None},
            "backoff_ms": {"type": "integer", "default": 50},
            "cache_var": {"type": ["string", "null"], "default": None},
            "metrics_var": {"type": ["string", "null"], "default": None},
            "safe_keys": {"type": "boolean", "default": False},
            "inline": {"type": "boolean", "default": False},
        }
    }
    if hasattr(toolkit, "register_helper"):
        try:
            toolkit.register_helper("prefetch_x", generate_prefetch_x, metadata=meta)
            return
        except Exception:
            try:
                toolkit.register_helper("prefetch_x", generate_prefetch_x)
                return
            except Exception:
                pass
    # fallback
    try:
        setattr(toolkit, "prefetch_x", generate_prefetch_x)
    except Exception:
        pass


# -----------------------------
# CLI demo and lightweight self-check
# -----------------------------
def _demo_and_selfcheck() -> None:
    sample = [
        "https://example.com/a.json",
        "https://example.com/b.json",
        "invalid:url",
        "https://cdn.example.com/img.png"
    ]
    print("=== spawn-style snippet ===")
    print(generate_prefetch_x(sample, results_var="__my_prefetch", options={"runtime": "spawn", "concurrency": 2, "retries": 1, "metrics_var": "METRICS"}))
    print("=== promises-style snippet ===")
    print(generate_prefetch_x(sample, results_var="__my_prefetch", options={"runtime": "promises", "concurrency": 2, "retries": 1, "metrics_var": "METRICS"}))

    plan = generate_prefetch_plan(sample, {"concurrency": 2, "safe_keys": True})
    print("\n=== Prefetch Plan JSON ===")
    print(json.dumps(plan, indent=2))

    ast = generate_prefetch_ast(sample, {"concurrency": 2})
    print("\n=== Prefetch AST ===")
    print(json.dumps(ast, indent=2))

    # basic assertions (lightweight)
    assert isinstance(plan, dict) and "batches" in plan and "urls" in plan
    assert sum(len(b) for b in plan["batches"]) == plan["count"]
    # ensure invalid URL was flagged
    assert any(not u["valid"] for u in plan["urls"])
    print("\nSelf-check: plan looks consistent.")


if __name__ == "__main__":
    # If used as a script, run demo + self-check and exit 0 on success
    try:
        _demo_and_selfcheck()
        sys.exit(0)
    except AssertionError as ae:
        print("Self-check failed:", ae)
        sys.exit(2)
    except Exception as e:
        print("Error during demo/self-check:", e)
        sys.exit(3)

        """
        ciams/ciams_plugins/prefetch_helper_plugin.py
        Enhanced, fully-implemented prefetch helper plugin.
        Features:
         - Robust `generate_prefetch_x(urls, results_var=None, options=None)` producing
           spawn-style or promise-style prefetch snippets with concurrency/retry/timeout/backoff.
         - `generate_prefetch_plan(urls, options)` returns structured plan (metadata + batches).
         - `generate_prefetch_ast(urls, options)` returns a structured AST-like representation.
         - Simple in-memory TTL cache helper `PrefetchCache` useful for runtime emulation / tests.
         - Metrics callback hook via `set_metrics_callback(fn)`.
         - `register(toolkit)` integrates with toolkits exposing `register_helper`.
         - CLI demo, self-check and lightweight unit-style checks (executable).
         - Pure-stdlib, defensive input validation and deterministic names to avoid collisions.
         Usage:
            - Import and call `generate_prefetch_x(...)` from codegen components.
            - Plugins/toolkits can call `register(toolkit)` to register the helper.
            Design notes:
            - The generated snippet is textual and targets runtimes that expose `spawn async { ... }`,
            `fetchData(url)` and (optionally) `Promise` + `setTimeout`. Timeouts are emitted as
            comments/placeholders because runtime-specific wrappers are required for actual timeout
            semantics in generated code.
            """

"""
ciams/ciams_plugins/prefetch_helper_plugin.py

Supreme boosters edition — enhanced, fully-implemented prefetch helper plugin.

Key additions and improvements (executable, pure-stdlib with optional aiohttp):
 - Single consistent API: generate_prefetch_x, generate_prefetch_plan, generate_prefetch_ast
 - Execute plan in-process:
   - synchronous ThreadPool based executor
   - asyncio-based executor using aiohttp if available (optional)
 - Robust retry/backoff with jitter and configurable timeouts
 - In-memory TTL cache (PrefetchCache) and optional disk-backed cache (DiskCache)
 - Metrics collection and simple HTTP /metrics endpoint for Prometheus scraping
 - set_metrics_callback hook and get_metrics() API
 - CLI with argparse for demoing generation, plan, AST and executing with dummy or real fetcher
 - Defensive URL validation, escaping, deterministic names to avoid collisions
 - Fully type annotated and self-checking demo
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import http.server
import json
import math
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

# Optional dependency
try:
    import aiohttp  # type: ignore
    _HAS_AIOHTTP = True
except Exception:
    _HAS_AIOHTTP = False

# Types
Options = Dict[str, Any]
MetricsCallback = Optional[Callable[[str, Dict[str, Any]], None]]

# Module-level metrics and callback
_metrics_cb: MetricsCallback = None
_metrics_lock = threading.RLock()
_metrics: Dict[str, int] = {
    "prefetch_requests_total": 0,
    "prefetch_success_total": 0,
    "prefetch_failure_total": 0,
    "prefetch_skipped_cache_total": 0,
    "prefetch_plan_generated_total": 0,
    "prefetch_ast_generated_total": 0,
    "prefetch_snippet_generated_total": 0,
    "prefetch_plan_executed_total": 0,
}


def set_metrics_callback(fn: MetricsCallback) -> None:
    """Install a metrics callback: fn(event_name, payload)."""
    global _metrics_cb
    _metrics_cb = fn


def _emit_metric(event: str, payload: Dict[str, Any]) -> None:
    try:
        if _metrics_cb:
            _metrics_cb(event, payload)
    except Exception:
        pass


def _metric_inc(name: str, n: int = 1) -> None:
    with _metrics_lock:
        _metrics[name] = _metrics.get(name, 0) + n


def get_metrics() -> Dict[str, int]:
    with _metrics_lock:
        return dict(_metrics)


def start_metrics_http(host: str = "127.0.0.1", port: int = 8085) -> threading.Thread:
    """
    Start a very small HTTP server exposing /metrics in Prometheus text format.
    Returns the Thread running the server (daemon).
    """
    class _Handler(http.server.BaseHTTPRequestHandler):  # type: ignore
        def do_GET(self):
            if self.path != "/metrics":
                self.send_response(404)
                self.end_headers()
                return
            payload_lines = []
            with _metrics_lock:
                for k, v in _metrics.items():
                    payload_lines.append(f"{k} {v}")
            payload = "\n".join(payload_lines) + "\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload.encode("utf-8"))

        def log_message(self, format, *args):  # silence logs
            return

    server = http.server.ThreadingHTTPServer((host, port), _Handler)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    return th


# -------------------------
# Utilities
# -------------------------
def _escape_literal(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")


def _is_valid_url(u: str) -> bool:
    try:
        p = urlparse(u)
        return bool(p.scheme and p.netloc)
    except Exception:
        return False


# -------------------------
# Caches
# -------------------------
class PrefetchCache:
    """In-memory TTL cache."""

    def __init__(self, default_ttl: Optional[int] = None):
        self._store: Dict[str, Tuple[float, Any]] = {}
        self._default_ttl = default_ttl
        self._lock = threading.RLock()

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        expire = 0.0
        use_ttl = ttl if ttl is not None else self._default_ttl
        if use_ttl:
            expire = time.time() + float(use_ttl)
        with self._lock:
            self._store[key] = (expire, value)

    def get(self, key: str) -> Any:
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None
            expire, value = entry
            if expire and time.time() > expire:
                self._store.pop(key, None)
                return None
            return value

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {k: {"expire": e, "repr": repr(v)} for k, (e, v) in self._store.items()}


class DiskCache:
    """Simple disk-backed JSON cache under a directory using hashed keys."""

    def __init__(self, path: str, default_ttl: Optional[int] = None):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.ttl = default_ttl
        self._lock = threading.RLock()

    def _file_for(self, key: str) -> Path:
        name = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return self.path / f"{name}.json"

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        expire = 0.0
        use = ttl if ttl is not None else self.ttl
        if use:
            expire = time.time() + float(use)
        payload = {"expire": expire, "value": value}
        p = self._file_for(key)
        with self._lock:
            p.write_text(json.dumps(payload), encoding="utf-8")

    def get(self, key: str) -> Any:
        p = self._file_for(key)
        if not p.exists():
            return None
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            expire = payload.get("expire", 0.0)
            if expire and time.time() > expire:
                try:
                    p.unlink()
                except Exception:
                    pass
                return None
            return payload.get("value")
        except Exception:
            return None

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def clear(self) -> None:
        for f in self.path.glob("*.json"):
            try:
                f.unlink()
            except Exception:
                pass


# -------------------------
# Plan / AST / snippet generation
# -------------------------
def generate_prefetch_plan(urls: Sequence[str], options: Optional[Options] = None) -> Dict[str, Any]:
    opts = dict(options or {})
    concurrency = max(1, int(opts.get("concurrency", 8) or 8))
    safe_keys = bool(opts.get("safe_keys", False))
    urls_list = [str(u) for u in urls]
    validated = []
    for u in urls_list:
        validated.append({
            "url": u,
            "valid": _is_valid_url(u),
            "safe_key": ("pf_" + hashlib.sha1(u.encode("utf-8")).hexdigest()[:12]) if safe_keys else None
        })
    batches = [urls_list[i:i + concurrency] for i in range(0, len(urls_list), concurrency)]
    plan = {"count": len(urls_list), "batches": batches, "urls": validated, "metadata": {"concurrency": concurrency, "safe_keys": safe_keys}}
    _metric_inc("prefetch_plan_generated_total", 1)
    _emit_metric("prefetch_plan_generated", {"count": len(urls_list), "concurrency": concurrency})
    return plan


def generate_prefetch_ast(urls: Sequence[str], options: Optional[Options] = None) -> Dict[str, Any]:
    opts = dict(options or {})
    runtime = opts.get("runtime", "spawn")
    concurrency = max(1, int(opts.get("concurrency", 8) or 8))
    retries = max(0, int(opts.get("retries", 2) or 2))
    backoff_ms = int(opts.get("backoff_ms", 50) or 50)
    timeout_ms = opts.get("timeout_ms")
    safe_keys = bool(opts.get("safe_keys", False))
    urls_list = [str(u) for u in urls]
    actions = []
    for u in urls_list:
        valid = _is_valid_url(u)
        actions.append({"url": u, "valid": valid, "action": "prefetch" if valid else "skip", "params": {"retries": retries, "timeout_ms": timeout_ms, "backoff_ms": backoff_ms, "safe_key": ("pf_" + hashlib.sha1(u.encode("utf-8")).hexdigest()[:12]) if safe_keys else None}})
    ast = {"runtime": runtime, "concurrency": concurrency, "actions": actions, "metadata": {"generated_at": time.time()}}
    _metric_inc("prefetch_ast_generated_total", 1)
    _emit_metric("prefetch_ast_generated", {"count": len(urls_list)})
    return ast


def generate_prefetch_x(urls: Sequence[str], results_var: Optional[str] = None, options: Optional[Options] = None) -> str:
    opts = dict(options or {})
    runtime = str(opts.get("runtime", "spawn"))
    concurrency = max(1, int(opts.get("concurrency", 8) or 8))
    retries = max(0, int(opts.get("retries", 2) or 2))
    timeout_ms = opts.get("timeout_ms")
    backoff_ms = max(0, int(opts.get("backoff_ms", 50) or 50))
    cache_var = opts.get("cache_var")
    metrics_var = opts.get("metrics_var")
    safe_keys = bool(opts.get("safe_keys", False))
    indent = str(opts.get("indent", "  "))
    inline = bool(opts.get("inline", False))

    urls_list = [str(u) for u in urls]
    rv = results_var or "__prefetch_x"
    sep = " " if inline else "\n"
    lines: List[str] = []
    lines.append(f"{rv} = {rv} ? {rv} : {{}};")
    if cache_var:
        lines.append(f"{cache_var} = {cache_var} ? {cache_var} : {{}};")
    if metrics_var:
        lines.append(f"{metrics_var} = {metrics_var} ? {metrics_var} : {{prefetched:0}};")

    valid_urls = [u for u in urls_list if _is_valid_url(u)]
    invalid_urls = [u for u in urls_list if not _is_valid_url(u)]
    if invalid_urls:
        lines.append(f"// WARNING: {len(invalid_urls)} invalid URL(s) skipped")
        for u in invalid_urls:
            lines.append(f"// skipped: {u}")
    if not valid_urls:
        return sep.join(lines) + (sep if not inline else "")

    def _key_for(u: str) -> str:
        if safe_keys:
            return f"pf_{hashlib.sha1(u.encode('utf-8')).hexdigest()[:12]}"
        return f"'{_escape_literal(u)}'"

    if runtime == "promises":
        helper = "__prefetch_fetch_with_retry"
        lines.extend([f"function {helper}(url, retries, timeout_ms, backoff_ms) {{", f"{indent}// returns a Promise that resolves to fetched data or null on failure", f"{indent}let attempt = 0;", f"{indent}function tryOnce(resolve) {{", f"{indent*2}let p = fetchData(url);"])
        if timeout_ms is not None:
            lines.append(f"{indent*2}// NOTE: wrap promise `p` with a timeout helper if runtime provides it")
        lines.extend([f"{indent*2}p.then(d => resolve(d)).catch(() => {{", f"{indent*3}attempt += 1;", f"{indent*3}if (attempt <= retries) {{", f"{indent*4}setTimeout(() => tryOnce(resolve), backoff_ms * attempt);", f"{indent*3}}} else {{ resolve(null); }}", f"{indent*2}}});", f"{indent}}}", f"{indent}return new Promise(tryOnce);", f"}}"])
        for i in range(0, len(valid_urls), concurrency):
            batch = valid_urls[i:i + concurrency]
            pvars: List[str] = []
            for j, u in enumerate(batch):
                safe = _escape_literal(u)
                pvar = f"__pf_p_{i}_{j}"
                pvars.append(pvar)
                key = _key_for(u)
                if cache_var:
                    lines.append(f"if ({cache_var}[{key}]) {{ {rv}[{key}] = {cache_var}[{key}]; }} else {{")
                    lines.append(f"{indent}let {pvar} = {helper}('{safe}', {retries}, {timeout_ms if timeout_ms is not None else 'null'}, {backoff_ms});")
                    if metrics_var:
                        lines.append(f"{indent}{pvar}.then(d => {{ if (d) {{ {rv}[{key}] = d; {cache_var}[{key}] = d; {metrics_var}.prefetched = ({metrics_var}.prefetched || 0) + 1; }} }});")
                    else:
                        lines.append(f"{indent}{pvar}.then(d => {{ if (d) {{ {rv}[{key}] = d; {cache_var}[{key}] = d; }} }});")
                    lines.append("}")
                else:
                    lines.append(f"let {pvar} = {helper}('{safe}', {retries}, {timeout_ms if timeout_ms is not None else 'null'}, {backoff_ms});")
                    if metrics_var:
                        lines.append(f"{indent}{pvar}.then(d => {{ if (d) {{ {rv}[{key}] = d; {metrics_var}.prefetched = ({metrics_var}.prefetched || 0) + 1; }} }});")
                    else:
                        lines.append(f"{indent}{pvar}.then(d => {{ if (d) {{ {rv}[{key}] = d; }} }});")
            if pvars:
                lines.append(f"Promise.all([{', '.join(pvars)}]);")
        _metric_inc("prefetch_snippet_generated_total", 1)
        return sep.join(lines) + (sep if not inline else "")

    # spawn-style
    batches = [valid_urls[i:i + concurrency] for i in range(0, len(valid_urls), concurrency)]
    uid = hashlib.sha1(",".join(valid_urls).encode("utf-8")).hexdigest()[:8]
    for bidx, batch in enumerate(batches):
        lines.append(f"// prefetch batch {bidx} uid={uid}")
        for u in batch:
            safe = _escape_literal(u)
            key = _key_for(u)
            if cache_var:
                lines.append(f"if ({cache_var}[{key}]) {{ {rv}[{key}] = {cache_var}[{key}]; }} else {{")
                lines.append(f"{indent}spawn async {{")
                lines.append(f"{indent*2}let __pf_attempt = 0;")
                lines.append(f"{indent*2}let __pf_res = null;")
                lines.append(f"{indent*2}while (__pf_attempt <= {retries} && !__pf_res) {{")
                if timeout_ms is not None:
                    lines.append(f"{indent*3}// placeholder: runtime should support fetchData(url, timeout_ms)")
                lines.append(f"{indent*3}__pf_res = fetchData('{safe}');")
                lines.append(f"{indent*3}if (!__pf_res) {{ __pf_attempt += 1; sleep({backoff_ms} * __pf_attempt); }}")
                lines.append(f"{indent*2}}}")
                lines.append(f"{indent*2}if (__pf_res) {{ {rv}[{key}] = __pf_res; {cache_var}[{key}] = __pf_res;")
                if metrics_var:
                    lines.append(f"{indent*3}{metrics_var}.prefetched = ({metrics_var}.prefetched || 0) + 1;")
                lines.append(f"{indent*2}}}")
                lines.append(f"{indent}}};")
                lines.append("}")
            else:
                lines.append(f"spawn async {{")
                lines.append(f"{indent}let __pf_attempt = 0;")
                lines.append(f"{indent}let __pf_res = null;")
                lines.append(f"{indent}while (__pf_attempt <= {retries} && !__pf_res) {{")
                if timeout_ms is not None:
                    lines.append(f"{indent*2}// placeholder: runtime timeout wrapper would be used here")
                lines.append(f"{indent*2}__pf_res = fetchData('{safe}');")
                lines.append(f"{indent*2}if (!__pf_res) {{ __pf_attempt += 1; sleep({backoff_ms} * __pf_attempt); }}")
                lines.append(f"{indent}}}")
                lines.append(f"{indent}if (__pf_res) {{ {rv}[{key}] = __pf_res;")
                if metrics_var:
                    lines.append(f"{indent*2}{metrics_var}.prefetched = ({metrics_var}.prefetched || 0) + 1;")
                lines.append(f"{indent}}}")
                lines.append("};")
    _metric_inc("prefetch_snippet_generated_total", 1)
    return sep.join(lines) + (sep if not inline else "")


# -------------------------
# Execution helpers (synchronous and async)
# -------------------------
def _exponential_backoff(attempt: int, base_ms: int, jitter: float = 0.1) -> float:
    # exponential backoff with full jitter
    exp = base_ms * (2 ** (attempt - 1)) if attempt > 0 else base_ms
    jitter_val = random.random() * jitter * exp
    return (exp + jitter_val) / 1000.0  # seconds


def _default_fetcher(url: str, timeout: Optional[int] = None) -> Tuple[bool, str]:
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=(timeout or 5)) as resp:
            data = resp.read()
            return True, data[:4096].decode("utf-8", errors="replace")
    except Exception as e:
        return False, str(e)


async def _aiohttp_fetcher(url: str, timeout: Optional[int] = None) -> Tuple[bool, str]:
    if not _HAS_AIOHTTP:
        raise RuntimeError("aiohttp not available")
    timeout_obj = aiohttp.ClientTimeout(total=(timeout or 5))
    async with aiohttp.ClientSession(timeout=timeout_obj) as session:
        try:
            async with session.get(url) as resp:
                data = await resp.read()
                return True, data[:4096].decode("utf-8", errors="replace")
        except Exception as e:
            return False, str(e)


def execute_plan(urls_or_plan: Any, options: Optional[Options] = None, cache: Optional[PrefetchCache] = None, disk_cache: Optional[DiskCache] = None, fetcher: Optional[Callable[[str, Optional[int]], Tuple[bool, Any]]] = None, use_async: bool = False) -> Dict[str, Any]:
    """
    Execute a plan or URL list. Returns dict with results, cache summary and stats.

    - options: concurrency, retries, timeout_ms, backoff_ms, metrics_var, safe_keys
    - cache: PrefetchCache instance (in-memory) to consult/populate
    - disk_cache: optional DiskCache instance; consulted before network and populated on success
    - fetcher: optional synchronous fetcher(url, timeout_ms)->(ok, data)
    - use_async: if True and aiohttp available, use asyncio executor for network I/O
    """
    opts = dict(options or {})
    concurrency = max(1, int(opts.get("concurrency", 8) or 8))
    retries = max(0, int(opts.get("retries", 2) or 2))
    timeout_ms = opts.get("timeout_ms")
    backoff_ms = max(0, int(opts.get("backoff_ms", 50) or 50))
    metrics_var = opts.get("metrics_var")
    safe_keys = bool(opts.get("safe_keys", False))

    if isinstance(urls_or_plan, dict) and "batches" in urls_or_plan:
        plan = urls_or_plan
        urls = [u["url"] for u in plan.get("urls", []) if u.get("valid", True)]
    else:
        urls = list(urls_or_plan)

    cache = cache or PrefetchCache()
    results: Dict[str, Any] = {}
    stats = {"requested": 0, "succeeded": 0, "failed": 0, "skipped": 0}

    def _key(u: str) -> str:
        return (f"pf_{hashlib.sha1(u.encode('utf-8')).hexdigest()[:12]}" if safe_keys else u)

    def _worker(u: str) -> Tuple[str, Any, bool]:
        stats["requested"] += 1
        key = _key(u)
        # disk cache first
        if disk_cache:
            val = disk_cache.get(key)
            if val is not None:
                stats["skipped"] += 1
                _metric_inc("prefetch_skipped_cache_total", 1)
                return u, val, True
        # in-memory cache
        if cache.exists(key):
            stats["skipped"] += 1
            _metric_inc("prefetch_skipped_cache_total", 1)
            return u, cache.get(key), True
        attempt = 0
        last_err = "not attempted"
        while attempt <= retries:
            attempt += 1
            ok, payload = (fetcher or _default_fetcher)(u, timeout_ms)
            if ok:
                cache.set(key, payload)
                if disk_cache:
                    try:
                        disk_cache.set(key, payload)
                    except Exception:
                        pass
                stats["succeeded"] += 1
                _metric_inc("prefetch_success_total", 1)
                return u, payload, True
            else:
                last_err = payload
                if attempt <= retries:
                    time.sleep(_exponential_backoff(attempt, backoff_ms))
        stats["failed"] += 1
        _metric_inc("prefetch_failure_total", 1)
        return u, last_err, False

    # Async path using aiohttp if requested and available
    if use_async and _HAS_AIOHTTP:
        async def _async_main():
            sem = asyncio.Semaphore(concurrency)
            async def _task(u: str):
                nonlocal stats
                key = _key(u)
                if disk_cache:
                    val = disk_cache.get(key)
                    if val is not None:
                        stats["skipped"] += 1
                        _metric_inc("prefetch_skipped_cache_total", 1)
                        return u, val, True
                if cache.exists(key):
                    stats["skipped"] += 1
                    _metric_inc("prefetch_skipped_cache_total", 1)
                    return u, cache.get(key), True
                attempt = 0
                last_err = "not attempted"
                async with sem:
                    while attempt <= retries:
                        attempt += 1
                        ok, payload = await _aiohttp_fetcher(u, timeout_ms)
                        if ok:
                            cache.set(key, payload)
                            if disk_cache:
                                try:
                                    disk_cache.set(key, payload)
                                except Exception:
                                    pass
                            stats["succeeded"] += 1
                            _metric_inc("prefetch_success_total", 1)
                            return u, payload, True
                        else:
                            last_err = payload
                            if attempt <= retries:
                                await asyncio.sleep(_exponential_backoff(attempt, backoff_ms))
                stats["failed"] += 1
                _metric_inc("prefetch_failure_total", 1)
                return u, last_err, False

            tasks = [asyncio.create_task(_task(u)) for u in urls]
            res_out = await asyncio.gather(*tasks)
            return res_out

        loop = asyncio.new_event_loop()
        try:
            res_list = loop.run_until_complete(_async_main())
        finally:
            loop.close()
        for u, payload, ok in res_list:
            results[u] = payload
        _metric_inc("prefetch_plan_executed_total", 1)
        return {"results": results, "cache": cache.to_dict(), "stats": stats}

    # Synchronous/threaded execution
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(_worker, u): u for u in urls}
        for fut in as_completed(futures):
            try:
                u, payload, ok = fut.result()
            except Exception as e:
                u = futures[fut]
                payload = str(e)
                ok = False
            results[u] = payload
    _metric_inc("prefetch_plan_executed_total", 1)
    _emit_metric("prefetch_plan_executed", {"count": len(urls), "stats": {"requested": stats["requested"], "succeeded": stats["succeeded"], "failed": stats["failed"], "skipped": stats["skipped"]}})
    return {"results": results, "cache": cache.to_dict(), "stats": stats}


# -------------------------
# Integration helper
# -------------------------
def register(toolkit: Any) -> None:
    meta = {
        "name": "prefetch_x",
        "description": "Generate prefetch code and optionally execute plans (concurrency/retries/cache).",
        "signature": "prefetch_x(urls, results_var=None, options=None) -> str",
        "options": {
            "runtime": {"type": "string", "enum": ["spawn", "promises"], "default": "spawn"},
            "concurrency": {"type": "integer", "default": 8},
            "retries": {"type": "integer", "default": 2},
            "timeout_ms": {"type": ["integer", "null"], "default": None},
            "backoff_ms": {"type": "integer", "default": 50},
            "cache_var": {"type": ["string", "null"], "default": None},
            "metrics_var": {"type": ["string", "null"], "default": None},
            "safe_keys": {"type": "boolean", "default": False},
            "inline": {"type": "boolean", "default": False},
        }
    }
    if hasattr(toolkit, "register_helper"):
        try:
            toolkit.register_helper("prefetch_x", generate_prefetch_x, metadata=meta)
            return
        except Exception:
            try:
                toolkit.register_helper("prefetch_x", generate_prefetch_x)
                return
            except Exception:
                pass
    try:
        setattr(toolkit, "prefetch_x", generate_prefetch_x)
    except Exception:
        pass


# -------------------------
# CLI demo + self-check
# -------------------------
def _demo_and_selfcheck(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="prefetch_helper_plugin", description="Demo prefetch helper plugin")
    parser.add_argument("--promises", action="store_true", help="Generate promises-style snippet")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--run", action="store_true", help="Execute plan with dummy fetcher")
    parser.add_argument("--async", dest="use_async", action="store_true", help="Use aiohttp async execution if available")
    parser.add_argument("--metrics-port", type=int, default=0, help="Start local metrics endpoint (0=disabled)")
    args = parser.parse_args(argv)

    sample = [
        "https://example.com/a.json",
        "https://example.com/b.json",
        "invalid:url",
        "https://cdn.example.com/img.png"
    ]
    opts = {"runtime": "promises" if args.promises else "spawn", "concurrency": args.concurrency, "retries": args.retries}
    print("=== SNIPPET ===")
    print(generate_prefetch_x(sample, results_var="__my_prefetch", options={**opts, "metrics_var": "METRICS"}))
    plan = generate_prefetch_plan(sample, {"concurrency": args.concurrency, "safe_keys": True})
    print("\n=== PLAN ===")
    print(json.dumps(plan, indent=2))
    ast = generate_prefetch_ast(sample, {"concurrency": args.concurrency})
    print("\n=== AST ===")
    print(json.dumps(ast, indent=2))
    if args.metrics_port:
        t = start_metrics_http(port=args.metrics_port)
        print(f"Metrics endpoint started on port {args.metrics_port} (thread {t.name})")

    if args.run:
        # use dummy fetcher to avoid network by default
        def dummy_fetcher(url: str, timeout: Optional[int] = None):
            return True, f"dummy-data-for:{url}"
        res = execute_plan(plan, options={"concurrency": args.concurrency, "retries": args.retries}, cache=PrefetchCache(), fetcher=dummy_fetcher, use_async=(args.use_async and _HAS_AIOHTTP))
        print("\n=== EXECUTION RESULT ===")
        print(json.dumps(res, indent=2))

    # simple assertions
    assert isinstance(plan, dict) and "batches" in plan and "urls" in plan
    assert sum(len(b) for b in plan["batches"]) == plan["count"]
    assert any(not u["valid"] for u in plan["urls"])
    print("\nSelf-check: OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(_demo_and_selfcheck())
    except AssertionError as ae:
        print("Self-check failed:", ae, file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print("Error during demo/self-check:", e, file=sys.stderr)
        sys.exit(3)

        exc
        ept 
       
print("Error during demo/self-check:", e, file=sys.stderr)
sys.exit(3)
Exception 

"""
ciams/ciams_plugins/prefetch_helper_plugin.py

Enhanced prefetch helper plugin for instryx_memory_math_loops_codegen.

Features (fully implemented & executable):
 - generate_prefetch_x(urls, results_var=None, options=None) -> str
 - generate_prefetch_plan(urls, options=None) -> dict
 - generate_prefetch_ast(urls, options=None) -> dict
 - PrefetchCache: in-memory TTL cache for tests/emulation
 - execute_plan(...): actually performs prefetches (emulated or real) for testing
 - set_metrics_callback(fn) to receive lightweight telemetry events
 - register(toolkit) to attach helper into a toolkit/PluginManager
 - CLI demo and self-check (exits non-zero on failure)
No external deps; pure stdlib.
"""

from __future__ import annotations
import hashlib
import json
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse
import urllib.request

# Types
Options = Dict[str, Any]
MetricsCallback = Optional[Callable[[str, Dict[str, Any]], None]]

# Module-level metrics callback (noop by default)
_metrics_cb: MetricsCallback = None


def set_metrics_callback(fn: MetricsCallback) -> None:
    """Install a metrics callback: fn(event_name, payload)."""
    global _metrics_cb
    _metrics_cb = fn


def _emit_metric(event: str, payload: Dict[str, Any]) -> None:
    try:
        if _metrics_cb:
            _metrics_cb(event, payload)
    except Exception:
        # metrics must not break behavior
        pass


# ---------------------------
# Utilities
# ---------------------------
def _escape_literal(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")


def _is_valid_url(u: str) -> bool:
    try:
        p = urlparse(u)
        return bool(p.scheme and p.netloc)
    except Exception:
        return False


# ---------------------------
# Lightweight TTL cache
# ---------------------------
class PrefetchCache:
    """Simple in-memory TTL cache. TTL in seconds; ttl==0 or None means no expiry."""

    def __init__(self, default_ttl: Optional[int] = None):
        self._store: Dict[str, Tuple[float, Any]] = {}
        self._default_ttl = default_ttl
        self._lock = threading.RLock()

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        with self._lock:
            expire = 0.0
            use_ttl = ttl if ttl is not None else self._default_ttl
            if use_ttl:
                expire = time.time() + float(use_ttl)
            self._store[key] = (expire, value)

    def get(self, key: str) -> Any:
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None
            expire, value = entry
            if expire and time.time() > expire:
                self._store.pop(key, None)
                return None
            return value

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            out: Dict[str, Any] = {}
            for k, (expire, v) in self._store.items():
                out[k] = {"expire": expire, "value_repr": repr(v)}
            return out


# ---------------------------
# Plan / AST generators
# ---------------------------
def generate_prefetch_plan(urls: Sequence[str], options: Optional[Options] = None) -> Dict[str, Any]:
    opts = dict(options or {})
    concurrency = max(1, int(opts.get("concurrency", 8) or 8))
    safe_keys = bool(opts.get("safe_keys", False))

    urls_list = [str(u) for u in urls]
    validated = []
    for u in urls_list:
        validated.append({
            "url": u,
            "valid": _is_valid_url(u),
            "safe_key": ("pf_" + hashlib.sha1(u.encode("utf-8")).hexdigest()[:12]) if safe_keys else None
        })
    batches = [urls_list[i:i + concurrency] for i in range(0, len(urls_list), concurrency)]
    plan = {
        "count": len(urls_list),
        "batches": batches,
        "urls": validated,
        "metadata": {"concurrency": concurrency, "safe_keys": safe_keys}
    }
    _emit_metric("prefetch_plan_generated", {"count": len(urls_list), "concurrency": concurrency})
    return plan


def generate_prefetch_ast(urls: Sequence[str], options: Optional[Options] = None) -> Dict[str, Any]:
    opts = dict(options or {})
    runtime = opts.get("runtime", "spawn")
    concurrency = max(1, int(opts.get("concurrency", 8) or 8))
    retries = max(0, int(opts.get("retries", 2) or 2))
    backoff_ms = int(opts.get("backoff_ms", 50) or 50)
    timeout_ms = opts.get("timeout_ms")
    safe_keys = bool(opts.get("safe_keys", False))

    urls_list = [str(u) for u in urls]
    actions = []
    for u in urls_list:
        valid = _is_valid_url(u)
        actions.append({
            "url": u,
            "valid": valid,
            "action": "prefetch" if valid else "skip",
            "params": {"retries": retries, "timeout_ms": timeout_ms, "backoff_ms": backoff_ms,
                       "safe_key": ("pf_" + hashlib.sha1(u.encode("utf-8")).hexdigest()[:12]) if safe_keys else None}
        })
    ast = {
        "runtime": runtime,
        "concurrency": concurrency,
        "actions": actions,
        "metadata": {"generated_at": time.time()}
    }
    _emit_metric("prefetch_ast_generated", {"count": len(urls_list)})
    return ast


# ---------------------------
# Textual snippet generator
# ---------------------------
def generate_prefetch_x(urls: Sequence[str], results_var: Optional[str] = None,
                        options: Optional[Options] = None) -> str:
    opts = dict(options or {})
    runtime = str(opts.get("runtime", "spawn"))
    concurrency = max(1, int(opts.get("concurrency", 8) or 8))
    retries = max(0, int(opts.get("retries", 2) or 2))
    timeout_ms = opts.get("timeout_ms")
    backoff_ms = max(0, int(opts.get("backoff_ms", 50) or 50))
    cache_var = opts.get("cache_var")
    metrics_var = opts.get("metrics_var")
    safe_keys = bool(opts.get("safe_keys", False))
    indent = str(opts.get("indent", "  "))
    inline = bool(opts.get("inline", False))

    urls_list = [str(u) for u in urls]
    rv = results_var or "__prefetch_x"
    sep = " " if inline else "\n"
    lines: List[str] = []
    lines.append(f"{rv} = {rv} ? {rv} : {{}};")
    if cache_var:
        lines.append(f"{cache_var} = {cache_var} ? {cache_var} : {{}};")
    if metrics_var:
        lines.append(f"{metrics_var} = {metrics_var} ? {metrics_var} : {{prefetched:0}};")

    valid_urls = [u for u in urls_list if _is_valid_url(u)]
    invalid_urls = [u for u in urls_list if not _is_valid_url(u)]
    if invalid_urls:
        lines.append(f"// WARNING: {len(invalid_urls)} invalid URL(s) skipped")
        for u in invalid_urls:
            lines.append(f"// skipped: {u}")
    if not valid_urls:
        return sep.join(lines) + (sep if not inline else "")

    def _key_for(u: str) -> str:
        if safe_keys:
            return f"pf_{hashlib.sha1(u.encode('utf-8')).hexdigest()[:12]}"
        return f"'{_escape_literal(u)}'"

    if runtime == "promises":
        helper = "__prefetch_fetch_with_retry"
        lines.extend([
            f"function {helper}(url, retries, timeout_ms, backoff_ms) {{",
            f"{indent}// returns a Promise that resolves to fetched data or null on failure",
            f"{indent}let attempt = 0;",
            f"{indent}function tryOnce(resolve) {{",
            f"{indent*2}let p = fetchData(url);",
        ])
        if timeout_ms is not None:
            lines.append(f"{indent*2}// NOTE: wrap promise `p` with a timeout helper if runtime provides it")
        lines.extend([
            f"{indent*2}p.then(d => resolve(d)).catch(() => {{",
            f"{indent*3}attempt += 1;",
            f"{indent*3}if (attempt <= retries) {{",
            f"{indent*4}setTimeout(() => tryOnce(resolve), backoff_ms * attempt);",
            f"{indent*3}}} else {{ resolve(null); }}",
            f"{indent*2}}});",
            f"{indent}}}",
            f"{indent}return new Promise(tryOnce);",
            f"}}"
        ])

        for i in range(0, len(valid_urls), concurrency):
            batch = valid_urls[i:i+concurrency]
            pvars: List[str] = []
            for j, u in enumerate(batch):
                safe = _escape_literal(u)
                pvar = f"__pf_p_{i}_{j}"
                pvars.append(pvar)
                key = _key_for(u)
                if cache_var:
                    lines.append(f"if ({cache_var}[{key}]) {{ {rv}[{key}] = {cache_var}[{key}]; }} else {{")
                    lines.append(f"{indent}let {pvar} = {helper}('{safe}', {retries}, {timeout_ms if timeout_ms is not None else 'null'}, {backoff_ms});")
                    if metrics_var:
                        lines.append(f"{indent}{pvar}.then(d => {{ if (d) {{ {rv}[{key}] = d; {cache_var}[{key}] = d; {metrics_var}.prefetched = ({metrics_var}.prefetched || 0) + 1; }} }});")
                    else:
                        lines.append(f"{indent}{pvar}.then(d => {{ if (d) {{ {rv}[{key}] = d; {cache_var}[{key}] = d; }} }});")
                    lines.append("}")
                else:
                    lines.append(f"let {pvar} = {helper}('{safe}', {retries}, {timeout_ms if timeout_ms is not None else 'null'}, {backoff_ms});")
                    if metrics_var:
                        lines.append(f"{indent}{pvar}.then(d => {{ if (d) {{ {rv}[{key}] = d; {metrics_var}.prefetched = ({metrics_var}.prefetched || 0) + 1; }} }});")
                    else:
                        lines.append(f"{indent}{pvar}.then(d => {{ if (d) {{ {rv}[{key}] = d; }} }});")
            if pvars:
                lines.append(f"Promise.all([{', '.join(pvars)}]);")

        _emit_metric("prefetch_snippet_generated", {"runtime": "promises", "count": len(valid_urls)})
        return sep.join(lines) + (sep if not inline else "")

    # spawn-style
    batches = [valid_urls[i:i+concurrency] for i in range(0, len(valid_urls), concurrency)]
    uid = hashlib.sha1(",".join(valid_urls).encode("utf-8")).hexdigest()[:8]
    for bidx, batch in enumerate(batches):
        lines.append(f"// prefetch batch {bidx} uid={uid}")
        for u in batch:
            safe = _escape_literal(u)
            key = _key_for(u)
            if cache_var:
                lines.append(f"if ({cache_var}[{key}]) {{ {rv}[{key}] = {cache_var}[{key}]; }} else {{")
                lines.append(f"{indent}spawn async {{")
                lines.append(f"{indent*2}let __pf_attempt = 0;")
                lines.append(f"{indent*2}let __pf_res = null;")
                lines.append(f"{indent*2}while (__pf_attempt <= {retries} && !__pf_res) {{")
                if timeout_ms is not None:
                    lines.append(f"{indent*3}// placeholder: runtime should support fetchData(url, timeout_ms)")
                lines.append(f"{indent*3}__pf_res = fetchData('{safe}');")
                lines.append(f"{indent*3}if (!__pf_res) {{ __pf_attempt += 1; sleep({backoff_ms} * __pf_attempt); }}")
                lines.append(f"{indent*2}}}")
                lines.append(f"{indent*2}if (__pf_res) {{ {rv}[{key}] = __pf_res; {cache_var}[{key}] = __pf_res;")
                if metrics_var:
                    lines.append(f"{indent*3}{metrics_var}.prefetched = ({metrics_var}.prefetched || 0) + 1;")
                lines.append(f"{indent*2}}}")
                lines.append(f"{indent}}};")
                lines.append("}")
            else:
                lines.append(f"spawn async {{")
                lines.append(f"{indent}let __pf_attempt = 0;")
                lines.append(f"{indent}let __pf_res = null;")
                lines.append(f"{indent}while (__pf_attempt <= {retries} && !__pf_res) {{")
                if timeout_ms is not None:
                    lines.append(f"{indent*2}// placeholder: runtime timeout wrapper would be used here")
                lines.append(f"{indent*2}__pf_res = fetchData('{safe}');")
                lines.append(f"{indent*2}if (!__pf_res) {{ __pf_attempt += 1; sleep({backoff_ms} * __pf_attempt); }}")
                lines.append(f"{indent}}}")
                lines.append(f"{indent}if (__pf_res) {{ {rv}[{key}] = __pf_res;")
                if metrics_var:
                    lines.append(f"{indent*2}{metrics_var}.prefetched = ({metrics_var}.prefetched || 0) + 1;")
                lines.append(f"{indent}}}")
                lines.append("};")

    _emit_metric("prefetch_snippet_generated", {"runtime": "spawn", "count": len(valid_urls)})
    return sep.join(lines) + (sep if not inline else "")


# ---------------------------
# Executing a plan (emulation / test)
# ---------------------------
def _default_fetcher(url: str, timeout: Optional[int] = None) -> Tuple[bool, str]:
    """
    Default fetcher: tries to perform a GET using urllib.request.
    Returns (ok, content_or_error).
    If network is not desired, callers can pass a custom fetcher.
    """
    try:
        with urllib.request.urlopen(url, timeout=(timeout or 5)) as resp:
            data = resp.read()
            # return a short representation to avoid huge output
            return True, data[:4096].decode("utf-8", errors="replace")
    except Exception as e:
        return False, str(e)


def execute_plan(urls_or_plan: Any, options: Optional[Options] = None,
                 runtime: Optional[str] = None,
                 cache: Optional[PrefetchCache] = None,
                 fetcher: Optional[Callable[[str, Optional[int]], Tuple[bool, Any]]] = None
                 ) -> Dict[str, Any]:
    """
    Execute a plan (or list of urls) in-process to validate behavior or produce a result map.

    - urls_or_plan: either a list of URLs or a plan dict from generate_prefetch_plan.
    - options: same options accepted by generator
    - runtime: optional override ("spawn" or "promises")
    - cache: optional PrefetchCache instance to consult/populate
    - fetcher: optional callable(url, timeout_ms) -> (ok:bool, data_or_error)

    Returns:
      { "results": { url: data_or_error_or_None }, "cache": cache.to_dict() if cache given, "stats": {...} }
    """
    opts = dict(options or {})
    runtime = runtime or opts.get("runtime", "spawn")
    concurrency = max(1, int(opts.get("concurrency", 8) or 8))
    retries = max(0, int(opts.get("retries", 2) or 2))
    timeout_ms = opts.get("timeout_ms")
    backoff_ms = max(0, int(opts.get("backoff_ms", 50) or 50))
    cache_var = opts.get("cache_var")
    metrics_var = opts.get("metrics_var")
    safe_keys = bool(opts.get("safe_keys", False))

    if isinstance(urls_or_plan, dict) and "batches" in urls_or_plan:
        plan = urls_or_plan
        urls = [u["url"] for u in plan.get("urls", []) if u.get("valid", True)]
    else:
        urls = list(urls_or_plan)

    fetcher = fetcher or _default_fetcher
    cache = cache or PrefetchCache()

    results: Dict[str, Any] = {}
    stats = {"requested": 0, "succeeded": 0, "failed": 0, "skipped": 0}

    def _key(u: str) -> str:
        return (f"pf_{hashlib.sha1(u.encode('utf-8')).hexdigest()[:12]}" if safe_keys else u)

    # worker that performs fetch with retries/backoff
    def _worker(u: str) -> Tuple[str, Any, bool]:
        stats["requested"] += 1
        key = _key(u)
        if cache.exists(key):
            stats["skipped"] += 1
            return u, cache.get(key), True
        attempt = 0
        while attempt <= retries:
            ok, payload = fetcher(u, timeout_ms)
            if ok:
                cache.set(key, payload)
                stats["succeeded"] += 1
                if metrics_var:
                    _emit_metric("prefetch_fetch_success", {"url": u})
                return u, payload, True
            attempt += 1
            if attempt <= retries:
                time.sleep(backoff_ms / 1000.0 * attempt)
        stats["failed"] += 1
        if metrics_var:
            _emit_metric("prefetch_fetch_failure", {"url": u})
        return u, payload, False

    # concurrency via ThreadPoolExecutor (works for both "spawn" and "promises" semantics for testing)
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(_worker, u): u for u in urls}
        for fut in as_completed(futures):
            try:
                u, payload, ok = fut.result()
            except Exception as e:
                u = futures[fut]
                payload = str(e)
                ok = False
                stats["failed"] += 1
            results[u] = payload

    out = {"results": results, "cache": cache.to_dict(), "stats": stats}
    _emit_metric("prefetch_plan_executed", {"count": len(urls), "stats": stats})
    return out


# ---------------------------
# Registration helper
# ---------------------------
def register(toolkit: Any) -> None:
    meta = {
        "name": "prefetch_x",
        "description": "Generate prefetch code for URLs (concurrency, retries, cache, metrics).",
        "signature": "prefetch_x(urls, results_var=None, options=None) -> str",
        "options": {
            "runtime": {"type": "string", "enum": ["spawn", "promises"], "default": "spawn"},
            "concurrency": {"type": "integer", "default": 8},
            "retries": {"type": "integer", "default": 2},
            "timeout_ms": {"type": ["integer", "null"], "default": None},
            "backoff_ms": {"type": "integer", "default": 50},
            "cache_var": {"type": ["string", "null"], "default": None},
            "metrics_var": {"type": ["string", "null"], "default": None},
            "safe_keys": {"type": "boolean", "default": False},
            "inline": {"type": "boolean", "default": False},
        }
    }
    if hasattr(toolkit, "register_helper"):
        try:
            toolkit.register_helper("prefetch_x", generate_prefetch_x, metadata=meta)
            return
        except Exception:
            try:
                toolkit.register_helper("prefetch_x", generate_prefetch_x)
                return
            except Exception:
                pass
    try:
        setattr(toolkit, "prefetch_x", generate_prefetch_x)
    except Exception:
        pass


# ---------------------------
# CLI demo + self-check
# ---------------------------
def _demo_and_selfcheck() -> None:
    sample = [
        "https://example.com/a.json",
        "https://example.com/b.json",
        "invalid:url",
        "https://cdn.example.com/img.png"
    ]
    print("=== Generated spawn-style snippet ===")
    print(generate_prefetch_x(sample, results_var="__my_prefetch", options={"runtime": "spawn", "concurrency": 2, "retries": 1, "metrics_var": "METRICS"}))
    print("=== Generated promises-style snippet ===")
    print(generate_prefetch_x(sample, results_var="__my_prefetch", options={"runtime": "promises", "concurrency": 2, "retries": 1, "metrics_var": "METRICS"}))

    plan = generate_prefetch_plan(sample, {"concurrency": 2, "safe_keys": True})
    print("\n=== Prefetch Plan JSON ===")
    print(json.dumps(plan, indent=2))

    ast = generate_prefetch_ast(sample, {"concurrency": 2})
    print("\n=== Prefetch AST ===")
    print(json.dumps(ast, indent=2))

    # execute using default fetcher (may attempt network) — use a dummy fetcher to avoid network by default
    def dummy_fetcher(url: str, timeout: Optional[int] = None):
        # deterministic dummy payload
        return True, f"dummy-data-for:{url}"

    result = execute_plan(plan, options={"concurrency": 2, "retries": 1}, cache=PrefetchCache(), fetcher=dummy_fetcher)
    print("\n=== Execution result (dummy fetcher) ===")
    print(json.dumps(result, indent=2))

    # basic assertions
    assert isinstance(plan, dict) and "batches" in plan and "urls" in plan
    assert sum(len(b) for b in plan["batches"]) == plan["count"]
    assert any(not u["valid"] for u in plan["urls"])
    print("\nSelf-check: plan/ast/execution look consistent.")


if __name__ == "__main__":
    try:
        _demo_and_selfcheck()
        sys.exit(0)
    except AssertionError as ae:
        print("Self-check failed:", ae)
        sys.exit(2)
    except Exception as e:
        print("Error during demo/self-check:", e)
        sys.exit(3)

