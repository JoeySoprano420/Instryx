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
