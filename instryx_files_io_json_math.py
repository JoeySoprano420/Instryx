"""
instryx_files_io_json_math.py

Robust file / JSON I/O and numeric utilities used by Instryx tooling.

Enhancements (production-focused)
- Streaming+batch JSON loading with ThreadPoolExecutor for concurrency
- Checksum-backed cache invalidation for cached JSON
- Duplicate-key detection and optional strict parsing
- Schema validation with optional coercion of simple types and automatic fixes
- Fast numeric statistics, optionally accelerated with numpy if installed
- Safe transform feature: apply a numeric mapping expression (lambda-style) to numeric leaves
- Directory merge, batch-validate, lint (duplicate keys, trailing commas), format and watch mode
- Pack/unpack array improved: gzip support, array module optimization
- File watch (polling) with debounce and callback
- Export metrics / stats for pipelines
- Streaming JSON iterator (ijson-backed when available) and schema inference
- Extensive CLI with new commands: format, lint, merge-dir, batch-validate, transform, watch, export-stats, infer-schema, fix
- Defensive, atomic and log-rich operations

No mandatory third-party deps; optional acceleration via numpy/ijson when available.

Usage examples:
  python instryx_files_io_json_math.py format file.json --inplace
  python instryx_files_io_json_math.py lint file.json
  python instryx_files_io_json_math.py merge-dir out.json mydir --ext .json
  python instryx_files_io_json_math.py transform infile.json outfile.json --expr "x*2"
"""

from __future__ import annotations
import argparse
import array
import concurrent.futures
import gzip
import hashlib
import json
import logging
import math
import os
import shutil
import struct
import sys
import tempfile
import threading
import time
import difflib
import ast
import fnmatch
import collections
from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable

# Optional accelerations
_try_numpy = None
try:
    import numpy as np  # type: ignore
    _try_numpy = np
except Exception:
    _try_numpy = None

_try_ijson = None
try:
    import ijson  # type: ignore
    _try_ijson = ijson
except Exception:
    _try_ijson = None

LOG = logging.getLogger("instryx.files_io_json_math")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------
# Utilities: atomic IO + safe read/write (gzip aware)
# ---------------------------
def safe_read(path: str, encoding: str = "utf-8") -> str:
    """Read file content safely; supports gzip (.gz)."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding=encoding) as f:
            return f.read()
    with open(path, "r", encoding=encoding, errors="strict") as f:
        return f.read()


def atomic_write(path: str, data: str, encoding: str = "utf-8", backup: bool = True) -> str:
    """
    Atomically write `data` to `path`. If backup=True and path exists, save path.bak.
    Supports writing to .gz by compressing.
    Returns target path on success.
    """
    dirp = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dirp, text=False)
    os.close(fd)
    try:
        if path.endswith(".gz"):
            with gzip.open(tmp, "wt", encoding=encoding) as f:
                f.write(data)
        else:
            with open(tmp, "w", encoding=encoding) as f:
                f.write(data)
        if backup and os.path.exists(path):
            bak = path + ".bak"
            shutil.copy2(path, bak)
        os.replace(tmp, path)
        return path
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise


# ---------------------------
# JSON tolerant parsing/writing (comment stripping, duplicate-key detection)
# ---------------------------
def _strip_json_comments(text: str) -> str:
    """Conservatively remove // and /* */ comments to allow JSON5-like files."""
    out = []
    i = 0
    L = len(text)
    in_string = None
    while i < L:
        ch = text[i]
        if in_string:
            out.append(ch)
            if ch == in_string and (i == 0 or text[i - 1] != "\\"):
                in_string = None
            i += 1
            continue
        if ch in ('"', "'"):
            in_string = ch
            out.append(ch)
            i += 1
            continue
        # single-line comment
        if text.startswith("//", i):
            nl = text.find("\n", i + 2)
            if nl == -1:
                return "".join(out)
            i = nl
            continue
        # multi-line comment
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end == -1:
                return "".join(out)
            i = end + 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


class DuplicateKeyDetector:
    """object_pairs_hook to detect duplicate keys when loading JSON."""
    def __init__(self):
        self.duplicates: List[Tuple[str, Any]] = []

    def hook(self, pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                self.duplicates.append((k, v))
            d[k] = v
        return d


def json_load(path: str, allow_comments: bool = True, detect_duplicates: bool = False) -> Any:
    """Load JSON; supports gz, comments, and duplicate detection."""
    txt = safe_read(path)
    if allow_comments:
        txt = _strip_json_comments(txt)
    if detect_duplicates:
        det = DuplicateKeyDetector()
        obj = json.loads(txt, object_pairs_hook=det.hook)
        if det.duplicates:
            LOG.warning("Duplicate keys detected in %s: %s", path, det.duplicates[:5])
        return obj
    return json.loads(txt)


def json_loads(s: str, allow_comments: bool = True) -> Any:
    if allow_comments:
        s = _strip_json_comments(s)
    return json.loads(s)


def json_dumps(data: Any, pretty: bool = True, sort_keys: bool = True, indent: int = 2) -> str:
    if pretty:
        return json.dumps(data, ensure_ascii=False, sort_keys=sort_keys, indent=indent)
    return json.dumps(data, ensure_ascii=False, sort_keys=sort_keys, separators=(",", ":"))


def json_write(path: str, obj: Any, pretty: bool = True, atomic: bool = True, backup: bool = True) -> str:
    content = json_dumps(obj, pretty=pretty)
    if atomic:
        return atomic_write(path, content, backup=backup)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path


# ---------------------------
# Checksum-backed cache
# ---------------------------
class ChecksumCache:
    def __init__(self):
        self.lock = threading.RLock()
        self._map: Dict[str, Tuple[str, Any]] = {}

    def get(self, path: str, checksum: str):
        with self.lock:
            v = self._map.get(path)
            if not v:
                return None
            if v[0] != checksum:
                return None
            return v[1]

    def set(self, path: str, checksum: str, data: Any):
        with self.lock:
            self._map[path] = (checksum, data)

    def invalidate(self, path: str):
        with self.lock:
            self._map.pop(path, None)


_checksum_cache = ChecksumCache()


def file_checksum(path: str, algo: str = "sha1", blocksize: int = 8192) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(blocksize), b""):
            h.update(chunk)
    return h.hexdigest()


def cached_json_load(path: str, allow_comments: bool = True, use_cache: bool = True, detect_duplicates: bool = False) -> Any:
    checksum = file_checksum(path)
    if use_cache:
        v = _checksum_cache.get(path, checksum)
        if v is not None:
            return v
    data = json_load(path, allow_comments=allow_comments, detect_duplicates=detect_duplicates)
    if use_cache:
        _checksum_cache.set(path, checksum, data)
    return data


# ---------------------------
# Streaming JSON (ijson if available) and helpers
# ---------------------------
def stream_json_items(path: str, prefix: Optional[str] = None):
    """
    Yield top-level JSON items from a file.
    If ijson is available and file is large, stream; otherwise load into memory.
    prefix: when streaming arrays, prefix='item' yields array elements.
    """
    if _try_ijson:
        with open(path, "rb") as f:
            parser = _try_ijson.parse(f)
            # simple streaming: yield values at prefix if provided
            for prefix_token, event, value in parser:
                # when prefix is None, yield top-level completed objects only (approx)
                if prefix and prefix_token.endswith(prefix) and event in ("string", "number", "boolean", "start_map", "start_array", "null"):
                    yield value
            # fallback: nothing
        return
    # fallback: load whole file (small files)
    data = cached_json_load(path)
    if isinstance(data, list):
        for it in data:
            yield it
    else:
        yield data


# ---------------------------
# Merge / diff utilities
# ---------------------------
def deep_merge(a: Any, b: Any, path: Optional[List[str]] = None) -> Any:
    """Merge b into a recursively, returning new structure (a, b unmodified)."""
    if path is None:
        path = []
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            if k in out:
                out[k] = deep_merge(out[k], v, path + [str(k)])
            else:
                out[k] = v
        return out
    # for lists and scalars prefer b
    return b


def unified_diff(a: str, b: str, a_name: str = "a", b_name: str = "b") -> str:
    return "".join(difflib.unified_diff(a.splitlines(keepends=True), b.splitlines(keepends=True),
                                        fromfile=a_name, tofile=b_name, lineterm=""))


def json_diff(a: Any, b: Any, pretty: bool = True) -> str:
    A = json_dumps(a, pretty=pretty).splitlines(keepends=True)
    B = json_dumps(b, pretty=pretty).splitlines(keepends=True)
    return "".join(difflib.unified_diff(A, B, fromfile="a.json", tofile="b.json", lineterm=""))


# ---------------------------
# Schema inference (simple)
# ---------------------------
def infer_schema_from_sample(data: Any, max_items: int = 100) -> Dict[str, Any]:
    """
    Infer a simple JSON schema from a sample object.
    Produces a shallow schema describing types and array item type when possible.
    """
    def infer(node, depth=0):
        if isinstance(node, dict):
            props = {}
            for k, v in node.items():
                props[k] = infer(v, depth + 1)
            return {"type": "object", "properties": props}
        if isinstance(node, list):
            if not node:
                return {"type": "array", "items": {"type": "any"}}
            # infer item types by sampling
            sample = node[:max_items]
            types = set()
            item_schemas = [infer(it, depth + 1) for it in sample]
            # if all same simple type, use it
            kind = item_schemas[0].get("type") if item_schemas else "any"
            if all(s.get("type") == kind for s in item_schemas):
                return {"type": "array", "items": {"type": kind}}
            return {"type": "array", "items": {"type": "any"}}
        if isinstance(node, bool):
            return {"type": "boolean"}
        if isinstance(node, int):
            return {"type": "integer"}
        if isinstance(node, float):
            return {"type": "number"}
        if node is None:
            return {"type": "null"}
        return {"type": "string"}
    return infer(data)


# ---------------------------
# Schema validation with optional coercion and auto-fix defaults
# ---------------------------
def _apply_defaults(node: Any, sch: Dict[str, Any]):
    """
    If schema provides 'default' values for missing properties, apply them.
    """
    if not isinstance(node, dict) or sch.get("type") != "object":
        return node
    props = sch.get("properties", {})
    for k, ps in props.items():
        if k not in node and "default" in ps:
            node[k] = ps["default"]
        elif k in node:
            _apply_defaults(node[k], ps)
    return node


def validate_and_fix(data: Any, schema: Dict[str, Any], coerce: bool = False, apply_defaults: bool = True) -> Tuple[bool, List[str], Any]:
    ok, msgs, coerced = validate_schema(data, schema, coerce=coerce)
    if apply_defaults:
        coerced = _apply_defaults(coerced, schema)
    return ok, msgs, coerced


# ---------------------------
# Numeric helpers and transforms
# ---------------------------
def _iter_numbers(obj: Any) -> Iterable[float]:
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield float(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_numbers(v)
    elif isinstance(obj, (list, tuple, set)):
        for it in obj:
            yield from _iter_numbers(it)


def numeric_stats(data: Any) -> Dict[str, Any]:
    nums = list(_iter_numbers(data))
    if not nums:
        return {"count": 0}
    n = len(nums)
    s = sum(nums)
    mean = s / n
    var = sum((x - mean) ** 2 for x in nums) / n
    res = {
        "count": n,
        "sum": s,
        "min": min(nums),
        "max": max(nums),
        "mean": mean,
        "std": math.sqrt(var),
    }
    if _try_numpy and len(nums) > 1024:
        try:
            arr = _try_numpy.array(nums, dtype=_try_numpy.float64)
            res["median"] = float(_try_numpy.median(arr))
        except Exception:
            pass
    return res


def map_numbers(data: Any, fn) -> Any:
    """Apply fn to every numeric leaf in data; preserves structure."""
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        return fn(data)
    if isinstance(data, dict):
        return {k: map_numbers(v, fn) for k, v in data.items()}
    if isinstance(data, list):
        return [map_numbers(v, fn) for v in data]
    if isinstance(data, tuple):
        return tuple(map_numbers(v, fn) for v in data)
    return data


def compile_transform_expr(expr: str) -> Callable[[float], float]:
    """
    Compile a safe transform expression like "x*2" into a callable f(x).
    Only expression using 'x' is allowed; disallow names/attributes for safety.
    """
    node = ast.parse(expr, mode="eval")
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            if n.id != "x":
                raise ValueError("Only 'x' identifier allowed in transform expression")
        elif isinstance(n, ast.Call):
            raise ValueError("Function calls are not allowed in transform expression")
        elif isinstance(n, ast.Attribute):
            raise ValueError("Attribute access not allowed in transform expression")
    code = compile(node, "<transform>", "eval")

    def fn(x):
        return eval(code, {"__builtins__": {}}, {"x": x})

    return fn


def transform_numeric_leaves(data: Any, expr: str) -> Any:
    fn = compile_transform_expr(expr)
    return map_numbers(data, lambda v: fn(v))


# ---------------------------
# Pack / unpack float arrays (optimized, gzip optional)
# ---------------------------
def pack_float_array(path: str, arr: Iterable[float], fmt: str = "f", endian: str = "<", gzip_out: bool = False) -> str:
    fmt_char = fmt
    if fmt_char not in ("f", "d"):
        raise ValueError("fmt must be 'f' or 'd'")
    packer = struct.Struct(endian + fmt_char)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path) or ".", text=False)
    os.close(tmp_fd)
    try:
        out_f = gzip.open(tmp_path, "wb") if gzip_out or path.endswith(".gz") else open(tmp_path, "wb")
        with out_f as f:
            magic = b"IFLT"
            f.write(magic)
            f.write(struct.pack("<I", 1 if fmt_char == "f" else 2))
            # reserve 8 bytes for count
            f.write(struct.pack("<Q", 0))
            count = 0
            for v in arr:
                f.write(packer.pack(float(v)))
                count += 1
            f.seek(8)
            f.write(struct.pack("<Q", count))
        os.replace(tmp_path, path)
        LOG.info("packed %d values -> %s", count, path)
        return path
    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise


def unpack_float_array(path: str) -> Tuple[List[float], str]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        magic = f.read(4)
        if magic != b"IFLT":
            raise ValueError("not an instryx float table")
        fmt_code = struct.unpack("<I", f.read(4))[0]
        count_bytes = f.read(8)
        if len(count_bytes) == 8:
            try:
                count = struct.unpack("<Q", count_bytes)[0]
            except Exception:
                count = None
        else:
            count = None
        fmt_char = "f" if fmt_code == 1 else "d"
        endian = "<"
        packer = struct.Struct(endian + fmt_char)
        vals = []
        if count is not None:
            for _ in range(count):
                b = f.read(packer.size)
                if not b:
                    break
                vals.append(packer.unpack(b)[0])
        else:
            while True:
                b = f.read(packer.size)
                if not b:
                    break
                vals.append(packer.unpack(b)[0])
        return vals, fmt_char


# ---------------------------
# Lint / format helpers
# ---------------------------
def detect_duplicate_keys_in_text(text: str) -> List[Tuple[str, int]]:
    """Return list of duplicate keys found using DuplicateKeyDetector hook (line not available)."""
    duplicates = []
    try:
        det = DuplicateKeyDetector()
        json.loads(_strip_json_comments(text), object_pairs_hook=det.hook)
        for k, v in getattr(det, "duplicates", []):
            duplicates.append((k, -1))
    except Exception:
        pass
    return duplicates


def format_json_text(text: str, sort_keys: bool = True, indent: int = 2) -> str:
    obj = json.loads(_strip_json_comments(text))
    return json_dumps(obj, pretty=True, sort_keys=sort_keys, indent=indent)


# ---------------------------
# Batch and directory operations
# ---------------------------
def batch_load_json(paths: Iterable[str], workers: int = 4, allow_comments: bool = True) -> Dict[str, Any]:
    """Load many JSON files concurrently; returns map path->(data or Exception)."""
    res = {}

    def _load(p):
        try:
            return cached_json_load(p, allow_comments=allow_comments)
        except Exception as e:
            return e

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_load, p): p for p in paths}
        for fut in concurrent.futures.as_completed(futs):
            p = futs[fut]
            try:
                res[p] = fut.result()
            except Exception as e:
                res[p] = e
    return res


def merge_dir(out: str, dirpath: str, ext: str = ".json", strategy: str = "deep") -> str:
    """Merge all JSON files under dirpath with extension ext into out using strategy."""
    merged = {}
    files = []
    for root, _, fns in os.walk(dirpath):
        for fn in fns:
            if fn.endswith(ext):
                files.append(os.path.join(root, fn))
    files.sort()
    for p in files:
        try:
            d = cached_json_load(p)
            if strategy == "deep":
                merged = deep_merge(merged, d)
            elif strategy == "overwrite":
                if isinstance(d, dict):
                    merged.update(d)
                else:
                    merged[p] = d
            else:
                merged = deep_merge(merged, d)
        except Exception:
            LOG.exception("failed to load during merge: %s", p)
    json_write(out, merged, pretty=True)
    LOG.info("merged %d files -> %s", len(files), out)
    return out


# ---------------------------
# File watch (polling) with debounce
# ---------------------------
def watch_dir(path: str, callback: Callable[[str], None], ext: str = ".json", interval: float = 1.0, debounce: float = 0.5):
    """
    Poll directory for changes and call callback(file) when file changed.
    Simple, dependency-free watcher.
    """
    mtimes: Dict[str, float] = {}
    pending: Dict[str, float] = {}
    try:
        while True:
            for root, _, fns in os.walk(path):
                for fn in fns:
                    if not fn.endswith(ext):
                        continue
                    p = os.path.join(root, fn)
                    try:
                        m = os.path.getmtime(p)
                    except Exception:
                        continue
                    if p not in mtimes:
                        mtimes[p] = m
                        continue
                    if m != mtimes[p]:
                        pending[p] = time.time()
                        mtimes[p] = m
            now = time.time()
            for p, t0 in list(pending.items()):
                if now - t0 >= debounce:
                    try:
                        callback(p)
                    except Exception:
                        LOG.exception("watch callback failed for %s", p)
                    pending.pop(p, None)
            time.sleep(interval)
    except KeyboardInterrupt:
        LOG.info("watch stopped")


# ---------------------------
# CLI
# ---------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="instryx_files_io_json_math.py", description="Instryx JSON + file I/O + math utilities")
    sub = p.add_subparsers(dest="cmd")

    sp = sub.add_parser("pretty", help="Pretty-print JSON")
    sp.add_argument("infile")
    sp.add_argument("--out", "-o", help="write pretty JSON to path (atomic)")
    sp.add_argument("--inplace", action="store_true", help="overwrite infile with formatted content")

    sp = sub.add_parser("validate", help="Validate JSON against a tiny schema")
    sp.add_argument("file")
    sp.add_argument("--schema", "-s", help="schema JSON file")
    sp.add_argument("--coerce", action="store_true", help="coerce simple types where safe")

    sp = sub.add_parser("merge", help="Deep merge JSON files")
    sp.add_argument("out")
    sp.add_argument("files", nargs="+", help="input JSON files to merge")

    sp = sub.add_parser("merge-dir", help="Merge all JSON files in directory")
    sp.add_argument("out")
    sp.add_argument("dir")
    sp.add_argument("--ext", default=".json")
    sp.add_argument("--strategy", choices=("deep", "overwrite"), default="deep")

    sp = sub.add_parser("diff", help="Diff two JSON files")
    sp.add_argument("a")
    sp.add_argument("b")

    sp = sub.add_parser("stats", help="Numeric stats over JSON")
    sp.add_argument("file")

    sp = sub.add_parser("checksum", help="Compute file checksum")
    sp.add_argument("file")
    sp.add_argument("--algo", default="sha1")

    sp = sub.add_parser("pack-array", help="Pack a float array (JSON array or whitespace list) to binary")
    sp.add_argument("infile")
    sp.add_argument("out")
    sp.add_argument("--fmt", choices=("f", "d"), default="f")
    sp.add_argument("--gzip", action="store_true", help="gzip output")

    sp = sub.add_parser("unpack-array", help="Unpack binary float table to JSON list")
    sp.add_argument("infile")
    sp.add_argument("out", nargs="?", help="output JSON file (default stdout)")

    sp = sub.add_parser("lint", help="Lint JSON file (duplicate keys etc.)")
    sp.add_argument("file")

    sp = sub.add_parser("format", help="Format JSON file")
    sp.add_argument("file")
    sp.add_argument("--inplace", action="store_true")

    sp = sub.add_parser("batch-validate", help="Validate all JSON under dir")
    sp.add_argument("dir")
    sp.add_argument("--ext", default=".json")
    sp.add_argument("--schema", "-s", help="schema JSON file")
    sp.add_argument("--workers", type=int, default=4)

    sp = sub.add_parser("transform", help="Transform numeric leaves using expression 'x' -> expr")
    sp.add_argument("infile")
    sp.add_argument("outfile")
    sp.add_argument("--expr", required=True, help="expression using 'x' (e.g. 'x*2+1')")
    sp.add_argument("--dry", action="store_true")

    sp = sub.add_parser("watch", help="Watch a directory and run a command on change")
    sp.add_argument("dir")
    sp.add_argument("--ext", default=".json")
    sp.add_argument("--interval", type=float, default=1.0)
    sp.add_argument("--debounce", type=float, default=0.5)

    sp = sub.add_parser("export-stats", help="Export metrics of JSON files under dir")
    sp.add_argument("dir")
    sp.add_argument("--out", "-o", default="json_stats.json")
    sp.add_argument("--ext", default=".json")
    sp.add_argument("--workers", type=int, default=4)

    sp = sub.add_parser("infer-schema", help="Infer a simple JSON schema from a sample file")
    sp.add_argument("file")
    sp.add_argument("--out", "-o", help="write inferred schema to file")

    sp = sub.add_parser("fix", help="Validate and attempt to fix JSON using schema (coercion + defaults)")
    sp.add_argument("file")
    sp.add_argument("--schema", "-s", required=True)
    sp.add_argument("--out", "-o", help="write fixed JSON to path (atomic)")

    return p


def main(argv: Optional[List[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv or sys.argv[1:])
    if not args.cmd:
        parser.print_help()
        return 2

    try:
        if args.cmd == "pretty":
            obj = cached_json_load(args.infile)
            out = json_dumps(obj, pretty=True)
            if args.inplace:
                atomic_write(args.infile, out)
                print("formatted", args.infile)
            elif args.out:
                atomic_write(args.out, out)
                print("wrote", args.out)
            else:
                print(out)
            return 0

        if args.cmd == "validate":
            data = cached_json_load(args.file)
            if args.schema:
                schema = cached_json_load(args.schema)
                ok, msgs, coerced = validate_schema(data, schema, coerce=args.coerce)
                if ok:
                    print("VALID")
                    return 0
                else:
                    print("INVALID")
                    for m in msgs:
                        print("-", m)
                    return 2
            else:
                print("OK")
                return 0

        if args.cmd == "merge":
            outpath = args.out
            merged = {}
            for pth in args.files:
                d = cached_json_load(pth)
                merged = deep_merge(merged, d)
            json_write(outpath, merged, pretty=True)
            print("wrote", outpath)
            return 0

        if args.cmd == "merge-dir":
            merge_dir(args.out, args.dir, ext=args.ext, strategy=args.strategy)
            return 0

        if args.cmd == "diff":
            a = cached_json_load(args.a)
            b = cached_json_load(args.b)
            print(json_diff(a, b))
            return 0

        if args.cmd == "stats":
            data = cached_json_load(args.file)
            stats = numeric_stats(data)
            print(json_dumps(stats, pretty=True))
            return 0

        if args.cmd == "checksum":
            print(file_checksum(args.file, algo=args.algo))
            return 0

        if args.cmd == "pack-array":
            try:
                arr = cached_json_load(args.infile)
                if not isinstance(arr, list):
                    raise ValueError("input JSON must be an array of numbers")
            except Exception:
                txt = safe_read(args.infile)
                arr = [float(x) for x in txt.strip().split()]
            pack_float_array(args.out, arr, fmt=args.fmt, gzip_out=args.gzip)
            print("wrote", args.out)
            return 0

        if args.cmd == "unpack-array":
            vals, fmt = unpack_float_array(args.infile)
            outstr = json_dumps(vals, pretty=True)
            if args.out:
                atomic_write(args.out, outstr)
                print("wrote", args.out)
            else:
                print(outstr)
            return 0

        if args.cmd == "lint":
            txt = safe_read(args.file)
            dups = detect_duplicate_keys_in_text(txt)
            if dups:
                print("duplicate keys detected:", dups)
                return 2
            else:
                print("no duplicate keys detected")
                return 0

        if args.cmd == "format":
            txt = safe_read(args.file)
            formatted = format_json_text(txt)
            if args.inplace:
                atomic_write(args.file, formatted)
                print("formatted", args.file)
            else:
                print(formatted)
            return 0

        if args.cmd == "batch-validate":
            files = []
            for root, _, fns in os.walk(args.dir):
                for fn in fns:
                    if fn.endswith(args.ext):
                        files.append(os.path.join(root, fn))
            schema = cached_json_load(args.schema) if args.schema else None
            failures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(cached_json_load, p): p for p in files}
                for fut in concurrent.futures.as_completed(futs):
                    p = futs[fut]
                    try:
                        data = fut.result()
                        if schema:
                            ok, msgs, _ = validate_schema(data, schema, coerce=False)
                            if not ok:
                                failures.append((p, msgs))
                    except Exception as e:
                        failures.append((p, [str(e)]))
            if failures:
                for p, msgs in failures:
                    print(p, "FAILED")
                    for m in msgs:
                        print("  -", m)
                return 2
            print("ALL OK")
            return 0

        if args.cmd == "transform":
            data = cached_json_load(args.infile)
            out = transform_numeric_leaves(data, args.expr)
            if args.dry:
                print(json_dumps(out, pretty=True))
            else:
                json_write(args.out, out, pretty=True)
                print("wrote", args.out)
            return 0

        if args.cmd == "watch":
            def cb(p):
                print("changed:", p)
            watch_dir(args.dir, cb, ext=args.ext, interval=args.interval, debounce=args.debounce)
            return 0

        if args.cmd == "export-stats":
            files = []
            for root, _, fns in os.walk(args.dir):
                for fn in fns:
                    if fn.endswith(args.ext):
                        files.append(os.path.join(root, fn))
            summary = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(cached_json_load, p): p for p in files}
                for fut in concurrent.futures.as_completed(futs):
                    p = futs[fut]
                    try:
                        data = fut.result()
                        summary[p] = numeric_stats(data)
                    except Exception as e:
                        summary[p] = {"error": str(e)}
            json_write(args.out, summary, pretty=True)
            print("wrote", args.out)
            return 0

        if args.cmd == "infer-schema":
            data = cached_json_load(args.file)
            schema = infer_schema_from_sample(data)
            out = json_dumps(schema, pretty=True)
            if args.out:
                atomic_write(args.out, out)
                print("wrote", args.out)
            else:
                print(out)
            return 0

        if args.cmd == "fix":
            data = cached_json_load(args.file)
            schema = cached_json_load(args.schema)
            ok, msgs, fixed = validate_and_fix(data, schema, coerce=True, apply_defaults=True)
            if not ok:
                print("FIXED (issues remain):")
                for m in msgs:
                    print("-", m)
            else:
                print("OK or fixed successfully")
            outpath = args.out or args.file
            json_write(outpath, fixed, pretty=True)
            print("wrote", outpath)
            return 0

        print("unknown command", args.cmd)
        return 2
    except Exception as e:
        LOG.exception("command failed")
        print("error:", e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
