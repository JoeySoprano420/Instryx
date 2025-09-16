"""
ciams/ai-engine.py

Extended CIAMS AI assistant.

This file expands the original lightweight heuristic assistant with:
- Many new heuristic rules (profile, inline, assert, unroll, cache hints)
- Batch analysis and batch-apply over directories (concurrent)
- Interactive apply mode (choose suggestions to apply)
- Patch generation (unified diff) and patch application (write .ai.patch and .ai.ix)
- Simple HTTP JSON API server to query suggestions and apply them remotely
- Undo support (backups + restore)
- Local caching of analysis results (LRU)
- More robust memory and logging
- Unit tests / verification runner
- Plugin system for custom rules
- HTML report generation + static server
- Fully executable with Python stdlib

Usage:
  python ciams/ai-engine.py suggest file.ix
  python ciams/ai-engine.py batch-suggest src_dir --max 5
  python ciams/ai-engine.py interactive file.ix
  python ciams/ai-engine.py serve --port 8787
  python ciams/ai-engine.py report myreport.html file.ix
  python ciams/ai-engine.py test

Notes:
- This remains a text-based helper. For robust integration use AST-aware macros and parser.
- Macro application uses macro_overlay module when available.
"""

from __future__ import annotations
import re
import json
import argparse
import shutil
import tempfile
import os
import sys
import time
import importlib
import logging
import concurrent.futures
import threading
import http.server
import socketserver
import urllib.parse
import difflib
import html
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, Any

# Paths
MODULE_DIR = os.path.dirname(__file__)
MEMORY_PATH = os.path.join(MODULE_DIR, "ai_memory.json")
LOG_PATH = os.path.join(MODULE_DIR, "ai_engine.log")
PLUGINS_DIR = os.path.join(MODULE_DIR, "plugins")

# Optional macro overlay integration (lazy)
_macro_overlay = None

# Configure logging
logging.basicConfig(level=logging.INFO, filename=LOG_PATH, filemode="a",
                    format="%(asctime)s [%(levelname)s] %(message)s")


def _try_import_macro_overlay():
    global _macro_overlay
    if _macro_overlay is None:
        try:
            _macro_overlay = importlib.import_module("macro_overlay")
        except Exception:
            _macro_overlay = None
    return _macro_overlay


@dataclass
class Suggestion:
    macro_name: str
    args: List[str]
    reason: str
    score: float  # 0.0 - 1.0
    snippet: Optional[str] = None  # source snippet where suggestion applies
    location: Optional[Tuple[int, int]] = None  # (start_offset, end_offset)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AISimpleMemory:
    """Tiny local JSON-backed memory to count patterns and record accepted suggestions."""

    def __init__(self, path: str = MEMORY_PATH):
        self.path = path
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
            logging.exception("Failed to save memory")

    def record_pattern(self, key: str):
        self._data.setdefault("patterns", {})
        self._data["patterns"][key] = self._data["patterns"].get(key, 0) + 1
        self.save()

    def record_accepted(self, suggestion: Suggestion, filename: Optional[str] = None):
        entry = {"time": int(time.time()), "suggestion": suggestion.to_dict(), "file": filename}
        self._data.setdefault("accepted", []).append(entry)
        self.save()

    def pattern_count(self, key: str) -> int:
        return int(self._data.get("patterns", {}).get(key, 0))

    def export(self) -> Dict[str, Any]:
        return self._data

    def import_data(self, data: Dict[str, Any], merge: bool = True):
        if not merge:
            self._data = data
        else:
            # merge patterns
            patterns = data.get("patterns", {})
            for k, v in patterns.items():
                self._data.setdefault("patterns", {})
                self._data["patterns"][k] = self._data["patterns"].get(k, 0) + v
            # append accepted
            self._data.setdefault("accepted", []).extend(data.get("accepted", []))
        self.save()


class LRUCache:
    """Tiny LRU cache for analysis results (thread-safe)."""

    def __init__(self, capacity: int = 128):
        self.capacity = capacity
        self.lock = threading.Lock()
        self.map: Dict[str, Any] = {}
        self.order: List[str] = []

    def get(self, key: str):
        with self.lock:
            if key in self.map:
                self.order.remove(key)
                self.order.insert(0, key)
                return self.map[key]
            return None

    def set(self, key: str, value: Any):
        with self.lock:
            if key in self.map:
                self.order.remove(key)
            self.map[key] = value
            self.order.insert(0, key)
            if len(self.order) > self.capacity:
                old = self.order.pop()
                self.map.pop(old, None)

    def clear(self):
        with self.lock:
            self.map.clear()
            self.order.clear()


class AIAssistant:
    """
    Heuristic AI assistant that suggests CIAMS macros and can apply them.
    """

    def __init__(self, memory: Optional[AISimpleMemory] = None):
        self.memory = memory or AISimpleMemory()
        self.cache = LRUCache(256)
        # Rules (ordered) - extended with many heuristics
        self.rules = [
            self._rule_sql_injection_risk,
            self._rule_transactional_db_writes,
            self._rule_wraptry_network_calls,
            self._rule_retry_backoff_network_calls,
            self._rule_async_for_background_tasks,
            self._rule_memoize_pure_functions,
            self._rule_suggest_cache_ttl,
            self._rule_lazy_injection_candidates,
            self._rule_debug_for_public_funcs,
            self._rule_defer_for_resource_cleanup,
            self._rule_ffi_export_for_prefix_funcs,
            self._rule_profile_hotpath,
            self._rule_inline_small_call,
            self._rule_assert_non_null,
            self._rule_unroll_small_loop,
            self._rule_circuit_breaker,
            self._rule_batch_operations,
            self._rule_prefetch_sequences,
            self._rule_rate_limit,
            self._rule_sanitize_html_output,
            self._rule_idempotency,
            self._rule_feature_flag_guard,
            self._rule_requires_ensures,
            self._rule_audit_sensitive_ops,
            self._rule_suggest_logging_context,
        ]

    # -------------------------
    # High-level APIs
    # -------------------------
    def analyze_source(self, source: str, filename: Optional[str] = None, max_suggestions: int = 12) -> List[Suggestion]:
        cache_key = f"{filename or '<anon>'}:{hash(source)}"
        cached = self.cache.get(cache_key)
        if cached:
            logging.debug("cache hit for analyze_source")
            return cached

        suggestions: List[Suggestion] = []
        for rule in self.rules:
            try:
                suggestions.extend(rule(source, filename))
            except Exception:
                logging.exception("rule failed")

        # Score boosting by memory
        for s in suggestions:
            mem_boost = 0.05 * min(5, self.memory.pattern_count(s.macro_name))
            s.score = max(0.0, min(1.0, s.score + mem_boost))

        # Deduplicate
        seen = set()
        unique: List[Suggestion] = []
        for s in sorted(suggestions, key=lambda x: -x.score):
            key = (s.macro_name, tuple(s.args), s.snippet or "", s.location)
            if key in seen:
                continue
            seen.add(key)
            unique.append(s)
            if len(unique) >= max_suggestions:
                break

        self.cache.set(cache_key, unique)
        return unique

    def preview_application(self, source: str, suggestion: Suggestion, filename: Optional[str] = None) -> Tuple[bool, str, Optional[List[Dict[str, Any]]]]:
        mo = _try_import_macro_overlay()
        if mo is None:
            return False, source, None

        # Insert macro invocation near suggestion.location or snippet
        macro_invocation = f"@{suggestion.macro_name} " + (", ".join(suggestion.args) if suggestion.args else "") + ";\n"

        transformed_source = source
        if suggestion.location:
            start, _end = suggestion.location
            # insert before start
            transformed_source = source[:start] + macro_invocation + source[start:]
        elif suggestion.snippet:
            idx = source.find(suggestion.snippet)
            if idx != -1:
                transformed_source = source[:idx] + macro_invocation + source[idx:]
            else:
                transformed_source = macro_invocation + source
        else:
            transformed_source = macro_invocation + source

        try:
            apply_fn = getattr(mo, "applyMacrosWithDiagnostics", None) or getattr(mo, "applyMacros", None)
            if apply_fn is None:
                return False, transformed_source, None
            res = apply_fn(transformed_source, mo.createFullRegistry() if hasattr(mo, "createFullRegistry") else mo.createDefaultRegistry(), {"filename": filename})
            # support coroutine
            if hasattr(res, "__await__"):
                import asyncio
                res = asyncio.get_event_loop().run_until_complete(res)
            if isinstance(res, dict) and "result" in res:
                transformed = res["result"]["transformed"]
                diagnostics = res.get("diagnostics", [])
                return True, transformed, diagnostics
            if isinstance(res, tuple) and len(res) >= 1:
                return True, res[0], None
            return False, transformed_source, None
        except Exception as e:
            logging.exception("apply failed")
            return False, transformed_source, [{"type": "error", "message": f"apply failed: {e}"}]

    def apply_suggestion_to_file(self, filepath: str, suggestion: Suggestion, inplace: bool = False, backup: bool = True) -> Tuple[bool, str]:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                src = f.read()
        except Exception as e:
            return False, f"read failed: {e}"

        ok, transformed, diagnostics = self.preview_application(src, suggestion, filename=filepath)
        if not ok:
            return False, "preview/apply failed (macro_overlay unavailable or error)"

        out_path = filepath if inplace else (filepath + ".ai.ix")
        if inplace and backup:
            bak = filepath + ".bak"
            shutil.copy2(filepath, bak)

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(transformed)
        except Exception as e:
            return False, f"write failed: {e}"

        self.memory.record_accepted(suggestion, filename=filepath)
        return True, out_path

    def generate_patch(self, original: str, transformed: str, filename: str) -> str:
        a_lines = original.splitlines(keepends=True)
        b_lines = transformed.splitlines(keepends=True)
        return "".join(difflib.unified_diff(a_lines, b_lines, fromfile=filename, tofile=filename + ".ai.ix", lineterm=""))

    # -------------------------
    # Batch helpers
    # -------------------------
    def batch_analyze(self, directory: str, pattern: str = ".ix", max_per_file: int = 5, workers: int = 4) -> Dict[str, List[Suggestion]]:
        results: Dict[str, List[Suggestion]] = {}
        files = []
        for root, _, filenames in os.walk(directory):
            for fn in filenames:
                if fn.endswith(pattern):
                    files.append(os.path.join(root, fn))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(self._analyze_file, f, max_per_file): f for f in files}
            for fut in concurrent.futures.as_completed(futures):
                f = futures[fut]
                try:
                    results[f] = fut.result()
                except Exception as e:
                    logging.exception("batch analyze failed for %s", f)
                    results[f] = []
        return results

    def _analyze_file(self, path: str, max_suggestions: int) -> List[Suggestion]:
        try:
            src = open(path, "r", encoding="utf-8").read()
        except Exception:
            return []
        return self.analyze_source(src, filename=path, max_suggestions=max_suggestions)

    def batch_apply_top(self, directory: str, suggestion_index: int = 0, inplace: bool = False, workers: int = 4) -> Dict[str, Tuple[bool, str]]:
        results: Dict[str, Tuple[bool, str]] = {}
        files = []
        for root, _, filenames in os.walk(directory):
            for fn in filenames:
                if fn.endswith(".ix"):
                    files.append(os.path.join(root, fn))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {}
            for f in files:
                futures[ex.submit(self._apply_top_for_file, f, suggestion_index, inplace)] = f
            for fut in concurrent.futures.as_completed(futures):
                f = futures[fut]
                try:
                    results[f] = fut.result()
                except Exception as e:
                    logging.exception("batch apply failed for %s", f)
                    results[f] = (False, str(e))
        return results

    def _apply_top_for_file(self, path: str, idx: int, inplace: bool) -> Tuple[bool, str]:
        try:
            src = open(path, "r", encoding="utf-8").read()
        except Exception as e:
            return False, f"read failed: {e}"
        suggestions = self.analyze_source(src, filename=path, max_suggestions=16)
        if not suggestions or idx >= len(suggestions):
            return False, "no suggestion"
        ok, out = self.apply_suggestion_to_file(path, suggestions[idx], inplace=inplace)
        return ok, out

    # -------------------------
    # Rules (heuristics)
    # -------------------------
    def _rule_sql_injection_risk(self, source: str, filename: Optional[str]) -> List[Suggestion]:
        """
        Detect string concatenation into SQL-like patterns and suggest parameterization or assert.
        """
        suggestions: List[Suggestion] = []
        # look for common concatenation patterns building SQL
        for m in re.finditer(r'(["\'].*?(SELECT|INSERT|UPDATE|DELETE).*?["\']\s*\+\s*[A-Za-z_][\w]*)', source, re.I | re.S):
            snippet = self._extract_line(source, m.start())
            suggestions.append(Suggestion(macro_name="assert", args=[f"sql_params_safe()"], reason="possible unparameterized SQL (string concat)", score=0.95, snippet=snippet, location=(m.start(), m.end())))
            self.memory.record_pattern("sql_injection")
        # also detect f-strings / interpolations like "...{id}..."
        for m in re.finditer(r'["\'].*\{[^\}]+\}.*["\']', source):
            line = self._extract_line(source, m.start())
            if re.search(r'\b(select|insert|update|delete)\b', line, re.I):
                suggestions.append(Suggestion(macro_name="assert", args=["params_are_safe()"], reason="string-interpolated SQL detected", score=0.9, snippet=line, location=(m.start(), m.end())))
                self.memory.record_pattern("sql_injection")
        return suggestions

    def _rule_transactional_db_writes(self, source: str, filename: Optional[str]) -> List[Suggestion]:
        """
        Detect DB write calls and suggest transactional wrap (wraptry/quarantine).
        """
        suggestions: List[Suggestion] = []
        for m in re.finditer(r"\b(db\.insert|db\.update|db\.save|execute\s*\(.*\bINSERT\b|execute\s*\(.*\bUPDATE\b)", source, re.I):
            snippet = self._extract_line(source, m.start())
            suggestions.append(Suggestion(macro_name="wraptry", args=[snippet.strip().rstrip(';')], reason="DB write detected; consider transactional wrap", score=0.9, snippet=snippet, location=(m.start(), m.end())))
            self.memory.record_pattern("transactional_db")
        return suggestions

    def _rule_wraptry_network_calls(self, source: str, filename: Optional[str]) -> List[Suggestion]:
        suggestions: List[Suggestion] = []
        for m in re.finditer(r"\b(net\.request|fetchData|fetch\(|http\.get|axios\.|request\()", source):
            snippet = self._extract_line(source, m.start())
            suggestions.append(Suggestion(macro_name="wraptry", args=[snippet.strip().rstrip(';')], reason="network or unstable call", score=0.88, snippet=snippet, location=(m.start(), m.end())))
            suggestions.append(Suggestion(macro_name="async", args=[snippet.strip().rstrip(';')], reason="run network call async", score=0.6, snippet=snippet, location=(m.start(), m.end())))
            self.memory.record_pattern("wraptry")
        return suggestions

    def _rule_retry_backoff_network_calls(self, source: str, filename: Optional[str]) -> List[Suggestion]:
        """
        Suggest retry/backoff wrappers for network calls.
        """
        suggestions: List[Suggestion] = []
        for m in re.finditer(r"\b(fetchData|net\.request|http\.get|request\()", source):
            snippet = self._extract_line(source, m.start())
            # propose wraptry with backoff hint argument (handled by macro overlay if supported)
            suggestions.append(Suggestion(macro_name="wraptry", args=[snippet.strip().rstrip(';')], reason="network call: consider retry/backoff", score=0.88, snippet=snippet, location=(m.start(), m.end())))
            self.memory.record_pattern("retry_backoff")
        return suggestions

    def _rule_async_for_background_tasks(self, source: str, filename: Optional[str]) -> List[Suggestion]:
        suggestions: List[Suggestion] = []
        for m in re.finditer(r"\b(backgroundTask|schedule|setTimeout|spawn|fork)\b", source):
            snippet = self._extract_line(source, m.start())
            suggestions.append(Suggestion(macro_name="async", args=[snippet.strip().rstrip(';')], reason="background/task scheduling detected", score=0.75, snippet=snippet, location=(m.start(), m.end())))
            self.memory.record_pattern("async")
        return suggestions

    def _rule_memoize_pure_functions(self, source: str, filename: Optional[str]) -> List[Suggestion]:
        suggestions: List[Suggestion] = []
        for m in re.finditer(r"\bfunc\s+([A-Za-z_][\w]*)\s*\(([^)]*)\)\s*{", source):
            name = m.group(1)
            start = m.end()
            body = self._capture_block(source, start - 1)
            if body is None:
                continue
            body_text = body.strip()
            if re.search(r"\bprint\b|\bdb\.|net\.|system\.|malloc|alloc\b", body_text):
                continue
            if len(body_text) < 300 and re.search(r"[+\-*/%]", body_text):
                snippet = source[m.start(): m.end() + min(80, len(body_text))]
                suggestions.append(Suggestion(macro_name="memoize", args=[name], reason="small pure/compute function", score=0.72, snippet=snippet, location=(m.start(), m.end())))
                self.memory.record_pattern("memoize")
        return suggestions

    def _rule_suggest_cache_ttl(self, source: str, filename: Optional[str]) -> List[Suggestion]:
        suggestions: List[Suggestion] = []
        # suggest caching for repeated computations or network responses
        for m in re.finditer(r'\b(expensiveCompute|compute|fetchData|net\.request)\b', source):
            snippet = self._extract_line(source, m.start())
            suggestions.append(Suggestion(macro_name="memoize", args=[snippet.strip().split("(")[0]], reason="expensive operation; consider cache/memoize", score=0.5, snippet=snippet, location=(m.start(), m.end())))
        return suggestions

    def _rule_lazy_injection_candidates(self, source: str, filename: Optional[str]) -> List[Suggestion]:
        """
        Suggest lazy injection for heavy dependencies detected (e.g. net.api, db.conn) if used infrequently.
        Heuristic: if dependency appears only once or inside uncommon branch.
        """
        suggestions: List[Suggestion] = []
        deps = {}
        for m in re.finditer(r"\b([a-zA-Z_][\w]*(?:\.[a-zA-Z_][\w]*)+)\b", source):
            token = m.group(1)
            if token.count(".") >= 1:
                deps.setdefault(token, 0)
                deps[token] += 1
        for dep, cnt in deps.items():
            if cnt <= 1 and (dep.startswith("net.") or dep.startswith("db.") or "api" in dep):
                snippet = dep
                suggestions.append(Suggestion(macro_name="inject", args=[dep], reason="candidate for lazy injection (infrequent use)", score=0.68, snippet=snippet))
                suggestions.append(Suggestion(macro_name="inject_as", args=[dep, dep.replace(".", "_")], reason="inject with alias", score=0.65, snippet=snippet))
                self.memory.record_pattern("lazy_inject")
        return suggestions

    def _rule_debug_for_public_funcs(self, source: str, filename: Optional[str]) -> List[Suggestion]:
        suggestions: List[Suggestion] = []
        for m in re.finditer(r"\bfunc\s+((?:handle|on|process|do|run)_[A-Za-z_][\w]*)\s*\(", source):
            name = m.group(1)
            suggestions.append(Suggestion(macro_name="debug", args=[name], reason="instrument public/handler function", score=0.56, snippet=self._extract_line(source, m.start()), location=(m.start(), m.end())))
            self.memory.record_pattern("debug")
        return suggestions

    def _rule_defer_for_resource_cleanup(self, source: str, filename: Optional[str]) -> List[Suggestion]:
        suggestions: List[Suggestion] = []
        for m in re.finditer(r"\b([A-Za-z_][\w]*\.(?:close|shutdown)|close\()", source):
            snippet = self._extract_line(source, m.start())
            suggestions.append(Suggestion(macro_name="defer", args=[snippet.strip().rstrip(';')], reason="resource cleanup detected; consider defer", score=0.62, snippet=snippet, location=(m.start(), m.end())))
            self.memory.record_pattern("defer")
        return suggestions

    def _rule_ffi_export_for_prefix_funcs(self, source: str, filename: Optional[str]) -> List[Suggestion]:
        suggestions: List[Suggestion] = []
        if re.search(r"\bextern\b|\bffi\b", source):
            for fm in re.finditer(r"\bfunc\s+([A-Za-z_][\w]*)\s*\(", source):
                name = fm.group(1)
                suggestions.append(Suggestion(macro_name="ffi", args=[name], reason="extern/ffi references present; expose function", score=0.52, snippet=self._extract_line(source, fm.start()), location=(fm.start(), fm.end())))
                self.memory.record_pattern("ffi")
        return suggestions

    def _rule_profile_hotpath(self, source: str, filename: Optional[str]) -> List[Suggestion]:
        suggestions: List[Suggestion] = []
        # Find heavy loops or nested loops heuristically
        for m in re.finditer(r"\bfor\b.*\{|\bwhile\b.*\{", source):
            # basic distance heuristic: if loop body length > 80 chars suggest profile
            start = m.end()
            body = self._capture_block(source, start - 1)
            if body and len(body) > 80:
                snippet = self._extract_line(source, m.start())
                suggestions.append(Suggestion(macro_name="profile", args=[snippet.strip().rstrip(';')], reason="long loop; consider profiling", score=0.45, snippet=snippet, location=(m.start(), m.end())))
                self.memory.record_pattern("profile")
        return suggestions

    def _rule_inline_small_call(self, source: str, filename: Optional[str]) -> List[Suggestion]:
        suggestions: List[Suggestion] = []
        # Suggest @inline for very small functions or callsites often used
        for m in re.finditer(r"\b([A-Za-z_][\w]*)\s*\(", source):
            name = m.group(1)
            # small heuristic: function name like add, mul, cmp
            if name in ("add", "mul", "cmp", "sum", "min", "max"):
                snippet = self._extract_line(source, m.start())
                suggestions.append(Suggestion(macro_name="inline", args=[name], reason="small utility; consider inline", score=0.3, snippet=snippet, location=(m.start(), m.end())))
        return suggestions

    def _rule_assert_non_null(self, source: str, filename: Optional[str]) -> List[Suggestion]:
        suggestions: List[Suggestion] = []
        # Find dereferences or uses of variables that may be null; heuristic: var.property or var.method()
        for m in re.finditer(r"\b([A-Za-z_][\w]*)\.[A-Za-z_][\w]*\b", source):
            var = m.group(1)
            snippet = self._extract_line(source, m.start())
            suggestions.append(Suggestion(macro_name="assert", args=[f"{var} != null"], reason="possible null deref; add assert", score=0.25, snippet=snippet, location=(m.start(), m.end())))
        return suggestions

    def _rule_unroll_small_loop(self, source: str, filename: Optional[str]) -> List[Suggestion]:
        suggestions: List[Suggestion] = []
        # detect loops with small numeric bounds e.g. for (i = 0; i < 4; i++)
        for m in re.finditer(r"for\s*\([^;]+;\s*([A-Za-z0-9_]+)\s*<\s*([0-9]+)\s*;[^)]+\)\s*\{", source):
            bound = int(m.group(2))
            if 0 < bound <= 8:
                snippet = self._extract_line(source, m.start())
                suggestions.append(Suggestion(macro_name="unroll", args=[str(bound)], reason=f"small fixed loop ({bound}); consider unrolling", score=0.35, snippet=snippet, location=(m.start(), m.end())))
        return suggestions

    def _rule_circuit_breaker(self, source: str, filename: Optional[str]) -> List[Suggestion]:
        suggestions: List[Suggestion] = []
        # detect repeated network calls in short span (heuristic)
        calls = list(re.finditer(r"\b(net\.request|fetchData|fetch\()", source))
        if len(calls) >= 3:
            # suggest circuit wrapper for the repeated call cluster
            for c in calls:
                snippet = self._extract_line(source, c.start())
                suggestions.append(Suggestion(macro_name="wraptry", args=[snippet.strip().rstrip(';')], reason="repeated network calls; consider circuit-breaker", score=0.65, snippet=snippet, location=(c.start(), c.end())))
            self.memory.record_pattern("circuit_breaker")
        return suggestions

    def _rule_batch_operations(self, source: str, filename: Optional[str]) -> List[Suggestion]:
        suggestions: List[Suggestion] = []
        # detect loops that call network/db per item
        for m in re.finditer(r"for\s*\([^)]*\)\s*\{(.*?)\}", source, re.S):
            body = m.group(1)
            if re.search(r"\b(net\.request|db\.insert|db\.update|fetchData|http\.get)\b", body):
                snippet = self._extract_line(source, m.start())
                suggestions.append(Suggestion(macro_name="batch", args=[snippet.strip().rstrip(';')], reason="per-item external calls in loop; consider batching", score=0.66, snippet=snippet, location=(m.start(), m.end())))
                self.memory.record_pattern("batch")
        return suggestions

    def _rule_prefetch_sequences(self, source: str, filename: Optional[str]) -> List[Suggestion]:
        suggestions: List[Suggestion] = []
        # detect sequence of fetches in same block
        for m in re.finditer(r"(fetchData\([^)]*\);(?:\s*fetchData\([^)]*\);)+)", source):
            snippet = self._extract_line(source, m.start())
            suggestions.append(Suggestion(macro_name="prefetch", args=[snippet.strip().rstrip(';')], reason="consecutive fetches detected; consider prefetch", score=0.55, snippet=snippet, location=(m.start(), m.end())))
            self.memory.record_pattern("prefetch")
        return suggestions

    def _rule_rate_limit(self, source: str, filename: Optional[str]) -> List[Suggestion]:
        suggestions: List[Suggestion] = []
        # look for high-frequency external calls w/in loops
        for m in re.finditer(r"for\s*\([^)]*\)\s*\{(.*?)\}", source, re.S):
            body = m.group(1)
            if len(re.findall(r"\b(net\.request|fetchData|http\.get)\b", body)) >= 1:
                snippet = self._extract_line(source, m.start())
                suggestions.append(Suggestion(macro_name="ratelimit", args=["10,1"], reason="external calls in loop; consider rate-limiting", score=0.6, snippet=snippet, location=(m.start(), m.end())))
                self.memory.record_pattern("ratelimit")
        return suggestions

    def _rule_sanitize_html_output(self, source: str, filename: Optional[str]) -> List[Suggestion]:
        suggestions: List[Suggestion] = []
        # detect direct printing of HTML with user input concatenation
        for m in re.finditer(r'print\s*:\s*["\'].*(<[^>]+>).*["\']\s*\+\s*[A-Za-z_][\w]*', source, re.S):
            snippet = self._extract_line(source, m.start())
            suggestions.append(Suggestion(macro_name="sanitize", args=["user_input"], reason="HTML output with concatenated user input; sanitize", score=0.9, snippet=snippet, location=(m.start(), m.end())))
            self.memory.record_pattern("sanitize")
        return suggestions

    def _rule_idempotency(self, source: str, filename: Optional[str]) -> List[Suggestion]:
        suggestions: List[Suggestion] = []
        # detect create_* or post_* functions that do DB writes
        for m in re.finditer(r"\bfunc\s+(create|post|submit)_[A-Za-z_][\w]*\s*\(", source):
            name = m.group(0)
            snippet = self._extract_line(source, m.start())
            suggestions.append(Suggestion(macro_name="idempotent", args=[], reason="create/submit endpoint detected; consider idempotency", score=0.6, snippet=snippet, location=(m.start(), m.end())))
            self.memory.record_pattern("idempotent")
        return suggestions

    def _rule_feature_flag_guard(self, source: str, filename: Optional[str]) -> List[Suggestion]:
        suggestions: List[Suggestion] = []
        # detect new/experimental-looking functions by name
        for m in re.finditer(r"\bfunc\s+(exp|beta|experimental)_[A-Za-z_][\w]*\s*\(", source):
            snippet = self._extract_line(source, m.start())
            suggestions.append(Suggestion(macro_name="feature", args=["flag_name"], reason="experimental function; wrap in feature flag", score=0.4, snippet=snippet, location=(m.start(), m.end())))
        return suggestions

    def _rule_requires_ensures(self, source: str, filename: Optional[str]) -> List[Suggestion]:
        suggestions: List[Suggestion] = []
        # detect functions lacking obvious precondition checks
        for m in re.finditer(r"\bfunc\s+([A-Za-z_][\w]*)\s*\(([^)]*)\)\s*{", source):
            params = m.group(2).strip()
            if params and not re.search(r"\bassert\b", source[m.end(): m.end()+200]):
                snippet = self._extract_line(source, m.start())
                suggestions.append(Suggestion(macro_name="assert", args=[f"{params.split(',')[0].strip()} != null"], reason="function missing precondition checks; add assert/requires", score=0.5, snippet=snippet, location=(m.start(), m.end())))
        return suggestions

    def _rule_audit_sensitive_ops(self, source: str, filename: Optional[str]) -> List[Suggestion]:
        suggestions: List[Suggestion] = []
        # detect likely sensitive operations: delete, transfer, withdraw
        for m in re.finditer(r"\b(delete|transfer|withdraw|close_account)\b", source, re.I):
            snippet = self._extract_line(source, m.start())
            suggestions.append(Suggestion(macro_name="audit", args=[snippet.strip().rstrip(';')], reason="sensitive operation detected; add audit", score=0.9, snippet=snippet, location=(m.start(), m.end())))
            self.memory.record_pattern("audit")
        return suggestions

    def _rule_suggest_logging_context(self, source: str, filename: Optional[str]) -> List[Suggestion]:
        suggestions: List[Suggestion] = []
        # add context logging where errors or alerts are present
        for m in re.finditer(r'\b(alert|fail|error|panic)\b', source):
            snippet = self._extract_line(source, m.start())
            suggestions.append(Suggestion(macro_name="debug", args=[snippet.strip().rstrip(';')], reason="error path detected; consider debug logging", score=0.4, snippet=snippet, location=(m.start(), m.end())))
        return suggestions

    # -------------------------
    # Utility helpers
    # -------------------------
    def _extract_line(self, source: str, idx: int, pad: int = 140) -> str:
        start = source.rfind("\n", 0, idx) + 1
        end = source.find("\n", idx)
        if end == -1:
            end = len(source)
        line = source[start:end].strip()
        if len(line) > pad:
            return line[:pad] + "..."
        return line

    def _capture_block(self, source: str, brace_open_index: int) -> Optional[str]:
        i = source.find("{", brace_open_index)
        if i == -1:
            return None
        depth = 0
        j = i
        while j < len(source):
            ch = source[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return source[i + 1: j]
            elif ch == '"' or ch == "'":
                quote = ch
                j += 1
                while j < len(source) and not (source[j] == quote and source[j - 1] != "\\"):
                    j += 1
            j += 1
        return None

# -------------------------
# HTTP API server (simple)
# -------------------------
class SuggestHandler(http.server.BaseHTTPRequestHandler):
    assistant: Optional[AIAssistant] = None

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/suggest":
            qs = urllib.parse.parse_qs(parsed.query)
            file = qs.get("file", [None])[0]
            maxn = int(qs.get("max", [8])[0])
            if not file or not os.path.exists(file):
                self._send_json({"error": "file missing or not found"}, 400)
                return
            src = open(file, "r", encoding="utf-8").read()
            suggestions = self.assistant.analyze_source(src, filename=file, max_suggestions=maxn)
            self._send_json({"suggestions": [s.to_dict() for s in suggestions]})
        else:
            self._send_json({"error": "unknown endpoint"}, 404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/apply":
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
                src = open(file, "r", encoding="utf-8").read()
                suggestions = self.assistant.analyze_source(src, filename=file, max_suggestions=32)
                if not suggestions or idx >= len(suggestions):
                    self._send_json({"error": "invalid suggestion index"}, 400)
                    return
                ok, out = self.assistant.apply_suggestion_to_file(file, suggestions[idx], inplace=inplace)
                self._send_json({"ok": ok, "out": out})
            except Exception as e:
                logging.exception("HTTP apply failed")
                self._send_json({"error": str(e)}, 500)
        else:
            self._send_json({"error": "unknown endpoint"}, 404)

    def _send_json(self, obj, status=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

def serve_api(assistant: AIAssistant, host: str = "127.0.0.1", port: int = 8787):
    SuggestHandler.assistant = assistant
    with socketserver.TCPServer((host, port), SuggestHandler) as httpd:
        logging.info("AI engine API serving on %s:%d", host, port)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logging.info("AI engine API stopped")

# -------------------------
# Plugins, Undo, Patch, Report tools
# -------------------------
class PluginManager:
    """Simple plugin loader for custom heuristic rules/plugins."""

    def __init__(self, plugins_dir: Optional[str] = None):
        self.plugins_dir = plugins_dir or PLUGINS_DIR
        os.makedirs(self.plugins_dir, exist_ok=True)
        self.loaded: Dict[str, Any] = {}

    def discover(self) -> List[str]:
        names = []
        for f in os.listdir(self.plugins_dir):
            if f.endswith(".py"):
                names.append(os.path.splitext(f)[0])
        return names

    def load(self, name: str) -> Tuple[bool, str]:
        path = os.path.join(self.plugins_dir, f"{name}.py")
        if not os.path.exists(path):
            return False, f"plugin {name} not found in {self.plugins_dir}"
        try:
            spec = importlib.util.spec_from_file_location(f"ciams.plugins.{name}", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            if hasattr(mod, "register") and callable(mod.register):
                mod.register(self)  # plugin may register hooks
            self.loaded[name] = mod
            return True, f"loaded {name}"
        except Exception as e:
            logging.exception("plugin load failed")
            return False, str(e)

    def unload(self, name: str) -> Tuple[bool, str]:
        if name not in self.loaded:
            return False, "not loaded"
        mod = self.loaded.pop(name)
        try:
            if hasattr(mod, "unregister") and callable(mod.unregister):
                mod.unregister(self)
            return True, f"unloaded {name}"
        except Exception as e:
            logging.exception("plugin unload failed")
            return False, str(e)


class UndoManager:
    """Manage simple backups and restore (undo)."""

    def list_backups(self, filepath: str) -> List[str]:
        base = filepath
        dirp = os.path.dirname(base)
        name = os.path.basename(base)
        res = []
        for f in os.listdir(dirp or "."):
            if f.startswith(name) and (f.endswith(".bak") or f.endswith(".ai.ix.bak")):
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
            return False, str(e)


class PatchManager:
    """Generate and apply patches used by this AI engine."""

    def create_patch(self, original_path: str, transformed_text: str) -> str:
        with open(original_path, "r", encoding="utf-8") as f:
            original = f.read()
        patch_text = "".join(difflib.unified_diff(original.splitlines(keepends=True),
                                                  transformed_text.splitlines(keepends=True),
                                                  fromfile=original_path,
                                                  tofile=original_path + ".ai.ix",
                                                  lineterm=""))
        patch_path = original_path + ".ai.patch"
        with open(patch_path, "w", encoding="utf-8") as pf:
            pf.write(patch_text)
        return patch_path

    def apply_patch(self, original_path: str, patch_path: str) -> Tuple[bool, str]:
        # simple apply: write transformed (.ai.ix) by reading patch and parsing hunks via difflib
        try:
            with open(patch_path, "r", encoding="utf-8") as pf:
                patch_text = pf.read()
            # compute transformed by applying diff (use difflib.restore)
            patched_lines = list(difflib.restore(patch_text.splitlines(keepends=True), 2))
            out_path = original_path + ".ai.ix"
            with open(out_path, "w", encoding="utf-8") as of:
                of.writelines(patched_lines)
            return True, out_path
        except Exception as e:
            logging.exception("apply_patch failed")
            return False, str(e)


class ReportGenerator:
    """Generate a simple HTML report for suggestions."""

    def generate_html(self, suggestions_map: Dict[str, List[Suggestion]], out_path: str):
        parts = ["<html><head><meta charset='utf-8'><title>CIAMS AI Report</title></head><body>"]
        parts.append("<h1>CIAMS AI Suggestions Report</h1>")
        parts.append(f"<p>Generated: {time.asctime()}</p>")
        for fname, suggs in suggestions_map.items():
            parts.append(f"<h2>{fname} ({len(suggs)})</h2><ul>")
            for s in suggs:
                parts.append("<li><b>{}</b> {} (score={:.2f})<br/><code>{}</code></li>".format(
                    s.macro_name, s.reason, s.score, (s.snippet or "").replace("<", "&lt;").replace(">", "&gt;")
                ))
            parts.append("</ul>")
        parts.append("</body></html>")
        html_text = "\n".join(parts)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_text)
        return out_path

    def serve_report(self, html_path: str, host: str = "127.0.0.1", port: int = 8001):
        prev = os.getcwd()
        os.chdir(os.path.dirname(html_path) or ".")
        handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer((host, port), handler) as httpd:
            logging.info("serving report on http://%s:%d/%s", host, port, os.path.basename(html_path))
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                pass
        os.chdir(prev)


# -------------------------
# CLI & Interactive helpers
# -------------------------
def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_file(path: str, data: str, backup: bool = True) -> None:
    if backup and os.path.exists(path):
        shutil.copy2(path, path + ".bak")
    with open(path, "w", encoding="utf-8") as f:
        f.write(data)


def print_suggestions(suggestions: List[Suggestion]) -> None:
    if not suggestions:
        print("[ai] no suggestions")
        return
    for i, s in enumerate(suggestions):
        loc = f"loc={s.location}" if s.location else ""
        print(f"[{i}] {s.macro_name} {s.args} (score={s.score:.2f}) - {s.reason} {loc}")
        if s.snippet:
            print("     snippet:", s.snippet)


def interactive_apply(assistant: AIAssistant, path: str):
    src = _read_file(path)
    suggestions = assistant.analyze_source(src, filename=path, max_suggestions=32)
    if not suggestions:
        print("No suggestions")
        return
    print_suggestions(suggestions)
    while True:
        choice = input("Select index to apply (q to quit, a to apply all top-N): ").strip()
        if choice == "q":
            break
        if choice == "a":
            try:
                n = int(input("Apply top N suggestions, N = ").strip())
            except Exception:
                print("invalid number")
                continue
            for i in range(min(n, len(suggestions))):
                ok, out = assistant.apply_suggestion_to_file(path, suggestions[i], inplace=False)
                print(f"Applied [{i}] -> {out} (ok={ok})")
            break
        try:
            idx = int(choice)
            if idx < 0 or idx >= len(suggestions):
                print("invalid index")
                continue
            ok, out = assistant.apply_suggestion_to_file(path, suggestions[idx], inplace=False)
            print(f"Applied -> {out} (ok={ok})")
        except ValueError:
            print("invalid input")


# -------------------------
# Unit tests
# -------------------------
def run_unit_tests(verbose: bool = True) -> bool:
    ai = AIAssistant()
    sample = """-- Demo
net.request("https://api.example.com/data");
db.conn.query("select *");
func fib(n) { if n <= 1 { n } else { fib(n-1) + fib(n-2) } };
func handle_event(ev) { print: "handled"; };
file_handle = open("log.txt");
file_handle.close();
for (i = 0; i < 4; i++) { doWork(i); }
sql = "SELECT * FROM users WHERE id=" + id;
html_out = "<div>" + user_input + "</div>";
"""
    suggestions = ai.analyze_source(sample, filename="<test>", max_suggestions=40)
    ok = len(suggestions) >= 6
    if verbose:
        print("Unit test suggestions count:", len(suggestions))
        print_suggestions(suggestions)
    # Test preview with macro overlay (if available) must not throw
    mo = _try_import_macro_overlay()
    if mo and suggestions:
        try:
            s = suggestions[0]
            ok2, transformed, diagnostics = ai.preview_application(sample, s, filename="<test>")
            if verbose:
                print("Preview transformed length:", len(transformed))
        except Exception as e:
            print("preview failed:", e)
            ok = False
    return ok


# -------------------------
# CLI entry point (enhanced)
# -------------------------
def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(prog="ciams/ai-engine.py", description="CIAMS AI assistant (extended)")
    sub = parser.add_subparsers(dest="cmd")

    p_suggest = sub.add_parser("suggest", help="Suggest macros for a file")
    p_suggest.add_argument("file", help="source file path")
    p_suggest.add_argument("--max", type=int, default=12)

    p_preview = sub.add_parser("preview", help="Preview applying a suggestion")
    p_preview.add_argument("file")
    p_preview.add_argument("--suggestion", type=int, default=0)

    p_apply = sub.add_parser("apply", help="Apply a suggestion to a file (writes new file by default)")
    p_apply.add_argument("file")
    p_apply.add_argument("--suggestion", type=int, default=0)
    p_apply.add_argument("--inplace", action="store_true")

    p_batch = sub.add_parser("batch-suggest", help="Run suggestions over a directory")
    p_batch.add_argument("dir")
    p_batch.add_argument("--max", type=int, default=5)
    p_batch.add_argument("--workers", type=int, default=4)

    p_batch_apply = sub.add_parser("batch-apply", help="Apply top suggestion across dir")
    p_batch_apply.add_argument("dir")
    p_batch_apply.add_argument("--index", type=int, default=0)
    p_batch_apply.add_argument("--inplace", action="store_true")
    p_batch_apply.add_argument("--workers", type=int, default=4)

    p_interactive = sub.add_parser("interactive", help="Interactive apply session for a file")
    p_interactive.add_argument("file")

    p_serve = sub.add_parser("serve", help="Start HTTP suggestion API")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8787)

    p_report = sub.add_parser("report", help="Generate HTML report for files")
    p_report.add_argument("out", help="output html file")
    p_report.add_argument("files", nargs="+", help="files to analyze")

    p_serve_report = sub.add_parser("serve-report", help="Serve a generated HTML report")
    p_serve_report.add_argument("html", help="path to html file")
    p_serve_report.add_argument("--host", default="127.0.0.1")
    p_serve_report.add_argument("--port", type=int, default=8001)

    p_export_mem = sub.add_parser("export-memory", help="Export memory to JSON file")
    p_export_mem.add_argument("out", help="path to export")

    p_import_mem = sub.add_parser("import-memory", help="Import memory from JSON file")
    p_import_mem.add_argument("src", help="path to import")
    p_import_mem.add_argument("--merge", action="store_true", help="merge into existing memory")

    p_undo = sub.add_parser("undo", help="List or restore backups")
    p_undo.add_argument("file", help="target file")
    p_undo.add_argument("--restore", help="backup path to restore")

    p_patch_apply = sub.add_parser("apply-patch", help="Apply a patch file produced by this tool")
    p_patch_apply.add_argument("original", help="original file path")
    p_patch_apply.add_argument("patch", help="patch file path")

    p_plugins = sub.add_parser("plugins", help="plugin operations")
    p_plugins.add_argument("action", choices=["list", "load", "unload"])
    p_plugins.add_argument("--name", help="plugin name (for load/unload)")

    p_test = sub.add_parser("test", help="Run unit tests")

    args = parser.parse_args(argv)
    assistant = AIAssistant()
    plugin_mgr = PluginManager()
    undo_mgr = UndoManager()
    patch_mgr = PatchManager()
    report_gen = ReportGenerator()

    if args.cmd == "suggest":
        src = _read_file(args.file)
        suggestions = assistant.analyze_source(src, filename=args.file, max_suggestions=args.max)
        print_suggestions(suggestions)
        return 0

    if args.cmd == "preview":
        src = _read_file(args.file)
        suggestions = assistant.analyze_source(src, filename=args.file, max_suggestions=32)
        if not suggestions:
            print("no suggestions")
            return 0
        idx = args.suggestion
        if idx < 0 or idx >= len(suggestions):
            print("invalid suggestion index")
            return 2
        ok, transformed, diagnostics = assistant.preview_application(src, suggestions[idx], filename=args.file)
        print("==== transformed preview ====")
        print(transformed)
        if diagnostics:
            print("diagnostics:", diagnostics)
        return 0

    if args.cmd == "apply":
        src = _read_file(args.file)
        suggestions = assistant.analyze_source(src, filename=args.file, max_suggestions=32)
        if not suggestions:
            print("no suggestions")
            return 2
        idx = args.suggestion
        if idx < 0 or idx >= len(suggestions):
            print("invalid suggestion index")
            return 2
        ok, out = assistant.apply_suggestion_to_file(args.file, suggestions[idx], inplace=args.inplace)
        if ok:
            print(f"applied -> {out}")
            return 0
        else:
            print(f"apply failed: {out}")
            return 2

    if args.cmd == "batch-suggest":
        results = assistant.batch_analyze(args.dir, max_per_file=args.max, workers=args.workers)
        for f, sug in results.items():
            print(f"File: {f} -> {len(sug)} suggestions")
        return 0

    if args.cmd == "batch-apply":
        results = assistant.batch_apply_top(args.dir, suggestion_index=args.index, inplace=args.inplace, workers=args.workers)
        for f, (ok, msg) in results.items():
            print(f"{f}: ok={ok} msg={msg}")
        return 0

    if args.cmd == "interactive":
        interactive_apply(assistant, args.file)
        return 0

    if args.cmd == "serve":
        print(f"Serving AI suggestion API on {args.host}:{args.port}")
        serve_thread = threading.Thread(target=serve_api, args=(assistant, args.host, args.port), daemon=True)
        serve_thread.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Server stopping...")
        return 0

    if args.cmd == "report":
        suggestions_map = {}
        for f in args.files:
            try:
                src = _read_file(f)
            except Exception as e:
                print(f"read failed: {e}")
                continue
            suggestions_map[f] = assistant.analyze_source(src, filename=f, max_suggestions=32)
        out = report_gen.generate_html(suggestions_map, args.out)
        print(f"report written to {out}")
        return 0

    if args.cmd == "serve-report":
        if not os.path.exists(args.html):
            print("html file not found")
            return 2
        print(f"Serving report {args.html} on {args.host}:{args.port}")
        report_gen.serve_report(args.html, host=args.host, port=args.port)
        return 0

    if args.cmd == "export-memory":
        data = assistant.memory.export()
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"memory exported to {args.out}")
        return 0

    if args.cmd == "import-memory":
        try:
            with open(args.src, "r", encoding="utf-8") as f:
                data = json.load(f)
            assistant.memory.import_data(data, merge=args.merge)
            print("memory imported")
            return 0
        except Exception as e:
            print("import failed:", e)
            return 2

    if args.cmd == "undo":
        if args.restore:
            ok, msg = undo_mgr.restore(args.restore, None)
            if ok:
                print(f"restored -> {msg}")
                return 0
            else:
                print("restore failed:", msg)
                return 2
        else:
            b = undo_mgr.list_backups(args.file)
            if not b:
                print("no backups found")
            else:
                for bi in b:
                    print(bi)
            return 0

    if args.cmd == "apply-patch":
        ok, out = patch_mgr.apply_patch(args.original, args.patch)
        if ok:
            print(f"applied patch -> {out}")
            return 0
        else:
            print("apply failed:", out)
            return 2

    if args.cmd == "plugins":
        action = args.action
        name = args.name
        if action == "list":
            print("available plugins:", plugin_mgr.discover())
            print("loaded plugins:", plugin_mgr.loaded.keys())
            return 0
        if action == "load":
            if not name:
                print("specify --name")
                return 2
            ok, msg = plugin_mgr.load(name)
            print(msg)
            return 0 if ok else 2
        if action == "unload":
            if not name:
                print("specify --name")
                return 2
            ok, msg = plugin_mgr.unload(name)
            print(msg)
            return 0 if ok else 2

    if args.cmd == "test":
        ok = run_unit_tests(verbose=True)
        print("All tests passed" if ok else "Some tests failed")
        return 0 if ok else 2

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

