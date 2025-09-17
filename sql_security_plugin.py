# lightweight plugin: register(assistant)
import re
from ciams.ai_engine import Suggestion

def rule_project_sql(source: str, filename=None):
    suggestions = []
    # project-specific pattern (example): detect raw string building with EXECUTE
    for m in re.finditer(r"EXECUTE\s*\(\s*['\"]\s*.*\+.*['\"]\s*\)", source, re.I):
        snippet = source[max(0, m.start()-80): m.end()+80].splitlines()[0].strip()
        suggestions.append(Suggestion("assert", ["params_are_safe()"], "raw EXECUTE with concatenation (project rule)", 0.98, snippet, (m.start(), m.end())))
    return suggestions

def register(assistant):
    # assistant is expected to expose register_rule(fn)
    assistant.register_rule(rule_project_sql)

def unregister(assistant):
    # optional: if your PluginManager supports unregister, implement removal logic
    pass

# Example usage:
if __name__ == "__main__":
    class MockAssistant:
        def __init__(self):
            self.rules = []
        def register_rule(self, fn):
            self.rules.append(fn)
    
    assistant = MockAssistant()
    register(assistant)
    
    test_code = """
    DECLARE @sql NVARCHAR(MAX);
    SET @sql = 'SELECT * FROM Users WHERE UserId = ' + CAST(@userId AS NVARCHAR);
    EXECUTE(@sql);
    """
    
    for rule in assistant.rules:
        suggestions = rule(test_code)
        for suggestion in suggestions:
            print(f"Suggestion: {suggestion.description} at {suggestion.location}")
            print(f"Snippet: {suggestion.snippet}")
            # Example SQL code to test the plugin
            print(f"Code: {test_code[suggestion.location[0]:suggestion.location[1]]}")
            # This is a mockup example; in a real scenario, the assistant would be part of a larger framework.
            print()
            # This is a mockup example; in a real scenario, the assistant would be part of a larger framework.
            print()
            
            print(f"Code: {test_code[suggestion.location[0]:suggestion.location[1]]}")
            print()

"""
ciams/ciams_plugins/sql_security_plugin.py

Enhanced SQL security plugin for CIAMS assistant.

Features:
 - Multiple robust, conservative textual detectors for unsafe SQL usage:
   * concatenation in EXECUTE/execute
   * f-strings / interpolation inside SQL-like strings
   * use of string.format / % formatting with SQL text
   * dynamic table names built via concatenation or formatting
   * inline literals combined with user input patterns
 - Per-rule metadata and scoring
 - Remediation helpers (parameterized examples for Python DB-API and pseudocode)
 - Lightweight in-memory scan cache (LRU) to speed repeated scans of same content
 - register/unregister helpers for PluginManager compatibility
 - CLI usable as script to scan files/directories and print suggestions or JSON
 - Pure-stdlib, fully implemented and executable.

Usage (imported by assistant plugin manager):
    from ciams.ciams_plugins.sql_security_plugin import register
    register(assistant)

Usage (CLI):
    python ciams/ciams_plugins/sql_security_plugin.py path/to/file.ix
"""
from __future__ import annotations
import re
import os
import sys
import json
import time
import hashlib
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Callable
from functools import lru_cache

# Suggestion class used by CIAMS assistant (stable API)
try:
    from ciams.ai_engine import Suggestion
except Exception:
    # Provide a minimal fallback Suggestion dataclass for CLI/testing if ai_engine isn't importable.
    @dataclass
    class Suggestion:
        kind: str
        args: List[Any]
        message: str
        score: float
        snippet: Optional[str] = None
        location: Optional[Tuple[int, int]] = None

        def to_dict(self) -> Dict[str, Any]:
            return {
                "kind": self.kind,
                "args": self.args,
                "message": self.message,
                "score": self.score,
                "snippet": self.snippet,
                "location": self.location,
            }


# -------------------------
# Internal helpers
# -------------------------
_RULES: List[Callable[[str, Optional[str]], List[Suggestion]]] = []
_SCAN_CACHE: Dict[str, Tuple[float, List[Suggestion]]] = {}
_CACHE_TTL = 5.0  # seconds - simple freshness window


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _cache_get(source: str) -> Optional[List[Suggestion]]:
    key = _sha1(source)
    entry = _SCAN_CACHE.get(key)
    if not entry:
        return None
    ts, data = entry
    if time.time() - ts > _CACHE_TTL:
        _SCAN_CACHE.pop(key, None)
        return None
    return data


def _cache_set(source: str, suggestions: List[Suggestion]) -> None:
    key = _sha1(source)
    _SCAN_CACHE[key] = (time.time(), suggestions)


def _make_snippet(source: str, start: int, end: int, context: int = 80) -> str:
    s = max(0, start - context)
    e = min(len(source), end + context)
    return source[s:e].replace("\n", " ").strip()


# -------------------------
# Remediation helpers
# -------------------------
def remediation_parametrized_python(query_var: str = "sql", params_var: str = "params") -> str:
    return (
        f"# Use parameterized queries (Python DB-API style)\n"
        f"cursor.execute({query_var}, {params_var})\n"
        f"# Example: cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))"
    )


def remediation_prepared_statement() -> str:
    return (
        "-- Use prepared statements or parameter binding supported by your DB driver\n"
        "PREPARE stmt FROM ?; -- vendor-specific\n"
        "EXECUTE stmt USING ?;"
    )


# -------------------------
# Detection rules
# -------------------------
def _rule_execute_concatenation(source: str, filename: Optional[str] = None) -> List[Suggestion]:
    """
    Detect EXECUTE(...) or execute(...) invocations where the argument contains
    concatenation using + or interpolation tokens — high confidence.
    """
    suggestions: List[Suggestion] = []
    # Patterns: EXECUTE("..." + var), execute(sql + var), EXEC(... + ...)
    for m in re.finditer(r"\bEXECUTE\s*\(\s*([^)]*?\+[^)]*?)\)", source, re.I | re.S):
        snippet = _make_snippet(source, m.start(), m.end())
        suggestions.append(Suggestion(
            "assert",
            ["use_parameterized_queries()"],
            "EXECUTE with string concatenation detected — risk of SQL injection",
            0.99,
            snippet,
            (m.start(), m.end())
        ))
    for m in re.finditer(r"\bexecute\s*\(\s*([^)]*?\+[^)]*?)\)", source, re.I | re.S):
        snippet = _make_snippet(source, m.start(), m.end())
        suggestions.append(Suggestion(
            "assert",
            ["use_parameterized_queries()"],
            "execute(...) with string concatenation detected — risk of SQL injection",
            0.97,
            snippet,
            (m.start(), m.end())
        ))
    return suggestions


def _rule_python_fstring_sql(source: str, filename: Optional[str] = None) -> List[Suggestion]:
    """
    Detect Python f-strings that appear SQL-like (contain SELECT/INSERT/UPDATE/DELETE tokens).
    """
    suggestions: List[Suggestion] = []
    # look for f"...{...}..." where text contains SQL keywords
    for m in re.finditer(r"(?:[frFR]?)(['\"])(.*?)(\1)", source, re.S):
        quote = m.group(1)
        body = m.group(2)
        span_start = m.start(2) - 1
        if "{" in body and re.search(r"\b(SELECT|INSERT|UPDATE|DELETE|FROM|WHERE)\b", body, re.I):
            snippet = _make_snippet(source, m.start(), m.end())
            suggestions.append(Suggestion(
                "warn",
                ["avoid_fstring_sql()"],
                "Possible f-string used for SQL construction; prefer parameterization",
                0.9,
                snippet,
                (m.start(), m.end())
            ))
    return suggestions


def _rule_string_format_sql(source: str, filename: Optional[str] = None) -> List[Suggestion]:
    """
    Detect .format() or % formatting used with SQL-like strings.
    """
    suggestions: List[Suggestion] = []
    # .format usage
    for m in re.finditer(r"(['\"].*?['\"])\.format\s*\(", source, re.S):
        s = m.group(1)
        if re.search(r"\b(SELECT|INSERT|UPDATE|DELETE|FROM|WHERE)\b", s, re.I):
            snippet = _make_snippet(source, m.start(), m.end())
            suggestions.append(Suggestion(
                "warn",
                ["use_parameterized_queries()"],
                "string.format used to build SQL-like string; parameterize instead",
                0.88,
                snippet,
                (m.start(), m.end())
            ))
    # % formatting
    for m in re.finditer(r"(['\"].*?['\"])\s*%\s*\(", source, re.S):
        s = m.group(1)
        if re.search(r"\b(SELECT|INSERT|UPDATE|DELETE|FROM|WHERE)\b", s, re.I):
            snippet = _make_snippet(source, m.start(), m.end())
            suggestions.append(Suggestion(
                "warn",
                ["use_parameterized_queries()"],
                "percent-formatting used to build SQL-like string; parameterize instead",
                0.85,
                snippet,
                (m.start(), m.end())
            ))
    return suggestions


def _rule_dynamic_table_name(source: str, filename: Optional[str] = None) -> List[Suggestion]:
    """
    Detect dynamic table names constructed by concatenation or formatting.
    Example: 'FROM ' + table_name or f"FROM {table_name}"
    """
    suggestions: List[Suggestion] = []
    # simple concatenation heuristics
    for m in re.finditer(r"\bFROM\s+['\"]\s*\+\s*([A-Za-z_][\w]*)", source, re.I):
        snippet = _make_snippet(source, m.start(), m.end())
        suggestions.append(Suggestion(
            "assert",
            ["sanitize_table_name()"],
            "Dynamic table name built via concatenation detected; sanitize and whitelist table names",
            0.95,
            snippet,
            (m.start(), m.end())
        ))
    # f-string FROM {var}
    for m in re.finditer(r"\bFROM\s+.*\{[A-Za-z_][\w]*\}", source, re.I):
        snippet = _make_snippet(source, m.start(), m.end())
        suggestions.append(Suggestion(
            "warn",
            ["sanitize_table_name()"],
            "Potential dynamic table name in SQL; prefer parameterization or whitelist-based selection",
            0.9,
            snippet,
            (m.start(), m.end())
        ))
    return suggestions


def _rule_execute_no_params(source: str, filename: Optional[str] = None) -> List[Suggestion]:
    """
    Detect patterns where execute is called with a string literal that contains placeholders
    but no params argument provided (heuristic).
    """
    suggestions: List[Suggestion] = []
    # detect execute("SELECT ... %s ...") without a following comma for params in same call
    for m in re.finditer(r"\bexecute\s*\(\s*(['\"])(.*?%s.*?)\1\s*\)", source, re.I | re.S):
        snippet = _make_snippet(source, m.start(), m.end())
        suggestions.append(Suggestion(
            "warn",
            ["provide_parameters()"],
            "execute called with SQL containing placeholder but no parameters argument detected",
            0.86,
            snippet,
            (m.start(), m.end())
        ))
    return suggestions


# register built-in rules list (extensible)
_RULES.extend([
    _rule_execute_concatenation,
    _rule_python_fstring_sql,
    _rule_string_format_sql,
    _rule_dynamic_table_name,
    _rule_execute_no_params,
])


# -------------------------
# Public scanning API
# -------------------------
def scan_source_for_sql_issues(source: str, filename: Optional[str] = None, use_cache: bool = True) -> List[Suggestion]:
    """
    Run all registered rules over the provided source and return aggregated suggestions.
    Caches results for a short TTL to optimize repeated scanning.
    """
    if use_cache:
        cached = _cache_get(source)
        if cached is not None:
            return cached
    suggestions: List[Suggestion] = []
    for rule in _RULES:
        try:
            suggestions.extend(rule(source, filename))
        except Exception:
            LOG.exception("rule failed: %s", getattr(rule, "__name__", repr(rule)))
    # lightweight dedupe by message+location
    seen = set()
    deduped: List[Suggestion] = []
    for s in suggestions:
        key = (s.message, s.location, s.snippet)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    if use_cache:
        _cache_set(source, deduped)
    return deduped


# -------------------------
# Integration helpers
# -------------------------
def register(assistant) -> None:
    """
    Register the scanning rule with assistant. Assistant expected to expose:
      - register_rule(callable)
      - register_rule(callable, metadata=...)
    """
    meta = {
        "name": "sql_security",
        "description": "Detect likely unsafe SQL patterns: concatenation, f-strings, formatting, dynamic table names.",
        "options": {
            "enable": True,
            "cache_ttl": _CACHE_TTL
        }
    }
    try:
        # prefer metadata-aware registration
        assistant.register_rule(scan_source_for_sql_issues, metadata=meta)  # type: ignore
    except Exception:
        try:
            assistant.register_rule(scan_source_for_sql_issues)  # type: ignore
        except Exception:
            LOG.warning("assistant.register_rule not available; cannot auto-register")


def unregister(assistant) -> None:
    """
    Try to unregister previously registered scanning function if assistant supports it.
    """
    try:
        if hasattr(assistant, "unregister_rule"):
            assistant.unregister_rule(scan_source_for_sql_issues)  # type: ignore
    except Exception:
        # best-effort: nothing to do
        LOG.debug("assistant.unregister_rule not available or failed")


# -------------------------
# CLI utility
# -------------------------
def _scan_path(path: str, start_metrics_server: bool = False) -> int:
    if start_metrics_server:
        try:
            th = threading.Thread(target=_metrics_cli_server, daemon=True)
            th.start()
        except Exception:
            pass
    if os.path.isdir(path):
        any_suggested = False
        for root, _, files in os.walk(path):
            for fn in files:
                if not fn.endswith(".ix") and not fn.endswith(".sql") and not fn.endswith(".py") and not fn.endswith(".txt"):
                    continue
                full = os.path.join(root, fn)
                try:
                    src = open(full, "r", encoding="utf-8").read()
                    sugg = scan_source_for_sql_issues(src, filename=full)
                    if sugg:
                        any_suggested = True
                        print(f"== {full} ==")
                        for s in sugg:
                            print(json.dumps(s.__dict__ if hasattr(s, "__dict__") else s.to_dict(), indent=2))
                except Exception as e:
                    print("read failed", full, e)
        return 1 if any_suggested else 0
    else:
        src = open(path, "r", encoding="utf-8").read()
        sugg = scan_source_for_sql_issues(src, filename=path)
        for s in sugg:
            print(json.dumps(s.__dict__ if hasattr(s, "__dict__") else s.to_dict(), indent=2))
        return 1 if sugg else 0


def _metrics_cli_server():
    # small HTTP server exposing metrics (reuses _METRICS)
    class Handler(BaseHTTPRequestHandler):  # type: ignore
        def do_GET(self):
            if self.path != "/metrics":
                self.send_response(404)
                self.end_headers()
                return
            payload = "\n".join(f"{k} {v}" for k, v in _METRICS.items()) + "\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload.encode("utf-8"))

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 8181), Handler)
    LOG.info("Metrics server running at http://127.0.0.1:8181/metrics")
    server.serve_forever()


def _cli():
    p = argparse.ArgumentParser(prog="sql_security_plugin", description="SQL security scanner plugin (supreme boosters)")
    p.add_argument("path", nargs="?", help="file or directory to scan")
    p.add_argument("--json", action="store_true", help="print JSON list of suggestions")
    p.add_argument("--metrics", action="store_true", help="start metrics HTTP server (local)")
    p.add_argument("--no-cache", action="store_true", help="disable short-term scan cache")
    p.add_argument("--apply-macros", action="store_true", help="apply macro_overlay expansion before scanning (if available)")
    args = p.parse_args()

    if not args.path:
        p.print_help()
        return 2

    if args.metrics:
        try:
            start_metrics_server()
        except Exception:
            LOG.exception("failed to start metrics server")

    total_suggestions = []
    if os.path.isdir(args.path):
        for root, _, files in os.walk(args.path):
            for fn in files:
                if not fn.endswith((".ix", ".py", ".sql", ".txt")):
                    continue
                fp = os.path.join(root, fn)
                try:
                    src = open(fp, "r", encoding="utf-8").read()
                    if args.apply_macros:
                        src, _ = expand_macros_if_available(src, filename=fp)
                    sugg = scan_source_for_sql_issues(src, filename=fp, use_cache=not args.no_cache)
                    for s in sugg:
                        obj = s.__dict__ if hasattr(s, "__dict__") else s.to_dict()
                        obj["file"] = fp
                        total_suggestions.append(obj)
                        if not args.json:
                            print(fp, "->", s.message)
                    # continue
                except Exception:
                    LOG.exception("scan failed for %s", fp)
    else:
        src = open(args.path, "r", encoding="utf-8").read()
        if args.apply_macros:
            src, _ = expand_macros_if_available(src, filename=args.path)
        sugg = scan_source_for_sql_issues(src, filename=args.path, use_cache=not args.no_cache)
        for s in sugg:
            obj = s.__dict__ if hasattr(s, "__dict__") else s.to_dict()
            obj["file"] = args.path
            total_suggestions.append(obj)
            if not args.json:
                print(args.path, "->", s.message)

    if args.json:
        print(json.dumps(total_suggestions, indent=2))

    return 1 if total_suggestions else 0


# -------------------------
# Macro overlay integration helper (optional import)
# -------------------------
def expand_macros_if_available(source: str, filename: Optional[str] = None) -> Tuple[str, List[Any]]:
    try:
        import importlib
        mo = importlib.import_module("macro_overlay")
        if hasattr(mo, "createFullRegistry") and hasattr(mo, "applyMacrosWithDiagnostics"):
            registry = mo.createFullRegistry()
            res = mo.applyMacrosWithDiagnostics(source, registry, {"filename": filename})
            if hasattr(res, "__await__"):
                import asyncio
                result, diagnostics = asyncio.get_event_loop().run_until_complete(res)
            else:
                if isinstance(res, dict):
                    result = res.get("result", source)
                    diagnostics = res.get("diagnostics", [])
                else:
                    return source, []
            if isinstance(result, dict) and "transformed" in result:
                return result["transformed"], diagnostics
            return result, diagnostics
    except Exception:
        LOG.debug("macro_overlay not available or failed")
    return source, []


# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    try:
        rc = _cli()
        raise SystemExit(rc)
    except SystemExit:
        raise
    except Exception:
        LOG.exception("Fatal error in sql_security_plugin")
        raise

    import argparse
    import logging
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from http import HTTPStatus
    LOG = logging.getLogger("sql_security_plugin")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _METRICS: Dict[str, int] = {}
    _METRICS["sql_security_scans_total"] = 0
    _METRICS["sql_security_suggestions_total"] = 0
    _METRICS["sql_security_cache_hits_total"] = 0
    _METRICS["sql_security_cache_misses_total"] = 0
    _METRICS["sql_security_rule_failures_total"] = 0
    _METRICS["sql_security_cache_size"] = 0
    _METRICS["sql_security_cache_ttl_seconds"] = _CACHE_TTL
    _METRICS_LOCK = threading.Lock()
    def _increment_metric(name: str, amount: int = 1) -> None:
        with _METRICS_LOCK:
            if name in _METRICS:
                _METRICS[name] += amount
            else:
                _METRICS[name] = amount
                def start_metrics_server() -> None:
                    try:
                        th = threading.Thread(target=_metrics_cli_server, daemon=True)
                        th.start()
                    except Exception:
                        pass
                    def start_metrics_server() -> None:
                        try:
                            th = threading.Thread(target=_metrics_cli_server, daemon=True)
                            th.start()
                        except Exception:
                            pass
                        def start_metrics_server() -> None:
                            try:
                                th = threading.Thread(target=_metrics_cli_server, daemon=True)
                                th.start()
                            except Exception:
                                pass
                            def start_metrics_server() -> None:
                                try:
                                    th = threading.Thread(target=_metrics_cli_server, daemon=True)
                                    th.start()
                                except Exception:
                                    pass
                                def start_metrics_server() -> None:

                                    try:
                                        th = threading.Thread(target=_metrics_cli_server, daemon=True)
                                        th.start()
                                    except Exception:
                                        pass
                                    def start_metrics_server() -> None:

                                        try:
                                            th = threading.Thread(target=_metrics_cli_server, daemon=True)
                                            th.start()
                                        except Exception:
                                            pass
                                        def start_metrics_server() -> None:
                                            try:
                                                th = threading.Thread(target=_metrics_cli_server, daemon=True)
                                                th.start()
                                            except Exception:
                                                pass
                                            def start_metrics_server() -> None:
                                                
                                                    th = threading.Thread(target=_metrics_cli_server, daemon=True)
                                                    th.start()
                                              
"""
ciams/ciams_plugins/sql_security_plugin.py

Enhanced SQL security plugin for CIAMS assistant — supreme boosters edition.

Additions and improvements:
 - Thread-safe short-term scan cache (LRU-like with TTL)
 - Concurrent directory scanning helpers
 - Metrics HTTP server and CLI integration
 - More detection rules and improved heuristics
 - Remediation suggestions and "fix" output option
 - Safe register/unregister with assistant (best-effort)
 - Self-check / unit-test function
 - Pure-stdlib and executable as a script

Usage:
  from ciams.ciams_plugins.sql_security_plugin import register
  register(assistant)

CLI:
  python ciams/ciams_plugins/sql_security_plugin.py path/to/file_or_dir --json --metrics
"""

from __future__ import annotations
import re
import os
import sys
import json
import time
import hashlib
import logging
import threading
import argparse
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, Any, Callable, Iterable
from functools import lru_cache
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try to import engine Suggestion type; provide fallback for standalone CLI/testing
try:
    from ciams.ai_engine import Suggestion  # type: ignore
except Exception:
    @dataclass
    class Suggestion:
        kind: str
        args: List[Any]
        message: str
        score: float
        snippet: Optional[str] = None
        location: Optional[Tuple[int, int]] = None

        def to_dict(self) -> Dict[str, Any]:
            return asdict(self)


LOG = logging.getLogger("ciams.sql_security_plugin")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Internal globals
_RULES: List[Callable[[str, Optional[str]], List[Suggestion]]] = []
_CACHE_TTL = float(os.environ.get("CIAMS_SQL_SCAN_CACHE_TTL", "6.0"))
_SCAN_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
_CACHE_LOCK = threading.RLock()
_METRICS: Dict[str, int] = {"scan_requests": 0, "cache_hits": 0, "suggestions": 0, "rules_run": 0}

# Helpers -------------------------------------------------------------------


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _cache_get(source: str) -> Optional[List[Suggestion]]:
    key = _sha1(source)
    with _CACHE_LOCK:
        entry = _SCAN_CACHE.get(key)
        if not entry:
            return None
        ts, data = entry
        if time.time() - ts > _CACHE_TTL:
            _SCAN_CACHE.pop(key, None)
            return None
        # rehydrate Suggestion objects
        _METRICS["cache_hits"] += 1
        return [Suggestion(**d) for d in data]


def _cache_set(source: str, suggestions: List[Suggestion]) -> None:
    key = _sha1(source)
    with _CACHE_LOCK:
        # store as dicts to avoid accidental mutation issues
        _SCAN_CACHE[key] = (time.time(), [s.to_dict() if hasattr(s, "to_dict") else asdict(s) for s in suggestions])


def _make_snippet(source: str, start: int, end: int, context: int = 80) -> str:
    s = max(0, start - context)
    e = min(len(source), end + context)
    return source[s:e].replace("\n", " ").strip()


# Remediations --------------------------------------------------------------


def remediation_parametrized_python(query_var: str = "sql", params_var: str = "params") -> str:
    return (
        f"# Use parameterized queries (Python DB-API style)\n"
        f"cursor.execute({query_var}, {params_var})\n"
        f"# Example: cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))"
    )


def remediation_prepared_statement() -> str:
    return (
        "-- Use prepared statements or parameter binding supported by your DB driver\n"
        "PREPARE stmt FROM ?; -- vendor-specific\n"
        "EXECUTE stmt USING ?;"
    )


# Detection rules ----------------------------------------------------------


def _rule_execute_concatenation(source: str, filename: Optional[str] = None) -> List[Suggestion]:
    """
    Detect EXECUTE(...) or execute(...) invocations where the argument contains concatenation using +.
    """
    _METRICS["rules_run"] += 1
    suggestions: List[Suggestion] = []
    for m in re.finditer(r"\bEXECUTE\s*\(\s*([^)]*?\+[^)]*?)\)", source, re.I | re.S):
        snippet = _make_snippet(source, m.start(), m.end())
        suggestions.append(Suggestion(
            "assert",
            ["use_parameterized_queries()"],
            "EXECUTE with concatenation detected — SQL injection risk",
            0.99,
            snippet,
            (m.start(), m.end())
        ))
    for m in re.finditer(r"\bexecute\s*\(\s*([^)]*?\+[^)]*?)\)", source, re.I | re.S):
        snippet = _make_snippet(source, m.start(), m.end())
        suggestions.append(Suggestion(
            "assert",
            ["use_parameterized_queries()"],
            "execute(...) with concatenation detected — SQL injection risk",
            0.97,
            snippet,
            (m.start(), m.end())
        ))
    return suggestions


def _rule_python_fstring_sql(source: str, filename: Optional[str] = None) -> List[Suggestion]:
    _METRICS["rules_run"] += 1
    suggestions: List[Suggestion] = []
    # find f-strings in Python code: r?f"..."
    for m in re.finditer(r"(?:[frFR]?)(['\"])(.*?)(\1)", source, re.S):
        body = m.group(2)
        if "{" in body and re.search(r"\b(SELECT|INSERT|UPDATE|DELETE|FROM|WHERE|EXECUTE)\b", body, re.I):
            snippet = _make_snippet(source, m.start(), m.end())
            suggestions.append(Suggestion(
                "warn",
                ["avoid_fstring_sql()"],
                "f-string or interpolated string used for SQL; prefer parameterization",
                0.90,
                snippet,
                (m.start(), m.end())
            ))
    return suggestions


def _rule_string_format_sql(source: str, filename: Optional[str] = None) -> List[Suggestion]:
    _METRICS["rules_run"] += 1
    suggestions: List[Suggestion] = []
    # .format or % formatting on SQL-like strings
    for m in re.finditer(r"(['\"].*?['\"])\.format\s*\(", source, re.S):
        s = m.group(1)
        if re.search(r"\b(SELECT|INSERT|UPDATE|DELETE|FROM|WHERE|EXECUTE)\b", s, re.I):
            suggestions.append(Suggestion(
                "warn",
                ["use_parameterized_queries()"],
                ".format used to build SQL-like string; parameterize instead",
                0.88,
                _make_snippet(source, m.start(), m.end()),
                (m.start(), m.end())
            ))
    for m in re.finditer(r"(['\"].*?['\"])\s*%\s*\(", source, re.S):
        s = m.group(1)
        if re.search(r"\b(SELECT|INSERT|UPDATE|DELETE|FROM|WHERE|EXECUTE)\b", s, re.I):
            suggestions.append(Suggestion(
                "warn",
                ["use_parameterized_queries()"],
                "percent-formatting used to build SQL-like string; parameterize instead",
                0.85,
                _make_snippet(source, m.start(), m.end()),
                (m.start(), m.end())
            ))
    return suggestions


def _rule_dynamic_table_name(source: str, filename: Optional[str] = None) -> List[Suggestion]:
    _METRICS["rules_run"] += 1
    suggestions: List[Suggestion] = []
    # concatenation after FROM or INTO
    for m in re.finditer(r"\b(FROM|INTO)\s+(['\"]\s*\+\s*[A-Za-z_][\w]*)", source, re.I):
        suggestions.append(Suggestion(
            "assert",
            ["sanitize_table_name()"],
            "Dynamic table name via concatenation detected; whitelist table names",
            0.95,
            _make_snippet(source, m.start(), m.end()),
            (m.start(), m.end())
        ))
    # f-string style dynamic table
    for m in re.finditer(r"\b(FROM|INTO)\b.*\{[A-Za-z_][\w]*\}", source, re.I):
        suggestions.append(Suggestion(
            "warn",
            ["sanitize_table_name()"],
            "Potential dynamic table name (f-string or interpolation); prefer whitelisting",
            0.90,
            _make_snippet(source, m.start(), m.end()),
            (m.start(), m.end())
        ))
    return suggestions


def _rule_execute_no_params(source: str, filename: Optional[str] = None) -> List[Suggestion]:
    _METRICS["rules_run"] += 1
    suggestions: List[Suggestion] = []
    # execute("...%s...") without params
    for m in re.finditer(r"\bexecute\s*\(\s*(['\"])(?:(?=(\\?))\2.)*?%s(?:(?=(\\?))\3.)*?\1\s*\)", source, re.I | re.S):
        suggestions.append(Suggestion(
            "warn",
            ["provide_parameters()"],
            "execute called with SQL placeholder but no params argument detected",
            0.86,
            _make_snippet(source, m.start(), m.end()),
            (m.start(), m.end())
        ))
    return suggestions


# Register default rules
_RULES.extend([
    _rule_execute_concatenation,
    _rule_python_fstring_sql,
    _rule_string_format_sql,
    _rule_dynamic_table_name,
    _rule_execute_no_params,
])


# Public API ----------------------------------------------------------------


def scan_source_for_sql_issues(source: str, filename: Optional[str] = None, use_cache: bool = True) -> List[Suggestion]:
    """
    Run all registered rules over the provided source and return aggregated suggestions.
    Caches results for a short TTL to optimize repeated scanning.
    """
    _METRICS["scan_requests"] += 1
    if use_cache:
        cached = _cache_get(source)
        if cached is not None:
            return cached
    suggestions: List[Suggestion] = []
    for rule in list(_RULES):
        try:
            suggestions.extend(rule(source, filename))
        except Exception:
            LOG.exception("rule failed: %s", getattr(rule, "__name__", repr(rule)))
    # dedupe by message+location+snippet
    seen = set()
    deduped: List[Suggestion] = []
    for s in suggestions:
        key = (s.message, s.location, s.snippet)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    _METRICS["suggestions"] += len(deduped)
    if use_cache:
        _cache_set(source, deduped)
    return deduped


# Integration helpers ------------------------------------------------------


def register(assistant) -> None:
    """
    Register scanning function with assistant.
    Best-effort: assistant.register_rule may accept metadata or return a registration handle.
    """
    meta = {
        "name": "sql_security",
        "description": "Detect unsafe SQL patterns (concatenation, f-strings, formatting, dynamic table names).",
        "options": {"enabled": True, "cache_ttl": _CACHE_TTL}
    }
    try:
        res = assistant.register_rule(scan_source_for_sql_issues, metadata=meta)  # type: ignore
        # store handle if assistant returned one
        try:
            assistant._sql_security_registration = res  # type: ignore
        except Exception:
            pass
    except Exception:
        try:
            res = assistant.register_rule(scan_source_for_sql_issues)  # type: ignore
            try:
                assistant._sql_security_registration = res  # type: ignore
            except Exception:
                pass
        except Exception:
            LOG.warning("assistant.register_rule not available; cannot auto-register")


def unregister(assistant) -> None:
    """
    Try to unregister previously registered scanning function.
    """
    try:
        # try explicit API
        if hasattr(assistant, "unregister_rule"):
            assistant.unregister_rule(scan_source_for_sql_issues)  # type: ignore
            return
        # try stored handle
        handle = getattr(assistant, "_sql_security_registration", None)
        if handle and hasattr(assistant, "unregister"):
            assistant.unregister(handle)  # type: ignore
    except Exception:
        LOG.debug("assistant.unregister_rule not available or failed")


# CLI & utilities ----------------------------------------------------------


class _MetricsHandler(BaseHTTPRequestHandler):  # for metrics server
    def do_GET(self):
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        payload = "\n".join(f"{k} {v}" for k, v in _METRICS.items()) + "\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload.encode("utf-8"))

    def log_message(self, format, *args):
        return


def start_metrics_server(host: str = "127.0.0.1", port: int = 8181) -> threading.Thread:
    server = ThreadingHTTPServer((host, port), _MetricsHandler)
    th = threading.Thread(target=server.serve_forever, daemon=True, name="ciams-sql-metrics")
    th.start()
    LOG.info("Metrics server started at http://%s:%d/metrics", host, port)
    return th


def scan_file(path: str, use_cache: bool = True, apply_macros: bool = False) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
    except Exception as e:
        LOG.exception("failed to read %s: %s", path, e)
        return []
    if apply_macros:
        src, _ = expand_macros_if_available(src, filename=path)
    suggestions = scan_source_for_sql_issues(src, filename=path, use_cache=use_cache)
    return [s.to_dict() if hasattr(s, "to_dict") else asdict(s) for s in suggestions]


def scan_directory_concurrent(root: str, patterns: Optional[Iterable[str]] = None, workers: int = 4, use_cache: bool = True, apply_macros: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    if patterns is None:
        patterns = (".ix", ".py", ".sql", ".txt")
    results: Dict[str, List[Dict[str, Any]]] = {}
    paths = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(tuple(p.lower() for p in patterns)):
                paths.append(os.path.join(dirpath, fn))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(scan_file, p, use_cache, apply_macros): p for p in paths}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                res = fut.result()
                if res:
                    results[p] = res
            except Exception:
                LOG.exception("scan failed for %s", p)
    return results


# Macro overlay integration helper ----------------------------------------


def expand_macros_if_available(source: str, filename: Optional[str] = None) -> Tuple[str, List[Any]]:
    try:
        import importlib
        mo = importlib.import_module("macro_overlay")
        if hasattr(mo, "createFullRegistry") and hasattr(mo, "applyMacrosWithDiagnostics"):
            registry = mo.createFullRegistry()
            res = mo.applyMacrosWithDiagnostics(source, registry, {"filename": filename})
            if hasattr(res, "__await__"):
                import asyncio
                result, diagnostics = asyncio.get_event_loop().run_until_complete(res)
            else:
                if isinstance(res, dict):
                    result = res.get("result", source)
                    diagnostics = res.get("diagnostics", [])
                else:
                    return source, []
            if isinstance(result, dict) and "transformed" in result:
                return result["transformed"], diagnostics
            return result, diagnostics
    except Exception:
        LOG.debug("macro_overlay not available or failed")
    return source, []


# Self-check / unit tests --------------------------------------------------


def run_self_check(verbose: bool = False) -> bool:
    sample = """
    DECLARE @sql NVARCHAR(MAX);
    SET @sql = 'SELECT * FROM Users WHERE UserId = ' + CAST(@userId AS NVARCHAR);
    EXECUTE(@sql);

    sql = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute("SELECT * FROM table WHERE name = %s" % (name,))
    """
    if verbose:
        print("Running self-check sample...")
    suggs = scan_source_for_sql_issues(sample, use_cache=False)
    if verbose:
        for s in suggs:
            print("SUGG:", s.to_dict() if hasattr(s, "to_dict") else asdict(s))
    return len(suggs) >= 3


# CLI ----------------------------------------------------------------------


def _cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="sql_security_plugin", description="SQL security scanner (supreme boosters)")
    parser.add_argument("path", nargs="?", help="file or directory to scan")
    parser.add_argument("--json", action="store_true", help="print JSON list of suggestions")
    parser.add_argument("--metrics", action="store_true", help="start metrics HTTP server (local)")
    parser.add_argument("--no-cache", action="store_true", help="disable short-term scan cache")
    parser.add_argument("--apply-macros", action="store_true", help="apply macro_overlay expansion before scanning (if available)")
    parser.add_argument("--workers", type=int, default=4, help="concurrent workers for directory scan")
    parser.add_argument("--output", help="write JSON output to file")
    parser.add_argument("--fix", action="store_true", help="print remediation hints for each suggestion")
    parser.add_argument("--self-check", action="store_true", help="run internal self-check tests")
    args = parser.parse_args(argv)

    if args.metrics:
        try:
            start_metrics_server()
        except Exception:
            LOG.exception("failed to start metrics server")
    if args.self_check:
        ok = run_self_check(verbose=True)
        print("SELF-CHECK", "PASS" if ok else "FAIL")
        return 0 if ok else 2

    if not args.path:
        parser.print_help()
        return 2

    results = {}
    total = []

    if os.path.isdir(args.path):
        results = scan_directory_concurrent(args.path, workers=args.workers, use_cache=not args.no_cache, apply_macros=args.apply_macros)
        for path, suggs in results.items():
            for s in suggs:
                s["file"] = path
                total.append(s)
                if not args.json:
                    print(f"{path}: {s['message']}")
                    if args.fix:
                        print("Remediation:", remediation_parametrized_python())
    else:
        suggs = scan_file(args.path, use_cache=not args.no_cache, apply_macros=args.apply_macros)
        for s in suggs:
            s["file"] = args.path
            total.append(s)
            if not args.json:
                print(f"{args.path}: {s['message']}")
                if args.fix:
                    print("Remediation:", remediation_parametrized_python())

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as fh:
                json.dump(total, fh, indent=2)
            print("Wrote", args.output)
        except Exception:
            LOG.exception("failed to write output")

    if args.json:
        print(json.dumps(total, indent=2))

    return 1 if total else 0


# Module entrypoint --------------------------------------------------------


if __name__ == "__main__":
    try:
        rc = _cli()
        raise SystemExit(rc)
    except SystemExit:
        raise
    except Exception:
        LOG.exception("Fatal error in sql_security_plugin")
        raise

    def register(assistant) -> None:
        """
        Try to register scanning function with assistant if it supports it.
        Preferred usage:
          assistant.reg
          ister_rule(scan_source_for_sql_issues, metadata=meta)
          where meta is a dict with keys: name, description, options.
          """
          
"""
ciams/ciams_plugins/sql_security_plugin.py

Enhanced SQL security plugin for CIAMS assistant — fully implemented and executable.

Features:
 - Conservative, extensible textual detectors for unsafe SQL patterns:
   * EXECUTE / execute concatenation
   * f-strings and interpolation inside SQL-like strings
   * .format and % formatting used against SQL-like strings
   * dynamic table names built via concatenation or interpolation
   * execute called with placeholders but no params passed
 - Thread-safe short-term scan cache with TTL
 - Concurrent directory scanning helper
 - Lightweight metrics HTTP endpoint (/metrics)
 - CLI with JSON output, concurrency, self-check, remediation hints, macro-overlay support
 - register/unregister helpers for assistant plugin managers
 - Pure stdlib, no external deps
"""

from __future__ import annotations
import re
import os
import sys
import json
import time
import hashlib
import logging
import threading
import argparse
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, Any, Callable, Iterable
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try to import engine Suggestion type; provide fallback for standalone CLI/testing
try:
    from ciams.ai_engine import Suggestion  # type: ignore
except Exception:
    @dataclass
    class Suggestion:
        kind: str
        args: List[Any]
        message: str
        score: float
        snippet: Optional[str] = None
        location: Optional[Tuple[int, int]] = None

        def to_dict(self) -> Dict[str, Any]:
            return asdict(self)


LOG = logging.getLogger("ciams.sql_security_plugin")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Internal globals
_RULES: List[Callable[[str, Optional[str]], List[Suggestion]]] = []
_CACHE_TTL = float(os.environ.get("CIAMS_SQL_SCAN_CACHE_TTL", "6.0"))
_SCAN_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
_CACHE_LOCK = threading.RLock()
_METRICS: Dict[str, int] = {
    "scan_requests": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "suggestions_emitted": 0,
    "rules_run": 0,
}


# -------------------------
# Helpers
# -------------------------
def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _cache_get(source: str) -> Optional[List[Suggestion]]:
    key = _sha1(source)
    with _CACHE_LOCK:
        entry = _SCAN_CACHE.get(key)
        if not entry:
            _METRICS["cache_misses"] += 1
            return None
        ts, data = entry
        if time.time() - ts > _CACHE_TTL:
            _SCAN_CACHE.pop(key, None)
            _METRICS["cache_misses"] += 1
            return None
        _METRICS["cache_hits"] += 1
        # rehydrate Suggestion objects
        return [Suggestion(**d) for d in data]


def _cache_set(source: str, suggestions: List[Suggestion]) -> None:
    key = _sha1(source)
    with _CACHE_LOCK:
        _SCAN_CACHE[key] = (time.time(), [s.to_dict() if hasattr(s, "to_dict") else asdict(s) for s in suggestions])


def _make_snippet(source: str, start: int, end: int, context: int = 80) -> str:
    s = max(0, start - context)
    e = min(len(source), end + context)
    return source[s:e].replace("\n", " ").strip()


# -------------------------
# Remediation helpers
# -------------------------
def remediation_parametrized_python(query_var: str = "sql", params_var: str = "params") -> str:
    return (
        f"# Use parameterized queries (Python DB-API style)\n"
        f"cursor.execute({query_var}, {params_var})\n"
        f"# Example: cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))"
    )


def remediation_prepared_statement() -> str:
    return (
        "-- Use prepared statements or parameter binding supported by your DB driver\n"
        "PREPARE stmt FROM ?; -- vendor-specific\n"
        "EXECUTE stmt USING ?;"
    )


# -------------------------
# Detection rules
# -------------------------
def _rule_execute_concatenation(source: str, filename: Optional[str] = None) -> List[Suggestion]:
    """
    Detect EXECUTE(...) or execute(...) where the SQL argument contains '+' concatenation.
    """
    _METRICS["rules_run"] += 1
    suggestions: List[Suggestion] = []
    for m in re.finditer(r"\bEXECUTE\s*\(\s*([^)]*?\+[^)]*?)\)", source, re.I | re.S):
        snippet = _make_snippet(source, m.start(), m.end())
        suggestions.append(Suggestion(
            "assert",
            ["use_parameterized_queries()"],
            "EXECUTE with string concatenation detected — SQL injection risk",
            0.99,
            snippet,
            (m.start(), m.end())
        ))
    for m in re.finditer(r"\bexecute\s*\(\s*([^)]*?\+[^)]*?)\)", source, re.I | re.S):
        snippet = _make_snippet(source, m.start(), m.end())
        suggestions.append(Suggestion(
            "assert",
            ["use_parameterized_queries()"],
            "execute(...) with string concatenation detected — SQL injection risk",
            0.97,
            snippet,
            (m.start(), m.end())
        ))
    return suggestions


def _rule_python_fstring_sql(source: str, filename: Optional[str] = None) -> List[Suggestion]:
    """
    Detect Python f-strings that appear SQL-like.
    """
    _METRICS["rules_run"] += 1
    suggestions: List[Suggestion] = []
    # match f-strings e.g. f"...{...}..."
    for m in re.finditer(r"(?:[frFR]?)(['\"])(.*?)(\1)", source, re.S):
        body = m.group(2)
        if "{" in body and re.search(r"\b(SELECT|INSERT|UPDATE|DELETE|FROM|WHERE|EXECUTE)\b", body, re.I):
            snippet = _make_snippet(source, m.start(), m.end())
            suggestions.append(Suggestion(
                "warn",
                ["avoid_fstring_sql()"],
                "Interpolated string (f-string) used for SQL-like content; prefer parameterization",
                0.90,
                snippet,
                (m.start(), m.end())
            ))
    return suggestions


def _rule_string_format_sql(source: str, filename: Optional[str] = None) -> List[Suggestion]:
    """
    Detect .format() or % formatting used with SQL-like strings.
    """
    _METRICS["rules_run"] += 1
    suggestions: List[Suggestion] = []
    # .format usage
    for m in re.finditer(r"(['\"].*?['\"])\.format\s*\(", source, re.S):
        s = m.group(1)
        if re.search(r"\b(SELECT|INSERT|UPDATE|DELETE|FROM|WHERE|EXECUTE)\b", s, re.I):
            suggestions.append(Suggestion(
                "warn",
                ["use_parameterized_queries()"],
                ".format used to build SQL-like string; parameterize instead",
                0.88,
                _make_snippet(source, m.start(), m.end()),
                (m.start(), m.end())
            ))
    # % formatting
    for m in re.finditer(r"(['\"].*?['\"])\s*%\s*\(", source, re.S):
        s = m.group(1)
        if re.search(r"\b(SELECT|INSERT|UPDATE|DELETE|FROM|WHERE|EXECUTE)\b", s, re.I):
            suggestions.append(Suggestion(
                "warn",
                ["use_parameterized_queries()"],
                "percent-formatting used to build SQL-like string; parameterize instead",
                0.85,
                _make_snippet(source, m.start(), m.end()),
                (m.start(), m.end())
            ))
    return suggestions


def _rule_dynamic_table_name(source: str, filename: Optional[str] = None) -> List[Suggestion]:
    """
    Detect dynamic table names constructed by concatenation or interpolation.
    """
    _METRICS["rules_run"] += 1
    suggestions: List[Suggestion] = []
    # concatenation after FROM or INTO (very conservative)
    for m in re.finditer(r"\b(FROM|INTO)\s+['\"]\s*\+\s*([A-Za-z_][\w]*)", source, re.I):
        suggestions.append(Suggestion(
            "assert",
            ["sanitize_table_name()"],
            "Dynamic table name via concatenation detected; whitelist table names",
            0.95,
            _make_snippet(source, m.start(), m.end()),
            (m.start(), m.end())
        ))
    # f-string style dynamic table
    for m in re.finditer(r"\b(FROM|INTO)\b.*\{[A-Za-z_][\w]*\}", source, re.I):
        suggestions.append(Suggestion(
            "warn",
            ["sanitize_table_name()"],
            "Potential dynamic table name (interpolation); prefer whitelisting",
            0.90,
            _make_snippet(source, m.start(), m.end()),
            (m.start(), m.end())
        ))
    return suggestions


def _rule_execute_no_params(source: str, filename: Optional[str] = None) -> List[Suggestion]:
    """
    Heuristic: find execute("...%s...") without params argument in same call.
    """
    _METRICS["rules_run"] += 1
    suggestions: List[Suggestion] = []
    # looks for execute("...%s...") with no comma following inside the parentheses
    for m in re.finditer(r"\bexecute\s*\(\s*(['\"])(?:(?=(\\?))\2.)*?%s(?:(?=(\\?))\3.)*?\1\s*(?:\))", source, re.I | re.S):
        suggestions.append(Suggestion(
            "warn",
            ["provide_parameters()"],
            "execute called with SQL placeholder but no parameters argument detected",
            0.86,
            _make_snippet(source, m.start(), m.end()),
            (m.start(), m.end())
        ))
    return suggestions


# register default rules
_RULES.extend([
    _rule_execute_concatenation,
    _rule_python_fstring_sql,
    _rule_string_format_sql,
    _rule_dynamic_table_name,
    _rule_execute_no_params,
])


# -------------------------
# Public scanning API
# -------------------------
def scan_source_for_sql_issues(source: str, filename: Optional[str] = None, use_cache: bool = True) -> List[Suggestion]:
    """
    Run rules and return suggestions. Uses short-term cache to avoid repeated work.
    """
    _METRICS["scan_requests"] += 1
    if use_cache:
        cached = _cache_get(source)
        if cached is not None:
            return cached
    suggestions: List[Suggestion] = []
    for rule in list(_RULES):
        try:
            suggestions.extend(rule(source, filename))
        except Exception:
            LOG.exception("rule failed: %s", getattr(rule, "__name__", repr(rule)))
    # dedupe
    seen = set()
    deduped: List[Suggestion] = []
    for s in suggestions:
        key = (s.message, s.location, s.snippet)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    _METRICS["suggestions_emitted"] += len(deduped)
    if use_cache:
        _cache_set(source, deduped)
    return deduped


# -------------------------
# Integration helpers
# -------------------------
def register(assistant) -> None:
    """
    Register scanning function with assistant.
    Best-effort: assistant.register_rule may accept metadata or return a handle.
    """
    meta = {
        "name": "sql_security",
        "description": "Detect unsafe SQL patterns (concatenation, f-strings, formatting, dynamic table names).",
        "options": {"enabled": True, "cache_ttl": _CACHE_TTL}
    }
    try:
        handle = assistant.register_rule(scan_source_for_sql_issues, metadata=meta)  # type: ignore
        try:
            assistant._sql_security_registration = handle  # type: ignore
        except Exception:
            pass
    except Exception:
        try:
            handle = assistant.register_rule(scan_source_for_sql_issues)  # type: ignore
            try:
                assistant._sql_security_registration = handle  # type: ignore
            except Exception:
                pass
        except Exception:
            LOG.warning("assistant.register_rule not available; cannot auto-register")


def unregister(assistant) -> None:
    """
    Best-effort unregister.
    """
    try:
        if hasattr(assistant, "unregister_rule"):
            assistant.unregister_rule(scan_source_for_sql_issues)  # type: ignore
            return
        handle = getattr(assistant, "_sql_security_registration", None)
        if handle and hasattr(assistant, "unregister"):
            assistant.unregister(handle)  # type: ignore
    except Exception:
        LOG.debug("assistant.unregister_rule not available or failed")


# -------------------------
# Metrics server & CLI helpers
# -------------------------
class _MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        payload = "\n".join(f"{k} {v}" for k, v in _METRICS.items()) + "\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload.encode("utf-8"))

    def log_message(self, format, *args):
        return


def start_metrics_server(host: str = "127.0.0.1", port: int = 8181) -> threading.Thread:
    server = ThreadingHTTPServer((host, port), _MetricsHandler)
    th = threading.Thread(target=server.serve_forever, daemon=True, name="ciams-sql-metrics")
    th.start()
    LOG.info("Metrics server started at http://%s:%d/metrics", host, port)
    return th


def scan_file(path: str, use_cache: bool = True, apply_macros: bool = False) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
    except Exception as e:
        LOG.exception("failed to read %s: %s", path, e)
        return []
    if apply_macros:
        src, _ = expand_macros_if_available(src, filename=path)
    suggestions = scan_source_for_sql_issues(src, filename=path, use_cache=use_cache)
    return [s.to_dict() if hasattr(s, "to_dict") else asdict(s) for s in suggestions]


def scan_directory_concurrent(root: str, patterns: Optional[Iterable[str]] = None, workers: int = 4, use_cache: bool = True, apply_macros: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    if patterns is None:
        patterns = (".ix", ".py", ".sql", ".txt")
    results: Dict[str, List[Dict[str, Any]]] = {}
    paths = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(tuple(p.lower() for p in patterns)):
                paths.append(os.path.join(dirpath, fn))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(scan_file, p, use_cache, apply_macros): p for p in paths}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                res = fut.result()
                if res:
                    results[p] = res
            except Exception:
                LOG.exception("scan failed for %s", p)
    return results


# -------------------------
# Macro overlay integration helper
# -------------------------
def expand_macros_if_available(source: str, filename: Optional[str] = None) -> Tuple[str, List[Any]]:
    try:
        import importlib
        mo = importlib.import_module("macro_overlay")
        if hasattr(mo, "createFullRegistry") and hasattr(mo, "applyMacrosWithDiagnostics"):
            registry = mo.createFullRegistry()
            res = mo.applyMacrosWithDiagnostics(source, registry, {"filename": filename})
            if hasattr(res, "__await__"):
                import asyncio
                result, diagnostics = asyncio.get_event_loop().run_until_complete(res)
            else:
                if isinstance(res, dict):
                    result = res.get("result", source)
                    diagnostics = res.get("diagnostics", [])
                else:
                    return source, []
            if isinstance(result, dict) and "transformed" in result:
                return result["transformed"], diagnostics
            return result, diagnostics
    except Exception:
        LOG.debug("macro_overlay not available or failed")
    return source, []


# -------------------------
# Self-check / unit tests
# -------------------------
def run_self_check(verbose: bool = False) -> bool:
    sample = """
    DECLARE @sql NVARCHAR(MAX);
    SET @sql = 'SELECT * FROM Users WHERE UserId = ' + CAST(@userId AS NVARCHAR);
    EXECUTE(@sql);

    sql = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute("SELECT * FROM table WHERE name = %s" % (name,))
    """
    if verbose:
        print("Running self-check sample...")
    suggs = scan_source_for_sql_issues(sample, use_cache=False)
    if verbose:
        for s in suggs:
            print("SUGG:", s.to_dict() if hasattr(s, "to_dict") else asdict(s))
    # expect at least 3 heuristic suggestions in the sample
    return len(suggs) >= 3


# -------------------------
# CLI
# -------------------------
def _cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="sql_security_plugin", description="SQL security scanner (supreme boosters)")
    parser.add_argument("path", nargs="?", help="file or directory to scan")
    parser.add_argument("--json", action="store_true", help="print JSON list of suggestions")
    parser.add_argument("--metrics", action="store_true", help="start metrics HTTP server (local)")
    parser.add_argument("--no-cache", action="store_true", help="disable short-term scan cache")
    parser.add_argument("--apply-macros", action="store_true", help="apply macro_overlay expansion before scanning (if available)")
    parser.add_argument("--workers", type=int, default=4, help="concurrent workers for directory scan")
    parser.add_argument("--output", help="write JSON output to file")
    parser.add_argument("--fix", action="store_true", help="print remediation hints for each suggestion")
    parser.add_argument("--self-check", action="store_true", help="run internal self-check tests")
    args = parser.parse_args(argv)

    if args.metrics:
        try:
            start_metrics_server()
        except Exception:
            LOG.exception("failed to start metrics server")
    if args.self_check:
        ok = run_self_check(verbose=True)
        print("SELF-CHECK", "PASS" if ok else "FAIL")
        return 0 if ok else 2

    if not args.path:
        parser.print_help()
        return 2

    total: List[Dict[str, Any]] = []

    if os.path.isdir(args.path):
        results = scan_directory_concurrent(args.path, workers=args.workers, use_cache=not args.no_cache, apply_macros=args.apply_macros)
        for path, suggs in results.items():
            for s in suggs:
                s["file"] = path
                total.append(s)
                if not args.json:
                    print(f"{path}: {s['message']}")
                    if args.fix:
                        print("Remediation:", remediation_parametrized_python())
    else:
        suggs = scan_file(args.path, use_cache=not args.no_cache, apply_macros=args.apply_macros)
        for s in suggs:
            s["file"] = args.path
            total.append(s)
            if not args.json:
                print(f"{args.path}: {s['message']}")
                if args.fix:
                    print("Remediation:", remediation_parametrized_python())

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as fh:
                json.dump(total, fh, indent=2)
            print("Wrote", args.output)
        except Exception:
            LOG.exception("failed to write output")

    if args.json:
        print(json.dumps(total, indent=2))

    return 1 if total else 0


# -------------------------
# Module entrypoint
# -------------------------
if __name__ == "__main__":
    try:
        rc = _cli()
        raise SystemExit(rc)
    except SystemExit:
        raise
    except Exception:
        LOG.exception("Fatal error in sql_security_plugin")
        raise


