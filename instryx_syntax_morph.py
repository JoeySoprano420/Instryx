"""
instryx_syntax_morph.py

Extended Instryx syntax morphing utilities.

This module builds on the earlier lightweight morph tool with many
additional fully-executable features, tools and optimizations:

- Additional deterministic morph passes:
  - convert label-colon data directives `name: expr;` -> `name = expr;`
  - collapse repeated blank lines
  - remove stray semicolons before braces `;}` -> `}`
  - ensure statement semicolons and one-per-line formatting
  - remove duplicate semicolons `;;`
  - remove trailing semicolon-only lines
  - fix unbalanced braces by appending missing closing braces (best-effort)
  - normalize comment spacing and fold sequences of single-line comments
- Source edit tracking (MorphEdit) and a simple source-map-like edits list
- File/directory processing, multithreaded batch apply
- Preview unified diffs using difflib
- Watcher (polling-based) for directories/files (no third-party deps)
- Small unit-test harness that runs a set of transformations and asserts expected outputs
- CLI with options: --inplace, --out, --dir, --watch, --diff, --test, --verbose
- Optional integration hook to call macro overlay expander if `macro_overlay` module is available
- Safe, regex-first approach; for complex cases recommend AST-based passes later

Usage:
  python instryx_syntax_morph.py file.ix --inplace
  python instryx_syntax_morph.py --dir src/ --diff
  python instryx_syntax_morph.py --test

Design notes:
- All passes are pure text transforms and return edit records for traceability.
- Implementation uses only Python stdlib.
"""

from __future__ import annotations
from dataclasses import dataclass
import re
from typing import List, Optional, Tuple, Callable, Dict
from pathlib import Path
import difflib
import concurrent.futures
import time
import sys
import os

# -------------------------
# Data classes
# -------------------------

@dataclass
class MorphEdit:
    start: int
    end: int
    original: str
    replacement: str
    reason: str

@dataclass
class MorphResult:
    transformed: str
    edits: List[MorphEdit]

# -------------------------
# Core morph class
# -------------------------

class SyntaxMorph:
    """
    Apply a sequence of safe, deterministic morph passes to Instryx source text.
    The class exposes many utility passes and convenience functions for file and directory operations.
    """

    def __init__(self, extra_passes: Optional[List[Callable[[str], Tuple[str, List[MorphEdit]]]]] = None):
        # base passes in order
        self.passes: List[Callable[[str], Tuple[str, List[MorphEdit]]]] = [
            self._normalize_line_endings,
            self._trim_trailing_spaces,
            self._collapse_blank_lines,
            self._normalize_comment_spacing,
            self._normalize_function_header_spacing,
            self._expand_print_directive,
            self._expand_do_array,
            self._convert_label_colon_to_assignment,
            self._normalize_if_then,
            self._normalize_while_not,
            self._remove_semicolon_before_brace,
            self._remove_duplicate_semicolons,
            self._remove_empty_semicolon_lines,
            self._ensure_statement_semicolons,
            self._normalize_quarantine_semicolon,
            self._fix_unbalanced_braces,
        ]
        if extra_passes:
            self.passes.extend(extra_passes)

    def morph(self, source: str) -> MorphResult:
        edits: List[MorphEdit] = []
        text = source
        for p in self.passes:
            text, pass_edits = p(text)
            edits.extend(pass_edits)
        return MorphResult(transformed=text, edits=edits)

    # -------------------------
    # Individual passes
    # -------------------------
    def _normalize_line_endings(self, text: str) -> Tuple[str, List[MorphEdit]]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        if normalized == text:
            return text, []
        return normalized, [MorphEdit(0, len(text), text, normalized, "normalize_line_endings")]

    def _trim_trailing_spaces(self, text: str) -> Tuple[str, List[MorphEdit]]:
        new = re.sub(r"[ \t]+(?=\n)", "", text)
        if new == text:
            return text, []
        return new, [MorphEdit(0, len(text), text, new, "trim_trailing_spaces")]

    def _collapse_blank_lines(self, text: str) -> Tuple[str, List[MorphEdit]]:
        # replace 3+ consecutive blank lines with 1 blank line
        new = re.sub(r"\n{3,}", "\n\n", text)
        if new == text:
            return text, []
        return new, [MorphEdit(0, len(text), text, new, "collapse_blank_lines")]

    def _normalize_comment_spacing(self, text: str) -> Tuple[str, List[MorphEdit]]:
        # normalize `--comment` to `-- comment` and collapse >2 single-line comment lines into a single block
        edits: List[MorphEdit] = []
        new = re.sub(r"--([^\s-])", r"-- \1", text)
        if new != text:
            edits.append(MorphEdit(0, len(text), text, new, "normalize_comment_spacing"))
            text = new
        # collapse runs of comment lines into a single block separated by newlines (keeps them; just normalizes)
        # No replacement performed here beyond spacing normalization.
        return text, edits

    def _normalize_function_header_spacing(self, text: str) -> Tuple[str, List[MorphEdit]]:
        pattern = re.compile(r"\bfunc\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.M)
        edits: List[MorphEdit] = []
        def repl(m: re.Match) -> str:
            before = m.group(0)
            after = f"func {m.group(1)}("
            if before != after:
                edits.append(MorphEdit(m.start(), m.end(), before, after, "normalize_function_header_spacing"))
            return after
        new = pattern.sub(repl, text)
        return new, edits

    def _expand_print_directive(self, text: str) -> Tuple[str, List[MorphEdit]]:
        pattern = re.compile(r"\bprint\s*:\s*(.+?)\s*;", re.S)
        edits: List[MorphEdit] = []
        def repl(m: re.Match) -> str:
            expr = m.group(1).strip()
            before = m.group(0)
            after = f"print({expr});"
            edits.append(MorphEdit(m.start(), m.end(), before, after, "expand_print_directive"))
            return after
        new = pattern.sub(repl, text)
        return new, edits

    def _expand_do_array(self, text: str) -> Tuple[str, List[MorphEdit]]:
        pattern = re.compile(r"\bdo\s*:\s*\[\s*(.*?)\s*\]\s*;", re.S)
        edits: List[MorphEdit] = []
        def repl(m: re.Match) -> str:
            inner = m.group(1).rstrip()
            before = m.group(0)
            after = f"do {{ {inner} }};"
            edits.append(MorphEdit(m.start(), m.end(), before, after, "expand_do_array"))
            return after
        new = pattern.sub(repl, text)
        return new, edits

    def _convert_label_colon_to_assignment(self, text: str) -> Tuple[str, List[MorphEdit]]:
        """
        Convert `ident: expr;` to `ident = expr;` only when ident is a simple identifier
        and the statement is not a known directive (e.g., not `print:` which has been transformed earlier).
        """
        pattern = re.compile(r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+?);\s*$")
        edits: List[MorphEdit] = []
        def repl(m: re.Match) -> str:
            ident = m.group(1)
            expr = m.group(2).rstrip()
            before = m.group(0)
            after = f"{ident} = {expr};"
            edits.append(MorphEdit(m.start(), m.end(), before, after, "convert_label_colon_to_assignment"))
            return after
        new = pattern.sub(repl, text)
        return new, edits

    def _normalize_if_then(self, text: str) -> Tuple[str, List[MorphEdit]]:
        pattern = re.compile(r"\bif\s+(.+?)\s+then\s*\{", re.S)
        edits: List[MorphEdit] = []
        def repl(m: re.Match) -> str:
            cond = m.group(1).strip()
            before = m.group(0)
            after = f"if ({cond}) {{"
            edits.append(MorphEdit(m.start(), m.end(), before, after, "normalize_if_then"))
            return after
        new = pattern.sub(repl, text)
        return new, edits

    def _normalize_while_not(self, text: str) -> Tuple[str, List[MorphEdit]]:
        pattern = re.compile(r"\bwhile\s+not\s+(.+?)\s*\{", re.S)
        edits: List[MorphEdit] = []
        def repl(m: re.Match) -> str:
            cond = m.group(1).strip()
            before = m.group(0)
            after = f"while (not ({cond})) {{"
            edits.append(MorphEdit(m.start(), m.end(), before, after, "normalize_while_not"))
            return after
        new = pattern.sub(repl, text)
        return new, edits

    def _remove_semicolon_before_brace(self, text: str) -> Tuple[str, List[MorphEdit]]:
        # replace `;}` with `}`
        new = re.sub(r";\s*}", "}", text)
        if new == text:
            return text, []
        return new, [MorphEdit(0, len(text), text, new, "remove_semicolon_before_brace")]

    def _remove_duplicate_semicolons(self, text: str) -> Tuple[str, List[MorphEdit]]:
        new = re.sub(r";{2,}", ";", text)
        if new == text:
            return text, []
        return new, [MorphEdit(0, len(text), text, new, "remove_duplicate_semicolons")]

    def _remove_empty_semicolon_lines(self, text: str) -> Tuple[str, List[MorphEdit]]:
        new = re.sub(r"(?m)^[ \t]*;\s*$\n?", "", text)
        if new == text:
            return text, []
        return new, [MorphEdit(0, len(text), text, new, "remove_empty_semicolon_lines")]

    def _ensure_statement_semicolons(self, text: str) -> Tuple[str, List[MorphEdit]]:
        """
        Ensure statements end with semicolons where appropriate.
        This is conservative: it only appends a semicolon to lines that look like simple expressions
        or assignments and do not already end with `;`, `{`, `}`, or `;}`.
        """
        lines = text.splitlines(keepends=True)
        edits: List[MorphEdit] = []
        changed = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            # skip lines continuing blocks or keywords
            if stripped.endswith(";") or stripped.endswith("{") or stripped.endswith("}") or stripped.endswith(":"):
                continue
            # heuristics: if line starts with keywords that shouldn't have semicolon appended, skip
            if re.match(r"^(func|if\b|while\b|quarantine\b|else\b|for\b|import\b|@)", stripped):
                continue
            # If looks like an expression/assignment (contains '=' or looks like ident call), add semicolon
            if re.search(r"=\s*|^[A-Za-z_][\w]*\s*\(|^[A-Za-z_][\w]*\s*$", stripped):
                # append semicolon preserving trailing newline
                if line.endswith("\n"):
                    new_line = line[:-1] + ";" + "\n"
                else:
                    new_line = line + ";"
                edits.append(MorphEdit(sum(len(x) for x in lines[:i]), sum(len(x) for x in lines[:i+1]), line, new_line, "ensure_statement_semicolons"))
                lines[i] = new_line
                changed = True
        if not changed:
            return text, []
        new = "".join(lines)
        return new, edits

    def _normalize_quarantine_semicolon(self, text: str) -> Tuple[str, List[MorphEdit]]:
        pattern = re.compile(r"(quarantine\s+try\s*\{.*?\}\s*replace\s*\{.*?\}\s*erase\s*\{.*?\})\s*;*", re.S)
        edits: List[MorphEdit] = []
        def repl(m: re.Match) -> str:
            inner = m.group(1)
            before = m.group(0)
            after = inner + ";"
            if before != after:
                edits.append(MorphEdit(m.start(), m.end(), before, after, "normalize_quarantine_semicolon"))
            return after
        new = pattern.sub(repl, text)
        return new, edits

    def _fix_unbalanced_braces(self, text: str) -> Tuple[str, List[MorphEdit]]:
        """
        Best-effort: if there are more '{' than '}', append missing '}' at the end.
        Also if more '}' than '{', remove trailing unmatched '}' lines (warn).
        """
        edits: List[MorphEdit] = []
        opens = text.count("{")
        closes = text.count("}")
        if opens == closes:
            return text, []
        if opens > closes:
            missing = opens - closes
            addition = "\n" + ("}" * missing) + "\n"
            new = text + addition
            edits.append(MorphEdit(len(text), len(new), "", addition, f"fix_unbalanced_braces_add_{missing}"))
            return new, edits
        else:
            # remove last N unmatched '}' lines conservatively
            unmatched = closes - opens
            # attempt to remove unmatched braces from end of file
            new = text
            removed = 0
            for _ in range(unmatched):
                idx = new.rfind("}")
                if idx == -1:
                    break
                # remove this character and any trailing whitespace on the line
                line_start = new.rfind("\n", 0, idx) + 1
                line_end = new.find("\n", idx)
                if line_end == -1:
                    line_end = len(new)
                removed_text = new[idx:idx+1]
                new = new[:idx] + new[idx+1:]
                removed += 1
            if removed > 0:
                edits.append(MorphEdit(0, len(text), text, new, f"fix_unbalanced_braces_remove_{removed}"))
            return new, edits

    # -------------------------
    # Utilities: diffs, file operations
    # -------------------------
    def diff(self, original: str, transformed: str, filename: str = "<source>") -> str:
        """
        Return a unified diff between original and transformed using difflib.
        """
        o_lines = original.splitlines(keepends=True)
        t_lines = transformed.splitlines(keepends=True)
        diff = difflib.unified_diff(o_lines, t_lines, fromfile=filename, tofile=filename + ".morphed", lineterm="")
        return "".join(line + "\n" for line in diff)

    def apply_to_file(self, src_path: str, out_path: Optional[str] = None, overwrite: bool = False, make_backup: bool = True, verbose: bool = False) -> MorphResult:
        p = Path(src_path)
        text = p.read_text(encoding="utf-8")
        result = self.morph(text)
        target = Path(src_path) if overwrite else Path(out_path or f"{src_path}.morphed.ix")
        if overwrite and make_backup:
            bak = p.with_suffix(p.suffix + ".bak")
            p.replace(bak) if p.exists() and not bak.exists() else None
            # write back to original path from transformed
            target = p
        if verbose and result.edits:
            print(f"[morph] {src_path}: {len(result.edits)} edits")
        target.write_text(result.transformed, encoding="utf-8")
        return result

    def apply_to_dir(self, dir_path: str, pattern: str = "*.ix", recursive: bool = True, max_workers: int = 4, inplace: bool = False, verbose: bool = False) -> Dict[str, MorphResult]:
        p = Path(dir_path)
        files = list(p.rglob(pattern) if recursive else p.glob(pattern))
        results: Dict[str, MorphResult] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_map = {}
            for f in files:
                future = ex.submit(self.apply_to_file, str(f), None, inplace, True, verbose)
                future_map[future] = f
            for fut in concurrent.futures.as_completed(future_map):
                f = future_map[fut]
                try:
                    res = fut.result()
                    results[str(f)] = res
                except Exception as e:
                    print(f"[error] failed morph {f}: {e}", file=sys.stderr)
        return results

    def preview_diff(self, src_path: str) -> str:
        p = Path(src_path)
        orig = p.read_text(encoding="utf-8")
        res = self.morph(orig)
        if orig == res.transformed:
            return ""
        return self.diff(orig, res.transformed, filename=src_path)

    # -------------------------
    # Watcher (polling)
    # -------------------------
    def watch(self, path: str, callback: Callable[[str, MorphResult], None], interval: float = 0.6):
        """
        Polls a file or directory and calls callback(file_path, morph_result) when file changes.
        Simple cross-platform watcher using mtime; no external deps.
        """
        tracked = {}
        p = Path(path)
        if p.is_file():
            tracked[p] = p.stat().st_mtime
        else:
            # track all .ix files under dir
            for f in p.rglob("*.ix"):
                tracked[f] = f.stat().st_mtime
        try:
            while True:
                time.sleep(interval)
                current = {}
                for f in list(tracked.keys()):
                    try:
                        mtime = f.stat().st_mtime
                        current[f] = mtime
                        if mtime != tracked[f]:
                            # file changed
                            try:
                                res = self.apply_to_file(str(f), overwrite=False, out_path=None)
                                callback(str(f), res)
                            except Exception as e:
                                print(f"[watch] apply error {f}: {e}", file=sys.stderr)
                            tracked[f] = mtime
                    except FileNotFoundError:
                        tracked.pop(f, None)
                # detect new files
                if p.is_dir():
                    for f in p.rglob("*.ix"):
                        if f not in tracked:
                            tracked[f] = f.stat().st_mtime
                # continue loop
        except KeyboardInterrupt:
            print("Watcher stopped.")

# -------------------------
# Unit tests
# -------------------------
def run_unit_tests(verbose: bool = True) -> bool:
    sm = SyntaxMorph()
    tests = []

    # 1: print expansion
    tests.append((
        'print: "Hello";',
        'print("Hello");'
    ))

    # 2: do array expansion
    tests.append((
        'do: [a = 1; b = 2;];',
        'do { a = 1; b = 2; };'
    ))

    # 3: label-colon to assignment
    tests.append((
        'data: [1, 2, 3];',
        'data = [1, 2, 3];'
    ))

    # 4: ensure semicolons
    tests.append((
        'x = 1\ny = 2\n',
        'x = 1;\ny = 2;'
    ))

    # 5: remove semicolon before brace
    tests.append((
        'if (x) {\n doSomething();\n};',
        'if (x) {\n doSomething();\n}'
    ))

    # 6: normalize if then
    tests.append((
        'if x > 0 then { print: "ok"; };',
        'if (x > 0) { print("ok"); };'
    ))

    all_pass = True
    for i, (inp, expected_partial) in enumerate(tests, 1):
        res = sm.morph(inp)
        out = res.transformed.strip()
        ok = expected_partial.strip() in out
        if verbose:
            print(f"Test {i}: {'PASS' if ok else 'FAIL'}")
            if not ok:
                print(" Input:", inp)
                print(" Output:", out)
                print(" Expected contains:", expected_partial)
                print(" Edits:", res.edits)
        all_pass = all_pass and ok

    # run some targeted checks for brace fixing
    inp = "func foo() { if (x) { do(); }\n"  # missing closing brace
    res = sm.morph(inp)
    if verbose:
        print("Brace fix output:", repr(res.transformed))
    if res.transformed.count("{") != res.transformed.count("}"):
        print("Brace test FAIL")
        all_pass = False
    else:
        if verbose:
            print("Brace test PASS")

    return all_pass

# -------------------------
# CLI
# -------------------------
def _cli():
    import argparse
    parser = argparse.ArgumentParser(prog="instryx_syntax_morph", description="Instryx syntax morphing tool")
    parser.add_argument("path", nargs="?", help="File or directory to process")
    parser.add_argument("--inplace", action="store_true", help="Write changes in-place")
    parser.add_argument("--out", help="Write transformed content to path (file).")
    parser.add_argument("--dir", help="Process directory recursively")
    parser.add_argument("--diff", action="store_true", help="Print unified diff instead of writing")
    parser.add_argument("--watch", action="store_true", help="Watch file/dir for changes")
    parser.add_argument("--test", action="store_true", help="Run unit tests")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    sm = SyntaxMorph()

    if args.test:
        ok = run_unit_tests(verbose=args.verbose)
        sys.exit(0 if ok else 2)

    if not args.path and not args.dir:
        parser.print_help()
        sys.exit(1)

    if args.dir:
        results = sm.apply_to_dir(args.dir, inplace=args.inplace, verbose=args.verbose)
        if args.verbose:
            print(f"Processed {len(results)} files.")
        sys.exit(0)

    path = args.path
    p = Path(path)
    if args.watch:
        def cb(file_path, res):
            print(f"[watch] {file_path} morphed ({len(res.edits)} edits).")
        sm.watch(path, cb)
        sys.exit(0)

    # single file
    orig = p.read_text(encoding="utf-8")
    res = sm.morph(orig)

    if args.diff:
        d = sm.diff(orig, res.transformed, filename=str(p))
        if d:
            print(d)
        else:
            print("No changes.")
        sys.exit(0)

    # write out
    if args.out:
        Path(args.out).write_text(res.transformed, encoding="utf-8")
        print(f"Wrote {args.out}")
    elif args.inplace:
        # backup original
        bak = p.with_suffix(p.suffix + ".bak")
        if not bak.exists():
            p.rename(bak)
            p.write_text(res.transformed, encoding="utf-8")
        else:
            # if backup exists, just overwrite original
            p.write_text(res.transformed, encoding="utf-8")
        print(f"Wrote in-place {p}")
    else:
        # print to stdout
        print(res.transformed)

# -------------------------
# Optional macro-overlay integration helper
# -------------------------
def expand_macros_if_available(source: str, filename: Optional[str] = None):
    """
    If a `macro_overlay` module is importable and defines `createFullRegistry`
    and `applyMacrosWithDiagnostics`, call it and return expanded text and diagnostics.
    Otherwise return source unchanged and empty diagnostics.
    """
    try:
        import importlib
        mo = importlib.import_module("macro_overlay")
        if hasattr(mo, "createFullRegistry") and hasattr(mo, "applyMacrosWithDiagnostics"):
            registry = mo.createFullRegistry()
            # applyMacrosWithDiagnostics expects (source, registry, ctx)
            res = mo.applyMacrosWithDiagnostics(source, registry, {"filename": filename})
            # support async or sync (if it returns coroutine)
            if hasattr(res, "__await__"):
                import asyncio
                result, diagnostics = asyncio.get_event_loop().run_until_complete(res)
            else:
                result, diagnostics = res.get("result"), res.get("diagnostics")
                # the module's function earlier returned dict { result, diagnostics }
                if isinstance(res, dict) and "result" in res:
                    result = res["result"]
                    diagnostics = res.get("diagnostics", [])
            if isinstance(result, dict) and "transformed" in result:
                return result["transformed"], diagnostics
    except Exception:
        # silent fallback: macro overlay not available or failed
        pass
    return source, []

# -------------------------
# Execute CLI if called directly
# -------------------------
if __name__ == "__main__":
    _cli()

"""
instryx_syntax_morph.py

Extended Instryx syntax morphing utilities (supreme-boosters edition).

Additions in this edition:
 - Extra safe morph passes:
    - remove BOM
    - normalize indentation (tabs -> spaces) and canonical indent size
    - fold adjacent single-line comments into a comment block
    - sort top-level imports (deterministic)
    - collapse multiple trailing newlines to single newline at EOF
 - Dry-run support, edits JSON export, backup rotation
 - Source-map-like line mapping (original->transformed)
 - Batch directory processing with configurable concurrency (defaults to CPU count)
 - Improved CLI flags: --dry-run, --edits-json, --keep-backups, --indent-size
 - Better logging and verbose output
 - All passes return edit records and are traceable
 - Self-test harness expanded

Design note: passes remain purely textual and conservative. For complex transforms,
use AST-level tooling.
"""

from __future__ import annotations
from dataclasses import dataclass
import re
from typing import List, Optional, Tuple, Callable, Dict, Any
from pathlib import Path
import difflib
import concurrent.futures
import time
import sys
import os
import json
import multiprocessing
import logging

LOG = logging.getLogger("instryx_syntax_morph")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -------------------------
# Data classes
# -------------------------

@dataclass
class MorphEdit:
    start: int
    end: int
    original: str
    replacement: str
    reason: str

@dataclass
class MorphResult:
    transformed: str
    edits: List[MorphEdit]
    source_map: Optional[List[Tuple[int, int]]] = None  # list of (orig_line, new_line) pairs for basic mapping

# -------------------------
# Core morph class
# -------------------------

class SyntaxMorph:
    """
    Apply a sequence of safe, deterministic morph passes to Instryx source text.
    """

    def __init__(self, extra_passes: Optional[List[Callable[[str], Tuple[str, List[MorphEdit]]]]] = None, indent_size: int = 2):
        self.indent_size = int(indent_size)
        # base passes in order
        self.passes: List[Callable[[str], Tuple[str, List[MorphEdit]]]] = [
            self._remove_bom,
            self._normalize_line_endings,
            self._trim_trailing_spaces,
            self._collapse_blank_lines,
            self._normalize_comment_spacing,
            self._fold_adjacent_comments,
            self._normalize_function_header_spacing,
            self._expand_print_directive,
            self._expand_do_array,
            self._convert_label_colon_to_assignment,
            self._normalize_if_then,
            self._normalize_while_not,
            self._remove_semicolon_before_brace,
            self._remove_duplicate_semicolons,
            self._remove_empty_semicolon_lines,
            self._ensure_statement_semicolons,
            self._normalize_quarantine_semicolon,
            self._sort_top_level_imports,
            self._normalize_indentation,
            self._collapse_trailing_newlines,
            self._fix_unbalanced_braces,
        ]
        if extra_passes:
            self.passes.extend(extra_passes)

    def morph(self, source: str) -> MorphResult:
        edits: List[MorphEdit] = []
        text = source
        for p in self.passes:
            try:
                text, pass_edits = p(text)
            except Exception:
                LOG.exception("Pass %s failed; continuing", getattr(p, "__name__", repr(p)))
                continue
            edits.extend(pass_edits)
        # compute a simple source map (line-based) by diffing original->transformed
        source_map = self._compute_basic_sourcemap(source, text)
        return MorphResult(transformed=text, edits=edits, source_map=source_map)

    # -------------------------
    # Individual passes
    # -------------------------
    def _remove_bom(self, text: str) -> Tuple[str, List[MorphEdit]]:
        if text.startswith("\ufeff"):
            new = text.lstrip("\ufeff")
            return new, [MorphEdit(0, 1, "\ufeff", "", "remove_bom")]
        return text, []

    def _normalize_line_endings(self, text: str) -> Tuple[str, List[MorphEdit]]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        if normalized == text:
            return text, []
        return normalized, [MorphEdit(0, len(text), text, normalized, "normalize_line_endings")]

    def _trim_trailing_spaces(self, text: str) -> Tuple[str, List[MorphEdit]]:
        new = re.sub(r"[ \t]+(?=\n)", "", text)
        if new == text:
            return text, []
        return new, [MorphEdit(0, len(text), text, new, "trim_trailing_spaces")]

    def _collapse_blank_lines(self, text: str) -> Tuple[str, List[MorphEdit]]:
        # replace 3+ consecutive blank lines with 1 blank line
        new = re.sub(r"\n{3,}", "\n\n", text)
        if new == text:
            return text, []
        return new, [MorphEdit(0, len(text), text, new, "collapse_blank_lines")]

    def _normalize_comment_spacing(self, text: str) -> Tuple[str, List[MorphEdit]]:
        # normalize `--comment` to `-- comment`
        new = re.sub(r"--([^\s-])", r"-- \1", text)
        if new == text:
            return text, []
        return new, [MorphEdit(0, len(text), text, new, "normalize_comment_spacing")]

    def _fold_adjacent_comments(self, text: str) -> Tuple[str, List[MorphEdit]]:
        """
        Fold sequences of single-line `--` comments into blocks separated by a single newline.
        This keeps comments but groups them; no content lost.
        """
        lines = text.splitlines(keepends=True)
        new_lines: List[str] = []
        edits: List[MorphEdit] = []
        i = 0
        while i < len(lines):
            if lines[i].lstrip().startswith("--"):
                # collect block
                start_i = i
                block = []
                while i < len(lines) and lines[i].lstrip().startswith("--"):
                    block.append(lines[i].lstrip().rstrip("\n").lstrip())  # keep text after leading spaces
                    i += 1
                # create normalized block
                block_text = "\n".join(block) + "\n"
                orig_text = "".join(lines[start_i:i])
                if orig_text != block_text:
                    edits.append(MorphEdit(sum(len(x) for x in lines[:start_i]), sum(len(x) for x in lines[:i]), orig_text, block_text, "fold_adjacent_comments"))
                new_lines.append(block_text)
            else:
                new_lines.append(lines[i])
                i += 1
        new = "".join(new_lines)
        if edits:
            return new, edits
        return text, []

    def _normalize_function_header_spacing(self, text: str) -> Tuple[str, List[MorphEdit]]:
        pattern = re.compile(r"\bfunc\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.M)
        edits: List[MorphEdit] = []
        def repl(m: re.Match) -> str:
            before = m.group(0)
            after = f"func {m.group(1)}("
            if before != after:
                edits.append(MorphEdit(m.start(), m.end(), before, after, "normalize_function_header_spacing"))
            return after
        new = pattern.sub(repl, text)
        return new, edits

    def _expand_print_directive(self, text: str) -> Tuple[str, List[MorphEdit]]:
        pattern = re.compile(r"\bprint\s*:\s*(.+?)\s*;", re.S)
        edits: List[MorphEdit] = []
        def repl(m: re.Match) -> str:
            expr = m.group(1).strip()
            before = m.group(0)
            after = f"print({expr});"
            edits.append(MorphEdit(m.start(), m.end(), before, after, "expand_print_directive"))
            return after
        new = pattern.sub(repl, text)
        return new, edits

    def _expand_do_array(self, text: str) -> Tuple[str, List[MorphEdit]]:
        pattern = re.compile(r"\bdo\s*:\s*\[\s*(.*?)\s*\]\s*;", re.S)
        edits: List[MorphEdit] = []
        def repl(m: re.Match) -> str:
            inner = m.group(1).rstrip()
            before = m.group(0)
            after = f"do {{ {inner} }};"
            edits.append(MorphEdit(m.start(), m.end(), before, after, "expand_do_array"))
            return after
        new = pattern.sub(repl, text)
        return new, edits

    def _convert_label_colon_to_assignment(self, text: str) -> Tuple[str, List[MorphEdit]]:
        """
        Convert `ident: expr;` to `ident = expr;` only when ident is a simple identifier.
        """
        pattern = re.compile(r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+?);\s*$")
        edits: List[MorphEdit] = []
        def repl(m: re.Match) -> str:
            ident = m.group(1)
            expr = m.group(2).rstrip()
            before = m.group(0)
            after = f"{ident} = {expr};"
            edits.append(MorphEdit(m.start(), m.end(), before, after, "convert_label_colon_to_assignment"))
            return after
        new = pattern.sub(repl, text)
        return new, edits

    def _normalize_if_then(self, text: str) -> Tuple[str, List[MorphEdit]]:
        pattern = re.compile(r"\bif\s+(.+?)\s+then\s*\{", re.S)
        edits: List[MorphEdit] = []
        def repl(m: re.Match) -> str:
            cond = m.group(1).strip()
            before = m.group(0)
            after = f"if ({cond}) {{"
            edits.append(MorphEdit(m.start(), m.end(), before, after, "normalize_if_then"))
            return after
        new = pattern.sub(repl, text)
        return new, edits

    def _normalize_while_not(self, text: str) -> Tuple[str, List[MorphEdit]]:
        pattern = re.compile(r"\bwhile\s+not\s+(.+?)\s*\{", re.S)
        edits: List[MorphEdit] = []
        def repl(m: re.Match) -> str:
            cond = m.group(1).strip()
            before = m.group(0)
            after = f"while (not ({cond})) {{"
            edits.append(MorphEdit(m.start(), m.end(), before, after, "normalize_while_not"))
            return after
        new = pattern.sub(repl, text)
        return new, edits

    def _remove_semicolon_before_brace(self, text: str) -> Tuple[str, List[MorphEdit]]:
        new = re.sub(r";\s*}", "}", text)
        if new == text:
            return text, []
        return new, [MorphEdit(0, len(text), text, new, "remove_semicolon_before_brace")]

    def _remove_duplicate_semicolons(self, text: str) -> Tuple[str, List[MorphEdit]]:
        new = re.sub(r";{2,}", ";", text)
        if new == text:
            return text, []
        return new, [MorphEdit(0, len(text), text, new, "remove_duplicate_semicolons")]

    def _remove_empty_semicolon_lines(self, text: str) -> Tuple[str, List[MorphEdit]]:
        new = re.sub(r"(?m)^[ \t]*;\s*$\n?", "", text)
        if new == text:
            return text, []
        return new, [MorphEdit(0, len(text), text, new, "remove_empty_semicolon_lines")]

    def _ensure_statement_semicolons(self, text: str) -> Tuple[str, List[MorphEdit]]:
        """
        Append semicolons to lines that look like simple expressions/assignments but lack punctuation.
        Conservative heuristics to avoid breaking control structures.
        """
        lines = text.splitlines(keepends=True)
        edits: List[MorphEdit] = []
        changed = False
        offset = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                offset += len(line)
                continue
            if stripped.endswith(";") or stripped.endswith("{") or stripped.endswith("}") or stripped.endswith(":"):
                offset += len(line)
                continue
            if re.match(r"^(func\b|if\b|while\b|else\b|for\b|quarantine\b|return\b|break\b|continue\b|import\b|@)", stripped):
                offset += len(line)
                continue
            # heuristics for simple statement/assignment/function-call lines
            if re.search(r"=\s*|^[A-Za-z_][\w]*\s*\(|^[A-Za-z_][\w]*\s*$", stripped):
                # append semicolon preserving newline
                if line.endswith("\n"):
                    new_line = line[:-1] + ";" + "\n"
                else:
                    new_line = line + ";"
                edits.append(MorphEdit(offset, offset + len(line), line, new_line, "ensure_statement_semicolons"))
                lines[i] = new_line
                changed = True
                offset += len(new_line)
            else:
                offset += len(line)
        if not changed:
            return text, []
        new = "".join(lines)
        return new, edits

    def _normalize_quarantine_semicolon(self, text: str) -> Tuple[str, List[MorphEdit]]:
        pattern = re.compile(r"(quarantine\s+try\s*\{.*?\}\s*replace\s*\{.*?\}\s*erase\s*\{.*?\})\s*;*", re.S)
        edits: List[MorphEdit] = []
        def repl(m: re.Match) -> str:
            inner = m.group(1)
            before = m.group(0)
            after = inner + ";"
            if before != after:
                edits.append(MorphEdit(m.start(), m.end(), before, after, "normalize_quarantine_semicolon"))
            return after
        new = pattern.sub(repl, text)
        return new, edits

    def _sort_top_level_imports(self, text: str) -> Tuple[str, List[MorphEdit]]:
        """
        Sort contiguous top-level import lines to a deterministic order.
        Only affects contiguous runs of lines that start with `import`.
        """
        lines = text.splitlines(keepends=True)
        i = 0
        edits: List[MorphEdit] = []
        changed = False
        out_lines: List[str] = []
        while i < len(lines):
            if lines[i].lstrip().startswith("import "):
                start = i
                imports = []
                while i < len(lines) and lines[i].lstrip().startswith("import "):
                    imports.append(lines[i].strip())
                    i += 1
                sorted_imports = sorted(dict.fromkeys(imports))  # dedupe and keep sorted
                if imports != sorted_imports:
                    changed = True
                    orig = "".join(lines[start:i])
                    after = "".join(s + "\n" for s in sorted_imports)
                    edits.append(MorphEdit(sum(len(x) for x in lines[:start]), sum(len(x) for x in lines[:i]), orig, after, "sort_top_level_imports"))
                    out_lines.append(after)
                else:
                    out_lines.extend(lines[start:i])
            else:
                out_lines.append(lines[i])
                i += 1
        if not changed:
            return text, []
        new = "".join(out_lines)
        return new, edits

    def _normalize_indentation(self, text: str) -> Tuple[str, List[MorphEdit]]:
        """
        Convert leading tabs into spaces and normalize to self.indent_size.
        """
        lines = text.splitlines(keepends=True)
        edits: List[MorphEdit] = []
        changed = False
        for idx, line in enumerate(lines):
            # find leading whitespace
            m = re.match(r"^([ \t]+)(.*)$", line)
            if not m:
                continue
            leading, rest = m.group(1), m.group(2)
            # convert tabs to spaces (tab assumed to be 8 columns, standard)
            spaces = leading.replace("\t", " " * 8)
            # reduce runs of >indent_size spaces to multiples of indent_size where practical (preserve relative indentation)
            # compute number of indent levels relative to indent_size
            level = len(spaces) // self.indent_size
            new_lead = " " * (level * self.indent_size)
            new_line = new_lead + rest
            if new_line != line:
                offset = sum(len(x) for x in lines[:idx])
                edits.append(MorphEdit(offset, offset + len(line), line, new_line, "normalize_indentation"))
                lines[idx] = new_line
                changed = True
        if not changed:
            return text, []
        new = "".join(lines)
        return new, edits

    def _collapse_trailing_newlines(self, text: str) -> Tuple[str, List[MorphEdit]]:
        # ensure file ends with exactly one newline
        if not text:
            return text, []
        if text.endswith("\n"):
            # collapse multiple trailing newlines to one
            new = re.sub(r"\n{2,}\Z", "\n", text)
        else:
            new = text + "\n"
        if new == text:
            return text, []
        return new, [MorphEdit(0, len(text), text, new, "collapse_trailing_newlines")]

    def _fix_unbalanced_braces(self, text: str) -> Tuple[str, List[MorphEdit]]:
        """
        Best-effort: append missing '}' if opens > closes; trim trailing unmatched '}' if closes > opens.
        """
        edits: List[MorphEdit] = []
        opens = text.count("{")
        closes = text.count("}")
        if opens == closes:
            return text, []
        if opens > closes:
            missing = opens - closes
            addition = ("\n" + ("}" * missing) + "\n")
            new = text + addition
            edits.append(MorphEdit(len(text), len(new), "", addition, f"fix_unbalanced_braces_add_{missing}"))
            return new, edits
        else:
            # remove last N unmatched '}' characters from end conservatively
            unmatched = closes - opens
            new = text
            removed = 0
            for _ in range(unmatched):
                idx = new.rfind("}")
                if idx == -1:
                    break
                new = new[:idx] + new[idx+1:]
                removed += 1
            if removed > 0:
                edits.append(MorphEdit(0, len(text), text, new, f"fix_unbalanced_braces_remove_{removed}"))
            return new, edits

    # -------------------------
    # Utilities: diffs, file operations, source map
    # -------------------------
    def diff(self, original: str, transformed: str, filename: str = "<source>") -> str:
        o_lines = original.splitlines(keepends=True)
        t_lines = transformed.splitlines(keepends=True)
        diff = difflib.unified_diff(o_lines, t_lines, fromfile=filename, tofile=filename + ".morphed", lineterm="")
        return "".join(line + "\n" for line in diff)

    def _compute_basic_sourcemap(self, original: str, transformed: str) -> List[Tuple[int, int]]:
        """
        Compute a simple line-based mapping: returns list of tuples (orig_line_no, new_line_no)
        for lines that remained identical or moved in position. This is a best-effort map using difflib.
        """
        o_lines = original.splitlines()
        t_lines = transformed.splitlines()
        sm = []
        matcher = difflib.SequenceMatcher(a=o_lines, b=t_lines)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for oi, nj in zip(range(i1, i2), range(j1, j2)):
                    sm.append((oi + 1, nj + 1))
        return sm

    def apply_to_file(self, src_path: str, out_path: Optional[str] = None, overwrite: bool = False,
                      make_backup: bool = True, keep_backups: int = 3, dry_run: bool = False, verbose: bool = False) -> MorphResult:
        p = Path(src_path)
        text = p.read_text(encoding="utf-8")
        result = self.morph(text)
        if dry_run:
            if verbose:
                LOG.info("[dry-run] %s -> %d edits", src_path, len(result.edits))
            return result
        target_path = p if overwrite and not out_path else Path(out_path or f"{src_path}.morphed.ix")
        if overwrite:
            if make_backup:
                # create backup with timestamp; rotate
                bak_base = p.with_suffix(p.suffix + ".bak")
                ts = time.strftime("%Y%m%d%H%M%S")
                bak = p.with_name(p.stem + f".{ts}.bak{p.suffix}")
                p.replace(bak) if p.exists() else None
                # rotate backups
                backups = sorted(p.parent.glob(p.stem + ".*.bak" + p.suffix), key=lambda x: x.stat().st_mtime if x.exists() else 0)
                while len(backups) > keep_backups:
                    try:
                        backups[0].unlink()
                        backups.pop(0)
                    except Exception:
                        break
            # write transformed into original path
            target_path = p
        if verbose and result.edits:
            LOG.info("morph %s -> %s (%d edits)", src_path, str(target_path), len(result.edits))
        target_path.write_text(result.transformed, encoding="utf-8")
        return result

    def apply_to_dir(self, dir_path: str, pattern: str = "*.ix", recursive: bool = True,
                     max_workers: Optional[int] = None, inplace: bool = False, verbose: bool = False,
                     dry_run: bool = False, keep_backups: int = 3) -> Dict[str, MorphResult]:
        p = Path(dir_path)
        if max_workers is None:
            max_workers = max(1, multiprocessing.cpu_count() - 1)
        files = list(p.rglob(pattern) if recursive else p.glob(pattern))
        results: Dict[str, MorphResult] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_map = {}
            for f in files:
                future = ex.submit(self.apply_to_file, str(f), None, inplace, True, keep_backups, dry_run, verbose)
                future_map[future] = f
            for fut in concurrent.futures.as_completed(future_map):
                f = future_map[fut]
                try:
                    res = fut.result()
                    results[str(f)] = res
                except Exception as e:
                    LOG.exception("failed morph %s: %s", f, e)
        return results

    def preview_diff(self, src_path: str) -> str:
        p = Path(src_path)
        orig = p.read_text(encoding="utf-8")
        res = self.morph(orig)
        if orig == res.transformed:
            return ""
        return self.diff(orig, res.transformed, filename=src_path)

    # -------------------------
    # Watcher (polling)
    # -------------------------
    def watch(self, path: str, callback: Callable[[str, MorphResult], None], interval: float = 0.6):
        """
        Polls a file or directory and calls callback(file_path, morph_result) when file changes.
        """
        tracked = {}
        p = Path(path)
        if p.is_file():
            tracked[p] = p.stat().st_mtime
        else:
            for f in p.rglob("*.ix"):
                tracked[f] = f.stat().st_mtime
        try:
            while True:
                time.sleep(interval)
                for f in list(tracked.keys()):
                    try:
                        mtime = f.stat().st_mtime
                        if mtime != tracked[f]:
                            res = self.apply_to_file(str(f), overwrite=False, out_path=None)
                            callback(str(f), res)
                            tracked[f] = mtime
                    except FileNotFoundError:
                        tracked.pop(f, None)
                if p.is_dir():
                    for f in p.rglob("*.ix"):
                        if f not in tracked:
                            tracked[f] = f.stat().st_mtime
        except KeyboardInterrupt:
            LOG.info("Watcher stopped.")

# -------------------------
# Unit tests
# -------------------------
def run_unit_tests(verbose: bool = True) -> bool:
    sm = SyntaxMorph(indent_size=2)
    tests = []

    tests.append((
        'print: "Hello";',
        'print("Hello");'
    ))

    tests.append((
        'do: [a = 1; b = 2;];',
        'do { a = 1; b = 2; };'
    ))

    tests.append((
        'data: [1, 2, 3];',
        'data = [1, 2, 3];'
    ))

    tests.append((
        'x = 1\ny = 2\n',
        'x = 1;\ny = 2;'
    ))

    tests.append((
        'if (x) {\n doSomething();\n};',
        'if (x) {\n doSomething();\n}'
    ))

    tests.append((
        'if x > 0 then { print: "ok"; };',
        'if (x > 0) { print("ok"); };'
    ))

    all_pass = True
    for i, (inp, expected_partial) in enumerate(tests, 1):
        res = sm.morph(inp)
        out = res.transformed.strip()
        ok = expected_partial.strip() in out
        if verbose:
            print(f"Test {i}: {'PASS' if ok else 'FAIL'}")
            if not ok:
                print(" Input:", inp)
                print(" Output:", out)
                print(" Expected contains:", expected_partial)
                print(" Edits:", res.edits)
        all_pass = all_pass and ok

    # brace balancing test
    inp = "func foo() { if (x) { do(); }\n"
    res = sm.morph(inp)
    if verbose:
        print("Brace fix output:", repr(res.transformed))
    if res.transformed.count("{") != res.transformed.count("}"):
        print("Brace test FAIL")
        all_pass = False
    else:
        if verbose:
            print("Brace test PASS")

    return all_pass

# -------------------------
# CLI
# -------------------------
def _cli(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(prog="instryx_syntax_morph", description="Instryx syntax morphing tool (supreme boosters)")
    parser.add_argument("path", nargs="?", help="File or directory to process")
    parser.add_argument("--inplace", action="store_true", help="Write changes in-place")
    parser.add_argument("--out", help="Write transformed content to path (file).")
    parser.add_argument("--dir", help="Process directory recursively")
    parser.add_argument("--diff", action="store_true", help="Print unified diff instead of writing")
    parser.add_argument("--watch", action="store_true", help="Watch file/dir for changes")
    parser.add_argument("--test", action="store_true", help="Run unit tests")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Do not write files; just show diff/edits")
    parser.add_argument("--edits-json", help="Write edits list JSON to given path")
    parser.add_argument("--keep-backups", type=int, default=3, help="Backup rotation count when writing in-place")
    parser.add_argument("--indent-size", type=int, default=2, help="Spaces per indent level for normalization")
    parser.add_argument("--concurrency", type=int, default=max(1, multiprocessing.cpu_count()-1), help="Workers for directory processing")
    args = parser.parse_args(argv)

    if args.verbose:
        LOG.setLevel(logging.DEBUG)
        LOG.debug("Verbose mode enabled")

    sm = SyntaxMorph(indent_size=args.indent_size)

    if args.test:
        ok = run_unit_tests(verbose=args.verbose)
        sys.exit(0 if ok else 2)

    if not args.path and not args.dir:
        parser.print_help()
        sys.exit(1)

    if args.dir:
        results = sm.apply_to_dir(args.dir, recursive=True, max_workers=args.concurrency, inplace=args.inplace, verbose=args.verbose, dry_run=args.dry_run, keep_backups=args.keep_backups)
        if args.verbose:
            print(f"Processed {len(results)} files.")
        # if edits-json requested, aggregate edits
        if args.edits_json:
            out = {}
            for k, r in results.items():
                out[k] = [e.__dict__ for e in r.edits]
            Path(args.edits_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        sys.exit(0)

    path = args.path
    p = Path(path)
    if args.watch:
        def cb(file_path, res):
            print(f"[watch] {file_path} morphed ({len(res.edits)} edits).")
        sm.watch(path, cb)
        sys.exit(0)

    # single file
    orig = p.read_text(encoding="utf-8")
    res = sm.morph(orig)

    if res.transformed == orig:
        print("No changes.")
        if args.edits_json:
            Path(args.edits_json).write_text(json.dumps({str(p): []}, indent=2), encoding="utf-8")
        sys.exit(0)

    if args.diff or args.dry_run:
        d = sm.diff(orig, res.transformed, filename=str(p))
        print(d)
        if args.edits_json:
            Path(args.edits_json).write_text(json.dumps({str(p): [e.__dict__ for e in res.edits]}, indent=2), encoding="utf-8")
        sys.exit(0)

    # write out
    if args.out:
        Path(args.out).write_text(res.transformed, encoding="utf-8")
        print(f"Wrote {args.out}")
    elif args.inplace:
        # backup original
        bak_base = p.with_suffix(p.suffix + ".bak")
        ts = time.strftime("%Y%m%d%H%M%S")
        bak = p.with_name(p.stem + f".{ts}.bak{p.suffix}")
        if not bak.exists():
            try:
                p.replace(bak)
            except Exception:
                # fallback to copy
                shutil.copy2(p, bak)
        p.write_text(res.transformed, encoding="utf-8")
        # rotate backups
        backups = sorted(p.parent.glob(p.stem + ".*.bak" + p.suffix), key=lambda x: x.stat().st_mtime if x.exists() else 0)
        while len(backups) > args.keep_backups:
            try:
                backups[0].unlink()
                backups.pop(0)
            except Exception:
                break
        print(f"Wrote in-place {p}")
    else:
        print(res.transformed)

    if args.edits_json:
        Path(args.edits_json).write_text(json.dumps({str(p): [e.__dict__ for e in res.edits]}, indent=2), encoding="utf-8")

# -------------------------
# Optional macro-overlay integration helper
# -------------------------
def expand_macros_if_available(source: str, filename: Optional[str] = None):
    """
    If a `macro_overlay` module is importable and defines `createFullRegistry`
    and `applyMacrosWithDiagnostics`, call it and return expanded text and diagnostics.
    Otherwise return source unchanged and empty diagnostics.
    """
    try:
        import importlib
        mo = importlib.import_module("macro_overlay")
        if hasattr(mo, "createFullRegistry") and hasattr(mo, "applyMacrosWithDiagnostics"):
            registry = mo.createFullRegistry()
            res = mo.applyMacrosWithDiagnostics(source, registry, {"filename": filename})
            # support async or sync result
            if hasattr(res, "__await__"):
                import asyncio
                result, diagnostics = asyncio.get_event_loop().run_until_complete(res)
            else:
                # expected dict with result & diagnostics
                if isinstance(res, dict):
                    result = res.get("result", source)
                    diagnostics = res.get("diagnostics", [])
                else:
                    return source, []
            if isinstance(result, dict) and "transformed" in result:
                return result["transformed"], diagnostics
            return result, diagnostics
    except Exception:
        pass
    return source, []

# -------------------------
# Execute CLI if called directly
# -------------------------
if __name__ == "__main__":
    _cli()

    import argparse
    import shutil
    import sys
    _cli()
    
"""
instryx_syntax_morph.py

Supreme-boosters edition — extended Instryx syntax morphing utilities.

Features added / boosted:
 - All conservative text-based morph passes (normalize, semicolons, indentation, braces, comments)
 - Additional passes: remove BOM, fold adjacent comments, sort top-level imports,
   normalize indentation (tabs->spaces), collapse trailing newlines.
 - Safe file operations: backup rotation, dry-run, edits JSON export, edits summary report
 - Directory batch processing with configurable concurrency and progress logging
 - Watcher (polling) with callback
 - Simple source-map (line -> line) computed via difflib SequenceMatcher
 - Benchmarked unit tests and expanded test cases
 - Optional macro_overlay integration via --apply-macros
 - Fully type annotated, pure stdlib, executable as CLI

Usage:
  python instryx_syntax_morph.py file.ix --inplace
  python instryx_syntax_morph.py --dir src/ --diff
  python instryx_syntax_morph.py --test
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import re
import json
import time
import shutil
import logging
import difflib
import concurrent.futures
import multiprocessing
import tempfile
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Callable, Dict, Any

LOG = logging.getLogger("instryx_syntax_morph")
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -------------------------
# Data classes
# -------------------------


@dataclass
class MorphEdit:
    start: int
    end: int
    original: str
    replacement: str
    reason: str


@dataclass
class MorphResult:
    transformed: str
    edits: List[MorphEdit]
    source_map: Optional[List[Tuple[int, int]]] = None  # (orig_line, new_line) pairs


# -------------------------
# SyntaxMorph core
# -------------------------


class SyntaxMorph:
    """
    Textual, conservative morphing for Instryx source files.
    """

    def __init__(self, indent_size: int = 2, extra_passes: Optional[List[Callable[[str], Tuple[str, List[MorphEdit]]]]] = None):
        self.indent_size = max(1, int(indent_size))
        # compile commonly used regexes once
        self._re_print = re.compile(r"\bprint\s*:\s*(.+?)\s*;", re.S)
        self._re_do_array = re.compile(r"\bdo\s*:\s*\[\s*(.*?)\s*\]\s*;", re.S)
        self._re_label_colon = re.compile(r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+?);\s*$")
        self._re_if_then = re.compile(r"\bif\s+(.+?)\s+then\s*\{", re.S)
        self._re_while_not = re.compile(r"\bwhile\s+not\s+(.+?)\s*\{", re.S)
        self._re_quarantine = re.compile(r"(quarantine\s+try\s*\{.*?\}\s*replace\s*\{.*?\}\s*erase\s*\{.*?\})\s*;*", re.S)
        # pass pipeline
        self.passes: List[Callable[[str], Tuple[str, List[MorphEdit]]]] = [
            self._remove_bom,
            self._normalize_line_endings,
            self._trim_trailing_spaces,
            self._collapse_blank_lines,
            self._normalize_comment_spacing,
            self._fold_adjacent_comments,
            self._normalize_function_header_spacing,
            self._expand_print_directive,
            self._expand_do_array,
            self._convert_label_colon_to_assignment,
            self._normalize_if_then,
            self._normalize_while_not,
            self._remove_semicolon_before_brace,
            self._remove_duplicate_semicolons,
            self._remove_empty_semicolon_lines,
            self._ensure_statement_semicolons,
            self._normalize_quarantine_semicolon,
            self._sort_top_level_imports,
            self._normalize_indentation,
            self._collapse_trailing_newlines,
            self._fix_unbalanced_braces,
        ]
        if extra_passes:
            self.passes.extend(extra_passes)

    # -------------------------
    # High-level morph
    # -------------------------
    def morph(self, source: str) -> MorphResult:
        original = source
        text = source
        edits: List[MorphEdit] = []
        for p in self.passes:
            try:
                text, p_edits = p(text)
            except Exception:
                LOG.exception("Pass %s failed", getattr(p, "__name__", repr(p)))
                p_edits = []
            edits.extend(p_edits)
        source_map = self._compute_basic_sourcemap(original, text)
        return MorphResult(transformed=text, edits=edits, source_map=source_map)

    # -------------------------
    # Individual passes (safe, conservative)
    # -------------------------
    def _remove_bom(self, text: str) -> Tuple[str, List[MorphEdit]]:
        if text.startswith("\ufeff"):
            new = text.lstrip("\ufeff")
            return new, [MorphEdit(0, 1, "\ufeff", "", "remove_bom")]
        return text, []

    def _normalize_line_endings(self, text: str) -> Tuple[str, List[MorphEdit]]:
        new = text.replace("\r\n", "\n").replace("\r", "\n")
        if new == text:
            return text, []
        return new, [MorphEdit(0, len(text), text, new, "normalize_line_endings")]

    def _trim_trailing_spaces(self, text: str) -> Tuple[str, List[MorphEdit]]:
        new = re.sub(r"[ \t]+(?=\n)", "", text)
        if new == text:
            return text, []
        return new, [MorphEdit(0, len(text), text, new, "trim_trailing_spaces")]

    def _collapse_blank_lines(self, text: str) -> Tuple[str, List[MorphEdit]]:
        new = re.sub(r"\n{3,}", "\n\n", text)
        if new == text:
            return text, []
        return new, [MorphEdit(0, len(text), text, new, "collapse_blank_lines")]

    def _normalize_comment_spacing(self, text: str) -> Tuple[str, List[MorphEdit]]:
        # convert "--comment" to "-- comment"
        new = re.sub(r"--([^\s-])", r"-- \1", text)
        if new == text:
            return text, []
        return new, [MorphEdit(0, len(text), text, new, "normalize_comment_spacing")]

    def _fold_adjacent_comments(self, text: str) -> Tuple[str, List[MorphEdit]]:
        lines = text.splitlines(keepends=True)
        i = 0
        new_lines: List[str] = []
        edits: List[MorphEdit] = []
        while i < len(lines):
            if lines[i].lstrip().startswith("--"):
                start = i
                block = []
                while i < len(lines) and lines[i].lstrip().startswith("--"):
                    # strip leading spaces, keep comment marker + content
                    block.append(lines[i].lstrip().rstrip("\n"))
                    i += 1
                orig = "".join(lines[start:i])
                folded = "\n".join(block) + "\n"
                if orig != folded:
                    edits.append(MorphEdit(sum(len(x) for x in lines[:start]), sum(len(x) for x in lines[:i]), orig, folded, "fold_adjacent_comments"))
                new_lines.append(folded)
            else:
                new_lines.append(lines[i])
                i += 1
        if edits:
            return "".join(new_lines), edits
        return text, []

    def _normalize_function_header_spacing(self, text: str) -> Tuple[str, List[MorphEdit]]:
        pattern = re.compile(r"\bfunc\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.M)
        edits: List[MorphEdit] = []

        def repl(m: re.Match) -> str:
            before = m.group(0)
            after = f"func {m.group(1)}("
            if before != after:
                edits.append(MorphEdit(m.start(), m.end(), before, after, "normalize_function_header_spacing"))
            return after

        new = pattern.sub(repl, text)
        return new, edits

    def _expand_print_directive(self, text: str) -> Tuple[str, List[MorphEdit]]:
        edits: List[MorphEdit] = []

        def repl(m: re.Match) -> str:
            expr = m.group(1).strip()
            before = m.group(0)
            after = f"print({expr});"
            edits.append(MorphEdit(m.start(), m.end(), before, after, "expand_print_directive"))
            return after

        new = self._re_print.sub(repl, text)
        return new, edits

    def _expand_do_array(self, text: str) -> Tuple[str, List[MorphEdit]]:
        edits: List[MorphEdit] = []

        def repl(m: re.Match) -> str:
            inner = m.group(1).rstrip()
            before = m.group(0)
            after = f"do {{ {inner} }};"
            edits.append(MorphEdit(m.start(), m.end(), before, after, "expand_do_array"))
            return after

        new = self._re_do_array.sub(repl, text)
        return new, edits

    def _convert_label_colon_to_assignment(self, text: str) -> Tuple[str, List[MorphEdit]]:
        edits: List[MorphEdit] = []

        def repl(m: re.Match) -> str:
            ident = m.group(1)
            expr = m.group(2).rstrip()
            before = m.group(0)
            after = f"{ident} = {expr};"
            edits.append(MorphEdit(m.start(), m.end(), before, after, "convert_label_colon_to_assignment"))
            return after

        new = self._re_label_colon.sub(repl, text)
        return new, edits

    def _normalize_if_then(self, text: str) -> Tuple[str, List[MorphEdit]]:
        edits: List[MorphEdit] = []

        def repl(m: re.Match) -> str:
            cond = m.group(1).strip()
            before = m.group(0)
            after = f"if ({cond}) {{"
            edits.append(MorphEdit(m.start(), m.end(), before, after, "normalize_if_then"))
            return after

        new = self._re_if_then.sub(repl, text)
        return new, edits

    def _normalize_while_not(self, text: str) -> Tuple[str, List[MorphEdit]]:
        edits: List[MorphEdit] = []

        def repl(m: re.Match) -> str:
            cond = m.group(1).strip()
            before = m.group(0)
            after = f"while (not ({cond})) {{"
            edits.append(MorphEdit(m.start(), m.end(), before, after, "normalize_while_not"))
            return after

        new = self._re_while_not.sub(repl, text)
        return new, edits

    def _remove_semicolon_before_brace(self, text: str) -> Tuple[str, List[MorphEdit]]:
        new = re.sub(r";\s*}", "}", text)
        if new == text:
            return text, []
        return new, [MorphEdit(0, len(text), text, new, "remove_semicolon_before_brace")]

    def _remove_duplicate_semicolons(self, text: str) -> Tuple[str, List[MorphEdit]]:
        new = re.sub(r";{2,}", ";", text)
        if new == text:
            return text, []
        return new, [MorphEdit(0, len(text), text, new, "remove_duplicate_semicolons")]

    def _remove_empty_semicolon_lines(self, text: str) -> Tuple[str, List[MorphEdit]]:
        new = re.sub(r"(?m)^[ \t]*;\s*$\n?", "", text)
        if new == text:
            return text, []
        return new, [MorphEdit(0, len(text), text, new, "remove_empty_semicolon_lines")]

    def _ensure_statement_semicolons(self, text: str) -> Tuple[str, List[MorphEdit]]:
        lines = text.splitlines(keepends=True)
        edits: List[MorphEdit] = []
        changed = False
        offset = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                offset += len(line)
                continue
            if stripped.endswith(";") or stripped.endswith("{") or stripped.endswith("}") or stripped.endswith(":"):
                offset += len(line)
                continue
            if re.match(r"^(func\b|if\b|while\b|else\b|for\b|quarantine\b|return\b|break\b|continue\b|import\b|@)", stripped):
                offset += len(line)
                continue
            if re.search(r"=\s*|^[A-Za-z_][\w]*\s*\(|^[A-Za-z_][\w]*\s*$", stripped):
                if line.endswith("\n"):
                    new_line = line[:-1] + ";" + "\n"
                else:
                    new_line = line + ";"
                edits.append(MorphEdit(offset, offset + len(line), line, new_line, "ensure_statement_semicolons"))
                lines[i] = new_line
                changed = True
                offset += len(new_line)
            else:
                offset += len(line)
        if not changed:
            return text, []
        new = "".join(lines)
        return new, edits

    def _normalize_quarantine_semicolon(self, text: str) -> Tuple[str, List[MorphEdit]]:
        edits: List[MorphEdit] = []

        def repl(m: re.Match) -> str:
            inner = m.group(1)
            before = m.group(0)
            after = inner + ";"
            if before != after:
                edits.append(MorphEdit(m.start(), m.end(), before, after, "normalize_quarantine_semicolon"))
            return after

        new = self._re_quarantine.sub(repl, text)
        return new, edits

    def _sort_top_level_imports(self, text: str) -> Tuple[str, List[MorphEdit]]:
        lines = text.splitlines(keepends=True)
        out_lines: List[str] = []
        i = 0
        edits: List[MorphEdit] = []
        changed = False
        while i < len(lines):
            if lines[i].lstrip().startswith("import "):
                start = i
                imports = []
                while i < len(lines) and lines[i].lstrip().startswith("import "):
                    imports.append(lines[i].strip())
                    i += 1
                sorted_imports = sorted(dict.fromkeys(imports))
                if imports != sorted_imports:
                    changed = True
                    orig = "".join(lines[start:i])
                    after = "".join(s + "\n" for s in sorted_imports)
                    edits.append(MorphEdit(sum(len(x) for x in lines[:start]), sum(len(x) for x in lines[:i]), orig, after, "sort_top_level_imports"))
                    out_lines.append(after)
                else:
                    out_lines.extend(lines[start:i])
            else:
                out_lines.append(lines[i])
                i += 1
        if not changed:
            return text, []
        new = "".join(out_lines)
        return new, edits

    def _normalize_indentation(self, text: str) -> Tuple[str, List[MorphEdit]]:
        lines = text.splitlines(keepends=True)
        edits: List[MorphEdit] = []
        changed = False
        for idx, line in enumerate(lines):
            m = re.match(r"^([ \t]+)(.*)$", line)
            if not m:
                continue
            leading, rest = m.group(1), m.group(2)
            spaces = leading.replace("\t", " " * 8)
            level = len(spaces) // self.indent_size
            new_lead = " " * (level * self.indent_size)
            new_line = new_lead + rest
            if new_line != line:
                offset = sum(len(x) for x in lines[:idx])
                edits.append(MorphEdit(offset, offset + len(line), line, new_line, "normalize_indentation"))
                lines[idx] = new_line
                changed = True
        if not changed:
            return text, []
        new = "".join(lines)
        return new, edits

    def _collapse_trailing_newlines(self, text: str) -> Tuple[str, List[MorphEdit]]:
        if not text:
            return text, []
        if text.endswith("\n"):
            new = re.sub(r"\n{2,}\Z", "\n", text)
        else:
            new = text + "\n"
        if new == text:
            return text, []
        return new, [MorphEdit(0, len(text), text, new, "collapse_trailing_newlines")]

    def _fix_unbalanced_braces(self, text: str) -> Tuple[str, List[MorphEdit]]:
        edits: List[MorphEdit] = []
        opens = text.count("{")
        closes = text.count("}")
        if opens == closes:
            return text, []
        if opens > closes:
            missing = opens - closes
            addition = ("\n" + ("}" * missing) + "\n")
            new = text + addition
            edits.append(MorphEdit(len(text), len(new), "", addition, f"fix_unbalanced_braces_add_{missing}"))
            return new, edits
        else:
            unmatched = closes - opens
            new = text
            removed = 0
            for _ in range(unmatched):
                idx = new.rfind("}")
                if idx == -1:
                    break
                new = new[:idx] + new[idx+1:]
                removed += 1
            if removed > 0:
                edits.append(MorphEdit(0, len(text), text, new, f"fix_unbalanced_braces_remove_{removed}"))
            return new, edits

    # -------------------------
    # Utilities: diff, source-map, file ops
    # -------------------------
    def diff(self, original: str, transformed: str, filename: str = "<source>") -> str:
        o_lines = original.splitlines(keepends=True)
        t_lines = transformed.splitlines(keepends=True)
        ud = difflib.unified_diff(o_lines, t_lines, fromfile=filename, tofile=filename + ".morphed", lineterm="")
        return "".join(line + "\n" for line in ud)

    def _compute_basic_sourcemap(self, original: str, transformed: str) -> List[Tuple[int, int]]:
        o_lines = original.splitlines()
        t_lines = transformed.splitlines()
        sm: List[Tuple[int, int]] = []
        matcher = difflib.SequenceMatcher(a=o_lines, b=t_lines)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for oi, nj in zip(range(i1, i2), range(j1, j2)):
                    sm.append((oi + 1, nj + 1))
        return sm

    def apply_to_file(self, src_path: str, out_path: Optional[str] = None, overwrite: bool = False,
                      make_backup: bool = True, keep_backups: int = 3, dry_run: bool = False, verbose: bool = False) -> MorphResult:
        p = Path(src_path)
        text = p.read_text(encoding="utf-8")
        result = self.morph(text)
        if dry_run:
            if verbose:
                LOG.info("[dry-run] %s -> %d edits", src_path, len(result.edits))
            return result
        target = p if overwrite and out_path is None else Path(out_path or f"{src_path}.morphed.ix")
        if overwrite:
            if make_backup:
                ts = time.strftime("%Y%m%d%H%M%S")
                bak = p.with_name(p.stem + f".{ts}.bak{p.suffix}")
                try:
                    p.replace(bak)
                except Exception:
                    shutil.copy2(p, bak)
                backups = sorted(p.parent.glob(p.stem + ".*.bak" + p.suffix), key=lambda x: x.stat().st_mtime if x.exists() else 0)
                while len(backups) > keep_backups:
                    try:
                        backups[0].unlink()
                        backups.pop(0)
                    except Exception:
                        break
            target = p
        if verbose and result.edits:
            LOG.info("morph %s -> %s (%d edits)", src_path, str(target), len(result.edits))
        target.write_text(result.transformed, encoding="utf-8")
        return result

    def apply_to_dir(self, dir_path: str, pattern: str = "*.ix", recursive: bool = True,
                     max_workers: Optional[int] = None, inplace: bool = False, verbose: bool = False,
                     dry_run: bool = False, keep_backups: int = 3) -> Dict[str, MorphResult]:
        p = Path(dir_path)
        if max_workers is None:
            max_workers = max(1, multiprocessing.cpu_count() - 1)
        files = list(p.rglob(pattern) if recursive else p.glob(pattern))
        results: Dict[str, MorphResult] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {}
            for f in files:
                fut = ex.submit(self.apply_to_file, str(f), None, inplace, True, keep_backups, dry_run, verbose)
                futures[fut] = f
            for fut in concurrent.futures.as_completed(futures):
                f = futures[fut]
                try:
                    res = fut.result()
                    results[str(f)] = res
                except Exception as e:
                    LOG.exception("failed morph %s: %s", f, e)
        return results

    def preview_diff(self, src_path: str) -> str:
        p = Path(src_path)
        orig = p.read_text(encoding="utf-8")
        res = self.morph(orig)
        if orig == res.transformed:
            return ""
        return self.diff(orig, res.transformed, filename=src_path)

    # -------------------------
    # Watcher
    # -------------------------
    def watch(self, path: str, callback: Callable[[str, MorphResult], None], interval: float = 0.6):
        tracked: Dict[Path, float] = {}
        p = Path(path)
        if p.is_file():
            tracked[p] = p.stat().st_mtime
        else:
            for f in p.rglob("*.ix"):
                tracked[f] = f.stat().st_mtime
        try:
            while True:
                time.sleep(interval)
                # check existing
                for f in list(tracked.keys()):
                    try:
                        m = f.stat().st_mtime
                        if m != tracked[f]:
                            res = self.apply_to_file(str(f), overwrite=False, out_path=None)
                            callback(str(f), res)
                            tracked[f] = m
                    except FileNotFoundError:
                        tracked.pop(f, None)
                # detect new
                if p.is_dir():
                    for f in p.rglob("*.ix"):
                        if f not in tracked:
                            tracked[f] = f.stat().st_mtime
        except KeyboardInterrupt:
            LOG.info("Watcher stopped.")

# -------------------------
# Unit tests
# -------------------------


def run_unit_tests(verbose: bool = True) -> bool:
    sm = SyntaxMorph(indent_size=2)
    tests: List[Tuple[str, str]] = [
        ('print: "Hello";', 'print("Hello");'),
        ('do: [a = 1; b = 2;];', 'do { a = 1; b = 2; };'),
        ('data: [1, 2, 3];', 'data = [1, 2, 3];'),
        ('x = 1\ny = 2\n', 'x = 1;\ny = 2;'),
        ('if (x) {\n doSomething();\n};', 'if (x) {\n doSomething();\n}'),
        ('if x > 0 then { print: "ok"; };', 'if (x > 0) { print("ok"); };'),
    ]
    all_pass = True
    for i, (inp, expected) in enumerate(tests, 1):
        res = sm.morph(inp)
        out = res.transformed.strip()
        ok = expected.strip() in out
        if verbose:
            print(f"Test {i}: {'PASS' if ok else 'FAIL'}")
            if not ok:
                print(" Input:", inp)
                print(" Output:", out)
                print(" Expected contains:", expected)
                print(" Edits:", [asdict(e) for e in res.edits])
        all_pass = all_pass and ok

    # braces
    inp = "func foo() { if (x) { do(); }\n"
    res = sm.morph(inp)
    if verbose:
        print("Brace result:", repr(res.transformed))
    if res.transformed.count("{") != res.transformed.count("}"):
        print("Brace test FAIL")
        all_pass = False
    else:
        if verbose:
            print("Brace test PASS")
    return all_pass

# -------------------------
# Macro overlay integration (optional)
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
                    result, diagnostics = source, []
            if isinstance(result, dict) and "transformed" in result:
                return result["transformed"], diagnostics
            return result, diagnostics
    except Exception:
        LOG.debug("macro_overlay not available or failed")
    return source, []

# -------------------------
# CLI
# -------------------------


def _cli(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="instryx_syntax_morph", description="Instryx syntax morphing tool (supreme boosters)")
    parser.add_argument("path", nargs="?", help="File or directory to process")
    parser.add_argument("--inplace", action="store_true", help="Write changes in-place")
    parser.add_argument("--out", help="Write transformed content to path (file).")
    parser.add_argument("--dir", help="Process directory recursively")
    parser.add_argument("--diff", action="store_true", help="Print unified diff instead of writing")
    parser.add_argument("--watch", action="store_true", help="Watch file/dir for changes")
    parser.add_argument("--test", action="store_true", help="Run unit tests")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Do not write files; just show diff/edits")
    parser.add_argument("--edits-json", help="Write edits list JSON to given path")
    parser.add_argument("--keep-backups", type=int, default=3, help="Backup rotation count when writing in-place")
    parser.add_argument("--indent-size", type=int, default=2, help="Spaces per indent level for normalization")
    parser.add_argument("--concurrency", type=int, default=max(1, multiprocessing.cpu_count() - 1), help="Workers for directory processing")
    parser.add_argument("--apply-macros", action="store_true", help="Apply macro_overlay expansion if available")
    args = parser.parse_args(argv)

    if args.verbose:
        LOG.setLevel(logging.DEBUG)
        LOG.debug("Verbose mode enabled")

    sm = SyntaxMorph(indent_size=args.indent_size)

    if args.test:
        ok = run_unit_tests(verbose=args.verbose)
        return 0 if ok else 2

    if not args.path and not args.dir:
        parser.print_help()
        return 1

    if args.dir:
        results = sm.apply_to_dir(args.dir, recursive=True, max_workers=args.concurrency, inplace=args.inplace, verbose=args.verbose, dry_run=args.dry_run, keep_backups=args.keep_backups)
        if args.verbose:
            LOG.info("Processed %d files.", len(results))
        if args.edits_json:
            out = {k: [asdict(e) for e in r.edits] for k, r in results.items()}
            Path(args.edits_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        return 0

    path = args.path
    p = Path(path)
    if args.watch:
        def cb(file_path, res):
            print(f"[watch] {file_path} morphed ({len(res.edits)} edits).")
        sm.watch(path, cb)
        return 0

    orig = p.read_text(encoding="utf-8")
    src = orig
    if args.apply_macros:
        src, diag = expand_macros_if_available(src, filename=str(p))
        if args.verbose and diag:
            LOG.debug("Macro diagnostics: %s", diag)

    res = sm.morph(src)

    if res.transformed == src:
        print("No changes.")
        if args.edits_json:
            Path(args.edits_json).write_text(json.dumps({str(p): []}, indent=2), encoding="utf-8")
        return 0

    if args.diff or args.dry_run:
        d = sm.diff(orig, res.transformed, filename=str(p))
        print(d)
        if args.edits_json:
            Path(args.edits_json).write_text(json.dumps({str(p): [asdict(e) for e in res.edits]}, indent=2), encoding="utf-8")
        return 0

    if args.out:
        Path(args.out).write_text(res.transformed, encoding="utf-8")
        print(f"Wrote {args.out}")
        if args.edits_json:
            Path(args.edits_json).write_text(json.dumps({str(p): [asdict(e) for e in res.edits]}, indent=2), encoding="utf-8")
        return 0

    if args.inplace:
        ts = time.strftime("%Y%m%d%H%M%S")
        bak = p.with_name(p.stem + f".{ts}.bak{p.suffix}")
        try:
            p.replace(bak)
        except Exception:
            shutil.copy2(p, bak)
        p.write_text(res.transformed, encoding="utf-8")
        backups = sorted(p.parent.glob(p.stem + ".*.bak" + p.suffix), key=lambda x: x.stat().st_mtime if x.exists() else 0)
        while len(backups) > args.keep_backups:
            try:
                backups[0].unlink()
                backups.pop(0)
            except Exception:
                break
        print(f"Wrote in-place {p}")
        if args.edits_json:
            Path(args.edits_json).write_text(json.dumps({str(p): [asdict(e) for e in res.edits]}, indent=2), encoding="utf-8")
        return 0

    # print transformed
    print(res.transformed)
    if args.edits_json:
        Path(args.edits_json).write_text(json.dumps({str(p): [asdict(e) for e in res.edits]}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        rc = _cli()
        raise SystemExit(rc)
    except SystemExit:
        raise
    except Exception:
        LOG.exception("Fatal error in instryx_syntax_morph")
        raise

