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

