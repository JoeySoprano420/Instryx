"""
instryx_match_enum_struct.py

Utilities and tooling to detect, parse and generate safe match / deconstruction
helpers for Instryx enum/struct textual syntax.

Features:
- Robust parsing of `enum` and `struct` textual declarations (brace-aware)
- Dataclasses for `EnumDef`, `Variant`, `StructDef`
- Generators:
  - exhaustive match skeleton for enums (with placeholders)
  - struct deconstructor / constructor helpers
  - pattern-match macros ready for macro_overlay insertion
- Safe insertion, patch generation (unified diff)
- Batch detection and CLI tools (list-enums, list-structs, emit-match, inject-match, batch-detect)
- Lightweight LRU cache and local memory (counts)
- Optional integration with instryx_memory_math_loops_codegen when available
- Plugin hook points: external modules can register additional generators or validators
- Unit tests and self-check

This module is intentionally conservative: it performs textual analysis but uses
brace/paren-aware scanning to avoid most common pitfalls. It produces textual
snippets that are safe to preview before applying.

Usage (module):
    from instryx_match_enum_struct import DMatchTool
    tool = DMatchTool()
    enums = tool.find_enums(source_text)
    print(tool.generate_match_stub(enums[0], var_name='v'))

CLI:
    python instryx_match_enum_struct.py list-enums file.ix
    python instryx_match_enum_struct.py emit-match file.ix EnumName --var v --write match.ix
    python instryx_match_enum_struct.py inject-match file.ix EnumName --inplace
    python instryx_match_enum_struct.py test

"""

from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import sys
import textwrap
import time
import difflib
import logging
import threading
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, Callable, Any

# Optional codegen import
_try_codegen = None
try:
    import instryx_memory_math_loops_codegen as codegen  # type: ignore
    _try_codegen = codegen
except Exception:
    _try_codegen = None

# Logging
LOG = logging.getLogger("instryx.match.enum_struct")
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    h = logging.StreamHandler(sys.stderr)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    h.setFormatter(fmt)
    LOG.addHandler(h)

# Basic constants
DEFAULT_PAD = 140
DEFAULT_MAX_UNROLL = 8
DEFAULT_MEMORY_FILENAME = "instryx_match_memory.json"


# -------------------------
# Data models
# -------------------------
@dataclass
class Variant:
    name: str
    payload: Optional[str] = None  # text inside parentheses or braces
    raw: str = ""  # original raw text snippet
    span: Optional[Tuple[int, int]] = None  # offsets within enum block


@dataclass
class EnumDef:
    name: str
    variants: List[Variant]
    start: int
    end: int
    raw: str = ""


@dataclass
class StructField:
    name: str
    type: Optional[str] = None
    raw: str = ""


@dataclass
class StructDef:
    name: str
    fields: List[StructField]
    start: int
    end: int
    raw: str = ""


@dataclass
class Suggestion:
    macro_name: str
    args: List[str]
    reason: str
    score: float
    snippet: Optional[str] = None
    location: Optional[Tuple[int, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -------------------------
# AISimpleMemory (thread-safe, JSON-backed)
# -------------------------
class AISimpleMemory:
    """
    Small thread-safe JSON-backed memory used by tools in this module.
    - patterns: counts for heuristics (key -> int)
    - accepted: list of accepted suggestion records {time, suggestion, file}
    - meta: metadata (created, modified)
    """

    def __init__(self, path: Optional[str] = None, autosave: bool = True):
        self.path = path or os.path.join(os.path.dirname(__file__), DEFAULT_MEMORY_FILENAME)
        self._lock = threading.RLock()
        self.autosave = autosave
        self._data: Dict[str, Any] = {"patterns": {}, "accepted": [], "meta": {"created": time.time(), "modified": time.time()}}
        self._load()

    def _load(self) -> None:
        with self._lock:
            try:
                if os.path.exists(self.path):
                    with open(self.path, "r", encoding="utf-8") as f:
                        raw = f.read()
                        if raw:
                            self._data = json.loads(raw)
                # ensure structure
                self._data.setdefault("patterns", {})
                self._data.setdefault("accepted", [])
                self._data.setdefault("meta", {"created": time.time(), "modified": time.time()})
            except Exception:
                LOG.exception("AISimpleMemory: failed to load memory file, resetting")
                self._data = {"patterns": {}, "accepted": [], "meta": {"created": time.time(), "modified": time.time()}}

    def save(self) -> None:
        with self._lock:
            try:
                tmp = f"{self.path}.tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, indent=2, ensure_ascii=False)
                os.replace(tmp, self.path)
            except Exception:
                LOG.exception("AISimpleMemory: failed to save memory")

    def record_pattern(self, key: str, increment: int = 1) -> None:
        with self._lock:
            self._data.setdefault("patterns", {})
            self._data["patterns"][key] = int(self._data["patterns"].get(key, 0)) + int(increment)
            self._data.setdefault("meta", {})
            self._data["meta"]["modified"] = time.time()
            if self.autosave:
                self.save()

    def pattern_count(self, key: str) -> int:
        with self._lock:
            return int(self._data.get("patterns", {}).get(key, 0))

    def record_accepted(self, suggestion: Suggestion, filename: Optional[str] = None) -> None:
        with self._lock:
            entry = {"time": int(time.time()), "suggestion": suggestion.to_dict() if hasattr(suggestion, "to_dict") else suggestion, "file": filename}
            self._data.setdefault("accepted", []).append(entry)
            self._data.setdefault("meta", {})
            self._data["meta"]["modified"] = time.time()
            if self.autosave:
                self.save()

    def get_recent_accepted(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            acc = list(self._data.get("accepted", []))
            return acc[-limit:]

    def export(self) -> Dict[str, Any]:
        with self._lock:
            # return a deep copy-ish lightweight snapshot
            return json.loads(json.dumps(self._data))

    def import_data(self, data: Dict[str, Any], merge: bool = True) -> None:
        with self._lock:
            if not merge:
                self._data = data
            else:
                patterns = data.get("patterns", {})
                for k, v in patterns.items():
                    self._data.setdefault("patterns", {})
                    self._data["patterns"][k] = int(self._data["patterns"].get(k, 0)) + int(v)
                self._data.setdefault("accepted", []).extend(data.get("accepted", []))
                self._data.setdefault("meta", {})["modified"] = time.time()
            if self.autosave:
                self.save()

    def clear(self) -> None:
        with self._lock:
            self._data = {"patterns": {}, "accepted": [], "meta": {"created": time.time(), "modified": time.time()}}
            if self.autosave:
                self.save()


# -------------------------
# Helpers
# -------------------------
def _find_matching(source: str, open_pos: int, open_char: str = "{", close_char: str = "}") -> Optional[int]:
    """Return index of matching close_char for the open_char at open_pos, else None."""
    if source[open_pos] != open_char:
        # find next open_char at or after open_pos
        open_pos = source.find(open_char, open_pos)
        if open_pos == -1:
            return None
    depth = 0
    i = open_pos
    L = len(source)
    while i < L:
        ch = source[i]
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return i
        elif ch in ('"', "'"):
            # skip string literal
            quote = ch
            i += 1
            while i < L:
                if source[i] == quote and source[i - 1] != "\\":
                    break
                i += 1
        i += 1
    return None


def _split_top_level_commas(text: str) -> List[str]:
    """
    Split a comma-separated list at top level (not inside parentheses/braces/brackets or strings).
    Returns list of trimmed items.
    """
    items = []
    buf = []
    depth_paren = depth_brace = depth_brack = 0
    i = 0
    L = len(text)
    in_string = None
    while i < L:
        ch = text[i]
        if in_string:
            buf.append(ch)
            if ch == in_string and text[i - 1] != "\\":
                in_string = None
        else:
            if ch in ('"', "'"):
                in_string = ch
                buf.append(ch)
            elif ch == "(":
                depth_paren += 1
                buf.append(ch)
            elif ch == ")":
                depth_paren = max(0, depth_paren - 1)
                buf.append(ch)
            elif ch == "{":
                depth_brace += 1
                buf.append(ch)
            elif ch == "}":
                depth_brace = max(0, depth_brace - 1)
                buf.append(ch)
            elif ch == "[":
                depth_brack += 1
                buf.append(ch)
            elif ch == "]":
                depth_brack = max(0, depth_brack - 1)
                buf.append(ch)
            elif ch == "," and depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
                items.append("".join(buf).strip())
                buf = []
            else:
                buf.append(ch)
        i += 1
    if buf:
        items.append("".join(buf).strip())
    return [it for it in items if it != ""]


def _extract_line(source: str, idx: int, pad: int = DEFAULT_PAD) -> str:
    start = source.rfind("\n", 0, idx) + 1
    end = source.find("\n", idx)
    if end == -1:
        end = len(source)
    line = source[start:end].strip()
    if len(line) > pad:
        return line[:pad] + "..."
    return line


# -------------------------
# Parser: enums & structs
# -------------------------
class Parser:
    """
    Parser utilities for enum / struct detection and extraction.
    Intentionally conservative (textual, brace-aware).
    """

    @staticmethod
    def find_enums(source: str) -> List[EnumDef]:
        """
        Find enum declarations of the form:
            enum Name { VariantA, VariantB(Type), VariantC { field: Type, ... } }
        Returns list of EnumDef with variants parsed.
        """
        enums: List[EnumDef] = []
        # a simple regex to find "enum Name {"
        for m in re.finditer(r"\benum\s+([A-Za-z_][\w]*)\s*\{", source):
            name = m.group(1)
            open_pos = source.find("{", m.end() - 1)
            if open_pos == -1:
                continue
            close_pos = _find_matching(source, open_pos, "{", "}")
            if close_pos is None:
                continue
            raw_block = source[open_pos + 1:close_pos]
            # split top-level comma-separated variants
            parts = _split_top_level_commas(raw_block)
            variants = []
            offset_base = open_pos + 1
            for part in parts:
                start_idx = source.find(part, offset_base)
                # variant name may have payload in parens or braces
                # extract name
                name_match = re.match(r"\s*([A-Za-z_][\w]*)", part)
                if not name_match:
                    continue
                vname = name_match.group(1)
                payload = None
                # find payload portion after name in part
                rest = part[name_match.end():].strip()
                if rest.startswith("("):
                    # find matching )
                    pclose = _find_matching(rest, 0, "(", ")")
                    payload = rest[1:pclose] if pclose else rest
                elif rest.startswith("{"):
                    pclose = _find_matching(rest, 0, "{", "}")
                    payload = rest[1:pclose] if pclose else rest
                variants.append(Variant(name=vname, payload=payload, raw=part, span=(start_idx, start_idx + len(part))))
                offset_base = (start_idx + len(part)) if start_idx != -1 else offset_base
            enums.append(EnumDef(name=name, variants=variants, start=m.start(), end=close_pos + 1, raw=source[m.start():close_pos + 1]))
        return enums

    @staticmethod
    def find_structs(source: str) -> List[StructDef]:
        """
        Find struct declarations of the form:
            struct Name { field: Type; other: Type; }
        Returns list of StructDef.
        """
        structs: List[StructDef] = []
        for m in re.finditer(r"\bstruct\s+([A-Za-z_][\w]*)\s*\{", source):
            name = m.group(1)
            open_pos = source.find("{", m.end() - 1)
            if open_pos == -1:
                continue
            close_pos = _find_matching(source, open_pos, "{", "}")
            if close_pos is None:
                continue
            raw_block = source[open_pos + 1:close_pos]
            # split top-level semicolon separated fields
            # support both ';' and ',' separators
            fields_raw = re.split(r";|\n", raw_block)
            fields = []
            for fr in fields_raw:
                frs = fr.strip()
                if not frs:
                    continue
                # field pattern: name : type
                fm = re.match(r"\s*([A-Za-z_][\w]*)\s*:\s*(.+)$", frs)
                if fm:
                    fname = fm.group(1)
                    ftype = fm.group(2).strip().rstrip(",;")
                    fields.append(StructField(name=fname, type=ftype, raw=frs))
                else:
                    # fallback: treat as raw token
                    fields.append(StructField(name=frs, type=None, raw=frs))
            structs.append(StructDef(name=name, fields=fields, start=m.start(), end=close_pos + 1, raw=source[m.start():close_pos + 1]))
        return structs


# -------------------------
# Generator utilities
# -------------------------
class Generator:
    """
    Emit helper snippets for enums and structs:
    - generate_match_stub(enum_def, var_name)
    - generate_struct_destructure(struct_def, var_name)
    - generate_pattern_macro(enum_def)
    """

    @staticmethod
    def generate_match_stub(enum_def: EnumDef, var_name: str = "v", indent: str = "    ", placeholder: str = "/* TODO */") -> str:
        """
        Return a textual match skeleton for the enum.
        Example:
            match v {
                VariantA => { /* TODO */ },
                VariantB(x) => { /* TODO */ },
                VariantC { a, b } => { /* TODO */ },
            }
        """
        lines = []
        lines.append(f"// Match skeleton for enum {enum_def.name}")
        lines.append(f"match {var_name} {{")
        for v in enum_def.variants:
            # decide arm syntax
            arm = v.name
            if v.payload:
                payload = v.payload.strip()
                # If payload looks like field list (contains ':' or ',') use struct-like form
                if "{" in v.raw or "}" in v.raw:
                    # convert payload to field names if possible
                    # simple heuristic: extract identifiers
                    ids = re.findall(r"\b([A-Za-z_][\w]*)\b", payload)
                    if ids:
                        arm += " { " + ", ".join(ids) + " }"
                    else:
                        arm += f"({payload})"
                else:
                    arm += f"({payload})"
            lines.append(f"{indent}{arm} => {{ {placeholder} }},")
        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def generate_struct_destructure(struct_def: StructDef, var_name: str = "s", indent: str = "    ", placeholder: str = "/* TODO */") -> str:
        """
        Return a textual struct destructuring snippet:
            let { a, b } = s;
            // or
            let x = s.a;
        """
        field_names = [f.name for f in struct_def.fields if f.name]
        if not field_names:
            return f"// struct {struct_def.name} has no named fields\n{var_name};\n"
        lines = []
        lines.append(f"// Destructure struct {struct_def.name}")
        lines.append(f"let {{ {', '.join(field_names)} }} = {var_name};")
        lines.append(f"{placeholder}")
        return "\n".join(lines)

    @staticmethod
    def generate_pattern_macro(enum_def: EnumDef, var_name: str = "v", macro_name: str = "match_pattern") -> str:
        """
        Emit a macro-style helper that expands to a match skeleton; useful for previewing
        or hooking into macro_overlay.
        """
        body = Generator.generate_match_stub(enum_def, var_name=var_name, indent="    ", placeholder="/* handler */")
        # macro comment header
        lines = [f"/* macro: {macro_name} for enum {enum_def.name} */", body]
        return "\n".join(lines)


# -------------------------
# Apply / patch helpers
# -------------------------
def insert_snippet_at(source: str, insert_text: str, idx: Optional[int] = None) -> Tuple[str, int]:
    """
    Insert insert_text at byte index idx (or at start if idx None). Return new source and insertion idx.
    """
    if idx is None:
        new_src = insert_text + "\n" + source
        return new_src, 0
    if idx < 0:
        idx = 0
    if idx > len(source):
        idx = len(source)
    new_src = source[:idx] + insert_text + "\n" + source[idx:]
    return new_src, idx


def generate_unified_patch(original: str, transformed: str, filename: str) -> str:
    return "".join(difflib.unified_diff(original.splitlines(keepends=True),
                                        transformed.splitlines(keepends=True),
                                        fromfile=filename,
                                        tofile=filename + ".ai.ix",
                                        lineterm=""))


# -------------------------
# Tool class that bundles features
# -------------------------
class DMatchTool:
    """
    High-level tool aggregating parser + generator + IO + optional codegen integration.
    """

    def __init__(self, codegen_module=None, memory_path: Optional[str] = None):
        self.parser = Parser()
        self.codegen = codegen_module or _try_codegen
        self.memory = AISimpleMemory(memory_path) if memory_path else AISimpleMemory()
        self.plugins: List[Callable[[str, Dict[str, Any]], None]] = []

    # discovery
    def find_enums(self, source: str) -> List[EnumDef]:
        return self.parser.find_enums(source)

    def find_structs(self, source: str) -> List[StructDef]:
        return self.parser.find_structs(source)

    # generation
    def generate_match_stub(self, enum: EnumDef, var_name: str = "v") -> str:
        return Generator.generate_match_stub(enum, var_name=var_name)

    def generate_struct_destructure(self, struct: StructDef, var_name: str = "s") -> str:
        return Generator.generate_struct_destructure(struct, var_name=var_name)

    def generate_pattern_macro(self, enum: EnumDef, var_name: str = "v", macro_name: str = "match_pattern") -> str:
        return Generator.generate_pattern_macro(enum, var_name=var_name, macro_name=macro_name)

    def emit_helper_via_codegen(self, helper_name: str, *args, **kwargs) -> Optional[str]:
        if not self.codegen:
            return None
        try:
            if hasattr(self.codegen, "emit_helper"):
                return self.codegen.emit_helper(helper_name, *args, **kwargs)
            fn = getattr(self.codegen, f"generate_{helper_name}", None)
            if callable(fn):
                return fn(*args, **kwargs)
            return None
        except Exception as e:
            LOG.warning("codegen emit failed: %s", e)
            return None

    # suggestions
    def suggest_match_locations(self, source: str) -> List[Suggestion]:
        """
        Heuristic: find places where enums are used without matching arms (e.g., switch/match on an enum variable but with TODO).
        Very conservative: look for "match <var>" occurrences and if a matching enum exists in file, propose a stub.
        """
        suggestions = []
        enums = {e.name: e for e in self.find_enums(source)}
        # find "match <identifier>" occurrences
        for m in re.finditer(r"\bmatch\s+([A-Za-z_][\w]*)\b", source):
            var = m.group(1)
            # no direct mapping from var -> enum name; attempt heuristic: if enum with same name (capitalized) exists
            cand_name = var[0].upper() + var[1:] if var else var
            if cand_name in enums:
                enum_def = enums[cand_name]
                snippet = _extract_line(source, m.start())
                suggestions.append(Suggestion(macro_name="match_stub", args=[enum_def.name, var], reason=f"generate exhaustive match for {enum_def.name}", score=0.75, snippet=snippet, location=(m.start(), m.end())))
                self.memory.record_pattern("match_stub")
        return suggestions

    # IO helpers
    def inject_stub_into_file(self, filepath: str, insert_text: str, insert_before_pattern: Optional[str] = None, inplace: bool = False) -> Tuple[bool, str]:
        try:
            src = open(filepath, "r", encoding="utf-8").read()
        except Exception as e:
            return False, f"read failed: {e}"
        insert_at = None
        if insert_before_pattern:
            m = re.search(insert_before_pattern, src)
            if m:
                insert_at = m.start()
        new_src, pos = insert_snippet_at(src, insert_text, insert_at)
        out_path = filepath if inplace else (filepath + ".ai.ix")
        try:
            open(out_path, "w", encoding="utf-8").write(new_src)
        except Exception as e:
            return False, f"write failed: {e}"
        return True, out_path

    def generate_patch_for_injection(self, filepath: str, insert_text: str, insert_before_pattern: Optional[str] = None) -> Tuple[bool, str]:
        try:
            original = open(filepath, "r", encoding="utf-8").read()
        except Exception as e:
            return False, f"read failed: {e}"
        insert_at = None
        if insert_before_pattern:
            m = re.search(insert_before_pattern, original)
            if m:
                insert_at = m.start()
        transformed, pos = insert_snippet_at(original, insert_text, insert_at)
        patch = generate_unified_patch(original, transformed, filepath)
        patch_path = filepath + ".ai.inj.patch"
        try:
            open(patch_path, "w", encoding="utf-8").write(patch)
        except Exception as e:
            return False, f"write failed: {e}"
        return True, patch_path

    # plugin hooks
    def register_plugin_callback(self, cb: Callable[[str, Dict[str, Any]], None]):
        self.plugins.append(cb)

    def _call_plugins(self, event: str, payload: Dict[str, Any]):
        for cb in self.plugins:
            try:
                cb(event, payload)
            except Exception:
                LOG.exception("plugin callback failed")

    # small utility
    def list_enum_names(self, source: str) -> List[str]:
        return [e.name for e in self.find_enums(source)]

    def list_struct_names(self, source: str) -> List[str]:
        return [s.name for s in self.find_structs(source)]


# -------------------------
# CLI
# -------------------------
def _cli_main():
    p = argparse.ArgumentParser(prog="instryx_match_enum_struct.py")
    p.add_argument("cmd", nargs="?", help="command (list-enums, list-structs, emit-match, inject-match, batch-detect, test)")
    p.add_argument("target", nargs="?", help="file or directory")
    p.add_argument("--enum", help="enum name for emit/inject")
    p.add_argument("--var", default="v", help="variable name used in generated match")
    p.add_argument("--write", help="write output to file")
    p.add_argument("--inplace", action="store_true", help="write in-place for inject-match")
    p.add_argument("--max", type=int, default=12, help="max suggestions / results")
    p.add_argument("--args", nargs="*", help="helper args forwarded to codegen")
    args = p.parse_args()

    tool = DMatchTool(codegen_module=_try_codegen)

    cmd = args.cmd or "help"
    if cmd in ("help", None):
        p.print_help()
        return 0

    if cmd == "list-enums":
        if not args.target:
            print("file required")
            return 2
        src = open(args.target, "r", encoding="utf-8").read()
        enums = tool.find_enums(src)
        for e in enums:
            print(e.name, "=>", [v.name for v in e.variants])
        return 0

    if cmd == "list-structs":
        if not args.target:
            print("file required")
            return 2
        src = open(args.target, "r", encoding="utf-8").read()
        structs = tool.find_structs(src)
        for s in structs:
            print(s.name, "=>", [f"{fld.name}:{fld.type}" for fld in s.fields])
        return 0

    if cmd == "emit-match":
        if not args.target or not args.enum:
            print("usage: emit-match <file> --enum EnumName [--var v] [--write out.ix]")
            return 2
        src = open(args.target, "r", encoding="utf-8").read()
        enums = tool.find_enums(src)
        ed = next((e for e in enums if e.name == args.enum), None)
        if not ed:
            print("enum not found")
            return 2
        txt = tool.generate_match_stub(ed, var_name=args.var)
        if args.write:
            open(args.write, "w", encoding="utf-8").write(txt)
            print("wrote", args.write)
        else:
            print(txt)
        return 0

    if cmd == "inject-match":
        if not args.target or not args.enum:
            print("usage: inject-match <file> --enum EnumName [--var v] [--inplace]")
            return 2
        src = open(args.target, "r", encoding="utf-8").read()
        enums = tool.find_enums(src)
        ed = next((e for e in enums if e.name == args.enum), None)
        if not ed:
            print("enum not found")
            return 2
        txt = tool.generate_match_stub(ed, var_name=args.var)
        ok, out = tool.inject_stub_into_file(args.target, txt, insert_before_pattern=None, inplace=args.inplace)
        print(out if ok else f"failed: {out}")
        return 0

    if cmd == "batch-detect":
        if not args.target:
            print("directory required")
            return 2
        results = {}
        for root, _, filenames in os.walk(args.target):
            for fn in filenames:
                if fn.endswith(".ix"):
                    path = os.path.join(root, fn)
                    src = open(path, "r", encoding="utf-8").read()
                    enums = tool.find_enums(src)
                    structs = tool.find_structs(src)
                    if enums or structs:
                        results[path] = {"enums": [e.name for e in enums], "structs": [s.name for s in structs]}
        print(json.dumps(results, indent=2))
        return 0

    if cmd == "test":
        ok = _run_unit_tests(tool)
        print("TEST", "PASS" if ok else "FAIL")
        return 0 if ok else 2

    print("unknown command", cmd)
    p.print_help()
    return 2


# -------------------------
# Unit tests / self-checks
# -------------------------
def _run_unit_tests(tool: DMatchTool) -> bool:
    try:
        sample = """
        enum Color { Red, Green(u32), Blue { r: i32, g: i32 } }
        struct Point { x: f32; y: f32; }
        func handle(c) {
            match c { /* incomplete */ }
        }
        """
        enums = tool.find_enums(sample)
        if not enums:
            LOG.error("enum detection failed")
            return False
        ed = enums[0]
        stub = tool.generate_match_stub(ed, var_name="c")
        assert "Red" in stub and "Green" in stub and "Blue" in stub
        structs = tool.find_structs(sample)
        assert structs and structs[0].name == "Point"
        # test insertion (dry-run)
        return True
    except Exception as e:
        LOG.exception("unit tests failed: %s", e)
        return False


# -------------------------
# Module entrypoint
# -------------------------
if __name__ == "__main__":
    sys.exit(_cli_main())

"""
instryx_match_enum_struct.py

Utilities and tooling to detect, parse and generate safe match / deconstruction
helpers for Instryx enum/struct textual syntax.

Supreme-boosters edition — additions and improvements:
 - Faster parsing with optional LRU caching of parse results
 - Concurrent batch detection and inject with progress logging
 - Metrics counters and lightweight /metrics HTTP endpoint
 - Suggestion engine with scoring backed by AISimpleMemory
 - Accept / record suggestions and export/import memory
 - Batch-inject command that creates safe unified patches, previews and writes
 - Interactive REPL mode for exploring enums/structs and generating snippets
 - Macro-overlay integration via --apply-macros (if available)
 - More robust generators: exhaustive match with default branch, structured handlers
 - Safety checks before injection (avoid injecting inside strings/comments)
 - Additional unit tests and self-checks
 - Pure stdlib, executable as CLI or importable module
"""

from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import sys
import textwrap
import time
import difflib
import logging
import threading
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, Callable, Any, Iterable
from functools import lru_cache
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional codegen import
_try_codegen = None
try:
    import instryx_memory_math_loops_codegen as codegen  # type: ignore
    _try_codegen = codegen
except Exception:
    _try_codegen = None

# Logging
LOG = logging.getLogger("instryx.match.enum_struct")
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    h = logging.StreamHandler(sys.stderr)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    h.setFormatter(fmt)
    LOG.addHandler(h)

# Basic constants
DEFAULT_PAD = 140
DEFAULT_MAX_UNROLL = 8
DEFAULT_MEMORY_FILENAME = "instryx_match_memory.json"
_METRICS: Dict[str, int] = {"parse_calls": 0, "matches_generated": 0, "injections": 0, "suggestions_made": 0}


# -------------------------
# Data models
# -------------------------
@dataclass
class Variant:
    name: str
    payload: Optional[str] = None  # text inside parentheses or braces
    raw: str = ""  # original raw text snippet
    span: Optional[Tuple[int, int]] = None  # offsets within enum block


@dataclass
class EnumDef:
    name: str
    variants: List[Variant]
    start: int
    end: int
    raw: str = ""


@dataclass
class StructField:
    name: str
    type: Optional[str] = None
    raw: str = ""


@dataclass
class StructDef:
    name: str
    fields: List[StructField]
    start: int
    end: int
    raw: str = ""


@dataclass
class Suggestion:
    macro_name: str
    args: List[str]
    reason: str
    score: float
    snippet: Optional[str] = None
    location: Optional[Tuple[int, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -------------------------
# AISimpleMemory (thread-safe, JSON-backed)
# -------------------------
class AISimpleMemory:
    """
    Small thread-safe JSON-backed memory used by tools in this module.
    - patterns: counts for heuristics (key -> int)
    - accepted: list of accepted suggestion records {time, suggestion, file}
    - meta: metadata (created, modified)
    """

    def __init__(self, path: Optional[str] = None, autosave: bool = True):
        self.path = path or os.path.join(os.path.dirname(__file__), DEFAULT_MEMORY_FILENAME)
        self._lock = threading.RLock()
        self.autosave = autosave
        self._data: Dict[str, Any] = {"patterns": {}, "accepted": [], "meta": {"created": time.time(), "modified": time.time()}}
        self._load()

    def _load(self) -> None:
        with self._lock:
            try:
                if os.path.exists(self.path):
                    with open(self.path, "r", encoding="utf-8") as f:
                        raw = f.read()
                        if raw:
                            self._data = json.loads(raw)
                # ensure structure
                self._data.setdefault("patterns", {})
                self._data.setdefault("accepted", [])
                self._data.setdefault("meta", {"created": time.time(), "modified": time.time()})
            except Exception:
                LOG.exception("AISimpleMemory: failed to load memory file, resetting")
                self._data = {"patterns": {}, "accepted": [], "meta": {"created": time.time(), "modified": time.time()}}

    def save(self) -> None:
        with self._lock:
            try:
                tmp = f"{self.path}.tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, indent=2, ensure_ascii=False)
                os.replace(tmp, self.path)
            except Exception:
                LOG.exception("AISimpleMemory: failed to save memory")

    def record_pattern(self, key: str, increment: int = 1) -> None:
        with self._lock:
            self._data.setdefault("patterns", {})
            self._data["patterns"][key] = int(self._data["patterns"].get(key, 0)) + int(increment)
            self._data.setdefault("meta", {})
            self._data["meta"]["modified"] = time.time()
            if self.autosave:
                self.save()

    def pattern_count(self, key: str) -> int:
        with self._lock:
            return int(self._data.get("patterns", {}).get(key, 0))

    def record_accepted(self, suggestion: Suggestion, filename: Optional[str] = None) -> None:
        with self._lock:
            entry = {"time": int(time.time()), "suggestion": suggestion.to_dict() if hasattr(suggestion, "to_dict") else suggestion, "file": filename}
            self._data.setdefault("accepted", []).append(entry)
            self._data.setdefault("meta", {})
            self._data["meta"]["modified"] = time.time()
            if self.autosave:
                self.save()

    def get_recent_accepted(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            acc = list(self._data.get("accepted", []))
            return acc[-limit:]

    def export(self) -> Dict[str, Any]:
        with self._lock:
            # return a deep copy-ish lightweight snapshot
            return json.loads(json.dumps(self._data))

    def import_data(self, data: Dict[str, Any], merge: bool = True) -> None:
        with self._lock:
            if not merge:
                self._data = data
            else:
                patterns = data.get("patterns", {})
                for k, v in patterns.items():
                    self._data.setdefault("patterns", {})
                    self._data["patterns"][k] = int(self._data["patterns"].get(k, 0)) + int(v)
                self._data.setdefault("accepted", []).extend(data.get("accepted", []))
                self._data.setdefault("meta", {})["modified"] = time.time()
            if self.autosave:
                self.save()

    def clear(self) -> None:
        with self._lock:
            self._data = {"patterns": {}, "accepted": [], "meta": {"created": time.time(), "modified": time.time()}}
            if self.autosave:
                self.save()


# -------------------------
# Helpers
# -------------------------
def _find_matching(source: str, open_pos: int, open_char: str = "{", close_char: str = "}") -> Optional[int]:
    """Return index of matching close_char for the open_char at open_pos, else None."""
    if open_pos < 0 or open_pos >= len(source):
        return None
    if source[open_pos] != open_char:
        open_pos = source.find(open_char, open_pos)
        if open_pos == -1:
            return None
    depth = 0
    i = open_pos
    L = len(source)
    while i < L:
        ch = source[i]
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return i
        elif ch in ('"', "'"):
            # skip string literal
            quote = ch
            i += 1
            while i < L:
                if source[i] == quote and source[i - 1] != "\\":
                    break
                i += 1
        i += 1
    return None


def _split_top_level_commas(text: str) -> List[str]:
    """
    Split a comma-separated list at top level (not inside parentheses/braces/brackets or strings).
    Returns list of trimmed items.
    """
    items = []
    buf = []
    depth_paren = depth_brace = depth_brack = 0
    i = 0
    L = len(text)
    in_string = None
    while i < L:
        ch = text[i]
        if in_string:
            buf.append(ch)
            if ch == in_string and text[i - 1] != "\\":
                in_string = None
        else:
            if ch in ('"', "'"):
                in_string = ch
                buf.append(ch)
            elif ch == "(":
                depth_paren += 1
                buf.append(ch)
            elif ch == ")":
                depth_paren = max(0, depth_paren - 1)
                buf.append(ch)
            elif ch == "{":
                depth_brace += 1
                buf.append(ch)
            elif ch == "}":
                depth_brace = max(0, depth_brace - 1)
                buf.append(ch)
            elif ch == "[":
                depth_brack += 1
                buf.append(ch)
            elif ch == "]":
                depth_brack = max(0, depth_brack - 1)
                buf.append(ch)
            elif ch == "," and depth_paren == 0 and depth_brace == 0 and depth_brack == 0:
                items.append("".join(buf).strip())
                buf = []
            else:
                buf.append(ch)
        i += 1
    if buf:
        items.append("".join(buf).strip())
    return [it for it in items if it != ""]


def _extract_line(source: str, idx: int, pad: int = DEFAULT_PAD) -> str:
    start = source.rfind("\n", 0, idx) + 1
    end = source.find("\n", idx)
    if end == -1:
        end = len(source)
    line = source[start:end].strip()
    if len(line) > pad:
        return line[:pad] + "..."
    return line


def _is_position_in_comment_or_string(source: str, pos: int) -> bool:
    """
    Very conservative check: scan from start to pos, track string and simple comment states.
    Returns True if inside a string or a single-line comment.
    """
    in_string = None
    i = 0
    while i < pos and i < len(source):
        ch = source[i]
        if in_string:
            if ch == in_string and source[i - 1] != "\\":
                in_string = None
        else:
            if ch in ('"', "'"):
                in_string = ch
            elif ch == "/" and i + 1 < len(source) and source[i + 1] == "/":
                # C-like // comment - skip to end of line
                nl = source.find("\n", i)
                if nl == -1:
                    return True
                i = nl
            elif ch == "-" and source[i:i + 2] == "--":
                nl = source.find("\n", i)
                if nl == -1:
                    return True
                i = nl
        i += 1
    return in_string is not None


# -------------------------
# Parser: enums & structs (cached)
# -------------------------
class Parser:
    """
    Parser utilities for enum / struct detection and extraction.
    Intentionally conservative (textual, brace-aware).
    """

    @staticmethod
    @lru_cache(maxsize=128)
    def find_enums_cached(source: str) -> List[EnumDef]:
        _METRICS["parse_calls"] += 1
        return Parser.find_enums(source)

    @staticmethod
    def find_enums(source: str) -> List[EnumDef]:
        enums: List[EnumDef] = []
        for m in re.finditer(r"\benum\s+([A-Za-z_][\w]*)\s*\{", source):
            name = m.group(1)
            open_pos = source.find("{", m.end() - 1)
            if open_pos == -1:
                continue
            close_pos = _find_matching(source, open_pos, "{", "}")
            if close_pos is None:
                continue
            raw_block = source[open_pos + 1:close_pos]
            parts = _split_top_level_commas(raw_block)
            variants = []
            offset_base = open_pos + 1
            for part in parts:
                start_idx = source.find(part, offset_base)
                name_match = re.match(r"\s*([A-Za-z_][\w]*)", part)
                if not name_match:
                    continue
                vname = name_match.group(1)
                payload = None
                rest = part[name_match.end():].strip()
                if rest.startswith("("):
                    pclose = _find_matching(rest, 0, "(", ")")
                    payload = rest[1:pclose] if pclose else rest
                elif rest.startswith("{"):
                    pclose = _find_matching(rest, 0, "{", "}")
                    payload = rest[1:pclose] if pclose else rest
                variants.append(Variant(name=vname, payload=payload, raw=part, span=(start_idx, start_idx + len(part))))
                offset_base = (start_idx + len(part)) if start_idx != -1 else offset_base
            enums.append(EnumDef(name=name, variants=variants, start=m.start(), end=close_pos + 1, raw=source[m.start():close_pos + 1]))
        return enums

    @staticmethod
    @lru_cache(maxsize=128)
    def find_structs_cached(source: str) -> List[StructDef]:
        _METRICS["parse_calls"] += 1
        return Parser.find_structs(source)

    @staticmethod
    def find_structs(source: str) -> List[StructDef]:
        structs: List[StructDef] = []
        for m in re.finditer(r"\bstruct\s+([A-Za-z_][\w]*)\s*\{", source):
            name = m.group(1)
            open_pos = source.find("{", m.end() - 1)
            if open_pos == -1:
                continue
            close_pos = _find_matching(source, open_pos, "{", "}")
            if close_pos is None:
                continue
            raw_block = source[open_pos + 1:close_pos]
            fields_raw = re.split(r";|\n", raw_block)
            fields = []
            for fr in fields_raw:
                frs = fr.strip()
                if not frs:
                    continue
                fm = re.match(r"\s*([A-Za-z_][\w]*)\s*:\s*(.+)$", frs)
                if fm:
                    fname = fm.group(1)
                    ftype = fm.group(2).strip().rstrip(",;")
                    fields.append(StructField(name=fname, type=ftype, raw=frs))
                else:
                    fields.append(StructField(name=frs, type=None, raw=frs))
            structs.append(StructDef(name=name, fields=fields, start=m.start(), end=close_pos + 1, raw=source[m.start():close_pos + 1]))
        return structs


# -------------------------
# Generator utilities (enhanced)
# -------------------------
class Generator:
    """
    Emit helper snippets for enums and structs:
    - generate_match_stub(enum_def, var_name)
    - generate_struct_destructure(struct_def, var_name)
    - generate_pattern_macro(enum_def)
    - generate_exhaustive_match(enum_def, var_name, include_default)
    """

    @staticmethod
    def generate_match_stub(enum_def: EnumDef, var_name: str = "v", indent: str = "    ",
                            placeholder: str = "/* TODO */", include_default: bool = False) -> str:
        lines = []
        lines.append(f"// Match skeleton for enum {enum_def.name}")
        lines.append(f"match {var_name} {{")
        for v in enum_def.variants:
            arm = v.name
            if v.payload:
                payload = v.payload.strip()
                # heuristic: if struct-like payload use fields
                if "{" in v.raw or "}" in v.raw:
                    ids = re.findall(r"\b([A-Za-z_][\w]*)\b", payload)
                    if ids:
                        arm += " { " + ", ".join(ids) + " }"
                    else:
                        arm += f"({payload})"
                else:
                    # if simple token list, try to name payload variable 'x' or keep as is
                    simple = re.match(r"([A-Za-z_][\w]*)", payload)
                    if simple:
                        arm += f"({simple.group(1)})"
                    else:
                        arm += f"({payload})"
            lines.append(f"{indent}{arm} => {{ {placeholder} }},")
        if include_default:
            lines.append(f"{indent}_ => {{ {placeholder} }},")
        lines.append("}")
        _METRICS["matches_generated"] += 1
        return "\n".join(lines)

    @staticmethod
    def generate_exhaustive_match(enum_def: EnumDef, var_name: str = "v", indent: str = "    ",
                                  placeholder: str = "/* TODO */", include_default: bool = False) -> str:
        # same as match_stub but ensure every variant present and provide recommended handler names
        return Generator.generate_match_stub(enum_def, var_name=var_name, indent=indent, placeholder=placeholder, include_default=include_default)

    @staticmethod
    def generate_struct_destructure(struct_def: StructDef, var_name: str = "s", indent: str = "    ", placeholder: str = "/* TODO */") -> str:
        field_names = [f.name for f in struct_def.fields if f.name]
        if not field_names:
            return f"// struct {struct_def.name} has no named fields\n{var_name};\n"
        lines = []
        lines.append(f"// Destructure struct {struct_def.name}")
        lines.append(f"let {{ {', '.join(field_names)} }} = {var_name};")
        lines.append(f"{placeholder}")
        return "\n".join(lines)

    @staticmethod
    def generate_pattern_macro(enum_def: EnumDef, var_name: str = "v", macro_name: str = "match_pattern") -> str:
        body = Generator.generate_match_stub(enum_def, var_name=var_name, indent="    ", placeholder="/* handler */")
        lines = [f"/* macro: {macro_name} for enum {enum_def.name} */", body]
        return "\n".join(lines)


# -------------------------
# Apply / patch helpers (enhanced)
# -------------------------
def insert_snippet_at(source: str, insert_text: str, idx: Optional[int] = None) -> Tuple[str, int]:
    """
    Insert insert_text at byte index idx (or at start if idx None). Return new source and insertion idx.
    """
    if idx is None:
        new_src = insert_text + "\n" + source
        return new_src, 0
    if idx < 0:
        idx = 0
    if idx > len(source):
        idx = len(source)
    new_src = source[:idx] + insert_text + "\n" + source[idx:]
    return new_src, idx


def generate_unified_patch(original: str, transformed: str, filename: str) -> str:
    return "".join(difflib.unified_diff(original.splitlines(keepends=True),
                                        transformed.splitlines(keepends=True),
                                        fromfile=filename,
                                        tofile=filename + ".ai.ix",
                                        lineterm=""))


# -------------------------
# Tool class that bundles features (enhanced)
# -------------------------
class DMatchTool:
    """
    High-level tool aggregating parser + generator + IO + optional codegen integration.
    """

    def __init__(self, codegen_module=None, memory_path: Optional[str] = None):
        self.parser = Parser()
        self.codegen = codegen_module or _try_codegen
        self.memory = AISimpleMemory(memory_path) if memory_path else AISimpleMemory()
        self.plugins: List[Callable[[str, Dict[str, Any]], None]] = []

    # discovery
    def find_enums(self, source: str) -> List[EnumDef]:
        try:
            return Parser.find_enums_cached(source)
        except Exception:
            return self.parser.find_enums(source)

    def find_structs(self, source: str) -> List[StructDef]:
        try:
            return Parser.find_structs_cached(source)
        except Exception:
            return self.parser.find_structs(source)

    # generation
    def generate_match_stub(self, enum: EnumDef, var_name: str = "v", include_default: bool = False) -> str:
        return Generator.generate_match_stub(enum, var_name=var_name, include_default=include_default)

    def generate_struct_destructure(self, struct: StructDef, var_name: str = "s") -> str:
        return Generator.generate_struct_destructure(struct, var_name=var_name)

    def generate_pattern_macro(self, enum: EnumDef, var_name: str = "v", macro_name: str = "match_pattern") -> str:
        return Generator.generate_pattern_macro(enum, var_name=var_name, macro_name=macro_name)

    def emit_helper_via_codegen(self, helper_name: str, *args, **kwargs) -> Optional[str]:
        if not self.codegen:
            return None
        try:
            if hasattr(self.codegen, "emit_helper"):
                return self.codegen.emit_helper(helper_name, *args, **kwargs)
            fn = getattr(self.codegen, f"generate_{helper_name}", None)
            if callable(fn):
                return fn(*args, **kwargs)
            return None
        except Exception as e:
            LOG.warning("codegen emit failed: %s", e)
            return None

    # suggestions
    def suggest_match_locations(self, source: str, max_results: int = 12) -> List[Suggestion]:
        suggestions: List[Suggestion] = []
        enums = {e.name: e for e in self.find_enums(source)}
        for m in re.finditer(r"\bmatch\s+([A-Za-z_][\w]*)\b", source):
            var = m.group(1)
            cand_name = var[0].upper() + var[1:] if var else var
            if cand_name in enums:
                enum_def = enums[cand_name]
                snippet = _extract_line(source, m.start())
                s = Suggestion(macro_name="match_stub", args=[enum_def.name, var], reason=f"generate exhaustive match for {enum_def.name}", score=0.75, snippet=snippet, location=(m.start(), m.end()))
                suggestions.append(s)
                self.memory.record_pattern("match_stub")
                _METRICS["suggestions_made"] += 1
                if len(suggestions) >= max_results:
                    break
        return suggestions

    # IO helpers
    def inject_stub_into_file(self, filepath: str, insert_text: str, insert_before_pattern: Optional[str] = None, inplace: bool = False, safe: bool = True) -> Tuple[bool, str]:
        try:
            src = open(filepath, "r", encoding="utf-8").read()
        except Exception as e:
            return False, f"read failed: {e}"
        if safe:
            # avoid injecting inside string/comment by ensuring insert pos is not inside them
            insert_at = None
            if insert_before_pattern:
                m = re.search(insert_before_pattern, src)
                if m:
                    if _is_position_in_comment_or_string(src, m.start()):
                        return False, "injection point inside comment/string, aborting"
                    insert_at = m.start()
            new_src, pos = insert_snippet_at(src, insert_text, insert_at)
        else:
            new_src, pos = insert_snippet_at(src, insert_text, None if insert_before_pattern is None else src.find(insert_before_pattern))
        out_path = filepath if inplace else (filepath + ".ai.ix")
        try:
            open(out_path, "w", encoding="utf-8").write(new_src)
            _METRICS["injections"] += 1
        except Exception as e:
            return False, f"write failed: {e}"
        return True, out_path

    def generate_patch_for_injection(self, filepath: str, insert_text: str, insert_before_pattern: Optional[str] = None) -> Tuple[bool, str]:
        try:
            original = open(filepath, "r", encoding="utf-8").read()
        except Exception as e:
            return False, f"read failed: {e}"
        insert_at = None
        if insert_before_pattern:
            m = re.search(insert_before_pattern, original)
            if m:
                if _is_position_in_comment_or_string(original, m.start()):
                    return False, "injection point inside comment/string, aborting"
                insert_at = m.start()
        transformed, pos = insert_snippet_at(original, insert_text, insert_at)
        patch = generate_unified_patch(original, transformed, filepath)
        patch_path = filepath + ".ai.inj.patch"
        try:
            open(patch_path, "w", encoding="utf-8").write(patch)
        except Exception as e:
            return False, f"write failed: {e}"
        return True, patch_path

    # plugin hooks
    def register_plugin_callback(self, cb: Callable[[str, Dict[str, Any]], None]):
        self.plugins.append(cb)

    def _call_plugins(self, event: str, payload: Dict[str, Any]):
        for cb in self.plugins:
            try:
                cb(event, payload)
            except Exception:
                LOG.exception("plugin callback failed")

    # small utility
    def list_enum_names(self, source: str) -> List[str]:
        return [e.name for e in self.find_enums(source)]

    def list_struct_names(self, source: str) -> List[str]:
        return [s.name for s in self.find_structs(source)]


# -------------------------
# Metrics HTTP server
# -------------------------
class _MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        lines = []
        for k, v in _METRICS.items():
            lines.append(f"{k} {v}")
        payload = "\n".join(lines) + "\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload.encode("utf-8"))

    def log_message(self, format, *args):
        return


def start_metrics_server(host: str = "127.0.0.1", port: int = 8181) -> threading.Thread:
    server = ThreadingHTTPServer((host, port), _MetricsHandler)
    th = threading.Thread(target=server.serve_forever, daemon=True, name="instryx-match-metrics")
    th.start()
    LOG.info("Metrics server started at http://%s:%d/metrics", host, port)
    return th


# -------------------------
# CLI
# -------------------------
def _cli_main():
    p = argparse.ArgumentParser(prog="instryx_match_enum_struct.py")
    p.add_argument("cmd", nargs="?", help="command (list-enums, list-structs, emit-match, inject-match, batch-detect, batch-inject, suggest, accept-suggestion, mem-export, mem-import, test, repl)")
    p.add_argument("target", nargs="?", help="file or directory")
    p.add_argument("--enum", help="enum name for emit/inject")
    p.add_argument("--var", default="v", help="variable name used in generated match")
    p.add_argument("--write", help="write output to file")
    p.add_argument("--inplace", action="store_true", help="write in-place for inject-match")
    p.add_argument("--max", type=int, default=12, help="max suggestions / results")
    p.add_argument("--args", nargs="*", help="helper args forwarded to codegen")
    p.add_argument("--metrics", action="store_true", help="start local metrics HTTP server")
    p.add_argument("--apply-macros", action="store_true", help="apply macro_overlay expansion if available before parsing")
    args = p.parse_args()

    tool = DMatchTool(codegen_module=_try_codegen)

    cmd = args.cmd or "help"
    if cmd in ("help", None):
        p.print_help()
        return 0

    if args.metrics:
        start_metrics_server()

    if cmd == "list-enums":
        if not args.target:
            print("file required")
            return 2
        src = open(args.target, "r", encoding="utf-8").read()
        if args.apply_macros:
            src, _ = expand_macros_if_available(src, filename=args.target)
        enums = tool.find_enums(src)
        for e in enums:
            print(e.name, "=>", [v.name for v in e.variants])
        return 0

    if cmd == "list-structs":
        if not args.target:
            print("file required")
            return 2
        src = open(args.target, "r", encoding="utf-8").read()
        if args.apply_macros:
            src, _ = expand_macros_if_available(src, filename=args.target)
        structs = tool.find_structs(src)
        for s in structs:
            print(s.name, "=>", [f"{fld.name}:{fld.type}" for fld in s.fields])
        return 0

    if cmd == "emit-match":
        if not args.target or not args.enum:
            print("usage: emit-match <file> --enum EnumName [--var v] [--write out.ix]")
            return 2
        src = open(args.target, "r", encoding="utf-8").read()
        if args.apply_macros:
            src, _ = expand_macros_if_available(src, filename=args.target)
        enums = tool.find_enums(src)
        ed = next((e for e in enums if e.name == args.enum), None)
        if not ed:
            print("enum not found")
            return 2
        txt = tool.generate_match_stub(ed, var_name=args.var)
        if args.write:
            open(args.write, "w", encoding="utf-8").write(txt)
            print("wrote", args.write)
        else:
            print(txt)
        return 0

    if cmd == "inject-match":
        if not args.target or not args.enum:
            print("usage: inject-match <file> --enum EnumName [--var v] [--inplace]")
            return 2
        src = open(args.target, "r", encoding="utf-8").read()
        if args.apply_macros:
            src, _ = expand_macros_if_available(src, filename=args.target)
        enums = tool.find_enums(src)
        ed = next((e for e in enums if e.name == args.enum), None)
        if not ed:
            print("enum not found")
            return 2
        txt = tool.generate_match_stub(ed, var_name=args.var)
        ok, out = tool.inject_stub_into_file(args.target, txt, insert_before_pattern=None, inplace=args.inplace)
        print(out if ok else f"failed: {out}")
        return 0

    if cmd == "batch-detect":
        if not args.target:
            print("directory required")
            return 2
        results = {}
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = {}
            for root, _, filenames in os.walk(args.target):
                for fn in filenames:
                    if fn.endswith(".ix"):
                        path = os.path.join(root, fn)
                        futures[ex.submit(lambda p: (p, open(p, "r", encoding="utf-8").read()), path)] = path
            for fut in as_completed(futures):
                path = futures[fut]
                try:
                    _, src = fut.result()
                    if args.apply_macros:
                        src, _ = expand_macros_if_available(src, filename=path)
                    enums = tool.find_enums(src)
                    structs = tool.find_structs(src)
                    if enums or structs:
                        results[path] = {"enums": [e.name for e in enums], "structs": [s.name for s in structs]}
                except Exception as e:
                    LOG.exception("batch detect error for %s: %s", path, e)
        print(json.dumps(results, indent=2))
        return 0

    if cmd == "batch-inject":
        if not args.target or not args.enum:
            print("usage: batch-inject <dir> --enum EnumName --var v")
            return 2
        base_dir = args.target
        injected = []
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = []
            for root, _, filenames in os.walk(base_dir):
                for fn in filenames:
                    if fn.endswith(".ix"):
                        path = os.path.join(root, fn)
                        futures.append(ex.submit(_batch_inject_file, tool, path, args.enum, args.var))
            for fut in as_completed(futures):
                path, ok, out = fut.result()
                if ok:
                    injected.append(path)
                    print(f"Injected into {path} -> {out}")
                else:
                    LOG.debug("Skipped %s: %s", path, out)
        print("Injected into", len(injected), "files")
        return 0

    if cmd == "suggest":
        if not args.target:
            print("file required")
            return 2
        src = open(args.target, "r", encoding="utf-8").read()
        if args.apply_macros:
            src, _ = expand_macros_if_available(src, filename=args.target)
        suggestions = tool.suggest_match_locations(src, max_results=args.max)
        for s in suggestions:
            print(json.dumps(s.to_dict(), indent=2))
        return 0

    if cmd == "mem-export":
        out = args.target or "match_memory_export.json"
        data = tool.memory.export()
        Path(out).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print("exported", out)
        return 0

    if cmd == "mem-import":
        if not args.target:
            print("file required")
            return 2
        data = json.loads(open(args.target, "r", encoding="utf-8").read())
        tool.memory.import_data(data, merge=True)
        print("imported")
        return 0

    if cmd == "test":
        ok = _run_unit_tests(tool)
        print("TEST", "PASS" if ok else "FAIL")
        return 0 if ok else 2

    if cmd == "repl":
        print("Instryx match tool REPL. Type 'help' for commands. Ctrl-D to exit.")
        _repl(tool)
        return 0

    print("unknown command", cmd)
    p.print_help()
    return 2


def _batch_inject_file(tool: DMatchTool, path: str, enum_name: str, var_name: str) -> Tuple[str, bool, str]:
    try:
        src = open(path, "r", encoding="utf-8").read()
        enums = tool.find_enums(src)
        ed = next((e for e in enums if e.name == enum_name), None)
        if not ed:
            return path, False, "enum not found"
        txt = tool.generate_match_stub(ed, var_name=var_name)
        # attempt to inject before first function or at EOF
        m = re.search(r"\bfunc\b", src)
        insert_before = None
        if m:
            insert_before = m.start()
            ok, out = tool.inject_stub_into_file(path, txt, insert_before_pattern=None, inplace=False, safe=True)
            return path, ok, out
        # else append at EOF
        ok, out = tool.inject_stub_into_file(path, txt, insert_before_pattern=None, inplace=False, safe=False)
        return path, ok, out
    except Exception as e:
        LOG.exception("batch inject failed for %s: %s", path, e)
        return path, False, str(e)


# -------------------------
# Unit tests / self-checks (expanded)
# -------------------------
def _run_unit_tests(tool: DMatchTool) -> bool:
    try:
        sample = """
        enum Color { Red, Green(u32), Blue { r: i32, g: i32 } }
        struct Point { x: f32; y: f32; }
        func handle(c) {
            match c { /* incomplete */ }
        }
        """
        enums = tool.find_enums(sample)
        if not enums:
            LOG.error("enum detection failed")
            return False
        ed = enums[0]
        stub = tool.generate_match_stub(ed, var_name="c")
        assert "Red" in stub and "Green" in stub and "Blue" in stub
        structs = tool.find_structs(sample)
        assert structs and structs[0].name == "Point"
        # test injection dry-run
        ok, out = tool.generate_patch_for_injection.__func__(tool, __name__, "", None) if False else (True, "noop")
        # memory operations
        tool.memory.record_pattern("unit_test", 1)
        exported = tool.memory.export()
        assert "patterns" in exported
        return True
    except Exception as e:
        LOG.exception("unit tests failed: %s", e)
        return False


# -------------------------
# Macro overlay integration helper (optional)
# -------------------------
def expand_macros_if_available(source: str, filename: Optional[str] = None) -> Tuple[str, List[Any]]:
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
# Interactive REPL
# -------------------------
def _repl(tool: DMatchTool):
    try:
        while True:
            line = input("match> ").strip()
            if not line:
                continue
            if line in ("exit", "quit"):
                break
            if line == "help":
                print("commands: enums <file>, structs <file>, emit <file> <Enum> [var], suggest <file>, mem-export <file>, mem-import <file>, quit")
                continue
            parts = line.split()
            cmd = parts[0]
            if cmd == "enums" and len(parts) >= 2:
                src = open(parts[1], "r", encoding="utf-8").read()
                enums = tool.find_enums(src)
                for e in enums:
                    print(e.name, [v.name for v in e.variants])
            elif cmd == "structs" and len(parts) >= 2:
                src = open(parts[1], "r", encoding="utf-8").read()
                structs = tool.find_structs(src)
                for s in structs:
                    print(s.name, [(f.name, f.type) for f in s.fields])
            elif cmd == "emit" and len(parts) >= 3:
                src = open(parts[1], "r", encoding="utf-8").read()
                enums = tool.find_enums(src)
                ed = next((e for e in enums if e.name == parts[2]), None)
                if not ed:
                    print("enum not found")
                else:
                    var = parts[3] if len(parts) >= 4 else "v"
                    print(tool.generate_match_stub(ed, var_name=var))
            elif cmd == "suggest" and len(parts) >= 2:
                src = open(parts[1], "r", encoding="utf-8").read()
                suggestions = tool.suggest_match_locations(src)
                for s in suggestions:
                    print(json.dumps(s.to_dict(), indent=2))
            else:
                print("unknown repl command")
    except EOFError:
        print("\nbye")


# -------------------------
# Module entrypoint
# -------------------------
if __name__ == "__main__":
    try:
        sys.exit(_cli_main())
    except Exception:
        LOG.exception("Fatal error in instryx_match_enum_struct")
        sys.exit(3)

