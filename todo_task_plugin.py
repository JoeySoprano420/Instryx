# register(assistant) style plugin that flags TODO comments as suggestions
from ciams.ai_engine import Suggestion

def rule_todos(source: str, filename=None):
    suggestions = []
    for m in __import__('re').finditer(r"//\s*TODO[: ]?(.*)$|#\s*TODO[: ]?(.*)$", source, __import__('re').M):
        txt = (m.group(1) or m.group(2) or "").strip()
        snippet = source[max(0, m.start()-60): m.end()+60].splitlines()[0].strip()
        suggestions.append(Suggestion("note", [txt], f"TODO found: {txt}", 0.2, snippet, (m.start(), m.end())))
    return suggestions

def register(assistant):
    assistant.register_rule(rule_todos)
    def unregister(assistant):
        # optional: if your PluginManager supports unregister, implement removal logic
        pass
    assistant.unregister_rule(rule_todos)
    # register(assistant) style plugin that flags TODO comments as suggestions
    """
    Registers the rule_todos function with the assistant to flag TODO comments in the source code.
    """

    return unregister
    
"""
Enhanced TODO task plugin for CIAMS assistant.

Features:
 - Detects TODO/FIXME/XXX/HACK/OPTIMIZE comments in single-line and block comments.
 - Parses metadata: assignee (@user), issue references (#123), priority ([P1]/priority:high),
   due dates (due:YYYY-MM-DD), freeform tags (@review, @perf).
 - Produces rich Suggestion objects with contextual multi-line snippets and precise
   (line, column, start_offset, end_offset) positions.
 - Exposes `parse_todos` for structured data and `rule_todos` for assistant integration.
 - register/unregister helpers for the assistant PluginManager.
 - Small convenience `scan_sources` to run the parser over multiple files.
"""

from __future__ import annotations
import re
import os
from typing import Any, Dict, List, Optional, Tuple

from ciams.ai_engine import Suggestion

# Regex building blocks
_SINGLE_LINE_COMMENT = r"(?P<prefix>//|#|--|;)\s*(?P<body>.*)$"
_BLOCK_COMMENT = r"/\*(?P<body>.*?)\*/"  # DOTALL used when applied

# Recognized markers (case-insensitive)
_MARKER_WORDS = r"(TODO|FIXME|XXX|HACK|OPTIMIZE|NOTE|REVIEW)"

# Inline marker pattern: captures marker and rest of line
_INLINE_MARKER_RE = re.compile(
    rf"(?P<prefix>//|#|--|;)\s*(?P<marker>{_MARKER_WORDS})[:\s]?(?P<rest>.*)$",
    re.IGNORECASE | re.MULTILINE
)

# Block comment search (will scan inside for markers)
_BLOCK_COMMENT_RE = re.compile(_BLOCK_COMMENT, re.DOTALL | re.IGNORECASE)

# Metadata extraction patterns
_ASSIGNEE_RE = re.compile(r"@(?P<assignee>[A-Za-z0-9_\-\.]+)")
_ISSUE_RE = re.compile(r"#(?P<issue>\d+)")
_PRIORITY_RE = re.compile(r"\b(?:\[P(?P<pnum>[0-9])\]|priority[:=]\s*(?P<pword>high|medium|low|p1|p2|p3))\b", re.IGNORECASE)
_DUE_RE = re.compile(r"\b(?:due[:=]\s*(?P<date>\d{4}-\d{2}-\d{2}))\b", re.IGNORECASE)
_TAG_RE = re.compile(r"@(?P<tag>[A-Za-z0-9_\-]+)")

# severity mapping by marker or metadata
_SEVERITY_BY_MARKER = {
    "FIXME": 0.95,
    "TODO": 0.60,
    "XXX": 0.90,
    "HACK": 0.45,
    "OPTIMIZE": 0.50,
    "NOTE": 0.30,
    "REVIEW": 0.65,
}

_PRIORITY_MAP = {
    "p1": 0.98, "p2": 0.9, "p3": 0.75,
    "high": 0.95, "medium": 0.7, "low": 0.4
}


def _offset_to_linecol(source: str, offset: int) -> Tuple[int, int]:
    """Convert byte offset to (line, col) 1-based."""
    if offset < 0:
        offset = 0
    # count lines
    prefix = source[:offset]
    line = prefix.count("\n") + 1
    # column is chars since last newline
    last_nl = prefix.rfind("\n")
    col = offset - (last_nl + 1) + 1 if last_nl != -1 else offset + 1
    return line, col


def _make_snippet(source: str, start: int, end: int, context_lines: int = 2) -> str:
    """Return context snippet with the TODO line highlighted with '>>' prefix."""
    lines = source.splitlines()
    # compute line indices (0-based)
    start_line, _ = _offset_to_linecol(source, start)
    end_line, _ = _offset_to_linecol(source, end)
    # convert to 0-based indices
    sidx = max(0, start_line - 1 - context_lines)
    eidx = min(len(lines), end_line + context_lines)
    snippet_lines = []
    for i in range(sidx, eidx):
        prefix = "   "
        if i >= (start_line - 1) and i <= (end_line - 1):
            prefix = ">> "
        snippet_lines.append(f"{prefix}{i+1:4d}: {lines[i]}")
    return "\n".join(snippet_lines)


def _normalize_marker(marker: Optional[str]) -> str:
    return (marker or "TODO").upper()


def _compute_confidence(marker: str, metadata: Dict[str, Any]) -> float:
    """Heuristic confidence based on marker and metadata (priority/due/issue)."""
    m = _normalize_marker(marker)
    base = _SEVERITY_BY_MARKER.get(m, 0.5)
    # bump for explicit priority
    pr = metadata.get("priority")
    if pr:
        if isinstance(pr, str):
            base = max(base, _PRIORITY_MAP.get(pr.lower(), base))
    # bump slightly if assignee present (actionable)
    if metadata.get("assignee"):
        base = min(0.99, base + 0.05)
    # bump if issue reference exists
    if metadata.get("issue"):
        base = min(0.98, base + 0.03)
    return round(base, 2)


def parse_todos(source: str, filename: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Parse source and return list of structured todo dictionaries.

    Each todo dict contains:
      - marker: TODO/FIXME/...
      - text: the freeform text after the marker
      - assignee: optional @user
      - issue: optional issue number (as string)
      - priority: optional (p1/p2/p3/high/medium/low)
      - due: optional YYYY-MM-DD
      - tags: list of @tags found
      - start_offset, end_offset: offsets in source
      - line, col: 1-based position of the marker
      - snippet: contextual snippet
      - filename: as passed in
    """
    todos: List[Dict[str, Any]] = []

    # 1) single-line comments with inline markers
    for m in _INLINE_MARKER_RE.finditer(source):
        marker = m.group("marker") or "TODO"
        rest = (m.group("rest") or "").strip()
        start, end = m.start(), m.end()
        # extract metadata
        assignee = (_ASSIGNEE_RE.search(rest).group("assignee")
                    if _ASSIGNEE_RE.search(rest) else None)
        issue = (_ISSUE_RE.search(rest).group("issue")
                 if _ISSUE_RE.search(rest) else None)
        pr_match = _PRIORITY_RE.search(rest)
        priority = None
        if pr_match:
            priority = pr_match.group("pnum") or pr_match.group("pword")
        due = (_DUE_RE.search(rest).group("date")
               if _DUE_RE.search(rest) else None)
        tags = [m.group("tag") for m in _TAG_RE.finditer(rest) if m.group("tag") != (assignee or "")]
        line, col = _offset_to_linecol(source, start)
        snippet = _make_snippet(source, start, end)
        todos.append({
            "marker": marker.upper(),
            "text": rest,
            "assignee": assignee,
            "issue": issue,
            "priority": priority,
            "due": due,
            "tags": tags,
            "start_offset": start,
            "end_offset": end,
            "line": line,
            "col": col,
            "snippet": snippet,
            "filename": filename
        })

    # 2) block comments search for markers inside
    for bm in _BLOCK_COMMENT_RE.finditer(source):
        body = bm.group("body") or ""
        # find markers inside body (may be multiple)
        for im in re.finditer(rf"(?P<marker>{_MARKER_WORDS})[:\s]?(?P<rest>.*)$", body, re.IGNORECASE | re.MULTILINE):
            marker = im.group("marker") or "TODO"
            rest = (im.group("rest") or "").strip()
            # compute offsets relative to source
            # find the position of the inner match within the full source
            inner_rel = im.start()
            start = bm.start() + 2 + inner_rel  # +2 accounts for '/*'
            end = start + (im.end() - im.start())
            assignee = (_ASSIGNEE_RE.search(rest).group("assignee")
                        if _ASSIGNEE_RE.search(rest) else None)
            issue = (_ISSUE_RE.search(rest).group("issue")
                     if _ISSUE_RE.search(rest) else None)
            pr_match = _PRIORITY_RE.search(rest)
            priority = None
            if pr_match:
                priority = pr_match.group("pnum") or pr_match.group("pword")
            due = (_DUE_RE.search(rest).group("date")
                   if _DUE_RE.search(rest) else None)
            tags = [m.group("tag") for m in _TAG_RE.finditer(rest) if m.group("tag") != (assignee or "")]
            line, col = _offset_to_linecol(source, start)
            snippet = _make_snippet(source, start, end)
            todos.append({
                "marker": marker.upper(),
                "text": rest,
                "assignee": assignee,
                "issue": issue,
                "priority": priority,
                "due": due,
                "tags": tags,
                "start_offset": start,
                "end_offset": end,
                "line": line,
                "col": col,
                "snippet": snippet,
                "filename": filename
            })

    # deduplicate by start_offset
    seen = set()
    unique: List[Dict[str, Any]] = []
    for t in todos:
        key = (t["filename"], t["start_offset"], t["end_offset"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(t)
    return unique


def rule_todos(source: str, filename: Optional[str] = None) -> List[Suggestion]:
    """
    Assistant integration entrypoint.

    Returns a list of Suggestion objects suitable for the assistant UI.
    Each Suggestion encodes the marker, parsed metadata and a recommended
    high-level action in the title/message.
    """
    suggestions: List[Suggestion] = []
    parsed = parse_todos(source, filename=filename)
    for t in parsed:
        marker = t["marker"]
        text = t["text"] or ""
        # tags for the suggestion: include marker and any discovered tags
        tags = [marker.lower()] + t.get("tags", [])
        if t.get("priority"):
            tags.append(f"priority:{t['priority']}")
        if t.get("assignee"):
            tags.append(f"assignee:{t['assignee']}")
        if t.get("issue"):
            tags.append(f"issue:{t['issue']}")

        # human-friendly title & description
        short = text.splitlines()[0] if text else ""
        title = f"{marker}: {short}" if short else f"{marker}"
        desc_lines = [
            f"File: {t['filename'] or '<unknown>'}  Line: {t['line']}  Col: {t['col']}",
            f"Text: {text}",
        ]
        if t.get("assignee"):
            desc_lines.append(f"Assignee: @{t['assignee']}")
        if t.get("issue"):
            desc_lines.append(f"Issue: #{t['issue']}")
        if t.get("priority"):
            desc_lines.append(f"Priority: {t['priority']}")
        if t.get("due"):
            desc_lines.append(f"Due: {t['due']}")
        desc = " | ".join(desc_lines)

        # confidence heuristic
        confidence = _compute_confidence(marker, t)

        snippet = t.get("snippet", "")
        # Suggestion constructor: (kind, tags:list, title, confidence:float, snippet, location_tuple)
        # keep location as (line, col, start_offset, end_offset) for richer UI
        location = (t["line"], t["col"], t["start_offset"], t["end_offset"])

        suggestions.append(Suggestion("task", tags, title, confidence, snippet, location))

    return suggestions


def scan_sources(paths: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convenience helper to scan files on disk. Returns mapping path -> parsed todos.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for p in paths:
        if os.path.isdir(p):
            # walk tree (ignores hidden directories)
            for root, dirs, files in os.walk(p):
                for f in files:
                    if f.startswith("."):
                        continue
                    full = os.path.join(root, f)
                    try:
                        with open(full, "r", encoding="utf-8") as fh:
                            src = fh.read()
                        parsed = parse_todos(src, filename=full)
                        if parsed:
                            out[full] = parsed
                    except Exception:
                        # best-effort; don't fail the whole scan
                        continue
        else:
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    src = fh.read()
                parsed = parse_todos(src, filename=p)
                if parsed:
                    out[p] = parsed
            except Exception:
                continue
    return out


def register(assistant) -> None:
    """
    Register the plugin with an assistant instance.

    The assistant is expected to provide:
      - register_rule(callable) -> registers a rule that accepts (source, filename) and returns [Suggestion]
      - unregister_rule(callable) -> optional, should remove the registered rule

    This function registers rule_todos and returns nothing. If the assistant
    supports unregistering, an external system may call `unregister(assistant)`.
    """
    assistant.register_rule(rule_todos)


def unregister(assistant) -> None:
    """Optional unregister routine to remove the plugin's rule from the assistant."""
    try:
        assistant.unregister_rule(rule_todos)
    except Exception:
        # best-effort: not all assistant/plugin managers implement unregister
        pass

"""
Enhanced TODO task plugin for CIAMS assistant (super boosters, enhancers, tooling).

Key additions:
 - Concurrency scanning (ThreadPoolExecutor) with file-mtime caching.
 - JSON + SARIF exporters for CI and static analysis integration.
 - Grouping, summary stats and prioritized sorting.
 - Quick-fix patch suggestion generator (edit/replace TODO with issue link or remove).
 - Issue payload generator for GitHub-style issue creation.
 - Register helpers expose utilities on assistant when available.
 - CLI entry when executed directly (scan paths, export).
"""

from __future__ import annotations
import re
import os
import json
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Iterable, Callable
from functools import lru_cache

from ciams.ai_engine import Suggestion

# --- existing regex / helpers (kept and extended) ---
_SINGLE_LINE_COMMENT = r"(?P<prefix>//|#|--|;)\s*(?P<body>.*)$"
_BLOCK_COMMENT = r"/\*(?P<body>.*?)\*/"  # DOTALL used when applied
_MARKER_WORDS = r"(TODO|FIXME|XXX|HACK|OPTIMIZE|NOTE|REVIEW|BUG|TASK)"
_INLINE_MARKER_RE = re.compile(
    rf"(?P<prefix>//|#|--|;)\s*(?P<marker>{_MARKER_WORDS})[:\s]?(?P<rest>.*)$",
    re.IGNORECASE | re.MULTILINE
)
_BLOCK_COMMENT_RE = re.compile(_BLOCK_COMMENT, re.DOTALL | re.IGNORECASE)

_ASSIGNEE_RE = re.compile(r"@(?P<assignee>[A-Za-z0-9_\-\.]+)")
_ISSUE_RE = re.compile(r"#(?P<issue>\d+)")
_PRIORITY_RE = re.compile(r"\b(?:\[P(?P<pnum>[0-9])\]|priority[:=]\s*(?P<pword>high|medium|low|p1|p2|p3))\b", re.IGNORECASE)
_DUE_RE = re.compile(r"\b(?:due[:=]\s*(?P<date>\d{4}-\d{2}-\d{2}))\b", re.IGNORECASE)
_TAG_RE = re.compile(r"@(?P<tag>[A-Za-z0-9_\-]+)")

_SEVERITY_BY_MARKER = {
    "FIXME": 0.95,
    "TODO": 0.60,
    "XXX": 0.90,
    "HACK": 0.45,
    "OPTIMIZE": 0.50,
    "NOTE": 0.30,
    "REVIEW": 0.65,
    "BUG": 0.96,
    "TASK": 0.55,
}

_PRIORITY_MAP = {
    "p1": 0.98, "p2": 0.9, "p3": 0.75,
    "high": 0.95, "medium": 0.7, "low": 0.4
}

# --- utility helpers ---
def _offset_to_linecol(source: str, offset: int) -> Tuple[int, int]:
    if offset < 0:
        offset = 0
    prefix = source[:offset]
    line = prefix.count("\n") + 1
    last_nl = prefix.rfind("\n")
    col = offset - (last_nl + 1) + 1 if last_nl != -1 else offset + 1
    return line, col

def _make_snippet(source: str, start: int, end: int, context_lines: int = 2) -> str:
    lines = source.splitlines()
    start_line, _ = _offset_to_linecol(source, start)
    end_line, _ = _offset_to_linecol(source, end)
    sidx = max(0, start_line - 1 - context_lines)
    eidx = min(len(lines), end_line + context_lines)
    snippet_lines = []
    for i in range(sidx, eidx):
        prefix = "   "
        if i >= (start_line - 1) and i <= (end_line - 1):
            prefix = ">> "
        snippet_lines.append(f"{prefix}{i+1:4d}: {lines[i]}")
    return "\n".join(snippet_lines)

def _normalize_marker(marker: Optional[str]) -> str:
    return (marker or "TODO").upper()

def _compute_confidence(marker: str, metadata: Dict[str, Any]) -> float:
    m = _normalize_marker(marker)
    base = _SEVERITY_BY_MARKER.get(m, 0.5)
    pr = metadata.get("priority")
    if pr:
        if isinstance(pr, str):
            base = max(base, _PRIORITY_MAP.get(pr.lower(), base))
    if metadata.get("assignee"):
        base = min(0.99, base + 0.05)
    if metadata.get("issue"):
        base = min(0.98, base + 0.03)
    return round(base, 2)

# --- parsing / core functionality ---
def parse_todos(source: str, filename: Optional[str] = None) -> List[Dict[str, Any]]:
    todos: List[Dict[str, Any]] = []

    # single-line marker matches
    for m in _INLINE_MARKER_RE.finditer(source):
        marker = m.group("marker") or "TODO"
        rest = (m.group("rest") or "").strip()
        start, end = m.start(), m.end()
        assignee = (_ASSIGNEE_RE.search(rest).group("assignee") if _ASSIGNEE_RE.search(rest) else None)
        issue = (_ISSUE_RE.search(rest).group("issue") if _ISSUE_RE.search(rest) else None)
        pr_match = _PRIORITY_RE.search(rest)
        priority = pr_match.group("pnum") or pr_match.group("pword") if pr_match else None
        due = (_DUE_RE.search(rest).group("date") if _DUE_RE.search(rest) else None)
        tags = [mo.group("tag") for mo in _TAG_RE.finditer(rest) if mo.group("tag") != (assignee or "")]
        line, col = _offset_to_linecol(source, start)
        snippet = _make_snippet(source, start, end)
        todos.append({
            "marker": marker.upper(),
            "text": rest,
            "assignee": assignee,
            "issue": issue,
            "priority": priority,
            "due": due,
            "tags": tags,
            "start_offset": start,
            "end_offset": end,
            "line": line,
            "col": col,
            "snippet": snippet,
            "filename": filename
        })

    # block comments
    for bm in _BLOCK_COMMENT_RE.finditer(source):
        body = bm.group("body") or ""
        for im in re.finditer(rf"(?P<marker>{_MARKER_WORDS})[:\s]?(?P<rest>.*)$", body, re.IGNORECASE | re.MULTILINE):
            marker = im.group("marker") or "TODO"
            rest = (im.group("rest") or "").strip()
            inner_rel = im.start()
            start = bm.start() + 2 + inner_rel
            end = start + (im.end() - im.start())
            assignee = (_ASSIGNEE_RE.search(rest).group("assignee") if _ASSIGNEE_RE.search(rest) else None)
            issue = (_ISSUE_RE.search(rest).group("issue") if _ISSUE_RE.search(rest) else None)
            pr_match = _PRIORITY_RE.search(rest)
            priority = pr_match.group("pnum") or pr_match.group("pword") if pr_match else None
            due = (_DUE_RE.search(rest).group("date") if _DUE_RE.search(rest) else None)
            tags = [mo.group("tag") for mo in _TAG_RE.finditer(rest) if mo.group("tag") != (assignee or "")]
            line, col = _offset_to_linecol(source, start)
            snippet = _make_snippet(source, start, end)
            todos.append({
                "marker": marker.upper(),
                "text": rest,
                "assignee": assignee,
                "issue": issue,
                "priority": priority,
                "due": due,
                "tags": tags,
                "start_offset": start,
                "end_offset": end,
                "line": line,
                "col": col,
                "snippet": snippet,
                "filename": filename
            })

    # dedupe by offsets
    seen = set()
    unique: List[Dict[str, Any]] = []
    for t in todos:
        key = (t["filename"], t["start_offset"], t["end_offset"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(t)
    return unique

# --- advanced boosters / tooling ---

# simple in-memory cache keyed by file path + mtime
_file_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}

def parse_todos_cached(path: str) -> List[Dict[str, Any]]:
    try:
        mtime = os.path.getmtime(path)
    except Exception:
        mtime = 0.0
    entry = _file_cache.get(path)
    if entry and entry[0] == mtime:
        return entry[1]
    try:
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
    except Exception:
        return []
    parsed = parse_todos(src, filename=path)
    _file_cache[path] = (mtime, parsed)
    return parsed

def scan_sources_concurrent(paths: Iterable[str], max_workers: int = 8, skip_hidden: bool = True) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, fs in os.walk(p):
                if skip_hidden:
                    dirs[:] = [d for d in dirs if not d.startswith(".")]
                for f in fs:
                    if f.startswith("."):
                        continue
                    files.append(os.path.join(root, f))
        else:
            files.append(p)
    # concurrent parse
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(parse_todos_cached, f): f for f in files}
        for fut in as_completed(futures):
            f = futures[fut]
            try:
                res = fut.result()
            except Exception:
                res = []
            if res:
                out[f] = res
    return out

def group_todos(todos: Iterable[Dict[str, Any]], by: str = "assignee") -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for t in todos:
        key = "<unassigned>"
        if by == "assignee":
            key = t.get("assignee") or "<unassigned>"
        elif by == "priority":
            key = str(t.get("priority") or "<none>")
        elif by == "marker":
            key = t.get("marker") or "<marker>"
        elif by == "file":
            key = t.get("filename") or "<unknown>"
        groups.setdefault(key, []).append(t)
    return groups

def summary_stats(todos: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    by_marker = {}
    by_priority = {}
    by_assignee = {}
    for t in todos:
        total += 1
        by_marker[t.get("marker")] = by_marker.get(t.get("marker"), 0) + 1
        by_priority[t.get("priority")] = by_priority.get(t.get("priority"), 0) + 1
        by_assignee[t.get("assignee")] = by_assignee.get(t.get("assignee"), 0) + 1
    return {"total": total, "by_marker": by_marker, "by_priority": by_priority, "by_assignee": by_assignee}

# quick-fix generator: returns a minimal patch dict {start,end,replacement} or None
def generate_quickfix_patch(todo: Dict[str, Any], action: str = "issue", issue_template_url: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    action:
      - "issue": replace TODO comment with "See issue #<generated>" placeholder (non-destructive suggestion)
      - "remove": remove the TODO line/comment
      - "reference": append " (tracked: <url>)" to the line
    """
    start = todo.get("start_offset")
    end = todo.get("end_offset")
    if start is None or end is None:
        return None
    original = todo.get("text", "")
    if action == "remove":
        replacement = ""
    elif action == "issue":
        # generate determinist hash id for offline suggestion
        h = hashlib.sha1((todo.get("filename","") + str(start) + original).encode("utf-8")).hexdigest()[:8]
        if issue_template_url:
            url = issue_template_url.rstrip("/") + f"/issues/new?title={re.escape(todo.get('marker','TODO'))}+{h}"
            replacement = f"{todo.get('marker')} (migrated -> {url})"
        else:
            replacement = f"{todo.get('marker')} (migrated -> ISSUE-{h})"
    elif action == "reference":
        url = issue_template_url or "https://example.com/issue"
        replacement = f"{todo.get('marker')}: {original} (tracked: {url})"
    else:
        return None
    return {"start": start, "end": end, "replacement": replacement}

def generate_issue_payload(todo: Dict[str, Any], repo: Optional[str] = None) -> Dict[str, Any]:
    title = todo.get("text", "").splitlines()[0] or f"{todo.get('marker')} in {os.path.basename(todo.get('filename') or '')}"
    body_lines = [
        f"File: {todo.get('filename') or '<unknown>'}",
        f"Line: {todo.get('line')}",
        "",
        "```\n" + (todo.get("snippet") or "") + "\n```",
        "",
        "Original TODO text:",
        todo.get("text", "")
    ]
    body = "\n".join(body_lines)
    labels = [todo.get("marker", "todo").lower()]
    if todo.get("priority"):
        labels.append(f"priority:{todo.get('priority')}")
    if todo.get("assignee"):
        labels.append(f"assignee:{todo.get('assignee')}")
    if repo:
        body = f"Repository: {repo}\n\n" + body
    return {"title": title, "body": body, "labels": labels, "assignee": todo.get("assignee")}

# export helpers
def export_json(todos: Iterable[Dict[str, Any]], path: str) -> str:
    data = list(todos)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"generated": time.time(), "todos": data}, fh, indent=2, default=str)
    return path

def export_sarif(todos: Iterable[Dict[str, Any]], path: str, tool_name: str = "ciams-todo-plugin") -> str:
    """
    Minimal SARIF v2 wrapper for TODO items so linters/CI can consume results.
    Produces a very small SARIF structure with rules generated from markers.
    """
    todos = list(todos)
    rules = {}
    results = []
    for t in todos:
        rule_id = t.get("marker") or "TODO"
        if rule_id not in rules:
            rules[rule_id] = {"id": rule_id, "shortDescription": {"text": rule_id}, "defaultConfiguration": {"level": "warning"}}
        result = {
            "ruleId": rule_id,
            "level": "warning",
            "message": {"text": t.get("text", "") or rule_id},
            "locations": [{
                "physicalLocation": {
                    "artifactLocation": {"uri": t.get("filename") or "<unknown>"},
                    "region": {"startLine": t.get("line", 1), "startColumn": t.get("col", 1)}
                }
            }]
        }
        results.append(result)
    sarif = {
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {"name": tool_name, "rules": list(rules.values())}},
            "results": results
        }]
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(sarif, fh, indent=2)
    return path

# assistant integration wrapper
def rule_todos(source: str, filename: Optional[str] = None) -> List[Suggestion]:
    suggestions: List[Suggestion] = []
    parsed = parse_todos(source, filename=filename)
    for t in parsed:
        marker = t["marker"]
        text = t["text"] or ""
        tags = [marker.lower()] + t.get("tags", [])
        if t.get("priority"):
            tags.append(f"priority:{t['priority']}")
        if t.get("assignee"):
            tags.append(f"assignee:{t['assignee']}")
        if t.get("issue"):
            tags.append(f"issue:{t['issue']}")
        short = text.splitlines()[0] if text else ""
        title = f"{marker}: {short}" if short else f"{marker}"
        desc_lines = [
            f"File: {t['filename'] or '<unknown>'}  Line: {t['line']}  Col: {t['col']}",
            f"Text: {text}",
        ]
        if t.get("assignee"):
            desc_lines.append(f"Assignee: @{t['assignee']}")
        if t.get("issue"):
            desc_lines.append(f"Issue: #{t['issue']}")
        if t.get("priority"):
            desc_lines.append(f"Priority: {t['priority']}")
        if t.get("due"):
            desc_lines.append(f"Due: {t['due']}")
        desc = " | ".join(desc_lines)
        confidence = _compute_confidence(marker, t)
        snippet = t.get("snippet", "")
        location = (t["line"], t["col"], t["start_offset"], t["end_offset"])
        # attach suggestion metadata in the suggestion 'message' or tags (conservative)
        # Suggestion signature: (kind, tags:list, title, confidence:float, snippet, location_tuple)
        suggestions.append(Suggestion("task", tags, title, confidence, snippet, location))
    return suggestions

# --- plugin registration and assistant helpers ---
def _expose_tools_on_assistant(assistant) -> None:
    """
    Attach utility helpers to assistant if it allows extension.
    This is non-invasive: if the assistant already exposes names it won't overwrite.
    """
    tools = {
        "parse_todos": parse_todos,
        "parse_todos_cached": parse_todos_cached,
        "scan_sources_concurrent": scan_sources_concurrent,
        "export_json": export_json,
        "export_sarif": export_sarif,
        "generate_quickfix_patch": generate_quickfix_patch,
        "generate_issue_payload": generate_issue_payload,
        "group_todos": group_todos,
        "summary_stats": summary_stats
    }
    for name, fn in tools.items():
        if hasattr(assistant, name):
            # do not overwrite existing attributes
            continue
        try:
            setattr(assistant, name, fn)
        except Exception:
            # best-effort; ignore if assignment not allowed
            continue

def register(assistant) -> Callable:
    assistant.register_rule(rule_todos)
    # attempt to expose tooling helpers for advanced workflows
    try:
        _expose_tools_on_assistant(assistant)
    except Exception:
        pass
    # return an unregister closure for plugin managers that support it
    def unregister_fn(asst=None):
        try:
            assistant.unregister_rule(rule_todos)
        except Exception:
            pass
        # remove helpers only if we added them
        try:
            for n in ("parse_todos", "parse_todos_cached", "scan_sources_concurrent", "export_json", "export_sarif", "generate_quickfix_patch", "generate_issue_payload", "group_todos", "summary_stats"):
                if getattr(assistant, n, None) in (parse_todos, parse_todos_cached, scan_sources_concurrent, export_json, export_sarif, generate_quickfix_patch, generate_issue_payload, group_todos, summary_stats):
                    try:
                        delattr(assistant, n)
                    except Exception:
                        pass
        except Exception:
            pass
    return unregister_fn

def unregister(assistant) -> None:
    try:
        assistant.unregister_rule(rule_todos)
    except Exception:
        pass

# --- CLI entry for ad-hoc runs ---
def _cli_main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="todo_task_plugin", description="Scan source paths for TODOs and export results.")
    parser.add_argument("paths", nargs="+", help="Files or directories to scan")
    parser.add_argument("--out-json", help="Write parsed todos to this JSON file")
    parser.add_argument("--out-sarif", help="Write SARIF to this path")
    parser.add_argument("--workers", type=int, default=8, help="Concurrency for file scanning")
    args = parser.parse_args(argv)
    results = scan_sources_concurrent(args.paths, max_workers=args.workers)
    all_todos = []
    for v in results.values():
        all_todos.extend(v)
    if args.out_json:
        export_json(all_todos, args.out_json)
    if args.out_sarif:
        export_sarif(all_todos, args.out_sarif)
    # print summary
    stats = summary_stats(all_todos)
    print(json.dumps(stats, indent=2))
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(_cli_main())

"""
CIAMS TODO Task Plugin — enhanced boosters, tooling, optimizations.

Adds:
 - All previous parsing, exporters, quick-fix and concurrency features.
 - Git blame-based assignee suggester.
 - GitHub issue poster (optional; uses GITHUB_TOKEN env).
 - Markdown & CSV report exporters.
 - .todoignore support and path-glob skipping.
 - Async scanning using asyncio/aiofiles when available (falls back to threadpool).
 - Prioritization / scoring function and sort helpers.
 - Apply quick-fix patches to files (atomic write).
 - Expose enhanced tools on assistant when registering.

Usage:
  register(assistant) attaches rule_todos and exposes helpers (if assistant allows).
  The CLI entry supports new flags for markdown/csv exports and GitHub dry-run issue creation.

Note: network operations (GitHub) require environment variables for auth (read README).
"""

from __future__ import annotations
import re
import os
import json
import csv
import hashlib
import time
import math
import shutil
import subprocess
import urllib.request
import urllib.error
import urllib.parse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Iterable, Callable
from functools import lru_cache

# Try to import aiofiles for async file IO; fallback if not available.
try:
    import aiofiles  # type: ignore
    _HAS_AIOFILES = True
except Exception:
    _HAS_AIOFILES = False

from ciams.ai_engine import Suggestion

# --- regex / constants ---
_MARKER_WORDS = r"(TODO|FIXME|XXX|HACK|OPTIMIZE|NOTE|REVIEW|BUG|TASK)"
_INLINE_MARKER_RE = re.compile(
    rf"(?P<prefix>//|#|--|;)\s*(?P<marker>{_MARKER_WORDS})[:\s]?(?P<rest>.*)$",
    re.IGNORECASE | re.MULTILINE
)
_BLOCK_COMMENT_RE = re.compile(r"/\*(?P<body>.*?)\*/", re.DOTALL | re.IGNORECASE)

_ASSIGNEE_RE = re.compile(r"@(?P<assignee>[A-Za-z0-9_\-\.]+)")
_ISSUE_RE = re.compile(r"#(?P<issue>\d+)")
_PRIORITY_RE = re.compile(r"\b(?:\[P(?P<pnum>[0-9])\]|priority[:=]\s*(?P<pword>high|medium|low|p1|p2|p3))\b", re.IGNORECASE)
_DUE_RE = re.compile(r"\b(?:due[:=]\s*(?P<date>\d{4}-\d{2}-\d{2}))\b", re.IGNORECASE)
_TAG_RE = re.compile(r"@(?P<tag>[A-Za-z0-9_\-]+)")

_SEVERITY_BY_MARKER = {
    "FIXME": 0.95,
    "TODO": 0.60,
    "XXX": 0.90,
    "HACK": 0.45,
    "OPTIMIZE": 0.50,
    "NOTE": 0.30,
    "REVIEW": 0.65,
    "BUG": 0.96,
    "TASK": 0.55,
}

_PRIORITY_MAP = {
    "p1": 0.98, "p2": 0.90, "p3": 0.75,
    "high": 0.95, "medium": 0.70, "low": 0.40
}

# in-memory caches
_file_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
_parse_cache_max = 1024


# --- low-level helpers ---
def _offset_to_linecol(source: str, offset: int) -> Tuple[int, int]:
    if offset < 0:
        offset = 0
    prefix = source[:offset]
    line = prefix.count("\n") + 1
    last_nl = prefix.rfind("\n")
    col = offset - (last_nl + 1) + 1 if last_nl != -1 else offset + 1
    return line, col


def _make_snippet(source: str, start: int, end: int, context_lines: int = 2) -> str:
    lines = source.splitlines()
    start_line, _ = _offset_to_linecol(source, start)
    end_line, _ = _offset_to_linecol(source, end)
    sidx = max(0, start_line - 1 - context_lines)
    eidx = min(len(lines), end_line + context_lines)
    snippet_lines = []
    for i in range(sidx, eidx):
        prefix = "   "
        if i >= (start_line - 1) and i <= (end_line - 1):
            prefix = ">> "
        snippet_lines.append(f"{prefix}{i+1:4d}: {lines[i]}")
    return "\n".join(snippet_lines)


def _normalize_marker(marker: Optional[str]) -> str:
    return (marker or "TODO").upper()


def _compute_confidence(marker: str, metadata: Dict[str, Any]) -> float:
    m = _normalize_marker(marker)
    base = _SEVERITY_BY_MARKER.get(m, 0.5)
    pr = metadata.get("priority")
    if pr:
        if isinstance(pr, str):
            base = max(base, _PRIORITY_MAP.get(pr.lower(), base))
    if metadata.get("assignee"):
        base = min(0.99, base + 0.05)
    if metadata.get("issue"):
        base = min(0.98, base + 0.03)
    # due date urgency: if due soon, boost
    due = metadata.get("due")
    if due:
        try:
            due_ts = time.mktime(time.strptime(due, "%Y-%m-%d"))
            days = (due_ts - time.time()) / 86400.0
            if days <= 0:
                base = min(0.999, base + 0.08)  # overdue
            elif days < 7:
                base = min(0.99, base + 0.05)
        except Exception:
            pass
    return round(base, 2)


# --- core parsing (robust, optimized) ---
def parse_todos(source: str, filename: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Parse source and collect structured TODO dictionaries.
    Optimized to avoid expensive allocations on large files.
    """
    todos: List[Dict[str, Any]] = []

    # single-line matches (fast)
    for m in _INLINE_MARKER_RE.finditer(source):
        marker = m.group("marker") or "TODO"
        rest = (m.group("rest") or "").strip()
        start, end = m.start(), m.end()
        assignee = (_ASSIGNEE_RE.search(rest).group("assignee") if _ASSIGNEE_RE.search(rest) else None)
        issue = (_ISSUE_RE.search(rest).group("issue") if _ISSUE_RE.search(rest) else None)
        pr_match = _PRIORITY_RE.search(rest)
        priority = pr_match.group("pnum") or pr_match.group("pword") if pr_match else None
        due = (_DUE_RE.search(rest).group("date") if _DUE_RE.search(rest) else None)
        tags = [mo.group("tag") for mo in _TAG_RE.finditer(rest) if mo.group("tag") != (assignee or "")]
        line, col = _offset_to_linecol(source, start)
        snippet = _make_snippet(source, start, end)
        todos.append({
            "marker": marker.upper(),
            "text": rest,
            "assignee": assignee,
            "issue": issue,
            "priority": priority,
            "due": due,
            "tags": tags,
            "start_offset": start,
            "end_offset": end,
            "line": line,
            "col": col,
            "snippet": snippet,
            "filename": filename
        })

    # block comments
    for bm in _BLOCK_COMMENT_RE.finditer(source):
        body = bm.group("body") or ""
        for im in re.finditer(rf"(?P<marker>{_MARKER_WORDS})[:\s]?(?P<rest>.*)$", body, re.IGNORECASE | re.MULTILINE):
            marker = im.group("marker") or "TODO"
            rest = (im.group("rest") or "").strip()
            inner_rel = im.start()
            start = bm.start() + 2 + inner_rel
            end = start + (im.end() - im.start())
            assignee = (_ASSIGNEE_RE.search(rest).group("assignee") if _ASSIGNEE_RE.search(rest) else None)
            issue = (_ISSUE_RE.search(rest).group("issue") if _ISSUE_RE.search(rest) else None)
            pr_match = _PRIORITY_RE.search(rest)
            priority = pr_match.group("pnum") or pr_match.group("pword") if pr_match else None
            due = (_DUE_RE.search(rest).group("date") if _DUE_RE.search(rest) else None)
            tags = [mo.group("tag") for mo in _TAG_RE.finditer(rest) if mo.group("tag") != (assignee or "")]
            line, col = _offset_to_linecol(source, start)
            snippet = _make_snippet(source, start, end)
            todos.append({
                "marker": marker.upper(),
                "text": rest,
                "assignee": assignee,
                "issue": issue,
                "priority": priority,
                "due": due,
                "tags": tags,
                "start_offset": start,
                "end_offset": end,
                "line": line,
                "col": col,
                "snippet": snippet,
                "filename": filename
            })

    # dedupe
    seen = set()
    unique: List[Dict[str, Any]] = []
    for t in todos:
        key = (t["filename"], t["start_offset"], t["end_offset"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(t)
    return unique


# --- caching / concurrent scanning / async support ---
def parse_todos_cached(path: str) -> List[Dict[str, Any]]:
    try:
        mtime = os.path.getmtime(path)
    except Exception:
        mtime = 0.0
    entry = _file_cache.get(path)
    if entry and entry[0] == mtime:
        return entry[1]
    try:
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
    except Exception:
        return []
    parsed = parse_todos(src, filename=path)
    # keep cache bounded using simple LRU behavior (dict + size check)
    _file_cache[path] = (mtime, parsed)
    if len(_file_cache) > _parse_cache_max:
        # drop oldest entries (not strictly LRU but simple)
        for k in list(_file_cache.keys())[: len(_file_cache)//4]:
            _file_cache.pop(k, None)
    return parsed


def _gather_files(paths: Iterable[str], skip_hidden: bool = True) -> List[str]:
    files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, fs in os.walk(p):
                if skip_hidden:
                    dirs[:] = [d for d in dirs if not d.startswith(".")]
                for f in fs:
                    if f.startswith("."):
                        continue
                    files.append(os.path.join(root, f))
        else:
            files.append(p)
    return files


def scan_sources_concurrent(paths: Iterable[str], max_workers: int = 8, skip_hidden: bool = True) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    files = _gather_files(paths, skip_hidden=skip_hidden)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(parse_todos_cached, f): f for f in files}
        for fut in as_completed(futures):
            f = futures[fut]
            try:
                res = fut.result()
            except Exception:
                res = []
            if res:
                out[f] = res
    return out


async def scan_sources_async(paths: Iterable[str], max_workers: int = 8, skip_hidden: bool = True) -> Dict[str, List[Dict[str, Any]]]:
    """
    Async scanning: uses aiofiles if available; otherwise falls back to threadpool.
    """
    files = _gather_files(paths, skip_hidden=skip_hidden)
    out: Dict[str, List[Dict[str, Any]]] = {}
    if _HAS_AIOFILES:
        sem = asyncio.Semaphore(max_workers)
        async def _read_and_parse(path: str):
            async with sem:
                try:
                    async with aiofiles.open(path, "r", encoding="utf-8") as fh:
                        src = await fh.read()
                except Exception:
                    return path, []
                parsed = parse_todos(src, filename=path)
                return path, parsed
        tasks = [asyncio.create_task(_read_and_parse(p)) for p in files]
        for t in await asyncio.gather(*tasks):
            if t[1]:
                out[t[0]] = t[1]
    else:
        # fallback
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [loop.run_in_executor(ex, parse_todos_cached, p) for p in files]
            for fut, path in zip(asyncio.as_completed(futures), files):
                res = await fut
                if res:
                    out[path] = res
    return out


# --- ignore patterns (.todoignore) ---
def load_todoignore(root_dir: str) -> List[str]:
    path = os.path.join(root_dir, ".todoignore")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip() and not ln.strip().startswith("#")]
        return lines
    except Exception:
        return []


def matches_ignore(path: str, ignore_patterns: Iterable[str]) -> bool:
    import fnmatch
    for p in ignore_patterns:
        if fnmatch.fnmatch(path, p) or fnmatch.fnmatch(os.path.basename(path), p):
            return True
    return False


# --- prioritization and scoring helpers ---
def score_todo(todo: Dict[str, Any]) -> float:
    """Compute a score (0..1) for prioritization."""
    conf = _compute_confidence(todo.get("marker", "TODO"), todo)
    # priority factor
    pr = todo.get("priority")
    pr_score = 0.0
    if pr:
        pr_score = _PRIORITY_MAP.get(str(pr).lower(), 0.0)
    # file age factor (older files less urgent unless due soon)
    age_days = 0.0
    try:
        mtime = os.path.getmtime(todo.get("filename")) if todo.get("filename") else None
        if mtime:
            age_days = (time.time() - mtime) / 86400.0
    except Exception:
        age_days = 0.0
    age_factor = 1.0 / (1.0 + math.log1p(age_days + 1.0))
    # due boost
    due = todo.get("due")
    due_boost = 0.0
    if due:
        try:
            due_ts = time.mktime(time.strptime(due, "%Y-%m-%d"))
            days_left = (due_ts - time.time()) / 86400.0
            if days_left <= 0:
                due_boost = 0.15
            elif days_left < 7:
                due_boost = 0.07
        except Exception:
            pass
    score = (conf * 0.6) + (pr_score * 0.25) + (age_factor * 0.1) + due_boost
    return min(0.999, round(score, 3))


def prioritize_todos(todos: Iterable[Dict[str, Any]], top_n: Optional[int] = None) -> List[Dict[str, Any]]:
    scored = []
    for t in todos:
        t = dict(t)  # copy
        t["_score"] = score_todo(t)
        scored.append(t)
    scored.sort(key=lambda x: (-x["_score"], x.get("priority") or "", x.get("filename") or ""))
    if top_n:
        return scored[:top_n]
    return scored


# --- Git helpers: git-blame assignee suggestion ---
def suggest_assignee_by_git_blame(path: str, line: int) -> Optional[str]:
    """
    Uses `git blame --porcelain -L line,line path` to extract the author name.
    Returns username (author-mail or author name) when available.
    """
    try:
        proc = subprocess.run(["git", "blame", "--porcelain", f"-L{line},{line}", path],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        out = proc.stdout or ""
        for ln in out.splitlines():
            if ln.startswith("author-mail "):
                mail = ln.split(" ", 1)[1].strip()
                # normalize <name@domain>
                return mail.strip("<>")
            if ln.startswith("author "):
                return ln.split(" ", 1)[1].strip()
    except Exception:
        pass
    return None


# --- quick-fix apply (atomic) ---
def apply_quickfix_patch_to_file(path: str, patch: Dict[str, Any], *, make_backup: bool = True) -> bool:
    """
    patch: {"start": int, "end": int, "replacement": str}
    Applies patch to file using byte offsets. Writes atomically and optionally keeps a backup.
    """
    start = patch.get("start"); end = patch.get("end"); repl = patch.get("replacement", "")
    if start is None or end is None:
        return False
    try:
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
    except Exception:
        return False
    if start < 0 or end > len(content) or start > end:
        return False
    new = content[:start] + repl + content[end:]
    if make_backup:
        bak = path + ".todo.bak"
        try:
            shutil.copy(path, bak)
        except Exception:
            pass
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(new)
        os.replace(tmp, path)
        return True
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        return False


# --- GitHub integration: optional poster (uses GITHUB_TOKEN env) ---
def post_github_issue(repo: str, payload: Dict[str, Any], token: Optional[str] = None, dry_run: bool = True) -> Dict[str, Any]:
    """
    Post an issue to GitHub repository "owner/repo". If dry_run=True, returns payload without posting.
    If token not provided, reads from GITHUB_TOKEN env var.
    Returns response dict or payload on dry-run/errors.
    """
    if token is None:
        token = os.environ.get("GITHUB_TOKEN")
    api = f"https://api.github.com/repos/{repo}/issues"
    if dry_run:
        return {"dry_run": True, "api": api, "payload": payload}
    if not token:
        raise RuntimeError("No GITHUB_TOKEN provided for posting issues")
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(api, data=data, method="POST", headers={
        "Authorization": f"token {token}",
        "User-Agent": "ciams-todo-plugin",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as he:
        return {"error": he.read().decode("utf-8"), "code": he.code}
    except Exception as e:
        return {"error": str(e)}


# --- report exporters ---
def export_markdown(todos: Iterable[Dict[str, Any]], path: str, title: str = "TODO Report") -> str:
    todos = list(todos)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# {title}\n\n")
        fh.write("| File | Line | Marker | Priority | Assignee | Summary |\n")
        fh.write("|---|---:|---|---|---|---|\n")
        for t in todos:
            fn = os.path.relpath(t.get("filename") or "<unknown>")
            line = t.get("line", 0)
            marker = t.get("marker", "")
            pr = t.get("priority") or ""
            ass = t.get("assignee") or ""
            summary = (t.get("text") or "").splitlines()[0][:80]
            fh.write(f"| {fn} | {line} | {marker} | {pr} | {ass} | {summary} |\n")
    return path


def export_csv(todos: Iterable[Dict[str, Any]], path: str) -> str:
    todos = list(todos)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "line", "col", "marker", "priority", "assignee", "issue", "text"])
        for t in todos:
            writer.writerow([t.get("filename") or "", t.get("line") or "", t.get("col") or "", t.get("marker") or "", t.get("priority") or "", t.get("assignee") or "", t.get("issue") or "", t.get("text") or ""])
    return path


# --- assistant integration wrapper ---
def rule_todos(source: str, filename: Optional[str] = None) -> List[Suggestion]:
    suggestions: List[Suggestion] = []
    parsed = parse_todos(source, filename=filename)
    for t in parsed:
        marker = t["marker"]
        text = t["text"] or ""
        tags = [marker.lower()] + t.get("tags", [])
        if t.get("priority"):
            tags.append(f"priority:{t['priority']}")
        if t.get("assignee"):
            tags.append(f"assignee:{t['assignee']}")
        if t.get("issue"):
            tags.append(f"issue:{t['issue']}")
        short = text.splitlines()[0] if text else ""
        title = f"{marker}: {short}" if short else f"{marker}"
        desc_lines = [
            f"File: {t['filename'] or '<unknown>'}  Line: {t['line']}  Col: {t['col']}",
            f"Text: {text}",
        ]
        if t.get("assignee"):
            desc_lines.append(f"Assignee: @{t['assignee']}")
        if t.get("issue"):
            desc_lines.append(f"Issue: #{t['issue']}")
        if t.get("priority"):
            desc_lines.append(f"Priority: {t['priority']}")
        if t.get("due"):
            desc_lines.append(f"Due: {t['due']}")
        desc = " | ".join(desc_lines)
        confidence = _compute_confidence(marker, t)
        snippet = t.get("snippet", "")
        location = (t["line"], t["col"], t["start_offset"], t["end_offset"])
        suggestions.append(Suggestion("task", tags, title, confidence, snippet, location))
    return suggestions


# --- plugin registration and exposing helpers to assistant ---
def _expose_tools_on_assistant(assistant) -> None:
    tools = {
        "parse_todos": parse_todos,
        "parse_todos_cached": parse_todos_cached,
        "scan_sources_concurrent": scan_sources_concurrent,
        "scan_sources_async": scan_sources_async,
        "export_json": lambda todos, p: export_json(list(todos), p),
        "export_sarif": export_sarif,
        "export_markdown": export_markdown,
        "export_csv": export_csv,
        "generate_quickfix_patch": generate_quickfix_patch,
        "apply_quickfix_patch_to_file": apply_quickfix_patch_to_file,
        "generate_issue_payload": generate_issue_payload,
        "post_github_issue": post_github_issue,
        "group_todos": group_todos,
        "summary_stats": summary_stats,
        "prioritize_todos": prioritize_todos,
        "suggest_assignee_by_git_blame": suggest_assignee_by_git_blame,
        "load_todoignore": load_todoignore,
    }
    for name, fn in tools.items():
        if hasattr(assistant, name):
            continue
        try:
            setattr(assistant, name, fn)
        except Exception:
            pass


def register(assistant) -> Callable:
    assistant.register_rule(rule_todos)
    try:
        _expose_tools_on_assistant(assistant)
    except Exception:
        pass

    def unregister_fn(asst=None):
        try:
            assistant.unregister_rule(rule_todos)
        except Exception:
            pass
        # remove helpers only if value equals our functions
        for n, fn in list(globals().items()):
            if n.startswith("_") or n in ("register", "unregister"):
                continue
            try:
                if getattr(assistant, n, None) is fn:
                    delattr(assistant, n)
            except Exception:
                pass

    return unregister_fn


def unregister(assistant) -> None:
    try:
        assistant.unregister_rule(rule_todos)
    except Exception:
        pass


# --- CLI helpers and exporters reused from above ---
def export_json(todos: Iterable[Dict[str, Any]], path: str) -> str:
    data = list(todos)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"generated": time.time(), "todos": data}, fh, indent=2, default=str)
    return path


def export_sarif(todos: Iterable[Dict[str, Any]], path: str, tool_name: str = "ciams-todo-plugin") -> str:
    todos = list(todos)
    rules = {}
    results = []
    for t in todos:
        rule_id = t.get("marker") or "TODO"
        if rule_id not in rules:
            rules[rule_id] = {"id": rule_id, "shortDescription": {"text": rule_id}, "defaultConfiguration": {"level": "warning"}}
        result = {
            "ruleId": rule_id,
            "level": "warning",
            "message": {"text": t.get("text", "") or rule_id},
            "locations": [{
                "physicalLocation": {
                    "artifactLocation": {"uri": t.get("filename") or "<unknown>"},
                    "region": {"startLine": t.get("line", 1), "startColumn": t.get("col", 1)}
                }
            }]
        }
        results.append(result)
    sarif = {
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {"name": tool_name, "rules": list(rules.values())}},
            "results": results
        }]
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(sarif, fh, indent=2)
    return path


# CLI entry
def _cli_main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="todo_task_plugin", description="Scan source paths for TODOs and export results.")
    parser.add_argument("paths", nargs="+", help="Files or directories to scan")
    parser.add_argument("--out-json", help="Write parsed todos to this JSON file")
    parser.add_argument("--out-sarif", help="Write SARIF to this path")
    parser.add_argument("--out-md", help="Write Markdown report")
    parser.add_argument("--out-csv", help="Write CSV report")
    parser.add_argument("--workers", type=int, default=8, help="Concurrency for file scanning")
    parser.add_argument("--github-repo", help="If provided, generate GitHub issue dry-run payloads")
    parser.add_argument("--apply-patches", action="store_true", help="Apply quickfix patches (use with caution)")
    args = parser.parse_args(argv)

    results = scan_sources_concurrent(args.paths, max_workers=args.workers)
    all_todos = []
    for v in results.values():
        all_todos.extend(v)
    if args.out_json:
        export_json(all_todos, args.out_json)
    if args.out_sarif:
        export_sarif(all_todos, args.out_sarif)
    if args.out_md:
        export_markdown(all_todos, args.out_md)
    if args.out_csv:
        export_csv(all_todos, args.out_csv)

    if args.github_repo:
        # produce issue payloads (dry-run)
        payloads = [generate_issue_payload(t, repo=args.github_repo) for t in all_todos]
        print(json.dumps({"issues": payloads}, indent=2))

    if args.apply_patches:
        for t in all_todos:
            patch = generate_quickfix_patch(t, action="issue")
            if patch:
                apply_quickfix_patch_to_file(t.get("filename"), patch)

    stats = summary_stats(all_todos)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli_main())
